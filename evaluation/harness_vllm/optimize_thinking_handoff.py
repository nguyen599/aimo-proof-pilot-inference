from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from openai import OpenAI
from transformers import AutoTokenizer

try:
    from evaluation.harness_vllm.thinking_handoff import (
        FINAL_PARTIAL_FORCE_MARKER,
        HANDOFF_ASSISTANT_PREFIX,
        HANDOFF_REQUIRED_SECTIONS,
        HANDOFF_SECTION_MAX_TOKENS,
        HANDOFF_VARIANTS,
        SavedProofGenerationCall,
        assemble_handoff,
        build_fresh_handoff_section_prompt_ids,
        build_handoff_instruction,
        build_handoff_repair_instruction,
        build_handoff_section_instruction,
        build_user_turn_prompt_ids,
        handoff_section_assistant_prefix,
        normalize_handoff_section,
        parse_handoff_response,
        parse_saved_proof_generation_call,
        remove_final_partial_force_text,
    )
except ModuleNotFoundError as exc:
    if exc.name != "evaluation":
        raise
    from thinking_handoff import (  # type: ignore[no-redef]
        FINAL_PARTIAL_FORCE_MARKER,
        HANDOFF_ASSISTANT_PREFIX,
        HANDOFF_REQUIRED_SECTIONS,
        HANDOFF_SECTION_MAX_TOKENS,
        HANDOFF_VARIANTS,
        SavedProofGenerationCall,
        assemble_handoff,
        build_fresh_handoff_section_prompt_ids,
        build_handoff_instruction,
        build_handoff_repair_instruction,
        build_handoff_section_instruction,
        build_user_turn_prompt_ids,
        handoff_section_assistant_prefix,
        normalize_handoff_section,
        parse_handoff_response,
        parse_saved_proof_generation_call,
        remove_final_partial_force_text,
    )


DEFAULT_TEMPERATURES = (1.0, 0.7, 0.6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize the thinking-budget handoff prompt using saved proof-generation "
            "contexts. This tool is independent of run.py and never regenerates the "
            "original long proof attempt."
        )
    )
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--base-url", action="append", default=[])
    parser.add_argument("--served-model-name", default="proof-model")
    parser.add_argument("--api-key", default="vllm-local")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-count", type=int, default=8)
    parser.add_argument(
        "--variant",
        action="append",
        choices=HANDOFF_VARIANTS,
        default=[],
    )
    parser.add_argument(
        "--temperature",
        action="append",
        type=float,
        default=[],
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--generation-mode",
        choices=("monolithic", "sectioned", "fresh_sectioned"),
        default="fresh_sectioned",
    )
    parser.add_argument(
        "--repair-invalid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror run.py by making one XML repair call after an invalid handoff.",
    )
    parser.add_argument("--max-token-drift", type=int, default=4)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--request-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def relative_parts(path: Path, root: Path) -> tuple[str, str]:
    relative = path.relative_to(root)
    rank = next(
        (part for part in relative.parts if part.startswith("rank")),
        "rank-unknown",
    )
    llm_index = relative.parts.index("llm_calls")
    problem = relative.parts[llm_index + 1]
    return rank, problem


def old_partial_is_parseable(record: SavedProofGenerationCall) -> bool:
    marker_index = record.output_text.rfind(FINAL_PARTIAL_FORCE_MARKER)
    if marker_index < 0:
        return False
    visible = record.output_text[marker_index:]
    return all(
        marker in visible
        for marker in (
            "</solution>",
            "<self_evaluation>",
            "</self_evaluation>",
            "<score>",
            "</score>",
        )
    )


def discover_cases(logs_root: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in sorted(logs_root.glob("rank*/llm_calls/*/*proof_gen*.txt")):
        raw = path.read_text(encoding="utf-8")
        if FINAL_PARTIAL_FORCE_MARKER not in raw:
            continue
        record = parse_saved_proof_generation_call(path)
        rank, problem = relative_parts(path, logs_root)
        cases.append(
            {
                "record": record,
                "rank": rank,
                "problem": problem,
                "old_parseable": old_partial_is_parseable(record),
                "source": str(path.relative_to(logs_root)),
            }
        )
    if not cases:
        raise ValueError(
            f"no thinking-budget proof-generation calls found in {logs_root}"
        )
    return cases


def select_diverse_cases(
    cases: list[dict[str, Any]],
    case_count: int,
) -> list[dict[str, Any]]:
    if case_count < 1:
        raise ValueError("--case-count must be positive")
    buckets: dict[tuple[str, str, bool], deque[dict[str, Any]]] = defaultdict(deque)
    for case in cases:
        buckets[
            (
                str(case["rank"]),
                str(case["problem"]),
                bool(case["old_parseable"]),
            )
        ].append(case)

    selected: list[dict[str, Any]] = []
    ordered_keys = sorted(buckets)
    while len(selected) < min(case_count, len(cases)):
        made_progress = False
        for key in ordered_keys:
            if buckets[key]:
                selected.append(buckets[key].popleft())
                made_progress = True
                if len(selected) >= min(case_count, len(cases)):
                    break
        if not made_progress:
            break
    return selected


def prepare_case(
    case: dict[str, Any],
    tokenizer: Any,
    *,
    max_token_drift: int = 4,
) -> dict[str, Any]:
    record: SavedProofGenerationCall = case["record"]
    pre_force_text = remove_final_partial_force_text(record.continuation_prompt)
    full_ids = tokenizer.encode(
        record.continuation_prompt,
        add_special_tokens=False,
    )
    pre_force_ids = tokenizer.encode(pre_force_text, add_special_tokens=False)
    if hasattr(full_ids, "tolist"):
        full_ids = full_ids.tolist()
    if hasattr(pre_force_ids, "tolist"):
        pre_force_ids = pre_force_ids.tolist()
    token_drift = len(full_ids) - int(record.continuation_prompt_tokens)
    canonical_text = tokenizer.decode(full_ids, skip_special_tokens=False)
    if canonical_text != record.continuation_prompt:
        raise ValueError(
            f"tokenizer round trip changed saved prompt text for {record.path}"
        )
    if abs(token_drift) > max_token_drift:
        raise ValueError(
            f"tokenizer token-count drift exceeds {max_token_drift} for "
            f"{record.path}: logged={record.continuation_prompt_tokens} "
            f"encoded={len(full_ids)}"
        )
    return {
        **case,
        "pre_force_text": pre_force_text,
        "pre_force_ids": [int(value) for value in pre_force_ids],
        "token_drift": token_drift,
    }


def case_run_id(source: str, variant: str, temperature: float) -> str:
    digest = hashlib.sha256(
        f"{source}|{variant}|{temperature:g}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{Path(source).stem}-{variant}-t{temperature:g}-{digest}"


def call_handoff(
    *,
    prepared: dict[str, Any],
    tokenizer: Any,
    base_url: str,
    api_key: str,
    served_model_name: str,
    variant: str,
    temperature: float,
    max_tokens: int,
    generation_mode: str,
    repair_invalid: bool,
    top_p: float,
    request_timeout_seconds: float,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    client = OpenAI(
        base_url=base_url.rstrip("/") + "/v1"
        if not base_url.rstrip("/").endswith("/v1")
        else base_url.rstrip("/"),
        api_key=api_key,
        timeout=request_timeout_seconds,
        max_retries=2,
    )
    attempts: list[dict[str, Any]] = []

    def complete(
        active_prompt_ids: list[int],
        attempt_name: str,
        *,
        assistant_prefix: str = HANDOFF_ASSISTANT_PREFIX,
        attempt_max_tokens: int = max_tokens,
    ) -> dict[str, Any]:
        response = client.completions.create(
            model=served_model_name,
            prompt=active_prompt_ids,
            temperature=temperature,
            top_p=top_p,
            max_tokens=attempt_max_tokens,
        )
        choice = response.choices[0]
        completion_text = choice.text or ""
        raw_output = assistant_prefix + completion_text
        usage = {
            key: getattr(response.usage, key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if response.usage is not None and hasattr(response.usage, key)
        }
        return {
            "name": attempt_name,
            "prompt_ids": active_prompt_ids,
            "completion_text": completion_text,
            "raw_output": raw_output,
            "parsed": parse_handoff_response(raw_output),
            "finish_reason": choice.finish_reason,
            "usage": usage,
        }

    if generation_mode in {"sectioned", "fresh_sectioned"}:
        sections: dict[str, str] = {}
        for section in HANDOFF_REQUIRED_SECTIONS:
            assistant_prefix = handoff_section_assistant_prefix(section)
            context_metadata: dict[str, Any] = {}
            if generation_mode == "fresh_sectioned":
                section_prompt_ids, context_metadata = (
                    build_fresh_handoff_section_prompt_ids(
                        tokenizer,
                        original_input_prompt=prepared["record"].input_prompt,
                        pre_force_text=prepared["pre_force_text"],
                        section=section,
                        variant=variant,
                    )
                )
            else:
                section_prompt_ids = build_user_turn_prompt_ids(
                    tokenizer,
                    prepared["pre_force_ids"],
                    build_handoff_section_instruction(section, variant),
                    close_open_thinking=True,
                    assistant_prefix=assistant_prefix,
                )
            attempt = complete(
                section_prompt_ids,
                section,
                assistant_prefix=assistant_prefix,
                attempt_max_tokens=min(
                    max_tokens,
                    HANDOFF_SECTION_MAX_TOKENS[section],
                ),
            )
            section_content = normalize_handoff_section(
                attempt["completion_text"],
                section,
            )
            attempt["section"] = section
            attempt["section_content"] = section_content
            attempt["context_metadata"] = context_metadata
            attempts.append(attempt)
            sections[section] = section_content
        prompt_ids = attempts[0]["prompt_ids"]
        raw_output = assemble_handoff(sections)
        parsed = parse_handoff_response(raw_output)
        finish_reason = "sectioned"
    elif generation_mode == "monolithic":
        prompt_ids = build_user_turn_prompt_ids(
            tokenizer,
            prepared["pre_force_ids"],
            build_handoff_instruction(variant),
            close_open_thinking=True,
        )
        attempts.append(complete(prompt_ids, "initial"))
        if repair_invalid and not attempts[-1]["parsed"]["is_valid"]:
            completion_ids = tokenizer.encode(
                attempts[-1]["completion_text"],
                add_special_tokens=False,
            )
            if hasattr(completion_ids, "tolist"):
                completion_ids = completion_ids.tolist()
            previous_context_ids = prompt_ids + [int(value) for value in completion_ids]
            repair_prompt_ids = build_user_turn_prompt_ids(
                tokenizer,
                previous_context_ids,
                build_handoff_repair_instruction(),
                close_open_thinking=False,
            )
            attempts.append(complete(repair_prompt_ids, "repair"))
        final_attempt = attempts[-1]
        raw_output = final_attempt["raw_output"]
        parsed = final_attempt["parsed"]
        finish_reason = final_attempt["finish_reason"]
    else:
        raise ValueError(f"unsupported handoff generation mode: {generation_mode!r}")

    usage = {
        key: sum(int(attempt["usage"].get(key) or 0) for attempt in attempts)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    run_id = case_run_id(prepared["source"], variant, temperature)
    call_path = output_dir / "calls" / f"{run_id}.txt"
    call_path.parent.mkdir(parents=True, exist_ok=True)
    call_path.write_text(
        "\n".join(
            [
                f"source: {prepared['source']}",
                f"rank: {prepared['rank']}",
                f"problem: {prepared['problem']}",
                f"old_parseable: {prepared['old_parseable']}",
                f"variant: {variant}",
                f"temperature: {temperature:g}",
                f"generation_mode: {generation_mode}",
                f"base_url: {base_url}",
                f"prompt_tokens: {len(prompt_ids)}",
                f"max_tokens: {max_tokens}",
                "",
                *(
                    line
                    for attempt_index, attempt in enumerate(attempts, start=1)
                    for line in (
                        f"===== ATTEMPT {attempt_index}: {attempt['name']} INPUT =====",
                        tokenizer.decode(
                            attempt["prompt_ids"],
                            skip_special_tokens=False,
                        ),
                        "",
                        f"===== ATTEMPT {attempt_index}: {attempt['name']} OUTPUT =====",
                        attempt["raw_output"],
                        "",
                    )
                ),
            ]
        ),
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "source": prepared["source"],
        "rank": prepared["rank"],
        "problem": prepared["problem"],
        "old_parseable": prepared["old_parseable"],
        "variant": variant,
        "temperature": temperature,
        "generation_mode": generation_mode,
        "base_url": base_url,
        "prompt_tokens": len(prompt_ids),
        "pre_force_tokens": len(prepared["pre_force_ids"]),
        "max_tokens": max_tokens,
        "finish_reason": finish_reason,
        "latency_s": time.monotonic() - started,
        "usage": usage,
        "parsed": parsed,
        "raw_output": raw_output,
        "repair_used": generation_mode == "monolithic" and len(attempts) > 1,
        "attempts": [
            {
                "name": attempt["name"],
                "section": attempt.get("section"),
                "context_metadata": attempt.get("context_metadata"),
                "finish_reason": attempt["finish_reason"],
                "usage": attempt["usage"],
                "parsed": attempt["parsed"],
                "raw_output": attempt["raw_output"],
            }
            for attempt in attempts
        ],
        "call_log": str(call_path),
        "error": None,
    }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_results(output_dir: Path, results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_text = "".join(
        json.dumps(result, ensure_ascii=False, default=str) + "\n" for result in results
    )
    atomic_write_text(output_dir / "results.jsonl", jsonl_text)

    rows = [
        {
            "run_id": result["run_id"],
            "source": result["source"],
            "rank": result["rank"],
            "problem": result["problem"],
            "old_parseable": result["old_parseable"],
            "variant": result["variant"],
            "temperature": result["temperature"],
            "generation_mode": result.get("generation_mode", "monolithic"),
            "base_url": result["base_url"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result.get("usage", {}).get("completion_tokens"),
            "finish_reason": result.get("finish_reason"),
            "latency_s": result.get("latency_s"),
            "repair_used": result.get("repair_used", False),
            "initial_valid": (
                result.get("attempts", [{}])[0].get("parsed", {}).get("is_valid", False)
            ),
            "valid_handoff": result.get("parsed", {}).get("is_valid", False),
            "missing_sections": ",".join(
                result.get("parsed", {}).get("missing_sections", [])
            ),
            "call_log": result.get("call_log"),
            "error": result.get("error"),
        }
        for result in results
    ]
    csv_output = io.StringIO(newline="")
    with csv_output as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
        csv_text = output.getvalue()
    atomic_write_text(output_dir / "results.csv", csv_text)

    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[(result["variant"], float(result["temperature"]))].append(result)
    summary = []
    for (variant, temperature), group in sorted(groups.items()):
        successful = [result for result in group if not result.get("error")]
        valid = sum(
            bool(result.get("parsed", {}).get("is_valid")) for result in successful
        )
        summary.append(
            {
                "variant": variant,
                "temperature": temperature,
                "cases": len(group),
                "successful": len(successful),
                "failed": len(group) - len(successful),
                "valid": valid,
                "valid_fraction": valid / len(successful) if successful else 0.0,
                "repair_count": sum(
                    bool(result.get("repair_used")) for result in successful
                ),
                "mean_completion_tokens": (
                    sum(
                        int(result.get("usage", {}).get("completion_tokens") or 0)
                        for result in successful
                    )
                    / len(successful)
                    if successful
                    else 0.0
                ),
                "mean_latency_s": (
                    sum(float(result["latency_s"]) for result in successful)
                    / len(successful)
                    if successful
                    else 0.0
                ),
            }
        )
    atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    review_path = output_dir / "REVIEW.md"
    atomic_write_text(
        review_path,
        "# Thinking-Budget Handoff Review\n\n"
        "Review each prompt/temperature group for:\n\n"
        "1. Fidelity to the saved reasoning, without invented lemmas.\n"
        "2. Preservation of reusable equations, notation, and proved facts.\n"
        "3. Precise description of failed approaches and the unresolved bottleneck.\n"
        "4. Concrete next steps that help a fresh solver avoid repetition.\n"
        "5. Concision and valid XML structure.\n\n"
        "Select a default only when it has no severe fidelity failure and is useful "
        "on at least six of the eight selected cases.\n",
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive")
    if args.max_token_drift < 0:
        raise ValueError("--max-token-drift cannot be negative")
    variants = args.variant or list(HANDOFF_VARIANTS)
    temperatures = args.temperature or list(DEFAULT_TEMPERATURES)
    if any(temperature < 0 for temperature in temperatures):
        raise ValueError("--temperature cannot be negative")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
    )
    discovered = discover_cases(args.logs_root)
    selected = select_diverse_cases(discovered, args.case_count)
    prepared = [
        prepare_case(
            case,
            tokenizer,
            max_token_drift=args.max_token_drift,
        )
        for case in selected
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selected_cases.json").write_text(
        json.dumps(
            [
                {
                    "source": case["source"],
                    "rank": case["rank"],
                    "problem": case["problem"],
                    "old_parseable": case["old_parseable"],
                    "logged_prompt_tokens": case["record"].continuation_prompt_tokens,
                    "pre_force_tokens": len(case["pre_force_ids"]),
                    "token_drift": case["token_drift"],
                    "finish_reason": case["record"].finish_reason,
                }
                for case in prepared
            ],
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    logging.info(
        "Prepared %d/%d saved cases with exact decoded-text round trips "
        "(token drift range %d..%d)",
        len(prepared),
        len(discovered),
        min(int(case["token_drift"]) for case in prepared),
        max(int(case["token_drift"]) for case in prepared),
    )
    if args.dry_run:
        return
    if not args.base_url:
        raise ValueError("at least one --base-url is required unless --dry-run is used")

    jobs = []
    endpoint_index = 0
    for case in prepared:
        for variant in variants:
            for temperature in temperatures:
                jobs.append(
                    {
                        "prepared": case,
                        "variant": variant,
                        "temperature": temperature,
                        "base_url": args.base_url[endpoint_index % len(args.base_url)],
                    }
                )
                endpoint_index += 1

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.max_workers, len(jobs))
    ) as executor:
        futures = {
            executor.submit(
                call_handoff,
                prepared=job["prepared"],
                tokenizer=tokenizer,
                base_url=job["base_url"],
                api_key=args.api_key,
                served_model_name=args.served_model_name,
                variant=job["variant"],
                temperature=job["temperature"],
                max_tokens=args.max_tokens,
                generation_mode=args.generation_mode,
                repair_invalid=args.repair_invalid,
                top_p=args.top_p,
                request_timeout_seconds=args.request_timeout_seconds,
                output_dir=args.output_dir,
            ): job
            for job in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                prepared_case = job["prepared"]
                result = {
                    "run_id": case_run_id(
                        prepared_case["source"],
                        job["variant"],
                        job["temperature"],
                    ),
                    "source": prepared_case["source"],
                    "rank": prepared_case["rank"],
                    "problem": prepared_case["problem"],
                    "old_parseable": prepared_case["old_parseable"],
                    "variant": job["variant"],
                    "temperature": job["temperature"],
                    "generation_mode": args.generation_mode,
                    "base_url": job["base_url"],
                    "prompt_tokens": None,
                    "pre_force_tokens": len(prepared_case["pre_force_ids"]),
                    "max_tokens": args.max_tokens,
                    "finish_reason": None,
                    "latency_s": None,
                    "usage": {},
                    "parsed": {
                        "is_valid": False,
                        "missing_sections": [],
                    },
                    "raw_output": "",
                    "call_log": None,
                    "error": repr(exc),
                }
                logging.exception(
                    "Handoff call failed source=%s variant=%s temperature=%s",
                    prepared_case["source"],
                    job["variant"],
                    job["temperature"],
                )
            results.append(result)
            results.sort(
                key=lambda item: (
                    item["source"],
                    item["variant"],
                    float(item["temperature"]),
                )
            )
            write_results(args.output_dir, results)
            if result.get("error"):
                logging.warning(
                    "Recorded failed %s error=%s",
                    result["run_id"],
                    result["error"],
                )
            else:
                logging.info(
                    "Completed %s valid=%s tokens=%s latency=%.1fs",
                    result["run_id"],
                    result["parsed"]["is_valid"],
                    result["usage"].get("completion_tokens"),
                    result["latency_s"],
                )

    logging.info(
        "Wrote %d handoff results to %s",
        len(results),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
