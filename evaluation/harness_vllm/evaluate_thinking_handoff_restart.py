from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from transformers import AutoTokenizer

try:
    from evaluation.harness_vllm.run import (
        merge_streamed_token_ids,
        parse_generation_response,
    )
    from evaluation.harness_vllm.thinking_handoff import (
        insert_restart_instruction_into_rendered_prompt,
        parse_saved_proof_generation_call,
    )
except ModuleNotFoundError as exc:
    if exc.name != "evaluation":
        raise
    from run import merge_streamed_token_ids, parse_generation_response
    from thinking_handoff import (  # type: ignore[no-redef]
        insert_restart_instruction_into_rendered_prompt,
        parse_saved_proof_generation_call,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether a saved thinking-budget handoff lets a fresh proof "
            "attempt finish before the next thinking cutoff."
        )
    )
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--handoff-results", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--base-url", action="append", default=[])
    parser.add_argument("--served-model-name", default="proof-model")
    parser.add_argument("--api-key", default="vllm-local")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--proof-temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--thinking-budget-tokens", type=int, default=122_000)
    parser.add_argument("--max-tokens", type=int, default=126_000)
    parser.add_argument("--case-count", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--request-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_handoffs(
    path: Path,
    *,
    variant: str,
    temperature: float,
    case_count: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        result = json.loads(line)
        if result.get("error"):
            continue
        if result.get("variant") != variant:
            continue
        if abs(float(result.get("temperature")) - temperature) > 1e-9:
            continue
        parsed = result.get("parsed") or {}
        if not parsed.get("is_valid") or not parsed.get("text"):
            continue
        selected.append(result)
    selected.sort(key=lambda item: str(item["source"]))
    if case_count > 0:
        selected = selected[:case_count]
    if not selected:
        raise ValueError(
            f"no valid handoffs for variant={variant} temperature={temperature:g}"
        )
    return selected


def completion_token_ids(choice: Any) -> list[int]:
    token_ids = getattr(choice, "token_ids", None)
    if token_ids is None:
        token_ids = (getattr(choice, "model_extra", None) or {}).get("token_ids")
    if token_ids is None:
        return []
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return [int(value) for value in token_ids]


def stream_segment(
    *,
    client: OpenAI,
    model: str,
    prompt_ids: list[int],
    temperature: float,
    top_p: float,
    max_tokens: int,
    stop_after_tokens: int | None,
) -> dict[str, Any]:
    stream = client.completions.create(
        model=model,
        prompt=prompt_ids,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=True,
        extra_body={"return_token_ids": True},
    )
    generated_ids: list[int] = []
    finish_reason = None
    stopped_by_budget = False
    try:
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            if getattr(choice, "finish_reason", None) is not None:
                finish_reason = choice.finish_reason
            incoming = completion_token_ids(choice)
            if incoming:
                generated_ids, _ = merge_streamed_token_ids(
                    generated_ids,
                    incoming,
                )
            if (
                stop_after_tokens is not None
                and finish_reason is None
                and len(generated_ids) >= stop_after_tokens
            ):
                stopped_by_budget = True
                break
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    if not generated_ids:
        raise RuntimeError("vLLM stream did not return token IDs")
    return {
        "generated_ids": generated_ids,
        "finish_reason": finish_reason,
        "stopped_by_budget": stopped_by_budget,
    }


def evaluate_restart(
    *,
    handoff_result: dict[str, Any],
    logs_root: Path,
    tokenizer: Any,
    base_url: str,
    api_key: str,
    served_model_name: str,
    proof_temperature: float,
    top_p: float,
    thinking_budget_tokens: int,
    max_tokens: int,
    request_timeout_seconds: float,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    source = str(handoff_result["source"])
    record = parse_saved_proof_generation_call(logs_root / source)
    handoff_text = str(handoff_result["parsed"]["text"])
    rendered_prompt = insert_restart_instruction_into_rendered_prompt(
        record.input_prompt,
        handoff_text,
        restart_round=1,
    )
    prompt_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    prompt_ids = [int(value) for value in prompt_ids]
    if tokenizer.decode(prompt_ids, skip_special_tokens=False) != rendered_prompt:
        raise ValueError(f"restart prompt tokenizer round trip changed text: {source}")

    client = OpenAI(
        base_url=(
            base_url.rstrip()
            if base_url.rstrip().endswith("/v1")
            else base_url.rstrip("/") + "/v1"
        ),
        api_key=api_key,
        timeout=request_timeout_seconds,
        max_retries=2,
    )
    first = stream_segment(
        client=client,
        model=served_model_name,
        prompt_ids=prompt_ids,
        temperature=proof_temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_after_tokens=thinking_budget_tokens,
    )
    generated_ids = list(first["generated_ids"])
    first_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    thinking_closed_at_budget = "</think>" in first_text.lower()
    hit_unfinished_thinking_budget = bool(
        first["stopped_by_budget"] and not thinking_closed_at_budget
    )
    finish_reason = first["finish_reason"]

    if first["stopped_by_budget"] and thinking_closed_at_budget:
        remaining_tokens = max_tokens - len(generated_ids)
        if remaining_tokens > 0:
            second = stream_segment(
                client=client,
                model=served_model_name,
                prompt_ids=prompt_ids + generated_ids,
                temperature=proof_temperature,
                top_p=top_p,
                max_tokens=remaining_tokens,
                stop_after_tokens=None,
            )
            generated_ids.extend(second["generated_ids"])
            finish_reason = second["finish_reason"]

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    parsed = parse_generation_response(
        output_text,
        require_self_evaluation=True,
    )
    latency_s = time.monotonic() - started
    call_name = Path(source).stem + ".txt"
    call_path = output_dir / "calls" / call_name
    call_path.parent.mkdir(parents=True, exist_ok=True)
    call_path.write_text(
        "\n".join(
            [
                f"source: {source}",
                f"handoff_run_id: {handoff_result['run_id']}",
                f"variant: {handoff_result['variant']}",
                f"handoff_temperature: {handoff_result['temperature']}",
                f"proof_temperature: {proof_temperature:g}",
                f"base_url: {base_url}",
                f"prompt_tokens: {len(prompt_ids)}",
                f"completion_tokens: {len(generated_ids)}",
                f"thinking_budget_tokens: {thinking_budget_tokens}",
                f"finish_reason: {finish_reason}",
                f"hit_unfinished_thinking_budget: {hit_unfinished_thinking_budget}",
                f"valid_proof: {parsed['is_valid_candidate_response']}",
                "",
                "===== HANDOFF =====",
                handoff_text,
                "",
                "===== RESTART PROMPT =====",
                rendered_prompt,
                "",
                "===== OUTPUT =====",
                output_text,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "source": source,
        "handoff_run_id": handoff_result["run_id"],
        "variant": handoff_result["variant"],
        "handoff_temperature": handoff_result["temperature"],
        "proof_temperature": proof_temperature,
        "base_url": base_url,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(generated_ids),
        "thinking_budget_tokens": thinking_budget_tokens,
        "finish_reason": finish_reason,
        "thinking_closed_at_budget": thinking_closed_at_budget,
        "hit_unfinished_thinking_budget": hit_unfinished_thinking_budget,
        "valid_proof": parsed["is_valid_candidate_response"],
        "verification_can_start": parsed["is_valid_candidate_response"],
        "proof_chars": len(str(parsed.get("proof") or "")),
        "self_score": parsed.get("self_score"),
        "latency_s": latency_s,
        "tokens_per_second": len(generated_ids) / latency_s if latency_s else 0.0,
        "call_log": str(call_path),
        "error": None,
    }


def write_results(output_dir: Path, results: list[dict[str, Any]]) -> None:
    ordered = sorted(results, key=lambda item: str(item["source"]))
    atomic_write_text(
        output_dir / "results.jsonl",
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered),
    )
    completed = [item for item in ordered if not item.get("error")]
    summary = {
        "cases": len(ordered),
        "completed": len(completed),
        "errors": len(ordered) - len(completed),
        "unfinished_thinking_budget_count": sum(
            bool(item.get("hit_unfinished_thinking_budget")) for item in completed
        ),
        "unfinished_thinking_budget_fraction": (
            sum(bool(item.get("hit_unfinished_thinking_budget")) for item in completed)
            / len(completed)
            if completed
            else 0.0
        ),
        "valid_proof_count": sum(bool(item.get("valid_proof")) for item in completed),
        "valid_proof_fraction": (
            sum(bool(item.get("valid_proof")) for item in completed) / len(completed)
            if completed
            else 0.0
        ),
        "verification_can_start_count": sum(
            bool(item.get("verification_can_start")) for item in completed
        ),
        "mean_completion_tokens": (
            sum(int(item.get("completion_tokens") or 0) for item in completed)
            / len(completed)
            if completed
            else 0.0
        ),
        "mean_tokens_per_second": (
            sum(float(item.get("tokens_per_second") or 0.0) for item in completed)
            / len(completed)
            if completed
            else 0.0
        ),
    }
    atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if not 0 < args.thinking_budget_tokens < args.max_tokens:
        raise ValueError(
            "--thinking-budget-tokens must be positive and below --max-tokens"
        )
    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive")
    handoffs = load_handoffs(
        args.handoff_results,
        variant=args.variant,
        temperature=args.temperature,
        case_count=args.case_count,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dry_run_rows = []
    for handoff in handoffs:
        record = parse_saved_proof_generation_call(
            args.logs_root / str(handoff["source"])
        )
        restarted = insert_restart_instruction_into_rendered_prompt(
            record.input_prompt,
            str(handoff["parsed"]["text"]),
            restart_round=1,
        )
        ids = tokenizer.encode(restarted, add_special_tokens=False)
        dry_run_rows.append(
            {
                "source": handoff["source"],
                "prompt_tokens": len(ids),
                "decoded_text_matches": (
                    tokenizer.decode(ids, skip_special_tokens=False) == restarted
                ),
            }
        )
    atomic_write_text(
        args.output_dir / "selected_cases.json",
        json.dumps(dry_run_rows, ensure_ascii=False, indent=2),
    )
    if args.dry_run:
        return
    if not args.base_url:
        raise ValueError("at least one --base-url is required")

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.max_workers, len(handoffs))
    ) as executor:
        futures = {}
        for index, handoff in enumerate(handoffs):
            base_url = args.base_url[index % len(args.base_url)]
            future = executor.submit(
                evaluate_restart,
                handoff_result=handoff,
                logs_root=args.logs_root,
                tokenizer=tokenizer,
                base_url=base_url,
                api_key=args.api_key,
                served_model_name=args.served_model_name,
                proof_temperature=args.proof_temperature,
                top_p=args.top_p,
                thinking_budget_tokens=args.thinking_budget_tokens,
                max_tokens=args.max_tokens,
                request_timeout_seconds=args.request_timeout_seconds,
                output_dir=args.output_dir,
            )
            futures[future] = (handoff, base_url)
        for future in concurrent.futures.as_completed(futures):
            handoff, base_url = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                logging.exception("Restart failed source=%s", handoff["source"])
                result = {
                    "source": handoff["source"],
                    "handoff_run_id": handoff["run_id"],
                    "variant": handoff["variant"],
                    "handoff_temperature": handoff["temperature"],
                    "proof_temperature": args.proof_temperature,
                    "base_url": base_url,
                    "error": repr(exc),
                }
            results.append(result)
            write_results(args.output_dir, results)
            logging.info(
                "Restart %d/%d source=%s cutoff=%s valid=%s tokens=%s error=%s",
                len(results),
                len(handoffs),
                result["source"],
                result.get("hit_unfinished_thinking_budget"),
                result.get("valid_proof"),
                result.get("completion_tokens"),
                result.get("error"),
            )


if __name__ == "__main__":
    main()
