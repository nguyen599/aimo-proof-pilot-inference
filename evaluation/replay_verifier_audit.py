#!/usr/bin/env python3
"""Replay selected proofs through the current verifier stack without generation."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from transformers import AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from evaluation.analyze_pipeline_run import parse_headers, split_output
    from evaluation.harness_vllm.run import (
        CFG,
        PROMPT_FAMILY_DEEPSEEK_MATH_V2,
        PROMPT_FAMILY_OPD,
        ChatScheduler,
        PipelineProgress,
        SamplingConfig,
        detect_id_column,
        detect_question_column,
        parse_deepseek_generation_response,
        parse_generation_response,
        read_input_table,
        run_verification_round,
    )
except ModuleNotFoundError as exc:
    if exc.name != "evaluation":
        raise
    from analyze_pipeline_run import (  # type: ignore[no-redef]
        parse_headers,
        split_output,
    )
    from harness_vllm.run import (  # type: ignore[no-redef]
        CFG,
        PROMPT_FAMILY_DEEPSEEK_MATH_V2,
        PROMPT_FAMILY_OPD,
        ChatScheduler,
        PipelineProgress,
        SamplingConfig,
        detect_id_column,
        detect_question_column,
        parse_deepseek_generation_response,
        parse_generation_response,
        read_input_table,
        run_verification_round,
    )


DETAIL_INDEX_PATTERN = re.compile(r"\b(candidate|round)=(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--proofs-path", type=Path)
    source.add_argument(
        "--llm-calls-dir",
        type=Path,
        help="Recursively replay completed proof_generation call logs.",
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--problem-id", action="append", default=[])
    parser.add_argument("--candidate-id", action="append", type=int, default=[])
    parser.add_argument("--round-index", action="append", type=int, default=[])
    parser.add_argument(
        "--proof-prompt-family",
        choices=("all", PROMPT_FAMILY_OPD, PROMPT_FAMILY_DEEPSEEK_MATH_V2),
        default=PROMPT_FAMILY_OPD,
        help="Filter raw proof calls; OPD is the current generation default.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Stop after this many parsed cases; zero keeps every matching call.",
    )
    parser.add_argument("--served-model-name", default="proof-model")
    parser.add_argument("--api-key", default="vllm-local")
    parser.add_argument("--verify-n", type=int, default=CFG.verify_n)
    parser.add_argument("--meta-n", type=int, default=1)
    parser.add_argument(
        "--meta-policy", choices=("all-reviews", "low-only"), default="all-reviews"
    )
    parser.add_argument("--max-concurrent-problems", type=int, default=3)
    parser.add_argument("--max-concurrent-requests", type=int, default=16)
    parser.add_argument("--verifier-max-tokens", type=int, default=32_000)
    parser.add_argument("--meta-max-tokens", type=int, default=32_000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--request-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--allow-case-errors",
        action="store_true",
        help=(
            "Write successful cases and structured failures, then exit zero. "
            "Without this flag, artifacts are still written before a nonzero exit."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def case_checkpoint_path(output_dir: Path, case: dict[str, Any]) -> Path:
    parts = [f"problem-{case['problem_id']}"]
    if case.get("source_candidate_id") is not None:
        parts.append(f"candidate-{case['source_candidate_id']}")
    if case.get("source_round_index") is not None:
        parts.append(f"round-{case['source_round_index']}")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", "_".join(parts)).strip("-")
    return output_dir / "cases" / f"{safe_name}.json"


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    frame = read_input_table(args.input_path)
    question_column = detect_question_column(frame, "auto")
    id_column = detect_id_column(frame)
    questions = {
        str(row[id_column] if id_column else index + 1): str(row[question_column])
        for index, row in frame.iterrows()
    }
    requested = {str(value) for value in args.problem_id}
    proofs_path = getattr(args, "proofs_path", None)
    llm_calls_dir = getattr(args, "llm_calls_dir", None)
    if bool(proofs_path) == bool(llm_calls_dir):
        raise ValueError("set exactly one of proofs_path or llm_calls_dir")

    cases = []
    if llm_calls_dir is not None:
        candidate_ids = set(getattr(args, "candidate_id", []) or [])
        round_indices = set(getattr(args, "round_index", []) or [])
        prompt_family_filter = str(
            getattr(args, "proof_prompt_family", PROMPT_FAMILY_OPD)
        )
        for path in sorted(Path(llm_calls_dir).rglob("*proof_gen*.txt")):
            text = path.read_text(encoding="utf-8", errors="replace")
            headers = parse_headers(text)
            if headers.get("stage") != "proof_generation":
                continue
            detail = headers.get("detail", "")
            indices = {
                key: int(value) for key, value in DETAIL_INDEX_PATTERN.findall(detail)
            }
            candidate_id = indices.get("candidate")
            round_index = indices.get("round")
            if candidate_id is None or round_index is None:
                continue
            if candidate_ids and candidate_id not in candidate_ids:
                continue
            if round_indices and round_index not in round_indices:
                continue
            prompt_family_match = re.search(r"\bprompt_family=([^ ]+)", detail)
            prompt_family = (
                prompt_family_match.group(1)
                if prompt_family_match is not None
                else PROMPT_FAMILY_OPD
            )
            if (
                prompt_family_filter != "all"
                and prompt_family != prompt_family_filter
            ):
                continue
            problem_id = str(path.parent.name)
            if requested and problem_id not in requested:
                continue
            if problem_id not in questions:
                raise KeyError(f"no problem text for proof id {problem_id!r}")
            metadata, output = split_output(text)
            if not metadata.get("success") or not output:
                continue
            parser = (
                parse_deepseek_generation_response
                if prompt_family == PROMPT_FAMILY_DEEPSEEK_MATH_V2
                else parse_generation_response
            )
            parsed = parser(output, require_self_evaluation=True)
            if not parsed.get("is_valid_candidate_response"):
                continue
            cases.append(
                {
                    "problem_id": problem_id,
                    "question": questions[problem_id],
                    "proof": parsed["proof"],
                    "self_evaluation": parsed.get("self_evaluation", ""),
                    "prompt_family": prompt_family,
                    "source_candidate_id": candidate_id,
                    "source_round_index": round_index,
                    "source_path": str(path),
                    "source_finish_reason": metadata.get("finish_reason"),
                    "old_internal_score": parsed.get("self_score"),
                    "old_internal_status": "raw_proof_generation",
                }
            )
        cases.sort(
            key=lambda case: (
                int(case["problem_id"]),
                int(case["source_candidate_id"]),
                int(case["source_round_index"]),
            )
        )
        max_cases = int(getattr(args, "max_cases", 0) or 0)
        if max_cases > 0:
            cases = cases[:max_cases]
        if not cases:
            raise ValueError(
                f"no parseable proof-generation calls under {llm_calls_dir}"
            )
    else:
        for record in read_jsonl(Path(proofs_path)):
            problem_id = str(record.get("problem_id"))
            if requested and problem_id not in requested:
                continue
            if problem_id not in questions:
                raise KeyError(f"no problem text for proof id {problem_id!r}")
            proof = str(record.get("final_proof") or "").strip()
            if not proof:
                raise ValueError(f"empty selected proof for problem {problem_id}")
            cases.append(
                {
                    "problem_id": problem_id,
                    "question": questions[problem_id],
                    "proof": proof,
                    "old_internal_score": record.get("final_score"),
                    "old_internal_status": record.get("final_status"),
                }
            )
    if requested - {case["problem_id"] for case in cases}:
        raise ValueError(f"missing requested proof ids: {sorted(requested)}")
    return sorted(cases, key=lambda case: case["problem_id"])


def replay_config(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        verify_n=max(1, int(args.verify_n)),
        meta_n=max(0, int(args.meta_n)),
        meta_policy=str(args.meta_policy),
        strict_pass_meta=bool(args.meta_n > 0),
        refine_rounds=0,
        refine_review_n=int(CFG.refine_review_n),
        min_valid_low=1,
        verification_early_stop=False,
        thinking_budget_enabled=True,
        verifier_thinking_budget_tokens=min(
            int(CFG.verifier_thinking_budget_tokens),
            max(1, int(args.verifier_max_tokens) - 1),
        ),
        verifier_thinking_budget_force_text=CFG.verifier_thinking_budget_force_text,
        deepseek_verifier_thinking_budget_force_text=(
            CFG.deepseek_verifier_thinking_budget_force_text
        ),
        meta_thinking_budget_tokens=min(
            int(CFG.meta_thinking_budget_tokens),
            max(1, int(args.meta_max_tokens) - 1),
        ),
        meta_thinking_budget_force_text=CFG.meta_thinking_budget_force_text,
    )


async def replay(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases(args)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), trust_remote_code=True
    )
    scheduler = ChatScheduler(
        base_urls=[str(url) for url in args.base_url],
        api_key=args.api_key,
        model=args.served_model_name,
        sampling=SamplingConfig(
            max_new_tokens=args.verifier_max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=-1,
            min_new_tokens=0,
            min_p=None,
        ),
        max_concurrent_requests=max(1, int(args.max_concurrent_requests)),
        stage_max_new_tokens={
            "proof_verify": int(args.verifier_max_tokens),
            "proof_meta_verify": int(args.meta_max_tokens),
        },
        request_timeout_seconds=float(args.request_timeout_seconds),
        request_worker_count=max(1, int(args.max_concurrent_requests)),
        stream_responses=True,
        context_length=int(CFG.num_ctx),
        tokenizer=tokenizer,
        llm_call_logdir=args.output_dir / "llm_calls",
        stream_interval_tokens=100,
    )
    cfg = replay_config(args)
    semaphore = asyncio.Semaphore(max(1, int(args.max_concurrent_problems)))

    async def evaluate(
        case: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        trace_problem_id = case["problem_id"]
        if case.get("source_candidate_id") is not None:
            trace_problem_id = (
                f"{case['problem_id']}-cand{case['source_candidate_id']}"
                f"-r{case['source_round_index']}"
            )
        progress = PipelineProgress(False, trace_problem_id, 1)
        failed_attempts = []
        try:
            for attempt_number in range(max(1, int(args.max_attempts))):
                replay_candidate_id = int(case["problem_id"]) * 100 + attempt_number
                if case.get("source_candidate_id") is not None:
                    replay_candidate_id = (
                        int(case["problem_id"]) * 1000
                        + int(case["source_candidate_id"]) * 10
                        + attempt_number
                    )
                async with semaphore:
                    (
                        verifier_results,
                        verifier_outputs,
                        meta_results,
                        meta_outputs,
                        aggregation,
                    ) = await run_verification_round(
                        case["question"],
                        case["proof"],
                        str(case.get("self_evaluation") or ""),
                        replay_candidate_id,
                        0,
                        scheduler,
                        cfg,
                        prompt_family=case.get("prompt_family", PROMPT_FAMILY_OPD),
                        progress=progress,
                    )
                parsed_scores = [result.get("score") for result in verifier_results]
                if len(parsed_scores) == cfg.verify_n and all(
                    score in {0.0, 0.5, 1.0} for score in parsed_scores
                ):
                    break
                failed_attempts.append(
                    {
                        "attempt": attempt_number + 1,
                        "verifier_scores": parsed_scores,
                    }
                )
            else:
                error = {
                    **case,
                    "error_type": "incomplete_verifier_scores",
                    "error": (
                        f"problem {case['problem_id']} has incomplete verifier "
                        f"scores after {max(1, int(args.max_attempts))} attempts"
                    ),
                    "failed_attempts": failed_attempts,
                }
                atomic_write_json(
                    case_checkpoint_path(args.output_dir, case),
                    {"schema_version": 1, "status": "error", "error": error},
                )
                return None, error
        except Exception as exc:
            error = {
                **case,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_attempts": failed_attempts,
            }
            atomic_write_json(
                case_checkpoint_path(args.output_dir, case),
                {"schema_version": 1, "status": "error", "error": error},
            )
            return None, error
        finally:
            progress.close()
        result = {
            **case,
            "replay_attempt": len(failed_attempts) + 1,
            "failed_attempts": failed_attempts,
            "verifier_results": verifier_results,
            "verifier_outputs": verifier_outputs,
            "meta_results_by_verifier": meta_results,
            "meta_outputs": meta_outputs,
            "aggregation": aggregation,
        }
        atomic_write_json(
            case_checkpoint_path(args.output_dir, case),
            {"schema_version": 1, "status": "ok", "result": result},
        )
        return result, None

    try:
        outcomes = await asyncio.gather(*(evaluate(case) for case in cases))
    finally:
        scheduler.close()
    results = [result for result, _ in outcomes if result is not None]
    errors = [error for _, error in outcomes if error is not None]
    return {
        "schema_version": 2,
        "settings": {
            "verify_n": cfg.verify_n,
            "meta_n": cfg.meta_n,
            "meta_policy": cfg.meta_policy,
            "verifier_max_tokens": args.verifier_max_tokens,
            "meta_max_tokens": args.meta_max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_attempts": max(1, int(args.max_attempts)),
        },
        "results": results,
        "errors": errors,
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Verifier audit replay",
        "",
        "| Problem | Candidate | Round | Prompt | Old score | New score | Status | Low cap | Fatal cap | Role scores |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for result in payload["results"]:
        aggregation = result["aggregation"]
        role_scores = ", ".join(
            f"{item.get('verifier_role')}={item.get('verifier_score')}"
            for item in aggregation.get("verifier_score_summaries") or []
        )
        lines.append(
            "| {problem_id} | {candidate} | {round_index} | {prompt_family} | {old} | {new} | {status} | {low_cap} | {fatal_cap} | {roles} |".format(
                problem_id=result["problem_id"],
                candidate=result.get("source_candidate_id", "-"),
                round_index=result.get("source_round_index", "-"),
                prompt_family=result.get("prompt_family", "-"),
                old=result.get("old_internal_score"),
                new=aggregation.get("final_score"),
                status=aggregation.get("final_status"),
                low_cap=aggregation.get("validated_low_score_cap_applied", False),
                fatal_cap=aggregation.get("fatal_score_cap_applied", False),
                roles=role_scores,
            )
        )
    errors = payload.get("errors") or []
    if errors:
        lines.extend(
            [
                "",
                "## Incomplete cases",
                "",
                "| Problem | Candidate | Round | Error | Attempts |",
                "| --- | ---: | ---: | --- | ---: |",
            ]
        )
        for error in errors:
            lines.append(
                "| {problem_id} | {candidate} | {round_index} | {error_type} | {attempts} |".format(
                    problem_id=error["problem_id"],
                    candidate=error.get("source_candidate_id", "-"),
                    round_index=error.get("source_round_index", "-"),
                    error_type=error.get("error_type", "unknown"),
                    attempts=len(error.get("failed_attempts") or []),
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    payload = asyncio.run(replay(args))
    atomic_write_json(args.output_dir / "result.json", payload)
    (args.output_dir / "REPORT.md").write_text(
        render_report(payload), encoding="utf-8"
    )
    if payload.get("errors") and not args.allow_case_errors:
        raise SystemExit(
            f"{len(payload['errors'])} verifier replay case(s) were incomplete; "
            "partial artifacts were written"
        )


if __name__ == "__main__":
    main()
