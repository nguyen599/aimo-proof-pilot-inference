#!/usr/bin/env python3
"""Replay selected proofs through the current verifier stack without generation."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from transformers import AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from evaluation.harness_vllm.run import (
        CFG,
        PROMPT_FAMILY_OPD,
        ChatScheduler,
        PipelineProgress,
        SamplingConfig,
        detect_id_column,
        detect_question_column,
        read_input_table,
        run_verification_round,
    )
except ModuleNotFoundError as exc:
    if exc.name != "evaluation":
        raise
    from harness_vllm.run import (  # type: ignore[no-redef]
        CFG,
        PROMPT_FAMILY_OPD,
        ChatScheduler,
        PipelineProgress,
        SamplingConfig,
        detect_id_column,
        detect_question_column,
        read_input_table,
        run_verification_round,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--proofs-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--problem-id", action="append", default=[])
    parser.add_argument("--served-model-name", default="proof-model")
    parser.add_argument("--api-key", default="vllm-local")
    parser.add_argument("--verify-n", type=int, default=4)
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


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    frame = read_input_table(args.input_path)
    question_column = detect_question_column(frame, "auto")
    id_column = detect_id_column(frame)
    questions = {
        str(row[id_column] if id_column else index + 1): str(row[question_column])
        for index, row in frame.iterrows()
    }
    requested = {str(value) for value in args.problem_id}
    cases = []
    for record in read_jsonl(args.proofs_path):
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
        refine_review_n=2,
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

    async def evaluate(case: dict[str, Any]) -> dict[str, Any]:
        progress = PipelineProgress(False, case["problem_id"], 1)
        try:
            failed_attempts = []
            for attempt_number in range(max(1, int(args.max_attempts))):
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
                        "",
                        int(case["problem_id"]) * 100 + attempt_number,
                        0,
                        scheduler,
                        cfg,
                        prompt_family=PROMPT_FAMILY_OPD,
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
                raise RuntimeError(
                    f"problem {case['problem_id']} has incomplete verifier scores "
                    f"after {max(1, int(args.max_attempts))} attempts: "
                    f"{failed_attempts}"
                )
        finally:
            progress.close()
        return {
            **case,
            "replay_attempt": len(failed_attempts) + 1,
            "failed_attempts": failed_attempts,
            "verifier_results": verifier_results,
            "verifier_outputs": verifier_outputs,
            "meta_results_by_verifier": meta_results,
            "meta_outputs": meta_outputs,
            "aggregation": aggregation,
        }

    try:
        results = await asyncio.gather(*(evaluate(case) for case in cases))
    finally:
        scheduler.close()
    return {
        "schema_version": 1,
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
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Verifier audit replay",
        "",
        "| Problem | Old score | New score | Status | Fatal cap | Role scores |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for result in payload["results"]:
        aggregation = result["aggregation"]
        role_scores = ", ".join(
            f"{item.get('verifier_role')}={item.get('verifier_score')}"
            for item in aggregation.get("verifier_score_summaries") or []
        )
        lines.append(
            "| {problem_id} | {old} | {new} | {status} | {cap} | {roles} |".format(
                problem_id=result["problem_id"],
                old=result.get("old_internal_score"),
                new=aggregation.get("final_score"),
                status=aggregation.get("final_status"),
                cap=aggregation.get("fatal_score_cap_applied", False),
                roles=role_scores,
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


if __name__ == "__main__":
    main()
