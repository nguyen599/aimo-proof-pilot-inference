"""Replay proof selection over an exported pipeline candidate pool.

This keeps expensive proof generation, verification, and refinement fixed while
allowing selector strategies to be compared on exactly the same candidates.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.harness_vllm import run as harness


@dataclass
class ReplayProblem:
    problem_id: str
    question: str
    grading_scheme: str
    candidates: list[dict[str, Any]]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def _unique_by_id(
    rows: list[dict[str, Any]],
    *,
    id_field: str,
    source: Path,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row.get(id_field) or "").strip()
        if not identifier:
            raise ValueError(f"missing {id_field!r} in {source}")
        if identifier in indexed:
            raise ValueError(f"duplicate {id_field} {identifier!r} in {source}")
        indexed[identifier] = row
    return indexed


def load_candidate_export(candidate_dir: Path) -> list[ReplayProblem]:
    records_path = candidate_dir / "records.jsonl"
    manifest_path = candidate_dir / "candidate_manifest.jsonl"
    rubrics_path = candidate_dir / "rubrics.jsonl"
    records = _unique_by_id(
        read_jsonl(records_path),
        id_field="problem_id",
        source=records_path,
    )
    manifests = _unique_by_id(
        read_jsonl(manifest_path),
        id_field="candidate_id",
        source=manifest_path,
    )
    rubrics = _unique_by_id(
        read_jsonl(rubrics_path),
        id_field="Problem ID",
        source=rubrics_path,
    )

    grouped: dict[str, list[dict[str, Any]]] = {}
    problem_metadata: dict[str, tuple[str, str]] = {}
    for candidate_id, record in records.items():
        manifest = manifests.get(candidate_id)
        rubric = rubrics.get(candidate_id)
        if manifest is None:
            raise ValueError(f"missing manifest row for {candidate_id!r}")
        if rubric is None:
            raise ValueError(f"missing rubric row for {candidate_id!r}")
        proof_version = str(
            record.get("proof_version") or manifest.get("proof_version") or "final"
        )
        if proof_version != "final":
            continue
        source_problem_id = str(
            record.get("source_problem_id") or manifest.get("problem_id") or ""
        ).strip()
        if not source_problem_id:
            raise ValueError(f"missing source problem ID for {candidate_id!r}")
        question = str(rubric.get("Problem") or "").strip()
        grading_scheme = str(rubric.get("Grading scheme") or "").strip()
        if not question or not grading_scheme:
            raise ValueError(f"incomplete rubric for {candidate_id!r}")
        previous_metadata = problem_metadata.setdefault(
            source_problem_id,
            (question, grading_scheme),
        )
        if previous_metadata != (question, grading_scheme):
            raise ValueError(
                f"inconsistent rubric rows for source problem {source_problem_id!r}"
            )

        final_score = manifest.get("final_score")
        candidate = {
            "candidate_id": candidate_id,
            "proof_solution": str(record.get("final_proof") or "").strip(),
            "attempt_idx": int(manifest["attempt_idx"]),
            "final_score": final_score,
            "pre_cap_score": manifest.get("pre_cap_score", final_score),
            "final_status": manifest.get("final_status"),
            "strict_pass": bool(manifest.get("strict_pass", False)),
            "selected_by_pipeline": bool(
                manifest.get("selected_by_pipeline", False)
            ),
            "planning_strategy": manifest.get("planning_strategy"),
            "prompt_family": manifest.get("prompt_family"),
            "selected_verification_round": manifest.get(
                "selected_verification_round"
            ),
        }
        if not candidate["proof_solution"]:
            raise ValueError(f"empty final proof for {candidate_id!r}")
        grouped.setdefault(source_problem_id, []).append(candidate)

    problems: list[ReplayProblem] = []
    for problem_id, candidates in grouped.items():
        candidates.sort(key=lambda candidate: int(candidate["attempt_idx"]))
        attempts = [int(candidate["attempt_idx"]) for candidate in candidates]
        if len(attempts) != len(set(attempts)):
            raise ValueError(f"duplicate final candidate attempt for problem {problem_id}")
        question, grading_scheme = problem_metadata[problem_id]
        problems.append(
            ReplayProblem(
                problem_id=problem_id,
                question=question,
                grading_scheme=grading_scheme,
                candidates=candidates,
            )
        )
    if not problems:
        raise ValueError(f"no final candidates found in {candidate_dir}")
    return sorted(problems, key=lambda problem: problem.problem_id)


async def replay_selectors(
    problems: list[ReplayProblem],
    scheduler: Any,
    selector_config: Any,
) -> list[dict[str, Any]]:
    async def select_problem(problem: ReplayProblem) -> dict[str, Any]:
        progress = harness.PipelineProgress(
            enabled=False,
            problem_id=problem.problem_id,
            pipeline_count=len(problem.candidates),
        )
        selected_index, selector_output = await harness.select_best_candidate(
            problem.question,
            problem.candidates,
            scheduler,
            selector_config,
            progress,
        )
        selected = problem.candidates[selected_index]
        return {
            "problem_id": problem.problem_id,
            "question": problem.question,
            "grading_scheme": problem.grading_scheme,
            "candidate_count": len(problem.candidates),
            "selected_index": selected_index,
            "selected_candidate_id": selected["candidate_id"],
            "selected_attempt_idx": selected["attempt_idx"],
            "selected_final_score": selected.get("final_score"),
            "selected_pre_cap_score": selected.get("pre_cap_score"),
            "selected_was_original": selected.get("selected_by_pipeline", False),
            "final_proof": selected["proof_solution"],
            "selector_output": selector_output,
        }

    return await asyncio.gather(*(select_problem(problem) for problem in problems))


def write_replay_outputs(
    output_dir: Path,
    results: list[dict[str, Any]],
    *,
    candidate_dir: Path,
    selector_config: Any,
) -> dict[str, Any]:
    records = [
        {
            "problem_id": result["problem_id"],
            "final_proof": result["final_proof"],
            "source_candidate_id": result["selected_candidate_id"],
            "source_attempt_idx": result["selected_attempt_idx"],
            "selector_mode": selector_config.selector_mode,
        }
        for result in results
    ]
    rubrics = [
        {
            "Problem ID": result["problem_id"],
            "Problem": result["question"],
            "Grading scheme": result["grading_scheme"],
        }
        for result in results
    ]
    selector_rows = [
        {key: value for key, value in result.items() if key != "final_proof"}
        for result in results
    ]
    summary = {
        "schema_version": 1,
        "candidate_dir": str(candidate_dir.resolve()),
        "selector_mode": selector_config.selector_mode,
        "problem_count": len(results),
        "selected_candidates": {
            result["problem_id"]: result["selected_candidate_id"]
            for result in results
        },
        "changed_from_original": sum(
            not bool(result["selected_was_original"]) for result in results
        ),
        "selector_config": {
            key: getattr(selector_config, key)
            for key in (
                "selector_max_candidate_chars",
                "selector_thinking_budget_tokens",
                "selection_temperature",
                "selector_tournament_group_size",
                "selector_tournament_rounds",
                "selector_tournament_max_candidates",
                "selector_tournament_threshold",
                "selector_tournament_force_wide_pool",
                "selector_score_window",
                "selector_vote_count",
            )
        },
    }
    write_jsonl(output_dir / "records.jsonl", records)
    write_jsonl(output_dir / "rubrics.jsonl", rubrics)
    write_jsonl(output_dir / "selector_results.jsonl", selector_rows)
    write_json(output_dir / "summary.json", summary)
    return summary


def build_selector_config(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        selector_mode=args.selector_mode,
        selector_max_candidate_chars=max(1_000, args.selector_max_candidate_chars),
        selector_thinking_budget_tokens=max(
            0, args.selector_thinking_budget_tokens
        ),
        selector_thinking_budget_force_text=(
            harness.CFG.selector_thinking_budget_force_text
        ),
        selection_temperature=args.selection_temperature,
        selector_tournament_group_size=max(2, args.selector_tournament_group_size),
        selector_tournament_rounds=max(1, args.selector_tournament_rounds),
        selector_tournament_max_candidates=max(
            2, args.selector_tournament_max_candidates
        ),
        selector_tournament_threshold=args.selector_tournament_threshold,
        selector_tournament_force_wide_pool=args.selector_tournament_force_wide_pool,
        selector_score_window=args.selector_score_window,
        selector_vote_count=max(1, args.selector_vote_count),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-url", action="append", default=[])
    parser.add_argument("--api-key", default=harness.DEFAULT_API_KEY)
    parser.add_argument("--model", default=harness.DEFAULT_SERVED_MODEL_NAME)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument(
        "--selector-mode",
        choices=("score", "llm", "llm_tournament", "llm_stratified_tournament"),
        default="llm_stratified_tournament",
    )
    parser.add_argument("--selector-max-candidate-chars", type=int, default=32_000)
    parser.add_argument("--selector-max-new-tokens", type=int, default=58_100)
    parser.add_argument(
        "--selector-thinking-budget-tokens",
        type=int,
        default=56_000,
    )
    parser.add_argument("--selection-temperature", type=float, default=0.3)
    parser.add_argument("--selector-tournament-group-size", type=int, default=4)
    parser.add_argument("--selector-tournament-rounds", type=int, default=64)
    parser.add_argument("--selector-tournament-max-candidates", type=int, default=10)
    parser.add_argument("--selector-tournament-threshold", type=float, default=0.5)
    parser.add_argument(
        "--selector-tournament-force-wide-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--selector-score-window", type=float, default=0.2)
    parser.add_argument("--selector-vote-count", type=int, default=16)
    parser.add_argument("--max-concurrent-requests", type=int, default=64)
    parser.add_argument("--request-worker-count", type=int, default=64)
    parser.add_argument("--request-timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--stream-responses",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--context-length", type=int, default=131_072)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.selector_mode != "score" and not args.mock_llm:
        if not args.base_url:
            parser.error("--base-url is required for LLM selector modes")
        if args.tokenizer_path is None:
            parser.error("--tokenizer-path is required for LLM selector modes")
    if not 0.0 < args.selector_tournament_threshold <= 1.0:
        parser.error("--selector-tournament-threshold must be in (0, 1]")
    if not 0.0 <= args.selector_score_window < 1.0:
        parser.error("--selector-score-window must be in [0, 1)")
    if args.selector_max_new_tokens < 1:
        parser.error("--selector-max-new-tokens must be positive")
    if args.selector_thinking_budget_tokens < 0:
        parser.error("--selector-thinking-budget-tokens cannot be negative")
    if (
        args.selector_thinking_budget_tokens > 0
        and args.selector_thinking_budget_tokens >= args.selector_max_new_tokens
    ):
        parser.error(
            "--selector-thinking-budget-tokens must be below "
            "--selector-max-new-tokens"
        )
    return args


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    problems = load_candidate_export(args.candidate_dir)
    selector_config = build_selector_config(args)
    tokenizer = None
    if not args.mock_llm and args.selector_mode != "score":
        tokenizer = harness.load_counting_tokenizer(args.tokenizer_path)
    scheduler = harness.ChatScheduler(
        base_urls=args.base_url,
        api_key=args.api_key,
        model=args.model,
        sampling=harness.SamplingConfig(
            max_new_tokens=args.selector_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_new_tokens=0,
            min_p=args.min_p,
        ),
        max_concurrent_requests=args.max_concurrent_requests,
        mock_llm=args.mock_llm or args.selector_mode == "score",
        stage_max_new_tokens={"selector": args.selector_max_new_tokens},
        request_timeout_seconds=args.request_timeout_seconds,
        request_worker_count=args.request_worker_count,
        stream_responses=args.stream_responses,
        context_length=args.context_length,
        tokenizer=tokenizer,
        llm_call_logdir=args.output_dir / "llm_calls",
    )
    try:
        results = await replay_selectors(problems, scheduler, selector_config)
    finally:
        scheduler.close()
    return write_replay_outputs(
        args.output_dir,
        results,
        candidate_dir=args.candidate_dir,
        selector_config=selector_config,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    summary = asyncio.run(async_main(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
