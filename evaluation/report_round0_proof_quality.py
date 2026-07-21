"""Prepare and summarize a round-zero proof-generation quality benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.harness_vllm.run import (
    PROOF_GENERATION_STRATEGY_PORTFOLIOS,
    parse_generation_response,
    resolve_proof_generation_strategy,
)
from evaluation.harness_vllm.thinking_handoff import (
    parse_saved_proof_generation_call,
)


PROOF_CALL_NAME = re.compile(r"^cand_(?P<candidate>\d+)_proof_gen_r0\.txt$")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def render_grading_scheme(items: Any) -> str:
    rendered: list[str] = []
    total_points = 0
    for index, item in enumerate(list(items), start=1):
        if not isinstance(item, dict):
            raise ValueError(f"grading scheme item {index} is not a mapping")
        missing = {"title", "points", "desc"} - set(item)
        if missing:
            raise ValueError(
                f"grading scheme item {index} missing fields: {sorted(missing)}"
            )
        points = int(item["points"])
        total_points += points
        rendered.append(f"{index}. [{points} pts] {item['title']}: {item['desc']}")
    if total_points != 7:
        raise ValueError(f"grading scheme totals {total_points}, expected 7")
    return "\n".join(rendered)


def load_rubrics(path: Path, problem_ids: list[str]) -> dict[str, dict[str, str]]:
    if path.suffix.lower() in {".parquet", ".pq"}:
        source_rows = pd.read_parquet(path).to_dict(orient="records")
        rows = [
            {
                "Problem ID": str(row["problem_idx"]),
                "Problem": str(row["problem"]),
                "Grading scheme": render_grading_scheme(row["grading_scheme"]),
            }
            for row in source_rows
        ]
    elif path.suffix.lower() in {".json", ".jsonl"}:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raise ValueError(f"unsupported rubric file: {path}")

    by_id = {str(row["Problem ID"]): row for row in rows}
    missing = [problem_id for problem_id in problem_ids if problem_id not in by_id]
    if missing:
        raise ValueError(f"rubrics missing problem IDs: {missing}")
    return {problem_id: by_id[problem_id] for problem_id in problem_ids}


def discover_calls(
    run_dir: Path,
    problem_ids: list[str],
    expected_candidates: int,
) -> dict[tuple[str, int], Path]:
    calls: dict[tuple[str, int], Path] = {}
    for path in sorted(run_dir.glob("logs/rank*/llm_calls/*/cand_*_proof_gen_r0.txt")):
        match = PROOF_CALL_NAME.fullmatch(path.name)
        if match is None:
            continue
        problem_id = path.parent.name
        if problem_id not in problem_ids:
            continue
        key = (problem_id, int(match.group("candidate")))
        if key in calls:
            raise RuntimeError(
                f"duplicate round-zero call for problem {problem_id} candidate "
                f"{key[1]}: {calls[key]} and {path}"
            )
        calls[key] = path

    expected = {
        (problem_id, candidate)
        for problem_id in problem_ids
        for candidate in range(expected_candidates)
    }
    missing = sorted(expected - set(calls))
    extra = sorted(set(calls) - expected)
    if missing or extra:
        raise RuntimeError(
            "round-zero call set differs from expectation: "
            f"missing={missing[:20]} (total={len(missing)}), "
            f"extra={extra[:20]} (total={len(extra)})"
        )
    return calls


def prepare(
    run_dir: Path,
    rubrics_file: Path,
    problem_ids: list[str],
    expected_candidates: int,
    strategy_portfolio: str = "baseline",
) -> dict[str, Any]:
    if strategy_portfolio not in PROOF_GENERATION_STRATEGY_PORTFOLIOS:
        raise ValueError(
            "strategy portfolio must be one of "
            + ", ".join(PROOF_GENERATION_STRATEGY_PORTFOLIOS)
        )
    calls = discover_calls(run_dir, problem_ids, expected_candidates)
    rubrics = load_rubrics(rubrics_file, problem_ids)
    strategy_cfg = SimpleNamespace(
        proof_generation_strategy_portfolio=strategy_portfolio
    )
    generation_rows: list[dict[str, Any]] = []
    grader_rows: list[dict[str, Any]] = []
    grader_rubrics: list[dict[str, Any]] = []
    eligible_manifest: list[dict[str, Any]] = []

    for problem_id in problem_ids:
        for candidate_index in range(expected_candidates):
            path = calls[(problem_id, candidate_index)]
            saved = parse_saved_proof_generation_call(
                path,
                allow_unintervened=True,
            )
            if saved.stage != "proof_generation":
                raise RuntimeError(f"unexpected stage {saved.stage!r} in {path}")
            if "thinking_budget_applied" not in saved.usage:
                raise RuntimeError(f"missing thinking_budget_applied in {path}")
            budget_reached = saved.usage["thinking_budget_applied"]
            if type(budget_reached) is not bool:
                raise RuntimeError(
                    f"thinking_budget_applied must be boolean in {path}"
                )

            parsed = parse_generation_response(
                saved.output_text,
                require_self_evaluation=True,
            )
            structurally_complete = bool(parsed["is_valid_candidate_response"])
            eligible = bool(not budget_reached and structurally_complete)
            candidate_id = f"p{problem_id}-c{candidate_index:02d}"
            planning_strategy = resolve_proof_generation_strategy(
                candidate_index,
                strategy_cfg,
                rubrics[problem_id]["Problem"],
            )
            relative_path = str(path.relative_to(run_dir))
            if budget_reached:
                rejection_reason = "thinking_budget_reached"
            elif not structurally_complete:
                rejection_reason = "invalid_candidate_response"
            else:
                rejection_reason = None

            row = {
                "candidate_id": candidate_id,
                "problem_id": problem_id,
                "candidate_index": candidate_index,
                "planning_strategy": planning_strategy,
                "source_log": relative_path,
                "source_sha256": sha256(path),
                "stage": saved.stage,
                "detail": saved.detail,
                "prompt_tokens": saved.prompt_tokens,
                "max_tokens": saved.max_tokens,
                "finish_reason": saved.finish_reason,
                "usage": saved.usage,
                "thinking_budget_reached": budget_reached,
                "structurally_complete": structurally_complete,
                "eligible_for_grading": eligible,
                "rejection_reason": rejection_reason,
                "proof_characters": len(str(parsed["proof"])),
                "self_score": parsed["self_score"],
            }
            generation_rows.append(row)
            if not eligible:
                continue

            proof = str(parsed["proof"]).strip()
            grader_rows.append(
                {
                    "problem_id": candidate_id,
                    "final_proof": proof,
                    "source_problem_id": problem_id,
                    "candidate_index": candidate_index,
                    "planning_strategy": planning_strategy,
                    "source_log": relative_path,
                    "source_sha256": row["source_sha256"],
                }
            )
            source_rubric = rubrics[problem_id]
            grader_rubrics.append(
                {
                    "Problem ID": candidate_id,
                    "Problem": source_rubric["Problem"],
                    "Grading scheme": source_rubric["Grading scheme"],
                }
            )
            eligible_manifest.append(
                {
                    "candidate_id": candidate_id,
                    "problem_id": problem_id,
                    "candidate_index": candidate_index,
                    "planning_strategy": planning_strategy,
                    "source_log": relative_path,
                    "source_sha256": row["source_sha256"],
                }
            )

    problem_summaries = []
    for problem_id in problem_ids:
        rows = [row for row in generation_rows if row["problem_id"] == problem_id]
        budget_count = sum(row["thinking_budget_reached"] for row in rows)
        no_budget_count = len(rows) - budget_count
        eligible_count = sum(row["eligible_for_grading"] for row in rows)
        invalid_no_budget = sum(
            not row["thinking_budget_reached"] and not row["structurally_complete"]
            for row in rows
        )
        strategy_summaries = []
        for strategy in dict.fromkeys(row["planning_strategy"] for row in rows):
            strategy_rows = [
                row for row in rows if row["planning_strategy"] == strategy
            ]
            strategy_summaries.append(
                {
                    "planning_strategy": strategy,
                    "total_candidates": len(strategy_rows),
                    "thinking_budget_not_reached_count": sum(
                        not row["thinking_budget_reached"] for row in strategy_rows
                    ),
                    "structurally_complete_nonbudget_count": sum(
                        row["eligible_for_grading"] for row in strategy_rows
                    ),
                    "grader_candidate_count": sum(
                        row["eligible_for_grading"] for row in strategy_rows
                    ),
                }
            )
        problem_summaries.append(
            {
                "problem_id": problem_id,
                "total_candidates": len(rows),
                "thinking_budget_reached_count": budget_count,
                "thinking_budget_reached_rate": rate(budget_count, len(rows)),
                "thinking_budget_not_reached_count": no_budget_count,
                "thinking_budget_not_reached_rate": rate(no_budget_count, len(rows)),
                "structurally_complete_nonbudget_count": eligible_count,
                "structurally_complete_nonbudget_rate": rate(
                    eligible_count, len(rows)
                ),
                "invalid_nonbudget_count": invalid_no_budget,
                "grader_candidate_count": eligible_count,
                "strategies": strategy_summaries,
            }
        )

    summary = {
        "schema_version": 1,
        "methodology": {
            "round": 0,
            "expected_candidates_per_problem": expected_candidates,
            "proof_generation_strategy_portfolio": strategy_portfolio,
            "thinking_budget_reached_source": "usage.thinking_budget_applied",
            "complete_response_definition": (
                "parse_generation_response(require_self_evaluation=True) accepted "
                "the final visible XML solution, self_evaluation, and score"
            ),
            "grading_eligibility": (
                "thinking budget not reached and structurally complete response"
            ),
        },
        "problem_ids": problem_ids,
        "problems": problem_summaries,
        "total_candidates": len(generation_rows),
        "grader_candidate_count": len(grader_rows),
    }
    write_jsonl(run_dir / "analysis" / "generation_records.jsonl", generation_rows)
    write_json(run_dir / "analysis" / "generation_summary.json", summary)
    write_jsonl(run_dir / "grader_input" / "records.jsonl", grader_rows)
    write_jsonl(run_dir / "grader_input" / "rubrics.jsonl", grader_rubrics)
    write_jsonl(
        run_dir / "grader_input" / "eligible_manifest.jsonl",
        eligible_manifest,
    )
    return summary


def format_mean(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.1f}"


def finalize(run_dir: Path, grading_dir: Path) -> dict[str, Any]:
    generation_summary = json.loads(
        (run_dir / "analysis" / "generation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    eligible = [
        json.loads(line)
        for line in (run_dir / "grader_input" / "eligible_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    grade_records = [
        json.loads(line)
        for line in (grading_dir / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    expected_ids = {row["candidate_id"] for row in eligible}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in grade_records:
        candidate_id = str(record["problem_id"])
        if candidate_id not in expected_ids:
            raise RuntimeError(f"unexpected graded candidate {candidate_id}")
        grouped[candidate_id].append(record)

    candidate_grades: list[dict[str, Any]] = []
    manifest_by_id = {row["candidate_id"]: row for row in eligible}
    for candidate_id in sorted(expected_ids):
        attempts = sorted(grouped[candidate_id], key=lambda row: row["attempt"])
        if [row["attempt"] for row in attempts] != [0, 1]:
            raise RuntimeError(
                f"candidate {candidate_id} does not have exactly attempts 0 and 1"
            )
        if any(row.get("error") is not None for row in attempts):
            raise RuntimeError(f"candidate {candidate_id} has a failed grade")
        scores = [int(row["score"]) for row in attempts]
        if any(score < 0 or score > 7 for score in scores):
            raise RuntimeError(f"candidate {candidate_id} has invalid scores {scores}")
        metadata = manifest_by_id[candidate_id]
        candidate_grades.append(
            {
                **metadata,
                "attempt_scores": scores,
                "mean_score_out_of_7": mean(scores),
            }
        )

    problem_results = []
    for generation_problem in generation_summary["problems"]:
        problem_id = generation_problem["problem_id"]
        candidates = [
            row for row in candidate_grades if row["problem_id"] == problem_id
        ]
        call_scores = [score for row in candidates for score in row["attempt_scores"]]
        call_counts = Counter(call_scores)
        candidate_counts = Counter(row["mean_score_out_of_7"] for row in candidates)
        strategy_results = []
        for generation_strategy in generation_problem.get("strategies", []):
            strategy = generation_strategy["planning_strategy"]
            strategy_candidates = [
                row
                for row in candidates
                if row.get("planning_strategy", "baseline") == strategy
            ]
            strategy_call_scores = [
                score
                for row in strategy_candidates
                for score in row["attempt_scores"]
            ]
            strategy_results.append(
                {
                    **generation_strategy,
                    "grader_calls": len(strategy_call_scores),
                    "average_score_out_of_7": (
                        mean(
                            row["mean_score_out_of_7"]
                            for row in strategy_candidates
                        )
                        if strategy_candidates
                        else None
                    ),
                    "best_mean_score_out_of_7": (
                        max(
                            row["mean_score_out_of_7"]
                            for row in strategy_candidates
                        )
                        if strategy_candidates
                        else None
                    ),
                    "candidate_count_at_or_above": {
                        str(threshold): sum(
                            row["mean_score_out_of_7"] >= threshold
                            for row in strategy_candidates
                        )
                        for threshold in (4, 5, 6, 7)
                    },
                    "grader_call_score_distribution": {
                        str(score): Counter(strategy_call_scores).get(score, 0)
                        for score in range(8)
                    },
                }
            )
        problem_results.append(
            {
                **generation_problem,
                "grader_calls": len(call_scores),
                "grader_failures": 0,
                "average_score_out_of_7": (
                    mean(row["mean_score_out_of_7"] for row in candidates)
                    if candidates
                    else None
                ),
                "best_mean_score_out_of_7": (
                    max(row["mean_score_out_of_7"] for row in candidates)
                    if candidates
                    else None
                ),
                "candidate_count_at_or_above": {
                    str(threshold): sum(
                        row["mean_score_out_of_7"] >= threshold
                        for row in candidates
                    )
                    for threshold in (4, 5, 6, 7)
                },
                "grader_call_score_distribution": {
                    str(score): call_counts.get(score, 0) for score in range(8)
                },
                "candidate_mean_score_distribution": {
                    format_mean(score / 2): candidate_counts.get(score / 2, 0)
                    for score in range(15)
                },
                "strategies": strategy_results,
            }
        )

    all_scores = [row["mean_score_out_of_7"] for row in candidate_grades]
    summary = {
        "schema_version": 1,
        "grader_model": "openai/gpt-5.6-sol",
        "grader_attempts_per_candidate": 2,
        "grader_failures": 0,
        "graded_candidates": len(candidate_grades),
        "average_score_out_of_7": mean(all_scores) if all_scores else None,
        "problems": problem_results,
        "candidate_grades": candidate_grades,
    }
    write_json(run_dir / "analysis" / "final_summary.json", summary)

    table_rows = []
    for row in problem_results:
        average = row["average_score_out_of_7"]
        table_rows.append(
            "| {problem_id} | {total_candidates} | {no_budget} | {no_budget_rate:.1%} "
            "| {complete} | {graded} | {average} | {best} |".format(
                problem_id=row["problem_id"],
                total_candidates=row["total_candidates"],
                no_budget=row["thinking_budget_not_reached_count"],
                no_budget_rate=row["thinking_budget_not_reached_rate"],
                complete=row["structurally_complete_nonbudget_count"],
                graded=row["grader_candidate_count"],
                average=f"{average:.3f}" if average is not None else "n/a",
                best=(
                    f"{row['best_mean_score_out_of_7']:.1f}"
                    if row["best_mean_score_out_of_7"] is not None
                    else "n/a"
                ),
            )
        )
    distribution_rows = []
    for row in problem_results:
        counts = row["grader_call_score_distribution"]
        distribution_rows.append(
            "| {problem_id} | {counts} |".format(
                problem_id=row["problem_id"],
                counts=" | ".join(str(counts[str(score)]) for score in range(8)),
            )
        )
    strategy_rows = []
    for problem in problem_results:
        for strategy in problem.get("strategies", []):
            average = strategy["average_score_out_of_7"]
            best = strategy["best_mean_score_out_of_7"]
            strategy_rows.append(
                "| {problem} | {strategy} | {total} | {complete} | {graded} | "
                "{average} | {best} | {ge6} | {ge7} |".format(
                    problem=problem["problem_id"],
                    strategy=strategy["planning_strategy"],
                    total=strategy["total_candidates"],
                    complete=strategy["structurally_complete_nonbudget_count"],
                    graded=strategy["grader_candidate_count"],
                    average=f"{average:.3f}" if average is not None else "n/a",
                    best=f"{best:.1f}" if best is not None else "n/a",
                    ge6=strategy["candidate_count_at_or_above"]["6"],
                    ge7=strategy["candidate_count_at_or_above"]["7"],
                )
            )
    overall = summary["average_score_out_of_7"]
    expected_candidates = generation_summary["methodology"][
        "expected_candidates_per_problem"
    ]
    problem_label = ", ".join(generation_summary["problem_ids"])
    strategy_portfolio = generation_summary["methodology"].get(
        "proof_generation_strategy_portfolio",
        "baseline",
    )
    report = "\n".join(
        [
            "# IMO 2025 round-0 proof-generation quality",
            "",
            f"Problems {problem_label} each have {expected_candidates} independent "
            f"round-0 proof calls using the `{strategy_portfolio}` planning portfolio. "
            "A candidate is graded only when `usage.thinking_budget_applied` is "
            "false and the production parser accepts its final visible XML response. "
            "This is a structural completeness check; GPT-5.6 performs the "
            "mathematical grading.",
            "",
            "| Problem | Total | No budget | No-budget rate | Parseable complete | Graded | Avg / 7 | Best / 7 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            *table_rows,
            "",
            f"Overall graded-candidate average: {overall:.3f}/7."
            if overall is not None
            else "No candidate was eligible for grading.",
            "",
            "## GPT-5.6 call-score distribution",
            "",
            "Two independent grader calls were made for every eligible candidate.",
            "",
            "| Problem | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *distribution_rows,
            "",
            "The exact per-candidate two-call scores and half-point mean "
            "distribution are in `analysis/final_summary.json`.",
            "",
            "## Planning-strategy outcomes",
            "",
            "| Problem | Strategy | Total | Complete | Graded | Avg / 7 | Best / 7 | >=6 | 7 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            *strategy_rows,
            "",
        ]
    )
    (run_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return summary


def parse_problem_ids(values: list[str]) -> list[str]:
    normalized = [str(value).strip() for value in values]
    if any(not value for value in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError("problem IDs must be nonempty and unique")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--run-dir", required=True, type=Path)
    prepare_parser.add_argument("--rubrics-file", required=True, type=Path)
    prepare_parser.add_argument("--problem-ids", nargs="+", default=["2", "4", "5"])
    prepare_parser.add_argument("--expected-candidates", type=int, default=64)
    prepare_parser.add_argument(
        "--strategy-portfolio",
        choices=PROOF_GENERATION_STRATEGY_PORTFOLIOS,
        default="baseline",
    )
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--run-dir", required=True, type=Path)
    finalize_parser.add_argument("--grading-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.command == "prepare":
        if args.expected_candidates <= 0:
            raise ValueError("expected candidates must be positive")
        result = prepare(
            args.run_dir,
            args.rubrics_file,
            parse_problem_ids(args.problem_ids),
            args.expected_candidates,
            args.strategy_portfolio,
        )
    else:
        result = finalize(args.run_dir, args.grading_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
