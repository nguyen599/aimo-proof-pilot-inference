"""Compare externally graded initial, round-one, and round-two proofs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


STAGE_ID = re.compile(
    r"^p(?P<problem>\d+)-c(?P<candidate>\d+)-r(?P<round>\d+)$"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_initial(path: Path) -> dict[tuple[str, int], float]:
    rows = load_json(path).get("candidate_grades", [])
    result: dict[tuple[str, int], float] = {}
    for row in rows:
        key = (str(row["problem_id"]), int(row["candidate_index"]))
        if key in result:
            raise RuntimeError(f"duplicate initial grade for {key}")
        result[key] = float(row["mean_score_out_of_7"])
    return result


def load_stage(path: Path, expected_round: int) -> dict[tuple[str, int], float]:
    result: dict[tuple[str, int], float] = {}
    for row in load_json(path).get("problems", []):
        match = STAGE_ID.fullmatch(str(row["problem_id"]))
        if match is None:
            raise RuntimeError(f"invalid stage problem ID {row['problem_id']!r}")
        round_index = int(match.group("round"))
        if round_index != expected_round:
            raise RuntimeError(
                f"expected round {expected_round}, found round {round_index}"
            )
        key = (match.group("problem"), int(match.group("candidate")))
        if key in result:
            raise RuntimeError(f"duplicate round-{expected_round} grade for {key}")
        result[key] = float(row["score_out_of_7"])
    return result


def score_label(score: float) -> str:
    return str(int(score)) if score.is_integer() else f"{score:.1f}"


def summarize_scores(scores: list[float]) -> dict[str, Any]:
    counts = Counter(scores)
    return {
        "candidate_count": len(scores),
        "mean_score_out_of_7": mean(scores) if scores else None,
        "best_score_out_of_7": max(scores) if scores else None,
        "score_distribution": {
            score_label(index / 2): counts.get(index / 2, 0)
            for index in range(15)
        },
    }


def paired_summary(
    before: dict[tuple[str, int], float],
    after: dict[tuple[str, int], float],
    problem_id: str,
) -> dict[str, Any]:
    keys = sorted(
        key for key in after if key in before and key[0] == problem_id
    )
    before_scores = [before[key] for key in keys]
    after_scores = [after[key] for key in keys]
    deltas = [after[key] - before[key] for key in keys]
    return {
        "paired_candidates": len(keys),
        "before_mean_score_out_of_7": mean(before_scores) if keys else None,
        "after_mean_score_out_of_7": mean(after_scores) if keys else None,
        "mean_delta": mean(deltas) if keys else None,
        "improved": sum(delta > 0 for delta in deltas),
        "tied": sum(delta == 0 for delta in deltas),
        "regressed": sum(delta < 0 for delta in deltas),
    }


def build_report(
    initial_summary: Path,
    round_one_summary: Path,
    round_two_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    initial = load_initial(initial_summary)
    round_one = load_stage(round_one_summary, 1)
    round_two = load_stage(round_two_summary, 2)
    problem_ids = sorted({problem_id for problem_id, _ in round_two})

    problems = []
    for problem_id in problem_ids:
        scores = [
            score for (source_problem, _), score in round_two.items()
            if source_problem == problem_id
        ]
        problems.append(
            {
                "problem_id": problem_id,
                "round_two": summarize_scores(scores),
                "initial_to_round_two": paired_summary(
                    initial, round_two, problem_id
                ),
                "round_one_to_round_two": paired_summary(
                    round_one, round_two, problem_id
                ),
            }
        )

    candidate_rows = []
    for (problem_id, candidate_index), round_two_score in sorted(round_two.items()):
        key = (problem_id, candidate_index)
        initial_score = initial.get(key)
        round_one_score = round_one.get(key)
        candidate_rows.append(
            {
                "problem_id": problem_id,
                "candidate_index": candidate_index,
                "initial_score_out_of_7": initial_score,
                "round_one_score_out_of_7": round_one_score,
                "round_two_score_out_of_7": round_two_score,
                "initial_to_round_two_delta": (
                    round_two_score - initial_score
                    if initial_score is not None
                    else None
                ),
                "round_one_to_round_two_delta": (
                    round_two_score - round_one_score
                    if round_one_score is not None
                    else None
                ),
            }
        )

    summary = {
        "schema_version": 1,
        "methodology": (
            "Every stage score is the arithmetic mean of two independent "
            "GPT-5.6 rubric grades. Paired deltas compare the same candidate."
        ),
        "problems": problems,
        "candidates": candidate_rows,
    }
    write_json(output_dir / "summary.json", summary)

    aggregate_rows = []
    for problem in problems:
        r2 = problem["round_two"]
        i2 = problem["initial_to_round_two"]
        r12 = problem["round_one_to_round_two"]
        aggregate_rows.append(
            "| {problem} | {n} | {avg:.3f} | {best:.1f} | {i_n} | {i_delta} | "
            "{r1_n} | {r1_delta} | {g}/{t}/{l} |".format(
                problem=problem["problem_id"],
                n=r2["candidate_count"],
                avg=r2["mean_score_out_of_7"],
                best=r2["best_score_out_of_7"],
                i_n=i2["paired_candidates"],
                i_delta=(
                    f"{i2['mean_delta']:+.3f}"
                    if i2["mean_delta"] is not None
                    else "n/a"
                ),
                r1_n=r12["paired_candidates"],
                r1_delta=(
                    f"{r12['mean_delta']:+.3f}"
                    if r12["mean_delta"] is not None
                    else "n/a"
                ),
                g=r12["improved"],
                t=r12["tied"],
                l=r12["regressed"],
            )
        )

    candidate_table = []
    for row in candidate_rows:
        def display(value: float | None, *, signed: bool = False) -> str:
            if value is None:
                return "n/a"
            return f"{value:+.1f}" if signed else f"{value:.1f}"

        candidate_table.append(
            "| P{problem} | {candidate} | {initial} | {r1} | {r2} | {d0} | {d1} |".format(
                problem=row["problem_id"],
                candidate=row["candidate_index"],
                initial=display(row["initial_score_out_of_7"]),
                r1=display(row["round_one_score_out_of_7"]),
                r2=display(row["round_two_score_out_of_7"]),
                d0=display(row["initial_to_round_two_delta"], signed=True),
                d1=display(row["round_one_to_round_two_delta"], signed=True),
            )
        )

    report = "\n".join(
        [
            "# Intermediate round-two proof quality",
            "",
            summary["methodology"],
            "",
            "| Problem | R2 n | R2 avg / 7 | R2 best / 7 | Initial pairs | Initial->R2 | R1 pairs | R1->R2 | R1->R2 gain/tie/loss |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *aggregate_rows,
            "",
            "## Candidate-level paired scores",
            "",
            "| Problem | Candidate | Initial | R1 | R2 | Initial->R2 | R1->R2 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            *candidate_table,
            "",
            "Full half-point score distributions are in `summary.json`.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-summary", required=True, type=Path)
    parser.add_argument("--round-one-summary", required=True, type=Path)
    parser.add_argument("--round-two-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build_report(
        args.initial_summary,
        args.round_one_summary,
        args.round_two_summary,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
