#!/usr/bin/env python3
"""Analyze paired external grades for initial and final pipeline proofs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PROOF_VERSIONS = {"initial", "final"}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in {path}")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else None


def format_number(value: Any, digits: int = 3) -> str:
    number = finite_float(value)
    if number is None:
        return "-"
    if number.is_integer():
        return str(int(number))
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def load_external_grades(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    summary = read_json(path)
    attempts_per_proof = int(summary["attempts_per_proof"])
    if attempts_per_proof < 1:
        raise ValueError("attempts_per_proof must be positive")
    grades: dict[str, dict[str, Any]] = {}
    for row in summary.get("problems") or []:
        candidate_id = str(row["problem_id"])
        if candidate_id in grades:
            raise RuntimeError(f"duplicate grader candidate ID {candidate_id!r}")
        attempt_scores = [int(value) for value in row.get("attempt_scores") or []]
        if len(attempt_scores) != attempts_per_proof:
            raise RuntimeError(
                f"candidate {candidate_id!r} has {len(attempt_scores)} grader "
                f"attempts, expected {attempts_per_proof}"
            )
        score = finite_float(row.get("score_out_of_7"))
        if score is None or not 0 <= score <= 7:
            raise RuntimeError(f"invalid external score for {candidate_id!r}")
        grades[candidate_id] = {
            "score": score,
            "attempt_scores": attempt_scores,
            "score_distribution": dict(row.get("score_distribution") or {}),
        }
    return grades, attempts_per_proof


def load_candidate_pairs(
    manifest_path: Path,
) -> tuple[dict[tuple[str, int], dict[str, dict[str, Any]]], str]:
    rows = read_jsonl(manifest_path)
    by_id: dict[str, dict[str, Any]] = {}
    pairs: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    run_ids: set[str] = set()
    for row in rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id in by_id:
            raise RuntimeError(f"duplicate manifest candidate ID {candidate_id!r}")
        by_id[candidate_id] = row
        proof_version = str(row.get("proof_version"))
        if proof_version not in PROOF_VERSIONS:
            raise RuntimeError(
                f"candidate {candidate_id!r} has invalid proof version "
                f"{proof_version!r}"
            )
        problem_id = str(row["problem_id"])
        attempt_idx = int(row["attempt_idx"])
        key = (problem_id, attempt_idx)
        if proof_version in pairs[key]:
            raise RuntimeError(
                f"duplicate {proof_version} manifest row for problem {problem_id!r} "
                f"attempt {attempt_idx}"
            )
        pairs[key][proof_version] = row
        run_ids.add(str(row["run_id"]))
    if len(run_ids) != 1:
        raise RuntimeError(f"expected one run ID in manifest, found {sorted(run_ids)}")
    incomplete = {
        key: sorted(PROOF_VERSIONS - set(versions))
        for key, versions in pairs.items()
        if set(versions) != PROOF_VERSIONS
    }
    if incomplete:
        first_key, missing = next(iter(sorted(incomplete.items())))
        raise RuntimeError(
            f"incomplete proof pair for problem {first_key[0]!r} attempt "
            f"{first_key[1]}: missing {missing}"
        )
    return dict(pairs), run_ids.pop()


def external_rank(score: float, all_scores: list[float]) -> int:
    return 1 + sum(other > score for other in all_scores)


def candidate_diagnosis(
    *,
    initial_best: float,
    final_best: float,
    selected_final: float | None,
    pass_threshold: float,
    degraded_initial_passes: int,
    strict_false_positives: int,
) -> list[str]:
    findings: list[str] = []
    if initial_best < pass_threshold:
        findings.append("initial_generation_ceiling")
    if initial_best >= pass_threshold and final_best < pass_threshold:
        findings.append("refinement_lost_all_credible_proofs")
    elif degraded_initial_passes:
        findings.append("refinement_degraded_credible_proofs")
    if final_best >= pass_threshold and (
        selected_final is None or selected_final < pass_threshold
    ):
        findings.append("selector_missed_credible_final")
    if strict_false_positives:
        findings.append("internal_strict_pass_false_positive")
    return findings or ["no_gate_failure_detected"]


def analyze_paired_grades(
    manifest_path: Path,
    grader_summary_path: Path,
    *,
    pass_threshold: float = 5.0,
) -> dict[str, Any]:
    if not 0 <= pass_threshold <= 7:
        raise ValueError("pass threshold must be between 0 and 7")
    pairs, run_id = load_candidate_pairs(manifest_path)
    grades, attempts_per_proof = load_external_grades(grader_summary_path)
    expected_ids = {
        str(row["candidate_id"])
        for versions in pairs.values()
        for row in versions.values()
    }
    missing_grades = sorted(expected_ids - set(grades))
    extra_grades = sorted(set(grades) - expected_ids)
    if missing_grades or extra_grades:
        raise RuntimeError(
            "grader/manifest candidate mismatch: "
            f"missing={missing_grades[:5]} extra={extra_grades[:5]}"
        )

    candidate_rows: list[dict[str, Any]] = []
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (problem_id, attempt_idx), versions in sorted(pairs.items()):
        initial_manifest = versions["initial"]
        final_manifest = versions["final"]
        stable_fields = (
            "run_id",
            "problem_id",
            "attempt_idx",
            "rank",
            "planning_strategy",
            "selected_by_pipeline",
        )
        for field in stable_fields:
            if initial_manifest.get(field) != final_manifest.get(field):
                raise RuntimeError(
                    f"paired manifest field {field!r} differs for problem "
                    f"{problem_id!r} attempt {attempt_idx}"
                )
        initial_grade = grades[str(initial_manifest["candidate_id"])]
        final_grade = grades[str(final_manifest["candidate_id"])]
        initial_score = float(initial_grade["score"])
        final_score = float(final_grade["score"])
        delta = final_score - initial_score
        row = {
            "run_id": run_id,
            "problem_id": problem_id,
            "attempt_idx": attempt_idx,
            "rank": int(final_manifest["rank"]),
            "planning_strategy": str(final_manifest["planning_strategy"]),
            "selected_by_pipeline": bool(final_manifest["selected_by_pipeline"]),
            "internal_final_score": finite_float(final_manifest.get("final_score")),
            "strict_pass": bool(final_manifest.get("strict_pass")),
            "all_verifiers_passed": bool(final_manifest.get("all_verifiers_passed")),
            "final_status": final_manifest.get("final_status"),
            "selected_verification_round": final_manifest.get(
                "selected_verification_round"
            ),
            "rollback_from_round": final_manifest.get("rollback_from_round"),
            "budget_restart_count": int(
                final_manifest.get("budget_restart_count") or 0
            ),
            "refine_budget_restart_count": int(
                final_manifest.get("refine_budget_restart_count") or 0
            ),
            "verifier_call_count": int(final_manifest.get("verifier_call_count") or 0),
            "meta_call_count": int(final_manifest.get("meta_call_count") or 0),
            "refinement_count": int(final_manifest.get("refinement_count") or 0),
            "initial_proof_characters": int(initial_manifest["proof_characters"]),
            "final_proof_characters": int(final_manifest["proof_characters"]),
            "initial_external_score": initial_score,
            "final_external_score": final_score,
            "external_delta": delta,
            "improved": delta > 1e-9,
            "degraded": delta < -1e-9,
            "unchanged": abs(delta) <= 1e-9,
            "initial_attempt_scores": initial_grade["attempt_scores"],
            "final_attempt_scores": final_grade["attempt_scores"],
            "initial_any_seven": 7 in initial_grade["attempt_scores"],
            "final_any_seven": 7 in final_grade["attempt_scores"],
            "initial_unanimous_seven": all(
                score == 7 for score in initial_grade["attempt_scores"]
            ),
            "final_unanimous_seven": all(
                score == 7 for score in final_grade["attempt_scores"]
            ),
            "initial_external_pass": initial_score >= pass_threshold,
            "final_external_pass": final_score >= pass_threshold,
        }
        candidate_rows.append(row)
        by_problem[problem_id].append(row)

    problem_rows: list[dict[str, Any]] = []
    for problem_id, rows in sorted(by_problem.items()):
        initial_scores = [row["initial_external_score"] for row in rows]
        final_scores = [row["final_external_score"] for row in rows]
        initial_best = max(initial_scores)
        final_best = max(final_scores)
        selected_rows = [row for row in rows if row["selected_by_pipeline"]]
        if len(selected_rows) != 1:
            raise RuntimeError(
                f"problem {problem_id!r} has {len(selected_rows)} selected candidates"
            )
        selected = selected_rows[0]
        selected["external_final_rank"] = external_rank(
            selected["final_external_score"], final_scores
        )
        selected["external_initial_rank"] = external_rank(
            selected["initial_external_score"], initial_scores
        )
        initial_pass_rows = [row for row in rows if row["initial_external_pass"]]
        final_pass_rows = [row for row in rows if row["final_external_pass"]]
        degraded_initial_passes = sum(
            row["initial_external_pass"] and row["degraded"] for row in rows
        )
        strict_rows = [row for row in rows if row["strict_pass"]]
        strict_true_positives = sum(row["final_external_pass"] for row in strict_rows)
        strict_false_positives = len(strict_rows) - strict_true_positives
        internal_external_pairs = [
            (row["internal_final_score"], row["final_external_score"] / 7.0)
            for row in rows
            if row["internal_final_score"] is not None
        ]
        problem_rows.append(
            {
                "problem_id": problem_id,
                "candidates": len(rows),
                "initial_mean": mean(initial_scores),
                "initial_best": initial_best,
                "initial_best_attempts": [
                    row["attempt_idx"]
                    for row in rows
                    if row["initial_external_score"] == initial_best
                ],
                "final_mean": mean(final_scores),
                "final_best": final_best,
                "final_best_attempts": [
                    row["attempt_idx"]
                    for row in rows
                    if row["final_external_score"] == final_best
                ],
                "mean_delta": mean(row["external_delta"] for row in rows),
                "improved": sum(row["improved"] for row in rows),
                "degraded": sum(row["degraded"] for row in rows),
                "unchanged": sum(row["unchanged"] for row in rows),
                "initial_pass_count": len(initial_pass_rows),
                "final_pass_count": len(final_pass_rows),
                "degraded_initial_pass_count": degraded_initial_passes,
                "initial_any_seven_count": sum(
                    row["initial_any_seven"] for row in rows
                ),
                "final_any_seven_count": sum(row["final_any_seven"] for row in rows),
                "initial_unanimous_seven_count": sum(
                    row["initial_unanimous_seven"] for row in rows
                ),
                "final_unanimous_seven_count": sum(
                    row["final_unanimous_seven"] for row in rows
                ),
                "strict_pass_count": len(strict_rows),
                "strict_pass_precision": (
                    strict_true_positives / len(strict_rows) if strict_rows else None
                ),
                "strict_pass_false_positives": strict_false_positives,
                "internal_external_pearson": pearson(
                    [float(item[0]) for item in internal_external_pairs],
                    [item[1] for item in internal_external_pairs],
                ),
                "selected_attempt": selected["attempt_idx"],
                "selected_initial_score": selected["initial_external_score"],
                "selected_final_score": selected["final_external_score"],
                "selected_delta": selected["external_delta"],
                "selected_final_rank": selected["external_final_rank"],
                "selected_final_tie_count": sum(
                    score == selected["final_external_score"] for score in final_scores
                ),
                "diagnosis": candidate_diagnosis(
                    initial_best=initial_best,
                    final_best=final_best,
                    selected_final=selected["final_external_score"],
                    pass_threshold=pass_threshold,
                    degraded_initial_passes=degraded_initial_passes,
                    strict_false_positives=strict_false_positives,
                ),
            }
        )

    overall_initial = [row["initial_external_score"] for row in candidate_rows]
    overall_final = [row["final_external_score"] for row in candidate_rows]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "grader_summary_path": str(grader_summary_path),
        "attempts_per_proof": attempts_per_proof,
        "pass_threshold": pass_threshold,
        "overview": {
            "problems": len(problem_rows),
            "paired_candidates": len(candidate_rows),
            "initial_mean": mean(overall_initial),
            "final_mean": mean(overall_final),
            "mean_delta": mean(row["external_delta"] for row in candidate_rows),
            "improved": sum(row["improved"] for row in candidate_rows),
            "degraded": sum(row["degraded"] for row in candidate_rows),
            "unchanged": sum(row["unchanged"] for row in candidate_rows),
            "initial_pass_count": sum(
                row["initial_external_pass"] for row in candidate_rows
            ),
            "final_pass_count": sum(
                row["final_external_pass"] for row in candidate_rows
            ),
            "initial_unanimous_seven_count": sum(
                row["initial_unanimous_seven"] for row in candidate_rows
            ),
            "final_unanimous_seven_count": sum(
                row["final_unanimous_seven"] for row in candidate_rows
            ),
        },
        "problems": problem_rows,
        "candidates": candidate_rows,
    }


def build_report(analysis: dict[str, Any]) -> str:
    overview = analysis["overview"]
    lines = [
        "# Paired proof-quality report",
        "",
        f"Run: `{analysis['run_id']}`",
        "",
        f"External pass threshold: `{format_number(analysis['pass_threshold'])}/7`; "
        f"grader calls per proof: `{analysis['attempts_per_proof']}`.",
        "",
        "## Stage gates",
        "",
        "| Problem | N | Initial mean / best | Final mean / best | Mean delta | Better / worse / same | Initial pass / 7-7 | Final pass / 7-7 | Selected final (rank) | Diagnosis |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for problem in analysis["problems"]:
        rendered = {
            **problem,
            "initial_mean": format_number(problem["initial_mean"]),
            "initial_best": format_number(problem["initial_best"]),
            "final_mean": format_number(problem["final_mean"]),
            "final_best": format_number(problem["final_best"]),
            "mean_delta": format_number(problem["mean_delta"]),
            "selected_final_score": format_number(problem["selected_final_score"]),
            "diagnosis": ", ".join(problem["diagnosis"]),
        }
        lines.append(
            "| {problem_id} | {candidates} | {initial_mean} / {initial_best} | "
            "{final_mean} / {final_best} | {mean_delta} | "
            "{improved} / {degraded} / {unchanged} | "
            "{initial_pass_count} / {initial_unanimous_seven_count} | "
            "{final_pass_count} / {final_unanimous_seven_count} | "
            "{selected_final_score} (#{selected_final_rank}) | {diagnosis} |".format(
                **rendered
            )
        )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- Paired candidates: {overview['paired_candidates']} across "
            f"{overview['problems']} problems.",
            f"- Mean external score: {format_number(overview['initial_mean'])} initially "
            f"and {format_number(overview['final_mean'])} finally "
            f"(delta {format_number(overview['mean_delta'])}).",
            f"- Candidate transitions: {overview['improved']} improved, "
            f"{overview['degraded']} degraded, {overview['unchanged']} unchanged.",
            f"- Credible proofs at threshold: {overview['initial_pass_count']} initially, "
            f"{overview['final_pass_count']} finally.",
            f"- Unanimous 7/7 proofs: {overview['initial_unanimous_seven_count']} initially, "
            f"{overview['final_unanimous_seven_count']} finally.",
            "",
            "## Calibration and selection",
            "",
            "| Problem | Strict passes | Strict precision | False positives | Internal/external r | Selected initial -> final |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for problem in analysis["problems"]:
        lines.append(
            "| {problem_id} | {strict_pass_count} | {precision} | "
            "{strict_pass_false_positives} | {correlation} | {initial} -> {final} |".format(
                **problem,
                precision=format_number(problem["strict_pass_precision"]),
                correlation=format_number(problem["internal_external_pearson"]),
                initial=format_number(problem["selected_initial_score"]),
                final=format_number(problem["selected_final_score"]),
            )
        )
    lines.extend(
        [
            "",
            "A `7-7` proof received score 7 from every independent grader call. "
            "This is the strongest candidate-pool evidence in the report; the lower "
            "pass threshold is retained to expose near-complete proofs that refinement "
            "might still repair.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--grader-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pass-threshold", type=float, default=5.0)
    args = parser.parse_args()
    analysis = analyze_paired_grades(
        args.candidate_manifest.resolve(),
        args.grader_summary.resolve(),
        pass_threshold=args.pass_threshold,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in analysis.items() if key != "candidates"}
    write_json(args.output_dir / "analysis.json", compact)
    write_csv(args.output_dir / "paired_candidates.csv", analysis["candidates"])
    (args.output_dir / "REPORT.md").write_text(build_report(analysis), encoding="utf-8")
    print(args.output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
