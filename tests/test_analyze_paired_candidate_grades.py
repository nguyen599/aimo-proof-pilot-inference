from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.analyze_paired_candidate_grades import (  # noqa: E402
    analyze_paired_grades,
    build_report,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def manifest_row(
    attempt: int,
    version: str,
    *,
    selected: bool,
    strict_pass: bool,
    internal_score: float,
) -> dict[str, object]:
    return {
        "candidate_id": f"p4-c{attempt:02d}-{version}",
        "run_id": "synthetic",
        "problem_id": "4",
        "attempt_idx": attempt,
        "rank": attempt % 2,
        "planning_strategy": "baseline",
        "proof_version": version,
        "selected_by_pipeline": selected,
        "final_score": internal_score,
        "strict_pass": strict_pass,
        "all_verifiers_passed": strict_pass,
        "final_status": "strict_pass" if strict_pass else "weighted_score_pass",
        "selected_verification_round": 4,
        "rollback_from_round": None,
        "budget_restart_count": 0,
        "refine_budget_restart_count": 0,
        "verifier_call_count": 40,
        "meta_call_count": 16,
        "refinement_count": 4,
        "proof_characters": 1000 + attempt,
    }


def grade_row(candidate_id: str, scores: list[int]) -> dict[str, object]:
    counts = {str(value): scores.count(value) for value in range(8)}
    return {
        "problem_id": candidate_id,
        "attempts": len(scores),
        "attempt_scores": scores,
        "score_distribution": counts,
        "score_out_of_7": sum(scores) / len(scores),
    }


def synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "candidate_manifest.jsonl"
    summary = tmp_path / "summary.json"
    manifest_rows = []
    for attempt, selected, strict_pass, internal_score in (
        (0, True, True, 1.0),
        (1, False, False, 0.6),
    ):
        for version in ("initial", "final"):
            manifest_rows.append(
                manifest_row(
                    attempt,
                    version,
                    selected=selected,
                    strict_pass=strict_pass,
                    internal_score=internal_score,
                )
            )
    write_jsonl(manifest, manifest_rows)
    score_map = {
        "p4-c00-initial": [7, 7],
        "p4-c00-final": [3, 3],
        "p4-c01-initial": [2, 2],
        "p4-c01-final": [6, 6],
    }
    write_json(
        summary,
        {
            "attempts_per_proof": 2,
            "problems": [
                grade_row(candidate_id, scores)
                for candidate_id, scores in score_map.items()
            ],
        },
    )
    return manifest, summary


def test_analyzes_refinement_calibration_and_selection_gates(tmp_path: Path) -> None:
    manifest, summary = synthetic_inputs(tmp_path)

    analysis = analyze_paired_grades(manifest, summary)

    assert analysis["overview"]["paired_candidates"] == 2
    assert analysis["overview"]["initial_mean"] == 4.5
    assert analysis["overview"]["final_mean"] == 4.5
    assert analysis["overview"]["improved"] == 1
    assert analysis["overview"]["degraded"] == 1
    problem = analysis["problems"][0]
    assert problem["initial_unanimous_seven_count"] == 1
    assert problem["final_unanimous_seven_count"] == 0
    assert problem["selected_final_score"] == 3
    assert problem["selected_final_rank"] == 2
    assert problem["strict_pass_false_positives"] == 1
    assert set(problem["diagnosis"]) == {
        "refinement_degraded_credible_proofs",
        "selector_missed_credible_final",
        "internal_strict_pass_false_positive",
    }
    report = build_report(analysis)
    assert "Initial pass / 7-7" in report
    assert "selector_missed_credible_final" in report


def test_rejects_incomplete_initial_final_pair(tmp_path: Path) -> None:
    manifest, summary = synthetic_inputs(tmp_path)
    rows = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    write_jsonl(
        manifest,
        [row for row in rows if row["candidate_id"] != "p4-c01-initial"],
    )

    with pytest.raises(RuntimeError, match="incomplete proof pair"):
        analyze_paired_grades(manifest, summary)


def test_rejects_grader_manifest_mismatch(tmp_path: Path) -> None:
    manifest, summary = synthetic_inputs(tmp_path)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["problems"] = payload["problems"][:-1]
    write_json(summary, payload)

    with pytest.raises(RuntimeError, match="grader/manifest candidate mismatch"):
        analyze_paired_grades(manifest, summary)
