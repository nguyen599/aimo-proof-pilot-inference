from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.export_pipeline_candidates import export_candidates  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_candidate(attempt_idx: int, **overrides: object) -> dict[str, object]:
    candidate: dict[str, object] = {
        "attempt_idx": attempt_idx,
        "prompt_family": "opd",
        "planning_strategy": "baseline",
        "generation_mode": "opd_xml",
        "generation_only": False,
        "proof_solution": f"Proof for candidate {attempt_idx}.",
        "proof_generation_output": {
            "parsed": {"proof": f"Initial proof for candidate {attempt_idx}."}
        },
        "proof_generation_outputs": [{}],
        "proof_handoffs": [],
        "proof_verify_output": [{}, {}],
        "proof_meta_verify_output": [{}],
        "proof_refine_output": [{}],
        "proof_refine_attempt_output": [{}],
        "proof_refine_handoffs": [],
        "validated_critiques": [{}],
        "final_score": 0.75,
        "pre_cap_score": 0.875,
        "final_status": "weighted_score_pass",
        "self_score": 1.0,
        "strict_pass": False,
        "all_verifiers_passed": False,
        "selected_verification_round": 1,
        "rollback_from_round": None,
        "verifier_score_summaries": [
            {
                "verifier_index": 0,
                "verifier_role": "generalist",
                "verifier_group": "generalist",
                "verifier_score": 1.0,
                "meta_factor": 0.5,
                "weighted_score": 0.5,
                "evaluation": "must not be exported",
            }
        ],
        "budget_restart_count": 0,
        "refine_budget_restart_count": 0,
    }
    candidate.update(overrides)
    return candidate


class ExportPipelineCandidatesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "runtime"
        self.output_dir = self.root / "grader_input"
        self.rubrics = self.root / "rubrics.jsonl"
        write_json(
            self.run_dir / "manifest.json",
            {
                "run_id": "test-run",
                "world_size": 2,
                "metadata": {"pipelines_per_problem": 4},
            },
        )
        self.rubrics.write_text(
            json.dumps(
                {
                    "Problem ID": "4",
                    "Problem": "Prove a statement.",
                    "Grading scheme": "1. [7 pts] Proof: Complete and correct.",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.write_payload(
            rank=0,
            assigned=[0, 2],
            candidates=[make_candidate(0)],
            failed=[{"attempt_idx": 2, "error": "invalid_generation"}],
        )
        self.write_payload(
            rank=1,
            assigned=[1, 3],
            candidates=[
                make_candidate(1, planning_strategy="exhaustive_transitions"),
                make_candidate(3, rollback_from_round=2),
            ],
        )
        results = self.run_dir / "logs" / "rank_0000" / "results.jsonl"
        results.parent.mkdir(parents=True)
        results.write_text(
            json.dumps({"id": "4", "selected_pipeline": 3}) + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_payload(
        self,
        *,
        rank: int,
        assigned: list[int],
        candidates: list[dict[str, object]],
        failed: list[dict[str, object]] | None = None,
    ) -> None:
        write_json(
            self.run_dir
            / "problems"
            / "0000_deadbeefdeadbeef"
            / f"rank_{rank:04d}.json",
            {
                "run_id": "test-run",
                "rank": rank,
                "world_size": 2,
                "problem_ordinal": 0,
                "problem_id": "4",
                "question_hash": "deadbeefdeadbeef",
                "assigned_attempts": assigned,
                "pipeline_result": {
                    "candidates": candidates,
                    "failed_attempts": failed or [],
                    "skipped_generations": [],
                    "cancelled_count": 0,
                },
            },
        )

    def test_exports_every_final_candidate_with_analysis_metadata(self) -> None:
        summary = export_candidates(
            self.run_dir,
            self.rubrics,
            self.output_dir,
            ["4"],
        )
        self.assertEqual(summary["exported_candidates"], 3)
        self.assertEqual(summary["exported_proof_versions"], 3)
        self.assertEqual(summary["problems"][0]["failed_candidates"], 1)

        records = [
            json.loads(line)
            for line in (self.output_dir / "records.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            [row["problem_id"] for row in records],
            ["p4-c00-final", "p4-c01-final", "p4-c03-final"],
        )
        manifest = [
            json.loads(line)
            for line in (self.output_dir / "candidate_manifest.jsonl")
            .read_text()
            .splitlines()
        ]
        by_attempt = {row["attempt_idx"]: row for row in manifest}
        self.assertEqual(by_attempt[1]["planning_strategy"], "exhaustive_transitions")
        self.assertTrue(by_attempt[3]["selected_by_pipeline"])
        self.assertEqual(by_attempt[0]["verifier_call_count"], 2)
        self.assertEqual(by_attempt[0]["refinement_count"], 1)
        self.assertEqual(by_attempt[0]["pre_cap_score"], 0.875)
        self.assertEqual(
            by_attempt[0]["verifier_score_summaries"][0]["verifier_score"],
            1.0,
        )
        self.assertNotIn(
            "evaluation",
            by_attempt[0]["verifier_score_summaries"][0],
        )

        rubrics = [
            json.loads(line)
            for line in (self.output_dir / "rubrics.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            [row["Problem ID"] for row in rubrics],
            [row["problem_id"] for row in records],
        )

    def test_exports_paired_initial_and_final_proofs(self) -> None:
        summary = export_candidates(
            self.run_dir,
            self.rubrics,
            self.output_dir,
            ["4"],
            ["initial", "final"],
        )
        self.assertEqual(summary["exported_candidates"], 3)
        self.assertEqual(summary["exported_proof_versions"], 6)
        records = [
            json.loads(line)
            for line in (self.output_dir / "records.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            [row["problem_id"] for row in records[:2]],
            ["p4-c00-initial", "p4-c00-final"],
        )
        self.assertEqual(
            {row["proof_version"] for row in records}, {"initial", "final"}
        )

    def test_problem_filter_skips_incomplete_unrequested_problem(self) -> None:
        write_json(
            self.run_dir
            / "problems"
            / "0001_feedfacefeedface"
            / "rank_0000.json",
            {
                "run_id": "test-run",
                "rank": 0,
                "world_size": 2,
                "problem_ordinal": 1,
                "problem_id": "5",
                "question_hash": "feedfacefeedface",
                "assigned_attempts": [0, 2],
                "pipeline_result": {
                    "candidates": [make_candidate(0)],
                    "failed_attempts": [],
                    "skipped_generations": [],
                    "cancelled_count": 0,
                },
            },
        )

        summary = export_candidates(
            self.run_dir,
            self.rubrics,
            self.output_dir,
            ["4"],
        )

        self.assertEqual(summary["problem_ids"], ["4"])
        self.assertEqual(summary["exported_candidates"], 3)

    def test_rejects_missing_rank_payload(self) -> None:
        (
            self.run_dir / "problems" / "0000_deadbeefdeadbeef" / "rank_0001.json"
        ).unlink()
        with self.assertRaisesRegex(RuntimeError, "expected 2 rank payloads"):
            export_candidates(self.run_dir, self.rubrics, self.output_dir)

    def test_rejects_duplicate_candidate_outcome(self) -> None:
        self.write_payload(
            rank=1,
            assigned=[1, 3],
            candidates=[make_candidate(1), make_candidate(1)],
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate outcome"):
            export_candidates(self.run_dir, self.rubrics, self.output_dir)

    def test_rejects_unaccounted_attempt(self) -> None:
        self.write_payload(
            rank=0,
            assigned=[0, 2],
            candidates=[make_candidate(0)],
        )
        with self.assertRaisesRegex(RuntimeError, "accounts for 3 of 4"):
            export_candidates(self.run_dir, self.rubrics, self.output_dir)

    def test_rejects_empty_final_proof(self) -> None:
        self.write_payload(
            rank=0,
            assigned=[0, 2],
            candidates=[make_candidate(0, proof_solution="")],
            failed=[{"attempt_idx": 2, "error": "invalid_generation"}],
        )
        with self.assertRaisesRegex(RuntimeError, "empty final proof"):
            export_candidates(self.run_dir, self.rubrics, self.output_dir)


if __name__ == "__main__":
    unittest.main()
