from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "evaluation" / "harness"
sys.path.insert(0, str(HARNESS))

from run_full_evaluation import audit_generation  # noqa: E402
from run_proof_search import DATASETS, dataset_path, load_requested_rows  # noqa: E402


class EvaluationOrchestratorTests(unittest.TestCase):
    def test_checked_in_manifests_select_exact_datasets_and_problems(self):
        first = REPO / "evaluation/manifests/imo-2025-problem-1.json"
        self.assertEqual(
            json.loads(first.read_text()),
            {"dataset": "imo_2025", "problem_ids": ["1"]},
        )
        rows = load_requested_rows(first)
        self.assertEqual([row["Problem ID"] for row in rows], ["1"])
        self.assertIn("sunny", rows[0]["Problem"])
        self.assertEqual(rows[0]["Points"], 7)
        self.assertEqual(len(rows[0]["Grading scheme"].splitlines()), 7)
        self.assertNotIn("Solution", rows[0])
        self.assertNotIn("Grading guidelines", rows[0])

        second = REPO / "evaluation/manifests/imo-2025-problem-2.json"
        second_rows = load_requested_rows(second)
        self.assertEqual([row["Problem ID"] for row in second_rows], ["2"])
        self.assertIn("circles", second_rows[0]["Problem"])

        aime = REPO / "evaluation/manifests/aime-2026-problem-10.json"
        self.assertEqual(
            json.loads(aime.read_text()),
            {"dataset": "aime_2026", "problem_ids": ["10"]},
        )
        aime_rows = load_requested_rows(aime)
        self.assertEqual([row["Problem ID"] for row in aime_rows], ["10"])
        self.assertEqual(aime_rows[0]["Competition"], "AIME I")
        self.assertEqual(aime_rows[0]["Year"], 2026)
        self.assertEqual(aime_rows[0]["Answer"], 156)
        self.assertIn("AB = 13", aime_rows[0]["Problem"])

    def test_datasets_are_the_pinned_matharena_parquets(self):
        import hashlib

        expected = {
            "imo_2025": "17592c82ae91049ae6215b3cece719fa62d37bcb82f9df16719d436797d03a6f",
            "aime_2026": "d91db799651b4cc1f0734f52792a695c9cc60dac342524b3d8e5b2ff31c3e957",
        }
        self.assertEqual(set(DATASETS), set(expected))
        for dataset, digest in expected.items():
            with self.subTest(dataset=dataset):
                self.assertEqual(
                    hashlib.sha256(DATASETS[dataset].read_bytes()).hexdigest(),
                    digest,
                )
        self.assertEqual(
            dataset_path(REPO / "evaluation/manifests/aime-2026-problem-10.json"),
            DATASETS["aime_2026"],
        )

    def test_orchestrator_exposes_one_config_ids_and_run_id_interface(self):
        source = (HARNESS / "run_full_evaluation.py").read_text()
        self.assertEqual(source.count('parser.add_argument("--'), 3)
        self.assertIn('parser.add_argument("--config"', source)
        self.assertIn('parser.add_argument("--ids-file"', source)
        self.assertIn('parser.add_argument("--run-id"', source)
        self.assertIn('run_root / "grader_models.json"', source)
        self.assertIn('os.environ["EVAL_SERVER_LOG"]', source)
        for stale in ("Basic", "Advanced", "shard", "notebook", "best-of-k"):
            self.assertNotIn(stale, source)

    def test_generation_audit_requires_lossless_calls_and_prompt_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            generation = Path(directory)
            problem_id = "1"
            root = generation / "problems" / problem_id
            (root / "prompts").mkdir(parents=True)
            (root / "proofs").mkdir()
            (generation / "records.jsonl").write_text(
                json.dumps({"problem_id": problem_id, "final_proof": "Proof."}) + "\n"
            )
            (root / "final.json").write_text(
                json.dumps({"problem_id": problem_id, "final_proof": "Proof."})
            )
            prompt_hash = "a" * 64
            (root / "prompts" / f"{prompt_hash}.json").write_text("[]\n")
            (root / "proofs" / "r01-p0000.json").write_text("{}\n")
            (root / "calls.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "round-01/generate/r01-p0000",
                        "prompt_sha256": prompt_hash,
                        "physical_request_count": 2,
                        "error": None,
                    }
                )
                + "\n"
            )
            self.assertEqual(
                audit_generation(generation, [problem_id]),
                {
                    "problem_count": 1,
                    "proof_count": 1,
                    "call_count": 1,
                    "physical_request_count": 2,
                    "failed_call_count": 0,
                },
            )

    def test_superseded_problem_source_is_absent(self):
        self.assertFalse((REPO / "evaluation/data/proofbench_v2.csv").exists())
        self.assertFalse(
            (REPO / "evaluation/manifests/proofbench-basic-001-002.json").exists()
        )


if __name__ == "__main__":
    unittest.main()
