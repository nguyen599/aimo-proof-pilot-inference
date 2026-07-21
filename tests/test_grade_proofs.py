from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "evaluation" / "harness"
sys.path.insert(0, str(HARNESS))

import grade_proofs  # noqa: E402
from eval_config import load_config  # noqa: E402
from grader import parse_score  # noqa: E402


class GradeProofsTests(unittest.TestCase):
    def test_imo_2025_parquet_has_six_complete_rubrics(self):
        rows = grade_proofs.load_requested_rows(
            REPO / "evaluation" / "data" / "imo_2025.parquet"
        )

        self.assertEqual([row["Problem ID"] for row in rows], list("123456"))
        for row in rows:
            self.assertTrue(row["Problem"].strip())
            self.assertIn("[", row["Grading scheme"])

    def test_arithmetic_mean_does_not_apply_zero_veto(self):
        self.assertEqual(grade_proofs.arithmetic_mean_score([0, 7], 2), 3.5)

    def test_retry_classification_stops_on_permanent_client_errors(self):
        class StatusError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code

        self.assertFalse(grade_proofs.is_retryable_error(StatusError(400)))
        self.assertTrue(grade_proofs.is_retryable_error(StatusError(429)))
        self.assertTrue(grade_proofs.is_retryable_error(StatusError(503)))
        self.assertTrue(grade_proofs.is_retryable_error(RuntimeError("parse")))

    def test_api_key_pool_loads_configured_environment_variables(self):
        grader = {"api_key_env": "FALLBACK", "api_key_envs": ["KEY_1", "KEY_2"]}
        with patch.dict(
            grade_proofs.os.environ,
            {"KEY_1": "secret-1", "KEY_2": "secret-2"},
            clear=True,
        ):
            resolved = grade_proofs.load_api_keys(grader)

        self.assertEqual(
            resolved,
            [("KEY_1", "secret-1"), ("KEY_2", "secret-2")],
        )

    def test_aggregate_includes_complete_score_distribution(self):
        records = [
            {"problem_id": "1", "attempt": 0, "score": 0, "error": None},
            {"problem_id": "1", "attempt": 1, "score": 7, "error": None},
        ]
        problem = grade_proofs.aggregate_grades(records, ["1"], 2)["problems"][0]

        self.assertEqual(problem["score_out_of_7"], 3.5)
        self.assertEqual(
            problem["score_distribution"],
            {"0": 1, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 1},
        )

    def test_strict_grader_parser_preserves_field_order(self):
        parsed = parse_score(
            json.dumps(
                {
                    "findings": ["The central lemma is justified."],
                    "grade": 6,
                    "reasoning": "One minor endpoint case is omitted.",
                }
            )
        )
        self.assertEqual(parsed["grade"], 6)

        with self.assertRaisesRegex(ValueError, "fields/order differ"):
            parse_score(
                json.dumps(
                    {
                        "grade": 6,
                        "findings": ["Correct."],
                        "reasoning": "Correct.",
                    }
                )
            )

    def test_preflight_aligns_parquet_and_selected_proofs_without_api(self):
        rows = grade_proofs.load_requested_rows(
            REPO / "evaluation" / "data" / "imo_2025.parquet"
        )
        with tempfile.TemporaryDirectory() as temporary:
            search_dir = Path(temporary)
            records_path = search_dir / "records.jsonl"
            records_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "problem_id": row["Problem ID"],
                            "final_proof": f"Proof for {row['Problem ID']}",
                        }
                    )
                    + "\n"
                    for row in rows
                )
            )
            config, loaded_rows, selected = grade_proofs.validate_grading_inputs(
                REPO / "config.yaml",
                REPO / "evaluation" / "data" / "imo_2025.parquet",
                search_dir,
            )

        self.assertEqual(config["grader"]["attempts_per_proof"], 32)
        self.assertEqual([row["Problem ID"] for row in loaded_rows], list("123456"))
        self.assertEqual(list(selected), list("123456"))

    def test_runtime_config_accepts_the_optional_grader_section(self):
        grader = load_config(REPO / "config.yaml")["grader"]
        self.assertEqual(grader["base_url"], "https://api.pinference.ai/api/v1")
        self.assertEqual(grader["model"], "openai/gpt-5.6-sol")
        self.assertEqual(grader["api_key_env"], "PRIME_API_KEY")
        self.assertEqual(
            grader["api_key_envs"],
            [
                "PRIME_API_KEY_1",
                "PRIME_API_KEY_2",
                "PRIME_API_KEY_3",
                "PRIME_API_KEY_4",
            ],
        )
        self.assertEqual(grader["reasoning"], "high")
        self.assertEqual(grader["concurrency"], 8)
        self.assertEqual(grader["request_retries"], 6)
        self.assertFalse(grader["prompt_cache_options_enabled"])
        self.assertFalse(grader["zero_veto"])

    def test_p5_grader_uses_eight_workers_per_api_key(self):
        grader = load_config(
            REPO
            / "evaluation"
            / "runs"
            / "imo2025-p245-round0-p64-sft750-20260721"
            / "grader-p5.yaml"
        )["grader"]

        self.assertEqual(len(grader["api_key_envs"]), 3)
        self.assertEqual(grader["concurrency"], 24)
        self.assertEqual(
            grader["concurrency"] // len(grader["api_key_envs"]),
            8,
        )


if __name__ == "__main__":
    unittest.main()
