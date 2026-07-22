from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.report_refinement_quality import build_report


class ReportRefinementQualityTests(unittest.TestCase):
    def test_builds_paired_candidate_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            initial = root / "initial.json"
            round_one = root / "round1.json"
            round_two = root / "round2.json"
            output = root / "report"
            initial.write_text(
                json.dumps(
                    {
                        "candidate_grades": [
                            {
                                "problem_id": "4",
                                "candidate_index": 2,
                                "mean_score_out_of_7": 2.0,
                            },
                            {
                                "problem_id": "4",
                                "candidate_index": 3,
                                "mean_score_out_of_7": 5.0,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            round_one.write_text(
                json.dumps(
                    {
                        "problems": [
                            {"problem_id": "p4-c2-r1", "score_out_of_7": 3.0},
                            {"problem_id": "p4-c3-r1", "score_out_of_7": 5.0},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            round_two.write_text(
                json.dumps(
                    {
                        "problems": [
                            {"problem_id": "p4-c02-r2", "score_out_of_7": 5.0},
                            {"problem_id": "p4-c03-r2", "score_out_of_7": 4.0},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = build_report(initial, round_one, round_two, output)

            problem = result["problems"][0]
            self.assertEqual(problem["round_two"]["mean_score_out_of_7"], 4.5)
            self.assertEqual(problem["initial_to_round_two"]["mean_delta"], 1.0)
            self.assertEqual(problem["round_one_to_round_two"]["mean_delta"], 0.5)
            self.assertEqual(problem["round_one_to_round_two"]["improved"], 1)
            self.assertEqual(problem["round_one_to_round_two"]["regressed"], 1)
            self.assertTrue((output / "REPORT.md").is_file())


if __name__ == "__main__":
    unittest.main()
