from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.harness_vllm import run  # noqa: E402


class InputLoadingTests(unittest.TestCase):
    def test_loads_problem_records_from_jsonl(self):
        rows = [
            {
                "problem_idx": "1",
                "problem": "Prove the first claim.",
                "reference_solution": "Reference one.",
            },
            {
                "problem_idx": "2",
                "problem": "Prove the second claim.",
                "reference_solution": "Reference two.",
            },
        ]

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "imo-2026.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            records = run.load_input_records(str(path), "auto")

        self.assertEqual([record.id for record in records], ["1", "2"])
        self.assertEqual(
            [record.question for record in records],
            ["Prove the first claim.", "Prove the second claim."],
        )
        self.assertEqual({record.question_column for record in records}, {"problem"})


if __name__ == "__main__":
    unittest.main()
