import csv
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.prepare_imo2026_p45 import load_problems, write_csv, write_jsonl


class PrepareImo2026P45Tests(unittest.TestCase):
    def test_outputs_only_problem_identifiers_and_statements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "imo-2026.jsonl"
            source.write_text(
                "".join(
                    json.dumps(
                        {
                            "problem_idx": index,
                            "problem": f"Problem {index}",
                            "reference_solution": f"Answer {index}",
                            "grading_scheme": [{"secret": index}],
                            "points": 7,
                        }
                    )
                    + "\n"
                    for index in range(1, 7)
                )
            )
            jsonl_output = root / "p45.jsonl"
            csv_output = root / "p45.csv"

            rows = load_problems(source)
            write_jsonl(jsonl_output, rows)
            write_csv(csv_output, rows)

            jsonl_rows = [json.loads(line) for line in jsonl_output.read_text().splitlines()]
            self.assertEqual(
                jsonl_rows,
                [
                    {"problem_idx": "4", "problem": "Problem 4"},
                    {"problem_idx": "5", "problem": "Problem 5"},
                ],
            )
            with csv_output.open(newline="") as input_file:
                csv_rows = list(csv.DictReader(input_file))
            self.assertEqual(
                csv_rows,
                [
                    {"id": "4", "problem": "Problem 4"},
                    {"id": "5", "problem": "Problem 5"},
                ],
            )
            serialized = jsonl_output.read_text() + csv_output.read_text()
            self.assertNotIn("Answer", serialized)
            self.assertNotIn("grading_scheme", serialized)

    def test_rejects_duplicate_or_missing_target_problems(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.jsonl"
            source.write_text(
                json.dumps({"problem_idx": 4, "problem": "first"})
                + "\n"
                + json.dumps({"problem_idx": 4, "problem": "duplicate"})
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "exactly problems 4 and 5"):
                load_problems(source)


if __name__ == "__main__":
    unittest.main()
