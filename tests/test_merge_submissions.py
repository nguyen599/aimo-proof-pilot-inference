import csv
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "evaluation" / "harness"
sys.path.insert(0, str(HARNESS))

from merge_submissions import merge_submissions


def write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["id", "proof"])
        writer.writerows(rows)


class MergeSubmissionsTests(unittest.TestCase):
    def test_merges_round_robin_shards_in_original_input_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "test.csv"
            input_path.write_text(
                "id,problem\n"
                "0,Problem 0\n"
                "1,Problem 1\n"
                "2,Problem 2\n"
                "3,Problem 3\n",
                encoding="utf-8",
            )
            even_path = root / "even.csv"
            odd_path = root / "odd.csv"
            output_path = root / "submission.csv"
            write_csv(even_path, [("0", "Proof 0"), ("2", "Proof 2")])
            write_csv(odd_path, [("1", "Proof 1"), ("3", "Proof 3")])

            merge_submissions(
                input_path,
                [odd_path, even_path],
                output_path,
            )

            with output_path.open(newline="", encoding="utf-8") as source:
                self.assertEqual(
                    list(csv.DictReader(source)),
                    [
                        {"id": "0", "proof": "Proof 0"},
                        {"id": "1", "proof": "Proof 1"},
                        {"id": "2", "proof": "Proof 2"},
                        {"id": "3", "proof": "Proof 3"},
                    ],
                )

    def test_rejects_missing_or_overlapping_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "test.csv"
            input_path.write_text(
                "id,problem\n0,Problem 0\n1,Problem 1\n",
                encoding="utf-8",
            )
            first_path = root / "first.csv"
            second_path = root / "second.csv"
            write_csv(first_path, [("0", "Proof 0")])
            write_csv(second_path, [("0", "Another proof 0")])

            with self.assertRaisesRegex(ValueError, "overlap"):
                merge_submissions(
                    input_path,
                    [first_path, second_path],
                    root / "duplicate.csv",
                )
            with self.assertRaisesRegex(ValueError, "missing"):
                merge_submissions(
                    input_path,
                    [first_path],
                    root / "missing.csv",
                )


if __name__ == "__main__":
    unittest.main()
