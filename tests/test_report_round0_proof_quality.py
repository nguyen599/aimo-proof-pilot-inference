from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.report_round0_proof_quality import finalize, prepare


VALID_OUTPUT = (
    "<solution>A complete proof.</solution>"
    "<self_evaluation>The argument is complete.</self_evaluation>"
    "<score>1</score>"
)


def write_call(path: Path, *, budget_reached: bool, output: str = VALID_OUTPUT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "thinking_budget_applied": budget_reached,
    }
    path.write_text(
        "\n".join(
            [
                "stage: proof_generation",
                "detail: candidate=0 round=0",
                "prompt_tokens: 10",
                "max_tokens: 20",
                "",
                "===== INPUT PROMPT =====",
                "prompt",
                "",
                "===== OUTPUT =====",
                "finish_reason: stop",
                f"usage: {json.dumps(usage)}",
                "",
                output,
                "",
            ]
        ),
        encoding="utf-8",
    )


class RoundZeroProofQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.run_dir = self.root / "run"
        self.rubrics = self.root / "rubrics.jsonl"
        rubric_rows = [
            {
                "Problem ID": problem_id,
                "Problem": f"Problem {problem_id}",
                "Grading scheme": "1. [7 pts] Complete proof: Correct.",
            }
            for problem_id in ("2", "4", "5")
        ]
        self.rubrics.write_text(
            "".join(json.dumps(row) + "\n" for row in rubric_rows),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def populate_calls(self) -> None:
        for problem_id in ("2", "4", "5"):
            for candidate_index in range(2):
                write_call(
                    self.run_dir
                    / "logs"
                    / f"rank_000{candidate_index}"
                    / "llm_calls"
                    / problem_id
                    / f"cand_{candidate_index}_proof_gen_r0.txt",
                    budget_reached=candidate_index == 1,
                )

    def test_prepare_and_finalize_exactly_two_grades(self) -> None:
        self.populate_calls()
        prepared = prepare(
            self.run_dir,
            self.rubrics,
            ["2", "4", "5"],
            expected_candidates=2,
        )
        self.assertEqual(prepared["total_candidates"], 6)
        self.assertEqual(prepared["grader_candidate_count"], 3)
        for problem in prepared["problems"]:
            self.assertEqual(problem["thinking_budget_not_reached_count"], 1)
            self.assertEqual(problem["thinking_budget_not_reached_rate"], 0.5)

        grading_dir = self.run_dir / "grading"
        records = []
        for problem_id, scores in (("2", [6, 7]), ("4", [2, 4]), ("5", [0, 1])):
            for attempt, score in enumerate(scores):
                records.append(
                    {
                        "problem_id": f"p{problem_id}-c00",
                        "attempt": attempt,
                        "score": score,
                        "error": None,
                    }
                )
        grading_dir.mkdir(parents=True)
        (grading_dir / "records.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        finalized = finalize(self.run_dir, grading_dir)
        self.assertEqual(finalized["graded_candidates"], 3)
        self.assertEqual(finalized["grader_failures"], 0)
        by_problem = {row["problem_id"]: row for row in finalized["problems"]}
        self.assertEqual(by_problem["2"]["average_score_out_of_7"], 6.5)
        self.assertEqual(
            by_problem["2"]["grader_call_score_distribution"]["6"], 1
        )
        self.assertEqual(
            by_problem["2"]["candidate_mean_score_distribution"]["6.5"], 1
        )
        self.assertTrue((self.run_dir / "REPORT.md").is_file())

    def test_prepare_rejects_missing_candidate(self) -> None:
        self.populate_calls()
        missing = (
            self.run_dir
            / "logs"
            / "rank_0001"
            / "llm_calls"
            / "5"
            / "cand_1_proof_gen_r0.txt"
        )
        missing.unlink()
        with self.assertRaisesRegex(RuntimeError, "call set differs"):
            prepare(
                self.run_dir,
                self.rubrics,
                ["2", "4", "5"],
                expected_candidates=2,
            )

    def test_prepare_accepts_standalone_llm_call_directory(self) -> None:
        for candidate_index in range(2):
            write_call(
                self.run_dir
                / "llm_calls"
                / "4"
                / f"cand_{candidate_index}_proof_gen_r0.txt",
                budget_reached=candidate_index == 1,
            )

        prepared = prepare(
            self.run_dir,
            self.rubrics,
            ["4"],
            expected_candidates=2,
        )

        self.assertEqual(prepared["total_candidates"], 2)
        self.assertEqual(prepared["grader_candidate_count"], 1)
        self.assertEqual(
            prepared["problems"][0]["thinking_budget_not_reached_count"],
            1,
        )

    def test_prepare_accepts_source_jsonl_and_labels_imo_2026(self) -> None:
        source_rubrics = self.root / "imo-2026.jsonl"
        source_rubrics.write_text(
            json.dumps(
                {
                    "problem_idx": "4",
                    "problem": "A clean problem statement.",
                    "reference_solution": "Must not enter grader inputs.",
                    "points": 7,
                    "grading_scheme": [
                        {
                            "title": "Complete proof",
                            "points": 7,
                            "desc": "The proof is correct and complete.",
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        for candidate_index in range(2):
            write_call(
                self.run_dir
                / "logs"
                / "rank_0000"
                / "llm_calls"
                / "4"
                / f"cand_{candidate_index}_proof_gen_r0.txt",
                budget_reached=candidate_index == 1,
            )

        prepared = prepare(
            self.run_dir,
            source_rubrics,
            ["4"],
            expected_candidates=2,
            contest_label="IMO 2026",
        )

        self.assertEqual(prepared["methodology"]["contest_label"], "IMO 2026")
        grader_rubric = json.loads(
            (self.run_dir / "grader_input" / "rubrics.jsonl")
            .read_text(encoding="utf-8")
            .strip()
        )
        self.assertEqual(grader_rubric["Problem ID"], "p4-c00")
        self.assertNotIn("reference_solution", grader_rubric)
        self.assertNotIn("Must not enter", json.dumps(grader_rubric))

    def test_prepare_reports_text_independent_adaptive_cycle(self) -> None:
        self.rubrics.write_text(
            json.dumps(
                {
                    "Problem ID": "4",
                    "Problem": (
                        "The sequence satisfies a_{n+1}=f(a_n). Determine all "
                        "possible initial values."
                    ),
                    "Grading scheme": "1. [7 pts] Complete proof: Correct.",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        for candidate_index in range(12):
            write_call(
                self.run_dir
                / "logs"
                / f"rank_000{candidate_index % 2}"
                / "llm_calls"
                / "4"
                / f"cand_{candidate_index}_proof_gen_r0.txt",
                budget_reached=False,
            )

        prepared = prepare(
            self.run_dir,
            self.rubrics,
            ["4"],
            expected_candidates=12,
            strategy_portfolio="adaptive",
        )

        strategies = {
            row["planning_strategy"]: row
            for row in prepared["problems"][0]["strategies"]
        }
        self.assertEqual(strategies["baseline"]["total_candidates"], 8)
        self.assertEqual(
            strategies["adversarial_quantifiers"]["total_candidates"],
            1,
        )
        self.assertEqual(
            strategies["exhaustive_transitions"]["total_candidates"],
            1,
        )
        self.assertEqual(strategies["counterexample_audit"]["total_candidates"], 1)
        self.assertEqual(
            strategies["independent_reformulation"]["total_candidates"],
            1,
        )
        self.assertTrue(
            all(
                row["grader_candidate_count"] == row["total_candidates"]
                for row in strategies.values()
            )
        )

    def test_finalize_rejects_missing_second_grade(self) -> None:
        self.populate_calls()
        prepare(
            self.run_dir,
            self.rubrics,
            ["2", "4", "5"],
            expected_candidates=2,
        )
        grading_dir = self.run_dir / "grading"
        grading_dir.mkdir(parents=True)
        (grading_dir / "records.jsonl").write_text(
            json.dumps(
                {
                    "problem_id": "p2-c00",
                    "attempt": 0,
                    "score": 7,
                    "error": None,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "exactly attempts 0 and 1"):
            finalize(self.run_dir, grading_dir)


if __name__ == "__main__":
    unittest.main()
