from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.export_refinement_candidates import export_refinements


VALID_OUTPUT = (
    "<solution>A complete proof.</solution>"
    "<self_evaluation>Every step is justified.</self_evaluation>"
    "<score>1</score>"
)


def write_call(
    path: Path,
    *,
    stage: str,
    budget_reached: bool,
    output: str = VALID_OUTPUT,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "thinking_budget_applied": budget_reached,
    }
    path.write_text(
        "\n".join(
            [
                f"stage: {stage}",
                "detail: candidate=0 round=2",
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


class ExportRefinementCandidatesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.run_dir = self.root / "run"
        self.output_dir = self.root / "output"
        self.rubrics = self.root / "rubrics.jsonl"
        self.rubrics.write_text(
            "".join(
                json.dumps(
                    {
                        "Problem ID": problem_id,
                        "Problem": f"Problem {problem_id}",
                        "Grading scheme": "1. [7 pts] Complete proof: Correct.",
                    }
                )
                + "\n"
                for problem_id in ("4", "5")
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def path(
        self,
        problem_id: str,
        candidate: int,
        round_index: int,
        *,
        finalize: bool = False,
        rank: int = 0,
    ) -> Path:
        suffix = "_finalize" if finalize else ""
        return (
            self.run_dir
            / "logs"
            / f"rank_{rank:04d}"
            / "llm_calls"
            / problem_id
            / f"cand_{candidate}_proof_refine{suffix}_r{round_index}.txt"
        )

    def test_exports_direct_round_two_call(self) -> None:
        write_call(
            self.path("4", 12, 2),
            stage="proof_refine",
            budget_reached=False,
        )
        summary = export_refinements(
            self.run_dir,
            self.rubrics,
            self.output_dir,
            ["4", "5"],
            min_round=2,
            max_round=2,
        )
        self.assertEqual(summary["grader_candidate_count"], 1)
        record = json.loads(
            (self.output_dir / "records.jsonl").read_text().strip()
        )
        self.assertEqual(record["problem_id"], "p4-c12-r2")
        self.assertEqual(record["source_stage"], "proof_refine")
        self.assertEqual(record["final_proof"], "A complete proof.")

    def test_budgeted_direct_uses_valid_finalize(self) -> None:
        write_call(
            self.path("5", 9, 1),
            stage="proof_refine",
            budget_reached=True,
        )
        write_call(
            self.path("5", 9, 1, finalize=True),
            stage="proof_refine_finalize",
            budget_reached=False,
            output=(
                "<solution>A finalized proof.</solution>"
                "<self_evaluation>Complete.</self_evaluation><score>1</score>"
            ),
        )
        export_refinements(
            self.run_dir,
            self.rubrics,
            self.output_dir,
            ["4", "5"],
            min_round=1,
        )
        record = json.loads(
            (self.output_dir / "records.jsonl").read_text().strip()
        )
        self.assertEqual(record["source_stage"], "proof_refine_finalize")
        self.assertEqual(record["final_proof"], "A finalized proof.")
        manifest = json.loads(
            (self.output_dir / "manifest.jsonl").read_text().strip()
        )
        self.assertEqual(manifest["selected_source"], "finalize")
        self.assertTrue(manifest["sources"]["direct"]["thinking_budget_reached"])

    def test_incomplete_direct_uses_valid_finalize(self) -> None:
        direct = self.path("5", 11, 2)
        direct.parent.mkdir(parents=True, exist_ok=True)
        direct.write_text("stage: proof_refine\n", encoding="utf-8")
        write_call(
            self.path("5", 11, 2, finalize=True),
            stage="proof_refine_finalize",
            budget_reached=False,
        )
        export_refinements(
            self.run_dir,
            self.rubrics,
            self.output_dir,
            ["4", "5"],
            min_round=2,
        )
        record = json.loads(
            (self.output_dir / "records.jsonl").read_text().strip()
        )
        self.assertEqual(record["source_stage"], "proof_refine_finalize")
        manifest = json.loads(
            (self.output_dir / "manifest.jsonl").read_text().strip()
        )
        self.assertIn("ValueError", manifest["sources"]["direct"]["parse_error"])

    def test_rejects_duplicate_rank_for_same_call(self) -> None:
        for rank in (0, 1):
            write_call(
                self.path("4", 3, 2, rank=rank),
                stage="proof_refine",
                budget_reached=False,
            )
        with self.assertRaisesRegex(RuntimeError, "duplicate refinement call"):
            export_refinements(
                self.run_dir,
                self.rubrics,
                self.output_dir,
                ["4", "5"],
                min_round=2,
            )

    def test_round_filter_excludes_other_rounds(self) -> None:
        for round_index in (1, 2, 3):
            write_call(
                self.path("4", round_index, round_index),
                stage="proof_refine",
                budget_reached=False,
            )
        summary = export_refinements(
            self.run_dir,
            self.rubrics,
            self.output_dir,
            ["4", "5"],
            min_round=2,
            max_round=2,
        )
        self.assertEqual(summary["grader_candidate_count"], 1)
        record = json.loads(
            (self.output_dir / "records.jsonl").read_text().strip()
        )
        self.assertEqual(record["round"], 2)


if __name__ == "__main__":
    unittest.main()
