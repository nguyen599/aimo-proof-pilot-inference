from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import run  # noqa: E402


class RunOpdPromptContractTests(unittest.TestCase):
    def test_checked_in_imo_parquet_is_supported(self):
        frame, problem_column, id_column = run.load_simple_input(
            REPO / "evaluation" / "data" / "imo_2025.parquet"
        )

        self.assertEqual(len(frame), 6)
        self.assertEqual(problem_column, "problem")
        self.assertEqual(id_column, "problem_idx")

    def test_long_context_defaults_stay_near_training_length(self):
        self.assertEqual(run.CFG.num_ctx, 262_144)
        self.assertLessEqual(run.CFG.max_new_tokens, 131_072)
        self.assertGreaterEqual(min(run.CFG.proof_generation_thinking_budgets), 120_000)
        self.assertLess(
            max(run.CFG.proof_generation_thinking_budgets),
            run.CFG.max_new_tokens,
        )
        self.assertEqual(run.CFG.verifier_max_new_tokens, 126_000)
        self.assertEqual(run.CFG.meta_max_new_tokens, 126_000)
        self.assertLess(
            run.CFG.verifier_thinking_budget_tokens,
            run.CFG.verifier_max_new_tokens,
        )
        self.assertLess(
            run.CFG.meta_thinking_budget_tokens,
            run.CFG.meta_max_new_tokens,
        )

    def test_imo2025_defaults_run_the_complete_candidate_pipeline(self):
        self.assertEqual(
            run.CFG.input_csv,
            REPO / "evaluation" / "data" / "imo_2025.parquet",
        )
        self.assertEqual(run.CFG.pipelines_per_problem, 14)
        self.assertEqual(run.CFG.proof_only_candidate_count, 0)
        self.assertFalse(run.CFG.skip_self_score_zero)
        self.assertFalse(run.CFG.stop_on_strict_pass)
        self.assertFalse(run.CFG.verification_early_stop)
        self.assertEqual(run.CFG.verify_n, 4)
        self.assertEqual(run.CFG.meta_n, 1)
        self.assertEqual(run.CFG.meta_policy, "all-reviews")
        self.assertEqual(run.CFG.refine_rounds, 1)
        self.assertEqual(run.CFG.refine_review_n, 2)
        self.assertLess(run.CFG.refine_review_n, run.CFG.verify_n)
        self.assertGreaterEqual(run.CFG.problem_timeout_seconds, 86_400)

    def test_prover_uses_trained_system_user_prompt(self):
        messages = run.build_opd_proof_generation_prompt("Prove the claim.")

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("mathematical proof generator", messages[0]["content"])
        self.assertIn("Problem:\nProve the claim.", messages[1]["content"])
        self.assertIn("<self_evaluation>", messages[1]["content"])

    def test_generation_parser_reads_opd_xml(self):
        parsed = run.parse_generation_response(
            "<think>private work</think>"
            "<solution>A complete proof.</solution>"
            "<self_evaluation>The proof is checked.</self_evaluation>"
            "<score>1</score>"
        )

        self.assertTrue(parsed["is_valid_candidate_response"])
        self.assertEqual(parsed["proof"], "A complete proof.")
        self.assertEqual(parsed["self_score"], 1.0)

    def test_verifier_requires_complete_xml_contract(self):
        valid = run.parse_verifier_response(
            "<evaluation>A fatal gap exists.</evaluation>"
            "<suggestions>Prove the missing lemma.</suggestions>"
            "<score>0</score>"
        )
        malformed = run.parse_verifier_response(
            "<evaluation>Claims to pass.</evaluation><score>1</score>"
        )

        self.assertTrue(valid["is_valid_verifier_response"])
        self.assertEqual(valid["score"], 0.0)
        self.assertFalse(malformed["is_valid_verifier_response"])
        self.assertIsNone(malformed["score"])

    def test_refiner_receives_xml_candidate_and_reviews(self):
        messages = run.build_opd_proof_refinement_prompt(
            "Problem.",
            "P3",
            "Candidate proof.",
            "Candidate audit.",
            [
                {
                    "score": 0.5,
                    "review": (
                        "<evaluation>Minor gap.</evaluation>"
                        "<suggestions>Fill it.</suggestions><score>0.5</score>"
                    ),
                }
            ],
        )
        user = messages[1]["content"]

        self.assertIn('<candidate id="P3">', user)
        self.assertIn('<verifier_review score="0.5">', user)
        self.assertIn("Candidate audit.", user)

    def test_selector_uses_trained_id_contract(self):
        messages = run.build_selection_prompt(
            "Problem.",
            [{"proof_solution": "First."}, {"proof_solution": "Second."}],
            10_000,
        )

        self.assertEqual(
            hashlib.sha256(
                (run.OPD_PROMPT_ROOT / "selector.txt").read_bytes()
            ).hexdigest(),
            "1cf13bb2c62cc15b3f92b4d65a5e25e8893736d6cc3afb49f01a49c13b45052b",
        )
        self.assertIn('<candidate id="R0">', messages[1]["content"])
        self.assertEqual(
            run.parse_selected_index("<selected_id>R1</selected_id>", 2), 1
        )
        self.assertIsNone(run.parse_selected_index("SELECTED_INDEX: 1", 2))

    def test_only_deepseek_meta_prompt_is_retained(self):
        self.assertTrue(hasattr(run, "build_deepseek_meta_verification_prompt"))
        self.assertFalse(hasattr(run, "build_deepseek_gold_proof_evaluation_prompt"))
        self.assertFalse(hasattr(run, "build_proof_architect_prompt"))
        self.assertFalse(hasattr(run, "build_sublemma_prover_prompt"))


if __name__ == "__main__":
    unittest.main()
