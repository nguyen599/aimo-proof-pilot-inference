from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.harness_vllm import run  # noqa: E402
from evaluation.harness_vllm.optimize_thinking_handoff import (  # noqa: E402
    discover_cases,
    prepare_case,
    select_diverse_cases,
)
from evaluation.harness_vllm.thinking_handoff import (  # noqa: E402
    FINAL_PARTIAL_FORCE_MARKER,
    HANDOFF_ASSISTANT_PREFIX,
    SavedProofGenerationCall,
    _parse_segment,
    append_restart_instruction,
    build_handoff_instruction,
    build_user_turn_prompt_ids,
    insert_restart_instruction_into_rendered_prompt,
    parse_handoff_response,
    parse_saved_proof_generation_call,
    remove_final_partial_force_text,
)


LOG_ROOT = (
    REPO
    / "evaluation"
    / "runs"
    / "imo2025-full-p16-r1-p2-sft750-parserfix4-20260716T173047Z-partial-20260717"
    / "logs"
)


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        continue_final_message,
        return_dict=False,
    ):
        del continue_final_message, return_dict
        rendered = "<bos>"
        for message in messages:
            rendered += (
                f"<{message['role']}>{message['content']}</{message['role']}><eos>"
            )
        if add_generation_prompt:
            rendered += "<assistant>"
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered

    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    @staticmethod
    def decode(token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(int(token_id)) for token_id in token_ids)


VALID_HANDOFF = (
    "<handoff>"
    "<established>A proved reduction.</established>"
    "<promising>An exact identity.</promising>"
    "<failed>A counting route lacks injectivity.</failed>"
    "<uncertain>A parity pattern is unproved.</uncertain>"
    "<bottleneck>The missing lower bound.</bottleneck>"
    "<next_steps>Prove the lower bound, then construct equality.</next_steps>"
    "</handoff>"
)
VALID_HANDOFF_AFTER_PREFIX = VALID_HANDOFF.removeprefix("<handoff>")


def pipeline_cfg(**overrides):
    values = {
        "deepseek_math_v2_candidate_count": 0,
        "thinking_budget_enabled": True,
        "proof_generation_thinking_budgets": [10],
        "proof_max_new_tokens": 20,
        "default_temperature": 0.7,
        "proof_generation_temperatures": [],
        "deepseek_thinking_budget_force_text": "## Solution\n",
        "thinking_budget_force_text": run.FINAL_PARTIAL_FORCE_TEXT,
        "thinking_budget_handoff_enabled": True,
        "thinking_budget_handoff_max_tokens": 4096,
        "thinking_budget_handoff_temperature": 0.7,
        "thinking_budget_handoff_prompt_variant": "evidence_first",
        "verify_n": 1,
        "meta_n": 0,
        "meta_policy": "all-reviews",
        "strict_pass_meta": False,
        "refine_rounds": 1,
        "refine_review_n": 1,
        "min_valid_low": 1,
        "verification_early_stop": False,
        "verifier_thinking_budget_tokens": 0,
        "verifier_thinking_budget_force_text": "<evaluation>\n",
        "deepseek_verifier_thinking_budget_force_text": (
            "Here is my evaluation of the solution:\n"
        ),
        "meta_thinking_budget_tokens": 0,
        "meta_thinking_budget_force_text": "",
        "wait_for_all_generations_before_verify": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class SavedCallParserTests(unittest.TestCase):
    def test_continuation_parser_preserves_prompt_trailing_newline(self):
        prompt = "decoded prompt with significant suffix\n"
        prompt_tokens, max_tokens, parsed = _parse_segment(
            f"prompt_tokens: 17\nmax_tokens: 23\n\n{prompt}\n\n"
        )
        self.assertEqual(prompt_tokens, 17)
        self.assertEqual(max_tokens, 23)
        self.assertEqual(parsed, prompt)

    def test_all_checked_in_budget_hit_logs_parse_and_drop_force_suffix(self):
        paths = []
        for path in LOG_ROOT.glob("rank*/llm_calls/*/*proof_gen*.txt"):
            if FINAL_PARTIAL_FORCE_MARKER in path.read_text(encoding="utf-8"):
                paths.append(path)

        self.assertEqual(len(paths), 17)
        for path in paths:
            record = parse_saved_proof_generation_call(path)
            pre_force = remove_final_partial_force_text(record.continuation_prompt)
            self.assertEqual(record.stage, "proof_generation")
            self.assertGreater(record.continuation_prompt_tokens, 120_000)
            self.assertNotIn(FINAL_PARTIAL_FORCE_MARKER, pre_force)
            self.assertTrue(pre_force.strip())
            self.assertIn("<think>", pre_force)

    def test_optimizer_selects_eight_diverse_cases(self):
        selected = select_diverse_cases(discover_cases(LOG_ROOT), 8)
        self.assertEqual(len(selected), 8)
        self.assertEqual({case["rank"] for case in selected}, {"rank0", "rank1"})
        self.assertEqual({case["problem"] for case in selected}, {"1", "2"})
        self.assertGreaterEqual(
            len({bool(case["old_parseable"]) for case in selected}),
            1,
        )

    def test_prepare_case_accepts_equivalent_token_segmentation(self):
        class CanonicalTokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                del add_special_tokens
                token_ids = []
                index = 0
                while index < len(text):
                    if text.startswith("ab", index):
                        token_ids.append(1_000_000)
                        index += 2
                    else:
                        token_ids.append(ord(text[index]))
                        index += 1
                return token_ids

            @staticmethod
            def decode(token_ids, skip_special_tokens=False):
                del skip_special_tokens
                return "".join(
                    "ab" if token_id == 1_000_000 else chr(token_id)
                    for token_id in token_ids
                )

        continuation_prompt = "abc" + run.FINAL_PARTIAL_FORCE_TEXT
        canonical_tokens = CanonicalTokenizer.encode(continuation_prompt)
        record = SavedProofGenerationCall(
            path=Path("saved.txt"),
            stage="proof_generation",
            detail="candidate=0 round=0",
            prompt_tokens=1,
            max_tokens=10,
            input_prompt="input",
            continuation_prompt=continuation_prompt,
            continuation_prompt_tokens=len(canonical_tokens) + 1,
            continuation_max_tokens=1,
            output_text=FINAL_PARTIAL_FORCE_MARKER,
            finish_reason="length",
            usage={},
        )
        prepared = prepare_case(
            {
                "record": record,
                "rank": "rank0",
                "problem": "1",
                "old_parseable": False,
                "source": "rank0/llm_calls/1/call.txt",
            },
            CanonicalTokenizer(),
            max_token_drift=1,
        )
        self.assertEqual(prepared["token_drift"], -1)


class HandoffPromptTests(unittest.TestCase):
    def test_transition_is_a_real_new_user_turn(self):
        tokenizer = FakeTokenizer()
        instruction = build_handoff_instruction("continuation_frontier")
        prompt_ids = build_user_turn_prompt_ids(
            tokenizer,
            tokenizer.encode("<bos><assistant><think>old reasoning"),
            instruction,
            close_open_thinking=True,
        )
        rendered = tokenizer.decode(prompt_ids)

        self.assertIn("</think></assistant><eos><user>", rendered)
        self.assertIn(instruction, rendered)
        self.assertTrue(rendered.endswith("<assistant>" + HANDOFF_ASSISTANT_PREFIX))

    def test_handoff_parser_requires_every_section(self):
        parsed = parse_handoff_response(VALID_HANDOFF)
        self.assertTrue(parsed["is_valid"])
        self.assertEqual(parsed["missing_sections"], [])

        invalid = parse_handoff_response(
            VALID_HANDOFF.replace(
                "<bottleneck>The missing lower bound.</bottleneck>",
                "",
            )
        )
        self.assertFalse(invalid["is_valid"])
        self.assertEqual(invalid["missing_sections"], ["bottleneck"])

    def test_restart_prompt_keeps_original_contract_and_adds_handoff(self):
        prompt = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Problem and XML contract."},
        ]
        restarted = append_restart_instruction(prompt, VALID_HANDOFF, 1)
        self.assertEqual(restarted[0], prompt[0])
        self.assertIn("Problem and XML contract.", restarted[-1]["content"])
        self.assertIn("<previous_attempt_handoff>", restarted[-1]["content"])
        self.assertIn("restart round 1", restarted[-1]["content"])

    def test_restart_instruction_can_be_inserted_into_saved_rendered_prompt(self):
        rendered = (
            "<｜begin▁of▁sentence｜><｜User｜>Problem text<｜Assistant｜><think>\n"
        )
        restarted = insert_restart_instruction_into_rendered_prompt(
            rendered,
            VALID_HANDOFF,
            1,
        )
        self.assertTrue(restarted.startswith("<｜begin▁of▁sentence｜><｜User｜>"))
        self.assertTrue(restarted.endswith("<｜Assistant｜><think>\n"))
        self.assertIn("<previous_attempt_handoff>", restarted)
        self.assertLess(
            restarted.index("<previous_attempt_handoff>"),
            restarted.index("<｜Assistant｜>"),
        )

    def test_restart_instruction_rejects_unknown_rendered_prompt(self):
        with self.assertRaisesRegex(ValueError, "assistant marker"):
            insert_restart_instruction_into_rendered_prompt(
                "plain prompt",
                VALID_HANDOFF,
                1,
            )


class BudgetRestartPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_budget_hit_handoff_restarts_before_only_verification_round(self):
        class FakeScheduler:
            tokenizer = FakeTokenizer()

            def __init__(self):
                self.calls = []
                self.generation_count = 0

            async def call(self, stage, prompt, **kwargs):
                self.calls.append((stage, prompt, kwargs))
                if stage == "proof_generation":
                    self.generation_count += 1
                    if self.generation_count == 1:
                        return {
                            "success": True,
                            "text": "<think>unfinished research",
                            "finish_reason": "thinking_budget_reached",
                            "usage": {
                                "completion_tokens": 10,
                                "estimated_prompt_tokens": 5,
                                "thinking_budget_applied": True,
                                "thinking_budget_action": "stop",
                                "thinking_budget_force_skipped_closed": False,
                            },
                            "_thinking_budget_context_ids": [1, 2, 3],
                        }
                    return {
                        "success": True,
                        "text": (
                            "<solution>A complete restarted proof.</solution>"
                            "<self_evaluation>All steps checked.</self_evaluation>"
                            "<score>1</score>"
                        ),
                        "finish_reason": "stop",
                        "usage": {},
                    }
                if stage == "proof_handoff":
                    return {
                        "success": True,
                        "text": (HANDOFF_ASSISTANT_PREFIX + VALID_HANDOFF_AFTER_PREFIX),
                        "finish_reason": "stop",
                        "usage": {},
                        "_completion_context_ids": [1, 2, 3, 4],
                    }
                if stage == "proof_verify":
                    return {
                        "success": True,
                        "text": (
                            "<evaluation>The proof is complete.</evaluation>"
                            "<suggestions>No repair needed.</suggestions>"
                            "<score>1</score>"
                        ),
                        "finish_reason": "stop",
                        "usage": {},
                    }
                raise AssertionError(stage)

        scheduler = FakeScheduler()
        result = await run.run_candidate_pipeline(
            "Prove the claim.",
            0,
            1,
            scheduler,
            pipeline_cfg(),
        )
        candidate = result["candidate"]

        self.assertEqual(
            [stage for stage, _, _ in scheduler.calls],
            [
                "proof_generation",
                "proof_handoff",
                "proof_generation",
                "proof_verify",
            ],
        )
        self.assertEqual(candidate["budget_restart_count"], 1)
        self.assertEqual(candidate["selected_verification_round"], 1)
        self.assertEqual(len(candidate["proof_generation_outputs"]), 2)
        self.assertEqual(len(candidate["proof_handoff_output"]), 1)
        self.assertEqual(candidate["proof_refine_output"], [])
        self.assertEqual(
            candidate["proof_solution"],
            "A complete restarted proof.",
        )

    async def test_final_round_keeps_existing_finalize_action(self):
        class FakeScheduler:
            def __init__(self):
                self.calls = []

            async def call(self, stage, prompt, **kwargs):
                self.calls.append((stage, prompt, kwargs))
                return {
                    "success": True,
                    "text": (
                        "<solution>A final partial proof.</solution>"
                        "<self_evaluation>A gap remains.</self_evaluation>"
                        "<score>0</score>"
                    ),
                    "finish_reason": "stop",
                    "usage": {},
                }

        scheduler = FakeScheduler()
        result = await run.generate_single_attempt(
            "Prove the claim.",
            0,
            1,
            scheduler,
            pipeline_cfg(refine_rounds=0),
        )

        self.assertIsNotNone(result)
        self.assertEqual(len(scheduler.calls), 1)
        self.assertEqual(
            scheduler.calls[0][2]["thinking_budget_action"],
            "finalize",
        )
        self.assertEqual(result["consumed_refine_rounds"], 0)
        self.assertEqual(result["handoff_outputs"], [])


if __name__ == "__main__":
    unittest.main()
