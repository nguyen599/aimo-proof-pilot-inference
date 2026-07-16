from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.harness_vllm import run  # noqa: E402
from evaluation.harness_vllm import evaluate_thinking_handoff_refinement  # noqa: E402
from evaluation.harness_vllm import evaluate_thinking_handoff_restart  # noqa: E402
from evaluation.harness_vllm.optimize_thinking_handoff import (  # noqa: E402
    call_handoff,
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
    append_final_output_discipline,
    assemble_handoff,
    build_empty_restart_handoff,
    build_fresh_handoff_section_prompt_ids,
    build_handoff_from_digests_prompt_ids,
    build_handoff_instruction,
    build_lossless_partial_handoff,
    build_restart_instruction,
    build_structured_partial_force_text,
    build_handoff_section_from_digests_prompt_ids,
    build_handoff_section_from_partial_progress_prompt_ids,
    build_handoff_section_instruction,
    build_research_window_digest_prompt_ids,
    build_user_turn_prompt_ids,
    insert_restart_instruction_into_rendered_prompt,
    normalize_research_digest,
    extract_forced_partial_progress,
    escape_handoff_control_tags,
    parse_handoff_response,
    parse_saved_proof_generation_call,
    prepare_handoff_research_windows,
    remove_final_partial_force_text,
    select_reasoning_token_windows,
    truncate_consecutive_token_repetition,
    truncate_low_novelty_token_tail,
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
        "thinking_budget_handoff_preserve_refine_rounds": False,
        "thinking_budget_handoff_max_tokens": 4096,
        "thinking_budget_handoff_temperature": 0.7,
        "thinking_budget_handoff_prompt_variant": "evidence_first",
        "thinking_budget_handoff_mode": "model",
        "thinking_budget_restart_strategy": "standard",
        "thinking_budget_final_round_tokens": 0,
        "thinking_budget_refine_handoff_enabled": False,
        "thinking_budget_refine_tokens": 0,
        "thinking_budget_refine_final_round_tokens": 0,
        "thinking_budget_refine_max_restarts": 1,
        "thinking_budget_refine_final_temperature": None,
        "thinking_budget_refine_visible_output_target_tokens": 0,
        "thinking_budget_refine_visible_output_limit_tokens": 0,
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
    def test_fresh_section_prompt_quotes_sampled_reasoning(self):
        original = (
            "<｜begin▁of▁sentence｜>System<｜User｜>Problem: prove X.\n\n"
            "Respond in EXACTLY this format:\n<solution>...</solution>"
            "<｜Assistant｜><think>\n"
        )
        prompt_ids, metadata = build_fresh_handoff_section_prompt_ids(
            FakeTokenizer(),
            original_input_prompt=original,
            pre_force_text=original + "useful unfinished reasoning",
            section="next_steps",
            variant="continuation_frontier",
            reasoning_total_tokens=16,
            reasoning_window_tokens=8,
        )
        rendered = FakeTokenizer.decode(prompt_ids)

        self.assertIn("Problem: prove X.", rendered)
        self.assertNotIn("Respond in EXACTLY this format", rendered)
        self.assertIn("<research_window", rendered)
        self.assertIn("latest extraction instruction", rendered)
        self.assertEqual(metadata["reasoning_original_tokens"], 27)

    def test_repetition_truncation_and_window_sampling(self):
        cleaned, repetition = truncate_consecutive_token_repetition(
            [1, 2] + [3, 4] * 4,
            block_sizes=(2,),
            minimum_repeats=4,
        )
        self.assertEqual(cleaned, [1, 2])
        self.assertEqual(repetition["block_tokens"], 2)

        windows = select_reasoning_token_windows(
            list(range(100)),
            total_tokens=20,
            window_tokens=10,
        )
        self.assertEqual(
            [(start, end) for start, end, _ in windows], [(0, 10), (90, 100)]
        )

    def test_map_reduce_prompts_keep_window_extraction_separate(self):
        original = (
            "<｜begin▁of▁sentence｜>System<｜User｜>Problem: prove X.\n\n"
            "Respond in EXACTLY this format:\n<solution>...</solution>"
            "<｜Assistant｜><think>\n"
        )
        problem, windows, metadata = prepare_handoff_research_windows(
            FakeTokenizer(),
            original_input_prompt=original,
            pre_force_text=original + "first fact; failed route; next idea",
            reasoning_total_tokens=18,
            reasoning_window_tokens=9,
        )
        digest_prompt = FakeTokenizer.decode(
            build_research_window_digest_prompt_ids(
                FakeTokenizer(),
                window=windows[0],
            )
        )
        final_prompt = FakeTokenizer.decode(
            build_handoff_from_digests_prompt_ids(
                FakeTokenizer(),
                problem=problem,
                digests=[
                    {
                        "start": windows[0]["start"],
                        "end": windows[0]["end"],
                        "text": "A proved local fact.",
                    }
                ],
                variant="evidence_first",
            )
        )

        self.assertEqual(len(windows), 2)
        self.assertEqual(metadata["window_ranges"], [(0, 9), (26, 35)])
        self.assertIn("one chronological window", digest_prompt)
        self.assertNotIn("below 1,200 words", digest_prompt)
        self.assertIn("A proved local fact.", final_prompt)
        self.assertIn("below 1,200 words", final_prompt)
        self.assertEqual(
            normalize_research_digest("<digest>P | Useful fact.</digest>ignored"),
            "P | Useful fact.",
        )
        section_prompt = FakeTokenizer.decode(
            build_handoff_section_from_digests_prompt_ids(
                FakeTokenizer(),
                digests=[
                    {
                        "start": windows[0]["start"],
                        "end": windows[0]["end"],
                        "text": "P | A proved local fact.",
                    }
                ],
                section="established",
                variant="evidence_first",
            )
        )
        self.assertNotIn("Original problem:", section_prompt)
        self.assertIn("P | A proved local fact.", section_prompt)

    def test_forced_partial_progress_becomes_fresh_section_source(self):
        report = extract_forced_partial_progress(
            "hidden reasoning\n"
            + run.FINAL_PARTIAL_FORCE_TEXT
            + "- Proved a reduction.\n- A construction still has a gap.\n"
            + "</solution><self_evaluation>ignored</self_evaluation>"
        )
        prompt = FakeTokenizer.decode(
            build_handoff_section_from_partial_progress_prompt_ids(
                FakeTokenizer(),
                partial_progress=report,
                section="failed",
                variant="lemma_ledger",
            )
        )

        self.assertIn("- Proved a reduction.", report)
        self.assertNotIn("self_evaluation", report)
        self.assertIn("<partial_progress_report>", prompt)
        self.assertIn("Treat every claim as untrusted", prompt)
        self.assertIn("lemma ledger", prompt)

    def test_low_novelty_tail_truncates_repetitive_loop(self):
        prefix = list(range(3_000))
        repeated = [7, 8, 9, 10] * 2_000
        cleaned, metadata = truncate_low_novelty_token_tail(
            prefix + repeated,
            window_tokens=512,
            stride_tokens=128,
            minimum_prefix_tokens=1_024,
        )

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["kind"], "low_novelty")
        self.assertGreaterEqual(len(cleaned), len(prefix) - 512)
        self.assertLess(len(cleaned), len(prefix) + 512)

    def test_optimizer_builds_map_reduce_handoff(self):
        requests = []
        responses = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text=f"P | Digest {index}.</digest>",
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=10,
                    total_tokens=110,
                ),
            )
            for index in range(8)
        ]
        responses.extend(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text=f"Section content {index}.",
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=200,
                    completion_tokens=10,
                    total_tokens=210,
                ),
            )
            for index in range(6)
        )

        class FakeCompletions:
            def create(self, **kwargs):
                requests.append(kwargs)
                return responses.pop(0)

        class FakeOpenAI:
            def __init__(self, **kwargs):
                del kwargs
                self.completions = FakeCompletions()

        original = (
            "<｜begin▁of▁sentence｜>System<｜User｜>Problem: prove X."
            "<｜Assistant｜><think>\n"
        )
        reasoning = "".join(chr(0x10000 + index) for index in range(40_000))
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "evaluation.harness_vllm.optimize_thinking_handoff.OpenAI",
                FakeOpenAI,
            ):
                result = call_handoff(
                    prepared={
                        "record": SimpleNamespace(input_prompt=original),
                        "pre_force_text": original + reasoning,
                        "pre_force_ids": [],
                        "source": "rank0/llm_calls/1/call.txt",
                        "rank": "rank0",
                        "problem": "1",
                        "old_parseable": False,
                    },
                    tokenizer=FakeTokenizer(),
                    base_url="http://localhost:8000",
                    api_key="test",
                    served_model_name="proof-model",
                    variant="evidence_first",
                    temperature=0.7,
                    max_tokens=4096,
                    generation_mode="map_reduce",
                    repair_invalid=True,
                    top_p=0.95,
                    request_timeout_seconds=10,
                    output_dir=Path(directory),
                )

        self.assertEqual(len(result["attempts"]), 14)
        self.assertTrue(result["parsed"]["is_valid"])
        self.assertEqual(result["usage"]["completion_tokens"], 140)
        self.assertEqual(
            [request["temperature"] for request in requests[:8]],
            [0.2] * 8,
        )
        self.assertEqual(
            [request["temperature"] for request in requests[8:]],
            [0.7] * 6,
        )
        self.assertEqual(
            [request["max_tokens"] for request in requests[8:]],
            [320, 256, 192, 160, 128, 192],
        )
        self.assertEqual(
            [attempt["digest"] for attempt in result["attempts"][:2]],
            ["P | Digest 0.", "P | Digest 1."],
        )

    def test_optimizer_builds_sectioned_handoff(self):
        responses = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text=f"Section content {index}.",
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=10,
                    total_tokens=110,
                ),
            )
            for index in range(6)
        ]

        class FakeCompletions:
            def create(self, **kwargs):
                del kwargs
                return responses.pop(0)

        class FakeOpenAI:
            def __init__(self, **kwargs):
                del kwargs
                self.completions = FakeCompletions()

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "evaluation.harness_vllm.optimize_thinking_handoff.OpenAI",
                FakeOpenAI,
            ):
                result = call_handoff(
                    prepared={
                        "pre_force_ids": FakeTokenizer.encode("<assistant><think>work"),
                        "source": "rank0/llm_calls/1/call.txt",
                        "rank": "rank0",
                        "problem": "1",
                        "old_parseable": False,
                    },
                    tokenizer=FakeTokenizer(),
                    base_url="http://localhost:8000",
                    api_key="test",
                    served_model_name="proof-model",
                    variant="evidence_first",
                    temperature=0.7,
                    max_tokens=4096,
                    generation_mode="sectioned",
                    repair_invalid=True,
                    top_p=0.95,
                    request_timeout_seconds=10,
                    output_dir=Path(directory),
                )

        self.assertEqual(len(result["attempts"]), 6)
        self.assertFalse(result["repair_used"])
        self.assertTrue(result["parsed"]["is_valid"])
        self.assertEqual(result["usage"]["completion_tokens"], 60)

    def test_optimizer_builds_partial_sectioned_handoff(self):
        responses = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text=f"Partial section {index}.",
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=10,
                    total_tokens=110,
                ),
            )
            for index in range(6)
        ]

        class FakeCompletions:
            def create(self, **kwargs):
                del kwargs
                return responses.pop(0)

        class FakeOpenAI:
            def __init__(self, **kwargs):
                del kwargs
                self.completions = FakeCompletions()

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "evaluation.harness_vllm.optimize_thinking_handoff.OpenAI",
                FakeOpenAI,
            ):
                result = call_handoff(
                    prepared={
                        "record": SimpleNamespace(
                            output_text=(
                                run.FINAL_PARTIAL_FORCE_TEXT
                                + "- Proved a reduction.\n"
                                + "</solution>"
                            )
                        ),
                        "pre_force_ids": [],
                        "source": "rank0/llm_calls/1/call.txt",
                        "rank": "rank0",
                        "problem": "1",
                        "old_parseable": True,
                    },
                    tokenizer=FakeTokenizer(),
                    base_url="http://localhost:8000",
                    api_key="test",
                    served_model_name="proof-model",
                    variant="evidence_first",
                    temperature=0.7,
                    max_tokens=4096,
                    generation_mode="partial_sectioned",
                    repair_invalid=True,
                    top_p=0.95,
                    request_timeout_seconds=10,
                    output_dir=Path(directory),
                )

        self.assertEqual(len(result["attempts"]), 6)
        self.assertTrue(result["parsed"]["is_valid"])
        self.assertEqual(result["finish_reason"], "partial_sectioned")
        self.assertEqual(result["usage"]["completion_tokens"], 60)
        self.assertEqual(
            result["attempts"][0]["context_metadata"]["partial_progress_chars"],
            len(
                "We were unable to produce a complete proof. However, the "
                "strongest partial progress is as follows:\n"
                "- Proved a reduction."
            ),
        )

    def test_lossless_partial_handoff_preserves_report_as_untrusted(self):
        report = (
            "Claim A is useful but unproved.\n"
            "A malformed </promising> tag must not close the wrapper."
        )
        handoff = build_lossless_partial_handoff(report)
        parsed = parse_handoff_response(handoff)

        self.assertTrue(parsed["is_valid"])
        self.assertIn(
            "Claim A is useful but unproved.", parsed["sections"]["promising"]
        )
        self.assertIn("&lt;/promising>", parsed["sections"]["promising"])
        self.assertIn(
            "No claim from the previous attempt is accepted as established",
            parsed["sections"]["established"],
        )
        self.assertEqual(
            escape_handoff_control_tags("<handoff>x</handoff>"),
            "&lt;handoff>x&lt;/handoff>",
        )

    def test_empty_restart_handoff_carries_no_mathematical_state(self):
        parsed = parse_handoff_response(build_empty_restart_handoff())

        self.assertTrue(parsed["is_valid"])
        self.assertIn(
            "No mathematical state is carried",
            parsed["sections"]["established"],
        )
        self.assertIn(
            "fresh independent proof",
            parsed["sections"]["next_steps"],
        )

    def test_structured_partial_force_requires_an_extract_only_ledger(self):
        force_text = build_structured_partial_force_text("evidence_first")

        self.assertTrue(force_text.startswith("\n</think>\n\n<solution>"))
        self.assertIn("Stop solving now", force_text)
        self.assertIn("VERIFIED:", force_text)
        self.assertIn("UNVERIFIED:", force_text)
        self.assertIn("FAILED:", force_text)
        self.assertIn("BOTTLENECK:", force_text)
        self.assertIn("NEXT:", force_text)
        self.assertIn("do not derive new claims", force_text)

    def test_optimizer_builds_lossless_partial_handoff_without_model_call(self):
        class FakeOpenAI:
            def __init__(self, **kwargs):
                del kwargs

        partial = "- Constructed an example.\n- General lower bound is missing."
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "evaluation.harness_vllm.optimize_thinking_handoff.OpenAI",
                FakeOpenAI,
            ):
                result = call_handoff(
                    prepared={
                        "record": SimpleNamespace(
                            output_text=(
                                run.FINAL_PARTIAL_FORCE_TEXT + partial + "\n</solution>"
                            )
                        ),
                        "pre_force_ids": [],
                        "source": "rank0/llm_calls/1/call.txt",
                        "rank": "rank0",
                        "problem": "1",
                        "old_parseable": True,
                    },
                    tokenizer=FakeTokenizer(),
                    base_url="http://localhost:8000",
                    api_key="test",
                    served_model_name="proof-model",
                    variant="evidence_first",
                    temperature=0.7,
                    max_tokens=4096,
                    generation_mode="partial_passthrough",
                    repair_invalid=True,
                    top_p=0.95,
                    request_timeout_seconds=10,
                    output_dir=Path(directory),
                )

        self.assertTrue(result["parsed"]["is_valid"])
        self.assertEqual(result["finish_reason"], "partial_passthrough")
        self.assertEqual(result["usage"]["completion_tokens"], 0)
        self.assertTrue(result["attempts"][0]["context_metadata"]["lossless"])
        self.assertIn(partial, result["parsed"]["sections"]["promising"])

    def test_optimizer_builds_empty_restart_baseline_without_model_call(self):
        class FakeOpenAI:
            def __init__(self, **kwargs):
                del kwargs

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "evaluation.harness_vllm.optimize_thinking_handoff.OpenAI",
                FakeOpenAI,
            ):
                result = call_handoff(
                    prepared={
                        "record": SimpleNamespace(output_text="unused"),
                        "pre_force_ids": [],
                        "source": "rank0/llm_calls/1/call.txt",
                        "rank": "rank0",
                        "problem": "1",
                        "old_parseable": True,
                    },
                    tokenizer=FakeTokenizer(),
                    base_url="http://localhost:8000",
                    api_key="test",
                    served_model_name="proof-model",
                    variant="evidence_first",
                    temperature=0.7,
                    max_tokens=4096,
                    generation_mode="empty_baseline",
                    repair_invalid=True,
                    top_p=0.95,
                    request_timeout_seconds=10,
                    output_dir=Path(directory),
                )

        self.assertTrue(result["parsed"]["is_valid"])
        self.assertEqual(result["finish_reason"], "empty_baseline")
        self.assertEqual(result["prompt_tokens"], 0)
        self.assertEqual(
            result["attempts"][0]["context_metadata"]["baseline"],
            "fresh_restart_without_mathematical_handoff",
        )

    def test_optimizer_wraps_structured_forced_report_losslessly(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    text=(
                        "VERIFIED:\n- Exact construction A.\n\n"
                        "UNVERIFIED:\n- General claim B.\n\n"
                        "FAILED:\n- Route C lacks a bound.\n\n"
                        "BOTTLENECK:\n- Missing lower bound.\n\n"
                        "NEXT:\n- Prove the lower bound."
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=120_000,
                completion_tokens=80,
                total_tokens=120_080,
            ),
        )

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return response

        class FakeOpenAI:
            def __init__(self, **kwargs):
                del kwargs
                self.completions = FakeCompletions()

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "evaluation.harness_vllm.optimize_thinking_handoff.OpenAI",
                FakeOpenAI,
            ):
                result = call_handoff(
                    prepared={
                        "record": SimpleNamespace(output_text="unused"),
                        "pre_force_ids": FakeTokenizer.encode("prior reasoning"),
                        "source": "rank0/llm_calls/1/call.txt",
                        "rank": "rank0",
                        "problem": "1",
                        "old_parseable": False,
                    },
                    tokenizer=FakeTokenizer(),
                    base_url="http://localhost:8000",
                    api_key="test",
                    served_model_name="proof-model",
                    variant="evidence_first",
                    temperature=0.6,
                    max_tokens=4096,
                    generation_mode="structured_force_passthrough",
                    repair_invalid=True,
                    top_p=0.95,
                    request_timeout_seconds=10,
                    output_dir=Path(directory),
                )

        self.assertTrue(result["parsed"]["is_valid"])
        self.assertEqual(result["finish_reason"], "structured_force_passthrough")
        self.assertEqual(result["usage"]["completion_tokens"], 80)
        self.assertIn(
            "Exact construction A.",
            result["parsed"]["sections"]["promising"],
        )
        self.assertEqual(
            result["attempts"][0]["context_metadata"]["pre_force_tokens"],
            len("prior reasoning"),
        )

    def test_optimizer_repairs_one_invalid_handoff(self):
        responses = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text="repeated draft without closing tags",
                        finish_reason="length",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=10,
                    total_tokens=110,
                ),
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text=VALID_HANDOFF_AFTER_PREFIX,
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=120,
                    completion_tokens=20,
                    total_tokens=140,
                ),
            ),
        ]

        class FakeCompletions:
            def create(self, **kwargs):
                del kwargs
                return responses.pop(0)

        class FakeOpenAI:
            def __init__(self, **kwargs):
                del kwargs
                self.completions = FakeCompletions()

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "evaluation.harness_vllm.optimize_thinking_handoff.OpenAI",
                FakeOpenAI,
            ):
                result = call_handoff(
                    prepared={
                        "pre_force_ids": FakeTokenizer.encode("<assistant><think>work"),
                        "source": "rank0/llm_calls/1/call.txt",
                        "rank": "rank0",
                        "problem": "1",
                        "old_parseable": False,
                    },
                    tokenizer=FakeTokenizer(),
                    base_url="http://localhost:8000",
                    api_key="test",
                    served_model_name="proof-model",
                    variant="evidence_first",
                    temperature=0.7,
                    max_tokens=4096,
                    generation_mode="monolithic",
                    repair_invalid=True,
                    top_p=0.95,
                    request_timeout_seconds=10,
                    output_dir=Path(directory),
                )

        self.assertTrue(result["repair_used"])
        self.assertEqual(len(result["attempts"]), 2)
        self.assertFalse(result["attempts"][0]["parsed"]["is_valid"])
        self.assertTrue(result["parsed"]["is_valid"])
        self.assertEqual(result["usage"]["completion_tokens"], 30)

    def test_sectioned_handoff_assembles_all_required_sections(self):
        handoff = assemble_handoff(
            {
                "established": "A proved reduction.",
                "promising": "An exact identity.",
                "failed": "A route lacks injectivity.",
                "uncertain": "A parity pattern is unproved.",
                "bottleneck": "The lower bound is missing.",
                "next_steps": "Prove the lower bound.",
            }
        )
        parsed = parse_handoff_response(handoff)

        self.assertTrue(parsed["is_valid"])
        instruction = build_handoff_section_instruction(
            "bottleneck",
            "continuation_frontier",
        )
        self.assertIn("one paragraph of at most 120 words", instruction)
        self.assertIn("narrowest unresolved frontier", instruction)

    def test_handoff_prompt_has_strict_compression_contract(self):
        instruction = build_handoff_instruction("evidence_first")

        self.assertIn("below 1,200 words", instruction)
        self.assertIn("Do not restate the full problem", instruction)
        self.assertIn("at most 6 bullets in established", instruction)
        self.assertIn("Close every XML tag", instruction)

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

    def test_deadline_restart_requires_voluntary_finalization(self):
        instruction = build_restart_instruction(
            VALID_HANDOFF,
            1,
            strategy="deadline_aware",
        )

        self.assertIn("final proof-writing attempt", instruction)
        self.assertIn("well before the external token cutoff", instruction)
        self.assertIn("Reserve enough budget", instruction)
        self.assertIn("strongest rigorous partial proof", instruction)

    def test_final_output_discipline_forbids_visible_search(self):
        prompt = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Problem and XML contract."},
        ]
        updated = append_final_output_discipline(prompt, 12_000)

        self.assertEqual(updated[0], prompt[0])
        self.assertIn("at most 12,000 tokens", updated[-1]["content"])
        self.assertIn(
            "Do not narrate search",
            updated[-1]["content"],
        )
        self.assertIn(
            "Always close `<solution>`",
            updated[-1]["content"],
        )

    def test_visible_output_limit_footer_closes_partial_proof_honestly(self):
        partial = "<think>research</think><solution>A partial argument."
        footer = run.build_visible_output_limit_footer(partial)
        parsed = run.parse_generation_response(partial + footer)

        self.assertTrue(parsed["is_valid_candidate_response"])
        self.assertEqual(parsed["self_score"], 0.0)
        self.assertIn("A partial argument.", parsed["proof"])
        self.assertIn("must not be treated", parsed["proof"])
        self.assertIn("closed automatically", parsed["self_evaluation"])

    def test_visible_output_limit_footer_does_not_change_complete_response(self):
        complete = (
            "<solution>A complete proof.</solution>"
            "<self_evaluation>Every step is justified.</self_evaluation>"
            "<score>1</score>"
        )

        self.assertEqual(run.build_visible_output_limit_footer(complete), "")

    def test_restart_finalization_force_reserves_visible_solution(self):
        force_text = (
            evaluate_thinking_handoff_restart.RESTART_FINALIZE_FORCE_TEXT
        )

        self.assertIn("stop exploratory reasoning now", force_text)
        self.assertIn("</think>", force_text)
        self.assertIn("<solution>", force_text)
        self.assertIn("Do not continue searching", force_text)

    def test_restart_finalization_uses_only_remaining_completion_budget(self):
        first_ids = FakeTokenizer.encode("unfinished")
        final_text = (
            "A rigorous proof.</solution>"
            "<self_evaluation>The proof is complete.</self_evaluation>"
            "<score>1</score>"
        )
        final_ids = FakeTokenizer.encode(final_text)
        record = SavedProofGenerationCall(
            path=Path("rank0/llm_calls/1/cand_0_proof_gen_r0.txt"),
            stage="proof_generation",
            detail="candidate=0 round=0",
            prompt_tokens=1,
            max_tokens=10,
            input_prompt=(
                "<｜begin▁of▁sentence｜><｜User｜>Problem text"
                "<｜Assistant｜><think>\n"
            ),
            continuation_prompt="",
            continuation_prompt_tokens=0,
            continuation_max_tokens=0,
            output_text="",
            finish_reason="length",
            usage={},
        )
        handoff_result = {
            "source": "rank0/llm_calls/1/cand_0_proof_gen_r0.txt",
            "run_id": "test",
            "variant": "partial_passthrough",
            "temperature": 0.0,
            "parsed": {"text": VALID_HANDOFF},
        }
        force_ids = FakeTokenizer.encode(
            evaluate_thinking_handoff_restart.RESTART_FINALIZE_FORCE_TEXT
        )
        max_tokens = len(first_ids) + len(force_ids) + len(final_ids) + 32

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(
                    evaluate_thinking_handoff_restart,
                    "parse_saved_proof_generation_call",
                    return_value=record,
                ),
                patch.object(
                    evaluate_thinking_handoff_restart,
                    "OpenAI",
                    return_value=object(),
                ),
                patch.object(
                    evaluate_thinking_handoff_restart,
                    "stream_segment",
                    side_effect=[
                        {
                            "generated_ids": first_ids,
                            "finish_reason": None,
                            "stopped_by_budget": True,
                        },
                        {
                            "generated_ids": final_ids,
                            "finish_reason": "stop",
                            "stopped_by_budget": False,
                        },
                    ],
                ) as stream_segment,
            ):
                result = evaluate_thinking_handoff_restart.evaluate_restart(
                    handoff_result=handoff_result,
                    logs_root=Path(tmpdir),
                    tokenizer=FakeTokenizer(),
                    base_url="http://127.0.0.1:8000",
                    api_key="test",
                    served_model_name="test",
                    proof_temperature=1.0,
                    restart_strategy="deadline_aware",
                    force_finalize_at_budget=True,
                    top_p=0.95,
                    thinking_budget_tokens=len(first_ids),
                    max_tokens=max_tokens,
                    request_timeout_seconds=30,
                    output_dir=Path(tmpdir),
                )

        second_call = stream_segment.call_args_list[1].kwargs
        self.assertEqual(
            second_call["max_tokens"],
            max_tokens - len(first_ids) - len(force_ids),
        )
        self.assertEqual(
            second_call["prompt_ids"][-len(force_ids) :],
            force_ids,
        )
        self.assertTrue(result["budget_forced_finalization"])
        self.assertEqual(result["forced_finalization_tokens"], len(final_ids))
        self.assertTrue(result["valid_proof"])
        self.assertTrue(result["verification_can_start"])

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


class VisibleOutputLimitSchedulerTests(unittest.TestCase):
    def test_streaming_limit_forces_an_audited_partial_xml_closure(self):
        class FakeStream:
            def __init__(self, text):
                token_ids = FakeTokenizer.encode(text)
                self.chunks = [
                    SimpleNamespace(
                        usage=None,
                        choices=[
                            SimpleNamespace(
                                finish_reason=None,
                                text=character,
                                token_ids=token_ids[: index + 1],
                                model_extra={},
                            )
                        ],
                    )
                    for index, character in enumerate(text)
                ]

            def __iter__(self):
                return iter(self.chunks)

            def close(self):
                return None

        class FakeCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                del kwargs
                self.calls += 1
                return FakeStream(
                    "abcdefghij" if self.calls == 1 else "0123456789ABCDEFGHIJ"
                )

        completions = FakeCompletions()
        fake_openai = SimpleNamespace(completions=completions)
        with patch.object(run, "OpenAI", return_value=fake_openai):
            scheduler = run.ChatScheduler(
                base_urls=["http://test/v1"],
                api_key="test",
                model="test",
                sampling=run.SamplingConfig(
                    max_new_tokens=1_000,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=-1,
                    min_new_tokens=0,
                    min_p=None,
                ),
                max_concurrent_requests=1,
                tokenizer=FakeTokenizer(),
                stream_responses=True,
            )

        result = scheduler._call_sync(
            "proof_refine",
            [{"role": "user", "content": "Prove the claim."}],
            0,
            0.6,
            thinking_budget_tokens=5,
            thinking_budget_force_text="\n</think>\n\n<solution>\n",
            visible_output_limit_tokens=10,
        )
        parsed = run.parse_generation_response(result["text"])

        self.assertEqual(completions.calls, 2)
        self.assertEqual(result["finish_reason"], "visible_output_limit_reached")
        self.assertTrue(result["usage"]["visible_output_limit_applied"])
        self.assertEqual(
            result["usage"]["visible_output_limit_effective_tokens"],
            10,
        )
        self.assertTrue(
            result["usage"]["visible_output_forced_partial_closure"]
        )
        self.assertGreater(result["usage"]["visible_output_forced_tokens"], 0)
        self.assertTrue(parsed["is_valid_candidate_response"])
        self.assertEqual(parsed["self_score"], 0.0)


class BudgetRestartPipelineTests(unittest.IsolatedAsyncioTestCase):
    def test_final_restart_round_can_use_a_smaller_reasoning_budget(self):
        cfg = pipeline_cfg(
            proof_max_new_tokens=512,
            proof_generation_thinking_budgets=[480],
            thinking_budget_final_round_tokens=100,
        )

        self.assertEqual(
            run.resolve_thinking_budget_tokens(
                0,
                cfg,
                solve_round_idx=0,
                can_restart=True,
            ),
            480,
        )
        self.assertEqual(
            run.resolve_thinking_budget_tokens(
                0,
                cfg,
                solve_round_idx=1,
                can_restart=False,
            ),
            100,
        )

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

    async def test_budget_restart_can_preserve_verifier_refinement_round(self):
        class FakeScheduler:
            tokenizer = FakeTokenizer()

            def __init__(self):
                self.calls = []
                self.generation_count = 0
                self.verification_count = 0

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
                            "<solution>A restarted proof with a gap.</solution>"
                            "<self_evaluation>The proof may have a gap.</self_evaluation>"
                            "<score>0.5</score>"
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
                    self.verification_count += 1
                    score = 0 if self.verification_count == 1 else 1
                    suggestion = (
                        "Prove the omitted lower-bound lemma."
                        if score == 0
                        else "No repair needed."
                    )
                    return {
                        "success": True,
                        "text": (
                            f"<evaluation>Verifier score {score}.</evaluation>"
                            f"<suggestions>{suggestion}</suggestions>"
                            f"<score>{score}</score>"
                        ),
                        "finish_reason": "stop",
                        "usage": {},
                    }
                if stage == "proof_refine":
                    return {
                        "success": True,
                        "text": (
                            "<solution>A repaired complete proof.</solution>"
                            "<self_evaluation>The omitted lemma is proved.</self_evaluation>"
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
            pipeline_cfg(
                thinking_budget_handoff_preserve_refine_rounds=True,
            ),
        )
        candidate = result["candidate"]

        self.assertEqual(
            [stage for stage, _, _ in scheduler.calls],
            [
                "proof_generation",
                "proof_handoff",
                "proof_generation",
                "proof_verify",
                "proof_refine",
                "proof_verify",
            ],
        )
        self.assertEqual(candidate["budget_restart_count"], 1)
        self.assertEqual(candidate["selected_verification_round"], 1)
        self.assertEqual(len(candidate["proof_refine_output"]), 1)
        self.assertEqual(candidate["proof_solution"], "A repaired complete proof.")

    async def test_refinement_budget_handoff_restarts_before_reverification(self):
        class FakeScheduler:
            tokenizer = FakeTokenizer()

            def __init__(self):
                self.calls = []
                self.refinement_count = 0
                self.verification_count = 0

            @staticmethod
            def _token_ids_to_list(token_ids):
                return list(token_ids)

            async def call(self, stage, prompt, **kwargs):
                self.calls.append((stage, prompt, kwargs))
                if stage == "proof_generation":
                    return {
                        "success": True,
                        "text": (
                            "<solution>A proof with a gap.</solution>"
                            "<self_evaluation>A lemma is missing.</self_evaluation>"
                            "<score>0.5</score>"
                        ),
                        "finish_reason": "stop",
                        "usage": {},
                    }
                if stage == "proof_verify":
                    self.verification_count += 1
                    score = 0 if self.verification_count == 1 else 1
                    return {
                        "success": True,
                        "text": (
                            f"<evaluation>Verifier score {score}.</evaluation>"
                            "<suggestions>Prove the missing lemma.</suggestions>"
                            f"<score>{score}</score>"
                        ),
                        "finish_reason": "stop",
                        "usage": {},
                    }
                if stage == "proof_refine":
                    self.refinement_count += 1
                    if self.refinement_count == 1:
                        return {
                            "success": True,
                            "text": "<think>unfinished verifier-guided repair",
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
                            "<solution>A repaired complete proof.</solution>"
                            "<self_evaluation>The missing lemma is proved.</self_evaluation>"
                            "<score>1</score>"
                        ),
                        "finish_reason": "stop",
                        "usage": {},
                    }
                if stage == "proof_refine_finalize":
                    return {
                        "success": True,
                        "text": (
                            run.FINAL_PARTIAL_FORCE_TEXT
                            + "The repair reduced the problem to one exact lemma."
                        ),
                        "finish_reason": "stop",
                        "usage": {
                            "completion_tokens": 5,
                            "estimated_prompt_tokens": 3,
                        },
                        "server_url": "http://test",
                        "latency_s": 1.0,
                    }
                raise AssertionError(stage)

        scheduler = FakeScheduler()
        result = await run.run_candidate_pipeline(
            "Prove the claim.",
            0,
            1,
            scheduler,
            pipeline_cfg(
                thinking_budget_handoff_mode="lossless_partial",
                thinking_budget_restart_strategy="deadline_aware",
                thinking_budget_refine_handoff_enabled=True,
                thinking_budget_refine_tokens=10,
                thinking_budget_refine_final_round_tokens=10,
                thinking_budget_refine_max_restarts=1,
                thinking_budget_refine_final_temperature=0.6,
                thinking_budget_refine_visible_output_target_tokens=12_000,
                thinking_budget_refine_visible_output_limit_tokens=12_000,
            ),
        )
        candidate = result["candidate"]

        self.assertEqual(
            [stage for stage, _, _ in scheduler.calls],
            [
                "proof_generation",
                "proof_verify",
                "proof_refine",
                "proof_refine_finalize",
                "proof_refine",
                "proof_verify",
            ],
        )
        self.assertEqual(candidate["refine_budget_restart_count"], 1)
        self.assertEqual(len(candidate["proof_refine_attempt_output"]), 1)
        self.assertEqual(len(candidate["proof_refine_handoff_output"]), 1)
        self.assertEqual(
            candidate["proof_refine_handoff_output"][0]["finish_reason"],
            "partial_passthrough",
        )
        self.assertEqual(candidate["selected_verification_round"], 1)
        self.assertEqual(candidate["proof_solution"], "A repaired complete proof.")
        refinement_calls = [
            kwargs
            for stage, _, kwargs in scheduler.calls
            if stage == "proof_refine"
        ]
        self.assertEqual(
            [call["temperature"] for call in refinement_calls],
            [None, 0.6],
        )
        self.assertEqual(
            [call["visible_output_limit_tokens"] for call in refinement_calls],
            [None, 12_000],
        )
        self.assertIn(
            "at most 12,000 tokens",
            str(scheduler.calls[4][1]),
        )

    async def test_resume_refinement_handoff_runs_only_final_repair_and_verifier(self):
        class FakeScheduler:
            def __init__(self):
                self.calls = []

            async def call(self, stage, prompt, **kwargs):
                self.calls.append((stage, prompt, kwargs))
                if stage == "proof_refine":
                    return {
                        "success": True,
                        "text": (
                            "<solution>A repaired complete proof.</solution>"
                            "<self_evaluation>The missing lemma is proved.</self_evaluation>"
                            "<score>1</score>"
                        ),
                        "finish_reason": "stop",
                        "usage": {},
                    }
                if stage == "proof_verify":
                    return {
                        "success": True,
                        "text": (
                            "<evaluation>The repair is complete.</evaluation>"
                            "<suggestions>No repair needed.</suggestions>"
                            "<score>1</score>"
                        ),
                        "finish_reason": "stop",
                        "usage": {},
                    }
                raise AssertionError(stage)

        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "candidate": {
                            "final_score": 0.0,
                            "final_status": "validated_low_score",
                            "proof_solution": "A proof with a gap.",
                            "validated_critiques": [
                                {
                                    "score": 0.0,
                                    "verifier_index": 0,
                                    "evaluation": "The lower-bound lemma is missing.",
                                }
                            ],
                            "proof_refine_handoffs": [
                                {"text": VALID_HANDOFF}
                            ],
                            "proof_refine_output": [],
                            "proof_verify_output": [],
                            "proof_meta_verify_output": [],
                        }
                    }
                ),
                encoding="utf-8",
            )
            scheduler = FakeScheduler()
            candidate, details = (
                await evaluate_thinking_handoff_refinement.resume_final_refinement(
                    path=result_path,
                    question="Prove the claim.",
                    initial_parsed={
                        "proof": "A proof with a gap.",
                        "self_evaluation": "The lower-bound lemma may be missing.",
                    },
                    scheduler=scheduler,
                    cfg=pipeline_cfg(
                        thinking_budget_refine_final_round_tokens=10,
                        thinking_budget_refine_final_temperature=0.6,
                        thinking_budget_refine_visible_output_target_tokens=12_000,
                        thinking_budget_refine_visible_output_limit_tokens=12_000,
                    ),
                )
            )

        self.assertEqual(
            [stage for stage, _, _ in scheduler.calls],
            ["proof_refine", "proof_verify"],
        )
        self.assertTrue(details["verification_ran"])
        self.assertEqual(candidate["selected_verification_round"], 1)
        self.assertEqual(candidate["proof_solution"], "A repaired complete proof.")
        self.assertEqual(scheduler.calls[0][2]["temperature"], 0.6)
        self.assertEqual(
            scheduler.calls[0][2]["visible_output_limit_tokens"],
            12_000,
        )
        self.assertIn("at most 12,000 tokens", str(scheduler.calls[0][1]))

    async def test_lossless_partial_handoff_reserves_final_output_budget(self):
        class FakeScheduler:
            tokenizer = FakeTokenizer()

            def __init__(self):
                self.calls = []
                self.generation_count = 0

            @staticmethod
            def _token_ids_to_list(token_ids):
                return list(token_ids)

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
                if stage == "proof_finalize":
                    return {
                        "success": True,
                        "text": (
                            run.FINAL_PARTIAL_FORCE_TEXT
                            + "A useful reduction was found, but its final "
                            "lower-bound lemma remains unproved."
                        ),
                        "finish_reason": "stop",
                        "usage": {
                            "completion_tokens": 20,
                            "estimated_prompt_tokens": 3,
                        },
                        "server_url": "http://test",
                        "latency_s": 1.0,
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
            pipeline_cfg(
                proof_max_new_tokens=512,
                proof_generation_thinking_budgets=[480],
                thinking_budget_handoff_mode="lossless_partial",
                thinking_budget_restart_strategy="deadline_aware",
                thinking_budget_final_round_tokens=100,
            ),
        )
        candidate = result["candidate"]

        self.assertEqual(
            [stage for stage, _, _ in scheduler.calls],
            [
                "proof_generation",
                "proof_finalize",
                "proof_generation",
                "proof_verify",
            ],
        )
        restarted_call = scheduler.calls[2]
        self.assertEqual(restarted_call[2]["thinking_budget_tokens"], 100)
        self.assertEqual(
            restarted_call[2]["thinking_budget_force_text"],
            run.RESTART_FINALIZE_FORCE_TEXT,
        )
        self.assertIn(
            "Treat this as the final proof-writing attempt",
            str(restarted_call[1]),
        )
        self.assertEqual(candidate["budget_restart_count"], 1)
        self.assertEqual(
            candidate["proof_handoff_output"][0]["finish_reason"],
            "partial_passthrough",
        )
        self.assertIn(
            "A useful reduction was found",
            candidate["proof_handoffs"][0]["text"],
        )
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
