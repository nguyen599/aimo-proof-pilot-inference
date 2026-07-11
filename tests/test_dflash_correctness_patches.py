from __future__ import annotations

import unittest
from array import array
from types import SimpleNamespace

from sglang_patches.patch_speculative_finish import (
    FINISH_MARKER,
    KV_MARKER,
    patch_batch_result_text,
    patch_schedule_batch_text,
)

try:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler_components.batch_result_processor import (
        _trim_dflash_finished_committed_tail,
    )
    from sglang.srt.sampling.sampling_params import SamplingParams
except ImportError:
    Req = None
    SamplingParams = None
    _trim_dflash_finished_committed_tail = None


class PatchTransformTests(unittest.TestCase):
    def test_schedule_transform_is_fail_closed_and_idempotent(self) -> None:
        source = """class Req:
    def update_finish_state(self, new_accepted_len: int = 1):
        old_behavior = True

    def reset_for_retract(self):
        pass
"""
        patched = patch_schedule_batch_text(source)
        self.assertIn(FINISH_MARKER, patched)
        self.assertEqual(patch_schedule_batch_text(patched), patched)

        with self.assertRaisesRegex(RuntimeError, "source layout changed"):
            patch_schedule_batch_text("class Req: pass\n")

    def test_batch_result_transform_adds_helper_and_call_once(self) -> None:
        source = """logger = logging.getLogger(__name__)


class Processor:
    def process(self):
            req.update_finish_state(new_accepted_len)

            self._handle_finish_state_updated_req(req, batch, result, i, logits_output)
"""
        patched = patch_batch_result_text(source)
        self.assertIn(KV_MARKER, patched)
        self.assertIn("def _trim_dflash_finished_committed_tail", patched)
        self.assertEqual(patch_batch_result_text(patched), patched)


class _Tokenizer:
    eos_token_id = 2
    additional_stop_token_ids: list[int] = []
    table = {
        2: "<eos>",
        10: "a",
        11: "b",
        12: "c",
        13: "d",
        14: "e",
        15: "f",
        20: "STOP",
    }

    def decode(self, ids) -> str:
        return "".join(self.table[int(token_id)] for token_id in ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text)))


@unittest.skipIf(Req is None, "patched SGLang runtime is not installed")
class SpeculativeFinishRegressionTests(unittest.TestCase):
    def make_req(
        self,
        output: list[int],
        *,
        max_new_tokens: int = 5,
        stop: list[str] | None = None,
        stop_regex: list[str] | None = None,
        eos_ids: set[int] | None = None,
    ):
        tokenizer = _Tokenizer()
        params = SamplingParams(
            max_new_tokens=max_new_tokens, stop=stop, stop_regex=stop_regex
        )
        params.normalize(tokenizer=tokenizer)
        req = Req(
            rid="probe",
            origin_input_text="",
            origin_input_ids=array("q", [100]),
            sampling_params=params,
            eos_token_ids=eos_ids or set(),
            vocab_size=1000,
        )
        req.tokenizer = tokenizer
        req.output_ids = array("q", output)
        return req

    def test_eos_before_length_wins_inside_speculative_chunk(self) -> None:
        req = self.make_req([10, 11, 2, 12, 13, 14, 15], eos_ids={2})
        req.update_finish_state(new_accepted_len=7)

        self.assertEqual(type(req.finished_reason).__name__, "FINISH_MATCHED_TOKEN")
        self.assertEqual(req.finished_len, 3)
        self.assertEqual(list(req.output_ids_through_stop), [10, 11, 2])

    def test_stop_string_before_length_wins_inside_speculative_chunk(self) -> None:
        req = self.make_req([10, 11, 20, 12, 13, 14, 15], stop=["STOP"])
        req.update_finish_state(new_accepted_len=7)

        self.assertEqual(type(req.finished_reason).__name__, "FINISH_MATCHED_STR")
        self.assertEqual(req.finished_len, 3)
        self.assertEqual(list(req.output_ids_through_stop), [10, 11, 20])

    def test_length_hides_eos_after_the_visible_limit(self) -> None:
        req = self.make_req([10, 11, 12, 13, 14, 2, 15], eos_ids={2})
        req.update_finish_state(new_accepted_len=7)

        self.assertEqual(type(req.finished_reason).__name__, "FINISH_LENGTH")
        self.assertEqual(req.finished_len, 5)
        self.assertEqual(list(req.output_ids_through_stop), [10, 11, 12, 13, 14])

    def test_earliest_boundary_wins_across_stop_types(self) -> None:
        req = self.make_req([10, 20, 11, 2, 12, 13], stop=["STOP"], eos_ids={2})
        req.update_finish_state(new_accepted_len=6)

        self.assertEqual(type(req.finished_reason).__name__, "FINISH_MATCHED_STR")
        self.assertEqual(req.finished_len, 2)
        self.assertEqual(list(req.output_ids_through_stop), [10, 20])

    def test_first_token_boundary_wins_even_when_stop_list_is_reversed(self) -> None:
        req = self.make_req(
            [10, 11, 12, 13],
            max_new_tokens=10,
            stop=["cd", "ab"],
        )
        req.update_finish_state(new_accepted_len=4)

        self.assertEqual(type(req.finished_reason).__name__, "FINISH_MATCHED_STR")
        self.assertEqual(req.finished_len, 2)
        self.assertEqual(list(req.output_ids_through_stop), [10, 11])

    def test_earlier_regex_wins_over_later_stop_string(self) -> None:
        req = self.make_req(
            [10, 11, 12, 13],
            max_new_tokens=10,
            stop=["cd"],
            stop_regex=["ab"],
        )
        req.update_finish_state(new_accepted_len=4)

        self.assertEqual(type(req.finished_reason).__name__, "FINISHED_MATCHED_REGEX")
        self.assertEqual(req.finished_len, 2)

    def test_existing_output_prefix_is_not_replayed(self) -> None:
        req = self.make_req(
            [10, 11, 20, 12, 13],
            max_new_tokens=10,
            stop=["STOP"],
        )
        req.update_finish_state(new_accepted_len=3)

        self.assertEqual(type(req.finished_reason).__name__, "FINISH_MATCHED_STR")
        self.assertEqual(req.finished_len, 3)
        self.assertEqual(list(req.output_ids_through_stop), [10, 11, 20])

    def test_length_wins_when_eos_is_exactly_at_max(self) -> None:
        req = self.make_req([10, 11, 2], max_new_tokens=3, eos_ids={2})
        req.update_finish_state(new_accepted_len=3)

        self.assertEqual(type(req.finished_reason).__name__, "FINISH_LENGTH")
        self.assertEqual(req.finished_len, 3)

    def test_length_wins_when_stop_string_is_exactly_at_max(self) -> None:
        req = self.make_req([10, 11], max_new_tokens=2, stop=["ab"])
        req.update_finish_state(new_accepted_len=2)

        self.assertEqual(type(req.finished_reason).__name__, "FINISH_LENGTH")
        self.assertEqual(req.finished_len, 2)

    def test_post_stop_kv_tail_becomes_overallocated(self) -> None:
        req = SimpleNamespace(
            output_ids=array("q", [10, 11, 2, 12, 13, 14, 15]),
            finished_len=3,
            kv_committed_len=108,
            kv_allocated_len=112,
            cache_protected_len=100,
        )
        discarded = _trim_dflash_finished_committed_tail(req, new_accepted_len=7)

        self.assertEqual(discarded, 4)
        self.assertEqual(req.kv_committed_len, 104)

    def test_kv_trim_restores_canonical_commit_for_every_block_position(self) -> None:
        prompt_len = 100
        prior_output_len = 5
        block_len = 8
        raw_output_len = prior_output_len + block_len
        committed_before_trim = prompt_len + raw_output_len - 1

        for stop_position in range(1, block_len + 1):
            with self.subTest(stop_position=stop_position):
                req = SimpleNamespace(
                    output_ids=array("q", range(raw_output_len)),
                    finished_len=prior_output_len + stop_position,
                    kv_committed_len=committed_before_trim,
                    kv_allocated_len=committed_before_trim + 8,
                    cache_protected_len=64,
                )
                discarded = _trim_dflash_finished_committed_tail(
                    req, new_accepted_len=block_len
                )
                expected = prompt_len + req.finished_len - 1
                self.assertEqual(discarded, block_len - stop_position)
                self.assertEqual(req.kv_committed_len, expected)
                self.assertGreaterEqual(
                    req.kv_committed_len, req.cache_protected_len
                )
                self.assertLessEqual(req.kv_committed_len, req.kv_allocated_len)

    def test_kv_trim_refuses_to_decommit_before_current_verify_chunk(self) -> None:
        req = SimpleNamespace(
            output_ids=array("q", range(13)),
            finished_len=2,
            kv_committed_len=112,
            kv_allocated_len=120,
            cache_protected_len=64,
        )
        with self.assertRaisesRegex(RuntimeError, "current verify chunk"):
            _trim_dflash_finished_committed_tail(req, new_accepted_len=8)

    def test_kv_trim_refuses_to_decommit_protected_prefix(self) -> None:
        req = SimpleNamespace(
            output_ids=array("q", range(8)),
            finished_len=4,
            kv_committed_len=108,
            kv_allocated_len=112,
            cache_protected_len=105,
        )
        with self.assertRaisesRegex(RuntimeError, "protected prefix"):
            _trim_dflash_finished_committed_tail(req, new_accepted_len=8)

    def test_no_kv_trim_without_a_finished_tail(self) -> None:
        req = SimpleNamespace(
            output_ids=array("q", [10, 11]),
            finished_len=2,
            kv_committed_len=103,
            kv_allocated_len=104,
            cache_protected_len=100,
        )
        self.assertEqual(
            _trim_dflash_finished_committed_tail(req, new_accepted_len=1), 0
        )
        self.assertEqual(req.kv_committed_len, 103)
