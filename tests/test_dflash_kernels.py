from __future__ import annotations

import math
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang_patches.patch_dflash_sampling import (
    SEED_MARKER,
    WORKER_SEED_MARKER,
    patch_dflash_seeded_sampling_text,
    patch_dflash_worker_text,
)

TORCH_IMPORT_ERROR = None
DFLASH_UTILS_IMPORT_ERROR = None
DFLASH_WORKER_IMPORT_ERROR = None
TRITON_ACCEPT_IMPORT_ERROR = None

try:
    import torch
except ImportError as error:
    torch = None
    TORCH_IMPORT_ERROR = error

try:
    from sglang.srt.speculative.dflash_utils import (
        compute_dflash_correct_drafts_and_bonus,
        compute_dflash_sampling_correct_drafts_and_bonus,
        is_dflash_sampling_verify_available,
    )
except ImportError as error:
    DFLASH_UTILS_IMPORT_ERROR = error
    compute_dflash_correct_drafts_and_bonus = None
    compute_dflash_sampling_correct_drafts_and_bonus = None
    is_dflash_sampling_verify_available = lambda: False

try:
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
except ImportError as error:
    DFLASH_WORKER_IMPORT_ERROR = error
    DFlashWorkerV2 = None

try:
    from sglang.srt.speculative.triton_ops.dflash_accept_bonus import (
        _compute_dflash_accept_bonus_triton_unchecked,
    )
except ImportError as error:
    TRITON_ACCEPT_IMPORT_ERROR = error
    _compute_dflash_accept_bonus_triton_unchecked = None


def greedy_reference(
    candidates: list[int], target_top1: list[int]
) -> tuple[int, int, list[int]]:
    if len(candidates) != len(target_top1) or not candidates:
        raise ValueError("candidate and target rows must have the same positive length")
    accept_len = 0
    for candidate, predicted in zip(candidates[1:], target_top1[:-1]):
        if candidate != predicted:
            break
        accept_len += 1
    bonus = target_top1[accept_len]
    packed = list(candidates[1:]) + [0]
    packed[accept_len] = bonus
    return accept_len, bonus, packed


def inverse_cdf(probs: list[float], uniform: float) -> int:
    cumulative = 0.0
    for token_id, probability in enumerate(probs):
        cumulative += probability
        if uniform <= cumulative:
            return token_id
    return len(probs) - 1


def sampling_reference(
    *,
    candidates: list[int],
    target_probs: list[list[float]],
    accept_uniforms: list[float],
    final_uniform: float,
) -> tuple[int, int]:
    block_size = len(candidates)
    if len(target_probs) != block_size:
        raise ValueError("one target distribution is required per block position")
    for proposal_index in range(1, block_size):
        distribution = target_probs[proposal_index - 1]
        proposal = candidates[proposal_index]
        if accept_uniforms[proposal_index - 1] <= distribution[proposal]:
            continue
        residual = list(distribution)
        residual[proposal] = 0.0
        total = sum(residual)
        if total <= 0:
            raise ValueError("rejected deterministic proposal has no residual mass")
        residual = [value / total for value in residual]
        return proposal_index - 1, inverse_cdf(residual, final_uniform)
    return block_size - 1, inverse_cdf(target_probs[-1], final_uniform)


def closed_form_outcome_probabilities(
    candidates: list[int], target_probs: list[list[float]]
) -> dict[tuple[int, ...], float]:
    outcomes: dict[tuple[int, ...], float] = {}
    prefix_probability = 1.0
    accepted_prefix: list[int] = []
    for proposal_index in range(1, len(candidates)):
        distribution = target_probs[proposal_index - 1]
        proposal = candidates[proposal_index]
        for token_id, probability in enumerate(distribution):
            if token_id == proposal:
                continue
            outcome = tuple(accepted_prefix + [token_id])
            outcomes[outcome] = outcomes.get(outcome, 0.0) + (
                prefix_probability * probability
            )
        prefix_probability *= distribution[proposal]
        accepted_prefix.append(proposal)

    for token_id, probability in enumerate(target_probs[-1]):
        outcome = tuple(accepted_prefix + [token_id])
        outcomes[outcome] = outcomes.get(outcome, 0.0) + (
            prefix_probability * probability
        )
    return outcomes


def filtered_probs_reference(
    logits: list[float], *, temperature: float, top_k: int, top_p: float
) -> list[float]:
    scaled = [value / temperature for value in logits]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]

    if top_k < len(weights):
        keep = set(
            sorted(
                range(len(weights)), key=weights.__getitem__, reverse=True
            )[:top_k]
        )
        weights = [
            weight if index in keep else 0.0
            for index, weight in enumerate(weights)
        ]

    total = sum(weights)
    probs = [weight / total for weight in weights]
    if top_p < 1.0:
        ranked = sorted(
            range(len(probs)), key=probs.__getitem__, reverse=True
        )
        cumulative_before = 0.0
        keep = set()
        for index in ranked:
            if cumulative_before <= top_p:
                keep.add(index)
            cumulative_before += probs[index]
        probs = [
            probability if index in keep else 0.0
            for index, probability in enumerate(probs)
        ]
        total = sum(probs)
        probs = [probability / total for probability in probs]
    return probs


class SamplingSeedPatchTransformTests(unittest.TestCase):
    def test_seeded_utils_transform_is_fail_closed_and_idempotent(self) -> None:
        source = '''from sglang.srt.layers.sampler import apply_custom_logit_processor

def compute_dflash_sampling_correct_drafts_and_bonus(
    *,
    candidates: torch.Tensor,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    max_top_k: Optional[int] = None,
):
    device = next_token_logits.device

    if uniform_samples is None:
        uniform_samples = torch.rand(
            (bs, draft_token_num), dtype=torch.float32, device=device
        )
    else:
        pass

    if uniform_samples_for_final_sampling is None:
        uniform_samples_for_final_sampling = torch.rand(
            (bs,), dtype=torch.float32, device=device
        )
    else:
        pass
'''
        patched = patch_dflash_seeded_sampling_text(source)
        self.assertIn(SEED_MARKER, patched)
        self.assertIn("verify_positions: Optional[torch.Tensor]", patched)
        self.assertIn("murmur_hash32", patched)
        self.assertIn("4294967296.0", patched)
        self.assertIn("needs_seeded_uniforms", patched)
        self.assertEqual(patch_dflash_seeded_sampling_text(patched), patched)

        with self.assertRaisesRegex(RuntimeError, "source layout changed"):
            patch_dflash_seeded_sampling_text("def unrelated(): pass\n")

    def test_worker_transform_passes_absolute_positions_once(self) -> None:
        source = '''            accept_len, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
                candidates=candidates,
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                max_top_k=draft_input.max_top_k,
            )
'''
        patched = patch_dflash_worker_text(source)
        self.assertIn(WORKER_SEED_MARKER, patched)
        self.assertIn("verify_positions=positions_2d", patched)
        self.assertEqual(patch_dflash_worker_text(patched), patched)


class ConfiguredRuntimeAvailabilityTests(unittest.TestCase):
    """The production H200 correctness run must fail, never skip, if incomplete."""

    def test_cuda_runtime_is_available(self) -> None:
        self.assertIsNone(TORCH_IMPORT_ERROR, repr(TORCH_IMPORT_ERROR))
        self.assertIsNotNone(torch)
        self.assertTrue(torch.cuda.is_available(), "CUDA is required by this suite")

    def test_dflash_sampling_verifier_is_available(self) -> None:
        self.assertIsNone(
            DFLASH_UTILS_IMPORT_ERROR, repr(DFLASH_UTILS_IMPORT_ERROR)
        )
        self.assertIsNotNone(compute_dflash_sampling_correct_drafts_and_bonus)
        self.assertTrue(
            is_dflash_sampling_verify_available(),
            "target-only speculative sampling kernel is required",
        )

    def test_worker_and_triton_accept_kernel_are_available(self) -> None:
        self.assertIsNone(
            DFLASH_WORKER_IMPORT_ERROR, repr(DFLASH_WORKER_IMPORT_ERROR)
        )
        self.assertIsNone(
            TRITON_ACCEPT_IMPORT_ERROR, repr(TRITON_ACCEPT_IMPORT_ERROR)
        )
        self.assertIsNotNone(DFlashWorkerV2)
        self.assertIsNotNone(_compute_dflash_accept_bonus_triton_unchecked)


@unittest.skipIf(torch is None, "torch/SGLang runtime is not installed")
class GreedyVerificationTests(unittest.TestCase):
    def test_all_accept_lengths_for_multiple_block_sizes(self) -> None:
        for block_size in (1, 2, 3, 8, 11, 17):
            rows = []
            targets = []
            expected_accept = []
            expected_bonus = []
            for accept_len in range(block_size):
                target = [10_000 + 100 * accept_len + i for i in range(block_size)]
                candidate = [900 + accept_len] + [0] * (block_size - 1)
                for index in range(block_size - 1):
                    candidate[index + 1] = target[index]
                if accept_len < block_size - 1:
                    candidate[accept_len + 1] = target[accept_len] + 1
                rows.append(candidate)
                targets.append(target)
                expected_accept.append(accept_len)
                expected_bonus.append(target[accept_len])

            candidates = torch.tensor(rows, dtype=torch.int64)
            target_top1 = torch.tensor(targets, dtype=torch.int64)
            accept, bonus = compute_dflash_correct_drafts_and_bonus(
                candidates=candidates,
                target_predict=target_top1,
            )
            self.assertEqual(accept.tolist(), expected_accept)
            self.assertEqual(bonus.tolist(), expected_bonus)

    def test_reference_rejects_only_at_first_mismatch(self) -> None:
        self.assertEqual(
            greedy_reference([99, 10, 999, 12], [10, 11, 12, 13]),
            (1, 11, [10, 11, 12, 0]),
        )


@unittest.skipUnless(
    torch is not None and torch.cuda.is_available(),
    "CUDA is required for Triton DFlash kernel tests",
)
class TritonAcceptBonusTests(unittest.TestCase):
    def test_kernel_matches_eager_for_every_accept_length_and_batch_shape(self) -> None:
        device = torch.device("cuda:0")
        for block_size in (1, 2, 3, 8, 11, 17):
            for batch_size in (1, 2, 7, 17, 48):
                rows = []
                targets = []
                for row in range(batch_size):
                    accept_len = row % block_size
                    target = [
                        20_000 + row * 100 + index for index in range(block_size)
                    ]
                    candidate = [700 + row] + target[:-1]
                    if accept_len < block_size - 1:
                        candidate[accept_len + 1] = target[accept_len] + 1
                    rows.append(candidate)
                    targets.append(target)

                candidates = torch.tensor(rows, dtype=torch.int64, device=device)
                target_top1 = torch.tensor(targets, dtype=torch.int64, device=device)
                prefix_lens = torch.arange(
                    101, 101 + batch_size, dtype=torch.int64, device=device
                )
                accept_out = torch.empty(batch_size, dtype=torch.int32, device=device)
                commit_out = torch.empty(batch_size, dtype=torch.int32, device=device)
                bonus_out = torch.empty(batch_size, dtype=torch.int32, device=device)
                packed_out = torch.empty(
                    (batch_size, block_size), dtype=torch.int64, device=device
                )
                new_lens_out = torch.empty(
                    batch_size, dtype=torch.int64, device=device
                )

                _compute_dflash_accept_bonus_triton_unchecked(
                    candidates=candidates,
                    target_top1=target_top1,
                    accept_lens_out=accept_out,
                    commit_lens_out=commit_out,
                    bonus_ids_out=bonus_out,
                    out_tokens_out=packed_out,
                    prefix_lens=prefix_lens,
                    new_seq_lens_out=new_lens_out,
                )
                torch.cuda.synchronize(device)

                eager_accept, eager_bonus = compute_dflash_correct_drafts_and_bonus(
                    candidates=candidates,
                    target_predict=target_top1,
                )
                self.assertEqual(accept_out.cpu().tolist(), eager_accept.cpu().tolist())
                self.assertEqual(bonus_out.cpu().tolist(), eager_bonus.cpu().tolist())
                self.assertEqual(
                    commit_out.cpu().tolist(),
                    (eager_accept + 1).cpu().tolist(),
                )
                self.assertEqual(
                    new_lens_out.cpu().tolist(),
                    (prefix_lens + eager_accept + 1).cpu().tolist(),
                )

                for row in range(batch_size):
                    expected = greedy_reference(rows[row], targets[row])[2]
                    self.assertEqual(packed_out[row].cpu().tolist(), expected)


@unittest.skipUnless(
    torch is not None
    and torch.cuda.is_available()
    and is_dflash_sampling_verify_available(),
    "CUDA target-only speculative sampling kernel is unavailable",
)
class SamplingVerificationTests(unittest.TestCase):
    @staticmethod
    def sampling_info(
        batch_size: int,
        device: torch.device,
        *,
        temperatures=None,
        top_ks=None,
        top_ps=None,
        sampling_seed=None,
    ):
        if temperatures is None:
            temperatures = torch.ones((batch_size, 1), device=device)
        if top_ks is None:
            top_ks = torch.full(
                (batch_size,), 1 << 30, dtype=torch.int32, device=device
            )
        if top_ps is None:
            top_ps = torch.ones((batch_size,), device=device)
        return SimpleNamespace(
            need_top_k_sampling=bool(
                torch.any(top_ks < (1 << 30)).item()
            ),
            need_top_p_sampling=bool(torch.any(top_ps < 1.0).item()),
            temperatures=temperatures,
            top_ks=top_ks,
            top_ps=top_ps,
            sampling_seed=sampling_seed,
        )

    @staticmethod
    def seeded_case(batch_size: int, block_size: int, device: torch.device):
        generator = torch.Generator(device=device).manual_seed(20260711)
        vocab_size = 11
        logits = torch.randn(
            (batch_size, block_size, vocab_size),
            generator=generator,
            device=device,
        )
        candidates = torch.randint(
            0,
            vocab_size,
            (batch_size, block_size),
            generator=generator,
            device=device,
        )
        seeds = torch.arange(7000, 7000 + batch_size, device=device)
        positions = torch.arange(
            100,
            100 + batch_size * block_size,
            dtype=torch.int64,
            device=device,
        ).view(batch_size, block_size)
        return logits, candidates, seeds, positions

    def test_randomized_cuda_verifier_matches_scalar_reference(self) -> None:
        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(20260710)
        batch_size = 1024
        block_size = 8
        vocab_size = 7

        raw = torch.rand(
            (batch_size, block_size, vocab_size),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        probs = raw / raw.sum(dim=-1, keepdim=True)
        logits = probs.log().reshape(batch_size * block_size, vocab_size)
        candidates = torch.randint(
            0,
            vocab_size,
            (batch_size, block_size),
            generator=generator,
            device=device,
            dtype=torch.int64,
        )
        accept_uniforms = torch.rand(
            (batch_size, block_size),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        final_uniforms = torch.rand(
            (batch_size,),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        accept, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
            candidates=candidates,
            next_token_logits=logits,
            sampling_info=self.sampling_info(batch_size, device),
            threshold_single=1.0,
            threshold_acc=1.0,
            uniform_samples=accept_uniforms,
            uniform_samples_for_final_sampling=final_uniforms,
            use_sparse_topk=False,
        )
        actual_accept = accept.cpu().tolist()
        actual_bonus = bonus.cpu().tolist()
        probs_cpu = probs.cpu().tolist()
        candidates_cpu = candidates.cpu().tolist()
        accept_uniforms_cpu = accept_uniforms.cpu().tolist()
        final_uniforms_cpu = final_uniforms.cpu().tolist()

        for row in range(batch_size):
            expected = sampling_reference(
                candidates=candidates_cpu[row],
                target_probs=probs_cpu[row],
                accept_uniforms=accept_uniforms_cpu[row],
                final_uniform=final_uniforms_cpu[row],
            )
            self.assertEqual(
                (actual_accept[row], actual_bonus[row]),
                expected,
                f"sampling mismatch at row {row}",
            )

    def test_zero_probability_proposal_is_never_accepted_at_uniform_zero(self) -> None:
        device = torch.device("cuda:0")
        probs = torch.tensor(
            [[[0.0, 0.25, 0.75], [0.2, 0.3, 0.5]]],
            dtype=torch.float32,
            device=device,
        )
        candidates = torch.tensor([[2, 0]], dtype=torch.int64, device=device)
        accept, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
            candidates=candidates,
            next_token_logits=probs.log().reshape(2, 3),
            sampling_info=self.sampling_info(1, device),
            threshold_single=1.0,
            threshold_acc=1.0,
            uniform_samples=torch.zeros((1, 2), device=device),
            uniform_samples_for_final_sampling=torch.tensor([0.4], device=device),
            use_sparse_topk=False,
        )
        self.assertEqual(accept.item(), 0)
        self.assertNotEqual(bonus.item(), 0)

    def test_tiny_positive_probability_is_not_changed_by_endpoint_guard(self) -> None:
        device = torch.device("cuda:0")
        epsilon = torch.finfo(torch.float32).eps
        proposal_probability = 0.75 * epsilon
        probs = torch.tensor(
            [
                [proposal_probability, 1.0 - proposal_probability, 0.0],
                [0.2, 0.3, 0.5],
            ],
            dtype=torch.float32,
            device=device,
        )
        candidates = torch.tensor([[2, 0]], dtype=torch.int64, device=device)
        accept, _ = compute_dflash_sampling_correct_drafts_and_bonus(
            candidates=candidates,
            next_token_logits=probs.log(),
            sampling_info=self.sampling_info(1, device),
            threshold_single=1.0,
            threshold_acc=1.0,
            uniform_samples=torch.tensor(
                [[0.5 * epsilon, 0.5]], dtype=torch.float32, device=device
            ),
            uniform_samples_for_final_sampling=torch.tensor([0.4], device=device),
            use_sparse_topk=False,
        )
        self.assertEqual(accept.item(), 1)

    def test_seeded_verifier_is_repeatable_and_global_rng_independent(self) -> None:
        device = torch.device("cuda:0")
        logits, candidates, seeds, positions = self.seeded_case(64, 8, device)
        info = self.sampling_info(64, device, sampling_seed=seeds)

        first_raw = compute_dflash_sampling_correct_drafts_and_bonus(
            candidates=candidates,
            next_token_logits=logits.reshape(64 * 8, -1),
            sampling_info=info,
            verify_positions=positions,
            threshold_single=1.0,
            threshold_acc=1.0,
            use_sparse_topk=False,
        )
        first = tuple(tensor.clone() for tensor in first_raw)
        torch.rand((4096,), device=device)
        second = compute_dflash_sampling_correct_drafts_and_bonus(
            candidates=candidates,
            next_token_logits=logits.reshape(64 * 8, -1),
            sampling_info=info,
            verify_positions=positions,
            threshold_single=1.0,
            threshold_acc=1.0,
            use_sparse_topk=False,
        )
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_seeded_verifier_is_stable_under_batch_reordering(self) -> None:
        device = torch.device("cuda:0")
        batch_size, block_size = 48, 8
        logits, candidates, seeds, positions = self.seeded_case(
            batch_size, block_size, device
        )
        info = self.sampling_info(batch_size, device, sampling_seed=seeds)
        expected_raw = compute_dflash_sampling_correct_drafts_and_bonus(
            candidates=candidates,
            next_token_logits=logits.reshape(batch_size * block_size, -1),
            sampling_info=info,
            verify_positions=positions,
            threshold_single=1.0,
            threshold_acc=1.0,
            use_sparse_topk=False,
        )
        expected = tuple(tensor.clone() for tensor in expected_raw)

        permutation = torch.randperm(batch_size, device=device)
        reordered_info = self.sampling_info(
            batch_size, device, sampling_seed=seeds[permutation]
        )
        actual = compute_dflash_sampling_correct_drafts_and_bonus(
            candidates=candidates[permutation],
            next_token_logits=logits[permutation].reshape(
                batch_size * block_size, -1
            ),
            sampling_info=reordered_info,
            verify_positions=positions[permutation],
            threshold_single=1.0,
            threshold_acc=1.0,
            use_sparse_topk=False,
        )
        inverse = torch.argsort(permutation)
        self.assertTrue(torch.equal(expected[0], actual[0][inverse]))
        self.assertTrue(torch.equal(expected[1], actual[1][inverse]))

    def test_seed_or_absolute_position_changes_stateless_stream(self) -> None:
        device = torch.device("cuda:0")
        batch_size, block_size = 256, 2
        logits = torch.zeros(
            (batch_size, block_size, 2),
            dtype=torch.float32,
            device=device,
        )
        candidates = torch.zeros(
            (batch_size, block_size), dtype=torch.int64, device=device
        )
        seeds = torch.arange(
            batch_size, dtype=torch.int64, device=device
        ) + 13
        positions = torch.arange(
            500,
            500 + batch_size * block_size,
            dtype=torch.int64,
            device=device,
        ).view(batch_size, block_size)

        def run(run_seeds, run_positions):
            output = compute_dflash_sampling_correct_drafts_and_bonus(
                candidates=candidates,
                next_token_logits=logits.reshape(
                    batch_size * block_size, 2
                ),
                sampling_info=self.sampling_info(
                    batch_size, device, sampling_seed=run_seeds
                ),
                verify_positions=run_positions,
                threshold_single=1.0,
                threshold_acc=1.0,
                use_sparse_topk=False,
            )
            return tuple(tensor.clone() for tensor in output)

        baseline = run(seeds, positions)
        different_seed = run(seeds + 1, positions)
        different_position = run(seeds, positions + 97)
        self.assertTrue(
            torch.any(baseline[0] != different_seed[0])
            or torch.any(baseline[1] != different_seed[1])
        )
        self.assertTrue(
            torch.any(baseline[0] != different_position[0])
            or torch.any(baseline[1] != different_position[1])
        )

    def test_seeded_verifier_requires_absolute_positions(self) -> None:
        device = torch.device("cuda:0")
        logits, candidates, seeds, _ = self.seeded_case(2, 2, device)
        with self.assertRaisesRegex(
            ValueError, "requires absolute verify_positions"
        ):
            compute_dflash_sampling_correct_drafts_and_bonus(
                candidates=candidates,
                next_token_logits=logits.reshape(4, -1),
                sampling_info=self.sampling_info(
                    2, device, sampling_seed=seeds
                ),
                threshold_single=1.0,
                threshold_acc=1.0,
                use_sparse_topk=False,
            )

    def test_fully_injected_uniforms_override_seed_without_positions(self) -> None:
        device = torch.device("cuda:0")
        logits, candidates, seeds, _ = self.seeded_case(4, 8, device)
        accept_uniforms = torch.full((4, 8), 0.41, device=device)
        final_uniforms = torch.full((4,), 0.73, device=device)
        expected_raw = compute_dflash_sampling_correct_drafts_and_bonus(
            candidates=candidates,
            next_token_logits=logits.reshape(4 * 8, -1),
            sampling_info=self.sampling_info(4, device),
            threshold_single=1.0,
            threshold_acc=1.0,
            uniform_samples=accept_uniforms.clone(),
            uniform_samples_for_final_sampling=final_uniforms.clone(),
            use_sparse_topk=False,
        )
        expected = tuple(tensor.clone() for tensor in expected_raw)
        actual = compute_dflash_sampling_correct_drafts_and_bonus(
            candidates=candidates,
            next_token_logits=logits.reshape(4 * 8, -1),
            sampling_info=self.sampling_info(
                4, device, sampling_seed=seeds
            ),
            threshold_single=1.0,
            threshold_acc=1.0,
            uniform_samples=accept_uniforms.clone(),
            uniform_samples_for_final_sampling=final_uniforms.clone(),
            use_sparse_topk=False,
        )
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(torch.equal(expected[1], actual[1]))

    def test_filters_temperature_and_block_sizes_match_reference(self) -> None:
        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(8675309)
        vocab_size = 11
        modes = ("temperature", "top_p", "top_k")
        for block_size in (1, 2, 8, 17):
            batch_size = 4
            logits = torch.randn(
                (batch_size, block_size, vocab_size),
                generator=generator,
                device=device,
            )
            candidates = torch.randint(
                0,
                vocab_size,
                (batch_size, block_size),
                generator=generator,
                device=device,
            )
            accept_uniforms = torch.full(
                (batch_size, block_size), 0.37, device=device
            )
            final_uniforms = torch.full(
                (batch_size,), 0.61, device=device
            )
            for mode in modes:
                temperatures = torch.tensor(
                    [[0.55], [0.8], [1.2], [1.75]], device=device
                )
                top_ks = torch.full(
                    (batch_size,),
                    1 << 30,
                    dtype=torch.int32,
                    device=device,
                )
                top_ps = torch.ones(batch_size, device=device)
                if mode == "top_p":
                    top_ps = torch.tensor(
                        [0.5, 0.7, 0.9, 0.95], device=device
                    )
                elif mode == "top_k":
                    top_ks = torch.tensor(
                        [2, 3, 5, 7], dtype=torch.int32, device=device
                    )

                info = self.sampling_info(
                    batch_size,
                    device,
                    temperatures=temperatures,
                    top_ks=top_ks,
                    top_ps=top_ps,
                )
                accept, bonus = (
                    compute_dflash_sampling_correct_drafts_and_bonus(
                        candidates=candidates,
                        next_token_logits=logits.reshape(
                            batch_size * block_size, vocab_size
                        ),
                        sampling_info=info,
                        max_top_k=int(top_ks.max().item()),
                        uniform_top_k_value=None,
                        threshold_single=1.0,
                        threshold_acc=1.0,
                        uniform_samples=accept_uniforms.clone(),
                        uniform_samples_for_final_sampling=(
                            final_uniforms.clone()
                        ),
                        use_sparse_topk=True,
                    )
                )
                logits_cpu = logits.cpu().tolist()
                candidates_cpu = candidates.cpu().tolist()
                for row in range(batch_size):
                    target_probs = [
                        filtered_probs_reference(
                            logits_cpu[row][position],
                            temperature=float(temperatures[row].item()),
                            top_k=min(
                                int(top_ks[row].item()), vocab_size
                            ),
                            top_p=float(top_ps[row].item()),
                        )
                        for position in range(block_size)
                    ]
                    expected = sampling_reference(
                        candidates=candidates_cpu[row],
                        target_probs=target_probs,
                        accept_uniforms=(
                            accept_uniforms[row].cpu().tolist()
                        ),
                        final_uniform=float(final_uniforms[row].item()),
                    )
                    self.assertEqual(
                        (accept[row].item(), bonus[row].item()),
                        expected,
                        f"{mode=} {block_size=} {row=}",
                    )

    def test_closed_form_joint_outcomes_sum_to_one(self) -> None:
        candidates = [6, 1, 2, 3]
        target_probs = [
            [0.05, 0.4, 0.1, 0.1, 0.1, 0.15, 0.1],
            [0.1, 0.1, 0.35, 0.15, 0.1, 0.1, 0.1],
            [0.15, 0.1, 0.1, 0.25, 0.1, 0.1, 0.2],
            [0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.2],
        ]
        outcomes = closed_form_outcome_probabilities(candidates, target_probs)
        self.assertTrue(outcomes)
        self.assertTrue(all(probability >= 0 for probability in outcomes.values()))
        self.assertTrue(math.isclose(sum(outcomes.values()), 1.0, abs_tol=1e-12))


@unittest.skipIf(
    DFlashWorkerV2 is None, "patched DFlash worker is unavailable"
)
class WorkerSamplingGuardTests(unittest.TestCase):
    def test_nonunit_thresholds_fail_closed(self) -> None:
        worker = object.__new__(DFlashWorkerV2)
        batch = SimpleNamespace(
            sampling_info=SimpleNamespace(is_all_greedy=False)
        )
        args = SimpleNamespace(
            speculative_accept_threshold_single=0.9,
            speculative_accept_threshold_acc=1.0,
        )
        with patch(
            "sglang.srt.speculative.dflash_worker_v2.get_global_server_args",
            return_value=args,
        ):
            with self.assertRaisesRegex(ValueError, "thresholds"):
                worker._validate_phase1_sampling_support(batch)

    def test_unavailable_sampling_kernel_fails_closed(self) -> None:
        worker = object.__new__(DFlashWorkerV2)
        batch = SimpleNamespace(
            sampling_info=SimpleNamespace(is_all_greedy=False)
        )
        args = SimpleNamespace(
            speculative_accept_threshold_single=1.0,
            speculative_accept_threshold_acc=1.0,
        )
        with patch(
            "sglang.srt.speculative.dflash_worker_v2.get_global_server_args",
            return_value=args,
        ), patch(
            "sglang.srt.speculative.dflash_worker_v2."
            "is_dflash_sampling_verify_available",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "refusing"):
                worker._validate_phase1_sampling_support(batch)


@unittest.skipUnless(
    torch is not None and torch.cuda.is_available() and DFlashWorkerV2 is not None,
    "CUDA and the patched DFlash worker are required",
)
class DraftRingPropertyTests(unittest.TestCase):
    def make_worker(self, *, page_size: int = 1):
        worker = object.__new__(DFlashWorkerV2)
        worker.device = torch.device("cuda:0")
        worker.draft_window_size = 512
        worker.page_size = page_size
        worker.draft_ring_size = 528
        worker._draft_ring_num_req_slots = 4
        return worker

    def test_ring_slots_wrap_without_crossing_request_regions(self) -> None:
        worker = self.make_worker()
        reqs = torch.tensor([0, 1, 3, 5], device=worker.device)
        positions = torch.tensor(
            [
                [0, 511, 527, 528, 529],
                [527, 528, 1055, 1056, 1057],
                [0, 1, 527, 528, 529],
                [0, 511, 527, 528, 529],
            ],
            device=worker.device,
        )
        slots = worker._ring_slots_2d(reqs, positions).cpu()
        expected_regions = [0, 1, 3, 1]
        for row, region in enumerate(expected_regions):
            expected = region * 528 + (positions[row].cpu() % 528)
            self.assertEqual(slots[row].tolist(), expected.tolist())

    def test_segment_gather_preserves_request_order_at_wrap(self) -> None:
        worker = self.make_worker()
        slots = worker._ring_slots_segments(
            req_pool_indices=torch.tensor([0, 2], device=worker.device),
            start=torch.tensor([526, 527], device=worker.device),
            lengths=torch.tensor([4, 3], device=worker.device),
        )
        self.assertEqual(
            slots.cpu().tolist(),
            [526, 527, 0, 1, 2 * 528 + 527, 2 * 528, 2 * 528 + 1],
        )

    def test_compact_lengths_cover_window_and_page_alignment(self) -> None:
        seq_lens = torch.tensor([0, 1, 511, 512, 513, 1025], device="cuda:0")
        unpaged = self.make_worker(page_size=1)
        self.assertEqual(
            unpaged._compute_compact_draft_seq_lens(seq_lens).cpu().tolist(),
            [0, 1, 511, 512, 512, 512],
        )

        paged = self.make_worker(page_size=256)
        compact = paged._compute_compact_draft_seq_lens(seq_lens).cpu().tolist()
        self.assertEqual(compact, [0, 1, 511, 512, 513, 513])


if __name__ == "__main__":
    unittest.main()
