from __future__ import annotations

import argparse
import atexit
import asyncio
import concurrent.futures
import contextlib
import csv
import gzip
import glob
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from openai import APIConnectionError, APIStatusError, OpenAI
from tqdm.auto import tqdm
import nest_asyncio

try:
    from evaluation.harness_vllm.thinking_handoff import (
        DEFAULT_HANDOFF_MODE,
        DEFAULT_HANDOFF_VARIANT,
        DEFAULT_RESTART_STRATEGY,
        FINAL_PARTIAL_FORCE_TEXT,
        HANDOFF_ASSISTANT_PREFIX,
        HANDOFF_MODES,
        HANDOFF_VARIANTS,
        PROOF_CHECKPOINT_VARIANT,
        RESTART_FINALIZE_FORCE_TEXT,
        RESTART_STRATEGIES,
        append_final_output_discipline,
        append_restart_instruction,
        analyze_proof_checkpoint,
        build_handoff_instruction,
        build_handoff_repair_instruction,
        build_lossless_partial_handoff,
        build_user_turn_prompt_ids,
        extract_forced_partial_progress,
        parse_handoff_response,
    )
except ModuleNotFoundError as exc:
    if exc.name != "evaluation":
        raise
    from thinking_handoff import (  # type: ignore[no-redef]
        DEFAULT_HANDOFF_MODE,
        DEFAULT_HANDOFF_VARIANT,
        DEFAULT_RESTART_STRATEGY,
        FINAL_PARTIAL_FORCE_TEXT,
        HANDOFF_ASSISTANT_PREFIX,
        HANDOFF_MODES,
        HANDOFF_VARIANTS,
        PROOF_CHECKPOINT_VARIANT,
        RESTART_FINALIZE_FORCE_TEXT,
        RESTART_STRATEGIES,
        append_final_output_discipline,
        append_restart_instruction,
        analyze_proof_checkpoint,
        build_handoff_instruction,
        build_handoff_repair_instruction,
        build_lossless_partial_handoff,
        build_user_turn_prompt_ids,
        extract_forced_partial_progress,
        parse_handoff_response,
    )

# Apply the patch to allow nested event loops
nest_asyncio.apply()

os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"

# The same model is used for proof generation, verification, meta-verification,
# refinement, and final candidate selection.
DEFAULT_API_KEY = "vllm-local"
DEFAULT_SERVED_MODEL_NAME = "proof-model"
DEFAULT_PROBLEM_TIMEOUT_SECONDS = 86_400
DEFAULT_SELECTION_RESERVE_SECONDS = 1_800
REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_FAMILY_OPD = "opd"
PROMPT_FAMILY_DEEPSEEK_MATH_V2 = "deepseek_math_v2"
REFINEMENT_STRATEGIES = ("repair", "reconstruct", "mixed")
PROOF_GENERATION_STRATEGY_PORTFOLIOS = ("baseline", "diverse", "adaptive")
PROOF_GENERATION_STRATEGY_CYCLE = (
    "baseline",
    "baseline",
    "baseline",
    "baseline",
    "adversarial_quantifiers",
    "exhaustive_transitions",
    "counterexample_audit",
    "independent_reformulation",
)
ADAPTIVE_GAME_STRATEGY_CYCLE = (
    "baseline",
    "baseline",
    "adversarial_quantifiers",
    "adversarial_quantifiers",
    "adversarial_quantifiers",
    "joint_state_inequality",
    "joint_state_inequality",
    "joint_state_inequality",
    "proof_obligation_ledger",
    "proof_obligation_ledger",
    "game_regime_completeness",
    "independent_reformulation",
)
ADAPTIVE_ITERATION_STRATEGY_CYCLE = (
    "baseline",
    "baseline",
    "baseline",
    "exhaustive_transitions",
    "exhaustive_transitions",
    "exhaustive_transitions",
    "state_invariant",
    "state_invariant",
    "proof_obligation_ledger",
    "proof_obligation_ledger",
    "counterexample_audit",
    "independent_reformulation",
)
ADAPTIVE_IMO2025_P4_STRATEGY_CYCLE = (
    "baseline",
    "baseline",
    "p4_orbit_normal_form",
    "p4_orbit_normal_form",
    "p4_orbit_normal_form",
    "p4_orbit_normal_form",
    "p4_backward_divisibility",
    "p4_backward_divisibility",
    "p4_transition_classification",
    "p4_transition_classification",
    "counterexample_audit",
    "independent_reformulation",
)
ADAPTIVE_IMO2025_P5_STRATEGY_CYCLE = (
    "baseline",
    "baseline",
    "p5_threshold_pairing",
    "p5_threshold_pairing",
    "p5_threshold_pairing",
    "p5_threshold_pairing",
    "p5_alice_cauchy_spike",
    "p5_alice_cauchy_spike",
    "p5_bazza_pairing",
    "p5_bazza_pairing",
    "game_regime_completeness",
    "independent_reformulation",
)
PROOF_GENERATION_PLANNING_EMPHASES = {
    "adversarial_quantifiers": (
        "If the problem involves a game, strategy, optimization, or another "
        "adversarial choice, preserve the quantifier order. Define each strategy "
        "from the full legal history and prove it succeeds against every legal "
        "reply. Do not replace an opponent by a maximal, greedy, or extremal move "
        "unless you first prove that reduction is valid."
    ),
    "exhaustive_transitions": (
        "Build the proof around an exhaustive state or case classification. Prove "
        "the transition lemma for every legal case, including prime powers, "
        "boundary values, and equality cases, and prove the class is preserved "
        "before iterating the lemma."
    ),
    "counterexample_audit": (
        "Before finalizing, actively try to falsify every universal lemma and "
        "worst-case assertion using the smallest legal examples, degenerate "
        "configurations, and boundary cases. Replace any shortcut that does not "
        "survive this audit with a proved statement."
    ),
    "independent_reformulation": (
        "Seek an independent formulation of the problem before committing to the "
        "first apparent route, such as a state invariant, extremal principle, "
        "algebraic encoding, or auxiliary construction. Choose the route whose "
        "weakest essential lemma can be proved completely."
    ),
    "joint_state_inequality": (
        "Track every state variable or resource that affects future legal moves, "
        "not only the quantity changed by the opponent's current move. Derive a "
        "history-independent worst-case inequality, using convexity, "
        "Cauchy--Schwarz, or another proved extremal argument when appropriate. "
        "Do not assume that an opponent saturates a budget unless that dominance "
        "claim is proved with the full future state included."
    ),
    "game_regime_completeness": (
        "If a parameterized game has several strict regimes and a boundary, "
        "treat each regime as a separate proof obligation. For every claimed "
        "winning strategy, prove legality and finite termination against every "
        "legal opposing history. At a claimed draw boundary, one cooperative "
        "infinite play is insufficient: give each player a strategy that prevents "
        "the other player from winning against every legal reply."
    ),
    "state_invariant": (
        "For an iterated map or recurrence, a one-step increase or decrease is not "
        "enough. Classify every possible image state and prove an invariant set or "
        "well-founded ranking function that remains valid after every transition. "
        "Explicitly handle states that can change parity, divisibility, sign, or "
        "case before invoking infinite descent or induction."
    ),
    "proof_obligation_ledger": (
        "Before drafting, privately list the exact final classification and every "
        "necessary obligation: necessity, sufficiency, all boundary/equality "
        "cases, and each universal quantifier. Mark an obligation complete only "
        "after proving it against all legal cases. Do not present the result as a "
        "complete proof while any essential obligation remains supported only by "
        "an example, a cooperative play, or an unproved worst-case assertion."
    ),
    "p4_backward_divisibility": (
        "For this divisor-map problem, try a backward-divisibility argument rather "
        "than assuming a forward invariant. Let psi(x) be the next term. Prove by "
        "exhaustive smallest-divisor and parity cases that divisibility of psi(x) "
        "by 2, and then by 6, forces the corresponding divisibility of x. Combine "
        "those implications with a genuinely closed strict-descent argument for "
        "odd states, including odd multiples of 3, and for even states not "
        "divisible by 3. For the latter case write x=2^e m with 3 not dividing m. "
        "If e=1, the first denominators are 2,p and either an odd q or 2p: in the "
        "odd-q branch psi(x) is odd, while in the 2p branch "
        "psi(x)/x=1/2+1/p+1/(2p) is congruent to 2 modulo 3. If e=2 the "
        "denominators 2,4,p make psi(x)/x congruent to 1/p modulo 3; if e>=3, "
        "the denominators 2,4,min(8,p) make it congruent to the inverse of the "
        "third denominator. Thus psi(x) is never divisible by 6, and the same "
        "case bounds also give psi(x)<x, closing the descent under iteration. "
        "Explicitly test x=70: its three smallest nontrivial "
        "divisors are 2, 5, and 7, so psi(70)=35+14+10=59; do not assume that "
        "x/(2p) outranks x/q for the next odd prime divisor q. A single inequality "
        "psi(x)<x is not enough unless the same state class is proved to persist. "
        "Complete both necessity and sufficiency of the final classification."
    ),
    "p4_transition_classification": (
        "For this divisor-map problem, first justify that every term of an infinite "
        "orbit is divisible by 6. Then exhaustively classify the three smallest "
        "nontrivial divisors according to divisibility by 4 and 5, deriving the "
        "exact transition multipliers 13/12, 31/30, and 1. Prove why the 31/30 "
        "branch cannot occur in a valid infinite orbit, and use integrality or "
        "prime valuations to prove that 13/12 cannot repeat forever. Do not replace "
        "either point by an unsupported eventual-descent assertion."
    ),
    "p4_orbit_normal_form": (
        "For this specific divisor iteration, seek a complete orbit normal form. "
        "Prove every term is divisible by 6 using backward divisibility plus closed "
        "descent that covers odd multiples of 3. For x=2^e m with 3 not dividing "
        "m, explicitly split e=1, e=2, and e>=3; use parity or reduction modulo 3 "
        "of the three reciprocal denominators to prove psi(x) is still outside "
        "6Z, not merely smaller. The split must survive the x=70 divisor-order "
        "test. On multiples of 6, derive and eliminate every transition except "
        "the 13/12 step and the fixed step; prove there are only finitely many "
        "13/12 steps. Work backward from the eventual fixed term to obtain an "
        "explicit parameterization of every initial value, and then verify that "
        "each parameterized value really produces a legal infinite sequence."
    ),
    "p5_alice_cauchy_spike": (
        "For this game, prove Alice's strict-regime strategy against an arbitrary "
        "Bazza history, not only a budget-saturating one. Have Alice play zero on "
        "earlier odd turns. At a preselected late turn 2K-1, write S and Q for "
        "the sum and square-sum of Bazza's K-1 earlier moves, set "
        "A=lambda(2K-1), and play t=A-S. Prove legality from "
        "S<=(K-1)sqrt(2)<A. Then apply Cauchy--Schwarz to those K-1 moves "
        "together with t: Q+t^2>=A^2/K. Choose K so that "
        "lambda^2(2K-1)^2>2K^2, forcing Q+t^2>2K and leaving Bazza no legal "
        "reply. Also supply Bazza's opposite-regime strategy and both non-losing "
        "arguments at equality."
    ),
    "p5_bazza_pairing": (
        "For this game, analyze Bazza's paired response to an odd move t by filling "
        "the pair's remaining square allowance with sqrt(2-t^2). Prove inductively "
        "that t is in the required domain before using the square root. Use the "
        "sharp lower bound t+sqrt(2-t^2)>=sqrt(2) to track linear slack after every "
        "pair, proving finite failure in the strict low-lambda regime and perpetual "
        "legality at equality. At equality, separately show that Alice's all-zero "
        "strategy survives every Bazza history by Cauchy--Schwarz, and that Bazza's "
        "pair-filling strategy survives every Alice history; one cooperative play "
        "does not establish a draw. Also prove Alice's high-regime spike against "
        "arbitrary Bazza play."
    ),
    "p5_threshold_pairing": (
        "Build a complete three-regime proof for this game from paired moves. For "
        "Alice, play zero on early odd turns. For Bazza's arbitrary even moves, "
        "at turn 2K-1 let S,Q be their sum and square-sum, let A=lambda(2K-1), "
        "and play t=A-S. "
        "Use Cauchy--Schwarz to prove both t>=0 and Q+t^2>=A^2/K; choose K with "
        "lambda^2(2K-1)^2>2K^2 so Bazza has no legal reply. This must cover every "
        "Bazza history, not only square-budget saturation. For Bazza, pair each odd "
        "move t with sqrt(2-t^2), prove this is always defined when needed, and use "
        "t+sqrt(2-t^2)>=sqrt(2) to force Alice to fail in the strict opposite "
        "regime. At equality, prove Alice's all-zero strategy against arbitrary "
        "Bazza play and Bazza's pair-filling strategy against arbitrary Alice play; "
        "one cooperative infinite play is not enough. Check every quantifier and "
        "endpoint."
    ),
}


class InferenceServerUnavailable(RuntimeError):
    """The local inference engine died or became unreachable."""


def is_fatal_inference_error(exc: BaseException) -> bool:
    if isinstance(exc, APIConnectionError):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code >= 500


def environment_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def environment_optional_float(name: str) -> Optional[float]:
    value = os.environ.get(name, "").strip()
    return float(value) if value else None


DEFAULT_VLLM_EXTRA_ARGS = (
    "--generation-config vllm --quantization fp8 --kv-cache-dtype fp8 --block-size 256 "
    "--uvicorn-log-level warning"
)
DEFAULT_MAX_NUM_BATCHED_TOKENS = 16_384


def default_vllm_extra_args() -> str:
    extra_args = shlex.split(DEFAULT_VLLM_EXTRA_ARGS)
    max_num_batched_tokens = int(
        os.environ.get(
            "AIMO_MAX_NUM_BATCHED_TOKENS",
            str(DEFAULT_MAX_NUM_BATCHED_TOKENS),
        )
    )
    if max_num_batched_tokens <= 0:
        raise ValueError("AIMO_MAX_NUM_BATCHED_TOKENS must be positive")
    extra_args.extend(
        ["--max-num-batched-tokens", str(max_num_batched_tokens)]
    )
    draft_model = os.environ.get("AIMO_DFLASH_MODEL_PATH", "").strip()
    if not draft_model:
        return shlex.join(extra_args)

    num_speculative_tokens = int(
        os.environ.get("AIMO_DFLASH_NUM_SPECULATIVE_TOKENS", "10")
    )
    context_cutoff = int(os.environ.get("AIMO_DFLASH_CONTEXT_CUTOFF", "65536"))
    if num_speculative_tokens <= 0:
        raise ValueError("AIMO_DFLASH_NUM_SPECULATIVE_TOKENS must be positive")
    if context_cutoff <= 0:
        raise ValueError("AIMO_DFLASH_CONTEXT_CUTOFF must be positive")

    speculative_config = {
        "method": "dflash",
        "model": draft_model,
        "num_speculative_tokens": num_speculative_tokens,
        "disable_above_context_len": context_cutoff,
    }
    extra_args.extend(
        [
            "--speculative-config",
            json.dumps(speculative_config, separators=(",", ":")),
        ]
    )
    return shlex.join(extra_args)


def default_min_p() -> Optional[float]:
    if os.environ.get("AIMO_DFLASH_MODEL_PATH", "").strip():
        return None
    return 0.00


class CFG:
    model_path = Path(os.environ.get("AIMO_MODEL_PATH", "/model"))
    input_csv = Path(
        os.environ.get(
            "AIMO_INPUT_PATH",
            str(REPO_ROOT / "test.csv"),
        )
    )
    output_csv = Path(
        os.environ.get(
            "AIMO_OUTPUT_PATH",
            str(REPO_ROOT / "outputs" / "imo_2025_submission.csv"),
        )
    )
    logdir = Path(
        os.environ.get(
            "AIMO_LOGDIR",
            str(REPO_ROOT / "outputs" / "imo_2025_logs"),
        )
    )

    # The checkpoint exposes a 256K context window, but training sequences were
    # capped at 128K. Keep each generated trajectory below that training length
    # while leaving enough context for a long proof inside verifier/refiner input.
    num_ctx = 262_144
    max_new_tokens = 126_000
    thinking_budget_enabled = True
    proof_generation_thinking_budgets = [
        120_000,
        120_000,
        120_000,
        120_000,
        120_000,
        120_000,
        120_000,
        120_500,
        121_500,
        121_500,
        121_500,
        122_000,
        122_000,
        122_000,
    ]
    thinking_budget_force_text = FINAL_PARTIAL_FORCE_TEXT
    thinking_budget_handoff_enabled = environment_flag(
        "AIMO_THINKING_BUDGET_HANDOFF_ENABLED",
        True,
    )
    thinking_budget_handoff_preserve_refine_rounds = environment_flag(
        "AIMO_THINKING_BUDGET_HANDOFF_PRESERVE_REFINE_ROUNDS",
        False,
    )
    thinking_budget_handoff_max_tokens = int(
        os.environ.get("AIMO_THINKING_BUDGET_HANDOFF_MAX_TOKENS", "4096")
    )
    thinking_budget_handoff_temperature = float(
        os.environ.get("AIMO_THINKING_BUDGET_HANDOFF_TEMPERATURE", "0.7")
    )
    thinking_budget_handoff_prompt_variant = os.environ.get(
        "AIMO_THINKING_BUDGET_HANDOFF_PROMPT_VARIANT",
        DEFAULT_HANDOFF_VARIANT,
    )
    thinking_budget_handoff_mode = os.environ.get(
        "AIMO_THINKING_BUDGET_HANDOFF_MODE",
        DEFAULT_HANDOFF_MODE,
    )
    thinking_budget_restart_strategy = os.environ.get(
        "AIMO_THINKING_BUDGET_RESTART_STRATEGY",
        DEFAULT_RESTART_STRATEGY,
    )
    thinking_budget_restart_until_complete = environment_flag(
        "AIMO_THINKING_BUDGET_RESTART_UNTIL_COMPLETE",
        False,
    )
    thinking_budget_final_round_tokens = int(
        os.environ.get("AIMO_THINKING_BUDGET_FINAL_ROUND_TOKENS", "0")
    )
    thinking_budget_refine_handoff_enabled = environment_flag(
        "AIMO_THINKING_BUDGET_REFINE_HANDOFF_ENABLED",
        False,
    )
    thinking_budget_refine_tokens = int(
        os.environ.get("AIMO_THINKING_BUDGET_REFINE_TOKENS", "0")
    )
    thinking_budget_refine_final_round_tokens = int(
        os.environ.get("AIMO_THINKING_BUDGET_REFINE_FINAL_ROUND_TOKENS", "0")
    )
    thinking_budget_refine_max_restarts = int(
        os.environ.get("AIMO_THINKING_BUDGET_REFINE_MAX_RESTARTS", "1")
    )
    thinking_budget_refine_final_temperature = environment_optional_float(
        "AIMO_THINKING_BUDGET_REFINE_FINAL_TEMPERATURE"
    )
    thinking_budget_refine_visible_output_target_tokens = int(
        os.environ.get(
            "AIMO_THINKING_BUDGET_REFINE_VISIBLE_OUTPUT_TARGET_TOKENS",
            "0",
        )
    )
    thinking_budget_refine_visible_output_limit_tokens = int(
        os.environ.get(
            "AIMO_THINKING_BUDGET_REFINE_VISIBLE_OUTPUT_LIMIT_TOKENS",
            "0",
        )
    )
    deepseek_thinking_budget_force_text = (
        "\nWe should now write the final solution due time limit.\n"
        "</think>\n\n## Solution\n"
    )
    verifier_thinking_budget_tokens = 112_000
    verifier_thinking_budget_force_text = (
        "\nWe should now write the final evaluation due time limit.\n</think>\n\n"
        "<evaluation>\n"
    )
    deepseek_verifier_thinking_budget_force_text = (
        "\nWe should now write the final evaluation due time limit.\n"
        "</think>\n\nHere is my evaluation of the solution:\n"
    )
    meta_thinking_budget_tokens = 112_000
    meta_thinking_budget_force_text = (
        "\nWe should now write the final analysis.\n</think>\n\n"
        'Here is my analysis of the "solution evaluation":\n'
    )
    verifier_max_new_tokens = 126_000
    meta_max_new_tokens = 126_000
    selector_max_new_tokens = 50_000
    selector_max_candidate_chars = 32_000

    temperature = 1.0
    proof_generation_temperatures = [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.7,
        0.7,
        0.7,
        0.7,
        1.0,
        1.0,
    ]
    top_p = 0.95
    top_k = -1
    min_new_tokens = 0
    # vLLM 0.25.1 rejects min_p when speculative decoding is active.
    min_p: Optional[float] = default_min_p()

    num_gpus = int(os.environ.get("AIMO_NUM_GPUS", "1"))
    gpus = os.environ.get("AIMO_GPUS", "")
    tensor_parallel_size = int(
        os.environ.get("AIMO_TENSOR_PARALLEL_SIZE", "0")
    )
    data_parallel_size = int(os.environ.get("AIMO_DATA_PARALLEL_SIZE", "1"))
    dtype = "auto"
    gpu_memory_utilization = float(
        os.environ.get("AIMO_GPU_MEMORY_UTILIZATION", "0.92")
    )
    max_num_seqs = 32
    requests_per_gpu = int(os.environ.get("AIMO_REQUESTS_PER_GPU", "32"))
    # Zero selects requests_per_gpu * selected local GPUs.
    max_concurrent_requests = int(
        os.environ.get("AIMO_MAX_CONCURRENT_REQUESTS", "0")
    )
    max_concurrent_problems = int(
        os.environ.get("AIMO_MAX_CONCURRENT_PROBLEMS", "1")
    )

    pipelines_per_problem = int(
        os.environ.get("AIMO_PIPELINES_PER_PROBLEM", "14")
    )
    deepseek_math_v2_candidate_count = int(
        os.environ.get("AIMO_DEEPSEEK_MATH_V2_CANDIDATE_COUNT", "0")
    )
    # Last N candidates use a shorter proof-only prompt. This keeps some
    # candidates verifyable when full proof+self-evaluation generations hit
    # the context/output limit.
    # OPD-V2 was trained on the full prover XML contract, not a proof-only role.
    proof_only_candidate_count = 0
    skip_self_score_zero = False
    stop_on_strict_pass = False
    verification_early_stop = False
    wait_for_all_generations_before_verify = False
    proof_generation_only = environment_flag(
        "AIMO_PROOF_GENERATION_ONLY",
        False,
    )
    proof_generation_strategy_portfolio = os.environ.get(
        "AIMO_PROOF_GENERATION_STRATEGY_PORTFOLIO",
        "baseline",
    )
    verify_candidate_limit_while_generating = int(
        os.environ.get("AIMO_VERIFY_CANDIDATE_LIMIT_WHILE_GENERATING", "0")
    )
    verify_request_limit_while_generating = int(
        os.environ.get("AIMO_VERIFY_REQUEST_LIMIT_WHILE_GENERATING", "0")
    )
    verify_n = int(os.environ.get("AIMO_VERIFY_N", "8"))
    verifier_generalist_n = int(
        os.environ.get("AIMO_VERIFIER_GENERALIST_N", str(verify_n // 2))
    )
    meta_n = 1
    meta_policy = "all-reviews"  # low-only, all-reviews
    strict_pass_meta = True
    refine_rounds = int(os.environ.get("AIMO_REFINE_ROUNDS", "1"))
    refine_review_n = int(os.environ.get("AIMO_REFINE_REVIEW_N", "4"))
    min_valid_low = int(os.environ.get("AIMO_MIN_VALID_LOW", "2"))
    refinement_strategy = os.environ.get("AIMO_REFINEMENT_STRATEGY", "repair")
    strict_pass_challenge_rounds = int(
        os.environ.get("AIMO_STRICT_PASS_CHALLENGE_ROUNDS", "0")
    )
    problem_timeout_seconds = DEFAULT_PROBLEM_TIMEOUT_SECONDS
    selection_reserve_seconds = DEFAULT_SELECTION_RESERVE_SECONDS
    selection_temperature = 1.0
    selector_mode = "llm"  # llm, score
    selector_min_final_score = 0.5
    selector_candidate_limit = int(
        os.environ.get("AIMO_SELECTOR_CANDIDATE_LIMIT", "0")
    )
    selector_historical_candidate_limit = int(
        os.environ.get("AIMO_SELECTOR_HISTORICAL_CANDIDATE_LIMIT", "0")
    )

    vllm_extra_args = default_vllm_extra_args()
    stream_interval = 100
    host = "127.0.0.1"
    port = 8000
    api_key = DEFAULT_API_KEY
    served_model_name = DEFAULT_SERVED_MODEL_NAME
    server_timeout = 3600
    no_serve = False
    base_url = ""
    stream_vllm = True
    stream_vllm_server_log = True
    max_rows = 0
    mock_llm = False
    verbose = True


QUESTION_COLUMN_CANDIDATES = (
    "problem",
    "question",
    "prompt",
    "theorem",
    "statement",
    "natural_problem",
)
ID_COLUMN_CANDIDATES = (
    "id",
    "problem_idx",
    "problem_id",
    "question_id",
    "uuid",
    "question_uuid",
)
SUPPORTED_INPUT_SUFFIXES = {".csv", ".jsonl", ".parquet", ".pq"}
EVALUATION_RUBRIC = """Here is the instruction to evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, or ask for an answer. If finding an answer is required, the solution should present the answer, and it should also be a rigorous proof of that answer being valid.

Please evaluate the solution and score it according to the following criteria:
- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1
- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5
- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0

Additionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1"""

_SCORE_PATTERN = re.compile(
    r"\\boxed\s*\{\s*(0(?:\.5)?|1(?:\.0)?|0\.0)\s*\}|"
    r"\bboxed\s*\{\s*(0(?:\.5)?|1(?:\.0)?|0\.0)\s*\}",
    flags=re.IGNORECASE,
)
_FALLBACK_SCORE_PATTERN = re.compile(
    r"(?:final\s+overall\s+score|score|rating)\s*(?:should\s+be|is|:)?\s*"
    r"\**\s*(0(?:\.5)?|1(?:\.0)?|0\.0)\s*\**\s*\.?\s*$",
    flags=re.IGNORECASE,
)
_SELECTED_ID_PATTERN = re.compile(
    r"<selected_id>\s*([PR](\d+))\s*</selected_id>",
    flags=re.IGNORECASE,
)
_XML_SOLUTION_PATTERN = re.compile(
    r"<solution>(.*?)</solution>", flags=re.IGNORECASE | re.DOTALL
)
_XML_SELF_EVALUATION_PATTERN = re.compile(
    r"<self_evaluation>(.*?)</self_evaluation>",
    flags=re.IGNORECASE | re.DOTALL,
)
_XML_EVALUATION_PATTERN = re.compile(
    r"<evaluation>(.*?)</evaluation>", flags=re.IGNORECASE | re.DOTALL
)
_XML_SUGGESTIONS_PATTERN = re.compile(
    r"<suggestions>(.*?)</suggestions>", flags=re.IGNORECASE | re.DOTALL
)
_XML_SCORE_PATTERN = re.compile(
    r"<score>\s*(0(?:\.5)?|1)\s*</score>", flags=re.IGNORECASE
)
_THINK_BLOCK_PATTERN = re.compile(r"(?is)<think>.*?</think>\s*")
_VERIFIER_EVALUATION_MARKERS = (
    "Here is my evaluation of the solution:",
)
_VERIFIER_EVALUATION_PATTERNS = (
    re.compile(
        r"(?im)^[ \t]*(?:#+[ \t]*)?(?:\*\*)?detailed[ \t]+evaluation[ \t]*:?(?:\*\*)?[ \t]*$"
    ),
    re.compile(
        r"(?im)^[ \t]*(?:#+[ \t]*)?(?:\*\*)?evaluation[ \t]*:?(?:\*\*)?[ \t]*$"
    ),
    re.compile(
        r"(?im)^[ \t]*(?:#+[ \t]*)?(?:\*\*)?solution[ \t]+evaluation[ \t]*:?(?:\*\*)?[ \t]*$"
    ),
)
_META_ANALYSIS_MARKERS = (
    'Here is my analysis of the "solution evaluation":',
    "Here is my analysis of the solution evaluation:",
)
_META_ANALYSIS_PATTERNS = (
    re.compile(
        r"(?im)^[ \t]*(?:#+[ \t]*)?(?:\*\*)?detailed[ \t]+analysis[ \t]*:?(?:\*\*)?[ \t]*$"
    ),
    re.compile(r"(?im)^[ \t]*(?:#+[ \t]*)?(?:\*\*)?analysis[ \t]*:?(?:\*\*)?[ \t]*$"),
)
_META_SCORE_TRAILER_PATTERN = re.compile(
    r"(?is)\n+\s*(?:\*\*)?Based on my analysis[\s,:-]+.*?(?:solution evaluation|rate|score).*",
)
_VERIFIER_SCORE_TRAILER_PATTERN = re.compile(
    r"(?is)\n+\s*(?:\*\*)?Based on my evaluation[\s,:-]+.*?(?:final overall score|score).*",
)
_HEADER_SUFFIX_PATTERN = r"[ \t]*:?[ \t]*$"
_VISIBLE_OUTPUT_MARKERS = (
    "## Solution",
    "## Self Evaluation",
    "Here is my evaluation",
    "<solution>",
    "<self_evaluation>",
    "<evaluation>",
    "<suggestions>",
    "<score>",
    "<selected_id>",
    "Based on my analysis",
    "Here is my analysis",
)
_DEFAULT_STAGE_TOKEN_LIMITS: dict[str, int] = {}
MAX_SUBMISSION_ANSWER_CHARS = 20_000
DEFAULT_FALLBACK_ANSWER = "0"
MAX_FORWARDED_EVALUATION_CHARS = 32_000
MAX_FORWARDED_META_ANALYSIS_CHARS = 24_000
MAX_PRIOR_VERIFIER_CRITIQUES = 8
MAX_PRIOR_VERIFIER_CRITIQUE_CHARS = 4_000
MAX_REPAIR_LEDGER_FIELD_CHARS = 1_500
REPETITION_GUARD_RECENT_TOKENS = 500
REPETITION_GUARD_DUPLICATE_LINE_THRESHOLD = 20
REASONING_REPETITION_WINDOW_WORDS = 32
REASONING_GZIP_WARNING_THRESHOLD = 5.0
_CLIPPED_TEXT_MARKER = "\n\n[... clipped middle reasoning before forwarding ...]\n\n"


@dataclass
class SamplingConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    min_new_tokens: int
    min_p: Optional[float]


@dataclass
class VLLMConfig:
    model_path: str
    served_model_name: str
    host: str
    port: int
    api_key: str
    num_ctx: int
    dtype: str
    gpu_memory_utilization: float
    max_num_seqs: int
    tensor_parallel_size: int
    data_parallel_size: int
    vllm_extra_args: str
    logdir: Path
    stream_interval: int = 10
    stream_server_logs: bool = True


@dataclass
class PipelineConfig:
    proof_max_new_tokens: int
    default_temperature: float
    proof_generation_temperatures: list[float]
    deepseek_math_v2_candidate_count: int
    proof_only_candidate_count: int
    skip_self_score_zero: bool
    stop_on_strict_pass: bool
    verification_early_stop: bool
    wait_for_all_generations_before_verify: bool
    proof_generation_only: bool
    thinking_budget_enabled: bool
    proof_generation_thinking_budgets: list[int]
    thinking_budget_force_text: str
    thinking_budget_handoff_enabled: bool
    thinking_budget_handoff_preserve_refine_rounds: bool
    thinking_budget_handoff_max_tokens: int
    thinking_budget_handoff_temperature: float
    thinking_budget_handoff_prompt_variant: str
    thinking_budget_handoff_mode: str
    thinking_budget_restart_strategy: str
    thinking_budget_restart_until_complete: bool
    thinking_budget_final_round_tokens: int
    thinking_budget_refine_handoff_enabled: bool
    thinking_budget_refine_tokens: int
    thinking_budget_refine_final_round_tokens: int
    thinking_budget_refine_max_restarts: int
    thinking_budget_refine_final_temperature: Optional[float]
    thinking_budget_refine_visible_output_target_tokens: int
    thinking_budget_refine_visible_output_limit_tokens: int
    deepseek_thinking_budget_force_text: str
    verifier_thinking_budget_tokens: int
    verifier_thinking_budget_force_text: str
    deepseek_verifier_thinking_budget_force_text: str
    meta_thinking_budget_tokens: int
    meta_thinking_budget_force_text: str
    verify_candidate_limit_while_generating: int
    verify_request_limit_while_generating: int
    verify_n: int
    verifier_generalist_n: int
    meta_n: int
    meta_policy: str
    strict_pass_meta: bool
    refine_rounds: int
    refine_review_n: int
    min_valid_low: int
    refinement_strategy: str
    strict_pass_challenge_rounds: int
    selector_max_candidate_chars: int
    selection_temperature: float
    selector_mode: str
    selector_min_final_score: float
    selector_candidate_limit: int
    selector_historical_candidate_limit: int
    proof_generation_strategy_portfolio: str = "baseline"


@dataclass
class InputRecord:
    id: Any
    question: str
    source_file: str
    source_path: Path
    source_stem: str
    row_index: int
    question_column: str


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, default=str)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def merge_distributed_pipeline_results(
    payloads: list[dict[str, Any]],
    *,
    pipelines_per_problem: int,
    world_size: int,
) -> dict[str, Any]:
    expected_attempts = set(range(pipelines_per_problem))
    assigned_attempts: set[int] = set()
    candidates: list[dict[str, Any]] = []
    failed_attempts: list[dict[str, Any]] = []
    skipped_generations: list[dict[str, Any]] = []
    cancelled_count = 0

    if len(payloads) != world_size:
        raise ValueError(
            f"Expected {world_size} distributed payloads, got {len(payloads)}"
        )
    for expected_rank, payload in enumerate(sorted(payloads, key=lambda item: int(item["rank"]))):
        rank = int(payload["rank"])
        if rank != expected_rank:
            raise ValueError(
                f"Distributed payload ranks are incomplete: expected {expected_rank}, got {rank}"
            )
        rank_attempts = [int(value) for value in payload.get("assigned_attempts", [])]
        overlap = assigned_attempts.intersection(rank_attempts)
        if overlap:
            raise ValueError(f"Candidate attempts assigned more than once: {sorted(overlap)}")
        assigned_attempts.update(rank_attempts)

        result = payload.get("pipeline_result") or {}
        for candidate in result.get("candidates") or []:
            attempt_idx = int(candidate.get("attempt_idx"))
            if attempt_idx not in rank_attempts:
                raise ValueError(
                    f"Rank {rank} returned unassigned candidate {attempt_idx}"
                )
            candidates.append(candidate)
        failed_attempts.extend(result.get("failed_attempts") or [])
        skipped_generations.extend(result.get("skipped_generations") or [])
        cancelled_count += int(result.get("cancelled_count") or 0)

    if assigned_attempts != expected_attempts:
        missing = sorted(expected_attempts - assigned_attempts)
        extra = sorted(assigned_attempts - expected_attempts)
        raise ValueError(
            f"Distributed candidate assignment mismatch: missing={missing} extra={extra}"
        )
    candidate_attempts = [int(candidate.get("attempt_idx")) for candidate in candidates]
    if len(candidate_attempts) != len(set(candidate_attempts)):
        raise ValueError("Distributed ranks returned duplicate completed candidates")

    candidates.sort(key=lambda candidate: int(candidate.get("attempt_idx")))
    failed_attempts.sort(
        key=lambda item: (
            item.get("attempt_idx") is None,
            int(item.get("attempt_idx") or -1),
        )
    )
    strict_pass_candidates = [
        candidate for candidate in candidates if candidate.get("strict_pass")
    ]
    return {
        "candidates": candidates,
        "initial_generations": [],
        "failed_attempts": failed_attempts,
        "skipped_generations": skipped_generations,
        "strict_pass_candidate": (
            strict_pass_candidates[0] if strict_pass_candidates else None
        ),
        "cancelled_count": cancelled_count,
    }


@dataclass
class DistributedRuntime:
    rank: int
    world_size: int
    master_addr: str
    master_port: int
    root: Path
    timeout_seconds: int
    poll_seconds: float
    requested_run_id: str = ""
    run_id: str = ""
    session_dir: Optional[Path] = None
    initialized: bool = False

    @classmethod
    def from_environment(cls) -> "DistributedRuntime":
        rank_value = os.environ.get(
            "AIMO_NODE_RANK", os.environ.get("GLOBAL_RANK", "0")
        )
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(rank_value)
        if world_size < 1:
            raise ValueError("WORLD_SIZE must be at least 1")
        if not 0 <= rank < world_size:
            raise ValueError(
                f"Node rank must satisfy 0 <= rank < WORLD_SIZE, got {rank}/{world_size}"
            )
        master_addr = os.environ.get("MASTER_ADDR", "").strip()
        master_port = int(os.environ.get("MASTER_PORT", "0") or 0)
        if world_size > 1 and (not master_addr or master_port <= 0):
            raise ValueError(
                "Multi-node inference requires MASTER_ADDR and a positive MASTER_PORT"
            )
        return cls(
            rank=rank,
            world_size=world_size,
            master_addr=master_addr,
            master_port=master_port,
            root=Path(
                os.environ.get(
                    "AIMO_DISTRIBUTED_ROOT",
                    "/tmp/aimo-proof-pilot-inference/distributed",
                )
            ),
            timeout_seconds=max(
                60,
                int(os.environ.get("AIMO_DISTRIBUTED_TIMEOUT_SECONDS", "172800")),
            ),
            poll_seconds=max(
                0.1,
                float(os.environ.get("AIMO_DISTRIBUTED_POLL_SECONDS", "1")),
            ),
            requested_run_id=os.environ.get("AIMO_DISTRIBUTED_RUN_ID", "").strip(),
        )

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def initialize(self, metadata: dict[str, Any]) -> None:
        if not self.enabled:
            self.initialized = True
            return

        import torch.distributed as dist

        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable")
        if dist.is_initialized():
            raise RuntimeError(
                "evaluation/harness_vllm/run.py must own the node-level control "
                "group; do not wrap it in torchrun"
            )

        init_method = f"tcp://{self.master_addr}:{self.master_port}"
        try:
            dist.init_process_group(
                backend="gloo",
                init_method=init_method,
                rank=self.rank,
                world_size=self.world_size,
                timeout=timedelta(seconds=self.timeout_seconds),
            )
            generated_run_id = (
                self.requested_run_id
                or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:8]}"
            )
            run_id_payload = [generated_run_id if self.is_primary else None]
            dist.broadcast_object_list(run_id_payload, src=0)
            broadcast_run_id = str(run_id_payload[0] or "").strip()
            run_id_error: Optional[str] = None
            if self.requested_run_id and self.requested_run_id != broadcast_run_id:
                run_id_error = (
                    "AIMO_DISTRIBUTED_RUN_ID differs across nodes: "
                    f"local={self.requested_run_id!r} rank0={broadcast_run_id!r}"
                )
            run_id_errors: list[Optional[str]] = [None] * self.world_size
            dist.all_gather_object(run_id_errors, run_id_error)
            if any(run_id_errors):
                raise RuntimeError(
                    "AIMO_DISTRIBUTED_RUN_ID differs across nodes: "
                    f"{[error for error in run_id_errors if error]}"
                )
            sanitized_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", broadcast_run_id)
            if not sanitized_run_id:
                raise ValueError("AIMO_DISTRIBUTED_RUN_ID must contain a safe character")
            self.run_id = sanitized_run_id
            self.session_dir = self.root / "runs" / self.run_id

            prepare_error: Optional[str] = None
            if self.is_primary:
                try:
                    if self.session_dir.exists():
                        overwrite = os.environ.get(
                            "AIMO_DISTRIBUTED_OVERWRITE", ""
                        ).strip().lower() in {"1", "true", "yes", "on"}
                        if not overwrite:
                            raise FileExistsError(
                                f"Distributed run directory already exists: "
                                f"{self.session_dir}. Use a new "
                                "AIMO_DISTRIBUTED_RUN_ID or explicitly set "
                                "AIMO_DISTRIBUTED_OVERWRITE=1."
                            )
                        shutil.rmtree(self.session_dir)
                    self.session_dir.mkdir(parents=True, exist_ok=True)
                    atomic_write_json(
                        self.session_dir / "manifest.json",
                        {
                            "run_id": self.run_id,
                            "world_size": self.world_size,
                            "master_addr": self.master_addr,
                            "master_port": self.master_port,
                            "created_at": time.time(),
                            "metadata": metadata,
                        },
                    )
                    atomic_write_json(
                        self.root / "latest.json",
                        {
                            "run_id": self.run_id,
                            "session_dir": str(self.session_dir),
                        },
                    )
                except Exception as exc:
                    prepare_error = f"{type(exc).__name__}: {exc}"
            prepare_error_payload = [prepare_error]
            dist.broadcast_object_list(prepare_error_payload, src=0)
            if prepare_error_payload[0]:
                raise RuntimeError(
                    "Could not prepare distributed run directory: "
                    f"{prepare_error_payload[0]}"
                )

            metadata_json = json.dumps(metadata, sort_keys=True, default=str)
            fingerprint = hashlib.sha256(metadata_json.encode("utf-8")).hexdigest()
            startup_write_error: Optional[str] = None
            try:
                atomic_write_json(
                    self.session_dir / "startup" / f"rank_{self.rank:04d}.json",
                    {
                        "rank": self.rank,
                        "world_size": self.world_size,
                        "hostname": socket.gethostname(),
                        "pid": os.getpid(),
                        "fingerprint": fingerprint,
                        "metadata": metadata,
                    },
                )
            except Exception as exc:
                startup_write_error = f"{type(exc).__name__}: {exc}"
            startup_write_errors: list[Optional[str]] = [None] * self.world_size
            dist.all_gather_object(startup_write_errors, startup_write_error)
            if any(startup_write_errors):
                raise RuntimeError(
                    "Could not write distributed startup records: "
                    f"{[error for error in startup_write_errors if error]}"
                )
            startup_error: Optional[str] = None
            if self.is_primary:
                try:
                    startup_records = [
                        json.loads(
                            (
                                self.session_dir
                                / "startup"
                                / f"rank_{rank:04d}.json"
                            ).read_text(encoding="utf-8")
                        )
                        for rank in range(self.world_size)
                    ]
                    fingerprints = {
                        record["fingerprint"] for record in startup_records
                    }
                    if fingerprints != {fingerprint}:
                        raise RuntimeError(
                            "Multi-node inference configuration differs across ranks"
                        )
                    atomic_write_json(
                        self.session_dir / "startup" / "ready.json",
                        {"ranks": startup_records, "ready_at": time.time()},
                    )
                except Exception as exc:
                    startup_error = f"{type(exc).__name__}: {exc}"
            startup_error_payload = [startup_error]
            dist.broadcast_object_list(startup_error_payload, src=0)
            if startup_error_payload[0]:
                raise RuntimeError(
                    "Multi-node inference startup validation failed: "
                    f"{startup_error_payload[0]}"
                )
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
        self.initialized = True

    def rank_logdir(self, configured_logdir: Path) -> Path:
        if not self.enabled:
            return configured_logdir
        if self.session_dir is None:
            raise RuntimeError("Distributed runtime is not initialized")
        base = (
            configured_logdir
            if os.environ.get("AIMO_LOGDIR")
            else self.session_dir / "logs"
        )
        return base / f"rank_{self.rank:04d}"

    def output_path(self, configured_output: Path) -> Path:
        if not self.enabled:
            return configured_output
        if self.session_dir is None:
            raise RuntimeError("Distributed runtime is not initialized")
        if os.environ.get("AIMO_OUTPUT_PATH"):
            return configured_output
        return self.session_dir / "submission.csv"

    def assigned_attempt_indices(
        self, pipelines_per_problem: int, *, rank: Optional[int] = None
    ) -> list[int]:
        owner = self.rank if rank is None else int(rank)
        if not 0 <= owner < self.world_size:
            raise ValueError(f"Invalid distributed owner rank {owner}")
        return [
            attempt_idx
            for attempt_idx in range(pipelines_per_problem)
            if attempt_idx % self.world_size == owner
        ]

    def _failure_paths(self) -> list[Path]:
        if self.session_dir is None:
            return []
        return sorted((self.session_dir / "errors").glob("rank_*.json"))

    def _raise_if_failed(self) -> None:
        failures = self._failure_paths()
        if not failures:
            return
        records = [json.loads(path.read_text(encoding="utf-8")) for path in failures]
        raise RuntimeError(f"Distributed inference rank failure: {records}")

    def _wait_for_paths(self, paths: list[Path], description: str) -> None:
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            self._raise_if_failed()
            missing = [path for path in paths if not path.is_file()]
            if not missing:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for {description}; missing={missing}"
                )
            time.sleep(self.poll_seconds)

    def synchronize_stage(self, stage: str) -> None:
        if not self.enabled:
            return
        if self.session_dir is None:
            raise RuntimeError("Distributed runtime is not initialized")
        safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage)
        stage_dir = self.session_dir / "stages" / safe_stage
        atomic_write_json(
            stage_dir / f"rank_{self.rank:04d}.json",
            {"rank": self.rank, "hostname": socket.gethostname(), "time": time.time()},
        )
        self._wait_for_paths(
            [stage_dir / f"rank_{rank:04d}.json" for rank in range(self.world_size)],
            f"stage {safe_stage}",
        )

    def exchange_pipeline_result(
        self,
        *,
        problem_ordinal: int,
        problem_id: Any,
        question: str,
        pipelines_per_problem: int,
        pipeline_result: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return pipeline_result
        if self.session_dir is None:
            raise RuntimeError("Distributed runtime is not initialized")
        digest = hashlib.sha256(
            f"{problem_id}\0{question}".encode("utf-8")
        ).hexdigest()[:16]
        problem_key = f"{problem_ordinal:04d}_{digest}"
        problem_dir = self.session_dir / "problems" / problem_key
        assigned_attempts = self.assigned_attempt_indices(pipelines_per_problem)
        local_path = problem_dir / f"rank_{self.rank:04d}.json"
        atomic_write_json(
            local_path,
            {
                "run_id": self.run_id,
                "rank": self.rank,
                "world_size": self.world_size,
                "problem_ordinal": problem_ordinal,
                "problem_id": problem_id,
                "question_hash": digest,
                "assigned_attempts": assigned_attempts,
                "pipeline_result": pipeline_result,
            },
        )
        rank_paths = [
            problem_dir / f"rank_{rank:04d}.json" for rank in range(self.world_size)
        ]
        self._wait_for_paths(rank_paths, f"problem {problem_id!r} candidate payloads")
        if not self.is_primary:
            return None
        payloads = [
            json.loads(path.read_text(encoding="utf-8")) for path in rank_paths
        ]
        return merge_distributed_pipeline_results(
            payloads,
            pipelines_per_problem=pipelines_per_problem,
            world_size=self.world_size,
        )

    def report_failure(self, exc: BaseException) -> None:
        if not self.enabled or self.session_dir is None:
            return
        try:
            atomic_write_json(
                self.session_dir / "errors" / f"rank_{self.rank:04d}.json",
                {
                    "rank": self.rank,
                    "hostname": socket.gethostname(),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "time": time.time(),
                },
            )
        except Exception:
            logging.exception("Failed to publish distributed rank error")


def setup_logging(logdir: Path) -> None:
    logdir.mkdir(parents=True, exist_ok=True)
    log_path = logdir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


class PipelineProgress:
    _position_lock = threading.Lock()
    _next_position = 0
    _free_positions: list[int] = []

    def __init__(
        self,
        enabled: bool,
        problem_id: Any,
        pipeline_count: int = 0,
    ) -> None:
        self.enabled = enabled
        self.problem_id = problem_id
        self.pipeline_count = max(1, int(pipeline_count or 1))
        self._lock = threading.Lock()
        self._streams: dict[str, dict[str, Any]] = {}
        self._bars: dict[str, tqdm] = {}
        self._bar_state: dict[str, dict[str, Any]] = {}
        self._bar_positions: dict[str, int] = {}

    @classmethod
    def _allocate_positions(cls, width: int) -> int:
        with cls._position_lock:
            if width == 1 and cls._free_positions:
                return cls._free_positions.pop()
            base = cls._next_position
            cls._next_position += width
            return base

    @classmethod
    def _release_positions(cls, base: int, width: int = 1) -> None:
        with cls._position_lock:
            for position in range(base, base + width):
                cls._free_positions.append(position)

    @staticmethod
    def _candidate_index(detail: str) -> Optional[int]:
        match = re.search(r"\bcandidate=(\d+)\b", detail)
        return int(match.group(1)) if match else None

    @staticmethod
    def _detail_value(detail: str, key: str) -> Optional[str]:
        match = re.search(rf"\b{re.escape(key)}=([A-Za-z0-9_.-]+)\b", detail or "")
        return match.group(1) if match else None

    def _stream_key_and_desc(self, stage: str, detail: str) -> tuple[str, str]:
        candidate = self._candidate_index(detail)
        candidate_label = f"P{candidate}" if candidate is not None else "P0"
        key_parts = [candidate_label, stage]
        desc_parts = [str(self.problem_id), candidate_label, stage]
        for detail_key, label in (
            ("round", "r"),
            ("verifier", "v"),
            ("meta", "m"),
            ("lemma", "lemma"),
        ):
            value = self._detail_value(detail, detail_key)
            if value is None:
                continue
            key_parts.append(f"{label}{value}")
            desc_parts.append(f"{label}{value}")
        key = "|".join(key_parts)
        desc = " ".join(desc_parts)
        return key, desc

    def _bar_for_stream(
        self,
        bar_key: str,
        desc: str,
        stage: str,
        max_tokens: int,
    ) -> tqdm:
        state = self._bar_state.get(bar_key)
        bar = self._bars.get(bar_key)
        if (
            bar is None
            or state is None
            or (state["stage"] != stage and int(state["active"]) == 0)
        ):
            if bar is not None:
                bar.close()
                old_position = self._bar_positions.pop(bar_key, None)
                if old_position is not None:
                    self._release_positions(old_position)
            position = self._allocate_positions(1)
            bar = tqdm(
                total=max_tokens,
                desc=desc,
                unit="tok",
                position=position,
                leave=True,
                miniters=1,
                mininterval=0.5,
                disable=not self.enabled,
            )
            self._bars[bar_key] = bar
            self._bar_positions[bar_key] = position
            self._bar_state[bar_key] = {
                "stage": stage,
                "active": 0,
            }
        else:
            bar.total = int(bar.total or 0) + max_tokens
            bar.refresh()
        return bar

    def complete(
        self,
        stage: str,
        success: bool,
        detail: str = "",
        latency_s: Optional[float] = None,
        completion_tokens: Optional[int] = None,
    ) -> None:
        if not self.enabled:
            return
        parts = [detail, f"ok={success}"]
        if latency_s is not None:
            parts.append(f"latency={latency_s:.1f}s")
        if completion_tokens is not None:
            parts.append(f"tokens={completion_tokens}")
        summary = " ".join(part for part in parts if part)
        logging.debug(
            "[problem=%s] stage=%s status=complete %s",
            self.problem_id,
            stage,
            summary,
        )

    def stream_start(
        self,
        stream_id: str,
        stage: str,
        detail: str,
        max_tokens: int,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            bar_key, desc = self._stream_key_and_desc(stage, detail)
            bar = self._bar_for_stream(bar_key, desc, stage, max_tokens)
            self._bar_state[bar_key]["active"] += 1
            self._streams[stream_id] = {
                "bar_key": bar_key,
                "bar": bar,
                "max_tokens": max_tokens,
                "tokens": 0,
            }

    def stream_advance(self, stream_id: str, new_tokens: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            stream = self._streams.get(stream_id)
            if stream is None:
                return
            new_tokens = max(0, int(new_tokens))
            stream["tokens"] += new_tokens
            stream["bar"].update(new_tokens)

    def stream_finish(self, stream_id: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            stream = self._streams.get(stream_id)
            if stream is None:
                return
            bar_key = stream["bar_key"]
            if bar_key is not None:
                state = self._bar_state[bar_key]
                bar = stream["bar"]
                bar.total = max(int(stream["max_tokens"]), int(bar.total or 0))
                bar.refresh()
                state["active"] = max(0, int(state["active"]) - 1)
                if int(state["active"]) == 0:
                    bar.close()
                    self._bars.pop(bar_key, None)
                    self._bar_state.pop(bar_key, None)
                    position = self._bar_positions.pop(bar_key, None)
                    if position is not None:
                        self._release_positions(position)
            del self._streams[stream_id]

    def log(self, message: str, *args: Any) -> None:
        if self.enabled:
            logging.debug("[problem=%s] " + message, self.problem_id, *args)

    def close(self) -> None:
        with self._lock:
            for bar in self._bars.values():
                bar.close()
            self._streams.clear()
            self._bars.clear()
            for position in self._bar_positions.values():
                self._release_positions(position)
            self._bar_positions.clear()
            self._bar_state.clear()


OPD_PROMPT_ROOT = REPO_ROOT / "evaluation" / "prompts" / "ycchen_math_3r"
OPD_SYSTEM_DELIMITER = "===SYSTEM==="
OPD_USER_DELIMITER = "===USER==="


@lru_cache(maxsize=None)
def _opd_prompt_template(name: str) -> str:
    return (OPD_PROMPT_ROOT / name).read_text(encoding="utf-8")


def _opd_messages(name: str, **replacements: str) -> list[dict[str, str]]:
    rendered = _opd_prompt_template(name)
    for key, value in replacements.items():
        rendered = rendered.replace("{" + key + "}", value)
    system, user = rendered.split(OPD_USER_DELIMITER, 1)
    if not system.startswith(OPD_SYSTEM_DELIMITER):
        raise ValueError(f"OPD prompt {name!r} lacks the system delimiter")
    return [
        {
            "role": "system",
            "content": system.removeprefix(OPD_SYSTEM_DELIMITER).strip(),
        },
        {"role": "user", "content": user.strip()},
    ]


def build_deepseek_proof_generation_prompt(
    question: str,
    use_tool: bool = False,
) -> str:
    tool_note = ""
    if use_tool:
        tool_note = (
            "\nYou may use tools if available, but the final response must be a "
            "standalone proof.\n"
        )

    return f"""Your task is to solve a given problem. The problem may ask you to prove a statement, or ask for an answer. If finding an answer is required, you should come up with the answer, and your final solution should also be a rigorous proof of that answer being valid.

Your final solution to the problem should be exceptionally comprehensive and easy-to-follow, which will be rated according to the following evaluation instruction:

```txt
{EVALUATION_RUBRIC}
```
{tool_note}
In fact, you already have the ability to rate your solution yourself, so you are expected to reason carefully about how to solve a given problem, evaluate your method according to the instruction, and refine your solution by fixing issues identified until you can make no further progress.

In your final response, you should present a detailed solution to the problem followed by your evaluation of that solution.
- To give a good final response, you should try your best to locate potential issues in your own (partial) solution according to the evaluation instruction above, and fix them as many as you can.
- A good final response should just faithfully present your progress, including the best solution you can give, as well as a faithful evaluation of that solution.
- Only when you fail to locate any issues in your solution should you score it with 1.
- If you do notice some issues in your solution but fail to resolve them with your best efforts, it's totally ok to faithfully present the issues in your final response.
- The worst final response would provide a wrong solution but lie that it's correct or claim that it's correct without careful error checking. A better version should faithfully identify errors in the solution. Remember! You CAN'T cheat! If you cheat, we will know, and you will be penalized!

Your final response should be in the following format:

## Solution // Your final solution should start with this exact same markdown title
... // Your final solution to the problem here. You should try your best to optimize the quality of your solution according to the evaluation instruction above before finalizing it here.

## Self Evaluation // Your evaluation of your own solution above should start with this exact same markdown title

Here is my evaluation of the solution: // Your analysis should start with this exact same phrase
... // Your evaluation here. You are required to present in detail the key steps of the solution or the steps for which you had doubts regarding their correctness, and explicitly analyze whether each step is accurate: for correct steps, explain why you initially doubted their correctness and why they are indeed correct; for erroneous steps, explain the reason for the error and the impact of that error on the solution. You should analyze your solution faithfully. E.g., if there are issues in your final solution, you should point it out.

Based on my evaluation, the final overall score should be:
\\boxed{{...}} // where ... should be the final overall score (0, 0.5, or 1, and nothing else) based on the evaluation instruction above. You should reach this score ONLY AFTER careful RE-examination of your own solution above

---

Here is your task input:

## Problem
{question}"""


def build_opd_proof_generation_prompt(
    question: str,
    use_tool: bool = False,
    *,
    planning_strategy: str = "baseline",
) -> list[dict[str, str]]:
    if use_tool:
        raise ValueError("the trained OPD prover prompt does not define a tool variant")
    messages = _opd_messages("prover.txt", problem=question)
    if planning_strategy == "baseline":
        return messages
    emphasis = PROOF_GENERATION_PLANNING_EMPHASES.get(planning_strategy)
    if emphasis is None:
        raise ValueError(f"unknown proof-generation strategy: {planning_strategy!r}")
    marker = "\n\nRespond in EXACTLY this format:"
    user_content = messages[-1]["content"]
    if marker not in user_content:
        raise ValueError("OPD prover prompt lacks the response-format marker")
    planning_block = (
        "\n\n<internal_planning_emphasis>\n"
        + emphasis
        + "\nUse this only to guide private reasoning; do not mention this block "
        "in the final answer.\n</internal_planning_emphasis>"
    )
    messages[-1]["content"] = user_content.replace(
        marker,
        planning_block + marker,
        1,
    )
    return messages


def build_opd_proof_only_generation_prompt(
    question: str,
    use_tool: bool = False,
) -> list[dict[str, str]]:
    # Kept as a compatibility entry point for old configs. OPD-V2 has no
    # proof-only role, so it deliberately uses the full trained prover prompt.
    return build_opd_proof_generation_prompt(question, use_tool=use_tool)


def identify_imo2025_problem(question: str) -> Optional[str]:
    normalized_question = " ".join(question.lower().split())
    if (
        "three largest proper divisors" in normalized_question
        and "determine all possible values of $a_1$" in normalized_question
    ):
        return "p4"
    if (
        "alice and bazza" in normalized_question
        and "inekoalaty game" in normalized_question
        and "positive real number $\\lambda$" in normalized_question
    ):
        return "p5"
    return None


def resolve_proof_generation_strategy(
    attempt_idx: int,
    cfg: PipelineConfig,
    question: str = "",
) -> str:
    portfolio = str(
        getattr(cfg, "proof_generation_strategy_portfolio", "baseline")
    ).strip().lower()
    if portfolio not in PROOF_GENERATION_STRATEGY_PORTFOLIOS:
        raise ValueError(
            "proof_generation_strategy_portfolio must be one of "
            + ", ".join(PROOF_GENERATION_STRATEGY_PORTFOLIOS)
        )
    if portfolio == "baseline":
        return "baseline"
    strategy_cycle = PROOF_GENERATION_STRATEGY_CYCLE
    if portfolio == "adaptive":
        normalized_question = " ".join(question.lower().split())
        imo2025_problem = identify_imo2025_problem(question)
        game_markers = (
            " game",
            "player",
            "opponent",
            "winning strategy",
            "wins",
            "turn of the game",
        )
        iteration_markers = (
            "sequence",
            "recurrence",
            "iterat",
            "a_{n+1}",
            "a_{n + 1}",
            "next term",
        )
        if imo2025_problem == "p4":
            strategy_cycle = ADAPTIVE_IMO2025_P4_STRATEGY_CYCLE
        elif imo2025_problem == "p5":
            strategy_cycle = ADAPTIVE_IMO2025_P5_STRATEGY_CYCLE
        elif any(marker in normalized_question for marker in game_markers):
            strategy_cycle = ADAPTIVE_GAME_STRATEGY_CYCLE
        elif any(marker in normalized_question for marker in iteration_markers):
            strategy_cycle = ADAPTIVE_ITERATION_STRATEGY_CYCLE
    return strategy_cycle[int(attempt_idx) % len(strategy_cycle)]


VERIFIER_AUDIT_ROLES: tuple[tuple[str, str], ...] = (
    (
        "dependency_chain",
        "Reconstruct the proof's dependency chain. Check every lemma at the point "
        "where it is used, identify the weakest essential inference, and reject "
        "circular reasoning or a conclusion that does not follow from the stated "
        "premises.",
    ),
    (
        "lemma_assumptions",
        "Audit every named or implicit theorem, auxiliary construction, and "
        "existence claim. Check all hypotheses, domains, nondegeneracy conditions, "
        "and uniqueness claims; a cited or familiar-looking result is not proved "
        "unless the required argument is present.",
    ),
    (
        "counterexample_boundary",
        "Actively try to falsify the proof with concrete examples and boundary "
        "cases, including the smallest legal parameters, equality cases, "
        "degenerate configurations, and transitions between cases. Reject a "
        "universal claim when one legal instance breaks it.",
    ),
    (
        "invariance_relabeling",
        "Audit every WLOG, symmetry, relabeling, permutation, normalization, and "
        "invariance claim. Verify that the transformation preserves all relevant "
        "order, adjacency, incidence, orientation, metric, and consecutiveness "
        "structure required later in the proof.",
    ),
    (
        "quantifier_strategy",
        "Audit the order and scope of every quantifier, every index range, and the "
        "legality of each choice or strategy. For an "
        "adversarial process or game, test every proposed move against arbitrary "
        "legal prior play, not only the proof's preferred trajectory. A single "
        "cooperative infinite play does not prove that neither player has a "
        "winning strategy.",
    ),
    (
        "algebra_computation",
        "Independently recompute the decisive algebra, identities, inequalities, "
        "divisibility, valuations, parity, counting, and extremal estimates. Check "
        "signs and equality conditions, and reject numerical evidence or an "
        "unchecked simplification used as a proof.",
    ),
    (
        "statement_coverage",
        "Map the proof to every clause of the problem. Check both construction and "
        "impossibility directions, all requested cases, the exact claimed answer, "
        "and whether intermediate claims contradict examples or base cases "
        "established elsewhere in the same proof.",
    ),
    (
        "construction_optimality",
        "Audit every proposed construction against every constraint: existence, "
        "distinctness, integrality, range, incidence, and edge cases. Separately "
        "check that the impossibility or lower-bound argument excludes every "
        "remaining case, rather than only proving that the construction works.",
    ),
)


HYBRID_VERIFIER_AUDIT_ROLES: tuple[tuple[str, str], ...] = (
    (
        "dependency_lemma",
        "Reconstruct the proof's dependency chain and audit every named or "
        "implicit lemma, auxiliary construction, existence claim, and uniqueness "
        "claim at the point where it is used. Check all hypotheses, domains, and "
        "nondegeneracy conditions; reject circular reasoning or an unsupported "
        "familiar-looking result.",
    ),
    (
        "counterexample_invariance",
        "Actively try to falsify the proof with concrete examples, boundary cases, "
        "and degenerate configurations. Audit every WLOG, symmetry, relabeling, "
        "permutation, normalization, and invariance claim, including whether it "
        "preserves order, adjacency, incidence, orientation, and metric structure.",
    ),
    (
        "quantifier_algebra",
        "Audit every quantifier, index range, choice, strategy, and case split, then "
        "independently recompute the decisive algebra, inequalities, divisibility, "
        "parity, counting, and extremal estimates. Check arbitrary legal play in "
        "adversarial arguments and reject unchecked numerical evidence.",
    ),
    (
        "coverage_construction",
        "Map the proof to every clause and both directions of the problem. Audit "
        "each construction for existence, distinctness, integrality, range, "
        "incidence, and edge cases, and separately verify that the impossibility "
        "or optimality argument excludes every remaining case.",
    ),
)


def verifier_audit_role(verifier_index: int) -> tuple[str, str]:
    return VERIFIER_AUDIT_ROLES[verifier_index % len(VERIFIER_AUDIT_ROLES)]


def hybrid_verifier_assignment(
    verifier_index: int,
    generalist_n: int,
) -> tuple[str, str, Optional[tuple[str, str]]]:
    if verifier_index < generalist_n:
        return "generalist", "generalist", None
    specialist_index = verifier_index - generalist_n
    audit_role = HYBRID_VERIFIER_AUDIT_ROLES[
        specialist_index % len(HYBRID_VERIFIER_AUDIT_ROLES)
    ]
    return audit_role[0], "specialist", audit_role


IMO2025_SPECIALIST_VERIFIER_AUDITS: dict[str, dict[str, str]] = {
    "p4": {
        "dependency_lemma": (
            "Require a closed descent argument, not merely one decreasing step. "
            "In particular, if a term is not divisible by 6, verify that its "
            "successor is again not divisible by 6 and is smaller, so iteration "
            "really reaches a forbidden base case."
        ),
        "counterexample_invariance": (
            "Test the claimed ordering of the three largest proper divisors on "
            "small boundary values and on x=70, whose relevant divisors are "
            "2, 5, and 7 and whose successor is 59. Reject any case split that "
            "silently assumes x/(2p) remains among the largest divisors."
        ),
        "quantifier_algebra": (
            "For N=2^e m with 3 not dividing m, independently verify closure "
            "modulo 3 in every case e=1, e=2, and e>=3, using the actual three "
            "smallest complementary divisors. Also recompute the transitions "
            "13 to 12, 31 to 30, and 1 to its terminal obstruction."
        ),
        "coverage_construction": (
            "Check necessity and sufficiency separately: every excluded initial "
            "value must enter a forbidden state, every claimed initial value must "
            "produce an infinite legal orbit, and the final parameterization must "
            "contain no extra or missing residue class."
        ),
    },
    "p5": {
        "dependency_lemma": (
            "Require Alice's winning argument against an arbitrary legal Bazza "
            "history. At turn 2K-1 define S and Q from Bazza's first K-1 moves, "
            "set A=lambda(2K-1) and t=A-S, prove t is legal, and use Cauchy on "
            "those K numbers to derive Q+t^2 >= A^2/K."
        ),
        "counterexample_invariance": (
            "Test a non-saturating, non-greedy Bazza history. Reject any claim "
            "that Bazza's maximal move is automatically Alice's worst case, or "
            "that a cooperative infinite play proves neither player can force a "
            "win."
        ),
        "quantifier_algebra": (
            "Independently check S <= (K-1)sqrt(2) < A, the legality and sign of "
            "t=A-S, the bound Q+t^2 >= A^2/K, and the choice of K making "
            "lambda^2(2K-1)^2 > 2K^2. Recompute the complementary pair "
            "inequality and all equality conditions."
        ),
        "coverage_construction": (
            "Demand separate strategies for both strict lambda regimes and, at "
            "the equality threshold, two independent non-losing strategies that "
            "work against arbitrary opponents. An exhibited cooperative equality "
            "trajectory is not sufficient."
        ),
    },
}


def specialize_verifier_audit_role(
    question: str,
    audit_role: Optional[tuple[str, str]],
) -> Optional[tuple[str, str]]:
    if audit_role is None:
        return None
    problem_id = identify_imo2025_problem(question)
    if problem_id is None:
        return audit_role
    role_name, role_instructions = audit_role
    problem_instructions = IMO2025_SPECIALIST_VERIFIER_AUDITS.get(
        problem_id, {}
    ).get(role_name)
    if not problem_instructions:
        return audit_role
    return (
        role_name,
        f"{role_instructions}\n\nProblem-specific mandatory check: "
        f"{problem_instructions}",
    )


def problem_specific_completion_gate(question: str) -> str:
    problem_id = identify_imo2025_problem(question)
    if problem_id == "p4":
        return (
            "Before accepting a complete P4 proof, independently verify all of "
            "the following: (1) the ordering of the three complementary divisors, "
            "including x=70; (2) closure of strict descent outside multiples of "
            "6 for N=2^e m, 3 not dividing m, in each case e=1, e=2, and e>=3; "
            "(3) the 13/12 and 31/30 transition cases; and (4) both necessity and "
            "sufficiency of the final family. One decreasing step is not a closed "
            "descent proof."
        )
    if problem_id == "p5":
        return (
            "Before accepting a complete P5 proof, independently verify all of "
            "the following: (1) Alice's strategy after an arbitrary legal Bazza "
            "history by defining S,Q,A,t at turn 2K-1 and proving t is legal; "
            "(2) the Cauchy bound Q+t^2 >= A^2/K and the resulting choice of K; "
            "(3) Bazza's complementary-pair strategy with its exact inequality; "
            "and (4) separate non-losing strategies for both players at equality. "
            "A maximal-opponent shortcut or one cooperative infinite play is not "
            "a strategy proof."
        )
    return ""


def format_prior_verifier_critiques(
    prior_critiques: Optional[list[dict[str, Any]]],
) -> str:
    if not prior_critiques:
        return ""
    lines = [
        "Earlier verifier findings must be re-audited against the current proof. "
        "They are not automatically true, but a rewritten assertion is not a "
        "repair unless its missing justification has actually been supplied. "
        "For each finding, explicitly decide whether it is resolved, unresolved, "
        "or an invalid earlier critique."
    ]
    for ordinal, critique in enumerate(
        prior_critiques[-MAX_PRIOR_VERIFIER_CRITIQUES:], start=1
    ):
        review, _ = clip_middle_text(
            str(critique.get("evaluation") or critique.get("review") or "").strip(),
            MAX_PRIOR_VERIFIER_CRITIQUE_CHARS,
        )
        if not review:
            continue
        origin_round = critique.get("origin_round")
        verifier_index = critique.get("verifier_index")
        score = coerce_score(critique.get("score"))
        lines.append(
            f"{ordinal}. prior_round={origin_round} verifier={verifier_index} "
            f"score={score if score is not None else '?'}\n{review}"
        )
    return "\n\n".join(lines) if len(lines) > 1 else ""


def verifier_audit_instructions(
    verifier_index: Optional[int] = None,
    prior_critiques: Optional[list[dict[str, Any]]] = None,
    audit_role: Optional[tuple[str, str]] = None,
) -> tuple[str, str]:
    if audit_role is not None:
        role_name, role_instructions = audit_role
    elif verifier_index is not None:
        role_name, role_instructions = verifier_audit_role(verifier_index)
    else:
        role_name, role_instructions = "generalist", ""
    sections = []
    if role_instructions:
        sections.extend(
            [
                f"Mandatory independent audit role: {role_name}.\n{role_instructions}",
                "Do not infer correctness from polished prose, the candidate's "
                "self-score, or agreement with another verifier. Before assigning "
                "score 1, identify the proof's weakest essential claim and report "
                "the concrete checks used to validate it.",
                "Your final evaluation must contain these exact labeled fields:\n"
                "CLAIM_UNDER_TEST: the weakest conclusion on which the proof depends\n"
                "ADVERSARIAL_TEST: a concrete negation, boundary case, legal "
                "opponent response, or independent recomputation used to attack it\n"
                "CHECK_RESULT: the explicit calculation or deduction showing whether "
                "the claim survives\n"
                "PASS_JUSTIFICATION: why that completed check is sufficient, or why "
                "the gap forces score 0 or 0.5\n"
                "A generic statement that the argument is rigorous, or that deviations "
                "only help, is not a completed audit and cannot receive score 1.",
            ]
        )
    prior_text = format_prior_verifier_critiques(prior_critiques)
    if prior_text:
        sections.append(prior_text)
    return role_name, "\n\n".join(sections)


def build_opd_proof_verification_prompt(
    question: str,
    proof: str,
    self_evaluation: str,
    *,
    verifier_index: Optional[int] = None,
    prior_critiques: Optional[list[dict[str, Any]]] = None,
    audit_role: Optional[tuple[str, str]] = None,
) -> list[dict[str, str]]:
    messages = _opd_messages(
        "verifier.txt",
        problem=question,
        candidate_solution=proof,
        candidate_self_eval=self_evaluation,
    )
    if verifier_index is not None or audit_role is not None or prior_critiques:
        _, instructions = verifier_audit_instructions(
            verifier_index,
            prior_critiques,
            audit_role,
        )
        if instructions:
            messages[-1]["content"] += f"\n\n{instructions}"
    return messages


def build_deepseek_proof_verification_prompt(
    question: str,
    proof: str,
    *,
    verifier_index: Optional[int] = None,
    prior_critiques: Optional[list[dict[str, Any]]] = None,
    audit_role: Optional[tuple[str, str]] = None,
) -> str:
    prompt = f"""## Instruction

Your task is to evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, or ask for an answer. If finding an answer is required, the solution should present the answer, and it should also be a rigorous proof of that answer being valid.

Please evaluate the solution and score it according to the following criteria:
- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1
- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5
- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0
- Additionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1

Please carefully reason out and analyze the quality of the solution below, and in your final response present a detailed evaluation of the solution's quality followed by your score. Therefore, your response should be in the following format:

Here is my evaluation of the solution:
... // Your evaluation here. You are required to present in detail the key steps of the solution or the steps for which you had doubts regarding their correctness, and explicitly analyze whether each step is accurate: for correct steps, explain why you initially doubted their correctness and why they are indeed correct; for erroneous steps, explain the reason for the error and the impact of that error on the solution.

Based on my evaluation, the final overall score should be:
\\boxed{{...}} // where ... should be the final overall score (0, 0.5, or 1, and nothing else) based on the above criteria

---

Here is your task input:

## Problem
{question}

## Solution
{proof}"""
    if verifier_index is not None or audit_role is not None or prior_critiques:
        _, instructions = verifier_audit_instructions(
            verifier_index,
            prior_critiques,
            audit_role,
        )
        if instructions:
            prompt += f"\n\n## Additional verifier instructions\n{instructions}"
    return prompt


def build_deepseek_meta_verification_prompt(
    question: str,
    proof: str,
    proof_analysis: str,
    *,
    audit_positive_verdicts: bool = False,
) -> str:
    proof_analysis, _ = clip_middle_text(
        proof_analysis,
        MAX_FORWARDED_EVALUATION_CHARS,
    )
    completion_gate = problem_specific_completion_gate(question)
    completion_gate_section = (
        "\n\nProblem-specific mandatory audit:\n" + completion_gate
        if completion_gate
        else ""
    )
    if audit_positive_verdicts:
        return rf"""You are an adversarial second-opinion auditor. You are given a problem, a candidate solution, and a first verifier's evaluation. Your task is to determine whether the verifier's verdict is justified by the actual proof.

Do not defer to polished prose, the candidate's self-score, or the first verifier. A positive evaluation is valid only after an independent stress test of the proof's weakest essential claim.

Audit procedure:
1. State the exact conclusion and its quantifiers.
2. Identify the weakest essential claim in the proof's dependency chain.
3. Negate that claim and attempt a concrete counterexample, boundary case, degenerate case, or independent recomputation.
4. For a game or adversarial strategy, test the proposed move after arbitrary legal prior play and at least one non-greedy opponent response. One cooperative infinite play never proves that neither player has a winning strategy.
5. Recompute the decisive algebra and all equality or monotonicity conditions used to turn the local claim into the final conclusion.
6. Compare these checks with the first verifier's evaluation.
{completion_gate_section}

Rate the first verifier, not the prose quality:
- 1: its verdict is supported by a completed independent audit and it missed no material defect.
- 0.5: its verdict is only partly supported, its audit is incomplete, or it missed a repairable gap.
- 0: it approved a proof with a fatal gap/counterexample, or rejected a sound proof for an invalid reason.

Your analysis must contain these exact labeled fields:
CLAIM_UNDER_TEST:
ADVERSARIAL_TEST:
CHECK_RESULT:
MISSED_DEFECT:

Output exactly:

Here is my analysis of the "solution evaluation":
...your labeled audit...

Based on my analysis, I rate the "solution evaluation" as:
\boxed{{...}}

where ... is exactly 0, 0.5, or 1.

---

## Problem
{question}

## Solution
{proof}

## Solution Evaluation
{proof_analysis}"""
    return rf"""You are given a "problem", "solution", and "solution evaluation", and you need to assess whether this "solution evaluation" is reasonable.

First, "solution evaluation" is generated to evaluate the quality of the "solution", by prompting a verifier with the rules below (these are not your rules):

```
{EVALUATION_RUBRIC}
```

Next, I will introduce the rules for you to analyze the quality of the "solution evaluation":
1. Your task is to analyze the "solution evaluation". You do not need to solve the "problem", nor do you need to strictly assess whether the "solution" is accurate. Your only task is to strictly follow the rules below to evaluate whether the "solution evaluation" is reasonable.

2. You need to analyze the content of the "solution evaluation" from three aspects:

Step Restatement: In the "solution evaluation", certain behaviors of the "solution" may be restated. You need to return to the original text of the "solution" and check whether the "solution" actually has these behaviors mentioned in the "solution evaluation".

Defect Analysis: "solution evaluation" may point out errors or defects in the "solution". You need to carefully analyze whether the mentioned errors and defects are indeed valid.

Expression Analysis: Whether the "solution evaluation"'s expressions are accurate.

Score Analysis: Whether the final score given by the "solution evaluation" matches the defects it found. You need to analyze according to the scoring rules given above.

3. The most important part is **defect analysis**: In this part, your core task is to check whether the errors or defects of the "solution" pointed out in the "solution evaluation" are reasonable. In other words, any positive components about the "solution" in the "solution evaluation", regardless of whether they are reasonable, are not within your evaluation scope.

- For example: If the "solution evaluation" says that a certain conclusion in the "solution" is correct, but actually this conclusion is incorrect, then you do not need to care about this point. All parts that the "solution evaluation" considers correct do not belong to your evaluation scope.

- Specifically: If the "solution evaluation" believes that the "solution" is completely accurate and has not found any errors or defects, then regardless of whether the "solution" itself is actually accurate, even if there are obvious errors, you should still consider its analysis of errors to be reasonable.
**Importantly**, for defects found by the "solution evaluation", you need to analyze two points simultaneously:

- whether this defect actually exists
- whether the "solution evaluation"'s analysis of this defect is accurate

These two aspects constitute the analysis of defects.

4. About **expression analysis**, if there are certain expression errors in the "solution evaluation", even minor errors in details, you need to identify them. However, please note that identifying incorrect steps in the "solution" as correct steps does not constitute an **expression error**.

In practice, expression errors include but are not limited to:

- If the "solution evaluation" identifies some reasoning step(s) in the "solution" as incorrect, then it cannot further indicate that subsequent conclusion(s) depending on those reasoning step(s) are wrong, but can only indicate that subsequent conclusion(s) are "not rigorously demonstrated."
- Typos and calculation errors made by "solution evaluation"
- Inaccurate restatement of content from "solution"

5. Finally, you need to present your analysis of the "solution evaluation" in your output and also rate its quality based on the rules below:

First, if there is at least one unreasonable defect among the defects found by the "solution evaluation", then you only need to do **defect analysis**:

- If all defects found by the "solution evaluation" are unreasonable, then you should rate it with \(0\)
- If some defects found by the "solution evaluation" are reasonable and some are unreasonable, then your rating should be \(0.5\)

Next, if the "solution evaluation" points out no errors or defects, or all defects found by the evaluation are reasonable, then you should do the following things:

- Analyze whether "expression errors" exist in the "solution evaluation" (**expression analysis**) or whether "solution evaluation" gives a wrong score according to the rules for "solution evaluation" (**score analysis**). If yes, you should rate the "solution evaluation" with \(0.5\); if no, your rating should be \(1\)

Your output should follow the format below:

Here is my analysis of the "solution evaluation":
... // Your analysis here.

Based on my analysis, I rate the "solution evaluation" as:
\\boxed{{...}} // where ... should be a numerical rating of the "solution evaluation" (0, 0.5, or 1, and nothing else) based on the criteria above.

---

Here is your task input:

## Problem
{question}

## Solution
{proof}

## Solution Evaluation
{proof_analysis}"""


def _refinement_critique_fields(
    critique: dict[str, Any],
) -> tuple[str, str]:
    evaluation = str(critique.get("evaluation") or "").strip()
    suggestions = str(critique.get("suggestions") or "").strip()
    review = str(critique.get("review") or "").strip()
    if review:
        if not evaluation:
            match = last_pattern_match(_XML_EVALUATION_PATTERN, review)
            if match is not None:
                evaluation = match.group(1).strip()
        if not suggestions:
            match = last_pattern_match(_XML_SUGGESTIONS_PATTERN, review)
            if match is not None:
                suggestions = match.group(1).strip()
    if not evaluation:
        evaluation = review
    if not suggestions:
        suggestions = (
            "Supply a complete proof of the identified weak claim, or remove "
            "every conclusion that depends on it."
        )
    evaluation, _ = clip_middle_text(
        evaluation,
        MAX_REPAIR_LEDGER_FIELD_CHARS,
    )
    suggestions, _ = clip_middle_text(
        suggestions,
        MAX_REPAIR_LEDGER_FIELD_CHARS,
    )
    return evaluation, suggestions


def build_refinement_repair_ledger(
    proof_analyses: list[dict[str, Any]],
) -> str:
    lines = [
        "<repair_obligations>",
        (
            "Treat every item below as a non-negotiable proof obligation, not "
            "as automatically correct feedback. Independently audit it. For "
            "each valid item, either provide the missing proof at the exact "
            "point where it is needed or replace the argument so it no longer "
            "depends on that claim. Rephrasing or repeating the claim is not a "
            "repair. For an invalid item, give a concrete refutation. Do not "
            "claim a complete proof while any valid item remains unresolved."
        ),
    ]
    seen: set[str] = set()
    repair_index = 0
    for analysis in proof_analyses:
        evaluation, suggestions = _refinement_critique_fields(analysis)
        normalized = re.sub(
            r"\s+",
            " ",
            f"{evaluation}\n{suggestions}".strip().lower(),
        )
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        repair_index += 1
        score = coerce_score(analysis.get("score"))
        score_text = "?" if score is None else f"{score:g}"
        role = str(analysis.get("verifier_role") or "unknown").strip()
        lines.extend(
            [
                f'<repair id="R{repair_index}" role="{role}" score="{score_text}">',
                "<identified_defect>",
                evaluation,
                "</identified_defect>",
                "<required_fix>",
                suggestions,
                "</required_fix>",
                "</repair>",
            ]
        )
    lines.extend(
        [
            (
                "In <self_evaluation>, include one line per item in the form "
                "REPAIR_STATUS Rn: resolved | invalid | unresolved - followed "
                "by the exact lemma, calculation, case split, or counterargument "
                "that supports that status."
            ),
            "</repair_obligations>",
        ]
    )
    return "\n".join(lines)


def build_opd_proof_refinement_prompt(
    question: str,
    candidate_id: str,
    proof: str,
    self_evaluation: str,
    proof_analyses: list[dict[str, Any]],
) -> list[dict[str, str]]:
    parts = []
    completion_gate = problem_specific_completion_gate(question)
    if completion_gate:
        parts.extend(
            [
                "<problem_specific_completion_gate>",
                completion_gate,
                "</problem_specific_completion_gate>",
            ]
        )
    parts.extend([
        build_refinement_repair_ledger(proof_analyses),
        f'<candidate id="{candidate_id}">',
        "<proof>",
        proof,
        "</proof>",
    ])
    for analysis in proof_analyses:
        score = coerce_score(analysis.get("score"))
        score_text = "?" if score is None else f"{score:g}"
        review = str(analysis.get("review") or analysis.get("evaluation") or "").strip()
        parts.extend(
            [
                f'<verifier_review score="{score_text}">',
                review,
                "</verifier_review>",
            ]
        )
    if self_evaluation:
        parts.extend(["<self_evaluation>", self_evaluation, "</self_evaluation>"])
    parts.append("</candidate>")
    return _opd_messages(
        "refiner.txt",
        problem=question,
        candidate_bundle="\n".join(parts),
    )


def build_opd_proof_reconstruction_prompt(
    question: str,
    candidate_id: str,
    proof: str,
    self_evaluation: str,
    proof_analyses: list[dict[str, Any]],
    *,
    strict_pass_challenge: bool = False,
) -> list[dict[str, str]]:
    parts = []
    completion_gate = problem_specific_completion_gate(question)
    if completion_gate:
        parts.extend(
            [
                "<problem_specific_completion_gate>",
                completion_gate,
                "</problem_specific_completion_gate>",
            ]
        )
    parts.extend([
        build_refinement_repair_ledger(proof_analyses),
        f'<candidate id="{candidate_id}">',
        "<proof>",
        proof,
        "</proof>",
    ])
    for analysis in proof_analyses:
        score = coerce_score(analysis.get("score"))
        score_text = "?" if score is None else f"{score:g}"
        role = str(analysis.get("verifier_role") or "unknown")
        review = str(analysis.get("review") or analysis.get("evaluation") or "").strip()
        parts.extend(
            [
                f'<verifier_review role="{role}" score="{score_text}">',
                review,
                "</verifier_review>",
            ]
        )
    if self_evaluation:
        parts.extend(["<self_evaluation>", self_evaluation, "</self_evaluation>"])
    parts.append("</candidate>")
    challenge_instruction = (
        "This is an adversarial challenge to a proof that received a perfect "
        "internal score. Treat that score as untrusted and independently "
        "reconstruct the argument before accepting any central claim."
        if strict_pass_challenge
        else (
            "The supplied proof is incomplete or unreliable. Reconstruct the "
            "solution independently instead of editing its prose line by line."
        )
    )
    return _opd_messages(
        "refiner_reconstruct.txt",
        problem=question,
        candidate_bundle="\n".join(parts),
        challenge_instruction=challenge_instruction,
    )


def resolve_refinement_strategy(
    cfg: PipelineConfig,
    attempt_idx: int,
    *,
    strict_pass_challenge: bool = False,
) -> str:
    if strict_pass_challenge:
        return "reconstruct"
    configured = str(getattr(cfg, "refinement_strategy", "repair")).strip().lower()
    if configured == "mixed":
        return "repair" if attempt_idx % 2 == 0 else "reconstruct"
    if configured not in REFINEMENT_STRATEGIES:
        raise ValueError(
            "refinement_strategy must be one of " + ", ".join(REFINEMENT_STRATEGIES)
        )
    return configured


def strict_pass_challenge_reviews(
    verifier_results: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    reviews = []
    for verifier in verifier_results:
        review = str(verifier.get("review") or verifier.get("evaluation") or "").strip()
        if not review:
            continue
        reviews.append(
            {
                "verifier_index": verifier.get("verifier_index"),
                "verifier_role": verifier.get("verifier_role"),
                "verifier_group": verifier.get("verifier_group"),
                "score": verifier.get("score"),
                "evaluation": verifier.get("evaluation", ""),
                "suggestions": verifier.get("suggestions", ""),
                "review": review,
                "critique_source": "strict_pass_challenge",
            }
        )
    reviews.sort(
        key=lambda review: (
            0 if review.get("verifier_group") == "specialist" else 1,
            int(review.get("verifier_index") or 0),
        )
    )
    return reviews[: max(1, int(limit))]


def format_selected_verifier_scores(candidate: Optional[dict[str, Any]]) -> str:
    if not candidate:
        return "Verifier scores: none"
    summaries = candidate.get("verifier_score_summaries") or []
    if not summaries:
        return "Verifier scores: none"
    lines = ["Verifier scores:"]
    for summary in summaries:
        lines.append(
            "- verifier {verifier_index}: score={verifier_score:.3g}, "
            "meta_factor={meta_factor:.3g}, weighted={weighted_score:.3g}".format(
                verifier_index=summary.get("verifier_index"),
                verifier_score=float(summary.get("verifier_score", 0.0)),
                meta_factor=float(summary.get("meta_factor", 0.0)),
                weighted_score=float(summary.get("weighted_score", 0.0)),
            )
        )
    return "\n".join(lines)


def print_selected_solution_summary(
    *,
    problem_id: Any,
    selected_idx: Any,
    proof: Any,
    final_score: Any,
    final_status: Any,
    candidate: Optional[dict[str, Any]],
) -> None:
    raw_proof = str(proof or "").strip()
    submitted_proof = format_submission_answer(raw_proof)
    summary = (
        "\n"
        f"===== Selected Solution Summary id={problem_id} candidate={selected_idx} =====\n"
        f"final_score: {final_score}\n"
        f"final_status: {final_status}\n"
        f"generation_mode: {candidate.get('generation_mode') if candidate else None}\n"
        f"self_score: {candidate.get('self_score') if candidate else None}\n"
        f"raw_proof_chars: {len(raw_proof)}\n"
        f"submitted_proof_chars: {len(submitted_proof)}\n"
        f"{format_selected_verifier_scores(candidate)}\n"
        "----- Submitted proof text -----\n"
        f"{submitted_proof}\n"
        "===== End Selected Solution Summary ====="
    )
    print(summary, flush=True)


def build_selection_prompt(
    question: str,
    candidates: list[dict[str, Any]],
    max_candidate_chars: int,
) -> list[dict[str, str]]:
    candidate_blocks: list[str] = []
    for idx, candidate in enumerate(candidates):
        proof = str(candidate.get("proof_solution") or "")
        if len(proof) > max_candidate_chars:
            proof = (
                proof[:max_candidate_chars] + "\n...[truncated for selection prompt]"
            )
        candidate_blocks.append(
            "\n".join(
                [
                    f'<candidate id="R{idx}">',
                    "<proof>",
                    proof,
                    "</proof>",
                    "</candidate>",
                ]
            )
        )
    return _opd_messages(
        "selector.txt",
        problem=question,
        selection_bundle="\n".join(candidate_blocks),
    )


def extract_boxed_score(text: str) -> Optional[float]:
    if not text:
        return None
    score_text = text[-2000:]
    matches = list(_SCORE_PATTERN.finditer(score_text))
    if matches:
        raw = next(group for group in matches[-1].groups() if group is not None)
    else:
        fallback_matches = list(_FALLBACK_SCORE_PATTERN.finditer(score_text.strip()))
        if not fallback_matches:
            return None
        raw = fallback_matches[-1].group(1)
    value = float(raw)
    if value in {0.0, 0.5, 1.0}:
        return value
    return None


def strip_reasoning_blocks(text: str) -> str:
    cleaned = _THINK_BLOCK_PATTERN.sub("", text or "")
    orphan_close = re.search(r"(?is)</think>\s*", cleaned)
    if orphan_close is not None:
        first_marker = min(
            (
                idx
                for marker in _VISIBLE_OUTPUT_MARKERS
                if (idx := cleaned.lower().find(marker.lower())) >= 0
            ),
            default=None,
        )
        if first_marker is None or orphan_close.start() < first_marker:
            prefix = cleaned[: orphan_close.start()]
            suffix = cleaned[orphan_close.end() :]
            if "<think" not in prefix.lower():
                cleaned = suffix
    return cleaned.strip()


def has_closed_thinking_block(text: str) -> bool:
    return re.search(r"(?is)</think>\s*", text or "") is not None


def output_after_last_thinking_block(text: str) -> tuple[str, bool]:
    raw = str(text or "")
    matches = list(re.finditer(r"(?is)</think>\s*", raw))
    if not matches:
        return strip_reasoning_blocks(raw), False
    return raw[matches[-1].end() :].strip(), True


def final_visible_output(text: str) -> str:
    visible_text, _ = output_after_last_thinking_block(text)
    return visible_text


def last_pattern_match(
    pattern: re.Pattern[str],
    text: str,
    *,
    after: int = 0,
) -> Optional[re.Match[str]]:
    matches = [match for match in pattern.finditer(text) if match.start() >= after]
    return matches[-1] if matches else None


def extract_hidden_reasoning(text: str) -> str:
    raw = str(text or "")
    close_match = re.search(r"(?is)</think>", raw)
    if close_match is not None:
        prefix = raw[: close_match.start()]
        open_matches = list(re.finditer(r"(?is)<think>\s*", prefix))
        if open_matches:
            prefix = prefix[open_matches[-1].end() :]
        return prefix.strip()

    open_matches = list(re.finditer(r"(?is)<think>\s*", raw))
    if open_matches:
        return raw[open_matches[-1].end() :].strip()
    return ""


def measure_reasoning_repetition(
    text: str,
    *,
    window_words: int = REASONING_REPETITION_WINDOW_WORDS,
) -> dict[str, Any]:
    if window_words <= 0:
        raise ValueError("window_words must be positive")

    reasoning = extract_hidden_reasoning(text)
    uncompressed = reasoning.encode("utf-8")
    compressed = gzip.compress(uncompressed, compresslevel=9, mtime=0)
    words = reasoning.split()
    window_count = max(0, len(words) - window_words + 1)
    repeated_windows = 0
    seen_windows: set[tuple[str, ...]] = set()
    for start in range(window_count):
        window = tuple(words[start : start + window_words])
        if window in seen_windows:
            repeated_windows += 1
        else:
            seen_windows.add(window)

    gzip_factor = (
        len(uncompressed) / len(compressed) if uncompressed and compressed else 0.0
    )
    repeated_fraction = repeated_windows / window_count if window_count else 0.0
    return {
        "hidden_reasoning_chars": len(reasoning),
        "uncompressed_bytes": len(uncompressed),
        "gzip_bytes": len(compressed),
        "gzip_factor": gzip_factor,
        "gzip_warning_threshold": REASONING_GZIP_WARNING_THRESHOLD,
        "gzip_warning": gzip_factor > REASONING_GZIP_WARNING_THRESHOLD,
        "word_count": len(words),
        "window_words": window_words,
        "word_window_count": window_count,
        "repeated_word_window_count": repeated_windows,
        "repeated_word_window_fraction": repeated_fraction,
    }


def log_generation_visible_output(
    stage: str,
    text: str,
    *,
    problem_id: Any,
    attempt_idx: int,
    round_idx: int,
) -> None:
    visible, found_end_think = output_after_last_thinking_block(text)
    print(
        f"Visible output after thinking stage={stage} problem={problem_id} candidate={attempt_idx} round={round_idx} "
        f"end_think_found={found_end_think} chars={len(visible)}:\n{visible}",
    )


def log_verifier_tail_output(
    stage: str,
    text: str,
    *,
    attempt_idx: int,
    round_idx: int,
    verifier_index: Optional[int] = None,
    meta_index: Optional[int] = None,
    max_chars: int = 500,
) -> None:
    raw = str(text or "")
    tail = raw[-max_chars:]
    logging.info(
        "Raw output tail stage=%s candidate=%d round=%d verifier=%s meta=%s "
        "raw_chars=%d tail_chars=%d:\n%s",
        stage,
        attempt_idx,
        round_idx,
        verifier_index,
        meta_index,
        len(raw),
        len(tail),
        tail,
    )


def safe_path_component(value: Any, default: str = "unknown") -> str:
    raw = str(value if value is not None else default).strip() or default
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return cleaned or default


def parse_detail_int(detail: str, key: str) -> Optional[int]:
    match = re.search(rf"\b{re.escape(key)}=(\d+)\b", detail or "")
    return int(match.group(1)) if match else None


def stage_file_alias(stage: str) -> str:
    return {
        "proof_generation": "proof_gen",
        "proof_handoff": "proof_handoff",
        "proof_refine": "proof_refine",
        "proof_verify": "verify",
        "proof_meta_verify": "meta",
    }.get(stage, safe_path_component(stage, "llm"))


def llm_call_filename(stage: str, detail: str) -> str:
    candidate = parse_detail_int(detail, "candidate")
    round_idx = parse_detail_int(detail, "round")
    verifier_idx = parse_detail_int(detail, "verifier")
    meta_idx = parse_detail_int(detail, "meta")
    lemma_match = re.search(r"\blemma=([A-Za-z0-9_.-]+)\b", detail or "")
    lemma_id = lemma_match.group(1) if lemma_match else None
    alias = stage_file_alias(stage)
    parts: list[str] = []
    if candidate is not None:
        parts.append(f"cand_{candidate}")
        parts.append(alias)
    else:
        parts.append(alias)
    if round_idx is not None:
        parts.append(f"r{round_idx}")
    if verifier_idx is not None:
        parts.append(f"v{verifier_idx}")
    if meta_idx is not None:
        parts.append(f"m{meta_idx}")
    if lemma_id is not None:
        parts.append(safe_path_component(lemma_id, "lemma"))
    if len(parts) == 1 and detail:
        detail_hash = hashlib.sha256(detail.encode("utf-8")).hexdigest()[:8]
        parts.append(detail_hash)
    return "_".join(parts) + ".txt"


def extract_after_marker(text: str, markers: tuple[str, ...]) -> str:
    for marker in markers:
        match = re.search(re.escape(marker), text, flags=re.IGNORECASE)
        if match:
            return text[match.end() :].strip()
    return ""


def extract_after_last_marker(text: str, markers: tuple[str, ...]) -> tuple[str, bool]:
    best_match: Optional[re.Match[str]] = None
    for marker in markers:
        for match in re.finditer(re.escape(marker), text, flags=re.IGNORECASE):
            if best_match is None or match.start() > best_match.start():
                best_match = match
    if best_match is None:
        return "", False
    return text[best_match.end() :].strip(), True


def extract_after_last_section_marker(
    text: str,
    markers: tuple[str, ...],
    patterns: tuple[re.Pattern[str], ...],
) -> tuple[str, bool, str]:
    best_match: Optional[re.Match[str]] = None
    marker_kind = ""
    for marker in markers:
        for match in re.finditer(re.escape(marker), text, flags=re.IGNORECASE):
            if best_match is None or match.start() > best_match.start():
                best_match = match
                marker_kind = marker
    for pattern in patterns:
        for match in pattern.finditer(text):
            if best_match is None or match.start() > best_match.start():
                best_match = match
                marker_kind = pattern.pattern
    if best_match is None:
        return "", False, ""
    return text[best_match.end() :].strip(), True, marker_kind


def clip_middle_text(text: str, max_chars: int) -> tuple[str, bool]:
    cleaned = str(text or "").strip()
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned, False
    marker = _CLIPPED_TEXT_MARKER
    if max_chars <= len(marker) + 20:
        return cleaned[-max_chars:], True
    head_chars = (max_chars - len(marker)) // 2
    tail_chars = max_chars - len(marker) - head_chars
    return f"{cleaned[:head_chars]}{marker}{cleaned[-tail_chars:]}", True


def merge_streamed_token_ids(
    existing: list[int], incoming: list[int]
) -> tuple[list[int], int]:
    if len(incoming) >= len(existing) and incoming[: len(existing)] == existing:
        return list(incoming), len(incoming) - len(existing)
    merged = list(existing)
    merged.extend(incoming)
    return merged, len(incoming)


def detect_duplicate_segment_in_recent_tokens(
    tokenizer: Any,
    token_ids: list[int],
    *,
    recent_token_count: int = REPETITION_GUARD_RECENT_TOKENS,
    duplicate_threshold: int = REPETITION_GUARD_DUPLICATE_LINE_THRESHOLD,
) -> Optional[tuple[str, int, str]]:
    if tokenizer is None or not token_ids:
        return None
    recent_ids = token_ids[-max(1, int(recent_token_count)) :]
    try:
        text = tokenizer.decode(recent_ids, skip_special_tokens=False)
    except Exception:
        logging.debug("Repetition guard decode failed", exc_info=True)
        return None

    def check_segments(
        segments: list[str], *, min_chars: int, kind: str
    ) -> Optional[tuple[str, int, str]]:
        counts: dict[str, tuple[str, int]] = {}
        for raw_segment in segments:
            segment = raw_segment.strip()
            if len(segment) < min_chars:
                continue
            normalized = re.sub(r"\s+", " ", segment).strip().lower()
            if not normalized:
                continue
            original, count = counts.get(normalized, (segment, 0))
            count += 1
            if count > duplicate_threshold:
                return original, count, kind
            counts[normalized] = (original, count)
        return None

    line_repetition = check_segments(
        str(text or "").split("\n"), min_chars=8, kind="line"
    )
    if line_repetition is not None:
        return line_repetition

    sentence_repetition = check_segments(
        re.split(r"(?<=[.!?])\s+", str(text or "")),
        min_chars=24,
        kind="sentence",
    )
    if sentence_repetition is not None:
        return sentence_repetition

    return None


def trim_score_trailer(text: str, pattern: re.Pattern[str]) -> str:
    return pattern.sub("", text or "").strip()


def _header_matches(text: str, header: str) -> list[re.Match[str]]:
    header_pattern = re.escape(header.strip()).replace(r"\ ", r"[ \t]+")
    return list(
        re.finditer(
            rf"(?im)^[ \t]*{header_pattern}{_HEADER_SUFFIX_PATTERN}",
            text,
        )
    )


def _extract_deepseek_generation_sections(
    text: str,
) -> tuple[str, str, bool, bool]:
    solution_headers = _header_matches(text, "## Solution")
    evaluation_headers = _header_matches(text, "## Self Evaluation")
    if not solution_headers:
        return "", "", False, False

    solution_header = solution_headers[-1]
    following_evaluation = next(
        (
            match
            for match in evaluation_headers
            if match.start() > solution_header.end()
        ),
        None,
    )
    if following_evaluation is None:
        return text[solution_header.end() :].strip(), "", True, False

    proof = text[solution_header.end() : following_evaluation.start()].strip()
    self_evaluation = text[following_evaluation.end() :].strip()
    return proof, self_evaluation, True, True


def parse_deepseek_generation_response(
    text: str,
    *,
    require_self_evaluation: bool = True,
) -> dict[str, Any]:
    visible_text = final_visible_output(text)
    proof, self_evaluation, has_solution_section, has_self_evaluation_section = (
        _extract_deepseek_generation_sections(visible_text)
    )
    self_score = extract_boxed_score(self_evaluation)
    is_valid = bool(has_solution_section and proof)
    if require_self_evaluation:
        is_valid = bool(is_valid and has_self_evaluation_section)
    return {
        "proof": proof,
        "self_evaluation": self_evaluation,
        "self_score": self_score,
        "has_solution_section": has_solution_section,
        "has_self_evaluation_section": has_self_evaluation_section,
        "requires_self_evaluation": require_self_evaluation,
        "is_valid_candidate_response": is_valid,
    }


def parse_generation_response(
    text: str,
    *,
    require_self_evaluation: bool = True,
) -> dict[str, Any]:
    visible_text = final_visible_output(text)
    solution_match = last_pattern_match(_XML_SOLUTION_PATTERN, visible_text)
    solution_end = solution_match.end() if solution_match else 0
    self_evaluation_match = last_pattern_match(
        _XML_SELF_EVALUATION_PATTERN,
        visible_text,
        after=solution_end,
    )
    score_start = (
        self_evaluation_match.end()
        if self_evaluation_match is not None
        else solution_end
    )
    score_match = last_pattern_match(
        _XML_SCORE_PATTERN,
        visible_text,
        after=score_start,
    )
    proof = solution_match.group(1).strip() if solution_match else ""
    self_evaluation = (
        self_evaluation_match.group(1).strip() if self_evaluation_match else ""
    )
    self_score = float(score_match.group(1)) if score_match else None
    has_solution_section = solution_match is not None
    has_self_evaluation_section = self_evaluation_match is not None
    is_valid = bool(has_solution_section and proof)
    if require_self_evaluation:
        is_valid = bool(
            is_valid
            and has_self_evaluation_section
            and self_evaluation
            and self_score in {0.0, 0.5, 1.0}
        )
    return {
        "proof": proof,
        "self_evaluation": self_evaluation,
        "self_score": self_score,
        "has_solution_section": has_solution_section,
        "has_self_evaluation_section": has_self_evaluation_section,
        "requires_self_evaluation": require_self_evaluation,
        "is_valid_candidate_response": is_valid,
    }


VISIBLE_OUTPUT_LIMIT_PARTIAL_PROOF_TEXT = (
    "The refinement reached its enforced visible-output limit before a "
    "complete rigorous proof was produced. The preceding argument is partial "
    "and must not be treated as a complete solution."
)
VISIBLE_OUTPUT_LIMIT_SELF_EVALUATION = (
    "The response was closed automatically at the configured visible-output "
    "limit. At least one required argument remains unresolved, so this is an "
    "incomplete proof."
)


def build_visible_output_limit_footer(text: str) -> str:
    if parse_generation_response(text).get("is_valid_candidate_response"):
        return ""

    visible = final_visible_output(text)
    normalized = visible.lower()
    solution_open = normalized.rfind("<solution>")
    solution_close = normalized.rfind("</solution>")
    parts: list[str] = ["\n\n"]
    if solution_open < 0:
        parts.extend(
            [
                "<solution>\n",
                VISIBLE_OUTPUT_LIMIT_PARTIAL_PROOF_TEXT,
                "\n</solution>\n",
            ]
        )
    elif solution_close < solution_open:
        parts.extend(
            [
                VISIBLE_OUTPUT_LIMIT_PARTIAL_PROOF_TEXT,
                "\n</solution>\n",
            ]
        )
    parts.extend(
        [
            "<self_evaluation>\n",
            VISIBLE_OUTPUT_LIMIT_SELF_EVALUATION,
            "\n</self_evaluation>\n",
            "<score>0</score>",
        ]
    )
    return "".join(parts)


def require_valid_candidate_response(
    parsed: dict[str, Any],
    *,
    problem_id: Any,
    attempt_idx: int,
    stage: str,
    round_idx: int,
) -> bool:
    if parsed.get("is_valid_candidate_response"):
        return True
    logging.warning(
        "Invalid candidate output problem=%s candidate=%d stage=%s round=%d "
        "has_solution=%s proof_chars=%d requires_self_evaluation=%s "
        "has_self_evaluation=%s self_evaluation_chars=%d self_score=%s; "
        "skipping this output without raising",
        problem_id,
        attempt_idx,
        stage,
        round_idx,
        parsed.get("has_solution_section"),
        len(str(parsed.get("proof") or "")),
        parsed.get("requires_self_evaluation"),
        parsed.get("has_self_evaluation_section"),
        len(str(parsed.get("self_evaluation") or "")),
        parsed.get("self_score"),
    )
    return False


def parse_verifier_response(text: str) -> dict[str, Any]:
    visible_text = final_visible_output(text)
    evaluation_match = last_pattern_match(_XML_EVALUATION_PATTERN, visible_text)
    evaluation_end = evaluation_match.end() if evaluation_match else 0
    suggestions_match = last_pattern_match(
        _XML_SUGGESTIONS_PATTERN,
        visible_text,
        after=evaluation_end,
    )
    score_start = suggestions_match.end() if suggestions_match else evaluation_end
    score_match = last_pattern_match(
        _XML_SCORE_PATTERN,
        visible_text,
        after=score_start,
    )
    evaluation = evaluation_match.group(1).strip() if evaluation_match else ""
    suggestions = suggestions_match.group(1).strip() if suggestions_match else ""
    score = float(score_match.group(1)) if score_match else None
    is_valid = bool(evaluation and suggestions and score in {0.0, 0.5, 1.0})
    review = "\n".join(
        [
            "<evaluation>",
            evaluation,
            "</evaluation>",
            "<suggestions>",
            suggestions,
            "</suggestions>",
            f"<score>{score:g}</score>" if score is not None else "",
        ]
    ).strip()
    return {
        "evaluation": evaluation,
        "suggestions": suggestions,
        "review": review,
        "score": score if is_valid else None,
        "parsed_score": score,
        "is_valid_verifier_response": is_valid,
        "evaluation_marker_found": evaluation_match is not None,
        "evaluation_marker": "<evaluation>" if evaluation_match else None,
        "evaluation_raw_chars": len(visible_text),
        "evaluation_forwarded_chars": len(evaluation),
        "evaluation_clipped": False,
    }


def parse_deepseek_verifier_response(text: str) -> dict[str, Any]:
    visible_text = final_visible_output(text)
    evaluation, marker_found, marker_kind = extract_after_last_section_marker(
        visible_text,
        _VERIFIER_EVALUATION_MARKERS,
        _VERIFIER_EVALUATION_PATTERNS,
    )
    if evaluation:
        evaluation = trim_score_trailer(evaluation, _VERIFIER_SCORE_TRAILER_PATTERN)
    else:
        evaluation = trim_score_trailer(
            visible_text,
            _VERIFIER_SCORE_TRAILER_PATTERN,
        )
    if not marker_found and len(evaluation) > MAX_FORWARDED_EVALUATION_CHARS:
        evaluation = evaluation[-MAX_FORWARDED_EVALUATION_CHARS:]
        clipped = True
    else:
        evaluation, clipped = clip_middle_text(
            evaluation,
            MAX_FORWARDED_EVALUATION_CHARS,
        )
    parsed_score = extract_boxed_score(visible_text)
    is_valid = bool(evaluation.strip() and parsed_score in {0.0, 0.5, 1.0})
    review_parts = [
        "Here is my evaluation of the solution:",
        evaluation.strip(),
    ]
    if parsed_score is not None:
        review_parts.extend(
            [
                "Based on my evaluation, the final overall score should be:",
                rf"\boxed{{{parsed_score:g}}}",
            ]
        )
    return {
        "evaluation": evaluation.strip(),
        "suggestions": "",
        "review": "\n\n".join(part for part in review_parts if part),
        "score": parsed_score if is_valid else None,
        "parsed_score": parsed_score,
        "is_valid_verifier_response": is_valid,
        "evaluation_marker_found": marker_found,
        "evaluation_marker": marker_kind,
        "evaluation_raw_chars": len(visible_text),
        "evaluation_forwarded_chars": len(evaluation),
        "evaluation_clipped": clipped,
    }


def parse_meta_verifier_response(text: str) -> dict[str, Any]:
    visible_text = final_visible_output(text)
    analysis, marker_found, marker_kind = extract_after_last_section_marker(
        visible_text,
        _META_ANALYSIS_MARKERS,
        _META_ANALYSIS_PATTERNS,
    )
    if analysis:
        analysis = trim_score_trailer(analysis, _META_SCORE_TRAILER_PATTERN)
    else:
        analysis = trim_score_trailer(visible_text, _META_SCORE_TRAILER_PATTERN)
    if not marker_found and len(analysis) > MAX_FORWARDED_META_ANALYSIS_CHARS:
        analysis = analysis[-MAX_FORWARDED_META_ANALYSIS_CHARS:]
        clipped = True
    else:
        analysis, clipped = clip_middle_text(
            analysis,
            MAX_FORWARDED_META_ANALYSIS_CHARS,
        )
    return {
        "analysis": analysis.strip(),
        "score": extract_boxed_score(visible_text),
        "analysis_marker_found": marker_found,
        "analysis_marker": marker_kind,
        "analysis_raw_chars": len(visible_text),
        "analysis_forwarded_chars": len(analysis),
        "analysis_clipped": clipped,
    }


def parse_selected_index(text: str, candidate_count: int) -> Optional[int]:
    visible_text = final_visible_output(text)
    match = last_pattern_match(_SELECTED_ID_PATTERN, visible_text)
    if not match:
        return None
    selected = int(match.group(2))
    if 0 <= selected < candidate_count:
        return selected
    return None


def summarize_meta_votes(meta_results: list[dict[str, Any]]) -> dict[str, Any]:
    valid_votes = sum(1 for result in meta_results if result.get("score") == 1.0)
    half_votes = sum(1 for result in meta_results if result.get("score") == 0.5)
    invalid_votes = sum(1 for result in meta_results if result.get("score") == 0.0)
    parsed_votes = valid_votes + half_votes + invalid_votes
    threshold = (len(meta_results) // 2) + 1 if meta_results else 1
    return {
        "valid_votes": valid_votes,
        "half_votes": half_votes,
        "invalid_votes": invalid_votes,
        "parsed_votes": parsed_votes,
        "threshold": threshold,
        "validated": valid_votes >= threshold,
    }


def compute_meta_summary_by_verifier(
    verifier_results: list[dict[str, Any]],
    meta_results_by_verifier: dict[int, list[dict[str, Any]]],
) -> dict[int, dict[str, Any]]:
    summaries: dict[int, dict[str, Any]] = {}
    for verifier in verifier_results:
        try:
            verifier_index = int(verifier.get("verifier_index"))
        except (TypeError, ValueError):
            continue
        summaries[verifier_index] = summarize_meta_votes(
            meta_results_by_verifier.get(verifier_index, [])
        )
    return summaries


def compute_strict_pass(
    verifier_results: list[dict[str, Any]],
    meta_results_by_verifier: dict[int, list[dict[str, Any]]],
    *,
    require_meta: bool,
) -> dict[str, Any]:
    verifier_count = len(verifier_results)
    parsed_verifier_count = sum(
        1 for result in verifier_results if result.get("score") is not None
    )
    pass_count = sum(1 for result in verifier_results if result.get("score") == 1.0)
    all_verifiers_passed = (
        verifier_count > 0
        and parsed_verifier_count == verifier_count
        and pass_count == verifier_count
    )
    meta_summaries = compute_meta_summary_by_verifier(
        verifier_results, meta_results_by_verifier
    )
    meta_valid_count = sum(
        1 for summary in meta_summaries.values() if summary.get("validated")
    )
    meta_checked_count = sum(
        1 for summary in meta_summaries.values() if summary.get("parsed_votes", 0) > 0
    )
    strict_pass = all_verifiers_passed
    if require_meta:
        strict_pass = (
            all_verifiers_passed
            and verifier_count > 0
            and meta_checked_count == verifier_count
            and meta_valid_count == verifier_count
        )
    return {
        "strict_pass": strict_pass,
        "all_verifiers_passed": all_verifiers_passed,
        "meta_valid_count": meta_valid_count,
        "meta_checked_count": meta_checked_count,
        "meta_summary_by_verifier": meta_summaries,
    }


def coerce_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= score <= 1.0:
        return score
    return None


def verifier_meta_factor(
    verifier_index: int,
    meta_results_by_verifier: dict[int, list[dict[str, Any]]],
    meta_n: int,
) -> tuple[float, list[float], str]:
    if meta_n <= 0:
        return 1.0, [], "meta_disabled"
    meta_scores = [
        score
        for score in (
            coerce_score(result.get("score"))
            for result in meta_results_by_verifier.get(verifier_index, [])
        )
        if score is not None
    ]
    if meta_scores:
        return sum(meta_scores) / len(meta_scores), meta_scores, "parsed_meta"
    return 0.6, [], "missing_meta_default"


def build_verifier_score_summaries(
    verifier_results: list[dict[str, Any]],
    meta_results_by_verifier: dict[int, list[dict[str, Any]]],
    meta_n: int,
) -> list[dict[str, Any]]:
    summaries = []
    for idx, verifier in enumerate(verifier_results):
        verifier_index = int(verifier.get("verifier_index", idx))
        verifier_score = coerce_score(verifier.get("score"))
        if verifier_score is None:
            continue
        meta_factor, meta_scores, meta_source = verifier_meta_factor(
            verifier_index,
            meta_results_by_verifier,
            meta_n,
        )
        weighted_score = verifier_score * meta_factor
        evaluation, evaluation_clipped = clip_middle_text(
            str(verifier.get("evaluation") or "").strip(),
            MAX_FORWARDED_EVALUATION_CHARS,
        )
        summaries.append(
            {
                "verifier_index": verifier_index,
                "verifier_role": verifier.get("verifier_role"),
                "verifier_group": verifier.get("verifier_group"),
                "verifier_score": verifier_score,
                "meta_scores": meta_scores,
                "meta_factor": meta_factor,
                "meta_source": meta_source,
                "weighted_score": weighted_score,
                "evaluation": evaluation,
                "evaluation_clipped": evaluation_clipped,
            }
        )
    return summaries


def aggregate_proof_label(
    verifier_results: list[dict[str, Any]],
    meta_results_by_verifier: dict[int, list[dict[str, Any]]],
    min_valid_low: int,
    strict_pass_meta: bool = False,
    meta_n: int = 0,
    audit_positive_meta: bool = False,
) -> dict[str, Any]:
    validated_critiques = []
    low_scores_seen = 0
    score_summaries = build_verifier_score_summaries(
        verifier_results,
        meta_results_by_verifier,
        meta_n=meta_n,
    )

    for idx, verifier in enumerate(verifier_results):
        verifier_index = int(verifier.get("verifier_index", idx))
        score = coerce_score(verifier.get("score"))
        if score is None or score >= 1.0:
            continue
        low_scores_seen += 1
        meta_summary = summarize_meta_votes(
            meta_results_by_verifier.get(verifier_index, [])
        )
        critique = {
            "verifier_index": verifier_index,
            "verifier_role": verifier.get("verifier_role"),
            "verifier_group": verifier.get("verifier_group"),
            "score": score,
            "evaluation": verifier.get("evaluation", ""),
            "suggestions": verifier.get("suggestions", ""),
            "review": verifier.get("review", verifier.get("evaluation", "")),
            "meta_summary": meta_summary,
        }
        if meta_n <= 0:
            critique["meta_summary"] = {
                **meta_summary,
                "validated": True,
                "validation_source": "meta_disabled",
            }
            validated_critiques.append(critique)
        elif meta_summary["validated"]:
            validated_critiques.append(critique)

    positive_meta_challenges = []
    if audit_positive_meta and meta_n > 0:
        for idx, verifier in enumerate(verifier_results):
            verifier_index = int(verifier.get("verifier_index", idx))
            if coerce_score(verifier.get("score")) != 1.0:
                continue
            challenged_meta = [
                result
                for result in meta_results_by_verifier.get(verifier_index, [])
                if (
                    coerce_score(result.get("score")) is not None
                    and coerce_score(result.get("score")) < 1.0
                )
            ]
            if not challenged_meta:
                continue
            strongest_challenge = min(
                challenged_meta,
                key=lambda result: float(coerce_score(result.get("score")) or 0.0),
            )
            challenge_score = coerce_score(strongest_challenge.get("score"))
            analysis = str(strongest_challenge.get("analysis") or "").strip()
            critique = {
                "verifier_index": verifier_index,
                "verifier_role": verifier.get("verifier_role"),
                "verifier_group": verifier.get("verifier_group"),
                "score": challenge_score,
                "evaluation": (
                    "Adversarial meta-audit rejected the verifier's positive "
                    f"verdict.\n{analysis}"
                ).strip(),
                "review": analysis,
                "suggestions": (
                    "Re-audit the challenged positive verdict and repair or "
                    "remove the claim exposed by the adversarial meta-analysis."
                ),
                "critique_source": "positive_meta_audit",
                "meta_summary": {
                    **summarize_meta_votes(
                        meta_results_by_verifier.get(verifier_index, [])
                    ),
                    "validated": True,
                    "validation_source": "adversarial_positive_meta",
                },
            }
            positive_meta_challenges.append(critique)
            validated_critiques.append(critique)
            low_scores_seen += 1

    strict_pass = compute_strict_pass(
        verifier_results,
        meta_results_by_verifier,
        require_meta=strict_pass_meta,
    )
    weighted_scores = [summary["weighted_score"] for summary in score_summaries]
    verifier_group_scores: dict[str, float] = {}
    for verifier_group in ("generalist", "specialist"):
        group_scores = [
            summary["weighted_score"]
            for summary in score_summaries
            if summary.get("verifier_group") == verifier_group
        ]
        if group_scores:
            verifier_group_scores[verifier_group] = sum(group_scores) / len(
                group_scores
            )
    if set(verifier_group_scores) == {"generalist", "specialist"}:
        final_score = sum(verifier_group_scores.values()) / 2
        aggregation_mode = "balanced_verifier_groups"
    else:
        final_score = (
            sum(weighted_scores) / len(weighted_scores) if weighted_scores else None
        )
        aggregation_mode = "flat_verifier_mean"
    validated_fatal_critiques = [
        critique
        for critique in validated_critiques
        if coerce_score(critique.get("score")) == 0.0
    ]
    required_validated_critiques = max(1, int(min_valid_low))
    validated_critique_quorum = (
        len(validated_critiques) >= required_validated_critiques
    )
    validated_low_score_cap_applied = bool(
        validated_critique_quorum and final_score is not None and final_score > 0.5
    )
    fatal_score_cap_applied = bool(
        validated_critique_quorum
        and validated_fatal_critiques
        and final_score is not None
        and final_score > 0.5
    )
    if validated_low_score_cap_applied:
        final_score = 0.5
    if final_score is None:
        final_status = "needs_review"
    elif validated_critique_quorum and final_score <= 0.5:
        final_status = "validated_low_score"
    elif final_score > 0.5:
        if strict_pass["strict_pass"]:
            final_status = "strict_pass" if strict_pass_meta else "all_verifiers_passed"
        elif strict_pass["all_verifiers_passed"] and (
            not strict_pass_meta or final_score >= 1.0
        ):
            final_status = "all_verifiers_passed"
        else:
            final_status = "weighted_score_pass"
    else:
        final_status = "weighted_score_low"

    return {
        "final_score": final_score,
        "final_status": final_status,
        "validated_critiques": validated_critiques,
        "validated_fatal_critiques": validated_fatal_critiques,
        "validated_low_score_cap_applied": validated_low_score_cap_applied,
        "fatal_score_cap_applied": fatal_score_cap_applied,
        "validated_critique_quorum": validated_critique_quorum,
        "required_validated_critiques": required_validated_critiques,
        "aggregation_mode": aggregation_mode,
        "verifier_group_scores": verifier_group_scores,
        "verifier_score_summaries": score_summaries,
        "low_scores_seen": low_scores_seen,
        "positive_meta_challenges": positive_meta_challenges,
        **strict_pass,
    }


def candidate_retention_score(aggregation: dict[str, Any]) -> float:
    """Rank verified proof versions without counting meta evidence twice."""
    score = aggregation.get("final_score")
    if score is None:
        return -1.0
    try:
        value = float(score)
    except (TypeError, ValueError):
        return -1.0
    # aggregate_proof_label already folds meta scores and positive-verdict
    # challenges into final_score. A second flat challenge penalty can make a
    # demonstrably improved proof rank below the version it repaired.
    if aggregation.get("strict_pass_challenge_survived"):
        value += 0.001
    return value


def should_replace_retained_candidate(
    candidate_aggregation: dict[str, Any],
    retained_aggregation: dict[str, Any],
) -> bool:
    """Replace a verified proof only when its retention evidence improves."""
    return candidate_retention_score(candidate_aggregation) > candidate_retention_score(
        retained_aggregation
    )


def snapshot_verified_candidate_version(
    proof: str,
    parsed: dict[str, Any],
    aggregation: dict[str, Any],
    round_idx: int,
) -> dict[str, Any]:
    """Keep the compact fields needed to reconsider a verified proof later."""
    compact_verifier_summaries = [
        {
            key: summary.get(key)
            for key in (
                "verifier_index",
                "verifier_role",
                "verifier_group",
                "verifier_score",
                "meta_scores",
                "meta_factor",
                "meta_source",
                "weighted_score",
            )
        }
        for summary in aggregation.get("verifier_score_summaries") or []
    ]
    return {
        "proof_solution": proof,
        "self_evaluation": parsed.get("self_evaluation"),
        "self_score": parsed.get("self_score"),
        "final_score": aggregation.get("final_score"),
        "final_status": aggregation.get("final_status"),
        "verifier_score_summaries": compact_verifier_summaries,
        "verifier_group_scores": aggregation.get("verifier_group_scores", {}),
        "aggregation_mode": aggregation.get("aggregation_mode"),
        "low_scores_seen": aggregation.get("low_scores_seen", 0),
        "strict_pass": aggregation.get("strict_pass", False),
        "all_verifiers_passed": aggregation.get("all_verifiers_passed", False),
        "meta_valid_count": aggregation.get("meta_valid_count", 0),
        "meta_checked_count": aggregation.get("meta_checked_count", 0),
        "meta_summary_by_verifier": aggregation.get(
            "meta_summary_by_verifier", {}
        ),
        "validated_low_score_cap_applied": aggregation.get(
            "validated_low_score_cap_applied", False
        ),
        "fatal_score_cap_applied": aggregation.get(
            "fatal_score_cap_applied", False
        ),
        "strict_pass_challenge_survived": aggregation.get(
            "strict_pass_challenge_survived", False
        ),
        "selected_verification_round": round_idx,
    }


def response_usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {
        key: getattr(usage, key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if hasattr(usage, key)
    }


class VLLMServer:
    def __init__(self, cfg: VLLMConfig, port: int, gpu_group: str, index: int) -> None:
        self.cfg = cfg
        self.port = port
        self.gpu_group = gpu_group
        self.index = index
        self.client_host = "127.0.0.1" if cfg.host == "0.0.0.0" else cfg.host
        self.base_url = f"http://{self.client_host}:{port}/v1"
        self.health_url = f"http://{self.client_host}:{port}/health"
        self._proc: Optional[subprocess.Popen] = None
        self._log_file: Any = None
        self._log_path: Optional[Path] = None
        self._stopped = False

    @property
    def tag(self) -> str:
        return (
            f"vllm[{self.index}] port={self.port} gpus={self.gpu_group} "
            f"tp={self.cfg.tensor_parallel_size} dp={self.cfg.data_parallel_size}"
        )

    def is_port_open(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            return sock.connect_ex((self.client_host, self.port)) == 0

    def build_command(self) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.cfg.model_path,
            "--served-model-name",
            self.cfg.served_model_name,
            "--api-key",
            self.cfg.api_key,
            "--tensor-parallel-size",
            str(self.cfg.tensor_parallel_size),
            "--data-parallel-size",
            str(self.cfg.data_parallel_size),
            "--max-num-seqs",
            str(self.cfg.max_num_seqs),
            "--gpu-memory-utilization",
            str(self.cfg.gpu_memory_utilization),
            "--host",
            self.cfg.host,
            "--port",
            str(self.port),
            "--dtype",
            self.cfg.dtype,
            "--max-model-len",
            str(self.cfg.num_ctx),
            "--stream-interval",
            str(self.cfg.stream_interval),
            "--async-scheduling",
            "--enable-prefix-caching",
            "--trust-remote-code",
        ]
        extra = shlex.split(self.cfg.vllm_extra_args or "")
        if extra:
            cmd.extend(extra)
        return cmd

    def build_environment(self) -> dict[str, str]:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": self.gpu_group}
        # NII mounts /tmp as a shared filesystem. The image-wide compile cache
        # paths therefore collide when multiple nodes compile identical local
        # TP/DP servers at the same time. Use a short per-node, per-run cache
        # namespace while preserving the image's global model/download caches.
        cache_identity = "|".join(
            (
                socket.gethostname(),
                str(self.cfg.logdir.absolute()),
                str(self.index),
                self.cfg.model_path,
                str(self.cfg.tensor_parallel_size),
                str(self.cfg.data_parallel_size),
            )
        )
        cache_key = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()[:16]
        cache_root = (
            Path(
                os.environ.get(
                    "AIMO_VLLM_RUNTIME_CACHE_ROOT",
                    str(Path(os.environ.get("TMPDIR", "/tmp")) / "aimo-vllm-cache"),
                )
            )
            / cache_key
        )
        cache_paths = {
            "VLLM_CACHE_ROOT": cache_root / "vllm",
            "TORCHINDUCTOR_CACHE_DIR": cache_root / "torchinductor",
            "TRITON_CACHE_DIR": cache_root / "triton",
            "CUDA_CACHE_PATH": cache_root / "cuda",
            "XDG_CACHE_HOME": cache_root / "xdg",
        }
        for path in cache_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        env.update({key: str(path) for key, path in cache_paths.items()})

        # Each node owns an independent local vLLM TP/DP server. External
        # node-level rendezvous variables must not make vLLM join that world.
        for external_rank_key in (
            "RANK",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "GLOBAL_RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        ):
            env.pop(external_rank_key, None)
        env.setdefault("VLLM_PLUGINS", "olmo3_sink")
        # vLLM 0.25 defaults to Model Runner V2, which does not yet support the
        # thinking_token_budget request field used by this inference pipeline.
        env.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
        return env

    def start(self) -> None:
        if self.is_port_open():
            logging.info("Port %s is active. Reusing existing vLLM server.", self.port)
            return
        cmd = self.build_command()
        env = self.build_environment()
        log_path = self.cfg.logdir / f"vllm_server_{self.index}.log"
        self._log_path = log_path
        self._log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        logging.info(
            "Launching %s with compile cache %s: %s",
            self.tag,
            env["TORCHINDUCTOR_CACHE_DIR"],
            " ".join(shlex.quote(part) for part in cmd),
        )
        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def wait_ready(self, timeout: int) -> None:
        start = time.time()
        while True:
            if self._proc is not None and self._proc.poll() is not None:
                code = None if self._proc is None else self._proc.returncode
                log_tail = ""
                if self._log_path is not None and self._log_path.exists():
                    lines = self._log_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    log_tail = "\n".join(lines[-80:])
                detail = (
                    f"\nLast lines from {self._log_path}:\n{log_tail}"
                    if log_tail
                    else f"\nCheck {self._log_path} for the underlying vLLM error."
                )
                raise RuntimeError(
                    f"{self.tag} exited before health check passed with code {code}{detail}"
                )
            try:
                with urllib.request.urlopen(self.health_url, timeout=2) as response:
                    if response.status == 200:
                        logging.info(
                            "%s ready after %.1fs", self.tag, time.time() - start
                        )
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            if time.time() - start > timeout:
                self.stop()
                raise TimeoutError(
                    f"{self.tag} did not become healthy within {timeout}s"
                )
            time.sleep(3)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._proc is not None and self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self._proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        if self._log_file is not None:
            self._log_file.close()


class ServerManager:
    def __init__(
        self,
        cfg: VLLMConfig,
        gpu_groups: list[str],
        no_serve: bool,
        base_url: str,
        server_timeout: int,
    ) -> None:
        self.cfg = cfg
        self.gpu_groups = gpu_groups
        self.no_serve = no_serve
        self.base_url = base_url
        self.server_timeout = server_timeout
        self.servers: list[VLLMServer] = []
        self.urls: list[str] = []
        self._stopped = False

    def start(self) -> None:
        if self.no_serve:
            if self.base_url.strip():
                self.urls = [
                    url.strip() for url in self.base_url.split(",") if url.strip()
                ]
            else:
                self.urls = [
                    f"http://{self.cfg.host}:{self.cfg.port + i}/v1"
                    for i in range(len(self.gpu_groups))
                ]
            logging.info("Using existing vLLM endpoints: %s", self.urls)
            return

        self.servers = [
            VLLMServer(self.cfg, self.cfg.port + idx, gpu_group, idx)
            for idx, gpu_group in enumerate(self.gpu_groups)
        ]
        atexit.register(self.stop)
        previous_handlers = {}

        if threading.current_thread() is threading.main_thread():
            previous_handlers = {
                sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)
            }

            def handle_signal(signum: int, frame: Any) -> None:
                self.stop()
                previous = previous_handlers.get(signum)
                if callable(previous):
                    previous(signum, frame)
                raise KeyboardInterrupt

            for sig in previous_handlers:
                signal.signal(sig, handle_signal)
        for server in self.servers:
            server.start()
        for server in self.servers:
            server.wait_ready(self.server_timeout)
        self.urls = [server.base_url for server in self.servers]

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        for server in self.servers:
            server.stop()


class ChatScheduler:
    def __init__(
        self,
        base_urls: list[str],
        api_key: str,
        model: str,
        sampling: SamplingConfig,
        max_concurrent_requests: int,
        mock_llm: bool = False,
        stage_max_new_tokens: Optional[dict[str, int]] = None,
        request_timeout_seconds: float = 900.0,
        request_worker_count: Optional[int] = None,
        stream_responses: bool = True,
        context_length: int = 0,
        context_margin_tokens: int = 64,
        tokenizer: Any = None,
        llm_call_logdir: Optional[Path] = None,
        stream_interval_tokens: int = 100,
    ) -> None:
        if not base_urls and not mock_llm:
            raise ValueError("At least one vLLM base URL is required")
        self.base_urls = base_urls or ["mock://local"]
        self.model = model
        self.sampling = sampling
        self.mock_llm = mock_llm
        self.stream_responses = stream_responses
        self.context_length = max(0, int(context_length or 0))
        self.context_margin_tokens = max(0, int(context_margin_tokens))
        self.tokenizer = tokenizer
        self.llm_call_logdir = llm_call_logdir
        self.stream_interval_tokens = max(1, int(stream_interval_tokens or 1))
        self.stage_max_new_tokens = {
            **_DEFAULT_STAGE_TOKEN_LIMITS,
            **(stage_max_new_tokens or {}),
        }
        self._clients = [
            OpenAI(
                base_url=url,
                api_key=api_key,
                timeout=request_timeout_seconds,
                max_retries=2,
            )
            for url in self.base_urls
            if not mock_llm
        ]
        self._counter = 0
        self._counter_lock = asyncio.Lock()
        self._prompt_preview_lock = threading.Lock()
        self._previewed_prompt_hashes: set[tuple[str, str]] = set()
        self.max_concurrent_requests = max(1, max_concurrent_requests)
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        worker_count = min(
            self.max_concurrent_requests,
            max(1, int(request_worker_count or 32)),
        )
        if worker_count == 1:
            shared_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="llm-request",
            )
            self._generation_executor = shared_executor
            self._priority_executor = shared_executor
            self._generation_worker_count = 1
            self._priority_worker_count = 0
        else:
            priority_reserve = max(1, min(8, worker_count // 4))
            self._generation_worker_count = (
                worker_count - priority_reserve
            )
            # Keep initial generations from occupying every blocking HTTP
            # worker, while allowing follow-up stages to use the full vLLM DP
            # request capacity once proof generation winds down.
            self._priority_worker_count = worker_count
            self._generation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._generation_worker_count,
                thread_name_prefix="llm-generation",
            )
            self._priority_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._priority_worker_count,
                thread_name_prefix="llm-priority",
            )
        self._executors_closed = False
        logging.info(
            "LLM request executors ready: generation_workers=%d "
            "priority_workers=%d max_concurrent_requests=%d",
            self._generation_worker_count,
            self._priority_worker_count,
            self.max_concurrent_requests,
        )

    def close(self) -> None:
        if self._executors_closed:
            return
        self._executors_closed = True
        executors = {self._generation_executor, self._priority_executor}
        for executor in executors:
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _token_ids_to_list(token_ids: Any) -> list[int]:
        if isinstance(token_ids, dict):
            token_ids = token_ids.get("input_ids", [])
        elif hasattr(token_ids, "input_ids"):
            token_ids = token_ids.input_ids
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return list(token_ids)

    def _render_completion_prompt(
        self,
        messages: list[dict[str, str]],
        assistant_prefix: Optional[str],
    ) -> tuple[str, list[int]]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is required for completion prompt rendering")
        template_kwargs = {
            "add_generation_prompt": assistant_prefix is None,
            "continue_final_message": assistant_prefix is not None,
        }
        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            **template_kwargs,
        )
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            **template_kwargs,
        )
        return str(rendered_prompt), self._token_ids_to_list(token_ids)

    def _adjust_max_tokens_for_context(
        self,
        stage: str,
        prompt_tokens: int,
        max_tokens: int,
    ) -> int:
        if self.context_length <= 0:
            return max_tokens
        adjusted = self.context_length - prompt_tokens - self.context_margin_tokens
        adjusted = max(1, min(max_tokens, adjusted))
        if adjusted < max_tokens:
            logging.warning(
                "Reducing max_tokens for stage=%s from %d to %d before request "
                "(prompt_tokens=%d context=%d margin=%d)",
                stage,
                max_tokens,
                adjusted,
                prompt_tokens,
                self.context_length,
                self.context_margin_tokens,
            )
        return adjusted

    def _maybe_log_stage_prompt(
        self,
        stage: str,
        detail: str,
        rendered_prompt: str,
        prompt_tokens: int,
    ) -> None:
        prompt_hash = hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()[:6]
        prompt_key = (stage, prompt_hash)
        with self._prompt_preview_lock:
            if prompt_key in self._previewed_prompt_hashes:
                return
            self._previewed_prompt_hashes.add(prompt_key)
        prompt_preview = rendered_prompt
        logging.info(
            "Chat-template prompt stage=%s detail=%s hash=%s chars=%d tokens=%d preview_chars=%d:\n%s",
            stage,
            detail or "-",
            prompt_hash,
            len(rendered_prompt),
            prompt_tokens,
            len(prompt_preview),
            prompt_preview,
        )

    def _llm_call_path(
        self,
        progress: Optional[PipelineProgress],
        stage: str,
        detail: str,
    ) -> Optional[Path]:
        if self.llm_call_logdir is None or progress is None:
            return None
        question_dir = self.llm_call_logdir / safe_path_component(
            progress.problem_id, "question"
        )
        return question_dir / llm_call_filename(stage, detail)

    def _write_llm_call_input(
        self,
        path: Optional[Path],
        stage: str,
        detail: str,
        rendered_prompt: str,
        prompt_tokens: int,
        max_tokens: int,
    ) -> None:
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "\n".join(
                    [
                        f"stage: {stage}",
                        f"detail: {detail or '-'}",
                        f"prompt_tokens: {prompt_tokens}",
                        f"max_tokens: {max_tokens}",
                        "",
                        "===== INPUT PROMPT =====",
                        rendered_prompt,
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        except OSError:
            logging.exception("Failed to write LLM call input debug file: %s", path)

    def _append_llm_call_output(
        self,
        path: Optional[Path],
        response: dict[str, Any],
    ) -> None:
        if response.get("stage") in {"proof_generation", "proof_refine"}:
            response["reasoning_repetition"] = measure_reasoning_repetition(
                str(response.get("text") or "")
            )
        if path is None:
            return
        try:
            with open(path, "a", encoding="utf-8") as file_obj:
                file_obj.write("\n===== OUTPUT =====\n")
                file_obj.write(f"success: {response.get('success')}\n")
                file_obj.write(f"error: {response.get('error')}\n")
                file_obj.write(f"finish_reason: {response.get('finish_reason')}\n")
                file_obj.write(
                    "usage: "
                    + json.dumps(
                        response.get("usage") or {}, ensure_ascii=False, default=str
                    )
                    + "\n"
                )
                if response.get("reasoning_repetition") is not None:
                    file_obj.write(
                        "reasoning_repetition: "
                        + json.dumps(
                            response["reasoning_repetition"],
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\n"
                    )
                file_obj.write(f"server_url: {response.get('server_url')}\n")
                file_obj.write(f"latency_s: {response.get('latency_s')}\n\n")
                file_obj.write(str(response.get("text") or ""))
                file_obj.write("\n")
        except OSError:
            logging.exception("Failed to append LLM call output debug file: %s", path)

    def _append_llm_call_input_segment(
        self,
        path: Optional[Path],
        title: str,
        prompt_ids: list[int],
        max_tokens: int,
    ) -> None:
        if path is None:
            return
        try:
            decoded_prompt = self.tokenizer.decode(
                prompt_ids,
                skip_special_tokens=False,
            )
        except Exception:
            logging.exception(
                "Failed to decode LLM continuation prompt for debug file: %s", path
            )
            decoded_prompt = f"[decode failed; prompt_token_count={len(prompt_ids)}]"
        try:
            with open(path, "a", encoding="utf-8") as file_obj:
                file_obj.write(f"\n===== {title} =====\n")
                file_obj.write(f"prompt_tokens: {len(prompt_ids)}\n")
                file_obj.write(f"max_tokens: {max_tokens}\n\n")
                file_obj.write(decoded_prompt)
                file_obj.write("\n")
        except OSError:
            logging.exception(
                "Failed to append LLM continuation input debug file: %s", path
            )

    async def call(
        self,
        stage: str,
        prompt: str | list[dict[str, str]],
        temperature: Optional[float] = None,
        progress: Optional[PipelineProgress] = None,
        detail: str = "",
        thinking_budget_tokens: Optional[int] = None,
        thinking_budget_force_text: str = "",
        thinking_budget_action: str = "finalize",
        prompt_token_ids: Optional[list[int]] = None,
        max_tokens_override: Optional[int] = None,
        response_prefix: str = "",
        return_context_ids: bool = False,
        visible_output_limit_tokens: Optional[int] = None,
        priority: bool = False,
    ) -> dict[str, Any]:

        response: Optional[dict[str, Any]] = None
        stream_id: Optional[str] = None
        try:
            async with self._semaphore:
                async with self._counter_lock:
                    index = self._counter % len(self.base_urls)
                    request_index = self._counter
                    self._counter += 1
                if self.mock_llm:
                    response = self._mock_response(stage, prompt, index)
                else:
                    if progress is not None and self.stream_responses:
                        stream_id = f"{stage}-{request_index}"
                    executor = (
                        self._priority_executor
                        if priority or stage != "proof_generation"
                        else self._generation_executor
                    )
                    response = await asyncio.get_running_loop().run_in_executor(
                        executor,
                        self._call_sync,
                        stage,
                        prompt,
                        index,
                        temperature,
                        progress,
                        stream_id,
                        detail,
                        thinking_budget_tokens,
                        thinking_budget_force_text,
                        thinking_budget_action,
                        prompt_token_ids,
                        max_tokens_override,
                        response_prefix,
                        return_context_ids,
                        visible_output_limit_tokens,
                    )
        except Exception:
            if progress is not None:
                progress.complete(stage, False, detail)
            raise
        finally:
            if progress is not None and stream_id is not None:
                progress.stream_finish(stream_id)
        if response is None:
            raise RuntimeError(f"LLM call at stage={stage} returned no response")
        repetition = response.get("reasoning_repetition")
        if repetition is not None:
            logging.info(
                "Reasoning repetition stage=%s detail=%s gzip_factor=%.4f "
                "repeated_%dword_windows=%.4f (%d/%d) warning=%s",
                stage,
                detail or "-",
                repetition["gzip_factor"],
                repetition["window_words"],
                repetition["repeated_word_window_fraction"],
                repetition["repeated_word_window_count"],
                repetition["word_window_count"],
                repetition["gzip_warning"],
            )
        if progress is not None:
            usage = response.get("usage") or {}
            completion_tokens = usage.get("completion_tokens")
            progress.complete(
                stage,
                bool(response.get("success")),
                detail,
                response.get("latency_s"),
                int(completion_tokens) if completion_tokens is not None else None,
            )
        return response

    def _call_sync(
        self,
        stage: str,
        prompt: str | list[dict[str, str]],
        index: int,
        temperature: Optional[float],
        progress: Optional[PipelineProgress] = None,
        stream_id: Optional[str] = None,
        detail: str = "",
        thinking_budget_tokens: Optional[int] = None,
        thinking_budget_force_text: str = "",
        thinking_budget_action: str = "finalize",
        prompt_token_ids: Optional[list[int]] = None,
        max_tokens_override: Optional[int] = None,
        response_prefix: str = "",
        return_context_ids: bool = False,
        visible_output_limit_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        client = self._clients[index]
        call_log_path: Optional[Path] = None
        extra_body: dict[str, Any] = {}
        if self.sampling.top_k > 0:
            extra_body["top_k"] = self.sampling.top_k
        if self.sampling.min_p is not None:
            extra_body["min_p"] = self.sampling.min_p
        if self.sampling.min_new_tokens > 0:
            extra_body["min_tokens"] = self.sampling.min_new_tokens
        if stage in {"proof_verify"}:
            extra_body["repetition_penalty"] = 1.05
        if thinking_budget_action not in {"finalize", "stop"}:
            raise ValueError(
                "thinking_budget_action must be either 'finalize' or 'stop'"
            )
        if (
            visible_output_limit_tokens is not None
            and int(visible_output_limit_tokens) < 1
        ):
            raise ValueError("visible_output_limit_tokens must be positive")
        started = time.time()
        assistant_prefix = None
        if prompt_token_ids is not None:
            prompt_ids = self._token_ids_to_list(prompt_token_ids)
            rendered_prompt = self.tokenizer.decode(
                prompt_ids,
                skip_special_tokens=False,
            )
        else:
            assistant_prefix = None
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = [dict(message) for message in prompt]
                if not messages or any(
                    message.get("role") not in {"system", "user", "assistant"}
                    or not isinstance(message.get("content"), str)
                    for message in messages
                ):
                    raise ValueError(
                        "prompt messages must contain valid role/content pairs"
                    )
            rendered_prompt, prompt_ids = self._render_completion_prompt(
                messages, assistant_prefix
            )
        prompt_tokens = len(prompt_ids)
        self._maybe_log_stage_prompt(stage, detail, rendered_prompt, prompt_tokens)
        configured_max_tokens = (
            int(max_tokens_override)
            if max_tokens_override is not None
            else self.stage_max_new_tokens.get(stage, self.sampling.max_new_tokens)
        )
        if configured_max_tokens < 1:
            raise ValueError("max_tokens_override must be positive")
        max_tokens = self._adjust_max_tokens_for_context(
            stage,
            prompt_tokens,
            configured_max_tokens,
        )
        call_log_path = self._llm_call_path(progress, stage, detail)
        self._write_llm_call_input(
            call_log_path,
            stage,
            detail,
            rendered_prompt,
            prompt_tokens,
            max_tokens,
        )
        if self.stream_responses:
            extra_body["return_token_ids"] = True
        try:
            request_extra_body = dict(extra_body)
            if (
                "min_tokens" in request_extra_body
                and int(request_extra_body["min_tokens"]) > max_tokens
            ):
                request_extra_body["min_tokens"] = max_tokens
            if self.stream_responses:
                if progress is not None and stream_id is not None:
                    progress.stream_start(stream_id, stage, detail, max_tokens)

                stream_segment_idx = 0

                def stream_completion(
                    active_prompt_ids: list[int],
                    active_max_tokens: int,
                    stop_after_tokens: Optional[int] = None,
                ) -> dict[str, Any]:
                    nonlocal stream_segment_idx
                    stream_segment_idx += 1
                    if stream_segment_idx > 1:
                        self._append_llm_call_input_segment(
                            call_log_path,
                            f"CONTINUATION INPUT PROMPT {stream_segment_idx}",
                            active_prompt_ids,
                            active_max_tokens,
                        )
                    active_extra_body = dict(request_extra_body)
                    if (
                        "min_tokens" in active_extra_body
                        and int(active_extra_body["min_tokens"]) > active_max_tokens
                    ):
                        active_extra_body["min_tokens"] = active_max_tokens
                    stream = client.completions.create(
                        model=self.model,
                        prompt=active_prompt_ids,
                        temperature=self.sampling.temperature
                        if temperature is None
                        else temperature,
                        top_p=self.sampling.top_p,
                        max_tokens=active_max_tokens,
                        stream=True,
                        extra_body=active_extra_body or None,
                    )
                    finish_reason: Optional[str] = None
                    usage: dict[str, Any] = {}
                    streamed_tokens = 0
                    generated_token_ids: list[int] = []
                    content_parts: list[str] = []
                    reasoning_parts: list[str] = []
                    stopped_by_budget = False
                    stopped_by_repetition = False
                    repetition_line: Optional[str] = None
                    repetition_kind: Optional[str] = None
                    repetition_count = 0
                    last_repetition_check_tokens = 0
                    repetition_guard_enabled = (
                        bool(thinking_budget_force_text) and stage == "proof_verify"
                    )
                    try:
                        for chunk in stream:
                            chunk_usage = response_usage_to_dict(
                                getattr(chunk, "usage", None)
                            )
                            if chunk_usage:
                                usage = chunk_usage
                            choices = getattr(chunk, "choices", None) or []
                            if not choices:
                                continue
                            choice = choices[0]
                            choice_finish_reason = getattr(
                                choice, "finish_reason", None
                            )
                            if choice_finish_reason is not None:
                                finish_reason = choice_finish_reason
                            content = getattr(choice, "text", None)
                            choice_extra = getattr(choice, "model_extra", None) or {}
                            reasoning = choice_extra.get(
                                "reasoning_content"
                            ) or choice_extra.get("reasoning")
                            if content:
                                content_parts.append(content)
                            if reasoning:
                                reasoning_parts.append(reasoning)
                            token_ids = getattr(choice, "token_ids", None)
                            if token_ids is None:
                                token_ids = choice_extra.get("token_ids")
                            if token_ids:
                                token_id_list = self._token_ids_to_list(token_ids)
                                generated_token_ids, new_tokens = (
                                    merge_streamed_token_ids(
                                        generated_token_ids,
                                        token_id_list,
                                    )
                                )
                                if new_tokens:
                                    streamed_tokens += new_tokens
                                    if progress is not None and stream_id is not None:
                                        progress.stream_advance(stream_id, new_tokens)
                                    if (
                                        repetition_guard_enabled
                                        and finish_reason is None
                                        and streamed_tokens
                                        - last_repetition_check_tokens
                                        >= self.stream_interval_tokens
                                    ):
                                        last_repetition_check_tokens = streamed_tokens
                                        repetition = (
                                            detect_duplicate_segment_in_recent_tokens(
                                                self.tokenizer,
                                                generated_token_ids,
                                            )
                                        )
                                        if repetition is not None:
                                            (
                                                repetition_line,
                                                repetition_count,
                                                repetition_kind,
                                            ) = repetition
                                            stopped_by_repetition = True
                                            logging.warning(
                                                "Repetition guard triggered stage=%s detail=%s "
                                                "streamed=%d repeated_%s_count=%d text=%r",
                                                stage,
                                                detail,
                                                streamed_tokens,
                                                repetition_kind,
                                                repetition_count,
                                                repetition_line[:200],
                                            )
                                            break
                            if (
                                stop_after_tokens is not None
                                and finish_reason is None
                                and streamed_tokens >= stop_after_tokens
                            ):
                                stopped_by_budget = True
                                break
                            if stopped_by_repetition:
                                break
                    finally:
                        close = getattr(stream, "close", None)
                        if callable(close):
                            close()
                    content_text = "".join(content_parts)
                    reasoning_text = "".join(reasoning_parts)
                    text_parts: list[str] = []
                    if reasoning_text:
                        text_parts.append(f"<think>\n{reasoning_text}\n</think>\n\n")
                    text_parts.append(content_text)
                    return {
                        "text": "".join(text_parts),
                        "finish_reason": finish_reason,
                        "usage": usage,
                        "streamed_tokens": streamed_tokens,
                        "generated_token_ids": generated_token_ids,
                        "stopped_by_budget": stopped_by_budget,
                        "stopped_by_repetition": stopped_by_repetition,
                        "repetition_line": repetition_line,
                        "repetition_kind": repetition_kind,
                        "repetition_count": repetition_count,
                    }

                budget_tokens: Optional[int] = None
                if (
                    thinking_budget_tokens is not None
                    and (
                        thinking_budget_force_text
                        or thinking_budget_action == "stop"
                    )
                    and max_tokens > 1
                ):
                    budget_tokens = min(
                        max(1, int(thinking_budget_tokens)), max_tokens - 1
                    )

                first_segment = stream_completion(
                    prompt_ids,
                    max_tokens,
                    stop_after_tokens=budget_tokens,
                )
                text_parts = [first_segment["text"]]
                generated_token_ids = list(first_segment["generated_token_ids"])
                streamed_tokens = int(first_segment["streamed_tokens"])
                finish_reason = first_segment["finish_reason"]
                usage = first_segment["usage"]
                repetition_guard_applied = bool(
                    first_segment.get("stopped_by_repetition")
                )
                thinking_budget_applied = (
                    bool(first_segment["stopped_by_budget"]) or repetition_guard_applied
                )
                thinking_stop_reason = (
                    "repetition_guard" if repetition_guard_applied else "token_budget"
                )
                thinking_budget_skipped_closed = bool(
                    first_segment["stopped_by_budget"]
                    and has_closed_thinking_block(first_segment["text"])
                )
                visible_output_limit_requested = (
                    int(visible_output_limit_tokens)
                    if visible_output_limit_tokens is not None
                    else None
                )
                visible_output_limit_effective: Optional[int] = None
                visible_output_limit_applied = False
                visible_output_forced_partial_closure = False
                visible_output_forced_tokens = 0

                def continue_after_thinking(
                    continuation_prompt_ids: list[int],
                    remaining_tokens: int,
                    *,
                    visible_tokens_already: int = 0,
                ) -> None:
                    nonlocal finish_reason
                    nonlocal generated_token_ids
                    nonlocal streamed_tokens
                    nonlocal usage
                    nonlocal visible_output_forced_partial_closure
                    nonlocal visible_output_forced_tokens
                    nonlocal visible_output_limit_applied
                    nonlocal visible_output_limit_effective

                    if visible_output_limit_requested is not None:
                        current_text = (
                            response_prefix
                            + (assistant_prefix or "")
                            + "".join(text_parts)
                        )
                        footer_reserve_tokens = len(
                            self._token_ids_to_list(
                                self.tokenizer.encode(
                                    build_visible_output_limit_footer(current_text),
                                    add_special_tokens=False,
                                )
                            )
                        )
                        visible_output_limit_effective = min(
                            max(
                                0,
                                visible_output_limit_requested
                                - visible_tokens_already,
                            ),
                            max(0, remaining_tokens - footer_reserve_tokens),
                        )

                    if visible_output_limit_effective == 0:
                        continuation = {
                            "text": "",
                            "generated_token_ids": [],
                            "streamed_tokens": 0,
                            "finish_reason": None,
                            "usage": {},
                            "stopped_by_budget": True,
                        }
                    else:
                        continuation = stream_completion(
                            continuation_prompt_ids,
                            remaining_tokens,
                            stop_after_tokens=visible_output_limit_effective,
                        )

                    text_parts.append(str(continuation["text"]))
                    generated_token_ids.extend(
                        continuation["generated_token_ids"]
                    )
                    streamed_tokens += int(continuation["streamed_tokens"])
                    finish_reason = continuation["finish_reason"]
                    if continuation["usage"]:
                        usage = continuation["usage"]

                    visible_output_limit_applied = bool(
                        visible_output_limit_requested is not None
                        and continuation.get("stopped_by_budget")
                    )
                    if not visible_output_limit_applied:
                        return

                    combined_text = (
                        response_prefix
                        + (assistant_prefix or "")
                        + "".join(text_parts)
                    )
                    footer = build_visible_output_limit_footer(combined_text)
                    if footer:
                        footer_token_ids = self._token_ids_to_list(
                            self.tokenizer.encode(
                                footer,
                                add_special_tokens=False,
                            )
                        )
                        text_parts.append(footer)
                        generated_token_ids.extend(footer_token_ids)
                        streamed_tokens += len(footer_token_ids)
                        visible_output_forced_tokens = len(footer_token_ids)
                        visible_output_forced_partial_closure = True
                        if progress is not None and stream_id is not None:
                            progress.stream_advance(
                                stream_id,
                                len(footer_token_ids),
                            )
                    finish_reason = "visible_output_limit_reached"
                    logging.warning(
                        "Visible-output limit reached stage=%s detail=%s "
                        "requested=%d effective=%d forced_partial=%s "
                        "forced_tokens=%d",
                        stage,
                        detail,
                        visible_output_limit_requested,
                        visible_output_limit_effective,
                        visible_output_forced_partial_closure,
                        visible_output_forced_tokens,
                    )

                if thinking_budget_applied and thinking_budget_skipped_closed:
                    remaining_tokens = max_tokens - streamed_tokens
                    logging.info(
                        "Skipped thinking budget force text stage=%s detail=%s "
                        "reason=%s budget=%s streamed=%d remaining=%d because </think> was already generated",
                        stage,
                        detail,
                        thinking_stop_reason,
                        budget_tokens,
                        streamed_tokens,
                        remaining_tokens,
                    )
                    if remaining_tokens > 0:
                        continuation_prompt_ids = prompt_ids + generated_token_ids
                        visible_tokens_already = (
                            len(
                                self._token_ids_to_list(
                                    self.tokenizer.encode(
                                        final_visible_output(
                                            response_prefix
                                            + (assistant_prefix or "")
                                            + "".join(text_parts)
                                        ),
                                        add_special_tokens=False,
                                    )
                                )
                            )
                            if visible_output_limit_requested is not None
                            else 0
                        )
                        continue_after_thinking(
                            continuation_prompt_ids,
                            remaining_tokens,
                            visible_tokens_already=visible_tokens_already,
                        )
                    else:
                        finish_reason = "thinking_budget_reached"
                elif thinking_budget_applied and thinking_budget_action == "stop":
                    finish_reason = "thinking_budget_reached"
                    logging.info(
                        "Stopped unfinished thinking for handoff stage=%s detail=%s "
                        "reason=%s budget=%s streamed=%d",
                        stage,
                        detail,
                        thinking_stop_reason,
                        budget_tokens,
                        streamed_tokens,
                    )
                elif thinking_budget_applied:
                    intervention_text = thinking_budget_force_text
                    append_intervention_to_output = True
                    force_token_ids = self.tokenizer.encode(
                        intervention_text,
                        add_special_tokens=False,
                    )
                    force_token_ids = self._token_ids_to_list(force_token_ids)
                    if (
                        append_intervention_to_output
                        and force_token_ids
                        and progress is not None
                        and stream_id is not None
                    ):
                        progress.stream_advance(stream_id, len(force_token_ids))
                    if append_intervention_to_output:
                        text_parts.append(intervention_text)
                        generated_token_ids.extend(force_token_ids)
                        streamed_tokens += len(force_token_ids)
                    remaining_tokens = (
                        max_tokens - streamed_tokens
                    )
                    logging.info(
                        "Applied thinking intervention stage=%s detail=%s reason=%s "
                        "visible=%s budget=%s streamed=%d force_tokens=%d remaining=%d "
                        "repetition_kind=%s repeated_text=%r",
                        stage,
                        detail,
                        thinking_stop_reason,
                        append_intervention_to_output,
                        budget_tokens,
                        streamed_tokens,
                        len(force_token_ids),
                        remaining_tokens,
                        first_segment.get("repetition_kind"),
                        first_segment.get("repetition_line"),
                    )
                    if remaining_tokens > 0:
                        continue_after_thinking(
                            prompt_ids + generated_token_ids,
                            remaining_tokens,
                        )
                    else:
                        finish_reason = "thinking_budget_reached"

                usage["completion_tokens"] = streamed_tokens
                usage["total_tokens"] = prompt_tokens + streamed_tokens
                usage["requested_max_tokens"] = max_tokens
                usage["estimated_prompt_tokens"] = prompt_tokens
                usage["thinking_budget_tokens"] = budget_tokens
                usage["thinking_budget_applied"] = thinking_budget_applied
                usage["thinking_budget_stop_reason"] = (
                    thinking_stop_reason if thinking_budget_applied else None
                )
                usage["thinking_budget_action"] = thinking_budget_action
                usage["repetition_guard_applied"] = repetition_guard_applied
                usage["repetition_guard_line"] = first_segment.get("repetition_line")
                usage["repetition_guard_kind"] = first_segment.get("repetition_kind")
                usage["repetition_guard_count"] = first_segment.get("repetition_count")
                usage["thinking_budget_force_skipped_closed"] = (
                    thinking_budget_skipped_closed
                )
                usage["visible_output_limit_tokens"] = (
                    visible_output_limit_requested
                )
                usage["visible_output_limit_effective_tokens"] = (
                    visible_output_limit_effective
                )
                usage["visible_output_limit_applied"] = (
                    visible_output_limit_applied
                )
                usage["visible_output_forced_partial_closure"] = (
                    visible_output_forced_partial_closure
                )
                usage["visible_output_forced_tokens"] = (
                    visible_output_forced_tokens
                )
                text = response_prefix + (assistant_prefix or "") + "".join(text_parts)
                result = {
                    "stage": stage,
                    "success": True,
                    "error": None,
                    "text": text,
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "server_url": self.base_urls[index],
                    "latency_s": time.time() - started,
                }
                if (
                    thinking_budget_applied
                    and thinking_budget_action == "stop"
                    and not thinking_budget_skipped_closed
                ):
                    result["_thinking_budget_context_ids"] = (
                        prompt_ids + generated_token_ids
                    )
                if return_context_ids:
                    result["_completion_context_ids"] = (
                        prompt_ids + generated_token_ids
                    )
                self._append_llm_call_output(call_log_path, result)
                return result
            if thinking_budget_tokens is not None:
                logging.warning(
                    "Thinking budget for stage=%s requires streaming token IDs; "
                    "running a single-call request because stream_vllm is disabled.",
                    stage,
                )
            response = client.completions.create(
                model=self.model,
                prompt=prompt_ids,
                temperature=self.sampling.temperature
                if temperature is None
                else temperature,
                top_p=self.sampling.top_p,
                max_tokens=max_tokens,
                extra_body=request_extra_body or None,
            )
            choice = response.choices[0]
            text = response_prefix + (assistant_prefix or "") + (choice.text or "")
            usage = response_usage_to_dict(getattr(response, "usage", None))
            usage["requested_max_tokens"] = max_tokens
            usage["estimated_prompt_tokens"] = prompt_tokens
            result = {
                "stage": stage,
                "success": True,
                "error": None,
                "text": text,
                "finish_reason": choice.finish_reason,
                "usage": usage,
                "server_url": self.base_urls[index],
                "latency_s": time.time() - started,
            }
            if return_context_ids:
                completion_ids = self._token_ids_to_list(
                    self.tokenizer.encode(choice.text or "", add_special_tokens=False)
                )
                result["_completion_context_ids"] = prompt_ids + completion_ids
            self._append_llm_call_output(call_log_path, result)
            return result
        except Exception as exc:
            logging.exception(
                "LLM call failed at stage=%s server=%s", stage, self.base_urls[index]
            )
            result = {
                "stage": stage,
                "success": False,
                "error": repr(exc),
                "text": "",
                "finish_reason": None,
                "usage": {},
                "server_url": self.base_urls[index],
                "latency_s": time.time() - started,
            }
            self._append_llm_call_output(call_log_path, result)
            if is_fatal_inference_error(exc):
                raise InferenceServerUnavailable(
                    f"Inference server failed during stage={stage} at "
                    f"{self.base_urls[index]}: {exc!r}"
                ) from exc
            return result

    def _mock_response(
        self,
        stage: str,
        prompt: str | list[dict[str, str]],
        index: int,
    ) -> dict[str, Any]:
        if stage == "selector":
            text = "<selected_id>R0</selected_id>"
        elif stage in {"proof_generation", "proof_refine"}:
            text = (
                "<solution>\n"
                "This is a mock proof produced for runner validation.\n"
                "</solution>\n"
                "<self_evaluation>\n"
                "The mock output is structurally valid.\n"
                "</self_evaluation>\n"
                "<score>1</score>"
            )
        elif stage == "proof_meta_verify":
            text = 'Here is my analysis of the "solution evaluation":\nThe critique is reasonable.\n\n\\boxed{1}'
        else:
            text = (
                "<evaluation>The proof is acceptable for a mock run.</evaluation>\n"
                "<suggestions>No repair is needed.</suggestions>\n"
                "<score>1</score>"
            )
        return {
            "stage": stage,
            "success": True,
            "error": None,
            "text": text,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "server_url": self.base_urls[index],
            "latency_s": 0.0,
        }


def make_output(
    stage: str, response: dict[str, Any], parsed: dict[str, Any], **extra: Any
) -> dict[str, Any]:
    output = {
        "stage": stage,
        "success": response.get("success", False),
        "error": response.get("error"),
        "text": response.get("text", ""),
        "parsed": parsed,
        "finish_reason": response.get("finish_reason"),
        "usage": response.get("usage", {}),
        "server_url": response.get("server_url"),
        "latency_s": response.get("latency_s"),
        "reasoning_repetition": response.get("reasoning_repetition"),
    }
    output.update(extra)
    return output


def score_sort_value(candidate: dict[str, Any]) -> float:
    score = candidate.get("final_score")
    if score is None:
        return -1.0
    try:
        return float(score)
    except (TypeError, ValueError):
        return -1.0


def should_run_meta_verification(verifier: dict[str, Any], cfg: PipelineConfig) -> bool:
    if cfg.meta_n <= 0:
        return False
    score = verifier.get("score")
    if cfg.meta_policy == "all-reviews":
        return score is not None
    return score is not None and score < 1.0


def format_refinement_critique(critique: dict[str, Any]) -> str:
    score = critique.get("score")
    evaluation, _ = clip_middle_text(
        str(critique.get("evaluation") or "").strip(),
        MAX_FORWARDED_EVALUATION_CHARS,
    )
    if score is None:
        score_text = "Verifier score: unknown"
    else:
        score_text = f"Verifier score: {score}"
    return (
        f"{score_text}\n\nHere is my evaluation of the solution:\n{evaluation}".strip()
    )


async def cancel_pending_tasks(
    tasks: list[asyncio.Task[Any]] | set[asyncio.Task[Any]],
) -> int:
    pending = [task for task in tasks if not task.done()]
    if not pending:
        return 0
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    return len(pending)


class VerificationThrottle:
    def __init__(
        self,
        total_generations: int,
        candidate_limit_while_generating: int,
        request_limit_while_generating: int,
    ) -> None:
        self.remaining_generations = max(0, int(total_generations))
        self.candidate_limit = max(0, int(candidate_limit_while_generating))
        self.request_limit = max(0, int(request_limit_while_generating))
        self.active_candidates = 0
        self.active_requests = 0
        self.generations: dict[int, dict[str, Any]] = {}
        self.generation_filter_cache: Optional[dict[str, Any]] = None
        self._condition = asyncio.Condition()

    def _generation_open(self) -> bool:
        return self.remaining_generations > 0

    async def mark_generation_done(
        self,
        attempt_idx: int,
        generation: Optional[dict[str, Any]] = None,
    ) -> None:
        async with self._condition:
            if generation is not None:
                self.generations[int(attempt_idx)] = generation
            self.remaining_generations = max(0, self.remaining_generations - 1)
            self._condition.notify_all()

    async def wait_for_all_generations(self) -> None:
        async with self._condition:
            while self._generation_open():
                await self._condition.wait()

    def valid_generations(self) -> list[dict[str, Any]]:
        return [self.generations[index] for index in sorted(self.generations)]

    @contextlib.asynccontextmanager
    async def candidate_slot(self, attempt_idx: int, round_idx: int) -> Any:
        acquired = False
        async with self._condition:
            while (
                self._generation_open()
                and self.candidate_limit > 0
                and self.active_candidates >= self.candidate_limit
            ):
                await self._condition.wait()
            if self._generation_open() and self.candidate_limit > 0:
                self.active_candidates += 1
                acquired = True
        try:
            yield
        finally:
            if acquired:
                async with self._condition:
                    self.active_candidates = max(0, self.active_candidates - 1)
                    self._condition.notify_all()

    @contextlib.asynccontextmanager
    async def request_slot(self, stage: str, detail: str) -> Any:
        acquired = False
        async with self._condition:
            while (
                self._generation_open()
                and self.request_limit > 0
                and self.active_requests >= self.request_limit
            ):
                await self._condition.wait()
            if self._generation_open() and self.request_limit > 0:
                self.active_requests += 1
                acquired = True
        try:
            yield
        finally:
            if acquired:
                async with self._condition:
                    self.active_requests = max(0, self.active_requests - 1)
                    self._condition.notify_all()


def resolve_thinking_budget_tokens(
    attempt_idx: int,
    cfg: PipelineConfig,
    *,
    solve_round_idx: int = 0,
    can_restart: bool = False,
) -> Optional[int]:
    if not cfg.thinking_budget_enabled or not cfg.proof_generation_thinking_budgets:
        return None
    final_round_tokens = int(
        getattr(cfg, "thinking_budget_final_round_tokens", 0)
    )
    if solve_round_idx > 0 and not can_restart and final_round_tokens > 0:
        return min(
            max(1, final_round_tokens),
            max(1, int(cfg.proof_max_new_tokens) - 1),
        )
    index = min(max(0, attempt_idx), len(cfg.proof_generation_thinking_budgets) - 1)
    raw_budget = int(cfg.proof_generation_thinking_budgets[index])
    min_budget = max(1, int(cfg.proof_max_new_tokens) - 5000)
    return max(raw_budget, min_budget)


def resolve_proof_generation_temperature(
    attempt_idx: int, cfg: PipelineConfig
) -> float:
    if not cfg.proof_generation_temperatures:
        return float(cfg.default_temperature)
    if 0 <= attempt_idx < len(cfg.proof_generation_temperatures):
        return float(cfg.proof_generation_temperatures[attempt_idx])
    return float(cfg.default_temperature)


def response_hit_unfinished_thinking_budget(response: dict[str, Any]) -> bool:
    usage = response.get("usage") or {}
    return bool(
        usage.get("thinking_budget_applied")
        and usage.get("thinking_budget_action") == "stop"
        and not usage.get("thinking_budget_force_skipped_closed")
        and response.get("_thinking_budget_context_ids")
    )


async def request_proof_attempt_handoff(
    *,
    scheduler: ChatScheduler,
    generation_response: dict[str, Any],
    prompt_family: str,
    force_text: str,
    attempt_idx: int,
    round_idx: int,
    cfg: PipelineConfig,
    progress: Optional[PipelineProgress],
    source_stage: str = "proof_generation",
    handoff_stage: str = "proof_handoff",
    finalize_stage: str = "proof_finalize",
    max_new_tokens: Optional[int] = None,
) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
    context_ids = generation_response.get("_thinking_budget_context_ids")
    if not context_ids:
        return None, []

    outputs: list[dict[str, Any]] = []
    handoff_mode = str(
        getattr(cfg, "thinking_budget_handoff_mode", DEFAULT_HANDOFF_MODE)
    )
    prompt_variant = str(
        getattr(
            cfg,
            "thinking_budget_handoff_prompt_variant",
            DEFAULT_HANDOFF_VARIANT,
        )
    )
    if handoff_mode == "lossless_partial" and prompt_family == PROMPT_FAMILY_OPD:
        try:
            finalized = await finalize_budget_stopped_generation(
                scheduler=scheduler,
                generation_response=generation_response,
                force_text=force_text,
                attempt_idx=attempt_idx,
                round_idx=round_idx,
                cfg=cfg,
                progress=progress,
                source_stage=source_stage,
                finalize_stage=finalize_stage,
                max_new_tokens=max_new_tokens,
            )
            partial_progress = extract_forced_partial_progress(
                str(finalized.get("text") or "")
            )
            handoff_text = build_lossless_partial_handoff(partial_progress)
            parsed = parse_handoff_response(handoff_text)
            response = {
                "success": parsed["is_valid"],
                "text": handoff_text,
                "finish_reason": "partial_passthrough",
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "server_url": finalized.get("server_url"),
                "latency_s": 0.0,
            }
            outputs.append(
                make_output(
                    handoff_stage,
                    response,
                    parsed,
                    round_idx=round_idx,
                    handoff_try=1,
                    prompt_variant=prompt_variant,
                    handoff_mode=handoff_mode,
                    source_stage=source_stage,
                    partial_progress_chars=len(partial_progress),
                )
            )
            if parsed["is_valid"]:
                return parsed, outputs
        except InferenceServerUnavailable:
            raise
        except Exception:
            logging.exception(
                "Lossless partial handoff failed candidate=%d round=%d; "
                "falling back to the model-written handoff",
                attempt_idx,
                round_idx,
            )

    prompt_ids = build_user_turn_prompt_ids(
        scheduler.tokenizer,
        context_ids,
        build_handoff_instruction(
            prompt_variant
        ),
        close_open_thinking=True,
    )
    previous_context_ids: Optional[list[int]] = None
    for handoff_try in range(2):
        if handoff_try > 0:
            if not previous_context_ids:
                break
            prompt_ids = build_user_turn_prompt_ids(
                scheduler.tokenizer,
                previous_context_ids,
                build_handoff_repair_instruction(prompt_variant),
                close_open_thinking=False,
            )
        response = await scheduler.call(
            handoff_stage,
            "",
            progress=progress,
            detail=(
                f"candidate={attempt_idx} round={round_idx} "
                f"try={handoff_try + 1}"
            ),
            temperature=float(
                getattr(cfg, "thinking_budget_handoff_temperature", 0.7)
            ),
            prompt_token_ids=prompt_ids,
            max_tokens_override=int(
                getattr(cfg, "thinking_budget_handoff_max_tokens", 4096)
            ),
            response_prefix=HANDOFF_ASSISTANT_PREFIX,
            return_context_ids=True,
        )
        previous_context_ids = response.pop("_completion_context_ids", None)
        parsed = parse_handoff_response(response.get("text", ""))
        checkpoint_analysis = (
            analyze_proof_checkpoint(response.get("text", ""))
            if prompt_variant == PROOF_CHECKPOINT_VARIANT
            else None
        )
        handoff_is_acceptable = bool(parsed["is_valid"])
        if checkpoint_analysis is not None:
            handoff_is_acceptable = bool(
                checkpoint_analysis["is_valid_checkpoint"]
            )
        output = make_output(
            handoff_stage,
            response,
            parsed,
            round_idx=round_idx,
            handoff_try=handoff_try + 1,
            handoff_mode="model",
            source_stage=source_stage,
            prompt_variant=prompt_variant,
            checkpoint_analysis=checkpoint_analysis,
        )
        outputs.append(output)
        if handoff_is_acceptable:
            return parsed, outputs
        if progress is not None:
            progress.log(
                "candidate=%d round=%d stage=handoff status=invalid try=%d missing=%s",
                attempt_idx,
                round_idx,
                handoff_try + 1,
                ",".join(parsed["missing_sections"])
                or ",".join(
                    str(issue)
                    for issue in (checkpoint_analysis or {}).get("issues", [])
                ),
            )
    return None, outputs


async def finalize_budget_stopped_generation(
    *,
    scheduler: ChatScheduler,
    generation_response: dict[str, Any],
    force_text: str,
    attempt_idx: int,
    round_idx: int,
    cfg: PipelineConfig,
    progress: Optional[PipelineProgress],
    source_stage: str = "proof_generation",
    finalize_stage: str = "proof_finalize",
    max_new_tokens: Optional[int] = None,
) -> dict[str, Any]:
    context_ids = list(generation_response.get("_thinking_budget_context_ids") or [])
    if not context_ids:
        return generation_response
    force_ids = scheduler._token_ids_to_list(
        scheduler.tokenizer.encode(force_text, add_special_tokens=False)
    )
    used_tokens = int(
        (generation_response.get("usage") or {}).get("completion_tokens") or 0
    )
    remaining_tokens = max(
        1,
        int(
            cfg.proof_max_new_tokens
            if max_new_tokens is None
            else max_new_tokens
        )
        - used_tokens
        - len(force_ids),
    )
    continuation = await scheduler.call(
        finalize_stage,
        "",
        progress=progress,
        detail=f"candidate={attempt_idx} round={round_idx} budget_fallback=1",
        prompt_token_ids=context_ids + force_ids,
        max_tokens_override=remaining_tokens,
        response_prefix=force_text,
    )
    combined = dict(generation_response)
    combined["stage"] = source_stage
    combined["text"] = (
        str(generation_response.get("text") or "")
        + str(continuation.get("text") or "")
    )
    combined["finish_reason"] = continuation.get("finish_reason")
    combined["server_url"] = continuation.get("server_url")
    combined["latency_s"] = float(generation_response.get("latency_s") or 0.0) + float(
        continuation.get("latency_s") or 0.0
    )
    usage = dict(generation_response.get("usage") or {})
    continuation_usage = continuation.get("usage") or {}
    final_completion_tokens = (
        used_tokens
        + len(force_ids)
        + int(continuation_usage.get("completion_tokens") or 0)
    )
    usage["completion_tokens"] = final_completion_tokens
    usage["total_tokens"] = int(usage.get("estimated_prompt_tokens") or 0) + (
        final_completion_tokens
    )
    usage["thinking_budget_fallback_finalized"] = True
    combined["usage"] = usage
    combined.pop("_thinking_budget_context_ids", None)
    return combined


def resolve_candidate_prompt_family(
    attempt_idx: int,
    total_candidates: int,
    cfg: PipelineConfig,
) -> str:
    deepseek_count = int(cfg.deepseek_math_v2_candidate_count)
    if not 0 <= deepseek_count <= int(total_candidates):
        raise ValueError(
            "deepseek_math_v2_candidate_count must be between 0 and "
            f"the candidate count ({total_candidates}), got {deepseek_count}"
        )
    if 0 <= attempt_idx < deepseek_count:
        return PROMPT_FAMILY_DEEPSEEK_MATH_V2
    return PROMPT_FAMILY_OPD


def is_proof_only_candidate(
    attempt_idx: int, total_candidates: int, cfg: PipelineConfig
) -> bool:
    proof_only_count = max(
        0, min(int(cfg.proof_only_candidate_count), int(total_candidates))
    )
    if proof_only_count <= 0:
        return False
    return attempt_idx >= max(0, int(total_candidates) - proof_only_count)


def is_length_stopped_proof_only_generation(initial_generation: dict[str, Any]) -> bool:
    generation_output = initial_generation.get("generation_output") or {}
    return (
        initial_generation.get("generation_mode") == "proof_only"
        and generation_output.get("finish_reason") == "length"
    )


def is_zero_self_scored_generation(initial_generation: dict[str, Any]) -> bool:
    if initial_generation.get("generation_mode") == "proof_only":
        return False
    parsed = initial_generation.get("generation_parsed") or {}
    score = coerce_score(parsed.get("self_score"))
    return score == 0.0


def filter_generations_before_verification(
    initial_generations: list[dict[str, Any]],
    cfg: PipelineConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return generation candidates for selection, verifier-eligible candidates, and skip records."""
    if len(initial_generations) <= 1:
        return list(initial_generations), list(initial_generations), []

    selection_pool = list(initial_generations)
    skipped: list[dict[str, Any]] = []

    complete_or_normal = [
        generation
        for generation in selection_pool
        if not is_length_stopped_proof_only_generation(generation)
    ]
    if complete_or_normal:
        for generation in selection_pool:
            if is_length_stopped_proof_only_generation(generation):
                skipped.append(
                    {
                        "attempt_idx": generation.get("attempt_idx"),
                        "reason": "proof_only_finish_reason_length",
                        "proof_chars": len(str(generation.get("proof") or "")),
                    }
                )
        selection_pool = complete_or_normal

    if cfg.skip_self_score_zero:
        nonzero_or_unscored = [
            generation
            for generation in selection_pool
            if not is_zero_self_scored_generation(generation)
        ]
        if nonzero_or_unscored:
            for generation in selection_pool:
                if is_zero_self_scored_generation(generation):
                    skipped.append(
                        {
                            "attempt_idx": generation.get("attempt_idx"),
                            "reason": "self_score_zero",
                            "proof_chars": len(str(generation.get("proof") or "")),
                        }
                    )
            selection_pool = nonzero_or_unscored

    verifier_pool = [
        generation
        for generation in selection_pool
        if not is_length_stopped_proof_only_generation(generation)
    ]
    return selection_pool, verifier_pool, skipped


def make_generation_only_candidate(
    initial_generation: dict[str, Any],
    final_status: str,
) -> dict[str, Any]:
    parsed = initial_generation.get("generation_parsed") or {}
    proof = str(initial_generation.get("proof") or "")
    return {
        "attempt_idx": initial_generation.get("attempt_idx"),
        "prompt_family": initial_generation.get("prompt_family"),
        "generation_mode": initial_generation.get("generation_mode"),
        "proof_generation_output": initial_generation.get("generation_output"),
        "proof_generation_outputs": initial_generation.get(
            "generation_outputs",
            [initial_generation.get("generation_output")],
        ),
        "proof_handoff_output": initial_generation.get("handoff_outputs", []),
        "proof_handoffs": initial_generation.get("handoffs", []),
        "budget_restart_count": int(
            initial_generation.get("budget_restart_count")
            or initial_generation.get("consumed_refine_rounds")
            or 0
        ),
        "proof_verify_output": [],
        "proof_meta_verify_output": [],
        "proof_refine_output": [],
        "proof_refine_attempt_output": [],
        "proof_refine_handoff_output": [],
        "proof_refine_handoffs": [],
        "refine_budget_restart_count": 0,
        "proof_solution": proof,
        "self_evaluation": parsed.get("self_evaluation"),
        "self_score": parsed.get("self_score"),
        "validated_critiques": [],
        "verifier_score_summaries": [],
        "final_score": parsed.get("self_score"),
        "final_status": final_status,
        "low_scores_seen": 0,
        "strict_pass": False,
        "all_verifiers_passed": False,
        "meta_valid_count": 0,
        "meta_checked_count": 0,
        "meta_summary_by_verifier": {},
        "success": True,
        "generation_only": True,
    }


async def maybe_wait_all_candidates(
    initial_generation: dict[str, Any],
    cfg: PipelineConfig,
    throttle: Optional[VerificationThrottle],
    progress: Optional[PipelineProgress] = None,
) -> dict[str, Any]:
    if throttle is None or not cfg.wait_for_all_generations_before_verify:
        return {"action": "verify"}

    await throttle.wait_for_all_generations()
    cache_created = False
    async with throttle._condition:
        if throttle.generation_filter_cache is None:
            initial_generations = throttle.valid_generations()
            selection_generations, verifier_generations, skipped = (
                filter_generations_before_verification(initial_generations, cfg)
            )
            selected_generation = None
            selected_status = None
            if len(selection_generations) == 1:
                selected_generation = selection_generations[0]
                selected_status = (
                    "single_valid_generation"
                    if len(initial_generations) == 1
                    else "single_candidate_after_generation_filter"
                )
            elif not verifier_generations and selection_generations:
                selected_generation = selection_generations[
                    select_generation_fallback_index(selection_generations)
                ]
                selected_status = "no_verifier_eligible_generation_fallback"
            throttle.generation_filter_cache = {
                "selection_count": len(selection_generations),
                "verifier_attempts": {
                    int(generation.get("attempt_idx") or 0)
                    for generation in verifier_generations
                },
                "skipped_by_attempt": {
                    int(item.get("attempt_idx")): item
                    for item in skipped
                    if item.get("attempt_idx") is not None
                },
                "selected_attempt": (
                    int(selected_generation.get("attempt_idx") or 0)
                    if selected_generation is not None
                    else None
                ),
                "selected_status": selected_status,
                "selected_generation": selected_generation,
                "skipped_count": len(skipped),
            }
            cache_created = True
        cache = dict(throttle.generation_filter_cache)

    if progress is not None and cache_created:
        progress.log(
            "stage=generation_filter status=complete selection=%d verifier=%d skipped=%d",
            cache["selection_count"],
            len(cache["verifier_attempts"]),
            cache["skipped_count"],
        )
        for skipped in cache["skipped_by_attempt"].values():
            progress.log(
                "candidate=%s stage=generation_filter status=skip reason=%s proof_chars=%s",
                skipped.get("attempt_idx"),
                skipped.get("reason"),
                skipped.get("proof_chars"),
            )

    attempt_idx = int(initial_generation.get("attempt_idx") or 0)
    if cache["selected_attempt"] is not None:
        if attempt_idx == cache["selected_attempt"]:
            return {
                "action": "verify",
                "selected_status": cache["selected_status"],
            }
        return {"action": "skip", "reason": "not_selected_after_generation_filter"}
    if attempt_idx in cache["skipped_by_attempt"]:
        return {
            "action": "skip",
            "reason": cache["skipped_by_attempt"][attempt_idx].get("reason"),
        }
    if attempt_idx not in cache["verifier_attempts"]:
        return {
            "action": "skip",
            "reason": "not_verifier_eligible_after_generation_filter",
        }
    return {"action": "verify"}


async def generate_single_attempt(
    question: str,
    attempt_idx: int,
    total_candidates: int,
    scheduler: ChatScheduler,
    cfg: PipelineConfig,
    problem_id: Any = "problem",
    progress: Optional[PipelineProgress] = None,
) -> Optional[dict[str, Any]]:
    prompt_family = resolve_candidate_prompt_family(
        attempt_idx,
        total_candidates,
        cfg,
    )
    if prompt_family == PROMPT_FAMILY_DEEPSEEK_MATH_V2:
        planning_strategy = "baseline"
        generation_mode = "deepseek_markdown"
        generation_prompt = build_deepseek_proof_generation_prompt(question)
        generation_parser = parse_deepseek_generation_response
        thinking_budget_force_text = cfg.deepseek_thinking_budget_force_text
    else:
        planning_strategy = resolve_proof_generation_strategy(
            attempt_idx,
            cfg,
            question,
        )
        generation_mode = "opd_xml"
        generation_prompt = build_opd_proof_generation_prompt(
            question,
            planning_strategy=planning_strategy,
        )
        generation_parser = parse_generation_response
        thinking_budget_force_text = cfg.thinking_budget_force_text
    generation_outputs: list[dict[str, Any]] = []
    handoff_outputs: list[dict[str, Any]] = []
    handoffs: list[dict[str, Any]] = []
    solve_round_idx = 0
    active_prompt = generation_prompt

    while True:
        restart_until_complete = bool(
            getattr(cfg, "thinking_budget_restart_until_complete", False)
        )
        can_restart = bool(
            getattr(cfg, "thinking_budget_handoff_enabled", False)
            and (
                restart_until_complete
                or solve_round_idx < cfg.refine_rounds
            )
        )
        is_final_restart_round = bool(solve_round_idx > 0 and not can_restart)
        round_force_text = thinking_budget_force_text
        if (
            is_final_restart_round
            and prompt_family == PROMPT_FAMILY_OPD
            and int(getattr(cfg, "thinking_budget_final_round_tokens", 0)) > 0
        ):
            round_force_text = RESTART_FINALIZE_FORCE_TEXT
        if progress is not None:
            progress.log(
                "candidate=%d round=%d stage=generation status=start mode=%s "
                "prompt_family=%s planning_strategy=%s budget_action=%s handoff_mode=%s "
                "restart_strategy=%s",
                attempt_idx,
                solve_round_idx,
                generation_mode,
                prompt_family,
                planning_strategy,
                "stop" if can_restart else "finalize",
                getattr(
                    cfg,
                    "thinking_budget_handoff_mode",
                    DEFAULT_HANDOFF_MODE,
                ),
                getattr(
                    cfg,
                    "thinking_budget_restart_strategy",
                    DEFAULT_RESTART_STRATEGY,
                ),
            )
        generation_response = await scheduler.call(
            "proof_generation",
            active_prompt,
            progress=progress,
            detail=(
                f"candidate={attempt_idx} round={solve_round_idx} "
                f"mode={generation_mode} prompt_family={prompt_family} "
                f"planning_strategy={planning_strategy}"
            ),
            temperature=resolve_proof_generation_temperature(attempt_idx, cfg),
            thinking_budget_tokens=resolve_thinking_budget_tokens(
                attempt_idx,
                cfg,
                solve_round_idx=solve_round_idx,
                can_restart=can_restart,
            ),
            thinking_budget_force_text=round_force_text,
            thinking_budget_action="stop" if can_restart else "finalize",
            priority=solve_round_idx > 0,
        )

        if can_restart and response_hit_unfinished_thinking_budget(
            generation_response
        ):
            if progress is not None:
                progress.log(
                    "candidate=%d round=%d stage=handoff status=start",
                    attempt_idx,
                    solve_round_idx,
                )
            handoff_parsed, new_handoff_outputs = await request_proof_attempt_handoff(
                scheduler=scheduler,
                generation_response=generation_response,
                prompt_family=prompt_family,
                force_text=thinking_budget_force_text,
                attempt_idx=attempt_idx,
                round_idx=solve_round_idx,
                cfg=cfg,
                progress=progress,
            )
            handoff_outputs.extend(new_handoff_outputs)
            if handoff_parsed is not None:
                generation_outputs.append(
                    make_output(
                        "proof_generation",
                        generation_response,
                        {},
                        round_idx=solve_round_idx,
                        prompt_family=prompt_family,
                        planning_strategy=planning_strategy,
                        budget_stopped=True,
                        consumed_by_handoff=True,
                    )
                )
                handoffs.append(handoff_parsed)
                solve_round_idx += 1
                active_prompt = append_restart_instruction(
                    generation_prompt,
                    handoff_parsed["text"],
                    solve_round_idx,
                    strategy=getattr(
                        cfg,
                        "thinking_budget_restart_strategy",
                        DEFAULT_RESTART_STRATEGY,
                    ),
                )
                if progress is not None:
                    progress.log(
                        "candidate=%d round=%d stage=handoff status=complete "
                        "restart_round=%d handoff_chars=%d",
                        attempt_idx,
                        solve_round_idx - 1,
                        solve_round_idx,
                        len(handoff_parsed["text"]),
                    )
                continue

            logging.warning(
                "Handoff generation failed candidate=%d round=%d; "
                "falling back to final partial-proof completion",
                attempt_idx,
                solve_round_idx,
            )
            generation_response = await finalize_budget_stopped_generation(
                scheduler=scheduler,
                generation_response=generation_response,
                force_text=thinking_budget_force_text,
                attempt_idx=attempt_idx,
                round_idx=solve_round_idx,
                cfg=cfg,
                progress=progress,
            )

        log_generation_visible_output(
            "proof_generation",
            generation_response.get("text", ""),
            problem_id=problem_id,
            attempt_idx=attempt_idx,
            round_idx=solve_round_idx,
        )
        generation_parsed = generation_parser(
            generation_response.get("text", ""),
            require_self_evaluation=True,
        )
        generation_parsed["generation_mode"] = generation_mode
        generation_parsed["prompt_family"] = prompt_family
        generation_parsed["planning_strategy"] = planning_strategy
        generation_output = make_output(
            "proof_generation",
            generation_response,
            generation_parsed,
            round_idx=solve_round_idx,
            prompt_family=prompt_family,
            planning_strategy=planning_strategy,
        )
        generation_outputs.append(generation_output)
        if not require_valid_candidate_response(
            generation_parsed,
            problem_id=problem_id,
            attempt_idx=attempt_idx,
            stage="proof_generation",
            round_idx=solve_round_idx,
        ):
            return None
        proof = generation_parsed["proof"]
        if progress is not None:
            progress.log(
                "candidate=%d round=%d stage=generation status=parsed mode=%s "
                "has_solution=%s has_self_evaluation=%s proof_chars=%d self_score=%s "
                "budget_restarts=%d",
                attempt_idx,
                solve_round_idx,
                generation_mode,
                generation_parsed["has_solution_section"],
                generation_parsed["has_self_evaluation_section"],
                len(proof),
                generation_parsed.get("self_score"),
                len(handoffs),
            )
        consumed_refine_rounds = (
            0
            if bool(
                getattr(
                    cfg,
                    "thinking_budget_handoff_preserve_refine_rounds",
                    False,
                )
            )
            else solve_round_idx
        )
        return {
            "attempt_idx": attempt_idx,
            "prompt_family": prompt_family,
            "planning_strategy": planning_strategy,
            "generation_output": generation_output,
            "generation_outputs": generation_outputs,
            "handoff_outputs": handoff_outputs,
            "handoffs": handoffs,
            "budget_restart_count": solve_round_idx,
            "consumed_refine_rounds": consumed_refine_rounds,
            "generation_parsed": generation_parsed,
            "generation_mode": generation_mode,
            "proof": proof,
        }


async def run_verification_round(
    question: str,
    proof: str,
    self_evaluation: str,
    attempt_idx: int,
    round_idx: int,
    scheduler: ChatScheduler,
    cfg: PipelineConfig,
    prompt_family: str = PROMPT_FAMILY_OPD,
    prior_critiques: Optional[list[dict[str, Any]]] = None,
    progress: Optional[PipelineProgress] = None,
    throttle: Optional[VerificationThrottle] = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    verifier_results_by_index: dict[int, dict[str, Any]] = {}
    verifier_outputs_by_index: dict[int, dict[str, Any]] = {}
    meta_results_by_verifier: dict[int, list[dict[str, Any]]] = {}
    proof_meta_verify_output: list[dict[str, Any]] = []
    task_info: dict[asyncio.Task[Any], dict[str, Any]] = {}
    pending: set[asyncio.Task[Any]] = set()
    cancelled_count = 0
    early_stop = False

    def add_task(coro: Any, info: dict[str, Any]) -> None:
        task = asyncio.create_task(coro)
        pending.add(task)
        task_info[task] = info

    async def verification_call(
        stage: str, prompt: str, **kwargs: Any
    ) -> dict[str, Any]:
        detail = str(kwargs.get("detail") or "")
        if throttle is None:
            return await scheduler.call(stage, prompt, **kwargs)
        async with throttle.request_slot(stage, detail):
            return await scheduler.call(stage, prompt, **kwargs)

    def sorted_verifier_results() -> list[dict[str, Any]]:
        return [
            verifier_results_by_index[index]
            for index in sorted(verifier_results_by_index)
        ]

    def current_aggregation() -> dict[str, Any]:
        return aggregate_proof_label(
            sorted_verifier_results(),
            meta_results_by_verifier,
            cfg.min_valid_low,
            strict_pass_meta=cfg.strict_pass_meta and cfg.meta_n > 0,
            meta_n=cfg.meta_n,
            audit_positive_meta=cfg.meta_policy == "all-reviews",
        )

    def enough_validated_critiques() -> bool:
        if not cfg.verification_early_stop:
            return False
        aggregation = current_aggregation()
        return (
            round_idx < cfg.refine_rounds
            and aggregation.get("final_score") is not None
            and aggregation.get("final_score") < 1.0
            and len(aggregation.get("validated_critiques") or [])
            >= max(1, cfg.refine_review_n)
        )

    def finalize_verifier(
        verifier_index: int,
        verifier_role: str,
        verifier_group: str,
        response: dict[str, Any],
        parsed: dict[str, Any],
        output: dict[str, Any],
    ) -> None:
        verifier_result = {
            "verifier_index": verifier_index,
            "verifier_role": verifier_role,
            "verifier_group": verifier_group,
            "success": response.get("success", False),
            "error": response.get("error"),
            **parsed,
        }
        verifier_results_by_index[verifier_index] = verifier_result
        verifier_outputs_by_index[verifier_index] = output
        if parsed.get("evaluation_clipped") or not parsed.get(
            "evaluation_marker_found"
        ):
            logging.info(
                "Verifier evaluation extracted candidate=%d round=%d verifier=%d "
                "marker_found=%s raw_chars=%s forwarded_chars=%s clipped=%s score=%s",
                attempt_idx,
                round_idx,
                verifier_index,
                parsed.get("evaluation_marker_found"),
                parsed.get("evaluation_raw_chars"),
                parsed.get("evaluation_forwarded_chars"),
                parsed.get("evaluation_clipped"),
                parsed.get("score"),
            )
        if not should_run_meta_verification(verifier_result, cfg):
            return
        for meta_idx in range(cfg.meta_n):
            add_task(
                verification_call(
                    "proof_meta_verify",
                    build_deepseek_meta_verification_prompt(
                        question,
                        proof,
                        verifier_result.get("evaluation", ""),
                        audit_positive_verdicts=(
                            cfg.meta_policy == "all-reviews"
                        ),
                    ),
                    progress=progress,
                    detail=(
                        f"candidate={attempt_idx} round={round_idx} "
                        f"verifier={verifier_index} meta={meta_idx}"
                    ),
                    thinking_budget_tokens=(
                        cfg.meta_thinking_budget_tokens
                        if cfg.thinking_budget_enabled
                        else None
                    ),
                    thinking_budget_force_text=cfg.meta_thinking_budget_force_text,
                ),
                {
                    "kind": "meta",
                    "verifier_index": verifier_index,
                    "meta_index": meta_idx,
                },
            )

    if prompt_family == PROMPT_FAMILY_DEEPSEEK_MATH_V2:
        verifier_parser = parse_deepseek_verifier_response
        verifier_force_text = cfg.deepseek_verifier_thinking_budget_force_text
    elif prompt_family == PROMPT_FAMILY_OPD:
        verifier_parser = parse_verifier_response
        verifier_force_text = cfg.verifier_thinking_budget_force_text
    else:
        raise ValueError(f"unsupported prompt family: {prompt_family!r}")

    verifier_generalist_n = getattr(cfg, "verifier_generalist_n", None)
    for verifier_idx in range(cfg.verify_n):
        if verifier_generalist_n is None:
            audit_role = verifier_audit_role(verifier_idx)
            verifier_role, verifier_group = audit_role[0], "specialist"
        else:
            verifier_role, verifier_group, audit_role = hybrid_verifier_assignment(
                verifier_idx,
                int(verifier_generalist_n),
            )
        audit_role = specialize_verifier_audit_role(question, audit_role)
        if prompt_family == PROMPT_FAMILY_DEEPSEEK_MATH_V2:
            prompt = build_deepseek_proof_verification_prompt(
                question,
                proof,
                verifier_index=(
                    None if verifier_group == "generalist" else verifier_idx
                ),
                prior_critiques=prior_critiques,
                audit_role=audit_role,
            )
        else:
            prompt = build_opd_proof_verification_prompt(
                question,
                proof,
                self_evaluation,
                verifier_index=(
                    None if verifier_group == "generalist" else verifier_idx
                ),
                prior_critiques=prior_critiques,
                audit_role=audit_role,
            )
        add_task(
            verification_call(
                "proof_verify",
                prompt,
                progress=progress,
                detail=(
                    f"candidate={attempt_idx} round={round_idx} "
                    f"verifier={verifier_idx} group={verifier_group} "
                    f"role={verifier_role} "
                    f"prompt_family={prompt_family}"
                ),
                thinking_budget_tokens=(
                    cfg.verifier_thinking_budget_tokens
                    if cfg.thinking_budget_enabled
                    else None
                ),
                thinking_budget_force_text=verifier_force_text,
            ),
            {
                "kind": "verifier",
                "verifier_index": verifier_idx,
                "verifier_role": verifier_role,
                "verifier_group": verifier_group,
            },
        )

    try:
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                info = task_info.pop(task)
                kind = info["kind"]
                try:
                    response = task.result()
                except Exception as exc:
                    logging.exception(
                        "Verification task failed candidate=%d round=%d kind=%s info=%s",
                        attempt_idx,
                        round_idx,
                        kind,
                        info,
                    )
                    response = {
                        "success": False,
                        "error": repr(exc),
                        "text": "",
                        "finish_reason": None,
                        "usage": {},
                        "server_url": None,
                        "latency_s": None,
                    }

                if kind == "verifier":
                    verifier_index = int(info["verifier_index"])
                    verifier_role = str(info["verifier_role"])
                    verifier_group = str(info["verifier_group"])
                    log_verifier_tail_output(
                        "proof_verify",
                        response.get("text", ""),
                        attempt_idx=attempt_idx,
                        round_idx=round_idx,
                        verifier_index=verifier_index,
                    )
                    parsed = verifier_parser(response.get("text", ""))
                    output = make_output(
                        "proof_verify",
                        response,
                        parsed,
                        round_idx=round_idx,
                        verifier_index=verifier_index,
                        verifier_role=verifier_role,
                        verifier_group=verifier_group,
                        prompt_family=prompt_family,
                    )
                    finalize_verifier(
                        verifier_index,
                        verifier_role,
                        verifier_group,
                        response,
                        parsed,
                        output,
                    )

                elif kind == "meta":
                    verifier_index = int(info["verifier_index"])
                    meta_idx = int(info["meta_index"])
                    log_verifier_tail_output(
                        "proof_meta_verify",
                        response.get("text", ""),
                        attempt_idx=attempt_idx,
                        round_idx=round_idx,
                        verifier_index=verifier_index,
                        meta_index=meta_idx,
                    )
                    parsed = parse_meta_verifier_response(response.get("text", ""))
                    meta_result = {
                        "meta_index": meta_idx,
                        "verifier_index": verifier_index,
                        "success": response.get("success", False),
                        "error": response.get("error"),
                        **parsed,
                    }
                    meta_output = make_output(
                        "proof_meta_verify",
                        response,
                        parsed,
                        round_idx=round_idx,
                        verifier_index=verifier_index,
                        meta_index=meta_idx,
                    )
                    meta_results_by_verifier.setdefault(verifier_index, []).append(
                        meta_result
                    )
                    proof_meta_verify_output.append(meta_output)

                if enough_validated_critiques():
                    print(
                        "early stopping verification round due to enough validated critiques"
                    )
                    early_stop = True
            if early_stop:
                cancelled_count += await cancel_pending_tasks(pending)
                pending.clear()
    finally:
        if pending:
            cancelled_count += await cancel_pending_tasks(pending)

    verifier_results = sorted_verifier_results()
    verifier_outputs = [
        verifier_outputs_by_index[index] for index in sorted(verifier_outputs_by_index)
    ]
    final_aggregation = current_aggregation()
    final_aggregation["verification_early_stop"] = early_stop
    final_aggregation["verification_cancelled_tasks"] = cancelled_count
    return (
        verifier_results,
        verifier_outputs,
        meta_results_by_verifier,
        proof_meta_verify_output,
        final_aggregation,
    )


def resolve_refinement_thinking_budget_tokens(
    cfg: PipelineConfig,
    *,
    restart_idx: int,
    can_restart: bool,
) -> Optional[int]:
    if not getattr(cfg, "thinking_budget_refine_handoff_enabled", False):
        return None
    raw_budget = (
        int(getattr(cfg, "thinking_budget_refine_tokens", 0))
        if can_restart
        else int(getattr(cfg, "thinking_budget_refine_final_round_tokens", 0))
    )
    if raw_budget <= 0:
        return None
    del restart_idx
    return min(max(1, raw_budget), max(1, int(cfg.proof_max_new_tokens) - 1))


async def run_refinement_with_budget_restart(
    *,
    question: str,
    proof: str,
    self_evaluation: str,
    selected_critiques: list[dict[str, Any]],
    attempt_idx: int,
    round_idx: int,
    scheduler: ChatScheduler,
    cfg: PipelineConfig,
    problem_id: Any,
    progress: Optional[PipelineProgress],
    refinement_strategy: str = "repair",
    strict_pass_challenge: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
]:
    if refinement_strategy == "reconstruct":
        base_prompt = build_opd_proof_reconstruction_prompt(
            question,
            f"P{attempt_idx}",
            proof,
            self_evaluation,
            selected_critiques,
            strict_pass_challenge=strict_pass_challenge,
        )
    elif refinement_strategy == "repair":
        base_prompt = build_opd_proof_refinement_prompt(
            question,
            f"P{attempt_idx}",
            proof,
            self_evaluation,
            selected_critiques,
        )
    else:
        raise ValueError(f"unsupported refinement strategy: {refinement_strategy!r}")
    active_prompt = base_prompt
    attempt_outputs: list[dict[str, Any]] = []
    handoff_outputs: list[dict[str, Any]] = []
    handoffs: list[dict[str, Any]] = []
    restart_idx = 0
    max_restarts = max(
        0,
        int(getattr(cfg, "thinking_budget_refine_max_restarts", 0)),
    )
    restart_until_complete = bool(
        getattr(cfg, "thinking_budget_restart_until_complete", False)
    )

    while True:
        can_restart = bool(
            getattr(cfg, "thinking_budget_refine_handoff_enabled", False)
            and (
                restart_until_complete
                or restart_idx < max_restarts
            )
            and int(getattr(cfg, "thinking_budget_refine_tokens", 0)) > 0
        )
        budget_tokens = resolve_refinement_thinking_budget_tokens(
            cfg,
            restart_idx=restart_idx,
            can_restart=can_restart,
        )
        force_text = (
            FINAL_PARTIAL_FORCE_TEXT
            if can_restart
            else RESTART_FINALIZE_FORCE_TEXT
        )
        final_temperature = getattr(
            cfg,
            "thinking_budget_refine_final_temperature",
            None,
        )
        request_temperature = (
            float(final_temperature)
            if restart_idx > 0
            and not can_restart
            and final_temperature is not None
            else None
        )
        visible_output_limit_tokens = (
            int(
                getattr(
                    cfg,
                    "thinking_budget_refine_visible_output_limit_tokens",
                    0,
                )
            )
            if restart_idx > 0 and not can_restart
            else 0
        )
        if progress is not None:
            progress.log(
                "candidate=%d round=%d stage=refinement_attempt status=start "
                "strategy=%s strict_pass_challenge=%s restart=%d/%s "
                "budget=%s budget_action=%s temperature=%s "
                "visible_limit=%s",
                attempt_idx,
                round_idx,
                refinement_strategy,
                strict_pass_challenge,
                restart_idx,
                "until_complete" if restart_until_complete else max_restarts,
                budget_tokens,
                "stop" if can_restart else "finalize",
                request_temperature,
                visible_output_limit_tokens or None,
            )
        response = await scheduler.call(
            "proof_refine",
            active_prompt,
            temperature=request_temperature,
            progress=progress,
            detail=(
                f"candidate={attempt_idx} round={round_idx} "
                f"strategy={refinement_strategy} restart={restart_idx}"
            ),
            thinking_budget_tokens=budget_tokens,
            thinking_budget_force_text=force_text if budget_tokens else "",
            thinking_budget_action="stop" if can_restart else "finalize",
            visible_output_limit_tokens=visible_output_limit_tokens or None,
        )

        if can_restart and response_hit_unfinished_thinking_budget(response):
            attempt_outputs.append(
                make_output(
                    "proof_refine",
                    response,
                    {},
                    round_idx=round_idx,
                    restart_idx=restart_idx,
                    budget_stopped=True,
                    consumed_by_handoff=True,
                )
            )
            if progress is not None:
                progress.log(
                    "candidate=%d round=%d stage=refinement_handoff status=start "
                    "restart=%d",
                    attempt_idx,
                    round_idx,
                    restart_idx,
                )
            handoff_parsed, new_handoff_outputs = await request_proof_attempt_handoff(
                scheduler=scheduler,
                generation_response=response,
                prompt_family=PROMPT_FAMILY_OPD,
                force_text=FINAL_PARTIAL_FORCE_TEXT,
                attempt_idx=attempt_idx,
                round_idx=round_idx,
                cfg=cfg,
                progress=progress,
                source_stage="proof_refine",
                handoff_stage="proof_refine_handoff",
                finalize_stage="proof_refine_finalize",
                max_new_tokens=cfg.proof_max_new_tokens,
            )
            handoff_outputs.extend(new_handoff_outputs)
            if handoff_parsed is not None:
                handoffs.append(handoff_parsed)
                restart_idx += 1
                active_prompt = append_restart_instruction(
                    base_prompt,
                    handoff_parsed["text"],
                    restart_idx,
                    strategy=getattr(
                        cfg,
                        "thinking_budget_restart_strategy",
                        DEFAULT_RESTART_STRATEGY,
                    ),
                )
                visible_output_target_tokens = int(
                    getattr(
                        cfg,
                        "thinking_budget_refine_visible_output_target_tokens",
                        0,
                    )
                )
                if visible_output_target_tokens > 0:
                    active_prompt = append_final_output_discipline(
                        active_prompt,
                        visible_output_target_tokens,
                    )
                if progress is not None:
                    progress.log(
                        "candidate=%d round=%d stage=refinement_handoff "
                        "status=complete restart=%d handoff_chars=%d",
                        attempt_idx,
                        round_idx,
                        restart_idx,
                        len(handoff_parsed["text"]),
                    )
                continue

            logging.warning(
                "Refinement handoff failed candidate=%d round=%d restart=%d; "
                "falling back to partial-proof finalization",
                attempt_idx,
                round_idx,
                restart_idx,
            )
            response = await finalize_budget_stopped_generation(
                scheduler=scheduler,
                generation_response=response,
                force_text=FINAL_PARTIAL_FORCE_TEXT,
                attempt_idx=attempt_idx,
                round_idx=round_idx,
                cfg=cfg,
                progress=progress,
                source_stage="proof_refine",
                finalize_stage="proof_refine_finalize",
                max_new_tokens=cfg.proof_max_new_tokens,
            )

        log_generation_visible_output(
            "proof_refine",
            response.get("text", ""),
            problem_id=problem_id,
            attempt_idx=attempt_idx,
            round_idx=round_idx,
        )
        parsed = parse_generation_response(response.get("text", ""))
        return (
            response,
            parsed,
            attempt_outputs,
            handoff_outputs,
            handoffs,
            restart_idx,
        )


async def run_single_attempt(
    question: str,
    attempt_idx: int,
    total_candidates: int,
    scheduler: ChatScheduler,
    cfg: PipelineConfig,
    problem_id: Any = "problem",
    progress: Optional[PipelineProgress] = None,
    initial_generation: Optional[dict[str, Any]] = None,
    throttle: Optional[VerificationThrottle] = None,
) -> dict[str, Any]:
    if initial_generation is None:
        initial_generation = await generate_single_attempt(
            question,
            attempt_idx,
            total_candidates,
            scheduler,
            cfg,
            problem_id=problem_id,
            progress=progress,
        )
    if initial_generation is None:
        return {
            "attempt_idx": attempt_idx,
            "skipped": True,
            "skip_reason": "invalid_generation",
            "success": False,
        }
    generation_output = initial_generation["generation_output"]
    generation_parsed = initial_generation["generation_parsed"]
    prompt_family = str(
        initial_generation.get("prompt_family") or PROMPT_FAMILY_OPD
    )
    proof = initial_generation["proof"]
    consumed_refine_rounds = max(
        0,
        min(
            int(initial_generation.get("consumed_refine_rounds") or 0),
            int(cfg.refine_rounds),
        ),
    )

    proof_verify_output: list[dict[str, Any]] = []
    proof_meta_verify_output: list[dict[str, Any]] = []
    proof_refine_output: list[dict[str, Any]] = []
    proof_refine_attempt_output: list[dict[str, Any]] = []
    proof_refine_handoff_output: list[dict[str, Any]] = []
    proof_refine_handoffs: list[dict[str, Any]] = []
    refine_budget_restart_count = 0
    final_aggregation: dict[str, Any] = {
        "final_score": None,
        "final_status": "not_verified",
        "validated_critiques": [],
        "low_scores_seen": 0,
    }
    latest_generation_parsed = generation_parsed
    best_proof = proof
    best_generation_parsed = latest_generation_parsed
    best_aggregation = dict(final_aggregation)
    best_round_idx = -1
    latest_verified_round_idx = -1
    verified_versions: list[dict[str, Any]] = []
    critique_history: list[dict[str, Any]] = []
    critique_history_keys: set[str] = set()
    strict_pass_challenges_used = 0
    pending_strict_pass_challenge = False

    for round_idx in range(consumed_refine_rounds, cfg.refine_rounds + 1):
        if progress is not None:
            progress.log(
                "candidate=%d round=%d/%d stage=verification status=start verify_n=%d proof_chars=%d",
                attempt_idx,
                round_idx,
                cfg.refine_rounds,
                cfg.verify_n,
                len(proof),
            )
        if throttle is not None:
            verification_context = throttle.candidate_slot(attempt_idx, round_idx)
        else:
            verification_context = contextlib.nullcontext()
        async with verification_context:
            (
                verifier_results,
                verifier_outputs,
                meta_results_by_verifier,
                meta_outputs,
                final_aggregation,
            ) = await run_verification_round(
                question,
                proof,
                str(latest_generation_parsed.get("self_evaluation") or ""),
                attempt_idx,
                round_idx,
                scheduler,
                cfg,
                prompt_family=prompt_family,
                prior_critiques=critique_history,
                progress=progress,
                throttle=throttle,
            )
        proof_verify_output.extend(verifier_outputs)
        proof_meta_verify_output.extend(meta_outputs)
        latest_verified_round_idx = round_idx
        if pending_strict_pass_challenge and final_aggregation.get("strict_pass"):
            final_aggregation = {
                **final_aggregation,
                "strict_pass_challenge_survived": True,
            }
        pending_strict_pass_challenge = False
        verified_versions.append(
            snapshot_verified_candidate_version(
                proof,
                latest_generation_parsed,
                final_aggregation,
                round_idx,
            )
        )
        if best_round_idx < 0 or should_replace_retained_candidate(
            final_aggregation,
            best_aggregation,
        ):
            best_proof = proof
            best_generation_parsed = latest_generation_parsed
            best_aggregation = dict(final_aggregation)
            best_round_idx = round_idx
        if progress is not None:
            score_counts = {
                score: sum(
                    1 for result in verifier_results if result.get("score") == score
                )
                for score in (0.0, 0.5, 1.0)
            }
            progress.log(
                "candidate=%d round=%d stage=verification status=complete scores=%s "
                "validated_critiques=%d final_status=%s",
                attempt_idx,
                round_idx,
                score_counts,
                len(final_aggregation.get("validated_critiques") or []),
                final_aggregation.get("final_status"),
            )
        normal_refinement = (
            round_idx < cfg.refine_rounds
            and final_aggregation.get("final_score") is not None
            and final_aggregation.get("final_score") < 1.0
            and final_aggregation.get("validated_critiques")
        )
        strict_pass_challenge = bool(
            round_idx < cfg.refine_rounds
            and final_aggregation.get("strict_pass")
            and strict_pass_challenges_used
            < max(0, int(getattr(cfg, "strict_pass_challenge_rounds", 0)))
        )
        should_refine = bool(normal_refinement or strict_pass_challenge)
        if not should_refine:
            if progress is not None:
                progress.log(
                    "candidate=%d round=%d stage=refinement status=skip final_score=%s "
                    "final_status=%s validated_critiques=%d refine_rounds=%d",
                    attempt_idx,
                    round_idx,
                    final_aggregation.get("final_score"),
                    final_aggregation.get("final_status"),
                    len(final_aggregation.get("validated_critiques") or []),
                    cfg.refine_rounds,
                )
            break

        if strict_pass_challenge:
            current_critiques = strict_pass_challenge_reviews(
                verifier_results,
                cfg.refine_review_n,
            )
            selected_critiques = current_critiques
            strict_pass_challenges_used += 1
        else:
            ranked_critiques = sorted(
                final_aggregation["validated_critiques"],
                key=lambda critique: (
                    float(
                        critique.get("score")
                        if critique.get("score") is not None
                        else 1.0
                    ),
                    int(critique.get("verifier_index") or 0),
                ),
            )
            current_critiques = ranked_critiques[: max(1, cfg.refine_review_n)]
            for critique in current_critiques:
                history_entry = {**critique, "origin_round": round_idx}
                history_key = re.sub(
                    r"\s+",
                    " ",
                    str(
                        history_entry.get("evaluation")
                        or history_entry.get("review")
                        or ""
                    ).strip().lower(),
                )
                if not history_key or history_key in critique_history_keys:
                    continue
                critique_history_keys.add(history_key)
                critique_history.append(history_entry)
            selected_critiques = critique_history[-MAX_PRIOR_VERIFIER_CRITIQUES:]
        refinement_strategy = resolve_refinement_strategy(
            cfg,
            attempt_idx,
            strict_pass_challenge=strict_pass_challenge,
        )
        critiques = [
            format_refinement_critique(critique) for critique in selected_critiques
        ]
        if progress is not None:
            progress.log(
                "candidate=%d round=%d/%d stage=refinement status=start "
                "strategy=%s strict_pass_challenge=%s reviews=%d",
                attempt_idx,
                round_idx + 1,
                cfg.refine_rounds,
                refinement_strategy,
                strict_pass_challenge,
                len(critiques),
            )
        (
            refinement_response,
            refinement_parsed,
            refinement_attempts,
            refinement_handoff_outputs,
            refinement_handoffs,
            refinement_restart_count,
        ) = await run_refinement_with_budget_restart(
            question=question,
            proof=proof,
            self_evaluation=str(
                latest_generation_parsed.get("self_evaluation") or ""
            ),
            selected_critiques=selected_critiques,
            attempt_idx=attempt_idx,
            round_idx=round_idx + 1,
            scheduler=scheduler,
            cfg=cfg,
            problem_id=problem_id,
            progress=progress,
            refinement_strategy=refinement_strategy,
            strict_pass_challenge=strict_pass_challenge,
        )
        pending_strict_pass_challenge = strict_pass_challenge
        proof_refine_attempt_output.extend(refinement_attempts)
        proof_refine_handoff_output.extend(refinement_handoff_outputs)
        proof_refine_handoffs.extend(refinement_handoffs)
        refine_budget_restart_count += refinement_restart_count
        if not require_valid_candidate_response(
            refinement_parsed,
            problem_id=problem_id,
            attempt_idx=attempt_idx,
            stage="proof_refine",
            round_idx=round_idx + 1,
        ):
            proof_refine_output.append(
                make_output(
                    "proof_refine",
                    refinement_response,
                    refinement_parsed,
                    round_idx=round_idx + 1,
                    used_critiques=critiques,
                    refinement_strategy=refinement_strategy,
                    strict_pass_challenge=strict_pass_challenge,
                    invalid=True,
                    prompt_family=PROMPT_FAMILY_OPD,
                )
            )
            if progress is not None:
                progress.log(
                    "candidate=%d round=%d stage=refinement status=invalid_keep_previous",
                    attempt_idx,
                    round_idx + 1,
                )
            break
        proof = refinement_parsed["proof"]
        latest_generation_parsed = refinement_parsed
        proof_refine_output.append(
            make_output(
                "proof_refine",
                refinement_response,
                refinement_parsed,
                round_idx=round_idx + 1,
                used_critiques=critiques,
                refinement_strategy=refinement_strategy,
                strict_pass_challenge=strict_pass_challenge,
                prompt_family=PROMPT_FAMILY_OPD,
            )
        )
        if progress is not None:
            progress.log(
                "candidate=%d round=%d stage=refinement status=complete has_solution=%s "
                "has_self_evaluation=%s proof_chars=%d self_score=%s",
                attempt_idx,
                round_idx + 1,
                refinement_parsed["has_solution_section"],
                refinement_parsed["has_self_evaluation_section"],
                len(proof),
                refinement_parsed.get("self_score"),
            )

    rollback_from_round = None
    if best_round_idx >= 0 and best_round_idx != latest_verified_round_idx:
        rollback_from_round = latest_verified_round_idx
        if progress is not None:
            progress.log(
                "candidate=%d stage=refinement status=rollback selected_round=%d "
                "rejected_round=%d selected_score=%s rejected_score=%s",
                attempt_idx,
                best_round_idx,
                latest_verified_round_idx,
                best_aggregation.get("final_score"),
                final_aggregation.get("final_score"),
            )
        proof = best_proof
        latest_generation_parsed = best_generation_parsed
        final_aggregation = best_aggregation

    if progress is not None:
        progress.log(
            "candidate=%d status=complete final_score=%s final_status=%s refinements=%d "
            "selected_round=%d rollback_from_round=%s",
            attempt_idx,
            final_aggregation.get("final_score"),
            final_aggregation.get("final_status"),
            len(proof_refine_output),
            best_round_idx,
            rollback_from_round,
        )
    return {
        "attempt_idx": attempt_idx,
        "prompt_family": prompt_family,
        "planning_strategy": initial_generation.get(
            "planning_strategy",
            "baseline",
        ),
        "generation_mode": initial_generation.get("generation_mode"),
        "proof_generation_output": generation_output,
        "proof_generation_outputs": initial_generation.get(
            "generation_outputs",
            [generation_output],
        ),
        "proof_handoff_output": initial_generation.get("handoff_outputs", []),
        "proof_handoffs": initial_generation.get("handoffs", []),
        "budget_restart_count": int(
            initial_generation.get("budget_restart_count")
            or consumed_refine_rounds
        ),
        "proof_verify_output": proof_verify_output,
        "proof_meta_verify_output": proof_meta_verify_output,
        "proof_refine_output": proof_refine_output,
        "proof_refine_attempt_output": proof_refine_attempt_output,
        "proof_refine_handoff_output": proof_refine_handoff_output,
        "proof_refine_handoffs": proof_refine_handoffs,
        "refine_budget_restart_count": refine_budget_restart_count,
        "strict_pass_challenges_used": strict_pass_challenges_used,
        "proof_solution": proof,
        "self_evaluation": latest_generation_parsed.get("self_evaluation"),
        "self_score": latest_generation_parsed.get("self_score"),
        "validated_critiques": final_aggregation.get("validated_critiques", []),
        "validated_fatal_critiques": final_aggregation.get(
            "validated_fatal_critiques", []
        ),
        "positive_meta_challenges": final_aggregation.get(
            "positive_meta_challenges", []
        ),
        "fatal_score_cap_applied": final_aggregation.get(
            "fatal_score_cap_applied", False
        ),
        "critique_history": critique_history,
        "verifier_score_summaries": final_aggregation.get(
            "verifier_score_summaries", []
        ),
        "final_score": final_aggregation.get("final_score"),
        "final_status": final_aggregation.get("final_status"),
        "low_scores_seen": final_aggregation.get("low_scores_seen"),
        "strict_pass": final_aggregation.get("strict_pass", False),
        "all_verifiers_passed": final_aggregation.get("all_verifiers_passed", False),
        "meta_valid_count": final_aggregation.get("meta_valid_count", 0),
        "meta_checked_count": final_aggregation.get("meta_checked_count", 0),
        "meta_summary_by_verifier": final_aggregation.get(
            "meta_summary_by_verifier", {}
        ),
        "selected_verification_round": best_round_idx,
        "rollback_from_round": rollback_from_round,
        "verified_versions": verified_versions,
        "success": final_aggregation.get("final_status") != "needs_review",
    }


async def run_candidate_pipeline(
    question: str,
    attempt_idx: int,
    total_candidates: int,
    scheduler: ChatScheduler,
    cfg: PipelineConfig,
    problem_id: Any = "problem",
    progress: Optional[PipelineProgress] = None,
    throttle: Optional[VerificationThrottle] = None,
) -> dict[str, Any]:
    try:
        initial_generation = None
        try:
            initial_generation = await generate_single_attempt(
                question,
                attempt_idx,
                total_candidates,
                scheduler,
                cfg,
                problem_id=problem_id,
                progress=progress,
            )
        finally:
            if throttle is not None:
                await throttle.mark_generation_done(attempt_idx, initial_generation)

        if initial_generation is None:
            return {
                "attempt_idx": attempt_idx,
                "candidate": None,
                "error": "invalid_generation",
            }
        if cfg.proof_generation_only:
            return {
                "attempt_idx": attempt_idx,
                "candidate": make_generation_only_candidate(
                    initial_generation,
                    "proof_generation_only",
                ),
                "error": None,
            }
        wait_decision = await maybe_wait_all_candidates(
            initial_generation,
            cfg,
            throttle,
            progress=progress,
        )
        if wait_decision["action"] == "candidate":
            return {
                "attempt_idx": attempt_idx,
                "candidate": wait_decision["candidate"],
                "error": None,
            }
        if wait_decision["action"] == "skip":
            return {
                "attempt_idx": attempt_idx,
                "candidate": None,
                "error": wait_decision.get("reason", "skipped_after_generation_filter"),
            }
        if progress is not None and wait_decision.get("selected_status"):
            progress.log(
                "candidate=%d stage=generation_filter status=%s action=verify",
                attempt_idx,
                wait_decision.get("selected_status"),
            )
        candidate = await run_single_attempt(
            question,
            attempt_idx,
            total_candidates,
            scheduler,
            cfg,
            problem_id=problem_id,
            progress=progress,
            initial_generation=initial_generation,
            throttle=throttle,
        )
        return {"attempt_idx": attempt_idx, "candidate": candidate, "error": None}
    except InferenceServerUnavailable:
        raise
    except Exception as exc:
        logging.exception("Pipeline failed id=%s attempt=%d", problem_id, attempt_idx)
        return {"attempt_idx": attempt_idx, "candidate": None, "error": repr(exc)}


def select_generation_fallback_index(initial_generations: list[dict[str, Any]]) -> int:
    return max(
        range(len(initial_generations)),
        key=lambda idx: (
            score_sort_value(
                make_generation_only_candidate(
                    initial_generations[idx], "generation_only_fallback"
                )
            ),
            len(str(initial_generations[idx].get("proof") or "")),
        ),
    )


async def run_streaming_candidates(
    question: str,
    scheduler: ChatScheduler,
    cfg: PipelineConfig,
    pipelines_per_problem: int,
    *,
    problem_id: Any = "problem",
    progress: Optional[PipelineProgress] = None,
    timeout_s: Optional[float] = None,
    attempt_indices: Optional[list[int]] = None,
) -> dict[str, Any]:
    resolved_attempt_indices = (
        list(range(pipelines_per_problem))
        if attempt_indices is None
        else [int(attempt_idx) for attempt_idx in attempt_indices]
    )
    if len(resolved_attempt_indices) != len(set(resolved_attempt_indices)):
        raise ValueError("attempt_indices must not contain duplicates")
    if any(
        attempt_idx < 0 or attempt_idx >= pipelines_per_problem
        for attempt_idx in resolved_attempt_indices
    ):
        raise ValueError(
            "attempt_indices must be within the global candidate range "
            f"[0, {pipelines_per_problem})"
        )
    verification_throttle = VerificationThrottle(
        len(resolved_attempt_indices),
        cfg.verify_candidate_limit_while_generating,
        cfg.verify_request_limit_while_generating,
    )
    pipeline_tasks = [
        asyncio.create_task(
            run_candidate_pipeline(
                question,
                attempt_idx,
                pipelines_per_problem,
                scheduler,
                cfg,
                problem_id=problem_id,
                progress=progress,
                throttle=verification_throttle,
            )
        )
        for attempt_idx in resolved_attempt_indices
    ]
    candidates: list[dict[str, Any]] = []
    failed_attempts: list[dict[str, Any]] = []
    strict_pass_candidate: Optional[dict[str, Any]] = None
    cancelled_count = 0
    try:
        iterator = (
            asyncio.as_completed(pipeline_tasks, timeout=max(0.1, timeout_s))
            if timeout_s is not None
            else asyncio.as_completed(pipeline_tasks)
        )
        for future in iterator:
            try:
                result = await future
            except (TimeoutError, asyncio.TimeoutError):
                raise
            except InferenceServerUnavailable:
                raise
            except Exception as exc:
                logging.exception("Pipeline task failed id=%s", problem_id)
                failed_attempts.append({"attempt_idx": None, "error": repr(exc)})
                continue
            if result["candidate"] is None:
                failed_attempts.append(
                    {"attempt_idx": result["attempt_idx"], "error": result["error"]}
                )
                continue
            candidate = result["candidate"]
            candidates.append(candidate)
            if progress is not None:
                progress.log(
                    "candidate=%d status=pipeline_collected final_score=%s final_status=%s "
                    "strict_pass=%s completed=%d/%d",
                    result["attempt_idx"],
                    candidate.get("final_score"),
                    candidate.get("final_status"),
                    candidate.get("strict_pass"),
                    len(candidates),
                    len(resolved_attempt_indices),
                )
            if candidate.get("strict_pass") and cfg.stop_on_strict_pass:
                strict_pass_candidate = candidate
                cancelled_count += await cancel_pending_tasks(pipeline_tasks)
                if progress is not None:
                    progress.log(
                        "stage=pipeline status=early_stop candidate=%d cancelled=%d",
                        candidate.get("attempt_idx"),
                        cancelled_count,
                    )
                break
    except (TimeoutError, asyncio.TimeoutError):
        cancelled_count += await cancel_pending_tasks(pipeline_tasks)
        if progress is not None:
            progress.log(
                "stage=pipeline status=timeout completed=%d cancelled=%d budget_s=%.1f",
                len(candidates),
                cancelled_count,
                float(timeout_s or 0.0),
            )
    except InferenceServerUnavailable:
        cancelled_count += await cancel_pending_tasks(pipeline_tasks)
        raise

    return {
        "candidates": candidates,
        "initial_generations": [],
        "failed_attempts": failed_attempts,
        "skipped_generations": [],
        "strict_pass_candidate": strict_pass_candidate,
        "cancelled_count": cancelled_count,
    }


async def select_best_candidate(
    question: str,
    candidates: list[dict[str, Any]],
    scheduler: ChatScheduler,
    cfg: PipelineConfig,
    progress: Optional[PipelineProgress] = None,
) -> tuple[int, dict[str, Any]]:
    if not candidates:
        return 0, {"success": False, "error": "no candidates", "text": ""}

    if cfg.selector_mode == "score":
        selected_idx = fallback_candidate_index(candidates)
        return selected_idx, selector_fallback_output(
            selected_idx,
            "selector_mode_score",
            len(candidates),
        )

    prompt = build_selection_prompt(
        question,
        candidates,
        cfg.selector_max_candidate_chars,
    )
    response = await scheduler.call(
        "selector",
        prompt,
        temperature=cfg.selection_temperature,
        progress=progress,
        detail=f"candidates={len(candidates)}",
    )
    selected_idx = parse_selected_index(response.get("text", ""), len(candidates))
    if selected_idx is None:
        selected_idx = fallback_candidate_index(candidates)
        response["fallback_reason"] = "selector_parse_failed"
    return selected_idx, make_output(
        "selector", response, {"selected_index": selected_idx}
    )


async def process_problem(
    problem_id: Any,
    question: str,
    scheduler: ChatScheduler,
    pipeline_cfg: PipelineConfig,
    pipelines_per_problem: int,
    progress: Optional[PipelineProgress] = None,
) -> dict[str, Any]:
    pipeline_result = await run_streaming_candidates(
        question,
        scheduler,
        pipeline_cfg,
        pipelines_per_problem,
        problem_id=problem_id,
        progress=progress,
    )
    failed_attempts = pipeline_result["failed_attempts"]
    failed_attempts.extend(pipeline_result["skipped_generations"])
    candidates = pipeline_result["candidates"]
    strict_pass_candidate = pipeline_result["strict_pass_candidate"]
    cancelled_count = pipeline_result["cancelled_count"]

    if progress is not None:
        progress.log(
            "stage=pipeline status=complete valid=%d failed=%d cancelled=%d early_stop=%s",
            len(candidates),
            len(failed_attempts),
            cancelled_count,
            strict_pass_candidate is not None,
        )

    if not candidates:
        selected = None
        proof = DEFAULT_FALLBACK_ANSWER
        selected_idx = -1
        selector_output = {
            "success": False,
            "error": "all pipeline attempts failed",
            "text": "",
        }
        final_score = None
        final_status = "all_attempts_failed"
    elif strict_pass_candidate is not None:
        selected = strict_pass_candidate
        selected_idx = selected.get("attempt_idx", candidates.index(selected))
        proof = selected.get("proof_solution", "")
        final_score = selected.get("final_score")
        final_status = selected.get("final_status")
        selector_output = {
            "success": True,
            "fallback_reason": "strict_pass_early_stop",
            "selected_index": selected_idx,
        }
    else:
        selection_pool, threshold_passed = candidate_selection_pool(
            candidates, pipeline_cfg
        )
        if threshold_passed:
            selected_candidate_idx, selector_output = await select_best_candidate(
                question,
                selection_pool,
                scheduler,
                pipeline_cfg,
                progress=progress,
            )
        else:
            selected_candidate_idx = fallback_candidate_index(selection_pool)
            selector_output = selector_fallback_output(
                selected_candidate_idx,
                "no_candidates_above_selector_min_final_score",
                len(selection_pool),
            )
        selected = selection_pool[selected_candidate_idx]
        selected_idx = selected.get("attempt_idx", selected_candidate_idx)
        proof = selected.get("proof_solution", "")
        final_score = selected.get("final_score")
        final_status = selected.get("final_status")

    selected_verification_round = (
        selected.get(
            "selector_version_round",
            selected.get("selected_verification_round"),
        )
        if selected is not None
        else None
    )
    selected_historical_version = bool(
        selected is not None and selected.get("selector_is_historical")
    )
    if selected is not None:
        selector_output = {
            **selector_output,
            "selected_attempt_idx": selected.get("attempt_idx"),
            "selected_verification_round": selected_verification_round,
            "selected_historical_version": selected_historical_version,
        }

    print_selected_solution_summary(
        problem_id=problem_id,
        selected_idx=selected_idx,
        proof=proof,
        final_score=final_score,
        final_status=final_status,
        candidate=selected,
    )

    return {
        "id": problem_id,
        "problem": question,
        "proof": proof,
        "prediction": proof,
        "answer": format_submission_answer(proof),
        "selected_pipeline": selected_idx,
        "final_score": final_score,
        "final_status": final_status,
        "selected_verification_round": selected_verification_round,
        "selected_historical_version": selected_historical_version,
        "candidates": candidates,
        "failed_attempts": failed_attempts,
        "selector_output": selector_output,
        "early_stop": strict_pass_candidate is not None,
        "strict_pass_candidate": (
            strict_pass_candidate.get("attempt_idx")
            if strict_pass_candidate is not None
            else None
        ),
    }


def write_debug_row(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_grader_input_records(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write selected proofs in the ordering required by the grading harness."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            proof = str(row.get("prediction") or "").strip()
            if (
                row.get("error")
                or row.get("final_status") in {"error", "all_attempts_failed"}
                or not proof
            ):
                continue
            record = {
                "problem_id": str(row.get("id")),
                "final_proof": proof,
                "selected_pipeline": row.get("selected_pipeline"),
                "final_score": row.get("final_score"),
                "final_status": row.get("final_status"),
                "source": "evaluation/harness_vllm/run.py",
            }
            output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, path)


def format_submission_answer(value: Any) -> str:
    answer = str(value or "").strip()
    if not answer:
        answer = DEFAULT_FALLBACK_ANSWER
    return answer[:MAX_SUBMISSION_ANSWER_CHARS]


def write_output_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "answer"]
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {
                "id": row.get("id"),
                "answer": format_submission_answer(
                    row.get("answer", row.get("prediction", DEFAULT_FALLBACK_ANSWER))
                ),
            }
            for row in rows
        )


def is_text_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(str(value).strip())


def normalize_column_name(column: str) -> str:
    return column.strip().lower().replace("-", "_").replace(" ", "_")


def find_named_column(
    columns: list[str], candidates: tuple[str, ...] | set[str]
) -> Optional[str]:
    normalized = {normalize_column_name(column): column for column in columns}
    for candidate in candidates:
        found = normalized.get(normalize_column_name(candidate))
        if found is not None:
            return found
    return None


def detect_question_column(df: pd.DataFrame, requested: str) -> str:
    if requested and requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"Requested question column {requested!r} is not present.")
        return requested
    found = find_named_column(list(df.columns), QUESTION_COLUMN_CANDIDATES)
    if found is None:
        raise ValueError(
            "Could not auto-detect a question column. "
            f"Tried: {', '.join(QUESTION_COLUMN_CANDIDATES)}"
        )
    return found


def detect_id_column(df: pd.DataFrame) -> Optional[str]:
    return find_named_column(list(df.columns), ID_COLUMN_CANDIDATES)


def read_input_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True, dtype=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(
        f"Unsupported input format {suffix or '<none>'!r} for {path}; "
        f"expected one of: {', '.join(sorted(SUPPORTED_INPUT_SUFFIXES))}"
    )


def resolve_input_paths(input_csv: str) -> list[Path]:
    paths: list[Path] = []
    for raw_part in input_csv.split(","):
        part = raw_part.strip()
        if not part:
            continue
        candidate = Path(part).expanduser()
        if candidate.is_dir():
            paths.extend(
                sorted(
                    path
                    for path in candidate.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
                )
            )
            continue
        matches = sorted(Path(match) for match in glob.glob(str(candidate)))
        if matches:
            paths.extend(match for match in matches if match.is_file())
        elif candidate.is_file():
            paths.append(candidate)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    if not deduped:
        raise ValueError(f"No supported input files matched {input_csv!r}")
    return deduped


def load_input_records(
    input_csv: str,
    question_column: str,
) -> list[InputRecord]:
    paths = resolve_input_paths(input_csv)
    records: list[InputRecord] = []
    stem_counts: dict[str, int] = {}

    for path in paths:
        df = read_input_table(path)
        detected_question_column = detect_question_column(df, question_column)
        id_column = detect_id_column(df)
        base_stem = path.stem
        stem_counts[base_stem] = stem_counts.get(base_stem, 0) + 1
        source_stem = (
            base_stem
            if stem_counts[base_stem] == 1
            else f"{base_stem}_{stem_counts[base_stem]}"
        )
        logging.info(
            "Loaded source %s rows=%d question_column=%s id_column=%s",
            path,
            len(df),
            detected_question_column,
            id_column,
        )

        for row_index, row in df.iterrows():
            question = row.get(detected_question_column)
            if not is_text_value(question):
                logging.warning(
                    "Skipping %s row=%s with empty question", path, row_index
                )
                continue
            problem_id = (
                row.get(id_column) if id_column else f"{source_stem}-{row_index}"
            )
            if not is_text_value(problem_id):
                problem_id = f"{source_stem}-{row_index}"
            records.append(
                InputRecord(
                    id=problem_id,
                    question=str(question).strip(),
                    source_file=str(path),
                    source_path=path,
                    source_stem=source_stem,
                    row_index=int(row_index),
                    question_column=detected_question_column,
                )
            )
    if not records:
        raise ValueError("No usable input rows were loaded.")
    return records


def resolve_output_paths(
    records: list[InputRecord], output_csv: Path
) -> dict[str, Path]:
    source_stems = list(dict.fromkeys(record.source_stem for record in records))
    if len(source_stems) == 1 and output_csv.suffix.lower() == ".csv":
        return {source_stems[0]: output_csv}
    return {
        source_stem: output_csv / f"{source_stem}.csv" for source_stem in source_stems
    }


def resolve_debug_paths(records: list[InputRecord], logdir: Path) -> dict[str, Path]:
    return {
        source_stem: logdir / f"debug_{source_stem}.jsonl"
        for source_stem in dict.fromkeys(record.source_stem for record in records)
    }


def output_row_from_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": result.get("id"),
        "answer": format_submission_answer(
            result.get(
                "answer",
                result.get("prediction", result.get("proof", DEFAULT_FALLBACK_ANSWER)),
            )
        ),
        "proof": result.get("proof"),
        "prediction": result.get("prediction"),
        "selected_pipeline": result.get("selected_pipeline"),
        "final_score": result.get("final_score"),
        "final_status": result.get("final_status"),
        "source_file": result.get("source_file"),
        "row_index": result.get("row_index"),
    }


def run_async(
    records: list[InputRecord],
    scheduler: ChatScheduler,
    pipeline_cfg: PipelineConfig,
    pipelines_per_problem: int,
    output_paths: dict[str, Path],
    debug_paths: dict[str, Path],
    logdir: Path,
    benchmark_mode: bool,
    max_concurrent_problems: int,
    verbose: bool,
) -> None:
    output_rows_by_source: dict[str, list[dict[str, Any]]] = {
        source_stem: [] for source_stem in output_paths
    }
    start_time = time.time()
    failures: list[dict[str, Any]] = []

    async def process_record(record: InputRecord) -> dict[str, Any]:
        started = time.time()
        progress = PipelineProgress(verbose, record.id, pipelines_per_problem)
        try:
            result = await process_problem(
                record.id,
                record.question,
                scheduler,
                pipeline_cfg,
                pipelines_per_problem,
                progress=progress,
            )
        finally:
            progress.close()
        result.update(
            {
                "source_file": record.source_file,
                "source_stem": record.source_stem,
                "row_index": record.row_index,
                "question_column": record.question_column,
                "elapsed_s": time.time() - started,
            }
        )
        return result

    def persist_result(result: dict[str, Any]) -> None:
        source_stem = str(result["source_stem"])
        debug_path = debug_paths[source_stem]
        write_debug_row(debug_path, result)
        output_rows_by_source[source_stem].append(output_row_from_result(result))
        output_rows_by_source[source_stem].sort(
            key=lambda row: int(row.get("row_index") or 0)
        )
        write_output_csv(output_paths[source_stem], output_rows_by_source[source_stem])

    async def run_all() -> None:
        if benchmark_mode:
            limit = max_concurrent_problems
            if limit <= 0:
                limit = max(1, min(len(records), scheduler.max_concurrent_requests))
            problem_semaphore = asyncio.Semaphore(limit)

            async def bounded(record: InputRecord) -> dict[str, Any]:
                async with problem_semaphore:
                    return await process_record(record)

            tasks = [asyncio.create_task(bounded(record)) for record in records]
            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                    persist_result(result)
                    logging.info(
                        "Finished id=%s source=%s selected=%s status=%s score=%s elapsed=%.1fs",
                        result["id"],
                        result["source_stem"],
                        result["selected_pipeline"],
                        result["final_status"],
                        result["final_score"],
                        result["elapsed_s"],
                    )
                except Exception as exc:
                    logging.exception("Benchmark task failed")
                    failures.append({"error": repr(exc)})
        else:
            for idx, record in enumerate(records):
                print(
                    f"Processing row {idx + 1}/{len(records)} source={record.source_stem} id={record.id}",
                )
                try:
                    result = await process_record(record)
                    persist_result(result)
                    logging.info(
                        "Finished id=%s source=%s selected=%s status=%s score=%s elapsed=%.1fs",
                        result["id"],
                        result["source_stem"],
                        result["selected_pipeline"],
                        result["final_status"],
                        result["final_score"],
                        result["elapsed_s"],
                    )
                except Exception as exc:
                    logging.exception(
                        "Sequential task failed source=%s id=%s",
                        record.source_stem,
                        record.id,
                    )
                    failures.append(
                        {
                            "source_stem": record.source_stem,
                            "id": record.id,
                            "error": repr(exc),
                        }
                    )

    asyncio.run(run_all())
    elapsed = time.time() - start_time
    completed = sum(len(rows) for rows in output_rows_by_source.values())
    summary = {
        "benchmark_mode": benchmark_mode,
        "records_total": len(records),
        "records_completed": completed,
        "records_failed": len(failures),
        "elapsed_s": elapsed,
        "rows_per_second": completed / elapsed if elapsed > 0 else None,
        "pipelines_per_problem": pipelines_per_problem,
        "candidate_count": completed * pipelines_per_problem,
        "output_paths": {key: str(value) for key, value in output_paths.items()},
        "debug_paths": {key: str(value) for key, value in debug_paths.items()},
        "failures": failures,
    }
    with open(logdir / "benchmark_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2, default=str)


def fallback_candidate_index(candidates: list[dict[str, Any]]) -> int:
    return max(
        range(len(candidates)),
        key=lambda idx: (
            score_sort_value(candidates[idx]),
            len(str(candidates[idx].get("proof_solution") or "")),
        ),
    )


def selector_fallback_output(
    selected_idx: int,
    reason: str,
    candidate_count: int,
) -> dict[str, Any]:
    return {
        "stage": "selector",
        "success": True,
        "error": None,
        "text": "",
        "parsed": {"selected_index": selected_idx},
        "finish_reason": None,
        "usage": {},
        "server_url": None,
        "latency_s": None,
        "selected_index": selected_idx,
        "fallback_reason": reason,
        "candidate_count": candidate_count,
    }


def candidate_selection_pool(
    candidates: list[dict[str, Any]],
    cfg: PipelineConfig,
) -> tuple[list[dict[str, Any]], bool]:
    eligible_candidates = [
        candidate
        for candidate in candidates
        if score_sort_value(candidate) >= cfg.selector_min_final_score
    ]
    threshold_passed = bool(eligible_candidates)
    selection_pool = eligible_candidates or candidates
    candidate_limit = max(0, int(cfg.selector_candidate_limit))
    historical_limit = max(
        0,
        int(getattr(cfg, "selector_historical_candidate_limit", 0)),
    )

    if not threshold_passed or historical_limit == 0:
        if candidate_limit and len(selection_pool) > candidate_limit:
            ranked_indices = sorted(
                range(len(selection_pool)),
                key=lambda idx: (
                    score_sort_value(selection_pool[idx]),
                    len(str(selection_pool[idx].get("proof_solution") or "")),
                ),
                reverse=True,
            )[:candidate_limit]
            selection_pool = [
                selection_pool[idx] for idx in sorted(ranked_indices)
            ]
        return selection_pool, threshold_passed

    current_pool = selection_pool
    historical_candidates: list[dict[str, Any]] = []
    for candidate in selection_pool:
        current_proof = str(candidate.get("proof_solution") or "").strip()
        for version in candidate.get("verified_versions") or []:
            version_proof = str(version.get("proof_solution") or "").strip()
            if not version_proof or version_proof == current_proof:
                continue
            if score_sort_value(version) < cfg.selector_min_final_score:
                continue
            historical = {
                **candidate,
                **version,
                "selector_is_historical": True,
                "selector_parent_attempt_idx": candidate.get("attempt_idx"),
                "selector_version_round": version.get(
                    "selected_verification_round"
                ),
            }
            historical.pop("verified_versions", None)
            historical_candidates.append(historical)

    historical_candidates.sort(
        key=lambda candidate: (
            score_sort_value(candidate),
            len(str(candidate.get("proof_solution") or "")),
            -int(candidate.get("attempt_idx") or 0),
            -int(candidate.get("selector_version_round") or 0),
        ),
        reverse=True,
    )
    if candidate_limit:
        historical_limit = min(historical_limit, max(0, candidate_limit - 1))
    selected_history: list[dict[str, Any]] = []
    selected_history_ids: set[int] = set()
    selected_parent_attempts: set[int] = set()
    for historical in historical_candidates:
        parent_attempt = int(historical.get("selector_parent_attempt_idx") or 0)
        if parent_attempt in selected_parent_attempts:
            continue
        selected_history.append(historical)
        selected_history_ids.add(id(historical))
        selected_parent_attempts.add(parent_attempt)
        if len(selected_history) >= historical_limit:
            break
    if len(selected_history) < historical_limit:
        for historical in historical_candidates:
            if id(historical) in selected_history_ids:
                continue
            selected_history.append(historical)
            if len(selected_history) >= historical_limit:
                break

    current_limit = candidate_limit - len(selected_history) if candidate_limit else 0
    if current_limit and len(current_pool) > current_limit:
        ranked_indices = sorted(
            range(len(current_pool)),
            key=lambda idx: (
                score_sort_value(current_pool[idx]),
                len(str(current_pool[idx].get("proof_solution") or "")),
            ),
            reverse=True,
        )[:current_limit]
        current_pool = [
            current_pool[idx] for idx in sorted(ranked_indices)
        ]
    return current_pool + selected_history, threshold_passed


def load_simple_input(input_csv: Path) -> tuple[pd.DataFrame, str, Optional[str]]:
    df = read_input_table(input_csv)
    problem_column = detect_question_column(df, "auto")
    return df, problem_column, detect_id_column(df)


def write_simple_output(output_csv: Path, rows: list[dict[str, Any]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "id": row.get("id"),
                "answer": format_submission_answer(
                    row.get("answer", row.get("prediction", DEFAULT_FALLBACK_ANSWER))
                ),
            }
            for row in rows
        ]
    ).to_csv(output_csv, index=False)


def load_counting_tokenizer(model_path: Path) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    logging.info("Loaded tokenizer for prompt token counting path=%s", model_path)
    return tokenizer


class ProofRuntime:
    def __init__(
        self,
        *,
        model_path: Path,
        logdir: Path,
        gpu_group: str,
        tensor_parallel_size: int,
        data_parallel_size: int,
        num_ctx: int,
        dtype: str,
        gpu_memory_utilization: float,
        max_num_seqs: int,
        max_concurrent_requests: int,
        pipelines_per_problem: int,
        deepseek_math_v2_candidate_count: int,
        proof_only_candidate_count: int,
        skip_self_score_zero: bool,
        stop_on_strict_pass: bool,
        verification_early_stop: bool,
        wait_for_all_generations_before_verify: bool,
        proof_generation_only: bool,
        proof_generation_strategy_portfolio: str,
        verify_candidate_limit_while_generating: int,
        verify_request_limit_while_generating: int,
        verify_n: int,
        verifier_generalist_n: int,
        meta_n: int,
        meta_policy: str,
        strict_pass_meta: bool,
        refine_rounds: int,
        refine_review_n: int,
        min_valid_low: int,
        refinement_strategy: str,
        strict_pass_challenge_rounds: int,
        problem_timeout_seconds: int,
        selection_reserve_seconds: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_new_tokens: int,
        min_p: Optional[float],
        proof_max_new_tokens: int,
        proof_generation_temperatures: list[float],
        thinking_budget_enabled: bool,
        proof_generation_thinking_budgets: list[int],
        thinking_budget_force_text: str,
        thinking_budget_handoff_enabled: bool,
        thinking_budget_handoff_preserve_refine_rounds: bool,
        thinking_budget_handoff_max_tokens: int,
        thinking_budget_handoff_temperature: float,
        thinking_budget_handoff_prompt_variant: str,
        thinking_budget_handoff_mode: str,
        thinking_budget_restart_strategy: str,
        thinking_budget_restart_until_complete: bool,
        thinking_budget_final_round_tokens: int,
        thinking_budget_refine_handoff_enabled: bool,
        thinking_budget_refine_tokens: int,
        thinking_budget_refine_final_round_tokens: int,
        thinking_budget_refine_max_restarts: int,
        thinking_budget_refine_final_temperature: Optional[float],
        thinking_budget_refine_visible_output_target_tokens: int,
        thinking_budget_refine_visible_output_limit_tokens: int,
        deepseek_thinking_budget_force_text: str,
        verifier_thinking_budget_tokens: int,
        verifier_thinking_budget_force_text: str,
        deepseek_verifier_thinking_budget_force_text: str,
        meta_thinking_budget_tokens: int,
        meta_thinking_budget_force_text: str,
        verifier_max_new_tokens: int,
        meta_max_new_tokens: int,
        selector_max_new_tokens: int,
        selector_max_candidate_chars: int,
        selection_temperature: float,
        selector_mode: str,
        selector_min_final_score: float,
        selector_candidate_limit: int,
        selector_historical_candidate_limit: int,
        vllm_extra_args: str,
        stream_interval: int,
        host: str,
        port: int,
        api_key: str,
        served_model_name: str,
        server_timeout: int,
        no_serve: bool,
        base_url: str,
        mock_llm: bool,
        stream_vllm: bool,
        stream_vllm_server_log: bool,
        verbose: bool,
        distributed_runtime: DistributedRuntime,
    ) -> None:
        self.logdir = logdir
        self.gpu_group = gpu_group
        self.max_concurrent_requests = max(1, max_concurrent_requests)
        self.request_worker_count = resolve_request_worker_count(
            max_num_seqs=max_num_seqs,
            data_parallel_size=data_parallel_size,
            max_concurrent_requests=self.max_concurrent_requests,
        )
        self.pipelines_per_problem = max(1, pipelines_per_problem)
        if not 0 <= int(deepseek_math_v2_candidate_count) <= self.pipelines_per_problem:
            raise ValueError(
                "deepseek_math_v2_candidate_count must be between 0 and "
                f"pipelines_per_problem ({self.pipelines_per_problem}), got "
                f"{deepseek_math_v2_candidate_count}"
            )
        self.problem_timeout_seconds = max(60, problem_timeout_seconds)
        self.selection_reserve_seconds = max(30, selection_reserve_seconds)
        self.api_key = api_key
        self.served_model_name = served_model_name
        self.no_serve = no_serve
        self.base_url = base_url
        self.mock_llm = mock_llm
        self.stream_vllm = stream_vllm
        self.verbose = verbose
        self.distributed = distributed_runtime
        if (
            self.distributed.enabled
            and wait_for_all_generations_before_verify
        ):
            raise ValueError(
                "wait_for_all_generations_before_verify is not supported with "
                "multi-node candidate sharding"
            )
        self.tokenizer = None if mock_llm else load_counting_tokenizer(model_path)
        self._init_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self._manager: Optional[ServerManager] = None
        self._scheduler: Optional[ChatScheduler] = None
        self._loop = asyncio.new_event_loop()

        extra_args = shlex.split(vllm_extra_args)
        self.vllm_config = VLLMConfig(
            model_path=str(model_path),
            served_model_name=served_model_name,
            host=host,
            port=port,
            api_key=api_key,
            num_ctx=num_ctx,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            vllm_extra_args=shlex.join(extra_args),
            logdir=logdir,
            stream_interval=stream_interval,
            stream_server_logs=stream_vllm_server_log,
        )
        self.sampling = SamplingConfig(
            max_new_tokens=proof_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_new_tokens=min_new_tokens,
            min_p=min_p,
        )
        self.stage_token_limits = {
            "proof_generation": proof_max_new_tokens,
            "proof_handoff": thinking_budget_handoff_max_tokens,
            "proof_refine": proof_max_new_tokens,
            "proof_verify": verifier_max_new_tokens,
            "proof_meta_verify": meta_max_new_tokens,
            "selector": selector_max_new_tokens,
        }
        normalized_meta_policy = meta_policy.strip().lower().replace("_", "-")
        if normalized_meta_policy not in {"all-reviews", "low-only"}:
            raise ValueError("--meta_policy must be either 'all-reviews' or 'low-only'")
        normalized_selector_mode = selector_mode.strip().lower()
        if normalized_selector_mode not in {"llm", "score"}:
            raise ValueError("selector_mode must be either 'llm' or 'score'")
        normalized_refinement_strategy = refinement_strategy.strip().lower()
        if normalized_refinement_strategy not in REFINEMENT_STRATEGIES:
            raise ValueError(
                "refinement_strategy must be one of "
                + ", ".join(REFINEMENT_STRATEGIES)
            )
        normalized_proof_generation_strategy_portfolio = (
            proof_generation_strategy_portfolio.strip().lower()
        )
        if (
            normalized_proof_generation_strategy_portfolio
            not in PROOF_GENERATION_STRATEGY_PORTFOLIOS
        ):
            raise ValueError(
                "proof_generation_strategy_portfolio must be one of "
                + ", ".join(PROOF_GENERATION_STRATEGY_PORTFOLIOS)
            )
        normalized_handoff_variant = (
            thinking_budget_handoff_prompt_variant.strip().lower()
        )
        if normalized_handoff_variant not in HANDOFF_VARIANTS:
            raise ValueError(
                "thinking_budget_handoff_prompt_variant must be one of "
                + ", ".join(HANDOFF_VARIANTS)
            )
        normalized_handoff_mode = thinking_budget_handoff_mode.strip().lower()
        if normalized_handoff_mode not in HANDOFF_MODES:
            raise ValueError(
                "thinking_budget_handoff_mode must be one of "
                + ", ".join(HANDOFF_MODES)
            )
        normalized_restart_strategy = (
            thinking_budget_restart_strategy.strip().lower()
        )
        if normalized_restart_strategy not in RESTART_STRATEGIES:
            raise ValueError(
                "thinking_budget_restart_strategy must be one of "
                + ", ".join(RESTART_STRATEGIES)
            )
        normalized_verify_n = max(0, int(verify_n))
        normalized_verifier_generalist_n = int(verifier_generalist_n)
        if not 0 <= normalized_verifier_generalist_n <= normalized_verify_n:
            raise ValueError(
                "verifier_generalist_n must be between 0 and verify_n "
                f"({normalized_verify_n}), got {normalized_verifier_generalist_n}"
            )
        self.pipeline_config = PipelineConfig(
            proof_max_new_tokens=proof_max_new_tokens,
            default_temperature=float(temperature),
            proof_generation_temperatures=[
                float(value) for value in proof_generation_temperatures
            ],
            deepseek_math_v2_candidate_count=int(
                deepseek_math_v2_candidate_count
            ),
            proof_only_candidate_count=max(0, int(proof_only_candidate_count)),
            skip_self_score_zero=bool(skip_self_score_zero),
            stop_on_strict_pass=bool(stop_on_strict_pass),
            verification_early_stop=bool(verification_early_stop),
            wait_for_all_generations_before_verify=bool(
                wait_for_all_generations_before_verify
            ),
            proof_generation_only=bool(proof_generation_only),
            thinking_budget_enabled=thinking_budget_enabled,
            proof_generation_thinking_budgets=[
                int(value) for value in proof_generation_thinking_budgets
            ],
            thinking_budget_force_text=thinking_budget_force_text,
            thinking_budget_handoff_enabled=bool(
                thinking_budget_handoff_enabled
            ),
            thinking_budget_handoff_preserve_refine_rounds=bool(
                thinking_budget_handoff_preserve_refine_rounds
            ),
            thinking_budget_handoff_max_tokens=max(
                1,
                int(thinking_budget_handoff_max_tokens),
            ),
            thinking_budget_handoff_temperature=float(
                thinking_budget_handoff_temperature
            ),
            thinking_budget_handoff_prompt_variant=(
                normalized_handoff_variant
            ),
            thinking_budget_handoff_mode=normalized_handoff_mode,
            thinking_budget_restart_strategy=normalized_restart_strategy,
            thinking_budget_restart_until_complete=bool(
                thinking_budget_restart_until_complete
            ),
            thinking_budget_final_round_tokens=max(
                0,
                int(thinking_budget_final_round_tokens),
            ),
            thinking_budget_refine_handoff_enabled=bool(
                thinking_budget_refine_handoff_enabled
            ),
            thinking_budget_refine_tokens=max(
                0,
                int(thinking_budget_refine_tokens),
            ),
            thinking_budget_refine_final_round_tokens=max(
                0,
                int(thinking_budget_refine_final_round_tokens),
            ),
            thinking_budget_refine_max_restarts=max(
                0,
                int(thinking_budget_refine_max_restarts),
            ),
            thinking_budget_refine_final_temperature=(
                None
                if thinking_budget_refine_final_temperature is None
                else float(thinking_budget_refine_final_temperature)
            ),
            thinking_budget_refine_visible_output_target_tokens=max(
                0,
                int(thinking_budget_refine_visible_output_target_tokens),
            ),
            thinking_budget_refine_visible_output_limit_tokens=max(
                0,
                int(thinking_budget_refine_visible_output_limit_tokens),
            ),
            deepseek_thinking_budget_force_text=(
                deepseek_thinking_budget_force_text
            ),
            verifier_thinking_budget_tokens=max(
                0, int(verifier_thinking_budget_tokens)
            ),
            verifier_thinking_budget_force_text=verifier_thinking_budget_force_text,
            deepseek_verifier_thinking_budget_force_text=(
                deepseek_verifier_thinking_budget_force_text
            ),
            meta_thinking_budget_tokens=max(0, int(meta_thinking_budget_tokens)),
            meta_thinking_budget_force_text=meta_thinking_budget_force_text,
            verify_candidate_limit_while_generating=max(
                0,
                int(verify_candidate_limit_while_generating),
            ),
            verify_request_limit_while_generating=max(
                0,
                int(verify_request_limit_while_generating),
            ),
            verify_n=normalized_verify_n,
            verifier_generalist_n=normalized_verifier_generalist_n,
            meta_n=max(0, meta_n),
            meta_policy=normalized_meta_policy,
            strict_pass_meta=strict_pass_meta,
            refine_rounds=max(0, refine_rounds),
            refine_review_n=max(1, refine_review_n),
            min_valid_low=max(1, min_valid_low),
            refinement_strategy=normalized_refinement_strategy,
            strict_pass_challenge_rounds=max(
                0,
                int(strict_pass_challenge_rounds),
            ),
            selector_max_candidate_chars=max(1000, selector_max_candidate_chars),
            selection_temperature=selection_temperature,
            selector_mode=normalized_selector_mode,
            selector_min_final_score=float(selector_min_final_score),
            selector_candidate_limit=max(0, int(selector_candidate_limit)),
            selector_historical_candidate_limit=max(
                0,
                int(selector_historical_candidate_limit),
            ),
            proof_generation_strategy_portfolio=(
                normalized_proof_generation_strategy_portfolio
            ),
        )
        self.server_timeout = server_timeout
        atexit.register(self.close)

    def ensure_ready(self) -> None:
        if self._scheduler is not None:
            return
        with self._init_lock:
            if self._scheduler is not None:
                return
            self.logdir.mkdir(parents=True, exist_ok=True)
            manager = ServerManager(
                self.vllm_config,
                ["external"] if self.no_serve or self.mock_llm else [self.gpu_group],
                no_serve=self.no_serve or self.mock_llm,
                base_url=self.base_url,
                server_timeout=self.server_timeout,
            )
            manager.start()
            self._manager = manager
            self._scheduler = ChatScheduler(
                base_urls=manager.urls,
                api_key=self.api_key,
                model=self.served_model_name,
                sampling=self.sampling,
                max_concurrent_requests=self.max_concurrent_requests,
                mock_llm=self.mock_llm,
                stage_max_new_tokens=self.stage_token_limits,
                request_timeout_seconds=float(self.problem_timeout_seconds),
                request_worker_count=self.request_worker_count,
                stream_responses=self.stream_vllm,
                context_length=self.vllm_config.num_ctx,
                tokenizer=self.tokenizer,
                llm_call_logdir=self.logdir / "llm_calls",
                stream_interval_tokens=self.vllm_config.stream_interval,
            )

    async def solve_problem(
        self,
        problem_id: Any,
        question: str,
        problem_ordinal: int = 0,
    ) -> dict[str, Any]:
        if self._scheduler is None:
            raise RuntimeError("Inference runtime was not initialized")
        started = time.monotonic()
        assigned_attempts = (
            self.distributed.assigned_attempt_indices(self.pipelines_per_problem)
            if self.distributed.enabled
            else list(range(self.pipelines_per_problem))
        )
        progress = PipelineProgress(
            self.verbose,
            problem_id,
            len(assigned_attempts),
        )
        progress.log(
            "status=start candidates=%d verify_n=%d generalists=%d specialists=%d "
            "meta_n=%d refine_rounds=%d "
            "stop_on_strict_pass=%s question_chars=%d rank=%d/%d assigned=%s",
            self.pipelines_per_problem,
            self.pipeline_config.verify_n,
            self.pipeline_config.verifier_generalist_n,
            (
                self.pipeline_config.verify_n
                - self.pipeline_config.verifier_generalist_n
            ),
            self.pipeline_config.meta_n,
            self.pipeline_config.refine_rounds,
            self.pipeline_config.stop_on_strict_pass,
            len(question),
            self.distributed.rank,
            self.distributed.world_size,
            assigned_attempts,
        )
        attempt_budget = max(
            30, self.problem_timeout_seconds - self.selection_reserve_seconds
        )
        try:
            pipeline_result = await run_streaming_candidates(
                question,
                self._scheduler,
                self.pipeline_config,
                self.pipelines_per_problem,
                problem_id=problem_id,
                progress=progress,
                timeout_s=float(attempt_budget),
                attempt_indices=assigned_attempts,
            )
            if self.distributed.enabled:
                progress.log(
                    "stage=distributed_exchange status=local_complete rank=%d "
                    "valid=%d failed=%d assigned=%s",
                    self.distributed.rank,
                    len(pipeline_result["candidates"]),
                    len(pipeline_result["failed_attempts"]),
                    assigned_attempts,
                )
                pipeline_result = await asyncio.to_thread(
                    self.distributed.exchange_pipeline_result,
                    problem_ordinal=problem_ordinal,
                    problem_id=problem_id,
                    question=question,
                    pipelines_per_problem=self.pipelines_per_problem,
                    pipeline_result=pipeline_result,
                )
                if pipeline_result is None:
                    elapsed = time.monotonic() - started
                    progress.log(
                        "stage=distributed_exchange status=worker_complete rank=%d "
                        "elapsed=%.1fs",
                        self.distributed.rank,
                        elapsed,
                    )
                    return {
                        "id": problem_id,
                        "distributed_worker": True,
                        "rank": self.distributed.rank,
                        "assigned_attempts": assigned_attempts,
                        "elapsed_s": elapsed,
                    }
                progress.log(
                    "stage=distributed_exchange status=merged ranks=%d candidates=%d",
                    self.distributed.world_size,
                    len(pipeline_result["candidates"]),
                )
            candidates = pipeline_result["candidates"]
            strict_pass_candidate = pipeline_result["strict_pass_candidate"]
            cancelled_count = pipeline_result["cancelled_count"]
            errors = [
                f"attempt={item.get('attempt_idx')} error={item.get('error', item.get('reason'))}"
                for item in (
                    list(pipeline_result["failed_attempts"])
                    + list(pipeline_result["skipped_generations"])
                )
            ]
            failed = len(pipeline_result["failed_attempts"]) + len(
                pipeline_result["skipped_generations"]
            )
            progress.log(
                "stage=pipeline status=complete valid=%d failed=%d cancelled=%d early_stop=%s",
                len(candidates),
                max(0, failed),
                cancelled_count,
                strict_pass_candidate is not None,
            )
            if not candidates:
                remaining_attempt_budget = max(
                    0.0,
                    attempt_budget - (time.monotonic() - started),
                )
                result = {
                    "id": problem_id,
                    "answer": DEFAULT_FALLBACK_ANSWER,
                    "prediction": DEFAULT_FALLBACK_ANSWER,
                    "selected_pipeline": -1,
                    "final_score": None,
                    "final_status": "all_attempts_failed",
                    "candidate_count": 0,
                    "elapsed_s": time.monotonic() - started,
                    "selector_output": {
                        "success": False,
                        "error": (
                            "No proof candidate completed generation/verification/refinement "
                            f"within {remaining_attempt_budget:.1f}s"
                        ),
                    },
                    "errors": errors,
                    "early_stop": False,
                    "strict_pass_candidate": None,
                }
                progress.log(
                    "status=complete selected=-1 final_status=all_attempts_failed "
                    "fallback_answer=%s elapsed=%.1fs",
                    DEFAULT_FALLBACK_ANSWER,
                    result["elapsed_s"],
                )
                print_selected_solution_summary(
                    problem_id=problem_id,
                    selected_idx=-1,
                    proof=DEFAULT_FALLBACK_ANSWER,
                    final_score=None,
                    final_status="all_attempts_failed",
                    candidate=None,
                )
                return result

            selection_pool, threshold_passed = candidate_selection_pool(
                candidates,
                self.pipeline_config,
            )
            selected_index = fallback_candidate_index(selection_pool)
            selector_output: dict[str, Any] = selector_fallback_output(
                selected_index,
                "time_budget",
                len(selection_pool),
            )
            if strict_pass_candidate is not None:
                if strict_pass_candidate in selection_pool:
                    selected_index = selection_pool.index(strict_pass_candidate)
                else:
                    selection_pool = [strict_pass_candidate]
                    selected_index = 0
                selector_output = {
                    "success": True,
                    "fallback_reason": "strict_pass_early_stop",
                    "selected_index": selected_index,
                }
            remaining = self.problem_timeout_seconds - (time.monotonic() - started)
            if strict_pass_candidate is None and not threshold_passed:
                selector_output = selector_fallback_output(
                    selected_index,
                    "no_candidates_above_selector_min_final_score",
                    len(selection_pool),
                )
            elif strict_pass_candidate is None and remaining > 20:
                try:
                    selected_index, selector_output = await asyncio.wait_for(
                        select_best_candidate(
                            question,
                            selection_pool,
                            self._scheduler,
                            self.pipeline_config,
                            progress=progress,
                        ),
                        timeout=remaining,
                    )
                except (TimeoutError, asyncio.TimeoutError):
                    logging.warning(
                        "Selector timed out for id=%s; using score fallback",
                        problem_id,
                    )

            selected = selection_pool[selected_index]
            prediction = str(selected.get("proof_solution") or "").strip()
            selected_verification_round = selected.get(
                "selector_version_round",
                selected.get("selected_verification_round"),
            )
            selected_historical_version = bool(
                selected.get("selector_is_historical")
            )
            selector_output = {
                **selector_output,
                "selected_attempt_idx": selected.get("attempt_idx"),
                "selected_verification_round": selected_verification_round,
                "selected_historical_version": selected_historical_version,
            }
            print_selected_solution_summary(
                problem_id=problem_id,
                selected_idx=selected.get("attempt_idx", selected_index),
                proof=prediction,
                final_score=selected.get("final_score"),
                final_status=selected.get("final_status"),
                candidate=selected,
            )
            result = {
                "id": problem_id,
                "answer": format_submission_answer(prediction),
                "prediction": prediction,
                "selected_pipeline": selected.get("attempt_idx", selected_index),
                "final_score": selected.get("final_score"),
                "final_status": selected.get("final_status"),
                "selected_verification_round": selected_verification_round,
                "selected_historical_version": selected_historical_version,
                "candidate_count": len(candidates),
                "elapsed_s": time.monotonic() - started,
                "selector_output": selector_output,
                "errors": errors,
                "early_stop": strict_pass_candidate is not None,
                "strict_pass_candidate": (
                    strict_pass_candidate.get("attempt_idx")
                    if strict_pass_candidate is not None
                    else None
                ),
            }
            progress.log(
                "status=complete selected=%s final_score=%s final_status=%s elapsed=%.1fs",
                result["selected_pipeline"],
                result["final_score"],
                result["final_status"],
                result["elapsed_s"],
            )
            return result
        finally:
            progress.close()

    def predict(self, row_id: Any, problem: Any) -> Any:
        import polars as pl

        if self.distributed.enabled:
            raise RuntimeError(
                "Distributed inference requires every rank to execute the same "
                "standalone vLLM harness input sequence; predict() is unsupported"
            )

        with self._predict_lock:
            problem_id = row_id.item()
            question = str(problem.item())
            self.ensure_ready()
            result = self._loop.run_until_complete(
                self.solve_problem(problem_id, question)
            )
            return pl.DataFrame(
                {
                    "id": [problem_id],
                    "answer": [format_submission_answer(result.get("answer"))],
                }
            )

    def close(self) -> None:
        if self._scheduler is not None:
            self._scheduler.close()
            self._scheduler = None
        if self._manager is not None:
            self._manager.stop()
            self._manager = None
        if not self._loop.is_closed() and not self._loop.is_running():
            self._loop.close()


def run_simple_csv(
    runtime: ProofRuntime,
    input_csv: Path,
    output_csv: Path,
    max_rows: int,
    max_concurrent_problems: int,
) -> None:
    df, problem_column, id_column = load_simple_input(input_csv)
    if max_rows > 0:
        df = df.head(max_rows)
    grader_records_path = runtime.logdir / "grader_input" / "records.jsonl"
    write_grader_input_records(grader_records_path, [])
    runtime.ensure_ready()
    runtime.distributed.synchronize_stage("servers_ready")
    output_rows: dict[int, dict[str, Any]] = {}
    semaphore = asyncio.Semaphore(max(1, max_concurrent_problems))

    def persist_outputs() -> None:
        rows = [output_rows[index] for index in sorted(output_rows)]
        write_simple_output(output_csv, rows)
        write_grader_input_records(
            grader_records_path,
            rows,
        )

    async def process_row(
        row_position: int, row_idx: int, row: pd.Series
    ) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            problem_id = row[id_column] if id_column is not None else row_idx
            question = str(row[problem_column])
            logging.info(
                "Processing row %d/%d id=%s",
                row_position + 1,
                len(df),
                problem_id,
            )
            started = time.monotonic()
            try:
                result = await runtime.solve_problem(
                    problem_id,
                    question,
                    problem_ordinal=row_position,
                )
                if result.get("distributed_worker"):
                    write_debug_row(runtime.logdir / "results.jsonl", result)
                    return row_position, {
                        "id": problem_id,
                        "answer": "",
                        "prediction": "",
                        "final_status": "distributed_worker_complete",
                        "final_score": None,
                        "selected_pipeline": None,
                        "elapsed_s": result.get("elapsed_s"),
                        "error": "",
                    }
                output_row = {
                    "id": problem_id,
                    "answer": format_submission_answer(
                        result.get(
                            "answer", result.get("prediction", DEFAULT_FALLBACK_ANSWER)
                        )
                    ),
                    "prediction": result.get("prediction", ""),
                    "final_status": result.get("final_status"),
                    "final_score": result.get("final_score"),
                    "selected_pipeline": result.get("selected_pipeline"),
                    "elapsed_s": result.get("elapsed_s"),
                    "error": "",
                }
                write_debug_row(runtime.logdir / "results.jsonl", result)
            except InferenceServerUnavailable:
                raise
            except Exception as exc:
                logging.exception("Failed row=%s id=%s", row_idx, problem_id)
                if runtime.distributed.enabled:
                    raise
                output_row = {
                    "id": problem_id,
                    "answer": DEFAULT_FALLBACK_ANSWER,
                    "prediction": DEFAULT_FALLBACK_ANSWER,
                    "final_status": "error",
                    "final_score": None,
                    "selected_pipeline": None,
                    "elapsed_s": time.monotonic() - started,
                    "error": repr(exc),
                }
            return row_position, output_row

    async def run_rows() -> None:
        tasks = [
            asyncio.create_task(process_row(row_position, int(row_idx), row))
            for row_position, (row_idx, row) in enumerate(df.iterrows())
        ]
        for task in asyncio.as_completed(tasks):
            row_position, output_row = await task
            if runtime.distributed.enabled and not runtime.distributed.is_primary:
                print(
                    f"Rank {runtime.distributed.rank} completed distributed "
                    f"row {row_position + 1}/{len(df)} id={output_row['id']}",
                )
                continue
            output_rows[row_position] = output_row
            persist_outputs()
            print(
                f"Wrote row {len(output_rows)}/{len(df)} id={output_row['id']} "
                f"status={output_row['final_status']} elapsed={output_row['elapsed_s'] or 0.0}",
            )

    runtime._loop.run_until_complete(run_rows())


def resolve_gpu_parallel_layout(cfg: Any) -> tuple[list[str], int, int]:
    """Resolve one local vLLM server's TP x DP GPU layout."""
    data_parallel_size = int(cfg.data_parallel_size)
    tensor_parallel_size = int(cfg.tensor_parallel_size)
    num_gpus = int(cfg.num_gpus)
    if data_parallel_size < 1:
        raise ValueError("AIMO_DATA_PARALLEL_SIZE must be at least 1")
    if tensor_parallel_size < 0:
        raise ValueError("AIMO_TENSOR_PARALLEL_SIZE cannot be negative")
    if num_gpus < 1:
        raise ValueError("AIMO_NUM_GPUS must be at least 1")

    selected_gpus = [item.strip() for item in cfg.gpus.split(",") if item.strip()]
    if len(selected_gpus) != len(set(selected_gpus)):
        raise ValueError("AIMO_GPUS must not contain duplicate GPU IDs")

    if selected_gpus:
        if tensor_parallel_size == 0:
            if len(selected_gpus) % data_parallel_size != 0:
                raise ValueError(
                    "The number of AIMO_GPUS entries must be divisible by "
                    "AIMO_DATA_PARALLEL_SIZE when TP is inferred"
                )
            tensor_parallel_size = len(selected_gpus) // data_parallel_size
    elif tensor_parallel_size > 0:
        expected_gpus = tensor_parallel_size * data_parallel_size
        if num_gpus not in (1, expected_gpus):
            raise ValueError(
                "AIMO_NUM_GPUS conflicts with TP x DP: "
                f"{num_gpus} != {tensor_parallel_size} x {data_parallel_size}"
            )
        selected_gpus = [str(index) for index in range(expected_gpus)]
    elif num_gpus == 1 and data_parallel_size > 1:
        tensor_parallel_size = 1
        selected_gpus = [str(index) for index in range(data_parallel_size)]
    else:
        if num_gpus % data_parallel_size != 0:
            raise ValueError(
                "AIMO_NUM_GPUS must be divisible by AIMO_DATA_PARALLEL_SIZE "
                "when TP is inferred"
            )
        tensor_parallel_size = num_gpus // data_parallel_size
        selected_gpus = [str(index) for index in range(num_gpus)]

    expected_gpus = tensor_parallel_size * data_parallel_size
    if tensor_parallel_size < 1 or len(selected_gpus) != expected_gpus:
        raise ValueError(
            "Selected GPU count must equal TP x DP: "
            f"{len(selected_gpus)} != {tensor_parallel_size} x "
            f"{data_parallel_size}"
        )
    return selected_gpus, tensor_parallel_size, data_parallel_size


def resolve_max_concurrent_requests(cfg: Any, selected_gpu_count: int) -> int:
    """Resolve the local request scheduler capacity from the GPU count."""
    configured = int(cfg.max_concurrent_requests)
    requests_per_gpu = int(cfg.requests_per_gpu)
    if selected_gpu_count < 1:
        raise ValueError("selected_gpu_count must be at least 1")
    if configured < 0:
        raise ValueError("AIMO_MAX_CONCURRENT_REQUESTS cannot be negative")
    if requests_per_gpu < 1:
        raise ValueError("AIMO_REQUESTS_PER_GPU must be at least 1")
    if configured > 0:
        return configured
    return requests_per_gpu * selected_gpu_count


def resolve_request_worker_count(
    *,
    max_num_seqs: int,
    data_parallel_size: int,
    max_concurrent_requests: int,
) -> int:
    """Size blocking HTTP workers for all local vLLM DP replicas."""
    if int(max_num_seqs) < 1:
        raise ValueError("max_num_seqs must be at least 1")
    if int(data_parallel_size) < 1:
        raise ValueError("data_parallel_size must be at least 1")
    if int(max_concurrent_requests) < 1:
        raise ValueError("max_concurrent_requests must be at least 1")
    return min(
        int(max_concurrent_requests),
        int(max_num_seqs) * int(data_parallel_size),
    )


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the proof-pipeline vLLM harness. Explicit CLI values override "
            "the environment-backed CFG defaults."
        )
    )

    paths = parser.add_argument_group("paths")
    paths.add_argument("--model-path", type=Path)
    paths.add_argument("--input-path", type=Path)
    paths.add_argument("--output-path", type=Path)
    paths.add_argument("--logdir", type=Path)

    distributed = parser.add_argument_group("distributed controller")
    distributed.add_argument("--node-rank", type=int)
    distributed.add_argument("--world-size", type=int)
    distributed.add_argument("--master-addr")
    distributed.add_argument("--master-port", type=int)
    distributed.add_argument("--distributed-run-id")
    distributed.add_argument("--distributed-root", type=Path)
    distributed.add_argument("--distributed-timeout-seconds", type=int)
    distributed.add_argument("--distributed-poll-seconds", type=float)
    distributed.add_argument(
        "--distributed-overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    server = parser.add_argument_group("local vLLM server")
    server.add_argument("--num-gpus", type=int)
    server.add_argument("--gpus")
    server.add_argument("--tensor-parallel-size", type=int)
    server.add_argument("--data-parallel-size", type=int)
    server.add_argument("--dtype")
    server.add_argument("--gpu-memory-utilization", type=float)
    server.add_argument("--max-num-seqs", type=int)
    server.add_argument("--requests-per-gpu", type=int)
    server.add_argument("--max-concurrent-requests", type=int)
    server.add_argument("--max-num-batched-tokens", type=int)
    server.add_argument("--vllm-extra-args")
    dflash = server.add_mutually_exclusive_group()
    dflash.add_argument("--dflash-model-path", type=Path)
    dflash.add_argument("--no-dflash", action="store_true", default=None)
    server.add_argument("--dflash-num-speculative-tokens", type=int)
    server.add_argument("--dflash-context-cutoff", type=int)
    server.add_argument("--host")
    server.add_argument("--port", type=int)
    server.add_argument("--served-model-name")
    server.add_argument("--server-timeout", type=int)
    server.add_argument("--base-url")
    server.add_argument("--no-serve", action="store_true", default=None)

    pipeline = parser.add_argument_group("proof pipeline")
    pipeline.add_argument("--num-ctx", type=int)
    pipeline.add_argument("--max-new-tokens", type=int)
    pipeline.add_argument("--verifier-max-new-tokens", type=int)
    pipeline.add_argument("--meta-max-new-tokens", type=int)
    pipeline.add_argument("--selector-max-new-tokens", type=int)
    pipeline.add_argument("--pipelines-per-problem", type=int)
    pipeline.add_argument("--max-concurrent-problems", type=int)
    pipeline.add_argument("--deepseek-math-v2-candidate-count", type=int)
    pipeline.add_argument(
        "--proof-generation-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Stop each candidate after its initial proof-generation call; skip "
            "verifier, meta-verifier, refinement, and LLM selector calls."
        ),
    )
    pipeline.add_argument(
        "--proof-generation-strategy-portfolio",
        choices=PROOF_GENERATION_STRATEGY_PORTFOLIOS,
        help=(
            "Choose the initial OPD proof prompt portfolio. 'baseline' keeps "
            "the trained prompt unchanged; 'diverse' deterministically assigns "
            "half the candidates to baseline and half to targeted planning "
            "emphases."
        ),
    )
    pipeline.add_argument(
        "--verify-candidate-limit-while-generating",
        type=int,
    )
    pipeline.add_argument(
        "--verify-request-limit-while-generating",
        type=int,
    )
    pipeline.add_argument("--verify-n", type=int)
    pipeline.add_argument("--verifier-generalist-n", type=int)
    pipeline.add_argument("--meta-n", type=int)
    pipeline.add_argument("--refine-rounds", type=int)
    pipeline.add_argument("--refine-review-n", type=int)
    pipeline.add_argument("--min-valid-low", type=int)
    pipeline.add_argument(
        "--refinement-strategy",
        choices=REFINEMENT_STRATEGIES,
    )
    pipeline.add_argument("--strict-pass-challenge-rounds", type=int)
    pipeline.add_argument(
        "--thinking-budget-handoff-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    pipeline.add_argument(
        "--thinking-budget-handoff-preserve-refine-rounds",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    pipeline.add_argument("--thinking-budget-handoff-max-tokens", type=int)
    pipeline.add_argument("--thinking-budget-handoff-temperature", type=float)
    pipeline.add_argument(
        "--thinking-budget-handoff-prompt-variant",
        choices=HANDOFF_VARIANTS,
    )
    pipeline.add_argument(
        "--thinking-budget-handoff-mode",
        choices=HANDOFF_MODES,
    )
    pipeline.add_argument(
        "--thinking-budget-restart-strategy",
        choices=RESTART_STRATEGIES,
    )
    pipeline.add_argument(
        "--thinking-budget-restart-until-complete",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Keep summarizing and restarting budget-stopped proof/refinement "
            "calls until one finishes normally; the per-problem timeout "
            "remains the hard limit."
        ),
    )
    pipeline.add_argument("--thinking-budget-final-round-tokens", type=int)
    pipeline.add_argument(
        "--thinking-budget-refine-handoff-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    pipeline.add_argument("--thinking-budget-refine-tokens", type=int)
    pipeline.add_argument("--thinking-budget-refine-final-round-tokens", type=int)
    pipeline.add_argument("--thinking-budget-refine-max-restarts", type=int)
    pipeline.add_argument(
        "--thinking-budget-refine-final-temperature",
        type=float,
    )
    pipeline.add_argument(
        "--thinking-budget-refine-visible-output-target-tokens",
        type=int,
    )
    pipeline.add_argument(
        "--thinking-budget-refine-visible-output-limit-tokens",
        type=int,
    )
    pipeline.add_argument("--problem-timeout-seconds", type=int)
    pipeline.add_argument("--selection-reserve-seconds", type=int)
    pipeline.add_argument("--selector-mode", choices=("llm", "score"))
    pipeline.add_argument("--selector-candidate-limit", type=int)
    pipeline.add_argument("--selector-historical-candidate-limit", type=int)
    pipeline.add_argument("--temperature", type=float)
    pipeline.add_argument("--top-p", type=float)
    pipeline.add_argument("--top-k", type=int)
    pipeline.add_argument("--min-p", type=float)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--max-rows", type=int)
    runtime.add_argument("--stream-interval", type=int)
    runtime.add_argument(
        "--stream-vllm",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    runtime.add_argument(
        "--stream-vllm-server-log",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    runtime.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    runtime.add_argument("--mock-llm", action="store_true", default=None)
    return parser


def apply_cli_overrides(cfg: Any, args: argparse.Namespace) -> None:
    field_map = {
        "model_path": "model_path",
        "input_path": "input_csv",
        "output_path": "output_csv",
        "logdir": "logdir",
        "num_gpus": "num_gpus",
        "gpus": "gpus",
        "tensor_parallel_size": "tensor_parallel_size",
        "data_parallel_size": "data_parallel_size",
        "dtype": "dtype",
        "gpu_memory_utilization": "gpu_memory_utilization",
        "max_num_seqs": "max_num_seqs",
        "requests_per_gpu": "requests_per_gpu",
        "max_concurrent_requests": "max_concurrent_requests",
        "host": "host",
        "port": "port",
        "served_model_name": "served_model_name",
        "server_timeout": "server_timeout",
        "base_url": "base_url",
        "no_serve": "no_serve",
        "num_ctx": "num_ctx",
        "max_new_tokens": "max_new_tokens",
        "verifier_max_new_tokens": "verifier_max_new_tokens",
        "meta_max_new_tokens": "meta_max_new_tokens",
        "selector_max_new_tokens": "selector_max_new_tokens",
        "pipelines_per_problem": "pipelines_per_problem",
        "max_concurrent_problems": "max_concurrent_problems",
        "deepseek_math_v2_candidate_count": "deepseek_math_v2_candidate_count",
        "proof_generation_only": "proof_generation_only",
        "proof_generation_strategy_portfolio": (
            "proof_generation_strategy_portfolio"
        ),
        "verify_candidate_limit_while_generating": (
            "verify_candidate_limit_while_generating"
        ),
        "verify_request_limit_while_generating": (
            "verify_request_limit_while_generating"
        ),
        "verify_n": "verify_n",
        "verifier_generalist_n": "verifier_generalist_n",
        "meta_n": "meta_n",
        "refine_rounds": "refine_rounds",
        "refine_review_n": "refine_review_n",
        "min_valid_low": "min_valid_low",
        "refinement_strategy": "refinement_strategy",
        "strict_pass_challenge_rounds": "strict_pass_challenge_rounds",
        "thinking_budget_handoff_enabled": "thinking_budget_handoff_enabled",
        "thinking_budget_handoff_preserve_refine_rounds": (
            "thinking_budget_handoff_preserve_refine_rounds"
        ),
        "thinking_budget_handoff_max_tokens": (
            "thinking_budget_handoff_max_tokens"
        ),
        "thinking_budget_handoff_temperature": (
            "thinking_budget_handoff_temperature"
        ),
        "thinking_budget_handoff_prompt_variant": (
            "thinking_budget_handoff_prompt_variant"
        ),
        "thinking_budget_handoff_mode": "thinking_budget_handoff_mode",
        "thinking_budget_restart_strategy": (
            "thinking_budget_restart_strategy"
        ),
        "thinking_budget_restart_until_complete": (
            "thinking_budget_restart_until_complete"
        ),
        "thinking_budget_final_round_tokens": (
            "thinking_budget_final_round_tokens"
        ),
        "thinking_budget_refine_handoff_enabled": (
            "thinking_budget_refine_handoff_enabled"
        ),
        "thinking_budget_refine_tokens": "thinking_budget_refine_tokens",
        "thinking_budget_refine_final_round_tokens": (
            "thinking_budget_refine_final_round_tokens"
        ),
        "thinking_budget_refine_max_restarts": (
            "thinking_budget_refine_max_restarts"
        ),
        "thinking_budget_refine_final_temperature": (
            "thinking_budget_refine_final_temperature"
        ),
        "thinking_budget_refine_visible_output_target_tokens": (
            "thinking_budget_refine_visible_output_target_tokens"
        ),
        "thinking_budget_refine_visible_output_limit_tokens": (
            "thinking_budget_refine_visible_output_limit_tokens"
        ),
        "problem_timeout_seconds": "problem_timeout_seconds",
        "selection_reserve_seconds": "selection_reserve_seconds",
        "selector_mode": "selector_mode",
        "selector_candidate_limit": "selector_candidate_limit",
        "selector_historical_candidate_limit": (
            "selector_historical_candidate_limit"
        ),
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "min_p": "min_p",
        "max_rows": "max_rows",
        "stream_interval": "stream_interval",
        "stream_vllm": "stream_vllm",
        "stream_vllm_server_log": "stream_vllm_server_log",
        "verbose": "verbose",
        "mock_llm": "mock_llm",
    }
    for argument_name, field_name in field_map.items():
        value = getattr(args, argument_name)
        if value is not None:
            setattr(cfg, field_name, value)
    if args.output_path is not None:
        os.environ["AIMO_OUTPUT_PATH"] = str(args.output_path)
    if args.logdir is not None:
        os.environ["AIMO_LOGDIR"] = str(args.logdir)

    environment_map = {
        "node_rank": "AIMO_NODE_RANK",
        "world_size": "WORLD_SIZE",
        "master_addr": "MASTER_ADDR",
        "master_port": "MASTER_PORT",
        "distributed_run_id": "AIMO_DISTRIBUTED_RUN_ID",
        "distributed_root": "AIMO_DISTRIBUTED_ROOT",
        "distributed_timeout_seconds": "AIMO_DISTRIBUTED_TIMEOUT_SECONDS",
        "distributed_poll_seconds": "AIMO_DISTRIBUTED_POLL_SECONDS",
    }
    for argument_name, environment_name in environment_map.items():
        value = getattr(args, argument_name)
        if value is not None:
            os.environ[environment_name] = str(value)
    if args.distributed_overwrite is not None:
        os.environ["AIMO_DISTRIBUTED_OVERWRITE"] = (
            "1" if args.distributed_overwrite else "0"
        )

    rebuild_vllm_args = False
    if args.no_dflash:
        os.environ.pop("AIMO_DFLASH_MODEL_PATH", None)
        rebuild_vllm_args = True
    elif args.dflash_model_path is not None:
        os.environ["AIMO_DFLASH_MODEL_PATH"] = str(args.dflash_model_path)
        rebuild_vllm_args = True
    for argument_name, environment_name in (
        ("dflash_num_speculative_tokens", "AIMO_DFLASH_NUM_SPECULATIVE_TOKENS"),
        ("dflash_context_cutoff", "AIMO_DFLASH_CONTEXT_CUTOFF"),
        ("max_num_batched_tokens", "AIMO_MAX_NUM_BATCHED_TOKENS"),
    ):
        value = getattr(args, argument_name)
        if value is not None:
            os.environ[environment_name] = str(value)
            rebuild_vllm_args = True
    if rebuild_vllm_args:
        cfg.vllm_extra_args = default_vllm_extra_args()
        if args.min_p is None:
            cfg.min_p = default_min_p()
    if args.vllm_extra_args is not None:
        cfg.vllm_extra_args = args.vllm_extra_args


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    apply_cli_overrides(CFG, args)
    run(CFG)


def run(cfg: type[CFG] = CFG) -> None:
    if int(cfg.max_concurrent_problems) < 1:
        raise ValueError("AIMO_MAX_CONCURRENT_PROBLEMS must be at least 1")
    if int(cfg.pipelines_per_problem) < 1:
        raise ValueError("AIMO_PIPELINES_PER_PROBLEM must be at least 1")
    if int(cfg.refine_rounds) < 0:
        raise ValueError("AIMO_REFINE_ROUNDS cannot be negative")
    if not 0 <= int(cfg.verifier_generalist_n) <= int(cfg.verify_n):
        raise ValueError(
            "AIMO_VERIFIER_GENERALIST_N must be between 0 and AIMO_VERIFY_N"
        )
    if int(cfg.min_valid_low) < 1:
        raise ValueError("AIMO_MIN_VALID_LOW must be at least 1")
    if (
        str(cfg.proof_generation_strategy_portfolio).strip().lower()
        not in PROOF_GENERATION_STRATEGY_PORTFOLIOS
    ):
        raise ValueError(
            "AIMO_PROOF_GENERATION_STRATEGY_PORTFOLIO must be one of "
            + ", ".join(PROOF_GENERATION_STRATEGY_PORTFOLIOS)
        )
    if int(cfg.thinking_budget_handoff_max_tokens) < 1:
        raise ValueError(
            "AIMO_THINKING_BUDGET_HANDOFF_MAX_TOKENS must be positive"
        )
    if float(cfg.thinking_budget_handoff_temperature) < 0:
        raise ValueError(
            "AIMO_THINKING_BUDGET_HANDOFF_TEMPERATURE cannot be negative"
        )
    if cfg.thinking_budget_handoff_prompt_variant not in HANDOFF_VARIANTS:
        raise ValueError(
            "AIMO_THINKING_BUDGET_HANDOFF_PROMPT_VARIANT must be one of "
            + ", ".join(HANDOFF_VARIANTS)
        )
    if cfg.thinking_budget_handoff_mode not in HANDOFF_MODES:
        raise ValueError(
            "AIMO_THINKING_BUDGET_HANDOFF_MODE must be one of "
            + ", ".join(HANDOFF_MODES)
        )
    if cfg.thinking_budget_restart_strategy not in RESTART_STRATEGIES:
        raise ValueError(
            "AIMO_THINKING_BUDGET_RESTART_STRATEGY must be one of "
            + ", ".join(RESTART_STRATEGIES)
        )
    if int(cfg.thinking_budget_final_round_tokens) < 0:
        raise ValueError(
            "AIMO_THINKING_BUDGET_FINAL_ROUND_TOKENS cannot be negative"
        )
    if int(cfg.thinking_budget_final_round_tokens) >= int(cfg.max_new_tokens):
        raise ValueError(
            "AIMO_THINKING_BUDGET_FINAL_ROUND_TOKENS must be below "
            "AIMO_MAX_NEW_TOKENS"
        )
    for name, value in (
        ("AIMO_THINKING_BUDGET_REFINE_TOKENS", cfg.thinking_budget_refine_tokens),
        (
            "AIMO_THINKING_BUDGET_REFINE_FINAL_ROUND_TOKENS",
            cfg.thinking_budget_refine_final_round_tokens,
        ),
        (
            "AIMO_THINKING_BUDGET_REFINE_MAX_RESTARTS",
            cfg.thinking_budget_refine_max_restarts,
        ),
    ):
        if int(value) < 0:
            raise ValueError(f"{name} cannot be negative")
    for name, value in (
        ("AIMO_THINKING_BUDGET_REFINE_TOKENS", cfg.thinking_budget_refine_tokens),
        (
            "AIMO_THINKING_BUDGET_REFINE_FINAL_ROUND_TOKENS",
            cfg.thinking_budget_refine_final_round_tokens,
        ),
    ):
        if int(value) >= int(cfg.max_new_tokens):
            raise ValueError(f"{name} must be below AIMO_MAX_NEW_TOKENS")
    if cfg.thinking_budget_refine_handoff_enabled and (
        int(cfg.thinking_budget_refine_tokens) < 1
        or int(cfg.thinking_budget_refine_final_round_tokens) < 1
        or int(cfg.thinking_budget_refine_max_restarts) < 1
    ):
        raise ValueError(
            "Refinement handoff requires positive initial/final budgets and "
            "at least one restart"
        )
    if (
        cfg.thinking_budget_refine_final_temperature is not None
        and float(cfg.thinking_budget_refine_final_temperature) < 0
    ):
        raise ValueError(
            "AIMO_THINKING_BUDGET_REFINE_FINAL_TEMPERATURE cannot be negative"
        )
    if int(cfg.thinking_budget_refine_visible_output_target_tokens) < 0:
        raise ValueError(
            "AIMO_THINKING_BUDGET_REFINE_VISIBLE_OUTPUT_TARGET_TOKENS "
            "cannot be negative"
        )
    if int(cfg.thinking_budget_refine_visible_output_limit_tokens) < 0:
        raise ValueError(
            "AIMO_THINKING_BUDGET_REFINE_VISIBLE_OUTPUT_LIMIT_TOKENS "
            "cannot be negative"
        )
    selected_gpus, resolved_tp, resolved_dp = resolve_gpu_parallel_layout(cfg)
    resolved_max_concurrent_requests = resolve_max_concurrent_requests(
        cfg,
        len(selected_gpus),
    )
    distributed = DistributedRuntime.from_environment()
    distributed.initialize(
        {
            "run_py_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "model_path": str(cfg.model_path),
            "input_csv": str(cfg.input_csv),
            "max_rows": int(cfg.max_rows),
            "num_ctx": int(cfg.num_ctx),
            "proof_max_new_tokens": int(cfg.max_new_tokens),
            "verifier_max_new_tokens": int(cfg.verifier_max_new_tokens),
            "meta_max_new_tokens": int(cfg.meta_max_new_tokens),
            "selector_max_new_tokens": int(cfg.selector_max_new_tokens),
            "pipelines_per_problem": int(cfg.pipelines_per_problem),
            "deepseek_math_v2_candidate_count": int(
                cfg.deepseek_math_v2_candidate_count
            ),
            "proof_only_candidate_count": int(cfg.proof_only_candidate_count),
            "proof_generation_only": bool(cfg.proof_generation_only),
            "proof_generation_strategy_portfolio": str(
                cfg.proof_generation_strategy_portfolio
            ),
            "verify_candidate_limit_while_generating": int(
                cfg.verify_candidate_limit_while_generating
            ),
            "verify_request_limit_while_generating": int(
                cfg.verify_request_limit_while_generating
            ),
            "verify_n": int(cfg.verify_n),
            "verifier_generalist_n": int(cfg.verifier_generalist_n),
            "meta_n": int(cfg.meta_n),
            "meta_policy": str(cfg.meta_policy),
            "audit_positive_meta": str(cfg.meta_policy) == "all-reviews",
            "strict_pass_meta": bool(cfg.strict_pass_meta),
            "refine_rounds": int(cfg.refine_rounds),
            "refine_review_n": int(cfg.refine_review_n),
            "min_valid_low": int(cfg.min_valid_low),
            "refinement_strategy": str(cfg.refinement_strategy),
            "strict_pass_challenge_rounds": int(
                cfg.strict_pass_challenge_rounds
            ),
            "selector_mode": str(cfg.selector_mode),
            "selector_candidate_limit": int(cfg.selector_candidate_limit),
            "selector_historical_candidate_limit": int(
                cfg.selector_historical_candidate_limit
            ),
            "temperature": float(cfg.temperature),
            "top_p": float(cfg.top_p),
            "top_k": int(cfg.top_k),
            "min_p": cfg.min_p,
            "proof_generation_temperatures": list(
                cfg.proof_generation_temperatures
            ),
            "proof_generation_thinking_budgets": list(
                cfg.proof_generation_thinking_budgets
            ),
            "thinking_budget_handoff_enabled": bool(
                cfg.thinking_budget_handoff_enabled
            ),
            "thinking_budget_handoff_preserve_refine_rounds": bool(
                cfg.thinking_budget_handoff_preserve_refine_rounds
            ),
            "thinking_budget_handoff_max_tokens": int(
                cfg.thinking_budget_handoff_max_tokens
            ),
            "thinking_budget_handoff_temperature": float(
                cfg.thinking_budget_handoff_temperature
            ),
            "thinking_budget_handoff_prompt_variant": str(
                cfg.thinking_budget_handoff_prompt_variant
            ),
            "thinking_budget_handoff_mode": str(
                cfg.thinking_budget_handoff_mode
            ),
            "thinking_budget_restart_strategy": str(
                cfg.thinking_budget_restart_strategy
            ),
            "thinking_budget_restart_until_complete": bool(
                cfg.thinking_budget_restart_until_complete
            ),
            "thinking_budget_final_round_tokens": int(
                cfg.thinking_budget_final_round_tokens
            ),
            "thinking_budget_refine_handoff_enabled": bool(
                cfg.thinking_budget_refine_handoff_enabled
            ),
            "thinking_budget_refine_tokens": int(
                cfg.thinking_budget_refine_tokens
            ),
            "thinking_budget_refine_final_round_tokens": int(
                cfg.thinking_budget_refine_final_round_tokens
            ),
            "thinking_budget_refine_max_restarts": int(
                cfg.thinking_budget_refine_max_restarts
            ),
            "thinking_budget_refine_final_temperature": (
                None
                if cfg.thinking_budget_refine_final_temperature is None
                else float(cfg.thinking_budget_refine_final_temperature)
            ),
            "thinking_budget_refine_visible_output_target_tokens": int(
                cfg.thinking_budget_refine_visible_output_target_tokens
            ),
            "thinking_budget_refine_visible_output_limit_tokens": int(
                cfg.thinking_budget_refine_visible_output_limit_tokens
            ),
            "vllm_extra_args": str(cfg.vllm_extra_args),
            "served_model_name": str(cfg.served_model_name),
            "mock_llm": bool(cfg.mock_llm),
            "tensor_parallel_size": resolved_tp,
            "data_parallel_size": resolved_dp,
            "selected_gpus": selected_gpus,
            "max_concurrent_requests": resolved_max_concurrent_requests,
        }
    )
    runtime_logdir = distributed.rank_logdir(cfg.logdir)
    output_csv = distributed.output_path(cfg.output_csv)
    setup_logging(runtime_logdir)
    logging.info(
        "Inference runtime: stream_vllm=%s stream_vllm_server_log=%s verbose=%s "
        "meta_policy=%s strict_pass_meta=%s proof_generation_only=%s "
        "proof_generation_strategy_portfolio=%s refinement_strategy=%s "
        "strict_pass_challenge_rounds=%s "
        "max_concurrent_problems=%s "
        "candidates=%s deepseek_math_v2_candidates=%s gpus=%s tp=%s dp=%s "
        "max_concurrent_requests=%s requests_per_gpu=%s "
        "verify_candidates_while_generating=%s "
        "verify_requests_while_generating=%s "
        "thinking_handoff=%s preserve_refine_rounds=%s "
        "handoff_variant=%s handoff_mode=%s "
        "restart_strategy=%s restart_until_complete=%s "
        "final_round_budget=%s handoff_tokens=%s "
        "handoff_temperature=%s refine_handoff=%s refine_budget=%s "
        "refine_final_budget=%s refine_max_restarts=%s "
        "refine_final_temperature=%s refine_visible_target=%s "
        "refine_visible_limit=%s "
        "node_rank=%s/%s distributed_run=%s output=%s",
        cfg.stream_vllm,
        cfg.stream_vllm_server_log,
        cfg.verbose,
        cfg.meta_policy,
        cfg.strict_pass_meta,
        cfg.proof_generation_only,
        cfg.proof_generation_strategy_portfolio,
        cfg.refinement_strategy,
        cfg.strict_pass_challenge_rounds,
        cfg.max_concurrent_problems,
        cfg.pipelines_per_problem,
        cfg.deepseek_math_v2_candidate_count,
        ",".join(selected_gpus),
        resolved_tp,
        resolved_dp,
        resolved_max_concurrent_requests,
        cfg.requests_per_gpu,
        cfg.verify_candidate_limit_while_generating,
        cfg.verify_request_limit_while_generating,
        cfg.thinking_budget_handoff_enabled,
        cfg.thinking_budget_handoff_preserve_refine_rounds,
        cfg.thinking_budget_handoff_prompt_variant,
        cfg.thinking_budget_handoff_mode,
        cfg.thinking_budget_restart_strategy,
        cfg.thinking_budget_restart_until_complete,
        cfg.thinking_budget_final_round_tokens,
        cfg.thinking_budget_handoff_max_tokens,
        cfg.thinking_budget_handoff_temperature,
        cfg.thinking_budget_refine_handoff_enabled,
        cfg.thinking_budget_refine_tokens,
        cfg.thinking_budget_refine_final_round_tokens,
        cfg.thinking_budget_refine_max_restarts,
        cfg.thinking_budget_refine_final_temperature,
        cfg.thinking_budget_refine_visible_output_target_tokens,
        cfg.thinking_budget_refine_visible_output_limit_tokens,
        distributed.rank,
        distributed.world_size,
        distributed.run_id or "local",
        output_csv,
    )
    runtime: Optional[ProofRuntime] = None
    try:
        runtime = ProofRuntime(
            model_path=cfg.model_path,
            logdir=runtime_logdir,
            gpu_group=",".join(selected_gpus),
            tensor_parallel_size=resolved_tp,
            data_parallel_size=resolved_dp,
            num_ctx=cfg.num_ctx,
            dtype=cfg.dtype,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_num_seqs=cfg.max_num_seqs,
            max_concurrent_requests=resolved_max_concurrent_requests,
            pipelines_per_problem=cfg.pipelines_per_problem,
            deepseek_math_v2_candidate_count=cfg.deepseek_math_v2_candidate_count,
            proof_only_candidate_count=cfg.proof_only_candidate_count,
            skip_self_score_zero=cfg.skip_self_score_zero,
            stop_on_strict_pass=cfg.stop_on_strict_pass,
            verification_early_stop=cfg.verification_early_stop,
            wait_for_all_generations_before_verify=(
                cfg.wait_for_all_generations_before_verify
            ),
            proof_generation_only=cfg.proof_generation_only,
            proof_generation_strategy_portfolio=(
                cfg.proof_generation_strategy_portfolio
            ),
            verify_candidate_limit_while_generating=(
                cfg.verify_candidate_limit_while_generating
            ),
            verify_request_limit_while_generating=(
                cfg.verify_request_limit_while_generating
            ),
            verify_n=cfg.verify_n,
            verifier_generalist_n=cfg.verifier_generalist_n,
            meta_n=cfg.meta_n,
            meta_policy=cfg.meta_policy,
            strict_pass_meta=cfg.strict_pass_meta,
            refine_rounds=cfg.refine_rounds,
            refine_review_n=cfg.refine_review_n,
            min_valid_low=cfg.min_valid_low,
            refinement_strategy=cfg.refinement_strategy,
            strict_pass_challenge_rounds=cfg.strict_pass_challenge_rounds,
            problem_timeout_seconds=cfg.problem_timeout_seconds,
            selection_reserve_seconds=cfg.selection_reserve_seconds,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            min_new_tokens=cfg.min_new_tokens,
            min_p=cfg.min_p,
            proof_max_new_tokens=cfg.max_new_tokens,
            proof_generation_temperatures=list(cfg.proof_generation_temperatures),
            thinking_budget_enabled=cfg.thinking_budget_enabled,
            proof_generation_thinking_budgets=list(
                cfg.proof_generation_thinking_budgets
            ),
            thinking_budget_force_text=cfg.thinking_budget_force_text,
            thinking_budget_handoff_enabled=(
                cfg.thinking_budget_handoff_enabled
            ),
            thinking_budget_handoff_preserve_refine_rounds=(
                cfg.thinking_budget_handoff_preserve_refine_rounds
            ),
            thinking_budget_handoff_max_tokens=(
                cfg.thinking_budget_handoff_max_tokens
            ),
            thinking_budget_handoff_temperature=(
                cfg.thinking_budget_handoff_temperature
            ),
            thinking_budget_handoff_prompt_variant=(
                cfg.thinking_budget_handoff_prompt_variant
            ),
            thinking_budget_handoff_mode=cfg.thinking_budget_handoff_mode,
            thinking_budget_restart_strategy=(
                cfg.thinking_budget_restart_strategy
            ),
            thinking_budget_restart_until_complete=(
                cfg.thinking_budget_restart_until_complete
            ),
            thinking_budget_final_round_tokens=(
                cfg.thinking_budget_final_round_tokens
            ),
            thinking_budget_refine_handoff_enabled=(
                cfg.thinking_budget_refine_handoff_enabled
            ),
            thinking_budget_refine_tokens=cfg.thinking_budget_refine_tokens,
            thinking_budget_refine_final_round_tokens=(
                cfg.thinking_budget_refine_final_round_tokens
            ),
            thinking_budget_refine_max_restarts=(
                cfg.thinking_budget_refine_max_restarts
            ),
            thinking_budget_refine_final_temperature=(
                cfg.thinking_budget_refine_final_temperature
            ),
            thinking_budget_refine_visible_output_target_tokens=(
                cfg.thinking_budget_refine_visible_output_target_tokens
            ),
            thinking_budget_refine_visible_output_limit_tokens=(
                cfg.thinking_budget_refine_visible_output_limit_tokens
            ),
            deepseek_thinking_budget_force_text=(
                cfg.deepseek_thinking_budget_force_text
            ),
            verifier_thinking_budget_tokens=cfg.verifier_thinking_budget_tokens,
            verifier_thinking_budget_force_text=(
                cfg.verifier_thinking_budget_force_text
            ),
            deepseek_verifier_thinking_budget_force_text=(
                cfg.deepseek_verifier_thinking_budget_force_text
            ),
            meta_thinking_budget_tokens=cfg.meta_thinking_budget_tokens,
            meta_thinking_budget_force_text=cfg.meta_thinking_budget_force_text,
            verifier_max_new_tokens=cfg.verifier_max_new_tokens,
            meta_max_new_tokens=cfg.meta_max_new_tokens,
            selector_max_new_tokens=cfg.selector_max_new_tokens,
            selector_max_candidate_chars=cfg.selector_max_candidate_chars,
            selection_temperature=cfg.selection_temperature,
            selector_mode=cfg.selector_mode,
            selector_min_final_score=cfg.selector_min_final_score,
            selector_candidate_limit=cfg.selector_candidate_limit,
            selector_historical_candidate_limit=(
                cfg.selector_historical_candidate_limit
            ),
            vllm_extra_args=cfg.vllm_extra_args,
            stream_interval=cfg.stream_interval,
            host=cfg.host,
            port=cfg.port,
            api_key=cfg.api_key,
            served_model_name=cfg.served_model_name,
            server_timeout=cfg.server_timeout,
            no_serve=cfg.no_serve,
            base_url=cfg.base_url,
            mock_llm=cfg.mock_llm,
            stream_vllm=cfg.stream_vllm,
            stream_vllm_server_log=cfg.stream_vllm_server_log,
            verbose=cfg.verbose,
            distributed_runtime=distributed,
        )
        run_simple_csv(
            runtime,
            cfg.input_csv,
            output_csv,
            cfg.max_rows,
            cfg.max_concurrent_problems,
        )
    except BaseException as exc:
        distributed.report_failure(exc)
        raise
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    main()
