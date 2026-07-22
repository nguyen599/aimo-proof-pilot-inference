from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.harness_vllm import run  # noqa: E402


class RunOpdPromptContractTests(unittest.TestCase):
    def test_cfg_reads_proof_generation_only_environment(self):
        env = dict(os.environ)
        env["AIMO_PROOF_GENERATION_ONLY"] = "true"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from evaluation.harness_vllm.run import CFG; "
                    "print(CFG.proof_generation_only)"
                ),
            ],
            cwd=REPO,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "True")

    def test_nii_launcher_accepts_adaptive_proof_portfolio(self):
        launcher = (REPO / "scripts" / "launch_nii_imo2025_all.sh").read_text()
        self.assertIn("baseline|diverse|adaptive", launcher)

    def test_cli_overrides_cfg_and_distributed_environment(self):
        cfg = SimpleNamespace(
            model_path=Path("/old-model"),
            input_csv=Path("/old-input.csv"),
            pipelines_per_problem=14,
            max_concurrent_problems=1,
            verify_candidate_limit_while_generating=2,
            verify_request_limit_while_generating=8,
            refine_rounds=1,
            proof_generation_strategy_portfolio="baseline",
        )
        args = run.build_cli_parser().parse_args(
            [
                "--model-path",
                "/models/current",
                "--input-path",
                "/data/imo.parquet",
                "--output-path",
                "/output/submission.csv",
                "--logdir",
                "/output/logs",
                "--pipelines-per-problem",
                "16",
                "--max-concurrent-problems",
                "2",
                "--verify-candidate-limit-while-generating",
                "4",
                "--verify-request-limit-while-generating",
                "16",
                "--refine-rounds",
                "1",
                "--proof-generation-strategy-portfolio",
                "diverse",
                "--thinking-budget-refine-final-temperature",
                "0.6",
                "--thinking-budget-refine-visible-output-target-tokens",
                "12000",
                "--thinking-budget-refine-visible-output-limit-tokens",
                "12000",
                "--node-rank",
                "1",
                "--world-size",
                "2",
                "--master-addr",
                "10.0.0.1",
                "--master-port",
                "29500",
            ]
        )

        with patch.dict(os.environ, {}, clear=True):
            run.apply_cli_overrides(cfg, args)
            self.assertEqual(os.environ["AIMO_NODE_RANK"], "1")
            self.assertEqual(os.environ["WORLD_SIZE"], "2")
            self.assertEqual(os.environ["MASTER_ADDR"], "10.0.0.1")
            self.assertEqual(os.environ["MASTER_PORT"], "29500")
            self.assertEqual(
                os.environ["AIMO_OUTPUT_PATH"], "/output/submission.csv"
            )
            self.assertEqual(os.environ["AIMO_LOGDIR"], "/output/logs")

        self.assertEqual(cfg.model_path, Path("/models/current"))
        self.assertEqual(cfg.input_csv, Path("/data/imo.parquet"))
        self.assertEqual(cfg.pipelines_per_problem, 16)
        self.assertEqual(cfg.max_concurrent_problems, 2)
        self.assertEqual(cfg.verify_candidate_limit_while_generating, 4)
        self.assertEqual(cfg.verify_request_limit_while_generating, 16)
        self.assertEqual(cfg.refine_rounds, 1)
        self.assertEqual(cfg.proof_generation_strategy_portfolio, "diverse")
        self.assertEqual(cfg.thinking_budget_refine_final_temperature, 0.6)
        self.assertEqual(
            cfg.thinking_budget_refine_visible_output_target_tokens,
            12_000,
        )
        self.assertEqual(
            cfg.thinking_budget_refine_visible_output_limit_tokens,
            12_000,
        )

    def test_cli_dflash_options_rebuild_vllm_args(self):
        cfg = SimpleNamespace(vllm_extra_args="", min_p=0.01)
        args = run.build_cli_parser().parse_args(
            [
                "--dflash-model-path",
                "/models/draft",
                "--dflash-num-speculative-tokens",
                "8",
                "--dflash-context-cutoff",
                "32768",
                "--max-num-batched-tokens",
                "24576",
            ]
        )

        with patch.dict(os.environ, {}, clear=True):
            run.apply_cli_overrides(cfg, args)

        vllm_args = shlex.split(cfg.vllm_extra_args)
        speculative_config = json.loads(
            vllm_args[vllm_args.index("--speculative-config") + 1]
        )
        self.assertEqual(speculative_config["model"], "/models/draft")
        self.assertEqual(speculative_config["num_speculative_tokens"], 8)
        self.assertEqual(speculative_config["disable_above_context_len"], 32768)
        self.assertEqual(
            vllm_args[vllm_args.index("--max-num-batched-tokens") + 1],
            "24576",
        )
        self.assertIsNone(cfg.min_p)

    def test_parquet_input_is_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "problems.parquet"
            run.pd.DataFrame(
                {
                    "problem_idx": [1, 2],
                    "problem": ["Prove the first claim.", "Prove the second claim."],
                }
            ).to_parquet(input_path)
            frame, problem_column, id_column = run.load_simple_input(input_path)

        self.assertEqual(len(frame), 2)
        self.assertEqual(problem_column, "problem")
        self.assertEqual(id_column, "problem_idx")

    def test_long_context_defaults_stay_near_training_length(self):
        self.assertEqual(run.CFG.num_ctx, 262_144)
        self.assertIn("--quantization fp8", run.CFG.vllm_extra_args)
        self.assertEqual(run.CFG.gpu_memory_utilization, 0.92)
        vllm_args = shlex.split(run.CFG.vllm_extra_args)
        self.assertEqual(
            vllm_args[vllm_args.index("--max-num-batched-tokens") + 1],
            "16384",
        )
        self.assertLessEqual(run.CFG.max_new_tokens, 131_072)
        self.assertGreaterEqual(min(run.CFG.proof_generation_thinking_budgets), 120_000)
        self.assertLess(
            max(run.CFG.proof_generation_thinking_budgets),
            run.CFG.max_new_tokens,
        )
        self.assertEqual(
            run.CFG.thinking_budget_force_text,
            "\n</think>\n\n<solution>\n"
            "We were unable to produce a complete proof. However, the strongest "
            "partial progress is as follows:\n",
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

    def test_dflash_can_be_enabled_from_environment(self):
        with patch.dict(
            os.environ,
            {
                "AIMO_DFLASH_MODEL_PATH": "/draft",
                "AIMO_DFLASH_NUM_SPECULATIVE_TOKENS": "10",
            },
        ):
            args = shlex.split(run.default_vllm_extra_args())

        config_index = args.index("--speculative-config") + 1
        self.assertEqual(
            json.loads(args[config_index]),
            {
                "method": "dflash",
                "model": "/draft",
                "num_speculative_tokens": 10,
                "disable_above_context_len": 65536,
            },
        )

    def test_max_num_batched_tokens_can_be_overridden(self):
        with patch.dict(
            os.environ,
            {"AIMO_MAX_NUM_BATCHED_TOKENS": "32768"},
        ):
            args = shlex.split(run.default_vllm_extra_args())

        self.assertEqual(
            args[args.index("--max-num-batched-tokens") + 1],
            "32768",
        )

    def test_tp2_dp4_auto_selects_eight_gpus(self):
        cfg = SimpleNamespace(
            gpus="",
            num_gpus=1,
            tensor_parallel_size=2,
            data_parallel_size=4,
        )

        selected_gpus, tp_size, dp_size = run.resolve_gpu_parallel_layout(cfg)

        self.assertEqual(selected_gpus, [str(index) for index in range(8)])
        self.assertEqual(tp_size, 2)
        self.assertEqual(dp_size, 4)

    def test_parallel_layout_infers_tp_from_explicit_gpus(self):
        cfg = SimpleNamespace(
            gpus="0,1,2,3,4,5,6,7",
            num_gpus=1,
            tensor_parallel_size=0,
            data_parallel_size=4,
        )

        selected_gpus, tp_size, dp_size = run.resolve_gpu_parallel_layout(cfg)

        self.assertEqual(len(selected_gpus), 8)
        self.assertEqual(tp_size, 2)
        self.assertEqual(dp_size, 4)

    def test_parallel_layout_rejects_incomplete_gpu_group(self):
        cfg = SimpleNamespace(
            gpus="0,1,2,3,4,5,6",
            num_gpus=1,
            tensor_parallel_size=2,
            data_parallel_size=4,
        )

        with self.assertRaisesRegex(ValueError, "GPU count must equal TP x DP"):
            run.resolve_gpu_parallel_layout(cfg)

    def test_request_concurrency_scales_with_selected_gpu_count(self):
        cfg = SimpleNamespace(
            max_concurrent_requests=0,
            requests_per_gpu=32,
        )

        self.assertEqual(run.resolve_max_concurrent_requests(cfg, 8), 256)

    def test_request_concurrency_supports_explicit_override(self):
        cfg = SimpleNamespace(
            max_concurrent_requests=96,
            requests_per_gpu=32,
        )

        self.assertEqual(run.resolve_max_concurrent_requests(cfg, 8), 96)

    def test_request_concurrency_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            run.resolve_max_concurrent_requests(
                SimpleNamespace(
                    max_concurrent_requests=-1,
                    requests_per_gpu=32,
                ),
                8,
            )
        with self.assertRaisesRegex(ValueError, "must be at least 1"):
            run.resolve_max_concurrent_requests(
                SimpleNamespace(
                    max_concurrent_requests=0,
                    requests_per_gpu=0,
                ),
                8,
            )

    def test_vllm_command_forwards_data_parallel_size(self):
        cfg = SimpleNamespace(
            model_path="/model",
            served_model_name="proof-model",
            api_key="key",
            tensor_parallel_size=2,
            data_parallel_size=4,
            max_num_seqs=32,
            gpu_memory_utilization=0.95,
            host="127.0.0.1",
            dtype="auto",
            num_ctx=262_144,
            stream_interval=100,
            vllm_extra_args="",
            logdir=REPO / "outputs" / "test-logs",
        )

        command = run.VLLMServer(
            cfg,
            port=8000,
            gpu_group="0,1,2,3,4,5,6,7",
            index=0,
        ).build_command()

        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "2")
        self.assertEqual(command[command.index("--data-parallel-size") + 1], "4")

    def test_dflash_disables_unsupported_min_p(self):
        with patch.dict(os.environ, {"AIMO_DFLASH_MODEL_PATH": "/draft"}):
            self.assertIsNone(run.default_min_p())
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(run.default_min_p(), 0.0)

    def test_imo2025_defaults_run_the_complete_candidate_pipeline(self):
        self.assertEqual(
            run.CFG.input_csv,
            REPO / "test.csv",
        )
        self.assertEqual(run.CFG.pipelines_per_problem, 14)
        self.assertEqual(run.CFG.deepseek_math_v2_candidate_count, 0)
        self.assertEqual(run.CFG.proof_only_candidate_count, 0)
        self.assertFalse(run.CFG.skip_self_score_zero)
        self.assertFalse(run.CFG.stop_on_strict_pass)
        self.assertFalse(run.CFG.verification_early_stop)
        self.assertEqual(run.CFG.verify_candidate_limit_while_generating, 0)
        self.assertEqual(run.CFG.verify_request_limit_while_generating, 0)
        self.assertEqual(run.CFG.verify_n, 8)
        self.assertEqual(run.CFG.verifier_generalist_n, 4)
        self.assertEqual(run.CFG.meta_n, 1)
        self.assertEqual(run.CFG.meta_policy, "all-reviews")
        self.assertEqual(run.CFG.refine_rounds, 1)
        self.assertEqual(run.CFG.refine_review_n, 4)
        self.assertEqual(run.CFG.min_valid_low, 2)
        self.assertEqual(
            run.CFG.verify_n - run.CFG.verifier_generalist_n,
            len(run.HYBRID_VERIFIER_AUDIT_ROLES),
        )
        self.assertLess(run.CFG.refine_review_n, run.CFG.verify_n)
        self.assertGreaterEqual(run.CFG.problem_timeout_seconds, 86_400)

    def test_verifiers_receive_distinct_adversarial_roles(self):
        prompts = [
            run.build_opd_proof_verification_prompt(
                "Problem.",
                "Candidate proof.",
                "Candidate self-review.",
                verifier_index=index,
            )
            for index in range(run.CFG.verify_n)
        ]

        user_prompts = [messages[-1]["content"] for messages in prompts]
        self.assertEqual(len(set(user_prompts)), 8)
        for role_name, _ in run.VERIFIER_AUDIT_ROLES:
            self.assertTrue(
                any(f"audit role: {role_name}" in prompt for prompt in user_prompts)
            )

    def test_hybrid_verifiers_keep_original_prompt_and_add_four_specialists(self):
        original_prompt = run.build_opd_proof_verification_prompt(
            "Problem.",
            "Candidate proof.",
            "Candidate self-review.",
        )[-1]["content"]
        assignments = [
            run.hybrid_verifier_assignment(
                index,
                run.CFG.verifier_generalist_n,
            )
            for index in range(run.CFG.verify_n)
        ]
        prompts = []
        for index, (_, group, audit_role) in enumerate(assignments):
            prompts.append(
                run.build_opd_proof_verification_prompt(
                    "Problem.",
                    "Candidate proof.",
                    "Candidate self-review.",
                    verifier_index=None if group == "generalist" else index,
                    audit_role=audit_role,
                )[-1]["content"]
            )

        self.assertEqual([group for _, group, _ in assignments].count("generalist"), 4)
        self.assertEqual([group for _, group, _ in assignments].count("specialist"), 4)
        self.assertTrue(all(prompt == original_prompt for prompt in prompts[:4]))
        self.assertEqual(len(set(prompts[4:])), 4)
        for role_name, _ in run.HYBRID_VERIFIER_AUDIT_ROLES:
            self.assertTrue(
                any(f"audit role: {role_name}" in prompt for prompt in prompts[4:])
            )
        for prompt in prompts[4:]:
            self.assertIn("CLAIM_UNDER_TEST:", prompt)
            self.assertIn("ADVERSARIAL_TEST:", prompt)
            self.assertIn("CHECK_RESULT:", prompt)
            self.assertIn("PASS_JUSTIFICATION:", prompt)

    def test_all_reviews_meta_independently_audits_positive_verdicts(self):
        prompt = run.build_deepseek_meta_verification_prompt(
            "Problem.",
            "Candidate proof.",
            "The proof is correct.",
            audit_positive_verdicts=True,
        )

        self.assertIn("adversarial second-opinion auditor", prompt)
        self.assertIn("One cooperative infinite play never proves", prompt)
        self.assertIn("CLAIM_UNDER_TEST:", prompt)
        self.assertNotIn(
            "positive components about the solution", prompt
        )

    def test_verifier_reaudits_prior_validated_critiques(self):
        messages = run.build_opd_proof_verification_prompt(
            "Problem.",
            "Rewritten proof.",
            "Now complete.",
            verifier_index=1,
            prior_critiques=[
                {
                    "origin_round": 0,
                    "verifier_index": 2,
                    "score": 0,
                    "evaluation": "The claimed symmetry does not preserve adjacency.",
                }
            ],
        )

        prompt = messages[-1]["content"]
        self.assertIn("resolved, unresolved, or an invalid earlier critique", prompt)
        self.assertIn("does not preserve adjacency", prompt)

    def test_validated_fatal_review_caps_aggregate_score(self):
        verifier_results = [
            {
                "verifier_index": index,
                "verifier_role": role[0],
                "score": score,
                "evaluation": "fatal issue" if score == 0 else "passes",
            }
            for index, (role, score) in enumerate(
                zip(run.VERIFIER_AUDIT_ROLES, [1.0, 1.0, 1.0, 0.0])
            )
        ]
        meta_results = {
            index: [{"score": 1.0}] for index in range(len(verifier_results))
        }

        result = run.aggregate_proof_label(
            verifier_results,
            meta_results,
            min_valid_low=1,
            strict_pass_meta=True,
            meta_n=1,
        )

        self.assertEqual(result["final_score"], 0.5)
        self.assertTrue(result["fatal_score_cap_applied"])
        self.assertTrue(result["validated_low_score_cap_applied"])
        self.assertEqual(result["final_status"], "validated_low_score")

    def test_validated_nonperfect_review_blocks_selector_eligibility(self):
        verifier_results = [
            {
                "verifier_index": index,
                "verifier_role": role[0],
                "score": score,
                "evaluation": "unresolved gap" if score == 0.5 else "passes",
            }
            for index, (role, score) in enumerate(
                zip(run.VERIFIER_AUDIT_ROLES, [1.0, 1.0, 1.0, 0.5])
            )
        ]
        meta_results = {
            index: [{"score": 1.0}] for index in range(len(verifier_results))
        }

        result = run.aggregate_proof_label(
            verifier_results,
            meta_results,
            min_valid_low=1,
            strict_pass_meta=True,
            meta_n=1,
        )

        self.assertEqual(result["final_score"], 0.5)
        self.assertTrue(result["validated_low_score_cap_applied"])
        self.assertFalse(result["fatal_score_cap_applied"])
        self.assertEqual(result["final_status"], "validated_low_score")

    def test_hybrid_aggregate_balances_generalist_and_specialist_groups(self):
        verifier_results = [
            {
                "verifier_index": index,
                "verifier_role": "generalist",
                "verifier_group": "generalist",
                "score": 1.0,
                "evaluation": "passes",
            }
            for index in range(3)
        ]
        verifier_results.append(
            {
                "verifier_index": 3,
                "verifier_role": "dependency_lemma",
                "verifier_group": "specialist",
                "score": 0.5,
                "evaluation": "minor gap",
            }
        )

        result = run.aggregate_proof_label(
            verifier_results,
            {},
            min_valid_low=2,
            meta_n=0,
        )

        self.assertEqual(result["aggregation_mode"], "balanced_verifier_groups")
        self.assertEqual(
            result["verifier_group_scores"],
            {"generalist": 1.0, "specialist": 0.5},
        )
        self.assertEqual(result["final_score"], 0.75)
        self.assertFalse(result["validated_low_score_cap_applied"])

    def test_hard_score_cap_requires_two_validated_critiques(self):
        def aggregate(specialist_scores):
            scores = [1.0] * 4 + specialist_scores
            verifier_results = [
                {
                    "verifier_index": index,
                    "verifier_role": (
                        "generalist" if index < 4 else f"specialist_{index}"
                    ),
                    "verifier_group": (
                        "generalist" if index < 4 else "specialist"
                    ),
                    "score": score,
                    "evaluation": "gap" if score < 1.0 else "passes",
                }
                for index, score in enumerate(scores)
            ]
            meta_results = {
                index: [{"score": 1.0}]
                for index, _ in enumerate(scores)
            }
            return run.aggregate_proof_label(
                verifier_results,
                meta_results,
                min_valid_low=2,
                strict_pass_meta=True,
                meta_n=1,
            )

        one_critique = aggregate([1.0, 1.0, 1.0, 0.0])
        two_critiques = aggregate([1.0, 1.0, 0.0, 0.0])

        self.assertGreater(one_critique["final_score"], 0.5)
        self.assertFalse(one_critique["validated_critique_quorum"])
        self.assertFalse(one_critique["validated_low_score_cap_applied"])
        self.assertEqual(two_critiques["final_score"], 0.5)
        self.assertTrue(two_critiques["validated_critique_quorum"])
        self.assertTrue(two_critiques["validated_low_score_cap_applied"])

    def test_rejected_positive_meta_verdicts_become_refinement_critiques(self):
        verifier_results = [
            {
                "verifier_index": index,
                "verifier_role": f"specialist_{index}",
                "verifier_group": "specialist",
                "score": 1.0,
                "evaluation": "The proof passes.",
            }
            for index in range(2)
        ]
        meta_results = {
            0: [
                {
                    "score": 0.0,
                    "analysis": "A legal counterexample breaks the strategy.",
                }
            ],
            1: [
                {
                    "score": 0.5,
                    "analysis": "The decisive monotonicity claim is unchecked.",
                }
            ],
        }

        result = run.aggregate_proof_label(
            verifier_results,
            meta_results,
            min_valid_low=2,
            strict_pass_meta=True,
            meta_n=1,
            audit_positive_meta=True,
        )

        self.assertEqual(len(result["positive_meta_challenges"]), 2)
        self.assertEqual(len(result["validated_critiques"]), 2)
        self.assertEqual(result["final_score"], 0.25)
        self.assertEqual(result["final_status"], "validated_low_score")
        self.assertFalse(result["strict_pass"])

    def test_retention_does_not_double_penalize_positive_meta_challenges(self):
        baseline = {
            "final_score": 0.25,
            "positive_meta_challenges": [],
        }
        improved_refinement = {
            "final_score": 0.5,
            "positive_meta_challenges": [{"score": 0.0}],
        }

        self.assertGreater(
            run.candidate_retention_score(improved_refinement),
            run.candidate_retention_score(baseline),
        )

    def test_retention_preserves_strict_pass_challenge_tiebreak(self):
        baseline = {"final_score": 1.0}
        survived_challenge = {
            "final_score": 1.0,
            "strict_pass_challenge_survived": True,
        }

        self.assertGreater(
            run.candidate_retention_score(survived_challenge),
            run.candidate_retention_score(baseline),
        )

    def test_validated_critique_preserves_requested_fix(self):
        result = run.aggregate_proof_label(
            [
                {
                    "verifier_index": 0,
                    "verifier_role": "transition_closure",
                    "verifier_group": "specialist",
                    "score": 0.0,
                    "evaluation": "The one-step descent is not closed.",
                    "suggestions": "Prove closure under every successor state.",
                }
            ],
            {},
            min_valid_low=1,
            meta_n=0,
        )

        self.assertEqual(
            result["validated_critiques"][0]["suggestions"],
            "Prove closure under every successor state.",
        )

    def test_prover_uses_trained_system_user_prompt(self):
        messages = run.build_opd_proof_generation_prompt("Prove the claim.")

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("mathematical proof generator", messages[0]["content"])
        self.assertIn(
            "Do not repeat yourself or brute-force a solution.",
            messages[0]["content"],
        )
        self.assertIn("Problem:\nProve the claim.", messages[1]["content"])
        self.assertIn("<self_evaluation>", messages[1]["content"])

    def test_baseline_proof_strategy_preserves_trained_prompt_exactly(self):
        implicit = run.build_opd_proof_generation_prompt("Prove the claim.")
        explicit = run.build_opd_proof_generation_prompt(
            "Prove the claim.",
            planning_strategy="baseline",
        )

        self.assertEqual(implicit, explicit)
        self.assertNotIn(
            "<internal_planning_emphasis>",
            implicit[-1]["content"],
        )

    def test_diverse_proof_strategy_cycle_keeps_half_baseline(self):
        cfg = SimpleNamespace(proof_generation_strategy_portfolio="diverse")

        strategies = [
            run.resolve_proof_generation_strategy(index, cfg)
            for index in range(8)
        ]

        self.assertEqual(
            strategies,
            list(run.PROOF_GENERATION_STRATEGY_CYCLE),
        )
        self.assertEqual(strategies.count("baseline"), 4)

    def test_adaptive_game_portfolio_targets_adversarial_history(self):
        cfg = SimpleNamespace(proof_generation_strategy_portfolio="adaptive")
        question = (
            "Alice and Bazza play a game. Determine which player has a "
            "winning strategy."
        )

        strategies = [
            run.resolve_proof_generation_strategy(index, cfg, question)
            for index in range(12)
        ]

        self.assertEqual(strategies, list(run.ADAPTIVE_GAME_STRATEGY_CYCLE))
        self.assertEqual(strategies.count("baseline"), 2)
        self.assertEqual(strategies.count("adversarial_quantifiers"), 3)
        self.assertEqual(strategies.count("joint_state_inequality"), 3)
        self.assertEqual(strategies.count("proof_obligation_ledger"), 2)
        self.assertEqual(strategies.count("game_regime_completeness"), 1)

    def test_adaptive_iteration_portfolio_targets_transition_closure(self):
        cfg = SimpleNamespace(proof_generation_strategy_portfolio="adaptive")
        question = "The sequence satisfies a_{n+1}=f(a_n). Determine a_1."

        strategies = [
            run.resolve_proof_generation_strategy(index, cfg, question)
            for index in range(12)
        ]

        self.assertEqual(
            strategies,
            list(run.ADAPTIVE_ITERATION_STRATEGY_CYCLE),
        )
        self.assertEqual(strategies.count("baseline"), 3)
        self.assertEqual(strategies.count("exhaustive_transitions"), 3)
        self.assertEqual(strategies.count("state_invariant"), 2)

    def test_adaptive_imo2025_p4_portfolio_targets_complete_orbit_normal_form(self):
        cfg = SimpleNamespace(proof_generation_strategy_portfolio="adaptive")
        question = (
            "The infinite sequence $a_1,a_2,\\ldots$ has at least three proper "
            "divisors per term. Each next term is the sum of the three largest "
            "proper divisors. Determine all possible values of $a_1$."
        )

        strategies = [
            run.resolve_proof_generation_strategy(index, cfg, question)
            for index in range(12)
        ]

        self.assertEqual(strategies, list(run.ADAPTIVE_IMO2025_P4_STRATEGY_CYCLE))
        self.assertEqual(strategies.count("baseline"), 2)
        self.assertEqual(strategies.count("p4_orbit_normal_form"), 4)
        self.assertEqual(strategies.count("p4_backward_divisibility"), 2)
        self.assertEqual(strategies.count("p4_transition_classification"), 2)

    def test_adaptive_imo2025_p5_portfolio_targets_both_player_strategies(self):
        cfg = SimpleNamespace(proof_generation_strategy_portfolio="adaptive")
        question = (
            "Alice and Bazza play the inekoalaty game depending on a positive "
            "real number $\\lambda$. Determine both players' winning regimes."
        )

        strategies = [
            run.resolve_proof_generation_strategy(index, cfg, question)
            for index in range(12)
        ]

        self.assertEqual(strategies, list(run.ADAPTIVE_IMO2025_P5_STRATEGY_CYCLE))
        self.assertEqual(strategies.count("baseline"), 2)
        self.assertEqual(strategies.count("p5_threshold_pairing"), 4)
        self.assertEqual(strategies.count("p5_alice_cauchy_spike"), 2)
        self.assertEqual(strategies.count("p5_bazza_pairing"), 2)

    def test_adaptive_generic_portfolio_falls_back_to_diverse_cycle(self):
        cfg = SimpleNamespace(proof_generation_strategy_portfolio="adaptive")

        strategies = [
            run.resolve_proof_generation_strategy(
                index,
                cfg,
                "Prove that the three circles are concurrent.",
            )
            for index in range(8)
        ]

        self.assertEqual(strategies, list(run.PROOF_GENERATION_STRATEGY_CYCLE))

    def test_adversarial_strategy_adds_private_quantifier_discipline(self):
        messages = run.build_opd_proof_generation_prompt(
            "Prove the game claim.",
            planning_strategy="adversarial_quantifiers",
        )
        user_prompt = messages[-1]["content"]

        self.assertIn("<internal_planning_emphasis>", user_prompt)
        self.assertIn("full legal history", user_prompt)
        self.assertIn("every legal reply", user_prompt)
        self.assertIn("Respond in EXACTLY this format:", user_prompt)
        self.assertLess(
            user_prompt.index("<internal_planning_emphasis>"),
            user_prompt.index("Respond in EXACTLY this format:"),
        )

    def test_joint_state_strategy_rejects_one_variable_worst_case_shortcuts(self):
        messages = run.build_opd_proof_generation_prompt(
            "Prove the game claim.",
            planning_strategy="joint_state_inequality",
        )
        user_prompt = messages[-1]["content"]

        self.assertIn("every state variable", user_prompt)
        self.assertIn("history-independent worst-case inequality", user_prompt)
        self.assertIn("opponent saturates a budget", user_prompt)

    def test_game_regime_strategy_requires_universal_boundary_play(self):
        messages = run.build_opd_proof_generation_prompt(
            "Classify the winner for every value of the game parameter.",
            planning_strategy="game_regime_completeness",
        )
        user_prompt = messages[-1]["content"]

        self.assertIn("each regime as a separate proof obligation", user_prompt)
        self.assertIn("against every legal opposing history", user_prompt)
        self.assertIn("one cooperative infinite play is insufficient", user_prompt)
        self.assertIn("prevents the other player from winning", user_prompt)

    def test_state_invariant_strategy_requires_transition_closure(self):
        messages = run.build_opd_proof_generation_prompt(
            "Determine the valid initial terms of the recurrence.",
            planning_strategy="state_invariant",
        )
        user_prompt = messages[-1]["content"]

        self.assertIn("one-step increase or decrease is not enough", user_prompt)
        self.assertIn("after every transition", user_prompt)
        self.assertIn("divisibility", user_prompt)

    def test_p4_orbit_strategy_requires_backward_closure_and_finite_growth(self):
        messages = run.build_opd_proof_generation_prompt(
            "Determine all possible initial terms.",
            planning_strategy="p4_orbit_normal_form",
        )
        user_prompt = messages[-1]["content"]

        self.assertIn("backward divisibility", user_prompt)
        self.assertIn("only finitely many", user_prompt)
        self.assertIn("verify that each parameterized value", user_prompt)

    def test_p5_threshold_strategy_requires_arbitrary_play_and_boundary(self):
        messages = run.build_opd_proof_generation_prompt(
            "Classify the winner in the game.",
            planning_strategy="p5_threshold_pairing",
        )
        user_prompt = messages[-1]["content"]

        self.assertIn("arbitrary even moves", user_prompt)
        self.assertIn("sqrt(2-t^2)", user_prompt)
        self.assertIn("non-losing strategy for each player", user_prompt)

    def test_invalid_proof_strategy_portfolio_is_rejected(self):
        cfg = SimpleNamespace(proof_generation_strategy_portfolio="unknown")

        with self.assertRaisesRegex(ValueError, "must be one of"):
            run.resolve_proof_generation_strategy(0, cfg)

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

    def test_generation_parser_ignores_xml_example_before_orphan_think_close(self):
        parsed = run.parse_generation_response(
            "Now I need to output:\n"
            "<solution>The complete rigorous proof.</solution>\n"
            "<self_evaluation>Briefly note fragile steps.</self_evaluation>\n"
            "<score>S</score>\n"
            "I will now write the actual answer.</think>\n"
            "<solution>The actual rigorous proof.</solution>\n"
            "<self_evaluation>The actual proof is checked.</self_evaluation>\n"
            "<score>1</score>"
        )

        self.assertTrue(parsed["is_valid_candidate_response"])
        self.assertEqual(parsed["proof"], "The actual rigorous proof.")
        self.assertEqual(
            parsed["self_evaluation"],
            "The actual proof is checked.",
        )
        self.assertEqual(parsed["self_score"], 1.0)

    def test_generation_parser_rejects_incomplete_final_output_after_thinking(self):
        parsed = run.parse_generation_response(
            "<solution>Prompt-format example.</solution>\n"
            "<self_evaluation>Example evaluation.</self_evaluation>\n"
            "<score>1</score>\n"
            "</think>\n"
            "<solution>The actual proof is unfinished"
        )

        self.assertFalse(parsed["is_valid_candidate_response"])
        self.assertEqual(parsed["proof"], "")
        self.assertIsNone(parsed["self_score"])

    def test_deepseek_prompt_and_generation_parser_use_markdown_contract(self):
        prompt = run.build_deepseek_proof_generation_prompt("Prove the claim.")
        parsed = run.parse_deepseek_generation_response(
            "<think>private work</think>\n"
            "## Solution\nA complete proof.\n\n"
            "## Self Evaluation\n"
            "Here is my evaluation of the solution: It is complete.\n"
            "Based on my evaluation, the final overall score should be:\n"
            "\\boxed{1}"
        )

        self.assertIn("## Solution", prompt)
        self.assertIn("## Self Evaluation", prompt)
        self.assertIn("## Problem\nProve the claim.", prompt)
        self.assertTrue(parsed["is_valid_candidate_response"])
        self.assertEqual(parsed["proof"], "A complete proof.")
        self.assertEqual(parsed["self_score"], 1.0)

    def test_deepseek_verifier_parser_normalizes_current_schema(self):
        prompt = run.build_deepseek_proof_verification_prompt(
            "Problem.",
            "Candidate proof.",
        )
        parsed = run.parse_deepseek_verifier_response(
            "<think>private work</think>\n"
            "Here is my evaluation of the solution:\nA fatal gap remains.\n\n"
            "Based on my evaluation, the final overall score should be:\n"
            "\\boxed{0}"
        )

        self.assertIn("## Solution\nCandidate proof.", prompt)
        self.assertTrue(parsed["is_valid_verifier_response"])
        self.assertEqual(parsed["score"], 0.0)
        self.assertEqual(parsed["evaluation"], "A fatal gap remains.")
        self.assertIn("Here is my evaluation", parsed["review"])

    def test_first_six_candidates_use_deepseek_prompt_family(self):
        cfg = SimpleNamespace(deepseek_math_v2_candidate_count=6)

        families = [
            run.resolve_candidate_prompt_family(index, 14, cfg) for index in range(14)
        ]

        self.assertEqual(
            families,
            [run.PROMPT_FAMILY_DEEPSEEK_MATH_V2] * 6 + [run.PROMPT_FAMILY_OPD] * 8,
        )
        all_opd = SimpleNamespace(deepseek_math_v2_candidate_count=0)
        all_deepseek = SimpleNamespace(deepseek_math_v2_candidate_count=14)
        self.assertEqual(
            run.resolve_candidate_prompt_family(0, 14, all_opd),
            run.PROMPT_FAMILY_OPD,
        )
        self.assertEqual(
            run.resolve_candidate_prompt_family(13, 14, all_deepseek),
            run.PROMPT_FAMILY_DEEPSEEK_MATH_V2,
        )
        with self.assertRaisesRegex(ValueError, "between 0"):
            run.resolve_candidate_prompt_family(
                0,
                14,
                SimpleNamespace(deepseek_math_v2_candidate_count=15),
            )

    def test_reasoning_repetition_metrics_use_hidden_reasoning_only(self):
        words = [f"word{index}" for index in range(32)] * 3
        text = (
            "<think>"
            + " ".join(words)
            + "</think><solution>Visible output is excluded.</solution>"
        )

        metrics = run.measure_reasoning_repetition(text)

        self.assertEqual(metrics["word_count"], 96)
        self.assertEqual(metrics["window_words"], 32)
        self.assertEqual(metrics["word_window_count"], 65)
        self.assertEqual(metrics["repeated_word_window_count"], 33)
        self.assertAlmostEqual(metrics["repeated_word_window_fraction"], 33 / 65)
        self.assertGreater(metrics["gzip_factor"], 1.0)

    def test_reasoning_repetition_supports_orphan_think_close(self):
        metrics = run.measure_reasoning_repetition(
            "private hidden work</think><solution>Visible.</solution>"
        )

        self.assertEqual(metrics["word_count"], 3)
        self.assertEqual(metrics["word_window_count"], 0)
        self.assertEqual(metrics["repeated_word_window_fraction"], 0.0)

    def test_proof_output_carries_reasoning_repetition_metrics(self):
        response = {
            "stage": "proof_generation",
            "success": True,
            "text": "<think>private work</think><solution>Proof.</solution>",
        }
        response["reasoning_repetition"] = run.measure_reasoning_repetition(
            response["text"]
        )

        output = run.make_output("proof_generation", response, {})

        self.assertEqual(output["reasoning_repetition"]["word_count"], 2)

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

    def test_verifier_parser_ignores_xml_example_before_orphan_think_close(self):
        parsed = run.parse_verifier_response(
            "<evaluation>Example evaluation.</evaluation>\n"
            "<suggestions>Example suggestion.</suggestions>\n"
            "<score>1</score>\n"
            "</think>\n"
            "<evaluation>The candidate has a fatal gap.</evaluation>\n"
            "<suggestions>Prove the missing lemma.</suggestions>\n"
            "<score>0</score>"
        )

        self.assertTrue(parsed["is_valid_verifier_response"])
        self.assertEqual(parsed["evaluation"], "The candidate has a fatal gap.")
        self.assertEqual(parsed["suggestions"], "Prove the missing lemma.")
        self.assertEqual(parsed["score"], 0.0)

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
        self.assertIn('<repair id="R1" role="unknown" score="0.5">', user)
        self.assertIn("Minor gap.", user)
        self.assertIn("Fill it.", user)
        self.assertIn("REPAIR_STATUS Rn", user)

    def test_reconstruction_prompt_treats_candidate_as_fallible(self):
        messages = run.build_opd_proof_reconstruction_prompt(
            "Problem.",
            "P5",
            "Candidate proof.",
            "Candidate audit.",
            [
                {
                    "verifier_role": "dependency_lemma",
                    "score": 1.0,
                    "review": "The weakest claim appears plausible.",
                }
            ],
            strict_pass_challenge=True,
        )

        self.assertIn("Re-solve the olympiad problem from first principles", messages[0]["content"])
        self.assertIn("replace it completely", messages[0]["content"])
        self.assertIn("perfect internal score", messages[1]["content"])
        self.assertIn(
            '<verifier_review role="dependency_lemma" score="1">',
            messages[1]["content"],
        )
        self.assertIn(
            '<repair id="R1" role="dependency_lemma" score="1">',
            messages[1]["content"],
        )
        self.assertIn(
            "repeating an unsupported claim",
            messages[0]["content"].lower(),
        )

    def test_refinement_repair_ledger_deduplicates_and_falls_back(self):
        critique = {
            "verifier_role": "transition_closure",
            "score": 0.0,
            "evaluation": "The descent lemma is not closed under iteration.",
            "suggestions": "Prove that every successor remains in the decreasing set.",
        }

        ledger = run.build_refinement_repair_ledger([critique, dict(critique)])
        fallback = run.build_refinement_repair_ledger(
            [{"score": 0.5, "review": "An unsupported boundary claim remains."}]
        )

        self.assertEqual(ledger.count('<repair id="R1"'), 1)
        self.assertNotIn('<repair id="R2"', ledger)
        self.assertIn("every successor remains", ledger)
        self.assertIn("Supply a complete proof", fallback)

    def test_mixed_refinement_strategy_splits_candidates_deterministically(self):
        cfg = SimpleNamespace(refinement_strategy="mixed")

        self.assertEqual(run.resolve_refinement_strategy(cfg, 0), "repair")
        self.assertEqual(run.resolve_refinement_strategy(cfg, 1), "reconstruct")
        self.assertEqual(
            run.resolve_refinement_strategy(
                cfg,
                0,
                strict_pass_challenge=True,
            ),
            "reconstruct",
        )

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

    def test_only_requested_deepseek_prompt_branches_are_retained(self):
        self.assertTrue(hasattr(run, "build_deepseek_proof_generation_prompt"))
        self.assertTrue(hasattr(run, "build_deepseek_proof_verification_prompt"))
        self.assertTrue(hasattr(run, "build_deepseek_meta_verification_prompt"))
        self.assertFalse(hasattr(run, "build_deepseek_gold_proof_evaluation_prompt"))
        self.assertFalse(hasattr(run, "build_proof_architect_prompt"))
        self.assertFalse(hasattr(run, "build_sublemma_prover_prompt"))

    def test_grader_input_uses_full_proofs_and_omits_failed_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "grader_input" / "records.jsonl"
            full_proof = "A" * (run.MAX_SUBMISSION_ANSWER_CHARS + 100)
            run.write_grader_input_records(
                path,
                [
                    {
                        "id": 1,
                        "answer": full_proof[: run.MAX_SUBMISSION_ANSWER_CHARS],
                        "prediction": full_proof,
                        "selected_pipeline": 3,
                        "final_score": 0.75,
                        "final_status": "refined",
                        "error": "",
                    },
                    {
                        "id": 2,
                        "prediction": "Fallback proof",
                        "final_status": "all_attempts_failed",
                        "error": "",
                    },
                ],
            )

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["problem_id"], "1")
            self.assertEqual(records[0]["final_proof"], full_proof)
            self.assertEqual(records[0]["selected_pipeline"], 3)
            self.assertFalse(path.with_suffix(".jsonl.tmp").exists())


class MixedPromptRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_strict_pass_gets_one_shadow_reconstruction_challenge(self):
        class FakeScheduler:
            def __init__(self) -> None:
                self.calls: list[tuple[str, object]] = []
                self.responses = iter(
                    [
                        (
                            "<evaluation>The proof appears complete; weakest claim is L.</evaluation>"
                            "<suggestions>Stress-test L.</suggestions><score>1</score>"
                        ),
                        (
                            "<solution>Independently reconstructed proof.</solution>"
                            "<self_evaluation>All cases rechecked.</self_evaluation>"
                            "<score>1</score>"
                        ),
                        (
                            "<evaluation>The reconstruction proves L independently.</evaluation>"
                            "<suggestions>None.</suggestions><score>1</score>"
                        ),
                    ]
                )

            async def call(self, stage: str, prompt: object, **kwargs: object):
                del kwargs
                self.calls.append((stage, prompt))
                return {
                    "success": True,
                    "error": None,
                    "text": next(self.responses),
                    "finish_reason": "stop",
                    "usage": {},
                    "server_url": "mock",
                    "latency_s": 0.0,
                }

        initial_text = (
            "<solution>Original proof.</solution>"
            "<self_evaluation>Looks complete.</self_evaluation><score>1</score>"
        )
        initial_parsed = run.parse_generation_response(initial_text)
        initial_generation = {
            "attempt_idx": 0,
            "prompt_family": run.PROMPT_FAMILY_OPD,
            "generation_mode": "opd_xml",
            "generation_output": run.make_output(
                "proof_generation",
                {"success": True, "text": initial_text},
                initial_parsed,
            ),
            "generation_parsed": initial_parsed,
            "proof": initial_parsed["proof"],
        }
        cfg = SimpleNamespace(
            verify_n=1,
            meta_n=0,
            meta_policy="all-reviews",
            strict_pass_meta=False,
            refine_rounds=1,
            refine_review_n=1,
            min_valid_low=1,
            verification_early_stop=False,
            thinking_budget_enabled=False,
            verifier_thinking_budget_tokens=0,
            verifier_thinking_budget_force_text="<evaluation>\n",
            deepseek_verifier_thinking_budget_force_text="",
            meta_thinking_budget_tokens=0,
            meta_thinking_budget_force_text="",
            refinement_strategy="mixed",
            strict_pass_challenge_rounds=1,
        )
        scheduler = FakeScheduler()

        result = await run.run_single_attempt(
            "Problem.",
            0,
            1,
            scheduler,
            cfg,
            initial_generation=initial_generation,
        )

        self.assertEqual(
            [stage for stage, _ in scheduler.calls],
            ["proof_verify", "proof_refine", "proof_verify"],
        )
        refinement_prompt = scheduler.calls[1][1]
        self.assertIn("independent mathematical proof researcher", refinement_prompt[0]["content"])
        self.assertEqual(result["strict_pass_challenges_used"], 1)
        self.assertEqual(result["selected_verification_round"], 1)
        self.assertEqual(result["proof_solution"], "Independently reconstructed proof.")
        self.assertTrue(result["proof_refine_output"][0]["strict_pass_challenge"])
        self.assertEqual(
            result["proof_refine_output"][0]["refinement_strategy"],
            "reconstruct",
        )

    async def test_failed_strict_pass_challenge_rolls_back_to_original(self):
        class FakeScheduler:
            def __init__(self) -> None:
                self.responses = iter(
                    [
                        "<evaluation>Accepted.</evaluation><suggestions>None.</suggestions><score>1</score>",
                        (
                            "<solution>Broken reconstruction.</solution>"
                            "<self_evaluation>Uncertain.</self_evaluation><score>0</score>"
                        ),
                        "<evaluation>Fatal gap.</evaluation><suggestions>Restart.</suggestions><score>0</score>",
                    ]
                )

            async def call(self, stage: str, prompt: object, **kwargs: object):
                del stage, prompt, kwargs
                return {
                    "success": True,
                    "error": None,
                    "text": next(self.responses),
                    "finish_reason": "stop",
                    "usage": {},
                    "server_url": "mock",
                    "latency_s": 0.0,
                }

        initial_text = (
            "<solution>Original proof.</solution>"
            "<self_evaluation>Looks complete.</self_evaluation><score>1</score>"
        )
        initial_parsed = run.parse_generation_response(initial_text)
        cfg = SimpleNamespace(
            verify_n=1,
            meta_n=0,
            meta_policy="all-reviews",
            strict_pass_meta=False,
            refine_rounds=1,
            refine_review_n=1,
            min_valid_low=1,
            verification_early_stop=False,
            thinking_budget_enabled=False,
            verifier_thinking_budget_tokens=0,
            verifier_thinking_budget_force_text="<evaluation>\n",
            deepseek_verifier_thinking_budget_force_text="",
            meta_thinking_budget_tokens=0,
            meta_thinking_budget_force_text="",
            refinement_strategy="mixed",
            strict_pass_challenge_rounds=1,
        )

        result = await run.run_single_attempt(
            "Problem.",
            0,
            1,
            FakeScheduler(),
            cfg,
            initial_generation={
                "attempt_idx": 0,
                "prompt_family": run.PROMPT_FAMILY_OPD,
                "generation_mode": "opd_xml",
                "generation_output": run.make_output(
                    "proof_generation",
                    {"success": True, "text": initial_text},
                    initial_parsed,
                ),
                "generation_parsed": initial_parsed,
                "proof": initial_parsed["proof"],
            },
        )

        self.assertEqual(result["proof_solution"], "Original proof.")
        self.assertEqual(result["selected_verification_round"], 0)
        self.assertEqual(result["rollback_from_round"], 1)

    async def test_refinement_preserves_prior_critique_audits(self):
        class FakeScheduler:
            def __init__(self) -> None:
                self.calls: list[tuple[str, object]] = []
                self.responses = iter(
                    [
                        (
                            "<evaluation>Missing lemma alpha.</evaluation>"
                            "<suggestions>Prove alpha.</suggestions><score>0</score>"
                        ),
                        (
                            "<solution>Proof after alpha repair.</solution>"
                            "<self_evaluation>Alpha is addressed.</self_evaluation>"
                            "<score>0.5</score>"
                        ),
                        (
                            "<evaluation>Missing lemma beta.</evaluation>"
                            "<suggestions>Prove beta.</suggestions><score>0</score>"
                        ),
                        (
                            "<solution>Proof after both repairs.</solution>"
                            "<self_evaluation>Both are addressed.</self_evaluation>"
                            "<score>1</score>"
                        ),
                        (
                            "<evaluation>Both lemmas are now proved.</evaluation>"
                            "<suggestions>None.</suggestions><score>1</score>"
                        ),
                    ]
                )

            async def call(self, stage: str, prompt: object, **kwargs: object):
                del kwargs
                self.calls.append((stage, prompt))
                return {
                    "success": True,
                    "error": None,
                    "text": next(self.responses),
                    "finish_reason": "stop",
                    "usage": {},
                    "server_url": "mock",
                    "latency_s": 0.0,
                }

        initial_text = (
            "<solution>Initial proof.</solution>"
            "<self_evaluation>Needs checking.</self_evaluation><score>0.5</score>"
        )
        initial_parsed = run.parse_generation_response(initial_text)
        initial_generation = {
            "attempt_idx": 0,
            "prompt_family": run.PROMPT_FAMILY_OPD,
            "generation_mode": "opd_xml",
            "generation_output": run.make_output(
                "proof_generation",
                {"success": True, "text": initial_text},
                initial_parsed,
            ),
            "generation_parsed": initial_parsed,
            "proof": initial_parsed["proof"],
        }
        cfg = SimpleNamespace(
            verify_n=1,
            meta_n=0,
            meta_policy="all-reviews",
            strict_pass_meta=False,
            refine_rounds=2,
            refine_review_n=1,
            min_valid_low=1,
            verification_early_stop=False,
            thinking_budget_enabled=False,
            verifier_thinking_budget_tokens=0,
            verifier_thinking_budget_force_text="<evaluation>\n",
            deepseek_verifier_thinking_budget_force_text="",
            meta_thinking_budget_tokens=0,
            meta_thinking_budget_force_text="",
        )
        scheduler = FakeScheduler()

        result = await run.run_single_attempt(
            "Problem.",
            0,
            1,
            scheduler,
            cfg,
            initial_generation=initial_generation,
        )

        verifier_prompts = [
            prompt for stage, prompt in scheduler.calls if stage == "proof_verify"
        ]
        self.assertEqual(len(verifier_prompts), 3)
        self.assertNotIn("Missing lemma alpha", verifier_prompts[0][-1]["content"])
        self.assertIn("Missing lemma alpha", verifier_prompts[1][-1]["content"])
        self.assertIn("Missing lemma alpha", verifier_prompts[2][-1]["content"])
        self.assertIn("Missing lemma beta", verifier_prompts[2][-1]["content"])
        self.assertEqual(len(result["critique_history"]), 2)

    async def test_generation_dispatches_prompt_parser_and_force_text_by_family(self):
        class FakeScheduler:
            def __init__(self) -> None:
                self.calls: list[tuple[object, dict[str, object]]] = []

            async def call(self, stage: str, prompt: object, **kwargs: object):
                self.assert_stage(stage)
                self.calls.append((prompt, kwargs))
                if isinstance(prompt, str):
                    text = (
                        "## Solution\nDeepSeek proof.\n\n"
                        "## Self Evaluation\nChecked.\n\\boxed{1}"
                    )
                else:
                    text = (
                        "<solution>OPD proof.</solution>"
                        "<self_evaluation>Checked.</self_evaluation>"
                        "<score>1</score>"
                    )
                return {"success": True, "text": text}

            @staticmethod
            def assert_stage(stage: str) -> None:
                if stage != "proof_generation":
                    raise AssertionError(stage)

        cfg = SimpleNamespace(
            deepseek_math_v2_candidate_count=6,
            thinking_budget_enabled=False,
            proof_generation_thinking_budgets=[],
            default_temperature=0.7,
            proof_generation_temperatures=[],
            deepseek_thinking_budget_force_text="## Solution\n",
            thinking_budget_force_text="<solution>\n",
        )
        scheduler = FakeScheduler()

        deepseek = await run.generate_single_attempt("Problem.", 0, 14, scheduler, cfg)
        opd = await run.generate_single_attempt("Problem.", 6, 14, scheduler, cfg)

        self.assertEqual(
            deepseek["prompt_family"],
            run.PROMPT_FAMILY_DEEPSEEK_MATH_V2,
        )
        self.assertEqual(opd["prompt_family"], run.PROMPT_FAMILY_OPD)
        self.assertIsInstance(scheduler.calls[0][0], str)
        self.assertIsInstance(scheduler.calls[1][0], list)
        self.assertEqual(
            scheduler.calls[0][1]["thinking_budget_force_text"],
            "## Solution\n",
        )
        self.assertEqual(
            scheduler.calls[1][1]["thinking_budget_force_text"],
            "<solution>\n",
        )

    async def test_deepseek_candidate_keeps_deepseek_verifier_after_opd_refine(self):
        class FakeScheduler:
            def __init__(self) -> None:
                self.calls: list[tuple[str, object]] = []
                self.responses = iter(
                    [
                        (
                            "Here is my evaluation of the solution:\n"
                            "A key justification is missing.\n\n"
                            "Based on my evaluation, the final overall score "
                            "should be:\n\\boxed{0}"
                        ),
                        (
                            "<solution>A repaired complete proof.</solution>"
                            "<self_evaluation>The gap is now filled.</self_evaluation>"
                            "<score>1</score>"
                        ),
                        (
                            "Here is my evaluation of the solution:\n"
                            "The repaired proof is complete.\n\n"
                            "Based on my evaluation, the final overall score "
                            "should be:\n\\boxed{1}"
                        ),
                    ]
                )

            async def call(self, stage: str, prompt: object, **kwargs: object):
                del kwargs
                self.calls.append((stage, prompt))
                return {
                    "success": True,
                    "error": None,
                    "text": next(self.responses),
                    "finish_reason": "stop",
                    "usage": {},
                    "server_url": "mock",
                    "latency_s": 0.0,
                }

        initial_text = (
            "## Solution\nAn incomplete proof.\n\n"
            "## Self Evaluation\nHere is my evaluation of the solution: "
            "a gap remains.\n\\boxed{0}"
        )
        initial_parsed = run.parse_deepseek_generation_response(initial_text)
        initial_generation = {
            "attempt_idx": 0,
            "prompt_family": run.PROMPT_FAMILY_DEEPSEEK_MATH_V2,
            "generation_mode": "deepseek_markdown",
            "generation_output": run.make_output(
                "proof_generation",
                {"success": True, "text": initial_text},
                initial_parsed,
                prompt_family=run.PROMPT_FAMILY_DEEPSEEK_MATH_V2,
            ),
            "generation_parsed": initial_parsed,
            "proof": initial_parsed["proof"],
        }
        cfg = SimpleNamespace(
            verify_n=1,
            meta_n=0,
            meta_policy="all-reviews",
            strict_pass_meta=False,
            refine_rounds=1,
            refine_review_n=1,
            min_valid_low=1,
            verification_early_stop=False,
            thinking_budget_enabled=False,
            verifier_thinking_budget_tokens=0,
            verifier_thinking_budget_force_text="<evaluation>\n",
            deepseek_verifier_thinking_budget_force_text=(
                "Here is my evaluation of the solution:\n"
            ),
            meta_thinking_budget_tokens=0,
            meta_thinking_budget_force_text="",
        )
        scheduler = FakeScheduler()

        result = await run.run_single_attempt(
            "Problem.",
            0,
            14,
            scheduler,
            cfg,
            initial_generation=initial_generation,
        )

        self.assertEqual(
            [stage for stage, _ in scheduler.calls],
            ["proof_verify", "proof_refine", "proof_verify"],
        )
        self.assertIsInstance(scheduler.calls[0][1], str)
        self.assertIsInstance(scheduler.calls[1][1], list)
        self.assertIsInstance(scheduler.calls[2][1], str)
        self.assertIn("## Instruction", str(scheduler.calls[0][1]))
        self.assertIn("## Instruction", str(scheduler.calls[2][1]))
        self.assertIn(
            '<candidate id="P0">',
            scheduler.calls[1][1][1]["content"],
        )
        self.assertEqual(
            result["prompt_family"],
            run.PROMPT_FAMILY_DEEPSEEK_MATH_V2,
        )
        self.assertEqual(result["proof_solution"], "A repaired complete proof.")


if __name__ == "__main__":
    unittest.main()
