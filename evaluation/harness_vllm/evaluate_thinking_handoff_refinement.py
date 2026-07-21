from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from transformers import AutoTokenizer

try:
    from evaluation.harness_vllm.run import (
        CFG,
        PROMPT_FAMILY_OPD,
        RESTART_FINALIZE_FORCE_TEXT,
        ChatScheduler,
        SamplingConfig,
        build_opd_proof_refinement_prompt,
        append_final_output_discipline,
        format_refinement_critique,
        make_output,
        parse_generation_response,
        run_verification_round,
        run_single_attempt,
    )
    from evaluation.harness_vllm.thinking_handoff import (
        append_restart_instruction,
        extract_rendered_problem_text,
        parse_saved_proof_generation_call,
    )
except ModuleNotFoundError as exc:
    if exc.name != "evaluation":
        raise
    from run import (  # type: ignore[no-redef]
        CFG,
        PROMPT_FAMILY_OPD,
        RESTART_FINALIZE_FORCE_TEXT,
        ChatScheduler,
        SamplingConfig,
        build_opd_proof_refinement_prompt,
        append_final_output_discipline,
        format_refinement_critique,
        make_output,
        parse_generation_response,
        run_verification_round,
        run_single_attempt,
    )
    from thinking_handoff import (  # type: ignore[no-redef]
        append_restart_instruction,
        extract_rendered_problem_text,
        parse_saved_proof_generation_call,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a saved parser-valid thinking-budget restart through the "
            "verifier, meta-verifier, and proof-refinement stages."
        )
    )
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--restart-results", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--served-model-name", default="proof-model")
    parser.add_argument("--api-key", default="vllm-local")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-index", type=int, default=0)
    parser.add_argument(
        "--resume-refinement-result",
        type=Path,
        help=(
            "Reuse the verifier critiques and lossless refinement handoff from "
            "a previous result.json, then run only the final refinement and "
            "its verification."
        ),
    )
    parser.add_argument("--verify-n", type=int, default=4)
    parser.add_argument(
        "--verifier-generalist-n",
        type=int,
        help=(
            "Use this many unmodified generalist verifier prompts before "
            "specialist prompts. Omit it to preserve the legacy all-specialist "
            "replay behavior."
        ),
    )
    parser.add_argument("--meta-n", type=int, default=1)
    parser.add_argument(
        "--meta-policy",
        choices=("all-reviews", "low-only"),
        default="low-only",
    )
    parser.add_argument("--refine-review-n", type=int, default=2)
    parser.add_argument("--refine-rounds", type=int, default=1)
    parser.add_argument("--min-valid-low", type=int, default=1)
    parser.add_argument(
        "--strict-pass-meta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require every verifier critique to pass meta review for a strict pass.",
    )
    parser.add_argument("--proof-max-tokens", type=int, default=65_000)
    parser.add_argument("--verifier-max-tokens", type=int, default=32_000)
    parser.add_argument("--meta-max-tokens", type=int, default=32_000)
    parser.add_argument(
        "--thinking-budget-refine-handoff-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--thinking-budget-refine-tokens", type=int, default=50_000)
    parser.add_argument(
        "--thinking-budget-refine-final-round-tokens",
        type=int,
        default=50_000,
    )
    parser.add_argument("--thinking-budget-refine-max-restarts", type=int, default=1)
    parser.add_argument(
        "--thinking-budget-refine-final-temperature",
        type=float,
        help=(
            "Sampling temperature used only for the final refinement after a "
            "thinking-budget handoff. By default it inherits --temperature."
        ),
    )
    parser.add_argument(
        "--thinking-budget-refine-visible-output-target-tokens",
        type=int,
        default=0,
        help=(
            "Add an explicit visible-answer token target and XML closure "
            "instruction to the final post-handoff refinement."
        ),
    )
    parser.add_argument(
        "--thinking-budget-refine-visible-output-limit-tokens",
        type=int,
        default=0,
        help=(
            "Hard-limit post-intervention visible tokens in the final "
            "post-handoff refinement. If the response remains incomplete, "
            "append an honest score-0 XML closure."
        ),
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--request-timeout-seconds", type=float, default=7200.0)
    return parser.parse_args()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_result(path: Path, index: int) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"restart result file is empty: {path}")
    if not 0 <= index < len(rows):
        raise IndexError(f"result index {index} is outside 0..{len(rows) - 1}")
    return rows[index]


def normalize_problem(rendered_prompt: str) -> str:
    problem = extract_rendered_problem_text(rendered_prompt)
    if problem.startswith("Problem:\n"):
        problem = problem.removeprefix("Problem:\n").strip()
    return problem


def score_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


async def resume_final_refinement(
    *,
    path: Path,
    question: str,
    initial_parsed: dict[str, Any],
    scheduler: ChatScheduler,
    cfg: SimpleNamespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    previous = json.loads(path.read_text(encoding="utf-8"))
    candidate = copy.deepcopy(previous["candidate"])
    handoffs = candidate.get("proof_refine_handoffs") or []
    if not handoffs or not handoffs[-1].get("text"):
        raise ValueError(f"resume result has no refinement handoff: {path}")
    ranked_critiques = sorted(
        candidate.get("validated_critiques") or [],
        key=lambda critique: (
            score_value(critique.get("score")),
            int(critique.get("verifier_index") or 0),
        ),
    )
    selected_critiques = ranked_critiques[: max(1, int(cfg.refine_review_n))]
    if not selected_critiques:
        raise ValueError(f"resume result has no validated critiques: {path}")

    base_prompt = build_opd_proof_refinement_prompt(
        question,
        "P0",
        str(initial_parsed.get("proof") or ""),
        str(initial_parsed.get("self_evaluation") or ""),
        selected_critiques,
    )
    prompt = append_restart_instruction(
        base_prompt,
        str(handoffs[-1]["text"]),
        1,
        strategy="deadline_aware",
    )
    if cfg.thinking_budget_refine_visible_output_target_tokens > 0:
        prompt = append_final_output_discipline(
            prompt,
            cfg.thinking_budget_refine_visible_output_target_tokens,
        )
    response = await scheduler.call(
        "proof_refine",
        prompt,
        temperature=cfg.thinking_budget_refine_final_temperature,
        detail="candidate=0 round=1 resumed_handoff=1",
        thinking_budget_tokens=int(
            cfg.thinking_budget_refine_final_round_tokens
        ),
        thinking_budget_force_text=RESTART_FINALIZE_FORCE_TEXT,
        thinking_budget_action="finalize",
        visible_output_limit_tokens=(
            cfg.thinking_budget_refine_visible_output_limit_tokens or None
        ),
    )
    parsed = parse_generation_response(response.get("text", ""))
    critiques = [
        format_refinement_critique(critique) for critique in selected_critiques
    ]
    output = make_output(
        "proof_refine",
        response,
        parsed,
        round_idx=1,
        used_critiques=critiques,
        resumed_handoff=True,
        invalid=not parsed["is_valid_candidate_response"],
        prompt_family=PROMPT_FAMILY_OPD,
    )
    candidate.setdefault("proof_refine_output", []).append(output)
    candidate["resumed_refinement_output"] = output
    resume_details: dict[str, Any] = {
        "source_result": str(path),
        "selected_critiques": selected_critiques,
        "refinement_output": output,
        "verification_ran": False,
    }
    if not parsed["is_valid_candidate_response"]:
        return candidate, resume_details

    (
        _verifier_results,
        verifier_outputs,
        _meta_results_by_verifier,
        meta_outputs,
        aggregation,
    ) = await run_verification_round(
        question,
        str(parsed["proof"]),
        str(parsed.get("self_evaluation") or ""),
        0,
        1,
        scheduler,
        cfg,
        prompt_family=PROMPT_FAMILY_OPD,
    )
    candidate.setdefault("proof_verify_output", []).extend(verifier_outputs)
    candidate.setdefault("proof_meta_verify_output", []).extend(meta_outputs)
    resume_details["verification_ran"] = True
    resume_details["aggregation"] = aggregation

    if score_value(aggregation.get("final_score")) >= score_value(
        candidate.get("final_score")
    ):
        candidate.update(
            {
                "proof_solution": parsed["proof"],
                "self_evaluation": parsed.get("self_evaluation"),
                "self_score": parsed.get("self_score"),
                "validated_critiques": aggregation.get(
                    "validated_critiques", []
                ),
                "verifier_score_summaries": aggregation.get(
                    "verifier_score_summaries", []
                ),
                "final_score": aggregation.get("final_score"),
                "final_status": aggregation.get("final_status"),
                "low_scores_seen": aggregation.get("low_scores_seen"),
                "strict_pass": aggregation.get("strict_pass", False),
                "all_verifiers_passed": aggregation.get(
                    "all_verifiers_passed", False
                ),
                "meta_valid_count": aggregation.get("meta_valid_count", 0),
                "meta_checked_count": aggregation.get("meta_checked_count", 0),
                "meta_summary_by_verifier": aggregation.get(
                    "meta_summary_by_verifier", {}
                ),
                "selected_verification_round": 1,
                "rollback_from_round": None,
            }
        )
    else:
        candidate["rollback_from_round"] = 1
    return candidate, resume_details


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    restart = load_result(args.restart_results, args.result_index)
    if restart.get("error"):
        raise ValueError(f"saved restart contains an error: {restart['error']}")
    source = str(restart["source"])
    record = parse_saved_proof_generation_call(args.logs_root / source)
    question = normalize_problem(record.input_prompt)
    raw_output = str(restart.get("raw_output") or "")
    parsed = parse_generation_response(raw_output, require_self_evaluation=True)
    if not parsed["is_valid_candidate_response"]:
        raise ValueError("saved restart is not a parser-valid OPD response")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
    )
    scheduler = ChatScheduler(
        base_urls=[str(url) for url in args.base_url],
        api_key=args.api_key,
        model=args.served_model_name,
        sampling=SamplingConfig(
            max_new_tokens=args.proof_max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=-1,
            min_new_tokens=0,
            min_p=None,
        ),
        max_concurrent_requests=max(
            1,
            args.verify_n + (args.verify_n * args.meta_n),
        ),
        stage_max_new_tokens={
            "proof_refine": args.proof_max_tokens,
            "proof_verify": args.verifier_max_tokens,
            "proof_meta_verify": args.meta_max_tokens,
        },
        request_timeout_seconds=args.request_timeout_seconds,
        stream_responses=True,
        context_length=262_144,
        tokenizer=tokenizer,
        stream_interval_tokens=100,
    )
    response = {
        "success": True,
        "error": None,
        "text": raw_output,
        "finish_reason": restart.get("finish_reason"),
        "usage": {
            "completion_tokens": int(restart.get("completion_tokens") or 0),
            "estimated_prompt_tokens": int(restart.get("prompt_tokens") or 0),
        },
        "server_url": restart.get("base_url"),
        "latency_s": restart.get("latency_s"),
    }
    generation_output = make_output(
        "proof_generation",
        response,
        parsed,
        round_idx=1,
        prompt_family=PROMPT_FAMILY_OPD,
        replayed_restart=True,
    )
    initial_generation = {
        "attempt_idx": 0,
        "prompt_family": PROMPT_FAMILY_OPD,
        "generation_mode": "opd_xml",
        "generation_output": generation_output,
        "generation_outputs": [generation_output],
        "handoff_outputs": [],
        "handoffs": [],
        "budget_restart_count": 1,
        "consumed_refine_rounds": 0,
        "generation_parsed": parsed,
        "proof": parsed["proof"],
    }
    verify_n = max(1, int(args.verify_n))
    verifier_generalist_n = args.verifier_generalist_n
    if verifier_generalist_n is not None:
        verifier_generalist_n = int(verifier_generalist_n)
        if not 0 <= verifier_generalist_n <= verify_n:
            raise ValueError(
                "verifier_generalist_n must be between 0 and verify_n"
            )

    cfg = SimpleNamespace(
        proof_max_new_tokens=max(1, int(args.proof_max_tokens)),
        default_temperature=float(args.temperature),
        proof_generation_temperatures=[],
        verify_n=verify_n,
        verifier_generalist_n=verifier_generalist_n,
        meta_n=max(0, int(args.meta_n)),
        meta_policy=str(args.meta_policy),
        strict_pass_meta=bool(args.strict_pass_meta),
        refine_rounds=max(0, int(args.refine_rounds)),
        refine_review_n=max(1, int(args.refine_review_n)),
        min_valid_low=max(1, int(args.min_valid_low)),
        verification_early_stop=False,
        thinking_budget_enabled=True,
        thinking_budget_handoff_max_tokens=int(
            CFG.thinking_budget_handoff_max_tokens
        ),
        thinking_budget_handoff_temperature=float(
            CFG.thinking_budget_handoff_temperature
        ),
        thinking_budget_handoff_prompt_variant=str(
            CFG.thinking_budget_handoff_prompt_variant
        ),
        thinking_budget_handoff_mode="lossless_partial",
        thinking_budget_restart_strategy="deadline_aware",
        thinking_budget_refine_handoff_enabled=bool(
            args.thinking_budget_refine_handoff_enabled
        ),
        thinking_budget_refine_tokens=max(
            0,
            int(args.thinking_budget_refine_tokens),
        ),
        thinking_budget_refine_final_round_tokens=max(
            0,
            int(args.thinking_budget_refine_final_round_tokens),
        ),
        thinking_budget_refine_max_restarts=max(
            0,
            int(args.thinking_budget_refine_max_restarts),
        ),
        thinking_budget_refine_final_temperature=(
            None
            if args.thinking_budget_refine_final_temperature is None
            else float(args.thinking_budget_refine_final_temperature)
        ),
        thinking_budget_refine_visible_output_target_tokens=max(
            0,
            int(args.thinking_budget_refine_visible_output_target_tokens),
        ),
        thinking_budget_refine_visible_output_limit_tokens=max(
            0,
            int(args.thinking_budget_refine_visible_output_limit_tokens),
        ),
        verifier_thinking_budget_tokens=min(
            int(CFG.verifier_thinking_budget_tokens),
            max(1, int(args.verifier_max_tokens) - 1),
        ),
        verifier_thinking_budget_force_text=CFG.verifier_thinking_budget_force_text,
        deepseek_verifier_thinking_budget_force_text=(
            CFG.deepseek_verifier_thinking_budget_force_text
        ),
        meta_thinking_budget_tokens=min(
            int(CFG.meta_thinking_budget_tokens),
            max(1, int(args.meta_max_tokens) - 1),
        ),
        meta_thinking_budget_force_text=CFG.meta_thinking_budget_force_text,
    )
    resume_details = None
    try:
        if args.resume_refinement_result is not None:
            candidate, resume_details = await resume_final_refinement(
                path=args.resume_refinement_result,
                question=question,
                initial_parsed=parsed,
                scheduler=scheduler,
                cfg=cfg,
            )
        else:
            candidate = await run_single_attempt(
                question,
                0,
                1,
                scheduler,
                cfg,
                initial_generation=initial_generation,
            )
    finally:
        scheduler.close()
    return {
        "source": source,
        "restart_results": str(args.restart_results),
        "question": question,
        "settings": {
            "verify_n": cfg.verify_n,
            "verifier_generalist_n": cfg.verifier_generalist_n,
            "meta_n": cfg.meta_n,
            "meta_policy": cfg.meta_policy,
            "strict_pass_meta": cfg.strict_pass_meta,
            "refine_rounds": cfg.refine_rounds,
            "refine_review_n": cfg.refine_review_n,
            "min_valid_low": cfg.min_valid_low,
            "proof_max_tokens": args.proof_max_tokens,
            "verifier_max_tokens": args.verifier_max_tokens,
            "meta_max_tokens": args.meta_max_tokens,
            "thinking_budget_refine_handoff_enabled": (
                cfg.thinking_budget_refine_handoff_enabled
            ),
            "thinking_budget_refine_tokens": cfg.thinking_budget_refine_tokens,
            "thinking_budget_refine_final_round_tokens": (
                cfg.thinking_budget_refine_final_round_tokens
            ),
            "thinking_budget_refine_max_restarts": (
                cfg.thinking_budget_refine_max_restarts
            ),
            "thinking_budget_refine_final_temperature": (
                cfg.thinking_budget_refine_final_temperature
            ),
            "thinking_budget_refine_visible_output_target_tokens": (
                cfg.thinking_budget_refine_visible_output_target_tokens
            ),
            "thinking_budget_refine_visible_output_limit_tokens": (
                cfg.thinking_budget_refine_visible_output_limit_tokens
            ),
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "initial_restart": {
            "completion_tokens": restart.get("completion_tokens"),
            "finish_reason": restart.get("finish_reason"),
            "proof_chars": len(str(parsed.get("proof") or "")),
            "self_score": parsed.get("self_score"),
        },
        "resume_refinement": resume_details,
        "candidate": candidate,
    }


def main() -> None:
    args = parse_args()
    result = asyncio.run(evaluate(args))
    atomic_write_json(args.output_dir / "result.json", result)
    summary = {
        "source": result["source"],
        "initial_restart": result["initial_restart"],
        "final_score": result["candidate"].get("final_score"),
        "final_status": result["candidate"].get("final_status"),
        "proof_chars": len(str(result["candidate"].get("proof_solution") or "")),
        "self_score": result["candidate"].get("self_score"),
        "verifier_calls": len(result["candidate"].get("proof_verify_output") or []),
        "meta_calls": len(result["candidate"].get("proof_meta_verify_output") or []),
        "refine_calls": len(result["candidate"].get("proof_refine_output") or []),
        "refine_budget_restarts": result["candidate"].get(
            "refine_budget_restart_count"
        ),
        "refine_handoff_calls": len(
            result["candidate"].get("proof_refine_handoff_output") or []
        ),
        "selected_verification_round": result["candidate"].get(
            "selected_verification_round"
        ),
        "rollback_from_round": result["candidate"].get("rollback_from_round"),
        "resumed_refinement": result.get("resume_refinement") is not None,
        "resumed_refinement_valid": (
            (
                result.get("resume_refinement", {})
                .get("refinement_output", {})
                .get("parsed", {})
                .get("is_valid_candidate_response")
            )
            if result.get("resume_refinement")
            else None
        ),
        "resumed_refinement_verified": (
            result.get("resume_refinement", {}).get("verification_ran")
            if result.get("resume_refinement")
            else None
        ),
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
