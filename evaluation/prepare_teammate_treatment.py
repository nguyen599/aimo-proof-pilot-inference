#!/usr/bin/env python3
"""Build the pinned teammate-harness config for the IMO 2026 P4/P5 treatment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml


def build_config(
    *,
    model_path: Path,
    draft_path: Path,
    tensor_parallel_size: int,
    data_parallel_size: int,
    server_port: int,
    proofs_per_round: int,
    verifications_per_proof: int,
    top_proofs: int,
    refine_parents: int,
    reviews_per_parent: int,
    max_rounds: int,
    max_running_requests: int,
    search_concurrency: int,
) -> dict[str, Any]:
    positive_values = {
        "tensor_parallel_size": tensor_parallel_size,
        "data_parallel_size": data_parallel_size,
        "server_port": server_port,
        "proofs_per_round": proofs_per_round,
        "verifications_per_proof": verifications_per_proof,
        "top_proofs": top_proofs,
        "refine_parents": refine_parents,
        "reviews_per_parent": reviews_per_parent,
        "max_rounds": max_rounds,
        "max_running_requests": max_running_requests,
        "search_concurrency": search_concurrency,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if top_proofs > proofs_per_round:
        raise ValueError("top_proofs cannot exceed proofs_per_round")
    if refine_parents > top_proofs:
        raise ValueError("refine_parents cannot exceed top_proofs")
    if reviews_per_parent > verifications_per_proof:
        raise ValueError("reviews_per_parent cannot exceed verifications_per_proof")

    return {
        "schema_version": 12,
        "models": {
            "bf16_target": str(model_path),
            "quantized_target": str(model_path),
            "bf16_draft": str(draft_path),
            "quantized_draft": str(draft_path),
        },
        "model": {
            "tensor_parallel_size": tensor_parallel_size,
            "data_parallel_size": data_parallel_size,
            "quantized": False,
            "dflash": True,
            "kv_cache_dtype": "auto",
        },
        "server": {
            "attention_backend": "fa3",
            "page_size": 1,
            "deterministic_inference": False,
            "host": "127.0.0.1",
            "port": server_port,
            "context_length": 262144,
            "mem_fraction_static": 0.82,
            "max_running_requests": max_running_requests,
            "swa_full_tokens_ratio": 0.2,
            "chunked_prefill_size": 2048,
            "stream_interval": 16,
            "watchdog_timeout": 1200,
            "prefill_cuda_graph_backend": "tc_piecewise",
            "dflash_block_size": 8,
            "dflash_num_draft_tokens": 8,
            "dflash_window_size": 512,
        },
        "search": {
            "proofs_per_round": proofs_per_round,
            "verifications_per_proof": verifications_per_proof,
            "top_proofs": top_proofs,
            "refine_parents": refine_parents,
            "reviews_per_refine_parent": reviews_per_parent,
            "refine_review_strategy": "random_nonideal",
            "max_rounds": max_rounds,
            # The teammate harness stops only for score > threshold. Exactly 1.0
            # therefore forces every configured round while remaining schema-valid.
            "early_stop_threshold": 1.0,
            "temperature": 1.0,
            "top_p": 0.95,
            "max_completion_tokens": 128000,
            "solution_continuation_tokens": 16384,
            "verifier_continuation_tokens": 16384,
            "min_valid_verifications": min(4, verifications_per_proof),
            "verifier_sees_self_evaluation": True,
            "refiner_sees_self_evaluation": False,
            "lenient_parsing": True,
            "filter_degenerate": True,
            "stream_detect": True,
            "request_timeout_seconds": 86400,
            "concurrency": search_concurrency,
            "seed": 0,
            "llm_selector": True,
            "selection_votes": 16,
            "selection_candidates": 4,
            "selection_max_tokens": 56000,
            "selection_continuation_tokens": 2048,
            "selection_tournament": True,
            "selection_tournament_threshold": 0.95,
            "selection_tournament_rounds": 64,
            "selection_tournament_max_candidates": 10,
            "selection_score_window": 0.2,
        },
    }


def write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--draft-path", required=True, type=Path)
    parser.add_argument("--tensor-parallel-size", required=True, type=int)
    parser.add_argument("--data-parallel-size", required=True, type=int)
    parser.add_argument("--server-port", required=True, type=int)
    parser.add_argument("--proofs-per-round", required=True, type=int)
    parser.add_argument("--verifications-per-proof", required=True, type=int)
    parser.add_argument("--top-proofs", required=True, type=int)
    parser.add_argument("--refine-parents", required=True, type=int)
    parser.add_argument("--reviews-per-parent", required=True, type=int)
    parser.add_argument("--max-rounds", required=True, type=int)
    parser.add_argument("--max-running-requests", required=True, type=int)
    parser.add_argument("--search-concurrency", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_config(
        args.output,
        build_config(
            model_path=args.model_path,
            draft_path=args.draft_path,
            tensor_parallel_size=args.tensor_parallel_size,
            data_parallel_size=args.data_parallel_size,
            server_port=args.server_port,
            proofs_per_round=args.proofs_per_round,
            verifications_per_proof=args.verifications_per_proof,
            top_proofs=args.top_proofs,
            refine_parents=args.refine_parents,
            reviews_per_parent=args.reviews_per_parent,
            max_rounds=args.max_rounds,
            max_running_requests=args.max_running_requests,
            search_concurrency=args.search_concurrency,
        ),
    )


if __name__ == "__main__":
    main()
