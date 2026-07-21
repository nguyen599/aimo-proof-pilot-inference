#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPO="${AIMO_SOURCE_REPO:-/tmp/aimo-proof-pilot-inference-runtime/repo}"
physical_rank="${GLOBAL_RANK:-${NODE_RANK:-}}"

case "$physical_rank" in
    2) logical_rank=0 ;;
    3) logical_rank=1 ;;
    *)
        echo "Expected physical NII rank 2 or 3, got ${physical_rank:-unset}" >&2
        exit 2
        ;;
esac

export AIMO_RUN_ID="${AIMO_RUN_ID:-imo2025-p45-adaptive-round0-p36-sft750-20260722}"
export AIMO_SOURCE_REF="${AIMO_SOURCE_REF:-main}"
export AIMO_NII_NODE_RANK="$logical_rank"
export AIMO_WORLD_SIZE=2
export MASTER_PORT="${MASTER_PORT:-29751}"
export AIMO_INPUT_PATH="${AIMO_INPUT_PATH:-${SOURCE_REPO}/evaluation/runs/imo2025-p45-adaptive-round0-p36-sft750-20260722/input.jsonl}"
export AIMO_MAX_CONCURRENT_PROBLEMS=2
export AIMO_REQUESTS_PER_GPU=32
export AIMO_MAX_NUM_SEQS_PER_DP=32
export AIMO_PIPELINES_PER_PROBLEM=36
export AIMO_REFINE_ROUNDS=0
export AIMO_PROOF_GENERATION_ONLY=true
export AIMO_PROOF_GENERATION_STRATEGY_PORTFOLIO=adaptive
export AIMO_THINKING_BUDGET_HANDOFF_ENABLED=true
export AIMO_SELECTOR_MODE=score

exec "${SOURCE_REPO}/scripts/launch_nii_imo2025_all.sh"
