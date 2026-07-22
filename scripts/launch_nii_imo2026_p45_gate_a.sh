#!/usr/bin/env bash
set -euo pipefail

# Measure whether the round-zero pool contains a correct P4/P5 proof before
# spending inference on verification, refinement, or selection.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPO="${AIMO_SOURCE_REPO:-/tmp/aimo-proof-pilot-inference-runtime/repo}"
VENV="${AIMO_VENV:-/tmp/aimo-proof-pilot-inference-runtime/venv-vllm-0.25.1}"
SOURCE_INPUT="${AIMO_SOURCE_INPUT:-${SOURCE_REPO}/imo-2026.jsonl}"
INPUT_DIR="${AIMO_INPUT_DIR:-/tmp/aimo-proof-pilot-inference-inputs}"
P45_INPUT="${INPUT_DIR}/imo2026-p45.jsonl"

if [ ! -x "$VENV/bin/python" ]; then
    echo "Missing vLLM environment: $VENV" >&2
    exit 3
fi
if [ ! -f "$SOURCE_INPUT" ]; then
    echo "Missing IMO 2026 input: $SOURCE_INPUT" >&2
    exit 3
fi

mkdir -p "$INPUT_DIR"
"$VENV/bin/python" "$SOURCE_REPO/evaluation/prepare_imo2026_p45.py" \
    --input "$SOURCE_INPUT" \
    --jsonl-output "$P45_INPUT"

export AIMO_NII_NODE_RANK=0
export AIMO_WORLD_SIZE=1
export AIMO_INPUT_PATH="$P45_INPUT"
export AIMO_GPUS="${AIMO_GPUS:-4,5,6,7}"
export AIMO_NUM_GPUS="${AIMO_NUM_GPUS:-4}"
export AIMO_TENSOR_PARALLEL_SIZE="${AIMO_TENSOR_PARALLEL_SIZE:-2}"
export AIMO_DATA_PARALLEL_SIZE="${AIMO_DATA_PARALLEL_SIZE:-2}"
export AIMO_MAX_NUM_SEQS_PER_DP="${AIMO_MAX_NUM_SEQS_PER_DP:-32}"
export AIMO_REQUESTS_PER_GPU="${AIMO_REQUESTS_PER_GPU:-32}"
export AIMO_MAX_CONCURRENT_PROBLEMS="${AIMO_MAX_CONCURRENT_PROBLEMS:-2}"
export AIMO_PIPELINES_PER_PROBLEM="${AIMO_PIPELINES_PER_PROBLEM:-64}"
export AIMO_PROOF_GENERATION_STRATEGY_PORTFOLIO=baseline
export AIMO_PROOF_GENERATION_ONLY=true
export AIMO_THINKING_BUDGET_HANDOFF_ENABLED=false
export AIMO_REFINE_ROUNDS=0
export AIMO_STRICT_PASS_CHALLENGE_ROUNDS=0
export AIMO_SELECTOR_MODE=score

exec "$SCRIPT_DIR/launch_nii_imo2025_all.sh"
