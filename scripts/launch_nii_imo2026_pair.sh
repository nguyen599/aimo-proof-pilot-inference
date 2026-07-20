#!/usr/bin/env bash
set -euo pipefail

# Run the proven IMO search configuration on physical NII nodes 2 and 3. The
# shared launcher maps them to distributed ranks 0 and 1 and starts TP2/DP4
# vLLM locally on each node when port 8000 is not already healthy.

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

export AIMO_NII_NODE_RANK="$logical_rank"
export AIMO_WORLD_SIZE=2
export AIMO_INPUT_PATH="${AIMO_INPUT_PATH:-${SOURCE_REPO}/imo-2026.jsonl}"
export AIMO_MAX_CONCURRENT_PROBLEMS="${AIMO_MAX_CONCURRENT_PROBLEMS:-6}"
export AIMO_REQUESTS_PER_GPU="${AIMO_REQUESTS_PER_GPU:-32}"
export AIMO_MAX_NUM_SEQS_PER_DP="${AIMO_MAX_NUM_SEQS_PER_DP:-32}"
export AIMO_VERIFY_CANDIDATE_LIMIT_WHILE_GENERATING="${AIMO_VERIFY_CANDIDATE_LIMIT_WHILE_GENERATING:-0}"
export AIMO_VERIFY_REQUEST_LIMIT_WHILE_GENERATING="${AIMO_VERIFY_REQUEST_LIMIT_WHILE_GENERATING:-0}"
export AIMO_VERIFY_N="${AIMO_VERIFY_N:-8}"
export AIMO_VERIFIER_GENERALIST_N="${AIMO_VERIFIER_GENERALIST_N:-4}"
export AIMO_REFINE_REVIEW_N="${AIMO_REFINE_REVIEW_N:-4}"
export AIMO_MIN_VALID_LOW="${AIMO_MIN_VALID_LOW:-2}"

exec "$SCRIPT_DIR/launch_nii_imo2025_all.sh"
