#!/usr/bin/env bash
set -euo pipefail

# Run our SFT-750 checkpoint through the teammate cumulative multi-parent
# refinement and saturation-gated tournament selector on one assigned NII node.

RUN_ID="${AIMO_RUN_ID:?set AIMO_RUN_ID to a unique run identifier}"
SOURCE_REPO="${AIMO_SOURCE_REPO:-/tmp/aimo-proof-pilot-inference-runtime/repo}"
SOURCE_INPUT="${AIMO_SOURCE_INPUT:-${SOURCE_REPO}/imo-2026.jsonl}"
TEAMMATE_COMMIT="${AIMO_TEAMMATE_COMMIT:-95a35c66eda41184b846a8489314b9e8f8f43b32}"
TEAMMATE_UPSTREAM="${AIMO_TEAMMATE_UPSTREAM:-https://github.com/fieldsmodelorg/AIMO-Proof-Pilot.git}"
TEAMMATE_ROOT="${AIMO_TEAMMATE_ROOT:-/tmp/aimo-proof-pilot-teammate-treatment/${TEAMMATE_COMMIT}}"
TEAMMATE_REPO="${AIMO_TEAMMATE_REPO:-${TEAMMATE_ROOT}/repo}"
VENV="${AIMO_TEAMMATE_VENV:-/tmp/chankhavu/venvs/infervenv}"
ACTIVATE_ENV="${VENV}/.runtime/activate-env.sh"
MODEL_PATH="${AIMO_MODEL_PATH:-/tmp/models/olmo3-opd-sft-750-vllm}"
DFLASH_MODEL_PATH="${AIMO_DFLASH_MODEL_PATH:-/tmp/models/dflash-32b-draft-v2test-phaseL}"
GPUS="${AIMO_GPUS:-4,5,6,7}"
TP_SIZE="${AIMO_TENSOR_PARALLEL_SIZE:-2}"
DP_SIZE="${AIMO_DATA_PARALLEL_SIZE:-2}"
SERVER_PORT="${AIMO_TEAMMATE_SERVER_PORT:-31000}"
PROOFS_PER_ROUND="${AIMO_TEAMMATE_PROOFS_PER_ROUND:-64}"
VERIFICATIONS_PER_PROOF="${AIMO_TEAMMATE_VERIFICATIONS_PER_PROOF:-8}"
TOP_PROOFS="${AIMO_TEAMMATE_TOP_PROOFS:-16}"
REFINE_PARENTS="${AIMO_TEAMMATE_REFINE_PARENTS:-4}"
REVIEWS_PER_PARENT="${AIMO_TEAMMATE_REVIEWS_PER_PARENT:-3}"
MAX_ROUNDS="${AIMO_TEAMMATE_MAX_ROUNDS:-4}"
MAX_RUNNING_REQUESTS="${AIMO_TEAMMATE_MAX_RUNNING_REQUESTS:-32}"
SEARCH_CONCURRENCY="${AIMO_TEAMMATE_SEARCH_CONCURRENCY:-64}"
PLAN_ONLY="${AIMO_TEAMMATE_PLAN_ONLY:-false}"
OUTPUT_ROOT="${AIMO_TEAMMATE_OUTPUT_ROOT:-/tmp/aimo-proof-pilot-teammate-runs}/${RUN_ID}"
INPUT_PATH="${OUTPUT_ROOT}/imo2026-p45.csv"
CONFIG_PATH="${OUTPUT_ROOT}/config.yaml"

for value_name in \
    TP_SIZE DP_SIZE SERVER_PORT PROOFS_PER_ROUND VERIFICATIONS_PER_PROOF \
    TOP_PROOFS REFINE_PARENTS REVIEWS_PER_PARENT MAX_ROUNDS \
    MAX_RUNNING_REQUESTS SEARCH_CONCURRENCY
do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$value_name must be a positive integer, got $value" >&2
        exit 2
    fi
done
if [ "$PLAN_ONLY" != "true" ] && [ "$PLAN_ONLY" != "false" ]; then
    echo "AIMO_TEAMMATE_PLAN_ONLY must be true or false, got $PLAN_ONLY" >&2
    exit 2
fi
IFS=',' read -r -a gpu_ids <<< "$GPUS"
if [ "${#gpu_ids[@]}" -ne $((TP_SIZE * DP_SIZE)) ]; then
    echo "AIMO_GPUS count must equal TP_SIZE x DP_SIZE" >&2
    exit 2
fi
if [ "$TOP_PROOFS" -gt "$PROOFS_PER_ROUND" ]; then
    echo "TOP_PROOFS cannot exceed PROOFS_PER_ROUND" >&2
    exit 2
fi
if [ "$REFINE_PARENTS" -gt "$TOP_PROOFS" ]; then
    echo "REFINE_PARENTS cannot exceed TOP_PROOFS" >&2
    exit 2
fi
if [ "$REVIEWS_PER_PARENT" -gt "$VERIFICATIONS_PER_PROOF" ]; then
    echo "REVIEWS_PER_PARENT cannot exceed VERIFICATIONS_PER_PROOF" >&2
    exit 2
fi
for path in "$SOURCE_INPUT" "$SOURCE_REPO/evaluation/prepare_imo2026_p45.py" \
    "$SOURCE_REPO/evaluation/prepare_teammate_treatment.py" \
    "$ACTIVATE_ENV" "$VENV/bin/python" "$MODEL_PATH" "$DFLASH_MODEL_PATH"
do
    if [ ! -e "$path" ]; then
        echo "Missing required path: $path" >&2
        exit 3
    fi
done

mkdir -p "$TEAMMATE_ROOT" "$OUTPUT_ROOT"
if [ ! -d "$TEAMMATE_REPO/.git" ]; then
    temporary_repo="${TEAMMATE_ROOT}/.repo.tmp.$$"
    cleanup() { rm -rf "$temporary_repo"; }
    trap cleanup EXIT
    git clone --filter=blob:none --no-checkout "$TEAMMATE_UPSTREAM" "$temporary_repo"
    git -C "$temporary_repo" checkout --detach "$TEAMMATE_COMMIT"
    mv "$temporary_repo" "$TEAMMATE_REPO"
    trap - EXIT
fi
actual_commit="$(git -C "$TEAMMATE_REPO" rev-parse HEAD)"
if [ "$actual_commit" != "$TEAMMATE_COMMIT" ]; then
    echo "Teammate checkout mismatch: expected $TEAMMATE_COMMIT, got $actual_commit" >&2
    exit 3
fi

"$VENV/bin/python" "$SOURCE_REPO/evaluation/prepare_imo2026_p45.py" \
    --input "$SOURCE_INPUT" \
    --csv-output "$INPUT_PATH"

"$VENV/bin/python" "$SOURCE_REPO/evaluation/prepare_teammate_treatment.py" \
    --output "$CONFIG_PATH" \
    --model-path "$MODEL_PATH" \
    --draft-path "$DFLASH_MODEL_PATH" \
    --tensor-parallel-size "$TP_SIZE" \
    --data-parallel-size "$DP_SIZE" \
    --server-port "$SERVER_PORT" \
    --proofs-per-round "$PROOFS_PER_ROUND" \
    --verifications-per-proof "$VERIFICATIONS_PER_PROOF" \
    --top-proofs "$TOP_PROOFS" \
    --refine-parents "$REFINE_PARENTS" \
    --reviews-per-parent "$REVIEWS_PER_PARENT" \
    --max-rounds "$MAX_ROUNDS" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --search-concurrency "$SEARCH_CONCURRENCY"

# The node-local SGLang runtime redirects all writable caches and exposes its
# patched CUDA/Python libraries through this activation script.
# shellcheck disable=SC1090
source "$ACTIVATE_ENV"
export VENV
export RUNTIME_ROOT="$VENV/.runtime"
export PYTHON="$VENV/bin/python"
export CUDA_VISIBLE_DEVICES="$GPUS"

echo "run_id=$RUN_ID"
echo "teammate_commit=$TEAMMATE_COMMIT"
echo "model=$MODEL_PATH draft=$DFLASH_MODEL_PATH"
echo "gpus=$GPUS tp=$TP_SIZE dp=$DP_SIZE port=$SERVER_PORT"
echo "search=proofs:${PROOFS_PER_ROUND} verifiers:${VERIFICATIONS_PER_PROOF} top:${TOP_PROOFS} parents:${REFINE_PARENTS} reviews:${REVIEWS_PER_PARENT} rounds:${MAX_ROUNDS}"
echo "input=$INPUT_PATH config=$CONFIG_PATH output=$OUTPUT_ROOT"

if [ "$PLAN_ONLY" = "true" ]; then
    exec "$TEAMMATE_REPO/scheduler.sh" --plan \
        "$CONFIG_PATH" "$OUTPUT_ROOT" "$INPUT_PATH"
fi
exec "$TEAMMATE_REPO/scheduler.sh" \
    "$CONFIG_PATH" "$OUTPUT_ROOT" "$INPUT_PATH"
