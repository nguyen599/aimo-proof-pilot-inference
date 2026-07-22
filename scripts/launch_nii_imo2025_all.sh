#!/usr/bin/env bash
set -euo pipefail

# Launch one distributed controller and one local TP2/DP4 vLLM server on each
# selected NII node. The default is the four-node "all" allocation; set
# AIMO_NII_NODE_RANK=0 and AIMO_WORLD_SIZE=1 for a single member node.

RUN_ID="${AIMO_RUN_ID:?set AIMO_RUN_ID to a unique run identifier}"
SOURCE_REPO="${AIMO_SOURCE_REPO:-/tmp/aimo-proof-pilot-inference-runtime/repo}"
SOURCE_REF="${AIMO_SOURCE_REF:-main}"
VENV="${AIMO_VENV:-/tmp/aimo-proof-pilot-inference-runtime/venv-vllm-0.25.1}"
MODEL_PATH="${AIMO_MODEL_PATH:-/tmp/models/olmo3-opd-sft-750-vllm}"
DFLASH_MODEL_PATH="${AIMO_DFLASH_MODEL_PATH:-/tmp/models/dflash-32b-draft-v2test-phaseL}"
DIST_ROOT="${AIMO_DISTRIBUTED_ROOT:-/tmp/aimo-proof-pilot-inference-distributed}"
LAUNCH_ROOT="${AIMO_LAUNCH_ROOT:-/tmp/aimo-proof-pilot-inference-launch}/${RUN_ID}"
MASTER_PORT="${MASTER_PORT:-29617}"
MAX_CONCURRENT_PROBLEMS="${AIMO_MAX_CONCURRENT_PROBLEMS:-6}"
TP_SIZE="${AIMO_TENSOR_PARALLEL_SIZE:-2}"
DP_SIZE="${AIMO_DATA_PARALLEL_SIZE:-4}"
GPUS="${AIMO_GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a gpu_ids <<< "$GPUS"
NUM_GPUS="${AIMO_NUM_GPUS:-${#gpu_ids[@]}}"
# vLLM applies max_num_seqs independently to every DP engine replica.
MAX_NUM_SEQS_PER_DP="${AIMO_MAX_NUM_SEQS_PER_DP:-32}"
REQUESTS_PER_GPU="${AIMO_REQUESTS_PER_GPU:-32}"
PIPELINES_PER_PROBLEM="${AIMO_PIPELINES_PER_PROBLEM:-36}"
REFINE_ROUNDS="${AIMO_REFINE_ROUNDS:-4}"
PROOF_GENERATION_ONLY="${AIMO_PROOF_GENERATION_ONLY:-false}"
PROOF_GENERATION_STRATEGY_PORTFOLIO="${AIMO_PROOF_GENERATION_STRATEGY_PORTFOLIO:-baseline}"
THINKING_BUDGET_HANDOFF_ENABLED="${AIMO_THINKING_BUDGET_HANDOFF_ENABLED:-true}"
SELECTOR_MODE="${AIMO_SELECTOR_MODE:-llm}"
SELECTOR_MAX_NEW_TOKENS="${AIMO_SELECTOR_MAX_NEW_TOKENS:-50000}"
SELECTOR_THINKING_BUDGET_TOKENS="${AIMO_SELECTOR_THINKING_BUDGET_TOKENS:-0}"
SELECTOR_CANDIDATE_LIMIT="${AIMO_SELECTOR_CANDIDATE_LIMIT:-0}"
SELECTOR_HISTORICAL_CANDIDATE_LIMIT="${AIMO_SELECTOR_HISTORICAL_CANDIDATE_LIMIT:-0}"
SELECTOR_TOURNAMENT_GROUP_SIZE="${AIMO_SELECTOR_TOURNAMENT_GROUP_SIZE:-8}"
SELECTOR_TOURNAMENT_ROUNDS="${AIMO_SELECTOR_TOURNAMENT_ROUNDS:-64}"
SELECTOR_TOURNAMENT_MAX_CANDIDATES="${AIMO_SELECTOR_TOURNAMENT_MAX_CANDIDATES:-10}"
SELECTOR_TOURNAMENT_THRESHOLD="${AIMO_SELECTOR_TOURNAMENT_THRESHOLD:-0.95}"
SELECTOR_TOURNAMENT_FORCE_WIDE_POOL="${AIMO_SELECTOR_TOURNAMENT_FORCE_WIDE_POOL:-false}"
SELECTOR_SCORE_WINDOW="${AIMO_SELECTOR_SCORE_WINDOW:-0.2}"
SELECTOR_VOTE_COUNT="${AIMO_SELECTOR_VOTE_COUNT:-16}"
SELECTOR_MIN_FINAL_SCORE="${AIMO_SELECTOR_MIN_FINAL_SCORE:-0.5}"
SELECTION_TEMPERATURE="${AIMO_SELECTION_TEMPERATURE:-1.0}"
VERIFY_CANDIDATE_LIMIT_WHILE_GENERATING="${AIMO_VERIFY_CANDIDATE_LIMIT_WHILE_GENERATING:-0}"
VERIFY_REQUEST_LIMIT_WHILE_GENERATING="${AIMO_VERIFY_REQUEST_LIMIT_WHILE_GENERATING:-0}"
VERIFY_N="${AIMO_VERIFY_N:-8}"
VERIFIER_GENERALIST_N="${AIMO_VERIFIER_GENERALIST_N:-4}"
REFINE_REVIEW_N="${AIMO_REFINE_REVIEW_N:-4}"
MIN_VALID_LOW="${AIMO_MIN_VALID_LOW:-2}"
REFINE_THINKING_BUDGET="${AIMO_THINKING_BUDGET_REFINE_TOKENS:-120000}"
REFINEMENT_STRATEGY="${AIMO_REFINEMENT_STRATEGY:-mixed}"
STRICT_PASS_CHALLENGE_ROUNDS="${AIMO_STRICT_PASS_CHALLENGE_ROUNDS:-1}"

for value_name in \
    TP_SIZE \
    DP_SIZE \
    MAX_NUM_SEQS_PER_DP \
    REQUESTS_PER_GPU \
    PIPELINES_PER_PROBLEM \
    VERIFY_N \
    REFINE_REVIEW_N \
    MIN_VALID_LOW \
    REFINE_THINKING_BUDGET \
    SELECTOR_MAX_NEW_TOKENS
do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$value_name must be a positive integer, got $value" >&2
        exit 2
    fi
done
for value_name in \
    REFINE_ROUNDS \
    STRICT_PASS_CHALLENGE_ROUNDS \
    SELECTOR_THINKING_BUDGET_TOKENS
do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "$value_name must be a nonnegative integer, got $value" >&2
        exit 2
    fi
done
if [ "$SELECTOR_THINKING_BUDGET_TOKENS" -gt 0 ] && \
   [ "$SELECTOR_THINKING_BUDGET_TOKENS" -ge "$SELECTOR_MAX_NEW_TOKENS" ]; then
    echo "SELECTOR_THINKING_BUDGET_TOKENS must be below SELECTOR_MAX_NEW_TOKENS" >&2
    exit 2
fi
for value_name in \
    PROOF_GENERATION_ONLY \
    THINKING_BUDGET_HANDOFF_ENABLED \
    SELECTOR_TOURNAMENT_FORCE_WIDE_POOL
do
    value="${!value_name}"
    if [ "$value" != "true" ] && [ "$value" != "false" ]; then
        echo "$value_name must be true or false, got $value" >&2
        exit 2
    fi
done
if [ "$SELECTOR_MODE" != "llm" ] && \
   [ "$SELECTOR_MODE" != "llm_tournament" ] && \
   [ "$SELECTOR_MODE" != "llm_stratified_tournament" ] && \
   [ "$SELECTOR_MODE" != "score" ]; then
    echo "SELECTOR_MODE must be llm, llm_tournament, llm_stratified_tournament, or score, got $SELECTOR_MODE" >&2
    exit 2
fi
case "$PROOF_GENERATION_STRATEGY_PORTFOLIO" in
    baseline|diverse|adaptive) ;;
    *)
        echo "PROOF_GENERATION_STRATEGY_PORTFOLIO must be baseline, diverse, or adaptive, got $PROOF_GENERATION_STRATEGY_PORTFOLIO" >&2
        exit 2
        ;;
esac
case "$REFINEMENT_STRATEGY" in
    repair|reconstruct|mixed) ;;
    *)
        echo "REFINEMENT_STRATEGY must be repair, reconstruct, or mixed; got $REFINEMENT_STRATEGY" >&2
        exit 2
        ;;
esac
for value_name in \
    VERIFY_CANDIDATE_LIMIT_WHILE_GENERATING \
    VERIFY_REQUEST_LIMIT_WHILE_GENERATING \
    VERIFIER_GENERALIST_N \
    SELECTOR_CANDIDATE_LIMIT \
    SELECTOR_HISTORICAL_CANDIDATE_LIMIT \
    SELECTOR_TOURNAMENT_GROUP_SIZE \
    SELECTOR_TOURNAMENT_ROUNDS \
    SELECTOR_TOURNAMENT_MAX_CANDIDATES \
    SELECTOR_VOTE_COUNT
do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "$value_name must be a nonnegative integer, got $value" >&2
        exit 2
    fi
done
if { [ "$SELECTOR_MODE" = "llm_tournament" ] || \
     [ "$SELECTOR_MODE" = "llm_stratified_tournament" ]; } && \
   [ "$SELECTOR_TOURNAMENT_GROUP_SIZE" -lt 2 ]; then
    echo "SELECTOR_TOURNAMENT_GROUP_SIZE must be at least 2 for tournament selectors" >&2
    exit 2
fi
if ! [[ "$SELECTOR_MIN_FINAL_SCORE" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]]; then
    echo "SELECTOR_MIN_FINAL_SCORE must be between 0 and 1, got $SELECTOR_MIN_FINAL_SCORE" >&2
    exit 2
fi
if ! [[ "$SELECTOR_TOURNAMENT_THRESHOLD" =~ ^(0[.][0-9]*[1-9][0-9]*|1([.]0+)?)$ ]]; then
    echo "SELECTOR_TOURNAMENT_THRESHOLD must be in (0, 1], got $SELECTOR_TOURNAMENT_THRESHOLD" >&2
    exit 2
fi
if ! [[ "$SELECTOR_SCORE_WINDOW" =~ ^(0([.][0-9]+)?)$ ]]; then
    echo "SELECTOR_SCORE_WINDOW must be in [0, 1), got $SELECTOR_SCORE_WINDOW" >&2
    exit 2
fi
if ! [[ "$SELECTION_TEMPERATURE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "SELECTION_TEMPERATURE must be nonnegative, got $SELECTION_TEMPERATURE" >&2
    exit 2
fi
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_GPUS must be a positive integer, got $NUM_GPUS" >&2
    exit 2
fi
if [ "${#gpu_ids[@]}" -ne "$NUM_GPUS" ]; then
    echo "AIMO_GPUS must contain NUM_GPUS entries: ${#gpu_ids[@]} != $NUM_GPUS" >&2
    exit 2
fi
if [ $((TP_SIZE * DP_SIZE)) -ne "$NUM_GPUS" ]; then
    echo "TP_SIZE x DP_SIZE must equal NUM_GPUS: $TP_SIZE x $DP_SIZE != $NUM_GPUS" >&2
    exit 2
fi
if [ "$REFINE_REVIEW_N" -gt "$VERIFY_N" ]; then
    echo "REFINE_REVIEW_N cannot exceed VERIFY_N" >&2
    exit 2
fi
if [ "$VERIFIER_GENERALIST_N" -gt "$VERIFY_N" ]; then
    echo "VERIFIER_GENERALIST_N cannot exceed VERIFY_N" >&2
    exit 2
fi

physical_rank="${GLOBAL_RANK:-${NODE_RANK:-}}"
case "$physical_rank" in
    0|1|2|3|"") ;;
    *)
        echo "Expected NII rank 0 through 3, got ${physical_rank:-unset}" >&2
        exit 2
        ;;
esac

node_rank="${AIMO_NII_NODE_RANK:-$physical_rank}"
world_size="${AIMO_WORLD_SIZE:-4}"
if ! [[ "$node_rank" =~ ^[0-9]+$ ]]; then
    echo "Expected a nonnegative logical node rank, got ${node_rank:-unset}" >&2
    exit 2
fi
if ! [[ "$world_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "Expected a positive world size, got ${world_size:-unset}" >&2
    exit 2
fi
if [ "$node_rank" -ge "$world_size" ]; then
    echo "Logical node rank $node_rank must be smaller than world size $world_size" >&2
    exit 2
fi

mkdir -p "$LAUNCH_ROOT"
rank_log="$LAUNCH_ROOT/rank_${node_rank}.log"
rank_status="$LAUNCH_ROOT/rank_${node_rank}.status"
rank_command="$LAUNCH_ROOT/rank_${node_rank}.command.sh"
ready_file="$LAUNCH_ROOT/launch.env"
code_dir="$LAUNCH_ROOT/repo"

exec > >(tee -a "$rank_log") 2>&1
rm -f "$rank_status"
on_exit() {
    rc=$?
    printf '%s\n' "$rc" > "$rank_status"
    echo "[$(date -u +%FT%TZ)] rank=${node_rank} exited rc=${rc}"
}
trap on_exit EXIT

echo "[$(date -u +%FT%TZ)] launch rank=${node_rank}/${world_size} physical_rank=${physical_rank:-unknown} host=$(hostname)"

if [ "$node_rank" -eq 0 ]; then
    if ! git -C "$SOURCE_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "Missing source checkout: $SOURCE_REPO" >&2
        exit 3
    fi
    if [ ! -x "$VENV/bin/python" ]; then
        echo "Missing vLLM environment: $VENV" >&2
        exit 3
    fi
    if [ ! -d "$MODEL_PATH" ] || [ ! -d "$DFLASH_MODEL_PATH" ]; then
        echo "Missing model path: model=$MODEL_PATH dflash=$DFLASH_MODEL_PATH" >&2
        exit 3
    fi

    input_path="${AIMO_INPUT_PATH:-}"
    if [ -z "$input_path" ]; then
        for candidate in \
            /tmp/aimo-proof-pilot-inference/test.csv \
            /tmp/aimo-proof-pilot-inference-runtime/repo/test.csv \
            /tmp/aimo-proof-pilot-inference-runtime/test.csv \
            /tmp/data/imo_2025.parquet \
            /tmp/data/imo_2025.csv
        do
            if [ -f "$candidate" ]; then
                input_path="$candidate"
                break
            fi
        done
    fi
    if [ -z "$input_path" ]; then
        input_path="$($VENV/bin/python - "$DIST_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "runs"
for manifest in sorted(root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
    try:
        metadata = json.loads(manifest.read_text()).get("metadata", {})
    except (OSError, ValueError):
        continue
    for key in ("input_csv", "input_path"):
        value = metadata.get(key)
        if value and Path(value).is_file():
            print(value)
            raise SystemExit
PY
)"
    fi
    if [ -z "$input_path" ] || [ ! -f "$input_path" ]; then
        echo "Could not locate the full IMO 2025 input; set AIMO_INPUT_PATH" >&2
        exit 3
    fi

    rm -rf "$code_dir.tmp"
    if [ ! -d "$code_dir/.git" ]; then
        git clone --shared --no-checkout "$SOURCE_REPO" "$code_dir.tmp"
        mv "$code_dir.tmp" "$code_dir"
    fi
    for attempt in 1 2 3 4 5; do
        git -C "$SOURCE_REPO" fetch origin "$SOURCE_REF" && break
        if [ "$attempt" -eq 5 ]; then
            echo "Failed to fetch source ref $SOURCE_REF" >&2
            exit 3
        fi
        sleep $((attempt * 3))
    done
    source_commit="$(git -C "$SOURCE_REPO" rev-parse FETCH_HEAD)"
    git -C "$code_dir" fetch "$SOURCE_REPO" "$source_commit"
    git -C "$code_dir" checkout --detach --force FETCH_HEAD

    master_addr="$($VENV/bin/python - <<'PY'
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("10.0.0.1", 9))
    print(s.getsockname()[0])
finally:
    s.close()
PY
)"
    tmp_ready="${ready_file}.tmp.$$"
    {
        printf 'MASTER_ADDR=%q\n' "$master_addr"
        printf 'AIMO_INPUT_PATH=%q\n' "$input_path"
        printf 'AIMO_CODE_DIR=%q\n' "$code_dir"
        printf 'AIMO_SOURCE_COMMIT=%q\n' "$(git -C "$code_dir" rev-parse HEAD)"
    } > "$tmp_ready"
    mv "$tmp_ready" "$ready_file"
else
    deadline=$((SECONDS + 900))
    while [ ! -s "$ready_file" ]; do
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "Timed out waiting for rank 0 launch metadata: $ready_file" >&2
            exit 3
        fi
        sleep 2
    done
fi

# shellcheck disable=SC1090
source "$ready_file"
export PATH="$VENV/bin:$PATH"
export VLLM_PLUGINS=olmo3_sink
unset AIMO_LOGDIR AIMO_OUTPUT_PATH

vllm_extra_args="$($VENV/bin/python - "$DFLASH_MODEL_PATH" <<'PY'
import json
import shlex
import sys

speculative_config = {
    "method": "dflash",
    "model": sys.argv[1],
    "num_speculative_tokens": 10,
    "disable_above_context_len": 65536,
}
args = [
    "--generation-config", "vllm",
    "--quantization", "fp8",
    "--kv-cache-dtype", "fp8",
    "--block-size", "256",
    "--uvicorn-log-level", "warning",
    "--max-num-batched-tokens", "16384",
    "--speculative-config", json.dumps(speculative_config, separators=(",", ":")),
    "--disable-custom-all-reduce",
]
print(shlex.join(args))
PY
)"

if curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null; then
    echo "Reusing healthy local vLLM endpoint on port 8000"
else
    echo "Local vLLM endpoint is down; this controller will start TP2/DP4 vLLM"
fi

args=(
    "$VENV/bin/python" -m evaluation.harness_vllm.run
    --node-rank "$node_rank"
    --world-size "$world_size"
    --master-addr "$MASTER_ADDR"
    --master-port "$MASTER_PORT"
    --distributed-run-id "$RUN_ID"
    --distributed-root "$DIST_ROOT"
    --model-path "$MODEL_PATH"
    --dflash-model-path "$DFLASH_MODEL_PATH"
    --input-path "$AIMO_INPUT_PATH"
    --num-gpus "$NUM_GPUS"
    --gpus "$GPUS"
    --tensor-parallel-size "$TP_SIZE"
    --data-parallel-size "$DP_SIZE"
    --gpu-memory-utilization 0.92
    --max-num-seqs "$MAX_NUM_SEQS_PER_DP"
    --requests-per-gpu "$REQUESTS_PER_GPU"
    --vllm-extra-args "$vllm_extra_args"
    --served-model-name proof-model
    --server-timeout 7200
    --pipelines-per-problem "$PIPELINES_PER_PROBLEM"
    --proof-generation-strategy-portfolio "$PROOF_GENERATION_STRATEGY_PORTFOLIO"
    --max-concurrent-problems "$MAX_CONCURRENT_PROBLEMS"
    --verify-candidate-limit-while-generating "$VERIFY_CANDIDATE_LIMIT_WHILE_GENERATING"
    --verify-request-limit-while-generating "$VERIFY_REQUEST_LIMIT_WHILE_GENERATING"
    --verify-n "$VERIFY_N"
    --verifier-generalist-n "$VERIFIER_GENERALIST_N"
    --refine-review-n "$REFINE_REVIEW_N"
    --min-valid-low "$MIN_VALID_LOW"
    --refine-rounds "$REFINE_ROUNDS"
    --refinement-strategy "$REFINEMENT_STRATEGY"
    --strict-pass-challenge-rounds "$STRICT_PASS_CHALLENGE_ROUNDS"
    --selector-mode "$SELECTOR_MODE"
    --selector-max-new-tokens "$SELECTOR_MAX_NEW_TOKENS"
    --selector-thinking-budget-tokens "$SELECTOR_THINKING_BUDGET_TOKENS"
    --selector-candidate-limit "$SELECTOR_CANDIDATE_LIMIT"
    --selector-historical-candidate-limit "$SELECTOR_HISTORICAL_CANDIDATE_LIMIT"
    --selector-tournament-group-size "$SELECTOR_TOURNAMENT_GROUP_SIZE"
    --selector-tournament-rounds "$SELECTOR_TOURNAMENT_ROUNDS"
    --selector-tournament-max-candidates "$SELECTOR_TOURNAMENT_MAX_CANDIDATES"
    --selector-tournament-threshold "$SELECTOR_TOURNAMENT_THRESHOLD"
    --selector-score-window "$SELECTOR_SCORE_WINDOW"
    --selector-vote-count "$SELECTOR_VOTE_COUNT"
    --selector-min-final-score "$SELECTOR_MIN_FINAL_SCORE"
    --selection-temperature "$SELECTION_TEMPERATURE"
)

if [ "$SELECTOR_TOURNAMENT_FORCE_WIDE_POOL" = "true" ]; then
    args+=(--selector-tournament-force-wide-pool)
else
    args+=(--no-selector-tournament-force-wide-pool)
fi

if [ "$PROOF_GENERATION_ONLY" = "true" ]; then
    args+=(--proof-generation-only)
fi
if [ "$THINKING_BUDGET_HANDOFF_ENABLED" = "true" ]; then
    args+=(
        --thinking-budget-handoff-enabled
        --thinking-budget-handoff-mode lossless_partial
        --thinking-budget-handoff-preserve-refine-rounds
        --thinking-budget-restart-strategy deadline_aware
        --thinking-budget-restart-until-complete
        --thinking-budget-final-round-tokens 0
        --thinking-budget-refine-handoff-enabled
        --thinking-budget-refine-tokens "$REFINE_THINKING_BUDGET"
        --thinking-budget-refine-final-round-tokens "$REFINE_THINKING_BUDGET"
        --thinking-budget-refine-max-restarts 1
    )
else
    args+=(--no-thinking-budget-handoff-enabled)
fi

printf '#!/usr/bin/env bash\nexec ' > "$rank_command"
printf '%q ' "${args[@]}" >> "$rank_command"
printf '\n' >> "$rank_command"
chmod 0755 "$rank_command"

echo "source_commit=$AIMO_SOURCE_COMMIT"
echo "input=$AIMO_INPUT_PATH"
echo "master=$MASTER_ADDR:$MASTER_PORT"
echo "vllm_capacity=gpus:${GPUS} tp${TP_SIZE}/dp${DP_SIZE} max_num_seqs_per_dp=${MAX_NUM_SEQS_PER_DP} aggregate_max_num_seqs=$((DP_SIZE * MAX_NUM_SEQS_PER_DP)) request_admission=$((NUM_GPUS * REQUESTS_PER_GPU))"
echo "pipeline=candidates:${PIPELINES_PER_PROBLEM} proof_generation_strategy_portfolio:${PROOF_GENERATION_STRATEGY_PORTFOLIO} refine_rounds:${REFINE_ROUNDS} refinement_strategy:${REFINEMENT_STRATEGY} strict_pass_challenges:${STRICT_PASS_CHALLENGE_ROUNDS} generation_only:${PROOF_GENERATION_ONLY} handoff:${THINKING_BUDGET_HANDOFF_ENABLED} selector:${SELECTOR_MODE} selector_max_new_tokens:${SELECTOR_MAX_NEW_TOKENS} selector_thinking_budget_tokens:${SELECTOR_THINKING_BUDGET_TOKENS} selector_candidate_limit:${SELECTOR_CANDIDATE_LIMIT} selector_historical_candidate_limit:${SELECTOR_HISTORICAL_CANDIDATE_LIMIT} selector_tournament_group_size:${SELECTOR_TOURNAMENT_GROUP_SIZE} selector_tournament_rounds:${SELECTOR_TOURNAMENT_ROUNDS} selector_tournament_max_candidates:${SELECTOR_TOURNAMENT_MAX_CANDIDATES} selector_tournament_threshold:${SELECTOR_TOURNAMENT_THRESHOLD} selector_tournament_force_wide_pool:${SELECTOR_TOURNAMENT_FORCE_WIDE_POOL} selector_score_window:${SELECTOR_SCORE_WINDOW} selector_vote_count:${SELECTOR_VOTE_COUNT} selector_temperature:${SELECTION_TEMPERATURE} selector_min_final_score:${SELECTOR_MIN_FINAL_SCORE}"
echo "verification_while_generating=candidates:${VERIFY_CANDIDATE_LIMIT_WHILE_GENERATING} requests:${VERIFY_REQUEST_LIMIT_WHILE_GENERATING} per_problem_per_rank"
echo "verification_per_candidate=verify_n:${VERIFY_N} generalists:${VERIFIER_GENERALIST_N} specialists:$((VERIFY_N - VERIFIER_GENERALIST_N)) refine_review_n:${REFINE_REVIEW_N} min_valid_low:${MIN_VALID_LOW}"
echo "command_file=$rank_command"
cd "$AIMO_CODE_DIR"
"${args[@]}"
