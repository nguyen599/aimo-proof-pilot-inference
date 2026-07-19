#!/usr/bin/env bash
set -euo pipefail

# Launch one distributed controller and one local TP2/DP4 vLLM server on each
# selected NII node. The default is the four-node "all" allocation; set
# AIMO_NII_NODE_RANK=0 and AIMO_WORLD_SIZE=1 for a single member node.

RUN_ID="${AIMO_RUN_ID:?set AIMO_RUN_ID to a unique run identifier}"
SOURCE_REPO="${AIMO_SOURCE_REPO:-/tmp/aimo-proof-pilot-inference-runtime/repo}"
SOURCE_REF="main"
VENV="${AIMO_VENV:-/tmp/aimo-proof-pilot-inference-runtime/venv-vllm-0.25.1}"
MODEL_PATH="${AIMO_MODEL_PATH:-/tmp/models/olmo3-opd-sft-750-vllm}"
DFLASH_MODEL_PATH="${AIMO_DFLASH_MODEL_PATH:-/tmp/models/dflash-32b-draft-v2test-phaseL}"
DIST_ROOT="${AIMO_DISTRIBUTED_ROOT:-/tmp/aimo-proof-pilot-inference-distributed}"
LAUNCH_ROOT="${AIMO_LAUNCH_ROOT:-/tmp/aimo-proof-pilot-inference-launch}/${RUN_ID}"
MASTER_PORT="${MASTER_PORT:-29617}"
MAX_CONCURRENT_PROBLEMS="${AIMO_MAX_CONCURRENT_PROBLEMS:-6}"

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
    if [ ! -d "$SOURCE_REPO/.git" ]; then
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
    git -C "$code_dir" fetch \
        "$SOURCE_REPO" "refs/remotes/origin/${SOURCE_REF}"
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
    --num-gpus 8
    --gpus 0,1,2,3,4,5,6,7
    --tensor-parallel-size 2
    --data-parallel-size 4
    --gpu-memory-utilization 0.92
    --max-num-seqs 32
    --requests-per-gpu 32
    --vllm-extra-args "$vllm_extra_args"
    --served-model-name proof-model
    --server-timeout 7200
    --pipelines-per-problem 36
    --max-concurrent-problems "$MAX_CONCURRENT_PROBLEMS"
    --refine-rounds 4
    --thinking-budget-handoff-enabled
    --thinking-budget-handoff-mode lossless_partial
    --thinking-budget-handoff-preserve-refine-rounds
    --thinking-budget-restart-strategy deadline_aware
    --thinking-budget-final-round-tokens 0
)

printf '#!/usr/bin/env bash\nexec ' > "$rank_command"
printf '%q ' "${args[@]}" >> "$rank_command"
printf '\n' >> "$rank_command"
chmod 0755 "$rank_command"

echo "source_commit=$AIMO_SOURCE_COMMIT"
echo "input=$AIMO_INPUT_PATH"
echo "master=$MASTER_ADDR:$MASTER_PORT"
echo "command_file=$rank_command"
cd "$AIMO_CODE_DIR"
"${args[@]}"
