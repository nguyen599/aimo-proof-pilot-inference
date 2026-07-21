#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/tmp/p45-adversarial-meta-stability-20260721}"
VENV="${VENV:-/tmp/aimo-proof-pilot-inference-runtime/venv-vllm-0.25.1}"
SOURCE_REPO="${SOURCE_REPO:-/tmp/aimo-proof-pilot-inference-runtime/repo}"
CODE="$ROOT/repo"
MODEL="${MODEL:-/tmp/models/olmo3-opd-sft-750-vllm}"
DFLASH="${DFLASH:-/tmp/models/dflash-32b-draft-v2test-phaseL}"
COMMIT="${COMMIT:-7f1beef2c6ddecf77cc2f71a65c9dea874a89fc2}"
RUN_ROOT=/tmp/aimo-proof-pilot-inference-distributed/runs/imo2025-full-p36-r4-p2-sft750-handoff-priority-2node-20260718T184959Z
P4_LOGS="$RUN_ROOT/logs/rank_0000/llm_calls"
P5_LOGS="$RUN_ROOT/logs/rank_0001/llm_calls"
P4_SOURCE=4/cand_24_proof_gen_r0.txt
P5_SOURCE=5/cand_15_proof_gen_r0.txt

mkdir -p "$ROOT"/audits "$ROOT"/full "$ROOT"/cache
printf '%s\n' starting > "$ROOT/status"
printf '%s\n' "$$" > "$ROOT/driver.pid"

cleanup_server() {
    [ -s "$ROOT/vllm.pid" ] || return 0
    local pid pgid left child cmd
    pid=$(cat "$ROOT/vllm.pid")
    [ -e "/proc/$pid/status" ] || return 0
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    case "$cmd" in
        *vllm.entrypoints.openai.api_server*) ;;
        *) echo "Refusing to stop unexpected vLLM leader pid=$pid cmd=$cmd"; return 1 ;;
    esac
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    [ -n "$pgid" ] || return 0
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 90); do
        left=$(ps -eo pid=,pgid= | awk -v p="$pgid" '$2==p {print $1}' | xargs)
        [ -z "$left" ] && break
        sleep 1
    done
    left=$(ps -eo pid=,pgid= | awk -v p="$pgid" '$2==p {print $1}' | xargs)
    for child in $left; do
        cmd=$(tr '\0' ' ' < "/proc/$child/cmdline" 2>/dev/null || true)
        case "$cmd" in
            *vllm*|VLLM::*|*multiprocessing.resource_tracker*)
                kill -KILL "$child" 2>/dev/null || true
                ;;
            *)
                echo "Refusing SIGKILL for unexpected group member pid=$child cmd=$cmd"
                ;;
        esac
    done
}
trap cleanup_server EXIT

git -C "$SOURCE_REPO" fetch origin main
if [ ! -d "$CODE/.git" ]; then
    rm -rf "$CODE.tmp"
    git clone --shared --no-checkout "$SOURCE_REPO" "$CODE.tmp"
    mv "$CODE.tmp" "$CODE"
fi
git -C "$CODE" fetch "$SOURCE_REPO" "$COMMIT"
git -C "$CODE" checkout --detach --force FETCH_HEAD
actual_commit=$(git -C "$CODE" rev-parse HEAD)
printf '%s\n' "$actual_commit" > "$ROOT/source_commit"
[ "$actual_commit" = "$COMMIT" ] || {
    echo "Unexpected source commit $actual_commit"
    exit 3
}

cd "$CODE"
PYTHONPATH="$CODE" "$VENV/bin/python" - \
    "$P4_LOGS" "$P4_SOURCE" "$ROOT/p4_restart.jsonl" \
    "$P5_LOGS" "$P5_SOURCE" "$ROOT/p5_restart.jsonl" <<'PY'
import json
import sys
from pathlib import Path

from evaluation.harness_vllm.thinking_handoff import (
    parse_saved_proof_generation_call,
)

for offset in (1, 4):
    logs_root = Path(sys.argv[offset])
    source = sys.argv[offset + 1]
    output_path = Path(sys.argv[offset + 2])
    record = parse_saved_proof_generation_call(
        logs_root / source,
        allow_unintervened=True,
    )
    usage = record.usage or {}
    row = {
        "source": source,
        "raw_output": record.output_text,
        "finish_reason": record.finish_reason,
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "prompt_tokens": int(
            usage.get("prompt_tokens") or record.prompt_tokens or 0
        ),
        "base_url": f"replayed://{source}",
        "latency_s": None,
    }
    output_path.write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
PY

export PATH="$VENV/bin:$PATH"
export VLLM_PLUGINS=olmo3_sink
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/torchinductor"
export VLLM_TORCH_COMPILE_CACHE_DIR="$ROOT/cache/vllm"
export CUDA_CACHE_PATH="$ROOT/cache/cuda"
spec=$(printf '{"method":"dflash","model":"%s","num_speculative_tokens":10,"disable_above_context_len":65536}' "$DFLASH")
setsid "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name proof-model \
    --api-key vllm-local \
    --tensor-parallel-size 2 \
    --data-parallel-size 4 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.92 \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 262144 \
    --stream-interval 100 \
    --async-scheduling \
    --enable-prefix-caching \
    --trust-remote-code \
    --generation-config vllm \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --uvicorn-log-level warning \
    --max-num-batched-tokens 16384 \
    --speculative-config "$spec" \
    --disable-custom-all-reduce \
    > "$ROOT/vllm.log" 2>&1 &
vpid=$!
printf '%s\n' "$vpid" > "$ROOT/vllm.pid"

deadline=$((SECONDS + 7200))
while ! curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null; do
    if ! kill -0 "$vpid" 2>/dev/null; then
        echo "vLLM exited before health"
        tail -n 200 "$ROOT/vllm.log"
        exit 4
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "Timed out waiting for vLLM"
        tail -n 200 "$ROOT/vllm.log"
        exit 5
    fi
    sleep 10
done
printf '%s\n' healthy > "$ROOT/vllm.status"
date -u +%FT%TZ > "$ROOT/vllm.ready_at"

run_replay() {
    local logs_root=$1 restart=$2 output=$3 rounds=$4 log=$5
    PYTHONPATH="$CODE" "$VENV/bin/python" \
        -m evaluation.harness_vllm.evaluate_thinking_handoff_refinement \
        --logs-root "$logs_root" \
        --restart-results "$restart" \
        --model-path "$MODEL" \
        --base-url http://127.0.0.1:8000/v1 \
        --served-model-name proof-model \
        --api-key vllm-local \
        --output-dir "$output" \
        --verify-n 8 \
        --verifier-generalist-n 4 \
        --meta-n 1 \
        --meta-policy all-reviews \
        --strict-pass-meta \
        --refine-rounds "$rounds" \
        --refine-review-n 4 \
        --min-valid-low 2 \
        --proof-max-tokens 126000 \
        --verifier-max-tokens 126000 \
        --meta-max-tokens 126000 \
        --thinking-budget-refine-handoff-enabled \
        --thinking-budget-refine-tokens 120000 \
        --thinking-budget-refine-final-round-tokens 120000 \
        --thinking-budget-refine-max-restarts 1 \
        --temperature 0.6 \
        --top-p 0.95 \
        --request-timeout-seconds 7200 \
        > "$log" 2>&1
}

printf '%s\n' auditing > "$ROOT/status"
audit_pids=()
for trial in 1 2 3; do
    run_replay \
        "$P5_LOGS" \
        "$ROOT/p5_restart.jsonl" \
        "$ROOT/audits/p5_trial_$trial" \
        0 \
        "$ROOT/audits/p5_trial_$trial.log" &
    audit_pids+=("$!")
done
run_replay \
    "$P4_LOGS" \
    "$ROOT/p4_restart.jsonl" \
    "$ROOT/audits/p4_safeguard" \
    0 \
    "$ROOT/audits/p4_safeguard.log" &
audit_pids+=("$!")
for pid in "${audit_pids[@]}"; do
    wait "$pid"
done
printf '%s\n' audits_complete > "$ROOT/status"
date -u +%FT%TZ > "$ROOT/audits_completed_at"

while [ ! -e "$ROOT/continue_full" ] && [ ! -e "$ROOT/stop_after_audits" ]; do
    sleep 5
done
if [ -e "$ROOT/stop_after_audits" ]; then
    printf '%s\n' stopped_after_audits > "$ROOT/status"
    exit 0
fi

printf '%s\n' full_replays > "$ROOT/status"
full_pids=()
for trial in 1 2 3; do
    run_replay \
        "$P5_LOGS" \
        "$ROOT/p5_restart.jsonl" \
        "$ROOT/full/p5_trial_$trial" \
        4 \
        "$ROOT/full/p5_trial_$trial.log" &
    full_pids+=("$!")
done
for pid in "${full_pids[@]}"; do
    wait "$pid"
done
printf '%s\n' complete > "$ROOT/status"
date -u +%FT%TZ > "$ROOT/completed_at"
