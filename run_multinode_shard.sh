#!/usr/bin/env bash
# Run one shard of a multi-node proof search and merge after every shard succeeds.
#
# Invoke the same command on every node, changing only SHARD_INDEX:
#   SHARD_COUNT=2 SHARD_INDEX=0 ./run_multinode_shard.sh CONFIG SHARED_OUTPUT [INPUT]
#   SHARD_COUNT=2 SHARD_INDEX=1 ./run_multinode_shard.sh CONFIG SHARED_OUTPUT [INPUT]
#
# Each node owns its local SGLang server. Outputs and completion markers live
# under SHARED_OUTPUT, which must be visible to every node.
set -uo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-}"
SHARED_OUTPUT="${2:-}"
INPUT="${3:-$REPO/evaluation/data/imo2026-latex-test.csv}"
SHARD_COUNT="${SHARD_COUNT:-2}"
SHARD_INDEX="${SHARD_INDEX:-0}"

[[ -n "$CONFIG" && -n "$SHARED_OUTPUT" ]] || {
    echo "usage: SHARD_COUNT=N SHARD_INDEX=I $0 CONFIG SHARED_OUTPUT [INPUT]" >&2
    exit 2
}
[[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || {
    echo "SHARD_COUNT must be an integer >= 1" >&2
    exit 2
}
[[ "$SHARD_INDEX" =~ ^[0-9]+$ ]] && (( SHARD_INDEX < SHARD_COUNT )) || {
    echo "SHARD_INDEX must be in [0, SHARD_COUNT)" >&2
    exit 2
}

mkdir -p "$SHARED_OUTPUT/shards"
SHARD_OUTPUT="$SHARED_OUTPUT/shards/shard-$SHARD_INDEX"
STATUS_PATH="$SHARED_OUTPUT/shards/shard-$SHARD_INDEX.status"

if [[ "${RESUME:-0}" == "1" ]]; then
    "$REPO/scheduler.sh" --resume "$SHARD_OUTPUT"
    status=$?
else
    "$REPO/scheduler.sh" \
        --shard-count "$SHARD_COUNT" \
        --shard-index "$SHARD_INDEX" \
        "$CONFIG" "$SHARD_OUTPUT" "$INPUT"
    status=$?
fi

status_tmp="$STATUS_PATH.tmp.$$"
printf '%s\n' "$status" >"$status_tmp"
mv -f "$status_tmp" "$STATUS_PATH"
if (( status != 0 )); then
    echo "[multinode] shard $SHARD_INDEX/$SHARD_COUNT failed with status $status" >&2
    exit "$status"
fi

exec 9>"$SHARED_OUTPUT/.merge.lock"
flock 9
shard_paths=()
for ((index = 0; index < SHARD_COUNT; index++)); do
    peer_status="$SHARED_OUTPUT/shards/shard-$index.status"
    if [[ ! -f "$peer_status" ]] || [[ "$(<"$peer_status")" != "0" ]]; then
        echo "[multinode] shard $SHARD_INDEX complete; waiting for shard $index"
        exit 0
    fi
    shard_paths+=("$SHARED_OUTPUT/shards/shard-$index/submission.csv")
done

PYTHON="${PYTHON:-${VENV:-/opt/pp/venv}/bin/python}"
"$PYTHON" "$REPO/evaluation/harness/merge_submissions.py" \
    --input "$INPUT" \
    --output "$SHARED_OUTPUT/submission.csv" \
    "${shard_paths[@]}"
printf '0\n' >"$SHARED_OUTPUT/merge.status"
echo "[multinode] all $SHARD_COUNT shards merged -> $SHARED_OUTPUT/submission.csv"
