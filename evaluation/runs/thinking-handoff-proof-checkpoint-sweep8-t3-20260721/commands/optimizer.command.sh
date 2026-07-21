#!/usr/bin/env bash
set +e
ROOT=/tmp/aimo-thinking-checkpoint/proof-checkpoint-sweep8-t3-20260721T114752Z
CODE=/tmp/aimo-thinking-checkpoint/code-db82c5a
LOGS=/tmp/aimo-thinking-checkpoint/code-db82c5a/evaluation/runs/imo2025-full-p16-r1-p2-sft750-parserfix4-20260716T173047Z-partial-20260717/logs
EXPECTED_SHA=db82c5aa0d927c86e7bcf7997aaae8e86eb9befc
STATUS="$ROOT/optimizer.status"
rm -f "$STATUS"
cd "$CODE" || { printf '%s\n' 90 > "$STATUS"; exit 90; }
ACTUAL_SHA=$(git rev-parse HEAD 2>/dev/null)
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
  echo "unexpected code sha: $ACTUAL_SHA" >&2
  printf '%s\n' 91 > "$STATUS"
  exit 91
fi
CUTOFF_FILES=$(find "$LOGS" -path '*llm_calls*' -name '*proof_gen*.txt' -type f | wc -l)
if [ "$CUTOFF_FILES" -lt 17 ]; then
  echo "insufficient saved proof calls: $CUTOFF_FILES" >&2
  printf '%s\n' 92 > "$STATUS"
  exit 92
fi
echo "run_id=proof-checkpoint-sweep8-t3-20260721T114752Z sha=$ACTUAL_SHA cutoff_files=$CUTOFF_FILES"
echo "endpoints=http://127.0.0.1:8000,http://172.17.7.55:8000"
PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  /tmp/aimo-proof-pilot-inference-runtime/venv-vllm-0.25.1/bin/python \
  evaluation/harness_vllm/optimize_thinking_handoff.py \
  --logs-root "$LOGS" \
  --model-path /tmp/models/olmo3-opd-sft-750-vllm \
  --base-url http://127.0.0.1:8000 \
  --base-url http://172.17.7.55:8000 \
  --served-model-name proof-model \
  --api-key vllm-local \
  --output-dir "$ROOT/output" \
  --case-count 8 \
  --variant proof_checkpoint \
  --temperature 1.0 \
  --temperature 0.7 \
  --temperature 0.6 \
  --max-tokens 32768 \
  --checkpoint-audit \
  --checkpoint-audit-max-tokens 8192 \
  --checkpoint-audit-temperature 0.2 \
  --generation-mode monolithic \
  --repair-invalid \
  --max-workers 16 \
  --request-timeout-seconds 7200 \
  --top-p 0.95
rc=$?
printf '%s\n' "$rc" > "$STATUS"
echo "optimizer_exit=$rc"
exit "$rc"
