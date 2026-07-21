#!/usr/bin/env bash
set -eu
cd "/tmp/aimo-thinking-checkpoint/proof-checkpoint-restart-cand6-base-20260721/repo"
exec "/tmp/aimo-proof-pilot-inference-runtime/venv-vllm-0.25.1/bin/python" -m evaluation.harness_vllm.evaluate_thinking_handoff_restart \
  --logs-root "/tmp/aimo-thinking-checkpoint/proof-checkpoint-restart-cand6-base-20260721/repo/evaluation/runs/imo2025-full-p16-r1-p2-sft750-parserfix4-20260716T173047Z-partial-20260717/logs" \
  --handoff-results "/tmp/aimo-thinking-checkpoint/lossless-restart-cand6-base-20260721/handoff_results.jsonl" \
  --model-path /tmp/models/olmo3-opd-sft-750-vllm \
  --base-url http://127.0.0.1:8000 \
  --served-model-name proof-model \
  --api-key vllm-local \
  --output-dir "/tmp/aimo-thinking-checkpoint/lossless-restart-cand6-base-20260721/output" \
  --variant lossless_partial \
  --temperature 0 \
  --proof-temperature 1.0 \
  --restart-strategy deadline_aware \
  --force-finalize-at-budget \
  --thinking-budget-tokens 100000 \
  --max-tokens 126000 \
  --case-count 1 \
  --max-workers 1 \
  --request-timeout-seconds 7200
