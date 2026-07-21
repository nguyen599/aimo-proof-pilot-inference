#!/usr/bin/env bash
set +e
echo $$ > "/tmp/aimo-proof-pilot-inference-submit/imo2025-p245-round0-p64-sft750-20260721T181144Z/node2/pid"
rm -f "/tmp/aimo-proof-pilot-inference-submit/imo2025-p245-round0-p64-sft750-20260721T181144Z/node2/status"
export GLOBAL_RANK="2"
export AIMO_RUN_ID="imo2025-p245-round0-p64-sft750-20260721T181144Z"
export AIMO_SOURCE_REPO="/tmp/aimo-proof-pilot-inference-source-4de9967"
export AIMO_SOURCE_REF="4de9967519e1aa4c7d8fb241e466f768abfd71ba"
export AIMO_VENV="/tmp/aimo-proof-pilot-inference-runtime/venv-vllm-0.25.1"
export AIMO_DISTRIBUTED_ROOT="/tmp/aimo-proof-pilot-inference-distributed"
export AIMO_LAUNCH_ROOT="/tmp/aimo-proof-pilot-inference-launch"
export MASTER_PORT=29645
bash "/tmp/aimo-proof-pilot-inference-source-4de9967/evaluation/runs/imo2025-p245-round0-p64-sft750-20260721/launch_nii_pair.sh"
rc=$?
printf '%s\n' "$rc" > "/tmp/aimo-proof-pilot-inference-submit/imo2025-p245-round0-p64-sft750-20260721T181144Z/node2/status"
exit "$rc"
