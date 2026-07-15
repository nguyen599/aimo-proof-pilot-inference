#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" "/var/log/portal/opd32b-eval.log"
. "${utils}/environment.sh"

cd /workspace/aimo-proof-pilot-eval
export VENV=/workspace/pp/venv
: "${CONFIG:?CONFIG is required and must point to config.yaml}"
exec /bin/bash serve_opd32b.sh \
  --config "$CONFIG"
