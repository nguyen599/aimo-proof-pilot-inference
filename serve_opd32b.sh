#!/bin/bash
# Launch the single YAML-configured OPD-32B server. BF16 target-only TP=2 is
# the default; quantization and DFlash are independent YAML opt-ins.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/workspace/pp/venv}"
CONFIG="$ROOT/evaluation/configs/nemotron_cascade2.yaml"

if [ "$#" -eq 2 ] && [ "$1" = "--config" ]; then
  CONFIG="$2"
elif [ "$#" -ne 0 ]; then
  echo "usage: $0 [--config PATH]" >&2
  exit 2
fi

exec "$VENV/bin/python" "$ROOT/evaluation/harness/launch_server.py" --config "$CONFIG"
