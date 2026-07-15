#!/bin/bash
# Launch the OPD-32B server from an explicitly supplied YAML configuration.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/workspace/pp/venv}"
if [ "$#" -eq 2 ] && [ "$1" = "--config" ]; then
  CONFIG="$2"
else
  echo "usage: $0 --config PATH" >&2
  exit 2
fi

exec "$VENV/bin/python" "$ROOT/evaluation/harness/launch_server.py" --config "$CONFIG"
