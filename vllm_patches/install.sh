#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-python}"

if [[ -d "$TARGET" ]]; then
  PYTHON_BIN="$TARGET/bin/python"
else
  PYTHON_BIN="$TARGET"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python interpreter is not executable: $PYTHON_BIN" >&2
  exit 1
fi

BUILD_ROOT="$(mktemp -d)"
trap 'rm -rf "$BUILD_ROOT"' EXIT
cp "$ROOT/pyproject.toml" "$ROOT/README.md" "$BUILD_ROOT/"
cp -R "$ROOT/src" "$BUILD_ROOT/src"

"$PYTHON_BIN" -m pip install \
  --no-deps \
  --no-build-isolation \
  --disable-pip-version-check \
  "$BUILD_ROOT"

VLLM_PLUGINS=olmo3_sink "$PYTHON_BIN" - <<'PY'
from vllm import ModelRegistry
from vllm.plugins import load_general_plugins

load_general_plugins()
architecture = "Olmo3SinkForCausalLM"
if architecture not in ModelRegistry.get_supported_archs():
    raise RuntimeError(f"vLLM plugin did not register {architecture}")
model_class = ModelRegistry._try_load_model_cls(architecture)
if model_class is None or model_class.__name__ != architecture:
    raise RuntimeError(f"vLLM could not import {architecture}")
print(f"[vllm-patch] registered {model_class.__module__}:{model_class.__name__}")
PY
