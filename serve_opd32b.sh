#!/bin/bash
# serve_opd32b.sh — serve the bf16 opd-32b-deploy (Olmo3Sink) with the patched
# sglang on an H200. Self-contained: bf16 only, one GPU per replica.
#
# Prereqs (see README): proof-pilot-env venv staged at $VENV and patched via
# sglang_patches/apply_patches.sh; model downloaded to $MODEL.
#
# Usage:  bash serve_opd32b.sh                              # GPU 0, port 30000
#         PORT=30001 CUDA_VISIBLE_DEVICES=1 bash serve_opd32b.sh   # second replica
set -euo pipefail

VENV="${VENV:-/workspace/pp/venv}"
MODEL="${MODEL:-/workspace/models/opd-32b-deploy}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
CTX="${CTX:-200000}"           # context length
MEMFRAC="${MEMFRAC:-0.88}"     # weights (61GB) + fp8 KV pool on a 143GB H200
MAXREQ="${MAXREQ:-48}"         # max concurrent requests (= decode cuda-graph max bs)
CHUNKED="${CHUNKED:-2048}"     # prefill chunk size (prefill graph buckets derive from it)
KV_SPLITS="${KV_SPLITS:-32}"   # triton decode kv-splits (long-ctx single-stream occupancy)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-9.0a}"   # H200 = sm90
export FLASHINFER_USE_CUDA_NORM=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# env-gated perf patches from sglang_patches/ (no-ops if not applied)
export SGLANG_DECODE_NUM_STAGES="${SGLANG_DECODE_NUM_STAGES:-3}"
export SGLANG_DECODE_BLOCK_N="${SGLANG_DECODE_BLOCK_N:-32}"
export SGLANG_GQA_PACKED_EXTEND="${SGLANG_GQA_PACKED_EXTEND:-1}"

# JIT robustness (no-op on a full-CUDA box): flashinfer's JIT link needs
# libcuda.so on LIBRARY_PATH, and NVRTC needs CCCL headers under the venv's
# bundled CUDA include root.
_link=/tmp/pp_link; mkdir -p "$_link"
if [ ! -e "$_link/libcuda.so" ]; then
  for _lc in /usr/local/cuda*/targets/*/lib/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so \
             /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/local/cuda*/compat/libcuda.so*; do
    if [ -e "$_lc" ]; then ln -s "$_lc" "$_link/libcuda.so"; break; fi
  done
fi
export LIBRARY_PATH="${_link}${LIBRARY_PATH:+:$LIBRARY_PATH}"
_cccl="$(ls -d "$VENV"/lib/python*/site-packages/flashinfer/data/cccl/libcudacxx/include 2>/dev/null | head -1)"
_cuinc="$(ls -d "$VENV"/lib/python*/site-packages/nvidia/cu13/include 2>/dev/null | head -1)"
if [ -n "$_cccl" ] && [ -n "$_cuinc" ] && [ ! -e "$_cuinc/cccl/cuda/std/cstdint" ]; then
  ln -sf "$_cccl" "$_cuinc/cccl"
fi

# capture every decode bs 1..16 (no padding for small batches) + sparse tail to MAXREQ
CG_BS_DECODE="$(for b in $(seq 1 16) 20 24 28 32 40 48 64 96 128; do if [ "$b" -le "$MAXREQ" ]; then printf '%s ' "$b"; fi; done)"

echo "[serve_opd32b] model=$MODEL gpu=$CUDA_VISIBLE_DEVICES port=$PORT ctx=$CTX memfrac=$MEMFRAC maxreq=$MAXREQ"

exec "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" \
  --attention-backend triton \
  --tp 1 --host "$HOST" --port "$PORT" \
  --mem-fraction-static "$MEMFRAC" \
  --chunked-prefill-size "$CHUNKED" \
  --context-length "$CTX" \
  --kv-cache-dtype fp8_e4m3 \
  --stream-interval 16 \
  --swa-full-tokens-ratio 0.1 \
  --max-running-requests "$MAXREQ" --cuda-graph-max-bs-decode "$MAXREQ" \
  --cuda-graph-bs-decode $CG_BS_DECODE \
  --cuda-graph-backend-prefill tc_piecewise --cuda-graph-bs-prefill 256 1024 "$CHUNKED" \
  --triton-attention-num-kv-splits "$KV_SPLITS" \
  --reasoning-parser deepseek-r1 ${EXTRA_ARGS:-}
