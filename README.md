# aimo-proof-pilot-inference

Inference for the **opd-32b-deploy** model (`Olmo3SinkForCausalLM` — Olmo3 32B
plus a trained per-head attention-sink logit in every layer, gpt-oss style,
with hybrid sliding-window attention and YaRN rope). Weights:
`ycchen/proof-pilot-deploy-bundle/opd-32b-deploy` on Hugging Face (bf16, 61 GB).

bf16 serving only. Verified on 2× H200 (one replica per GPU).

Stock sglang/vLLM don't know the `Olmo3Sink` architecture, so this runs a
patched sglang: `sglang_patches/` carries the upstream proof-pilot patch files
(sink-aware `olmo2.py` target model, DFlash speculative-decoding support,
SWA-eviction fix, env-gated triton decode/extend tuning) and an applier.

## Environment (not in this repo)

The server runs in the prebuilt `proof-pilot-env` venv (own Python 3.12,
torch 2.11 cu130, custom sglang 0.5.14 nightly build — no conda needed):

```bash
unzip proof-pilot-env.zip -d proof-pilot-env-x       # -> proof-pilot-env.bin (gzip tar)
mkdir -p /workspace/pp
tar -xzf proof-pilot-env-x/proof-pilot-env.bin -C /workspace/pp --strip-components=1
sed -i "s|^home = .*|home = /workspace/pp/pybase/bin|" /workspace/pp/venv/pyvenv.cfg
mkdir -p ~/.cache/flashinfer ~/.humming/cache
cp -rn /workspace/pp/flashinfer_cache/. ~/.cache/flashinfer/

# apply the sglang patches from this repo to the venv (idempotent)
bash sglang_patches/apply_patches.sh /workspace/pp/venv
```

Model download (needs HF_TOKEN):

```bash
hf download ycchen/proof-pilot-deploy-bundle --include "opd-32b-deploy/*" \
  --local-dir /workspace/models
```

## Serve + solve

```bash
bash serve_opd32b.sh &                                    # GPU 0, port 30000
PORT=30001 CUDA_VISIBLE_DEVICES=1 bash serve_opd32b.sh &  # GPU 1, port 30001
python solve_problems.py                                  # fans out across both
```

Each replica: bf16 weights (61 GB), fp8 KV cache, 200k context, triton
attention with in-kernel sinks, CUDA graphs up to batch 48, deepseek-r1
reasoning parser (`reasoning_content` separated from `content` in the API).
First boot JIT-compiles triton kernels for sm90 (~2 min); later boots reuse
the cache.

## Measured throughput (2× H200, 512-token generations)

| Concurrency per GPU | Total tok/s | Per stream |
|---|---|---|
| 1 | 78 | 38.8 |
| 4 | 352 | 44.0 |
| 16 | 1,160 | 36.2 |
| 48 | 3,772 | 39.3 |

Per-stream speed stays flat up to the configured `MAXREQ=48` cap — throughput
scales linearly with concurrency. Long-context requests decode slower as the
KV cache grows.

`sample_results_sglang.json` holds the outputs of the 6-problem sample run
(5/6 proved cleanly; the harmonic-sum problem thinks past the token cap —
known long-thinking tendency of this OPD checkpoint).
