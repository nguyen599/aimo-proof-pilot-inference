# SGLang FP8 KV parity

## What differed

The BF16 OLMo3 target and DFlash draft checkpoints do not contain KV-cache
scales. That is not itself an error: vLLM serves the same case with static unit
Q/K/V scales. The runtime contracts differed:

- vLLM statically quantizes Q and passes `q_descale`, `k_descale`, and
  `v_descale` to FA3, even when every scale is `1.0`.
- SGLang raw-cast Q to FP8, omitted `q_descale`, and left K/V descales absent
  when a BF16 checkpoint supplied no quantization metadata.
- The retained SGLang run with mean acceptance around `1.7` used the Triton
  attention backend. The vLLM run with acceptance around `4` used FA3, so that
  old result was not a backend-parity comparison.

## Patch behavior

`sglang_patches/apply_patches.sh` now patches the installed SGLang FA3 backend.
For an FP8-KV configuration, `evaluation/harness/launch_server.py` enables
`SGLANG_FP8_KV_VLLM_PARITY=1` by default. The custom target and draft install
non-persistent unit Q/K/V scale buffers, Q uses SGLang's static FP8 quantizer,
and FA3 receives all three descales.

BF16 KV is unchanged. Set `SGLANG_FP8_KV_VLLM_PARITY=0` only for a controlled
A/B run. Do not combine parity mode with the rejected
`SGLANG_LOAD_KV_SCALE=1` experiment.

Required startup evidence:

```text
fp8_kv_vllm_parity=true attention=fa3
SGLANG_FP8_KV_VLLM_PARITY target scales ready
SGLANG_FP8_KV_VLLM_PARITY draft scales ready
```

## H200 validation

Apply patches to the exact serving venv, then run three profiles with identical
weights, prompts, seeds, temperature, top-p, concurrency, and DFlash settings:

1. `kv_cache_dtype: auto` for the BF16-KV acceptance baseline.
2. `kv_cache_dtype: fp8_e4m3` with parity enabled (the default).
3. The same FP8 profile with `SGLANG_FP8_KV_VLLM_PARITY=0` as the old-contract
   control.

```bash
bash sglang_patches/apply_patches.sh "$VENV"
pytest -q tests/test_sglang_fp8_kv_parity_patch.py tests/test_runtime_config.py
```

Compare mean accepted length and output tokens/s after warmup. Source-level
parity is validated locally, but the numerical fix is not considered complete
until the FP8 parity profile recovers close to the BF16-KV baseline on H200.
