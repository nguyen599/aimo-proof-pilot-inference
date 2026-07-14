# OLMo3Sink vLLM Plugin

This directory provides the target-model support required to serve Proof Pilot
checkpoints whose `config.json` declares `Olmo3SinkForCausalLM`.

The plugin adds only the target model. It does not install or register a DFlash
draft model and does not enable speculative decoding.

## Behavior

- Registers `Olmo3SinkForCausalLM` through vLLM's `vllm.general_plugins`
  entry-point group. vLLM loads this plugin in the API, engine-core, and worker
  processes.
- Passes each layer's trained per-query-head sink logits to vLLM's sink-aware
  attention backend.
- Preserves OLMo3 hybrid sliding-window attention and applies YaRN only to
  full-attention layers.
- Shards the checkpoint's full sink vector by tensor-parallel rank.
- Fails model loading when a local layer is missing its sink tensor or a sink
  tensor has the wrong number of heads.

The implementation targets vLLM `0.23.1rc1.dev699` through `0.25.1`. The CUDA
attention backend must support sinks. On H200, vLLM's FlashAttention 3 backend
provides this support.

## Install

Install into the same Python environment that owns the `vllm` executable:

```bash
bash vllm_patches/install.sh /path/to/venv
```

The script installs the local plugin without resolving dependencies, then
starts a fresh Python process to verify model registration and importability.

To install with an explicit interpreter instead:

```bash
bash vllm_patches/install.sh /path/to/python
```

## Serve

`VLLM_PLUGINS` is optional when no plugin allowlist is configured. Setting it
explicitly makes the deployment deterministic:

```bash
VLLM_PLUGINS=olmo3_sink vllm serve /path/to/opd-32b-deploy \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --reasoning-parser deepseek_v4
```

Do not rename the checkpoint architecture to `Olmo3ForCausalLM`. The stock
model has no sink parameters and would silently change every attention
distribution.
