# IMO 2025 Q1 DFlash partial snapshot

This directory preserves the live `test-opd-200` inference run before the H200
node shutdown window on 2026-07-15. The run uses vLLM 0.25.1, online FP8 for
the target model, FP8 KV cache, the OLMo3Sink DFlash draft model, a 65,536-token
DFlash cutoff, and a 262,144-token model context.

Snapshot time: `2026-07-15T06:14:56+07:00`.

At this snapshot, one of the fourteen initial Q1 proof candidates had completed.
Candidate 13 completed after 52,244 generated tokens with a reasoning-only gzip
factor of `4.2641`; none of its 27,268 32-word windows was repeated. Candidate
10 had crossed the forced thinking-budget boundary and its partial output was
also on disk. The other proof-generation response files still contain their
inputs because those streaming calls had not finalized.

`logs/run.log` is the inference-pipeline log, `logs/vllm_server_0.log` is the
server log, and `logs/container.log` captures the live container output.
`logs/llm_calls/1/` contains every Q1 call artifact available at snapshot time,
including proof, verifier, meta-verifier, and refinement calls. Runtime Git and
container metadata are included under `logs/` for reproducibility.

This is intentionally a partial run artifact. No engine death, cutoff
`IndexError`, or `EngineDead` event had occurred when it was captured.
