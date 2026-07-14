# IMO 2025 Q1 DFlash proof-generation snapshot

This directory preserves the live `test-opd-200` inference run after all 14
initial proof candidates for IMO 2025 question 1 completed. Downstream verifier,
meta-verifier, and refinement work was still in progress. The run uses vLLM
0.25.1, online FP8 for the target model, FP8 KV cache, the OLMo3Sink DFlash
draft model, a 65,536-token DFlash cutoff, and a 262,144-token model context.

Snapshot time: `2026-07-15T06:43:52+07:00`.

The median reasoning-only gzip factor is `4.4407`; four candidates exceed the
`5.0` warning threshold. Candidate 10 is the clear failure case with a gzip
factor of `47.9394` and 94.99% repeated 32-word windows. Candidates 1 and 11
also show substantial repetition. Full per-candidate values are in
`gzip_report.md` and `gzip_report.json`.

`logs/run.log` is the inference-pipeline log, `logs/vllm_server_0.log` is the
server log, and `logs/container.log` captures the live container output.
`logs/llm_calls/1/` contains all 14 completed proof call artifacts and every
verifier, meta-verifier, and refinement artifact available at snapshot time.
Runtime Git, process, container, and GPU metadata are included under `logs/`
for reproducibility.

This is intentionally a partial end-to-end run artifact, but Q1 initial proof
generation is complete. No engine death, cutoff `IndexError`, or `EngineDead`
event had occurred when it was captured.
