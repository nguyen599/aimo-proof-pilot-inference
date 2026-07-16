# IMO 2025 SFT-750 parser-fix inference snapshot

This is a non-disruptive point-in-time snapshot of the live two-node NII run
`imo2025-full-p16-r1-p2-sft750-parserfix4-20260716T173047Z`, captured at `2026-07-16T18:18:22Z`.

Configuration:

- code commit: `461a4d5a9dd0a2c63d59fab1011556a367c9f439` (final XML is parsed only after the final `</think>`)
- model: `nguyen599/olmo3-opd-sft-750`
- dataset: full IMO 2025 problem list
- workers: relay nodes `node6` and `node7`
- parallel problems: 2
- pipelines per problem: 16
- refinement rounds: 1
- runtime: vLLM 0.25.1 with DFlash speculative decoding

At snapshot time, all **32/32 proof candidates** had completed. The copied
completed-call set also contains 52 verifier, 52 meta-verifier,
and 8 refinement traces. Rank 0 alive: `yes`; rank 1 alive:
`yes`. Liveness was verified on each rank's owning node because process IDs are
node-local. The inference process was not stopped or restarted.

Live parser validation before publication inspected 52 verifier prompts: zero
contained the old `The complete rigorous proof.` placeholder and zero had a
candidate solution shorter than 100 characters.

Contents:

- `logs/rank0/llm_calls/` and `logs/rank1/llm_calls/`: completed raw LLM calls only.
- `SNAPSHOT.txt`: machine-readable source paths, counts, and run state.
- `LLM_CALLS_SHA256SUMS.txt`: checksums for every copied trace.
