# Stopped Four-Node IMO 2025 Run

Run ID: `imo2025-full-p36-r4-p6-sft750-lossless-deadline-4node-retry-20260718T161415Z`

This is a complete point-in-time log snapshot made after the user stopped the run so the nodes could be released to teammates. It includes all four ranks' harness logs, vLLM logs, LLM-call files, distributed startup metadata, and launcher logs. The nested launcher repository checkout is intentionally excluded because it is source code, not a run log.

## Progress

- Requested work: 6 problems, 36 candidates per problem, 4 refinement rounds.
- LLM-call files created: 141.
- Completed proof-generation outputs: 13.
- Interrupted prompt-only files: 128.
- Started by problem: P1=36, P2=36, P3=36, P4=32, P5=1, P6=0.
- Completed by problem: P1=0, P2=4, P3=3, P4=6, P5=0, P6=0.
- Completed output tokens: 1,040,851 total; 56,936 minimum; 100,900 maximum; 80,065.5 mean.
- Closed `</think>` outputs: 13/13.
- Parseable `<solution>`, `<self_evaluation>`, and `<score>` triplets: 13/13.
- Model self-scores: {'1': 13}. These are not verified correctness results.
- Verifier calls: 0; meta-verifier calls: 0; refine calls: 0.
- No final submission was produced before shutdown.

The interrupted files contain exact prompts that had been scheduled but no `===== OUTPUT =====` marker. Raw in-flight model text was not persisted by the harness before those requests were terminated.

## Layout

- `logs/rank_*/run.log`: inference harness logs.
- `logs/rank_*/vllm_server_0.log`: complete local vLLM server logs.
- `logs/rank_*/llm_calls/`: every persisted proof-generation call record, including completed and interrupted calls.
- `launcher/`: launch environment, per-rank commands, launcher logs, submit logs, and PID records.
- `manifest.json`, `startup/`, and `stages/`: distributed-run coordination metadata.
- `PROGRESS.json`: machine-readable audit and per-call completion metadata.
- `SHA256SUMS.txt`: checksums for every snapshot artifact except the checksum file itself.
