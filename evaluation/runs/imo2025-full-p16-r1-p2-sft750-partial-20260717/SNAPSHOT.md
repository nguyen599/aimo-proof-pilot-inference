# IMO 2025 SFT-750 distributed inference snapshot

This directory is a point-in-time snapshot of the live two-node NII run
`imo2025-full-p16-r1-p2-sft750-20260716T144750Z`.

Configuration:

- model: `nguyen599/olmo3-opd-sft-750`
- dataset: full IMO 2025 problem list
- workers: relay nodes `node6` and `node7`
- parallel problems: 2
- pipelines per problem: 16
- refinement rounds: 1
- runtime: vLLM 0.25.1 with DFlash speculative decoding

Both ranks were still running when this snapshot was taken at
`2026-07-17T01:03:37+09:00`. The partial submission contains the two completed
problem outputs. Active requests observed during the status check were decoding
at approximately 76-78 tokens/s.

Contents:

- `artifacts/submission.csv`: partial two-row model output.
- `artifacts/problems/`: four per-rank JSON state files for the two completed
  problems.
- `logs/`: point-in-time launcher logs and rank PID files.
- `SNAPSHOT.txt`: source paths, process state, sizes, and submission hash.

Creating this snapshot did not stop or restart the live inference processes.
