# IMO 2025 P4/P5 checkpoint A/B

This experiment compares the current SFT-750 model with the teammate pipeline's
OPD step-225 checkpoint on the same IMO 2025 P4/P5 round-zero workload.

Run `launch_nii_checkpoint_ab.sh` on both physical NII nodes after the active
baseline releases them:

| Physical node | Checkpoint |
| --- | --- |
| 2 | `/tmp/models/olmo3-opd-sft-750-vllm` |
| 3 | `/tmp/chankhavu/models/opd-32b-bf16-step-225` |

Each node runs independently with `WORLD_SIZE=1` and generates 36 adaptive
initial proofs for each problem. The launch settings are identical except for
the target checkpoint. Both use TP2/DP4, the same DFlash draft, sampling,
thinking-budget handoff, and input file. No verifier, refinement, or selector
calls are made, so the result measures checkpoint generation quality rather
than downstream ranking quality.

Export every structurally complete, non-cutoff proof and grade it twice with
the same GPT-5.6 grader. Compare eligible rate, mean score, best-of-36 score,
and score distribution per problem. Do not promote step-225 from the IMO 2026
report alone: that report found step-225 and deploy statistically tied, while
the larger search width was the clearer source of its P5 gain.
