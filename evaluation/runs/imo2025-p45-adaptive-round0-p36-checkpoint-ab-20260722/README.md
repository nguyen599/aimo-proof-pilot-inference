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
the target checkpoint and isolated physical GPU IDs. Both use TP2/DP1, the
same DFlash draft, sampling, thinking-budget handoff, and input file. Node 2
uses GPUs 6-7 and node 3 uses GPUs 4-5 so the experiment does not disturb the
older vLLM workers occupying the other GPUs. No verifier, refinement, or
selector calls are made, so the result measures checkpoint generation quality
rather than downstream ranking quality.

Export every structurally complete, non-cutoff proof and grade it twice with
the same GPT-5.6 grader. Compare eligible rate, mean score, best-of-36 score,
and score distribution per problem. Do not promote step-225 from the IMO 2026
report alone: that report found step-225 and deploy statistically tied, while
the larger search width was the clearer source of its P5 gain.

## Results

All eligible proofs were graded twice with `openai/gpt-5.6-sol` at high
reasoning effort. The SFT run completed 118/118 calls and the step-225 run
completed 94/94 calls, with no failed, invalid, missing, or duplicate calls.

| Problem | Checkpoint | Eligible / 36 | Mean / 7 | Best / 7 | Candidates >= 5 |
| --- | --- | ---: | ---: | ---: | ---: |
| P4 | SFT-750 | 33 (91.7%) | 2.773 | 5.5 | 4 |
| P4 | OPD step-225 | 28 (77.8%) | 2.357 | 5.0 | 2 |
| P5 | SFT-750 | 26 (72.2%) | 2.538 | 5.0 | 3 |
| P5 | OPD step-225 | 19 (52.8%) | 2.395 | **7.0** | 2 |

SFT-750 is the stronger general initial-prover checkpoint: it has materially
higher completion rates and higher mean scores on both problems, and it wins
P4 on both mean and best score. Step-225 should nevertheless remain in the P5
initial pool because `p5-c30` (`p5_alice_cauchy_spike`) received `[7, 7]` and
is the only fully correct initial P5 proof in either 36-candidate run. Selecting
only by mean checkpoint quality would discard the most important tail event.

The next initial-pool experiment should therefore use SFT-750 for all P4
candidates and a heterogeneous P5 pool containing both SFT-750 and step-225.
P4 allocation should emphasize the orbit-normal-form and backward-divisibility
routes. P5 allocation should emphasize the Alice Cauchy spike and the complete
three-regime pairing proof. The experiment must retain checkpoint identity in
candidate metadata so later verifier and selector analysis can measure whether
the perfect step-225 tail is preserved.

## Recurring proof failures

The strongest P4 proof still omitted a valid treatment of `N=2m` when `m` is
composite, odd, and not divisible by 3. In particular, `N=70` has complementary
denominators `2, 5, 7`, so its next term is `35+14+10=59`; the common shortcut
using `m,2,1` is false. A complete necessity proof needs a closed strict descent
outside multiples of 6, not just one decreasing transition.

The P5 SFT pool spread the necessary ingredients across different proofs:

- Bazza's strict-low-regime complementary-pair strategy, including legality;
- Alice's arbitrary-history Cauchy spike in the strict-high regime; and
- one non-losing strategy for each player at equality.

Step-225 `p5-c30` assembled all three correctly. Its proof is retained in
`results/step225/grader_input/records.jsonl` with source hash
`19bbccfb8ce0b6fb8916b8e006668178a748dfc87de6dc07580ce485d1b50673`.

## Artifacts

`results/{sft750,step225}` contains generation summaries, eligible manifests,
proof texts with source hashes, grader summaries, compact grader findings, and
the generated per-checkpoint report. Full API response envelopes are omitted;
the compact records contain every score and finding needed to reproduce the
aggregate tables without storing multi-megabyte reasoning payloads.
