# IMO 2025 P4/P5 adaptive round-zero comparison

This experiment isolates initial-proof quality. It generates 36 candidates for
each of IMO 2025 P4 and P5, then stops before verifier, meta-verifier,
refinement, and selector calls.

The treatment changes only the private planning portfolio. Model weights,
sampling temperatures, thinking budgets, context, DFlash settings, and
candidate count match the measured 36-candidate baseline in
`../imo2025-p45-r4-mixed-challenge-p36-2node-20260721T212441Z`.

The targeted 12-candidate cycle is repeated three times. Exact problem
fingerprints keep these proof-route hints limited to IMO 2025 P4/P5; other
sequence and game problems retain the generic adaptive cycles.

| Problem | Allocation per cycle |
| --- | --- |
| P4 divisor iteration | 2 baseline, 4 full orbit-normal-form plans, 2 backward-divisibility plans, 2 transition-classification plans, 1 counterexample audit, 1 independent reformulation |
| P5 inekoalaty game | 2 baseline, 4 complete paired-threshold plans, 2 Alice Cauchy/spike plans, 2 Bazza pairing/slack plans, 1 regime/boundary completeness audit, 1 independent reformulation |

The P4 plans explicitly avoid the measured false shortcut that one decreasing
step implies eventual failure. They seek backward divisibility, classify the
`13/12`, `31/30`, and fixed transitions, and prove that growth cannot repeat
forever. The P5 plans prove the high-regime strategy against arbitrary Bazza
play, not a saturating proxy, and require separate non-losing strategies at the
boundary.

Run `launch_nii_pair.sh` on physical nodes 2 and 3 after the current four-round
baseline releases those GPUs. Export every structurally complete, non-cutoff
candidate and grade it with `grader.yaml`. The grader uses three API keys, 24
global workers (eight per key), and two independent calls per proof.

Promote the adaptive portfolio only if it improves best-of-36 score or produces
a correct candidate without materially reducing structurally complete proof
rate. Report mean score and per-strategy score distributions as diagnostics,
not as the sole acceptance criterion.

The promotion gate is fixed before launch from the two-call baseline:

| Problem | Baseline eligible | Baseline eligible mean | Baseline best | Minimum acceptable eligible rate |
| --- | ---: | ---: | ---: | ---: |
| P4 | 26/36 (72.2%) | 2.327/7 | 5.0/7 | 62.2% |
| P5 | 20/36 (55.6%) | 1.550/7 | 3.0/7 | 45.6% |

For each problem, promotion requires a best score above the baseline best (or
at least one `7/7` proof) while keeping the structurally complete, non-cutoff
rate within 10 absolute percentage points of baseline. Mean score and strategy
breakdowns diagnose regressions but do not override this best-proof gate.
