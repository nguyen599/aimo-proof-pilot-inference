# IMO 2025 P4/P5 adaptive round-zero comparison

This experiment isolates initial-proof quality. It generates 36 candidates for
each of IMO 2025 P4 and P5, then stops before verifier, meta-verifier,
refinement, and selector calls.

The treatment changes only the private planning portfolio. Model weights,
sampling temperatures, thinking budgets, context, DFlash settings, and
candidate count match the measured 36-candidate baseline in
`../imo2025-p45-r4-mixed-challenge-p36-2node-20260721T212441Z`.

The adaptive 12-candidate cycle is repeated three times:

| Problem | Allocation per cycle |
| --- | --- |
| P4 iterated sequence | 3 baseline, 3 exhaustive transitions, 2 state invariants, 2 proof-obligation ledgers, 1 counterexample audit, 1 independent reformulation |
| P5 adversarial game | 2 baseline, 3 adversarial-quantifier audits, 3 joint-state inequalities, 2 proof-obligation ledgers, 1 regime/boundary completeness audit, 1 independent reformulation |

Run `launch_nii_pair.sh` on physical nodes 2 and 3 after the current four-round
baseline releases those GPUs. Export every structurally complete, non-cutoff
candidate and grade it with `grader.yaml`. The grader uses three API keys, 24
global workers (eight per key), and two independent calls per proof.

Promote the adaptive portfolio only if it improves best-of-36 score or produces
a correct candidate without materially reducing structurally complete proof
rate. Report mean score and per-strategy score distributions as diagnostics,
not as the sole acceptance criterion.
