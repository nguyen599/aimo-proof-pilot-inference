# Legacy six-problem evaluation

This directory preserves the original six-problem AIMO Proof Pilot experiment.
It is historical evidence, not the active IMO ProofBench evaluation pipeline.
The active 60-problem dataset, harness, grader prompt, configuration, and future
run artifacts live in the parent [`evaluation/`](../) directory.

Contents:

- `problems.csv`: the six LaTeX-faithful markscheme problems.
- `run_legacy_eval.sh`: the original prove/verify/refine/select workflow, updated
  only to resolve input and output paths from this directory.
- `results/`: committed DIVALL and ALTORO submissions and trace summaries.

The runner requires the extracted `/workspace/proof-pilot-code-x` bundle and live
SGLang servers. It is retained for reproducibility and never imported by the
ProofBench harness.

```bash
bash evaluation/legacy-six-problem/run_legacy_eval.sh
```
