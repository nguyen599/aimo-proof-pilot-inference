# Experiment log

Running log of IMO-2026 inference experiments (newest first). Each entry pins the
model, config, problem selection, node, code commit, and the exact HF trace
directory so a run is reproducible and its outputs are findable. Companion to
[`NONDETERMINISM.md`](NONDETERMINISM.md), [`DEGENERATE_FILTER.md`](DEGENERATE_FILTER.md),
and [`../CHANGES_VS_UPSTREAM.md`](../CHANGES_VS_UPSTREAM.md).

---

## `imo2026-bugfix-p145` — P1/P4/P5 full-search on the bugfix merge

- **Started:** 2026-07-19 23:22 UTC · **Node:** node0 (`hnode495`), 8×H200 · **Status:** running
- **Purpose:** first full-search production run on the hardened harness — validate
  the merged production checkpoint on a representative quick-benchmark subset
  (P1 easy; P4/P5 harder) with streaming loop-abort + salvage, watchdog bump, and
  the gzip degenerate filter all active, plus the new `--problems` selector.
- **Model:** `opd-32b-bf16-merged-125-to-225-bugfix` (HF `fieldsmodelorg/Olmo-3.1-32B-Think-OPD-IMO`, subdir `opd-32b-bf16-merged-125-to-225-bugfix`) + DFlash draft `dflash-32b-draft-v2test-phaseL`.
- **Config:** [`config-nii-bugfix-p145.yaml`](../config-nii-bugfix-p145.yaml) — full production search: `max_rounds 4`, `proofs_per_round 32`, `verifications_per_proof 16`, `top_proofs 8`, `refine_parents 4`, `reviews_per_refine_parent 3`, `min_valid_verifications 4`; `temperature 1.0` / `top_p 0.95`; **nondeterministic** (`deterministic_inference: false`); `filter_degenerate: true`, `stream_detect: true`, `watchdog_timeout: 1200`; `seed 0`; tp2×dp4, fa3, ctx 262144.
- **Problems:** P1, P4, P5 — selected at run time via `--problems 1,4,5` (the new
  selector; nothing baked into the config). Input = the full 6-problem
  `evaluation/data/imo2026-latex-test.csv` (verified byte-identical to the
  `imo2026-deploy-r4` pinned input for P1/P4/P5). Artifacts index by position:
  `row-0000`=P1, `row-0001`=P4, `row-0002`=P5.
- **HF traces:** `imo2026-challenge/chankhavu-imo-reasoning-traces` : **`imo2026-bugfix-p145-20260719-231751`** (unique timestamped run_name).
- **Code:** branch `feature/streaming-loop-abort` @ `2635f51` (harness + SGLang patches from this repo).

### Launch (reproduce)

Server (node0):
```bash
opd-run sglp145 bash -c 'source /tmp/chankhavu/venvs/infervenv/.runtime/activate-env.sh && \
  exec python /tmp/chankhavu/imo-inference/evaluation/harness/launch_server.py \
    --config /tmp/chankhavu/imo-inference/config-nii-bugfix-p145.yaml'
```
Submission (after server healthy; `HF_TOKEN` exported before `activate-env.sh` so
the file-based `hf auth login` token survives the `HF_HOME` redirect):
```bash
opd-run subp145 bash -c 'export HF_TOKEN="$(hf auth token 2>/dev/null)"; \
  source /tmp/chankhavu/venvs/infervenv/.runtime/activate-env.sh && \
  exec python /tmp/chankhavu/imo-inference/evaluation/harness/run_submission.py \
    --config /tmp/chankhavu/imo-inference/config-nii-bugfix-p145.yaml \
    --input  /tmp/chankhavu/imo-inference/evaluation/data/imo2026-latex-test.csv \
    --output /tmp/chankhavu/out/imo2026-bugfix-p145-20260719-231751/submission.csv \
    --artifacts-dir /tmp/chankhavu/artifacts/imo2026-bugfix-p145-20260719-231751 \
    --problems 1,4,5'
```
Note: for a *unique* HF trace dir per run, set `traces.run_name` in the config to a
timestamped value (here `imo2026-bugfix-p145-20260719-231751`) and match the
`--output`/`--artifacts-dir` paths.

### Results

_To be filled in on completion (per-problem best self-verifier score; the runner
processes problems sequentially P1 → P4 → P5)._

| Problem | rounds | best mean_verifier_score | notes |
|---|---|---|---|
| P1 | — | — | in progress |
| P4 | — | — | pending |
| P5 | — | — | pending |
