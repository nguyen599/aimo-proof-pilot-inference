#!/bin/bash
# run_eval.sh — evaluate opd-32b-deploy on the 6 markscheme problems using the
# EXACT agentic loop + prompts from submission-32b-fix4.ipynb.
#
# run_v2.py and the whole proof_agent package (prover/verifier/refiner/selector
# prompt templates included) run UNMODIFIED from the extracted code bundle.
# All loop parameters below are copied verbatim from the notebook's CONFIG cell
# (LOOP dict + BUDGET_HIDDEN); only server location and file paths differ.
#
#   SPLIT=1 (default): 3 problems per GPU, two run_v2 processes (~3.5 h)
#   SPLIT=0          : all 6 sequential on GPU 0's server, as on Kaggle (~7 h)
set -euo pipefail

VENV=/workspace/pp/venv
REPO=/workspace/proof-pilot-code-x
MODEL=/workspace/models/opd-32b-deploy
EVAL=/workspace/eval
SPLIT="${SPLIT:-1}"

# notebook LOOP + BUDGET_HIDDEN, verbatim
LOOP_ARGS=(
  --budget-s "${BUDGET_S:-4200}" --select-reserve "${SELECT_RESERVE:-900}"
  --init-provers 6 --verify-k 3 --refine-inputs 4 --refine-min-seeds 1
  --select-bundle-n 4 --selectors 5 --call-cap 60000 --concurrency 12 --gen-cap 6
  --temperature 1.0 --top-p 0.95 --verify-temp 1.0 --select-temp 1.0
)

run_one() {  # run_one <input_csv> <tag> <base_url>
  "$VENV/bin/python" "$REPO/kaggle_deploy/final/notebook/run_v2.py" \
    --model_path "$MODEL" \
    --input_csv "$1" --output_csv "$EVAL/submission_$2.csv" \
    --logdir "$EVAL/logs_$2" --base-url "$3" \
    "${LOOP_ARGS[@]}"
}

if [ "$SPLIT" = 1 ]; then
  # 3+3 split preserving PDF order; the per-problem loop is identical, problems
  # just run on different (identical) server replicas
  head -n 1 "$EVAL/problems.csv" > "$EVAL/problems_a.csv"
  head -n 1 "$EVAL/problems.csv" > "$EVAL/problems_b.csv"
  "$VENV/bin/python" - <<'PY'
import csv
rows = list(csv.reader(open('/workspace/eval/problems.csv')))[1:]
for name, part in (('a', rows[:3]), ('b', rows[3:])):
    with open(f'/workspace/eval/problems_{name}.csv', 'a', newline='') as f:
        csv.writer(f).writerows(part)
PY
  run_one "$EVAL/problems_a.csv" a http://127.0.0.1:30000 &
  PA=$!
  run_one "$EVAL/problems_b.csv" b http://127.0.0.1:30001 &
  PB=$!
  wait $PA $PB
  # merge in original problem order
  "$VENV/bin/python" - <<'PY'
import csv
answers = {}
for t in ('a', 'b'):
    for r in csv.DictReader(open(f'/workspace/eval/submission_{t}.csv')):
        answers[r['id']] = r['answer']
order = [r['id'] for r in csv.DictReader(open('/workspace/eval/problems.csv'))]
with open('/workspace/eval/submission.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['id', 'answer'])
    for pid in order:
        w.writerow([pid, answers[pid]])
print('merged -> /workspace/eval/submission.csv')
PY
else
  run_one "$EVAL/problems.csv" all http://127.0.0.1:30000
  cp "$EVAL/submission_all.csv" "$EVAL/submission.csv"
fi
echo "[run_eval] done -> $EVAL/submission.csv"
