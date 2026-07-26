# OLMo3 no-sink prompt benchmark

This benchmark compares two prompt/parser profiles for
`nguyen599/allenai-Olmo-3.1-32B-Think-aimo-proof-pilot-sft-v2-ckpt-1000`
on IMO 2026 Problems 1, 4, and 5.

## Fixed setup

- Input: `evaluation/data/imo2026-p145.csv`
- Search: medium tournament budget
- Topology: SGLang TP2/DP4 on one 8-GPU NII node per arm
- Model: BF16, plain `Olmo3ForCausalLM`, no attention sinks
- Context: 65,536 tokens
- DFlash: disabled
- Meta verification: not used
- Refinement review deduplication: in-process MinHash-LSH

Both configs are identical except for `search.prompt_profile`, trace labels,
and descriptive comments:

| Arm | Config | Prompt profile |
|---|---|---|
| Default | `config-model-olmo31-sft-v2-ckpt1000-budget-medium-default.yaml` | `ycchen_math_3r` |
| Proof Pilot | `config-model-olmo31-sft-v2-ckpt1000-budget-medium-proof-pilot.yaml` | `proof_pilot_markdown` |

The default arm preserves the upstream XML prompts and parser. The Proof Pilot
arm uses the Markdown proof-generation, verification, and refinement contract
from `aimo-proof-pilot/src/run.py`. The selector remains unchanged so final
selection is controlled across both arms.

The Markdown parser deliberately selects the last `## Solution` section and the
last boxed score. This avoids grading a format example repeated in the model's
reasoning or response before its actual final proof.

## NII launch

The NII runtime is `/tmp/chankhavu/venvs/infervenv`. Download the model once to
the shared `/tmp/models` filesystem, then launch the default arm on node 2 and
the Proof Pilot arm on node 3. Use separate output directories; each node owns
its local port 30000.

Long commands must run under `nohup` with PID, status, and log files, following
`aimo-proof-pilot-inference/NII_CONTROL_PANEL.md`. Do not send credentials
through the Gradio relay.
