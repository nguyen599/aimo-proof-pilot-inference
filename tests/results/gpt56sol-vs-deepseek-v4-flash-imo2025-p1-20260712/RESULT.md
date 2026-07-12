# GPT-5.6 Sol versus DeepSeek V4 Flash grader check

## Setup

- Problem: IMO 2025 problem 1 from the pinned MathArena dataset.
- Proof: adversarial partial solution that states the correct answer and gives a
  valid `k=0` construction, but uses an invalid line-tilting construction for
  `k=1` and `k=3` and merely asserts the induction and impossibility steps.
- Expected grade: `1`, following the explicit one-point checkpoint for reaching
  the answer `k=0,1,3`.
- Prompt: the repository's strict `findings`, `grade`, `reasoning` grader prompt.
- Reasoning effort: high for both models.
- Retries: disabled.

GPT-5.6 Sol was called through the OpenAI Responses API with native Pydantic
structured parsing. DeepSeek V4 Flash was called through its Chat Completions
endpoint with JSON mode and the repository's strict ordered parser.

## Results

| Model | Grade | Expected | Latency | Input tokens | Output tokens | Reasoning tokens |
|---|---:|:---:|---:|---:|---:|---:|
| `gpt-5.6-sol` | 1 | yes | 8.179 s | 1,818 | 386 | 118 |
| `deepseek-v4-flash` | 1 | yes | 12.413 s | 1,773 | 972 | 756 |

Both responses conformed to the exact field order and correctly applied the
problem-specific marking scheme.

## Qualitative comparison

GPT-5.6 Sol gave the stronger critique on this sample. It distinguished the
valid vertical-line construction for `k=0` from the invalid constructions for
`k=1` and `k=3`, and explained the central geometric defect: a line through two
of the same lattice points is already determined, so it cannot simply be tilted
while retaining those incidences. It also mapped the omitted work to the
boundary, origin, hypotenuse, reduction, and final-contradiction checkpoints.

DeepSeek V4 Flash reached the same correct grade and correctly identified the
missing construction, induction, and impossibility arguments. Its findings were
sound but less precise about why the proposed tilting operation is impossible.

## Conclusion

For this one adversarial proof, `gpt-5.6-sol` is the better grader: equal scoring
accuracy, more specific mathematical fault localization, lower latency, and
fewer output tokens. This is an indicative result only. A production model
choice should use a balanced calibration set spanning expected grades 0 through
7, multiple mathematical domains, subtle false proofs, and repeated attempts.
