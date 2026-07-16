# SFT-750 thinking-budget handoff experiments

## Goal

Increase the fraction of proof candidates that finish with a parseable final
proof after an initial proof-generation call reaches its thinking budget.
With one refine round configured, the new path summarizes the unfinished round
0 research, resets the context, and spends round 1 on a fresh proof attempt.
The final success metric is not handoff validity alone: round 1 should finish
before its own thinking cutoff and emit valid proof XML.

## Fixed runtime

- Target: `/tmp/models/olmo3-opd-sft-750-vllm`
- Draft: `/tmp/models/dflash-32b-draft-v2test-phaseL`
- Runtime: vLLM 0.25.1
- Per node: 8 H200, `TP=2`, `DP=4`
- Target quantization: online FP8
- KV cache: FP8
- DFlash speculative tokens: 10
- DFlash context cutoff: 65,536
- Model context: 262,144
- Saved source calls:
  `imo2025-full-p16-r1-p2-sft750-parserfix4-20260716T173047Z-partial-20260717`

## Experiment 0: local parser and control-flow validation

Commit `aa3c350` added the handoff/restart path, standalone optimizer, and
focused tests.

Results:

- Parsed all 17 checked-in proof calls containing the old forced-partial marker.
- All 17 calls reached the round-0 unfinished-thinking cutoff. Only 6/17 old
  forced-partial continuations emitted structurally parseable final sections;
  this is a formatting baseline, not evidence of a complete proof.
- Confirmed the marker is absent after reconstruction.
- Confirmed a budget hit with one round remaining calls:
  proof generation, handoff, fresh proof generation, then verification.
- Confirmed the final round retains the old strongest-partial fallback.
- Full repository suite: 112 tests passed, 36 subtests passed.

## Experiment 1: exact SFT-750 tokenizer reconstruction

The first NII tokenizer dry run failed before any GPU server was started:

```text
logged continuation tokens: 121992
re-encoded continuation tokens: 121991
```

Root cause: the saved-call parser used `rstrip()` on the decoded continuation
prompt. The real prompt ended with a significant newline from the forced text,
so reconstruction dropped one token. The parser was changed to remove exactly
the two debug-log section delimiters while preserving prompt trailing
whitespace. A regression test now covers this case.

After preserving whitespace, canonical SFT-750 re-tokenization still used
between zero and three fewer tokens across the 17 contexts. For every context,
decoding the canonical token IDs reproduced the saved prompt text exactly.
This is equivalent tokenizer segmentation rather than parser loss. The
optimizer now requires exact decoded-text equality and separately bounds the
absolute token-count drift to four.

## Experiment 2: handoff prompt and temperature sweep

Planned matrix:

- Eight balanced saved contexts: both ranks, both IMO problems, and both
  parseable/unparseable old forced-partial outcomes.
- Prompt variants: `evidence_first`, `lemma_ledger`,
  `continuation_frontier`.
- Temperatures: `1.0`, `0.7`, `0.6`.
- Total short handoff calls: 72.

Record XML validity, fidelity, mathematical information retained, failed-route
precision, bottleneck quality, next-step usefulness, completion tokens, and
latency. Results will be appended below.

### Server launch attempt 1

Both NII nodes used vLLM 0.25.1 with the fixed runtime above. Target and draft
weights loaded, but startup failed during CUDA-graph memory profiling:

```text
Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:455
'an illegal memory access was encountered'
```

Node 6 also reported one secondary Inductor autotuning failure while the DP
engines were aborting:

```text
Failed to run autotuning code block: CUDA driver error: file not found
```

No handoff requests were submitted. The retry preserves compiled execution,
CUDA graphs, `TP=2`, `DP=4`, FP8 target/KV, and DFlash, but adds
`--disable-custom-all-reduce` and uses new per-node compile-cache directories.

### Live smoke 1: unconstrained handoff length

The retry started successfully on both nodes. One `evidence_first`,
temperature-0.7 handoff was generated from a 122,227-token prompt:

```text
finish_reason=length
completion_tokens=4096
is_valid=false
missing_sections=established,promising,failed,uncertain,bottleneck,next_steps
```

This was not a turn-transition or prefix failure. The output correctly began
with `</think><handoff><established>`, but it used all 4,096 tokens in the first
section. It repeatedly restated the problem and used phrases such as "the
previous attempt attempted" instead of compressing the mathematical state.

The next prompt revision keeps 4,096 as a safety ceiling but requires:

- fewer than 1,200 words total;
- hard bullet limits for every list section;
- at most two sentences per bullet;
- one bottleneck paragraph of at most 120 words;
- no full problem restatement, duplicated facts, or attempt narration;
- all XML tags to be closed.

### Live smoke 2: strict single-call compression

The same source context was retried with the strict compression contract:

```text
prompt_tokens=122377
completion_tokens=4096
finish_reason=length
is_valid=false
```

The output was denser and dropped most attempt narration, but it again remained
inside `<established>` until the token cap. It expanded into small-case
calculations instead of obeying the bullet and section limits. Increasing the
single-call cap would preserve this failure mode and consume more context.

The optimizer is therefore being aligned with the production path in `run.py`:
after an invalid first handoff, it makes one repair turn over the exact prior
context and draft. The repair instruction requires the same mathematical
content to be re-emitted in all six XML sections. Results retain both attempts
and aggregate their token cost, while the repaired artifact is what the restart
evaluation consumes.

### Live smoke 3: production-equivalent repair

The repair turn did not recover the XML:

```text
initial: prompt=122377 completion=4096 finish=length valid=false
repair:  prompt=126579 completion=4096 finish=length valid=false
aggregate completion tokens=8192
```

Both calls remained inside `<established>`. The repair was therefore more
expensive than the original forced-partial fallback without creating a usable
restart artifact.

The next design generates each handoff section independently from the same
pre-force context. Each request has a focused extraction prompt and a hard
section-specific cap:

| Section | Max tokens |
| --- | ---: |
| `established` | 768 |
| `promising` | 640 |
| `failed` | 512 |
| `uncertain` | 384 |
| `bottleneck` | 256 |
| `next_steps` | 384 |

The harness strips any accidental section tag, fills only genuinely empty
sections with an explicit no-information sentence, and assembles the six
sections into valid XML. This bounds total generated handoff content at 2,944
tokens and prevents one category from consuming the whole budget. All section
requests retain the full original reasoning context; vLLM prefix caching can
reuse their common 122K-token prefix.

### Live smoke 4: bounded sections in the original conversation

The six calls completed in 37.2 seconds and produced structurally valid XML:

```text
aggregate prompt tokens=732745
aggregate completion tokens=2838
valid=true
```

This established that prefix caching makes six extraction calls practical, but
manual review rejected the handoff quality:

- five of six sections reached their token cap;
- `established`, `promising`, and `uncertain` restated the problem and mixed in
  new speculative solving;
- `failed` repeated generic failure statements;
- `bottleneck` echoed its extraction instruction;
- `next_steps` was generic and unfinished.

The likely cause is role framing. The 122K-token attempt remains the model's own
previous assistant turn, so even a new user instruction leaves a strong
continuation bias.

### Fresh-context section extraction

The next smoke reframes the attempt as quoted, untrusted research notes in a
new conversation. It:

1. extracts the original problem without the final-answer XML contract;
2. removes an exact consecutive repeated-token suffix when detected;
3. samples eight chronological 4,096-token reasoning windows, at most 32,768
   tokens total;
4. places the focused section instruction after those windows;
5. starts a fresh assistant turn with the section tag already opened.

This reduces each extraction prompt from roughly 122K to roughly 33K tokens,
keeps early, middle, and late research visible, and moves the current
instruction into a clean role hierarchy.

### Live smoke 5: bounded sections from fresh sampled context

The fresh-context design sampled eight evenly spaced 4,096-token windows from
the unfinished reasoning:

```text
window ranges:
[0,4096], [16776,20872], [33553,37649], [50329,54425],
[67106,71202], [83882,87978], [100659,104755], [117435,121531]
aggregate prompt tokens=199705
aggregate completion tokens=2944
latency=27.0s
valid=true
```

No exact consecutive repeated-token suffix was detected. Structurally this was
cheaper and reliable, but manual review still rejected the content:

- all six sections reached their individual token caps;
- `established` mostly restated the problem and introduced a conjectured sunny
  line bound rather than extracting only proved state;
- `promising` expanded into new small-`n` enumeration;
- `failed` continued solving instead of identifying exact prior obstructions;
- `uncertain` largely restated the task;
- `bottleneck` and `next_steps` remained generic.

Fresh role framing alone therefore does not prevent the model from treating a
large collection of raw reasoning excerpts as a request to resume proof search.

### Map-reduce extraction plan

The next design separates evidence recovery from handoff organization:

1. keep the same eight chronological 4,096-token windows;
2. summarize each window independently at temperature `0.2`, with a 320-token
   cap and an extract-only contract;
3. feed only the eight short digests, plus the original problem, into the final
   handoff call;
4. test final-handoff temperatures `1.0`, `0.7`, and `0.6`;
5. retain every window prompt, digest, final prompt, raw completion, token
   count, and latency in the run artifacts.

This map stage is intentionally low-temperature because its job is faithful
state extraction, not proof search. Temperature variation is reserved for the
small reduce stage. A configuration is accepted only if its handoff is both
valid and useful under manual review, then improves fresh round-1 proof
completion before the next thinking cutoff.

### Live smoke 6: first map-reduce implementation

One `evidence_first`, temperature-0.7 case completed quickly:

```text
aggregate prompt tokens=38571
aggregate completion tokens=2678
latency=16.6s
final reduce prompt tokens=3340
finish_reason=stop
valid=false
```

The design was rejected for two independent reasons:

- every one of the eight 320-token map calls reached its length cap and mostly
  described or restated the problem instead of extracting state;
- the monolithic reducer copied the XML schema descriptions, closed
  `<established>` as `</handoff>`, and therefore produced invalid output.

Inspection of the exact saved source explains why even chronological sampling
was poor. For this case, useful reasoning occupies only the first roughly
8,000 characters. The remaining more than 230,000 characters repeatedly emit
variants of:

```text
(1,?) all; (2,2) missing;
```

The exact fixed-block repetition detector missed this variable-alignment loop.
The next revision therefore:

1. detects the first pair of low-novelty token windows and discards the tail
   before window selection;
2. omits the full original problem from map calls;
3. requires at most five typed extractive lines (`P`, `A`, `F`, `U`, `N`) and
   deterministically drops nonconforming prose;
4. reduces each final handoff section independently from the typed digests;
5. assembles final XML in code, so one malformed generation cannot invalidate
   the handoff.

### Live smoke 7: low-novelty truncation and typed digests

The revised extractor identified the repetitive tail with the real SFT-750
tokenizer:

```text
reasoning_original_tokens=121531
low_novelty_start=2560
reasoning_cleaned_tokens=2560
window_ranges=[[0,2560]]
aggregate prompt tokens=4261
aggregate completion tokens=1408
latency=6.0s
```

This confirms that low-novelty truncation removes the pathological loop and
makes extraction inexpensive. The configuration is still rejected:

- the one 160-token map completion did not emit any required typed line, so the
  deterministic parser correctly replaced it with a no-progress marker;
- the six section reducers ignored that empty evidence and hallucinated
  unrelated problems about sums of squares, functional equations, reciprocal
  inequalities, and harmonic numbers;
- deterministic XML assembly made the artifact structurally valid, but its
  mathematical fidelity was zero.

The next source is the already-generated strongest-partial continuation after
the old forced `</think>` marker. For this case it is roughly 13,000 characters
and contains concrete constructions, small-`n` checks, and explicit gaps. It is
far cleaner than the preceding repetitive hidden reasoning. The next smoke
quotes that partial-progress report in a fresh context, extracts six bounded
sections independently, and still treats every carried claim as untrusted.

### Live smoke 8: section extraction from forced partial progress

The sectioned extractor used the 4,462-token forced partial-progress report
rather than the repetitive hidden reasoning:

```text
partial_progress_chars=12947
partial_progress_tokens=4462
aggregate prompt tokens=28213
aggregate completion tokens=1210
latency=7.3s
valid=true
```

This source materially improved relevance: every section stayed on the sunny
lines problem and retained concrete small-`n` constructions. The generated
handoff is still rejected because the reducer changed mathematical state:

- `established` says that `n=3, k=1` is possible, while `failed` says it is
  impossible;
- speculative bounds on sunny-line coverage are promoted into apparent facts;
- five of six section calls reached their token caps and ended mid-argument;
- the same uncertain `n=4, k=1` construction is repeated with different
  confidence levels across sections.

The failure is therefore semantic rather than structural. Changing reducer
temperature cannot guarantee fidelity. The next baseline carries the complete
forced partial report losslessly as explicitly untrusted context, with
deterministic guard sections around it. The fresh solver, rather than another
summarizer, is responsible for auditing its claims.

### Live comparison: reducer temperatures versus lossless passthrough

The same forced partial report was tested with reducer temperatures `1.0`,
`0.7`, and `0.6`, plus a deterministic lossless passthrough:

| Mode | Temperature | Completion tokens | Latency | Structural result |
|---|---:|---:|---:|---|
| partial section reducer | 1.0 | 1,151 | 6.2 s | valid XML |
| partial section reducer | 0.7 | 1,230 | 11.2 s | valid XML |
| partial section reducer | 0.6 | 1,248 | 6.5 s | valid XML |
| lossless passthrough | n/a | 0 | 0.03 s | valid XML |

All reducer temperatures are rejected:

- temperature `1.0` states unproved sunny-line capacity bounds as established;
- temperature `0.7` starts an `established` bullet by claiming `n=4, k=1`,
  then retracts it inside the same bullet;
- temperature `0.6` contradicts itself between `established` and `failed` on
  whether `n=3, k=1` is possible;
- each temperature hit the completion cap in at least five of six sections and
  ended several mathematical statements mid-sentence.

The lossless mode preserved all 12,947 report characters inside an explicitly
untrusted block and made no new mathematical claim. It is the only acceptable
handoff candidate from this comparison. The fresh-round experiment compares it
against an empty restart control to distinguish useful carried state from the
benefit of merely resetting context.

### Live comparison: structured force directly from cutoff context

A final prompt design was tested without regenerating the original proof. It
removed the old force suffix from the saved 122K-token context, appended a
strict extract-only ledger prefix with `VERIFIED`, `UNVERIFIED`, `FAILED`,
`BOTTLENECK`, and `NEXT` headings, and sampled at three temperatures:

| Temperature | Completion tokens | Finish | Result |
|---:|---:|---|---|
| 1.0 | 4 | stop | immediately emitted `</solution>` |
| 0.7 | 4,096 | length | resumed proof search and hit the cap |
| 0.6 | 304 | stop | emitted a generic incomplete solution and score 0 |

None followed the requested research-ledger contract:

- temperature `1.0` treated the forced prefix as a complete answer and closed
  it without transferring state;
- temperature `0.7` restated the problem and continued a long small-`n`
  enumeration instead of summarizing;
- temperature `0.6` produced a generic list of observations, guessed that the
  answer might be `0` through `n`, and explicitly admitted that no proof was
  present.

The cutoff context is therefore too strongly conditioned toward completing the
original assistant answer for a structured force prompt to be reliable. The
old bounded strongest-partial continuation is imperfect but materially more
relevant. Preserve it losslessly, mark it untrusted, and perform all auditing
inside a fresh proof attempt.

## Experiment 3: fresh round-1 proof completion

After selecting the strongest handoff configuration, restart fresh proof
generation from the original problem plus handoff. Record:

- final `finish_reason`;
- whether the final thinking budget was reached;
- presence of `</think>`;
- parseable `<solution>`, `<self_evaluation>`, and `<score>`;
- proof character and token counts;
- whether verification can start.

This experiment determines whether the handoff improves proof completion rate,
rather than only producing cleaner summaries.

Implementation:

- `evaluate_thinking_handoff_restart.py` consumes one selected
  prompt/temperature group from the short handoff sweep.
- It inserts the handoff into the original rendered problem prompt before the
  assistant generation marker, then starts a fresh round-1 completion.
- It stops an unfinished reasoning stream at 122,000 tokens without applying
  the old strongest-partial force text. If `</think>` was genuinely emitted
  before that boundary, it lets the visible proof continue up to 126,000 total
  completion tokens.
- It records the raw prompt, handoff, output, finish reason, real cutoff state,
  parseable proof state, proof length, latency, and throughput for every case.
- The focused handoff suite passes with 31 tests after adding rendered-prompt
  insertion, restart-budget, and forced-finalization coverage.

### Fresh restart controls

The lossless handoff and an empty restart control were each tested on the same
saved sunny-lines problem with a 122,000-token reasoning boundary and 126,000
total completion tokens:

| Restart input | Prompt tokens | Completion tokens | Throughput | Closed thinking | Parseable proof |
|---|---:|---:|---:|---|---|
| lossless untrusted partial report | 5,199 | 122,036 | 136.34 tok/s | no | no |
| empty restart control | 631 | 122,070 | 140.19 tok/s | no | no |

Both fresh attempts reached the reasoning boundary without emitting
`</think>`. The empty control continued broad case enumeration. The lossless
handoff recovered more concrete construction work near its tail, but it also
continued researching until cutoff and therefore produced no visible
`<solution>`.

This comparison rules out two simple hypotheses:

- resetting context alone does not make this checkpoint voluntarily finish;
- carrying the previous partial report losslessly does not, by itself, reserve
  output budget.

The next test changes only the deadline policy. It caps fresh reasoning at
100,000 tokens, then appends an explicit `</think><solution>` transition and
spends the remainder of a 125,000-token completion allowance on the visible
proof. This leaves approximately 25,000 tokens for a rigorous answer while
keeping the full lossless handoff and deadline-aware restart instruction.

Implementation commit: `46af604` (`Reserve output budget in proof restarts`).
The evaluator records whether forced finalization ran and the number of tokens
generated after the transition. A behavioral unit test verifies that the
second request receives exactly
`max_tokens - reasoning_tokens - force_text_tokens`.

### Live result: deadline-aware restart without reserved output

The lossless handoff was also tested with a stronger deadline-aware restart
instruction, while retaining the original 122,000-token thinking boundary and
126,000-token total completion allowance:

```text
prompt_tokens=5199
completion_tokens=124073
finish_reason=stop
tokens_per_second=131.81
closed_thinking=true
parseable_proof=true
proof_chars=6333
self_score=1
```

This is a formatting completion improvement over both fresh controls, but it
is not a rigorous-proof success. The generated answer explicitly says that the
impossibility of `k=2` and `k>=4` is only sketched and appeals to an unstated
"official solution" for the missing combinatorial argument. It then
incorrectly assigns itself score `1`.

The measured outcomes for this case are therefore:

| Metric | Result |
|---|---:|
| Parser-valid XML | 1/1 |
| Reached a visible proof | 1/1 |
| Self-reported complete | 1/1 |
| Manually rigorous proof | 0/1 |

The deadline wording can induce a late transition out of thinking, but with
only about 2,000 tokens left it encourages an unsupported proof sketch. It is
not selected as the production policy by itself. The reserved-output
experiment is required to determine whether an earlier transition leaves
enough space to close the missing arguments honestly.

### Live result: force finalization after 100,000 reasoning tokens

The first reserved-output policy forced the transition after 100,000 reasoning
tokens within a 125,000-token allowance:

Artifact:
`/tmp/thinking-handoff-restart-passthrough-force100k-sft750-20260717`

```text
prompt_tokens=5282
completion_tokens=108782
finish_reason=stop
hit_unfinished_thinking_budget=true
budget_forced_finalization=true
forced_finalization_tokens=8710
tokens_per_second=143.54
parseable_proof=true
proof_chars=7186
self_score=1
```

The forced transition worked exactly as implemented, and the model stopped
well before exhausting the available visible-output budget. The proof is still
rejected under manual mathematical review:

- the `k=3`, `n=3` construction uses a line of slope `-1` while calling it
  sunny;
- the description of the six uncovered points first lists an incorrect
  variable-length row;
- the `k>=4` impossibility argument claims that the triangular lattice has no
  three collinear points on a non-forbidden slope, which is false in general;
- maximal coverage by mixed non-sunny line families is asserted without a
  sufficient extremal proof;
- the self-evaluation calls these steps complete and rigorous despite the
  errors.

Measured outcome:

| Metric | Result |
|---|---:|
| Forced transition executed | 1/1 |
| Parser-valid XML | 1/1 |
| Self-reported complete | 1/1 |
| Manually rigorous proof | 0/1 |

Reserving 25,000 output tokens is therefore sufficient for formatting, but the
model's late proof synthesis inherits unchecked claims from its long search.
The next comparison moves the forced transition earlier and strengthens the
finalization instruction: audit every construction and impossibility claim,
never appeal to an omitted or official solution, and prefer an honest partial
proof with score `0` or `0.5` over a false claim of completeness.

### Live result: stricter audit at 80,000 and 60,000 reasoning tokens

The next two runs kept the same problem, lossless handoff, 125,000-token total
completion allowance, temperature `1.0`, and top-p `0.95`. They used the
stricter final audit instruction and changed only the forced-transition
boundary:

Artifacts:

- 80,000:
  `/tmp/thinking-handoff-restart-force80k-auditv2-sft750-20260717`
- 60,000:
  `/tmp/thinking-handoff-restart-force60k-auditv2-sft750-20260717`

| Reasoning boundary | Completion tokens | Forced visible tokens | Finish reason | Closed `solution` | Valid XML |
|---|---:|---:|---|---|---|
| 80,000 | 125,000 | 44,822 | `length` | no | no |
| 60,000 | 125,000 | 64,861 | `length` | no | no |

Both runs emitted `</think><solution>` at the configured boundary, proving that
the control flow and reserved-token accounting worked. Neither emitted
`</solution>`, `<self_evaluation>`, or `<score>`. Their visible output continued
open-ended case analysis until the full 125,000-token request limit.

Moving the transition earlier therefore made formatting completion worse on
this case. The model did not use the larger visible-output reserve to synthesize
and audit a bounded proof; it resumed exploratory solving inside the
`solution` section.

Combined result for the tested forced boundaries:

| Boundary | Parser-valid proof | Manually rigorous proof |
|---|---:|---:|
| 100,000 | 1/1 | 0/1 |
| 80,000 | 0/1 | 0/1 |
| 60,000 | 0/1 | 0/1 |

No tested boundary increased rigorous proof completion on the sunny-lines
case. The 100,000-token boundary is the only parser-valid result, but its proof
contains fatal mathematical errors. The final-round budget remains opt-in
rather than becoming a production default.

## Current decision

- Select `lossless_partial` over model-written handoffs. Across temperatures
  `1.0`, `0.7`, and `0.6`, model reducers were structurally usable only after
  substantial repair and could introduce contradictions or lose mathematical
  state.
- Keep `deadline_aware` available for fresh proof restarts.
- Keep `thinking_budget_final_round_tokens=0` as the backward-compatible
  default. Values `60,000`, `80,000`, and `100,000` did not produce a rigorous
  proof in the live comparison.
- Treat parser validity and self-reported score as insufficient. A run counts
  as complete only when the XML closes and the mathematical proof survives
  external review.
- The next promising experiment is a separate bounded synthesis call over the
  lossless research record, rather than moving the same-context force boundary
  again.

## Experiment 4: preserve verifier-guided refinement

Code review found that `solve_round_idx` served two purposes: it counted
thinking-budget restarts and was also returned as `consumed_refine_rounds`.
Consequently, with `refine_rounds=1`, one context-reset restart left no
verifier-guided proof refinement after the restarted proof. The 100,000-token
run was verified only after its sole refinement allowance had already been
spent on the restart.

The opt-in `thinking_budget_handoff_preserve_refine_rounds` setting separates
these budgets without changing existing defaults:

- the handoff still permits at most the configured number of restart attempts;
- `budget_restart_count` records the real restart count;
- preserved refinement starts verification at round 0;
- a low verifier score with a validated critique can trigger `proof_refine`;
- the refined proof is verified again at round 1.

Focused validation executes this exact sequence:

```text
proof_generation
proof_handoff
proof_generation
proof_verify
proof_refine
proof_verify
```

The live experiment will use the lossless handoff, deadline-aware restart,
100,000-token final reasoning boundary, one verifier-guided refine round, and
the same SFT-750 runtime. This tests whether verification can repair the
parser-valid but mathematically incorrect restarted proof instead of spending
the only refinement allowance on context reset.

`evaluate_thinking_handoff_refinement.py` replays the saved 100,000-token
restart directly, so this experiment measures only verification and
refinement. It does not pay for or introduce sampling variance from another
long proof-generation call.

### Baseline replay result

Artifact:
`/tmp/thinking-handoff-refinement-replay100k-sft750-20260717`

The saved 100,000-token restart was checked by four verifiers and four
meta-verifiers. All four low-score critiques were validated. The aggregate
score was `0.125`, status was `validated_low_score`, and the critique set
identified both the invalid `n=3, k=3` construction and the false claim that a
sunny line contains at most two points of the residual triangle.

The verifier-guided refinement did run, but it repeated the original failure
mode:

```text
proof_refine completion_tokens=65000
finish_reason=length
has_solution=false
has_self_evaluation=false
proof_chars=0
```

The tail was still exploring whether `k=5` might be possible. Because the
refinement was parser-invalid, it was not re-verified and round 0 remained the
selected proof:

| Metric | Result |
|---|---:|
| Initial proof verifier calls | 4 |
| Validated low-score critiques | 4 |
| Refinement calls | 1 |
| Refinement parser-valid | 0/1 |
| Refinement reached visible proof | 0/1 |
| Selected verification round | 0 |
| Final aggregate score | 0.125 |
| Manually rigorous proof | 0/1 |

This isolates a second cutoff: preserving the verifier-guided round is
necessary but not sufficient, because the refiner itself can spend its entire
completion allowance in hidden reasoning.

### Experiment 5: lossless refinement restart

The next opt-in policy applies the same context-reset mechanism inside one
logical refinement round:

1. Stop the first refinement at 50,000 of 65,000 tokens.
2. Use the remaining allowance to emit an untrusted partial-progress report.
3. Start a fresh refinement with the original proof, selected verifier
   critiques, and the lossless report.
4. Force the fresh refinement out of thinking at 50,000 tokens, leaving about
   15,000 tokens for closed XML.
5. Re-verify only if the repaired proof parses.

The feature is disabled by default and controlled by
`thinking_budget_refine_handoff_enabled`,
`thinking_budget_refine_tokens`,
`thinking_budget_refine_final_round_tokens`, and
`thinking_budget_refine_max_restarts`. Trace records distinguish the stopped
attempt, deterministic handoff, and final refinement. A focused test verifies:

```text
proof_generation
proof_verify
proof_refine
proof_refine_finalize
proof_refine
proof_verify
```

#### Live result: 50,000-token refinement restart

The exact sunny-lines replay completed successfully at the process level, but
the repaired proof remained parser-invalid:

| Metric | Result |
|---|---:|
| Initial refinement completion tokens | 50,089 |
| Initial refinement finish reason | `thinking_budget_reached` |
| Lossless handoff characters | 41,721 |
| Fresh refinement completion tokens | 65,000 |
| Fresh refinement forced boundary | 50,000 |
| Visible-proof reserve | about 15,000 tokens |
| Fresh refinement finish reason | `length` |
| Closed `solution` section | no |
| Re-verification calls | 0 |
| Selected verification round | 0 |
| Final aggregate score | 0.0 |

The fresh call obeyed the control: it left hidden reasoning at the configured
boundary and started a structured proof. It emitted an opening `<solution>`
tag but did not close the solution before the 65,000-token request limit. This
isolates the remaining failure to token allocation rather than handoff
dispatch or force-text application.

### Experiment 6: resume with an earlier final boundary

Reuse the exact saved 41,721-character handoff and the same validated
critiques, but force the final refinement out of thinking after 20,000 tokens:

| Setting | Value |
|---|---:|
| Final completion allowance | 65,000 |
| Final thinking boundary | 20,000 |
| Approximate visible-proof reserve | 45,000 tokens |
| Temperature | 1.0 |
| Top-p | 0.95 |

The resume harness skips the initial verifier, first refinement, and handoff
generation. This makes the comparison test only whether a larger visible
answer reserve improves parser completion and verified mathematical quality.

#### Live result: 20,000-token final boundary

The earlier boundary fixed completion but not correctness:

| Metric | Result |
|---|---:|
| Completion tokens | 57,647 |
| Forced thinking boundary | 20,000 |
| Finish reason | `stop` |
| Parser-valid XML | yes |
| Refined proof characters | 6,272 |
| Model self-score | 1 |
| Verifier scores | 0, 0, 0, 0 |
| Validated low-score critiques | 3/4 |
| Meta scores | 1, 0.5, 1, 1 |
| Final aggregate score | 0.0 |
| Selected verification round | 1 |
| Manually rigorous proof | 0/1 |

All verifier and meta calls ended with `stop`, not `length`. The four
verifiers independently found fatal errors:

- The claimed reduction from arbitrary non-sunny lines to the first
  horizontal rows is unsupported and includes a false set-containment claim.
- The assertion that a non-integer-slope line meets the lattice triangle in at
  most two points is false.
- The proposed `k=3` construction misses required points.
- The proof does not analyze `k=2`, despite claiming an exhaustive answer.

The result demonstrates a format-completion gain: this sample changed from an
unclosed refinement at a 50,000-token boundary to a closed, re-verifiable
refinement at 20,000. It does not demonstrate a solved-proof gain. The model
also assigned itself score 1 despite the explicit finalizer instruction to
report unresolved gaps honestly.

### Experiment 7: restate critiques after the handoff

In the failed 20,000-token run, the selected verifier critiques appeared
before the 41,721-character handoff. The final proof then repeated the exact
claims those critiques rejected. The next prompt-ordering experiment repeats
the selected critiques after the handoff as mandatory repair obligations,
immediately before the fresh refinement begins. The final call must either
repair each item or mark it unresolved and avoid claiming a complete proof.

#### Live result: post-handoff critique restatement

This change regressed completion:

| Metric | Result |
|---|---:|
| Prompt tokens | 19,973 |
| Completion tokens | 65,000 |
| Forced thinking boundary | 20,000 |
| Finish reason | `length` |
| Open `solution` tags | 1 |
| Closed `solution` tags | 0 |
| Parser-valid XML | no |
| Re-verification calls | 0 |
| Selected verification round | 0 |

The final visible section continued exploratory work on the rejected reduction
and `k=2` until the request limit. Repeating the full critiques near the
generation boundary did not produce a concise repair and removed the
format-completion gain from Experiment 6. The change is therefore not retained
in the production restart prompt.

### Experiment 8: final-refinement temperature sweep

Keep the parser-successful Experiment 6 prompt and 20,000-token boundary, then
run final refinements at temperatures 0.7 and 0.6. The two runs reuse the same
saved handoff and critiques. They test whether lower sampling variance improves
voluntary closure or verifier score without changing token allocation.

#### Live result

| Final refinement temperature | Completion tokens | Finish reason | Parser-valid | Aggregate verifier score | Manually rigorous |
|---:|---:|---|---:|---:|---:|
| 1.0 | 57,647 | `stop` | yes | 0.0 | 0/1 |
| 0.7 | 65,000 | `length` | no | not run | 0/1 |
| 0.6 | 24,678 | `stop` | yes | 0.0625 | 0/1 |

Temperature `0.6` was the most token-efficient format completion on this one
hard case. Its 13,407-character proof received verifier scores
`0, 0.5, 0, 0` and meta-verifier scores `0, 0.5, 1, 1`. It still contained
fatal gaps:

- It did not prove the claimed extremal bound for mixtures of non-sunny line
  orientations.
- It replaced an arbitrary line cover by horizontal rows without a valid
  reduction.
- Its `k=2` argument assumed the complement was exactly a smaller lattice
  triangle.
- Its `n >= 6` case implicitly assumed `n-k >= 2` for every `k >= 4`.
- It omitted `n=4`.
- Its special `n=3, k=3` construction called a slope `-1` line sunny.
- It cited a "standard geometric argument" in place of the missing key proof.

The sweep therefore improved XML completion from `1/3` at the original
50,000-token final boundary to `2/3` at the 20,000-token boundary, but rigorous
proof completion remained `0/3`. Temperature `0.6` is exposed as an
independent final-refinement setting so it does not alter proof generation,
initial refinement, verification, or meta-verification sampling. It remains
opt-in pending a multi-problem completion-rate study.

Use:

```bash
python evaluation/harness_vllm/run.py \
  ... \
  --thinking-budget-refine-handoff-enabled \
  --thinking-budget-refine-tokens 50000 \
  --thinking-budget-refine-final-round-tokens 20000 \
  --thinking-budget-refine-final-temperature 0.6
```

#### Independent-temperature replay

A subsequent replay held the normal pipeline temperature at `1.0` and applied
`0.6` only to the final post-handoff refinement. This removes the earlier
confound where the verifier and meta-verifier also inherited temperature
`0.6`.

Artifact:
`/tmp/thinking-handoff-refinement-resume20k-finaltemp06-sft750-20260717`

| Metric | Result |
|---|---:|
| Base pipeline temperature | 1.0 |
| Final refinement temperature | 0.6 |
| Completion tokens | 65,000 |
| Forced thinking boundary | 20,000 |
| Finish reason | `length` |
| Open `solution` tags | 1 |
| Closed `solution` tags | 0 |
| Parser-valid XML | no |
| Re-verification calls | 0 |
| Selected verification round | 0 |

The model spent the entire approximately 45,000-token visible reserve
continuing exploratory case analysis inside `<solution>`. Thus temperature
`0.6` is not a deterministic completion fix: across the two runs that used
`0.6` for final refinement, one closed and one did not.

### Experiment 9: bound visible proof writing

The next opt-in prompt targets the failure observed above rather than moving
the hidden-reasoning boundary again. After a refinement handoff, it tells the
model to:

- finish all search during hidden reasoning;
- write only finalized claims after `<solution>` begins;
- target a complete visible response within 12,000 tokens;
- always close all three XML sections;
- stop early with an honest partial proof and score 0 or 0.5 when a key lemma
  remains missing.

It is controlled by
`thinking_budget_refine_visible_output_target_tokens` and is disabled at `0`.
The target is a prompt contract, not a hard truncation; the external 65,000
completion limit remains unchanged.

#### Prompt-only target replay

The replay used the independent final-refinement temperature from the previous
experiment and added a 12,000-token visible-output target.

Artifact:
`/tmp/thinking-handoff-refinement-resume20k-finaltemp06-visible12k-sft750-20260717`

| Metric | Result |
|---|---:|
| Base pipeline temperature | 1.0 |
| Final refinement temperature | 0.6 |
| Forced thinking boundary | 20,000 |
| Visible-output prompt target | 12,000 |
| Completion tokens | 65,000 |
| Finish reason | `length` |
| Open `solution` tags | 1 |
| Closed `solution` tags | 0 |
| Parser-valid XML | no |
| Re-verification calls | 0 |
| Selected verification round | 0 |

The model ignored the soft target and spent the full remaining completion
budget on exploratory combinatorics inside `<solution>`. This rules out a
prompt-only token target as a reliable cutoff defense. The next experiment
must enforce the visible-output boundary in the streaming client and append an
explicitly incomplete, score-0 XML footer when the model does not close the
response itself. That can improve structural completion but must not be
reported as a solved proof.

### Experiment 10: enforced visible-output boundary

The runner now has a separate opt-in
`thinking_budget_refine_visible_output_limit_tokens` control. It applies only
to the final refinement after a successful handoff. The streaming client stops
the post-thinking visible segment at the configured boundary. If the output is
already parser-valid it is left unchanged. Otherwise the client:

1. preserves all generated proof text;
2. states that the proof is incomplete;
3. closes `<solution>` and `<self_evaluation>`;
4. emits `<score>0</score>`; and
5. records the intervention in usage metadata.

This control is intended to separate two metrics:

- **structural completion:** whether the candidate reaches parseable XML
  without the external 65,000-token cutoff;
- **rigorous completion:** whether a verifier accepts the mathematical proof.

Only the second metric is evidence that the handoff improved solving. Forced
score-0 closure counts as structural completion and rigorous failure.

#### Live result: 12,000-token hard boundary

Artifact:
`/tmp/thinking-handoff-refinement-resume20k-finaltemp06-hardvisible12k-sft750-20260717-v1`

| Metric | Result |
|---|---:|
| Base pipeline temperature | 1.0 |
| Final refinement temperature | 0.6 |
| Forced thinking boundary | 20,000 |
| Visible-output prompt target | 12,000 |
| Visible-output hard limit | 12,000 |
| Completion tokens | 23,805 |
| Finish reason | `stop` |
| Parser-valid XML | yes |
| Refined proof characters | 10,632 |
| Model self-score | 1 |
| Hard limit reached | no |
| Forced score-0 closure | no |
| Verifier scores | 0, 0, 0, 0 |
| Meta-verifier scores | 0.5, 1, 0.5, 0.5 |
| Aggregate verifier score | 0.0 |
| Selected verification round | 1 |
| Structurally complete | 1/1 |
| Manually rigorous | 0/1 |

The final refinement stopped naturally after approximately 20,000 hidden
reasoning tokens and 3,805 visible tokens. Consequently,
`visible_output_limit_applied=false` and
`visible_output_forced_partial_closure=false`: the hard guard did not alter
this sample. This is a useful end-to-end check that enabling the guard leaves
a naturally completed response unchanged, but it is not live evidence for the
forced-closure branch.

All four verifiers rejected the proof. Their common objections were:

- the claimed maximum coverage by mixed non-sunny lines is unsupported and
  has counterexamples;
- the proof replaces an arbitrary uncovered set with a fixed triangular set
  without a valid reduction;
- the arguments for `k=2` and `k>=4` therefore do not apply to arbitrary line
  configurations; and
- the stated `n=3, k=3` construction incorrectly calls a slope `-1` line
  sunny.

The same prompt settings produced an unclosed 65,000-token response in the
previous replay. Since the hard limit was not reached here, the difference is
sampling variance and cannot be attributed to the guard. Across all final
refinement temperature and boundary experiments on this problem, structural
completion improved in some runs, but rigorous proof completion remains zero.
The hard limit is retained as an opt-in safety mechanism for producing an
honest trainable partial trace; it is not a proof-quality improvement.

## Current validation

- Targeted Ruff checks pass.
- Python compilation checks pass.
- Focused handoff and CLI suites: 73 tests passed.
- Full repository suite: 145 tests passed, 36 subtests passed.
