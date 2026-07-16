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

## Current validation

- Targeted Ruff checks pass.
- Python compilation checks pass.
- Full repository suite: 138 tests passed, 36 subtests passed.
