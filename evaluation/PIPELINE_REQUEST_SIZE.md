# Pipeline request-size derivation

## Scope

This document derives request sizes for the generate-verify-refine pipeline
under the checked-in serving semantics:

- the local SGLang server context is `C = 262,144` tokens;
- the first segment of every local logical call uses the configurable
  `max_completion_tokens = O = 65,536`;
- a length-truncated prover or refiner may use one configurable
  `solution_continuation_tokens = R_s = 16,384` native continuation; and
- a length-truncated verifier may use one independently configurable
  `verifier_continuation_tokens = R_v = 16,384` native continuation.

The client does not subtract prompt tokens from any configured output budget,
clamp either budget, truncate prompt material, or perform a context preflight.
SGLang's configured context length is the sole context enforcement point.

Here, **input payload** means the tokenized messages or explicit token IDs
submitted in one physical inference request. **Requested total context** means
that input plus that physical request's output budget. HTTP JSON bytes and
aggregate concurrent work are separate measurements.

## Pipeline structure

For one problem, round 1 makes 32 proof attempts. Every admitted proof is
verified 16 times. Later rounds select the cumulative top eight proofs, choose
the four lowest-rated verifier analyses for each parent, generate one refinement
from each analysis, and verify every admitted refinement 16 times. There are at
most four rounds.

A naturally completed candidate is admitted only when it matches the complete
ycchen XML contract. A length-truncated prover/refiner receives at most one
continuation:

- if `<solution>` has not started, the client appends a finalize instruction,
  `</think>`, and `<solution>` to the token prefix;
- if `<solution>` has started, the client continues the partial visible XML
  without inserting another solution tag; and
- the combined visible output must still match the complete XML contract.

Invalid candidates are disqualified without retries. Hidden thinking is stored
for audit and for the one continuation request only. It is never inserted into a
verifier or later refinement prompt.

A verifier at `length` follows the same one-continuation rule with
`<evaluation>` as its visible opening tag. Naturally malformed verifier XML and
combined output that remains invalid are logged and skipped. A proof is eligible
for ranking with at least the configurable four valid verifier responses; no
replacement calls or synthetic scores are used.

## Definitions

Let:

- `O` be the configured first-segment completion budget, 65,536;
- `R_s` be the configured solution-continuation budget, 16,384;
- `R_v` be the configured verifier-continuation budget, 16,384;
- `L_s = O + R_s = 81,920` be the maximum logical prover/refiner output across both
  physical segments;
- `L_v = O + R_v = 81,920` be the maximum retained verifier output across both
  physical segments;
- `B_r` be the parsed parent proof plus self-evaluation retained from round `r`;
- `V_{r,i}` be one selected verifier response for that parent;
- `F_g` be the fixed generation prompt;
- `F_v` be the verifier wrapper, problem, and chat-template overhead;
- `F_{r,1}` be the refinement wrapper, problem, candidate markup, chat template,
  and one empty review wrapper;
- `F_{cs}` be the solution force-close steering suffix; and
- `F_{cv}` be the verifier force-close steering suffix.

Using the live OPD tokenizer on IMO 2025 Problem 1:

| Fixed component | Tokens |
|---|---:|
| Generation prompt, `F_g` | 426 |
| Verifier with an empty candidate, `F_v` | 377 |
| Refiner with an empty parent and one empty review, `F_{r,1}` | 399 |
| Solution force-close suffix, `F_{cs}` | 51 |
| Verifier force-close suffix, `F_{cv}` | 52 |

These fixed counts are problem-, tokenizer-, and steering-text-specific. The
formulas remain the same when the configurable budgets or fixed counts change.

## Generation request

The first physical generation request has:

```text
input  = F_g
output = O
```

For Problem 1:

```text
input                      426
requested output        65,536
-------------------------------
requested total         65,962
server context         262,144
```

If this request reaches `length` without complete XML, its continuation has the
original prompt and generated prefix as input. The thinking-only force-close is
the larger fixed-overhead case:

```text
continuation_input <= F_g + O + F_{cs}
continuation_total <= F_g + O + F_{cs} + R_s
```

For Problem 1:

```text
original generation prompt       426
first generated prefix        65,536
force-close steering              51
------------------------------------
continuation input             66,013
continuation output            16,384
------------------------------------
continuation total             82,397
server context               262,144
```

A partial solution can span both segments, so structurally:

```text
tokens(B_1) <= L_s = 81,920
```

Reasoning remains in the separate `reasoning_content` artifact and is not part
of `B_1`.

## Verification request

Each verifier receives one parsed proof and its self-evaluation:

```text
verification_input <= F_v + tokens(B_r)
                   <= F_v + L_s
```

For Problem 1:

```text
fixed verifier wrapper       377
parent proof and self-eval 81,920
---------------------------------
maximum verifier input     82,297
requested output           65,536
---------------------------------
requested total           147,833
server context            262,144
```

If a verifier reaches `length` without complete XML, its continuation includes
the original verifier prompt, the first generated prefix, and its role-specific
force-close suffix:

```text
continuation_input <= F_v + L_s + O + F_{cv}
continuation_total <= F_v + L_s + O + F_{cv} + R_v
```

For Problem 1:

```text
fixed verifier wrapper          377
parent proof and self-eval   81,920
first generated prefix       65,536
verifier force-close suffix      52
-----------------------------------
continuation input          147,885
continuation output          16,384
-----------------------------------
continuation total          164,269
server context             262,144
remaining margin            97,875
```

A valid combined verifier response can span both segments:

```text
tokens(V_{r,i}) <= L_v = 81,920
```

## Refinement requests

The first refinement segment receives one parent bundle and one verifier
response:

```text
refinement_input <= F_{r,1} + tokens(B_r) + tokens(V_{r,i})
                 <= F_{r,1} + L_s + L_v
```

For Problem 1:

```text
parent proof and self-eval     81,920
one verifier response          81,920
fixed refinement wrapper          399
-------------------------------------
maximum first input           164,239
first requested output         65,536
-------------------------------------
maximum first total           229,775
server context                262,144
```

If that segment reaches `length` without complete XML, the continuation also
contains the first generated prefix and force-close suffix:

```text
continuation_input <= F_{r,1} + L_s + L_v + O + F_{cs}
continuation_total <= F_{r,1} + L_s + L_v + O + F_{cs} + R_s
```

For Problem 1:

```text
parent proof and self-eval     81,920
one verifier response          81,920
fixed refinement wrapper          399
first generated prefix         65,536
force-close steering               51
--------------------------------------
maximum continuation input    229,826
continuation output            16,384
--------------------------------------
maximum continuation total    246,210
server context                262,144
remaining structural margin    15,934
```

The equivalent three-output check is also below context:

```text
3 * 81,920 + 399 + 51 = 246,210 < 262,144
```

Neither calculation is enforced by a client-side prompt check. SGLang remains
the sole authority over the concrete request.

## Why later rounds do not grow recursively

A prover/refiner's combined output is capped again on every round:

```text
tokens(B_{r+1}) <= L_s
```

Its verifier responses are independently capped:

```text
tokens(V_{r+1,i}) <= L_v
```

The cumulative proof pool affects ranking only. Prompt construction does not
recursively dereference `parent_id`, and hidden thinking is not propagated.
Every later round therefore has the same structural bounds.

## Physical request accounting

The four-round full-width search still has at most 2,176 logical calls:

```text
4 * (32 prover/refiner calls + 32 * 16 verifier calls) = 2,176
```

Every logical call is either a prover/refiner or verifier and can require its
single continuation. If every call does, the physical request ceiling is:

```text
2,176 + 2,176 = 4,352
```

Invalid candidates and early stopping reduce these counts. Artifacts retain the
logical call ID while recording one or two physical request segments.

## External grader

Each of the 64 grader requests receives only the selected proof plus the
problem, official checkpoints, grading guidelines, and grader instructions. It
does not receive verifier responses, ancestry, or hidden thinking.

If `F_grader` is the external model's token count for its fixed material:

```text
grader_input <= L_s + F_grader
grader_requested_total <= L_s + F_grader + 65,536
```

The external grader model controls its own accepted context.

## Concurrency is not one payload

The local semaphore permits 32 independent requests. SGLang does not combine
them into one context window. If 32 structural worst-case refinement
continuations were simultaneously submitted:

```text
aggregate input <= 32 * 229,826 = 7,354,432 tokens
aggregate requested total <= 32 * 246,210 = 7,878,720 tokens
```

Those figures describe aggregate work and KV demand, not one request context.

## Tokenization caveat

Generated token IDs are decoded into `reasoning_content` and `content`, then
retokenized when constructing a native continuation or later prompt.
Decode-then-encode is not guaranteed to preserve the original token count
exactly, and chat-template boundaries can change tokenization. The authoritative
size is SGLang's tokenization of each concrete request.

The client intentionally performs no prompt-size subtraction or output-budget
adjustment based on this retokenization.
