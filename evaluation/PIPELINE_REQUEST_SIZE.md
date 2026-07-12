# Pipeline request-size derivation

## Scope

This document derives request sizes for the generate-verify-refine pipeline
under the checked-in serving semantics:

- the local SGLang server context is `C = 262,144` tokens;
- every local request sends `max_completion_tokens = O = 65,536`; and
- the client forwards `O` unchanged.

There is no client-side subtraction, clamp, prompt-size preflight, truncation,
or special overflow handling. SGLang's configured context length is the sole
context enforcement point.

Here, **input payload** means the tokenized chat messages submitted in one LLM
request. **Requested total context** means that input plus the fixed requested
output. HTTP JSON bytes are a separate transport measurement.

## Pipeline structure

For one problem, round 1 generates 32 proofs. Every admitted proof is verified
16 times. Later rounds select the cumulative top eight proofs, create four
refinements from each, and verify every admitted refinement 16 times. There
are at most four rounds.

The largest fan-in is one refinement prompt. It contains:

1. one parent proof;
2. that proof's self-evaluation; and
3. at most eight verifier responses selected from that parent's own 16
   verifications.

It does not contain the parent's parent, earlier verifier sets, or the complete
history of the proof pool.

## Definitions

Let:

- `B_r` be the parent proof plus self-evaluation retained from round `r`;
- `V_{r,i}` be selected verifier response `i` for that parent;
- `F_g` be the fixed generation prompt;
- `F_v` be the verifier wrapper, problem, and chat-template overhead; and
- `F_{r,8}` be the refinement wrapper, problem, candidate markup, chat
  template, and eight empty review wrappers.

Using the live OPD tokenizer on IMO 2025 Problem 1:

| Fixed component | Tokens |
|---|---:|
| Generation prompt, `F_g` | 426 |
| Verifier with an empty candidate, `F_v` | 377 |
| Refiner with an empty parent and eight empty reviews, `F_{r,8}` | 504 |

These fixed counts are problem- and tokenizer-specific. The formulas remain
the same when the counts change.

## Generation request

A generation request has:

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

Only a natural-stop response matching the required XML contract is admitted.
The retained proof and self-evaluation come from the same capped completion,
so structurally:

```text
tokens(B_1) <= O
```

Reasoning returned in the separate `reasoning_content` field is persisted but
is not inserted into verifier or refinement prompts.

## Verification request

Each verifier receives one proof and its self-evaluation:

```text
verification_input <= F_v + tokens(B_r)
                   <= F_v + O
```

For Problem 1:

```text
fixed verifier wrapper       377
parent proof and self-eval 65,536
---------------------------------
maximum verifier input     65,913
requested output           65,536
---------------------------------
requested total           131,449
server context            262,144
```

A maximum-size verification request remains within the server context. Each
parsed verifier response is stored in full and can contribute up to `O`
tokens to a later refinement prompt:

```text
tokens(V_{r,i}) <= O
```

## Refinement request

One refinement receives one parent bundle and at most eight verifier
responses:

```text
refinement_input
  <= F_{r,8} + tokens(B_r) + sum(tokens(V_{r,i}), i=1..8)
  <= F_{r,8} + O + 8O
  <= F_{r,8} + 9O
```

For Problem 1:

```text
parent proof and self-eval     65,536
eight verifier responses      524,288
fixed refinement wrapper          504
-------------------------------------
maximum refinement input      590,328
requested output               65,536
-------------------------------------
requested total               655,864
server context                262,144
```

The structural worst case exceeds the server context. The client still sends
`max_completion_tokens=65,536` unchanged and performs no special handling.
SGLang decides whether to reject the request according to its context policy.

## Required upstream invariant

To guarantee that every request can receive the full fixed output budget
without any clamp, the prompt-construction policy must eventually establish:

```text
prompt_tokens + O <= C
```

With the checked-in values:

```text
prompt_tokens <= C - O
prompt_tokens <= 262,144 - 65,536
prompt_tokens <= 196,608
```

For a Problem 1 refinement, the dynamic parent and review material must
therefore satisfy:

```text
tokens(B_r) + sum(tokens(V_{r,i}), i=1..8)
  <= 196,608 - F_{r,8}
  <= 196,104
```

This is a design obligation for later prompt construction. The current client
does not enforce it, reduce the output budget, truncate material, or catch a
context overflow specially.

## Why four rounds do not increase the structural bound

The refinement output for round `r + 1` is capped again:

```text
tokens(B_{r+1}) <= O
```

Its new verifier responses are independently capped:

```text
tokens(V_{r+1,i}) <= O
```

Therefore every later refinement satisfies the same recurrence:

```text
refinement_input_{r+2} <= F_{r,8} + O + 8O
```

By induction, the structural bound is unchanged for rounds 2, 3, and 4. A
model may copy older material into its new proof, but all copied material must
fit inside the new `O`-token output. The cumulative proof pool affects ranking
only; `refinement_messages()` does not recursively dereference `parent_id`.

## External grader

Each of the 64 grader requests receives only the selected proof plus the
problem, official checkpoints, grading guidelines, and grader instructions.
It does not receive verifier responses or ancestry. Its independent output cap
is also 65,536 tokens.

If `F_grader` is the external model's token count for that fixed material:

```text
grader_input <= O + F_grader
grader_requested_total <= O + F_grader + 65,536
```

The exact accepted context is controlled by the external grader model, not the
local SGLang context setting.

## Concurrency is not one payload

The local semaphore permits 32 independent requests. SGLang does not combine
them into one chat payload. If 32 structural worst-case refinements were
simultaneously submitted:

```text
aggregate input <= 32 * 590,328 = 18,890,496 tokens
aggregate requested total <= 32 * 655,864 = 20,987,648 tokens
```

Those figures describe aggregate submitted work, not one context window. Each
request is independently subject to SGLang's 262,144-token context.

## Tokenization caveat

Generated token IDs are decoded to text and tokenized again when embedded in a
later prompt. Decode-then-encode is not guaranteed to preserve the original
token count exactly, and chat-template boundaries can change tokenization.
Consequently, `9O + F_{r,8}` is structural accounting. The authoritative
prompt size is SGLang's tokenization of the concrete submitted messages.

The client intentionally does not use `/tokenize` inside `chat_raw()` and
does not alter `max_completion_tokens` based on prompt size.
