# Proof-checkpoint versus lossless handoff

## Goal

Compare the experimental `proof_checkpoint` handoff with the production
`lossless_partial` handoff on the same unfinished proof context. The comparison
tests both downstream proof quality and the extra inference cost introduced by
checkpoint extraction and audit.

## Setup

- Source: `rank0/llm_calls/2/cand_6_proof_gen_r0.txt` from IMO 2025 problem 2
- Model: OLMo3 OPD SFT-750 with online FP8 weights and FP8 KV cache
- Runtime: vLLM 0.25.1, TP2, no DFlash for the successful matched runs
- Proof temperature: `1.0`
- Restart strategy: `deadline_aware`
- Thinking cutoff: 100,000 tokens
- Finalization: forced after the restarted proof also reached its cutoff
- Checkpoint extraction temperature: `0.6`
- Quality grader: `openai/gpt-5.6-sol`, high reasoning, two attempts per proof
- Grading rubric: the official 7-point MathArena rubric for IMO 2025 problem 2

The proof-checkpoint handoff was selected from the earlier 24-job extraction
sweep. The baseline uses the exact unfinished partial text from the same source
without another model call.

## Results

| Metric | `proof_checkpoint` | `lossless_partial` |
| --- | ---: | ---: |
| Handoff model tokens | 248,720 | 0 |
| Fresh audit tokens | 2,116 | 0 |
| Restart prompt tokens | 1,592 | 1,335 |
| Restart completion tokens | 101,567 | 102,784 |
| Total post-cutoff tokens | 353,995 | 104,119 |
| Restart latency | 999.4 s | 1,011.4 s |
| Decode throughput | 101.626 tok/s | 101.624 tok/s |
| Forced finalization | yes | yes |
| Parsed proof length | 4,281 chars | 8,603 chars |
| Model self-score | 0.5 | 0.5 |
| Grader attempts | 0, 0 | 0, 0 |
| Mean MathArena score | **0/7** | **0/7** |

The checkpoint path consumed 3.40 times as many post-cutoff tokens as the
lossless path, an extra 249,876 tokens. Even if the audit is excluded, it used
3.38 times as many tokens.

## Quality analysis

The checkpoint did not preserve a reusable proved result. It relabeled the
entire target theorem as `[PROVED_LEMMA L1]`, cited unproved polar, Miquel, and
coaxality claims, and explicitly said that the detailed algebra was omitted.
The original structural parser accepted the wrapper, but a fresh-context audit
returned `REPROVE` and `OVERALL: FAIL`. Commit `8db529a` tightens the structural
gate so this explicit omission is now rejected before restart.

The restarted checkpoint proof remained incomplete. It asserted without proof
that the circumcenter of `BEF` has x-coordinate `d/2`, left an unresolved `?`
inside a displayed formula, normalized both `d=1` and `r=1` using only one
scaling degree of freedom, and omitted the final algebra.

The lossless proof was longer and retained more useful local work, including
the perpendicularities `MP perpendicular AC` and `NP perpendicular AD` and a
coordinate setup for `P` and `H`. It still failed: its expression for
`|v|^2` omitted an `a^2` factor, its claimed circumcenter formula was false,
and the final distance identity was only asserted after a "straightforward,
lengthy" simplification.

`numeric_counterexample.py` evaluates a valid configuration with
`d=1, r=1, R=1.5`. The theorem's tangency identity holds numerically, while
neither generated circumcenter is equidistant from `B`, `E`, and `F`. This
confirms that the failures are in the generated proofs rather than in the
problem instance.

## Decision

`proof_checkpoint` is not better than the current `lossless_partial` approach
on this matched test. It produced no score gain, supplied a shorter and less
useful final proof, and cost roughly 3.4 times more tokens. The broader
24-checkpoint sweep supports the same decision: zero complete checkpoints
passed fresh audit.

Keep `lossless_partial` as the production default. Reconsider checkpoints only
with all of the following constraints:

1. Extract short local lemmas instead of the original target theorem.
2. Require a successful fresh-context audit before using any lemma.
3. Fall back to the lossless partial whenever extraction or audit fails.
4. Demonstrate a downstream proof-score improvement, not only valid XML.

This live A/B comparison has one matched source and therefore does not estimate
population-level effect size. Its quality conclusion is nevertheless
consistent with the independent 24-case extraction sweep.

## Artifacts

- `comparison.json`: machine-readable metrics and decision
- `numeric_counterexample.py`: executable check of both generated centers
- `numeric_counterexample.txt`: captured checker output
- `grader/`: four successful grader attempts, rubric, and aggregate summary
- `node_bundle/`: unpacked review copy with trailing whitespace normalized
- `proof-checkpoint-ab-export-20260721.tar.gz`: byte-exact transferred bundle
- `SOURCE_ARCHIVE_SHA256.txt`: archive checksum
