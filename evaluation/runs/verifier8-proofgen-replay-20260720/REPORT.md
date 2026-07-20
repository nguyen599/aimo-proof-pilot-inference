# Eight-role verifier replay

This audit reuses completed proof-generation calls from the stopped IMO 2025
four-node run. It does not regenerate or select proofs. The current verifier
stack evaluates each parsed proof with eight distinct roles, followed by one
meta-verifier call per review:

1. dependency chain;
2. lemma assumptions;
3. counterexample and boundary cases;
4. invariance and relabeling;
5. quantifiers and strategy;
6. algebra and computation;
7. statement coverage;
8. construction and optimality.

The proof set contains every round-zero OPD generation that had valid proof and
self-evaluation XML in the source snapshot: 13 proofs, all with self-score
`1`. An independent blind review was locked before inspecting the new verifier
outputs. None of the 13 proofs was complete: eight were substantial partial
proofs and five had fatal errors.

## Result

The new ensemble is materially better at preventing false acceptance on this
adversarial set, but it is not yet a calibrated proof grader.

- Twelve cases completed all eight verifier and meta calls. Every one was
  classified at or below `0.5`, giving `12/12` recall for incomplete proofs and
  no false complete result.
- At least one role found the decisive blind-review flaw in all 12 completed
  cases. The diagnosis was exact in 11/12; for P2/C17 the roles correctly
  rejected the unchecked final identity but did not independently prove it
  false.
- The continuous replay score averaged `0.143`, versus a mean blind label of
  `0.292`. Bucketing exact zero as fatal and `(0, 0.5]` as partial gives only
  `6/12` three-way agreement. The verifier tends to turn localized repairable
  gaps into fatal judgments.
- Two individual role calls returned score `1`; both were false passes. The
  eight-role aggregate still rejected both proofs.
- P4/C18 never completed a valid eight-role round. Three 32k attempts parsed
  only `2/8`, `3/8`, and `2/8` scores. Three 8k retries parsed `0/8`; every 8k
  call ended by length. Reducing the token cap did not fix output compliance.

This is not a controlled A/B against the previous four-role verifier on the
same proofs. The direct baseline here is the source proof self-evaluation,
which accepted all 13 incorrect or incomplete proofs. A same-proof four-role
replay would be required to attribute the gain specifically to using eight
roles rather than to the other verifier-hardening changes.

## Cases

| Problem | Candidate | Blind grade | Blind label | Replay score | Outcome |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 8 | 5/7 | 0.5 | 0.0625 | rejected |
| 2 | 17 | 5/7 | 0.5 | 0 | rejected |
| 2 | 20 | 1/7 | 0 | 0.125 | rejected |
| 2 | 21 | 6/7 | 0.5 | 0 | rejected |
| 3 | 13 | 2/7 | 0 | 0 | rejected |
| 3 | 26 | 2/7 | 0 | 0.21875 | rejected |
| 3 | 33 | 2/7 | 0 | 0.0625 | rejected |
| 4 | 12 | 2/7 | 0 | 0.0625 | rejected |
| 4 | 13 | 5/7 | 0.5 | 0.3125 | rejected |
| 4 | 14 | 5/7 | 0.5 | 0.3125 | rejected |
| 4 | 15 | 5/7 | 0.5 | 0.4375 | rejected |
| 4 | 16 | 5/7 | 0.5 | 0.125 | rejected |
| 4 | 18 | 5/7 | 0.5 | n/a | verifier format failure |

## Role utility

The hit column records whether the role located the blind review's decisive
flaw. Agreement compares the role's `0`, `0.5`, or `1` score with the blind
label.

| Role | Flaw hits | Label agreement | Assessment |
| --- | ---: | ---: | --- |
| dependency chain | 8/12 | 6/12 | Useful dependency tracing, with speculative secondary claims in several cases. |
| lemma assumptions | 9/12 | 7/12 | Usually precise; missed the central circularity in P3/C33. |
| counterexample boundary | 10/12 | 7/12 | High recall, but fabricated a numerical counterexample for P2/C21. |
| invariance relabeling | 10/12 | 7/12 | High recall, but falsely passed P3/C26. |
| quantifier strategy | 10/12 | 9/12 | Best standalone balance; no material false critique in this sample. |
| algebra computation | 9/12 | 6/12 | Useful for explicit calculations, but prone to false secondary objections. |
| statement coverage | 9/12 | 6/12 | Uniquely found P4/C15's key gap, but falsely passed P4/C14. |
| construction optimality | 9/12 | 7/12 | Useful coverage, with one recurring false divisor-count objection. |

Role diversity mattered most on P4/C15: most roles pursued a missing factor-5
case, while statement coverage alone found the decisive nonmultiple-of-3
descent gap. Conversely, the false score-1 calls show why one role must not be
allowed to certify a proof by itself.

## Format failure

P4/C18 exposed a separate operational failure. At 32k tokens, most verifier
calls spent the response solving the problem rather than producing the required
audit XML. Some calls stopped before the limit but still lacked a complete
parseable response. At 8k tokens, all 24 retries ended by length with no parsed
score. The evidence supports these changes:

- preserve role diversity and ensemble aggregation;
- distinguish a localized repairable gap from a fatal result more explicitly
  in the scoring contract;
- require a bounded audit before any extended derivation, or enforce a
  structured final response independently of the reasoning budget;
- retry malformed outputs with a format-only continuation rather than simply
  lowering the total token limit;
- never discard successful cases because one proof fails the output contract.

The replay utility now checkpoints each case and writes both `results` and
structured `errors`, so an incomplete case cannot erase the completed audit.

## Reproduction

The source proofs are under
`evaluation/runs/imo2025-full-p36-r4-p6-sft750-lossless-deadline-4node-retry-20260718T161415Z-stopped-20260719/logs`.
The replay used two NII vLLM servers with the SFT-750 checkpoint, online FP8
weights, FP8 KV cache, TP2/DP4 per node, and this core command:

```bash
python evaluation/replay_verifier_audit.py \
  --input-path evaluation/data/imo_2025.parquet \
  --llm-calls-dir SOURCE_LOGS \
  --proof-prompt-family opd \
  --model-path /tmp/models/olmo3-opd-sft-750-vllm \
  --base-url http://NODE2:8012/v1 \
  --base-url http://NODE3:8012/v1 \
  --output-dir OUTPUT_DIR \
  --verify-n 8 \
  --meta-n 1 \
  --meta-policy all-reviews \
  --verifier-max-tokens 32000 \
  --meta-max-tokens 32000 \
  --temperature 1.0 \
  --max-attempts 3
```

`analysis.json` preserves the proof hashes, full role evaluations, meta
evaluations, score matrix, blind labels, and P4/C18 retry metadata used by this
report.
