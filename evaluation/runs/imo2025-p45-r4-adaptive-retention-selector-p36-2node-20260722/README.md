# P4/P5 four-round treatment

This treatment keeps the baseline two-node TP2/DP4 runtime, 36 candidates,
eight mixed-role verifiers, one meta review per verifier, and four mixed
refinement rounds. It changes only the measured P4/P5 interventions:

- adaptive initial-proof strategies for the missing P4 transition-closure and
  P5 game-regime obligations;
- P4/P5-specific mandatory checks for the four specialist verifiers while the
  four generalist prompts remain unchanged;
- the same decisive completion obligations carried into meta-audit, repair,
  and independent reconstruction prompts;
- meta-aware version retention without the duplicate challenge penalty; and
- conservative retention of the earlier proof when two verified versions have
  exactly equal internal evidence, except for strict-pass challenge survival;
- an inclusive `0.5` selector boundary with at most eight candidates in the
  final LLM comparison; four slots are reserved for strong earlier verified
  proof versions. On the baseline round-2 audit, the P5 proofs graded `6.0`
  and `5.5` by GPT-5.6 ranked only third and fourth under internal verifier
  evidence, so a two-slot history budget would remove useful proofs before
  the final selector could compare them.

Do not launch it while the baseline run
`imo2025-p45-r4-mixed-challenge-p36-2node-20260721T212441Z` owns nodes 2 and 3.
After completion, export and grade every initial/final candidate with the same
two-call GPT-5.6 grader used for the baseline. `grader.yaml` assigns 24 reusable
slots round-robin across three keys, giving each key exactly eight concurrent
calls.
