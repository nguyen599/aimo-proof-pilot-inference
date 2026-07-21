# Proof-checkpoint handoff sweep

## Goal

Test whether an unfinished 122K-token proof attempt can emit a compact,
proof-carrying checkpoint that lets a fresh solver reuse completed lemmas
without proving them again. This is a stricter alternative to the production
`lossless_partial` handoff.

The implementation was committed and pushed as `db82c5a` before this NII run.

## Setup

- Run ID: `proof-checkpoint-sweep8-t3-20260721T114752Z`
- Source: eight balanced saved IMO 2025 cutoff contexts from both ranks,
  problems, and old parseable/unparseable outcomes
- Prompt before the forced suffix: 121,521 to 122,486 tokens
- Variants: `proof_checkpoint` at temperatures `1.0`, `0.7`, and `0.6`
- Matrix: 8 cases x 3 temperatures = 24 jobs
- Checkpoint generation limit: 32,768 tokens
- Repair: one attempt for invalid checkpoint XML
- Fresh-context audit: 8,192 tokens at temperature `0.2`
- Runtime: vLLM 0.25.1, OLMo3 OPD SFT-750, online FP8 weights,
  FP8 KV cache, TP2/DP4 per 8xH200 node
- Concurrency: 16 optimizer workers split round-robin across two nodes

Node3 ran DFlash with its 65,536-token cutoff. Node2's initial DFlash startup
failed during Inductor AOT autotuning, so it was restarted without DFlash.
This does not change decoding for these prompts because all source contexts
already exceeded DFlash's cutoff.

## Results

| Temperature | XML valid | Checkpoint valid | Structural lemmas | Audit pass | Audited reusable lemmas | Mean generated tokens | Mean latency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6 | 5/8 | 4/8 | 2 | 0/8 | 1 | 24,086 | 310.7 s |
| 0.7 | 5/8 | 3/8 | 1 | 0/8 | 0 | 27,431 | 337.6 s |
| 1.0 | 6/8 | 5/8 | 0 | 0/8 | 0 | 19,913 | 263.7 s |
| **Total** | **16/24** | **12/24** | **3** | **0/24** | **1** | - | - |

All 24 HTTP jobs returned successfully and the optimizer exited with status
zero. The validity failures were model-output failures, not request failures.
Several invalid cases exhausted both generation and repair budgets, consuming
66K to 74K aggregate completion tokens.

The single audited reusable lemma was the elementary count
`n(n+1)/2` for positive lattice points with `a+b <= n+1`. It appeared in an
otherwise invalid checkpoint whose second claimed lemma was marked `REPROVE`,
so it did not produce an audit-passing checkpoint. Every complete checkpoint
that claimed a nontrivial lemma either failed the fresh audit or emitted an
audit response that did not satisfy the required wrapper.

Typical failures were:

- relabeling the original unsolved target as a proved lemma;
- replacing the missing argument with "straightforward but lengthy" algebra;
- citing an unproved inversion, polar, or Miquel construction as a full proof;
- spending the full 32,768-token budget inside one section, then repeating the
  same behavior during repair; and
- producing a plausible proof outline that the fresh auditor correctly marked
  `REPROVE`.

## Decision

Do not make `proof_checkpoint` the production default. It has not demonstrated
a single complete checkpoint whose claimed lemmas survive fresh-context audit,
and its repair path can consume more tokens than a fresh proof attempt.

Keep `lossless_partial` as the default handoff. A future checkpoint experiment
should narrow the task to extracting short local facts, require audit success
before restart, and measure whether an audited lemma actually improves the
next proof attempt. Structural XML validity alone is not a proof-quality
metric.

## Artifacts

- `summary.json`: aggregate metrics used in the table
- `results.jsonl` and `results.csv`: all 24 structured outcomes
- `selected_cases.json`: balanced source-case manifest
- `calls/`: full prompt, checkpoint, repair, and audit logs
- `logs/optimizer.log.txt`: request completion timeline
- `logs/vllm_node2_dflash_failed.log.txt`: preserved startup failure
- `logs/vllm_node2.log.txt` and `logs/vllm_node3.log.txt`: successful server logs
- `commands/`: exact optimizer and node2 server commands
- `proof-checkpoint-sweep8-t3-20260721T114752Z.tar.gz`: byte-exact NII
  archive; unpacked text files have only line-ending and trailing-whitespace
  normalization for Git review
- `SOURCE_ARCHIVE_SHA256.txt`: checksum of the transferred NII archive
