# BF16 DFlash memory-fraction sweep at concurrency 32

Date: 2026-07-12 UTC

## Scope

Two BF16 target-plus-DFlash servers ran simultaneously on separate H200 143771
MiB GPUs: OPD-32B on GPU 0 and the repaired SFT step-1000 checkpoint on GPU 1,
both at TP1.

Every benchmark used the same fixed serving workload:

- 32 prompts admitted at maximum concurrency 32;
- 512 random input tokens and 512 requested output tokens per prompt;
- exactly 16,384 input and 16,384 output tokens completed per server; and
- DFlash block size 8, eight draft tokens, and a 512-token draft window.

## Results

| Static fraction | OPD output tok/s | SFT output tok/s | KV token capacity | Peak GPU allocation |
|---:|---:|---:|---:|---:|
| 0.82 | 661.21 | 636.46 | 544,497 | 138,340 MiB |
| 0.84 | 671.99 | 642.87 | 573,005 | 141,200 MiB |
| 0.85 | 669.93 | 641.38 | 587,258 | 142,620 MiB |

All six runs reached 32 concurrent requests, completed the exact token contract,
kept both supervised servers healthy, and produced no CUDA or allocator failure.

## Selection

`mem_fraction_static=0.84` was selected. It increased the server KV pool while
leaving 2,571 MiB of physical GPU memory after the concurrency-32 workload. The
0.85 setting also passed, but left only 1,151 MiB and was rejected as unnecessarily
tight for production variance.

The checked-in policy also sets `server.max_running_requests=32` and
`search.concurrency=32`, which caps decode CUDA-graph capture at batch 32 and
removes the unused batch-40 and batch-48 graphs.

## Raw artifacts

- [`opd-mem082-c32.jsonl`](./opd-mem082-c32.jsonl)
- [`sft-mem082-c32.jsonl`](./sft-mem082-c32.jsonl)
- [`opd-mem084-c32.jsonl`](./opd-mem084-c32.jsonl)
- [`sft-mem084-c32.jsonl`](./sft-mem084-c32.jsonl)
- [`opd-mem085-c32.jsonl`](./opd-mem085-c32.jsonl)
- [`sft-mem085-c32.jsonl`](./sft-mem085-c32.jsonl)
