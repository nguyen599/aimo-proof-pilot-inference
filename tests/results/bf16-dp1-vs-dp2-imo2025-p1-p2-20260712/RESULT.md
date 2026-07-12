# BF16 DP1 versus DP2: target-only and DFlash

## Result

For this two-request workload, DP2 barely increased aggregate throughput over DP1:

| Inference | DP | GPUs | Wall time (s) | Aggregate tok/s | Problem 1 tok/s | Problem 2 tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Target-only | 1 | 1 | 174.941 | 93.654 | 46.827 | 46.827 |
| Target-only | 2 | 2 | 172.602 | 94.924 | 47.462 | 47.490 |
| DFlash | 1 | 1 | 114.767 | 142.759 | 71.380 | 85.151 |
| DFlash | 2 | 2 | 111.904 | 146.411 | 73.206 | 86.705 |

The direct comparisons are:

- Target-only DP2 versus DP1: **1.0136x**, or **+1.36%**.
- DFlash DP2 versus DP1: **1.0256x**, or **+2.56%**.
- DFlash versus target-only at DP1: **1.5243x**, or **+52.43%**.
- DFlash versus target-only at DP2: **1.5424x**, or **+54.24%**.

This does not mean data parallelism is broken. It means that, at concurrency two, one H200 already processes a batch of two requests almost as quickly in aggregate as two H200s process one request each.

## Experimental contract

Every case used the same workload:

- BF16 OPD target: `/workspace/models/opd-32b-deploy`.
- BF16 DFlash draft when enabled: `/workspace/models/dflash-32b-draft-v2test-phaseL`.
- Tensor parallel size: 1.
- MathArena IMO 2025 problems 1 and 2, submitted simultaneously.
- Exact ycchen prover messages already recorded by the evaluation harness.
- Prompt lengths: 426 and 461 tokens.
- Maximum completion: 8,192 tokens per request.
- Greedy decoding: temperature 0, top-p 1, seed 0.
- Total measured output: 16,384 tokens per case.
- Prefix cache flushed immediately before every case.
- No benchmark warm-up request.
- SGLang deterministic inference enabled.
- BF16 KV cache, 262,144-token context ceiling, and `mem_fraction_static: 0.84`.
- `max_running_requests: 32` per DP worker.
- DP2 used native SGLang round-robin routing. Live logs confirmed problem 1 ran on DP0/GPU0 and problem 2 ran on DP1/GPU1.

All completions hit the 8,192-token length limit. Target-only output was identical between DP1 and DP2 by the recorded character counts. DFlash output was also identical between DP1 and DP2. This controls for a different generated sequence changing the amount of work.

Aggregate throughput is defined as:

$$
\text{aggregate throughput}
=
\frac{\text{sum of completion tokens from all requests}}
     {\text{time from starting both requests until both are finished}}.
$$

The slowest request therefore determines the wall-time denominator. This matters for DFlash because speculative acceptance depends on the generated content: problem 2 finished earlier than problem 1 in both DP layouts.

## First principles: batch size is not data-parallel size

Three quantities must be kept separate:

1. **Active request batch size**: how many sequences a single model replica advances together.
2. **Data-parallel size (DP)**: how many complete model replicas exist.
3. **Tensor-parallel size (TP)**: how many GPUs cooperate on one replica and one forward pass.

In these tests TP is always 1.

### DP1 with two concurrent requests

DP1 means one copy of the target model is loaded on one GPU. It does not limit the server to one request. SGLang continuous batching places both active sequences in the same decode batch.

For one request, a linear layer conceptually computes:

$$
y_1 = W x_1.
$$

For two requests on one replica, their hidden states can be stacked:

$$
X =
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix},
\qquad
Y = X W^\mathsf{T}.
$$

The GPU reads the same weight matrix $W$ once for a larger matrix operation and produces one next-token state for each request. Each request still has its own:

- prompt and generated tokens;
- KV cache;
- sampling state;
- stop condition;
- response.

Only the forward computation is batched. Therefore DP1 can—and normally does—have batch size 2, 32, or another scheduler-controlled value.

The observed target-only server logs showed `#running-req: 2` and about 94–95 generated tokens per second during this case. Each request progressed at about 46.83 tok/s, while their combined rate was 93.65 tok/s.

### DP2 with two concurrent requests

DP2 creates two complete model replicas:

- DP0 loads one copy of all weights on GPU0.
- DP1 loads another copy of all weights on GPU1.

Round-robin routing sent one request to each worker. Each worker therefore decoded at local batch size 1:

$$
y_1 = W^{(0)} x_1,
\qquad
y_2 = W^{(1)} x_2.
$$

There is no tensor-parallel all-reduce between these replicas. They are independent inference engines behind one HTTP frontend.

## Why did target-only DP2 add only 1.36%?

Autoregressive decoding repeatedly reads a very large model to produce a small number of new token states. At low batch size this is often limited more by moving weights through the memory hierarchy than by the GPU's peak arithmetic rate.

With one request, the GPU reads the weights to advance one sequence. With two batched requests, it can reuse those weights to advance two sequences during the same layer operation. The incremental cost of the second row of activations is much smaller than reading a second full copy of the weights.

A simplified model is:

$$
T(B) \approx T_{\text{weights}} + B T_{\text{activation}},
$$

where $B$ is the active batch size. If $T_{\text{weights}}$ dominates, then:

$$
T(2) \approx T(1),
$$

so one replica at batch size 2 can produce nearly twice the aggregate tokens of one replica at batch size 1.

That is exactly what happened:

- One target-only TP1 request previously measured about 47.51 tok/s.
- One DP1 worker with two requests measured 93.65 aggregate tok/s.
- Two DP2 workers with one request each measured 94.92 aggregate tok/s.

DP2 did not make either request much faster because each request still ran through a TP1 model at batch size 1. It also did not greatly increase aggregate throughput because DP1 had already obtained nearly the full two-request batching gain on one GPU.

The cost-efficiency consequence is visible in throughput per allocated GPU:

| Inference | DP1 tok/s/GPU | DP2 tok/s/GPU |
|---|---:|---:|
| Target-only | 93.654 | 47.462 |
| DFlash | 142.759 | 73.206 |

DP2 uses twice the model memory and approximately twice the GPU allocation for only a small gain at concurrency two.

## Why does DFlash behave differently?

DFlash is speculative decoding. A smaller draft model proposes several future tokens, and the target model verifies a block of proposals in one operation. If several proposals are accepted, one expensive target pass advances the sequence by multiple tokens.

For a single speculative step, let $A$ be the number of accepted output tokens and let the step time be:

$$
T_{\text{step}}
=
T_{\text{draft}}
+
T_{\text{target verification}}
+
T_{\text{coordination}}.
$$

Then useful output throughput is approximately:

$$
\text{throughput}
\approx
\frac{A}{T_{\text{step}}}.
$$

Acceptance is content-dependent. That is why problem 2 ran faster than problem 1 even with identical server settings. DP changes where requests execute, but it does not make their acceptance patterns equal.

DFlash DP1 reached 142.76 aggregate tok/s, 52.43% above target-only DP1. DFlash DP2 reached 146.41 aggregate tok/s, 54.24% above target-only DP2. DP2 provided a slightly larger gain for DFlash than for target-only, but it was still only 2.56%.

The likely reason is that DFlash adds draft-model work, verification, KV materialization, and scheduling. Batching two speculative streams on one worker is still efficient, but it is not quite as perfectly amortized as ordinary target-only decode. Separating the streams gives a small improvement, not a near-2x improvement.

## What DP is useful for

These results cover exactly two simultaneous long generations. They do not imply that DP is useless at higher concurrency.

DP becomes useful when one worker can no longer efficiently absorb the offered load because of:

- compute saturation at larger active batches;
- KV-cache capacity;
- request queueing and latency targets;
- heterogeneous request lengths;
- prefill contention;
- scheduler or memory-pressure limits.

At that point, additional replicas increase cluster capacity and reduce queues. The correct scaling curve is therefore aggregate throughput versus concurrency for DP1 and DP2—not a single low-concurrency point.

For the current evaluation, which submits many proofs concurrently, the next useful experiment is a concurrency sweep such as 2, 8, 16, and 32 requests per configuration. DP2 should be judged by where it raises the throughput ceiling or improves tail latency, not by expecting one request to decode faster.

## Configuration support

The launcher now has a strict required `model.data_parallel_size` field. It passes both native SGLang settings explicitly:

```text
--tp <tensor_parallel_size>
--dp <data_parallel_size>
--load-balance-method round_robin
```

The required GPU count is:

$$
N_{\text{GPU}} = \text{TP} \times \text{DP}.
$$

The canonical configuration remains TP2/DP1. Changing to TP1/DP2 therefore requires setting both values deliberately; there is no inferred DP default or compatibility fallback.
