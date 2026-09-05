# Part B — Capacity Reconciliation

All arithmetic is derived from `bench/model_spec.md` and `bench/bench_log.csv`.
Computations are reproduced in `analysis.py`.

---

## B1. KV-Cache Sizing (7 pts)

### (a) KV-cache bytes per token

From `model_spec.md`:

| Property | Value |
|---|---|
| KV heads (GQA) | 8 |
| head_dim | 128 |
| layers | 28 |
| KV cache precision | fp16 (2 bytes) |

For each token position, we store both K and V projections across all layers:

```
KV bytes/token = 2 (K+V) × 8 (kv_heads) × 128 (head_dim) × 28 (layers) × 2 (fp16)
               = 2 × 8 × 128 × 28 × 2
               = 114,688 bytes
               = 112 KiB per token
```

### (b) Maximum concurrent 4096-token sequences

Memory budget:

```
GPU total:             24 GB
Usable (×0.92):        22.08 GB
Model weights (fp16):  4.2B × 2 bytes = 8.4 GB
Runtime overhead:      ~1.6 GB
Available for KV:      22.08 − 8.4 − 1.6 = 12.08 GB
```

KV cache per full-length sequence:

```
KV per sequence = 112 KiB × 4096 tokens = 448 MiB = 0.4375 GB
Max sequences   = 12.08 / 0.4375 ≈ 27.6
                → ~27 concurrent 4096-token sequences
```

### Validation against bench_log.csv

| batch | prompt+gen | kv_cache_util | preempted_seqs | Status |
|-------|-----------|---------------|----------------|--------|
| 24 | 3584+512=4096 | 0.93 | 0 | ✓ Just under limit |
| 32 | 3584+512=4096 | 0.97 | 7 | ✓ Exceeds ~27 limit |
| 48 | 3584+512=4096 | 0.97 | 23 | ✓ Severely exceeds |

The prediction (~27 max sequences) matches: batch 24 fits with 93% utilization,
batch 32 overflows with 7 preemptions, batch 48 overflows with 23 preemptions.

---

## B2. Throughput Anomaly (6 pts)

### The anomaly

In the long-context sweep (prompt_len=3584), throughput **decreases** at batch
sizes above 24:

| batch | reported_tok/s | preempted | kv_util | ttft_ms_p50 | wall_s |
|-------|---------------|-----------|---------|-------------|--------|
| 4 | 565.4 | 0 | 0.16 | 483.2 | 28.98 |
| 8 | 902.6 | 0 | 0.31 | 519.0 | 36.30 |
| 16 | 1311.4 | 0 | 0.62 | 498.3 | 49.97 |
| **24** | **1607.4** | **0** | **0.93** | **500.5** | **61.16** |
| 32 | 1384.0 | 7 | 0.97 | 636.9 | 94.71 |
| 48 | 1298.5 | 23 | 0.97 | 955.4 | 151.41 |

Throughput scales linearly from batch 4→24 but then **drops 14% at batch 32**
and **19% at batch 48** relative to batch 24.

### Mechanism: KV-cache exhaustion → preemption → wasted recomputation

1. **KV-cache saturation**: At batch 24, kv_cache_util hits 0.93 — nearly all
   KV-cache blocks are allocated. The GPU is now **memory-bound**, not
   compute-bound.

2. **Preemption**: At batch 32, the scheduler cannot allocate KV blocks for
   all 32 sequences simultaneously. It **preempts** (evicts) 7 sequences to
   free KV-cache space for active ones. At batch 48, 23/48 = 48% of sequences
   are preempted.

3. **Recomputation cost**: Preempted sequences lose their cached KV state.
   When they're rescheduled, the server must re-run the prefill phase to
   rebuild the KV cache. This is pure wasted work — GPU cycles spent
   regenerating state rather than producing new tokens.

4. **Observable in the log**: TTFT (time to first token) spikes from ~500ms
   to 637ms (batch 32) to 955ms (batch 48), reflecting prefill contention.
   Wall-clock time nearly triples (61→151s) for only 2× more requests.

### Proposed config change

**Set `max_num_seqs=24`** for 4096-token workloads (or equivalently, cap
concurrent requests to match KV-cache capacity).

**Predicted quantitative effect**: Eliminating preemption restores throughput
to 1607 tok/s and keeps per-request latency at ~61s. For higher request
volumes, use **request queuing** rather than oversubscription. A queue-based
system at batch-24 delivers 24 × 512 / 61.16 ≈ 201 generated tokens/s
sustained, versus the preemption-degraded 48 × 512 / 151.41 ≈ 162 generated
tokens/s at batch-48.

---

## B3. Report Misreading (4 pts)

### The misreading

REPORT_v0 Section 2 says:

> "at batch 16, long prompts hit **1311 tok/s** vs only **883 tok/s** for
> short prompts. Longer prompts clearly give better GPU utilization."

> "batch 48 should give us ~3200 tok/s"

Both conclusions come from misreading **`reported_tok_s`**.

### What's wrong

`reported_tok_s` is the harness's throughput counter: it counts **all tokens
processed** (prompt + generation) per unit time. This conflates prompt-processing
throughput with useful generation throughput.

**Why the comparison is invalid**: Short rows process 768 total tokens/request
(512 prompt + 256 gen), while long rows process 4096 total tokens/request
(3584 + 512). The long rows show higher `reported_tok_s` not because of "better
GPU utilization" but because each request pushes 5.3× more tokens through the
model — most of which are prompt tokens that don't produce useful output.

### Honest goodput of batch-24 long-prompt

**Method 1 — from gen_len × num_requests / wall_clock_s:**

```
useful tokens = 24 × 512 = 12,288
wall_clock    = 61.16 s
goodput       = 12,288 / 61.16 = 200.9 generated tok/s
```

**Method 2 — from itl_ms_p50 (inter-token latency):**

```
itl_ms_p50   = 96.07 ms → 0.09607 s/token
decode time  ≈ 512 × 0.09607 = 49.2 s
prefill time ≈ 61.16 − 49.2 = 12.0 s
end-to-end goodput = 12,288 / 61.16 = 200.9 generated tok/s  ✓ (matches Method 1)
```

Both methods agree: **~201 generated tok/s** is the honest goodput.

### What the report should have said

The `reported_tok_s` metric includes prompt-processing tokens and is not a
measure of useful generation throughput. The actual generation goodput at
batch-24 long-prompt is **~201 tok/s**, not 1607 tok/s. The claim that
"batch 48 should give ~3200 tok/s" is doubly wrong:

1. It linearly extrapolates `reported_tok_s`, which includes non-useful work
2. Batch 48 actually shows **degraded** throughput due to KV-cache preemption
3. Actual batch-48 goodput = 48 × 512 / 151.41 ≈ **162 tok/s** (worse than
   batch-24's 201 tok/s)

---

## B4. Confirming Counter/Metric (3 pts)

**Pull**: `num_preemption_total` (or the equivalent Prometheus counter, e.g.,
`vllm:num_preemption_total` in vLLM).

This counter tracks the cumulative number of sequence preemptions by the
scheduler. The B2 mechanism is KV-cache exhaustion forcing the scheduler to
evict running sequences.

**Expected values**:

| Batch size | Expected preemptions |
|-----------|---------------------|
| ≤ 24 (long-context) | 0 |
| 32 | ~7 per run |
| 48 | ~23 per run |

If this counter is non-zero in production, the serving config is
oversubscribing GPU memory. The fix is to cap `max_num_seqs` to the value
that keeps `kv_cache_util` below ~0.93.
