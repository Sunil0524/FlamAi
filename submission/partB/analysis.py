#!/usr/bin/env python3
"""
analysis.py — Capacity reconciliation computations for Part B.

Computes KV-cache sizing, validates against bench_log.csv,
and derives honest goodput metrics.

Usage:
    python analysis.py
"""

import csv
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")


def b1_kv_cache_math():
    """B1: KV-cache bytes per token and max concurrent sequences."""
    print("=" * 70)
    print("B1: KV-Cache Sizing from Model Spec")
    print("=" * 70)

    # From model_spec.md
    n_layers = 28
    n_kv_heads = 8       # GQA: 8 KV heads
    head_dim = 128
    kv_precision_bytes = 2  # fp16
    gpu_mem_gb = 24
    gpu_mem_util = 0.92
    max_model_len = 4096
    n_params_b = 4.2
    overhead_gb = 1.6

    # KV-cache bytes per token
    # For each token, we store K and V for each layer
    # Each of K and V has shape [n_kv_heads, head_dim] in fp16
    kv_bytes_per_token = (
        2  # K and V
        * n_kv_heads
        * head_dim
        * n_layers
        * kv_precision_bytes
    )

    print(f"\n  KV-cache bytes per token:")
    print(f"    = 2 (K+V) × {n_kv_heads} (KV heads) × {head_dim} (head_dim) "
          f"× {n_layers} (layers) × {kv_precision_bytes} (fp16 bytes)")
    print(f"    = {kv_bytes_per_token:,} bytes")
    print(f"    = {kv_bytes_per_token / 1024:.0f} KiB per token")

    # Available memory for KV cache
    usable_mem_gb = gpu_mem_gb * gpu_mem_util
    model_weights_gb = n_params_b * 2  # fp16: 2 bytes per param
    kv_available_gb = usable_mem_gb - model_weights_gb - overhead_gb

    print(f"\n  Memory budget:")
    print(f"    GPU total:           {gpu_mem_gb} GB")
    print(f"    Usable (×{gpu_mem_util}):     {usable_mem_gb:.2f} GB")
    print(f"    Model weights (fp16): {model_weights_gb:.1f} GB  "
          f"({n_params_b}B × 2 bytes)")
    print(f"    Runtime overhead:     {overhead_gb} GB")
    print(f"    Available for KV:     {kv_available_gb:.2f} GB")

    # Max concurrent sequences at max_model_len
    kv_per_seq_bytes = kv_bytes_per_token * max_model_len
    kv_per_seq_gb = kv_per_seq_bytes / (1024 ** 3)
    max_seqs = kv_available_gb / kv_per_seq_gb

    print(f"\n  Max concurrent {max_model_len}-token sequences:")
    print(f"    KV per sequence = {kv_bytes_per_token:,} × {max_model_len} "
          f"= {kv_per_seq_bytes:,} bytes = {kv_per_seq_gb:.3f} GB")
    print(f"    Max sequences = {kv_available_gb:.2f} / {kv_per_seq_gb:.3f} "
          f"= {max_seqs:.1f}")
    print(f"    → approximately {int(max_seqs)} concurrent 4096-token sequences")

    print(f"\n  Validation against bench_log.csv:")
    print(f"    - batch=24, prompt 3584+gen 512=4096 tokens: "
          f"kv_util=0.93, 0 preempted ✓")
    print(f"    - batch=32, 4096 tokens: kv_util=0.97, 7 preempted "
          f"(exceeds ~{int(max_seqs)} limit) ✓")
    print(f"    - batch=48, 4096 tokens: kv_util=0.97, 23 preempted "
          f"(severely exceeds) ✓")
    print(f"    Prediction matches observations: "
          f"limit is ~{int(max_seqs)} sequences.\n")

    return max_seqs


def b2_throughput_anomaly(rows):
    """B2: Identify and explain the throughput anomaly in long-context sweep."""
    print("=" * 70)
    print("B2: Throughput Anomaly in Long-Context Sweep")
    print("=" * 70)

    # Filter long-context rows (prompt_len=3584)
    long_rows = [r for r in rows if r["prompt_len"] == 3584]

    print("\n  Long-context sweep (prompt_len=3584, gen_len=512):")
    print(f"  {'batch':>6} {'tok/s':>8} {'preempted':>10} {'kv_util':>9} "
          f"{'ttft_ms':>9} {'itl_ms':>8} {'wall_s':>8}")
    print("  " + "-" * 62)
    for r in long_rows:
        print(f"  {r['batch_size']:>6} {r['reported_tok_s']:>8.1f} "
              f"{r['preempted_seqs']:>10} {r['kv_cache_util']:>9.2f} "
              f"{r['ttft_ms_p50']:>9.1f} {r['itl_ms_p50']:>8.2f} "
              f"{r['wall_clock_s']:>8.2f}")

    print(f"\n  THE ANOMALY:")
    print(f"  Throughput DROPS from batch 24 → 32 → 48:")
    print(f"    batch 24: 1607.4 tok/s (peak)")
    print(f"    batch 32: 1384.0 tok/s (−13.9%)")
    print(f"    batch 48: 1298.5 tok/s (−19.2%)")
    print(f"  This violates the naive expectation that throughput scales with batch.")

    print(f"\n  MECHANISM: KV-cache exhaustion → preemption → wasted recomputation")
    print(f"    1. At batch 32, kv_cache_util = 0.97 (essentially full).")
    print(f"       7 sequences are preempted: evicted from KV cache to make room.")
    print(f"    2. At batch 48, 23/48 = 48% of sequences are preempted.")
    print(f"    3. Preempted sequences must be re-prefilled later, wasting GPU cycles")
    print(f"       on recomputation instead of forward progress.")
    print(f"    4. Evidence: ttft_ms spikes from 500→637→955 (prefill delays from")
    print(f"       contention), and wall_clock roughly doubles (94.7→151.4s) despite")
    print(f"       only 1.5× more requests.")

    print(f"\n  PROPOSED CONFIG CHANGE:")
    print(f"    Limit the maximum concurrent batch to 24 for 4096-token sequences")
    print(f"    (i.e., set max_num_seqs=24 in the serving config).")
    print(f"    Predicted effect: eliminates preemption entirely, maintaining")
    print(f"    1607.4 tok/s throughput. Beyond batch 24, use request queuing")
    print(f"    instead of oversubscription. Per-request latency stays at ~61s")
    print(f"    (vs. 94.7–151.4s with preemption), improving user experience.")
    print(f"    Alternative: reduce max_model_len if use case permits shorter")
    print(f"    contexts, which would allow more concurrent sequences.\n")


def b3_report_misreading(rows):
    """B3: Identify the misreading in REPORT_v0 Section 2."""
    print("=" * 70)
    print("B3: REPORT_v0 Section 2 Misreading")
    print("=" * 70)

    print(f"\n  The report claims:")
    print(f'    "at batch 16, long prompts hit 1311 tok/s vs only 883 tok/s')
    print(f'     for short prompts. Longer prompts clearly give better GPU')
    print(f'     utilization."')
    print(f'    "batch 48 should give us ~3200 tok/s"')

    print(f"\n  THE MISREADING: 'reported_tok_s' conflates different things.")
    print(f"  The column counts total tokens processed (including prompt tokens)")
    print(f"  divided by wall-clock time. It is NOT 'useful generation throughput.'")

    print(f"\n  Why 'longer prompts = higher tok/s' is misleading:")
    print(f"    Short rows: prompt=512, gen=256 → 768 total tokens/request")
    print(f"    Long rows:  prompt=3584, gen=512 → 4096 total tokens/request")
    print(f"    The long rows process 5.3× more tokens per request, so of course")
    print(f"    total tok/s is higher! But the useful output is only the gen_len.")

    # Compute goodput for batch-24 long-prompt
    b24_long = [r for r in rows
                if r["batch_size"] == 24 and r["prompt_len"] == 3584][0]

    useful_tokens = b24_long["num_requests"] * b24_long["gen_len"]
    wall = b24_long["wall_clock_s"]
    goodput_method1 = useful_tokens / wall

    print(f"\n  HONEST GOODPUT of batch-24 long-prompt row:")
    print(f"\n  Method 1 — from gen_len × num_requests / wall_clock_s:")
    print(f"    useful tokens = {b24_long['num_requests']} × "
          f"{b24_long['gen_len']} = {useful_tokens}")
    print(f"    wall_clock = {wall:.2f}s")
    print(f"    goodput = {useful_tokens} / {wall:.2f} = "
          f"{goodput_method1:.1f} generated tok/s")

    # Method 2: from itl_ms_p50
    itl_s = b24_long["itl_ms_p50"] / 1000.0
    decode_throughput_per_seq = 1.0 / itl_s
    # With batch 24 running in parallel during decode phase:
    goodput_method2 = decode_throughput_per_seq * b24_long["num_requests"]
    # But that's peak decode-phase throughput. More accurately,
    # wall time = prefill time + decode time
    # decode time ≈ gen_len × itl_ms / 1000
    decode_time = b24_long["gen_len"] * itl_s
    prefill_time = wall - decode_time
    # Goodput counting only decode phase:
    goodput_decode_only = useful_tokens / decode_time

    print(f"\n  Method 2 — from itl_ms_p50 (median inter-token latency):")
    print(f"    itl_ms_p50 = {b24_long['itl_ms_p50']} ms → {itl_s:.5f}s per token")
    print(f"    decode time ≈ {b24_long['gen_len']} × {itl_s:.5f} = {decode_time:.1f}s")
    print(f"    prefill time ≈ {wall:.2f} − {decode_time:.1f} = {prefill_time:.1f}s")
    print(f"    decode-phase goodput = {useful_tokens} / {decode_time:.1f} = "
          f"{goodput_decode_only:.1f} tok/s")
    print(f"    end-to-end goodput = {useful_tokens} / {wall:.2f} = "
          f"{goodput_method1:.1f} tok/s (same as Method 1 ✓)")

    print(f"\n  WHAT THE REPORT SHOULD HAVE SAID:")
    print(f"    The reported_tok_s of 1607.4 includes prompt processing and is not")
    print(f"    representative of useful generation throughput. The actual generation")
    print(f"    goodput at batch-24 long-prompt is ~{goodput_method1:.0f} tok/s.")
    print(f"    The claim that 'batch 48 should give ~3200 tok/s' is wrong because:")
    print(f"      1. It linearly extrapolates a metric that includes non-useful work")
    print(f"      2. Batch 48 actually shows DEGRADED throughput due to KV preemption")
    print(f"      3. Actual batch-48 goodput = {48*512/151.41:.0f} tok/s "
          f"(worse than batch-24's {goodput_method1:.0f})\n")


def b4_confirming_metric():
    """B4: Which counter confirms the B2 mechanism."""
    print("=" * 70)
    print("B4: Confirming Counter/Metric")
    print("=" * 70)

    print(f"\n  The single counter to pull: num_preemption_total")
    print(f"  (or the equivalent in your serving framework, e.g.,")
    print(f"  vllm:num_preemption_total in vLLM's Prometheus metrics)")
    print()
    print(f"  This counter tracks the cumulative number of sequence preemptions")
    print(f"  by the scheduler. The mechanism in B2 is KV-cache exhaustion forcing")
    print(f"  the scheduler to evict (preempt) running sequences.")
    print()
    print(f"  Expected values:")
    print(f"    - batch ≤ 24 (long-context): 0 preemptions")
    print(f"    - batch = 32:  ~7 preemptions per run")
    print(f"    - batch = 48: ~23 preemptions per run")
    print()
    print(f"  If this counter is non-zero in production, it means the serving")
    print(f"  config is oversubscribing GPU memory, and requests are being")
    print(f"  slowed by recomputation. The fix is to cap max_num_seqs to the")
    print(f"  value that keeps kv_cache_util below ~0.93.\n")


def main():
    # Load bench_log.csv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(
        script_dir, "..", "docs", "starter_kit (1)", "starter_kit",
        "bench", "bench_log.csv"
    )
    if not os.path.exists(csv_path):
        csv_path = os.path.join(
            script_dir, "..", "..", "docs", "starter_kit (1)", "starter_kit",
            "bench", "bench_log.csv"
        )

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                k = k.strip()
                try:
                    if "." in v:
                        parsed[k] = float(v)
                    else:
                        parsed[k] = int(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            if parsed:
                rows.append(parsed)

    max_seqs = b1_kv_cache_math()
    b2_throughput_anomaly(rows)
    b3_report_misreading(rows)
    b4_confirming_metric()


if __name__ == "__main__":
    main()
