# Week 1 Experiment Log

This file records the first public-report data pass for the GDN prefill
tutorial. Raw JSONL files live in the same directory.

## Setup

- GPU: NVIDIA H200
- Clocks: SM 1500 MHz, memory 3201 MHz
- Layout: BTHD
- Workload family: `B=1`, `DK=DV=128`, `chunk_size=64`, `fp16`
- Timing: `benchmarks.benchmark_base.bench_kernel`
- Standard timing policy: 10 warmup, 100 repeat, 3 trials
- TileOps code: PR #1596 worktree, head `d09c8f2`

## Fair FLA Comparison

Raw data:

- `fair_fla_comparison_week1_gpu4_1500_20260618.jsonl`

FLA was run in BTHD layout with `output_final_state=True` when supported.

| Seq len | Heads | TileOps BTHD | FLA BTHD | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 32K | 16 | 2.8165 ms | 3.1320 ms | 1.11x |
| 64K | 16 | 5.5506 ms | 6.5554 ms | 1.18x |
| 128K | 16 | 11.2403 ms | 13.4351 ms | 1.20x |
| 128K | 32 | 18.6650 ms | 21.0881 ms | 1.13x |

Takeaway: the current BTHD production path beats the fair FLA baseline on the
target rows already measured for the tutorial.

## Public-Op Profiler Breakdown

Raw data:

- `bottleneck_profile_week1_gpu4_1500_20260618.jsonl`

Profiler caveat: PyTorch profiler reports an `Activity Buffer Request` event
that is not a TileOps kernel. The stage-level interpretation below ignores it.

### H16, S32K

| Kernel/event | Device time |
| --- | ---: |
| full public op wrapper | 3074.782 us |
| `h_recurrence_bthd_kernel` | 831.104 us |
| `prefill_prepare_w_u_bthd_kernel` | 451.039 us |
| `output_o_bthd_kernel` | 247.040 us |
| `chunk_cumsum_bthd_kernel_kernel` | 4.416 us |

Takeaway: at S32K, the non-scan public op is still h-recurrence dominated.

### H16, S128K

| Kernel/event | Device time |
| --- | ---: |
| full public op wrapper | 11441.934 us |
| `prefill_prepare_w_u_bthd_kernel` | 1764.322 us |
| `group_transition_summary_bthd_kernel` | 1720.706 us |
| `h_grouped_replay_bthd_kernel` | 1107.425 us |
| `output_o_bthd_kernel` | 970.337 us |
| `dense_group_start_scan_bthd_kernel` | 150.240 us |
| `chunk_cumsum_bthd_kernel_kernel` | 5.505 us |

Takeaway: the long-sequence H16 production path uses the scan decomposition:
transition summary, dense scan over group starts, then grouped replay.

## H-Recurrence Scale-Placement Ablation

Raw data:

- `scale_vnew_ablation_week1_gpu4_1500_20260618.jsonl`

Workload:

```text
B=1, H=16, S=32K, DK=DV=128, chunk64, fp16
h config: block_v=16, h_threads=128, h_num_stages=2
```

| Variant | H-only latency | Speedup vs old |
| --- | ---: | ---: |
| old: scale `k` into `k_scaled_s` | 2.2725 ms | 1.00x |
| current: scale `v_new` in place | 1.6277 ms | 1.40x |

Correctness:

- `v_new_max_abs_vs_current = 0.0`
- `final_state_max_abs_vs_current = 3.05e-5`

Takeaway: the algebraic rewrite keeps the output-equivalent h update but removes
the `k_scaled_s` shared buffer from the BTHD h recurrence. In the current
production h configuration, this gives a 1.40x h-only speedup for H16/S32K.

## Current Evidence Status

Done:

- fair TileOps-vs-FLA comparison for the v1 table
- public-op profiler evidence for S32K and S128K
- exact BTHD h-recurrence algebraic rewrite ablation

Still useful before final publication:

- naive-to-optimized ladder on one canonical workload
- prepare inverse comparison, preferably framed as Neumann vs naive forward
  solve vs FLA-style blocked solve
- scan guard sweep for H16 S32K/64K/128K and H32 S128K
- optional NCU resource snapshot for the h-recurrence rewrite
