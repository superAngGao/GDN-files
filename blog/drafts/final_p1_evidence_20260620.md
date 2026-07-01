# Final P1 Evidence: GDN Prefill TileLang Tuning

Date: 2026-06-20

This file archives the Priority 1 evidence needed before drafting the public
TileLang tuning article.

Note: this lives under `drafts/` because the existing `data/` directory is not
writable by the current user.

## Environment And Worktree

Production PR worktree:

```text
/home/ga/TileOPs-pr1596
```

Git state:

```text
HEAD: d09c8f2d297d8d8cc6badaa1df139014a1d7c4de
working tree: includes uncommitted PR changes in
  tests/ops/test_gated_deltanet_prefill.py
  tileops/kernels/gated_deltanet/gated_deltanet_prefill.py
```

Benchmark docker:

```text
directory: /home/ga/Documents/gdn_kernel_bench_2026-06-18
image: gdn-kernel-bench:nightly-tl019
GPU: NVIDIA H200
TileLang: 0.1.9
FLA: 0.5.1
```

Environment details are archived in:

```text
/home/ga/Documents/gdn_kernel_bench_2026-06-18/gdn_prefill_benchmark_environment.md
```

## Source Audit And Tests

Source audit command:

```bash
rg -n "triton|tl\." \
  /home/ga/TileOPs-pr1596/tileops/kernels/gated_deltanet/gated_deltanet_prefill.py
```

Result:

```text
no matches
```

Test command:

```bash
TMPDIR=/home/ga/tmp/tvm-test TILELANG_CLEANUP_TEMP_FILES=1 \
PYTHONPATH=/home/ga/TileOPs-pr1596 \
/home/ga/anaconda3/bin/python -m pytest \
  tests/ops/test_gated_deltanet_prefill.py -q
```

Result:

```text
8 passed, 2 warnings in 8.74s
```

## Final Full-Op Docker Benchmark

Latest PR-head rerun, 2026-06-22:

```text
PR head: 067edc7e0dc2da3220468f9ed5321a6d6c14406e
kernel sha256: ecd6d9931539989d67aa243528a4afb76e67c0c254808c9597caa7b9d35f5bea
Torch: 2.10.0+cu128
FLA: 0.5.1
GPU: NVIDIA H200
```

The rerun used BTHD layout, `B=1`, `DK=DV=128`, `chunk64`, `fp16`, and
`output_final_state=True` for FLA.

| Seq len | Heads | TileOps TileLang | FLA 0.5.1 | FLA / TileOps |
| --- | ---: | ---: | ---: | ---: |
| 32K | 16 | 2.39824722 ms | 2.44284368 ms | 1.01860x |
| 64K | 16 | 4.64622306 ms | 5.20679608 ms | 1.12065x |
| 128K | 16 | 9.35985710 ms | 10.87307050 ms | 1.16167x |
| 128K | 32 | 14.51952280 ms | 16.09163698 ms | 1.10828x |

The older archived 2026-06-20 H16-only run is kept below for provenance.

Raw output:

```text
/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/results/final_tilelang_bthd_vs_fla_docker_20260620.jsonl
```

Benchmark command shape:

```bash
python /workspace/gdn_prefill_ako/scripts/bench_prefill_bthd_op_vs_fla.py \
  --batch 1 \
  --heads 16 \
  --seq-len {32768,65536,131072} \
  --chunk-size 64 \
  --dim-k 128 \
  --dim-v 128 \
  --dtype float16 \
  --warmup 10 \
  --repeat 100 \
  --trials 3 \
  --output-final-state \
  --output /workspace/gdn_prefill_ako/results/final_tilelang_bthd_vs_fla_docker_20260620.jsonl
```

The docker run mounted:

```text
/home/ga/TileOPs-pr1596 -> /workspace/TileOps
/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako -> /workspace/gdn_prefill_ako
PYTHONPATH=/workspace/TileOps
```

Archived 2026-06-20 BTHD H16 full-op comparison:

| Seq len | TileOps TileLang | FLA 0.5.1 | FLA / TileOps |
| --- | ---: | ---: | ---: |
| 32K | 2.38432512 ms | 2.42574500 ms | 1.01737x |
| 64K | 4.64552674 ms | 5.27495597 ms | 1.13549x |
| 128K | 9.43092621 ms | 10.88170333 ms | 1.15383x |

Notes:

- FLA was run with `output_final_state=True`.
- Earlier notes contained a faster full-op table
  (`1.39987200/2.53315198/4.92172813 ms`), but no raw artifact was found for
  that table.
- The public article should use the 2026-06-22 PR-head rerun above as the
  current final table.

## Isolated Blocksolve-A Final Comparison

Raw output:

```text
/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/results/current_tilelang_blocksolve_vs_triton_after_018_s32k.jsonl
```

Workload:

```text
B=1, H=16, S=32768, chunk64, DK=128, fp16
warmup=5, repeat=50, trials=3
```

| Backend | Latency |
| --- | ---: |
| Triton local baseline | 0.14692616 ms |
| TileLang production | 0.11384284 ms |

Correctness:

```text
max_abs_diff_vs_triton: 0.000823974609375
mean_abs_diff_vs_triton: 4.738235475088004e-06
```

## Recompute Candidate Ladder

Raw artifacts:

```text
tilelang_recompute_005_copy_recount_s32k_h16_gpu4_20260620.jsonl
tilelang_recompute_008_no_store_s32k_h16_gpu4_20260620.jsonl
tilelang_recompute_010_shared_copy_store_s32k_h16_gpu4_20260620.jsonl
tilelang_recompute_014_swizzled_async_shared_copy_s32k_h16_gpu4_20260620.jsonl
tilelang_recompute_016_swizzled_async_shared_copy_confirm_s32k_h16_gpu4_20260620.jsonl
```

All paths are under:

```text
/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/results
```

Workload:

```text
B=1, H=16, S=32768, chunk64, DK=DV=128, fp16
```

| Candidate | Correctness | TileLang latency | Same-run Triton | Lesson |
| --- | --- | ---: | ---: | --- |
| 005 copy baseline | exact | 0.46791766 ms | 0.30507803 ms | Correct baseline, but far behind Triton. |
| 008 no-store diagnostic | skipped by design | 0.13233657 ms | 0.30706382 ms | Compute path was not the bottleneck; output store dominated. |
| 010 shared-copy store | exact | 0.37684782 ms | 0.30497479 ms | Routing stores through shared memory materially helped. |
| 014 swizzled async shared-copy | exact | 0.27223574 ms | 0.30525394 ms | Swizzled shared output staging plus async loads beat same-run Triton. |
| 016 confirmation | exact | 0.27507705 ms | 0.30761285 ms | Accepted isolated TileLang recompute candidate. |

Teaching point:

```text
T.async_copy/cp.async visibility was not enough. The decisive diagnosis was
that the generated output store route, not the input copy primitive alone, was
the real bottleneck.
```

## Blocksolve-A Candidate Ladder

Raw artifacts:

```text
blocksolve_a_tl_candidate_002_s32k.jsonl
blocksolve_a_tl_candidate_003_s32k.jsonl
blocksolve_a_tl_candidate_014_s32k.jsonl
blocksolve_a_tl_candidate_018_s32k.jsonl
current_tilelang_blocksolve_vs_triton_after_018_s32k.jsonl
```

All paths are under:

```text
/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/results
```

Workload:

```text
B=1, H=16, S=32768, chunk64, DK=128, fp16
```

| Candidate | Status | TileLang latency | Same-run Triton/reference | Lesson |
| --- | --- | ---: | ---: | --- |
| 001 fragment RR solve | compile failure | n/a | n/a | Stock TileLang 0.1.9 could not express direct Triton-style fragment recirculation. |
| 002 fp32 shared solve | correct | 0.34256142 ms | 0.15223822 ms | Shared solve was expressible but too slow. |
| 003 fp16 shared solve | correct | 0.18865764 ms | 0.14846226 ms | Half shared solve closed much of the gap. |
| 014 anchored gate precompute | correct | 0.16825072 ms | 0.19289172 ms | Safe anchored gate precompute reduced repeated exponent work. |
| 018 one-round Neumann | correct within tolerance | 0.12971870 ms | 0.15802626 ms | Tuning within the Neumann/inverse space produced the accepted candidate. |
| Final production compare | correct within tolerance | 0.11384284 ms | 0.14692616 ms | Production TileLang blocksolve-A beats local Triton isolated. |

Teaching point:

```text
The path was not "remove Triton at any cost." With the expert pipeline
understood, the AKO loop searched TileLang's expressible space and respected
the DSL boundary where direct fragment recirculation was not available.
```
