# Section 11 A-Producer Ablation: GDN Prefill 64K/H16

Purpose: provide the missing experiment for the blog's Section 11. This note
replaces the incorrect shortcut of using the V5 `5.3912 ms` bridge row as if it
were "FlashQLA-style A plus production replay." V5 is a first correct CP
adaptation, not the controlled A-producer ablation.

## Contract

- Shape: `B=1,T=65536,H=16,DK=128,DV=128,chunk=64,fp16,BTHD`
- GPU: H200 / `CUDA_VISIBLE_DEVICES=4`
- Timer: CUPTI kernel-only with L2 flush
- Timing contract: `warmup=5, repeat=20, trials=3`
- Public FlashQLA environment: TL0.1.8 docker, installed `flash_qla` package
- TileOps replay environment: current host TileOps harness using PR1596 root
- Public FlashQLA artifact:
  `/home/ga/Documents/gdn_kernel_bench_2026-06-18/results/flashqla_cross_ablation/artifacts/fq_tl018_64k_h16_seed20260630.pt`
- Artifact hash:
  `sha256:4ba1e0c0c92ade7cd415b04f57f7f8ab93ba4781437daa6f81ac899184053810`

## Results

| Row | A producer | Replay/output | Timing scope | Correctness reference | Latency ms |
| --- | --- | --- | --- | --- | ---: |
| public FlashQLA full | public FlashQLA TL0.1.8 KKT | public FlashQLA TL0.1.8 CP replay | full public op | public FlashQLA self row | 1.304489 |
| public FlashQLA producer | public FlashQLA TL0.1.8 KKT | producer only | `chunk_local_cumsum + kkt_solve` | component timing only | 0.471943 |
| public FlashQLA replay | exported public FlashQLA A/g | public FlashQLA TL0.1.8 CP replay | `cp_preprocess + fused_gdr_fwd` | component timing only | 0.864754 |
| `FQ18/TO` | exported public FlashQLA TL0.1.8 A/g | TileOps PR1596 CP replay | replay only | recorded vendored FLA reference | 0.542807 |
| `TO/TO replay` | TileOps blocksolve A | TileOps PR1596 CP replay | replay only | recorded vendored FLA reference | 0.542905 |
| `TO/TO full` | TileOps blocksolve A | TileOps PR1596 CP replay | include producers | recorded vendored FLA reference | 0.691642 |

All refreshed TileOps rows pass correctness against the recorded vendored FLA
reference.

## Rejected Measured Combined Rows

We also tried the row that would remove the need for the component-sum
estimate:

```text
current-TL FlashQLA-style KKT producer + TileOps PR1596 replay
```

This is exactly the "replace TileOps blocksolve / Neumann producer with a
FlashQLA-style producer under the same TileOps replay path" test. It is
measurable in the harness, but it is not correct at `64K/H16`.

| Row | GEMM compatibility mode | Timing scope | Latency ms | Correctness | Failure |
| --- | --- | --- | ---: | --- | --- |
| `FQ/TO` | `default` | include producers | 0.811018 | fail | `o` has 18,898,944 nonfinite values; final_state has 16,384 nonfinite values |
| `FQ/TO` | `legacy` | include producers | 1.958386 | fail | `o` has 31,887,360 nonfinite values; final_state differs beyond tolerance |
| `FQ/TO` | `wgmma` | include producers | 0.808363 | fail | `o` has 20,946,944 nonfinite values; final_state has 16,384 nonfinite values |

The root cause is the current-TL FlashQLA KKT producer, not the replay
handoff. A direct diagnostic on the same artifact showed:

```text
g_cum current-TL vs TL0.1.8 artifact: exact match
current-TL KKT A: 562 nonfinite values, range hits +/-65504
TL0.1.8 exported A: 0 nonfinite values, range [-0.269, 1.0]
```

Therefore, the measured combined row exists but is rejected. Until the
current-TL FlashQLA-style KKT producer is fixed, the correct evidence remains:

1. public TL0.1.8 FlashQLA producer component;
2. exported TL0.1.8 A/g artifact;
3. TileOps replay on that artifact.

That is why the `public FlashQLA producer + TileOps replay` number is still a
component-sum estimate rather than a measured full row.

## Derived Comparisons

TileOps replay is faster than the public FlashQLA replay component on this
shape:

```text
0.864754 ms / 0.542807 ms = 1.59x
```

The TileOps replay latency is essentially unchanged when driven by public
FlashQLA A/g or TileOps A/g:

```text
FQ18 A/g + TileOps replay:    0.542807 ms
TileOps A/g + TileOps replay: 0.542905 ms
```

This supports the claim that the replay/output implementation itself improved;
it is not merely faster because the A producer changed.

A conservative cross-environment estimate for public FlashQLA producer plus
TileOps replay is:

```text
0.471943 ms + 0.542807 ms = 1.014750 ms
```

That estimate is faster than the public FlashQLA full path:

```text
1.304489 ms / 1.014750 ms = 1.29x
```

The TileOps blocksolve producer plus the same TileOps replay path is faster
again:

```text
1.014750 ms / 0.691642 ms = 1.47x
```

The implied TileOps producer-side cost in this harness is:

```text
0.691642 ms - 0.542905 ms = 0.148737 ms
```

This is much lower than the public FlashQLA producer component:

```text
0.471943 ms / 0.148737 ms = 3.17x
```

## Supported Narrative

Use this sequence in the blog:

```text
local wall
-> V5 first correct CP adaptation, but still not FlashQLA-performance-near
-> public FlashQLA A/g + TileOps replay already beats public FlashQLA by a
   conservative component-sum estimate
-> TileOps blocksolve / Neumann-style A producer reduces the producer side
   further under the same TileOps replay path
```

This is stronger than saying "V5 was slow, then V6 was fast." It explains the
two independent axes:

1. TileOps replay/output became faster under the CP-split schedule family.
2. TileOps A producer became faster through the blocked-inverse /
   Neumann-style path.

## Caveats

The `public FlashQLA producer + TileOps replay` number is a cross-environment
component-sum estimate, not a single measured fused full path. The producer
component comes from the TL0.1.8 FlashQLA docker; the TileOps replay component
comes from the current TileOps harness.

We did attempt the single measured combined path in the current harness, but
the current-TL FlashQLA-style KKT producer failed correctness at `64K/H16`
under `default`, `legacy`, and `wgmma` compatibility modes. Those failed rows
should be reported as rejected diagnostics, not performance evidence.

The V5 bridge row should not be called "FlashQLA-style A plus production
replay." It uses a conservative generic TileOps A producer under a mixed
experiment adapter.

Current-TL FlashQLA migration KKT rows produced non-finite outputs at this
shape and are rejected for attribution.

## Evidence Files

- Public FlashQLA TL0.1.8 export:
  `/home/ga/Documents/gdn_kernel_bench_2026-06-18/results/flashqla_cross_ablation/fq_tl018_export_64k_h16.jsonl`
- Refreshed public FlashQLA A/g + TileOps replay:
  `experiments/gated_deltanet_prefill_blog_ladder/results/section11_a_producer_ablation_64k_h16_fq18_to_replay.jsonl`
- Refreshed TileOps A/g + TileOps replay:
  `experiments/gated_deltanet_prefill_blog_ladder/results/section11_a_producer_ablation_64k_h16_to_to_replay.jsonl`
- Refreshed TileOps full row:
  `experiments/gated_deltanet_prefill_blog_ladder/results/section11_a_producer_ablation_64k_h16_to_to_full.jsonl`
- Rejected measured current-TL FlashQLA-style producer + TileOps replay rows:
  `experiments/gated_deltanet_prefill_blog_ladder/results/section11_a_producer_ablation_64k_h16_fq_current_to_full.jsonl`
  `experiments/gated_deltanet_prefill_blog_ladder/results/section11_a_producer_ablation_64k_h16_fq_current_to_full_legacy.jsonl`
  `experiments/gated_deltanet_prefill_blog_ladder/results/section11_a_producer_ablation_64k_h16_fq_current_to_full_wgmma.jsonl`
