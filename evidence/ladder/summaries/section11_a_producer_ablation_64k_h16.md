# Section 11 A-Producer Ablation: GDN Prefill 64K/H16

Purpose: provide the missing experiment for the blog's Section 11. This note
replaces the incorrect shortcut of using the V5 `5.3912 ms` bridge row as if it
were "FlashQLA-style A plus production replay." V5 is a first correct CP
adaptation, not the controlled A-producer ablation.

## Contract

- Shape: `B=1,T=65536,H=16,DK=128,DV=128,chunk=64,fp16,BTHD`
- GPU: H200. The refreshed public FlashQLA anchor, TL0.1.8-lowering injection
  row, and same-scope TileOps blocksolve row were measured on
  `CUDA_VISIBLE_DEVICES=3`.
- Timer: `benchmarks.benchmark_base.bench_kernel` for the TileOps rows; CUPTI
  kernel-only with L2 flush and CUDA-event fallback.
- Timing contract: `warmup=5, repeat=20, trials=3`
- Public FlashQLA environment: TL0.1.8 docker, installed `flash_qla` package
- TileOps replay environment: current host TileOps harness using PR1596 root
- Public FlashQLA artifact:
  `/home/ga/Documents/gdn_kernel_bench_2026-06-18/results/flashqla_cross_ablation/artifacts/fq_tl018_64k_h16_seed20260630.pt`
- Artifact hash:
  `sha256:4ba1e0c0c92ade7cd415b04f57f7f8ab93ba4781437daa6f81ac899184053810`

## Main Full End-To-End Rows

These are the rows Section 11 should use in the main text.

| Row | Prepare-A producer | Replay/output | Timing scope | Correctness | Latency ms | Use |
| --- | --- | --- | --- | --- | ---: | --- |
| public FlashQLA full | public FlashQLA TL0.1.8 KKT | public FlashQLA TL0.1.8 CP replay | full public op | pass / public anchor | 1.306838 | external baseline |
| FlashQLA-style prepare A + TileOps replay | TL0.1.8 lowered FlashQLA KKT via external launcher | TileOps PR1596 CP replay | full combined row | pass vs public TL0.1.8 artifact | 0.815029 | measured no-Neumann combined row |
| TileOps prepare A + TileOps replay | TileOps blocksolve / Neumann-style A | TileOps PR1596 CP replay | full combined row | pass vs public TL0.1.8 artifact | 0.695237 | same benchmark-scope TileOps row |

The middle row is now filled with a measured, non-Neumann FlashQLA-style
prepare path. Instead of relying on the broken current-TL KKT migration, the
harness injects the public TL0.1.8 lowered KKT kernel through a small external
launcher, then feeds its `A` plus current `chunk_local_cumsum` into the
unchanged TileOps PR1596 replay path.

## Supporting Component Diagnostics

| Row | A producer | Replay/output | Timing scope | Correctness reference | Latency ms |
| --- | --- | --- | --- | --- | ---: |
| public FlashQLA full | public FlashQLA TL0.1.8 KKT | public FlashQLA TL0.1.8 CP replay | full public op | public FlashQLA self row | 1.306838 |
| public FlashQLA producer | public FlashQLA TL0.1.8 KKT | producer only | `chunk_local_cumsum + kkt_solve` | component timing only | 0.471233 |
| public FlashQLA replay | exported public FlashQLA A/g | public FlashQLA TL0.1.8 CP replay | `cp_preprocess + fused_gdr_fwd` | component timing only | 0.860569 |
| TL0.1.8-lowering external prepare | TL0.1.8 lowered KKT via external launcher | producer only | current `chunk_local_cumsum` + external `kkt_solve` | exact `A/g` vs public artifact | 0.470905 |
| TL0.1.8-lowering external full | TL0.1.8 lowered KKT via external launcher | TileOps PR1596 CP replay | full combined row | pass vs public TL0.1.8 artifact | 0.815029 |
| `FQ18/TO` | exported public FlashQLA TL0.1.8 A/g | TileOps PR1596 CP replay | replay only | recorded vendored FLA reference | 0.542807 |
| `TO/TO replay` | TileOps blocksolve A | TileOps PR1596 CP replay | replay only | recorded vendored FLA reference | 0.542905 |
| TileOps Neumann prepare | TileOps blocksolve A | producer only | current `chunk_local_cumsum` + blocksolve A | component timing only | 0.238202 |
| `TO/TO full` | TileOps blocksolve A | TileOps PR1596 CP replay | include producers | pass vs public TL0.1.8 artifact | 0.695237 |

The external-lowering rows pass against the public TL0.1.8 artifact; the older
replay-only TileOps diagnostics pass against the recorded vendored FLA
reference. These diagnostics explain the mechanism, but they should not replace
the main full-op Section 11 table.

## Rejected Measured Combined Rows

We also tried the row that would remove the need for the component-sum
estimate:

```text
current-TL FlashQLA-style KKT producer + TileOps PR1596 replay
```

This is exactly the "replace TileOps blocksolve / Neumann producer with a
FlashQLA-style producer under the same TileOps replay path" test. It is
measurable in the harness, but it is not correct at `64K/H16`, so its latency
is not performance evidence and should not be shown in the blog. The failed
diagnostics covered `default`, `legacy`, and `wgmma` compatibility modes; see
the JSON files listed below for raw failure details.

The root cause is the current-TL FlashQLA KKT producer, not the replay
handoff. A direct diagnostic on the same artifact showed:

```text
g_cum current-TL vs TL0.1.8 artifact: exact match
current-TL KKT A: 562 nonfinite values, range hits +/-65504
TL0.1.8 exported A: 0 nonfinite values, range [-0.269, 1.0]
```

Therefore, those current-TL measured combined rows exist but are rejected.
They should not overwrite the new passing TL0.1.8-lowering injection row.
The correct evidence is now:

1. public TL0.1.8 FlashQLA producer component;
2. TL0.1.8 lowered KKT injected through the external launcher;
3. unchanged TileOps PR1596 replay on the produced `A/g`;
4. exported TL0.1.8 artifact used for exact `A/g` and output correctness.

This is no longer only a component-sum estimate: the external-lowering row is
a measured full path in one host process.

## Derived Comparisons

TileOps replay is faster than the public FlashQLA replay component on this
shape:

```text
0.860569 ms / 0.542159 ms = 1.59x
```

The TileOps replay latency is essentially unchanged when driven by public
FlashQLA A/g or TileOps A/g:

```text
FQ18 A/g + TileOps replay:    0.542807 ms
External-lowering A/g replay: 0.542159 ms
TileOps A/g + TileOps replay: 0.542905 ms
```

This supports the claim that the replay/output implementation itself improved;
it is not merely faster because the A producer changed.

The measured TL0.1.8-lowering prepare row is effectively tied with, and very
slightly faster than, the refreshed public FlashQLA producer component:

```text
0.471233 ms / 0.470905 ms = 1.0007x
```

The measured TL0.1.8-lowering prepare plus TileOps replay full path is faster
than the refreshed public FlashQLA full path:

```text
1.306838 ms / 0.815029 ms = 1.60x
```

The TileOps blocksolve producer plus the same replay family is faster again:

```text
0.815029 ms / 0.695237 ms = 1.17x
```

The paired producer-only component rows show the prepare-side difference
without mixing in replay timing boundaries:

```text
0.470905 ms / 0.238202 ms = 1.98x
```

Do not derive a producer cost by subtracting the replay-only diagnostic from
the full row. The replay-only and full-row measurements are useful together,
but their event boundaries are not a strict additive decomposition.

## Supported Narrative

Use this sequence in the blog:

```text
local wall
-> V5 first correct CP adaptation, but still not FlashQLA-performance-near
-> TL0.1.8-lowering FlashQLA-style prepare A + TileOps replay/output full
   row: 0.815029 ms
-> TileOps blocksolve / Neumann-style A producer + TileOps replay/output full
   row: 0.695237 ms
```

This keeps Section 11 on full end-to-end rows. The replay-only and
component rows remain useful diagnostics, but the headline FlashQLA-style row
is now the measured TL0.1.8-lowering injection path.

## Caveats

The TL0.1.8-lowering injection row is a measured full path, but it is still an
external-kernel harness row rather than a production TileOps API row. It should
be described as "TL0.1.8 lowered FlashQLA KKT injected via external launcher +
TileOps replay," not as a native current-TL KKT port.

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
- Refreshed public FlashQLA TL0.1.8 same-GPU anchor:
  `/home/ga/Documents/gdn_kernel_bench_2026-06-18/results/flashqla_cross_ablation/tmp_rerun_gpu3_fq_tl018_64k_h16.jsonl`
- Measured TL0.1.8-lowering external prepare + TileOps replay full row:
  `evidence/ladder/results/section11_tileops_benchmark_ext_lowering_vs_neumann_64k_h16.jsonl`
- Direct-profiler diagnostic for TL0.1.8-lowering external row:
  `evidence/ladder/results/section11_tl018_lowering_ext_prepare_to_tileops_replay_64k_h16_gpu3.jsonl`
- Refreshed public FlashQLA A/g + TileOps replay:
  `evidence/ladder/results/section11_a_producer_ablation_64k_h16_fq18_to_replay.jsonl`
- Refreshed TileOps A/g + TileOps replay:
  `evidence/ladder/results/section11_a_producer_ablation_64k_h16_to_to_replay.jsonl`
- Refreshed TileOps full row:
  `evidence/ladder/results/section11_a_producer_ablation_64k_h16_to_to_full.jsonl`
- Rejected measured current-TL FlashQLA-style producer + TileOps replay rows:
  `evidence/ladder/results/section11_a_producer_ablation_64k_h16_fq_current_to_full.jsonl`
  `evidence/ladder/results/section11_a_producer_ablation_64k_h16_fq_current_to_full_legacy.jsonl`
  `evidence/ladder/results/section11_a_producer_ablation_64k_h16_fq_current_to_full_wgmma.jsonl`
