# A/Replay Cross-Ablation: GDN Prefill 64K/H16

Purpose: answer the blog-review question:

```text
If TileOps learned from the FlashQLA CP-split schedule, why does the generic-A
CP row not match FlashQLA? And if the final TileOps row beats FlashQLA, did the
replay side or the A-producer side improve?
```

This file is a writing-facing evidence note. It should be used to explain the
mechanism, not as a replacement for the TileOps experiment-adapter
rows in `blog_ladder_evidence_64k_h16.md`.

July 1 refresh: the TileOps replay rows below were rerun under the same
`warmup=5, repeat=20, trials=3` CUPTI/L2-flush timing contract as the public
FlashQLA TL0.1.8 export. These refreshed rows are supporting diagnostics for
Section 11, not the main full-op table.

We also attempted a true measured combined row using the current-TL migrated
FlashQLA-style KKT producer feeding TileOps replay. That row is measurable, but
it fails correctness at `64K/H16` in all tested GEMM compatibility modes, so it
is recorded as a rejected diagnostic rather than a performance row.

## Shape And Timer

- Shape: `B=1,T=65536,H=16,DK=128,DV=128,chunk=64,fp16,BTHD`
- GPU: H200 / `CUDA_VISIBLE_DEVICES=4`
- Timer: CUPTI kernel-only with L2 flush
- Public FlashQLA environment: TL0.1.8 docker, installed `flash_qla` package
- TileOps replay environment: current host TileOps harness using PR1596 root
- Public FlashQLA artifact:
  `/home/ga/Documents/gdn_kernel_bench_2026-06-18/results/flashqla_cross_ablation/artifacts/fq_tl018_64k_h16_seed20260630.pt`
- Artifact hash:
  `sha256:4ba1e0c0c92ade7cd415b04f57f7f8ab93ba4781437daa6f81ac899184053810`

## Results

This file is retained as a replay-side diagnostic. The headline Section 11
full end-to-end A-producer rows are maintained in
`evidence/ladder/summaries/section11_a_producer_ablation_64k_h16.md`, including
the measured TL0.1.8-lowering full row and the same benchmark-scope TileOps
Neumann row.

| Row | A producer | Replay/output | Timing scope | Correctness reference | Latency ms | Use |
| --- | --- | --- | --- | --- | ---: | --- |
| `FQ/FQ` | public FlashQLA TL0.1.8 KKT | public FlashQLA TL0.1.8 CP replay | full public op | public FlashQLA self row | 1.304489 | External anchor. |
| `FQ/FQ producer` | public FlashQLA TL0.1.8 KKT | producer-only row | `chunk_local_cumsum + kkt_solve` | component timing only | 0.471943 | Producer-side public anchor. |
| `FQ/FQ replay` | exported public FlashQLA A/g | public FlashQLA TL0.1.8 CP replay | `cp_preprocess + fused_gdr_fwd` | component timing only | 0.864754 | Replay-side public anchor. |
| `FQ18/TO` | exported public FlashQLA TL0.1.8 A/g | TileOps PR1596 CP replay | replay-only | recorded vendored FLA reference | 0.542807 | Refreshed formal Section 11 row. Tests public FlashQLA A/g plus TileOps replay. |
| `TO/TO replay` | TileOps blocksolve A | TileOps PR1596 CP replay | replay-only | recorded vendored FLA reference | 0.542905 | Refreshed formal Section 11 row. Same replay path with TileOps A/g. |
| `TO/TO full` | TileOps blocksolve A | TileOps PR1596 CP replay | include producers | recorded vendored FLA reference | 0.691642 | Older recorded-FLA diagnostic; not the headline Section 11 full row. |

## Immediate Interpretation

The old explanation was under-specified. The data says two things at once:

First, the FlashQLA-learning story should be presented as three nodes:

| Node | Evidence | Meaning |
| --- | --- | --- |
| local wall | direct fusion did not shorten the long replay dependency | local AKO needed an external schedule idea |
| first correct adaptation | V5 `tileops_owned_cp_generic_a = 5.3912 ms` | TileOps adapted the CP idea, but this was not performance-near FlashQLA |
| replay/output breakthrough before Neumann | public FlashQLA producer `0.471943 ms` + TileOps replay `0.542807 ms` gives a `1.014750 ms` estimate, faster than public FlashQLA full `1.304489 ms` | with FlashQLA-style A/KKT still in place, TileOps replay/output was already faster than the public FlashQLA back half |

1. The V5 `tileops_owned_cp_generic_a` row is not a faithful FlashQLA
   reproduction row. It is a controlled bridge row: generic TileOps A producer
   plus the TileOps-owned CP downstream ABI. Its `5.3912 ms` latency should not
   be used to say "learning the FlashQLA schedule only gets this far."

2. With public FlashQLA TL0.1.8 A/g fixed, TileOps replay is substantially
   faster than public FlashQLA replay under this benchmark method:

```text
0.864754 ms / 0.542807 ms = 1.59x
```

The same TileOps replay latency appears with either public FlashQLA A/g or
TileOps A/g:

```text
FQ18 A + TileOps replay: 0.542807 ms
TileOps A + TileOps replay: 0.542905 ms
```

So the replay/output speed improvement is not merely a side effect of using
TileOps A.

There is also an A-producer-side improvement. This older diagnostic used a
conservative cross-environment estimate for "public FlashQLA producer +
TileOps replay":

```text
0.471943 ms + 0.542807 ms = 1.014750 ms
```

That is faster than public FlashQLA full path:

```text
1.304489 ms / 1.014750 ms = 1.29x
```

but still slower than the older recorded-FLA TileOps full diagnostic:

```text
1.014750 ms / 0.691642 ms = 1.47x
```

The publication-facing full-row comparison is now the measured Section 11
table, not this component-sum estimate. This diagnostic still supports the
cleaner narrative:

```text
FlashQLA contributes the production-grade CP-split schedule idea.
TileOps later improves two implementation axes:
  1. replay/output implementation under the CP schedule;
  2. A producer via the blocked-inverse / Neumann-style path.
```

## Rejected Measured Combined Row

The row we would prefer to use is:

```text
current-TL FlashQLA-style KKT producer + TileOps replay
```

It was tested, but rejected by correctness:

| Row | GEMM compatibility mode | Latency ms | Correctness |
| --- | --- | ---: | --- |
| `FQ/TO` include producers | `default` | 0.811018 | fail, nonfinite output |
| `FQ/TO` include producers | `legacy` | 1.958386 | fail, nonfinite output |
| `FQ/TO` include producers | `wgmma` | 0.808363 | fail, nonfinite output |

Diagnostic summary:

```text
g_cum current-TL vs TL0.1.8 artifact: exact match
current-TL KKT A: 562 nonfinite values, range hits +/-65504
TL0.1.8 exported A: 0 nonfinite values, range [-0.269, 1.0]
```

This is why this older replay diagnostic could only use a component-sum
estimate for the FlashQLA-style prepare path. The current headline Section 11
row is no longer that estimate: it is the measured TL0.1.8-lowering KKT
external-launcher path plus TileOps replay, archived in
`evidence/ladder/summaries/section11_a_producer_ablation_64k_h16.md`.

## What This Does Not Prove

Do not claim that V5 is a full FlashQLA reproduction. It is not.

In this older diagnostic file, do not claim that the `producer + replay` sum
is a measured single fused full path. The producer part is measured in the
TL0.1.8 FlashQLA docker, while the TileOps replay part is measured in the
current TileOps harness. The sum is a useful cross-ablation estimate, not a
single-kernel measurement. The current headline Section 11 row is the measured
TL0.1.8-lowering external-launcher full path documented in
`evidence/ladder/summaries/section11_a_producer_ablation_64k_h16.md`.

Do not use the current-TL FlashQLA migration A producer as the public FlashQLA
producer at 64K/H16. In current-TL migration experiments, FQ A rows produced
non-finite output at 64K and are unsuitable for public attribution.

## Evidence Files

- Public FlashQLA TL0.1.8 export:
  `/home/ga/Documents/gdn_kernel_bench_2026-06-18/results/flashqla_cross_ablation/fq_tl018_export_64k_h16.jsonl`
- Refreshed public FlashQLA A/g + TileOps replay row:
  `evidence/ladder/results/section11_a_producer_ablation_64k_h16_fq18_to_replay.jsonl`
- Refreshed same-input TileOps full row:
  `evidence/ladder/results/section11_a_producer_ablation_64k_h16_to_to_full.jsonl`
- Refreshed same-input TileOps replay-only row:
  `evidence/ladder/results/section11_a_producer_ablation_64k_h16_to_to_replay.jsonl`
- Rejected measured current-TL FlashQLA-style producer + TileOps replay rows:
  `evidence/ladder/results/section11_a_producer_ablation_64k_h16_fq_current_to_full.jsonl`
  `evidence/ladder/results/section11_a_producer_ablation_64k_h16_fq_current_to_full_legacy.jsonl`
  `evidence/ladder/results/section11_a_producer_ablation_64k_h16_fq_current_to_full_wgmma.jsonl`
