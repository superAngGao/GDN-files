# Blog Ladder Evidence: GDN Prefill 64K/H16

Purpose: writing-facing evidence package for the GDN prefill blog. This file
separates experiment-adapter rows from final/external anchors so the
blog does not mix attribution lanes.

Source evidence:

- Historical local summary:
  `evidence/ladder/summaries/formal_64k_h16_historical_local.md`
- Current TileOps summary:
  `evidence/ladder/summaries/formal_64k_h16_current_gpu4_rerun.md`
- Section 11 ablation:
  `evidence/ladder/summaries/section11_a_producer_ablation_64k_h16.md`
- Historical local JSONL:
  `evidence/ladder/results/formal_64k_h16_historical_local.jsonl`
- Current TileOps JSONL:
  `evidence/ladder/results/formal_64k_h16_current_gpu4_rerun.jsonl`
- Shape: `B=1,T=65536,H=16,DK=128,DV=128,chunk=64,fp16,BTHD`
- Input artifact: local harness artifact
  `experiments/gated_deltanet_prefill_blog_ladder/results/artifacts/formal_64k_h16_seed20260630.pt`
  (large tensor artifact not mirrored into this repo; use the hash below for
  identity)
- Input hash: `sha256:a8987a2c6d16c658a1cb8ed95e409d973a3f736e2019d8719b143f18b4741513`
- Timer: CUPTI kernel-only with CUDA-event fallback; warmup `10`, repeat `50`, trials `3`
- GPU contract: H200 / GPU4

## Historical Local Full-Op Rows

These rows were rerun from detached historical worktrees under the same
`64K/H16` input artifact. They are story checkpoints for Level 2, not V5/V6
controlled producer-swap rows.

| Role | Variant | Blog meaning | Latency ms | Correctness | Use |
| --- | --- | --- | ---: | --- | --- |
| initial correctness | `local_initial_prefill_f147` | first serving prefill op checkpoint | 11.1762 | pass | first measurable end-to-end op |
| local prepare specialization | `local_prepare_specialized_00a60` | fixed-contract prepare specialization | 10.8353 | pass | local AKO positive full-op node |
| local h-tile diagnostic | `local_h_tile_tuned_827` | h tile tuning checkpoint | 10.1631 | fail | diagnostic only, not a positive story row |
| local wall | `local_bthd_wall_d09c` | optimized BTHD/local pre-CP path | 5.5566 | pass | closes Level 2 at the long-replay wall |

## Controlled Producer-Comparison Rows

These rows share the same formal input hash, pass correctness against the
recorded FLA reference, and are marked `causal_ladder_eligible=true` in the
harness. That machine-readable field means they are allowed into the controlled
experiment table; it does not mean every row should be a headline narrative
milestone.

| Role | Variant | Blog meaning | Latency ms | Speedup vs previous | Use |
| --- | --- | --- | ---: | ---: | --- |
| baseline | `generic_a_legacy` | Current-repo generic A producer plus legacy replay/output baseline. | 11.1906 | 1.00x | Starting point for controlled TileOps producer comparison. |
| first CP adaptation | `tileops_owned_cp_generic_a` | Same generic A producer class moved under the PR1596 CP downstream ABI and fused replay/output schedule. | 2.7674 | 4.04x | First correct TileOps-owned adaptation after studying FlashQLA; useful for V5/V6 comparison, not a claim that TileOps reproduced FlashQLA performance. |
| producer-swap adapter | `tileops_owned_cp_blocked_inverse_a` | Same CP downstream ABI as the intermediate row, but swaps in blocked-inverse / Neumann-style blocksolve A producer. | 0.715062 | 3.87x | Useful bridge evidence, but not the clean Section 11 A-producer ablation. |

Controlled end-to-end speedup from the baseline row to the producer-swap row:

```text
11.1906 ms / 0.715062 ms = 15.65x
```

Use this only as an experiment-adapter chain:

```text
generic_a_legacy
  -> tileops_owned_cp_generic_a
  -> tileops_owned_cp_blocked_inverse_a
```

Do not insert `tileops_final_dispatch` into this chain. It is a final candidate
/ production wrapper anchor, not a separate algorithmic step. Also do not write
V5 as "TileOps matched FlashQLA." V5 is the first correct adaptation; the
A/replay cross-ablation is the evidence for replay alignment and A-producer
attribution.

V5 is still useful evidence. Its poor performance relative to public FlashQLA,
together with the mixed TileOps-owned implementation path and conservative
generic A producer, supports the process claim that the agent was adapting a
schedule idea rather than reproducing a finished kernel. The incomplete
intermediate row motivated the cleaner A/replay attribution experiment.

The FlashQLA-learning sequence should be written as:

```text
local wall
  -> V5 first correct CP adaptation, still not performance-near FlashQLA
  -> TL0.1.8-lowering FlashQLA-style prepare A + TileOps replay/output full
     row: 0.815029 ms
  -> TileOps blocksolve / Neumann-style A producer full row: measured
```

The clean Section 11 table should use full end-to-end rows:

| Evidence | Latency ms | Meaning |
| --- | ---: | --- |
| public FlashQLA full | 1.306838 | external TL0.1.8 anchor, refreshed on GPU3 |
| FlashQLA-style prepare A + TileOps replay/output full row | 0.815029 | measured TL0.1.8 lowered KKT injected via external launcher plus TileOps replay |
| TileOps blocksolve A + TileOps replay/output full row | 0.695237 | same-scope measured TileOps prepare-A row |

Replay-only and component-sum rows can stay in supporting diagnostics, but
they should not replace the headline Section 11 rows. The native current-TL KKT
port remains rejected, but the missing no-Neumann combined row is filled by the
TL0.1.8-lowering injection measurement.

We also attempted a true measured combined row:

```text
current-TL FlashQLA-style KKT producer + TileOps replay
```

It failed correctness at `64K/H16` under `default`, `legacy`, and `wgmma`
GEMM compatibility modes, producing nonfinite outputs. That rejected diagnostic
remains supporting evidence rather than a performance point. The passing
combined row uses the TL0.1.8 lowered KKT via an external launcher.

## External And Final Anchors

These rows are useful context, but they should not be mixed into the controlled
experiment-adapter chain as if they were intermediate algorithmic steps. For
the main article's productionization section, the stronger claim is now the
refreshed production-surface sweep rather than the single `64K/H16` wrapper
anchor.

| Variant | Role | Latency ms | Correctness | Use in blog |
| --- | --- | ---: | --- | --- |
| `ref_fla_051` | External correctness oracle and FLA latency baseline. | 8.02574 | self/reference row | May be reported as the recorded vendored FLA reference baseline, with version caveat. |
| `tileops_final_dispatch` | Final production wrapper / dispatch context from PR1596. | 0.692026 historical anchor; 0.6951 in refreshed surface sweep | pass vs FLA reference | May be reported as the production dispatch surface, not as an experiment-adapter step. |

The historical `tileops_final_dispatch` anchor is slightly faster than the
explicit V6 adapter:

```text
0.715062 ms / 0.692026 ms = 1.03x
```

This is a production wrapper / dispatch-context observation. It should not be
written as a new algorithmic jump after the blocked-inverse A producer.

Production-surface evidence:

- TileOps vs FLA, GPU3/H200, TileOps benchmark infrastructure:
  `evidence/ladder/results/production_surface_tileops_vs_fla_20260701_tmpdir.jsonl`.
- Public FlashQLA TL0.1.8 Docker sweep:
  `evidence/ladder/results/production_surface_flashqla_20260701.jsonl`.

The measured TileOps production-dispatch rows are `0.3723 ms` at `32K/H16`,
`0.6951 ms` at `64K/H16`, `1.2284 ms` at `128K/H16`, `1.2238 ms` at
`64K/H32`, and `2.3085 ms` at `64K/H64`.

The corresponding public FlashQLA TL0.1.8 full-op rows are `0.5440 ms` at
`32K/H16`, `1.3073 ms` at `64K/H16`, `2.6055 ms` at `128K/H16`,
`2.5942 ms` at `64K/H32`, and `6.7233 ms` at `64K/H64`. Under the
public-environment comparison lane, TileOps production dispatch is `146%-291%`
of public FlashQLA throughput across this measured surface. This is still not a
same-lowering replay attribution experiment.

## Source / ABI Caveats

The safest blog wording is:

```text
V5 and V6 use the same CP downstream ABI and materialized A handoff
shape/layout, but they use different A producers. Both rows are full-op
correct against the recorded FLA reference.
```

Do not write:

```text
V5 and V6 have numerically equivalent A tensors.
```

The V5/V6 A comparison explicitly shows `allclose=false`.

| Pair / row | ABI/source fact | Evidence |
| --- | --- | --- |
| V5 `tileops_owned_cp_generic_a` | Experiment adapter using current-repo generic A producer plus PR1596 CP downstream. | `used_code_root.kind=mixed_experiment_roots`; generic A module `/home/ga/TileOPs/tileops/kernels/gated_deltanet/fused_prepare_compute_w_u.py`; CP downstream module `/home/ga/TileOPs-pr1596/tileops/kernels/gated_deltanet/gdn_prefill/fused_fwd.py`. |
| V6 `tileops_owned_cp_blocked_inverse_a` | Experiment adapter using PR1596 blocked-inverse / blocksolve A producer plus the same PR1596 CP downstream. | `used_code_root.kind=production_root_experiment_adapter`; blocked-inverse A module `/home/ga/TileOPs-pr1596/tileops/kernels/gated_deltanet/gated_deltanet_prefill.py`; CP downstream module `/home/ga/TileOPs-pr1596/tileops/kernels/gated_deltanet/gdn_prefill/fused_fwd.py`. |
| V5/V6 A comparison | Same materialized A handoff shape/layout, different producer math / numerics. | `A allclose=false`; `max_abs=0.117279`; V5 `max_rel=20583.9`; V6 `max_rel=29546.4`. |
| V6 dispatch wrapper scope | Explicit adapter row, not production dispatch wrapper. | `uses_production_dispatch_wrapper=false`; `uses_pr1596_cp_downstream=true`; `a_producer=blocked_inverse_blocksolve`. |
| Final dispatch | Production wrapper from PR1596. | `uses_production_dispatch_wrapper=true`; module `/home/ga/TileOPs-pr1596/tileops/ops/gated_deltanet.py`. |

## FLA Reference Caveat

All formal rows have:

```text
reference_version_verified=false
version_status=unverified_commit_based_reference
vendor_commit_file=91d2f468944842ab2d947350d280ca1db793db57
```

This does not invalidate the controlled TileOps internal ladder, because the
same recorded reference and same input hash are used consistently for
correctness. However, external FLA claims should be phrased conservatively:

```text
recorded vendored FLA reference
```

or should include a footnote that the requested `FLA 0.5.1` package identity was
not independently verified in this run.

## Suggested Blog Claims

Supported:

- `generic_a_legacy -> tileops_owned_cp_generic_a` supports: moving into the
  CP downstream family breaks the legacy replay/output wall under a controlled
  generic-A setup. It does not support claiming that V5 reproduced FlashQLA
  performance.
- V5's underperformance, mixed implementation path, and conservative generic A
  producer support a process claim: the agent was adapting an external schedule
  idea rather than reproducing a finished FlashQLA kernel, and the failed
  intermediate row helped identify the need for A/replay cross-ablation.
- The refreshed replay-only/component rows support diagnostics: with public
  FlashQLA A/g fixed, TileOps replay is `0.542807 ms`; with TL0.1.8-lowering
  external A/g, TileOps replay is `0.542159 ms`; with TileOps A/g fixed, the
  same replay is `0.542905 ms`. Under the same TileOps `bench_kernel` timing
  path, TL0.1.8-lowering prepare plus TileOps replay is `0.815029 ms`, and
  TileOps full producer plus replay is `0.695237 ms`.
- `tileops_owned_cp_generic_a -> tileops_owned_cp_blocked_inverse_a` supports
  only an experiment-adapter bridge under the same CP downstream ABI; it should
  not be presented as the main A-producer ablation.
- `tileops_owned_cp_blocked_inverse_a -> tileops_final_dispatch` supports:
  final production wrapper / dispatch context is consistent with the V6
  blocked-inverse path. Cite the refreshed production-surface sweep for the
  productionization claim.

Not supported:

- Claiming V5 and V6 have numerically equivalent A tensors.
- Treating `tileops_final_dispatch` as an additional causal algorithmic step.
- Stating an externally verified FLA 0.5.1 comparison without the current
  reference-version caveat.
