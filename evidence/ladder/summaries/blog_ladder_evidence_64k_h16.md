# Blog Ladder Evidence: GDN Prefill 64K/H16

Purpose: writing-facing evidence package for the GDN prefill blog. This file
separates experiment-adapter rows from final/external anchors so the
blog does not mix attribution lanes.

Source evidence:

- Summary: `experiments/gated_deltanet_prefill_blog_ladder/summaries/formal_64k_h16_v6_ladder.md`
- Section 11 ablation:
  `experiments/gated_deltanet_prefill_blog_ladder/summaries/section11_a_producer_ablation_64k_h16.md`
- JSONL: `experiments/gated_deltanet_prefill_blog_ladder/results/formal_64k_h16_v6_ladder.jsonl`
- Shape: `B=1,T=65536,H=16,DK=128,DV=128,chunk=64,fp16,BTHD`
- Input artifact: `experiments/gated_deltanet_prefill_blog_ladder/results/artifacts/formal_64k_h16_seed20260630.pt`
- Input hash: `sha256:a8987a2c6d16c658a1cb8ed95e409d973a3f736e2019d8719b143f18b4741513`
- Timer: CUPTI kernel-only with CUDA-event fallback; warmup `10`, repeat `50`, trials `3`
- GPU contract: H200 / GPU4

## Controlled Producer-Comparison Rows

These rows share the same formal input hash, pass correctness against the
recorded FLA reference, and are marked `causal_ladder_eligible=true` in the
harness. That machine-readable field means they are allowed into the controlled
experiment table; it does not mean every row should be a headline narrative
milestone.

| Role | Variant | Blog meaning | Latency ms | Speedup vs previous | Use |
| --- | --- | --- | ---: | ---: | --- |
| baseline | `generic_a_legacy` | Current-repo generic A producer plus legacy replay/output baseline. | 25.3849 | 1.00x | Starting point for controlled TileOps producer comparison. |
| first CP adaptation | `tileops_owned_cp_generic_a` | Same generic A producer class moved under the PR1596 CP downstream ABI and fused replay/output schedule. | 5.3912 | 4.71x | First correct TileOps-owned adaptation after studying FlashQLA; useful for V5/V6 comparison, not a claim that TileOps reproduced FlashQLA performance. |
| producer-swap adapter | `tileops_owned_cp_blocked_inverse_a` | Same CP downstream ABI as the intermediate row, but swaps in blocked-inverse / Neumann-style blocksolve A producer. | 0.746707 | 7.22x | Useful bridge evidence, but not the clean Section 11 A-producer ablation. |

Controlled end-to-end speedup from the baseline row to the producer-swap row:

```text
25.3849 ms / 0.746707 ms = 33.99x
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
  -> public FlashQLA producer + TileOps replay exceeds public FlashQLA estimate
  -> TileOps blocksolve / Neumann-style A producer reduces producer-side cost
```

The clean Section 11 numbers are:

| Evidence | Latency ms | Meaning |
| --- | ---: | --- |
| public FlashQLA full | 1.304489 | external TL0.1.8 anchor |
| public FlashQLA producer | 0.471943 | `chunk_local_cumsum + kkt_solve` component |
| public FlashQLA replay | 0.864754 | public `cp_preprocess + fused_gdr_fwd` component |
| public FlashQLA A/g + TileOps replay | 0.542807 | replay/output alignment and improvement row |
| TileOps A/g + TileOps replay | 0.542905 | same replay path with TileOps A/g |
| TileOps blocksolve A + TileOps replay, include producers | 0.691642 | A-producer-side improvement under TileOps replay |
| public FlashQLA producer + TileOps replay estimate | 1.014750 | cross-environment component estimate; not a single fused full path |

These numbers show that TileOps replay/output became faster before invoking the
blocked-inverse / Neumann-style A producer. They also show that the TileOps A
producer then reduced the producer-side cost further. This is the evidence to
use for Section 11, not `5.3912 ms -> 0.746707 ms` by itself.

## External And Final Anchors

These rows are useful context, but they should not be mixed into the controlled
experiment-adapter chain as if they were intermediate algorithmic steps.

| Variant | Role | Latency ms | Correctness | Use in blog |
| --- | --- | ---: | --- | --- |
| `ref_fla_051` | External correctness oracle and FLA latency baseline. | 18.7565 | self/reference row | May be reported as the recorded vendored FLA reference baseline, with version caveat. |
| `tileops_final_dispatch` | Final production wrapper / dispatch context from PR1596. | 0.722839 | pass vs FLA reference | May be reported as the final production dispatch row, not as an experiment-adapter step. |

`tileops_final_dispatch` is slightly faster than the explicit V6 adapter:

```text
0.746707 ms / 0.722839 ms = 1.03x
```

This is a production wrapper / dispatch-context observation. It should not be
written as a new algorithmic jump after the blocked-inverse A producer.

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
- The refreshed Section 11 cross-ablation supports: with public FlashQLA A/g
  fixed, TileOps replay is `0.542807 ms`; with TileOps A/g fixed, the same
  replay is `0.542905 ms`; and TileOps full producer plus replay is
  `0.691642 ms`. That is the cleaner evidence for the A/replay split.
- `tileops_owned_cp_generic_a -> tileops_owned_cp_blocked_inverse_a` supports
  only an experiment-adapter bridge under the same CP downstream ABI; it should
  not be presented as the main A-producer ablation.
- `tileops_owned_cp_blocked_inverse_a -> tileops_final_dispatch` supports:
  final production wrapper / dispatch context is consistent with the V6
  blocked-inverse path and is the row to cite for production-candidate latency.

Not supported:

- Claiming V5 and V6 have numerically equivalent A tensors.
- Treating `tileops_final_dispatch` as an additional causal algorithmic step.
- Stating an externally verified FLA 0.5.1 comparison without the current
  reference-version caveat.
