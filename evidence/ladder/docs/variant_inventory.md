# Variant Inventory

Date: 2026-06-30

Scope: controlled GDN prefill evidence ladder for the blog plan.  This inventory
does not modify `tutorial_v3.md` and does not change production dispatch.

## Summary

| Variant | Lane | Publication role | Causal ladder eligible? | Current status | Decision |
| --- | --- | --- | --- | --- | --- |
| `local_initial_prefill_f147` | historical_full_op | historical_local_ladder_row | no | historical worktree adapter implemented; formal full-op correctness pass | accept as Level 2 story checkpoint |
| `local_prepare_specialized_00a60` | historical_full_op | historical_local_ladder_row | no | historical worktree adapter implemented; formal full-op correctness pass | accept as Level 2 story checkpoint |
| `local_h_tile_tuned_827` | historical_full_op | historical_local_ladder_row | no | historical worktree adapter implemented; formal full-op correctness fail | diagnostic_only |
| `local_bthd_wall_d09c` | historical_full_op | historical_local_wall_row | no | historical worktree adapter implemented; formal full-op correctness pass | accept as Level 2 wall row |
| `generic_a_legacy` | controlled_full_op | causal_ladder_row | yes | implemented in current repo as `GatedDeltaNetFwdOp` | accept for rerun |
| `generic_a_local_ako_best` | conditional_full_op | conditional_causal_ladder_row | no | historical traces exist, no clean current selectable full-op adapter found | unavailable |
| `generic_a_direct_fused_correct` | conditional_full_op | conditional_boundary_diagnostic | no | no clean correct full-op implementation found | unavailable |
| `generic_a_direct_fused_failed` | negative_diagnostic | negative_diagnostic | no | historical rejected fused diagnostics exist | diagnostic_only |
| `flashqla_public_tl018` | external_anchor | external_anchor | no | requires public FlashQLA TL0.1.8 env | unavailable |
| `flashqla_port_current_tl` | external_anchor | migration_lowering_anchor | no | schedule notes exist, no clean runnable adapter found | unavailable |
| `tileops_owned_cp_generic_a` | controlled_full_op | causal_ladder_row | yes | experiment-only adapter implemented; smoke/formal full-op correctness pass | accept with ABI caveat |
| `tileops_owned_cp_blocked_inverse_a` | controlled_full_op | causal_ladder_row | yes | experiment-only adapter implemented; smoke/formal full-op correctness pass | accept as V6 producer-swap row |
| `tileops_final_dispatch` | controlled_full_op | final_candidate | no | runnable from `/home/ga/TileOPs-pr1596` as current production candidate | accept as final/current candidate |

## Audit Notes

### Historical local full-op checkpoints

Status: implemented as detached worktree adapters.

These variants were added after the blog roadmap was tightened to use only
end-to-end rows.  They are not controlled V5/V6 producer-swap rows; they are
Level 2 story checkpoints before FlashQLA is introduced.

Formal `64K/H16`, GPU4/H200, same input artifact:

| Variant | Commit | Correctness | Latency ms | Use |
| --- | --- | --- | ---: | --- |
| `local_initial_prefill_f147` | `f1472392` | pass | `11.1762` | first measurable serving prefill op |
| `local_prepare_specialized_00a60` | `00a60b19` | pass | `10.8353` | local prepare specialization full-op node |
| `local_h_tile_tuned_827` | `82707454` | fail | `10.1631` | diagnostic only |
| `local_bthd_wall_d09c` | `d09c8f2d` | pass | `5.5566` | Level 2 local wall row |

Source roots:

```text
/home/ga/TileOPs-gdn-history/initial-f1472392
/home/ga/TileOPs-gdn-history/prepare-00a60b19
/home/ga/TileOPs-gdn-history/htile-82707454
/home/ga/TileOPs-gdn-history/bthdwall-d09c8f2d
```

### `generic_a_legacy`

Status: implemented in current repo.

Code pointer:

```text
tileops.ops.GatedDeltaNetFwdOp
tileops/kernels/gated_deltanet/gated_deltanet_fwd.py
tileops/kernels/linear_attn/prepare_wy_repr.py
```

Classification:

- A producer: generic exact/KKT-style WY inverse.
- Schedule: legacy chunk replay/output.
- Layout: production op is BHSD/BHTD style; the harness uses canonical BTHD and
  converts inside the variant adapter.
- Output: op returns `(o, S, Aw, Au)`; harness records `S[:, :, -1]` as
  `final_state`.

Decision: accepted for controlled V1 reruns, with layout-conversion scope
explicitly recorded.

### `generic_a_local_ako_best`

Status: needs resurrection.

Evidence found:

```text
/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/results/
/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/
```

Historical local-AKO component and full-op logs exist, including scale/store and
resource-shape sweeps.  I did not find a clean current source-level variant that
can be selected as an end-to-end generic-A full-op row under the fixed contract.

Decision: unavailable for now.  It may enter the controlled ladder only after a
specific end-to-end full-op implementation is revived and rerun with correctness
and latency.

### `generic_a_direct_fused_correct`

Status: not found.

No current clean direct-fused no-CP full-op implementation was found that both
passes full-op correctness and exposes end-to-end latency.  Historical notes
contain fused h+output diagnostics, but not a row that satisfies this variant's
gate.

Decision: unavailable.

### `generic_a_direct_fused_failed`

Status: historical diagnostic only.

Evidence pointer:

```text
/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/restart_checkpoint_20260616.md
```

The notes record rejected fused h+output/direct-fusion diagnostics.  These rows
are useful for explaining rejected candidates, but they are not controlled
performance milestones and do not prove the recurrence argument by themselves.

Decision: diagnostic_only.

### `flashqla_public_tl018`

Status: external environment required.

The plan requires public FlashQLA TL0.1.8 as an external anchor.  That anchor is
not a controlled TileOps row because repository, TileLang version, lowering,
wrapper, runtime environment, and producer implementation all change together.

Decision: unavailable in this harness until the public environment is supplied;
external_anchor only.

## Reference Identity Note

`ref_fla_051` remains the plan's intended FLA 0.5.1 reference row, but the
current smoke environment does not expose installed FLA package metadata.  The
harness therefore records:

```text
environment.fla_reference.requested_version = "0.5.1"
environment.fla_reference.package_version = null
environment.fla_reference.vendor_commit_file = <vendored commit>
environment.fla_reference.version_status = "unverified_commit_based_reference"
```

Formal 64K ladder rows must either verify an installed `0.5.1` package version
or document the commit-to-release mapping before using `ref_fla_051` as a
publication reference.

### `flashqla_port_current_tl`

Status: needs resurrection.

Evidence pointers:

```text
/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/flashqla_source_reading_20260622.md
/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/flashqla_scheduling_skeleton_20260623.md
/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/results/flashqla_specialized/
```

I found schedule/migration notes and quick specialized results, but not a clean
current TileLang adapter that can be selected from this ladder harness.

Decision: unavailable.  Keep as external/migration/lowering anchor.

### `tileops_owned_cp_generic_a`

Status: implemented as an experiment-only V5 adapter.

This is the critical V5 row.  The harness now builds a controlled full-op from:

```text
current repo generic exact/KKT-style fused_prepare_compute_w_u_tl
-> materialized A decoded to canonical B,T,H,chunk layout
-> PR1596 _prefill_partitioned_initial_state_bthd
-> PR1596 gdn_prefill.fused_gdr_fwd
```

Recorded ABI:

- A logical shape: `[B, T, H, chunk_size]`.
- Physical layout: contiguous BTHC handoff; produced as BHSC then permuted.
- Triangular convention: unit diagonal plus lower-triangular inverse rows.
- `g/g_cum`: generic A uses `g_zero`; chunk-local `g_cum` is passed separately
  to CP replay, matching the PR1596 partitioned ABI.
- Beta folding: beta is folded inside A producer and also passed to downstream
  as `b`.
- Dtype/store policy: A stored in input dtype, with fp32 internal accumulators.
- Materialization/handoff: materialized A tensor is handed to CP preprocess and
  fused replay/output.

Validation:

```text
smoke B=1,T=512,H=16,DK=DV=128,chunk64,fp16: correctness pass
formal B=1,T=65536,H=16,DK=DV=128,chunk64,fp16: correctness pass
```

A equivalence caveat: materialized generic A and PR1596 blocksolve A are
compared in canonical BTHC layout.  The formal 64K/H16 comparison records
`allclose=false`, `max_abs=0.117279`, `max_rel=20583.9` at
`atol=rtol=5e-2`.  Full-op correctness against the FLA reference passes, so V5
is accepted as a controlled full-op row, but the A-producer numerical delta must
remain visible in summaries.

Decision: accept as V5 controlled causal row.  Do not substitute final dispatch
or historical component rows for V5.

### `tileops_owned_cp_blocked_inverse_a`

Status: implemented as an experiment-only V6 adapter.

Code pointer:

```text
experiments/gated_deltanet_prefill_blog_ladder/variants.py::tileops_owned_cp_blocked_inverse_a adapter
/home/ga/TileOPs-pr1596/tileops/kernels/gated_deltanet/gated_deltanet_prefill.py::_prefill_blocksolve_A_bthd
/home/ga/TileOPs-pr1596/tileops/kernels/gated_deltanet/gdn_prefill/fused_fwd.py::fused_gdr_fwd
```

The harness now builds the pure V6 producer-swap row without calling the
production dispatch wrapper:

```text
PR1596 blocksolve A with 16x16 diagonal Neumann-style solve
-> materialized A decoded to canonical B,T,H,chunk layout
-> PR1596 _prefill_partitioned_initial_state_bthd
-> PR1596 gdn_prefill.fused_gdr_fwd
```

Recorded ABI matches V5:

- A logical shape: `[B, T, H, chunk_size]`.
- Physical layout: contiguous BTHC handoff.
- Triangular convention: unit diagonal plus lower-triangular inverse rows.
- `g/g_cum`: blocked-inverse A uses `g_zero`; chunk-local `g_cum` is passed
  separately to CP replay.
- Beta folding: beta is folded inside A producer and also passed to downstream
  as `b`.
- Dtype/store policy: A stored in input dtype, with fp32/solve accumulators as
  in PR1596.
- Materialization/handoff: materialized A tensor is handed to CP preprocess and
  fused replay/output.

Validation:

```text
smoke B=1,T=512,H=16,DK=DV=128,chunk64,fp16: correctness pass
formal B=1,T=65536,H=16,DK=DV=128,chunk64,fp16: correctness pass
```

A comparison caveat: V6 blocked-inverse A and V5 generic A are compared in
canonical BTHC layout.  The formal 64K/H16 comparison records
`allclose=false`, `max_abs=0.117279`, `max_rel=29546.4` at
`atol=rtol=5e-2`.  This is expected producer-swap evidence, not a full-op
correctness failure.

Decision: accept as V6 controlled causal row.  Keep `tileops_final_dispatch`
separate as final candidate.

### `tileops_final_dispatch`

Status: runnable from PR1596.

Code pointer:

```text
/home/ga/TileOPs-pr1596/tileops.ops.GatedDeltaNetPrefillFwdOp
/home/ga/TileOPs-pr1596/tileops/kernels/gated_deltanet/gated_deltanet_prefill.py
```

The current handoff identifies `prepare008` in PR1596 as the accepted production
node.  This row can be used as the final/current candidate smoke row:

```bash
python experiments/gated_deltanet_prefill_blog_ladder/run_ladder.py \
  --variant tileops_final_dispatch \
  --production-root /home/ga/TileOPs-pr1596
```

Decision: accepted only as final/current production candidate.  Machine-readable
metadata sets `publication_role="final_candidate"` and
`causal_ladder_eligible=false`; scripts must not use this row as a replacement
for `tileops_owned_cp_generic_a` or as a pure A-producer swap claim.

## A-Producer Swap Status

The V5/V6 controlled swap is now available as experiment-only full-op rows:

| Check | Status |
| --- | --- |
| same A semantics | established; V5 generic exact/KKT vs V6 blocked-inverse |
| same A logical shape | both record `[B,T,H,chunk]` |
| same physical layout/strides | both record contiguous BTHC |
| same triangular convention | both record unit diagonal/lower inverse |
| same `g/g_cum` convention | both use `g_zero` A plus downstream `g_cum` |
| same beta folding convention | both fold beta in A and pass downstream `b` |
| same dtype/store policy | both store A in fp16 with fp32/solve accumulators |
| same materialization/handoff | both hand materialized A to CP preprocess |
| same CP preprocess API | both use PR1596 `_prefill_partitioned_initial_state_bthd` |
| same fused replay/output API | both use PR1596 `gdn_prefill.fused_gdr_fwd` |
| A or downstream `w/u` comparison | V5/V6 A comparison collected; allclose=false |

Remaining work:

1. Keep `tileops_final_dispatch` as `publication_role=final_candidate` and
   `causal_ladder_eligible=false`.
2. Optional: broaden the formal sweep to 32K/H16, 128K/H16, and 128K/H32.

## Smoke Rows

The first runnable smoke should include:

```text
ref_fla_051
generic_a_legacy
tileops_final_dispatch
```

If `/home/ga/TileOPs-pr1596` is unavailable, `tileops_final_dispatch` must be
recorded as `unavailable`; do not replace it silently with `generic_a_legacy`.
