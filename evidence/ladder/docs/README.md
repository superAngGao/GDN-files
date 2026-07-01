# GDN Prefill Blog Ladder Harness

Experiment-only harness for the controlled evidence ladder described in
`/home/ga/2026-06-15/gdn-prefill-ai-assisted-blog/drafts/gdn_prefill_blog_experiment_plan_20260630.md`.

This directory is intentionally separate from production dispatch.  Every run
must name a variant explicitly:

```bash
python experiments/gated_deltanet_prefill_blog_ladder/run_ladder.py \
  --variant ref_fla_051 \
  --variant tileops_final_dispatch \
  --seq-len 65536 --heads 16 --dim-k 128 --dim-v 128 --chunk-size 64 \
  --dtype fp16 \
  --output experiments/gated_deltanet_prefill_blog_ladder/results/ladder_64k_h16.jsonl
```

For a quick compile/runtime smoke:

```bash
python experiments/gated_deltanet_prefill_blog_ladder/run_ladder.py \
  --smoke \
  --variant ref_fla_051 \
  --variant generic_a_legacy \
  --variant tileops_final_dispatch \
  --output experiments/gated_deltanet_prefill_blog_ladder/results/smoke_ladder.jsonl
```

`--smoke` uses `--smoke-shape blog` by default:

```text
B=1, T=512, H=16, DK=DV=128, chunk64, fp16
```

It also lowers timing to `warmup=1, repeat=3, trials=1` unless those timing
arguments are supplied explicitly.  It does not override explicitly supplied
shape arguments.  Use `--smoke-shape tiny` for the smaller diagnostic shape:

```text
B=1, T=512, H=2, DK=DV=64, chunk64, fp16
```

Smoke rows are written with `run_role="smoke"` and
`publication_eligible=false`.  Non-smoke runs default to `run_role="formal"`;
formal rows become publication eligible only when the row decision remains
accepted and full-op correctness passes.  The gate also writes
`reference_version_verified`; an unverified FLA version is reported as a
warning rather than used as an automatic rejection.

## Contract

Canonical input layout is BTHD:

- `q/k`: `[B, T, H, DK]`
- `v/o`: `[B, T, H, DV]`
- `g/beta`: `[B, T, H]`
- `initial_state`: `[B, H, DK, DV]`, zero by default

Main row:

```text
B=1, T=65536, H=16, DK=DV=128, chunk64, fp16, BTHD, GPU4/H200, scale=1.0
```

Final sweep rows:

```text
32K/H16, 64K/H16, 128K/H16, 128K/H32
```

Input distributions are recorded in each JSONL row:

- `q/k/v`: `torch.randn * 0.1`
- `g`: `-torch.rand`, raw log-gate input in `[-1, 0)`
- `beta`: `torch.rand * 0.5`
- `initial_state`: zeros fp32

The harness records a content hash for all generated tensors.  Use `--artifact`
to persist tensors with `torch.save`.

## Evidence Lanes

- `controlled_full_op`: same-repo/same-contract TileOps rows only.
- `conditional_full_op`: rows that may join the ladder only after full-op
  correctness and latency pass.
- `component`: component-only diagnostics, never mixed into full-op rows.
- `external_anchor`: FLA/FlashQLA reference rows outside TileOps causal claims.
- `negative_diagnostic`: failed candidates and rejected directions.

Public FlashQLA rows remain external anchors.  Component rows stay out of the
full-op ladder.  `tileops_final_dispatch` is a final candidate/current
production smoke row; it is not a substitute for the V5/V6 A-producer swap.

Every row also carries:

- `publication_role`: table role such as `causal_ladder_row`,
  `final_candidate`, `external_anchor`, or `negative_diagnostic`.
- `causal_ladder_eligible`: machine-readable guard for causal-ladder
  aggregation.  `tileops_final_dispatch` is `false` even though it is a
  full-op row.
- `run_role` and `publication_eligible`: row-level guards that keep smoke or
  diagnostic runs out of publication tables even when the variant itself is a
  candidate for the formal ladder.
- `environment.used_code_root`: the actual implementation root used by the row.
  TileOps adapters explicitly reactivate their intended root before import, so
  variant order cannot silently swap current-repo and production-candidate
  modules.

Correctness rows use:

```text
pass_rule = torch.allclose(actual.float(), reference.float(), atol=atol, rtol=rtol)
```

`max_abs` and `max_rel` are diagnostic metrics.  Large max-relative values near
zero reference elements do not independently fail a row unless the `pass_rule`
fails.

Each correctness block records `reference_used_code_root`, so a non-FLA row can
be audited even when the command does not also include a separate `ref_fla_051`
row.

## Files

- `variant_inventory.md`: current availability audit.
- `variants.py`: stable variant registry and callable adapters.
- `run_ladder.py`: full-op correctness/latency JSONL runner.
- `summarize_ladder.py`: markdown summary and table-filter generator for
  publication, controlled causal, final-candidate, and external-anchor rows.
- `collect_component_breakdown.py`: placeholder component JSONL writer until
  event boundaries are wired.
- `results/`: JSONL outputs.
- `generated_code/`: future lowering/PTX/SASS archive.
- `summaries/`: future markdown summaries derived from JSONL.
