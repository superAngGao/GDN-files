# GDN Prefill Ladder Summary

- Source JSONL: `experiments/gated_deltanet_prefill_blog_ladder/results/formal_64k_h16_historical_local.jsonl`
- Rows: 5
- Publication-eligible evidence rows: 4
- Controlled causal rows: 0
- Final candidate rows: 0
- External anchor rows: 1
- Shape: B=1,T=65536,H=16,DK=128,DV=128,chunk=64
- Input artifact: `experiments/gated_deltanet_prefill_blog_ladder/results/artifacts/formal_64k_h16_seed20260630.pt`
- Input hash: `sha256:a8987a2c6d16c658a1cb8ed95e409d973a3f736e2019d8719b143f18b4741513`
- Evidence lane note: `final_candidate` rows and `causal_ladder_row` rows are reported separately; this summary does not compute causal speedup between them.

## All Rows

| variant | role | lane | causal | formal accepted | correct | latency_ms | ref verified | caveat | used root | ref root |
|---|---|---|---|---|---|---:|---|---|---|---|
| ref_fla_051 | correctness_oracle_and_fla_baseline | external_anchor | false | true | pass | 8.02445 | false | FLA version unverified | fla_vendor | fla_vendor |
| local_initial_prefill_f147 | historical_local_ladder_row | historical_full_op | false | true | pass | 11.1762 | false | FLA version unverified | historical_worktree | fla_vendor |
| local_prepare_specialized_00a60 | historical_local_ladder_row | historical_full_op | false | true | pass | 10.8353 | false | FLA version unverified | historical_worktree | fla_vendor |
| local_h_tile_tuned_827 | historical_local_ladder_row | historical_full_op | false | false | fail | 10.1631 | false | FLA version unverified | historical_worktree | fla_vendor |
| local_bthd_wall_d09c | historical_local_wall_row | historical_full_op | false | true | pass | 5.5566 | false | FLA version unverified | historical_worktree | fla_vendor |

## Publication-Eligible Evidence Rows

| variant | role | lane | causal | formal accepted | correct | latency_ms | ref verified | caveat | used root | ref root |
|---|---|---|---|---|---|---:|---|---|---|---|
| ref_fla_051 | correctness_oracle_and_fla_baseline | external_anchor | false | true | pass | 8.02445 | false | FLA version unverified | fla_vendor | fla_vendor |
| local_initial_prefill_f147 | historical_local_ladder_row | historical_full_op | false | true | pass | 11.1762 | false | FLA version unverified | historical_worktree | fla_vendor |
| local_prepare_specialized_00a60 | historical_local_ladder_row | historical_full_op | false | true | pass | 10.8353 | false | FLA version unverified | historical_worktree | fla_vendor |
| local_bthd_wall_d09c | historical_local_wall_row | historical_full_op | false | true | pass | 5.5566 | false | FLA version unverified | historical_worktree | fla_vendor |

## Controlled Causal Ladder Rows

| variant | role | lane | causal | formal accepted | correct | latency_ms | ref verified | caveat | used root | ref root |
|---|---|---|---|---|---|---:|---|---|---|---|

## Final Candidate Rows

| variant | role | lane | causal | formal accepted | correct | latency_ms | ref verified | caveat | used root | ref root |
|---|---|---|---|---|---|---:|---|---|---|---|

## External Anchor Rows

| variant | role | lane | causal | formal accepted | correct | latency_ms | ref verified | caveat | used root | ref root |
|---|---|---|---|---|---|---:|---|---|---|---|
| ref_fla_051 | correctness_oracle_and_fla_baseline | external_anchor | false | true | pass | 8.02445 | false | FLA version unverified | fla_vendor | fla_vendor |

## Warnings

These warnings are also summarized in the main tables' caveat column.

- ref_fla_051: FLA 0.5.1 package version is not verified; use recorded reference source identity
- local_initial_prefill_f147: FLA 0.5.1 package version is not verified; use recorded reference source identity
- local_prepare_specialized_00a60: FLA 0.5.1 package version is not verified; use recorded reference source identity
- local_h_tile_tuned_827: FLA 0.5.1 package version is not verified; use recorded reference source identity
- local_bthd_wall_d09c: FLA 0.5.1 package version is not verified; use recorded reference source identity

## Code Source Audit

| variant | used module | root match | reference module |
|---|---|---|---|
| ref_fla_051 | /home/ga/TileOPs/.github/runner/vendor/flash-linear-attention/fla/ops/gated_delta_rule/chunk.py | N/A | /home/ga/TileOPs/.github/runner/vendor/flash-linear-attention/fla/ops/gated_delta_rule/chunk.py |
| local_initial_prefill_f147 | /home/ga/TileOPs-gdn-history/initial-f1472392/tileops/ops/gated_deltanet.py | true | /home/ga/TileOPs/.github/runner/vendor/flash-linear-attention/fla/ops/gated_delta_rule/chunk.py |
| local_prepare_specialized_00a60 | /home/ga/TileOPs-gdn-history/prepare-00a60b19/tileops/ops/gated_deltanet.py | true | /home/ga/TileOPs/.github/runner/vendor/flash-linear-attention/fla/ops/gated_delta_rule/chunk.py |
| local_h_tile_tuned_827 | /home/ga/TileOPs-gdn-history/htile-82707454/tileops/ops/gated_deltanet.py | true | /home/ga/TileOPs/.github/runner/vendor/flash-linear-attention/fla/ops/gated_delta_rule/chunk.py |
| local_bthd_wall_d09c | /home/ga/TileOPs-gdn-history/bthdwall-d09c8f2d/tileops/ops/gated_deltanet.py | true | /home/ga/TileOPs/.github/runner/vendor/flash-linear-attention/fla/ops/gated_delta_rule/chunk.py |
