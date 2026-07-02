# GDN Prefill Ladder Summary

- Source JSONL: `experiments/gated_deltanet_prefill_blog_ladder/results/formal_64k_h16_current_gpu4_rerun.jsonl`
- Rows: 5
- Publication-eligible evidence rows: 5
- Controlled causal rows: 3
- Final candidate rows: 1
- External anchor rows: 1
- Shape: B=1,T=65536,H=16,DK=128,DV=128,chunk=64
- Input artifact: local harness artifact
  `experiments/gated_deltanet_prefill_blog_ladder/results/artifacts/formal_64k_h16_seed20260630.pt`
  (large tensor artifact not mirrored into this repo; use the hash below for
  identity)
- Input hash: `sha256:a8987a2c6d16c658a1cb8ed95e409d973a3f736e2019d8719b143f18b4741513`
- Evidence lane note: `final_candidate` rows and `causal_ladder_row` rows are reported separately; this summary does not compute causal speedup between them.

## All Rows

| variant | role | lane | causal | formal accepted | correct | latency_ms | ref verified | caveat | used root | ref root |
|---|---|---|---|---|---|---:|---|---|---|---|
| ref_fla_051 | correctness_oracle_and_fla_baseline | external_anchor | false | true | pass | 8.02574 | false | FLA version unverified | fla_vendor | fla_vendor |
| generic_a_legacy | causal_ladder_row | controlled_full_op | true | true | pass | 11.1906 | false | FLA version unverified | tileops_repo | fla_vendor |
| tileops_owned_cp_generic_a | causal_ladder_row | controlled_full_op | true | true | pass | 2.7674 | false | FLA version unverified | mixed_experiment_roots | fla_vendor |
| tileops_owned_cp_blocked_inverse_a | causal_ladder_row | controlled_full_op | true | true | pass | 0.715062 | false | FLA version unverified | production_root_experiment_adapter | fla_vendor |
| tileops_final_dispatch | final_candidate | controlled_full_op | false | true | pass | 0.692026 | false | FLA version unverified | production_root | fla_vendor |

## Publication-Eligible Evidence Rows

| variant | role | lane | causal | formal accepted | correct | latency_ms | ref verified | caveat | used root | ref root |
|---|---|---|---|---|---|---:|---|---|---|---|
| ref_fla_051 | correctness_oracle_and_fla_baseline | external_anchor | false | true | pass | 8.02574 | false | FLA version unverified | fla_vendor | fla_vendor |
| generic_a_legacy | causal_ladder_row | controlled_full_op | true | true | pass | 11.1906 | false | FLA version unverified | tileops_repo | fla_vendor |
| tileops_owned_cp_generic_a | causal_ladder_row | controlled_full_op | true | true | pass | 2.7674 | false | FLA version unverified | mixed_experiment_roots | fla_vendor |
| tileops_owned_cp_blocked_inverse_a | causal_ladder_row | controlled_full_op | true | true | pass | 0.715062 | false | FLA version unverified | production_root_experiment_adapter | fla_vendor |
| tileops_final_dispatch | final_candidate | controlled_full_op | false | true | pass | 0.692026 | false | FLA version unverified | production_root | fla_vendor |

## Controlled Causal Ladder Rows

| variant | role | lane | causal | formal accepted | correct | latency_ms | ref verified | caveat | used root | ref root |
|---|---|---|---|---|---|---:|---|---|---|---|
| generic_a_legacy | causal_ladder_row | controlled_full_op | true | true | pass | 11.1906 | false | FLA version unverified | tileops_repo | fla_vendor |
| tileops_owned_cp_generic_a | causal_ladder_row | controlled_full_op | true | true | pass | 2.7674 | false | FLA version unverified | mixed_experiment_roots | fla_vendor |
| tileops_owned_cp_blocked_inverse_a | causal_ladder_row | controlled_full_op | true | true | pass | 0.715062 | false | FLA version unverified | production_root_experiment_adapter | fla_vendor |

## Final Candidate Rows

| variant | role | lane | causal | formal accepted | correct | latency_ms | ref verified | caveat | used root | ref root |
|---|---|---|---|---|---|---:|---|---|---|---|
| tileops_final_dispatch | final_candidate | controlled_full_op | false | true | pass | 0.692026 | false | FLA version unverified | production_root | fla_vendor |

## External Anchor Rows

| variant | role | lane | causal | formal accepted | correct | latency_ms | ref verified | caveat | used root | ref root |
|---|---|---|---|---|---|---:|---|---|---|---|
| ref_fla_051 | correctness_oracle_and_fla_baseline | external_anchor | false | true | pass | 8.02574 | false | FLA version unverified | fla_vendor | fla_vendor |

## ABI / A-Producer Equivalence

| variant | abi status | A comparison | A allclose | A max_abs | A max_rel | note |
|---|---|---|---|---:|---:|---|
| tileops_owned_cp_generic_a | collected | materialized canonical logical A | false | 0.117279 | 20583.9 | current generic fused_prepare_compute_w_u_tl also computes unused w/u; latency is a conservative V5 full-op row, not an A-only microbenchmark |
| tileops_owned_cp_blocked_inverse_a | collected | materialized canonical logical A | false | 0.117279 | 29546.4 | explicit V6 adapter calls blocksolve A once and then the same CP downstream as V5 |

## Warnings

These warnings are also summarized in the main tables' caveat column.

- ref_fla_051: FLA 0.5.1 package version is not verified; use recorded reference source identity
- generic_a_legacy: FLA 0.5.1 package version is not verified; use recorded reference source identity
- tileops_owned_cp_generic_a: FLA 0.5.1 package version is not verified; use recorded reference source identity
- tileops_owned_cp_blocked_inverse_a: FLA 0.5.1 package version is not verified; use recorded reference source identity
- tileops_final_dispatch: FLA 0.5.1 package version is not verified; use recorded reference source identity

## Code Source Audit

| variant | used module | root match | reference module |
|---|---|---|---|
| ref_fla_051 | /home/ga/TileOPs/.github/runner/vendor/flash-linear-attention/fla/ops/gated_delta_rule/chunk.py | N/A | /home/ga/TileOPs/.github/runner/vendor/flash-linear-attention/fla/ops/gated_delta_rule/chunk.py |
| generic_a_legacy | /home/ga/TileOPs/tileops/ops/gated_deltanet.py | true | /home/ga/TileOPs/.github/runner/vendor/flash-linear-attention/fla/ops/gated_delta_rule/chunk.py |
| tileops_owned_cp_generic_a | /home/ga/TileOPs/experiments/gated_deltanet_prefill_blog_ladder/variants.py | true | /home/ga/TileOPs/.github/runner/vendor/flash-linear-attention/fla/ops/gated_delta_rule/chunk.py |
| tileops_owned_cp_blocked_inverse_a | /home/ga/TileOPs/experiments/gated_deltanet_prefill_blog_ladder/variants.py | true | /home/ga/TileOPs/.github/runner/vendor/flash-linear-attention/fla/ops/gated_delta_rule/chunk.py |
| tileops_final_dispatch | /home/ga/TileOPs-pr1596/tileops/ops/gated_deltanet.py | true | /home/ga/TileOPs/.github/runner/vendor/flash-linear-attention/fla/ops/gated_delta_rule/chunk.py |
