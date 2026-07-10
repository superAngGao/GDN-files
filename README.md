# GDN Files

Working documents for the Gated DeltaNet prefill article and supporting
evidence. This repository is meant for rendered reading and shared review, not
as a replacement for the TileOps source tree.

## Recommended Reading Order

0. `presentations/cp-split-neumann-zh.md`
   - Chinese internal talk outline for explaining CP split and
     blocked-inverse / Neumann prepare before publication.
1. `algorithm/gated_deltanet_algorithm_talk_zh.md`
   - Chinese algorithm note for Gated DeltaNet intuition, recurrence, erase
     mechanism, gates, and chunkwise computation.
2. `blog/drafts/tutorial_v3.md`
   - Current main article draft.
3. `evidence/ladder/summaries/blog_ladder_evidence_64k_h16.md`
   - Writing-facing `64K/H16` evidence package.
4. `evidence/ladder/summaries/a_replay_cross_ablation_64k_h16.md`
   - A/replay cross-ablation note explaining the FlashQLA attribution split.
5. `blog/drafts/gdn_prefill_blog_experiment_plan_20260630.md`
   - Experiment plan that defines the controlled rows and evidence lanes.
6. `blog/drafts/kernel_code_shapes_20260629.md`
   - Code-shape snippets for the writing agent and reviewer.

## Directory Map

- `algorithm/`
  - Standalone algorithm explanation drafts.
- `blog/drafts/`
  - Main tutorial drafts, plans, and supporting notes.
- `blog/reviews/`
  - Review requests, handoff notes, and external-review material.
- `blog/data/`
  - Historical log material.
- `presentations/`
  - Rendered talk outlines and internal presentation material.
- `evidence/ladder/docs/`
  - Ladder harness README and variant inventory.
- `evidence/ladder/summaries/`
  - Markdown summaries intended for writing and review.
- `evidence/ladder/results/`
  - Small JSONL result files. Large tensor artifacts are intentionally not
    included.

## Current Evidence Caveats

- The formal evidence currently centers on `64K/H16`.
- Broader-shape tables can be refreshed later.
- FLA should be described as a recorded vendored FLA reference unless the
  package identity is independently verified.
- FlashQLA comparisons are public-environment anchors, not same-lowering
  replay attribution by themselves.
- Neumann/blocksolve formulas should keep the implementation caveat: TileOps
  uses a blocked-inverse / Neumann-style producer, and the materialized `A` is
  not claimed to equal the generic exact/KKT-style producer.
- Low-level TMA/WGMMA/PTX/SASS behavior should only be claimed when generated
  code evidence is archived.
