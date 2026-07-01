# Claude Review Request: Blog Plan V3 Finalization Pass

We revised `drafts/blog_plan_v3.md` again after PR validation and the second
review. Please review whether the plan is now ready to drive a rewrite of
`drafts/tutorial_v2.md`.

## Materials To Review

Read in this order:

1. `drafts/blog_plan_v3.md`
   - This is the current narrative plan.
2. `drafts/tutorial_v2.md`
   - Current article draft. It still follows the older v2 narrative and has
     not yet been reorganized into the three-act v3 structure.
3. `drafts/h_replay_attribution_note_20260625.md`
   - Working note on whether TileOps' h/replay back half is faster than
     FlashQLA, and what can/cannot be claimed.
4. Prior review output, if available:
   - `/home/ga/.codex/attachments/054ae21e-06ab-4f1b-9646-77d6ce2464b8/pasted-text.txt`
5. Supporting evidence:
   - `/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/flashqla_specialized_ako_log.md`
   - `/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/flashqla_source_reading_20260622.md`
   - `/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/flashqla_scheduling_skeleton_20260623.md`
   - `/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/flashqla_pipeline_overlap_20260623.md`
   - `drafts/final_p1_evidence_20260620.md`

## Latest Changes Since V4

- Updated final PR-facing TileLang wording from `0.1.9` to `0.1.11`.
- Preserved historical `TL0.1.8 -> TL0.1.9` wording only where it describes
  the actual migration/debugging timeline.
- Added an alias-fix regression note:
  - CI-only scalar-alias fix commit: `7fb5f35d`.
  - GPU4/CUPTI four-row rerun showed no meaningful full-op performance
    regression.
- Clarified that the component breakdown table was recorded on `a329a6c1`,
  while the later alias fix was verified separately for full-op latency.
- Confirmed PR #1596 GPU smoke passed after the alias fix.

## Questions

Please answer directly:

1. Is the TileLang version wording now internally consistent?
2. Does the plan clearly distinguish historical TL0.1.9 migration work from
   final PR TL0.1.11 claims?
3. Is the alias-fix evidence treatment fair, or should the component breakdown
   be rerun before using Table D in the article?
4. Is the FlashQLA attribution still explicit enough after the latest edits?
5. Does the plan avoid implying that TileOps independently invented CP split?
6. Does the plan avoid underclaiming TileOps' productionization work?
7. Are the benchmark tables and component breakdown safe enough as planning
   evidence, assuming a final refresh before publication?
8. Is the h/replay attribution note cautious enough? In particular, does it
   avoid overclaiming a new replay algorithm while still acknowledging the
   measured TileOps fused replay event advantage?
9. What exact changes should be made before starting the `tutorial_v2.md`
   rewrite?
10. Which current `tutorial_v2.md` sections can be moved mostly intact?
11. Which sections should be rewritten from scratch?

## Desired Output Format

Use this structure:

1. Verdict: ready / almost ready / needs another plan pass
2. Remaining evidence risks
3. Remaining attribution risks
4. TileLang version wording issues
5. Rewrite map for `tutorial_v2.md`
6. Concrete edits still needed in `drafts/blog_plan_v3.md`
