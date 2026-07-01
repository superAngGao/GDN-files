# Claude Review Request: Blog Plan V3 After First Review

We revised `drafts/blog_plan_v3.md` after a first Claude review. Please review
the updated plan again, focusing on whether the previous high-risk issues were
actually resolved.

## Materials To Review

Read in this order:

1. `drafts/blog_plan_v3.md`
   - This is the revised plan after incorporating the prior review.
2. `drafts/tutorial_v2.md`
   - Current article draft. It has not yet been reorganized to match the new
     plan.
3. Prior review output, if available:
   - The first review flagged three major risks:
     - conflating original human algorithmic insight with FlashQLA reference
       adoption;
     - underspecifying final-vs-historical benchmark evidence;
     - compressing the local AKO story so much that the tuning wall was not
       clear.
4. Supporting evidence:
   - `/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/flashqla_specialized_ako_log.md`
   - `/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/flashqla_source_reading_20260622.md`
   - `/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/flashqla_scheduling_skeleton_20260623.md`
   - `/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/flashqla_pipeline_overlap_20260623.md`
   - `drafts/final_p1_evidence_20260620.md`

## What Changed In This Revision

The revised plan now:

- Adds a development timeline separating local AKO/Neumann prepare from the
  later FlashQLA study and migration phase.
- Strengthens code-level attribution to Qwen FlashQLA, including the CP-split
  schedule and `hopper/fused_fwd.py` producer/consumer skeleton.
- Separates contribution types:
  - original algorithmic insight: blocked inverse/Neumann prepare;
  - expert reference adoption: FlashQLA CP split;
  - engineering contribution: migration, TMA restoration, TileOps-owned port,
    dispatch tuning, validation.
- Reframes Act II as discovering the boundary of local tuning, not merely
  reporting modest wins.
- Reframes Act III as "Expert-Driven Search Space Changes."
- Adds a bridge explaining why FlashQLA was needed after prepare improved.
- Splits final benchmark presentation into:
  - TileOps TL0.1.9 vs FLA 0.5.1;
  - TileOps TL0.1.9 vs FlashQLA TL0.1.8 with compiler/lowering context.
- Adds a three-tier evidence policy:
  - final PR-head claims;
  - dated historical trajectory;
  - claims that require qualification.
- Adds a current PR-head benchmark snapshot from 2026-06-25.

## Questions

Please answer directly:

1. Did the revision sufficiently distinguish original human insight from
   FlashQLA reference adoption?
2. Is the FlashQLA attribution now explicit enough at both algorithm and code
   structure levels?
3. Does the plan still risk implying that TileOps independently invented the
   CP-split schedule?
4. Does it still risk underclaiming the TileOps engineering work after adopting
   the schedule?
5. Is Act II now strong enough as a story about discovering the local tuning
   wall?
6. Is the Bridge section enough to explain why prepare improvements did not
   solve long sequences?
7. Is the migration section substantial enough, especially the TL0.1.8 to
   TL0.1.9 TMA restoration story?
8. Are the two benchmark tables fair, given that FLA 0.5.1 and FlashQLA
   TL0.1.8 are different comparison types?
9. Is the evidence tier system clear enough to prevent stale numbers from
   becoming final claims?
10. What should be changed before using this plan to rewrite `tutorial_v2.md`?

## Desired Output Format

Use this structure:

1. Verdict: ready / almost ready / needs another plan pass
2. Remaining overclaim risks
3. Remaining underclaim risks
4. Benchmark and evidence risks
5. Rewrite guidance for `tutorial_v2.md`
6. Concrete edits still needed in `drafts/blog_plan_v3.md`
