# Claude Review Request: Blog Plan V3

We are revising the public article plan for our Gated DeltaNet prefill
optimization work in TileOps/TileLang.

The previous plan, `drafts/blog_plan_v2.md`, was written before the
FlashQLA-specialized phase was complete. The new plan, `drafts/blog_plan_v3.md`,
reorganizes the article around a larger story:

```text
make the operator measurable
-> use AKO for local TileLang kernel optimization
-> identify the boundary of local tuning
-> use human expert judgment and expert open-source kernels to change the
   search space
-> productionize the new schedule with correctness, benchmark, and dispatch
   gates
```

Please review whether this new plan is ready to drive a rewrite of
`drafts/tutorial_v2.md`.

## Materials To Review

Read in this order:

1. `drafts/blog_plan_v3.md`
   - The new proposed narrative plan.
2. `drafts/tutorial_v2.md`
   - Current working article draft. It mostly reflects the earlier v2 plan and
     does not yet fully incorporate the FlashQLA/CP-split story.
3. `drafts/blog_plan_v2.md`
   - Historical plan. Use it to identify what changed.
4. FlashQLA-specialized AKO evidence:
   - `/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/flashqla_specialized_ako_log.md`
   - Focus especially on:
     - Goal 3C / 3D migration and TMA findings;
     - Goal 4 CP-hybrid;
     - Goal 5 TileOps-owned CP split;
     - CP parameter sweep;
     - H64 follow-up;
     - Qwen four-line benchmark reproduction;
     - final attribution sections.
5. FlashQLA schedule notes:
   - `/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/flashqla_source_reading_20260622.md`
   - `/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/flashqla_scheduling_skeleton_20260623.md`
   - `/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/flashqla_pipeline_overlap_20260623.md`
6. Earlier local-AKO evidence:
   - `drafts/final_p1_evidence_20260620.md`
   - `/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/tilelang_recompute_lowering_decision_log.md`
   - `/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/latest_fla_prepare_decision_log.md`

## Current Intended Narrative

The public article should no longer be a chronological AKO diary. It should be
a tutorial about how kernel optimization progressed across layers:

1. Measurement and decomposition made the problem tunable.
2. Agentic local tuning improved fixed-contract kernels and revealed where
   local fusion was insufficient.
3. Human expert interventions changed the search space:
   - prepare/blocksolve-A as a blocked inverse / Neumann-style problem;
   - FlashQLA-inspired CP split as the production schedule for long-sequence
     replay.
4. Qwen's FlashQLA project should be explicitly credited for the CP-split GDN
   prefill schedule used as the main scheduling reference in the second half of
   the work.
5. TileOps' contribution should be described as:
   - understanding and validating the FlashQLA schedule;
   - porting it into TileOps-owned kernels;
   - combining it with TileOps' faster A producer;
   - tuning CP split and dispatch;
   - validating correctness/performance against FLA and FlashQLA.

## Critical Review Focus

Please focus on narrative correctness, not copyediting.

### 1. Human Expert Contributions

The plan places two human expert contributions near the end as the climax:

1. blocked inverse / Neumann-style prepare framing;
2. FlashQLA-inspired CP split plus prefix/butterfly study.

Please judge:

- Is this placement convincing?
- Does it make the article stronger than the old round-by-round layout?
- Does it fairly distinguish human insight from agentic implementation search?

### 2. FlashQLA / Qwen Attribution

The plan explicitly credits Qwen's FlashQLA project for the production-grade
CP-split schedule:

```text
compute corrected segment initial states, then run fused replay/output over
shorter segments with a producer/consumer pipeline
```

Please judge:

- Is the attribution strong enough?
- Is it too strong anywhere, making our work sound like a thin wrapper?
- Does the plan clearly identify our own contributions after adopting the
  schedule?

### 3. Agent Capability Boundary

The intended boundary is:

```text
Agents refine measurable implementation spaces. Human experts and expert
references reshape the search space. Once a new space exists, agents can tune
inside it.
```

Please judge whether the plan:

- underclaims the agent's actual usefulness;
- overclaims autonomous algorithm invention;
- gives enough examples of agentic work beyond parameter sweeps.

### 4. Evidence And Claims

Several benchmark sets exist across different dates. The plan says final public
claims must use refreshed PR-head data and should mark old Goal 4/5 numbers as
historical unless reproduced.

Please judge:

- Which old numbers are safe to use as historical evidence?
- Which numbers must be refreshed before public publication?
- What final table structure would avoid misleading comparisons between FLA,
  FlashQLA TL0.1.8, and TileOps TL0.1.9?

## Review Questions

Please answer these directly:

1. Does `drafts/blog_plan_v3.md` provide a stronger public narrative than v2?
2. Is the three-act structure clear?
3. Should the two human expert contributions both be placed in Act III, or
   should blocked inverse/Neumann appear earlier?
4. Is the FlashQLA credit wording fair and sufficiently explicit?
5. Does the plan avoid saying or implying that we independently invented
   FlashQLA's CP split?
6. Does it avoid underclaiming the TileOps work after adopting the schedule?
7. Are the planned sections enough to explain why direct fusion failed?
8. Are the planned sections enough to explain why butterfly/prefix scan was
   studied but not productionized?
9. Which parts of `tutorial_v2.md` should be preserved as-is?
10. Which parts of `tutorial_v2.md` should be moved or rewritten?
11. What diagrams are essential before a public draft?
12. What evidence is blocking?
13. What evidence is nice-to-have?
14. What is the biggest overclaiming risk?
15. What is the biggest underclaiming risk?

## Desired Output Format

Please organize your review as:

1. High-level verdict
2. Major narrative issues
3. Attribution and credit risks
4. Agent capability boundary risks
5. Benchmark/evidence risks
6. Suggested outline changes
7. Concrete edits to make in `drafts/blog_plan_v3.md`
8. Concrete rewrite guidance for `drafts/tutorial_v2.md`

Be specific. Quote or paraphrase any claim that should change and suggest
replacement wording.

