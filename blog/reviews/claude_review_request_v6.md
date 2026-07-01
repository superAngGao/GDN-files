# Claude Review Request: Blog Plan V3 Three-Level Narrative Pass

Please review whether the current blog plan is ready to drive a rewrite of
`drafts/tutorial_v2.md`.

## Materials To Read

Read these files:

1. `drafts/blog_plan_v3.md`
   - Current narrative plan.
   - It was revised from a "three-act" story into a "three-level capability"
     story.
2. `drafts/kernel_code_shapes_20260629.md`
   - Blog-facing code skeletons for the key algorithm/kernel nodes.
3. `drafts/h_replay_attribution_note_20260625.md`
   - Attribution and caution notes for FlashQLA, h/replay, CP split, and
     hierarchical-prefix negative results.
4. `drafts/tutorial_v2.md`
   - Current older draft. It has not yet been rewritten to match the new plan.

Optional supporting files if needed:

- `/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/flashqla_specialized_ako_log.md`
- `/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/flashqla_source_reading_20260622.md`
- `/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/flashqla_scheduling_skeleton_20260623.md`
- `/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/flashqla_pipeline_overlap_20260623.md`

## Most Recent Narrative Correction

The plan should not say:

```text
Act I: understand the operator
Act II: AKO local tuning including Neumann/blocksolve
Act III: FlashQLA / human input
```

That framing is wrong because:

- "understanding the operator" is not a standalone act. It is the first
  agent capability: reading papers/reference code and building a correct,
  measurable operator.
- blocked inverse / Neumann prepare must not appear before the external-input
  section. That search space was not autonomously discovered by the agent.
  It was a human mathematical reframing; AKO then implemented and tuned inside
  that externally supplied space.

The intended story is:

```text
1. Agent can read papers/reference code, understand the algorithm, and build a
   correct measurable TileLang operator.

2. Agent can autonomously search local implementation space inside a fixed
   semantic contract: memory movement, copy/store path, instruction/lowering
   choices, small fusion, shape parameters, etc.

3. Once local tuning hits a ceiling, external inputs reshape the search space:
   human mathematical insight (blocked inverse / Neumann prepare) and expert
   open-source schedules (Qwen FlashQLA CP-split fused replay).
```

The plan also now includes a hierarchical-prefix negative result:

```text
prefix scan improves dependency parallelism, but the tested full [b | M]
transition representation is too expensive for the production fused path.
```

## Questions

Please answer directly:

1. Does `blog_plan_v3.md` now reflect the intended three-level capability
   structure?
2. Does it correctly keep blocked inverse / Neumann prepare out of the
   autonomous-local-AKO section?
3. Does it fairly describe agent capability without overclaiming that the
   agent independently discovered the larger search spaces?
4. Is Qwen FlashQLA credited clearly enough for the CP-split schedule and
   fused replay/output skeleton?
5. Does the plan avoid underclaiming TileOps' work: A producer, port,
   validation, dispatch, benchmarking, and productionization?
6. Is the hierarchical-prefix negative result placed correctly and explained
   in a way a GPU/kernel reader will accept?
7. Is `kernel_code_shapes_20260629.md` the right level of code detail for the
   article, or should any snippet be added, removed, or rewritten?
8. Which parts of `tutorial_v2.md` can be moved mostly intact?
9. Which parts of `tutorial_v2.md` should be rewritten from scratch?
10. What exact edits should be made to `blog_plan_v3.md` before rewriting the
    tutorial?

## Desired Output Format

Use this structure:

1. Verdict: ready / almost ready / needs another plan pass
2. Narrative structure risks
3. Attribution risks
4. Code-shape / technical-explanation risks
5. Rewrite map for `tutorial_v2.md`
6. Exact edits recommended for `blog_plan_v3.md`
