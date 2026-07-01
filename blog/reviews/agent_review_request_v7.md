# Review Request V7: GDN Prefill AI-Assisted Blog Rewrite Plan

This document is a self-contained review prompt for another agent/reviewer.
Please review the current blog rewrite plan, not the TileOps PR itself.

## Source Documents To Read

Primary:

- `/home/ga/2026-06-15/gdn-prefill-ai-assisted-blog/drafts/blog_plan_v3.md`
- `/home/ga/2026-06-15/gdn-prefill-ai-assisted-blog/drafts/kernel_code_shapes_20260629.md`
- `/home/ga/2026-06-15/gdn-prefill-ai-assisted-blog/drafts/tutorial_v2.md`

Important supporting notes:

- `/home/ga/2026-06-15/gdn-prefill-ai-assisted-blog/drafts/h_replay_attribution_note_20260625.md`
- `/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/flashqla_specialized_ako_log.md`
- `/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/flashqla_specialization_plan_20260622.md`

## Current Thesis

The article should not be framed as "an AI invented a faster kernel."

The current thesis is:

```text
Agents are effective when the kernel problem is made measurable: they can read
papers/reference implementations, reconstruct equations, build tests, inspect
lowering, and tune local TileLang implementation choices inside a fixed
semantic contract.

But the largest search-space changes still required external input: human
mathematical insight for the blocked inverse / Neumann-style prepare producer,
and Qwen FlashQLA as the production-grade CP-split schedule reference for
long-sequence replay.
```

Short version:

```text
Agents can reconstruct and refine a measurable search space; experts and expert
kernels reshape it.
```

## Desired Article Structure

Use a three-level capability structure, not a chronological AKO round diary.

### Level 1: Correct, Measurable Operator

Explain how the agent helps turn Gated DeltaNet prefill from paper/reference
code into a correct TileLang operator:

- GDN as gated delta-memory;
- decode residual write intuition;
- chunkwise prefill: `prepare -> replay -> output`;
- `w/u` as effective chunk writes, not raw `beta*k` and `beta*v`;
- correctness/reference checks;
- CUPTI/component benchmark infrastructure;
- lowering inspection and logs.

### Level 2: Autonomous Local AKO Inside A Fixed Contract

Show what the agent could optimize without changing the mathematical search
space:

- h recurrence scale placement;
- recompute store-path diagnosis;
- local memory/copy/store-path choices;
- small fusion attempts;
- direct fused skeletons that removed some intermediates but did not shorten
  recurrence.

Boundary: do not put Neumann/blocksolve here. That was not discovered by
autonomous local AKO; it was introduced by human mathematical insight.

### Level 3: External Input Changes The Search Space

This is the climax. It has two major search-space expansions:

1. Human Neumann/blocksolve prepare.
2. Qwen FlashQLA CP-split replay schedule.

Also include hierarchical prefix/butterfly as a negative result: explored,
validated as an affine idea, but rejected for the current production path due
to representation and pipeline overhead.

## Attribution Rules

Must credit Qwen FlashQLA clearly:

```text
We credit Qwen's FlashQLA project (https://github.com/QwenLM/FlashQLA,
commit 6ef4858b) for the CP-split GDN prefill schedule that became the
production reference for the second half of this work. Our TileOps
`fused_fwd.py` kernel structure directly follows FlashQLA's
`hopper/fused_fwd.py` producer/consumer skeleton, adapted for the current
TileLang 0.1.11 lowering line and TileOps dispatch.
```

Important wording:

```text
FlashQLA showed the production-grade way to attack that long-replay bottleneck:
compute corrected segment initial states, then run fused replay/output over
shorter segments with a carefully partitioned producer/consumer pipeline.
```

Avoid:

- claiming we independently invented FlashQLA's CP-split schedule;
- claiming TileOps has a superior replay algorithm unless same-lowering,
  same-call-path evidence is shown;
- implying the work was only copying FlashQLA.

TileOps contributions to preserve:

- TileOps-owned BTHD production path;
- faster specialized A producer;
- CP-split port into TileOps-owned kernels;
- CP parameter tuning and dispatch;
- H64 dispatch correction;
- validation against FLA;
- production benchmark integration.

## Neumann / Blocksolve Section Requirements

This part should include more math than the purely engineering sections,
because it is the main original algorithmic reframing in this project.

Suggested formula ladder:

1. Effective writes:

```text
W = A (beta K)
U = A (beta V)
```

2. Treat chunk-local correction as a causal triangular inverse problem:

```text
A = (I + L)^(-1)
```

where `L` is a strictly lower-triangular / causal interaction term induced by
chunk-local gated key interactions.

3. Neumann-style expansion:

```text
(I + L)^(-1) = I - L + L^2 - L^3 + ...
```

For fixed chunk size, the strictly lower-triangular structure makes this finite
in principle. The engineering win is not "use approximation instead of exact
math"; it is changing the producer's computational shape from a generic
triangular solve view to a blocked causal inverse/update view that TileOps can
specialize.

Formula caution: this is the explanatory ladder, not final publication-ready
notation. Before approving final formulas, verify sign convention, exponent
placement, beta/gate placement, and matrix orientation against the TileOps
implementation. If exact notation is too noisy, use a schematic formula and
say so explicitly.

Suggested figure:

- left: dense-looking lower-triangular solve;
- right: block-lower-triangular 4x16 decomposition with local Neumann-style
  updates and block reuse.

Review concern: does this section explain the math enough without turning into
a proof or overclaiming novelty?

## Figure Plan

The rewrite should use figures as primary explanation, not decoration.

Priority order:

1. Direct Fused vs CP-Split Schedule.
   - Left: `h0 -> chunk0 -> chunk1 -> ... -> chunkN`, one long replay chain.
   - Right: explicitly show `prepare_h/correct_h0` producing corrected segment
     starts, then shorter replay/output chains. Do not make segments look
     naturally independent without the correction step.
   - Takeaway: fusion alone does not shorten recurrence; CP split does.

2. Blocksolve / Neumann Prepare.
   - Left: dense-looking lower-triangular solve.
   - Right: 4x16 block-lower structure with local inverse/Neumann-style
     updates and block reuse.
   - Takeaway: this is the human-supplied mathematical search-space change.

3. Measurement Gate / AKO Loop.
   - candidate idea -> TileLang edit -> compile gate -> correctness gate ->
     CUPTI/component benchmark -> lowering inspection -> decision log ->
     accept/reject.
   - Takeaway: AKO is gated engineering search, not free-form prompting.

4. Dispatch / Metadata / Shape Guard.
   - shape metadata -> dispatch policy -> `max_local_chunks`, CP segments,
     `block_DV`, H64 special case -> selected kernel + benchmark metadata.
   - Takeaway: production dispatch is part of the kernel behavior.

5. Hierarchical Prefix Negative Result.
   - left: dependency depth improves from serial `N` to prefix-like `log N`;
   - right: representation becomes heavier: `H` vs `M + b` vs `[b | M]`.
   - Takeaway: prefix improves dependency parallelism, not arithmetic or
     pipeline efficiency in this production shape.

6. Migration / Lowering Mismatch.
   - source-level: FlashQLA skeleton ~= TileOps port;
   - lowering-level: `T.copy` path loses intended TMA behavior, explicit
     `T.tma_copy(..., barrier=...)` restores it.
   - Takeaway: source similarity is not performance equality.

Suggested visual language:

- use one color for causal/recurrent dependency;
- one color for parallel CP segments;
- one color for global memory materialization;
- one color for measurement/evidence gates;
- avoid full kernel listings in figures.

## Code Shape Plan

`kernel_code_shapes_20260629.md` now covers:

- correct operator dataflow: prepare -> replay -> output;
- decode residual view;
- chunkwise `w/u`;
- local AKO scale placement;
- recompute store path;
- blocked inverse / Neumann prepare;
- FlashQLA CP-split schedule;
- direct fused without CP vs CP-split contrast;
- FlashQLA producer/consumer fused forward skeleton;
- hierarchical prefix negative result;
- measurement gate / AKO loop;
- dispatch metadata and shape guards;
- migration lesson: source copy vs TMA-restored path.

Please check whether these are enough for the rewrite, and whether any code
shape is misleading, too concrete, or missing an important caveat.

Known sensitive code-shape checks:

- decode residual snippet must be marked schematic unless exact exponent and
  orientation are verified;
- FLA should remain the main behavioral correctness reference; FlashQLA is
  primarily a schedule/source/performance reference unless explicitly stated;
- TMA/WGMMA snippets are TileLang-shaped pseudocode unless exact APIs and
  generated code are cited;
- generated CUDA/PTX should not be shown in the main text unless there is a
  narrow reason and an archived source for the claim.

## Benchmark And Evidence Policy

Do not call current tables final unless they are refreshed at latest PR head.

Use this evidence hierarchy:

- Tier 1: final claims require latest PR-head correctness and benchmark rows.
- Tier 2: historical trajectory may use dated rows, clearly labeled.
- Tier 3: FlashQLA and generated-code claims require caveats unless controlled
  evidence is shown.

For FlashQLA comparison rows, the article must include:

- `max_local_chunks`;
- CP segment count;
- `block_DV`;
- TileLang version;
- GPU model and GPU id;
- warmup, repeat, and trials;
- CUPTI vs wall-clock timer configuration and any L2-flush policy;
- TileOps commit and FlashQLA commit/environment;
- dtype, layout, and seed for public-facing rows.

Table B caveat must remain:

```text
TileOps vs FlashQLA is a public-environment comparison, not a controlled
same-lowering replay attribution experiment.
```

Speedup language guard:

```text
Use "faster" or "significantly faster" only for public-environment full-op
comparisons under stated shapes/timers/versions. Do not use that wording as
replay algorithm attribution unless CP split, call path, lowering, dispatch,
and timer are controlled.
```

## Rewrite Map For `tutorial_v2.md`

Mostly reusable:

- Sections 0-4: terminology, motivation, formulas, measurement loop.
- Section 5: scale placement local AKO.
- Section 6: recompute/store-path diagnostic.
- Section 7: blocksolve-A technical content, but relocate to Level 3.

Rewrite from scratch:

- introduction and thesis;
- capability summary;
- human algorithmic moves;
- production integration;
- benchmark section;
- negative results section.

Do not move Section 9 as-is. The scan/prefix formulation should appear only as
an explored-and-rejected branch, not as the source of the final production
algorithm. The final production schedule should be credited to Qwen FlashQLA's
CP-split schedule, followed by TileOps-owned port, A producer, CP tuning,
dispatch, validation, and productionization.

## Review Questions

Please focus on these questions:

1. Is the three-level capability structure clear and publishable?
2. Does the plan fairly credit Qwen FlashQLA while preserving TileOps'
   engineering contributions?
3. Is the Neumann/blocksolve section mathematically concrete enough, and does
   it avoid overclaiming?
4. Are the proposed figures sufficient and well placed?
5. Are any figures missing for important strategy changes?
6. Are benchmark claims and speedup language sufficiently scoped?
7. Is the hierarchical prefix result framed correctly as explored-and-rejected,
   not "prefix is impossible"?
8. Are the code shapes suitable for a blog, or are any too implementation-like?
9. What should be added before rewriting `tutorial_v2.md`?
10. What should be explicitly deleted or demoted from `tutorial_v2.md`?

## Desired Output From Reviewer

Please return:

1. High-level verdict: ready to rewrite / needs one more planning pass.
2. Top risks, ordered by severity.
3. Specific edits to `blog_plan_v3.md` or `kernel_code_shapes_20260629.md`.
4. Suggested figure changes.
5. Suggested formula/math changes for the Neumann section.
6. Any warning about attribution, benchmark, or novelty claims.
