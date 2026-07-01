# Claude Review Request: Blog Plan V2

We are revising the public article plan for our Gated DeltaNet prefill
optimization work in TileOps/TileLang.

The previous plan was too close to an experiment report. The new goal is a
community-facing TileLang tuning tutorial in the spirit of practical GEMM
optimization writeups: teach the method, then use GDN prefill as the detailed
case study.

This request has been updated after the first review pass. Please review the
revised plan and focus on whether the structural changes resolved the earlier
issues.

## Materials To Review

Read in this order:

1. `drafts/blog_plan_v2.md`
   - The new proposed positioning, article outline, claims policy, and
     experiment plan.
2. `planning.md`
   - The older planning document. Use it as historical context; do not assume
     its claims are still current.
3. `data/week1_experiment_log.md`
   - Earlier benchmark/profiler/ablation evidence.
4. Final AKO decision log:
   - `/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/tilelang_recompute_lowering_decision_log.md`
   - Focus on the recompute and blocksolve-A sections near the end.
5. Optional:
   - `drafts/tutorial.md`, the old article skeleton.

## Current Technical Status

The final PR worktree is:

```text
/home/ga/TileOPs-pr1596
```

The relevant production file is:

```text
/home/ga/TileOPs-pr1596/tileops/kernels/gated_deltanet/gated_deltanet_prefill.py
```

Current status:

- the discussed production prefill file has no `triton` or `tl.` references;
- BTHD blocksolve-A is now TileLang;
- tests pass:

```text
pytest tests/ops/test_gated_deltanet_prefill.py -q
Result: 8 passed
```

Final isolated blocksolve-A result, S32K/H16:

| Backend | Latency |
| --- | ---: |
| Triton local baseline | 0.14692616 ms |
| TileLang production | 0.11384284 ms |

Final docker BTHD full-op benchmark, `gdn-kernel-bench:nightly-tl019`,
`B=1`, `H=16`, `DK=DV=128`, `chunk64`, `fp16`,
`output_final_state=True`:

| Seq len | TileOps TileLang | FLA 0.5.1 | FLA / TileOps |
| --- | ---: | ---: | ---: |
| 32K | 2.38432512 ms | 2.42574500 ms | 1.01737x |
| 64K | 4.64552674 ms | 5.27495597 ms | 1.13549x |
| 128K | 9.43092621 ms | 10.88170333 ms | 1.15383x |

## Desired Narrative Shift

The old plan framed the article partly as:

```text
large local wins, close to latest FLA, not yet ahead
```

That is now outdated for the validated H16 BTHD rows.

The new plan should frame the article as:

```text
a practical TileLang tuning methodology: first build the measurement/gate
infrastructure, then use agents to optimize inside that measurable loop. Agents
can organize reference implementations and human-provided paper context into a
correctness-first baseline and test scaffold; during tuning they can perform
expert-kernel alignment, hardware-aware pipeline reasoning, TileLang lowering
search, and local algebraic rewrites. Human engineers introduce larger
algorithmic structures such as Neumann-style prepare framing and associative
scan recurrence; agents can then tune inside those human-proposed algorithm
spaces.
```

Important separation:

1. Agentic tuning system:
   - the scope, reference baseline, correctness gate, and latency gate are part
     of the tuning loop, not a separate narrative layer;
   - before optimization, agents can help turn reference code and
     human-provided paper context into runnable baselines, initial fusion
     hypotheses, and test scaffolds;
   - agents can tune at several levels: parameters, TileLang expressions,
     hardware-aware pipelines, and local algebraic rewrites.
2. Human algorithmic innovation:
   - changes the search space itself;
   - examples: Neumann-style prepare/inverse framing and associative prefix
     scan for chunk recurrence.
3. Hybrid bridge:
   - after humans define an algorithmic search space, agents can tune within
     it;
   - example: Neumann round reduction should be presented this way, not as
     independent agent algorithm invention.

## Critical Review Focus: Agent Capability Boundary

Please pay special attention to whether the plan captures the agent capability
boundary precisely.

The article should not portray agents as only doing parameter sweeps, because
that underclaims what happened. In this project, the agentic loop also helped:

- organize reference implementations with human-provided paper context;
- organize the algorithm into prepare / recurrence / output stages;
- write correctness-first naive implementations;
- propose first fusion boundaries from dataflow, with human review;
- build test scaffolds and benchmark harnesses;
- study expert Triton/FLA pipelines;
- reason about hardware-aware implementation structure;
- inspect generated/lowered code;
- search TileLang expressions for copy, wait, layout, shared-memory, and store
  behavior;
- perform local algebraic rewrites inside a component contract;
- tune inside a human-proposed algorithm space, such as reducing Neumann rounds
  after the Neumann-style prepare framing is already chosen.

But the article also should not imply that agents independently invented the
global algorithmic structure. Human algorithmic contributions include:

- framing prepare as a blocked inverse / Neumann-style problem;
- recognizing associative prefix scan structure in the chunk recurrence;
- deciding which algorithm family is worth tuning and where production
  dispatch should be guarded.

Please review whether this boundary is clear, useful, and defensible:

```text
Agents can help organize reference implementations and human-provided paper
context into a correctness-first operator path, propose initial fusion
boundaries from dataflow, and then align and refine implementations inside a
measurable component contract, including hardware pipeline alignment and
fixed-semantics local rewrites. Humans introduce new global algorithmic
structures and decide which algorithmic search space is worth exploring. Once
that space exists, agents can tune within it, but the article should label that
as a hybrid case.
```

## Review Questions

Please answer broadly, but prioritize these:

1. Does `drafts/blog_plan_v2.md` now read like a community tuning tutorial
   rather than an internal experiment report?
2. Does the revised ordering work: decompose GDN, build tuning infrastructure,
   present agent taxonomy, then demonstrate with AKO rounds?
3. Is Level 0 correctly scoped as pre-tuning scaffolding rather than an
   over-elevated standalone capability?
4. Is the agent tuning taxonomy useful and technically precise?
5. Is the separation between agentic tuning inside a measurable component
   contract and human algorithmic innovation clear?
6. Does the plan describe first-fusion ability accurately: strong enough to
   show agents can propose a fusion boundary from dataflow, but clear that
   humans review it and optimized fusion requires profiling?
7. Does the plan draw the right boundary for hardware pipeline alignment:
   strong enough to show the agent can reason about expert kernel pipelines,
   but not so strong that it implies arbitrary Triton/CUDA pipelines are
   automatically expressible in TileLang?
8. Does the plan draw the right boundary for local algorithm adjustment:
   strong enough to include scale placement and gate precompute, while treating
   Neumann round reduction as tuning inside a human-proposed Neumann/inverse
   algorithm space?
9. Are the three agentic teaching nodes the right ones?
   - move the scale, remove the buffer
   - find the real bottleneck, not the fashionable primitive
   - align with the expert pipeline, respect the DSL boundary
10. Does the article overclaim the final results anywhere?
11. Does it underclaim the final result now that production is fully TileLang on
   the validated path?
12. Which experiments are actually blocking before writing the public draft?
13. Which proposed experiments are nice-to-have but should not delay drafting?
14. What diagrams or tables would most improve readability?
15. Which sections should be shortened or merged?

## Desired Output Format

Please organize your review as:

1. High-level verdict
2. Major narrative issues
3. Overclaiming or underclaiming risks
4. Suggested outline changes
5. Experiment plan changes
6. Style and readability comments
7. Concrete edits to make in `drafts/blog_plan_v2.md`

Please be specific. Quote or paraphrase any claim that should change and
suggest replacement wording.
