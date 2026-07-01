# Blog Plan V2: Agentic TileLang Kernel Tuning For Gated DeltaNet Prefill

Draft status: planning document for review. This is not the public article yet.

## 1. Goal Of The Article

The article should teach a practical TileLang kernel tuning workflow, using
Gated DeltaNet prefill as the case study.

It should not read like a chronological experiment report. The reader should
come away with a reusable method:

1. decompose a nontrivial model operator into tunable kernel components;
2. define a fair reference implementation and measurement protocol;
3. use agents for high-throughput implementation-space search;
4. use human algorithmic insight to reshape the search space;
5. integrate only the variants that pass correctness, performance, and
   production-scope checks.

The technical result is important but secondary to the method:

```text
Starting from a mixed implementation, we used TileLang plus agentic kernel
optimization to reach a fully TileLang production GDN prefill path that beats
the fair FLA 0.5.1 baseline on the validated BTHD rows, including H16 and H32
cases.
```

## 2. Intended Audience

Primary audience:

- TileLang and TileOps developers.
- GPU kernel engineers interested in AI-assisted optimization workflows.
- ML systems engineers maintaining model-specific serving kernels.

Assumed background:

- understands basic GPU kernel concepts: shared memory, registers, occupancy,
  async copy, GEMM tiling;
- has seen Triton or CUDA kernels;
- does not need to know Gated DeltaNet internals before reading the article.

The article should explain enough GDN math to understand the optimization
targets, but it should not become a model paper summary.

## 3. Positioning And Core Thesis

Suggested title direction:

```text
Agentic TileLang Kernel Tuning: From Expert-Kernel Alignment To Production
Gated DeltaNet Prefill
```

Alternative shorter title:

```text
Tuning Gated DeltaNet Prefill In TileLang With Agents
```

Core thesis:

```text
Agents are currently most effective when the tuning problem is made measurable:
they can study expert kernels, reason about hardware-aware pipelines, explore
TileLang lowering choices, make local algebraic or algorithmic adjustments
inside a component contract, prune failed variants, and match or exceed a
reference implementation.

Human engineers still supply the larger algorithmic reframings that change the
search space itself: deciding that prepare should be viewed as a blocked
inverse problem, recognizing associative scan structure in the recurrence, and
choosing which algorithm family should be tuned next.
```

Short version:

```text
Agents explore and refine a measurable tuning space; humans introduce the new
algorithmic structures that create better tuning spaces.
```

This framing is deliberately narrower than "AI invented a faster kernel." It is
more credible and more useful for the community.

## 4. What Changed Since The Week 1 Plan

The previous planning document was written before the final TileLang migration
was complete. Its conservative message was:

```text
TileOps is close to the latest FLA baseline, but the public article should not
claim that it outperforms FLA.
```

That is now outdated for the validated target rows. The production worktree
now has a fully TileLang blocksolve-A path and no Triton components in
`tileops/kernels/gated_deltanet/gated_deltanet_prefill.py`.

Final accepted blocksolve-A AKO candidate:

```text
014: anchored gate precompute with numerically safe anchor-ratio scaling
018: one-round diagonal Neumann solve
```

Final isolated blocksolve-A comparison, S32K/H16:

| Backend | Latency |
| --- | ---: |
| Triton local baseline | 0.14692616 ms |
| TileLang production | 0.11384284 ms |

Targeted docker BTHD full-op benchmark on PR head `067edc7e`,
`gdn-kernel-bench:nightly-tl019`, `B=1`, `DK=DV=128`, `chunk64`, `fp16`,
`output_final_state=True`:

| Seq len | Heads | TileOps TileLang | FLA 0.5.1 | FLA / TileOps |
| --- | ---: | ---: | ---: | ---: |
| 32K | 16 | 2.39824722 ms | 2.44284368 ms | 1.01860x |
| 64K | 16 | 4.64622306 ms | 5.20679608 ms | 1.12065x |
| 128K | 16 | 9.35985710 ms | 10.87307050 ms | 1.16167x |
| 128K | 32 | 14.51952280 ms | 16.09163698 ms | 1.10828x |

Correctness and hygiene:

```text
pytest tests/ops/test_gated_deltanet_prefill.py -q
Result: 8 passed

rg -n "triton|tl\." tileops/kernels/gated_deltanet/gated_deltanet_prefill.py
Result: no matches
```

The article can now claim the fully TileLang production path beats the fair FLA
0.5.1 baseline on the validated BTHD rows above. It should still avoid implying
that the result generalizes to every shape.

## 5. Article Shape

The article should be a tuning tutorial, not an experiment diary.

The final public post should use experiments as evidence for transferable
lessons:

- how to define a tuning surface in TileLang;
- how agents can turn reference implementations and human-provided paper
  context into a runnable first fused operator baseline;
- how to build an agentic tuning loop with scope, metrics, and gates;
- how to compare against an expert reference kernel;
- how to distinguish local algorithm adjustments from new algorithmic
  reframings;
- how to turn local wins into production code.

Avoid a candidate-by-candidate chronological narrative except in compact
tables. The AKO logs should be cited as reproducibility artifacts, not copied
linearly into the article.

## 6. Proposed Outline

### 1. Why This Case Study

Purpose:

- Introduce Gated DeltaNet prefill as a realistic, nontrivial kernel tuning
  target.
- Explain why this is more instructive than a standalone GEMM benchmark:
  several kernels, recurrent structure, causal chunk-local work, production
  layout constraints, and a strong external FLA baseline.
- State the final outcome without overselling:
  a fully TileLang validated path that beats FLA 0.5.1 on target BTHD rows.

Key message:

```text
The interesting part is not only the final latency. It is the workflow that got
there: component profiling, agentic implementation search, human algorithmic
reframing, and production guards.
```

### 2. The TileLang Tuning Surface

Purpose:

- Explain why TileLang is a good platform for this workflow.
- Present TileLang as a middle ground between high-level tensor code and fully
  hand-written CUDA/Triton.

Drafting decision:

- Write this section after the computation graph and optimization rounds are
  drafted. The section should summarize what the case study proves about
  TileLang as a tuning surface, rather than becoming a generic TileLang API
  introduction.
- In the final article, every TileLang advantage introduced here should point
  to a later concrete moment: agent-editable implementation search, component
  benchmarks, lowering/source inspection, explicit memory/layout control,
  shape-specific dispatch, or an expressiveness boundary such as Triton-style
  fragment recirculation not mapping directly to stock TileLang.

Topics:

- explicit shared memory and fragments;
- `T.copy` and `T.async_copy`;
- `T.ptx_wait_group(0)` as the explicit wait before consuming async-copied
  shared tiles;
- layout annotations and swizzled shared memory;
- easy component kernels and shape-specific dispatch;
- Python-level experiment harnesses that agents can edit and benchmark.

Tone:

- Do not make this a TileLang API manual.
- Use only the features that matter for the GDN story.

### 3. Decompose GDN Prefill Before Tuning

Purpose:

- Give the reader a compact mental model of the operator.
- Define the components that later become tuning targets.

Inputs and outputs, BTHD:

```text
q, k: [B, T, H, K]
v:    [B, T, H, V]
g:    [B, T, H]
beta: [B, T, H]

o:           [B, T, H, V]
final_state: [B, H, K, V]
```

Pipeline:

1. chunk-local prepare / triangular system;
2. cross-chunk h recurrence;
3. output.

Important article choice:

- Keep formulas short.
- Use a figure for the three-stage pipeline.
- Make it clear which formulas are conceptual and which are implementation
  oriented.

### 4. Build The Agentic Tuning Infrastructure

Purpose:

- Teach readers that agentic search is only useful when the loop is measurable.
- Present the protocol, benchmark harness, and production gates as the
  infrastructure that makes automatic tuning possible.
- Establish the feedback system before describing what agents can do with it.

Loop structure:

```text
choose component and shape -> define correctness gate -> define latency gate ->
inspect reference pipeline -> edit TileLang candidate -> compile -> test ->
benchmark -> inspect lowering/source when needed -> accept/reject -> log
```

Measurement rules inside the loop:

- same BTHD layout as FLA/Qwen-style serving;
- same dtype and shape family;
- `output_final_state=True`;
- pinned or recorded GPU clocks when possible;
- component benchmarks separated from full-op benchmarks;
- correctness tolerance recorded next to each optimization;
- report validated shapes only.

Reference baseline:

- FLA 0.5.1 for full-op comparisons.
- Triton local baseline for component comparisons when replacing a Triton
  production helper.

Production gates:

- correctness test must pass;
- full prefill test file must pass before production acceptance;
- full-op benchmark must not regress;
- shape guard must match the validated domain;
- negative results stay in the log.

This section should make the first key methodological point:

```text
The scope, reference, correctness test, and stopping criteria are not separate
from agentic tuning. They are what make agentic tuning possible.
```

The infrastructure is not just final acceptance. It is the feedback system used
throughout AKO:

| Gate | Purpose | Example |
| --- | --- | --- |
| Compile gate | reject TileLang expressions that cannot lower | fragment RR blocksolve layout failures |
| Correctness gate | prevent locally fast but wrong candidates | unsafe gate product form rejected after full tests exposed NaNs |
| Component latency gate | make local search cheap and directional | isolated recompute and blocksolve-A benchmarks |
| Lowering/source gate | check whether intended hardware behavior appeared | cp.async, wait groups, swizzled shared store path |
| Full-op gate | ensure component wins survive integration | docker BTHD TileOps vs FLA benchmark |
| Scope gate | avoid unvalidated generalization | report only validated BTHD rows |

This infrastructure does double duty:

1. During optimization, it provides fast feedback for candidate evaluation:
   compile -> test -> benchmark -> accept/reject.
2. Before optimization, it makes pre-tuning work measurable: did the baseline
   implementation pass correctness, and can it be profiled?

The following section categorizes what agents can do at different depths when
this infrastructure exists.

### 5. Agent Tuning Levels: A Practical Taxonomy

Purpose:

- Summarize what agentic tuning can do at different depths.
- Give the reader a practical taxonomy they can reuse on other TileLang
  kernels.
- Avoid reducing agents to "parameter sweepers"; in this project they also
  helped reason about hardware pipelines and local transformations inside
  fixed component contracts.

Agent tuning levels shown by this project:

| Level | Philosophy name | What changes | Example |
| --- | --- | --- | --- |
| Level 0: pre-tuning scaffolding | make the problem measurable | organize reference code and human-provided paper context into an algorithm map, correctness-first baseline, tests, and initial fusion hypothesis with human review | GDN prepare / recurrence / output decomposition |
| Level 1: parameter/config search | search the cheap knobs first | thread count, warp policy, stages, tile sizes | blocksolve thread/policy probes |
| Level 2: TileLang expression search | express the same work differently | `T.copy` vs `T.async_copy`, explicit waits, shared store route, layout annotation | recompute lowering and swizzled shared output staging |
| Level 3: hardware-aware pipeline alignment | study first, then search | shared/register staging, cp.async/WGMMA overlap, output-store path, store pattern | recompute and blocksolve-A expert-pipeline alignment |
| Level 4: local algebraic rewrite | remove unnecessary data movement | mathematically equivalent changes inside one kernel contract | h recurrence scale-placement rewrite |
| Level 5: integration pruning | production decides what survives | check whether a local win survives full-op tests and shape guards | accepting 018 only after full tests and docker benchmark |

Tuning requirements:

| Level | What the agent needs | Boundary |
| --- | --- | --- |
| Level 0 | reference implementation, human-provided paper context, correctness oracle | agent can propose a first fusion boundary from dataflow; humans review decomposition and production viability; this is not performance-optimized fusion |
| Level 1 | stable component benchmark | cheap sweeps do not solve structural problems |
| Level 2 | source/lowering inspection plus latency gate | high-level intent does not guarantee the generated pipeline |
| Level 3 | expert reference kernel and generated-code checks | not every Triton/CUDA pipeline is expressible in TileLang |
| Level 4 | correctness oracle and tolerance | fixed-semantics rewrite, not a new global algorithm |
| Level 5 | full-op benchmark and scope guard | isolated microbenchmark wins are not enough |

Level 0 should be mentioned briefly in the article, not elevated into a major
standalone story unless we add strong evidence. The defensible claim is:

```text
Agents can help organize reference implementations and human-provided paper
context into a runnable baseline and test scaffold. They can propose initial
fusion boundaries from dataflow, but humans review correctness and production
viability, and performance-optimized fusion requires later profiling.
```

Important boundary:

```text
These are still optimizations inside a chosen component contract. They can
include local algebraic rewrites and hardware-aware pipeline changes, but they
are not the same as inventing a new global algorithmic decomposition.
```

Agent limitations shown by this project:

- it did not independently invent the high-level recurrence scan;
- it needed the problem scoped to a component and shape as part of the tuning
  loop;
- it needed a correctness/performance gate as part of the tuning loop;
- it could propose numerically risky ideas, such as unsafe gate factorization,
  that required human review and robust tests.

The agentic-tuning part of the article should elevate three representative
nodes, each with both a performance result and a tuning philosophy:

| Node | Performance role | Tuning philosophy | Boundary shown |
| --- | --- | --- | --- |
| Move the scale, remove the buffer | h recurrence improved by removing `k_scaled_s` | optimize memory traffic through local algebraic equivalence | agent can rewrite inside fixed semantics, not invent a new recurrence algorithm |
| Find the real bottleneck, not the fashionable primitive | recompute improved after store-path diagnosis | `async_copy` alone is not pipeline alignment; measure copy, wait, layout, and store behavior | agent can do hardware-aware attribution when component feedback is stable |
| Align with the expert pipeline, respect the DSL boundary | blocksolve-A became faster than Triton isolated | study Triton first, then search TileLang's expressible space | agent can match expert kernels when measurable, but cannot assume every Triton fragment pattern is expressible |

The following AKO sections demonstrate these tuning capabilities through
representative optimization rounds. In the current draft, recompute and
blocksolve-A should be split into separate rounds so each has a clear hardware
lesson and capability boundary. Each round combines a performance result with a
portable tuning philosophy. The public draft should present the rounds in the
order they were accepted, because we could not know ahead of time which layer
would tune successfully. It should still keep rejected candidate detail compact
and use tables instead of replaying the full AKO log linearly.

### 6. AKO Round 1: H-Recurrence Scale Placement

Purpose:

- First clean example of agentic implementation-space tuning.
- Show a local algebraic rewrite that improves memory behavior.

Original form:

```text
(k * exp(g_last - g_i))^T @ v_new
```

Rewritten form:

```text
k^T @ (v_new * exp(g_last - g_i))
```

Implementation effect:

- removes the `k_scaled_s` shared-memory buffer;
- reduces shared-memory traffic;
- keeps the mathematical result equivalent within tolerance.

Existing evidence:

| Variant | H-only latency | Speedup |
| --- | ---: | ---: |
| scale `k` into `k_scaled_s` | 2.2725 ms | 1.00x |
| scale `v_new` in place | 1.6277 ms | 1.40x |

Correctness:

```text
v_new_max_abs_vs_current = 0.0
final_state_max_abs_vs_current = 3.05e-5
```

Reader lesson:

```text
In a memory-sensitive kernel, the useful question is often not "how do we make
the GEMM faster?" but "why did we create this staging buffer at all?"  This is
the kind of fixed-semantics algebraic rewrite where an agent can be very
effective.
```

Needed before publication:

- rerun or confirm this ablation against the final production commit;
- optionally collect shared memory / register count before and after;
- generate a small before/after code snippet.

### 7. AKO Rounds 2-3: Recompute And Blocksolve-A

Purpose:

- Show agentic tuning at a larger implementation surface: replacing remaining
  expert-kernel components while preserving performance.
- Separate two lessons that should not be blurred: recompute is mainly a
  hardware pipeline diagnosis story, while blocksolve-A is mainly an
  expert-kernel alignment and DSL-boundary story.

Substory A: recompute `w/u` lowering

Key observations:

- simple TileLang expressions were correct but too slow;
- `T.async_copy` could emit cp.async, but async copy alone was not enough;
- direct fragment-to-global stores were a major bottleneck;
- the accepted TileLang expression routes output through swizzled shared memory
  before global store.

Important candidate ladder:

| Candidate idea | Result |
| --- | --- |
| simple `T.copy`/`T.gemm` variants | correct, about 0.42-0.47 ms |
| explicit `T.async_copy` ping-pong | emitted cp.async but still slow |
| no-store diagnostic | showed store path dominated |
| shared-copy output staging | major improvement |
| swizzled shared output staging | accepted isolated candidate, about 0.275 ms |

Article lesson:

```text
Matching a reference kernel is not about adding the fashionable primitive.  In
this case, `T.async_copy` and cp.async were visible, but the breakthrough came
from diagnosing the generated pipeline and finding that the output store route
was the real bottleneck.
```

Substory B: blocksolve-A replacement

Key observations from studying Triton:

- one program per `(batch, head, chunk)`;
- chunk64 is split into four 16-token sub-blocks;
- the expert pipeline computes the 10 lower-block Gram matrices;
- only lower 16x16 blocks are stored;
- the blocksolve is essentially a register-resident small-block triangular
  inverse pipeline.

TileLang constraint:

- stock TileLang 0.1.9 could not directly express the same register-resident
  fragment recirculation pattern for repeated RR GEMMs;
- the competitive TileLang path used shared-memory solve tiles instead.

Candidate ladder:

| Candidate | Lesson |
| --- | --- |
| 001 fragment RR solve | compile/layout failure; pure Triton-style fragment recirculation not directly expressible |
| 002 fp32 shared solve | correct but too slow |
| 003 fp16 shared solve | much closer, about 0.189 ms |
| 014 anchored gate precompute | reduced repeated exp work without unsafe fp16 overflow |

Final isolated result:

| Backend | Latency |
| --- | ---: |
| Triton blocksolve-A | 0.14692616 ms |
| TileLang blocksolve-A | 0.11384284 ms |

Article lesson:

```text
The agent did not blindly remove Triton. With the expert pipeline understood
(four 16-token sub-blocks, 10 lower Gram blocks, register-resident small-block
solve, and lower-block stores), it searched TileLang's expressible space.  The
boundary mattered: direct fragment recirculation was not expressible in stock
TileLang 0.1.9, so the competitive route used shared solve tiles instead.
```

Needed before publication:

- create a compact blocksolve candidate table with no more than 6-8 rows;
- include one diagram of the four 16-token sub-blocks and 10 lower Gram blocks;
- verify final component benchmark and full-op benchmark are generated from the
  same final production commit.

### 8. Human Algorithmic Moves: Changing The Search Space

Purpose:

- Distinguish agentic tuning inside an existing component contract from
  algorithmic changes that create a new contract or new execution structure.
- Show where human reasoning changed what was worth tuning.

Algorithmic move 1: Neumann-style prepare / inverse view

Treating chunk-local causal dependency as a blocked inverse problem enabled
GPU-friendly prepare variants. This is a mathematical reframing, not a
Triton-to-TileLang translation. The agent tuned within this space by reducing
diagonal Neumann rounds from 4 to 1, but the inverse framing itself was a human
algorithmic contribution.

Algorithmic move 2: associative prefix scan for chunk recurrence

Recognizing the affine composition structure changed long-sequence recurrence
from serial to scannable:

```text
(A_2, b_2) o (A_1, b_1) = (A_2 A_1, A_2 b_1 + b_2)
```

This requires shape-aware dispatch. H16 long sequences benefit from the current
scan formulation; earlier dense-transition H32 scan experiments did not. The
latest full-op H32/S128K production row is nevertheless faster than FLA because
production dispatch is allowed to choose the measured-safe path instead of
forcing the scan variant.

Article lesson:

```text
Agents can tune a kernel once the structure is chosen. Human algorithmic
insight can change the structure itself.
```

Needed before publication:

- include one "scan wins" and one "scan loses" row in the production dispatch
  discussion to avoid overgeneralization.

### 9. Production Integration: What Survived

Purpose:

- Show how local wins become production code.
- Keep this section concise and engineering-focused.

Final production statement:

```text
The validated BTHD production path in gated_deltanet_prefill.py is now fully
TileLang for the prefill kernel surface discussed here. A source search for
"triton" and "tl." in that file returns no matches.
```

On validated shapes, the fully TileLang path shows two levels of evidence:

- isolated blocksolve-A: 1.29x faster than the local Triton baseline;
- full-op BTHD validated rows: 1.02-1.16x faster than FLA 0.5.1.

Isolated blocksolve-A:

| Backend | Latency |
| --- | ---: |
| Triton local baseline | 0.14692616 ms |
| TileLang production | 0.11384284 ms |

Final full-op benchmark table:

| Seq len | Heads | TileOps TileLang | FLA 0.5.1 | FLA / TileOps |
| --- | ---: | ---: | ---: | ---: |
| 32K | 16 | 2.39824722 ms | 2.44284368 ms | 1.01860x |
| 64K | 16 | 4.64622306 ms | 5.20679608 ms | 1.12065x |
| 128K | 16 | 9.35985710 ms | 10.87307050 ms | 1.16167x |
| 128K | 32 | 14.51952280 ms | 16.09163698 ms | 1.10828x |

Testing:

```text
tests/ops/test_gated_deltanet_prefill.py: 8 passed
```

Production guard discussion:

- report only validated shapes;
- do not claim broad dominance over all FLA configurations;
- explain that shape-specific dispatch is a feature, not a weakness.

Needed before publication:

- decide whether to include older Week 1 table or only final docker table;
- if both are included, explain why numbers changed: different code state,
  final TileLang migration, and benchmark environment.

### 10. Performance Progression Against FLA

Purpose:

- Show the tuning trajectory against FLA without replaying implementation
  details.
- Make clear that progression rows can be directional across experiment days,
  while the final docker table is the strict publication-quality comparison.

Format:

- one compact S32K/H16 progression table;
- one final four-row docker table for the PR-head comparison;
- a short note that older rows are not a single controlled benchmark suite.

### 11. Negative Results That Teach Something

Purpose:

- Preserve the tuning method.
- Avoid a victory-only narrative.

Candidate negative results to include:

- naive TileLang forward solve was not a fair FLA-style solve;
- broad `T.gemm` policy sweeps did not solve blocksolve-A;
- `T.async_copy` alone did not fix recompute;
- fragment RR blocksolve could not be expressed directly in TileLang 0.1.9 due
  to layout inference constraints;
- sparse anchored gate precompute lost to dense precompute;
- broad scan dispatch loses outside validated shapes.

Format:

- one compact table;
- each row should have "what we learned", not just "slower".

### 12. Lessons For TileLang Developers

Suggested final lessons:

1. Start with decomposition, not code edits.
2. Define the fair baseline before optimizing.
3. Use agents for implementation-space search where feedback is measurable.
4. Study expert kernels before replacing them.
5. Inspect the generated pipeline; high-level intent is not enough.
6. Keep negative results in the decision log.
7. Separate algorithmic moves from local lowering choices.
8. Production dispatch should be shape-aware.

## 7. Suggested Figures And Tables

Required figures:

1. Three-stage GDN prefill pipeline:
   prepare -> h recurrence -> output.
2. AKO loop diagram:
   observe -> hypothesize -> edit -> compile -> test -> benchmark -> log.
3. Blocksolve-A 4x16 sub-block diagram:
   show 10 lower triangular Gram blocks.

Strong optional figures:

1. h recurrence before/after scale placement.
2. recompute output-store path:
   direct fragment store vs swizzled shared staging.
3. scan recurrence diagram:
   serial chunks vs affine transition scan.

Required tables:

1. agent tuning levels taxonomy;
2. final TileOps vs FLA full-op benchmark;
3. h recurrence scale-placement ablation;
4. recompute lowering candidate ladder;
5. blocksolve-A candidate ladder.

Optional tables:

1. scan guard sweep;
2. component breakdown before and after final migration;
3. NCU or generated-code summary for accepted TileLang kernels.

## 8. Experiments To Add Or Refresh

This section separates must-have evidence from publication-strengthening and
appendix material.

### Priority 1: Blocking For Public Draft

#### 1. Final full-op benchmark table from the final production commit

Current PR-head evidence exists from docker:

```text
S32K/H16:  TileOps 2.39824722 ms,  FLA 2.44284368 ms
S64K/H16:  TileOps 4.64622306 ms,  FLA 5.20679608 ms
S128K/H16: TileOps 9.35985710 ms,  FLA 10.87307050 ms
S128K/H32: TileOps 14.51952280 ms, FLA 16.09163698 ms
```

Before publication:

- rerun or archive the exact command, image, git commit, and raw JSONL;
- confirm `output_final_state=True`;
- confirm the mounted PR worktree is the final fully TileLang version;
- record GPU model and clock state.

#### 2. Final source audit

Needed evidence:

```bash
rg -n "triton|tl\." tileops/kernels/gated_deltanet/gated_deltanet_prefill.py
```

Expected result:

```text
no matches
```

Also record:

```bash
python -m pytest tests/ops/test_gated_deltanet_prefill.py -q
```

Expected result:

```text
8 passed
```

#### 3. Compact recompute candidate ladder

Use existing AKO logs, but make the final article table concise.

Need rows:

- simple TileLang baseline;
- async copy diagnostic;
- no-store diagnostic;
- shared-copy store;
- swizzled shared output accepted candidate.

Report:

- correctness;
- latency;
- one-line lesson.

#### 4. Compact blocksolve-A candidate ladder

Use existing AKO logs.

Need rows:

- Triton reference pipeline summary;
- 001 fragment RR compile failure;
- 002 fp32 shared solve;
- 003 fp16 shared solve;
- 014 anchored gate precompute;
- 018 one-round Neumann accepted candidate.

Report:

- correctness / compile status;
- isolated latency if available;
- one-line lesson.

### Priority 2: Strongly Recommended Before Publication

These experiments strengthen claims, but drafting can begin without them if
they are clearly marked as future-work items.

#### 5. H-recurrence AKO ablation against final production commit

Existing Week 1 data is good, but it predates the final all-TileLang prepare
migration.

Refresh if possible:

```text
B=1, H=16, S=32K, DK=DV=128, chunk64, fp16
old: scale k
new: scale v_new
```

Report:

- h-only latency;
- correctness max diff;
- shared memory and registers if easy to collect;
- whether full-op latency moves similarly.

#### 6. Final component breakdown

The article needs one table showing where final time goes.

Suggested rows:

```text
B=1, H=16, S=32K, DK=DV=128, chunk64, fp16
B=1, H=16, S=128K, DK=DV=128, chunk64, fp16
```

Components:

- chunk cumsum;
- prepare/blocksolve/recompute;
- h recurrence or scan summary/replay;
- output;
- full op.

Purpose:

- prevents the article from feeling like isolated microbenchmarks only;
- shows how local wins affect production.

### Priority 3: Nice-To-Have, Do Not Block Drafting

These are supplementary material or appendix candidates.

#### 7. Scan guard sweep

Suggested rows:

```text
H16/S32K: baseline should win or scan should stay disabled
H16/S64K: scan should win
H16/S128K: scan should win more clearly
H32/S128K: current dense scan should lose
```

Purpose:

- justify shape-aware dispatch;
- support the human algorithmic scan section.

#### 8. Generated-code or NCU evidence for key claims

Useful snapshots:

- recompute accepted candidate uses swizzled shared output path;
- blocksolve accepted candidate has no Triton dependency;
- `T.async_copy` lowering already emits commit group, so explicit
  `T.ptx_commit_group()` is redundant;
- `T.ptx_wait_group(0)` remains necessary before consuming async-copied shared
  memory.

Purpose:

- gives TileLang developers concrete lowering lessons.

#### 9. Shape generalization table

Optional but valuable:

```text
H in {4, 16, 32, 64}
S in {32K, 128K}
```

Purpose:

- clarify where the current production path is validated;
- avoid overclaiming.

#### 10. Reproducibility cleanup

Update:

- `REPRODUCE.md`;
- final benchmark JSONL names;
- benchmark command snippets;
- environment versions.

## 9. Raw Evidence To Reference

Existing planning and draft:

```text
planning.md
drafts/tutorial.md
drafts/tutorial_v2.md
drafts/final_p1_evidence_20260620.md
data/week1_experiment_log.md
```

Final AKO decision log:

```text
/home/ga/2026-06-15/TileOPs/experiments/gated_deltanet_prefill_ako/notes/tilelang_recompute_lowering_decision_log.md
```

Production file:

```text
/home/ga/TileOPs-pr1596/tileops/kernels/gated_deltanet/gated_deltanet_prefill.py
```

Benchmark docker:

```text
/home/ga/Documents/gdn_kernel_bench_2026-06-18
image: gdn-kernel-bench:nightly-tl019
```

## 10. Claims Policy

Allowed claims:

- The final validated BTHD production path is fully TileLang in the
  discussed prefill kernel file.
- On validated BTHD S32K/H16, S64K/H16, S128K/H16, and S128K/H32 rows, TileOps
  beats the fair FLA 0.5.1 baseline in the current docker benchmark.
- The TileLang blocksolve-A replacement is faster than the local Triton
  blocksolve-A baseline on S32K/H16 in isolated measurement.
- The agentic loop performed hardware-aware pipeline analysis: studying expert
  Triton kernels, diagnosing bottlenecks through lowering inspection, and
  searching TileLang's expressible space.
- Agentic tuning was effective at implementation-space optimization and expert
  kernel alignment.
- Human algorithmic insight supplied Neumann-style prepare framing and
  associative scan recurrence.

Avoid or qualify:

- "TileLang is faster than Triton" without specifying this component and shape.
- "TileOps beats FLA" without specifying FLA version, shape, layout, dtype, and
  `output_final_state=True`.
- "AI discovered the algorithm" for Neumann prepare or scan recurrence.
- "Agents can read papers and extract algorithms" without specifying that paper
  context was human-provided and the agent organized reference code and tests.
- "Agents proposed optimal fusion boundaries"; agents can propose first fusion
  from dataflow, but optimization requires profiling and performance modeling.
- claims about untested shapes.
- claims that `T.async_copy` alone explains the recompute win.

## 11. Proposed Claude Review Questions

Ask Claude to review this plan for:

1. Does the article now read like a community tuning tutorial rather than an
   internal experiment report?
2. Does infrastructure now appear before capabilities in a way that makes the
   agentic workflow understandable?
3. Is Level 0 scoped correctly as pre-tuning scaffolding rather than a major
   standalone claim?
4. Is the separation between agentic tuning inside a measurable component
   contract and human algorithmic innovation clear?
5. Are the three agentic teaching nodes the right ones:
   scale placement, recompute store-path diagnosis, and expert-pipeline
   alignment?
6. Is the final result stated strongly enough without overclaiming?
7. Which Priority 2 experiments should be completed before publication?
8. Which sections should be shortened to keep the article readable?
9. Does the proposed outline resemble a practical tuning guide in the spirit of
   classic GEMM optimization writeups?
