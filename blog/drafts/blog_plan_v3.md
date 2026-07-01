# Blog Plan V3: Agentic TileLang Kernel Tuning For Gated DeltaNet Prefill

Draft status: planning document for review. This supersedes `blog_plan_v2.md`
as the narrative plan, but keeps v2 as historical context.

## 1. Why V3 Exists

The v2 plan was written before the FlashQLA-specialized phase was complete.
Since then, the project added a second major body of work:

- reading and benchmarking Qwen's FlashQLA implementation;
- migrating key FlashQLA kernels across TileLang 0.1.8 and 0.1.9;
- learning the CP-split prefill schedule;
- porting that schedule into TileOps-owned kernels;
- tuning CP split parameters and H64 dispatch;
- reproducing Qwen-style benchmark rows against TileOps, FlashQLA, and FLA.

That changes the article shape. The public article should no longer read as a
linear sequence of AKO rounds ending with blocksolve-A. It should explain a
larger workflow:

```text
read the paper and existing code
-> build a correct, measurable TileLang operator
-> let AKO search local implementation space inside that fixed contract
-> identify the boundary of autonomous local tuning
-> use human mathematical insight and expert open-source schedules to change
   the search space
-> productionize the new path with correctness, benchmark, and dispatch gates
```

The article's main lesson is now stronger:

```text
Agents helped turn a paper operator into a correct, measurable TileLang
implementation, then made local kernel optimization systematic. The largest
search-space changes still required external input: human mathematical insight
for the prepare/blocksolve formulation, and Qwen's FlashQLA as the
production-grade CP-split schedule reference.
```

## 2. Core Thesis

Suggested title:

```text
Agentic TileLang Kernel Tuning: From Local AKO To CP-Split Gated DeltaNet
Prefill
```

Alternative title:

```text
How We Tuned Gated DeltaNet Prefill In TileLang With Agents, Expert Kernels,
And Human Algorithmic Insight
```

Core thesis:

```text
Agents are effective when the kernel problem is made measurable: they can
read papers and reference implementations, organize equations, build tests,
inspect lowering, search TileLang implementation choices, and tune local
memory, instruction, and small-fusion variants inside a fixed semantic
contract.

But agents do not automatically invent every useful search space. The largest
performance jumps came after external input changed the problem being searched:
first a human expert reframed prepare/blocksolve as a blocked
inverse/Neumann-style problem, then Qwen's FlashQLA supplied the
production-grade CP-split prefill schedule for long-sequence replay.
```

Short form:

```text
Agents can reconstruct and refine a measurable search space; experts and
expert kernels reshape it.
```

This is deliberately not an "AI invented a faster kernel" story. It is a
practical case study in collaboration between:

- agentic implementation search;
- human algorithmic judgment;
- expert open-source references such as FLA, Triton kernels, and Qwen
  FlashQLA;
- production correctness and benchmark gates.

## 3. Intended Audience

Primary audience:

- TileLang and TileOps developers;
- GPU kernel engineers interested in AI-assisted tuning workflows;
- ML systems engineers maintaining model-specific serving kernels;
- readers who want an honest account of what agents helped with and where
  expert intervention mattered.

Assumed background:

- understands GPU basics: shared memory, registers, occupancy, async/TMA copy,
  tensor cores, and kernel launch overhead;
- has seen Triton, CUDA, or TileLang-like DSL kernels;
- does not need prior Gated DeltaNet expertise.

The article should explain enough GDN math to understand the optimization
surface, but should not become a model-paper summary.

## 3.5 Development Timeline

The article should make the sequence of contributions explicit. The early
agent work produced a correct, measurable operator and explored local TileLang
implementation space. Neumann prepare was an externally supplied human
mathematical reframing that AKO then implemented and tuned. FlashQLA adoption
happened later as a separate expert-reference phase.

| Phase | Date Range | Key Milestones | Representative Result |
| --- | --- | --- | --- |
| Correct operator + local AKO | Early-mid June 2026 | Paper/code reading, correctness gates, H recurrence, recompute, early fusion | Pre-FlashQLA local path and component bottlenecks identified |
| Human Neumann prepare | Mid June 2026 | Blocked inverse framing, Neumann-style prepare, AKO implementation/tuning | Pre-FlashQLA path around `2.4 ms` on `32K/H16` historical rows |
| FlashQLA study | June 22-23, 2026 | Source reading, scheduling analysis, producer/consumer skeleton extraction | CP-split schedule understood and correctness target defined |
| FlashQLA migration | June 23-24, 2026 | TL0.1.8 -> TL0.1.9 migration, TMA restoration, lowering inspection | Restored FlashQLA-style parity around `1.31 ms` historical `64K/H16` rows |
| Hybrid proof | June 24, 2026 | TileOps A producer + FlashQLA CP back half | Around `1.05 ms` historical `64K/H16` proof row |
| TileOps CP port | June 24-25, 2026 | TileOps-owned CP kernels, validation, dispatch integration | Performance maintained in owned implementation |
| Tuning + H64 | June 25, 2026 | CP split sweep, H64 dispatch correction, final PR validation | Production-ready PR path and refreshed Qwen-style benchmark rows |

## 4. Attribution Policy

The article should explicitly credit Qwen's FlashQLA project.

Recommended wording:

```text
We credit Qwen's FlashQLA project (https://github.com/QwenLM/FlashQLA,
commit 6ef4858b) for the CP-split GDN prefill schedule that became the
production reference for the second half of this work. Our TileOps
`fused_fwd.py` kernel structure directly follows FlashQLA's
`hopper/fused_fwd.py` producer/consumer skeleton, adapted for the current
TileLang 0.1.11 lowering line and TileOps dispatch.
```

Direct non-originality statement to preserve in the article:

```text
TileOps did not invent the CP-split replay schedule. The contribution was to
study, validate, port, tune, dispatch, benchmark, and productionize the
FlashQLA-style schedule inside TileOps, combined with TileOps' own A producer.
```

Expanded wording:

```text
Our earlier tuning had already identified long recurrence replay as the
dominant obstacle and explored grouped/prefix-style ideas. FlashQLA showed the
production-grade way to attack that long-replay bottleneck: compute corrected
segment initial states, then run fused replay/output over shorter segments with
a carefully partitioned producer/consumer pipeline.
```

The article should separate contributions cleanly:

| Source | Contribution |
| --- | --- |
| FLA / earlier expert kernels | Full-op behavioral reference and component-level expert baselines. |
| Qwen FlashQLA | Production CP-split prefill schedule and fused replay/output skeleton. |
| Human expert work in this project | Blocked inverse/Neumann prepare framing (original algorithmic insight); recognizing long-sequence replay as a fundamental bottleneck beyond local tuning; strategic decision to study, validate, and port FlashQLA's CP-split schedule; treating CP split length as tunable dispatch; proposing/investigating butterfly or prefix composition as a research branch. |
| AKO / agentic loop | Paper/reference reading support, equation-to-kernel scaffolding, correctness tests, candidate implementation, lowering inspection, local memory/instruction/fusion tuning, parameter sweeps, benchmark logging, and productionization support. |
| TileOps/TileLang implementation | TileOps-owned BTHD production path, faster specialized A producer, CP-split port, shape-aware dispatch, and validation. |
| TileLang 0.1.11 compiler/runtime line | Current TileOps lowering/runtime environment for the owned port. Performance differences versus public FlashQLA TL0.1.8 should be framed as environment/implementation evidence, not fully attributed unless generated-code evidence is shown. |

Avoid these two bad framings:

- Do not imply we independently invented FlashQLA's CP-split production
  schedule.
- Do not imply the work was "just copying FlashQLA"; the TileOps-owned port,
  A producer, CP parameter tuning, H64 dispatch, validation, and FLA/FlashQLA
  comparisons are part of this work.

## 5. Final Article Shape

Use a three-level capability structure rather than a chronological experiment
diary. The levels are not "math, then kernels, then benchmarks"; they are
three different roles an agent can play in a serious kernel project.

### Level 1: From Paper Operator To Correct TileLang Kernel

Purpose:

- Show that the first useful agent capability is not performance tuning; it is
  reconstructing the operator from papers, reference implementations, and
  existing code into a correct, measurable implementation.
- Explain why GDN prefill is hard: delta residual writes, gate decay,
  chunk-local triangular work, cross-chunk recurrence, output replay,
  final-state production, layout sensitivity, and long-context serving shapes.
- Introduce the target path: BTHD, `B=1`, `DK=DV=128`, `chunk64`, `fp16`,
  long prefill.
- Explain why correctness gates and component benchmarks are the first
  engineering artifact.

Core message:

```text
Before an agent can tune a recurrent kernel, it must help make the paper
operator executable, testable, and decomposed into measurable components.
```

Sections likely in this level:

1. Why GDN prefill is important and hard.
2. Decode view: GDN as gated delta memory.
   - ordinary linear attention blindly accumulates;
   - GDN reads the old memory under `k_t` and writes only the residual;
   - `g` controls memory lifetime and `beta` controls write strength.
3. Chunkwise prefill formulas.
   - single-token decode: `w_t = beta_t k_t`, `u_t = beta_t v_t`;
   - chunkwise prepare: `W = A(beta K)`, `U = A(beta V)`;
   - fused replay/output: state contribution, local causal contribution, and
     state update.
4. TileLang as the tuning surface.
5. AKO infrastructure: correctness, compile, benchmark, lowering, logs.

Evidence to cite:

- `tutorial_v2.md` Sections 0-4;
- `kernel_code_shapes_20260629.md`;
- `serial_ako_protocol.md`;
- `final_p1_evidence_20260620.md`;
- current benchmark harness files in `gdn_kernel_bench_2026-06-18`.

### Level 2: Autonomous Local AKO Inside A Fixed Contract

Purpose:

- Show concrete AKO wins and failures inside fixed component contracts.
- Teach how agents can edit TileLang, run gates, inspect lowering, and refine
  candidates without changing the mathematical search space.
- Establish that local tuning reached a performance ceiling on long-sequence
  replay.
- Prove that removing intermediates is not the same as breaking the
  O(sequence_length) replay bottleneck.

Important boundary:

Do not put blocked inverse/Neumann prepare in this level. That search space was
not discovered by autonomous local AKO; it was introduced by external human
mathematical insight and belongs in Level 3.

Recommended teaching nodes:

1. **H recurrence: move the scale, remove the buffer.**
   - Local algebraic placement and memory traffic reduction.
   - Shows AKO can make a small semantic-preserving change that matters.

2. **Recompute: find the real bottleneck, not the fashionable primitive.**
   - Async/WGMMA-like structure was not enough; output/copy/store path mattered.
   - Shows why component benchmarks and lowering inspection are necessary.

3. **Early fusion and skeleton attempts: fusing more is not enough.**
   - Direct fused-HO and full-DV fused skeleton attempts removed some
     intermediates but still replayed the long sequence serially.
   - This should be the bridge into Level 3.

What to avoid:

- Do not over-list every AKO round.
- Use tables for failed candidates.
- Keep the story at the level of hypothesis -> gate -> lesson.

Candidate table:

| Candidate family | What it taught |
| --- | --- |
| h recurrence scale placement | Local algebraic movement can reduce memory traffic. |
| recompute copy/store variants | Matching a primitive is not the same as matching the pipeline. |
| direct fused serial / fused-HO | Removing intermediates does not break the long recurrence. |
| full-DV fused skeleton without CP split | Full-DV/TMA/WGMMA-looking lowering still loses if the schedule scans 64K serially. |

### Level 3: External Input Changes The Search Space

This level should be the climax of the article. It explains how the search
space was reshaped through two external channels: human mathematical insight
(blocked-inverse/Neumann prepare) and expert open-source schedule adoption
(FlashQLA's CP-split schedule).

Purpose:

- Show that local AKO reached a performance ceiling.
- Present the two ways the search space expanded.
- Distinguish original framing from reference study.
- Explain migration as engineering discovery, not a mechanical copy.

#### Contribution 1: Prepare As A Blocked Inverse / Neumann Problem

Classification: original algorithmic insight in this project.

Story:

- The prepare stage is not merely "write the same triangular solve in
  TileLang."
- Human analysis reframed it as a blocked lower-triangular inverse problem.
- Once that search space existed, AKO could tune inside it.

Technical content:

```text
64-token chunk
-> four 16-token Gram blocks
-> lower-triangle-only products
-> shared lower blocks
-> Neumann-style small-block solve
```

Important nuance:

The Neumann/inverse framing is a human algorithmic choice. The agentic loop
then explored implementable variants, compile gates, correctness gates, and
component performance.

Formula publication gate:

- Do not publish exact Neumann/blocksolve notation until it has been checked
  against `_prefill_blocksolve_A_bthd_tl`.
- Verify sign convention, gate/exponent placement, beta placement, matrix
  orientation, and whether `A` is expressed as a left- or right-multiply.
- If the exact notation becomes too noisy for the article, mark the formulas
  and figure caption as schematic: they describe the computational shape, not
  exact implementation notation.

Evidence:

- `tutorial_v2.md` Section 7;
- `kernel_code_shapes_20260629.md`;
- `latest_fla_prepare_decision_log.md`;
- `tilelang_recompute_lowering_decision_log.md`;
- `final_p1_evidence_20260620.md`.

#### Bridge: Why FlashQLA Was Needed

After Neumann prepare improved the A producer, TileOps was ahead of FLA on the
prepare side but still replayed long sequences with a schedule that did not
break the sequence-length dependency wall. Fusion attempts in the early
FlashQLA-specialized rounds removed intermediate buffers and improved local
data movement, but they still left a long replay loop in the hot path.

The important boundary discovered by AKO was:

```text
less materialization != shorter recurrence
```

Component profiling and failed fused candidates showed that replay/output, not
prepare alone, dominated the long-context rows. That made a schedule-level
change necessary. FlashQLA provided the production reference for that change.

#### Contribution 2: FlashQLA CP-Split Schedule

Classification: expert open-source reference adoption. We should explicitly
credit Qwen FlashQLA for the CP-split schedule and fused replay/output
skeleton; our contribution was understanding, validating, migrating, tuning,
and productionizing the schedule inside TileOps.

Story:

- Earlier work identified long recurrence replay as the obstacle and had
  grouped/prefix-style ideas.
- Qwen FlashQLA gave a production reference for the right schedule:

```text
chunk_local_cumsum
-> kkt_solve(A)
-> get_warmup_chunks
-> prepare_h
-> correct_h0
-> fused_gdr_fwd
-> o + final_state
```

- The decisive insight is not merely "fuse output":

```text
first compute valid segment initial states
then run fused replay/output over shorter segments
```

Technical content:

- producer/store group and S/V/O consumer split;
- TMA loads and delayed output store;
- CP segment initial-state correction;
- replay/output over shorter segments;
- no global `w/u/S/v_new` materialization on the inference path.

Implementation story:

1. Understand FlashQLA's CP split and producer/consumer fused-forward skeleton.
2. Validate/migrate FlashQLA across TileLang 0.1.8 and the current TileLang
   0.1.x line used by TileOps.
3. Discover that lowering behavior, not source-level spelling, determines
   performance: plain `T.copy` in the migration did not match the old
   TMA-specialized paths, and explicit `T.tma_copy(..., barrier=...)` was
   needed to restore performance.
4. Use TileOps' faster A producer plus FlashQLA CP back half as a hybrid proof.
5. Port the CP back half into TileOps-owned kernels.
6. Tune CP split length and dispatch.
7. Study butterfly/prefix composition and reject the production-shaped prefix
   insertion path for now because every tested insertion point paid extra
   recurrence/summary/output-correction cost; the prefix idea remains
   mathematically valid, but the representation and pipeline overheads were
   too expensive in this kernel.

Historical trajectory evidence ladder, not final public benchmark claims:

| Milestone | Representative result |
| --- | ---: |
| Direct fused skeleton without CP split | `~4.15 ms` on 64K/H16 |
| TileOps A + FlashQLA CP back half | `~1.05 ms` on 64K/H16 |
| TileOps-owned CP split | `~1.05 ms` on 64K/H16 |
| CP split length sweep | old gate found `max_local_chunks=64` at `~0.94 ms` |
| H64 real CP follow-up | fixed earlier non-CP misdiagnosis |

Be careful with current numbers:

- The final public table should use the latest reproduced benchmark set, not
  the old Goal 5 single-row numbers alone.
- For every FlashQLA comparison row, record actual `max_local_chunks`, CP
  segment count, `block_DV`, TileLang version, and timer.

Evidence:

- `flashqla_specialization_plan_20260622.md`;
- `flashqla_source_reading_20260622.md`;
- `flashqla_scheduling_skeleton_20260623.md`;
- `flashqla_pipeline_overlap_20260623.md`;
- `flashqla_specialized_ako_log.md`;
- `kernel_code_shapes_20260629.md`;
- Qwen four-line benchmark result JSONL.

#### Negative Result: Hierarchical Prefix Is More Parallel But Too Heavy Here

Classification: human-suggested research branch, implemented and rejected by
AKO evidence for the current production path.

Story:

- Before and after studying FlashQLA, we explored whether recurrence replay
  could be turned into grouped transition composition / butterfly prefix scan.
- The mathematical idea is sound: a group transition can be written as
  `T(H) = M H + b`, and transitions compose associatively.
- The production blocker is the representation, not correctness:

```text
direct state:       H     has shape DK x DV
full transition:    M, b  have shape DK x DK and DK x DV
augmented summary:  [b | M] has width DV + DK
```

- On `DK=DV=128`, the full summary path roughly doubles the state width before
  considering prefix buffers, live state, and scheduling pressure.

Evidence:

- `h_replay_attribution_note_20260625.md`;
- `flashqla_specialized_ako_log.md` Rounds 031-035;
- Goal 7 probes:
  - full transition producer and compose lower in TileLang;
  - per-chunk `M @ H + b` is correct but about `1.5x` slower;
  - fused direct+summary is correct but about `2.2x` slower on `64K/H16`;
  - intra-CP prefix insertion validates the dependency idea but adds
    production-path overhead;
  - boundary prepass plus fused replay/output does not recover enough latency;
  - cluster2 fused and pair2 transition-state variants expose the same
    summary/composition cost;
  - sweeping `group_chunks=2/4/8/16` does not rescue the full `[b | M]`
    representation.

Compressed publication wording:

```text
We tested full transition summaries, intra-CP prefix insertion, boundary
prepass, cluster relay variants, pair2 transition-state, and fused
direct+summary. They validated the affine view, but every production-shaped
insertion paid extra recurrence/summary/output-correction cost; the full
`[b | M]` representation remained too heavy.
```

Blog lesson:

```text
Prefix scan improves dependency parallelism, not arithmetic efficiency. For
this GDN shape, the full affine transition representation is too expensive for
the production fused path. The idea remains a future research branch only if a
narrower transition representation is found.
```

Shape-scope sentence to preserve in the article:

```text
This rejects the tested full `[b | M]` transition representation for the
current `DK=DV=128`, `chunk64` production path, not a general claim that prefix
scan cannot work for GDN.
```

## 6. Proposed Public Outline

1. Introduction: why this case study matters.
2. GDN prefill in one page: state, chunks, prepare, replay, output.
3. The measurement loop: correctness, CUPTI, component breakdown, lowering.
4. What AKO did well:
   - h recurrence scale placement;
   - recompute pipeline diagnosis;
   - early fusion attempts and why they were insufficient.
5. Hitting the local tuning wall: why direct fusion was not enough.
6. Blocked inverse/Neumann prepare: the first search-space expansion.
7. Bridge: why prepare alone did not solve long sequences.
8. FlashQLA as the production reference:
   - explicit credit to Qwen;
   - CP split and corrected segment h0;
   - producer/consumer fused replay skeleton.
9. Migration engineering:
   - 9.1 Understanding FlashQLA's schedule and skeleton.
   - 9.2 Initial migration to the current TileOps TileLang line and
     performance regression.
   - 9.3 TMA restoration discovery: `T.copy` vs
     `T.tma_copy(..., barrier=...)`.
   - 9.4 Generated-code inspection showing explicit WGMMA and batched
     TMA-store.
   - 9.5 Lesson: source-level equality is not performance equality; lowering
     behavior determines speed.
10. TileOps productionization:
   - TileOps A producer (Neumann blocksolve from Level 3 Contribution 1,
     already around `1.29x` faster than the Triton isolated prepare baseline)
     combined with FlashQLA CP schedule;
   - Hybrid proof: TileOps A + FlashQLA CP back half showed schedule
     portability;
   - TileOps-owned CP split for full production ownership;
   - CP parameter tuning;
   - H64 dispatch correction;
   - butterfly/full-transition study and why the full `[b | M]` representation
     was deferred as future work rather than adopted for production.
11. Final benchmark and attribution:
   - TileOps vs FLA as the apples-to-apples production comparison;
   - TileOps vs FlashQLA with compiler/version differences called out;
   - component breakdown explaining where speed comes from.
   - hard evidence gate: the final comparison table must include
     `max_local_chunks`, CP segment count, `block_DV`, TileLang version,
     commit hash, and benchmark timer; otherwise the FlashQLA comparison does
     not enter the main article text.
   - h/replay attribution must use the cautious wording in
     `drafts/h_replay_attribution_note_20260625.md`: current component runs
     show a faster TileOps fused replay event, but the source skeleton follows
     FlashQLA closely, so frame the gap as implementation/lowering/dispatch
     evidence rather than a new high-level replay algorithm.
   - explicit contribution/attribution table.
12. Lessons:
   - agents refine measurable spaces;
   - humans and expert references reshape the spaces;
   - always inspect generated code;
   - negative results prevent wrong narratives;
   - production dispatch is part of the kernel.

## 7. Evidence And Data Plan

### Must Refresh Before Public Draft

1. Latest PR-head correctness.
   - Public FLA reference correctness for validated shapes.
   - Include max abs and state max abs.

2. Latest three-way benchmark.
   - TileOps on the current PR TileLang 0.1.11 environment.
   - FLA 0.5.1 in the same current benchmark environment.
   - FlashQLA in its intended TileLang 0.1.8 docker.
   - Use GPU4 if that remains the agreed measurement device.
   - Use TileOps benchmark infrastructure and CUPTI timer.

3. Component breakdown for final rows.
   - A/KKT producer.
   - CP preprocess: get_warmup, prepare_h, correct_h0.
   - fused replay/output.
   - final-state or initialization overhead if visible.

4. Dispatch metadata for each row.
   - actual `max_local_chunks`;
   - CP segment count;
   - `block_DV`;
   - TileLang version;
   - FLA version;
   - commit hash;
   - benchmark timer configuration.

5. FlashQLA attribution sanity check.
   - Verify source-level references and generated-code claims used in the
     article.
   - Avoid claiming PTX/SASS facts unless we actually inspect PTX/SASS.

6. Component breakdown for current PR-head snapshot.
   - Current snapshot exists as Table D below; refresh it if the PR head,
     runtime docker, or dispatch heuristic changes.
   - Report: A/KKT producer, CP preprocess (`get_warmup + prepare_h +
     correct_h0`), fused replay/output, and overhead.
   - **Publication gate:** a current breakdown must exist before claiming
     "replay dominated" in the Bridge section or justifying butterfly deferral
     in Section 10.
   - Use the measured proportions, not a target percentage. The current rows
     show fused replay/output at roughly `61-70%` of filtered kernel time, and
     `correct_h0` alone at roughly `1.6-4.3%`.

### Nice To Have

- Standalone diagram of FlashQLA CP split.
- Diagram comparing "direct fused replay" vs "CP split + replay".
- Short timeline table from initial production path to final CP-split path.
- Appendix table of rejected candidate families.

### Evidence Tier System

Use a three-tier evidence hierarchy so the article does not mix historical
trajectory numbers with final public claims.

**Tier 1: Final Claims (latest PR-head required).**

- TileOps vs FLA full-op comparison.
- TileOps vs FlashQLA comparison, with compiler/runtime differences stated.
- Component breakdown for final rows.
- Correctness validation against FLA.
- Dispatch metadata: `max_local_chunks`, CP segment count, `block_DV`,
  TileLang version, FLA version, commit hash, and timer.

**Tier 2: Historical Trajectory (dated evidence allowed).**

- Pre-FlashQLA local AKO baseline: around `2.4 ms` on historical 2026-06-20
  `32K/H16` rows.
- Hybrid proof milestone: around `1.05 ms` on historical 2026-06-24
  `64K/H16` rows.
- CP sweep milestone: historical sweep found `max_local_chunks=64` around
  `0.94 ms`.

Label these explicitly as historical evidence with dates.

**Tier 3: Cannot Use Without Qualification.**

- Any FlashQLA comparison without noting TL0.1.8 vs the current TileOps
  TileLang 0.1.11 environment.
- Any "faster than FlashQLA" claim without compiler context.
- Any PTX/SASS claim unless the generated code was inspected and archived.

### Current PR-Head Benchmark Snapshot

Current candidate evidence snapshot as of 2026-06-25, H200/GPU4, CUPTI timer,
`warmup=10`, `repeat=100`, `trials=7`, BTHD, `B=1`, `DK=DV=128`,
`chunk=64`, `fp16`, seed 0. Use this for planning and rewrite structure, but
refresh before publication if the PR head, runtime docker, TileLang wheel,
dispatch heuristic, or benchmark timer changes.

Publication refresh trigger:

```text
Rerun Tier-1 correctness and benchmark tables if the PR head, TileLang wheel,
docker/runtime, dispatch heuristic, benchmark timer, GPU, or FlashQLA/FLA
environment changes.
```

Table A: TileOps vs FLA, apples-to-apples production comparison.

| Case | TileOps TL0.1.11 | FLA 0.5.1 | TileOps throughput speedup |
| --- | ---: | ---: | ---: |
| `32K/H16` | `0.369577 ms` | `2.364970 ms` | `6.399x` |
| `64K/H16` | `0.690859 ms` | `5.265457 ms` | `7.622x` |
| `128K/H16` | `1.227298 ms` | `10.899666 ms` | `8.881x` |
| `128K/H32` | `2.384966 ms` | `16.687767 ms` | `6.997x` |

Table B: TileOps vs FlashQLA, same high-level CP-split schedule family but
different compiler/runtime environment.

| Case | TileOps TL0.1.11 | FlashQLA TL0.1.8 | TileOps throughput speedup | Note |
| --- | ---: | ---: | ---: | --- |
| `32K/H16` | `0.369577 ms` | `0.583178 ms` | `1.578x` | FlashQLA uses TL0.1.8 public lowering path. |
| `64K/H16` | `0.690859 ms` | `1.331969 ms` | `1.928x` | Same schedule family; compiler/lowering differences matter. |
| `128K/H16` | `1.227298 ms` | `2.623757 ms` | `2.138x` | Do not frame this as independent algorithm invention. |
| `128K/H32` | `2.384966 ms` | `5.395038 ms` | `2.262x` | Use component breakdown to explain the gap. |

This table is a three-way public-environment comparison, not a controlled
same-lowering replay attribution experiment.

Caption requirement for Table B:

```text
TileOps uses the current TileLang 0.1.11 PR environment. FlashQLA uses its
public TileLang 0.1.8 environment. Both implement the same CP-split fused
replay schedule family. Performance differences should be discussed as
implementation/compiler/dispatch differences, not as a claim that TileOps
invented a different high-level algorithm. Mention explicit WGMMA or batched
TMA-store only when the article also shows the generated-code evidence.
```

Important: in the final article, Table B must appear with the caption above or
equivalent wording. The caption is load-bearing: without it, readers may
misinterpret the speedup as algorithmic superiority. Article reviewers should
verify this caption is present and unambiguous before publication.

Table C: Correctness validation against FLA 0.5.1, same 2026-06-25 snapshot,
`atol=rtol=5e-2`.

| Case | `o` max abs diff | `final_state` max abs diff | Status |
| --- | ---: | ---: | --- |
| `32K/H16` | `0.0011291504` | `0.0002846699` | ok |
| `64K/H16` | `0.0012512207` | `0.0006084554` | ok |
| `128K/H16` | `0.0012817383` | `0.0004399344` | ok |
| `128K/H32` | `0.0011291504` | `0.0005266964` | ok |

These values are small enough to support the final benchmark claim; refresh
them again if the PR head or FLA reference environment changes.

Alias-fix regression check: after the CI-only TileLang scalar-alias fix
(`7fb5f35d`), the same four TileOps rows were rerun with GPU4/CUPTI
(`warmup=10`, `repeat=50`, `trials=3`) and showed no meaningful performance
regression relative to the component snapshot: `32K/H16 = 0.371738 ms`,
`64K/H16 = 0.694596 ms`, `128K/H16 = 1.229067 ms`, and
`128K/H32 = 2.386888 ms`.

Table D: Current TileOps component breakdown, same shape set, PR head
`a329a6c1`, H200/GPU4, TileLang 0.1.11 PR environment. This table is from a
flush-aligned CUPTI profiler pass (`warmup=5`, `repeat=20`, `trials=3`) and
uses median filtered CUDA kernel time, so totals are for attribution rather
than the canonical full-op latency table. The later alias fix was separately
verified not to change full-op latency; rerun Table D before publication if
the final article requires component proportions at the exact final commit.

| Case | Filtered total | A/KKT producer | CP preprocess | Fused replay/output | Other |
| --- | ---: | ---: | ---: | ---: | ---: |
| `32K/H16` | `0.372823 ms` | `0.079771 ms` (`21.4%`) | `0.048397 ms` (`13.0%`) | `0.228162 ms` (`61.2%`) | `0.016493 ms` (`4.4%`) |
| `64K/H16` | `0.694473 ms` | `0.147712 ms` (`21.3%`) | `0.085818 ms` (`12.4%`) | `0.443514 ms` (`63.9%`) | `0.017429 ms` (`2.5%`) |
| `128K/H16` | `1.234902 ms` | `0.285147 ms` (`23.1%`) | `0.089768 ms` (`7.3%`) | `0.841582 ms` (`68.1%`) | `0.018405 ms` (`1.5%`) |
| `128K/H32` | `2.383350 ms` | `0.564967 ms` (`23.7%`) | `0.136112 ms` (`5.7%`) | `1.661549 ms` (`69.7%`) | `0.020722 ms` (`0.9%`) |

CP preprocess here means `get_warmup_chunks + prepare_h + correct_h0`.
`correct_h0` alone is small (`~1.6-4.3%` of filtered total on these rows), so a
butterfly optimization limited only to that correction stage would not move the
full-op much. The later Goal 7 hierarchical-prefix experiments give the
stronger reason to defer full prefix productionization: the full `[b | M]`
transition representation is too expensive even when fused with direct
replay/output. The dominant component is fused replay/output, which supports
the Level 2/Level 3 claim that shortening the replay schedule mattered more
than local fusion alone.
For TileOps-vs-FlashQLA back-half attribution, use
`drafts/h_replay_attribution_note_20260625.md`: current component data shows
TileOps' fused replay/output event is shorter on measured H16/H32 rows, but
the source skeleton is largely FlashQLA-derived, so the blog should frame the
gap as an implementation/lowering/dispatch result, not an independently
invented h/replay algorithm.

A separate synced CP/fusion ablation showed that when CP split and fused replay
are aligned, TileOps and FlashQLA replay can be essentially same-speed on the
tested rows; therefore any public FlashQLA gap must be attributed cautiously to
environment, call path, lowering, and dispatch unless matched generated-code
or profiler evidence is shown.

## 8. Claim Policy

Use scoped claims:

```text
On the validated BTHD inference prefill rows with B=1, DK=DV=128, chunk64,
fp16, and output_final_state=True, the TileOps path reaches ...
```

Avoid broad claims:

- "TileOps is faster than FlashQLA" without shape/timer/version scope.
- "The agent discovered CP split."
- "T.gemm_v1 vs T.gemm explains all performance."
- "Butterfly prefix is unnecessary in general."

Preferred wording:

```text
In our measured rows, the main long-replay schedule jump came from adopting a
FlashQLA-style CP-split schedule. Hierarchical prefix remains an interesting
research direction, but our tested full `[b | M]` transition representation was
too expensive for the production fused path; tuning `group_chunks` did not
rescue it.
```

Calibrated speedup language for same-schedule comparisons:

- `1.0-1.3x`: "comparable" or "similar performance".
- `1.3-2.0x`: "faster" or "notable speedup".
- `2.0-3.0x`: "significantly faster" or "substantial speedup".
- `3.0x+`: "much faster" or "large performance gap".

Apply this language only to public-environment full-op comparisons. Do not use
these words as replay algorithm attribution unless the comparison is controlled
for CP split, call path, lowering, dispatch, and timer.

For TileOps vs FlashQLA in the current `1.6-2.3x` range, use "faster",
"notably faster", or "significantly faster" only for the rows above `2x`.
Avoid "far superior", "dramatically faster", or "dominates" when the
high-level schedule family is the same and the gap reflects
compiler/implementation/dispatch differences rather than a different high-level
algorithm.

## 9. Figures And Tables

Recommended figures:

1. GDN chunkwise prefill dataflow.
2. AKO loop: hypothesis -> edit -> correctness -> benchmark -> lowering -> log.
3. Local tuning ladder: h recurrence / recompute / early fusion.
4. Blocksolve-A lower-triangle 4x16 Gram and Neumann-style solve.
5. Direct fused replay vs CP-split replay.
6. FlashQLA producer/consumer skeleton: producer/store + S/V/O consumers.
7. Hierarchical prefix negative result: dependency depth vs full transition
   representation cost.
8. Final component breakdown: A/KKT, CP preprocess, fused replay/output.
9. Attribution map: FLA as behavioral reference, FlashQLA as CP-split
   schedule/skeleton reference, human expert as Neumann/blocksolve reframing,
   AKO as gated implementation search, and TileOps as owned port, dispatch,
   validation, and productionization.

Recommended tables:

1. Final validated benchmark rows.
2. Candidate ladder with accepted/rejected decisions.
3. FlashQLA migration lessons: TMA-lost vs TMA-restored.
4. CP split parameter sweep.
5. Hierarchical prefix / full `[b | M]` rejection table.
6. Contribution/attribution table.

Rewrite warning for `tutorial_v2.md`:

- Section 9 must not be moved into the new article as-is.
- The scan/prefix formulation should appear only as an explored-and-rejected
  branch, not as the source of the final production algorithm.
- The final production schedule should be credited to Qwen FlashQLA's CP-split
  schedule, followed by TileOps' owned port, A producer, parameter tuning,
  dispatch, validation, and productionization work.

## 10. Claude Review Questions

Ask Claude to review v3 for:

1. Does this plan now have a stronger narrative than v2?
2. Are the two human-expert contributions placed at the right point in the
   article?
3. Is Qwen FlashQLA credited clearly and fairly?
4. Does the plan avoid both overclaiming independence and underclaiming our
   TileOps productionization work?
5. Is the agent capability boundary precise?
6. Is the three-level capability structure easier to follow than the old
   round-based tutorial structure?
7. Which sections of `tutorial_v2.md` should be moved, rewritten, or deleted?
8. What evidence is still blocking before public publication?
9. Which benchmark numbers should be treated as historical versus final?
10. What diagrams would most help a reader understand CP split?
