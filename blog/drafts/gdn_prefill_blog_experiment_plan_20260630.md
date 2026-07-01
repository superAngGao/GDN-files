# GDN Prefill Blog Experiment Plan

Date: 2026-06-30

Status: review draft. This plan is meant to define the missing experiment
ladder before rewriting the performance narrative in `tutorial_v3.md`.

## 1. Goal

The blog should not stitch together scattered historical numbers. It should
present a controlled kernel ladder:

```text
generic-A TileOps path
  -> local AKO improvements
  -> FlashQLA schedule alignment
  -> TileOps-owned FlashQLA-style schedule
  -> add blocked-inverse / Neumann-style A producer
  -> final shape-aware dispatch
```

The key ordering requirement is:

1. First show the TileOps path with the generic exact/KKT-style A producer.
2. Align that path as far as possible to FlashQLA's CP-split replay schedule.
3. Only after the schedule is understood and measured, swap in the human
   blocked-inverse / Neumann-style A producer and measure the incremental gain.

This keeps the blog honest:

- FlashQLA is credited for the CP-split replay/output schedule.
- Human expert work is credited for the blocked-inverse / Neumann-style A
  producer.
- AKO is credited for implementation, variant search, lowering inspection,
  correctness gates, benchmark gates, and dispatch tuning.

## 2. Narrative Alignment

The experiment ladder must match the article's explanatory order. The article
is not a pure chronology; it is a causal explanation of how the search space
changed. Therefore the benchmark ladder should be organized as:

```text
correct operator
  -> local generic-A AKO
  -> generic-A local wall
  -> FlashQLA schedule anchors
  -> TileOps-owned CP split with generic A
  -> add blocked-inverse / Neumann-style A producer
  -> final dispatch / reference comparison
```

This means the early performance roadmap in Section 0 should not show
blocked-inverse / Neumann-style producer numbers before the FlashQLA schedule
rows. The roadmap should first show the generic-A path aligning toward
FlashQLA, and only then show the increment from the human A-producer insight.

It also must not place public FlashQLA inside the controlled TileOps causal
ladder. Public FlashQLA changes the repository, TileLang version, lowering,
wrapper, runtime environment, and producer implementation at once. It is an
external anchor for schedule attribution and performance context, not a row in
the controlled generic-A TileOps descent.

### 2.1 Section-To-Experiment Map

| Article section | Narrative question | Experiment rows that support it | Allowed claim |
| --- | --- | --- | --- |
| Section 0: story / roadmap | What is the global performance path? | controlled TileOps ladder: V1 -> V2 full-op row if available -> V3 diagnostic -> V5 -> V6/V7; V4 external anchors shown separately | Performance improved by changing both schedule and A producer; rows must be labeled by evidence tier and lane. |
| Section 4: Level 1 | Did the agent make the operator measurable? | correctness harness, reference setup, benchmark gate | No performance claim needed. |
| Sections 5-7: Level 2 local AKO | What can the agent improve without external search-space changes? | V2 scale/store component rows | Local component improvements under the same operator contract. |
| Section 8: Level 2 wall | Why was local fusion insufficient? | V3 correct direct-fused row if available; otherwise dependency-chain diagram plus failed-candidate diagnostic | This is a boundary diagnostic, not a win. |
| Section 10: FlashQLA CP split | What external schedule changed the long-replay problem? | V4 external FlashQLA anchor and migrated FlashQLA skeleton; V5 `tileops_owned_cp_generic_a` as the controlled TileOps row | FlashQLA supplied the CP-split replay/output schedule; the controlled TileOps row still uses the generic A producer. |
| Section 11: Human math insight | What does the blocked-inverse / Neumann-style producer add? | V5 vs V6 controlled A-producer swap | Human insight improved the prepare-side A producer; keep the replay schedule fixed. |
| Section 12: productionization | What turns the kernel into a production path? | V7 shape sweep and dispatch metadata | Dispatch and metadata are part of the kernel evidence. |
| Section 14: evidence snapshot | What are the final scoped comparisons? | V7 vs FLA and FlashQLA on Qwen rows | Public-environment comparison only; avoid replay-algorithm attribution. |

### 2.2 Opening Roadmap Table Shape

After the experiments are collected, Section 0 should use two small tables, not
one overloaded table. The first table is the controlled TileOps full-op ladder.
The second table contains component evidence and external anchors.

Controlled TileOps full-op ladder:

| Order | Variant | Uses blocked-inverse A? | Schedule | Blog meaning | `64K/H16` latency | Perf vs FLA (%) | Perf vs FlashQLA, public-env only (%) |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | `generic_a_legacy` | no | legacy replay/output | starting point after correctness gates | TBD | TBD | N/A |
| 2 | `tileops_owned_cp_generic_a` | no | TileOps-owned CP split | FlashQLA schedule family ported with generic A producer | TBD | TBD | TBD |
| 3 | `tileops_owned_cp_blocked_inverse_a` | yes | same TileOps-owned CP split | human A-producer insight added last | TBD | TBD | TBD |
| 4 | `tileops_final_dispatch` | yes | shape-aware production dispatch | final candidate row | TBD | TBD | TBD |

Conditional full-op rows:

| Variant | Include only if | Blog meaning | `64K/H16` latency | Perf vs FLA (%) | Perf vs FlashQLA, public-env only (%) |
| --- | --- | --- | ---: | ---: | ---: |
| `generic_a_local_ako_best` | an end-to-end generic-A full-op path exists and passes correctness | agent-only local implementation search | TBD | TBD | N/A |
| `generic_a_direct_fused_correct` | it passes full-op correctness and has end-to-end latency | diagnostic wall, not an achievement | TBD | TBD | N/A |

Component evidence and external anchors:

| Row | Variant | Evidence lane | Latency | Blog meaning |
| --- | --- | --- | ---: | --- |
| A | `generic_a_scale_k` vs `generic_a_scale_v` | component | TBD | scale placement shows local AKO can improve a component |
| B | store-path variants | component | TBD | store-path diagnostics show arithmetic primitive matching was insufficient |
| C | `generic_a_direct_fused_failed` | negative diagnostic | N/A | failed fused candidates explain candidate rejection, not the recurrence claim |
| D | `flashqla_public_tl018` | external anchor | TBD | Qwen FlashQLA is the public CP-split schedule/performance reference |
| E | `flashqla_port_current_tl` | migration/lowering anchor | TBD | source-level schedule migration must be checked against generated code |

Rules for this table:

- Controlled rows before `tileops_owned_cp_blocked_inverse_a` are the generic-A
  route toward FlashQLA schedule alignment.
- Public FlashQLA is an external anchor, not a controlled TileOps row.
- `generic_a_local_ako_best` enters the controlled full-op ladder only if an
  end-to-end generic-A full-op path exists. Otherwise local AKO evidence stays
  in the component table.
- `generic_a_direct_fused_correct` enters the controlled full-op ladder only if
  it passes full-op correctness and has end-to-end latency. Otherwise keep only
  `generic_a_direct_fused_failed` as a negative diagnostic.
- The direct-fused row, if present, must be styled as a diagnostic wall, not as
  a performance milestone.
- Component rows should not be mixed into the full-op latency ladder unless an
  end-to-end variant is measured.
- `tileops_owned_cp_blocked_inverse_a` is the first row where the human
  blocked-inverse / Neumann-style insight is
  allowed to appear.
- `Perf vs FlashQLA, public-env only (%)` is optional context for controlled
  TileOps rows. It must not be used to support causal claims inside the
  controlled TileOps ladder; use `N/A` unless the row is explicitly being
  compared against the public FlashQLA anchor.
- The final dispatch row is allowed to include dispatch tuning, but it should
  not obscure the
  controlled V5 -> V6 A-producer comparison.
- If a row cannot be implemented cleanly, leave it as `not available` and
  explain why; do not replace it with a different kernel silently.

### 2.3 Why This Fits The Current Draft

The current `tutorial_v3.md` section order is compatible with this plan:

```text
Level 1 operator measurement
Level 2 local AKO
Level 2 wall
Level 3 FlashQLA schedule
Level 3 Neumann A producer
Productionization
Final evidence
```

The missing piece is not prose structure; it is a controlled experiment set.
Once the V1-V7 rows exist, the draft can replace the current mixed historical
milestone table with the Section 0 roadmap table above, and each later section
can pick only the rows that support its claim.

## 3. Common Benchmark Contract

All accepted rows in the main ladder should use the same measurement contract
unless explicitly marked as historical or component-only.

| Field | Contract |
| --- | --- |
| GPU | GPU4, H200 |
| Layout | BTHD |
| Batch | `B=1` |
| Main shape | `T=65536`, `H=16`, `DK=DV=128`, `chunk64`, `fp16` |
| Secondary shapes | `32K/H16`, `128K/H16`, `128K/H32`; optional `64K/H64` |
| Correctness reference | FLA 0.5.1 full op |
| FlashQLA reference | public FlashQLA TL0.1.8 environment |
| TileOps environment | current PR environment, TileLang 0.1.11 line unless changed |
| Timer | TileOps official benchmark infrastructure, CUPTI kernel-only timing via `BenchmarkBase.bench_kernel` or the current official equivalent |
| Timing metadata | warmup, repeat, trials, L2 flush policy, seed, clocks, GPU name |
| Output contract | `o + final_state` for inference prefill rows |

If an older or component-only row is used, it must be labeled as such and must
not be mixed into a final speedup table.

Input generation contract:

| Field | Requirement |
| --- | --- |
| tensor reuse | all variants and references for a row must use the same generated `q/k/v/g/beta/initial_state` tensors |
| tensor artifact | persist generated tensors or record a content hash and artifact path for each benchmark row |
| q/k/v distribution | distribution, scaling, dtype, and device must be recorded |
| g distribution | raw distribution, clamp/range, and chunk-local cumulative convention must be recorded |
| beta distribution | distribution/range and whether beta is pre-applied anywhere must be recorded |
| initial state | default zero state unless explicitly testing nonzero state; distribution recorded if nonzero |
| contiguity | tensors must be contiguous after layout conversion unless a row explicitly tests layout effects |
| layout helpers | BTHD/BHSD conversion helpers must be shared by all variants |

External anchor timing contract:

| Field | Requirement |
| --- | --- |
| timer type | record kernel-only CUPTI, CUDA-event fallback, or wrapper-level wall time |
| wrapper scope | record whether preprocessing, layout conversion, allocation, or output-final-state conversion is included |
| event filters | record profiler event filters used for component timing |
| profiler boundary | synchronize before timing/profiler step and document `profiler.step` placement |
| warmup/repeat/trials | record exact values |
| L2 flush | record whether flush is used and whether flush kernels are included/excluded |
| output contract | record whether `output_final_state` is included |
| environment | FlashQLA commit, docker/image, TileLang version, GPU, clocks if locked |
| schedule metadata | CP params, `block_DV`, and observable dispatch metadata if available |

## 4. A-Producer Definitions And ABI

### 4.1 Generic A Producer Definition

Reader-facing labels before the human math section should use "generic A
producer". The blocked-inverse / Neumann-style framing is introduced later in
the article, so earlier variants should not be named by a concept that did not
exist yet in the story.

For this experiment plan, a generic A producer means:

- it does not use the later blocked-inverse / Neumann-style 4x16 producer;
- it computes the same chunk-local A semantics expected by the downstream CP
  path;
- it may use a generic exact/KKT/triangular solve implementation;
- it must not reuse blocked-inverse-specific layout, staging, partial inverse
  blocks, or producer-side shortcuts unless the row is explicitly reclassified
  as an interface-change experiment;
- if the original generic implementation no longer exists, a resurrected
  experiment-only implementation must pass correctness and must be described as
  an experiment ladder row, not a production candidate.

### 4.2 A-Producer ABI For Controlled Swaps

The V5 -> V6 comparison is only meaningful if the A producer is the only
changed component. Therefore `tileops_owned_cp_generic_a` and
`tileops_owned_cp_blocked_inverse_a` must share an explicit A-producer ABI.

The ABI must fix:

| Field | Requirement |
| --- | --- |
| A semantics | same chunk-local correction matrix consumed by CP preprocess/replay |
| input tensors | identical `k`, chunk-local cumulative `g`, `beta`, shape, dtype, layout, and scale convention |
| output logical shape | must be filled with the exact A rank before running V5/V6, e.g. `[B, H, num_chunks, C, C]` or the exact production flattened equivalent |
| output physical layout | exact strides, chunk-major ordering, head/time ordering, and BTHD/BHSD mapping must be recorded |
| triangular convention | lower vs strict-lower, diagonal convention, and whether unused upper-triangle values are zero/ignored/undefined |
| gate convention | whether the input is raw `g` or chunk-local cumulative `g_cum`; exponent/sign convention must be recorded |
| beta convention | whether `beta` is folded into A or applied downstream must be identical |
| output dtype | identical dtype and accumulation-to-store policy |
| materialization | either both materialize A in the same global format, or both use the same fused handoff contract |
| staging/layout conversion | no extra layout conversion may be introduced in only one side of the comparison |
| downstream CP preprocess API | function signature, tensor order, and expected contiguity must be identical |
| downstream replay/output API | fused replay/output sees identical tensor contracts |

Allowed difference:

```text
A = producer(k, g_cum, beta)
```

The implementation of `producer` may differ. The downstream contract must not.
If the blocked-inverse / Neumann-style producer also changes A layout or
staging in a way that benefits the replay pipeline, the blog must report that
as "A producer + interface change", not as a pure mathematical gain.

A-producer equivalence checks:

| Case | Required check |
| --- | --- |
| A materialized by both producers | decode both outputs into the same canonical logical A layout before comparison; record triangular convention, comparison dtype, tolerance, max abs, and max rel |
| A fused or abstracted by either producer | compare downstream `w/u` or CP-preprocess outputs under identical inputs; record comparison dtype, tolerance, max abs, and max rel |
| CP preprocess uses temporary buffers/cache | verify both V5 and V6 expose the same downstream tensor contract; any one-sided cache/temp-buffer change must be classified as interface change |
| full-op validation | still required, but not sufficient by itself to claim a pure A-producer swap |

## 5. Evidence Rules

Every row in the main ladder needs:

| Requirement | Meaning |
| --- | --- |
| `variant_id` | stable name used in code, result JSONL, and blog table |
| code pointer | file/function/commit or patch reference for the exact kernel |
| correctness result | max abs/rel diff for `o` and `final_state` against FLA, or a clear reason why the row is component-only |
| latency result | median/min policy recorded by the official benchmark helper |
| component breakdown | at least prepare A, CP preprocess, fused replay/output, other, when applicable |
| dispatch metadata | CP segment count, `max_local_chunks`, `block_DV`, layout, dtype, seed |
| generated-code evidence | CUDA/PTX/SASS or lowering hash for claims about TMA/WGMMA/lowering parity |

Rows that fail correctness can appear only as negative diagnostics, not as
performance milestones.

Rows with different repositories, TileLang versions, lowering stacks, wrappers,
or runtime environments can appear as external anchors, but they must not be
drawn as controlled causal steps.

## 6. Variant Ladder

### V0. References

Purpose: establish the reference surfaces before changing TileOps.

| Variant | Description | Expected use |
| --- | --- | --- |
| `ref_fla_051` | FLA 0.5.1 `chunk_gated_delta_rule`, same BTHD inputs through wrapper | correctness oracle and FLA performance comparison |
| `ref_flashqla_tl018` | public Qwen FlashQLA TL0.1.8 path | schedule/performance reference |
| `tileops_current_pr` | current production candidate | final comparison anchor, not used as a causal explanation by itself |

Artifacts:

- full-op latency on Qwen four-row benchmark set;
- correctness against FLA where applicable;
- component breakdown for TileOps and FlashQLA if the profiler can split them
  under the same timer semantics.

### V1. Generic-A Baseline: Generic A Producer + Legacy Replay

Purpose: define the starting TileOps path with the generic exact/KKT-style A
producer, before FlashQLA CP split and before the later human
blocked-inverse / Neumann-style reframing.

Shape:

```text
chunk_local_cumsum
  -> generic exact/KKT-style A producer
  -> recompute w/u
  -> h replay
  -> output_o
```

Rules:

- This variant uses the generic exact/KKT-style A producer.
- It must not use the later blocked-inverse / Neumann-style 4x16 producer or
  its layout/staging shortcuts.
- If the original generic implementation no longer exists, recreate it as an
  experiment-only kernel or harness. Do not silently substitute current
  production A.
- This row is the baseline for the generic-A story.

Measurements:

- full-op `64K/H16`;
- component breakdown;
- correctness against FLA.

### V2. Generic-A Local AKO: Scale Placement And Store Path

Purpose: show what agentic local tuning can do before external schedule/math
input changes the search space.

Subvariants:

| Variant | Change | Measurement |
| --- | --- | --- |
| `generic_a_scale_k` | scale on K path | h replay component |
| `generic_a_scale_v` | scale on V path | h replay component |
| `generic_a_store_baseline` | baseline recompute output path | recompute component |
| `generic_a_store_shared` | shared output staging | recompute component |
| `generic_a_store_swizzled_async` | store-friendly swizzled/async path | recompute component |

Blog role:

- These are component wins.
- They should explain local AKO capability, not claim the full long-context
  problem is solved.

### V3. Generic-A Boundary Diagnostic: Direct Fusion Without CP Split

Purpose: demonstrate why fusion alone was not the final answer.

Shape:

```text
generic A producer
  -> direct fused replay/output
  -> o + final_state
```

Subvariants:

| Variant | Meaning | Allowed claim |
| --- | --- | --- |
| `generic_a_direct_fused_correct` | correct direct fused no-CP kernel, measured end to end | fusion reduced materialization but did not shorten the replay dependency enough |
| `generic_a_direct_fused_failed` | compile/runtime/correctness failure | this candidate was not viable; it does not by itself prove the recurrence argument |

Rules:

- Use the generic exact/KKT-style A producer.
- No CP split.
- The main performance ladder should use `generic_a_direct_fused_correct` only
  if it passes correctness.
- If the all-fused kernel is too large to compile or unstable, record
  `generic_a_direct_fused_failed` as a candidate failure, not as evidence that
  fusion cannot shorten recurrence.
- To support the "fusion alone did not shorten replay" claim, include either a
  correct-but-slow direct fused row or explicit algorithm/lowering evidence
  showing that the long chunk-to-chunk dependency chain is unchanged.

Blog role:

- This is a boundary diagnostic, not an achievement.
- It should be phrased as: "less materialization did not shorten the replay
  dependency."

### V4. FlashQLA Schedule Anchors

Purpose: establish the external schedule reference and migration/lowering
anchor. These rows are not controlled TileOps causal steps.

Shape:

```text
FlashQLA-style CP preprocess / corrected segment starts
  -> fused replay/output over CP segments
```

Required rows:

| Variant | Description |
| --- | --- |
| `flashqla_public_tl018` | external public FlashQLA reference, as released; not a controlled TileOps ladder row |
| `flashqla_port_current_tl` | FlashQLA skeleton migrated to current TileLang, no TileOps A producer substitution; migration/lowering anchor |

Key question:

```text
What schedule and generated-code behavior should the TileOps-owned CP split
try to match?
```

This is an anchor section, not the controlled schedule-delta row.

Measurements:

- public FlashQLA full-op `64K/H16` and Qwen four-row set if available;
- migrated FlashQLA skeleton latency if implemented;
- component breakdown when event boundaries are reliable;
- generated-code evidence for the migrated FlashQLA skeleton if we make any
  TMA/WGMMA claim.

### V5. TileOps-Owned CP Split With Generic A Producer

Purpose: isolate the CP-split schedule contribution inside the TileOps-owned
code shape while still using the generic exact/KKT-style A producer.

Shape:

```text
generic exact/KKT-style A producer
  -> TileOps-owned CP preprocess
  -> TileOps-owned fused replay/output
  -> TileOps dispatch metadata
```

Required row:

| Variant | Description |
| --- | --- |
| `tileops_owned_cp_generic_a` | TileOps-owned BTHD CP-split path with generic A producer |

Rules:

- Use the generic-A producer definition in Section 4.
- Do not use the later blocked-inverse / Neumann-style A producer or its
  layout/staging shortcuts.
- CP parameters must be recorded. If they differ from the FlashQLA anchor,
  report the difference as TileOps implementation/dispatch behavior, not as a
  new high-level replay algorithm.

Blog role:

- Shows engineering ownership and production code shape.
- Should not claim a new replay algorithm.

### V6. Same TileOps-Owned CP Row With Blocked-Inverse / Neumann-Style A Producer

Purpose: measure the human mathematical insight as a controlled delta.

Shape:

```text
blocked-inverse / Neumann-style A producer
  -> same TileOps-owned CP preprocess
  -> same TileOps-owned fused replay/output
  -> o + final_state
```

Critical control:

```text
V6 should change only the A producer relative to V5.
```

The A-producer ABI in Section 4 is mandatory for this comparison. If V6 also
changes the A tensor layout, materialization strategy, staging, or downstream
handoff, the comparison must be renamed and interpreted as an interface change
rather than a pure Neumann / blocked-inverse producer swap.

Required comparisons:

| Compare | Meaning |
| --- | --- |
| `tileops_owned_cp_generic_a` vs `tileops_owned_cp_blocked_inverse_a` | full-op gain from A producer change |
| generic A component vs blocked-inverse / Neumann-style A component | component gain from math reframing |
| correctness vs FLA | ensure the new producer is not changing semantics |

Blog role:

- This is where the human insight appears.
- It should come after the FlashQLA schedule ladder in the performance story,
  even if historically the Neumann work happened earlier.

### V7. Dispatch And Shape Sweep

Purpose: turn the best kernel into a production candidate.

Sweep dimensions:

| Dimension | Examples |
| --- | --- |
| CP segment policy | `max_local_chunks` values used by dispatch |
| `block_DV` | full `DV=128` vs tiled variants if supported |
| shape | `32K/H16`, `64K/H16`, `128K/H16`, `128K/H32`, optional `64K/H64` |
| layout | BTHD only for main blog claims |

Required output:

- final candidate table vs FLA and FlashQLA;
- dispatch metadata for every row;
- correctness table for every row.

## 7. Proposed Blog Tables After Experiments

### Table 1. Controlled TileOps Full-Op Ladder

This is the table that should appear near the beginning of the article.

Relative performance is reported as `reference_latency / variant_latency *
100%`. `100%` means parity with the reference; values above `100%` mean the
variant has higher throughput than that reference.

| Row | Variant | A producer | Schedule | Evidence tier | `64K/H16` latency | Perf vs FLA (%) | Perf vs FlashQLA, public-env only (%) |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | `generic_a_legacy` | generic exact/KKT | legacy replay/output | Tier 1 if rerun | TBD | TBD | N/A |
| 2 | `tileops_owned_cp_generic_a` | generic exact/KKT | TileOps-owned CP split | Tier 1 if rerun | TBD | TBD | TBD |
| 3 | `tileops_owned_cp_blocked_inverse_a` | blocked inverse / Neumann-style | same TileOps-owned CP split | Tier 1 if V5 -> V6 ABI is fixed | TBD | TBD | TBD |
| 4 | `tileops_final_dispatch` | final production A producer | shape-aware production dispatch | final candidate row | TBD | TBD | TBD |

Conditional full-op rows for Table 1:

| Variant | Include only if | Evidence tier | `64K/H16` latency | Perf vs FLA (%) | Perf vs FlashQLA, public-env only (%) |
| --- | --- | --- | ---: | ---: | ---: |
| `generic_a_local_ako_best` | an end-to-end generic-A full-op path exists and passes correctness | Tier 1 only if rerun end to end | TBD | TBD | N/A |
| `generic_a_direct_fused_correct` | it passes full-op correctness and has end-to-end latency | diagnostic, not a win | TBD | TBD | N/A |

### Table 2. Component Evidence And External Anchors

These rows explain local AKO and external references. They should not be drawn
as controlled full-op causal steps.

| Row | Variant | Lane | Latency | Evidence tier | Blog meaning |
| --- | --- | --- | ---: | --- | --- |
| A | `generic_a_scale_k` vs `generic_a_scale_v` | component | TBD | component diagnostic | local AKO can improve replay data movement |
| B | `generic_a_store_*` variants | component | TBD | component diagnostic | store path, not only GEMM primitive, mattered |
| C | `generic_a_direct_fused_failed` | negative diagnostic | N/A | failure diagnostic | failed fused candidates cannot prove the recurrence claim by themselves |
| D | `flashqla_public_tl018` | external anchor | TBD | external reference | Qwen FlashQLA is the public CP-split schedule/performance reference |
| E | `flashqla_port_current_tl` | migration/lowering anchor | TBD | generated-code evidence if claimed | source-equivalent migration still needs lowering validation |

### Table 3. Add Human Math Insight

| Row | Variant | Changed component | A-producer latency | Evidence tier | `64K/H16` full-op latency | Perf vs FLA (%) | Perf vs FlashQLA, public-env only (%) |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| 1 | `tileops_owned_cp_generic_a` | baseline for this comparison | TBD | Tier 1 if rerun | TBD | TBD | TBD |
| 2 | `tileops_owned_cp_blocked_inverse_a` | only A producer changes under the Section 4 ABI | TBD | Tier 1 if rerun | TBD | TBD | TBD |

### Table 4. Final Candidate Vs References

This is the public-facing table after all gates pass.

Caption requirement:

```text
TileOps/FlashQLA throughput is a public-environment comparison only, not a
controlled same-lowering replay attribution comparison. Table metadata must
state whether wrapper preprocessing, layout conversion, allocation, and
output-final-state conversion are included or excluded for each reference.
```

| Case | FLA 0.5.1 latency | FlashQLA TL0.1.8 latency | TileOps final latency | TileOps/FLA throughput (%) | TileOps/FlashQLA throughput, public-env only (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `32K/H16` | TBD | TBD | TBD | TBD | TBD |
| `64K/H16` | TBD | TBD | TBD | TBD | TBD |
| `128K/H16` | TBD | TBD | TBD | TBD | TBD |
| `128K/H32` | TBD | TBD | TBD | TBD | TBD |

## 8. Implementation Plan

### Step 1. Audit Existing Variant Availability

Find whether each variant already exists:

- generic exact/KKT A producer;
- current Neumann / blocked-inverse A producer;
- legacy replay/output path;
- direct fused no-CP diagnostic;
- FlashQLA migrated skeleton;
- TileOps-owned CP-split path;
- dispatch sweep hooks.

Output:

```text
experiments/gated_deltanet_prefill_blog_ladder/variant_inventory.md
```

### Step 2. Add Experiment-Only Harness

Create a harness that can select variants without polluting production
dispatch. Prefer explicit Python variant names over hidden environment
variables.

Suggested path:

```text
experiments/gated_deltanet_prefill_blog_ladder/
  README.md
  run_ladder.py
  variants.py
  collect_component_breakdown.py
  results/
  generated_code/
  summaries/
```

Variant selection should look like:

```bash
python experiments/gated_deltanet_prefill_blog_ladder/run_ladder.py \
  --variant tileops_owned_cp_generic_a \
  --seq-len 65536 --heads 16 \
  --output results/ladder_64k_h16.jsonl
```

### Step 3. Correctness Gate

For every full-op variant:

```text
compare variant(q,k,v,g,beta) against FLA 0.5.1
record o max abs / max rel
record final_state max abs / max rel
```

Correctness setup must be identical across rows:

| Field | Requirement |
| --- | --- |
| initial state | fixed and recorded; default zero state unless the row explicitly tests nonzero state |
| FLA output | confirm whether FLA 0.5.1 returns `final_state`; if not, use a documented reference path for final-state validation |
| scale | fixed, recorded, and shared across TileOps/FLA wrappers; default `scale=1.0` for the GDN prefill rows unless the final op contract says otherwise |
| layout conversion | exact BTHD/BHSD conversion helpers shared by all variants |
| dtype policy | same input dtype and output comparison dtype for every row |
| random seed | fixed and recorded |

Suggested tolerance for current fp16 rows:

```text
atol=rtol=5e-2
```

Any stricter tolerance that passes consistently should be recorded, but the
blog should not silently change tolerance row by row.

### Step 4. Benchmark Gate

Use the TileOps official benchmark timer. Record:

- warmup;
- repeat;
- trials;
- CUPTI vs fallback;
- L2 flush policy;
- seed;
- GPU;
- clocks if locked;
- docker image / wheel versions;
- TileOps commit;
- FlashQLA commit/environment.

### Step 5. Component Breakdown

For `64K/H16`, collect at least:

| Component | Required for |
| --- | --- |
| chunk local cumsum | all variants |
| A producer | generic vs Neumann comparison |
| recompute w/u or equivalent | legacy/local-AKO path |
| CP preprocess / corrected starts | CP-split variants |
| fused replay/output | CP-split variants |
| output only | legacy split path |

This is needed so the blog can say which part got faster without guessing.

### Step 6. Lowering Evidence

Only collect generated-code evidence where the blog makes lowering claims:

- FlashQLA TL0.1.8 public skeleton;
- migrated FlashQLA skeleton under current TileLang;
- TileOps-owned CP-split fused replay/output;
- A producer if we claim a specific GEMM/TMA/WGMMA behavior.

Do not put TMA/WGMMA language in the blog unless this evidence is archived.

## 9. Review Questions

Before implementing the harness, review these points:

1. Is `64K/H16` the right primary ladder row, with the Qwen four-row set used
   only for final candidate comparison?
2. What exactly should count as "no Neumann" for the A producer?
3. Do we need to resurrect an old generic A producer, or is there a current
   exact/KKT implementation we can select directly?
4. Can we produce a correct direct fused no-CP row, or only a failure
   diagnostic?
5. What metadata is required for the public FlashQLA anchor so readers do not
   mistake it for a controlled TileOps step?
6. Is the Section 4 A-producer ABI strict enough to isolate the V5 -> V6
   blocked-inverse / Neumann-style A-producer delta?
7. Which rows need generated-code evidence for publication, rather than just
   benchmark/correctness evidence?

## 10. Non-Goals

- Do not tune new algorithms while collecting the ladder. First make the
  evidence controlled.
- Do not use historical rows as final publication claims.
- Do not infer replay algorithm superiority from full-op TileOps vs FlashQLA
  speedups.
- Do not merge experiment-only switches into production dispatch unless they
  are separately justified.
