# AI-Assisted TileLang Kernel Optimization: Gated DeltaNet Prefill

This document collects the current planning notes for a tutorial-style report
about optimizing a Gated DeltaNet prefill kernel in TileOPs. It is not the final
blog post yet. Its purpose is to pin down the narrative, algorithm map,
optimization story, and missing experiments before drafting the public version.

## Proposed Positioning

The tutorial should not be positioned as only a Gated DeltaNet performance
report. The stronger framing is:

> A case study in AI-assisted TileLang kernel optimization: starting from a
> correct, FLA/Qwen-compatible Gated DeltaNet prefill kernel, we combine
> profiling, agent-driven AKO search, human mathematical insight, and careful
> production guards to close most of the gap to a fast moving mainstream FLA
> baseline, while keeping the optimization process reproducible and honest
> about version-sensitive results.

The intended audience is the TileOps/TileLang developer community. The goal is
to teach a kernel optimization workflow, not merely to report final numbers.

The central message:

> AI does not replace the kernel engineer. It amplifies local search,
> instrumentation, bookkeeping, and hypothesis pruning, while human judgment
> remains crucial for algorithmic reformulations and production decisions.

Current correction from the production PR rerun:

- Earlier planning notes briefly treated a source-baseline FLA 0.5.1 run as the
  final result. That was a stale intermediate measurement.
- The PR-head rerun on `067edc7e0dc2da3220468f9ed5321a6d6c14406e` uses
  `flash-linear-attention==0.5.1`, BTHD layout, `output_final_state=True`,
  H200, `B=1`, `DK=DV=128`, `chunk64`, and `fp16`.
- In that rerun, TileOps is faster on H16 S32K/S64K/S128K and H32 S128K. The
  article may claim this scoped result, but should still avoid broad dominance
  claims beyond these measured shapes and environment.

## What Is Assumed Before The Tutorial Starts

The tutorial should treat interface and layout alignment as an engineering
precondition, not as the main optimization story.

The starting point should already satisfy:

- FLA/Qwen-compatible `BTHD` input layout as the primary serving path.
- Optional TileOps head-major path, named consistently as `BHTD`.
- A dedicated prefill interface:
  - inputs: `q, k, v, g, beta`
  - outputs: `o, final_state`
- No training-forward-only artifacts exposed from the public prefill op.
- Fair benchmarking against FLA with `output_final_state=True` when supported.

The tutorial can briefly mention why this matters, but the core story is how we
optimize the kernel after the engineering target is clear.

## Algorithm Decomposition

Before showing optimization steps or benchmark ladders, the tutorial should give
readers a compact map of the Gated DeltaNet prefill algorithm. The goal is not
to rederive the paper, but to explain all intermediate quantities used later.
The final tutorial should pair this section with a visual three-stage pipeline
figure, because the notation is dense.

Suggested references:

- Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule*, ICLR
  2025 / arXiv 2412.06464.
- FLA `chunk_gated_delta_rule` implementation as the practical baseline.

Notation conventions to clarify in the final tutorial:

- We use the implementation convention where each `(batch, head)` stream owns a
  recurrent state `h` with shape `[K, V]`.
- `@` denotes matrix multiplication.
- `*` denotes elementwise multiplication and broadcasts over the missing axes.
- `tril(x, -1)` denotes the strictly lower-triangular part.
- Vectors such as `k_t`, `q_t`, and `v_new_t` should be treated according to
  the local implementation convention; the tutorial should state whether a
  formula is mathematical shorthand or exact code orientation.

### Inputs And Outputs

Using the FLA/Qwen-style `BTHD` convention:

```text
Inputs:
  q, k: [B, T, H, K]
  v:    [B, T, H, V]
  g:    [B, T, H]      log-space decay gate
  beta: [B, T, H]      delta-rule update gate

Outputs:
  o:           [B, T, H, V]
  final_state: [B, H, K, V]
```

The implementation stores the recurrent state as a `[K, V]` matrix per
`(batch, head)` stream.

### Token-Level Intuition

At token level, the gated delta rule can be viewed as a decay/update recurrence:

```text
alpha_t = exp(g_t)

read_t  = h_{t-1}^T k_t
v_new_t = beta_t * (v_t - alpha_t * read_t)

h_t = alpha_t * h_{t-1} + k_t v_new_t^T

o_t = alpha_t * h_{t-1}^T q_t + (q_t^T k_t) v_new_t
```

The exact left/right orientation can follow the implementation convention. The
important point is that each token both reads from and writes to a recurrent
state.

### Chunkwise Prefill Pipeline

For prefill, we process a long sequence in chunks. The practical kernel is
decomposed into three stages.

#### Stage 1: Prepare WY / Chunk-Local Triangular Solve

Within each chunk, `v_new[i]` depends causally on previous tokens in the same
chunk. The prepare stage removes this chunk-local dependency by building a WY
or equivalent representation:

```text
w_c, u_c = Prepare(k_c, v_c, g_c, beta_c)
```

The core mathematical object is a chunk-local lower-triangular system, roughly:

```text
A_c = I + tril(beta_i * exp(g_i - g_j) * k_i^T k_j, -1)
```

The implementation choice here matters:

- TileOps current path uses a fused Neumann-style explicit inverse/prepare.
- FLA uses an optimized blocked triangular-solve style pipeline.
- A naive forward solve is useful as a diagnostic, but it is not a fair model
  of FLA's implementation.

This stage produces `w` and `u`, which are consumed by the recurrence stage.

#### Stage 2: H Recurrence Across Chunks

After prepare, each chunk still updates the cross-chunk recurrent state:

```text
v_new_c = u_c - ((w_c * exp(g_c + g_last)[..., None]) @ h_c)
# shapes: [C,V] = [C,V] - (([C,K] * [C,1]) @ [K,V])

h_{c+1} = exp(g_last) * h_c
          + (k_c * exp(g_last - g_c))^T @ v_new_c
```

This stage produces:

- boundary states for each chunk, used by output
- `v_new`, the corrected values used by output
- `final_state`, the last recurrent state

Two major optimization opportunities live here:

- local algebraic rewrite of the h update
- associative prefix scan over chunk transitions

The scan view is:

```text
h_{c+1} = A_c h_c + b_c
```

The affine transition composition is associative, which enables a prefix-scan
formulation for long sequences.

#### Stage 3: Output

The output stage combines recurrent contribution from previous chunks with
causal intra-chunk contribution:

```text
o_c = exp(g_c) * q_c @ h_c
      + causal((q_c k_c^T) * exp(g_i - g_j)) @ v_new_c
```

This stage consumes:

- `q, k, g`
- boundary states from the recurrence stage
- `v_new`

It returns `o`; the public op also returns `final_state`.

### Optimization Map

This decomposition gives the rest of the tutorial a clear map:

- prepare optimizations target the chunk-local triangular system.
- local h-recurrence optimizations target the state update kernel.
- prefix scan optimizations target the serial dependency across chunks.
- output optimizations target the final combination of boundary state and
  chunk-local causal terms.

## Main Optimization Story

The main tutorial should present three core optimization loops.

### 1. Agent-Discovered Algebraic Rewrite In H Recurrence

The AKO loop found an equivalent h-update rewrite:

```text
(k * exp(g_last - g_i))^T @ v_new
```

can be computed as:

```text
k^T @ (v_new * exp(g_last - g_i))
```

This keeps the math unchanged but moves the scaling from `k` to `v_new`.

Why it matters:

- avoids a separate `k_scaled` shared-memory buffer
- reduces shared-memory footprint
- reduces register pressure
- improves h recurrence latency

This is the cleanest example of an AI-assisted local optimization: the agent
used profiling and repeated small hypotheses to find a non-obvious but local
algebraic improvement.

### 2. Human-Driven Prepare Inverse Analysis

The prepare stage is a chunk-local triangular-system problem. A naive forward
solve in TileLang was much slower, but that result must not be interpreted as
"forward solve is bad." It only means the naive implementation is not a fair
model of FLA's optimized blocked solve.

The tutorial should compare:

- TileOps fused Neumann-style prepare
- naive TileLang forward solve
- FLA-style pipeline:
  - `chunk_scaled_dot_kkt_fwd`
  - `solve_tril`
  - `recompute_w_u_fwd`

The story here is human mathematical judgment:

- identify the true mathematical object
- avoid overgeneralizing from a poor implementation of an algorithm
- compare against the right decomposition
- decide whether replacing the current prepare path is justified

Current conclusion:

- naive forward solve should not be used
- FLA-style solve is worth understanding, but current TileOps fused prepare is
  competitive at H>=16, DK=DV=128 workloads in our current benchmark set
- future fair work would need a blocked solve or associative triangular-solve
  design, not a naive serial solve

### 3. Human-Identified Prefix Scan For Chunk Recurrence

The recurrence across chunks can be represented as affine transitions:

```text
h_{c+1} = A_c h_c + b_c
```

Transition composition is associative:

```text
(A_2, b_2) ∘ (A_1, b_1)
  = (A_2 A_1, A_2 b_1 + b_2)
```

This enables a prefix-scan path for long sequences. The production version uses
a narrow guard rather than enabling scan everywhere.

Current accepted scan guard:

```text
layout == BTHD
batch == 1
heads == 16
seq_len >= 65536
chunk_size == 64
dim_k == dim_v == 128
dtype == float16
seq_len % (chunk_size * 64) == 0
```

Each guard should be justified in the tutorial:

| Condition | Reason |
| --- | --- |
| `layout == BTHD` | The scan helper kernels were implemented for the FLA/Qwen-compatible serving layout. |
| `batch == 1` | The current target is single-request long-prefill serving; batch effects need separate validation. |
| `heads == 16` | H16 long sequences benefit from the current scan path; H32 dense-transition scan lost in earlier isolated sweeps, so production should not force that path. |
| `seq_len >= 65536` | At S32K the scan overhead dominates or is not worth enabling. |
| `chunk_size == 64` | The current summary/replay grouping and benchmarks are tuned for chunk64. |
| `dim_k == dim_v == 128` | The dense transition shape and resource use assume 128x128 states. |
| `dtype == float16` | This is the validated fast path and matches the target Qwen-like serving dtype. |
| `seq_len % (chunk_size * 64) == 0` | Avoids unvalidated scan tail handling for the current production guard. |

Why narrow guards matter:

- H16 long sequences benefit.
- H16 S32K should remain on baseline h recurrence.
- H32 S128K does not benefit from the current dense-transition scan design.
- production kernels should use shape-specific dispatch when algorithmic
  overheads are shape-dependent.

## Supporting Engineering Lessons

These are not the three core mathematical optimizations, but they make the
tutorial useful for TileLang developers.

### Multi-Agent Review Is Part Of The Workflow

After each meaningful milestone, the draft or experiment summary should go
through a Claude review pass before we build the next layer on top of it.

Suggested review checkpoints:

- after each major experiment summary is added
- after each optimization section is drafted
- after the full tutorial skeleton becomes readable end to end
- before publishing the public repo

Each review pass should be archived with:

- the exact document or section reviewed
- the reviewer feedback
- what we changed in response
- what we intentionally left unchanged and why

This is part of the tutorial's meta-story: AI-assisted kernel work benefits
from separation of roles. Codex can run experiments, edit code, and maintain
the working draft; Claude can act as an independent reader that checks
structure, clarity, missing evidence, and narrative overclaiming.

### Stable Measurement Comes First

AI-assisted optimization needs stable feedback. The tutorial should emphasize:

- lock GPU clocks
- compare against the same layout
- use `output_final_state=True` for FLA when comparing prefill
- separate full-op benchmarks from component benchmarks
- use profiler/NCU to locate the real bottleneck
- keep an AKO decision log

Measurement rigor to define before publication:

- number of warmup iterations, repeats, and trials per benchmark
- whether latency is reported as median trial mean
- allowed run-to-run variance for calling a result stable
- what to do if a reproduced result contradicts the planned narrative

Provisional measurement policy for the first experiment pass:

- use 10 warmup iterations, 100 repeat iterations, and 3 trials per config when
  runtime allows
- report the median of trial means
- treat trial-to-trial standard deviation below 5% of the median as stable
- if a long experiment is too expensive, explicitly record the reduced
  warmup/repeat/trial counts next to the result

The answer to the last point should be explicit: the tutorial should report the
contradiction, narrow the claim, or move that optimization to a negative-result
section rather than forcing the story.

### Shape-Aware Dispatch Is Part Of Production

The fastest kernel family can change by shape. The production kernel should not
pretend one configuration wins everywhere.

Examples:

- small-stream paths need different chunk/thread choices
- scan should only be enabled at long H16 sequences
- H32 dense-transition scan lost in earlier sweeps despite being
  algorithmically attractive; this is separate from the latest full-op
  H32/S128K production benchmark, where the guarded path beats FLA.

### Negative Results Are Part Of The Method

The tutorial should include a compact "what did not work" section:

- BV8 increased CTA count but worsened resource/WGMMA shape.
- broad BV32/BV64 changes were not universally better.
- fused h+output increased recomputation and serialization.
- sparse checkpointing saved some state bytes but duplicated h-update work.
- naive forward solve was not a fair FLA-style solve.
- H32 dense scan group-size sweep did not recover performance.

This is important because it teaches developers how to prune the search space,
not just how to copy final code.

## Experiments To Add Before Writing The Final Tutorial

The next step is not broad new AKO. It is to fill the evidence gaps needed for a
clear tutorial.

To avoid scope creep, split experiments into a minimum viable tutorial set and
supplementary material.

Blocking for a first draft:

1. naive-to-optimized ladder
2. algebraic rewrite ablation
3. fair FLA comparison

Strongly useful, but can be appendix/supplementary if schedule gets tight:

1. component breakdown
2. prepare inverse comparison
3. prefix scan guard sweep

### 1. Naive-To-Optimized Ladder

Create a table with the same workload across key milestones:

- naive dedicated prefill
- resource-tuned baseline
- algebraic h-recurrence rewrite
- prepare inverse / fused Neumann prepare state
- scan dispatch
- FLA baseline

Recommended workloads:

```text
B=1, H=16, S=32K,  DK=DV=128, fp16
B=1, H=16, S=64K,  DK=DV=128, fp16
B=1, H=16, S=128K, DK=DV=128, fp16
B=1, H=32, S=128K, DK=DV=128, fp16
```

### 2. Component Breakdown

For key versions, measure:

- chunk-local cumsum
- prepare
- h recurrence
- output
- full op

This should show which stage each optimization affects.

### 3. Algebraic Rewrite Ablation

Compare before/after:

- scale `k`
- scale `v_new`

Metrics:

- full latency
- h recurrence latency
- registers/thread
- shared memory
- correctness max error

Suggested shapes:

```text
B=1, H=16, S=32K,  DK=DV=128, fp16
B=1, H=16, S=128K, DK=DV=128, fp16
```

### 4. Prepare Inverse Comparison

Compare:

- TileOps fused Neumann prepare
- naive TileLang forward solve
- FLA-style prepare pipeline

Suggested sweep:

```text
S=32K, chunk64, DK=DV=128, fp16
H in {4, 16, 32, 64, 96}
```

Reason for the head counts:

- `H=4`: small-head underfill baseline
- `H=16`: primary tutorial and Qwen-like target
- `H=32`: secondary larger-head target
- `H=64`: high-stream regime
- `H=96`: wider edge case for prepare scaling

Report both latency and numerical agreement.

### 5. Prefix Scan Guard Sweep

Measure:

```text
H=16, S=32K:  baseline should win or scan should stay disabled
H=16, S=64K:  scan should win
H=16, S=128K: scan should win more clearly
H=32, S=128K: current dense scan should lose
```

This supports the production scan guard.

### 6. Fair FLA Comparison

Final benchmark table:

- TileOps BTHD vs FLA BTHD
- FLA with `output_final_state=True`
- same dtype
- same GPU clocks
- same timing infrastructure

Suggested rows:

```text
S=32K,  H=16
S=64K,  H=16
S=128K, H=16
S=128K, H=32
```

## Draft Article Structure

Default v1 structure:

1. Introduction: why AI-assisted kernel optimization matters
2. Prerequisites: TileLang basics, Gated DeltaNet, BTHD prefill, hardware setup
3. Algorithm decomposition
4. Baseline profiling and optimization workflow
5. Ladder: naive to optimized
6. Optimization 1: agent-discovered algebraic rewrite, with "what we tried"
7. Optimization 2: human-driven prepare analysis, with "what we tried"
8. Optimization 3: human-identified prefix scan, with "what we tried"
9. Lessons for TileLang kernel developers

Appendices:

- Appendix A: production notes, including dispatch, tests, guards, and PR hygiene
- Appendix B: component breakdown if completed
- Appendix C: prepare inverse deep dive if completed

This embeds negative results inside each optimization story, which should make
the search process feel natural rather than bolted on at the end.

## Visual And Code Assets

The tutorial should include visual aids and code snippets, not just benchmark
tables.

Critical v1 figures:

- three-stage GDN prefill pipeline
- optimization ladder from naive to final
- before/after h-recurrence resource usage

Strong v1 figure if time permits:

- scan guard / sequence-length threshold chart

V2 or appendix figures:

- BTHD vs BHTD memory-layout diagram
- profiler/NCU bottleneck screenshot or summarized trace
- prepare inverse comparison chart

Minimum code snippets:

- pseudo-TileLang before/after for the h-recurrence algebraic rewrite
- benchmark harness snippet showing fair FLA comparison
- scan guard dispatch snippet
- component profiler integration snippet showing how to isolate h recurrence

## Experiment Execution Order

Suggested execution plan:

Current Week 1 status:

- Fair FLA comparison has been rerun against `flash-linear-attention==0.5.1`
  for H16 S32K/64K/128K and H32 S128K on PR head `067edc7e`. TileOps is faster
  on all four rows in that run.
- Public-op torch profiler breakdown has been run for H16 S32K and H16 S128K.
- BTHD h-recurrence scale-placement ablation has been run for H16 S32K.
- The first curated summary is in `data/week1_experiment_log.md`.

### Week 1: Establish Baseline

1. Lock down measurement rigor: warmup/repeat/trial counts and variance policy.
2. Run fair FLA comparison to establish the target.
3. Profile the current TileOps version to confirm the h recurrence bottleneck.

### Week 2: Core Optimization Evidence

1. Run the algebraic rewrite ablation.
2. Run the naive-to-optimized ladder.
3. Generate the pipeline diagram, ladder chart, and h-recurrence resource figure.

### Week 3: Draft V1

1. Write sections 1-8 with the blocking experiments.
2. Add code snippets.
3. Do an internal review pass.

### Week 4: Supplementary Material

1. Run component breakdown if needed.
2. Run prepare inverse comparison if needed.
3. Run scan sweep if needed.
4. Add supplementary results as appendices.

## Suggested Repository Layout

For a public repository under `superAngGao`, a simple structure could be:

```text
gdn-prefill-ai-assisted-blog/
  LICENSE
  README.md
  REPRODUCE.md
  requirements.txt
  drafts/
    tutorial.md
    planning.md
  code/
    kernels/
    benchmarks/
  figures/
    pipeline.svg
    optimization-ladder.svg
    resource-usage.svg
  data/
    final_benchmarks.jsonl
    component_breakdown.jsonl
    prepare_inverse_comparison.jsonl
    scan_guard_sweep.jsonl
  scripts/
    summarize_benchmarks.py
  references/
    links.md
```

The current document can become `drafts/planning.md`.

## REPRODUCE.md Outline

The public repo should include a reproducibility guide with:

1. Hardware requirements: H100/H200-class GPU, CUDA version, locked-clock note.
2. Environment setup: Python environment, requirements, TileLang/TileOps commit.
3. Quick start: run the preconfigured optimization ladder benchmark.
4. Component benchmarks: how to isolate prepare, h recurrence, and output.
5. Fair FLA comparison: layout convention and `output_final_state=True`.
6. Expected results: latency ranges, variance notes, and accepted tolerances.
7. Troubleshooting: GPU clock drift, FLA version mismatch, JIT cache behavior,
   and long first-run compile times.

## Attribution Framing

For public writing, avoid overemphasizing individual credit in the main text.
Instead, use a methodological framing:

- Agent-assisted search found and validated local transformations.
- Human mathematical insight identified prepare/inversion and scan structure.
- The final production kernel came from combining both with careful profiling
  and guard design.

Internally, the current attribution is:

- AKO/agent-discovered: h-recurrence transposed multiplication rewrite.
- Human-proposed algorithmic improvements: prepare inverse analysis and prefix
  scan recurrence.
- Joint engineering: layout-compatible production path, benchmark fairness,
  dispatch guards, tests, and upstream PR polish.
