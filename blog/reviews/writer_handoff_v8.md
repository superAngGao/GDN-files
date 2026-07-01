# Writer Handoff V8: Tutorial Rewrite Constraints

This is the merged instruction sheet for the agent that will rewrite
`drafts/tutorial_v2.md`. It combines the current plan, external review, and our
final judgment.

Status:

```text
Ready to start rewrite.
Not ready for publication until the blockers below are resolved.
```

## Publication Blockers

Before publication, all five conditions must hold:

1. Neumann/blocksolve formula notation is either verified against the TileOps
   implementation or explicitly marked schematic.
2. Tier-1 correctness and benchmark tables are refreshed if any refresh trigger
   fires.
3. The CP-split non-originality sentence is included.
4. The hierarchical-prefix negative result is scoped to the tested production
   shape.
5. FlashQLA comparison keeps the public-environment / non-replay-attribution
   caveat.

## Main Narrative

Use the three-level capability structure:

1. **Level 1: Correct, measurable operator.**
   - Agent reads paper/reference code.
   - Reconstructs GDN prefill as `prepare -> replay -> output`.
   - Builds correctness, CUPTI benchmark, component breakdown, lowering, and
     logging gates.

2. **Level 2: Autonomous local AKO inside a fixed contract.**
   - h recurrence scale placement.
   - recompute store-path diagnosis.
   - small fusion and direct-fused attempts.
   - Key lesson: removing intermediates is not the same as shortening
     recurrence.

3. **Level 3: External input changes the search space.**
   - Qwen FlashQLA provides the production CP-split replay schedule and fused
     replay/output skeleton.
   - Human expert introduces blocked inverse / Neumann-style prepare framing
     as a stronger A producer inside the same schedule family.
   - TileOps ports, validates, tunes, dispatches, benchmarks, and
     productionizes the resulting path.

## Attribution Rules

Must include this direct sentence:

```text
TileOps did not invent the CP-split replay schedule. The contribution was to
study, validate, port, tune, dispatch, benchmark, and productionize the
FlashQLA-style schedule inside TileOps, combined with TileOps' own A producer.
```

Preserve both sides:

- Qwen FlashQLA: CP-split schedule and fused replay/output skeleton.
- TileOps: owned port, A producer, CP parameter tuning, H64 dispatch,
  validation, benchmark integration, and productionization.

Avoid both bad framings:

- "TileOps invented a better replay algorithm than FlashQLA."
- "We just copied FlashQLA."

## Neumann / Blocksolve Requirements

This is the novelty-sensitive math section.

Do not publish the following formula ladder as exact notation unless it has
been checked against `_prefill_blocksolve_A_bthd_tl`:

```text
W = A (beta K)
U = A (beta V)
A = (I + L)^(-1)
(I + L)^(-1) = I - L + L^2 - L^3 + ...
```

Verify before using as exact notation:

- `L` sign convention;
- gate / exponent placement;
- beta placement;
- matrix orientation;
- whether `A` is represented as a left- or right-multiply;
- consistency with the TileOps implementation.

If not verified, write:

```text
Schematic notation:
```

or:

```text
The following formulas describe the computational shape, not exact
implementation notation.
```

Use the same caveat in the figure caption.

## Prefix Negative Result

Must include this shape-scope sentence:

```text
This rejects the tested full `[b | M]` transition representation for the
current `DK=DV=128`, `chunk64` production path, not a general claim that prefix
scan cannot work for GDN.
```

Correct framing:

- affine transition idea is mathematically valid;
- experiments validate correctness/lowerability;
- production-shaped insertion loses due to representation and pipeline
  overhead;
- future narrower transition or different pipeline remains possible.

## Benchmark And Evidence Refresh

Publication refresh trigger:

```text
Rerun Tier-1 correctness and benchmark tables if the PR head, TileLang wheel,
docker/runtime, dispatch heuristic, benchmark timer, GPU, or FlashQLA/FLA
environment changes.
```

Current tables should be called:

```text
current candidate evidence snapshot
```

Do not call them final public claims unless refreshed.

FlashQLA comparison rows must carry:

- `max_local_chunks`;
- CP segment count;
- `block_DV`;
- TileLang version;
- TileOps commit;
- FlashQLA commit / docker environment;
- FLA version;
- GPU model / GPU id;
- warmup / repeat / trials;
- CUPTI / L2 flush / timer configuration;
- dtype / layout / seed.

## Speedup Language

Allowed for public-environment full-op comparison with shapes/timers/versions:

- faster;
- notably faster;
- significantly faster.

Do not apply these words to replay algorithm attribution unless CP split, call
path, lowering, dispatch, and timer are all controlled.

Table B caveat must remain:

```text
TileOps vs FlashQLA is a public-environment comparison, not a controlled
same-lowering replay attribution experiment.
```

## Figure Requirements

Required or strongly recommended figures:

1. **Direct Fused vs CP-Split Schedule**
   - Must have.
   - Right side must show `prepare_h/correct_h0`; do not imply CP segments are
     naturally independent.

2. **Blocksolve / Neumann Prepare**
   - Required figure or inset.
   - Left: dense-looking lower-triangular solve.
   - Right: 4x16 block-lower decomposition with local Neumann-style updates.

3. **Measurement Gate / AKO Loop**
   - Show candidate idea -> TileLang edit -> compile gate -> correctness gate
     -> CUPTI/component benchmark -> lowering inspection -> decision log ->
     accept/reject.

4. **Dispatch / Metadata / Shape Guard**
   - Show shape metadata -> dispatch policy -> selected kernel and benchmark
     metadata.

5. **Hierarchical Prefix Negative Result**
   - Show dependency depth gets shorter while representation/pipeline cost gets
     heavier.

6. **Migration / Lowering Mismatch**
   - Show source similarity does not imply performance equality.

7. **Attribution Map**
   - Optional but strongly useful.
   - FLA -> behavioral correctness reference.
   - FlashQLA -> CP-split schedule and fused replay skeleton.
   - Human expert -> Neumann/blocksolve mathematical reframing.
   - AKO -> gated implementation search and tuning.
   - TileOps -> owned port, A producer, dispatch, validation,
     productionization.

## Code Shape Caveats

Preserve these caveats during rewrite:

- decode residual snippet is schematic unless exponent/orientation is verified;
- FLA is the main behavioral correctness reference;
- FlashQLA is mainly schedule/source/performance reference unless explicitly
  stated otherwise;
- TMA/WGMMA snippets are TileLang-shaped pseudocode unless exact API and
  generated code are archived;
- do not paste full kernel listings or generated CUDA/PTX into the main text.

## `tutorial_v2.md` Rewrite Map

Reusable with adaptation:

- Sections 0-4: terminology, motivation, formulas, measurement loop.
- Section 5: scale placement local AKO.
- Section 6: recompute/store-path diagnostic.
- Section 7: blocksolve-A technical content, moved to Level 3.

Must rewrite:

- introduction and thesis;
- capability summary;
- human algorithmic moves;
- production integration;
- benchmark section;
- negative results section.

Do not directly copy Section 9.

The scan/prefix formulation should appear only as an explored-and-rejected
branch, not as the final production algorithm source.
