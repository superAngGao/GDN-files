# Kernel Code Shapes For Tutorial Rewrite

Status: working note for rewriting `tutorial_v2.md`.

Purpose: identify the code shapes worth showing in the article. These are not
full source excerpts. They are blog-facing skeletons: enough code structure to
make each optimization node concrete without turning the article into a kernel
listing.

## 1. Correct Operator Development: Chunkwise GDN Dataflow

Source anchors:

- `/home/ga/TileOPs-pr1596/tileops/kernels/gated_deltanet/gated_deltanet_prefill.py`
  - `_prefill_recompute_w_u_from_A_bthd_tl`
  - `_prefill_grouped_replay_bthd_tl`
  - `_prefill_output_o_bthd_tl`
- `/home/ga/TileOPs-pr1596/tileops/kernels/gated_deltanet/gdn_prefill/fused_fwd.py`

Blog code shape:

```python
# Stage 1: chunk-local prepare
A = blocksolve_A(k, g, beta)
w, u = recompute_w_u_from_A(k, v, beta, A)

# Stage 2: replay recurrent state across chunks
for chunk in chunks:
    v_new = u_chunk - exp(g_i + g_last) * (w_chunk @ h)
    h = exp(g_last) * h + k_chunk.T @ (exp(g_last - g_i) * v_new)

# Stage 3: output
o_state = exp(g_i) * (q_chunk @ h_in)
o_local = causal(q_chunk @ k_chunk.T * exp(g_i - g_j)) @ v_new
o = o_state + o_local
```

Article role:

- Introduce the operator as `prepare -> replay -> output`.
- Explain that `w/u` are effective chunk writes, not raw `beta*k` and
  `beta*v` except in the decode/single-token case.

## 2. Decode View: Residual Write

Source anchors:

- Paper-level explanation, cross-checked against replay formulas in
  `gated_deltanet_prefill.py`.

Blog code shape:

This is an implementation-convention schematic for explaining the residual
write mechanism. Before publication, verify the exponent placement and matrix
orientation against the exact code path being quoted.

```python
w_t = beta_t * k_t
u_t = beta_t * v_t

prediction = w_t @ h_prev
residual = u_t - exp(2 * g_t) * prediction

o_t = exp(g_t) * (q_t @ h_prev) + (q_t @ k_t.T) * residual
h_t = exp(g_t) * h_prev + k_t.T @ residual
```

Article role:

- Explain why GDN has better memory signal-to-noise than vanilla linear
  attention: it writes residuals instead of blindly accumulating values.
- Explain `g` as memory lifetime/coordinate scaling, and `beta` as write
  strength.

## 3. Chunkwise `w/u`: Effective Writes

Source anchors:

- `_prefill_recompute_w_u_from_A_bthd_tl` in
  `gated_deltanet_prefill.py`.
- older fused prepare shape around `S_shared`, `P_shared`, `k_beta_shared`,
  `v_beta_shared`.

Blog code shape:

```python
k_beta = beta[:, None] * k
v_beta = beta[:, None] * v

# A encodes the lower-triangular chunk-local delta-rule solve.
w = A @ k_beta
u = A @ v_beta
```

TileLang-shaped snippet:

```python
T.async_copy(A[chunk, :, :], A_s)

for tile in v_tiles:
    T.async_copy(v_beta_tile, x_s)
    T.gemm(A_s, x_s, out_frag)
    T.copy(out_frag, out_s)
    T.copy(out_s, u_tile)

for tile in k_tiles:
    T.async_copy(k_beta_tile, x_s)
    T.gemm(A_s, x_s, out_frag)
    T.copy(out_frag, out_s)
    T.copy(out_s, w_tile)
```

Article role:

- Show that chunkwise prefill is not simply vectorized decode.
- Explain why the prepare stage exists: it absorbs intra-chunk causal delta
  dependencies into `w/u`.

## 4. Local AKO Win: Move Scale From K Path To V Path

Source anchors:

- `_prefill_grouped_replay_bthd_tl` and related replay kernels.
- AKO notes around h recurrence scale placement.

Blog code shape:

```python
# Old shape
k_scaled = k_chunk * exp(g_last - g_i)[:, None]
h += k_scaled.T @ v_new

# Accepted shape
v_scaled = v_new * exp(g_last - g_i)[:, None]
h += k_chunk.T @ v_scaled
```

TileLang-shaped snippet:

```python
for i, j in T.Parallel(block_C, BV):
    v_new_c[i, j] *= T.exp2((g_last - g_c[i]) * LOG2E)

T.gemm(k_c, v_new_c, h_next_frag, transpose_A=True)
```

Article role:

- A small algebraic rewrite changes the shared-memory path.
- This belongs to "agent autonomous local AKO inside a fixed contract."

## 5. Local AKO Diagnostic: Recompute Store Path

Source anchors:

- `_prefill_recompute_w_u_from_A_bthd_tl`.
- recompute decision logs and no-store diagnostic.

Blog code shape:

```python
T.gemm(A_s, x_s, out_frag)

# Important shape: route fragment through a store-friendly shared tile.
T.copy(out_frag, out_s)
T.copy(out_s, global_out_tile)
```

Article role:

- Matching the arithmetic primitive was not enough.
- The store path and layout mattered as much as the GEMM body.

## 6. External Human Insight: Blocked Inverse / Neumann Prepare

Source anchors:

- `_prefill_blocksolve_A_bthd_tl` in `gated_deltanet_prefill.py`.
- `latest_fla_prepare_decision_log.md`.
- `tilelang_recompute_lowering_decision_log.md`.

Blog code shape:

```python
# 64-token chunk split into four 16-token blocks.
G00 = k0 @ k0.T
G10 = k1 @ k0.T
G11 = k1 @ k1.T
...
G33 = k3 @ k3.T

# Build lower blocks, then solve/invert small lower-triangular blocks.
I0 = inverse_or_neumann(G00)
I1 = inverse_or_neumann(G11)
I2 = inverse_or_neumann(G22)
I3 = inverse_or_neumann(G33)

# Compose lower off-diagonal blocks in shared memory.
A = blocked_lower_inverse(I0, I1, I2, I3, lower_blocks)
```

Article role:

- This is not "agent local search"; it is a human-supplied mathematical search
  space.
- AKO then becomes useful inside the new space: implement variants, run
  correctness gates, inspect lowering, benchmark.

## 7. FlashQLA / TileOps CP-Split Schedule

Source anchors:

- `/home/ga/TileOPs-pr1596/tileops/kernels/gated_deltanet/gdn_prefill/prepare_h.py`
- `/home/ga/TileOPs-pr1596/tileops/kernels/gated_deltanet/gdn_prefill/cp_fwd.py`
- `/home/ga/TileOPs-pr1596/tileops/kernels/gated_deltanet/gdn_prefill/fused_fwd.py`
- `flashqla_scheduling_skeleton_20260623.md`
- `flashqla_pipeline_overlap_20260623.md`

Blog code shape:

```python
# Production long-prefill schedule, credited to Qwen FlashQLA.
g = chunk_local_cumsum(g)
A = kkt_solve_or_blocksolve(k, g, beta)

warmup_chunks = get_warmup_chunks(...)
h_warmup = prepare_h(k, v, A, g, beta, warmup_chunks)
h0 = correct_h0(initial_state, h_warmup, cp_metadata)

o, final_state = fused_gdr_fwd(
    q, k, v, A, g, beta,
    initial_state=h0,
    cp_seq_map=cp_seq_map,
)
```

Schedule contrast:

| Shape | Dependency structure | What it proves |
| --- | --- | --- |
| Direct fused without CP | One long recurrence replay loop from the original initial state. | Fusion alone removes some global materialization, but it does not shorten the causal replay dependency. |
| FlashQLA-style CP split | Compute corrected segment initial states, then run fused replay/output over shorter segments. | The key schedule-level win is shortening the replay dependency while keeping replay/output fused. |

Minimal pseudocode:

```python
# Direct fused path: one long dependency chain.
h = initial_state
for chunk in all_chunks:
    o[chunk], h = replay_output_chunk(chunk, h)

# CP-split path: segment starts are prepared first, then replay is local.
h0_segments = prepare_corrected_segment_states(all_chunks, initial_state)
parallel_for segment in cp_segments:
    h = h0_segments[segment]
    for chunk in segment.chunks:
        o[chunk], h = replay_output_chunk(chunk, h)
```

Article role:

- Credit FlashQLA for the CP-split schedule and fused replay skeleton.
- Explain the actual schedule change:
  - compute corrected segment initial states;
  - replay/output over shorter CP segments;
  - avoid global `w/u/S/v_new` materialization on the inference path.

## 8. FlashQLA Fused Forward Skeleton

Source anchors:

- `gdn_prefill/fused_fwd.py`
- FlashQLA source notes.

Blog code shape:

```python
with T.Kernel(grid=(batch, head, dv_tile), threads=512):
    # producer / store side
    load q, k, v, A, g, beta tiles with TMA

    # consumer side
    for chunk in cp_segment:
        compute local QK / S / V pieces
        update h fragment or shared state
        accumulate output

    store o and final_state
```

Article role:

- Do not overquote implementation.
- Show the schedule shape: producer/consumer split and delayed output store.
- Tie this to the source-level vs lowering-level lesson:
  source similarity does not guarantee same TMA/WGMMA behavior.

## 9. Negative Result: Full Hierarchical Prefix

Source anchors:

- `/home/ga/Documents/gdn_kernel_bench_2026-06-18/probe_tilelang_transition_gemm.py`
- `/home/ga/Documents/gdn_kernel_bench_2026-06-18/probe_tilelang_transition_compose2.py`
- `/home/ga/Documents/gdn_kernel_bench_2026-06-18/bench_pair2_transition_candidate.py`
- `/home/ga/Documents/gdn_kernel_bench_2026-06-18/bench_pair2_direct_summary_candidate.py`
- `flashqla_specialized_ako_log.md` Rounds 031-035.

Blog code shape:

```python
# Full transition summary for a group.
M = exp(g_last) * I - exp(2 * g_last) * (K.T @ W)
b = K.T @ (exp(g_last - g_i)[:, None] * U)

# Associative composition.
M_21 = M2 @ M1
b_21 = M2 @ b1 + b2
```

Rejected fused candidate shape:

```python
for group in groups:
    # Direct replay/output path.
    o, h_out = direct_replay_output(group, h_start)

    # Extra full summary path.
    summary[group] = transition_summary_over_augmented_state(group)
```

Article role:

- Prefix is more parallel in dependency depth, but not cheaper here.
- Full transition summary carries `[b | M]` with width `DV + DK`.
- Evidence: correct and lowerable, but too expensive for the production fused
  path under `DK=DV=128`.

## 10. Measurement Gate / AKO Loop

Source anchors:

- `flashqla_specialized_ako_log.md`
- TileOps benchmark scripts and CUPTI timing utilities used in the PR
  validation.

Blog code shape:

```python
candidate = build_tilelang_kernel(config)
correctness_ref = fla_reference
schedule_ref = flashqla_source_if_needed

correct = check_against_reference(
    candidate,
    reference=correctness_ref,
    shapes=validated_shapes,
)

if correct:
    latency = cupti_bench(
        candidate,
        warmup=warmup,
        repeat=repeat,
        trials=trials,
    )
    lowering = inspect_generated_code(candidate)
else:
    latency = None
    lowering = None

decision_log.write({
    "config": config,
    "correct": correct,
    "latency_ms": latency,
    "lowering_notes": lowering.summary if lowering else None,
    "decision": accept_or_reject(correct, latency, lowering),
})
```

Article role:

- Show that AKO is gated engineering search, not free-form prompting.
- Explain why each candidate needs correctness, benchmark, lowering, and log
  evidence before it can influence the production story.
- Tie historical rows to dated evidence and final claims to latest PR-head
  evidence.

## 11. Dispatch Metadata And Shape Guards

Source anchors:

- `/home/ga/TileOps-pr1596/tileops/kernels/gated_deltanet/gated_deltanet_prefill.py`
- `/home/ga/TileOps-pr1596/tileops/kernels/gated_deltanet/gdn_prefill/`
- PR benchmark metadata records.

Blog code shape:

```python
meta = resolve_gdn_prefill_dispatch(
    B=B,
    T=T,
    H=H,
    DK=DK,
    DV=DV,
    dtype=dtype,
    layout="BTHD",
)

kernel = select_prefill_kernel(
    use_cp=meta.use_cp,
    max_local_chunks=meta.max_local_chunks,
    block_DV=meta.block_DV,
)

record_benchmark_metadata({
    "max_local_chunks": meta.max_local_chunks,
    "cp_segments": meta.cp_segments,
    "block_DV": meta.block_DV,
    "tilelang_version": tilelang.__version__,
    "commit": git_commit,
    "timer": timer_config,
})
```

Article role:

- Explain why production dispatch is part of the kernel, not a bookkeeping
  afterthought.
- Support the hard publication gate for FlashQLA comparison rows:
  `max_local_chunks`, CP segment count, `block_DV`, TileLang version, commit,
  and timer must be visible.
- Connect the H64 dispatch correction to a concrete shape-guard story.

## 12. Migration Lesson: Source Copy vs TMA Restored Path

Source anchors:

- FlashQLA migration notes and generated-code inspection logs.
- TileLang 0.1.8 vs 0.1.11 migration experiments.

Blog code shape:

Pseudo TileLang-shaped snippet:

```python
# Migration shape that looked source-equivalent but lost the intended pipeline.
T.copy(global_tile, shared_tile)

# Restored TMA-specialized path.
T.tma_copy(global_tile, shared_tile, barrier=barrier)
T.wait_tma_barrier(barrier)  # pseudocode

# Only claim TMA behavior after inspecting generated code.
lowering = inspect_generated_cuda(kernel)
assert lowering.contains_expected_tma_path
```

Article role:

- Explain why FlashQLA migration was engineering discovery, not mechanical
  copying.
- Support the lesson that source-level equality is not performance equality.
- Tie Table B's compiler/runtime caveat to concrete lowering evidence.

## 13. Code Shapes To Avoid In The Main Text

Avoid full kernel listings for:

- the entire 512-thread fused forward;
- the full blocked inverse implementation;
- generated CUDA/PTX.

Use diagrams or abbreviated skeletons instead. Full paths can go in references
or appendix notes. The article should teach the dataflow and search-space
boundary, not force readers to parse hundreds of lines of TileLang.
