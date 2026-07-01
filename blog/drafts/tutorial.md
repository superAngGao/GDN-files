# AI-Assisted TileLang Kernel Optimization: From GDN Prefill To Faster-Than-FLA

> Draft status: skeleton with first evidence pass. This is not the final public
> article yet.

## 1. Why This Case Study Exists

This tutorial is about optimizing a TileLang kernel in an AI-assisted
engineering environment. The concrete example is Gated DeltaNet prefill in
TileOps, but the broader topic is the workflow: profile, hypothesize, test,
record, prune, and turn mathematical insight into guarded production code.

The starting point is assumed to be an engineering-correct prefill op:

- BTHD input layout aligned with FLA and Qwen-style serving
- BHTD retained as a TileOps-compatible path
- public inputs `q, k, v, g, beta`
- public outputs `o, final_state`
- fair FLA comparison with `output_final_state=True`

The optimization story starts after those interface conditions are satisfied.

## 2. Algorithm Map

For BTHD inputs:

```text
q, k: [B, T, H, K]
v:    [B, T, H, V]
g:    [B, T, H]
beta: [B, T, H]
```

The op returns:

```text
o:           [B, T, H, V]
final_state: [B, H, K, V]
```

Each `(batch, head)` stream owns a recurrent state `h` with shape `[K, V]`.
The prefill implementation decomposes the work into three stages.

### Stage 1: Prepare

The prepare stage resolves chunk-local causal dependencies and produces
intermediate `w` and `u` tensors:

```text
w_c, u_c = Prepare(k_c, v_c, g_c, beta_c)
```

This is a chunk-local triangular-system problem. TileOps currently uses a
fused Neumann-style prepare; FLA uses an optimized blocked triangular-solve
pipeline. A naive forward solve is a diagnostic, not a fair representation of
FLA.

### Stage 2: H Recurrence

The h recurrence updates boundary states between chunks:

```text
v_new_c = u_c - ((w_c * exp(g_c + g_last)[..., None]) @ h_c)

h_{c+1} = exp(g_last) * h_c
          + (k_c * exp(g_last - g_c))^T @ v_new_c
```

This stage produces both the chunk boundary states and `v_new`, which are then
consumed by the output kernel.

### Stage 3: Output

The output stage combines cross-chunk state and intra-chunk causal attention:

```text
o_c = exp(g_c) * q_c @ h_c
      + causal((q_c k_c^T) * exp(g_i - g_j)) @ v_new_c
```

## 3. Baseline Measurement

The first measurement rule was simple: compare the same layout, same dtype, and
same output contract. On H200 with SM locked to 1500 MHz:

| Seq len | Heads | TileOps BTHD | FLA BTHD | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 32K | 16 | 2.8165 ms | 3.1320 ms | 1.11x |
| 64K | 16 | 5.5506 ms | 6.5554 ms | 1.18x |
| 128K | 16 | 11.2403 ms | 13.4351 ms | 1.20x |
| 128K | 32 | 18.6650 ms | 21.0881 ms | 1.13x |

The S32K profiler points to h recurrence as the largest component, while S128K
shows the long-sequence scan path split into transition summary, group-start
scan, and grouped replay.

## 4. Optimization 1: Move The Scale

The AKO loop found a local algebraic rewrite in the h update:

```text
(k * exp(g_last - g_i))^T @ v_new
```

can be computed as:

```text
k^T @ (v_new * exp(g_last - g_i))
```

The mathematical result is the same, but the implementation effect is large:
we no longer need a `k_scaled_s` shared-memory buffer.

On the BTHD H16/S32K h recurrence:

| Variant | H-only latency | Speedup vs old |
| --- | ---: | ---: |
| old: scale `k` into `k_scaled_s` | 2.2725 ms | 1.00x |
| current: scale `v_new` in place | 1.6277 ms | 1.40x |

The two variants match with `v_new_max_abs = 0` and final-state max difference
`3.05e-5`.

## 5. Optimization 2: Understand Prepare Before Replacing It

The prepare stage is tempting to describe as "Neumann versus forward solve",
but that framing is incomplete. A naive TileLang forward solve can be much
slower without proving that FLA's blocked solve strategy is worse. The correct
lesson is to identify the mathematical object first, then compare similarly
optimized implementations.

This section still needs the final prepare comparison table.

## 6. Optimization 3: Scan The Chunk Recurrence

The cross-chunk recurrence can be written as an affine transition:

```text
h_{c+1} = A_c h_c + b_c
```

Composition of these transitions is associative:

```text
(A_2, b_2) o (A_1, b_1) = (A_2 A_1, A_2 b_1 + b_2)
```

That enables a prefix-scan path for long sequences. The production kernel keeps
this behind a narrow guard, because scan overhead is shape-dependent.

Current measured S128K/H16 profiler stages:

| Kernel/event | Device time |
| --- | ---: |
| `prefill_prepare_w_u_bthd_kernel` | 1764.322 us |
| `group_transition_summary_bthd_kernel` | 1720.706 us |
| `h_grouped_replay_bthd_kernel` | 1107.425 us |
| `output_o_bthd_kernel` | 970.337 us |
| `dense_group_start_scan_bthd_kernel` | 150.240 us |

This section still needs the final guard sweep table.

## 7. What Did Not Work

Negative results are part of the method. The final article should include the
failed or rejected branches:

- BV8 increased CTA count but lost overall.
- fused h+output serialized output work and recomputed too much.
- sparse checkpointing saved boundary-state bytes but duplicated h updates.
- naive forward solve was not a fair FLA-style solve.
- broad scan dispatch loses on shapes outside the current guard.

## 8. Lessons For TileLang Developers

Early lessons from this case study:

- measure with the target layout, not a convenient internal layout
- isolate the bottleneck before starting AKO
- keep a decision log so negative results remain useful
- let AI search local code transformations
- keep humans responsible for algebraic framing and production guards
- validate algorithmic optimizations by shape before enabling dispatch broadly
