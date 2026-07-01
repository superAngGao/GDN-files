# H/Replay Attribution Note, 2026-06-25

Status: working note for blog planning. Updated after the synced component
ablation below. The earlier unsynced component gaps should not be used as a
strong attribution claim.

## Question

The final full-op numbers show TileOps faster than FlashQLA. A/KKT is clearly
different because TileOps uses its own A producer. The open question is whether
the h/replay back half is also faster, and if so, why.

## Current Evidence

Existing component attribution files:

- `replay_attribution_tileops_128k_h16_h32_20260625.jsonl`
- `replay_attribution_flashqla_128k_h16_h32_20260625.jsonl`
- `breakdown_tileops_h16_20260625.jsonl`
- `breakdown_flashqla_tl018_h16_h64_20260625.jsonl`

All rows below are H200/GPU4, CUPTI-style filtered kernel time. TileOps is the
current TileOps PR environment; FlashQLA is the public TL0.1.8 environment.

| Case | TileOps fused replay/output | FlashQLA fused replay/output | Ratio |
| --- | ---: | ---: | ---: |
| `64K/H16` | `0.444991 ms` | `0.741983 ms` | `1.67x` |
| `128K/H16` | `0.837670 ms` | `1.490566 ms` | `1.78x` |
| `128K/H32` | `1.661218 ms` | `3.060147 ms` | `1.84x` |

The measured event names are:

- TileOps: `tilelang_fused_chunk_gdr_fwd_kernel_kernel`
- FlashQLA: `fused_chunk_gdr_fwd`

This initially supported the narrow statement:

```text
On the measured rows, TileOps' fused replay/output kernel event is shorter than
FlashQLA TL0.1.8's fused replay/output kernel event.
```

It does not by itself prove a different high-level h/replay algorithm.

Important correction: a later ablation found that single-kernel component
profiling can suffer from profiler step boundary bleed if the queued warmup
kernels are not synchronized before `profiler.step()`. In that case, the event
count can become close to `2 * repeat`, inflating some component numbers. The
synced ablation below is the stronger evidence.

## Synced CP/Fusion Ablation, 2026-06-25

Script:

```text
ablate_cp_fused_replay_official.py
```

Method:

- same fixed q/k/v/g/beta inputs
- precompute only `g_cum` and `A`
- use L2 flush + CUPTI + warmup/repeat/trials
- add `torch.cuda.synchronize()` before each `profiler.step()` so component
  event counts equal `repeat`

Output files:

```text
results/flashqla_specialized/ablate_cp_fused_replay_tileops_sync_20260625.jsonl
results/flashqla_specialized/ablate_cp_fused_replay_flashqla_sync_20260625.jsonl
```

Clean component results:

| Case | Backend | CP schedule total | fused CP replay | fused no-CP replay | no-CP / CP |
| --- | --- | ---: | ---: | ---: | ---: |
| `64K/H16` | TileOps | `0.537757 ms` | `0.440062 ms` | `2.152506 ms` | `4.89x` |
| `64K/H16` | FlashQLA | `0.524259 ms` | `0.436775 ms` | `2.083542 ms` | `4.77x` |
| `128K/H16` | TileOps | `0.939820 ms` | `0.831321 ms` | `4.199742 ms` | `5.05x` |
| `128K/H16` | FlashQLA | `0.920950 ms` | `0.827502 ms` | `4.168005 ms` | `5.04x` |
| `128K/H32` | TileOps | `1.809344 ms` | `1.663766 ms` | `4.188219 ms` | `2.52x` |
| `128K/H32` | FlashQLA | `1.782718 ms` | `1.650833 ms` | `4.174342 ms` | `2.53x` |

TileOps-only non-fused no-CP comparison:

| Case | non-fused no-CP total | h/output-side subset |
| --- | ---: | ---: |
| `64K/H16` | `2.552870 ms` | `2.099001 ms` |
| `128K/H16` | `5.045389 ms` | `4.162569 ms` |

Conclusions from this ablation:

- CP-split is the dominant h/replay-side factor. Removing it makes the fused
  replay kernel about `4.8-5.0x` slower on H16 and `2.5x` slower on H32.
- Once CP-split is aligned, TileOps and FlashQLA fused replay are essentially
  the same speed on these rows. The ratio is within about `1%`.
- Fusion still matters: TileOps' old non-fused no-CP path is slower than the
  fused no-CP path and also materializes more intermediate state.
- Therefore the blog should not claim TileOps has a faster high-level replay
  algorithm than FlashQLA. The high-level back-half speedup comes from adopting
  FlashQLA's CP-split + fused replay schedule. TileOps' remaining full-op win
  must be explained primarily by the A/KKT producer and production integration,
  not by a fundamentally better replay skeleton.

## Source-Level Comparison

`tileops/kernels/gated_deltanet/gdn_prefill/fused_fwd.py` is intentionally very
close to FlashQLA's:

`FlashQLA-tl018-src/flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py`

The meaningful source-level differences are small:

- TileOps imports `prepare_chunk_offsets` from its local `gdn_prefill.utils`.
- TileOps adds compile flags:
  `["-O3", "-DENABLE_BF16", "-include", "tl_templates/cuda/gemm.h"]`.
- TileOps adds an environment override for `block_DV`.
- TileOps has style-only comments/noqa changes.

Therefore the current evidence should be framed as:

```text
Same schedule family and very similar fused replay skeleton, but a faster
TileOps instantiation under the current TileLang/compiler/runtime and dispatch
environment.
```

not:

```text
TileOps invented a better fused replay algorithm.
```

## Factor-by-Factor Breakdown, 2026-06-25

This section breaks down the suspected causes one by one. The working rule is:
do not attribute the fused replay gap to an item unless either source diff,
controlled benchmark, or generated-code evidence supports it.

### 1. Local Import

TileOps imports:

```text
from .utils import prepare_chunk_offsets
```

FlashQLA imports:

```text
from flash_qla.utils import prepare_chunk_offsets
```

The actual helper implementation is identical except import ordering. A direct
diff between FlashQLA's `utils/index.py` and TileOps'
`gdn_prefill/utils.py` only shows reordered imports.

Conclusion: this is not a plausible fused replay performance cause. It is a
packaging/ownership change.

### 2. `block_DV` Override Hook

TileOps adds an environment override:

```text
TILEOPS_GDN_PREFILL_BLOCK_DV
TILEOPS_GDN_PREFILL_CP_BLOCK_DV
```

Without the override, TileOps uses FlashQLA's same default rule:

```text
if grid_size >= TARGET_NUM_CTAS: block_DV = 128
elif grid_size * 2 >= TARGET_NUM_CTAS: block_DV = 64
else: block_DV = 32
```

For the measured H16/H32 rows on H200:

| Case | CP segments | grid | default `block_DV` | Same as FlashQLA? |
| --- | ---: | ---: | ---: | --- |
| `64K/H16` | `32` | `512` | `128` | yes |
| `128K/H16` | `32` | `512` | `128` | yes |
| `128K/H32` | `32` | `1024` | `128` | yes |

Controlled TileOps run on `64K/H16`, GPU4, current docker, commit `7fb5f35d`:

`results/flashqla_specialized/h_replay_factor_tileops_blockdv_20260625.jsonl`

| Forced `block_DV` | fused replay/output |
| ---: | ---: |
| `128` | `0.443763 ms` |
| `64` | `0.635999 ms` |
| `32` | `1.033272 ms` |

Conclusion: the override hook is not a hidden advantage on H16/H32; the default
is already `128`. The important performance property is full-DV CTA coverage,
which avoids V-tile local-attention recomputation. H64 is different because
TileOps has a separate dispatch policy that keeps CP enabled for long H64 rows,
while public FlashQLA disables CP when `Be * H > 56`.

### 3. `gemm_v1` Compat / Compile Flags

TileOps uses a local `T.gemm_v1` compatibility helper. It can choose:

- `default`: delegate to current `T.gemm`
- `wgmma`: call `T.wgmma_gemm` explicitly
- `legacy`: call a `tl::gemm_ss` helper via `T.call_extern`

TileOps also JITs fused/prepare-h with:

```text
["-O3", "-DENABLE_BF16", "-include", "tl_templates/cuda/gemm.h"]
```

Controlled TileOps run on `64K/H16`, GPU4, current docker, commit `7fb5f35d`:

`results/flashqla_specialized/h_replay_factor_tileops_gemm_modes_20260625.jsonl`

| `TILEOPS_GDN_PREFILL_GEMM_V1_MODE` | fused replay/output |
| --- | ---: |
| `default` | `0.442963 ms` |
| `wgmma` | `0.445948 ms` |
| `legacy` | `1.600719 ms` |

Generated CUDA in TileOps cache confirms the distinction:

- `default` / `wgmma`: generated `tl::wgmma_ss` calls
- `legacy`: generated `tl::gemm_ss` calls

Conclusion: current TileOps speed is not explained by resurrecting the old
`gemm_ss` path. On this fused replay kernel, the old-style `gemm_ss` helper is
much slower. The fast TileOps path is the current WGMMA-style lowering.

### 4. TileLang / Runtime / Lowering Environment

Existing migration records show two different signals:

Old non-flush kernel breakdown:

| Row | fused replay/output |
| --- | ---: |
| FlashQLA TL0.1.8 | `0.439040 ms` |
| FlashQLA TL0.1.9 migration default | `0.441271 ms` |
| FlashQLA TL0.1.9 migration wgmma | `0.441206 ms` |

Later flush-aligned CUPTI breakdown:

| Row | fused replay/output |
| --- | ---: |
| FlashQLA TL0.1.8 anchor | `0.721201 ms` |
| FlashQLA TL0.1.9 after 3C | `0.749473 ms` |
| TileOps current, same shape | `0.443-0.445 ms` |

This means the gap cannot yet be cleanly attributed to TileLang version alone.
The early non-flush measurement had FlashQLA fused replay near `0.44 ms`, while
the later flush-aligned measurement had FlashQLA near `0.72-0.75 ms`. TileOps
current remains near `0.44 ms` under the flush-aligned TileOps component
profiler.

Conclusion: this is the unresolved part. The next evidence must compare
generated CUDA/PTX/SASS and the exact call path under the same timer semantics.
Current source diff is too small to justify a high-level algorithm claim.

### 5. Current Best Explanation

For H16/H32, the safe explanation is:

```text
FlashQLA supplied the CP-split schedule and fused replay skeleton. TileOps
keeps the same schedule family and full-DV CTA shape, but its current
TileOps-owned instantiation runs the fused replay/output event faster in our
component measurements. The most likely causes are implementation/lowering and
measurement/call-path details, not a different h/replay algorithm.
```

For H64, include the extra qualifier:

```text
H64 also includes a dispatch-policy difference: TileOps keeps CP enabled for
the long-sequence H64 row, while public FlashQLA disables CP at high Be*H.
```

## CP Parameter Context

For the H16/H32 rows, TileOps and FlashQLA appear to use the same CP split
formula family. The obvious H64 difference is separate:

- FlashQLA disables CP when `Be * H > 56`.
- TileOps has an H64 long-sequence special case that keeps CP enabled and
  raises `max_local_chunks` to at least `256`.

So H64 should not be used as evidence that TileOps' same-schedule fused replay
kernel is intrinsically faster. It is partly a dispatch policy difference.

## Prefix-Scan / Butterfly Clarification, 2026-06-25

Important correction: the quick PyTorch Hillis-Steele experiment below is not
a fair replacement for the pre-FlashQLA TileOps scan idea.

Before studying FlashQLA, the TileOps prefix-scan idea was not "call PyTorch
matmul in a Hillis-Steele loop." It was a TileLang-native execution model:

```text
group_transition_summary_bthd
  -> dense_group_start_scan_bthd
  -> h_grouped_replay_bthd
  -> output_o_bthd
```

The mathematical object was the same affine transition:

```text
h_out = A @ h_in + b
```

but the implementation was a fused/blocked TileLang decomposition with a narrow
shape guard. Week-1 profiler evidence showed this path active on `128K/H16`:

| Kernel/event | Device time |
| --- | ---: |
| `group_transition_summary_bthd_kernel` | `1720.706 us` |
| `h_grouped_replay_bthd_kernel` | `1107.425 us` |
| `output_o_bthd_kernel` | `970.337 us` |
| `dense_group_start_scan_bthd_kernel` | `150.240 us` |

So the useful pre-FlashQLA contribution was recognizing and validating the
associative chunk-transition formulation, then dispatching it only on shapes
where the extra summary/scan/replay work paid off. It was not an arbitrary
Python-level butterfly scan.

### TileLang Butterfly Migration Probe

After the unfair PyTorch prototype, we migrated the correct version: a
TileLang-native butterfly scan over the same grouped transition layout. The
implementation uses global ping-pong prefix buffers:

```text
group_scan_init
  -> group_scan_stage(offset=1)
  -> group_scan_stage(offset=2)
  -> ...
  -> group_scan_finalize
```

Each stage composes dense affine transitions:

```text
A_new = A_after @ A_before
b_new = A_after @ b_before + b_after
```

This is a fairer test of the pre-FlashQLA prefix-scan idea because both dense
scan and butterfly scan run as TileLang/CUDA kernels on the same summary
layout.

Script:

```text
ablate_butterfly_group_scan.py
```

Method:

- reuse TileOps' existing `group_transition_summary_bthd` output
- treat each group summary as an affine transition:
  `h_out = A @ h_in + b`
- compare `dense_group_start_scan_bthd` with a TileLang-native butterfly
  composition
- first compare the group-start scan in isolation, then compare the full
  grouped back-half path with only the scan implementation changed

Correctness: the TileLang butterfly output matched the dense TileOps scan
exactly on the tested rows (`max_abs = 0`).

Timing:

| Case | `group_chunks` | `num_groups` | dense scan | TileLang butterfly scan | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `64K/H16` | `64` | `16` | `0.097107 ms` | `0.144419 ms` | `1.49x` slower |
| `128K/H16` | `16` | `128` | `0.562860 ms` | `1.251753 ms` | `2.22x` slower |
| `128K/H16` | `32` | `64` | `0.326094 ms` | `0.584813 ms` | `1.79x` slower |
| `128K/H16` | `64` | `32` | `0.173385 ms` | `0.284239 ms` | `1.64x` slower |
| `128K/H16` | `128` | `16` | `0.095777 ms` | `0.141729 ms` | `1.48x` slower |

Minimal `block_v` sweep on `128K/H16`, `group_chunks=64`:

| `block_v` | dense scan | TileLang butterfly scan | Ratio |
| ---: | ---: | ---: | ---: |
| `32` | `0.108593 ms` | `0.310653 ms` | `2.86x` slower |
| `64` | `0.172762 ms` | `0.286048 ms` | `1.66x` slower |
| `128` | `0.277930 ms` | `0.268248 ms` | `0.97x` |

Full grouped back-half sweep, 2026-06-26:

Output files:

```text
results/flashqla_specialized/ablate_tilelang_butterfly_full_sweep_20260626.jsonl
results/flashqla_specialized/ablate_tilelang_butterfly_full_512k_20260626.jsonl
```

This run times the whole grouped back-half:

```text
group_transition_summary_bthd
  -> dense or butterfly group-start scan
  -> h_grouped_replay_bthd
  -> output_o_bthd
```

Correctness: both `o` and `states` matched the dense TileOps path exactly on
all rows below (`max_abs = 0`).

Best rows from the parameter sweep:

| Case | `group_chunks` | `block_v` | `num_groups` | dense full back-half | butterfly full back-half | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `128K/H16` | `128` | `128` | `16` | `4.442346 ms` | `4.427754 ms` | `0.997x` |
| `256K/H16` | `128` | `128` | `32` | `9.176986 ms` | `9.169259 ms` | `0.999x` |
| `512K/H16` | `256` | `128` | `32` | `17.347299 ms` | `17.367866 ms` | `1.001x` |

Representative losing rows:

| Case | `group_chunks` | `block_v` | `num_groups` | full-path ratio |
| --- | ---: | ---: | ---: | ---: |
| `128K/H16` | `32` | `64` | `64` | `1.061x` slower |
| `256K/H16` | `32` | `64` | `128` | `1.083x` slower |
| `256K/H16` | `32` | `128` | `128` | `1.017x` slower |
| `512K/H16` | `128` | `128` | `64` | `1.003x` slower |

### Intra-CP-Segment Prefix Probe, 2026-06-26

Important correction: the previous butterfly migration changed the global
group-start scan in the old grouped path. The human suggestion was sharper:
apply prefix/butterfly inside each CP segment, where FlashQLA-style fused
forward still runs a shorter but sequential recurrence.

New experiment:

```text
ablate_inner_prefix_cp.py
```

Schedule:

```text
production CP initial-state construction
  -> reshape each CP segment as a batch item
  -> group_transition_summary_bthd inside the segment
  -> butterfly prefix over inner groups, initialized from cp_h0
  -> h_grouped_replay_bthd
  -> output_o_bthd
```

This introduces two explicit tuning dimensions:

- `max_local_chunks`: CP segment length
- `inner_group_chunks`: prefix group size inside each CP segment

The helper computes group starts as:

```text
group_start[0] = cp_h0
group_start[g] = A_prefix[g-1] @ cp_h0 + b_prefix[g-1]
```

Output files:

```text
results/flashqla_specialized/ablate_inner_prefix_cp_sweep64k_20260626.jsonl
results/flashqla_specialized/ablate_inner_prefix_cp_long_20260626.jsonl
results/flashqla_specialized/ablate_inner_prefix_cp_fused_sweep64k_20260626.jsonl
results/flashqla_specialized/ablate_inner_prefix_cp_fused_long_20260626.jsonl
```

64K/H16 parameter sweep, `block_v=128`:

| `max_local_chunks` | `inner_group_chunks` | inner groups | fused CP baseline | inner prefix path | Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `64` | `4` | `16` | `0.448742 ms` | `3.702323 ms` | `8.25x` slower |
| `64` | `8` | `8` | `0.451366 ms` | `2.814595 ms` | `6.24x` slower |
| `64` | `16` | `4` | `0.446518 ms` | `2.412995 ms` | `5.40x` slower |
| `64` | `32` | `2` | `0.449862 ms` | `2.253059 ms` | `5.01x` slower |
| `64` | `64` | `1` | `0.449114 ms` | `2.211523 ms` | `4.92x` slower |
| `128` | `4` | `32` | `0.439267 ms` | `4.017680 ms` | `9.15x` slower |
| `128` | `8` | `16` | `0.437603 ms` | `2.972093 ms` | `6.79x` slower |
| `128` | `16` | `8` | `0.437610 ms` | `2.496000 ms` | `5.70x` slower |
| `128` | `32` | `4` | `0.439757 ms` | `2.290954 ms` | `5.21x` slower |
| `128` | `64` | `2` | `0.438931 ms` | `2.332717 ms` | `5.31x` slower |

Longer-sequence confirmation using the best real-prefix point
(`max_local_chunks=64`, `inner_group_chunks=32`, `block_v=128`):

| Case | CP segments | inner groups | fused CP baseline | inner prefix path | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `128K/H16` | `32` | `2` | `0.876533 ms` | `4.600725 ms` | `5.25x` slower |
| `256K/H16` | `64` | `2` | `1.705739 ms` | `8.787744 ms` | `5.15x` slower |

Fused replay/output follow-up:

The separated version above still used:

```text
h_grouped_replay_bthd -> output_o_bthd
```

So we added an experimental fused replay/output kernel:

```text
fused_group_replay_output_bthd
```

It keeps the same inner prefix starts, then computes `v_new`, output `o`, and
the segment final state in one kernel, avoiding global `states/v_new`
materialization.

Focused 64K/H16 sweep, `block_v=128`:

| `max_local_chunks` | `inner_group_chunks` | inner groups | fused CP baseline | separated inner prefix | fused inner prefix | Fused ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `64` | `16` | `4` | `0.449699 ms` | `2.408502 ms` | `2.149280 ms` | `4.78x` slower |
| `64` | `32` | `2` | `0.449197 ms` | `2.261155 ms` | `2.017898 ms` | `4.49x` slower |
| `64` | `64` | `1` | `0.450102 ms` | `2.212115 ms` | `2.085475 ms` | `4.63x` slower |
| `128` | `16` | `8` | `0.441168 ms` | `2.497341 ms` | `2.238301 ms` | `5.07x` slower |
| `128` | `32` | `4` | `0.439136 ms` | `2.303078 ms` | `2.061571 ms` | `4.69x` slower |
| `128` | `64` | `2` | `0.438499 ms` | `2.343562 ms` | `2.011453 ms` | `4.59x` slower |

Longer-sequence confirmation, `max_local_chunks=64`, `inner_group_chunks=32`,
`block_v=128`:

| Case | CP segments | inner groups | fused CP baseline | separated inner prefix | fused inner prefix | Fused ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `128K/H16` | `32` | `2` | `0.879221 ms` | `4.613995 ms` | `4.108288 ms` | `4.67x` slower |
| `256K/H16` | `64` | `2` | `1.707477 ms` | `8.811872 ms` | `7.780523 ms` | `4.56x` slower |

Boundary-prepass follow-up:

The fused replay/output version still paid for full affine summaries and a
butterfly prefix over dense `(A, b)` objects. A lighter AKO candidate removed
those matrices:

```text
group_start_prepass_bthd
  -> fused_group_replay_output_bthd
```

`group_start_prepass_bthd` starts from `cp_h0`, runs only the recurrent h update
to each group boundary, and writes `group_start`. This duplicates some h work
but removes global `(A,b)` summary materialization, butterfly ping-pong buffers,
and prefix-stage synchronization.

Output files:

```text
results/flashqla_specialized/ablate_inner_prefix_cp_prepass_sweep64k_20260626.jsonl
results/flashqla_specialized/ablate_inner_prefix_cp_prepass_long_20260626.jsonl
```

64K/H16 sweep, `block_v=128`:

| `max_local_chunks` | `inner_group_chunks` | inner groups | fused CP baseline | fused inner prefix | prepass + fused | Prepass ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `64` | `16` | `4` | `0.450234 ms` | `2.149773 ms` | `1.424746 ms` | `3.16x` slower |
| `64` | `32` | `2` | `0.449933 ms` | `2.011261 ms` | `1.426086 ms` | `3.17x` slower |
| `64` | `64` | `1` | `0.448880 ms` | `2.087805 ms` | `1.439462 ms` | `3.21x` slower |
| `128` | `16` | `8` | `0.439142 ms` | `2.241638 ms` | `1.422989 ms` | `3.24x` slower |
| `128` | `32` | `4` | `0.438899 ms` | `2.060928 ms` | `1.421773 ms` | `3.24x` slower |
| `128` | `64` | `2` | `0.438704 ms` | `2.008838 ms` | `1.433715 ms` | `3.27x` slower |

Longer-sequence confirmation, `max_local_chunks=64`, `inner_group_chunks=32`,
`block_v=128`:

| Case | CP segments | inner groups | fused CP baseline | fused inner prefix | prepass + fused | Prepass ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `128K/H16` | `32` | `2` | `0.871659 ms` | `4.105280 ms` | `3.104363 ms` | `3.56x` slower |
| `256K/H16` | `64` | `2` | `1.698229 ms` | `7.739328 ms` | `5.543723 ms` | `3.26x` slower |

This is the best prefix-insertion candidate so far. It validates the user's
diagnosis: the earlier prefix variants were dominated by the way prefix was
inserted into the pipeline, not by the associative formulation itself.
However, even the lighter prepass still runs an extra recurrence pass before
the fused replay/output pass. That duplicate recurrence is enough to remain
about `3.2-3.6x` slower than FlashQLA-style fused CP.

Correctness note: the experimental path injects `cp_h0` through the existing
fp16 grouped-replay path, while production fused CP consumes fp32 initial
states. The observed diffs are small but nonzero (`o_max_abs` around
`0.02-0.026`, `final_max_abs` around `0.01`), so this is a schedule/performance
probe rather than a production-correct candidate. The fused replay/output and
prepass follow-ups have the same diff profile as the separated version.

Conclusion:

- The affine composition is mathematically valid and has now been migrated to a
  TileLang-native helper.
- The TileLang helper is much more reasonable than the PyTorch prototype, and
  the correct intra-CP placement has now been tested. Current tuning still does
  not justify replacing production fused CP. With `block_v=128`, butterfly can
  beat dense scan in scan-only timing, but the full grouped back-half
  improvement is at most about `0.3%` in the global grouped-path rows and
  disappears on the 512K probe. In the true intra-CP-segment experiment, the
  separated prefix/replay/output path is about `5-9x` slower than the fused CP
  baseline.
- The likely reason is structural: butterfly reduces dependency depth but adds
  multiple kernel launches and maintains dense `A` and `b` ping-pong buffers.
  In the intra-CP version, fusing replay/output recovers part of FlashQLA's data
  reuse and improves the probe by about `10-13%`, but the schedule still pays
  for separate summary and prefix stages and loses the tight producer/consumer
  pipeline of FlashQLA's fused forward. Replacing summary/prefix with a
  boundary prepass is much better, but it still duplicates recurrence work
  before replay/output.
- The current production win still comes from FlashQLA-style CP-split/fused
  replay plus TileOps' A/KKT producer and dispatch integration. The earlier
  TileOps grouped prefix-scan path remains an important pre-FlashQLA
  algorithmic exploration, but not the final production schedule. A future
  prefix attempt would need to compute boundary state and output in one
  producer/consumer pipeline, or use CTA/cluster-level cooperation. A separate
  boundary pass is still too expensive.

Follow-up: Goal 6 cluster2 fused boundary probe

We then tested the most direct version of "put the boundary state inside the
fused kernel" using TileLang cluster launch:

```text
rank 0: replay/output first half of each CP segment, write h_mid
cluster_sync
rank 1: replay/output second half from h_mid, write final_state
```

This proved that TileLang cluster launch and `T.cluster_sync()` work in the
current docker, and that boundary state can move inside one fused cluster
kernel. It did not produce a production candidate:

| Case | fused CP baseline | prepass + fused | cluster2 fused | cluster2 / baseline |
| --- | ---: | ---: | ---: | ---: |
| `64K/H16` | `0.441037 ms` | `1.429757 ms` | `1.632618 ms` | `3.70x` slower |
| `128K/H16` | `0.860301 ms` | `3.102362 ms` | `2.973792 ms` | `3.46x` slower |
| `256K/H16` | `1.682035 ms` | `5.546595 ms` | `5.904765 ms` | `3.51x` slower |

The result sharpens the diagnosis: simply avoiding the external prepass kernel
is not enough. A two-CTA relay still serializes the real recurrence, doubles
CTA resources, and leaves the second CTA mostly idle. A worthwhile
hierarchical-prefix design would need rank 1 to do useful independent work
before the boundary state arrives, for example producing its local affine
transition or preparing local replay data, then composing with the received
state inside the same pipeline.

Follow-up: prefetching one chunk during the cluster wait

We tried a small improvement to the cluster2 relay: let rank 1 prefetch the
first chunk of the second half and compute its local `QK`/masked attention
before `h_mid` arrives. This fills part of the waiting window but does not
change the recurrence dataflow.

| Case | fused CP baseline | prepass + fused | cluster2 fused | cluster2 prefetch1 |
| --- | ---: | ---: | ---: | ---: |
| `64K/H16` | `0.446864 ms` | `1.431763 ms` | `1.621213 ms` | `1.577485 ms` |
| `128K/H16` | `0.859216 ms` | `3.103302 ms` | `2.958755 ms` | `2.878912 ms` |
| `256K/H16` | `1.683514 ms` | `5.543277 ms` | `5.894278 ms` | `6.299027 ms` |

The result is useful precisely because it is not a win. Rank-1 prefetch gives a
small positive signal on 64K/128K, but it reverses at 256K. Sweeping shorter
CP segments also failed to close the gap. This means the missing piece is not
just "do something while waiting"; the prefix/scan idea has to be co-designed
with the producer-side transition computation and the replay/output pipeline.
Otherwise it adds shared/register pressure without removing the serial state
progression that output still needs.

Transition formula check

To make that statement concrete, we verified the chunk-level affine transition
used by the grouped replay recurrence:

```text
h_next = M_chunk @ h + b_chunk

M_chunk = exp(g_last) * I - exp(2 * g_last) * K^T @ W
b_chunk = K^T @ (U * exp(g_last - g_i))
```

Using TileOps production `chunk_cumsum -> blocksolve_A -> recompute_w_u`
outputs, a PyTorch probe reproduced direct grouped replay exactly:

| Case | chunks | max abs |
| --- | ---: | ---: |
| `4K/H1` | `64` | `0.0` |
| `8K/H1` | `128` | `0.0` |

This gives the right next abstraction: a fused hierarchical-prefix attempt
should let the later CTA compute a local transition representation
`(M_local, b_local)` while waiting for the incoming boundary state. But the
full representation is large: `M_local` is `128 x 128`, and `b_local` is
`128 x block_DV`. For `block_DV=128`, that is already about `64 KiB` in fp16
before q/k/w/u/output buffers. This is why prefix cannot be treated as an
isolated scan add-on; the producer, transition representation, and
replay/output pipeline must be designed together.

Transition feasibility checkpoint

We checked whether this transition-aware route is immediately blocked by
storage or compute:

| Probe | Result |
| --- | --- |
| cluster2 base buffers, `block_DV=128` | `120.125 KiB` shared estimate |
| cluster2 + full `(M,b)`, `block_DV=128` | `184.125 KiB` shared estimate |
| cluster2 + prefetch1 + full `(M,b)`, `block_DV=128` | `256.250 KiB` shared estimate |
| TileLang shared-memory smoke, `block_DV=128`, full `(M,b)` | passes |
| TileLang shared-memory smoke, `block_DV=128`, prefetch1 + full `(M,b)` | passes |

So the first blocker is not simply "M and b cannot fit". The more useful proxy
is existing TileLang `prepare_h`, which already computes `ht/mt` with a
split-left/right `M_local` strategy. Running it over the second half
(`warmup_chunks=32`) gives:

| Case | prepare_h transition proxy |
| --- | ---: |
| `64K/H16` | `0.223462 ms` |
| `128K/H16` | `0.402826 ms` |
| `256K/H16` | `0.763808 ms` |

This is not free, but it is small enough to be interesting if overlapped with
first-half replay/output. It is too expensive as a separate stage. The remaining
hard part is output: a transition representation gives the final state, but
each output chunk still needs the corresponding incoming state. Therefore a
real hierarchical-prefix kernel would need local prefix states inside the
cluster or an output decomposition that can be prepared before `h_mid` arrives
and corrected afterward.

Output decomposition

We verified such an output decomposition:

```text
o_chunk = local_o_chunk + L_chunk @ h_in

local_o_chunk = attn @ U
L_chunk = Q * exp(g_i) - (attn * exp(g_j + g_last)) @ W
```

This reproduced direct replay/output up to numerical noise:

| Case | output max abs | final max abs |
| --- | ---: | ---: |
| `4K/H1` | `3.73e-09` | `0.0` |
| `8K/H1` | `5.12e-09` | `0.0` |

So the complete transition-aware view is:

```text
state:  h_out = M @ h_in + b
output: o     = local_o + L @ h_in
```

But the storage story matters. For a half segment of 32 chunks, full-half
`local_o` and `L` would require roughly `1 MiB` at `block_DV=128`, far beyond a
reasonable shared-memory design. A one-chunk streaming version is feasible:
`local_o_chunk + L_chunk` is about `32 KiB` at `block_DV=128`, and a TileLang
shared-memory smoke with `(M,b)` plus one output chunk passes. This pushes the
next design toward streaming correction, not full-buffered prefix.

Full-transition TileLang feasibility

We then tested the actual full-DK transition building blocks rather than only
their storage footprint. `probe_tilelang_transition_gemm.py` computes
`M_chunk,b_chunk`; `probe_tilelang_transition_compose2.py` computes two chunk
transitions and composes them as `M_1 @ M_0`, `M_1 @ b_0 + b_1`.

Under the TileOps nightly TL0.1.9 docker on GPU4/H200 with L2-flush event
timing:

| Case | transition producer | compose2 |
| --- | ---: | ---: |
| `4K/H1` | `0.050800 ms` | `0.091440 ms` |
| `64K/H16` | `0.323859 ms` | `0.496781 ms` |
| `128K/H16` | `0.619904 ms` | `0.963866 ms` |
| `256K/H16` | `1.219091 ms` | `1.886419 ms` |

This is an important refinement. Full affine transition is not blocked by
TileLang lowering, but the naive prefix work is already expensive enough that a
separate summary/prefix/replay pipeline would mostly add overhead. The
butterfly/prefix-scan idea is therefore still mathematically valid, but it has
to be fused into the producer/replay pipeline so that loads, QK/local attention,
W/U reads, and output correction are reused.

Pair2 transition-state replay candidate

We also tested a first fused replay/output candidate that actually replaces the
state update with the affine transition formula inside the output kernel. For
each two-chunk group it keeps the normal direct output computation, but updates
the state as:

```text
h_next = M_chunk @ h + b_chunk
```

This candidate is numerically exact against the direct pair2 fused
replay/output path on the tested rows, but it is not a performance win:

| Case | block_DV | direct pair2 | transition-state pair2 | Ratio |
| --- | ---: | ---: | ---: | ---: |
| `4K/H1` | 128 | `0.111072 ms` | `0.079440 ms` | `0.72x` |
| `64K/H16` | 128 | `0.919469 ms` | `1.380762 ms` | `1.50x` |
| `128K/H16` | 128 | `1.802298 ms` | `2.716294 ms` | `1.51x` |
| `256K/H16` | 128 | `3.592605 ms` | `5.415293 ms` | `1.51x` |
| `64K/H16` | 64 | `1.292342 ms` | `2.098592 ms` | `1.62x` |

This distinguishes two claims. The affine transition is a correct way to
describe the recurrence. But using full `M_chunk @ h` as the per-chunk update is
the wrong insertion point because it adds `K^T W`, `K^T U`, and `M @ h` without
removing enough work from the output path. A useful hierarchical-prefix design
should keep the cheap direct recurrence inside a small group and use transition
composition only across group boundaries where it can remove long sequential
dependencies.

Boundary summary fused with direct replay

We then tested the more plausible insertion point: keep direct replay/output
inside a two-chunk group, but also produce the group transition summary in the
same kernel. This fused kernel is exact against direct pair2 output/final-state
and exact against the existing group-summary kernel.

The performance result is sobering:

| Case | direct pair2 | summary only | separate direct+summary | fused direct+summary |
| --- | ---: | ---: | ---: | ---: |
| `64K/H16` | `0.918112 ms` | `1.245443 ms` | `2.146794 ms` | `2.106118 ms` |
| `128K/H16` | `1.804797 ms` | `2.461706 ms` | `4.256531 ms` | `4.186080 ms` |

Fusing saves only about two percent versus running direct and summary as
separate kernels. The full augmented summary path is still larger than the
direct replay/output path because it carries `DV + DK` columns. So the blocker
is not primarily launch overhead or duplicated global loads; it is the full
transition representation itself. The next design has to reduce summary count,
summary width, or both.

Group-size sweep

We swept the fused direct+summary candidate over larger groups on `64K/H16`:

| group_chunks | direct replay/output | summary only | fused direct+summary | fused/direct |
| ---: | ---: | ---: | ---: | ---: |
| 2 | `0.918112 ms` | `1.245443 ms` | `2.106118 ms` | `2.29x` |
| 4 | `0.910726 ms` | `1.114778 ms` | `1.981616 ms` | `2.18x` |
| 8 | `0.897315 ms` | `1.071347 ms` | `1.957696 ms` | `2.18x` |
| 16 | `0.886045 ms` | `1.060659 ms` | `1.950458 ms` | `2.20x` |

And checked `128K/H16`, `group_chunks=16`:

```text
direct = 1.745930 ms
fused direct+summary = 4.111075 ms
fused/direct = 2.35x
```

So tuning group size alone does not rescue the full `[b | M]` representation.
The summary count drops, but the kernel still runs the recurrence over the
large augmented state. This is the practical reason to keep FlashQLA-style
CP-split as the production path and treat hierarchical prefix as future work
unless we find a narrower transition representation.

## What We Know

- TileOps full-op is faster on the reported rows.
- TileOps A/KKT producer is faster and algorithmically different.
- Existing component attribution shows TileOps fused replay/output event is
  also shorter on `64K/H16`, `128K/H16`, and `128K/H32`.
- TileOps fused replay source is largely a port of FlashQLA's skeleton, not an
  independently invented back-half algorithm.

## What We Do Not Yet Know

- How much of the fused replay gap comes from TileLang version/lowering.
- How much comes from compile flags and `gemm.h` compatibility.
- How much comes from `block_DV` selection or hidden dispatch metadata.
- Whether a migrated FlashQLA implementation in the exact TileOps TL0.1.11
  environment would close the fused replay gap.
- Whether generated PTX/SASS differs in tensor-core/store/barrier structure in
  a way that explains the gap.

## Blog-Safe Wording

Use this:

```text
FlashQLA provided the CP-split schedule and fused replay skeleton. In our
component attribution runs, the TileOps instantiation of that back half also
ran faster than the public FlashQLA TL0.1.8 kernel on the measured H16/H32
rows. Because the source skeletons are very close, we treat that gap as an
implementation/lowering/dispatch result, not as a new high-level replay
algorithm.
```

Avoid this:

```text
We designed a better h/replay algorithm than FlashQLA.
```

## Evidence Needed Before Stronger Claims

1. Rerun TileOps and FlashQLA component attribution with explicit metadata:
   `max_local_chunks`, CP segment count, `block_DV`, TileLang version, and
   commit hash.
2. Run the TL0.1.9/TL0.1.11 FlashQLA migration, if viable, with the same
   component timer.
3. Archive generated CUDA/PTX or lowering summaries for the fused replay
   kernel on the same row.
4. Compare only matched rows where CP split and `block_DV` are aligned.
