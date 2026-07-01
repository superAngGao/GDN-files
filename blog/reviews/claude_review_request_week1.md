# Claude Review Request: Week 1 Planning And Evidence

We are preparing a public tutorial/report about AI-assisted TileLang kernel
optimization, using our Gated DeltaNet prefill work in TileOps as the case
study.

Please review the attached materials directly, not only this summary. The goal
is to get detailed, actionable feedback before we continue with the next
experiment block.

## Materials To Review

Please read these files in order:

1. `planning.md`
   - Current positioning, algorithm decomposition, optimization narrative,
     experiment plan, and multi-agent review workflow.
2. `drafts/tutorial.md`
   - Early tutorial skeleton with first evidence inserted.
3. `data/week1_experiment_log.md`
   - Curated summary of the Week 1 benchmark/profiler/ablation results.
4. `REPRODUCE.md`
   - Reproduction commands and measurement setup.
5. `code/benchmarks/bench_h_recurrence_scale_vnew_ablation.py`
   - Exact BTHD h-recurrence scale-placement ablation script.

Optional raw data files if you want to inspect exact JSONL records:

- `data/fair_fla_comparison_week1_gpu4_1500_20260618.jsonl`
- `data/bottleneck_profile_week1_gpu4_1500_20260618.jsonl`
- `data/scale_vnew_ablation_week1_gpu4_1500_20260618.jsonl`

## Current Positioning

This is not meant to be only a Gated DeltaNet performance report. The intended
public article is a tutorial-style case study for TileOps/TileLang developers:

- start from a correct FLA/Qwen-compatible BTHD prefill interface
- profile the real bottleneck
- use AI-assisted AKO for local kernel search and hypothesis pruning
- combine that with human mathematical insight
- turn successful ideas into guarded production dispatch
- preserve negative results as part of the method

Interface/layout alignment is treated as an engineering precondition, not the
main optimization story.

## Current Evidence Collected

The current evidence is summarized in `data/week1_experiment_log.md`, but the
main results are:

Fair BTHD TileOps vs FLA comparison on H200, SM locked to 1500 MHz, fp16,
`B=1`, `DK=DV=128`, `chunk64`:

| Seq len | Heads | TileOps BTHD | FLA BTHD | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 32K | 16 | 2.8165 ms | 3.1320 ms | 1.11x |
| 64K | 16 | 5.5506 ms | 6.5554 ms | 1.18x |
| 128K | 16 | 11.2403 ms | 13.4351 ms | 1.20x |
| 128K | 32 | 18.6650 ms | 21.0881 ms | 1.13x |

Public-op profiler:

- H16/S32K is h-recurrence dominated.
- H16/S128K shows the scan decomposition: transition summary, dense group-start
  scan, grouped replay.

Exact BTHD h-recurrence scale-placement ablation:

| Variant | H-only latency | Speedup vs old |
| --- | ---: | ---: |
| old: scale `k` into `k_scaled_s` | 2.2725 ms | 1.00x |
| current: scale `v_new` in place | 1.6277 ms | 1.40x |

Correctness:

- `v_new_max_abs_vs_current = 0.0`
- `final_state_max_abs_vs_current = 3.05e-5`

## Review Scope

Please review broadly. You are not limited to the questions below. We especially
want detailed, actionable feedback on:

- behavioral logic of the optimization workflow
- narrative structure and whether the reader can follow the technical causality
- language style and whether the tone is appropriate for a public
  TileOps/TileLang tutorial
- experimental validity, fairness, missing controls, and possible overclaims
- clarity of the AI-assisted development story
- whether the attribution between agent-discovered local optimization and human
  mathematical/algorithmic insight is fair and precise
- concrete edits we should make to the planning document or tutorial skeleton
- specific experiments or tables that should be added, removed, or reframed

## Specific Questions

1. Is the tutorial positioning clear and compelling for the TileLang/TileOps
   developer community?
2. Are we overclaiming anywhere based on the evidence so far?
3. Which missing experiment is most important before drafting the first public
   version?
4. Is the three-part optimization narrative correct, or should another
   engineering optimization be elevated?
5. What should we be careful about when explaining AI contribution vs human
   mathematical insight?
6. Does `drafts/tutorial.md` have the right shape for a public article, or does
   it currently read too much like an internal report?
7. Are the reproduction instructions in `REPRODUCE.md` sufficient for a reader
   who has access to a compatible TileOps environment?

## Desired Output Format

Please organize your review into:

1. High-level verdict
2. Major issues or risks, ordered by severity
3. Concrete suggested edits to the narrative
4. Concrete suggested edits to the experiment plan
5. Comments on language/style
6. Any optional ideas that are useful but not blocking

Please be specific. If you recommend changing a claim, quote or paraphrase the
claim and suggest a replacement.
