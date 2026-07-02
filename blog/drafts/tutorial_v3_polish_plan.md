# Tutorial V3 Polish Plan

This plan tracks the remaining work needed to turn `tutorial_v3.md` from a
reviewable draft into a publishable tutorial. The main rule is simple:

```text
Main text carries the story. SI carries the detailed evidence.
```

Canonical files:

| File | Role |
| --- | --- |
| `tutorial_v3.md` | Main tutorial draft. Keep narrative, roadmap, core formulas, figures, and takeaways here. |
| `tutorial_v3_si.md` | Supporting information. Keep detailed evidence tables, rejected rows, ABI caveats, and attribution diagnostics here. |
| `../../evidence/ladder/docs/variant_inventory.md` | Maintains variant-to-code / variant-to-evidence mapping. |
| `../../evidence/ladder/summaries/blog_ladder_evidence_64k_h16.md` | Writing-facing evidence summary for the main 64K/H16 ladder. |

## Current State

The tutorial now has the intended high-level structure:

1. Roadmap and evidence contract.
2. Operator understanding.
3. Level 1: make the operator measurable.
4. Level 2: local AKO inside a fixed contract.
5. Level 3: external input changes the search space.
6. Takeaways.

The first roadmap table now uses story-level nodes rather than internal variant
IDs. Latency and performance columns are public-facing, while exact kernel /
variant mapping is maintained in SI and evidence inventory.

External review status: P0 is closed as of `f003792`, and P1 is now underway.
The opening now starts with problem/result/core insight, the roadmap is
split into controlled rows / anchors / production surface, high-risk formula
wording is schematic, and intermediate-A details are kept out of the main
claim. Remaining work is P1/P2 polish: stronger agentic episodes, error
reporting, reproduction entry, publication figures, and final compression.

## Polish Priorities

### P0 - Opening And Reader Mental Model

- [x] Rewrite the opening so the first screen gives:
      problem, result, and core insight.
- [x] Move dense evidence-management details out of the opening. Keep only a
      short "How to read the numbers" callout after the reader understands the
      problem.
- [x] Avoid putting hash, GPU id, historical worktree, publication blocker, or
      internal checkpoint language before the operator mental model.
- [x] Add an early one-sentence mental model:
      GDN prefill is hard because it wants parallel throughput while preserving
      a recurrent KV memory that must match token-by-token decode.
- [ ] Each major section should open with a result-oriented sentence, for
      example:
      "This section explains why local fusion helped less than expected."

### P0 - Narrative Consistency

- [ ] Read `tutorial_v3.md` end to end and remove any remaining internal
      checkpoint language.
- [ ] Ensure every performance claim has one of these scopes:
      controlled TileOps row, public-environment FlashQLA comparison,
      component diagnostic, or rejected diagnostic.
- [ ] Keep FlashQLA attribution explicit:
      Qwen FlashQLA supplied the production CP-split schedule family.
- [ ] Keep human expert attribution explicit:
      human expert insight supplied the blocked-inverse / Neumann-style prepare
      search-space change.
- [ ] Keep V5 / first CP adaptation out of the headline roadmap; mention it only
      as process evidence in SI.
- [x] Replace "public FlashQLA SOTA" with "public FlashQLA TL0.1.8 anchor" or
      "public FlashQLA baseline" unless the article defines a strict SOTA
      benchmark scope.
- [ ] Reduce repeated attribution disclaimers. Keep one strong CP-split
      non-originality statement in the main text, then rely on figures/tables and
      SI for detail.

### P0 - Benchmark Table Structure

- [x] Do not make one visual table look like a single causal speedup ladder if it
      mixes controlled rows, external anchors, and production-surface rows.
- [x] Split the current roadmap evidence into reader-friendly blocks:
      1. single-shape controlled narrative rows;
      2. public FLA / FlashQLA external anchors;
      3. production dispatch surface sweep.
- [ ] Main text should show only table rows that directly support the narrative.
      Put internal row mapping and non-headline process rows in SI.
- [ ] Every performance table must state whether the comparison is:
      controlled same-input TileOps, public-environment FlashQLA, component
      diagnostic, or rejected diagnostic.

### P0 - Neumann / Blocked-Inverse Section

- [ ] Review the math against the implementation one more time.
- [x] Distinguish schematic formulas from implementation-tied formulas.
- [ ] Explain the problem it solves before presenting the block algorithm.
- [ ] Make clear why this is a search-space change, not a local code tweak.
- [ ] Preserve the caveat that TileOps does not claim materialized `A` equality
      with the generic exact/KKT-style producer.
- [x] Fix the publication-blocking formula in Section 4.2. In particular, the
      schematic lower-triangular term currently written like
      `M[i, j] = beta[i] * exp(g[i] - g[j]) * ...` must include the missing key
      interaction / Gram term or be rewritten as intentionally schematic.
- [x] Remove or resolve any wording that says row/column orientation or gate
      exponent placement "should be checked" before publication. Published text
      should either be implementation-verified or explicitly schematic.

### P0 - Intermediate A Mismatch Explanation

- [x] Add a main-text callout or short subsection:
      "Why intermediate A mismatch is acceptable here."
- [x] Explain that V5/V6 materialized `A` tensors are not claimed equivalent,
      because the validated claim is full-op correctness under the chosen ABI.
- [x] State what is checked: output, final state, dtype, tolerance, input
      distribution, and reference path.
- [x] Move detailed `A allclose=false`, max-abs, max-rel, and producer ABI facts
      to SI, but make the main text understandable enough that readers do not
      suspect tolerance is hiding an error.

### P1 - Figures

Replace Mermaid placeholders or prepare publication-quality redraws for:

| Figure | Purpose | Status |
| --- | --- | --- |
| AKO gated loop | Show hypothesis -> edit -> correctness -> benchmark -> lowering -> decision log. | Placeholder |
| Logical GDN prefill decomposition | Show prepare / replay / output / final state boundaries. | Placeholder |
| Local replay wall vs CP split | Show why fusion alone does not shorten replay, and why CP split changes dependency depth. | Placeholder |
| FlashQLA schedule -> TileOps replay adaptation -> Neumann prepare | Show attribution and search-space expansion sequence. | Needed |
| Production dispatch surface | Show shape metadata -> dispatch policy -> selected kernel -> benchmark metadata. | Placeholder |
| Prefix negative result | Show dependency-depth benefit vs full-transition representation cost. | SI / optional |

The figures should explain strategy changes, not decorate the text.

### P1 - Agentic Evidence

- [x] Add one or two concrete AKO episodes so "Agentic Kernel Optimization" is
      not just branding.
- [x] Each episode should include:
      hypothesis, TileLang/code pattern changed, correctness gate, benchmark
      result, and accept/reject decision.
- [ ] Good candidate episodes:
      scale placement / buffer removal;
      store-path diagnostic;
      direct fusion rejected by the replay wall;
      V5 as first correct but incomplete FlashQLA-inspired adaptation.
- [ ] Show where human intervention changed the search space, rather than
      implying the agent discovered every global direction alone.

Current status: main text now includes two compact Level-2 AKO episodes
(`scale placement` and `direct fusion`) using the hypothesis/edit/gate/decision
shape, with repo-relative evidence pointers and scoped equivalence wording for
the scale-placement case. Further polish may add one Level-3 episode or move
detailed episode evidence to SI.

### P1 - Evidence Hygiene

- [ ] Ensure every main-text number appears in either `tutorial_v3_si.md` or the
      evidence summary.
- [ ] Keep raw JSONL links in repo-relative form under `../../evidence/ladder`.
- [ ] Keep internal variant IDs out of the public roadmap table.
- [ ] Keep `FLA 0.5.1` wording conservative unless package identity is verified.
- [ ] Keep FlashQLA comparisons labeled as public-environment comparisons unless
      the row is explicitly a controlled same-input / same-lowering experiment.
- [ ] Add an error-reporting table or SI section with:
      max abs, max rel, p99 error if available, final-state error, output dtype,
      input distribution, and tolerance.
- [ ] Every benchmark table should state:
      whether latency is min/median/mean, whether it is kernel-only, whether
      allocation/layout conversion is included, synchronization policy, clock
      setting, warmup/repeat/trials, and GPU.
- [ ] Any TMA/WGMMA/PTX/SASS claim must link to generated-code evidence or be
      downgraded to source-level schedule wording.
- [ ] Keep PR numbers, GPU3/GPU4 details, internal variant IDs, and dirty-worktree
      audit fields out of the main narrative unless they are essential.

### P1 - Terminology And Mapping

- [ ] Add a compact producer/replay/environment table, preferably in SI and
      optionally summarized in the main text:

| Term | Producer | Replay/output | Environment | Allowed claim |
| --- | --- | --- | --- | --- |
| public FlashQLA anchor | FlashQLA TL0.1.8 KKT | FlashQLA TL0.1.8 CP replay | public-env Docker | schedule/performance reference |
| TL0.1.8-lowering + TileOps replay | TL0.1.8-lowered KKT via external launcher | TileOps replay/output | external-lowering harness | no-Neumann combined row |
| TileOps Neumann row | blocked-inverse / Neumann-style prepare | TileOps replay/output | TileOps harness | prepare improvement under same replay family |
| production dispatch surface | TileOps production A producer | selected TileOps production replay path | production wrapper | dispatchable kernel-family claim |

- [ ] Ensure terms like `FlashQLA-style A`, `TL0.1.8-lowering KKT`,
      `TileOps prepare A`, and `blocked-inverse / Neumann-style blocksolve A`
      are introduced once and then used consistently.

### P1 - Code Shape Snippets

Keep code snippets short and structural. The tutorial should include only the
code shapes needed to explain:

- how the measurable operator is decomposed;
- how the AKO loop gates candidates;
- how CP split changes the replay schedule;
- how blocked inverse / Neumann-style prepare changes the producer;
- how dispatch metadata makes production results auditable.

Avoid pasting full kernels in the main text. Larger snippets belong in SI or
separate code-shape notes.

### P2 - Language And Layout

- [ ] Make the roadmap table readable on a narrow screen.
- [ ] Add a compact glossary if terms like `CP split`, `A producer`,
      `TL0.1.8-lowering`, and `public-env` still feel dense.
- [ ] Replace "Level 1/2/3" wording when it appears without context.
- [ ] Smooth transitions:
      local AKO wall -> FlashQLA schedule reference -> human expert prepare
      insight -> production surface.
- [ ] Final pass for tone: avoid "AI magic", avoid overstating autonomy, and
      avoid reducing expert/community contributions to implementation details.
- [ ] Replace Mermaid placeholders with publication figures before external
      release, or clearly mark the rendered page as a draft.
- [ ] Reduce duplicate wording around "not inventing FlashQLA" and "not just
      copying FlashQLA"; retain the strongest statement once and support it with
      evidence.

### P2 - Reproducibility Entry

- [x] Add a "Reproduce the headline rows" section or SI entry.
- [x] Include environment, commit, data artifact, command, expected output, and
      where raw JSONL is stored.
- [ ] Keep the reproduction path minimal: one command for the main 64K/H16 rows
      and one command for the five-shape production surface, if practical.

Current status: `tutorial_v3_si.md` now has `SI.3.6 Reproduce The Headline Rows`
with archived raw JSONL paths and a rerun command for the formal `64K/H16`
harness rows. The five-shape production surface is archived as raw JSONL; a
single-command public rerun remains a tooling TODO.

## Review Gates

Use these review checkpoints before publication:

| Gate | Reviewer question |
| --- | --- |
| Narrative review | Does the article read as a coherent tutorial rather than an experiment log? |
| Attribution review | Are FlashQLA, human expert insight, and TileOps implementation contributions clearly separated? |
| Evidence review | Can every number in the main text be traced to SI / evidence files? |
| Math review | Are Neumann / blocked-inverse formulas accurate and properly scoped? |
| Figure review | Do figures explain strategy changes without adding misleading claims? |
| Publication caveat review | Are FLA identity, FlashQLA public-env comparisons, and PTX/TMA/WGMMA claims properly guarded? |
| Reader review | Can an external kernel engineer understand the mental model before seeing hashes, row names, and caveats? |
| Reproducibility review | Can a reader find the command and artifact behind the headline rows? |

## Done Definition

The tutorial is ready for final external review when:

- the main text has no large evidence dumps;
- `tutorial_v3_si.md` contains the detailed evidence needed to audit the story;
- all roadmap values are traceable and consistently scoped;
- roadmap / benchmark evidence is split so controlled rows, public anchors, and
  production-surface rows do not visually collapse into one causal ladder;
- FlashQLA CP-split credit is explicit;
- human expert Neumann / blocked-inverse insight is explicit;
- Neumann / blocked-inverse formulas are implementation-checked or explicitly
  schematic;
- intermediate `A` mismatch is explained clearly enough for an external reader;
- correctness error reporting is available in SI;
- reproduction instructions exist for headline rows;
- figures are either final assets or clearly marked placeholders;
- no internal variant ID is required to understand the main story.
