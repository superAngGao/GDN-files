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

## Polish Priorities

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

### P0 - Neumann / Blocked-Inverse Section

- [ ] Review the math against the implementation one more time.
- [ ] Distinguish schematic formulas from implementation-tied formulas.
- [ ] Explain the problem it solves before presenting the block algorithm.
- [ ] Make clear why this is a search-space change, not a local code tweak.
- [ ] Preserve the caveat that TileOps does not claim materialized `A` equality
      with the generic exact/KKT-style producer.

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

### P1 - Evidence Hygiene

- [ ] Ensure every main-text number appears in either `tutorial_v3_si.md` or the
      evidence summary.
- [ ] Keep raw JSONL links in repo-relative form under `../../evidence/ladder`.
- [ ] Keep internal variant IDs out of the public roadmap table.
- [ ] Keep `FLA 0.5.1` wording conservative unless package identity is verified.
- [ ] Keep FlashQLA comparisons labeled as public-environment comparisons unless
      the row is explicitly a controlled same-input / same-lowering experiment.

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

## Done Definition

The tutorial is ready for final external review when:

- the main text has no large evidence dumps;
- `tutorial_v3_si.md` contains the detailed evidence needed to audit the story;
- all roadmap values are traceable and consistently scoped;
- FlashQLA CP-split credit is explicit;
- human expert Neumann / blocked-inverse insight is explicit;
- figures are either final assets or clearly marked placeholders;
- no internal variant ID is required to understand the main story.
