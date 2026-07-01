# Review Request: tutorial_v3 Checkpoint 3

Please review the third checkpoint of the rewritten tutorial:

- `/home/ga/2026-06-15/gdn-prefill-ai-assisted-blog/drafts/tutorial_v3.md`

Context documents:

- `/home/ga/2026-06-15/gdn-prefill-ai-assisted-blog/reviews/writer_handoff_v8.md`
- `/home/ga/2026-06-15/gdn-prefill-ai-assisted-blog/drafts/blog_plan_v3.md`
- `/home/ga/2026-06-15/gdn-prefill-ai-assisted-blog/drafts/kernel_code_shapes_20260629.md`

This checkpoint includes:

- opening / thesis;
- Level 1: correct, measurable operator;
- Level 2: autonomous local AKO inside a fixed contract;
- Level 3: external input changes the search space.

Level 3 currently covers:

- attribution map;
- FlashQLA CP-split schedule and corrected segment starts;
- human blocked inverse / Neumann-style prepare framing;
- production dispatch and benchmark metadata.

After the main Level 3 path, the checkpoint also covers guardrail lessons:

- migration/lowering lesson;
- hierarchical prefix negative result;

Please focus on:

1. Does Level 3 keep the attribution boundaries clear?
2. Does the FlashQLA CP-split section clearly credit Qwen and explain why
   `prepare_h/correct_h0` is required?
3. Is the Neumann/blocksolve section useful while remaining safely schematic?
4. Does the productionization section make dispatch metadata feel technically
   necessary rather than bureaucratic?
5. Do the migration/lowering and prefix sections feel properly placed as
   guardrail lessons after the main path?
6. Does the migration/lowering section avoid unsupported generated-code
   claims?
7. Is the prefix negative result correctly scoped to the tested
   `DK=DV=128`, `chunk64` production path?
8. Are any claims too strong before the final benchmark refresh?
9. What should be changed before writing final benchmark and conclusion
   sections?

Known constraints:

- Exact Neumann/blocksolve notation remains schematic unless verified against
  implementation.
- Current benchmark tables are candidate evidence snapshots, not final public
  claims.
- TileOps vs FlashQLA must remain a public-environment comparison, not a
  controlled same-lowering replay attribution experiment.
