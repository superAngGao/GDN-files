# Review Request: tutorial_v3 Checkpoint 2

Please review the second checkpoint of the rewritten tutorial:

- `/home/ga/2026-06-15/gdn-prefill-ai-assisted-blog/drafts/tutorial_v3.md`

Context documents:

- `/home/ga/2026-06-15/gdn-prefill-ai-assisted-blog/reviews/writer_handoff_v8.md`
- `/home/ga/2026-06-15/gdn-prefill-ai-assisted-blog/drafts/blog_plan_v3.md`
- `/home/ga/2026-06-15/gdn-prefill-ai-assisted-blog/drafts/kernel_code_shapes_20260629.md`

This checkpoint includes:

- opening / thesis;
- Level 1: correct, measurable operator;
- Level 2: autonomous local AKO inside a fixed contract.

Level 2 currently covers:

- scale placement in replay;
- recompute/store-path diagnostic;
- direct fusion wall: less materialization is not shorter recurrence.

Please focus on:

1. Does Level 2 preserve the boundary between local AKO and Level 3 external
   search-space changes?
2. Does the scale-placement example feel concrete without becoming too
   implementation-heavy?
3. Does the recompute section clearly show that the store path, not just GEMM,
   mattered?
4. Does the fusion section make the `less materialization != shorter
   recurrence` lesson intuitive?
5. Are historical diagnostics scoped safely, without becoming final benchmark
   claims?
6. Does the bridge into human Neumann/blocksolve and Qwen FlashQLA feel earned?
7. Is anything missing before writing Level 3?

Known constraints:

- Do not ask for exact Neumann formulas yet; Level 3 will introduce them with a
  schematic/exact-notation caveat.
- Do not ask for final benchmark rows yet; final publication still requires a
  Tier-1 refresh.
- FlashQLA attribution must remain explicit when Level 3 is added.
