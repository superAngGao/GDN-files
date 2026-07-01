# AI-Assisted TileLang Kernel Optimization: GDN Prefill

This repository collects material for a tutorial-style report on optimizing a
Gated DeltaNet prefill kernel in TileOps with an AI-assisted engineering loop.

The goal is not only to publish final latency numbers. The goal is to document
a workflow:

- start from a correct FLA/Qwen-compatible BTHD prefill interface
- profile the real bottleneck
- use agent-driven AKO to search local kernel transformations
- use human mathematical insight for algorithmic changes
- keep production dispatch guarded by measured shape behavior

The current case study centers on three optimization themes:

1. moving the h-recurrence scale from `k` to `v_new`
2. analyzing the prepare stage as a chunk-local triangular-system problem
3. applying an associative prefix scan to long-sequence chunk recurrence

## Layout

```text
code/benchmarks/  Benchmark and profiling scripts used by the tutorial
code/kernels/     Standalone kernel snippets, if needed for the article
data/             Raw JSONL measurements and curated experiment logs
drafts/           Planning notes and tutorial drafts
figures/          Diagrams and generated plots
references/       Paper and implementation links
scripts/          Summarization and plotting utilities
```

The main planning document is [planning.md](planning.md). The first experiment
summary is [data/week1_experiment_log.md](data/week1_experiment_log.md).
