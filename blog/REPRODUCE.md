# Reproducing The Current Measurements

This document records the measurement setup used for the first tutorial data
pass. Commands assume a TileOps checkout containing PR #1596 mounted as
`/workspace/TileOPs` and this repository mounted as `/workspace/blog`.

## Environment

- GPU: NVIDIA H200
- Locked clocks for reported data: SM 1500 MHz, memory 3201 MHz
- Docker image: `tileops-runner-sshd:nightly-tl019-fullstack-no-tileops-ldfix-registered-tmux`
- Timing method: `benchmarks.benchmark_base.bench_kernel`
- Default stable timing policy: 10 warmup, 100 repeat, 3 trials
- Layout: BTHD
- Dtype: float16

## Fair FLA Comparison

```bash
for cfg in \
  "16 32768" \
  "16 65536" \
  "16 131072" \
  "32 131072"; do
  set -- $cfg
  python /workspace/blog/code/benchmarks/bench_prefill_bthd_op_vs_fla.py \
    --batch 1 --heads "$1" --seq-len "$2" \
    --chunk-size 64 --dim-k 128 --dim-v 128 --dtype float16 \
    --warmup 10 --repeat 100 --trials 3 \
    --output-final-state \
    --output /workspace/blog/data/fair_fla_comparison_week1_gpu4_1500_20260618.jsonl
done
```

## Public-Op Profiler Breakdown

```bash
for cfg in \
  "16 32768" \
  "16 131072"; do
  set -- $cfg
  python /workspace/blog/code/benchmarks/profile_prefill_bthd_op_torch_profiler.py \
    --batch 1 --heads "$1" --seq-len "$2" \
    --chunk-size 64 --dim-k 128 --dim-v 128 --dtype float16 \
    --warmup 3 --top-k 30 \
    --output /workspace/blog/data/bottleneck_profile_week1_gpu4_1500_20260618.jsonl
done
```

## H-Recurrence Scale-Placement Ablation

```bash
python /workspace/blog/code/benchmarks/bench_h_recurrence_scale_vnew_ablation.py \
  --batch 1 --heads 16 --seq-len 32768 \
  --chunk-size 64 --dim-k 128 --dim-v 128 --dtype float16 \
  --block-v 16 --h-num-stages 2 --h-threads 128 \
  --warmup 10 --repeat 100 --trials 3 \
  --output /workspace/blog/data/scale_vnew_ablation_week1_gpu4_1500_20260618.jsonl
```
