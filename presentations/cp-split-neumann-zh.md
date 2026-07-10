# CP Split 与 Neumann Prepare：算法和 GPU 实现讲解

目标：给同事讲清楚 Gated DeltaNet prefill 中两个核心实现问题：

1. **CP split** 如何把长 replay 依赖拆成 corrected segment starts + 短 segment replay。
2. **Blocked-inverse / Neumann-style prepare** 如何把 chunk 内因果修正变成更适合 GPU 的 block/GEMM 形状。

建议时长：25-35 分钟。  
建议定位：讲算法和 GPU 实现，避免展开项目叙事和 benchmark 证据。

---

## 0. 总览

Gated DeltaNet prefill 的核心约束：

> prefill 要并行处理长序列，但 `o` 和 `final_state` 必须等价于 token-by-token decode。

这会产生两个实现问题：

| 层级 | 问题 | 解决方向 |
| --- | --- | --- |
| 跨 chunk / segment | replay 是长状态依赖链 | CP split |
| chunk 内 | prepare-A 是因果三角修正 | blocked-inverse / Neumann-style prepare |

---

## 1. GDN prefill 的状态依赖

### 直觉图

```text
decode:

h0 -> token0 -> token1 -> token2 -> ... -> tokenT
```

每一步都依赖前一步状态：

```text
read old state
compute residual write
update state
produce output
```

prefill 的目标不是改语义，而是把等价计算换成更并行的形状。

---

## 2. 为什么直接 replay 会慢

把长序列切成 chunk 后，最朴素的 replay 仍然是一条长链：

```text
h0 -> chunk0 -> chunk1 -> chunk2 -> ... -> chunkN
```

每个 chunk 的 initial state 来自前一个 chunk 的 final state。  
这会限制 GPU 并行度：

- chunk 间不能自然并行；
- replay kernel 即使融合，也仍然受长依赖链限制；
- 减少 store 不等于缩短 recurrence。

---

## 3. CP split 的核心思想

CP split 的做法：

1. 先用 preprocess/correction 算出每个 segment 的正确起始状态；
2. 再让每个 segment 从自己的 corrected start state 开始做短 replay。

```text
without CP split:

h0 -> chunk0 -> chunk1 -> chunk2 -> chunk3 -> ... -> chunkN

with CP split:

          corrected h_start[0] -> segment0 short replay
prepare -> corrected h_start[1] -> segment1 short replay
          corrected h_start[2] -> segment2 short replay
```

关键点：

> segment 不是天然独立；它们因为 corrected initial state 正确，才可以分段 replay。

---

## 4. CP split 伪代码

### 朴素 serial replay

```python
h = h0
for chunk in chunks:
    o[chunk], h = replay_output(chunk, h)
```

### CP split replay

```python
# 1. 每个 segment/chunk 先给出 summary
segment_summaries = summarize_segments(k, v, A, g, beta)

# 2. correction step 计算每个 segment 的正确起始状态
h_start = correct_segment_starts(segment_summaries, h0)

# 3. 每个 segment 内部做短 replay
for segment in segments:          # segment 之间可以并行调度
    h = h_start[segment]
    for chunk in segment:         # segment 内仍然是短 recurrence
        o[chunk], h = fused_replay_output(chunk, h)
```

CP split 没有消除因果性。  
它把长链的一部分变成了 segment-start correction。

---

## 5. GPU 实现中的 CP split pipeline

可以把实现拆成三类 kernel / 阶段：

```text
input q/k/v/g/beta
  |
  v
prepare-A / effective writes
  |
  v
segment summary / prepare_h
  |
  v
correct segment starts
  |
  v
fused replay/output over short segments
  |
  v
o, final_state
```

每个阶段的 GPU 角色：

| 阶段 | GPU 上的工作形状 |
| --- | --- |
| prepare-A | chunk-local triangular/block computation |
| segment summary | per segment / per head summary state |
| correct starts | prefix/correction over segment summaries |
| fused replay/output | many shorter local replay chains |

---

## 6. corrected segment starts 为什么有效

假设有 4 个 segment：

```text
S0, S1, S2, S3
```

serial replay 需要：

```text
h_start[S0] = h0
h_start[S1] = final_state(S0)
h_start[S2] = final_state(S1)
h_start[S3] = final_state(S2)
```

CP split 的目标是直接构造：

```text
h_start[S0], h_start[S1], h_start[S2], h_start[S3]
```

使得每个 segment 的 replay 结果等价于 serial replay。

一旦 `h_start[Si]` 正确，segment 内部就只需要处理局部 chunk 链。

---

## 7. replay/output kernel 的并行收益

没有 CP split：

```text
one very long chain
low inter-segment parallelism
large dependency depth
```

有 CP split：

```text
many shorter chains
more CTAs / work units can be scheduled
dependency depth per replay worker is smaller
```

GPU 上的收益来自：

- 更短的 serial chain；
- 更好的 occupancy / scheduling opportunity；
- fused replay/output 可以专注局部 segment；
- correction 的成本被 preprocess 阶段吸收。

---

## 8. 为什么 prepare-A 仍然关键

CP split 解决 replay 的跨 segment 依赖，但 replay 需要有效写入。

chunk 内 token 之间仍然有因果修正：

```text
raw k/v/beta/g
  -> prepare-A
  -> effective writes / correction matrix
  -> replay/output
```

如果 prepare-A producer 慢，整个 CP-split pipeline 仍然会受限。

所以第二个问题是：

> 如何把 chunk-local causal correction 做成 GPU 友好的 producer？

---

## 9. chunk-local correction 的逻辑形状

在一个实现约定下，可以把 chunk 内交互看成严格下三角矩阵：

```math
M_{i,j} =
\begin{cases}
\beta_i \exp(g_i - g_j)\langle k_i, k_j\rangle, & i > j, \\
0, & i \le j .
\end{cases}
```

有效写入可以写成：

```math
A = (I + M)^{-1}
```

注意：这是逻辑视图。实际 CP path 的 ABI 会拆 factor：

- materialized `A` 使用 `g_zero` convention；
- `g_cum` 单独传给 replay；
- correctness 看 full `o` / `final_state`。

---

## 10. 为什么 Neumann 视角成立

`M` 是严格下三角矩阵，所以它是 nilpotent：

```math
M^C = 0
```

因此：

```math
(I + M)^{-1}
= I - M + M^2 - M^3 + \cdots + (-1)^{C-1}M^{C-1}
```

这不是无限近似。  
在 chunk 长度为 `C` 时，级数最多到 `M^{C-1}`，然后精确结束。

实现意义：

> 可以把 triangular inverse / correction 写成固定 block update 结构。

---

## 11. blocksolve 的分块结构

以 `chunk64` 为例，把 token 维切成 4 个 `16-token` block：

```text
chunk64 -> [block0, block1, block2, block3]
```

lower block matrix：

```text
[B0  0   0   0 ]
[L10 B1  0   0 ]
[L20 L21 B2  0 ]
[L30 L31 L32 B3]
```

对角 block：

```math
A_{r,r} = B_r^{-1}
```

非对角 block：

```math
A_{r,s} =
-B_r^{-1}\sum_{m=s}^{r-1} L_{r,m}A_{m,s}, \quad r > s
```

这就是 block inverse / block composition 的固定形状。

---

## 12. GPU 上 blocksolve 怎么跑

核心工作拆成两部分：

1. 计算 lower Gram blocks；
2. 组合 diagonal inverse 和 off-diagonal blocks。

```text
G00 = k0 @ k0.T
G10 = k1 @ k0.T
G11 = k1 @ k1.T
...
G33 = k3 @ k3.T
```

然后：

```text
B0, B1, B2, B3 -> local inverse
L10, L20, ...  -> off-diagonal composition
```

GPU 友好点：

- 主要形状是 `16 x 16` dense block；
- 可以用 regular GEMM-shaped computation；
- off-diagonal composition 是固定小 GEMM 序列；
- 比 token-by-token forward substitution 更适合并行后端。

---

## 13. MAC accounting：不是少算一切

以 `chunk64, DK=128` 为例：

| 项 | MACs per chunk/head |
| --- | ---: |
| full dense `64 x 64` Gram | `524,288` |
| strict causal off-diagonal interaction | `258,048` |
| lower triangular including diagonal | `266,240` |
| TileOps 10 个 dense `16 x 16` Gram block | `327,680` |
| TileOps block inverse/composition tail | `98,304` |
| TileOps prepare-A GEMM-shaped work total | `425,984` |

解释：

- blocksolve 比严格下三角 interaction 做了更多 dense block 内计算；
- 但它远少于 full dense grid；
- 更重要的是，它把工作变成 regular block/GEMM shape。

---

## 14. 和 forward solve 的差异

只看 solve/combine tail：

| Producer tail | MACs per chunk/head |
| --- | ---: |
| FlashQLA-style forward solve/combine tail | `89,600` |
| TileOps blocked-inverse / Neumann-style tail | `98,304` |

TileOps tail 算术量略大。  
这符合直觉：形成 reusable blocked inverse / composition 需要额外组合。

但它的实现形状更 regular：

```text
fixed small GEMM sequence
shared-memory block composition
less forward-substitution-shaped dependency
```

---

## 15. CP split + Neumann 的组合方式

整体数据流：

```text
q/k/v/g/beta
  |
  | prepare-A producer
  v
A, g_cum, beta
  |
  | CP preprocess / corrected segment starts
  v
h_start[segment]
  |
  | fused replay/output over short segments
  v
o, final_state
```

两个技术分别管不同问题：

| 技术 | 管的问题 | 输出给下一阶段 |
| --- | --- | --- |
| Neumann/blocksolve prepare | chunk 内 causal correction | `A` / effective writes |
| CP split | segment 间 replay dependency | corrected `h_start` |
| fused replay/output | segment 内短链 replay | `o`, `final_state` |

---

## 16. prepare-A 到 replay 的接口

CP split replay 不需要知道 A 是怎么来的。它只需要看到满足 ABI 的输入：

| 张量 / 元数据 | 来源 | replay 怎么用 |
| --- | --- | --- |
| `A` | prepare-A producer | chunk-local causal correction |
| `g_cum` | chunk-local gate prefix | replay 中处理 gate/decay |
| `beta` | 原始输入 | write strength / residual update |
| `q/k/v` | 原始输入 | output read 和 state update |
| `initial_state` / `h_start` | CP preprocess / correction | 每个 segment 的正确起始状态 |
| `cp_seq_map`, `cu_seqlens` 等 metadata | CP preprocess | 指示 segment replay 的 token 范围和调度方式 |

这就是为什么 prepare-A producer 可以替换：

```text
prepare-A producer
  -> A, g_cum, beta, metadata
  -> fused replay/output
```

只要接口语义一致，replay/output kernel 可以保持同一个调度族。

---

## 17. 最终总结

一句话：

> CP split 改 replay 的依赖形状；Neumann/blocksolve 改 prepare-A 的计算形状。

更具体地说：

| 问题 | 原始形状 | 改后形状 |
| --- | --- | --- |
| replay | 一条长 chunk chain | corrected starts + 多个短 segment replay |
| prepare-A | chunk 内三角因果修正 | lower-blocked GEMM-shaped producer |

GPU 实现收益来自：

- dependency depth 变短；
- work units 更容易分段调度；
- prepare-A 变成 regular block/GEMM shape；
- fused replay/output 只处理短 segment。

---

## 18. 常见问题

### Q1: CP split 是不是消除了因果性？

不是。它把跨 segment 的因果性挪到 corrected segment starts。

### Q2: segment 为什么可以并行？

因为 correction 已经算出了每个 segment 的正确初始状态。  
没有这个 corrected state，segment 不独立。

### Q3: Neumann 是近似吗？

这里不是随机近似。  
chunk 内 `M` 严格下三角，所以 `M^C=0`，级数有限终止。

### Q4: blocksolve 为什么可能更快？

不是因为 abstract FLOP 更少，而是因为它把工作变成更适合 GPU 的 block/GEMM shape。

### Q5: materialized A 和公式 A 完全相同吗？

不一定。公式是逻辑视图。  
production CP path 中 `A` 使用 `g_zero` convention，`g_cum` 单独传给 replay。

---

## 19. 白板讲解顺序

1. 画 serial replay 长链。
2. 画 CP split 的 corrected segment starts。
3. 画 pipeline：prepare-A -> correct starts -> fused replay/output。
4. 写严格下三角 `M`。
5. 写 `M^C=0` 和有限 Neumann。
6. 画 `chunk64 -> 4 x 16` lower block matrix。
7. 写 MAC accounting：不是少算一切，而是 GPU shape 更好。
8. 画 prepare-A 到 replay 的张量接口。
