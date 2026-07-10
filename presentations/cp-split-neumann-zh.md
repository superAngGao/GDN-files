# CP Split 和 Neumann Prepare：内部讲解提纲

目标：给同事讲清楚 Gated DeltaNet prefill 中两个关键 search-space expansion：

1. **CP split**：为什么它解决的是 replay 依赖链形状问题。
2. **Blocked-inverse / Neumann-style prepare**：为什么它解决的是 prepare-A producer 的并行形状问题。

建议时长：25-35 分钟。  
建议定位：不是 benchmark review，而是机制讲解。数字只做锚点。

---

## 0. 一句话主线

GDN prefill 的困难不是“算一个 GEMM”，而是：

> 我们想并行处理很多 token，但输出和 final_state 必须等价于 token-by-token decode。

这带来两条长依赖：

- replay/output 的跨 chunk 状态依赖；
- chunk 内 prepare-A 的因果修正依赖。

CP split 改 replay 依赖链。  
Neumann / blocksolve 改 prepare-A 的计算形状。

---

## 1. Slide: GDN prefill 为什么难

### 图

```text
decode view:

h0 -> token0 -> token1 -> token2 -> ... -> tokenT

prefill wants:

many tokens / chunks in parallel
but same o and same final_state
```

### 讲法

GDN 有 recurrent memory。每个 token 会读旧状态、写 residual，再影响后面的 token。

所以 prefill 不能只把所有 token 当成普通 attention 矩阵乘。它必须保留 causal state 语义。

我们优化的核心不是“消灭因果性”，而是改变因果性在哪个阶段、以什么形状被计算。

---

## 2. Slide: local AKO 的墙

### 图

```text
local optimization:

scale placement
store path
small fusion

helps local cost
but replay chain is still:

h0 -> chunk0 -> chunk1 -> chunk2 -> ... -> chunkN
```

### 讲法

Agent 很擅长在固定 contract 下做局部搜索：

- scale 放在 key 还是 value 路径；
- accumulator 怎么写 shared/global；
- 哪些中间结果可以少 materialize。

这些是有效优化，但它们不改变 replay 的长依赖链。  
这就是 local wall。

### 关键句

> less materialization 不等于 shorter recurrence。

---

## 3. Slide: CP split 的核心想法

### 图

```text
without CP split:

h0 -> chunk0 -> chunk1 -> chunk2 -> chunk3 -> ... -> chunkN

with CP split:

          corrected h_start[0] -> segment0 short replay
prepare -> corrected h_start[1] -> segment1 short replay
          corrected h_start[2] -> segment2 short replay
```

### 讲法

CP split 的关键不是“每个 segment 天然独立”。  
它们不天然独立。

关键是先计算每个 segment 应该从什么状态开始：

```python
h_start = correct_segment_starts(segment_summaries, h0)
```

一旦 `h_start[segment]` 正确，每个 segment 内部只需要跑短链。

### 关键句

> CP split does not remove causality. It moves part of causality into segment-start correction.

---

## 4. Slide: CP split 伪代码

### 伪代码

```python
# serial replay: one long dependency chain
h = h0
for chunk in chunks:
    o[chunk], h = replay_output(chunk, h)

# CP split: compute valid segment starts first
h_start = correct_segment_starts(segment_summaries, h0)

for segment in segments:       # independent once h_start is known
    h = h_start[segment]
    for chunk in segment:      # short local recurrence
        o[chunk], h = fused_replay_output(chunk, h)
```

### 讲法

这改变了后端看到的工作形状：

- 原来是一条很长的 serial replay；
- 现在是 preprocess/correction 加多个短 replay；
- fused replay/output kernel 可以在更短 segment 上工作。

这就是 FlashQLA 对我们故事的关键贡献：提供了 CP-split replay schedule family。

---

## 5. Slide: CP split 对 attribution 的含义

### 表

| 问题 | CP split 说明了什么 | CP split 没说明什么 |
| --- | --- | --- |
| replay wall | 长 replay 依赖需要 schedule-level 改写 | 不是普通 fusion 能解决 |
| FlashQLA credit | schedule family 来自 FlashQLA | 不是说 TileOps 直接调用 FlashQLA |
| TileOps contribution | 重建、适配、dispatch、验证 | 不是“发明 CP split” |

### 讲法

我们对外必须区分：

- schedule idea 的来源；
- TileOps owned implementation；
- final dispatch surface；
- A producer 的进一步改进。

---

## 6. Slide: 为什么还需要 prepare-A

CP split 解决 replay 依赖链形状，但 replay 需要有效写入。

在 chunk 内，后面的 token 会受到前面 residual write 的影响。  
所以 prepare 阶段需要构造一个 chunk-local correction：

```text
raw k/v/beta/g
  -> prepare-A
  -> effective writes W, U
  -> replay/output consumes W, U / A / g_cum
```

如果 prepare-A producer 慢，整个 CP-split pipeline 仍然受限。

---

## 7. Slide: chunk-local correction 的数学形状

### 逻辑视图

在实现约定下，可以把 chunk 内相互作用写成一个严格下三角矩阵：

```math
M_{i,j} =
\begin{cases}
\beta_i \exp(g_i - g_j)\langle k_i, k_j\rangle, & i > j, \\
0, & i \le j .
\end{cases}
```

有效写入形如：

```math
A = (I + M)^{-1}
```

### 讲法

这里不要把公式说成唯一 ABI。  
实际 production CP path 会拆 factor：

- materialized `A` 使用 `g_zero` convention；
- `g_cum` 单独传给 replay；
- correctness claim 是 full `o` / `final_state`，不是所有中间 A 完全相同。

---

## 8. Slide: 为什么是 Neumann

因为 `M` 是严格下三角矩阵。

严格下三角矩阵是 nilpotent：

```math
M^C = 0
```

所以：

```math
(I + M)^{-1}
= I - M + M^2 - M^3 + \cdots + (-1)^{C-1}M^{C-1}
```

### 讲法

这不是随机近似，也不是无限级数截断的猜测。  
在固定 chunk 长度 `C` 内，它是有限的，因为严格下三角结构保证 `M^C = 0`。

### 关键句

> Neumann 视角不是为了少算一个数学算子，而是暴露一个可 block、可并行的逆/更新结构。

---

## 9. Slide: blocksolve 的计算形状

### 图

```text
chunk64
  -> 4 blocks of 16 tokens

lower block structure:

[B0  0   0   0 ]
[L10 B1  0   0 ]
[L20 L21 B2  0 ]
[L30 L31 L32 B3]
```

### block recurrence

```math
A_{r,r} = B_r^{-1}
```

```math
A_{r,s} =
-B_r^{-1}\sum_{m=s}^{r-1} L_{r,m}A_{m,s}, \quad r > s
```

### 讲法

把 `64 x 64` 的 causal correction 拆成四个 `16 x 16` block。  
对角 block 做 local inverse / Neumann-style update。  
off-diagonal block 用固定的小 GEMM 组合出来。

---

## 10. Slide: 计算量对比，不是“少算一切”

### 数字

以 `chunk64, DK=128` 为例：

| 项 | MACs per chunk/head |
| --- | ---: |
| full dense `64 x 64` Gram | `524,288` |
| strict causal off-diagonal interaction | `258,048` |
| lower triangular including diagonal | `266,240` |
| TileOps 10 个 dense `16 x 16` Gram block | `327,680` |
| TileOps block inverse/composition tail | `98,304` |
| TileOps prepare-A GEMM-shaped work total | `425,984` |

### 讲法

TileOps blocksolve 不一定减少总 MAC。  
它比严格下三角 interaction 多做一些 dense block 内计算。

优势在于：

- 工作变成 regular GEMM-shaped blocks；
- dependency shape 更适合 GPU backend；
- 小 block composition 是固定结构，容易调度。

### 关键句

> 这里的收益是 scheduling/backend-shape win，不是 abstract FLOP reduction。

---

## 11. Slide: 和 FlashQLA-style forward solve 的关系

### 数字

只比较 solve/combine tail：

| Producer tail | MACs per chunk/head |
| --- | ---: |
| FlashQLA-style forward solve/combine tail | `89,600` |
| TileOps blocked-inverse / Neumann-style tail | `98,304` |

### 讲法

直觉上，TileOps tail 算术量略大，这不奇怪。

它要形成可复用的 blocked inverse / composition。  
代价是 tail MAC 稍多。  
收益是并行形状和 backend friendliness。

---

## 12. Slide: 三个 A/replay rows 怎么读

### 表

| Row | 含义 | 用途 |
| --- | --- | --- |
| Public FlashQLA full path | public FlashQLA TL0.1.8 full path | external context |
| FlashQLA-style A on TileOps replay | TL0.1.8-lowered KKT A/g + TileOps replay | 控制 replay，观察 FlashQLA-style A |
| TileOps blocked-inverse A on TileOps replay | TileOps Neumann/blocksolve A + 同一个 TileOps replay | 观察 A producer 改进 |

### 数字

```text
Public FlashQLA full path:                 1.306838 ms
FlashQLA-style A on TileOps replay:        0.8245 ms
TileOps blocked-inverse A on TileOps replay: 0.7474 ms
```

### 讲法

不要把这三行读成同一个 causal ladder。  
它们回答三个不同问题：

- public FlashQLA 是外部 context；
- middle row 说明 TileOps replay/output 已经很强；
- last row 说明在同一个 TileOps replay family 下，TileOps A producer 继续改进。

---

## 13. Slide: 最终一句话总结

CP split 和 Neumann 分别解决不同层级的问题：

| 问题 | 技术 | 改变了什么 |
| --- | --- | --- |
| replay 长链 | CP split | 把长 replay 改成 corrected starts + 短 segment replay |
| prepare-A 形状 | blocked-inverse / Neumann | 把 causal correction 改成 lower-blocked GEMM-shaped producer |

最终贡献不是某个孤立 speedup，而是：

> 一个可审计的 kernel optimization loop：operator contract 固定，search-space expansion 清楚，credit boundary 清楚，结果可通过 correctness / benchmark / attribution gates 检查。

---

## 14. 可能被问的问题

### Q1: CP split 是不是消除了因果性？

不是。它把部分因果性挪到 segment-start correction。  
segment 只有在 corrected initial state 正确时才可以并行/分段 replay。

### Q2: Neumann 是不是近似？

在这里不是随机近似。  
因为 chunk-local `M` 是严格下三角，`M^C=0`，所以 Neumann series 是有限的。

### Q3: 为什么算得更多还可能更快？

GPU 上不仅看 MAC 数，还看 dependency、数据布局、kernel shape、shared memory、compiler/backend 能不能生成高效路径。  
TileOps blocksolve 把 prepare-A 变成更 regular 的 small GEMM / block composition。

### Q4: materialized A 和公式里的 A 是同一个吗？

不是无条件同一个。  
公式是 operator-level / ABI-scoped 视图。  
production CP path 中 materialized `A` 使用 `g_zero` convention，`g_cum` 单独给 replay。

### Q5: FlashQLA 和 TileOps 的 credit 怎么讲？

FlashQLA 提供 CP-split schedule family。  
TileOps 做 owned implementation、A producer、dispatch、correctness、benchmark、evidence lane。

---

## 15. 建议白板顺序

如果只用白板讲，按这个顺序：

1. 画 decode recurrent chain。
2. 画 without CP split 的 chunk chain。
3. 画 with CP split 的 corrected segment starts。
4. 写 `M` 严格下三角。
5. 写 `M^C=0` 和有限 Neumann。
6. 画 `chunk64 -> 4 x 16` lower block matrix。
7. 写三行 A/replay comparison。

这样同事会先理解“为什么要改 search space”，再看数字。
