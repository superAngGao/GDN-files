# CP Split 与 Neumann Prepare：算法和 GPU 实现说明

这份材料只讨论 Gated DeltaNet prefill 中两个实现问题：

1. CP split 如何把跨 chunk 的长 replay 依赖改造成 corrected segment starts 加短 segment replay。
2. blocked-inverse / Neumann-style prepare 如何把 chunk 内因果三角求解改造成更适合 GPU 的 block computation。

不讨论 local AKO 过程、benchmark story 或 agent 叙事。这里的目标是给同事建立算法和 GPU 实现层面的 mental model。

---

## 1. 背景：Chunkwise GDN Prefill 分成哪些步骤

Gated DeltaNet decode 的语义是一条状态链：

```text
h0 -> token0 -> token1 -> token2 -> ... -> tokenT
```

每一步都会读取旧状态、计算 residual write、更新 state，并产生当前 token 的输出。Prefill 的目标不是改变这个语义，而是在保持 `o` 和 `final_state` 等价于 token-by-token decode 的前提下，把计算改造成更适合 GPU 并行执行的形状。

chunkwise prefill 先做了一层重写：它把 token 级 recurrence 压到 chunk 内部，用 chunk-local correction 表达 token 之间的因果关系。

```text
token-level:
h0 -> token0 -> token1 -> ... -> tokenT

chunkwise:
h0 -> chunk0 -> chunk1 -> ... -> chunkN
```

从实现上看，chunkwise GDN prefill 可以拆成三类工作：

```text
1. prepare-A / effective writes
   处理 chunk 内 token-token causal correction

2. replay / state update
   从 chunk 的 h_start 出发，更新 recurrent state

3. output read
   用 q 读取 replay 得到的 state / residual contribution
```

这三类工作对应的主要输入不同：

| 阶段 | 主要输入 | 主要输出 | 后面对应的问题 |
|---|---|---|---|
| prepare-A | `K, g, beta`，以及 chunk 内 interaction | chunk-local `A` / effective writes | Neumann / blocksolve |
| replay/state update | `K, V, A, g, beta, h_start` | chunk states / final state | CP split |
| output read | `Q, K, A, g` 和 replay states | `o` | fused replay/output |

更具体地说，对一个 chunk 可以用下面的 schematic 表达：

```text
inputs:
    Q, K, V, g, beta, h_start

prepare-A:
    L[i,j] = beta[i] * gate(i,j) * <K[i], K[j]>,  i > j
    A      = (I + L)^(-1)

effective write / residual:
    W = scale_g_beta(A @ K)      # [C, C] @ [C, DK] -> [C, DK]
    U = scale_g_beta(A @ V)      # [C, C] @ [C, DV] -> [C, DV]

replay/state update:
    H[0]   = h_start
    H[t+1] = decay(g[t]) * H[t] + outer(W[t], U[t])
           = decay(g[t]) * H[t] + W[t][:, None] @ U[t][None, :]

output read:
    o[t] = Q[t] @ H[t] + local_residual(Q[t], K, U, A, g)
```

这里的 `scale_g_beta`、`decay(g)`、`gate(i,j)` 是实现约定下的折叠写法。不同
ABI 可以把 beta / gate factor 放在 `A`、effective write 或 replay 中不同位置；
本文只需要看清三类 workload 的矩阵乘加形状。

所以本文后面两个优化点分别对应：

- CP split：处理 replay 的跨 chunk / segment 状态依赖；
- Neumann / blocksolve：处理 prepare-A 的 chunk 内三角求解。

chunkwise 已经解决了 chunk 内大量细粒度 token 依赖，但没有解决跨 chunk 的状态依赖：

```text
chunk i 的 h_start = chunk i-1 的 h_final
```

因此，朴素 replay 仍然是一条很长的 chunk 链：

```text
h0 -> chunk0 -> chunk1 -> chunk2 -> ... -> chunkN
```

这会限制 GPU 并行度：chunk 间不能自然并行，replay kernel 即使融合，也仍然受长依赖链限制；减少 store 不等于缩短 recurrence depth。

---

## 2. Segment 的 Affine Summary

CP split 的出发点是：chunk / segment 对 recurrent state 的作用不是任意函数，
而是可以写成 affine transition。因此多个 chunk / segment 的状态转移可以组合。

对一个 chunk 或 segment，可以把它对输入 state 的作用写成：

```text
H_out = H_local + M @ H_in
```

其中：

| 符号 | 含义 |
|---|---|
| `H_in` | segment 的起始 recurrent state |
| `H_local` | 假设 `H_in = 0`，本 segment 自己写出的 state |
| `M` | 本 segment 对输入 state 的传播、衰减和修正 |
| `H_out` | segment 结束后的 state |

两个相邻 transition 可以合成。假设：

```text
T0(H) = B0 + M0 @ H
T1(H) = B1 + M1 @ H
```

那么：

```text
T1(T0(H))
= B1 + M1 @ (B0 + M0 @ H)
= (B1 + M1 @ B0) + (M1 @ M0) @ H
```

也就是说，合成以后仍然是同一种形式：

```text
T10(H) = B10 + M10 @ H

B10 = B1 + M1 @ B0
M10 = M1 @ M0
```

这个结合结构是缩短递推依赖链的核心。我们不必把所有 chunk 都完整 replay
一遍才能知道后面 segment 的起点；可以先把一段压缩成 `(H_local, M)`，
再在这些 summary 上做 correction / prefix。

因此每个 segment 先生成一个 summary：

```text
summary[s] = (H_local[s], M[s])
```

`H_local` 和 `M` 的依赖不同：

| summary | 表示什么 | 主要依赖 |
|---|---|---|
| `H_local` | 本 segment 新写入的 state | `K, V, A, g, beta` |
| `M` | 输入 state 如何传到段尾 | `K, A, g, beta` |

`H_local` 需要 `V`，因为它描述“本 segment 写入了什么内容”。`M` 描述旧 state 如何被本 segment 传播到段尾，通常不需要 `V`，主要由 key、chunk-local correction、gate 和 beta 决定。

同一个 segment 内，`H_local` 和 `M` 的计算仍然要按 chunk 顺序累计：

```text
chunk0 transition -> chunk1 transition -> ... -> chunk31 transition
```

但不同 segment 可以并行计算自己的 summary：

```text
segment0: chunk0  -> ... -> chunk31
segment1: chunk32 -> ... -> chunk63
segment2: chunk64 -> ... -> chunk95
...
```

---

## 3. CP Split 的整体 Pipeline

有了 segment affine summary 这个对象后，CP split 的实现流程就比较清楚了。prefill path 可以看成下面几个阶段：

```text
q/k/v/g/beta
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

关键点仍然是：segment 不是直接独立执行，而是先生成 `(H_local, M)` summary，再 correction 出正确 `h_start`，最后才做 segment-local replay/output。

---

## 4. Corrected Segment Starts

有了每个 segment 的 affine summary 后，就可以在 segment 粒度上计算 corrected starts：

```text
h_start[0] = h0

h_start[1] = H_local[0] + M[0] @ h_start[0]
h_start[2] = H_local[1] + M[1] @ h_start[1]
h_start[3] = H_local[2] + M[2] @ h_start[2]
...
```

也就是：

```text
h_start[s + 1] = H_local[s] + M[s] @ h_start[s]
```

这一步仍然有因果性，但它只在 segment summary 上传播，而不是在所有 token/chunk 的完整 replay 上传播。

拿到 corrected starts 后，replay 可以分段执行：

```text
h_start[0] -> segment0 short replay
h_start[1] -> segment1 short replay
h_start[2] -> segment2 short replay
```

所以 CP split 的语义边界是：

> segment 不是天然独立；它们因为 corrected start 正确，才可以分段 replay。

---

## 5. CP Split 带来的并行和代价

以 64K/H16/chunk64 为例：

```text
T = 65536
chunk_size = 64
num_chunks = 1024
max_local_chunks = 32
CP segments = 32
chunks/segment = 32
```

原始 replay 是一条 1024-chunk 的长链。CP split 后，依赖形状变成：

```text
32 个 segment summary 短链
+ 32 个 segment summary 上的 correction scan
+ 32 个 segment-local replay 短链
```

并行主要来自两个地方：

1. 不同 segment 的 summary 可以先并行准备。
2. 每个 segment 拿到自己的 corrected `h_start` 后，可以独立执行短 replay。

它不是 work-efficient 的免费优化。CP split 会多做一遍 summary work：

```text
summary pass:
    计算 H_local / M，用来得到 corrected h_start

replay/output pass:
    从 corrected h_start 出发，再跑 segment-local replay/output
```

因此一部分 `K/V/A/g/beta` 相关的 state-transition work 会重复。CP split 的收益不是减少所有计算量，而是用更多 work 换更短 critical path 和更大的 GPU work surface。

```text
work 增加
critical path 缩短
GPU work surface 变大
```

---

## 6. 为什么 Prepare-A 仍然关键

CP split 改的是 replay 的跨 segment 依赖形状，但 replay 仍然需要 chunk 内有效写入：

```text
raw k/v/beta/g
  -> prepare-A
  -> effective writes / correction matrix
  -> replay/output
```

如果 prepare-A producer 慢，整个 CP-split pipeline 仍然会受限。因此第二个问题是：如何把 chunk-local causal correction 做成 GPU 友好的 producer。

在一个逻辑视图中，chunk 内交互可以写成严格下三角矩阵：

```math
M_{i,j} =
\begin{cases}
\beta_i \exp(g_i - g_j)\langle k_i, k_j\rangle, & i > j, \\
0, & i \le j .
\end{cases}
```

有效写入需要：

```math
A = (I + M)^{-1}
```

这里 `M` 是 strictly lower triangular。实际 production CP path 会按 ABI 拆 factor，例如 materialized `A` 可以使用 `g_zero` convention，`g_cum` 单独传给 replay；这里的公式用于说明 chunk 内因果修正的逻辑形状。

---

## 7. 串行 Forward Solve

从：

```math
(I + M)A = I
```

可以逐行解出 `A`：

```text
A0 = e0
A1 = e1 - M10 A0
A2 = e2 - M20 A0 - M21 A1
...
Ai = ei - sum_{j<i} Mij Aj
```

这是一种 forward substitution 形状：

```text
row0 -> row1 -> row2 -> ... -> row63
```

行内可能有并行，但行之间存在细粒度顺序依赖。对于 GPU kernel 来说，这种形状不如固定 block GEMM / block composition 规则。

---

## 8. Neumann 视角

因为 `M` 是 strictly lower triangular，它是 nilpotent：

```math
M^C = 0
```

其中 `C` 是 chunk 长度。因此：

```math
(I + M)^{-1}
= I - M + M^2 - M^3 + \cdots + (-1)^{C-1}M^{C-1}
```

这不是无限近似。在有限 chunk 内，级数会精确截断。Neumann 视角的意义是：triangular inverse 可以被改写成固定的矩阵乘加结构。

不过，直接对完整 `64 x 64` 矩阵做展开仍然不一定是最好的 GPU 形状。实现里进一步把 chunk64 分成 4 个 16-token block。

---

## 9. 分块结构

把 `chunk64` 切成 4 个 16-token block：

```text
chunk64 = [B0, B1, B2, B3]
```

对应的 block lower-triangular matrix：

```text
T =
[ D0   0    0    0
  L10  D1   0    0
  L20  L21  D2   0
  L30  L31  L32  D3 ]
```

其中 `D0..D3` 是每个 16-token block 内部的 causal lower-triangular 结构，`L10, L20, ...` 是 block 之间的 lower interaction。因为未来 block 不影响过去 block，所以上三角为 0。

逆矩阵仍然是 block lower triangular：

```text
X = T^{-1}

X =
[ X00  0    0    0
  X10  X11  0    0
  X20  X21  X22  0
  X30  X31  X32  X33 ]
```

---

## 10. 对角块求逆

对角块满足：

```text
X00 = D0^{-1}
X11 = D1^{-1}
X22 = D2^{-1}
X33 = D3^{-1}
```

每个 `Di` 都是：

```text
Di = I + strictly_lower
```

所以每个 `Di^{-1}` 可以用有限 Neumann 展开：

```math
D_i^{-1} = I - L_i + L_i^2 - \cdots + (-1)^{15}L_i^{15}
```

这四个对角块逻辑上彼此独立。实现里对应 `16 x 16` block 上的固定乘加结构。

---

## 11. 非对角块求逆

非对角块由 `T X = I` 逐层推出。

第一层：

```text
L10 X00 + D1 X10 = 0
```

所以：

```text
X10 = - D1^{-1} L10 X00
```

同理：

```text
X21 = - D2^{-1} L21 X11
X32 = - D3^{-1} L32 X22
```

第二层以 `X20` 为例：

```text
L20 X00 + L21 X10 + D2 X20 = 0
```

所以：

```text
X20 = - D2^{-1} (L20 X00 + L21 X10)
```

这里包含两条 causal path：

```text
B0 -> B2 direct
B0 -> B1 -> B2
```

最远的 `X30`：

```text
X30 = - D3^{-1} (L30 X00 + L31 X10 + L32 X20)
```

依赖层级是：

```text
level 0: X00, X11, X22, X33
level 1: X10, X21, X32
level 2: X20, X31
level 3: X30
```

---

## 12. 分块求逆的通用公式

对 block lower-triangular matrix：

```text
T[i,i] = Di
T[i,j] = Lij, i > j
```

逆矩阵 `X = T^{-1}` 满足：

```text
X[i,i] = Di^{-1}
```

```text
X[i,j] =
- Di^{-1} * sum_{k=j}^{i-1} L[i,k] X[k,j],
  i > j
```

也就是说，求逆被改写成：

```text
block matmul + add + scale/sign
```

它不是通用 `inverse()`，也不是 64 行 forward substitution。它是固定 block DAG：先算对角块，再按 block subdiagonal 层级算非对角块。

最后需要讲清楚的核心是：CP split 利用 affine transition 的结合结构缩短 replay 依赖链；Neumann/blocksolve 利用严格下三角的有限级数和分块求逆，把 prepare-A 改写成 block matmul/add 结构。
