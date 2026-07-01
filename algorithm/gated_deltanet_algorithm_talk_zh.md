# Gated DeltaNet 算法说明

> 目标：解释 Gated DeltaNet 的算法动机、状态更新公式和 chunkwise 计算结构。
>
> 阅读主线：从“线性注意力是一种 key-value 记忆”开始，依次引出
> Mamba2 的遗忘、DeltaNet 的定点改写，最后说明 Gated DeltaNet 为什么是二者互补。

## 0. 核心问题

Gated DeltaNet 可以先不从模型结构图或 benchmark 理解，而是从一个更基础的问题
进入：

```text
模型能不能维护一个可读、可写、可遗忘的 key-value 记忆？
```

标准 attention 是把历史 token 都摊开，用 query 去和所有 key 做匹配。Gated
DeltaNet 走的是另一条路：它不显式保留所有历史 token，而是维护一个固定大小的
状态矩阵 `S_t`。每来一个 token，就更新这个矩阵；需要输出时，就用当前 query
从这个矩阵里读。

本文的主线是：

```text
线性注意力：会写，但不太会删
Mamba2：会忘，但忘得比较粗
DeltaNet：会定点改写，但缺少快速清空能力
Gated DeltaNet：把“遗忘门”和“delta-rule 定点改写”合起来
```

最后再说明 Gated DeltaNet 的 chunkwise 计算结构，以及实际 block 中常见的参数化。

## 1. 从 softmax attention 到线性注意力

先从 softmax attention 开始。

对第 `t` 个 token，标准 causal attention 大致是：

```text
o_t = sum_{i<=t} softmax(q_t^T k_i) v_i
```

当前 query `q_t` 要和所有历史 key `k_i` 做匹配，再用匹配分数对
历史 value `v_i` 加权求和。

这个机制很强，因为它显式保留了所有 token 之间的 pairwise interaction。当前
token 可以直接回看任意历史 token。

但代价也很明显：

```text
prefill / training: 需要处理 T x T 的 token 交互
decode: KV cache 随上下文长度 T 线性增长
```

现代 attention kernel 可以避免真的把完整 attention matrix 全部落到显存里，但
计算结构本身仍然是“当前 token 和许多历史 token 做匹配”。当上下文变得很长时，
这个结构会越来越贵。

线性注意力的目标是改变这种计算组织方式：

```text
不要每次都显式扫描所有历史 token；
把历史 token 压缩进一个可递推更新的状态。
```

这里不展开 kernelized attention 的完整推导，只保留最关键的直觉：线性注意力把
query-key 的相似度设计成一种可以“先汇总历史、再用当前 query 读取”的形式。

于是所有历史 token 可以先被压缩到一个状态里：

```text
S_t = sum_{i<=t} v_i phi(k_i)^T
```

这个 `S_t` 就是递推状态。每来一个 token，只需要把新的 `v_t phi(k_t)^T` 写进去：

```text
S_t = S_{t-1} + v_t phi(k_t)^T
o_t = S_t phi(q_t)
```

为了让后续公式更轻，下面省略 `phi`，直接写成：

```text
S_t = S_{t-1} + v_t k_t^T
o_t = S_t q_t
```

这就是线性注意力相对于 softmax attention 的核心优势：

```text
它把“对所有历史 token 做 attention”变成了“维护一个固定大小的状态矩阵”。
```

从复杂度直觉上看：

- softmax attention 的 prefill 需要处理大量 pairwise token interaction；
- 线性注意力的 recurrent 形式按 token 更新状态，序列长度方向是线性的；
- decode 时不需要无限增长的完整 KV cache，而是维护固定大小的 recurrent state。

当然，这个优势不是免费的。线性注意力把历史压缩进 `S_t`，就不再显式保留每个
token 的独立位置。于是问题变成：

```text
这个压缩状态怎么写、怎么读、怎么忘、怎么改？
```

Gated DeltaNet 要解决的正是这个问题。

## 2. 状态矩阵是一种关联记忆

考虑最简单的线性注意力递推：

```text
S_t = S_{t-1} + v_t k_t^T
o_t = S_t q_t
```

这里 `S_t` 是一个状态矩阵。它可以被理解成一个压缩的 key-value memory：

- `k_t` 是地址；
- `v_t` 是内容；
- `v_t k_t^T` 是把内容写到这个地址相关的方向上；
- `q_t` 是读取地址；
- `S_t q_t` 是从当前状态里读出输出。

如果把递推展开：

```text
S_t = v_1 k_1^T + v_2 k_2^T + ... + v_t k_t^T
```

用任意一个向量 `x` 读取这个状态：

```text
S_t x = sum_i v_i (k_i^T x)
```

这个式子说明 `S_t x` 不是一个额外假设，而是 value 的相似度加权和：

```text
把所有历史 value 加权求和；
权重来自历史 key k_i 和当前读取向量 x 的相似度 k_i^T x。
```

当 `x = q_t` 时，对应正常的输出读取：

```text
o_t = sum_i v_i (k_i^T q_t)
```

query 和过去的 key 越相似，对应 value 的权重越大。

但是这个形式有一个明显问题：它只会加，不会删。

如果序列很长，状态里会叠很多东西。不同 key 方向之间不完全正交时，信息会互相
污染。更麻烦的是，当同一个 key 对应的 value 发生变化时，普通线性注意力没有
一个明确的“覆盖旧值”的动作。

例如：

```text
Alice -> Paris
Bob   -> London
Alice -> Berlin
```

如果模型需要记住最新事实，那么第二次看到 `Alice -> Berlin` 的时候，它最好能
把 `Alice -> Paris` 覆盖掉。普通线性注意力更像是又追加了一条记录，而不是精准
改写。

因此，最简单的线性注意力状态存在一个核心问题：

```text
只写不删的状态矩阵，长期使用会变脏。
```

## 3. Mamba2 的思路：给状态加一个遗忘门

一种直接做法是：每次写入新信息之前，先把旧状态衰减一下。

Mamba2 风格的 gated linear attention 可以抽象成：

```text
S_t = alpha_t S_{t-1} + v_t k_t^T
o_t = S_t q_t
```

其中：

```text
alpha_t in (0, 1)
```

`alpha_t` 是一个数据相关的 decay gate。它控制旧状态保留多少。

其作用可以概括为：

- `alpha_t` 接近 1：旧记忆基本保留；
- `alpha_t` 接近 0：旧记忆被快速清空；
- `alpha_t` 随 token 变化：模型可以根据上下文决定什么时候忘。

这解决了一个问题：状态不再只能无限累积。模型可以在段落切换、任务切换、上下文
变得无关时，快速降低旧状态的影响。

但是它的遗忘方式比较粗。因为 `alpha_t S_{t-1}` 是对整个状态一起缩放。

可以把这看作对白板的整体擦淡：Mamba2 的 gate 能降低旧状态的整体影响，但它不
直接回答一个更细的问题：

```text
如果只想更新 Alice 这个 key 对应的 value，应该怎么做？
```

所以 Mamba2 的优点是会忘，缺点是定点改写不够明确。

## 4. DeltaNet 的思路：用 delta rule 定点改写

DeltaNet 解决的是另一个问题：如何让状态矩阵像一个可以更新的关联记忆。

线性注意力的状态已经支持读取：

```text
S_t x = sum_i v_i (k_i^T x)
```

DeltaNet 的关键是把当前 key `k_t` 自己也作为读取向量，先对旧状态做一次
self-read：

```text
S_{t-1} k_t = sum_{i<t} v_i (k_i^T k_t)
```

这一步可以理解成：

```text
在写入当前 token 之前，先问旧状态：
如果用当前 key 作为地址，你现在会读出什么旧预测？
```

如果旧状态在这个 key 附近已经存过相关信息，`S_{t-1} k_t` 会读出一个旧预测。
Delta rule 后续写入的，正是当前 value 和这个旧预测之间的差。

它的递推可以写成：

```text
S_t = S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T
```

这里：

- `beta_t` 是写入强度；
- `k_t` 通常会做归一化，方便把 `k_t k_t^T` 看成 key 方向上的投影；
- `S_{t-1} k_t` 是旧状态按当前 key 读出来的旧预测。

将公式展开：

```text
S_t = S_{t-1} - beta_t (S_{t-1} k_t) k_t^T + beta_t v_t k_t^T
```

再合并一下：

```text
S_t = S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T
```

这就是 delta rule 的核心机制：

```text
先用当前 key 从旧记忆里读出预测值；
再计算新 value 和旧预测之间的残差；
最后只把这个残差写回当前 key 的方向。
```

也就是说，DeltaNet 写的不是完整的 `v_t`，而是：

```text
v_t - S_{t-1} k_t
```

如果旧状态已经能正确预测当前 value，就没有必要重复写入完整 value；如果
旧状态预测错了，就写入 correction。

回到前面的例子：

```text
Alice -> Paris
Bob   -> London
Alice -> Berlin
```

当第二次看到 Alice 时，DeltaNet 会先用 Alice 的 key 去读旧记忆。如果读出来
接近 Paris，而现在 value 是 Berlin，那么它会写入一个 residual，让 Alice 方向
的记忆从 Paris 被改向 Berlin。

因此，DeltaNet 的优势是：

```text
它不是简单追加信息，而是对当前 key 对应的关联进行定点改写。
```

但是 DeltaNet 自己也有问题。它擅长 key-specific update，却没有 Mamba2 那种
直接的全局遗忘门。长序列里，如果状态已经装了很多不再相关的信息，只靠 delta
rule 不一定能快速清理掉。

所以 DeltaNet 的优点是会改写，缺点是缺少快速遗忘。

## 5. Gated DeltaNet：把“会忘”和“会改写”合起来

Gated DeltaNet 的核心公式非常短：

```text
S_t = S_{t-1} [ alpha_t (I - beta_t k_t k_t^T) ] + beta_t v_t k_t^T
```

可以理解为：

```text
旧状态先经过 gate 衰减，
再沿当前 key 的方向做 delta-rule 改写，
最后写入新的 value 信息。
```

如果稍微展开：

```text
S_t = alpha_t S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T
```

这条公式包含两个互补机制：

```text
alpha_t：控制旧状态整体保留多少
beta_t：控制当前 key 方向改写多少
```

可以概括为：

```text
gate 负责 memory lifetime；
delta rule 负责 key-value association update。
```

这就是 Gated DeltaNet 最核心的算法动机。

它不是凭空发明了一个复杂递推，而是把两个已经很自然的机制放到同一个状态更新里：

- Mamba2 类方法说明：状态需要自适应遗忘；
- DeltaNet 类方法说明：状态需要按 key 做精准 residual update；
- Gated DeltaNet 的结论是：这两个机制不是竞争关系，而是互补关系。

## 6. Toy Example：上下文更新

考虑如下上下文：

```text
User profile:
Alice lives in Paris.
Bob lives in London.

Update:
Alice moved to Berlin.

Question:
Where does Alice live now?
```

普通线性注意力的状态像是不断往记忆里追加：

```text
Alice -> Paris
Bob   -> London
Alice -> Berlin
```

如果 Alice 的两个 key 表示很接近，读的时候可能会混到两个 value。

Mamba2 的 gate 可以在进入 `Update:` 之后衰减旧状态，让旧 profile 的影响变小。
但是这种衰减偏整体，不是专门针对 Alice。

DeltaNet 在看到 `Alice moved to Berlin` 时，会用 Alice 的 key 读出旧 value，
发现旧预测和新 value 不一致，于是沿 Alice 这个 key 方向写入 residual，把
Alice 的关联改掉。

Gated DeltaNet 两件事都能做：

```text
看到上下文阶段变化时，用 gate 降低旧状态影响；
看到 Alice 的新事实时，用 delta rule 改写 Alice 对应的关联。
```

这个例子对应的机制分工是：

```text
Mamba2-like gating: clear irrelevant context
DeltaNet-like update: overwrite specific association
Gated DeltaNet: both
```

## 7. Online Learning 视角：状态更新是在做一步回归

Delta rule 也可以从 online learning 角度理解。这里需要注意因果方向：

在普通线性注意力里，`S k_t` 只是一次相似度加权读出：

```text
S k_t = sum_i v_i (k_i^T k_t)
```

如果 key 很干净、接近正交，读出来可能主要是和 `k_t` 最相关的 value。但这不是
天然保证，也不是线性注意力自动满足的性质。

Delta rule 做的是另一件事：它主动把状态矩阵 `S` 当成一个正在在线学习的线性
映射，并给当前 token 施加一个局部目标：

```text
S k_t ≈ v_t
```

不是“因为 `S k_t` 本来就应该等于 `v_t`，所以有 delta rule”；而是
DeltaNet 选择让状态朝这个目标更新：

```text
给定当前 key，状态读出来的 value 尽量接近当前 value。
```

一个简单的平方损失是：

```text
L(S) = 1/2 || S k_t - v_t ||^2
```

对这个目标做一步梯度更新，就会得到类似：

```text
S_t = S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T
```

因此，delta rule 的 residual write 不是任意构造的。它可以理解成：

```text
每个 token 都给状态矩阵提出一个局部监督：
当前 key 应该读出当前 value。
```

Gated DeltaNet 在这个基础上加了 `alpha_t`，相当于允许模型根据当前 token 调整对
旧状态的约束强度。当旧状态不再可靠或不再相关时，gate 可以让状态更快偏离过去。

本节要点是：

```text
delta rule 是一种带 residual correction 的在线记忆更新。
```

## 8. Chunkwise 并行训练的 workload

Token-by-token 递推适合 decode，但训练或长 prefill 不能简单按 token 串行更新：

```text
for t in 1..T:
    update S_t
    compute o_t
```

那就很难用满 GPU。序列很长时，串行依赖会变成瓶颈。

因此，论文中的另一个重要部分是：如何把 Gated DeltaNet 改写成 chunkwise
parallel training 的形式。

核心想法是把序列切成 chunk。每个 chunk 内部仍然有因果依赖，但可以把依赖整理成
几类矩阵 workload：

```text
1. chunk-local prepare：块内因果依赖、三角 solve / inverse
2. cross-chunk state replay：块与块之间传播 recurrent state
3. output replay：把块初始状态和块内写入合起来，算每个 token 的输出
```

下面不展开完整推导，只列出主要中间量，用来说明计算由哪些 workload 组成。

假设一个 chunk 长度是 `C`。在这个 chunk 里包含如下矩阵：

```text
Q, K: [C, d_k]
V:    [C, d_v]
beta: [C]
alpha / gate: [C]
```

第一步是构造块内 token 之间的相互作用。首先形成一个 Gram-like 矩阵：

```text
G = K K^T
G_ij = k_i^T k_j
```

`G_ij` 表示 chunk 内第 `i` 个 token 的 key 和第 `j` 个 token 的 key 有多相似。
Delta rule 的块内依赖还需要把三件事合进去：

```text
causal mask：只允许过去 token j 影响未来 token i，也就是 i > j
beta_j：第 j 个 token 的写入强度
gate decay：从 j 写入到 i 读取之间，旧状态被 alpha 衰减了多少
```

因此可以把块内依赖矩阵记成一个严格下三角矩阵 `L`：

```text
L_ij ≈ 1[i > j] * decay(j -> i) * beta_j * (k_i^T k_j)
```

这里的 `decay(j -> i)` 是从 token `j` 到 token `i` 之间 gate 的累计衰减。不同实现
会把 gate scaling 放在不同的张量上，但它表达的是同一件事：较早写入的状态在被
后面 token 读取之前，会先经过若干步遗忘。

`L` 描述的是：

```text
chunk 内第 j 个 token 的写入，会怎样影响后面第 i 个 token 的旧状态读取。
```

然后 prepare 阶段会对这个下三角结构做 solve 或 inverse-like 计算。不同论文和
实现里的符号略有差别，本文用 `A` 表示这个块内 correction 矩阵：

```text
A ≈ inverse_or_triangular_solve(I + L)
```

这个 `A` 不是模型参数，而是每个 chunk 根据当前 `K, beta, gate` 临时算出来的。
它的作用是把串行 delta-rule 里“一步影响下一步”的依赖，压成一个块内矩阵计算。

有了 `A` 以后，prepare 阶段会产生两组有效写入，通常可以粗略写成：

```text
W = A @ (beta * K)
U = A @ (beta * V)
```

这里是示意写法。真实 Gated DeltaNet 里还会把 gate 的 cumulative decay 放到
对应位置，所以实现里可能会看到带箭头的 decay-scaled `K`、`Q`、`U`、`W`。
但从 workload 角度看，重点是：

```text
W：这个 chunk 对 key/state 转移的有效影响
U：这个 chunk 对 value/write 内容的有效影响
```

可以把 `W, U` 理解成“已经修正过块内因果依赖的写入包”。如果没有这一步，chunk
内部只能按 token 串行地读旧值、算 residual、写回；有了 `A -> W,U`，块内
很多工作就能变成矩阵乘法和三角 solve。

第二步是跨 chunk 传播状态。每个 chunk 不是从空状态开始，它有一个进入 chunk 的
初始状态 `S_in`。prepare 阶段给出 `W, U` 后，chunk 结束时的状态可以示意成：

```text
R = U - W @ S_in^T
S_out = decay_chunk(S_in) + R^T @ K
```

这里 `R` 可以理解成这个 chunk 真的要写入状态的 residual 包。`U` 来自当前 value，
`W @ S_in^T` 则是在问：如果用 chunk 里的 key 去读进入 chunk 之前的旧状态，会
读出什么旧预测？两者相减，正好对应前面 token-by-token 里的 residual 直觉。

等价地：

```text
新状态 = 衰减后的旧状态 + 这个 chunk 的有效 residual writes
```

这部分仍然有跨 chunk 的 recurrent 味道：第 `n+1` 个 chunk 的 `S_in` 来自第 `n`
个 chunk 的 `S_out`。所以长 prefill 的实现里，经常会看到 replay、scan、segment
replay 之类的调度。

第三步是输出 replay。每个 token 的输出由两部分组成：

```text
1. 从 chunk 初始状态 S_in 读到的历史贡献；
2. 从本 chunk 内前面 token 的有效写入读到的局部贡献。
```

示意写成：

```text
R = U - W @ S_in^T
O = Q @ S_in^T + causal(Q @ K^T) @ R
```

其中 `Q @ S_in^T` 是跨 chunk 历史状态贡献，`causal(Q @ K^T) @ R` 是块内因果
贡献。真实 Gated DeltaNet 还会在这些项上放入 gate decay，但 workload 形状仍然
可以这样记：

```text
QK^T：块内 query-key 相似度
W @ S_in^T：从旧状态读出的块内旧预测
U - W @ S_in^T：块内 residual writes
causal(QK^T) @ R：本 chunk 内部写入对输出的贡献
```

因此，token-by-token 递推和 chunkwise prefill 的差异可以概括为：

```text
token-by-token 递推：
    每个 token 读旧状态 -> 算 residual -> 写状态 -> 算输出

chunkwise prefill：
    每个 chunk 构造块内依赖 -> solve/inverse 得到 A
    A 生成 W, U
    W, U 用来做跨块 state replay 和块内 output replay
```

这里最难的是 chunk 内部的 delta-rule 依赖。因为 token `i` 的写入会影响 token
`j` 后续读到的旧状态。DeltaNet 论文线用的是 WY representation 来并行化这类
Householder-like 更新。Gated DeltaNet 则在这个基础上把 `alpha_t` 的 cumulative
decay 也合进去。

核心结论是：

```text
推理时：递推形式简单，适合 token-by-token decode。
训练 / prefill 时：用 chunkwise parallel，把块内依赖变成 A、W、U 和几组矩阵乘法。
```

关于 WY representation，可以保留一个简化理解：

```text
它的作用是避免显式展开一长串 (I - beta k k^T) 的矩阵乘积，
而是把这些 rank-1 update 组织成更适合并行矩阵运算的形式。
```

完整 WY 推导不是理解 Gated DeltaNet 算法动机的必要前提；对多数读者来说，先
把 `A -> W,U -> state/output replay` 的 workload 分解看清楚更重要。

## 9. 实际 Gated DeltaNet block 里发生了什么

论文里的 Gated DeltaNet block 大体沿用 Llama 风格的宏结构：

```text
token mixer layer + SwiGLU MLP
```

区别是 token mixer 不用 full attention，而是用 gated delta rule。

在 token mixer 里：

- `q, k, v` 来自线性投影；
- `q, k, v` 路径上会有 short convolution 和 SiLU；
- `q, k` 会做 L2 normalization，帮助训练稳定；
- `alpha` 和 `beta` 由线性投影产生；
- 输出通常还会经过 normalization、output gate 和 output projection。

这部分不是算法公式的核心，但它回答了一个工程问题：

```text
Gated DeltaNet 不是只把公式塞进模型；
它还需要合适的参数化、归一化、局部卷积和输出门控来稳定训练。
```

此外，论文也讨论了 hybrid 架构，例如把 Gated DeltaNet 和 sliding window
attention 组合。原因也很自然：

- recurrent linear state 擅长长程压缩和线性复杂度；
- sliding window attention 擅长局部 token 比较和短程 pattern；
- hybrid model 可以让两种机制各做擅长的事情。

因此，Gated DeltaNet 的实际使用方式不一定是“全模型都替换成 GDN 层”。更常见的
理解是：它是 hybrid long-context model 里的一个强 token mixer。

## 10. 和 Mamba2、DeltaNet 的对比总结

下表总结了几类相关方法的差异：

| 方法 | 状态更新直觉 | 优点 | 局限 |
| --- | --- | --- | --- |
| Linear attention | 不断累积 `v k^T` | 简单，可并行化 | 只写不删，状态容易混 |
| Mamba2-like gated update | 对旧状态做 decay，再写入 | 能快速遗忘无关历史 | 遗忘较整体，缺少定点覆盖 |
| DeltaNet | 读旧预测，写 residual | 擅长 key-value 关联改写 | 缺少快速全局清理 |
| Gated DeltaNet | decay + delta-rule residual write | 兼顾遗忘和定点改写 | 训练实现更复杂，需要 chunkwise 算法 |

核心 takeaway 是：

```text
Gated DeltaNet 的算法核心不是“多了一个 gate”这么简单，
而是把 memory management 和 associative update 放进同一个递推。
```

## 11. 阅读摘要

理解 Gated DeltaNet 时，最关键的过渡是从普通线性注意力的累积写入：

```text
S_t = S_{t-1} + v_t k_t^T
```

过渡到 DeltaNet 的 residual 写入：

```text
S_t = S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T
```

这一跳说明：状态更新不再只是追加 `v_t`，而是先用当前 key 从旧状态读出预测，
再写入当前 value 和旧预测之间的 residual。

在此基础上，Gated DeltaNet 再加入 `alpha_t` gate：

```text
S_t = alpha_t S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T
```

这使得状态既能按 key 做定点改写，也能根据上下文快速遗忘。

## 12. 核心结论

第一：

```text
线性注意力里的状态矩阵，本质上可以看成一个压缩的 key-value memory。
```

第二：

```text
Delta rule 的关键不是写入 v，而是写入 v 和旧状态预测之间的 residual。
```

第三：

```text
Gated DeltaNet 让状态既能按 key 精确改写，也能根据上下文快速遗忘。
```

## 13. 参考

- Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule*, ICLR 2025.
  <https://arxiv.org/abs/2412.06464>
- Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*.
- Dao and Gu, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*.
- Schlag et al., *Linear Transformers Are Secretly Fast Weight Programmers*.
