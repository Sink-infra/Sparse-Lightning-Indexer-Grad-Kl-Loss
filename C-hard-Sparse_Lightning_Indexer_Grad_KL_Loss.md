# Sparse Lightning Indexer Grad KL Loss 算子

## 1. 算子说明

SparseLightningIndexerGradKLLoss 是 LightningIndexer（稀疏 Top-k 索引选择算子）的**反向梯度算子**，并融合了 KL 散度损失计算。它通过 KL 散度约束 LightningIndexer 的 Top-k 选择分布逼近主注意力（main attention）的 softmax 分布，计算 loss 及其对 queryIndex/keyIndex/weights 的梯度，用于训练 LightningIndexer。稀疏场景下由 `sparseIndices` 指定每个 query token 选中的 Top-k 个 key 位置，仅对这些位置做 gather/计算，未选中位置梯度为 0。

### 1.1 背景术语（题目内自解释）

- **LightningIndexer 与 sparseIndices**：为每个 query token 从上下文中选出相关度最高的 Top-k 个 key 位置索引，存放在 `sparseIndices` 中，减少长序列 Attention 计算量。本算子是其训练阶段的反向。
- **主注意力（main attention）与 target 分布 `p`**：用 query/key 做标准注意力 `softmax(Q@K^T/√d)`，对所有头求和后沿上下文方向 L1 正则化，得到 target 分布 `p`（长度为 key 序列长度）。本算子主注意力为 MQA（key 头数 `N2=1`）。
- **稀疏 gather / scatter**：根据 `sparseIndices` 从 keyIndex 中 gather 出选中的 key 向量参与计算；反向时将 dKeyIndex scatter 回原位置（累加），未选中位置梯度为 0。
- **Indexer 得分 `I` 与预测分布**：用 queryIndex/keyIndex 计算 `I = W @ ReLU(q̃ @ K̃^T)`，再做 `softmax(I)` 得预测分布。
- **KL 散度**：`D_KL(a||b) = Σ_i a_i log(a_i/b_i)`。本算子在被选 Top-k 位置上计算 target 条件分布与预测分布的 KL 散度。
- **MQA**：Multi-Query Attention。主注意力 key 头数 `N2=1`，所有 query 头共享一组 key；indexer keyIndex 头数 `Nidx2=1`，group size = Nidx1。
- **RoPE**：旋转位置编码，query/key 各带一份 RoPE 向量（维度 `DRope`）。
- **维度符号**：`B`=Batch Size，`S1`=query 序列长度，`S2`=key 序列长度，`N1`=query 头数，`N2`=key 头数（恒为 1），`Nidx1`=queryIndex 头数，`Nidx2`=keyIndex 头数（恒为 1），`DQuery`=query/key 每头维度（=512），`DQueryIndex`=queryIndex/keyIndex 每头维度（=128），`DRope`=RoPE 维度（=64），`K`=sparse_size 每个 query token 选取的 key 位置数，`G=N1/N2`。
- **数据排布**：本算子仅支持 `BSND` 排布，即 `(B, S, N, D)`。

### 1.2 计算公式

1. **Indexer 得分**（Top-k value）：

```
I[t,:] = W[t,:] @ ReLU( q̃[t,:] @ K̃[:,t,:]^T )
```

其中 `W` 为 weights，`q̃` 为 queryIndex，`K̃` 为 keyIndex（仅取 sparseIndices 选中的位置）。

2. **target 分布**（主注意力 softmax，各头求和后 L1 正则化）：

```
p[t,:] = L1norm( Σ_head softmax( q[t,:] @ K[:,t,:]^T / √d ) )
```

3. **KL Loss**（在被选 Top-k 位置上，target 取条件分布）：

```
Loss = Σ_t D_KL( p_sel[t,:] || softmax(I[t,:]) ) = Σ_t Σ_i p_sel_i log( p_sel_i / softmax(I)_i )
```

其中 `p_sel` 为 `p` 在 sparseIndices 选中位置上的条件分布（重新 L1 正则化）。

4. **梯度**（由 KL 散度对 I 求导，再链式传播）：

```
dI[t,:]   = softmax(I[t,:]) - p_sel[t,:]
dW[t,:]   = dI[t,:] @ ReLU(S[t,:])^T          # S = q̃ @ K̃^T
dq̃[t,:]  = dS[t,:] @ K̃[:,t,:]
dK̃[:,t,:] = (dS[t,:])^T @ q̃[:,t,:]
```

其中 `S = q̃ @ K̃^T` 为 indexer 相似度（仅被选位置），`dS` 由 `dI` 经 `W` 与 ReLU 反向得到；dK̃ 需 scatter 回原位置累加。

---

## 2. 输入输出说明

### 2.1 输入规格

| 输入 | 必选/可选 | 类型 | 形状 | 数据类型 | 含义 |
|---|---|---|---|---|---|
| query | 必选 | 张量 | (B, S1, N1, DQuery) | float16/bfloat16 | 主注意力输入 Q。N1∈{32,64,128}，DQuery=512 |
| key | 必选 | 张量 | (B, S2, N2, DQuery) | float16/bfloat16 | 主注意力输入 K。N2=1（MQA），DQuery=512 |
| queryIndex | 必选 | 张量 | (B, S1, Nidx1, DQueryIndex) | float16/bfloat16 | LightningIndexer 的 query。Nidx1∈{8,16,32,64}，DQueryIndex=128 |
| keyIndex | 必选 | 张量 | (B, S2, Nidx2, DQueryIndex) | float16/bfloat16 | LightningIndexer 的 key。Nidx2=1（MQA），DQueryIndex=128 |
| weights | 必选 | 张量 | (B, S1, Nidx1) | float16/bfloat16/float32 | group 内每个头的权重 W |
| sparseIndices | 必选 | 张量 | (B, S1, Nidx2, K) | int32 | Top-k 索引，选每个 query 对应的 key。Nidx2=1，K∈{1024,2048,3072,4096,5120,6144,7168,8192} |
| softmaxMax | 必选 | 张量 | (B, N2, S1, G) | float32 | 来自 SparseFlashAttention 前向 pass 的 softmax 行最大值，由外部传入（避免重算）。G=N1/N2 |
| softmaxSum | 必选 | 张量 | (B, N2, S1, G) | float32 | 来自 SparseFlashAttention 前向 pass 的 softmax 行求和，由外部传入 |
| queryRope | 必选 | 张量 | (B, S1, N1, DRope) | float16/bfloat16 | query 的 RoPE 信息，DRope=64 |
| keyRope | 必选 | 张量 | (B, S2, N2, DRope) | float16/bfloat16 | key 的 RoPE 信息，DRope=64 |
| actual_seq_lengths_query | 必选 | 数组 | (B,) | int64 | 每个 batch query 的有效 token 数 |
| actual_seq_lengths_key | 必选 | 数组 | (B,) | int64 | 每个 batch key 的有效 token 数 |
| scaleValue | 必选 | 属性 | 标量 | double | 注意力缩放系数，对应公式中的 1/√d |
| layout | 必选 | 属性 | 标量 | string | 数据排布格式，默认 "BSND"，本算子仅支持 BSND |
| sparseMode | 必选 | 属性 | 标量 | int64 | 掩码模式，仅支持 3（rightDownCausal 下三角掩码） |
| deterministic | 必选 | 属性 | 标量 | bool | 确定性计算，优先使用整网确定性配置，该参数实际不产生效果，确定性由整网配置决定 |

> 说明：属性 `pre_tokens`、`next_tokens` 用于稀疏计算的滑窗位置（仅 `sparseMode=0/4` 时生效，本算子仅支持 3 故不生效），仅支持默认最大值 2^63-1，参赛者无需修改。

### 2.2 输出规格

| 输出类型 | 必选/可选 | 形状 | 数据类型 | 含义 |
|---|---|---|---|---|
| dQueryIndex | 必选 | (B, S1, Nidx1, DQueryIndex) | float16/bfloat16 | queryIndex 的梯度 dq̃ |
| dKeyIndex | 必选 | (B, S2, Nidx2, DQueryIndex) | float16/bfloat16 | keyIndex 的梯度 dK̃（未选位置为 0） |
| dWeights | 必选 | (B, S1, Nidx1) | float16/bfloat16/float32 | weights 的梯度 dW |
| loss | 必选 | (1,) | float32 | KL 散度损失值 |

### 2.3 形状约束

- 仅支持训练场景。Ascend 950PR/950DT、Atlas A2/A3 系列产品均支持。
- `DQuery=512`（query/key 每头维度），`DQueryIndex=128`（queryIndex/keyIndex 每头维度），`DRope=64`。
- query/key/queryIndex/keyIndex 数据类型必须一致；weights 不为 float32 时，五者数据类型必须一致。
- query 为空 Tensor 时直接返回。
- `sparseMode` 仅支持 3（rightDownCausal）。
- query 头数 `N1`∈{32,64,128}；queryIndex 头数 `Nidx1`∈{8,16,32,64}；key 头数 `N2=1`、keyIndex 头数 `Nidx2=1`（均为 MQA）。
- `B` 取值 1~256；`S1` 取值 1~8K，`S2` 取值 1~512K，支持不等长且 `S1 ≤ S2`。
- `K`（sparse_size）取值 1024/2048/3072/4096/5120/6144/7168/8192。
- Ascend 950PR/950DT：`N1` 额外支持 48、`Nidx1` 额外支持 24，且二者仅允许 (48,24) 组合，禁止其余配对；B/S1/S2 均支持泛化。
- 仅支持 ND 数据格式（任意多维非结构化连续格式）。
- 默认非确定性实现，可通过 `aclrtCtxSetSysParamOpt` 开启确定性。

### 2.4 输出形状计算公式

- `dQueryIndex`：与 queryIndex 形状一致 `(B, S1, Nidx1, DQueryIndex)`。
- `dKeyIndex`：与 keyIndex 形状一致 `(B, S2, Nidx2, DQueryIndex)`，未被 sparseIndices 选中的位置梯度为 0。
- `dWeights`：与 weights 形状一致 `(B, S1, Nidx1)`。
- `loss`：标量 `(1,)`。

---

## 3. 算子逻辑说明

### 3.1 target 分布 p（主注意力）

对每个 query token，用 query/key 计算主注意力 score `q@K^T/√d`（MQA，各 query 头共享 key），做 softmax，对所有头求和后沿 key 方向 L1 正则化，得到 target 分布 `p`（长度 S2）。实际算子用输入 `softmaxMax/softmaxSum` 重建 softmax 以避免重算。

### 3.2 稀疏 gather 与预测分布 softmax(I)

根据 `sparseIndices` 从 keyIndex 中 gather 出选中的 Top-k 个 key 向量 `K̃`；计算 indexer 相似度 `S = q̃ @ K̃^T`，ReLU 激活，再用 weights 加权得 indexer 得分 `I = W @ ReLU(S)`，做 `softmax(I)` 得预测分布。将被选位置上的 target `p` 重新 L1 正则化得条件分布 `p_sel`。

### 3.3 KL Loss

`Loss = Σ_t D_KL(p_sel || softmax(I)) = Σ_t Σ_i p_sel_i log(p_sel_i / softmax(I)_i)`，约束预测分布逼近 target 条件分布。

### 3.4 反向梯度（链式求导 + scatter）

- `dI = softmax(I) - p_sel`（KL 散度对 I 的梯度）
- `dW = dI @ ReLU(S)^T`（weights 梯度）
- `dS = outer(W, dI) ⊙ (S>0)`，即 `dS[i,j] = W[i]·dI[j]·(S[i,j]>0)`（经 W 反传与 ReLU 反向）
- `dq̃ = dS @ K̃`（queryIndex 梯度）
- `dK̃ = dS^T @ q̃`（keyIndex 梯度，需 scatter 回原位置累加，未选位置为 0）

### 3.5 掩码与 RoPE

- `sparseMode=3`（rightDownCausal）对主注意力与 indexer score 应用 query 右端对齐 key 右端的下三角掩码，屏蔽不可见位置。
- RoPE（queryRope/keyRope）按旋转位置编码对 query/key 施加旋转并融合到主注意力计算。

---

## 4. 决赛任务要求(现在不作要求)

- **精度保障**：针对不同 `N1/Nidx1`、变长序列、`sparseMode=3` 掩码与稀疏 gather，设计算子逻辑，保证 KL loss 及 dQueryIndex/dKeyIndex/dWeights 梯度精度正确（可用数值梯度校验）。
- **性能优化**：利用输入 softmaxMax/Sum 中间结果还原主注意力分布，避免 softmax 规约操作的重算（但仍需计算 Q@K^T 获取打分），优化稀疏 gather/scatter 的离散访存与 indexer 矩阵乘/反向的并行度，算子性能更优。
- **切分最优**：探索 query 序列维度、头维度与稀疏索引维度在多 batch/变长场景下的切分方式，找到不同输入 shape 场景下的最优解。

---

## 5. 功能示例

```python
import numpy as np

def sparse_lightning_indexer_grad_kl_loss(query, key, query_index, key_index, weights,
                                          sparse_indices, scale_value):
    # query:(B,S1,N1,DQuery) key:(B,S2,N2,DQuery) N2=1 (main MQA)
    # query_index:(B,S1,Nidx1,DQueryIndex) key_index:(B,S2,Nidx2,DQueryIndex) Nidx2=1
    # weights:(B,S1,Nidx1) sparse_indices:(B,S1,Nidx2,K)
    # 输出 dQueryIndex:(B,S1,Nidx1,DQueryIndex) dKeyIndex:(B,S2,Nidx2,DQueryIndex) dWeights:(B,S1,Nidx1) loss:标量
    # 聚焦核心 KL loss + indexer 反向(稀疏gather/scatter)；main 用 softmaxMax/Sum 重建避免重算(此处简化重算);
    # 省略 mask(sparseMode=3) 与 RoPE 融合(见 3.5 节)
    q = query.astype(np.float32); k = key.astype(np.float32)
    qi = query_index.astype(np.float32); ki = key_index.astype(np.float32); w = weights.astype(np.float32)
    B, S1, N1, Dq = q.shape; _, S2, N2, _ = k.shape; _, _, Nidx1, Dqi = qi.shape
    dQi = np.zeros_like(qi); dKi = np.zeros_like(ki); dW = np.zeros_like(w); loss = 0.0
    for b in range(B):
        for s in range(S1):
            # target 分布 p：主注意力各头 softmax 求和后 L1 正则化
            S_main = (q[b, s] @ k[b, :, 0, :].T) * scale_value            # (N1, S2)
            sm = np.exp(S_main - S_main.max(-1, keepdims=True)); sm /= sm.sum(-1, keepdims=True)
            p_raw = sm.sum(0); p = p_raw / p_raw.sum()                      # target (S2,)
            # 稀疏 gather：按 sparseIndices 取被选 key 向量
            idx = sparse_indices[b, s, 0]                                   # (K,)
            valid = idx[(idx >= 0) & (idx < S2)]
            Ksel = ki[b, valid, 0, :]                                       # (m, Dqi)
            S_idx = qi[b, s] @ Ksel.T                                       # (Nidx1, m) indexer 相似度
            relu = np.maximum(S_idx, 0); I = w[b, s] @ relu                # (m,) indexer 得分
            softmax_I = np.exp(I - I.max()); softmax_I /= softmax_I.sum()
            p_sel = p[valid] / p[valid].sum()                              # 被选位置条件 target 分布
            # KL loss: D_KL(p_sel || softmax_I)
            eps = 1e-12
            loss += np.sum(p_sel * np.log((p_sel + eps) / (softmax_I + eps)))
            # 反向：dI = softmax_I - p_sel，链式求 dW/dq̃/dK̃
            dI = softmax_I - p_sel
            dW[b, s] = dI @ relu.T                                         # (Nidx1,)
            dS = np.outer(w[b, s], dI) * (S_idx > 0)                       # (Nidx1, m) ReLU 反向
            dQi[b, s] = dS @ Ksel                                          # (Nidx1, Dqi)
            for ii, pos in enumerate(valid):                              # scatter 回原位置(累加)
                dKi[b, pos, 0, :] += dS[:, ii] @ qi[b, s]
    return dQi.astype(np.float16), dKi.astype(np.float16), dW.astype(np.float16), loss

np.set_printoptions(precision=4, suppress=True)

# 示例1：稀疏 gather 与反向梯度（BSND，B=1, S1=1, N1=32, N2=1, Nidx1=8, Nidx2=1, DQuery=512, DQueryIndex=128, S2=4, K=2）
# 为可读性使用较小 K，实际算子要求 K∈{1024,2048,3072,4096,5120,6144,7168,8192}；示例仅为演示逻辑，实际参赛实现需满足 K 的取值约束
Dq = 512; Dqi = 128; scale = 1.0 / np.sqrt(Dq)
B, S1, N1, N2, Nidx1, Nidx2, S2, K = 1, 1, 32, 1, 8, 1, 4, 2
query       = np.full((B, S1, N1, Dq), 0.1, dtype=np.float16)
key         = np.tile((np.arange(1, S2 + 1) * 0.1).reshape(1, S2, 1, 1), (1, 1, N2, Dq)).astype(np.float16)
query_index = np.full((B, S1, Nidx1, Dqi), 0.1, dtype=np.float16)
key_index   = np.tile((np.arange(1, S2 + 1) * 0.1).reshape(1, S2, 1, 1), (1, 1, Nidx2, Dqi)).astype(np.float16)
weights     = np.full((B, S1, Nidx1), 0.1, dtype=np.float16)
sparse_indices = np.array([[[[0, 1]]]], dtype=np.int32)    # (1,1,1,2) 选位置 0/1
dQi, dKi, dW, loss = sparse_lightning_indexer_grad_kl_loss(query, key, query_index, key_index, weights,
                                                            sparse_indices, scale)
# 输出：loss=0.0743；dQueryIndex=(1,1,8,128) dKeyIndex=(1,4,1,128) dWeights=(1,1,8)
# 因各头 queryIndex 相同，dWeights 各头相同（均≈0.2294）
# dKeyIndex：被选位置 0/1 非零（∓0.0143/维），未选位置 2/3 为 0（体现稀疏 gather/scatter）
print('loss:', round(loss, 4))
print('dWeights(8 维):', dW[0, 0])

# 示例2：多 batch 场景（B=2，两 batch 结构相同，各自独立计算）
B = 2
query       = np.full((B, S1, N1, Dq), 0.1, dtype=np.float16)
key         = np.tile((np.arange(1, S2 + 1) * 0.1).reshape(1, S2, 1, 1), (B, 1, N2, Dq)).astype(np.float16)
query_index = np.full((B, S1, Nidx1, Dqi), 0.1, dtype=np.float16)
key_index   = np.tile((np.arange(1, S2 + 1) * 0.1).reshape(1, S2, 1, 1), (B, 1, Nidx2, Dqi)).astype(np.float16)
weights     = np.full((B, S1, Nidx1), 0.1, dtype=np.float16)
sparse_indices = np.tile(np.array([[[[0, 1]]]], dtype=np.int32), (B, 1, 1, 1))  # (2,1,1,2)
dQi, dKi, dW, loss = sparse_lightning_indexer_grad_kl_loss(query, key, query_index, key_index, weights,
                                                            sparse_indices, scale)
# 输出形状：dQueryIndex=(2,1,8,128) dKeyIndex=(2,4,1,128) dWeights=(2,1,8)
# 两 batch 结构相同 => loss 为示例1的 2 倍（0.1486）
print('多 batch loss:', round(loss, 4))
```
