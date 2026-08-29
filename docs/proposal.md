# CLIR 方法说明：Consistency-Localized Intrinsic Rewards

## 研究目标

CLIR 研究如何从 frozen LLM 的 token hidden states 学习一个轻量 reward model，用于 query-level Best-of-N trajectory 选择。它以 SWIFT-style token reward / gate aggregation 为骨架，并加入三类监督：

1. 同语义、不同 style/domain 表达之间的 reward consistency；
2. 从 hallucination onset 起降低 continuation token value；
3. 用 key support 与 complete support 两张 prior map 约束证据定位。

当前仓库实现的是一个可运行的研究假设，不是已经证明优于 baseline 的最终方法。`configs/best_current.json` 是唯一默认整合配置；“best”表示当前工程和方法取舍最清晰，不表示已取得稳定的 Best-of-N 增益。

## 输入与 exact-token 假设

对 query `x`、可选 context `c` 和生成 trajectory `y=(y_1,...,y_T)`，上游生成过程必须保存原始：

```text
prompt_token_ids
output_token_ids
```

特征抽取对两者拼接后做一次无 padding、teacher-forced causal forward，读取 embedding 和全部 transformer blocks 的 hidden states。第 `t` 个生成 token 的原始特征为：

```text
h_t^raw = concat(h_t^0, h_t^1, ..., h_t^(L-1)) ∈ R^(L·d)
```

当前位置只由保存的 token IDs 决定，不允许从 response 文本重新 tokenize 后再映射标签。对默认 Phi 配置，`L=33`、`d=3072`，因此 raw width 为 `101376`。

## Layer-axis 特征编码

直接在 `101376` 维上使用 condition attention 或多层 reward heads 会造成不必要的参数和显存开销。默认 encoder 先把每个 token reshape 为 `[L,d]`：

```text
E_t = reshape(h_t^raw) ∈ R^(L×d)
```

然后依次执行：

1. 每层共享的 `d→256` 投影；
2. learned layer position；
3. 2-block、8-head layer-axis Transformer；
4. 4 个 learned pooling queries；
5. pooling 输出拼接并投影到 `model_dim=768`。

得到 compact token feature `e_t∈R^768`。同一个 encoder 同时处理 trajectory 和 condition states。Identity encoder 仅用于 toy 或已经压缩的输入。

## 条件化 token feature

对 prompt/context compact states `C=(c_1,...,c_M)`，每个 trajectory token 通过 256 维瓶颈计算 cross-attention：

```text
q_t = normalize(W_q e_t)
k_j = normalize(W_k c_j)
α_tj = softmax(q_t · k_j / temperature)
u_t = Σ_j α_tj W_v c_j
```

再把 token projection、context、逐维乘积、差值和 relevance 拼接，经小型 fusion network 产生 residual delta：

```text
f_t = LayerNorm(e_t + Δ(e_t, u_t))
```

若 row 没有 condition，`f_t=e_t`。这个瓶颈保留逐 token 条件相关性，同时避免在 raw all-layer width 上构造平方级参数层。

## SWIFT-style reward backbone

模型从 `f_t` 预测 gate logit 和 token reward：

```text
g_t = sigmoid(w_g^T f_t + b_g)
r_t = w_r^T f_t + b_r
p_t = w_h^T f_t + b_h              hallucination logit
a_t = w_a^T f_t + b_a              progress
```

token value 定义为：

```text
v_t = r_t + η a_t
```

当前唯一默认配置令 `η=0`，所以 `v_t=r_t`；progress head 虽然仍输出，但不会在无独立证据时偷偷改变 scalar score。

trajectory score 为：

```text
R(y|x,c) = Σ_t g_t v_t / max(Σ_t g_t, ε) + w_res^T mean_masked(f_t)
```

训练默认对有 `correctness` 标签的 row 使用 pointwise BCE：

```text
L_final = BCEWithLogits(R, correctness)
```

缺失 correctness 的 row 通过 `correctness_mask` 跳过，不会当成 incorrect。

## 模块一：semantic/style consistency

令 `z_i` 是 pooled trajectory feature 经 projector 后的归一化表示，`u_i` 是 `semantic_id`，`s_i` 是 `style_id`。

同语义、不同 style 的表示应接近：

```text
L_pos = mean_[u_i=u_j, s_i≠s_j] (1 - cos(z_i,z_j))
```

不同语义、相同 style 的表示应低于 margin `m`：

```text
L_neg = mean_[u_i≠u_j, s_i=s_j] relu(cos(z_i,z_j) - m)
```

同语义 rewrite 的 scalar score 也应稳定：

```text
L_score = mean_[u_i=u_j, s_i≠s_j] (R_i-R_j)^2
```

当前配置使用：

```text
L_cons = L_pos + 1.0·L_neg + 0.1·L_score
m = 0.2
```

`SemanticGroupBatchSampler` 保证同 semantic 的样本能进入同一 mini-batch。接口不限定 augmentation 的来源：可以是经过验证的 rewrite，也可以是同一模型生成、经独立裁决确认 reasoning-equivalent 的 on-policy trajectories。v6.1 已用400个双 AI Silver 训练正对和150+150个 held-out 正负关系完成三 seed C0/C1 复测；当前支持的是 hard-negative separation 和去塌缩的部分机制证据，不能外推成广义 style/domain invariance 或 ranking efficacy。

## 模块二：hallucination onset 与 negative tail

当前实现恢复 `main` 的方法定义。对具有可靠 onset 标签的 trajectory：

```text
τ = 首个 unsupported 或 contradicted material claim 对应的生成 token 索引
```

已知 clean trajectory 使用 `τ=-1`；字段缺失表示未标注，而不是 clean。

H head 的 token target 为：

```text
q_t* = 1[t≥τ]    若 τ≥0
q_t* = 0         若 τ=-1
```

并使用：

```text
L_hall = BCEWithLogits(p_t, q_t*)
```

同一个真实 onset 还约束 score 使用的 token value。若存在外部 `token_advantage`，它提供已知位置的 value target；无论是否存在 advantage，tail target 都被覆盖为 `-γ`：

```text
v_t* = -γ,  t≥τ
L_value = MSE(v_t, v_t*) on known positions and tail
L_tail = mean_[t≥τ] relu(v_t + γ)^2
```

默认 `γ=0.5`，`L_hall/L_value/L_tail` 的外层权重分别为 `1.0/0.5/0.5`。因为 `v_t` 直接进入 gate-weighted score，negative-tail 监督能通过 value path 影响 scalar reward；H probability 本身不会直接乘到 score 上。

这是当前待检验的核心假设：首错后的 continuation value 应整体降低。历史实验显示旧 absolute-margin 实现可能通过全局 value shift 满足约束，relative 和 clean-matched 修复也没有过门。因此当前实现恢复了方法身份，但尚无新证据证明这种 shaping 改善 ranking 或 locality。

### 默认关闭的弱监督

若只有 path label，可以定义稳定 log-space noisy-or MIL：

```text
P(path clean) = Π_t (1-sigmoid(p_t))
L_MIL = BCE(1-P(path clean), path_hallucinated)
```

也可以取首个超过阈值的 token 为 pseudo onset，再施加低权重 tail loss。当前 `mil_weight=0`、`pseudo_tail_weight=0`：boundary head 尚未在独立数据上通过前，不允许用自身预测循环生成 reward target。

## 模块三：key/complete dual prior

模型预测两张完整 trajectory 上的 masked-softmax attention map：

```text
A_key  = softmax_masked(l_key)
A_comp = softmax_masked(l_complete)
```

- `A_key`：最关键的 evidence、decisive step 或 decisive flaw。
- `A_comp`：形成完整支持链所需的更广 token span。

有外部 binary token targets 时，两个 head 分别使用 masked BCE：

```text
L_key_direct
L_complete_direct
```

当前还保留双向 stop-gradient mutual distillation。每条 trajectory 内先对 token squared error 求和，再在 trajectory 间取均值，避免目标强度随序列长度被 `1/T` 稀释：

```text
L_mutual = MSE(A_key, stopgrad(A_comp))
         + MSE(A_comp, stopgrad(A_key))
```

比较只在 key/complete 共同具有 label coverage 的 token 上进行，但两张 attention map 仍是在完整有效 trajectory 上归一化，不会对子集重新 softmax。默认 direct 权重均为 `1.0`，mutual 权重为 `.25`，训练 phase 为 `joint`。

融合 prior 为：

```text
A_fused = normalize(0.5·A_key + 0.5·A_comp)
```

默认同时保留 `origin/main` 的 shared-gradient gate coupling：

```text
A_gate = normalize(sigmoid(gate_logits))
L_gate = squared-L2(A_gate, stopgrad(A_fused))
```

`gate_prior_weight=.25`。它不会在推理时把 prior 额外加到 score，而是在训练时让 fused
prior 约束本来就参与 scalar score 的 reward gate；梯度更新 gate head 和共享 encoder，
不通过该 loss 更新 detached prior target。该 `.25` 是在当前开发 population 上按冻结规则
选出的工程默认值，也与 `origin/main` 的内部系数相同；clean 外层 `prior_weight=1`，所以
绝对 coupling 系数为 `.25`，而原 main 的绝对系数为 `.25×.25=.0625`。

external reconstruction target 接口仍保留但权重为 0，因为没有可靠外部 target。禁止用
同一 candidate 的 pooled feature 构造平凡自重构 target。

## 当前默认总目标

在一批数据具备对应监督时，`best_current` 的 active objective 是：

```text
L = 1.0·L_final
  + 1.0·L_cons
  + 1.0·L_hall
  + 0.5·L_value
  + 0.5·L_tail
  + 1.0·(L_key_direct + L_complete_direct
          + 0.25·L_mutual + 0.25·L_gate)
```

这是一种 sparse multi-task objective：不同 row 可以只具有其中一部分标签。每种监督有独立 mask；没有 target 就不计算相应分量。

默认不进入 total objective 的分量为：

```text
path MIL
pseudo-onset tail
progress regression
progress contribution to score
complete reconstruction
```

## 证据与假设边界

| 命题 | 当前状态 |
|---|---|
| Exact-ID 抽取可保持 token/feature 对齐 | 有代码与回归测试支持的工程事实 |
| Layer-axis encoder 可在真实 raw width 下构建小于一千万参数的模型 | 有配置和测试支持的工程事实 |
| Consistency loss 能改变 held-out relation geometry | v6.1 400 train positives、150 heldout positives、150 hard negatives 的三 seed C0/C1 中，冻结均值 separation 增量 `+.1760`、relation bootstrap `[+.1389,+.2155]`；但正对 cosine 下降、AUROC seed 方向混合，只支持 hard-negative separation/去塌缩的部分机制结论 |
| Sparse-span H head 在 16-row dev 上超过位置基线 | 只有点估计；bootstrap 跨 0，后续 blind/position control 失败；且不是当前默认实现 |
| Main onset-tail shaping 会改善 reward ranking | 未验证假设；历史 absolute/relative/clean-matched 实现均暴露问题 |
| Direct key/complete targets 可学习 | 48/16、3 seeds 的 standalone gate 通过 |
| Mutual distillation 降低 branch discrepancy 且不明显损伤 localization | standalone 3/3 seeds 保护门通过 |
| Shared gate-prior alignment 改善 Best-of-N | 未建立。v2 同 dev 调参中 `.25` 的 BoN@16 `.9187` vs P0 `.9180`，两个配对区间跨 0；`10` 的 raw point estimate `.9207` 也未形成独立验证。默认 `.25` 是方法身份约束下的 dev-tuned 工程值，不是 efficacy 结论 |
| 三模块联合优于 correctness-only | 未建立；历史 JALL BoN@16 `.912`，J0 `.920`，扩展门失败 |

这些证据主要来自 Phi 生成的 GSM8K/MATH 算术与数学推理数据；Consistency 的关系规模已扩大，但仍不能支持跨生成器、广义跨领域或 Best-of-N 结论。工程 pipeline 运行、auxiliary target 可学习和 Best-of-N 改善必须分开报告。

## 评价设计

所有效果比较应使用 query-disjoint 数据、冻结 candidate order 和至少 3 个训练 seeds。核心指标包括：

- Best-of-N accuracy@`k` 及 query bootstrap 区间；
- query 内 correct-vs-wrong pairwise accuracy；
- consistency held-out relation cosine/score gap；
- hallucination onset/boundary、token localization 和 selected-trajectory hallucination rate；
- key/complete prior AP，以及与简单 token position baseline 的比较。

推荐 matched ablation：

```text
correctness only
+ consistency
+ main hallucination onset/tail
+ direct dual prior
+ mutual distillation
+ main-style prior-to-gate alignment
full active integration
```

下一轮扩大 prior 数据后，应把 `.25` 固定为 on cell，与 gate-off cell 做独立配对复测；
不再在当前 16-row mechanism dev / 500-query ranking dev 上连续扫描 weight、epoch、margin、
threshold 或 routing。任何其他默认关闭分量仍应在独立 protocol 下只改变一个因素后重开。

## 预期贡献与当前表述

若后续证据成立，CLIR 的贡献将不是扩大 reward model，而是把更结构化的监督接入轻量 hidden-state scoring：

1. 对可验证的语义等价变化保持 reward 稳定；
2. 显式定位 reasoning 从何处失效；
3. 用关键证据与完整支持链共同约束 token-level credit assignment。

当前可以声称的是：仓库已经实现了一个 exact-token、全层特征、可恢复训练、三模块接口完整的研究平台，并把历史通过和失败部分整理成一个单一配置。当前不能声称的是：三模块联合已经带来稳定 Best-of-N 增益，或 main hallucination tail 已被证明优于 sparse diagnostic head。
