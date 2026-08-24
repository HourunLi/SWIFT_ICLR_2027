# CLIR 三模块阶段性报告

- 日期：2026-08-24
- 分支：`clir-clean-integration`
- 报告基线 commit：`572d3625fd942a266edfa91c4fb3b90db8aa8d33`
- 证据等级：`small-scale real screening`

> 2026-08-24 gate 更新：用户将“保留 `main` 原始 prior→reward-gate shared-gradient
> coupling 且默认开启”确定为方法身份约束。六权重 v2 工程选择已完成，当前
> `gate_prior_weight=.25`；这是同一 dev 上选出的工程默认值，不是独立 efficacy 结论。
> 本报告正文已同步当前路由；原 P0/P1/Full 数字仍对应当时 gate-off 的冻结消融，不会
> 因默认改变而重解释。详见
> [`clean_gate_tuning_v2_results.md`](clean_gate_tuning_v2_results.md)。

本报告回答五个问题：当前接入的三个模块分别是什么、代码上如何实现、如何进入训练与最终
Best-of-N 选择、各自目前有什么效果、这些监督数据最初如何得到，以及组合训练后发生了什么。
下一步如何扩充数据只在末尾列出待讨论的决策点，不在本报告中提前冻结新协议。

## 1. 阶段性结论

当前可以明确区分三个层次：

1. **工程实现已经完成。** Consistency、Hallucination 和 Dual Prior 都有独立 head/loss、严格
   mask、真实全层特征训练、checkpoint、打分输出和机制诊断；完整训练不存在 OOM、NaN 或漏接 loss。
2. **部分辅助目标在小样本上可学。** H onset BCE 有 token/path 排序信号；direct key/complete
   prior 明显学到了小 dev 上的 membership；Consistency 能产生训练内几何变化。
3. **三个模块改善最终 Best-of-N 的证据尚未建立。** C1 和 H0 有正向点估计，但置信区间跨 0；
   direct/mutual prior 没有 ranking 增量；gold tail 回退；Full 不优于 correctness-only；C1 与 H0
   的干净组合 CH0 出现负交互信号。

因此，`configs/best_current.json` 表示“当前完整、可维护的方法实现”，不表示“已经被实验选出的
最优效果配置”。当前适合做阶段性模块筛选，不适合写成三模块联合有效的最终结论。

## 2. 共同骨架：最终到底按什么分数选择

### 2.1 输入不是文本，而是 frozen LLM 的 exact-token hidden states

当前任务模型是 frozen 的 Phi-3.5-mini 路径。每条 trajectory 保存原始 `prompt_token_ids` 和
`output_token_ids`，用一次 teacher-forced forward 提取：

```text
embedding + 32 transformer blocks = 33 层
每层宽度 3072
每个 output token 的原始特征宽度 = 33 × 3072 = 101376
存储 dtype = BF16
```

模型不重新分词，也不从解码文本猜 token 位置。trajectory feature 只覆盖生成 token；prompt-only
condition feature 单独保存，并在模型中通过 token-specific cross-attention 融入每个生成 token。

`layer_transformer` encoder 先沿 33 层这一轴把 `101376` 维特征压到 `model_dim=768`。三个模块和
correctness backbone 共享这一套 encoder、condition fusion 和 token feature。换句话说，辅助 loss
即使不直接写进最终公式，也可能通过更新共享表示间接改变最终分数。

### 2.2 最终 scalar score

对第 `t` 个生成 token 的共享特征 `f_t`，基础 reward backbone 预测：

```text
gate_t  = sigmoid(g_t)
value_t = v_t
```

当前 progress score weight 为 0，因此 `value_t` 就是 token reward head 的输出。trajectory score 为：

```text
token_score = Σ_t gate_t × value_t / Σ_t gate_t
final_score = token_score + pooled-trajectory residual
```

训练中的 correctness BCE 直接让 `final_score` 预测答案是否正确。打分时，对同一个 query 的前 K 个
冻结候选分别计算 `final_score`，选择分数最大的一个；BoN@K 检查这个被选 trajectory 的
correctness。候选顺序和 tie 规则是冻结的。

下面是三模块与最终分数的关系：

```text
exact-token all-layer features
          │
          ▼
 shared encoder + condition fusion ───────────────┐
          │                                       │
          ├─ gate/value ──► scalar score ──► argmax@K
          │
          ├─ Consistency：训练表示，并直接约束同义轨迹的 scalar score
          │
          ├─ H onset head：H0 只经共享表示间接影响 score
          │                 H1 tail 直接训练 score 使用的 value_t
          │
          └─ Key/Complete heads：direct/mutual 经共享表示间接影响 score
                                fused prior 还以 .25 loss 对齐 reward gate
                                gate 再直接参与 scalar score
```

一个容易混淆但非常重要的事实是：推理时不会把 `hallucination_probability` 直接从 reward 中减掉，
也不会把 `key_prior/complete_prior` 直接乘到 reward 上。prior 的“直接接入”发生在训练时：detached
fused prior 约束本来就参与 score 的 reward gate。所有 cell 的 forward score 公式仍相同，差别来自
训练时哪些 loss 改变了共享参数、gate、value 或 residual。

## 3. 模块一：Semantic/Style Consistency

### 3.1 想解决的问题

同一道题、推理实质相同但表达长短或 style 不同的两条 trajectory，应该得到相近的表示和分数；仅仅
共享某种表面 style、但语义不同的轨迹，不应该因为风格相似就聚在一起。

### 3.2 代码如何实现

模型先对每条 trajectory 的 token features 做 masked mean pooling，再经过 projector 和 L2
normalization，得到表示 `z_i`。Consistency loss 构造两类 batch 内 pair：

- positive：`semantic_id` 相同、`style_id` 不同；
- negative：`semantic_id` 不同、`style_id` 相同。

loss 包含三项：

```text
positive representation = mean(1 - cosine(z_i, z_j))
negative separation      = mean(relu(cosine(z_i, z_j) - margin))
score consistency        = mean((score_i - score_j)²) on positive pairs
```

当前 margin 为 `.2`，positive/negative 外层权重为 `1/1`，score consistency 权重为 `.1`。
`SemanticGroupBatchSampler` 把同一 semantic group 的 views 放进同一个 mini-batch，否则有标签也产生不了
positive-pair loss。缺少 `semantic_id/style_id` 的普通 correctness row 继续训练 outcome BCE，但不被当作
consistency negative。

实现入口是 `src/consistency_localized_reward.py` 中的
`prism_style_consistency_loss`，数据组批入口是 `src/clir_data.py` 中的
`SemanticGroupBatchSampler`。

### 3.3 如何影响最终选择

Consistency 没有独立的 inference-time 加分项，但有两条训练路径：

1. representation loss 更新共享 encoder/condition features，随后 gate、value 和 residual 都可能改变；
2. score-consistency 项直接约束同语义不同 style 的最终 scalar score 接近。

因此它希望减少 reward 对表达风格的敏感性。但如果 negative separation 相对过强，或者监督组太少，
它也可能为了拉开跨语义表示而重排原本有用的高分极值。

### 3.4 当前效果

三 seed 的 matched screen 中：

| 指标 | C0 correctness | C1 + consistency | 变化 |
|---|---:|---:|---:|
| BoN@16 | `91.73%` | `92.20%` | `+0.47` point |
| within-query pairwise | `68.60%` | `69.37%` | `+0.77` point |

BoN@16 三个 seed 的增量为 `+0.2/+1.0/+0.2` point，方向一致；fixed-seed query
bootstrap 95% interval 为 `[-0.47,+1.40]` points，仍跨 0。

阶段裁决是：**保留为值得扩量复测的候选，但还不能声称有效。** 当前没有 held-out consistency
relation set，所以不能用训练中的 relation cosine gap 证明它学会了 unseen 语义/风格泛化。

### 3.5 训练数据怎样得到

当前 27 对 relation 来自 Phi 自身的 on-policy 候选，不是把一句话机械替换词汇得到的合成 rewrite：

1. 从原始 GSM8K train-primary query 的 8 个 Phi 候选中，寻找同题、correctness 相同、归一化答案相同、
   文本不同的 trajectory pair；
2. 历史高召回阶段得到 125 个候选 pair。早期 flat verifier 给出的 121 个 accept 后来因没有真正执行
   双向 material-claim 审核而被撤销，**这些旧标签没有继续用于当前训练**；
3. 对当时发布 roster 中的 31 对重新制作 blind items，由两份独立标注检查前提、方法、关键中间量、
   结论和内部错误是否保持；
4. 两份判断一致 30/31，唯一分歧单独裁决，最终留下 27 accept、4 reject；
5. 每对按 exact output-token 长度标成 `native_compact/native_expanded`，不再用候选序号冒充 style；
6. 把通过的 27 对映射回 base outcome manifest 中已经存在的 54 行，只增加
   `semantic_id/style_id`，不复制 trajectory，也不继承 H 或 prior 标签。

所以 Consistency 的真正独立监督单位只有 **27 个关系组**，54 个 view 全部是 correct trajectory。
训练 manifest 虽能构造许多 different-semantic/same-style 对照，但它们不是数百个独立人工 relation
标签；当前也没有独立 relation dev。

## 4. 模块二：Hallucination Localization 与 Negative Tail

### 4.1 想解决的问题

最终答案 correctness 只告诉模型整条轨迹对不对，不能区分“前面有价值、从某一步开始出错”。H 模块希望
定位最早一个 unsupported 或 contradicted reasoning claim，并把首错后的 continuation 与首错前分开。

当前字段语义是：

- `hallucination_onset=k`：第 `k` 个生成 token 是首个问题 claim 的起点；
- `hallucination_onset=-1`：这条 trajectory 经明确标注为 clean；
- 字段缺失：未标注，必须 mask，不能当 clean；
- 从 `k` 起监督 H target 为 1，严格含义是 contaminated tail，不是声称尾部每个 token 都独立产生了幻觉。

### 4.2 代码如何实现

Hallucination head 对每个 `f_t` 输出一个 H logit。显式 onset 产生 token target：

```text
y_t = 0, t < onset
y_t = 1, t >= onset
clean row: 所有有效 token 的 y_t = 0
```

`Hallucination onset BCE` 对这些 token 做 BCE。它就是 clean 消融中的 H0。

同一个 gold onset 还可以直接约束 score 使用的 `value_t`：

```text
target value_t = -0.5, t >= onset
tail margin    = relu(value_t + 0.5)²
```

`token_reward_weight=.5` 加 MSE，`tail_weight=.5` 加 margin；这就是 H1 相比 H0 多出的
`Onset→negative tail reward`。因为 `value_t` 进入 gate-weighted scalar score，这条路径是三模块中当前
最直接的 reward coupling。

代码还保留：

- Path MIL：只有整条 path 是否 hallucinated 的弱标签，用 log-space noisy-or 训练 H head；
- Pseudo-onset tail：没有 gold onset 时，用当前 H head 第一次过阈值的位置产生 tail loss。

这两项当前权重都是 0。现有 boundary 在独立数据上没有通过，开启 pseudo-tail 会把 head 自己的定位错误
循环写回 reward；Path MIL 也只能说明“某处有问题”，不能给出可靠首错位置。

### 4.3 如何影响最终选择

必须把 H0 与 H1 分开：

- **H0 onset BCE**：`hallucination_logits` 本身不进入 scalar score。它只通过共享 encoder 的梯度间接改变
  gate/value/residual，所以 H0 的 ranking 变化不是“检测到幻觉后直接扣分”。
- **H1 onset BCE + gold tail**：除了 H0 的间接路径，还直接把 gold onset 后的 `value_t` 训练为负值；
  推理时不需要 gold onset，期待模型把学到的局部 value 行为泛化到新 trajectory。

### 4.4 当前效果

| 指标 | C0 | H0 onset BCE | H1 onset + tail |
|---|---:|---:|---:|
| BoN@16 | `91.73%` | `92.67%` | `91.87%` |
| pairwise | `68.60%` | `67.53%` | `67.78%` |

H0 相对 C0 的 BoN@16 点增量为 `+0.93` point，三个 seed 都为正，但 95% interval
`[-0.13,+2.07]` points 跨 0。H0 的 pairwise 反而下降 `1.07` points，说明它目前更像改变了
top-of-16 选择，而不是全面改善所有 correct/wrong pair。

H1 相对 H0 三个 seed 都下降，均值 `-0.80` point。机制 dev 上 H1 的
`post-onset value - pre-onset value ≈ 0`，但所有 token value 整体移到约 `-.62`。也就是说，当前 tail
没有学成“只在首错后下降”，而是发生了 global value shift。

H0/H1 有一些 token/path AP、AUROC 信号，但三 seed 汇总的 onset `±5` 仍为 `0/6`。阶段裁决是：

- H0 保留为扩量后的定位/排序候选；
- 当前 gold-tail coupling 未通过 locality 和 ranking 增量门；
- MIL/pseudo-tail 不开启；
- 不能把 H0 的正向点估计写成稳定收益。

### 4.5 训练数据怎样得到

H 数据不是从 correctness 自动推导的，生产流程是：

1. 从 4096-row base train 中冻结抽 64 条不同 query 的 Phi 原生 trajectory；correct/incorrect 各 32 条，
   每类按 output-token 长度分四个 bin 抽样，并排除 Consistency 重审过的 31 个 query；
2. blind item 只显示 problem 和 trajectory，不显示 correctness；
3. primary 标注器逐步标 `supported/contradicted/unsupported/non-claim/uncertain`，给出第一个问题 claim
   的 exact quote；程序只能做受限的 whitespace/index 合同修复，不能改语义判断；
4. 第二份标注在同一 64 条上独立盲标。原始 path agreement 为 `52/64=.8125`，kappa `.577`；双方共同
   判 positive 的 15 条中，onset 在 `±5` 内只一致 `5/15`，表明 boundary 本身噪声很大；
5. 对 path/onset 冲突制作 22-row A/B package，内部盲裁后得到 41 clean、23 hallucinated；该结果明确
   标记为 `not_gold_internal_blinded_pipeline_pilot`，不是论文级人工 Gold；
6. 按 query 划成 48 train（31 clean、17 positive onset）和 16 mechanism dev（10 clean、6 positive），
   onset 通过保存的 Phi `output_token_ids` 精确映射，不重新分词。

当前 clean 训练实际用的是这套 explicit onset/clean 标签。历史 `panzhixin` 后期保留的 sparse reviewed-span
H 是另一种 target，不能与当前 onset-to-full-tail 结果合并解释。

## 5. 模块三：Key/Complete Dual Prior

### 5.1 想解决的问题

Dual Prior 希望模型区分两张证据图：

- `key`：最小的决定性证据或决定性错误；
- `complete`：从 prompt 到 trajectory 结论所需的较完整、非冗余支持链。

错误 trajectory 也可以标 prior：决定性 flaw 可以是 key，模型实际使用的 material attempted chain 可以是
complete。因此 prior 表示“应重点看哪里”，correctness 表示“看完后结果对不对”，二者不是同一个标签。

### 5.2 代码如何实现

共享 token feature 上有两个独立 linear head：

```text
key_prior_logits       -> sigmoid membership probability
complete_prior_logits  -> sigmoid membership probability
```

direct loss 分别对外部二值 token target 做 BCE。相同 logits 再沿完整 trajectory 做 masked softmax，得到
`key_prior` 和 `complete_prior` attention map。

当前 P1 还保留原始双向 stop-gradient mutual distillation：

```text
key map      拟合 stopgrad(complete map)
complete map 拟合 stopgrad(key map)
```

权重为 `.25`。`P0` 只有 direct BCE，`P1` 是 direct BCE + mutual。

代码还实现了 `0.5 key + 0.5 complete` fused prior、reward gate 对齐和 complete reconstruction 接口。
当前 `gate_prior_weight=.25`：归一化 reward gate 对 detached fused prior 做 full-trajectory squared-L2，
更新 gate head 和共享 encoder；`reconstruction_weight=0`。reconstruction 没有独立外部 768-d target；
用同一 candidate 的 pooled feature 会形成平凡自重构 shortcut，因此保持关闭。

### 5.3 如何影响最终选择

在当前 clean 配置里，key/complete/fused prior 仍不作为独立项进入 forward score 公式，但
prior→reward gate 对齐已经默认开启。影响路径分两部分：direct/mutual loss 通过共享 encoder 间接改变
gate/value/residual；同时 detached fused prior 以 `.25` 的 alignment loss 训练 reward gate，而 gate
直接决定 score 中各 token 的权重。

这解释了为什么“prior target 能学会”“gate 对齐变好”和“BoN 提升”仍是三件不同的事。对齐路径开启
只证明 prior 获得了一条更直接的训练路由，不自动证明这个 inductive bias 对最终选择有益。

### 5.4 当前效果

| 指标 | C0 | P0 direct | P1 direct + mutual |
|---|---:|---:|---:|
| BoN@16 | `91.73%` | `91.80%` | `91.80%` |
| pairwise | `68.60%` | `68.91%` | `69.50%` |
| key AUROC（16-row dev） | `.508` | `.663` | `.662` |
| complete AUROC（16-row dev） | `.517` | `.869` | `.870` |

P0 相对 C0 的 BoN@16 只有 `+0.07` point，seed 方向不稳定；P0→P1 的 BoN@16 均值为 0。P1 没有进一步
提高 key/complete AP/AUROC，也没有降低 held-out map discrepancy。

阶段裁决是：**direct membership 在这个小 dev 上可学；ranking efficacy 和 mutual 的增量价值未建立。**
冻结的 P0/P1 数字都来自 gate-off cell。后续独立 gate v2 中，`.25` 的 BoN@16 为 `91.87%`，
P0 为 `91.80%`，差 `+0.07` point 且两个配对区间跨 0；`10` 的 raw point estimate 为
`92.07%`。因此当前默认打开 `.25` 是方法身份和保守工程选择，不是 ranking efficacy；
reconstruction 仍关闭。

### 5.5 训练数据怎样得到

Prior 为减少 token identity 工程风险，复用了 H 的同一 64 条 trajectory 和 48/16 query split，但进行的是
一套完全独立的 evidence annotation，不继承 correctness、H path 或 onset 判断：

1. deterministic segmenter 先按非空行、句末标点和标题把 trajectory 切成固定 reasoning units；
2. 两份独立标注都只看到 problem、trajectory 和 unit list，各自选择 key/complete unit indices；
3. 两份标注在结构和可用性门上通过；key macro unit F1 `.547`，complete `.846`；
4. 16/64 完全一致，另外 48 条进入 role-blind A/B adjudication：adopt-A 13、adopt-B 24、synthesize 11；
5. unit selection 通过冻结 Phi token identity 映射成逐 token 二值 target。最终 64/64 usable，63/64 满足
   严格 `key ⊂ complete`；
6. train 48 条包含每个 head 14,307 个 token target，dev 16 条包含 5,008 个；key/complete 的正 token
   micro fraction 约为 `.084/.341`。

这里也有已知 shortcut 风险：44/64 的 key units 全部位于轨迹最后四分之一。因此评价必须保留 position-only
baseline，且不能把 14,307 个强相关 token 当成 14,307 个独立样本。真正独立的 prior train trajectory
仍只有 48 条。

## 6. 三套监督怎样合并进当前训练集

基础 outcome 数据原为 512 个 GSM8K train query × 8 个 Phi 候选，共 4096 行。为把 16 个 mechanism dev
query 完整隔离，训练中删除这 16 个 query 的全部 128 个候选，得到：

```text
train:             496 queries × 8  = 3968 rows
mechanism dev:      16 queries × 1  =   16 rows
ranking validation:500 queries × 16 = 8000 rows
```

所有 3968 行都有 checker v5 correctness，当前分布为 3590 correct / 378 incorrect。模块监督以
exact row ID、query ID 和 output-token hash overlay 到已有 trajectory：

| 训练行类型 | 行数 | 监督 |
|---|---:|---|
| correctness only | 3866 | correctness BCE |
| correctness + consistency | 54 | 27 个 relation pair；与 mechanism 行不重叠 |
| correctness + H + prior | 48 | 17 positive onset + 31 clean；key/complete target |

缺失字段保持缺失，各自有独立 mask。普通 correctness row 不会被当作 clean H、negative consistency 或
全零 prior。每个 row 每 epoch 只访问一次，没有为了稀疏辅助监督而 oversample。

ranking validation 是另一个 query-disjoint 的 500×16 冻结候选池。当前 screening 的 train correctness
来自 checker v5，而 ranking population 的历史 label 是 checker v4；所有 cell 一致，因此可做 matched
screen，但正式复测应统一 checker。

## 7. 单模块和组合结果总表

所有下列 clean cell 都使用同一 3968-row train、16-row mechanism dev、500×16 ranking population、
3 epochs 和 seeds 42/43/44，只改变 loss-family 权重。

| Cell | 实际组成 | BoN@16 mean ± seed SD | 相对 C0 | 阶段解释 |
|---|---|---:|---:|---|
| C0 | correctness | `91.73 ± .61%` | — | 共同 backbone baseline |
| C1 | C0 + consistency | `92.20 ± .40%` | `+0.47` pt | 三 seed 同向，区间跨 0 |
| H0 | C0 + onset BCE | `92.67 ± 1.10%` | `+0.93` pt | 最佳点估计，pairwise 下降，区间跨 0 |
| H1 | H0 + gold negative tail | `91.87 ± .58%` | `+0.13` pt | 相对 H0 `-0.80` pt，tail 拒绝 |
| P0 | C0 + direct priors | `91.80 ± .72%` | `+0.07` pt | target 可学，ranking 不稳定 |
| P1 | P0 + mutual prior | `91.80 ± .72%` | `+0.07` pt | mutual 无增量 |
| CH0 | C1 + H0 | `91.53 ± .42%` | `-0.20` pt | C×H0 负交互筛选信号 |
| Full | C1 + H1 + P1 | `91.60 ± .80%` | `-0.13` pt | 不优于 correctness-only |

原 7-cell 对 C0 的 paired query-bootstrap interval 全部跨 0。因此表中的正负数是筛选点估计，不是
统计上已经稳定的模块效果。

## 8. 模块组合后发生了什么

### 8.1 C1 与 H0：单独点估计为正，组合却下降

CH0 是专门补跑的干净二因子 cell：

```text
C0  = correctness
C1  = correctness + consistency
H0  = correctness + onset BCE
CH0 = correctness + consistency + onset BCE
```

CH0 的 BoN@16 为 `91.53%`，相对 C0/C1/H0 分别为 `-0.20/-0.67/-1.13` points。H0→CH0
三个 seed 都下降。二因子交互：

```text
CH0 - C1 - H0 + C0 = -1.60 points
```

三个 seed 的交互为 `-1.4/-1.2/-2.2` points。固定 seed 的 query bootstrap interval 排除 0，但把
训练 seed 也视为泛化维度后 interval 为 `[-3.40,+.13]` points，仍跨 0。

CH0 的 pairwise 为 `69.42%`，高于 H0 的 `67.53%`，却在 BoN@16 上更差。这说明组合没有让所有
correct/wrong pair 全面变坏，而是没有保住 H0 在每组最高分候选上的行为。Consistency 可能平滑或重排了
score 极值，这是与观测一致的解释，但现有实验没有证明因果机制。

### 8.2 H0 与 negative tail：定位 head 的点收益被直接 coupling 抹掉

H0→H1 三 seed 全部回退，且 H1 把整条 trajectory 的 token value 推向负区间，却没有形成稳定的
post-onset 相对下降。当前结果说明问题不是“tail loss 没接上”，而是它接上后学到了不希望的全局平移。

### 8.3 Direct 与 mutual prior：可学不等于能帮助选择

P0 在 16-row dev 上明显提高 key/complete target AUROC，但 BoN@16 几乎不变；P1 的 mutual 又没有增量。
这些冻结 cell 当时没有 gate coupling，所以该结果在机制上并不矛盾。当前默认新增了
fused-prior→gate 的训练路由，但 v2 的 `.25−P0` 仍只有 `+0.07` point 且区间跨 0：辅助 head
可以准确、甚至 gate 可以被拉近，最终 reward 仍未必获得可泛化的候选排序信号。

### 8.4 Full：所有训练路径同时存在，但收益没有相加

当前 clean Full 是：

```text
Full = C1 + H1 + P1
     = correctness
       + consistency
       + onset BCE
       + gold negative tail
       + direct key/complete priors
       + mutual prior distillation
```

Full 不是 H0，也不是单独 H1。它的 BoN@16 为 `91.60%`，低于 C0 的 `91.73%`。机制 dev 上 prior target
仍有信号，但比 P0 弱；H boundary 仍失败；tail 仍呈 global shift。旧 `panzhixin` seed-42 联合实验也曾
出现 Full/JALL 低于 correctness-only、H 和 key prior 在联合环境中退化的现象。因此“没有叠加”已经不是
单次偶然观察，但现有样本仍不足以判定某一对模块必然冲突。

### 8.5 当前最合理的原因假设

目前证据支持把以下因素作为下一轮要区分的假设，而不是已经证明的原因：

1. **共享表示梯度竞争。** 三个 auxiliary loss 都更新同一 encoder/condition features；C 的跨语义分离、
   H 的 onset 分类、prior 的证据 membership 未必需要同一几何结构。
2. **独立监督过少。** 27 个 relation groups 和 48 个 mechanism trajectories 相比 3968 个 outcome rows
   太稀疏；更多 epoch 主要是在重复同一批标签。
3. **目标与最终指标错位。** H/prior AP 可以提高，但 BoN 只关心每个 query 的最高分候选；平均 token
   分类或 pairwise 改善不保证 top-1-of-16 改善。
4. **Tail objective 的全局平移。** 当前 absolute negative margin 有低成本 global-shift 解，已在多轮
   实验中复现。
5. **标注 shortcut 与噪声。** H onset 双标距离大，prior key 偏后，Consistency 没有 held-out relation；
   模型可能学习位置、长度或训练组特征，而不是预期机制。

## 9. 当前可以和不可以写进结论的内容

可以写：

- 三模块代码、数据 mask、训练、评分和机制诊断已完整接入；
- C1 和 H0 在当前 screen 有正向 BoN@16 点估计，值得扩量复测；
- direct key/complete target 在小 dev 上可学；
- current gold tail、mutual prior 增量和 Full integration 没有通过效果门；
- C1×H0 在当前数据上有三个 seed 同向的负交互筛选信号。

不可以写：

- Consistency、H0 或 Dual Prior 已被证明稳定提高 Best-of-N；
- H head 已准确定位首错边界；
- key/complete prior 已通过 reward gate 改善最终选择；
- 三模块天然不兼容；
- `best_current` 是效果最优配置；
- 增加当前数据上的 epoch 可以替代扩充独立监督。

## 10. 下一步扩数据讨论的入口

下一轮不应直接继续跑旧 Full。建议讨论并冻结四类决策：

1. **Consistency 数据怎么扩。** 继续挖 Phi 原生等价候选，还是生成受控 style rewrites；relation verifier
   用什么门；如何建立真正 held-out 的 semantic/style relation set。
2. **H 数据怎么扩。** 首先增加可靠 positive onset 与 explicit clean，还是先修改 claim-boundary 标注协议；
   如何提高双标 onset 一致性，并控制 correctness、长度和绝对位置 shortcut。
3. **Prior 是否同步扩。** 是先用更多独立 trajectory 复核 direct target，还是暂时把 prior 从第一轮 C/H
   2×2 中拿开；如何平衡 key 位置并保留错误轨迹的 decisive-flaw 标签。
4. **Outcome/ranking population 怎么扩。** 统一 checker v5，增加独立 ranking query，明确 validation 与
   protected test；新数据必须让 `C0/C1/H0/CH0` 全部从头同预算重跑，不能把新 CH0 与旧 baseline 拼接。

现有结果文档见 [clean_ablation_v1_results.md](clean_ablation_v1_results.md)；当前方法设计见
[proposal.md](proposal.md)；历史数据生产、双标、裁决和联合训练的版本化产物保留在 `panzhixin`
分支的 `configs/`、`docs/` 与本地 `run_artifacts/` 中。
