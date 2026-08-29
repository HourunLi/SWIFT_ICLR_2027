# CLIR Consistency v6.1：C0/C1 训练与留出关系评估协议

状态：`AUTHORIZED_NOT_YET_EXECUTED`

授权日期：2026-08-29

机器授权：`configs/data_expansion_scale_v6/consistency_training_v6_1/authorization.json`

## 1. 这一轮到底回答什么

这轮只回答一个窄问题：扩充后的 Consistency 数据，能不能让模型在**没参与训练的题目**上，把“同一道题、同一种解法的简写和详写”识别得更相似，同时把“不同题但长度相近的解答”区分开。

它不是三模块联合实验，也不是新的 Best-of-N 效果实验。原因很简单：这轮有新的 Consistency 关系留出集，但没有一套新的、独立的多候选 ranking 验证集。因此本轮可以判断“这个辅助目标有没有学会它自己的任务”，不能据此写“最终选答案能力提高了多少”。

## 2. 两个完全匹配的训练组

| 组 | final correctness | Consistency | 其他模块 |
|---|---:|---:|---:|
| C0 | 1.0 | 0.0 | 全关 |
| C1 | 1.0 | 1.0 | 全关 |

C0 和 C1 使用完全相同的数据行、batch 顺序、网络、优化器、学习率、seed 和 epoch。配置唯一差异是 `consistency_weight=0` 或 `1`。这样 C1 相对 C0 的变化才可以归到 Consistency，而不是归到“多看了数据”或“换了训练顺序”。

固定 seed 为 42、43、44，固定训练 3 个 epoch。seed 42 的两个组先各跑 1 个 epoch，目的只是发现 OOM、NaN、坏路径或断点续训问题；只按工程门决定是否从同一 full-state checkpoint 续到 epoch 3，不能根据第 1 个 epoch 的效果好坏临时停训、延长或改参数。

## 3. 训练数据怎么拼

共享训练清单由两部分构成：

1. 旧的 3,968 条 correctness 训练轨迹，继续教模型“最终答案对不对”；
2. 新的 400 对 Consistency 训练关系，共 800 条轨迹。它们都通过冻结 numeric checker，因此 correctness 记为 1；同一对共享 `semantic_id=relation_id`。

总计 4,768 条。旧数据里原先少量的 Consistency、H 和 Prior 字段全部从本轮构造清单中去掉，避免偷偷混入旧辅助监督。H、Dual Prior 和 Full 不训练。

每个新正对中的短回答叫 `relative_compact`，长回答叫 `relative_expanded`，只比较保存的 `output_token_ids` 数量；若长度恰好相等，就用 trajectory ID 做固定 tie-break。这里的 compact/expanded 只是“这一对里相对短/长”，不是新增 AI 标签，也不声称短回答一定更好。

batch size 固定为 4。分组 sampler 每个 batch 放两个 relation：

- 同 relation 的 compact/expanded 是应该拉近的正对；
- 不同 relation 的 compact/compact、expanded/expanded 是应该分开的 batch 内负对。

3,968 条旧行没有 `semantic_id/style_id`，只贡献 correctness loss。C0 仍使用同一个分组 sampler，因此两组看到的 batch 身份和顺序匹配。

## 4. 留出集怎么用

冻结留出集包含：

- 150 对正关系：同题、同方法的简写/详写；
- 150 对 hard negative：不同题、长度和题源尽量匹配，专门防止模型只靠长度或题源猜。

训练关系与留出关系的 query 和 template cluster 交集都必须为 0。旧 3,968 条 correctness 题与新训练/留出池按 source identity 核对也必须为 0。

留出正负关系本身并非 300 个完全独立实验单位：负关系构造时复用了留出池的一部分 view，目前正负集合共享 43 个 endpoint、44 个 query/cluster。评估会逐 relation 做 C0/C1 配对，并把正、负两层分别 bootstrap；区间只能叫“relation-level 描述性不确定性”，不能假装消除了 endpoint/query 相关性。

## 5. 指标怎么读

主指标是：

```text
正关系平均 cosine 相似度 - 负关系平均 cosine 相似度
```

越大越好。C1-C0 若为正，表示打开 Consistency 后，同解法正对相对更近、hard negative 相对更远。还会报告：

- cosine 区分正负关系的 AUROC/AP；
- 正对和负对各自的 cosine 分布；
- 正对 scalar score 差距，以及负对与正对的 score-gap separation；
- representation centroid norm 和坐标方差，用来发现“所有表示挤成一个点”的塌缩。

确认性判断规则固定为：3 个 seed 的主指标增量都大于 0，且固定-seed relation bootstrap 95% 区间下界大于 0，才写“支持 C1 改善留出关系分离”；否则写“不确定”或“方向相反”。无论结果如何都完成既定 3 seed，不拿效果门挑 seed。

`score_consistency_weight=0.1` 会让同一正对的 scalar score 更接近，所以这轮也能直接观察 C 对 score-gap 的影响。但它仍不等于 Best-of-N：留出关系没有 correct-vs-wrong 候选排序任务。

## 6. 执行门与固定顺序

正式训练前必须：

1. hash 核对旧训练清单、1,357 条 extracted feature、400/150/150 三份关系和独立 verifier；
2. 确定性生成 4,768-row 共享训练清单、300-row 正对 validation view 和 557-row 留出 endpoint 清单；
3. 独立重算三个清单并核对行、顺序、文件 hash、query/cluster 边界；
4. 在 clean commit 上对 C0/C1 各跑一个真实 `[B,T,101376]` BF16 forward/backward，检查所有关键输出、loss 和梯度 finite；
5. 先完成 toy/full test，再跑 seed 42 的 1-epoch 双组 pilot；
6. pilot 的数据身份、checkpoint、loss、梯度和评估器均通过后，从原 checkpoint 续到 epoch 3；同时跑 seed 43/44 到 epoch 3；
7. 每个最终 checkpoint 都在相同 150+150 关系上评估，最后做配对多 seed 汇总。

任何 hash、split、全宽、finite 或 checkpoint provenance 门失败，都停止并修实现；不得换样本、改关系、改阈值或只保留好看的 seed。

## 7. 证据边界

允许的表述：

- C0/C1 训练和断点续训是否在扩充数据上稳定跑通；
- Consistency loss 是否降低；
- C1 相对 C0 是否改善冻结留出正负关系的 representation/score-gap 分离；
- 3 seed 的方向与 relation-level 区间。

禁止的表述：

- 不得说本轮证明 Best-of-N 或最终选答案性能提高；
- 不得说 H、Dual Prior 或 Full 得到验证；
- 不得把双 AI Silver 标签称为 Gold 或人工验证；
- 不得把 relation bootstrap 当成 query 完全独立的统计证明；
- 不得根据本轮结果回头修改已冻结样本或只汇报最好的 seed。

本协议完成后，下一步才是决定：Consistency 机制若可学，是否值得另外构建新的 query-disjoint ranking population，重跑 C0/C1/H0/CH0 的最终效果实验。
