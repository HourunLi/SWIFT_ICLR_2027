# Prior v12 事后探索子集协议

## 为什么另开名字

Prior v12 原协议已经终止，状态仍是
`STOP_PRIOR_V12_STRICT_CONSENSUS_DATA_GATE_FAILURE`。它失败的事实、失败门和原始
报告都不修改。本协议只响应后续明确指令“V12吧”，回答一个更窄的问题：现有 v12
标注里有没有一批足够保守、可以先做训练诊断的 Silver 数据。

因此，这批数据只能称为“事后探索性双 AI Silver 子集”，不能称为 Gold、人工验证、
v12 通过或确认性证据。

## 机械选择规则

800 条 natural proposal 全部进入固定分母。按下列规则一次性筛选，并保留全部合格行：

1. GPT-5.6-sol xhigh 与 Claude Opus 5 high 都判 `usable`，且都不是 low confidence；
2. 两边都只给一个 Key，并且 Key unit 完全相同；
3. 两边 Complete 都非空，并且 Complete unit 集合逐项完全相同；
4. 如果某条 natural row 被任一标注者抽到 self-repeat，而该 repeat 的
   eligibility、Key 或 Complete 与原标注不完全相同，则排除该 natural row；
5. 不按 CLIR score、正确/错误、置信度高低、题源或结果好坏补配额；按原先冻结的
   `selection_priority` 排序，保留所有合格行。

这条规则得到 253 条：202 train、51 dev。它们覆盖两个题源以及数值匹配/不匹配，
但明显偏向“两位 AI 都容易判断”的推理，因此不代表原始 800 条的自然分布。

## token target

Key/Complete 标注仍在 unit 层。materialization 只用保存的 output token IDs 与冻结的
unit `[token_start, token_end)`：被选 unit 覆盖的 token target 为 1，其余完整输出轴 token
为 0，Key/Complete mask 都是全 1。模型的 Key/Complete attention 仍在完整 trajectory
上归一化，不在正样本或 material-claim 子集上重新归一化。

## 第一阶段实验

只做 matched `R0 vs P0`：两格共享完全相同的 3,968 条历史 correctness 数据和 202 条
新 Prior train 行，结构、初始化、采样、epoch 与学习率一致。

- R0：只开最终 correctness loss；
- P0：在 R0 上只增加 direct Key BCE 和 direct Complete BCE；
- mutual distillation、gate coupling、Consistency、H0/H1、MIL、pseudo-tail 全部关闭。

51 条 dev 只看 Key/Complete 是否可学以及 correctness 是否明显退化。只有 direct Prior
可学，才允许另冻下一阶段 gate-off/on；不会根据本轮结果在同一 dev 上调权重。

排序效果如后续执行，只能复用与 v12 query/cluster 隔离的 v7.4 fully-labeled ranking
population，属于探索性复用，不是新的确认性测试。

## 已知限制

- v12 原门中 annotator A controls 只有 11/16，A/B self-repeat 都只有 51/80；
- 当前规则只排除了“被抽到且 repeat 失败”的行，未被 repeat 抽到不等于已验证稳定；
- 253 条由事后完全一致筛选产生，存在显著 easy-sample selection bias；
- dev 的数值不匹配行较少，不能据此宣称 Prior 对错误推理普遍有效；
- 没有人工复核，不能声称标签客观准确。
