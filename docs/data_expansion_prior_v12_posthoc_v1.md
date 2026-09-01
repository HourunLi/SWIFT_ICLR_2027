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

这里的 3,968 行沿用项目原训练清单，其中已有 48 条早期 Key/Complete 标签。换句话说，
R0 仍只使用全部 4,170 行的 correctness；P0 使用“旧 48 + 新 202 = 250 条”Prior
监督。这是把扩量数据接到原模块数据上的口径，不是只用新 202 条替换旧数据。精确监督量
已另行冻结在 `posthoc_v1/training_authorization.json`。

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

## 执行结果

这条路线已经执行完，不再是待运行协议。

- 800 条原始 proposal 中，266 条同时满足 A/B exact singleton-Key 与 exact nonempty
  Complete；再排除 13 条任一已观测 self-repeat target 漂移的 parent，最终为 253 条，
  其中 train/dev=`202/51`。
- selected-only feature inventory、最长样本全宽 preflight、506 个 trajectory/condition
  tensor 和完整复验均通过。raw BF16 feature 为 `21,209,075,712` bytes，约 `19.75 GiB`。
- matched R0/P0 六次训练均完成 3 epochs，checkpoint 可加载、state tensor 全 finite。
  两边共享 4,170 行；P0 实际使用旧 48 + 新 202 =250 条 direct Prior supervision。
- 51-row dev 上，P0 三 seed 的 Key AUROC/AP/BCE 均值为
  `.9038/.5956/.1453`，Complete 为 `.9612/.9352/.2702`；R0 分别为
  `.4883/.0625/.6160` 与 `.5274/.3805/.6826`。因此 direct target 可学。
- 冻结的 892-query ×16-candidate 复用排序评估也已完成。主指标 BoN@16 为
  R0 `.85725`、P0 `.85538`，P0−R0=`-.00187`，fixed-seed query 95% interval
  `[-.01308,+.00897]`，exploratory hierarchical interval `[-.01570,+.01121]`。
  BoN@8 为 R0 `.86173`、P0 `.84604`，三个 seed 都下降；两区间为
  `[-.02653,-.00486]` 与 `[-.02952,-.00224]`。pairwise 从 `.66820` 降到 `.65937`。

最终状态是
`COMPLETE_PRIOR_V12_POSTHOC_EXPLORATORY_R0_P0_RANKING_EVALUATION`。本地冻结排序
summary SHA-256 为 `dd8cc22b11e3ff201b316c0c4c1b4a268f710fb9c680106476511335c9ff5bf2`。
结论只能写成：post-hoc exact 双 AI Silver Prior 在 held-out 子集上非常可学，但 gate-off
共享表示路径没有改善最终排序，K=8 反而稳定回退。排序 population 是复用的探索性数据；
原 v12/v13 失败状态、无人工复核和 easy-sample selection bias 全部保留。mutual、gate、Full
仍未解锁。
