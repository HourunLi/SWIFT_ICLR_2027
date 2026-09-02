# Prior v15 fresh role-only smoke（冻结版）

## 目的

v14 的 Complete F1/IoU/coverage 已通过，但候选依赖边漏失率和错误链 singleton Key 没过门。
只读诊断进一步确认：11 个 Key 分歧全部来自两个 AI 对“最早致命错误”的不同判断；把错误定位同时放进
H 和 Prior 会让模块职责重叠。

v15 改成更简单的结构目标：AI 只审 block 角色、path 和 final block；程序把全部 `main_step` 映射为
Complete，把 final main-step block 映射为 Key。错误从哪里开始只归 H。本轮不改模型、loss、Gate、
exact-token 映射或训练 manifest 接口。

## Fresh population

- 来源仍为冻结 v12 acquisition 的 train-only GSM8K/MATH rows；
- 排除 v12 的 800、v13 的 48、v14 的 48 个 proposal query/template cluster，共 896 个排除项；
- 新取 48 个 query、48 个 cluster；
- GSM8K/MATH × numeric match/mismatch × medium/long 八格各 6 条；
- 每题最多一条，选择只使用冻结 source/checker/length/SHA 字段。

这些 48 题是新的 prompt-development smoke，永远不能进入 scale train/dev/ranking/protected test。

## 标注人口

A/B 各四个 shard，每个固定为 12 natural +2 fresh hidden controls +4 cross-shard repeats，共 18 行；
每侧 72 行。A 使用用户报告的 GPT-5.6-sol/max，B 使用升级后的 Claude Opus/max；两侧必须隔离，
不得查看 `PRIVATE_*`、另一侧、checker、参考答案、历史标签、代码或 evaluator。

AI 每行只能输出 eligibility、path_status、所有 block role、final_block_id、confidence、rationale。
不能输出 dependency edge、Key 或 Complete。候选引入的变量/别名/方程只要后面使用，必须算
`main_step`；错误路径中 Key 仍是 final structural step，而不是 earliest error。

## 冻结 raw gates

必须全部满足：

| 指标 | 门槛 |
|---|---:|
| controls | A/B 各 8/8 |
| exact target repeat | A/B 各 ≥15/16 |
| common usable non-low | ≥40/48 |
| path agreement | ≥.90 |
| final-block agreement | ≥.90 |
| structural Key macro F1 | ≥.90 |
| role-derived Complete macro F1 | ≥.90 |
| Complete positive IoU | ≥.80 |
| Complete mask coverage | ≥.90 |
| role agreement | ≥.85 |
| Complete union=全部 material | ≤.25 |

任何失败都终止 v15：不修 label、prompt、control 或阈值，不裁决/重标，不挑子集，不抽 feature，
不训练。全部通过也只允许另冻 scale-v16；本 smoke 永远不是训练数据，也不支持 Prior/Gate/Full 的
Best-of-N 效果结论。

## 执行顺序

```bash
python prepare_clir_prior_role_v15.py prepare
python prepare_clir_prior_role_v15.py verify
```

两个 package population 都完成后，才允许只运行一次：

```bash
python prepare_clir_prior_role_v15.py evaluate
```
