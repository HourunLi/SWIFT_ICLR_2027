# Prior v17：机械 Key + 剩余 block 二分类 smoke

v16 的 600 条规模化双标没有通过。问题主要不是 JSON、切分或 token 映射，而是两个 AI 对“主链应该
有多宽”的理解不同：一个经常把计划句、题面复述和通用公式算进主链，另一个经常删掉它们。继续用
同一套七类角色 prompt 重标，只会把已经看过的失败批次变成事后调参。

v17 因而换了一个更简单的前瞻目标，并永久排除 v12--v16 用过的全部 query 和模板簇。程序先做：

- 找出最后一条明确包含候选数值答案的计算，把整块固定为 Key；
- 固定排除明显的计划/标题、纯题面复述、没有代入的通用公式、完整重复、答案包装；
- Key 后面只允许没有新计算的答案总结，并全部固定为 Complete 负例；
- 如果找不到安全 Key、Key 后还有新计算，或剩余待判 block 少于 2，候选直接不进 smoke。

AI 只看剩余的 Key 前 block，并逐个回答 `used|not_used`。标准只有一个：从固定 Key 往回追，删掉
这块后，还能不能沿候选自己写出的路线理解和核验 Key。算错的路线也照写出的实际依赖判断，不修题；
AI 不再输出 path、首错、七类 role、final block、依赖边、Key 或 Complete。

两边一致标 `used` 的剩余块与机械 Key 构成 Complete 正例；一致标 `not_used` 的剩余块与机械排除块
构成负例；分歧块遮住。这个规则保持现有 exact-token Key/Complete、Prior loss 和 `.25` main-style
Gate 接口不变，只简化标签生产。

## 冻结样本与质量门

v17 是 96 条全新 natural 的非训练 smoke：84 GSM8K、12 MATH，同时覆盖 numeric match/mismatch
和 medium/long。每侧 6 个 shard，每个 `16 natural +2 hidden controls +4 cross-shard repeats=22`，
共 132 行；两侧分别使用 GPT-5.6 Sol/max 与升级后的 Claude Opus/max，互盲且不得查看 private、
checker、参考答案、历史标签或对方输出。

冻结门包括：control 至少 11/12；24 个 self-repeat exact 至少 `.95`；low-confidence 不超过 `.10`；
自然样本残余 block agreement 至少 `.90`、Cohen κ 至少 `.65`、整行 exact 至少 `.45`；Complete
unit IoU 至少 `.85`、共识 mask coverage 至少 `.95`；双方共同 used 与共同 not-used 各至少占 `.05`，
且 Complete union 覆盖全部 material block 的退化比例不超过 `.25`。

v17 失败后不能改 prompt、门或挑容易子集；按用户授权，届时只能另冻、另命名
`v16-posthoc replay`，原 v16 的失败结论保持不变。v17 即使通过也只是证明这个简化定义在双 AI 上
可稳定操作；96 条 natural 永远不训练，通过只允许另立 fresh scale-v18，不能直接宣称 Prior 标签
准确、Prior/Gate 有效或 Best-of-N 提升。

## 执行顺序

```bash
python prepare_clir_prior_binary_v17.py prepare
python prepare_clir_prior_binary_v17.py verify
```

独立双标完成后才允许执行一次：

```bash
python prepare_clir_prior_binary_v17.py evaluate
```

完整冻结字段、父文件 hash、配额、模型边界和 fallback 见
`configs/data_expansion_prior_v17/protocol.json`。
