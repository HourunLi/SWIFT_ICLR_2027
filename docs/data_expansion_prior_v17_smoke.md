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

## 标注前冻结结果（2026-09-02）

代码与协议提交为 `b20831f9cf10a803d46d64542fa6fc704a20e31d`。正式包状态是
`PASS_PRIOR_V17_FRESH_BLIND_PACKAGES_READY`，随后从源 acquisition、历史排除清单和冻结 selector
独立重算，得到 `PASS_PRIOR_V17_PACKAGE_INDEPENDENT_RECOMPUTE`、`mismatches=[]`。两侧合计 12 个
公开 shard、264 行；每侧 natural 的残余二分类判断量为 855 个，不需要 GPU。

proposal/package-report/verification/private-index 的 SHA-256 依次为
`885fff97…4545` / `b1866207…298d` / `6860f26b…cd4a` / `e9d7be2…541d`。当前没有 label、evaluation、
feature 或训练；下一步只能让两个独立模型分别完成 A/B 公共包，再运行一次冻结 evaluator。

## 冻结执行结果

12 个 label shard 全部完成。严格前检确认 A/B 各 132 行、ID 与 package population 完全一致、四字段
schema 合法，并按序覆盖全部 residual block；两侧均无 low confidence。冻结 evaluator 随后只运行
一次，得到 `STOP_PRIOR_V17_MECHANICAL_KEY_BINARY_SMOKE`。raw report SHA-256 为
`3235aec6712705b4c505708201a6c7c7da8d7594b6fc5499cb6dd28d8cc2064c`。

自然数据本身的结果很好：两侧 self-repeat 都是 `24/24`；96 条 natural 的 855 个 residual 判断中，
agreement=`.9602`、Cohen κ=`.9194`、整行 exact=`.7813`、Complete unit IoU=`.9519`、mask
coverage=`.9804`，双方共同 used/not-used 分别占 `.5368/.4234`，没有退化或 low-confidence。所有
natural gate 都通过。

唯一失败项是隐藏 controls：A/B 均为 `8/12 < 11/12`。只读检查显示，我在四道题里把题面已经提供、
而 Key 又自包含的数量复述预设成 `used`，但 prompt 明确写着“题面始终可见，纯复述不应算 used”；
两位 AI 都按文字规则稳定判为 `not_used`。另有一条单位换算事实和一条冗余方程存在边界差异。这说明
本次 STOP 主要是预注册控制答案与书面删除规则错位，而不是自然样本双标再次崩坏；但冻结协议要求
all gates，所以仍必须终止，不能事后改 control 答案让 v17 通过，也不能训练这 96 条。

按用户在结果前给出的 fallback，下一步另立并明确命名 `v16-posthoc replay`：保留原 v16/v17 结果，
把修正后的删除规则重新用于 v16 的 600 条旧 population。该 replay 是事后开发/训练数据路线，不能
称为 prospective v16 或 v17 pass，后续效果仍需全新 query/cluster population 验证。
