# CLIR Dual Prior v11：逐单元验算后的全新确认轮

## 1. 为什么再做一轮

v10 已按冻结协议终止，不能改标签或挑 54 条一致行救场。它的 60 条自然样本实际上通过了全部语义、数量、coverage、repeat 和反退化门：Key/Complete F1 为 `.9000/.9253`，非低置信 exact Key 为 `54/60`，Complete unit agreement 为 `.9291`。唯一失败是标注者 A 在一条隐藏题里漏看 `7+5=13`，控制题为 `7/8`，低于预注册 `8/8`；B 为 `8/8`。

用户随后明确授权“再试一次”。v11 是全新确认轮，不改 v10 结论，也不重用它的自然题或控制题。

## 2. 唯一 prompt 改动

Key/Complete 定义、模型、loss、partial mask、mutual 和固定 `.25` main-style gate 路径全部不变。v11 只要求标注者在选标签前按 unit 顺序独立检查：

- 每个算术与代数等式；
- 单位与换算；
- 人物、物体和数量关系；
- 题目实际询问的量；
- 后续错误是否只是在传播更早的错误。

错误行的 rationale 必须指出具体错误，例如“6×8 应为 48，不是 46”。这不增加标签字段，只让已有“最早致命错误”规则真正建立在逐单元核验上。

## 3. 全新数据与盲包

本轮继续使用已经完成 checker v3 与 exact-token unitizer v2 的 v6 16,000-row 池，不重新 rollout、不使用 GPU。选择前排除 v6.1 C、v7 H、v7 ranking、v8、v9、v10 的全部 query 与 template cluster。

固定选择 60 条：GSM8K/MATH × numeric match/mismatch 各 15；每个 query 和 cluster 至多一条，要求 train-acquisition、normal stop、可监督、unitization=ok、6–40 个 material claim。选择只按冻结 hash，不看 CLIR 分数、参考答案或 AI 标签。60 个 query 永久作为 prompt-development smoke 烧掉。

每位标注者收到一个 80 行盲包：

- 60 条全新 natural；
- 8 条全新隐藏控制，覆盖早期算术错、后期算术错、单位/所求量错、拆分计算、题面复述、未用旁枝、重复结果和 answer-only；
- 12 条各自独立 blind repeat。

A 仍为用户报告的 GPT-5.6-sol xhigh，B 仍为 Claude Opus 5 high。两者必须在独立新上下文中完成，不能看另一方文件、`PRIVATE_*`、checker、参考答案或历史标签。

## 4. 标签与门槛完全不放松

- usable Key 恰好一个：错误链取逐单元验算发现的最早致命错误；没有错误则取首次完成候选答案的最后非包装步骤。
- Complete 仍是候选实际链的 canonical backward slice；拆开的代入式和求值结果都保留，自包含计算不重复收结果，题面复述、计划、未用旁枝、重复与包装排除。
- Key 仅在双方 usable、非 low、singleton exact 时可供未来训练。
- Complete 的交集为正、并集外为负、对称差 mask；attention 始终在完整轨迹归一化。

raw 门与 v10 相同：Key/Complete F1 各 ≥`.90`，exact Key、Complete 共识与 paired support 各 ≥50，Complete unit agreement ≥`.90`、ambiguous ≤`.10`、positive IoU ≥`.80`、coverage ≥`.90`，A/B 控制各 `8/8`，self-repeat 各 ≥`.95`，并保留反全集退化门。60 条 natural 全部进入冻结分母，不裁决、不用第三模型修复。

## 5. 判定

1. 全门通过：只允许另行准备独立 scale 协议。
2. 仅预注册数量/yield 门失败，而语义、控制、repeat 和反退化门全过：允许另冻前瞻性 oversampled strict-consensus scale 协议。
3. 任一语义、控制或 repeat 门失败：停止 Prior 扩量；不能改本轮标签、降低门槛或挑子集。

v11 smoke 自身永远不是训练数据。即使通过，也只说明两个 AI 能按这套定义稳定操作；标签仍只能称 `silver_dual_ai_verified_canonical_prior_v11`，不能称 Gold、人工验证、事实准确或模型效果证据。

## 6. 入口

```bash
python prepare_clir_prior_v11.py prepare-smoke
python prepare_clir_prior_v11.py verify-smoke
python prepare_clir_prior_v11.py evaluate-labels
```

准备与复算阶段只用 CPU。包验证通过后，分别把 `launch_prompt_a.txt` 和 `launch_prompt_b.txt` 发给两位模型；绝不能发送私有索引。
