# CLIR Consistency 提示词修复 v4

冻结日期：2026-08-25 UTC

当前状态：`READY_FOR_DEVELOPMENT_REPLAY`

机器契约：[`../configs/data_expansion_smoke_v4/protocol.json`](../configs/data_expansion_smoke_v4/protocol.json)

机器契约 SHA-256：`64157eb895a498865234e80b07b886b58ae837c1bfaea55d646fff7515ce5f82`

唯一提示词：[`../configs/data_expansion_smoke_v4/consistency_prompt.md`](../configs/data_expansion_smoke_v4/consistency_prompt.md)

提示词 SHA-256：`d5ef2f9d021168014dbd36306b9345ec2da816ea5fd3d32ec15bde1ca413317c`

14-ID 清单 SHA-256：`ff969e3407cea18918b63b403b3bcf24bfa8b4b62dcaa533b60f73a36f2fe136`

## 1. 为什么单独开 v4

smoke-v3 已按冻结规则终止，不能修改提示词后重标同一批数据并宣称 v3 通过。但它提供了清晰的开发诊断：
v2/v3 的 40 个 C pair 中有 36 个完全相同；B 对这 36 条的判断没有变化，A 却把 9 条从 accept 改成
reject，其中 8 条理由是“近乎照抄”。因此当前问题主要是同数学骨架与 near-copy 边界混在一个主观判断
里，不是新题源或 Consistency 模块本身突然失效。

v4 不加入相似度算法，也不改变模型、数据 schema 或训练 loss。它只用一份更明确的两关提示词：先判断
数学路径，再判断表达差异，并明确“相同公式、数字和必然顺序不能单独作为照抄证据”。

## 2. 阶段一：14 条开发回放

14 条输入就是 v3 的自然分歧项，只用于调试提示词。A/B 使用不同模型系列、两个全新上下文、完全相同的
提示词；两边不得看历史标签、raw report 或对方输出。

预注册通过门：

- 两边都恰好输出 14 个冻结 item_id，schema 与 rationale 前缀全部合法；
- decision agreement 至少 13/14；
- 每位 annotator 的 review 不超过 1；
- 每位 annotator 至少有 2 个 accept 和 2 个 reject，防止“全接收/全拒绝”伪一致；
- 不调用第三模型，也不允许裁决救活失败。

即使全部通过，这 14 条也不进入训练、不能称为新的可靠性证据；它们已经参与提示词设计。

## 3. 阶段二：30 条全新确认

只有阶段一通过才制作 30 个 query-distinct 新 pair。它们不得复用任何 v2/v3 C item_id 或 query_id，且在
A/B 标注前冻结 manifest 与提示词 hash。

预注册最低门是 27/30 decision agreement；每边 review 不超过 2，并至少产生 3 个 accept 和3 个 reject。
raw gate 失败仍不能由第三模型救活。通过只授权另发正式 Consistency 扩量协议，不直接证明 C 模块提高
Best-of-N，也不自动解锁 H/P 或 full 训练。

## 4. 当前唯一操作

分别把 `launch_prompt_a.txt` 和 `launch_prompt_b.txt` 的完整内容交给 GPT-5.5-sol/xhigh 与 Claude Opus 5/high 的
两个全新上下文。两者都直接读取同一自然 pair 文件与同一 14-ID 清单，分别把结果写入 machine protocol
规定的本地路径。标签回来后先做机械校验与冻结门判定，再决定是否生成 30 条新 pair。

两份标签回来后运行：

```bash
cd /prodcpfs/user/panzhixin/ICLR_2027
python prepare_clir_smoke.py consistency-prompt-check
```

输出只会是 `PASS_DEVELOPMENT_REPLAY` 或 `STOP_REPLAY_FAILURE`；前者也只允许进入 30 条全新确认。
