# CLIR 排序评测与 H0 扩量协议 v7

状态：32,000 条原始回复与 v7.1 一次性补采样均已完成并通过哈希校验，但 H0 proposal
仍有三个格子产率不足；v7.2 全新补充题池已通过只读容量审计，尚未生成，也尚未送给 AI
标注、抽特征或训练。机器可执行的权威契约是 `configs/ranking_expansion_v7/protocol.json`、
`configs/ranking_expansion_v7/yield_rescue_amendment_v7_1.json` 和
`configs/ranking_expansion_v7/supplement_protocol_v7_2.json`；本文只解释设计，不替代 JSON
契约。

## 1. 这一轮要回答什么

上一轮已经说明 Consistency 能学到“同一道题的简写和详写应该保持同一种解题关系”，但样本仍不足以判断三个模块放在一起为什么互相干扰。本轮先补两块基础设施：

1. 建一个更大、从未用于本项目训练和调参的排序评测池；
2. 扩充 Hallucination onset（第一次出现不受题目或前文支持的实质性说法）标签，只训练 H0 的定位损失。

H1 的负尾部 reward、Dual Prior 和完整三模块组合本轮都不启用。最终只做四个完全匹配的训练条件：C0、C1、H0、C1+H0。这样能先看清 Consistency 和 H0 单独是否有效、合用时有没有负交互。

## 2. 题目与隔离

- 只使用固定版本的 MATH train 与 GSM8K train；不读取或选择官方 test。
- MATH 使用 7 个数值答案子科目，限 level 3–5、官方解答不少于 45 个英文词、最终答案能由冻结 checker 解析为单一数值。
- GSM8K 只取长链题：参考推理不少于 45 个词，至少 2 个计算标记和 3 个不同中间数值。
- 历史排除清单与 Consistency-v6 已使用的 2,000 条题全部进入排除锚点；任何与它们同属精确、模板或近重复簇的题也排除。
- 每个模板簇最多选一题。排序评测与 H0 采集不仅 query 不重叠，模板簇也不重叠。

只读容量审计结果为 4,062 条可选题，足够冻结以下两个池：

| 用途 | MATH | GSM8K | 合计 | 每题候选 |
|---|---:|---:|---:|---:|
| 新排序评测 | 700 | 800 | 1,500 | 16 |
| H0 标签采集 | 600 | 400 | 1,000 | 8 |

排序池是“新鲜、query/模板簇隔离的验证集”，不是官方 test，也不能拿来反复调 v7 参数。

## 3. 生成与存储

- 生成器固定为精确版本的 `microsoft/Phi-3.5-mini-instruct`。
- 温度 1.0、top-p 0.9、最多 1,024 个新 token；每个 query 使用由 query ID 确定的独立 seed。
- 共 50 个可恢复分片，每片 50 道题：排序 30 片、H0 采集 20 片。
- 总计 32,000 条原始候选。每片写入后都核对候选编号、query 顺序、prompt token、模型版本、代码提交和文件哈希；不自动覆盖残缺文件。
- 禁止给所有原始候选抽完整 33 层特征。后续只抽 24,000 条排序候选、冻结后的 800 条 H0 proposal，以及每题一条 condition。

## 4. H0 候选和双 AI 标注

H0 不是“答案做错就算幻觉”。它问的是：推理中是否出现了第一条不能由题面或此前有效推理支持的实质性说法。例如算错一个中间数、凭空添加比例、把人物或单位换掉，都可能是 positive；单纯省略常规算术展开不算。

生成前已经按题源、checker 结果目标和 train/dev 划好 1,000 个 cell。生成后只做机械选择：在每个预分配 cell 中，按冻结哈希顺序取第一条满足 checker 状态、未截断、可切出至少 5 个 material units 的候选。不能看 AI 标签后换题。

计划冻结 800 条 proposal，其中先开放 80 条 train smoke：

- 40 条目标为 numeric mismatch：MATH 35、GSM8K 5；
- 40 条目标为 numeric match：MATH 15、GSM8K 25；
- 剩余 720 条在 smoke 通过前保持未打开。

两位标注者分别是用户报告的 GPT-5.6-sol xhigh 和 Claude Opus 5 high。二者互盲、都不是 Phi 系列，不使用第三模型裁决。标签统一称为 `silver_dual_ai_h_onset_v7_no_human_verification`，不能称 Gold 或人工验证。

smoke 门：两 AI 的 path 原始一致率至少 90%；共同判 positive 的样本中，首错 unit 精确一致率至少 75%，且分母不少于 30；每位标注者 8 个已知控制项全对；约 10% 重复项的自一致率至少 95%。失败则停在标注协议/产率诊断，不改题目清单来凑通过。

smoke 通过后，最终目标是：

| split | positive | clean |
|---|---:|---:|
| train | 200 | 200 |
| mechanism dev | 100 | 100 |

只接收两 AI exact agreement 且都不是 low confidence 的行，再按预冻结哈希顺序取数。共同不确定或分歧行不进入训练标签。

### 4.1 首次 FAIL-yield 与一次性补采样

每题最初 8 条回复做完机械筛选后，有 4 个预分配格子凑不满：MATH 的 numeric-match
train/dev 分别只有 51/100 和 29/50 道题存活；GSM8K 的 numeric-mismatch train/dev
分别只有 16/17 和 13/33 道题存活。这不是标注一致性失败，也不是 unitizer 故障，而是
部分题目的 8 次采样里没有出现目标 checker 结果。

v7.1 只允许一次补采样：固定上述短缺格子中全部 191 道零产出题，每题再生成 24 条，
候选编号固定为 8–31，共 4,584 条。题目、query split、模板簇、checker、unitizer、至少
5 个 material units 的门槛、800 条配额和哈希选择规则都不变；仍然是每题最多入选一条。
补采样前没有查看任何 AI 标签。若这一轮后仍凑不满，必须以 FAIL-yield 停止并另建新题池，
不能继续追采样或降低门槛。

v7.1 补采样按上述规则执行后仍有三个格子不足：MATH numeric-match train/dev 分别差
27/7 道，GSM8K numeric-mismatch dev 差 14 道。因此旧 191 道题正式停止采样，转入新的
v7.2 题池，不把这次结果当成协议或标签失败。

### 4.2 v7.2 全新补充题池

v7.2 在任何 AI 标签打开前冻结 180 道新题，每题从一开始固定生成 16 条，共 2,880 条：

- 80 道 MATH numeric-match train；
- 30 道 MATH numeric-match dev；
- 70 道 GSM8K numeric-mismatch dev。

所有历史题、v7 排序题和原 H0 题都作为模板簇排除锚点，新题与它们 query 和模板簇均不
重叠。GSM8K 仍用原来的长链门槛；MATH 为了补充较稀缺的数值正确路径，扩展到 level 2，
但仍要求官方解答至少 45 个英文词并可解析出单一数值。只允许这一批预冻结生成，不再按
结果追加采样；checker、unitizer、至少 5 个实质步骤、每题最多一条和最终 800 条格子配额
全部保持不变。机器契约见 `configs/ranking_expansion_v7/supplement_protocol_v7_2.json`。

### 4.3 reserve 首轮失败与 v7.3 一次性全量重标

80 条 smoke 的两 AI path 一致率为 96.25%，共同 positive 的首错 unit 精确一致率为
82.22%；两位标注者的 8 个控制项和 8 个盲重复项都全对，因此按原门槛开放 720 条
reserve。

reserve 首轮没有通过。A 在 720 条自然样本中把 527 条标成 `uncertain`，并在包括隐藏
控制项在内的 578 行重复同一句笼统兜底理由，控制项只有 1/8；B 的控制项为 8/8，但盲
重复为 68/72（94.44%），比冻结的 95% 门槛少一条。两者 path 一致率因此只有 25.97%。
该轮已按 `FAIL_H0_V7_RESERVE` 永久保留，未生成训练标签，也不降低任何门槛。

v7.3 只允许一次完整重标：同样的两位模型在新的独立会话中，对原封不动的同一批 800
个公开条目全部重做。为避免长文件末尾被占位符补齐，每份公开包仅按原顺序机械切成
16 个 50 行 shard；item ID、题目、trajectory、units、隐藏控制项和盲重复项均不改变。
禁止读取首轮标签、对方包或 private index，禁止只修失败行，也禁止混用两轮结果。若
无法完成后续 shard，必须让文件缺失，不能批量填 `uncertain`。第二轮仍使用原来的 path、
onset、控制项、盲重复和最终 200/200/100/100 门槛；再失败就停止，不允许第三次尝试。
冻结契约见 `configs/ranking_expansion_v7/reannotation_amendment_v7_3.json`。

### 4.4 v7.3 完整重标结果：终止于自一致性门

两位标注者的 16 个 shard 均完整落盘，每边 800 条；严格 schema、item ID、unit index、
原始包重建与哈希校验全部通过。A/B 分别给出 466/458 条 `hallucinated` 和 332/338 条
`clean`，每位标注者中相同理由的最大重复次数都是 1，没有再次出现首轮的批量兜底。

自然 reserve 的 path 一致率为 `698/720=96.94%`（kappa `0.9408`），共同判为 positive
的支持数为 403，首错 unit 精确一致率为 `310/403=76.92%`；两边 8 个隐藏控制项均全部
正确。这些指标都通过原门槛。但盲重复自一致性 A 为 `65/72=90.28%`，B 为
`64/72=88.89%`，都低于冻结的 95%。因此最终报告状态为 `FAIL_H0_V7_RESERVE`，最终
Silver 选择没有运行，feature extraction 与 training 都保持禁止。

失败并不只是 unit 边界差一格：A 的 7 个重复分歧中有 6 个是同判 hallucinated 但 onset
相差 3–15 个 unit，另 1 个发生 path 翻转；B 的 8 个分歧中有 5 个 onset 相差 3–43 个
unit，另 3 个发生 path 翻转。也就是说，下一版若重做 H0，需要让“第一条已无法挽回的坏
主张”有更机械的证据链或候选集合，而不是简单把 exact 门改成 ±1。

这次是修正案允许的最后一次完整重标。不得再发起第三次重标、只修不一致行、混用两轮
标签、改重复项分母或降低 95% 门槛。完整终止报告位于本地
`run_artifacts/ranking_expansion_v7/pre_annotation/final_v7_3/finalization_report.json`，
SHA-256 为 `932606832aedb8de1781f20404989fc4f587aa5b0d919f8ba9eb7d731fcd2c01`；
原始标签与报告不推送远端。

## 5. 后续训练与允许的结论

原先预注册的训练矩阵为 C0、C1、H0、C1+H0；seed 42/43/44；每个条件 3 epoch。H0
只用 onset BCE，H1 负尾部关闭，Dual Prior 关闭，不在看到结果后调 epoch、margin 或
loss 权重。由于 v7.3 H0 reserve 门失败，这个四格矩阵现在不能执行；若只做 C0/C1，
必须另写一份不使用本轮 H 标签的新授权，不能把它冒充原四格实验的完成版。

主排序指标是新鲜 1,500 题上的 Best-of-16；同时报告各 K、随机选择基线、Oracle、题源分层和 seed 区间。交互量固定为：

`(C1+H0) - C1 - H0 + C0`

如果四格实验完成，它原本最多能支持“在该冻结生成器、题源、Silver 标签定义和新鲜
验证池下，H0/Consistency 是否呈现可重复增益以及二者是否有负交互”。但本轮已在 H0
标注门终止，所以实际只支持数据流水线、自然标签互相一致性与自重复失败的诊断；不能
给出新的 H0、Consistency 或交互 Best-of-N 结论，也不支持对官方 test、其他生成器或
所有数学推理任务作泛化声明。
