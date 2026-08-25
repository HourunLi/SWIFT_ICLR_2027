# 可直接复制给外部 AI 的 CLIR 扩量协议审查提示词 v1

> 历史状态：`review_completed`。两份互盲审查均已返回并共同 block v1；不要再把本提示词当成当前协议，
> 更不能把它当作正式数据标注 prompt。当前裁决见
> [`../data_expansion_smoke_review_resolution_20260825.md`](../data_expansion_smoke_review_resolution_20260825.md)。

下面代码块内的内容应作为一个完整 prompt 一次性发送给外部 AI。它用于独立审查协议，不是让该 AI
直接进行数据标注。

```text
你现在是一个独立的机器学习研究协议审查员。请审查下面这套 CLIR 数据扩充 smoke 协议，重点找会导致数据泄漏、标签不可信、实验结论不成立或正式扩量浪费预算的问题。不要只复述方案，也不要因为方案写得完整就默认同意。

项目背景：
1. CLIR 是一个利用冻结语言模型 token hidden states 给候选推理过程打分、再做 Best-of-N 选择的 reward model。
2. 三个辅助模块分别是：
   - Consistency：同一道题、同一种推理，只是表达长短或风格不同，表示和最终分数应尽量稳定；
   - Hallucination onset：找出推理第一次出现明确错误或无依据主张的位置；
   - Key/Complete 双先验：Key 是最决定性的步骤，Complete 是足够审计整条推理的最小完整步骤集合。
3. 当前小数据实验只有 496 个 outcome queries、27 个 Consistency 正 pair、17 条正 onset +31 条 clean、48 条 Key/Complete。小样本点估计中 Consistency 和 onset BCE 各自可能有帮助，但区间都跨 0；两者组合出现负交互信号；negative tail 回退；prior 可学但没有稳定排序增益。
4. 当前 prior-to-reward gate 保留 main 原始共享梯度路径并默认开启，固定权重 .25。它只是开发集选出的工程默认值，不代表已经证明有效。本次 smoke 不允许继续调 gate、loss 权重或 epoch。
5. 没有人工复核；计划采用两个 AI 独立标注，对所有分歧做盲裁。所有最终标签只能叫 dual-AI Silver，不能叫 Gold。

冻结的 smoke 方案：
A. 数据源
- 只用 GSM8K 官方 train 和 ASDiv-A arithmetic subset 做训练侧 smoke；GSM8K 官方 test 不访问。
- SVAMP 完整保留为未来外部鲁棒性测试，不进入训练和本轮调参。
- ASDiv 是 CC BY-NC 4.0，本项目默认是非商业论文研究。
- 暂不加入 MATH/AQuA，避免同时改变答案形态与 checker。

B. 最终合格数量
- 50 个 outcome queries：30 GSM8K +20 ASDiv-A；每题 8 条 Phi rollout，共 400 trajectories。
- 30 个 Consistency groups：18 GSM8K +12 ASDiv-A；每组两个同 query 的 Phi native candidates，一个相对 compact、一个相对 expanded；30 组来自 30 个不同 query。
- 20 条正 onset +20 条 explicit clean：每类 12 GSM8K +8 ASDiv-A；共 40 条来自 40 个不同 query。
- 同一批 40 条分别、盲地再标 Key/Complete；H 标注者与 prior 标注者互相看不到对方标签。
- 这些是最终合格数。最多先提议 40 个 Consistency pair、60 条 H/P candidate；若 400 条 rollout 内无法按固定标准凑足，不降低标准，smoke 直接判未通过。

C. query、split 与去重
- 在生成前按 original query 切分；同一 query 的全部候选、视图、机制标签永远在同一 split。
- query_id 带数据源命名空间；query_id 只用于候选池和 split，semantic_id 只用于 Consistency。
- 排除旧 outcome train、旧 ranking population、旧 mechanism queries、官方 test 和 protected SVAMP。
- 规范化文本完全重复的题硬删除；近重复或只换数字/实体的模板题由双 AI 判断，任一认为同底题就保守删除后出现者。
- 合格题按 sha256("clir-smoke-v1|" + query_id) 排序取样，不按模型表现挑题。

D. rollout
- microsoft/Phi-3.5-mini-instruct，model/tokenizer revision 固定为 2fe192450127e6a83f7441aef6e3ca586c338b77。
- vLLM；8 candidates；temperature=1.0；top_p=.9；max_new_tokens=1024；max_model_length=4096；seed=42。
- prompt 要求逐步解题并用 boxed answer。
- 必须保存原始 prompt_token_ids/output_token_ids；它们是 token 位置的唯一真相，禁止从文本重新 tokenize。

E. correctness checker
- 新建并冻结 clir_numeric_multisource_v1；GSM8K 复用现有 numeric-v5 语义；ASDiv 从 Answer 解析单个有限数值。
- ASDiv 单位作为 provenance 保存，本轮 numeric correctness 不强制候选输出单位。
- AI 不投票覆盖 checker；如果 checker 规则错，修 checker、升协议版本并全量重标。

F. Consistency 定义
- pair 的 numeric correctness 和规范化最终答案相同，token 长度比至少 1.25，而且有真实表达/组织差异。
- 必须保持同一核心前提、同一解题方法、关键中间量和最终结论。
- 如果原推理有错误，两条必须保持同一个错误机制、语义位置和下游影响。
- 换独立解法、修复旧错、引入新错或近乎复制都 reject。
- 标注 AI 看不到 checker、reference answer、长度选择理由和另一标注者输出。

G. H 与 Key/Complete exact-token 标注
- 程序先把 trajectory 切成固定 material-claim units；每个 unit 绑定保存的 output token 区间 [token_start, token_end)。
- AI 只能选 unit index，不能自己重切文本。unitization 无法表达必要 claim 时，样本 ineligible。
- H 的 clean 表示所有实质主张都有支持；hallucinated 表示存在明确错误或缺前提；onset 是第一条明确坏 claim。正确最终答案不保证 clean，错误最终答案不自动给 onset。
- positive onset materialize 为所选 unit 的 token_start；explicit clean 用 hallucination_onset=-1；未知必须字段缺失，不能用 -1。
- Complete 是可重建和审计实际推理链的最小非冗余 unit 集；Key 是 Complete 中最决定结论是否成立的最小子集；必须 key⊆complete。

H. 双 AI 与裁决
- A/B 独立调用，优先不同模型系列；固定 model revision、prompt hash、temperature、seed 和原始响应。
- 二者互盲，并且看不到 checker、reference、历史结果或预期通过率。
- 完全一致且 confidence 不是 low 的行可以直接接收；所有其他行进入匿名顺序的盲裁。
- 裁决只处理分歧；最好用第三模型。如果只能复用较强的 A/B 模型，必须新建 clean context 并如实记录。
- 裁决仍 uncertain/low 的样本丢弃，不硬凑。所有标签命名 silver_dual_ai_v1。

I. 预注册通过门
- 数据、split、candidate 顺序、reference parse、exact-token/unit span 契约必须 100% 通过。
- Consistency：A/B decision agreement≥90%，需裁决≤25%，最后得到30组。
- H：path agreement≥80%，A/B 都判 positive 时 exact onset-unit agreement≥60%，需裁决≤50%，最后得到20 positive+20 clean。
- Prior：eligibility agreement≥95%；Key macro unit-set F1≥.60；Complete F1≥.80；key⊆complete=100%；需裁决≤75%。
- 先报告 A/B 原始一致性，再报告裁决结果，不能用裁决后的统一率冒充原始一致率。
- 任一门失败都不能正式扩量；必须修 unitization/指南/模型组合并发布新协议版本。

J. 证据边界
- smoke 通过仅证明多题源生成、checker、双 AI 标注和 exact-token materialization 流水线可用，不证明三个模块有效。
- 文本、checker、unitization、标注门全部通过后才抽 33×3072 BF16 hidden states，避免提前浪费存储。
- smoke 通过后，正式规模和 GSM8K/ASDiv 比例仍根据各题源错误率、标注 yield、裁决成本重新冻结。
- 下一轮效果实验仍完整重跑 C0/C1/H0/CH0；negative tail、MIL、pseudo-onset tail 不进核心矩阵；prior 在新数据上做 direct 学习性与固定 .25 gate off/on。

请按下面格式回答：
1. 总裁决：approve / approve_with_changes / block，给一句核心理由。
2. Blocking issues：只列如果不改就不能执行的问题；没有就明确写“无”。
3. 分别审查：数据源选择、跨数据集重复、split 泄漏、checker 语义、Consistency 构造、H onset 定义、Key/Complete 定义、双 AI 独立性与裁决。
4. 检查数量和约束是否自洽，尤其是能否从 50×8 rollout 中得到所需的 30 groups 和 20+20 H/P，同时保持 distinct-query 约束。
5. 审查这些一致性/F1/裁决率门槛是否过松或不现实；如果要改，给出具体数值及原因。
6. 明确说明：没有人工复核时，这批数据最多支持哪些表述、绝不能支持哪些表述。
7. 给出最小修改清单，区分“必须改”和“可优化”。不要泛泛建议“增加更多数据”或“请专家复核”，因为本轮已明确没有人工复核。

请用中文和大白话，但技术判断要具体。不要直接替我们执行标注，也不要虚构你没有看到的代码、数据或实验结果。
```
