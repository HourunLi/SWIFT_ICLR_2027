# CLIR smoke v3 双 AI 标注提示词

这份文件只规定标注语义。实际发送时，每个 AI、每个任务都使用一个全新对话，共六个隔离上下文：
`A/C`、`A/H`、`A/Prior`、`B/C`、`B/H`、`B/Prior`。提示词可以让标注 AI 直接读取对应 JSONL，
但不得给它 `PRIVATE_*` 文件、checker 结果、参考答案、另一位 AI 的输出、候选 strata 或通过门。

## 六个任务共用的开头

> 请读取我指定的 JSONL 文件，逐行完成全部 item。只输出 JSONL：每个输入 item 恰好一行 JSON，顺序可变，
> 但不得漏项、重复或输出 Markdown 代码围栏。你只能根据 `problem`、候选推理文本和已经冻结的 `units`
> 判断；不要自行重切 unit，也不要查询参考答案。`confidence` 只能是 `high`、`medium` 或 `low`；
> `rationale` 用一两句大白话说明理由。若格式校验失败，我最多只会让你修两次格式，不会告诉你正确标签。

实际六次发送分别把文件路径换为：

- `.../blind_packages/annotator_a/consistency.jsonl`
- `.../blind_packages/annotator_a/hallucination.jsonl`
- `.../blind_packages/annotator_a/prior.jsonl`
- `.../blind_packages/annotator_b/consistency.jsonl`
- `.../blind_packages/annotator_b/hallucination.jsonl`
- `.../blind_packages/annotator_b/prior.jsonl`

## Consistency（C）

在共用开头后追加：

> 判断 `left` 与 `right` 是否是同一道题、同一种实质解法和同一组关键中间量，只在措辞、拆步、解释详细度
> 或组织方式上不同。只有同时满足这些条件才 `accept`。以下都必须 `reject`：换了解法；修掉或新增错误；
> 关键中间量、最终结论或错误机制不同；只是碰巧答案相同；近乎逐字复制；只靠无意义尾话拉长。
> 两条都错也可以 accept，但必须从同一个最早位置以同一种方式错下去。
> 只有可见信息确实不足、无法稳定二选一时才用 `review`，不要为了提高一致率硬猜。
>
> 每行输出：
> `{"item_id":"...","decision":"accept|reject|review","confidence":"high|medium|low","rationale":"..."}`

## Hallucination onset（H）

在共用开头后追加：

> 这里找的是“第一个明确坏掉的实质推理 unit”，不是猜第一个坏 token。
>
> - `clean`：每个 material claim 都能由题面、前面已经成立的推导或普通数学规则支持。
> - `hallucinated`：至少一个 material claim 可明确反驳，或者凭空引入会影响结论的新事实/假设。
> - 省略常规算术展开不等于缺前提；但明确写错数字、单位、对象、等式或推理关系算坏 claim。
> - 多处错误取最早的 material unit；后面自己改正也不抹掉第一次错误。
> - 如果最早坏点被一个 unit 与另一条独立 claim 粘在一起，输出 `insufficient_unitization`。
> - 只有答案、拒答或没有可审计推理时，不要硬判 clean，输出 `no_auditable_reasoning`；无法稳定判断时
>   输出 `uncertain`。
>
> `clean` 时 `first_bad_unit_index` 必须是 `null`；`hallucinated` 时必须引用一个 material unit。
> 每行输出：
> `{"item_id":"...","status":"hallucinated|clean|uncertain|insufficient_unitization|no_auditable_reasoning","first_bad_unit_index":3或null,"confidence":"high|medium|low","rationale":"..."}`

## Key / Complete Prior（P）

在共用开头后追加：

> 题面始终在场，不需要把原样复述题面的 unit 选进证据集合。先判断这条候选是否有可审计的推理；若只有
> 答案、拒答或 unit 切分不足，标成相应 ineligible，两个数组都留空。
>
> `Complete` 不是“聪明人重新算这题时最短能写几步”，而是候选**实际走过的依赖链**中，所有唯一、
> 不重复、确实被后一步使用的中间变换或中间结果。保留候选先算 A、再用 A 算 B 的两步，即使你可以把它们
> 合并成一个式子。排除：原样复述题面、计划话术、没有参与后续计算的旁枝、重复等式和重复 final wrapper。
>
> `Key` 必须是 `Complete` 的子集：对正确链，选最直接决定最终答案的最小步骤；对错误链，选最早或因果上
> 最决定性的错误/无依据步骤。若多个集合等价，先选元素更少的，再选排序后 unit-index 数组字典序更小的。
>
> 例子：候选先算 `2+3=5`，再算 `5-1=4`，最后只重复“答案是 4”。那么 Complete 是前两条计算，
> Key 通常是直接得到 4 的第二条；不能因为 `2+3-1=4` 可以一步重算，就只把第二条当 Complete。
>
> 每行输出：
> `{"item_id":"...","eligibility":"usable|insufficient_unitization|no_auditable_reasoning","key_unit_indices":[3],"complete_unit_indices":[2,3],"confidence":"high|medium|low","rationale":"..."}`
>
> 数组必须升序、无重复，只能引用 material claim，并且 `key_unit_indices` 必须是
> `complete_unit_indices` 的子集。

## 第三模型

第三模型先读取 `third_independent/<task>.jsonl`，只按上面同一任务规则独立作答；此时看不到 A/B。
独立结果落盘并通过 schema 后，才能读取匿名 adjudicator 包。裁决时可以采用 option 1、采用 option 2、
综合成新的合法标签或 unresolved；不得猜匿名选项来自哪家模型。裁决不会改写或“救活”已经失败的 A/B
原始一致性门。
