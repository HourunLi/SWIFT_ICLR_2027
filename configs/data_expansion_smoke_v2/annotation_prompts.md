# CLIR smoke v2 标注提示词（A/B/第三模型）

这份文件是给执行者复制的提示词，不是标注结果。A 与 B 必须来自两个不同模型系列，分别收到各自
`annotator_a/`、`annotator_b/` 目录；两边不能看对方输出、checker、参考答案、题目来源配额或通过门。
同一个模型重复调用不能冒充 A/B。H 与 Prior 必须开两个全新的独立上下文，不能在同一对话连续标。

## 当前 27 对去重：一键复制版本

把下面整段和 `run_artifacts/data_expansion_smoke_v2/dedup/candidates.jsonl` 一起发给 A；再在另一个模型系列
的全新对话中原样发给 B。不要把第一份回答放进第二个对话。

```text
你是 CLIR smoke v2 的独立题目去重标注者。我会附上一份 JSONL，每行是一对原始数学题。只根据题目文本
判断，不搜索答案、不解题、不调用外部 checker，也不要猜另一位标注者会怎么答。

如果两题只是换了数字、人物/物品名称或轻微措辞，而核心故事模板、已知量关系和所求量相同，decision 写
duplicate；如果运算关系、条件结构或所求量实质不同，写 distinct；确实无法稳定判断才写 uncertain。

逐行返回 JSONL，一条输入对应一条输出；pair_id 原样保留，不要 Markdown 代码围栏，不要额外解释，不要
漏行、加行、合并或改顺序。confidence 只能是 high、medium、low，rationale 用一句话说明核心关系为何
相同或不同。输出 schema 必须是：
{"pair_id":"原ID","decision":"duplicate|distinct|uncertain","confidence":"high|medium|low","rationale":"一句理由"}
```

## 通用前缀（每次都复制）

```text
你是 CLIR smoke v2 的独立 Silver 数据标注者。只根据我附上的 JSONL item 做判断，不搜索答案，不调用
外部 checker，不猜另一位标注者会怎么答。逐行返回 JSONL：一条输入对应一条输出，item_id 原样保留；
不要 Markdown 代码围栏，不要额外解释，不要漏行、加行或改顺序。confidence 只能是 high、medium、low，
rationale 用一句简短理由说明你依据了哪条可见推理。若某项确实无法判断，必须按任务允许的 uncertain/
insufficient 状态输出，不能为了凑一致率硬猜。
```

## 去重候选（rollout 前单独执行）

A 与 B 在两个完全独立的上下文里收到同一份 `candidates.jsonl`；不要把 A 的输出转发给 B，也不要把 B
的输出转发给 A。二者都必须返回 manifest 中的全部行，不能只回“看起来像重复”的子集。

输入是两个原始题目。若它们只是换了数字、人物/物品名称或轻微措辞，核心故事模板、已知量关系和所求量
相同，则判 `duplicate`；若运算关系、条件结构或所求量实质不同，则判 `distinct`。无法稳定判断才用
`uncertain`。不要解题，也不要参考答案。

每行只输出：

```json
{"pair_id":"原ID","decision":"duplicate|distinct|uncertain","confidence":"high|medium|low","rationale":"一句理由"}
```

A/B 完成后由脚本生成只含未解决 pair、完全不含 A/B 答案的 `third_independent_items.jsonl`。第三模型仍按
同一标准独立判断，但每行额外确认自己没有先看主标注者答案：

```json
{"pair_id":"原ID","decision":"duplicate|distinct","confidence":"high|medium|low","rationale":"一句理由","independent_answer_completed":true}
```

## Consistency（单独的新上下文）

把“同一种推理换一种展开方式”判为 `accept`。必须同时满足：同一道题、相同最终数值、相同解题方法、
相同关键中间量；如果两条都错，还必须是同一种错误机制。长短、复述、验算和组织顺序可以不同。

以下情况判 `reject`：修正或新引入错误；换了一种解法；关键中间量不同；只碰巧得到同一答案；两条几乎
逐字复制。信息不足才用 `review`。

每行只输出：

```json
{"item_id":"原ID","decision":"accept|reject|review","confidence":"high|medium|low","rationale":"一句理由"}
```

## Hallucination / first-bad-unit（单独的新上下文）

逐个看 `material_claim`。坏 claim 包括：与题面冲突的事实、题面没有支持的新事实/必要前提、错误算式或
推导、错误实体、明确写错的单位。没有把普通算术小步骤全部展开，不算“缺少前提”。最终数字对不代表
整条链 clean；最终数字错也不自动说明能定位到首错。

- 所有 material claim 都可由题面和前面正确步骤支持：`clean`，index 为 `null`。
- 能定位：`hallucinated`，选择最早的坏 unit；同 unit 内不再细分 token。
- 一个 unit 混了两个无法分别判断的 claim：`insufficient_unitization`。
- 没有可审查推理：`no_auditable_reasoning`。
- 证据不足：`uncertain`。

每行只输出：

```json
{"item_id":"原ID","status":"hallucinated|clean|uncertain|insufficient_unitization|no_auditable_reasoning","first_bad_unit_index":整数或null,"confidence":"high|medium|low","rationale":"一句理由"}
```

## Key / Complete Prior（与 H 完全分开的新上下文）

这里判断的是“这条候选自己声称的结论依赖哪些步骤”，即使候选结论是错的也照样标。

- `Complete`：删掉其中任一 unit 后，只凭剩余 unit 就不能核验候选所声称的结论；在满足这一条件的集合
  中先取元素最少者，多解时取排序后 index 数组字典序最小者。
- `Key`：`Complete` 内对结论最关键、最直接决定成立与否的最小子集；同样先最少、再字典序最小。
- 必须 `Key ⊆ Complete`，数组升序、无重复，只能引用 material claim。
- 只有答案、拒答、或没有可审查推理时用 `no_auditable_reasoning`；unit 粘连到无法按规则选集合时用
  `insufficient_unitization`。这两种情况下两个数组都为空。

每行只输出：

```json
{"item_id":"原ID","eligibility":"usable|insufficient_unitization|no_auditable_reasoning","key_unit_indices":[整数],"complete_unit_indices":[整数],"confidence":"high|medium|low","rationale":"一句理由"}
```

## 第三模型第一阶段：独立回答

第三模型先收到 `third_independent/<task>.jsonl`。它不知道哪些是 A/B 分歧、哪些是 15% 自动一致抽检，
也看不到 A/B 输出。按上面对应任务的原始提示词返回 JSONL。完成并保存后，才允许进入下一阶段。

## 第三模型第二阶段：匿名裁决

此时输入来自 `adjudicator/<task>.jsonl`，内含第三模型刚才的 `independent_annotation`，以及随机顺序的
`option_1`、`option_2`。先核对自己的独立答案，再选择一个匿名方案；两边都不合适时可 synthesize 一个
符合原任务 schema 的 annotation；仍无法确定则 unresolved。不要猜哪个 option 来自哪个模型。

每行只输出：

```json
{"item_id":"原ID","resolution":"adopt_option_1|adopt_option_2|synthesize|unresolved","annotation":null或完整任务annotation,"independent_answer_completed":true,"independent_annotation_sha256":"输入中原样复制","confidence":"high|medium|low","rationale":"一句理由"}
```

格式错误最多只做两次“只修 JSON 格式”的重试；不能在重试时透露答案或另一模型身份。所有原始尝试、
模型 ID、模型系列、revision/date alias、temperature=0、prompt hash 和调用时间都要留在本地产物中。
