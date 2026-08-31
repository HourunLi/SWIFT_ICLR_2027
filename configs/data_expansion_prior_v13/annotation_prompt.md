# CLIR Prior v13：只审核局部主链，Complete 由程序生成

你只标当前公开 JSONL。机器已经把原始推理切成较大的 `block`，并为每个 block 给了角色提示、为每一步最多提议两条父依赖。提示只是候选，不是真相；你要结合题目和回复重新验算并纠正。

禁止查看 `PRIVATE_*`、代码、测试、协议、checker、参考答案、历史标签、另一位标注者的包或输出。不要直接输出 Complete；程序会从你保留的依赖边自动向前回溯，生成 Complete。

## 1. 先验算，再判能否审核

从前到后重算回复中的算术、代数、单位、对象和所求量。

- 有可审计推理：`eligibility="usable"`。
- 只有答案/拒答：`eligibility="no_auditable_reasoning"`。
- 即使看过 block 仍无法分清实际主线：`eligibility="insufficient_structure"`。
- 不可用时，`path_status/final_block_id/key_unit_index` 为 `null`，三个数组都为空。

## 2. 给每个 block 定一个角色

必须按 block_id 顺序给全：

- `main_step`：候选实际用来推出其答案的主线步骤；步骤算错也仍可属于主线。
- `premise_restatement`：只复述题面；后续算式已经直接写出所需数字/关系。
- `plan_or_heading`：计划、标题或“下面计算……”，没有真正产生量或关系。
- `formula_only`：未代入本题数值/对象的通用公式。
- `duplicate`：重复前面已经完整表达的同一结果。
- `answer_wrapper`：只说“所以答案是……”或只包一层 `boxed`，没有新计算/选择。
- `unused_branch`：算了一个量，但最终答案实际没用它。

机器的 `role_hint` 只是提醒。若提示错了，以你的判断为准。

## 3. 找最终主线 block

`final_block_id` 是最后一个真正完成候选所声称答案的 `main_step`：可能是最后一次计算，也可能是“正数/最大值/满足条件者”的选择。不要选纯 `answer_wrapper`。

## 4. 审核局部依赖边

对 `structure.candidate_edges` 中每一条边按原顺序输出决定：

- `keep`：child 的实际推导确实使用了 parent 新产生的数值、关系、约束或分支选择。
- `drop`：只是相邻、同主题、题面复述、通用公式、重复、验算或未使用旁枝，并非实际依赖。
- `uncertain`：只有在语义确实无法稳定确定时使用，不能用来省事。

若机器漏掉了主线必需的直接依赖，可在 `missing_edges` 增加 `[parent_block_id, child_block_id]`，按升序、最多两条。不要补传递边：已有 `A→B`、`B→C` 时，不必再补 `A→C`。保留边和补充边的两端都必须是你标出的 `main_step`。

## 5. 判 path 和 Key

程序会从 `final_block_id` 沿 `keep + missing_edges` 回溯，得到候选实际 Complete 主链。

- 主链有致命错误：`path_status="flawed"`，`key_unit_index` 选主链里最早的致命错误原始 unit。后面只是沿用早错时不能选后果。
- 主链没有致命错误：`path_status="supported"`，Key 选 final block 内第一次真正完成候选答案的原始 unit。
- Key 必须恰好一个原始 `unit_index`，并且必须位于程序回溯出的主链内。

`rationale` 用一句话说明最早错误的具体算术/语义，或说明哪个步骤完成答案。

## 6. 输出格式

每个输入行输出一个单行 JSON；item_id 原样复制；不得输出 Markdown、额外字段或 Complete：

```json
{"item_id":"原样复制","eligibility":"usable","path_status":"supported","block_roles":[{"block_id":0,"role":"main_step"},{"block_id":1,"role":"answer_wrapper"}],"final_block_id":0,"edge_decisions":[],"missing_edges":[],"key_unit_index":0,"confidence":"high","rationale":"block 0 完成了答案，block 1 只是包装"}
```

字段与取值必须严格如下：

- `eligibility`: `usable|no_auditable_reasoning|insufficient_structure`
- `path_status`: `supported|flawed|null`
- `block_roles`: 每个 block 恰好一次且按序
- `final_block_id`: 整数或 `null`
- `edge_decisions`: 每条候选边恰好一次且按输入顺序；对象字段为 `parent_block_id,child_block_id,decision`
- `missing_edges`: 最多两个 `[parent,child]`
- `key_unit_index`: 一个原始 unit 整数或 `null`
- `confidence`: `high|medium|low`
- `rationale`: 非空字符串

普通难题不要一律标 low；只有按规则仍确实无法稳定判断时才用 low。
