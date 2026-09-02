# CLIR Prior v14：审核局部主链，Complete 仍由程序回溯生成

你只标当前公开 JSONL。机器把原始推理切成 `block`，并给每个 child 提供最多 6 个可能的直接父依赖。机器现在会尽量保留数值、分数和变量的来源，但所有角色、分数、强弱和候选边都只是提示，不是真相。

禁止查看 `PRIVATE_*`、代码、测试、协议、checker、参考答案、历史标签、另一位标注者的包或输出。不要输出 Complete；程序会从你保留的依赖边向前回溯生成它。

## 1. 三遍完成一行

第一遍：从头验算算术、代数、单位、对象和所求量，并判断能否审核。

- 有可审计推理：`eligibility="usable"`。
- 只有答案或拒答：`eligibility="no_auditable_reasoning"`。
- 即使读完 block 仍无法辨认实际主线：`eligibility="insufficient_structure"`。

第二遍：先定每个 block 的角色和最终主线 block，再从最终 block 倒着问：“这个 child 真正用了哪些前面新算出的数值、关系或选择？”据此逐条审核候选边。

第三遍：检查输出契约，尤其是每个 block/候选边是否都恰好出现一次，以及不可用行是否彻底清空结构字段。

## 2. Block 角色

必须按 `block_id` 顺序给全：

- `main_step`：候选实际用来推出其答案的主线步骤；算错也仍可属于主线。
- `premise_restatement`：只复述题面，后续算式已经直接写出需要的题面数字或关系。
- `plan_or_heading`：计划、标题或“接下来计算……”，没有产生新量或关系。
- `formula_only`：没有代入本题数值/对象的通用公式。
- `duplicate`：重复前面已经完整表达的同一结果。
- `answer_wrapper`：只说答案或只包一层 `boxed`，没有新计算/选择。
- `unused_branch`：算了一个量，但最终答案没有使用它。

`role_hint` 以及候选边里的 `strength/score/mandatory/evidence` 都不具约束力。

## 3. 最终 block 和直接依赖边

`final_block_id` 是最后一个真正完成候选所声称答案的 `main_step`，不能选纯答案包装。

对 `structure.candidate_edges` 每条边按输入顺序输出：

- `keep`：child 的实际推导直接使用了 parent 新产生的数值、关系、约束或选择。
- `drop`：只是相邻、同主题、题面复述、计划、通用公式、重复、验算或未使用旁枝。
- `uncertain`：只有语义确实无法稳定判断时使用。

不要因为机器写了 `mandatory` 就自动 keep。不要补传递边：若已有 `A→B`、`B→C`，通常不再补 `A→C`。先检查全部候选；只有真正缺失的直接主线依赖才写入 `missing_edges`，最多两条、升序且不与候选边重复。keep 和 missing 的两端都必须是 `main_step`。

## 4. Path 和 Key

程序从 `final_block_id` 沿 `keep + missing_edges` 回溯得到该标注者的 Complete 主链。

- 主链有致命错误：`path_status="flawed"`；Key 选主链里最早的致命错误原始 `unit_index`。
- 主链没有致命错误：`path_status="supported"`；Key 选 final block 内第一次真正完成候选答案的原始 `unit_index`。

Key 必须恰好一个，并位于回溯主链内。`rationale` 用一句话写明具体错误，或写明哪个步骤完成答案。

## 5. 严格输出格式

每个输入行输出一个单行 JSON；`item_id` 原样复制；不得输出 Markdown、额外说明或 Complete。

可用行示例：

```json
{"item_id":"原样复制","eligibility":"usable","path_status":"supported","block_roles":[{"block_id":0,"role":"main_step"},{"block_id":1,"role":"answer_wrapper"}],"final_block_id":0,"edge_decisions":[],"missing_edges":[],"key_unit_index":0,"confidence":"high","rationale":"block 0 完成答案，block 1 只包装答案"}
```

不可用行必须使用下面这种“完全清空”形式，不能保留任何角色或边：

```json
{"item_id":"原样复制","eligibility":"no_auditable_reasoning","path_status":null,"block_roles":[],"final_block_id":null,"edge_decisions":[],"missing_edges":[],"key_unit_index":null,"confidence":"high","rationale":"只有答案，没有可审核推理"}
```

字段取值：

- `eligibility`: `usable|no_auditable_reasoning|insufficient_structure`
- `path_status`: `supported|flawed|null`
- `block_roles`: usable 时每个 block 恰好一次且按序；不可用时为空
- `final_block_id`: 整数或 `null`
- `edge_decisions`: usable 时每条候选边恰好一次且按输入顺序；对象字段只能是 `parent_block_id,child_block_id,decision`
- `missing_edges`: 最多两个 `[parent,child]`
- `key_unit_index`: 一个原始 unit 整数或 `null`
- `confidence`: `high|medium|low`
- `rationale`: 非空字符串

普通难题不要一律标 low；只有按上述三遍仍无法稳定判断时才用 low。
