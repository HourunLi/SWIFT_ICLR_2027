# CLIR Consistency prompt repair v4

状态：冻结前开发回放提示词。它只用于修复 Consistency 的标注边界，不改变已终止的 smoke-v3。

## A/B 共用的唯一判断提示词

下面正文必须原样发送给 A 和 B；两边只替换“输出路径”，其余内容、例子和顺序完全一致。

```text
你是 CLIR 的 Consistency 正样本审核员。请读取我指定的自然 pair JSONL 和 item_id 清单，只处理清单里的
14 个 item。目标是判断 left 和 right 是否适合作为“同一数学推理、不同表达方式”的 Consistency pair。

这不是比较谁写得更好，也不是单纯比较文字像不像。请严格按下面两关依次判断。

第一关：数学推理是否相同

只有同时满足以下条件，才能进入第二关：

1. 是同一道题，声称的最终答案相同；
2. 使用同一种实质解法；
3. 关键中间量及其依赖关系相同；
4. 代数上等价的写法、乘法交换顺序、同一步骤的合并或拆分，仍算相同；
5. 如果两条推理都错，必须是同一个最早错误、同一种错误机制；一条修正或新增了另一条没有的实质错误，
   必须 reject。

第一关不满足时直接 reject。

第二关：表达方式是否存在真实差异

请先忽略相同数字、数学公式、必然相同的因果顺序和题面专有名词，再判断是否“近乎照抄”。这些内容相同
是 Consistency pair 的正常特征，不能单独作为 reject 理由。

以下任意一种都可以算真实表达差异：

- 一条是简洁推导，另一条解释得更详细；
- 一条以公式为主，另一条以自然语言解释为主；
- 同一推理被重新分组、拆步或组织；
- 一条增加了不改变原推理的解释或验算；
- 简短版本对长版本进行了真正的概括改写，而不是只截取原句。

只有在下面情况才以“近乎照抄” reject：忽略公式、数字和格式后，两边的非数学文字仍基本逐句相同，
差异只剩步骤编号、换行、标点、删除少量词或替换少量同义词，而且没有真实的解释、组织或详细程度差异。

兜底规则：

- 数学路径相同，并且你能指出至少一个真实表达差异：accept。
- 数学路径相同，但只能指出编号、格式或少量同义词变化：reject。
- “约等于 106”和“等于 106”在 106 确实是精确结果且不影响后续推理时，不视为修正错误；真正改变
  精度、舍入方式或结论时才 reject。
- 只有可见信息确实不足时才 review；不要为了提高一致率硬猜。

例 1，accept：

left：“3×8=24，因此30−24=6。”
right：“三件商品每件8元，一共花24元。从30元中减去24元，还剩6元。”

数学骨架相同，但一个是紧凑公式，一个是文字解释。

例 2，reject，近乎照抄：

left：“Step 1: 3×8=24. Step 2: 30−24=6.”
right：“1. 3×8=24。2. 30−24=6。”

只有编号和格式不同。

例 3，reject，解法不同：

left：“先算3×8=24，再算30−24=6。”
right：“连续计算30−8−8−8=6。”

答案相同，但使用的中间量和推理组织不同。

例 4，错误机制：两边都把2+3错误算成6，只是一个用公式、一个用文字解释，可以 accept；如果一条写
2+3=6，另一条改成2+3=5，则必须 reject。

只允许读取：

/prodcpfs/user/panzhixin/ICLR_2027/run_artifacts/data_expansion_smoke_v3/proposals/annotation_consistency_natural.jsonl
/prodcpfs/user/panzhixin/ICLR_2027/configs/data_expansion_smoke_v4/consistency_prompt_replay_ids.txt
/prodcpfs/user/panzhixin/ICLR_2027/configs/data_expansion_smoke_v4/consistency_prompt.md

不得读取 v2/v3 的 labels_a、labels_b、triage、raw report、README、handoff、PRIVATE 文件或另一位标注者
输出。旧答案会污染这次提示词回放。

输出恰好 14 行 JSONL，每个清单 item_id 恰好一行，不得漏项、重复、增加其他 item 或输出 Markdown
代码围栏。confidence 只能是 high、medium、low。rationale 必须以与 decision 对应的前缀开头：

- accept：`[ACCEPT_STYLE]`
- 因数学路径、关键中间量、答案或错误机制不同而 reject：`[REJECT_REASONING]`
- 因近乎照抄而 reject：`[REJECT_COPY]`
- review：`[REVIEW]`

每行 schema：

{"item_id":"原ID","decision":"accept|reject|review","confidence":"high|medium|low","rationale":"[对应前缀] 一句话决定性理由"}

把结果写到我指定的输出路径。除了该输出 JSONL，不修改任何文件。完成后只报告输出路径和行数，不要在
聊天中复述 14 条标签。
```

## 给 A 的一键提示词

```text
直接复制并发送：
/prodcpfs/user/panzhixin/ICLR_2027/configs/data_expansion_smoke_v4/launch_prompt_a.txt
```

## 给 B 的一键提示词

```text
直接复制并发送：
/prodcpfs/user/panzhixin/ICLR_2027/configs/data_expansion_smoke_v4/launch_prompt_b.txt
```

## 解释边界

- 这 14 条已经用于分析失败原因，只能做 prompt development regression，不能作为新可靠性证据或训练数据。
- A/B 必须是两个不同模型系列并使用两个全新上下文；不能把 A 的输出给 B。
- 本轮不调用第三模型。回放通过只授权冻结提示词并制作 30 条新 pair，不授权训练。
