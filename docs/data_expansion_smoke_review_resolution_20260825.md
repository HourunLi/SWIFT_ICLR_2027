# CLIR 扩量 smoke v1 双审查裁决

日期：2026-08-25 UTC

裁决：`v1 block -> publish v2`

输入：两份互相不可见的外部 AI 协议审查

边界：审查者均未查看代码、数据或实验 artifact；本文把可验证事实与审查建议分开处理。

## 一句话结论

两份审查在真正关键的地方已经收敛：v1 不能直接跑。主要问题不是简单的“50 题一定不够”，而是
proposal 怎么选、哪些行进入一致率分母、H 与 prior 怎么联合凑齐、unit/token 怎么精确绑定都没有锁死。
这些问题会允许执行者在看过标签后无意中留下最容易一致的样本。

v2 已发布为：

- `docs/data_expansion_smoke_protocol_v2.md`；
- `configs/data_expansion_smoke_v2/protocol.json`。

v1 从未执行，现为 `superseded_before_execution`。

## 共同意见：全部采纳

1. query 池扩大为 100（60 GSM8K +40 ASDiv-A），但自然 C/H/P 标注上限仍为 40/60；
2. 全部 smoke query 永久 train-only，不得进入后续 mechanism dev、ranking validation 或 test；
3. C 与 H/P 可在 query 层重叠但必须报告；同一 trajectory 不可复用；
4. C 和 H/P proposal 的枚举、自动过滤、每 query 上限、来源/数值 strata、hash tie-break 和 manifest
   必须在 A/B 标注前冻结；
5. H 与 prior 对同一 60 条全部独立标注，再从 joint usable 交集按冻结顺序取 20 positive +20 clean；
6. 所有 agreement、F1、裁决率以自然 proposal manifest 为分母；格式失败或 low/uncertain 不能静默移除；
7. unitizer 必须冻结 output-token 半开区间、完整覆盖、终止 token、原子 claim 和回归测试；
8. 截断 row 保留审计但不进任何训练/机制标签；
9. H 标签准确命名为 first-bad-unit start token，不宣称精确首错 token；
10. Key/Complete 补齐错误链、多个最小集合 tie-break 和反全集退化统计；
11. A/B 不同模型系列从“优先”升级为硬条件，并禁止与 Phi generator/backbone 同系列；
12. 裁决率显著收紧；结构合法率与标注质量指标分开；所有最终标签只叫 dual-AI Silver；
13. 增加合成 hidden controls、A 自一致性和 15% auto-agree 第三模型抽审；
14. `FAIL_PIPELINE`、`FAIL_YIELD`、`FAIL_DIVERSITY` 分开，防止 yield 不足后改定义凑数。

## 经核验后修正采纳

### 1. SVAMP 与 ASDiv-A

审查指出两者并不独立，这一事实成立：SVAMP
[论文](https://aclanthology.org/2021.naacl-main.168/)写明从 ASDiv-A 选 100 个 seed，再人工生成变化题。
但“只要训练 ASDiv-A，SVAMP 就完全作废”说得过头；SVAMP 原本就被设计成在相关 seed/source exposure
下检查变化鲁棒性的 challenge set。

v2 的处理是：继续不用 SVAMP 训练或调参，但把它准确降格为 **ASDiv-derived protected contrast set**，
不再叫独立来源 holdout。以后若需要独立泛化结论，另选 holdout；若要提升 SVAMP 的独立性，先做密封的
seed-family 排除审计。

### 2. checker 和单位

现有历史 checker 的稳定目标是数值匹配，并没有完整单位 ontology。v2 不假装一夜之间解决完整语义
correctness，而是把公开目标改名为 `numeric_value_match`；训练兼容字段 `correctness` 必须注明同一语义。
显式错误单位/实体由 H 当作坏 claim 并单独诊断。报告不得再把 numeric match 写成完整语义正确。

### 3. unitizer 复用范围

`panzhixin` 已有 deterministic 行/句切分、fixed unit、exact-ID 映射、F1 汇总与 role-blind adjudication，
可以移植思路。但旧实现主要保证可见字符覆盖，尚未冻结完整 output-token partition、terminal control token、
小数/缩写回归和“每 unit 一个 material claim”。因此 v2 明确要求新版本，不能直接把旧文件复制后宣布通过。

### 4. 阈值

两位审查者对阈值并未完全一致：一位要求 bootstrap CI 下界与更低 F1 floor，另一位主张更高点阈值。
v2 采用中间但可执行的预注册门：

- C raw agreement `.90`、裁决 `.20`；
- H path `.85`、exact unit `.70`、±1 unit `.85`、裁决 `.35`；
- Key/Complete macro F1 `.65/.82`、prior 裁决 `.40`；
- 所有区间、分子和分母必须报告，但几十行 smoke 不用区间下界作为唯一硬门。

## 没有照单全收的建议

### 1. 不把 trap accuracy 叫自然数据准确率下界

合成假等式或唯一短链能检验模型有没有照指南做，但它们比自然错误简单，不能估计自然 Silver label 的真实
accuracy。v2 保留 controls，却把结论限定为 protocol compliance；15% auto-agree 抽审也只叫三模型稳定性，
不叫真实错误率。

### 2. 不强制裁决者站 A/B 各 35%–65%

裁决偏向一方可能来自锚定，也可能因为一方确实更好。没有人类真值时，用 65% 直接判整轮无效缺少依据。
v2 改为：裁决先独立作答、再看匿名随机顺序方案，并完整报告 adopt-A/B/synthesize/unresolved；不设人为
五五开硬门。

### 3. 不把 Consistency 的 outcome-row 重叠当成归因泄漏

正式 C0/C1 matched ablation 中，两格看到相同 outcome rows 和 correctness labels，唯一变化是 C1 增加
relation metadata/loss；这种重叠本身不是泄漏。真正必须做的是 held-out relation evaluation 和 query-disjoint
ranking。v2 记录跨任务 overlap，但不要求 C 数据离开 outcome pool。

### 4. 不在数据 smoke 中悄悄发明新的 hard-negative loss

clean 当前已经对 different-semantic/same-style pairs 做 negative consistency。审查建议的“同 query 不同答案/
不同方法”很适合做 verifier control，但若直接加入训练 loss，就成了新方法。v2 把它保留为 rejection control；
训练 hard-negative 若要改变，必须单独预注册。

### 5. 不要求自然 C accept rate 人为落在 40%–85%

C proposal 本来经过机械预筛，目标就是富集合格 pair；强迫出现固定比例 reject 会把数据构造变成迎合统计量。
v2 报 raw prevalence、class-specific agreement 和 κ；在自然数据只有一个类别占绝对多数时，κ只作描述，
另用隐藏 reject controls 防止“全部 accept”偷懒。

## 执行前仍然真实缺失的东西

协议升级不等于流水线已经实现。当前 clean 分支仍缺：

- source freezer / cluster-level dedup；
- `clir_numeric_multisource_v2` 及回归测试；
- `clir_material_claim_unitizer_v2` 与 exact-token materializer；
- C/H/prior proposal builders；
- 三类 strict JSON validator、agreement/self-agreement/control report；
- 第三模型 blind audit/adjudication runner；
- query-sharded rollout 与最终 artifact publisher。

因此下一步是按 v2 写最小 deterministic pipeline 和 toy/tiny fixtures；不是现在就把两份标注 prompt 发出去，
更不是直接训练更多 epoch。
