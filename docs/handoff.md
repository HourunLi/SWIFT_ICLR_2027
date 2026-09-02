# CLIR clean integration 交接说明

本分支从 `main` 的十余文件结构重新出发，只整合 `panzhixin` 分支中可复用的工程进展和通过相应门的模块部分。它不是对 `panzhixin` 的压缩复制，也没有继承其大量标注流水账、版本化 runner 或失败实现。当前远端顶端只保留训练/打分/评测代码、测试、可运行配置、README、本 handoff、核心方法说明、关键 smoke 协议，以及 v6 Consistency、v7 排序/H0 和 v12-posthoc Prior 的关键扩量协议；阶段报告、PDF、v1 协议及审查过程材料的最后完整快照是提交 `596a5e4`，需要时可从 Git 历史恢复。rollout、双盲包、原始标签、feature、checkpoint 和大型 scored JSONL 继续只留在被忽略的 `run_artifacts/`；2026-08-26 重新生成的阶段 PDF 也只作为本地产物交付，不进入精简远端。

当前唯一运行配置是 `configs/best_current.json`。这里的 “best current” 指当前最清晰、最可维护的**整合方案**；历史上最高的单次联合矩阵 BoN@16 是 correctness-only J0 `.920`，不是三模块联合成功。

## 2026-08-23 复核状态

在 commit `8b116c4` 的 clean integration 上完成了完整测试、toy resume、旧真实 manifest 接入、全宽梯度 smoke、1-epoch 真实训练和 500×16 排名评估。审计中发现 CUDA 续训会失败：`torch.load(..., map_location=cuda)` 把 CPU RNG state 搬到 GPU，随后 `torch.set_rng_state` 拒绝该 tensor。当前工作树已改为 checkpoint 先在 CPU 加载，再由 `load_state_dict` 恢复 model/optimizer；针对性测试与完整测试均通过，最终为 `37 passed`。

真实数据门与试跑：

- train 为 3968 rows / 496 queries × 8，validation 为 8000 rows / 500 queries × 16，query overlap 为 0；代表性 BF16 feature shape 为 trajectory `[221,101376]`、condition `[105,101376]`。
- train 中有 3968 correctness rows、27 个 consistency positive pairs、702 个 negative pairs、48 个 onset rows，以及 48 行/14,307 token 的 paired key/complete supervision，覆盖当前所有 active objective。
- 默认模型实测 `5,347,593` trainable parameters；全宽两个模块 batch 的 loss/gradient finite，峰值显存不超过约 `2.94 GiB`。
- seed 42、1 epoch 的 train/mechanism-dev total 为 `.5374/3.6609`。checkpoint SHA-256 是 `e1dba08f91d6529213db1acadfc274a422a76c7c8d096a74da2576290c7c891f`。
- checker v4 的 ranking validation 上，BoN@16/random/oracle 为 `.906/.8925/.976`，within-query pairwise 为 `.6241`；selected-minus-random paired delta `+.0135`，10,000-query-bootstrap interval `[-.0045,+.03175]`。

证据等级是 `small-scale real pipeline pilot`。它证明 clean 代码可以消费旧全层 BF16 artifact、联合 loss 可以真实更新并产出完整 checkpoint/score；1 epoch/1 seed、区间跨 0、没有 matched correctness-only clean baseline，所以不能证明任何单模块或 full integration 增益。

`configs/clean_ablation_v1` 冻结了 7-cell 三 epoch 筛选矩阵：C0 correctness-only、C1 consistency、H0 onset BCE、H1 onset BCE+negative tail、P0 direct priors、P1 direct+mutual priors、full integration。所有 cell 的架构、初始化 RNG、sampler、优化器和预算相同，仅 loss-family 权重变化。预注册 staging 是先完整跑 seed 42；若数据/hash/finite gate 正常，则不按 seed-42 排名挑选，而是全矩阵扩到 seeds 43/44；只有所有 cell 的三 epoch 曲线均未饱和且 mechanism dev 未明显恶化时，才统一续到五 epoch。该协议现已执行完毕，结果见下一节。

数据承载边界也已量化：train correctness 为 3590 正/378 负；consistency 只有 27 个训练正 pair 且没有 held-out relation；H train 为 17 个正 onset + 31 个 clean，dev 为 6 + 10；prior 虽有每个 head 14,307 个 token 标签，但只来自 48 条相关 trajectory，dev 只有 16 条。故这套数据足够做 matched engineering/screening ablation，不足以做正式机制归因；更多 epoch 只会重复相同标签，不能替代扩标和独立 held-out mechanism set。

## 2026-08-24 三随机种子消融状态

`configs/clean_ablation_v1` 的 7 个 cell 已完整运行 seeds 42/43/44、每个 3 epochs，
没有按 seed-42 结果挑选扩跑对象。21 个 checkpoint 都绑定
`da913318dae92ed8d436564729b92afb4c93f44c`、`dirty=false`、配置/数据/环境/命令；
统一 500×16 ranking scoring 和严格 candidate parity 也已完成。新增
`summarize_clir_ablation.py` 做跨 cell/seed 的候选、标签、input/checkpoint hash 校验及
paired-query uncertainty，`evaluate_clir_mechanisms.py` 分开评估 H、onset、value shift
和 dual-prior target。

BoN@16 mean ± sample SD：C0 `.9173 ± .0061`、C1 `.9220 ± .0040`、H0
`.9267 ± .0110`、H1 `.9187 ± .0058`、P0/P1 均 `.9180 ± .0072`、full
`.9160 ± .0080`。相对 C0，C1 为 `+.47` points、H0 为 `+.93`、P1 为
`+.07`、full 为 `-.13`；所有 paired query bootstrap 区间跨 0。H0→H1 的 gold-tail
增量是 `-.80` points，三个 seed 都回退。H1 的 `post-onset − pre-onset` token value
约为 0，但所有 token value 整体约为 `-.62`，再次呈现 global shift 而非 locality。

direct priors 在 16-row dev 上可学（P0 key/complete AUROC `.663/.869`），但 mutual
没有增量机制或 ranking 收益；H0/H1 有弱 token/path 排序信号，onset ±5 仍为 0；
consistency 没有 held-out relation set。多个 auxiliary cell 的 mechanism-dev 随 epoch
恶化，故统一 3→5 epoch 门失败，没有继续训练。完整历史文档可用
`git show 596a5e4:docs/clean_ablation_v1_results.md` 查看；本 handoff 保留当前裁决。正确状态是：工程矩阵闭环
通过，部分 auxiliary target 在小 dev 上可学，任何模块的 held-out ranking efficacy 和
full integration 增益都尚未建立。

### CH0 交互补测

用户提出的缺失组合已按查看结果前冻结的协议完成：`CH0 = correctness + consistency +
onset BCE`，不含 gold negative tail、direct/mutual prior 或其他辅助 loss。它在 commit
`582da9af65da622608d796f68b035f523b13009d`、`dirty=false` 上使用与原矩阵完全相同的
train、16-row mechanism dev、500×16 ranking population、3 epochs 和 seeds 42/43/44。
三个 checkpoint、scored input、candidate identity/order/labels 和配置/数据 hash 均通过。

| Cell | BoN@16 mean ± SD | seed 42/43/44 | pairwise mean ± SD |
|---|---:|---:|---:|
| C0 correctness | `.9173 ± .0061` | `.916/.912/.924` | `.6860 ± .0200` |
| C1 consistency | `.9220 ± .0040` | `.918/.922/.926` | `.6937 ± .0088` |
| H0 onset BCE | `.9267 ± .0110` | `.932/.914/.934` | `.6753 ± .0174` |
| CH0 C1 + H0 | `.9153 ± .0042` | `.920/.912/.914` | `.6942 ± .0181` |

CH0 相对 C0/C1/H0 的 BoN@16 均值分别为 `-.20/-.67/-1.13` points；其中 H0→CH0
逐 seed 为 `-1.2/-.2/-2.0` points，fixed-seed query bootstrap 95% interval 为
`[-2.20,-.13]` points，seed+query hierarchical interval 为 `[-2.73,+.47]` points。
二因子交互 `CH0 - C1 - H0 + C0` 为 `-1.60` points，逐 seed
`-1.4/-1.2/-2.2`，fixed-seed query interval `[-3.07,-.20]`，hierarchical interval
`[-3.40,+.13]` points。准确裁决是“小数据筛选中没有加和，并出现一致的负交互信号”；
三个 seed 和稀疏辅助监督不足以声称两个模块天然不兼容。

CH0 的 pairwise `.6942` 与 C1 基本相当且高于 H0，但 BoN@16 低于三者，说明它并非让
所有排序全面变坏，更像是改变了最顶部候选的极值排序。机制 dev 上 CH0 的 H token
AP/AUROC 为 `.450/.749`、path AUROC `.872`，优于 H0 的小样本点估计；但阈值 `.5`
完全不报正 onset，onset ±5 仍为 `0/6`。因此辅助 H 排序诊断变好不能替代最终选择效果，
也不能证明 onset boundary 已学好。

这里的命名关系必须保持清楚：`H0 = correctness + onset BCE`；`H1 = H0 + gold tail`；
`CH0 = C1 + H0`；旧 `full = C1 + H1 + P1`。所以旧 full 既不是 H0 也不是单独 H1，
它同时包含 consistency、onset BCE、gold-tail、direct priors 和 mutual priors。CH0 才是
检验 C 与 H0 是否能组合的干净二因子 cell。

### Prior→reward gate 补测

`configs/clean_gate_ablation_v1` 已在查看新指标前冻结并完成：P0 是 correctness + direct
key/complete prior，PG0 只增加 `gate_prior_weight=.0625`。该值匹配 `origin/main` 的总
有效强度 `.25×.25`；mutual、C、H、tail、progress 和 reconstruction 全部关闭。两 cell
使用同一 3968-row train、16-row mechanism dev、500×16 ranking pool、3 epochs 与
seeds 42/43/44，checkpoint 绑定 commit
`649747f3605e820430d4c93d788e368676ff37ea`、`dirty=false`。重新训练的 P0 state dict
与旧 clean P0 逐 tensor bit-exact。

PG0 的 held-out gate→fused-prior squared-L2 从 P0 `.01195` 变为 `.01335`，只有 seed42
改善，seeds43/44 恶化；key AP `.2969→.2926`、complete AP `.9210→.9251`，prior
protection 与 gate-collapse guard 通过。BoN@16 从 `.9180→.9167`，逐 seed delta
`+.2/-.8/+.2` points，fixed-seed query interval `[-.87,+.60]`、seed+query interval
`[-1.20,+1.00]` points。pairwise 点估计约 `+.35` points，但没有转化为主指标增益。

gate 确实影响了 score：各 seed 有 `53.2%/75.6%/57.6%` query 更换最终候选；其中绝大
多数 correctness 不变，错→对与对→错合计为 `18 vs 20`，净少 2/1500。准确裁决是：
工程 direct coupling 闭环，但 main-scale objective 没建立 alignment learnability 或
ranking efficacy。该实验当时支持保持默认 0；它作为 `.0625` 单点的历史结果保留，完整
旧文档可用 `git show 596a5e4:docs/clean_gate_ablation_v1_results.md` 查看。

### Prior→reward gate 工程默认值选择 v2

用户随后把“保留 `main` 原始 shared-gradient coupling，并默认开启”定为方法身份约束，
同时授权在当前开发 population 上选一个相对保守的固定强度。新协议在查看结果前提交
`.25/1/4/10` 四点配置，复用已完成的 `0/.0625` anchors；12 个新 run 使用相同 3968-row
train、16-row mechanism dev、500×16 ranking development、3 epochs 和 seeds 42/43/44。
mutual、C、H/tail、progress 与 reconstruction 全关，只改变 `gate_prior_weight`。

所有正权重均通过 finite、key/complete AP protection、normalized entropy 与 effective
support 门。BoN@16 三-seed mean 为：`0=.9180`、`.0625=.9167`、`.25=.9187`、
`1=.9180`、`4=.9173`、`10=.9207`。`10` 是 raw best，但 `.25` 与它恰好相差冻结的
near-tie 阈值 `.002`，所以按“近似时选更小权重”规则固定 `.25`。`.25−P0` 仅 `+.07`
point，fixed-seed interval `[-.80,+.87]` points、hierarchical interval
`[-1.20,+1.53]` points，均跨 0。

因此 `RewardConfig` 和 `configs/best_current.json` 现在都默认 `.25`。它保持
`origin/main` 的 gate normalization、detached 50/50 fused prior、full-trajectory
squared-L2、shared mask 与 shared-encoder gradient 路径；没有引入新的推理分数项。
clean 外层 `prior_weight=1`，故绝对 coupling 系数 `.25`，是原 main 总有效系数 `.0625`
的 4 倍。这个值的证据标签只能是 `dev-tuned engineering default`：同一 500-query dev
参与了选择，而且选中点没有改善 held-out gate L2。扩大独立 prior/ranking 数据后应固定
`.25` 做 off/on 复测，不再消费当前 dev 调参。完整旧文档可用
`git show 596a5e4:docs/clean_gate_tuning_v2_results.md` 查看。

## 2026-08-25 多题源扩量 smoke v2

v1 经两份互盲外部 AI 审查后共同判为 block，且没有生成、标注或训练任何数据；现状态为
`superseded_before_execution`。v1 与逐项审查材料已归档出分支顶端，最后完整快照为
`596a5e4`；当前执行只认下述 v2。
修订后的 canonical 协议为
[`data_expansion_smoke_protocol_v2.md`](data_expansion_smoke_protocol_v2.md)，机器配置为
[`../configs/data_expansion_smoke_v2/protocol.json`](../configs/data_expansion_smoke_v2/protocol.json)。

v2 pipeline 已实现，并用 8-query/64-row 确定性 fixture 验证 source freeze、checker、exact-token
unitizer、C/H/P proposal、隐藏控制项、自一致性、第三模型先独立后匿名裁决和 final materialization。
真实源导出也已完成：7473 条 GSM8K train +1218 条 ASDiv-A =8691 条；旧
outcome/ranking/mechanism population 共形成 1108 个唯一 GSM8K query exclusions。固定 Phi tokenizer 的
小数/货币、缩写、LaTeX/Unicode、空行和 terminal EOS 回归均通过。

近重复检索共发现 29 对，其中 2 对两端都已经被历史排除，送标没有意义；剩余 27 对按 hash 冻结后完成了
A/B 互盲回答。两份文件的 population/schema 均通过，target 判断 27/27 一致，7 duplicate/20 distinct，
κ=1.0；`dedup-triage` 为 0 行，所以不需要第三模型。恰有一端被历史排除的 4 对保留在原始分母，因为一旦
判 duplicate，另一端也必须随整簇排除。用户报告 A 为 OpenAI `gpt-5.5-sol`（xhigh），B 为 Anthropic
`claude-opus-5`（high）：模型系列不同且均非 Phi，去重决定可以落盘；但产品界面不暴露精确 revision 与
temperature，本地 roster 必须保留 unknown/unverified，不能声称满足预注册 temperature=`0`。

随后真实执行已经推进到自然 proposal 冻结：100 个永久 train-only query（60 GSM8K +40 ASDiv-A）与
1108 个历史排除项零交集，选中 query 集合 hash 为 `08fc850d…df78b`。固定 Phi-3.5 revision 在单张
NVIDIA L20Z 上以 `vllm==0.5.3.post1`、TP=1、BF16、`max_num_seqs=32` 生成 800/800 条正常 stop rollout，
没有空输出或长度截断，有序 raw hash 为 `3b47bd39…101f`。checker 得到 680 numeric match、39 明确
numeric mismatch、78 candidate-not-numeric mismatch 与 3 conflicting-boxed-answer ineligible。
799/800 条通过 exact-token unitization；唯一失败行是生成 token 的非规范等价切分，而且自身已经因两个
冲突 boxed answer 排除，所以保留原始审计、不进入 proposal，不修改冻结 token IDs。

确定性 proposal 已发布：40 个 C pairs/hash=`6a97b2bf…9145`，60 个 H/P rows/hash=`36a2b380…80cb`。
H/P 恰为 GSM8K match/mismatch 各 18、ASDiv-A 各 12；C/H query overlap=24，trajectory overlap=0；
H/P material-claim unit 最少 5、中位数 12。A=`gpt-5.5-sol`/xhigh、B=`claude-opus-5`/high 已完成
C/H/P 六个隔离上下文；全部 label 文件通过 population/schema/index 校验，A 的三组 self-repeat 均为
100%。但 v2 按预注册规则失败，未进入第三模型、finalize、hidden-state 抽取或训练。冻结规模与预算为：

- outcome：60 GSM8K train +40 ASDiv-A queries，每题 8 条 Phi rollout，共 800 raw trajectories；
- Consistency：标 40 个 GSM8K natural proposals，按冻结顺序留下前 30 个最终 accept；
- H/P：对 60 条 distinct-query trajectories 完整、独立地做 H 和 prior 两套双标；
- final：只从 joint H+prior usable 交集中按冻结 source/hash 规则取 20 first-bad-unit positive +20 clean，
  每类每来源至少 5；
- 100 个 query 永久 train-only，不得进入 mechanism dev、ranking validation 或 test；C/H 可 query 重叠，
  但同一 trajectory 不可复用，所有 overlap 必须报告。

v2 解决了 v1 最危险的隐式筛选：C/H/P proposal 的枚举、机械过滤、source/numeric strata、每 query 上限、
hash tie-break、有序 manifest 和 metric 分母都必须在 A/B 看到任何标签前冻结；格式失败、low、uncertain
和 ineligible 不得从原始分母消失。H 与 prior 要先在全部 60 条上分别双标，再做 joint selection，不能先
拿到容易的 H 行后只把它们送 prior。

标注独立性也从偏好升级为硬门：A/B 必须是两个不同模型系列，且都不得与 Phi generator/backbone 同族；
同模型两次调用只可 debug。分歧需要第三个不同系列模型先独立作答，再查看匿名随机顺序方案；没有合格
第三模型就 drop，不复用 A/B 投票。最终标签只能叫 `silver_dual_ai_v2`，并保留
`label_source=auto_agree/adjudicated`。

H 不再宣称精确首错 token。`clir_material_claim_unitizer_v2` 必须把完整 `output_token_ids` 切成连续、
无重叠、左闭右开的 ranges，包括空白/标题/final wrapper/terminal control token；一个 material unit 只能有
一个可判断 claim。AI 选择 `first_bad_unit_index`，训练兼容 onset 只是该 unit 的 `token_start`。
`panzhixin` 的 deterministic 行/句 segmenter、fixed-unit F1、role-blind adjudication 可作为实现参考，但旧
unitizer 只保证可见字符覆盖，没有完整 v2 token-partition 与原子 claim 契约，不能原样搬来即宣布通过。

checker 升为 `clir_numeric_multisource_v2`。公开语义准确叫 `numeric_value_match`；兼容字段
`correctness` 必须声明同一语义，不能冒充完整单位/实体正确性。截断/空输出/非法 ID 保留审计但不进任何
训练或机制 proposal。文本/checker/unitizer/dual-AI gates 全过后才估算实际 token 数并抽 `33×3072`
BF16 feature。

SVAMP 的角色已纠正：论文明确由 100 个 ASDiv-A seeds 生成变化题。继续使用 ASDiv 训练时，SVAMP
只能是 protected **ASDiv-derived contrast/challenge set**，不能支持独立来源泛化；它仍不用于本轮训练/
调参。独立泛化需另选 holdout，或以后发布密封 seed-family 排除协议。

本轮仍是 `pipeline smoke`，不训练 CLIR，不给模块 efficacy、Best-of-N 或 gate 权重结论。`.25` gate
作为方法身份默认值保留，但本轮不使用它选样、训练或调参。

实现入口是 [`../prepare_clir_smoke.py`](../prepare_clir_smoke.py)，核心契约在
[`../src/clir_smoke.py`](../src/clir_smoke.py)，回归测试为
[`../tests/test_clir_smoke_v2_pipeline.py`](../tests/test_clir_smoke_v2_pipeline.py)。fixture 实测 64/64
token partition 通过、32 numeric match/32 mismatch、2 个 C 与4 个 H/P proposal 闭环。A/B 的盲包会
混入未公开 control 和仅 A 可见的 self-repeat；A/B 完成后，`triage` 才生成第三模型的独立 audit/dispute
包，第三模型独立输出落盘后，`adjudication-package` 才暴露匿名 Option 1/2。`finalize` 任一预注册门失败
就只写失败报告，不发布 `pre_extraction.jsonl`。

双标的原始诊断如下；这些是 `dual-AI Silver, no human verification` 的一致性结果，不是准确率：

| 任务 | 原始结果 | v2 门 |
|---|---|---|
| C | 36/40 decision 一致；A/B accept=34/38；κ=.459；controls 各 4/4 | agreement=.90、分歧=.10；κ 因 B reject<5 按协议 N/A |
| H | path 60/60；54 clean、6 common-positive；exact/±1 onset=5/6；controls 各 6/6 | common-positive 6<15、±1=.833<.85，且 final 20 positive 无解 |
| Prior | eligibility 60/60；Key F1=.928；Complete F1=.784；exact sets 27/60；controls 各 3/6 | control<1、Complete<.82、分歧/裁决候选 .55>.40 |

H 的 30 条所谓 numeric mismatch 中，24 条其实是正确链，只因输出 `\boxed{38 cents}` 一类 boxed
prose 被 v2 checker 当成不可解析并误记 0；A/B 都将它们判 clean，剩余 6 条真实 mismatch 才全部给出
positive onset。这是明确的 checker 系统性假阴性。按冻结协议，checker 改动必须升版、重建所有依赖
proposal 并重做标注，因此不允许用第三模型把 v2 的 raw gate “裁成通过”。v2 状态固定为
`FAIL_PIPELINE_CHECKER_AND_YIELD`，本地产物只留审计。

实现已补三项工程缺陷：`clir_numeric_multisource_v3` 能从 boxed 单位短语/答案句/等式/金额/复合时长中
提取受支配数值，同时 `materialize` 由协议显式钉住 checker，仍能复现 v2；set F1 在完全不相交时现在返回
0 而不再除零崩溃；`triage` 会原子写出裁决前 raw agreement/control/self-repeat 报告。用 v3 只读重算同一
800 rows 得到 754 match、42 parsed mismatch、1 non-numeric refusal、3 conflicting boxes。机制可用的
mismatch 只覆盖 13 个 GSM8K 和 1 个 ASDiv query，仍不足 20 个 query-distinct positive。Prior 的 33 条
Complete 分歧中 29 条是 B 为 A 的严格超集，而 Key 55/60 完全一致；新版必须把“沿候选实际依赖链的所有
非冗余步骤”与“可重新计算出的最短证明”分开写清，并使用真正检验多步链的 control。

v2 不继续消费第三模型预算；其标签只作失败诊断，不跨 manifest 复用。源数据、模型输出和其他大
artifact 全部留在 `run_artifacts/`，不推远端。

## 2026-08-25 多题源扩量 smoke v3

当前 canonical 协议为
[`data_expansion_smoke_protocol_v3.md`](data_expansion_smoke_protocol_v3.md)，机器契约与标注规则为
[`../configs/data_expansion_smoke_v3`](../configs/data_expansion_smoke_v3)。v3 在任何新标签出现前完成了
source、去重、query pool、primary/reserve、checker、Complete 语义、proposal strata 与停止规则的冻结。

数据侧完整保留 v2 的 100 个 train-only incumbent，并从 MATH train 的四个数值学科、level 3/4/5 中
严格筛出 scalar-numeric 长链题。每个“学科×难度”预选 5 primary +4 reserve，形成新增 60+48；完整 208
题 manifest hash=`bc57c065…9481`。production loader 只请求 pinned train parquet；冻结前一次开发性
`datasets` 预检曾把 test 写入本机 cache，但没有查看 test row/答案/模型表现，也没有进入选样或调参，
所以只能声称执行路径 train-only，不能声称工作区从未下载 test。

primary 新增 60×8=480 条固定 Phi rollout，raw hash=`8ed41fad…185`；与绑定 hash 的旧 800 条合并为
160 题/1,280 rows。v3 checker/unitizer 得到 995 match、240 parsed mismatch、10 conflicting answers、
14 parse failures、21 truncated，exact-token partition 为 1,280/1,280。截断占 1.64%<2%，全部排除；
parse failure 只审计，不充当 H mismatch。两条非规范等价 tokenization 通过 frozen-prefix decode fallback
映射回原始 ID，未改写 token 轴。

primary readiness 已通过：机制可用、query-distinct parsed mismatch 为 57，门为 30；40 C 与60 H/P 的
全部 strata 都能冻结，所以 48 reserve 未生成。C proposal hash=`1cd93bd…9a75`；H/P hash=
`cd5bd369…f3d`，其中 GSM8K match/mismatch=10/8、ASDiv match=12、MATH match/mismatch=8/22。

Complete 已改成候选实际依赖链里所有被后续使用的唯一非冗余中间步骤，不能压成可重算的最短证明；新的
prior hidden control 明确区分中间量与直接答案。盲包已经生成，A 的 C/H/P 为 52/78/78，B 为44/66/66，
含约 10% controls 与 A-only 20% self-repeat。用户指定 A=`gpt-5.5-sol`/xhigh、B=`claude-opus-5`/high；
六个隔离任务均已落盘并通过 population/schema/index 检查。所有 task/annotator hidden controls 都是
100%，A 的三类 self-repeat 也都是 100%。原始标签及报告仅在 `run_artifacts/`，不得推远端或发送
PRIVATE manifest。

raw triage 的终态是 `STOP_RAW_GATE_FAILURE`，失败项正好三条：C agreement=`26/40=.65<.90`、C 最低
裁决比例=`14/40=.35>.20`、Prior 最低裁决比例=`35/60=.5833>.40`。C 中 A/B accept 数为25/39，B 只
reject 1 条；14 个争议里 10 个的理由直接围绕“是否近乎照抄”，另 2 个围绕乘法结合/单位表示是否已改变
关键中间量，2 个围绕是否修正/新增了错误。也就是说主要问题是自然样本上的 near-copy/方法边界没有被
操作化到两模型同一阈值，不是格式错误或标注者随机漂移。

H 是清晰的正结果，但只限于标注可操作性：path=`59/60=.9833`、κ=`.9667`、共同 positive=30、五个以上
material units 上 exact 与 ±1 onset 均为 `26/30=.8667`、最低裁决比例 `5/60=.0833`，所有 raw H 门均过。
这不等于标签事实准确，也不是 H 模块有效性证据。

Prior 比 v2 明显改善：eligibility=`60/60`，Key/Complete macro F1=`.9167/.9267`，两边均 0 条
`Complete=全部 material units`；Key exact=`55/60`。但 Complete exact 只有 `26/60`，导致两类 target
同时 exact 的只有 `25/60`。在 Complete 分歧中，24 条是 B 为 A 的严格子集、4 条是 A 为 B 的严格子集、
6 条非嵌套，说明依赖链大体相近但自然链的可选 unit 仍未唯一化。

冻结协议禁止裁决救活 raw failure，所以 `third_model_send_allowed=false`：不得发送已经机械生成的第三模型
包，不运行 adjudication/finalize，不抽 hidden state，也不训练。`triage` 已补 raw-gate fail-fast 状态，
避免旧输出误报为“第三模型包已就绪”。

用户随后同意优先用提示词修复 C，而不是先加复杂相似度算法。新的
[`data_expansion_smoke_protocol_v4.md`](data_expansion_smoke_protocol_v4.md) 已冻结为 C-only prompt
development：同一提示词先做数学路径判断，再做表达差异判断，并明确公式/数字/必然顺序相同不等于照抄。
第一阶段只回放上述 14 条已检查争议，门为至少13/14 agreement、每边 review≤1 且 accept/reject 各≥2；
这些行不训练、不算新证据。通过后才制作 30 条不复用 v2/v3 C item/query 的新确认 pair，门为27/30。
两个全新 A/B 上下文已经完成，冻结检查器终态为 `STOP_REPLAY_FAILURE`。decision agreement 只有
`7/14=.50<13/14`，kappa=`-.0426`；A accept/reject=`8/6`，B=`9/5`，两边 review 都是 0，格式、理由
前缀和反塌缩门均通过。七个分歧里四个是近抄/真实展开边界，两个是单位表示或乘法分组是否算同一路径，
一个是回答夹带实质错误是否必须拒绝。因此失败不是输出质量问题，而是 prompt 没有把自然困难边界变成
跨模型稳定规则。按预注册决定，不制作 30 条新确认，不调用第三模型，不训练，也不把回放当可靠性证据；
prompt-only 修复路线到此关闭。

v5 随后按用户同意的简单方案把困难边界机械化，规范见
[`data_expansion_smoke_protocol_v5.md`](data_expansion_smoke_protocol_v5.md)，机器契约与两个启动提示词在
[`../configs/data_expansion_smoke_v5`](../configs/data_expansion_smoke_v5)。程序先要求两边 numeric match 和
规范化答案相同、至少 4 个 material claims、token 长度比 `[1.15,3]`、数学 trace 相似度≥`.60`、数字
trace 相似度≥`.75`，并把非数学 word-bigram Jaccard 固定在 `[.10,.40]`；AI 不再判断方法/风格/近抄，
只判断是否夹带实质错误。v4 的 14 条已看争议只作开发回归：机械规则放行 3 条，三条均为旧 A/B 共同
accept，不作为训练、准确率或可靠性证据。

实现和协议先冻结在干净提交 `a60b2cb`。随后消费此前完全未 rollout/未标注的 48 个 MATH reserve
queries，得到 384 条候选，生成 provenance 绑定 `a60b2cb` 且 `code_dirty=false`。384/384 exact-token
unitization 成功；checker 分布为 numeric match 201、parsed mismatch 154、parse failure 10、truncated 16、
ambiguous multiple answers 3。机械筛选在 38 个有合格正确候选的 query 中找到 16 个 query-distinct pairs，
按冻结 hash 顺序取前 12 个。自然 item manifest SHA-256 为 `e9017ef3…23a3f`，A/B package SHA-256 分别为
`e3e2e223…abe59` / `30dcb839…37b66`。

GPT-5.5-sol/xhigh 与 Claude Opus 5/high 已在两个隔离上下文中完成 A/B 包。冻结检查器终态为
`PASS_FRESH_MECHANICAL_AUDIT`，所有 raw gates 通过：自然 decision agreement=`12/12`，两边均为 12 accept、
0 review，controls 各 `4/4`，A self-repeat=`3/3`，理由前缀与 schema 全部合法。A/B label SHA-256 分别为
`7004f129…b149` / `185f0b15…ca1`，审计报告 SHA-256 为 `e88d389b…196f`。自然集是机械筛出的正 pair，
因此单类 12/12 accept 符合任务结构，但报告 κ=1 在单类分布下不提供额外的强统计证据；两个含明确错误的
controls 均被正确 reject，排除了“所有输入都 accept”的简单退化。

这个 PASS 只把下一道门从“能否稳定制作 C pair”推进到“另发正式扩量协议”。v5 自身仍明确
`eligible_for_training=false`、`third_model_allowed=false`，不发布训练 manifest、不抽 hidden state、不训练，
也不支持 Consistency 改善 Best-of-N 的结论。

## 2026-08-26/29 数据扩容主协议 v6/v6.1：关系与 selected feature 均 PASS

用户确认采用中档 Consistency 扩量方案。canonical 文档是
[`data_expansion_scale_protocol_v6.md`](data_expansion_scale_protocol_v6.md)，机器契约是
[`../configs/data_expansion_scale_v6/protocol.json`](../configs/data_expansion_scale_v6/protocol.json)。v6 不是
v5 数据的放大复制，也不复用那 12 对 smoke：它计划从所有历史 query 之外重新冻结 2,000 个 train-only
query，来源为 1,400 MATH train 与 600 个长链 GSM8K train，每题 8 条、共约 16,000 条 raw rollout。
生成前按 original query + near-duplicate template cluster 拆成 1,500 train-acquisition 与 500
heldout-acquisition，目标是 400 train positive relations、150 held-out positive relations，并从 heldout
query 确定性匹配 150 个 different-semantic/similar-surface hard negatives。

机械筛选完全复用 `clir_consistency_mechanical_v1` 的 v5 阈值：两边 numeric match/同答案、至少 4 个
material claims、长度比 `[1.15,3]`、math trace≥`.60`、numeric trace≥`.75`、surface bigram Jaccard
`[.10,.40]`，每题最多一对且 hash tie-break。AI 只审事实错误；A/B 必须是不同、非 Phi 模型系列，最多
50 个自然 pair 一 shard，带 4 个隐藏控制和约 10% 跨 shard 自重复。未来只有双方共同 accept 的 pair 可按
冻结顺序入选；agreement<.95、控制非 100%、自重复<.95 或 common accepts 不足 400/150 都 fail-closed，
不允许第三模型救活。无人类复核，所以标签只叫 `silver_dual_ai_consistency_v6`。

预算按真实 v3/v5 token 长度估算：全层 BF16 每 token `33×3072×2=202,752` bytes。2,000×8 若全部抽
feature 约 1.42 TB，因此 v6 明确先 rollout/check/unitize/filter/双标，最后只抽入选的 1,100 条 views 与
550 个共享 prompt。原先按 420 output token 估计约 105.23 GB；raw rollout 实测均值为 449.24，最终预算
必须等入选 views 确定后按真实长度重算。独立 ranking pool 的 TB 级 feature 另算，不能混入本轮预算。

2026-08-27 已补齐生成前执行契约和独立的 [`../prepare_clir_scale.py`](../prepare_clir_scale.py)；核心逻辑在
[`../src/clir_scale.py`](../src/clir_scale.py)。生成前版本只允许在 clean commit 上执行 `freeze/verify`，负责来源过滤、
永久排除传播、exact/near-template cluster、1,500/500 query split、40 个原子 shard、prompt 长度和 GPU/磁盘
预算复核。六份 manifest/hash 已通过，registry SHA-256 为 `7e6d6da9…352`。

2026-08-28 用户在收到 2,000 题、40 shard、约 16,000 rollout 与存储预算汇总后明确回复“开始”。为避免
改写已经绑定哈希的基础协议，授权单独记录在
[`../configs/data_expansion_scale_v6/rollout_authorization.json`](../configs/data_expansion_scale_v6/rollout_authorization.json)：
只允许 frozen rollout，materialization、标注、抽特征和训练仍为 false。`prepare_clir_scale.py` 因此新增逐
shard 原子生成、completion marker、只读复核和 40-shard fail-closed 合并；先用 `train-000` 校准，再在机械
运行门通过时最多 8 卡并行。

该 rollout 授权现已完整执行，生成代码与授权的 clean commit 是
`078eb6b1c3d1aa8c1c950030e4aeff496ea1f342`。40/40 shard、2,000/2,000 query、16,000/16,000 raw rows
通过独立 `verify-rollouts --require-complete` 和 `merge-rollouts`，终态为
`PASS_ALL_16000_RAW_ROLLOUTS_VERIFIED_V6`。合并文件 SHA-256 为
`f538373b3d99791001cbe2119466b0bd52e23a8337223053920371cda7e75139`，raw artifact 只留在 Git 忽略目录，
没有推到远端。

运行健康统计：15,393 条自然停止，607 条达到 1,024-token 上限（3.79%）；MATH/GSM8K 截断率分别为
5.35%/0.17%，train/heldout acquisition 分别为 3.78%/3.85%。1,999/2,000 个 query 至少有两条未截断
候选；另 1 个 query 以后按冻结规则排除，不能为凑数放松阈值。vLLM 展示文本统一比 token-ID decode 多一个
前导空格，strip 后 16,000/16,000 相等；保存的 token IDs 是唯一权威，没有发现 token 漂移。

这里完成的是 raw acquisition，不是可训练数据扩容。607 条截断输出尚未由材料化阶段排除，checker、
`clir_material_claim_unitizer_v2`、机械筛选、双 AI 审计、hidden-state extraction 和训练全部未启动，当前也
不能写“Consistency 已经有 400 对训练数据”。

用户随后明确授权“接着做，到要标数据再叫我”。独立的
[`../configs/data_expansion_scale_v6/pre_annotation_authorization.json`](../configs/data_expansion_scale_v6/pre_annotation_authorization.json)
把范围固定为：CPU 材料化、原样 v5 机械筛选、全部自然候选冻结和 A/B 隔离盲标包；真正调用两个 AI、标签
triage/finalize、抽 feature、训练、重生成 raw 或改阈值仍禁止。新入口每阶段拒绝覆盖旧 artifact，并提供独立
recompute verifier。每个标注 shard 最多 50 个自然 pair，另有 2 accept +2 reject 控制；A/B 各 10% 自重复
只放后续 shard，启动词要求一个全新上下文只处理一个 shard，防止上下文记忆虚高自一致性。执行前先把授权、实现、
启动词和 raw gate 评估器冻结；执行终点必须是 `PASS_PRE_ANNOTATION_PACKAGES_VERIFIED_V6`，然后通知用户启动双 AI。
本阶段不占 GPU，直到后续 selected-view feature extraction 和训练才重新需要 GPU。

标注前阶段已在 clean commit `5ab1fb743185a806f70269d66230768b7a4ad38d` 上执行完毕，并由独立
`verify-materialization`、`verify-proposals`、`verify-pre-annotation` 重算。材料化为 16,000/16,000 exact
partition，15,927 canonical mappings +73 frozen-prefix fallbacks；checker 统计为 numeric match/mismatch
`8877/5907`、truncated/parse-failed/ambiguous `607/350/259`，后 1,216 行均 audit-only。materialized SHA-256
是 `ca37d027…ea4fd`。

原样 v5 阈值留下 708 个 query-distinct natural pair：train 526、heldout 182；MATH 348、GSM8K 360；proposal
SHA-256 是 `795a1d47…33b73`。它们已打成 15 个 shard，每个 annotator 为 708 natural +60 controls +71
later-shard repeats =839 行；全部 A/B package 1,678 行复核通过。A/B 公共 index SHA-256 是
`160d8c7f…429a` / `e895cd29…b1c3`，private manifest 为 `07884ceb…904f`。标签目录仍不存在，报告明确
`annotation_started=false`。

双 AI 标注现已完成，当前阻塞点变为 deterministic raw-gate 与 hard-negative feasibility。最终 common-accept 至少要求 train 400/526（76.0%）、
heldout 150/182（82.4%），另需自然 decision agreement≥95%、review≤2%、两边各 60/60 controls 和自重复
≥95%。这些 raw gate 的评估逻辑已在看到任何标签前写入并测试，失败不允许第三模型补救。每个新上下文只能
处理一个 shard。首个 A/B shard 已存在后，用户澄清 A 实际为 GPT-5.6-sol/xhigh，不是基础索引中的 5.5；
[`../configs/data_expansion_scale_v6/annotation_model_amendment_v1.json`](../configs/data_expansion_scale_v6/annotation_model_amendment_v1.json)
绑定了当时两个 `shard-000`，SHA-256 为 `fadb3351…5a47`；该身份是 user-reported，exact revision/temperature
未验证。B 仍为 Claude Opus 5/high，包和 gate 都不变。两边分别重复使用
[`../configs/data_expansion_scale_v6/launch_prompt_a_5_6.txt`](../configs/data_expansion_scale_v6/launch_prompt_a_5_6.txt) 和
[`../configs/data_expansion_scale_v6/launch_prompt_b.txt`](../configs/data_expansion_scale_v6/launch_prompt_b.txt)，每次在全新会话中启动；
两边现均为 15 shard/839 行。commit `657d471` 上的冻结 raw evaluator 已完成：自然 agreement
`676/708=95.48%`，A/B controls=`60/60`，self-repeat=`71/71`/`69/71`，review=`7/708`/`0`，共同 accept
train/heldout=`474/167`；raw report SHA-256 为 `4c80626e…a598`，终态
`PASS_SCALE_V6_RAW_ANNOTATION_GATES`。

按冻结顺序选择的 400 train +150 heldout 正关系可行，source 分别为 train GSM8K/MATH=`210/190`、heldout
`85/65`，query 与 template cluster 跨 split overlap 都是 0。但 frozen hard-negative 阶段在 300 个 heldout
正视图上只有 10 条 eligible response-surface edge，greedy 只能选 8/150。因此 plan report SHA-256
`42052c40…bb0`，终态 `STOP_SCALE_V6_POST_ANNOTATION_PLAN`；没有发布任何 relation manifest，也没有抽
feature 或训练。raw PASS 只能证明 Silver 标注流程达到冻结一致性门，不能证明 C 有效，更不能覆盖后置 STOP。

用户在看到 v6 STOP 和只读可行性诊断后，明确同意
[`../configs/data_expansion_scale_v6/hard_negative_amendment_v6_1.json`](../configs/data_expansion_scale_v6/hard_negative_amendment_v6_1.json)
（SHA-256 `8cbcf62b…20b3`）。这是 post-failure engineering amendment，不是盲预注册：它不动任何 A/B 标签、raw
gate/分母、400/150 first-N 正关系顺序或 edge 阈值，只把负例来源从300 个正视图扩到所有现有 heldout
numeric-match 可监督视图，并把 greedy 换成固定 NetworkX 3.6.1 的 maximum-cardinality/preference matching。

实现与契约先冻结在 clean commit `bb992614319993617777a114645c2d0c871c7d7e`，再运行正式 plan。结果为
`PASS_SCALE_V6_1_POST_ANNOTATION_PLAN`：2,178 个端点、331,122 个 shared-bigram pair、605 条同阈值 edge、
maximum matching=167，最终选满150 条且300 个 endpoint 无复用；hard-negative source pair 为 MATH/MATH
138、GSM8K/GSM8K 12。原400/150 正关系有序 hash 仍是 `079d014b…f9d2` / `704a8317…73b6`，没有发生
post-hoc 重选。plan report SHA-256 为 `71a470e6…bb75`。

独立 `verify-final-relations-v6-1` 随后从父级标签、proposal 和16,000 条 materialized rows 重建图、匹配、
relation 与 inventory，逐行和 sidecar hash 全部相同，终态 `PASS_SCALE_V6_1_INDEPENDENT_RECOMPUTATION`，
verification report SHA-256 为 `48d69371…2756`。本地 Git-ignored manifest 文件 hash 为 train positives
`4709b143…009d`、heldout positives `fc88a094…96d1`、heldout hard negatives `fad547c5…48f2`、inventory
`0e3c796d…28db`。

精确 inventory 为1,357 trajectory +612 prompt，输出/提示 token=`460151/59952`，共520,103 feature token，
预算98.210 GiB（105.452 GB）。负例端点与正关系重叠43 个 view，故真正新增257 trajectory +62 prompt；旧的
101-prompt 粗估没有按 query 扣除已随正关系保存的44 个 prompt，现已纠正。这个结果只把 Consistency 数据构造
推进到“已发布并复核关系清单”，不是可学习性或 Best-of-N 证据。关系 plan 和 verifier 在其冻结时点明确
`feature_extraction_allowed=false`、`training_allowed=false`；后续只有新的窄授权才能只抽 inventory，始终禁止
抽全16,000条。

### Consistency v6.1 selected-inventory feature extraction：已完成并独立核验

用户在关系/inventory 独立复核完成后明确回复“ok，接着做”，授权只推进 exact feature extraction。机器记录
[`../configs/data_expansion_scale_v6/feature_extraction_authorization_v6_1.json`](../configs/data_expansion_scale_v6/feature_extraction_authorization_v6_1.json)
绑定 commit `a96b8441` 上的 v6.1 plan/verifier、三份关系、1,357-row inventory 与16,000-row materialized file；
范围只含 selected rows join、一个全宽 preflight、inventory-only 抽取、逐文件独立复核和最终 feature manifest。
完整16,000条抽取、重 rollout、改标签/关系/阈值以及训练都是 false。

新入口 `extract_clir_scale_features.py` 不改变通用 exact-ID 数学定义，只补正式规模所需的恢复与审计层：按 query
的总 feature token 做固定 largest-first 八路均衡；同题所有 view 在同一 GPU worker；tensor 与 query marker
分别原子发布；无 marker 的现有 payload 必须 reload 验证后才续用；8个 writer 全通过后，再由8个 CPU verifier
逐个重读1,969个 payload，检查 shape、BF16、contiguous、finiteness 和 SHA-256。完整执行规则见
[`feature_extraction_protocol_v6_1.md`](feature_extraction_protocol_v6_1.md)。

正式 run3 已在 code commit `64470c8c76ffdbeee3c1f810e8d0ea9d86752b95` 上完成：plan hash
`1a096175…67cf`、全宽 preflight hash `9080bd88…85d3`、最终报告 hash `a1ce2d9b…8daa`，终态
`PASS_SELECTED_FEATURE_EXTRACTION_AND_VERIFICATION_V6_1`。8个 writer 与8个独立 verifier 均通过；1,357个
trajectory +612个 condition 的 shape/BF16/contiguous/finiteness/SHA 全量一致，raw/serialized bytes 分别为
`105451923456` / `105455351485`。最终 extracted manifest hash 为 `ac1f35ff…7a8b`，有序行 hash
`1901e88c…c50`；`CLIRTrajectoryDataset` 额外成功加载普通行和最长980-token行。两个早期 preflight 停止目录
均为0 tensor，正式目录无 partial 文件，完整16,000条从未抽取。可提交的机器摘要是
[`../configs/data_expansion_scale_v6/feature_extraction_completion_v6_1.json`](../configs/data_expansion_scale_v6/feature_extraction_completion_v6_1.json)；
约102 GiB本地 artifact 保持 Git ignore。这个 PASS 不证明 Consistency 可学或提高 BoN，且最终报告继续
`training_allowed=false`；下一门是单独的 C-only 训练授权。

### Consistency v6.1 C0/C1 训练与 held-out relation 复测：已完成

用户随后明确回复“可以，你按照我们之前的计划做就行”，授权一轮窄范围、hash-bound 的 C-only
复测。实现、配置、协议与授权先冻结在 clean commit `2a55e02c43cab016eac0f6cd6bd2915ba63ba8ef`；
H、Prior、Full、新 feature、重标和 ranking efficacy 都不在授权内。

确定性训练视图由历史3,968条 correctness row 和400个新 train relation 的800个 endpoint
组成，共4,768 rows、896 queries。历史 auxiliary 字段全部去掉；新 relation 两端按保存的
output-token 相对长度得到 `relative_compact/relative_expanded`，同 relation 提供正对、不同 relation
同 style 提供 batch 内负对。C0/C1 共用完全相同的 manifest、sampler、batch、optimizer、seed 和 epoch，
配置只差 `consistency_weight=0/1`。train 与 heldout query/cluster overlap 均为0。

materialization、独立重算和两组真实 `[4,429,101376]` BF16 forward/backward 先通过；seed 42
完成一轮 pilot 后从同一个 full-state checkpoint 精确续到 epoch 3，seed 43/44 各从头跑3 epochs。
六个 checkpoint 的 model、optimizer、metrics、数据/config/seed/clean-commit provenance 全部复核通过。

150个 heldout 正关系与150个 hard negative 的三 seed 结果为：

| 指标 | C0 | C1 | C1−C0 |
|---|---:|---:|---:|
| 正对 cosine − 负对 cosine | .00016 | .17614 | +.17597 |
| cosine relation AUROC | .6875 | .7209 | +.0334，seed 方向混合 |
| 正对 cosine | .99991 | .98335 | −.01656 |
| 负对 cosine | .99975 | .80721 | −.19253 |
| score-gap separation | .18622 | .42970 | +.24348 |
| score-gap relation AUROC | .5954 | .7400 | +.1447 |

冻结主指标逐 seed 增量为 `+.0953/+.3110/+.1216`，relation bootstrap 95% 区间
`[+.1389,+.2155]`，按协议裁决为 `SUPPORTS_C1_HELDOUT_RELATION_SEPARATION`。score-gap
separation 三个 seed 也都改善，区间 `[+.1405,+.3457]`。

结论必须保守：C0 表示几乎整体塌缩，C1 可靠地展开表示并把 hard negative 推远；它没有让正对比 C0
更近，cosine AUROC 在 seed 42/43 下降、44 上升，且 C1 的 negative margin violation 仍为
98.0%/80.7%/96.7%。所以这是 hard-negative separation、去塌缩和 score-gap 结构的**部分机制证据**，
不是完整正对不变性，更不是 Best-of-N 证据。正负 relation 还共享43个 endpoint、44个 query/cluster，
bootstrap 只能解释为 relation-level 描述性不确定性。

机器摘要见
[`../configs/data_expansion_scale_v6/consistency_training_v6_1/completion.json`](../configs/data_expansion_scale_v6/consistency_training_v6_1/completion.json)。
本地 paired summary / final verifier SHA-256 分别为 `e8037676…3c139` / `2a3cda4f…e1555f`；大
manifest、checkpoint 和逐 relation 报告继续 Git ignore。

用户报告全部 shard 完成后，
[`../configs/data_expansion_scale_v6/post_annotation_authorization.json`](../configs/data_expansion_scale_v6/post_annotation_authorization.json)
（SHA-256 `7dcd096a…ab34`）在其阶段只解锁 deterministic label audit、条件式 400/150 选择和 frozen
hard-negative feasibility；它没有授权第三模型、裁决、修标签/阈值、feature extraction 或训练。后来的 selected
feature PASS 只来自上面的独立授权；后来的 C0/C1 训练只来自新的 C-only 授权，不能反向扩大这里的
post-annotation 授权。

## 与 `origin/main` / 历史 artifact 的兼容性

| 维度 | 结论 | 边界 |
|---|---|---|
| 最新 `origin/main` model API | 兼容 | identity 模式严格加载最新 main state dict；21 个共同 forward 输出在运行时对照中 bit-exact，clean 只额外输出 `layer_attention` |
| 旧 main checkpoint | 部分兼容 | condition bottleneck 修复之后的 identity checkpoint 可加载；修复前、缺少 `condition_hidden_proj/condition_delta_out` 的旧 checkpoint shape 不兼容 |
| clean real checkpoint | 不与 main raw-width 权重互换 | clean 新增 layer-axis encoder，真实参数 shape 与 main 直接吃 raw width 的模型不同 |
| 打分 checkpoint schema | 向后读取 | `score_clir.py` 接受 clean 的 `model_config` 或 main 的 `config`；是否能 strict-load 仍取决于具体架构版本 |
| 训练续训 schema | 不兼容 main | clean 需要 optimizer/epoch/RNG/data_state 的 full-state checkpoint；main 的 weights-only checkpoint 只能打分，不能精确 resume |
| CLI/config | 有意 breaking | main 的 loss-heavy CLI 改为一个 JSON 方法配置加少量运行覆盖项，旧命令不能原样复用 |
| `panzhixin` manifest | 已验证兼容 | nested `feature_metadata`、`feature_sha256`/`condition_sha256` 和全层 BF16 路径已在 3968/8000 manifests 上通过 |
| `panzhixin` 研究协议 | 不自动兼容 | sparse H、strict/encoded variants、versioned runner、标注与多 seed summarizer 没有迁入；不能把可读 manifest 等同于可复现旧协议 |
| evaluator | 工程兼容，已补多 seed 配对 | frozen prefix、stable tie、common population、random/oracle/pairwise aggregate、机制诊断、held-out consistency relation evaluator 及 parity-checked multi-seed paired summary 已有；formal 仍缺新的 protected ranking test 和完整 baseline 矩阵 |

## 迁移裁决

| 内容 | 当前处理 | 来源与理由 |
|---|---|---|
| SWIFT-style token reward/gate、trajectory residual | 保留 | `main` 的核心 score 语义 |
| 通用 semantic/style consistency | 保留 | `main` 接口不绑定某一种 rewrite 路线 |
| Hallucination onset BCE 与负 tail reward | 实现与整合默认保留，但分开裁决 | onset BCE 有弱诊断/排名点信号；2026-08-24 gold-tail 的 locality/ranking 门失败，不能因默认 active 就声称有效 |
| condition attention 256 维瓶颈 | 保留 | `main` 修复了 raw hidden width 上的二次参数爆炸 |
| exact prompt/output token ID 对齐 | 移植 | `panzhixin` 最重要的数据正确性经验 |
| embedding + 全部 block hidden states | 移植 | 真实 Phi 数据链已经跑通 |
| feature provenance/checksum 与旧 manifest schema | 精简后移植 | 新产物记录 model/revision/dtype/SHA-256，reader 解析嵌套 metadata，resume 识别旧 checksum 别名 |
| layer-axis Transformer encoder | 移植 | 把 `33×3072` 压到 768，真实配置少于一千万训练参数 |
| BF16 feature 原 dtype 读取 | 移植 | 避免 CPU 侧无意义扩成 FP32 |
| token target 严格等长、缺失标签 mask | 移植 | 禁止静默截断、补零和把未标注 row 当负样本 |
| semantic group sampler | 移植 | 确保 consistency pair 进入同一 batch |
| query-disjoint split | 移植 | 避免同 query 候选跨 train/validation |
| finite loss/gradient、grad clip | 移植 | 基础训练可靠性 |
| 原子 full-state checkpoint 和精确 resume | 精简后移植 | 保留 model/optimizer/RNG/data contract，不搬复杂 run-record 系统 |
| query-level Best-of-N evaluator | 精简后移植 | 固定 candidate prefix、tie 和 bootstrap 语义 |
| Dual-prior direct key/complete supervision | 默认启用 | standalone 与 clean 小 dev 均显示 target learnability；clean ranking 增益未建立 |
| 双向 stop-gradient mutual distillation | 默认启用，权重 `.25` | 历史保护门通过，但 clean P0→P1 没有机制或 ranking 增量，不应再写成 efficacy 证据 |
| shared-gradient gate-prior alignment | 公式保留，默认权重 `.25` | 用户要求保留 main 方法路径；v2 按冻结近优规则选出 `.25` 作为 dev-tuned 工程默认值。排名 efficacy 仍未建立，扩大数据后固定 off/on 复测 |
| sparse-span hallucination | 不迁移到当前核心 | 点估计小门通过，但 onset、blind transfer 和联合门失败；且用户指定回 main |
| online batch-local extraction | 暂不迁移 | 只有小样本等价性，没有大规模吞吐结论；会显著扩大 trainer |
| Strict / Encoded baseline model variants | 不迁移 | 保持 clean 主干单一模型；后续 matched ablation 在独立分支或最小 baseline 中重建，不把多 variant 类塞回核心 |
| annotation/adjudication/protocol/versioned scripts | 只新增单一 smoke-v2 入口 | 不迁移旧分支的大量历史 runner；当前仅保留经双审查冻结的 source/checker/unitizer/proposal/双标/盲裁硬门链路 |

不要整文件复制 `panzhixin` 的模型文件：那会覆盖 `main` 后续增加的 condition bottleneck。当前正确组合是：

```text
exact all-layer features
  → layer-axis encoder (101376 → 768)
  → main-style condition attention bottleneck (768 → 256 → 768)
  → reward / consistency / hallucination / dual-prior heads
```

## `panzhixin` 的真实数据规模

旧分支的磁盘规模很大，但统计监督规模很小，二者不能混为一谈。

| 数据部分 | 历史真实规模 |
|---|---:|
| outcome train | 496 queries × 8 candidates = 3968 trajectories |
| 每 epoch 生成 token | 1,116,541 |
| 5 epochs 每个 cell 的 token exposure | 5,582,705 |
| correctness-only rows | 3866，约 97.43% |
| consistency rows | 54，即 27 对，约 1.36% |
| hallucination + prior rows | 48，约 1.21% |
| 任意 auxiliary rows | 102/3968，约 2.57% |
| H train sparse supervised tokens | 6681：922 positive / 5759 negative |
| prior supervised token units | 14,307 |
| mechanism dev | 16 trajectories；H sparse tokens 2451 |
| ranking validation | 500 queries × 16 candidates = 8000 trajectories |
| ranking validation tokens | 2,315,243 |

全层 BF16 feature 被多轮重复保存后，旧分支本地 artifact 约 833 GB。这个数字主要反映 `[T,101376]` payload 的重复物化，不代表有海量独立训练样本。clean integration 不跟踪这些 feature 或 checkpoint，也不在配置中写机器绝对路径。

已对现有 3968-row manifest 做 clean reader 兼容性 smoke：schema 完整解析，首条真实 trajectory `[221,101376]` 和 condition `[105,101376]` 均以 BF16 读取，`33×3072` 约束一致。这证明旧 artifact 可作为数据接入起点，不证明新 extractor 全量跑通或新配置的训练效果。

## 联合训练的真实负结果

旧分支完成过 seed 42、5 epochs 的 J0/JP/JALL single-stream 矩阵。loss、监督计数、梯度和 checkpoint 工程闭环均成功，但冻结效果门失败。

| Cell | BoN@16 | H span AP | H claim AP | Key AP | Complete AP | C cosine gap |
|---|---:|---:|---:|---:|---:|---:|
| J0 correctness | `.920` | `.169` | `.421` | `.098` | `.277` | 约 0 |
| JP prior | `.918` | `.192` | `.172` | `.432` | `.946` | `.023` |
| JALL 三模块 | `.912` | `.272` | `.289` | `.314` | `.931` | `.789` |

关键解释：

- JALL 相对 J0 的 BoN@16 为 `-.008`，query-paired 95% bootstrap 区间为 `[-.026,+.010]`。没有建立正增益，区间跨 0 也不足以宣称稳定负效应。
- H span `.272` 低于冻结位置基线 `.393`，claim `.289` 低于 `.422`，两门均失败。
- JALL key AP 相对 JP 从 `.432` 降到 `.314`，下降 `.118`，超过允许的 `.05`。
- complete prior 和 consistency 的训练内 geometry 通过，但 consistency gap 只来自训练关系，不能当作 held-out 泛化证据。
- 因扩展门失败，seeds 43/44 没有继续跑。
- 后续 drop-one、supervision packing、condition gradient routing、frozen probe、temporal smoother和 H-v3/v3a 都没有修复核心问题。

因此新分支不能写成“三模块联合有效”。2026-08-24 的 clean 三 seed matched matrix 也已
完成，但 full 相对 C0 为 `-.13` points 且区间跨 0；准确状态是：三模块接口已整合，
部分 auxiliary target 有小样本 learnability，联合或单模块 ranking efficacy 未建立。

## 为什么 Hallucination 回到 `main`

`panzhixin` 的 S1 sparse reviewed-span BCE 在 48/16 小数据上得到 span AP `.416`、claim AP `.464`，点估计略高于位置基线 `.393/.422`；但 bootstrap 区间跨 0，exact onset `±5` 为 `0/6`。后续 mixed-domain blind validation 和 position-control 路线继续失败。

更重要的是，旧联合方案只让 sparse BCE 更新独立 H head 和共享表示：MIL、token reward、absolute/relative/pseudo tail、progress 都为 0，`hallucination_logits` 不进入 scalar score。因此它实现的是“幻觉诊断 head”，不是项目最初的“首错之后降低 reward”。

当前分支按用户裁决恢复 `main` 语义：

```text
hallucination_onset = k
  → H target 在 k 前为 0、从 k 起为 1
  → token_values 从 k 起被监督到负 margin
  → gate-weighted scalar score 的 value path 因此受到影响
```

需要同时保留反面证据：旧分支对 absolute-margin tail 做过多 fold / seed 复核，tail-specific locality 0/3 seeds 通过，存在全局 value shift；relative 和 clean-matched repair 也失败。2026-08-24 的 clean gold-tail 消融再次出现全局 value shift，且 H0→H1 的 BoN@16 三 seed 都回退。因此“回 main”只解释方法身份和为何保留实现；当前 objective 已未通过这轮 locality/ranking 筛选门，不应继续在同一 16-row dev 上调权重。

## 可选与关闭支线的当前裁决

| 支线 | 当前代码状态 | 默认 | 重新开启条件 |
|---|---|---:|---|
| path-level MIL | 保留稳定 log-space noisy-or | `0` | 更大、定义稳定的 path labels；单独 matched ablation |
| pseudo-onset tail | 保留 | `0` | H boundary 在独立数据通过后再启用，避免循环自训练 |
| progress | head 与 loss 保留 | loss `0`，score weight `0` | 有独立于 token advantage 的 target，并明确 reward/progress 分工 |
| gate-prior alignment | 原 shared-gradient 公式保留 | `.25`，开启 | 用户方法身份约束下的 dev-tuned 工程值；扩大 prior/ranking 数据后固定权重做 off/on 独立复测，不再在当前 dev 调参 |
| complete reconstruction | 仅外部 target 接口保留 | `0` | 获得独立 evidence/answer embedding；禁止同 trajectory 自重构 |
| sparse-span H | 未迁移 | 不适用 | 若重开需新实现、独立标签和与 onset-tail 的明确语义比较 |
| relative tail | 未迁移 | 不适用 | 新方法、新 validation，不继续消费旧 16-row dev |
| clean-matched tail | 未迁移 | 不适用 | 先解决优化隔离与 comparator 定义，再重新预注册 |
| H-v3/v3a、probe、smoother | 未迁移 | 不适用 | 不作为当前主线 |
| supervision packing、condition routing | 未迁移 | 不适用 | 只有新证据表明 schedule/routing 是主要瓶颈时才考虑 |

“保留开关”不代表推荐启用，也不代表历史负结果被抹去；“未迁移”也不等于永久否证整个研究假设。

## 核心代码入口

### `configs/best_current.json`

唯一默认配置。真实输入为 `33×3072`，layer encoder 输出 768，condition bottleneck 256。默认 active loss 是 final + consistency + main hallucination onset/tail + direct/mutual dual prior + `.25` main-style gate alignment；MIL、pseudo、progress 和 reconstruction 关闭。

### `src/clir_features.py`

- `IdentityFeatureEncoder`：toy 或已压缩 feature。
- `LayerAxisFeatureEncoder`：当前真实默认；reshape 为 `[B*T,L,D]`，共享投影、layer Transformer、learned-query pooling。
- `build_feature_encoder`：由 `RewardConfig.encoder_type` 选择。

### `src/clir_data.py`

- `CLIRTrajectoryDataset`：读取 inline/path feature，核对 exact output/prompt token 长度和 layer metadata，保留 BF16；兼容旧的嵌套 `feature_metadata` schema。
- `clir_collate`：padding 和每类监督的独立 mask。
- `SemanticGroupBatchSampler`：把同 semantic 的多 style row 放入同一 batch。
- `EpochRandomSampler`：以 `(seed,epoch)` 固定普通 shuffle，供精确 resume。

`query_id` 只用于候选分组与 split；`semantic_id/style_id` 只用于 consistency。不要互相 fallback。

### `src/consistency_localized_reward.py`

- `RewardConfig`：模型结构与 loss 权重。
- `ConsistencyLocalizedReward.forward`：encoder、condition fusion、score 和诊断 heads。
- `ConsistencyLocalizedReward.loss`：按标签存在性和权重路由三模块 loss。
- `prism_style_consistency_loss`：正/负 pair 和 score consistency。
- `hallucination_localization_losses`：当前 main onset-tail 实现。
- `path_level_hallucination_mil` / `pseudo_onset_tail_loss`：保留但默认关闭。
- `dual_prior_losses`：direct、mutual、gate、external reconstruction。

### 顶层命令

- `extract_hidden_states.py`：必填 `query_id` + strict exact integer IDs → all-layer token features；复用同 prompt condition，记录 revision/dtype/checksum，支持受控 `--overwrite`。
- `train_clir.py`：唯一 JSON config、query split、finite checks、原子 full-state checkpoint/resume；resume feature contract 兼容 `feature_sha256` / `condition_sha256` 旧别名。
- `score_clir.py`：默认 batch 2 + BF16，输出 checkpoint SHA-256、scalar/path-clean log/逐 token H-reward-value/prior membership/condition 诊断和每 query Best-of-N 标记；原子写入且默认不覆盖。
- `evaluate_clir.py`：candidate-prefix Best-of-N、bootstrap、pairwise accuracy；默认要求全部 query 满足 max K，仅 `--allow_incomplete_queries` 启用逐 K 过滤，报告记录输入 SHA-256。

### Ranking/H0 v7 原始终态与 v7.4 探索性子集

v7 已生成并核验 1,500-query、每题 16 候选的新排序池，以及经补样后冻结的 800 条 H0
proposal。80 条 smoke 通过全部门后开放 reserve。reserve 首轮因 A 大量兜底和控制失败而
终止；v7.3 一次性修正案又让 GPT-5.6-sol xhigh 与 Claude Opus 5 high 在独立新会话中
对同一批 800 条公开条目完整重标。

v7.3 的 32 个 label shard 全部通过 schema、ID、unit index、原包重建、重复理由和哈希
校验。自然 reserve 的 path agreement 为 `698/720=.96944`、kappa `.94081`，共同
positive 403 条，首错 unit exact agreement `.76923`，A/B controls 都为 `8/8`。但 A/B
self-repeat 只有 `65/72=.90278` 和 `64/72=.88889`，低于预注册 `.95`。最终状态是
`FAIL_H0_V7_RESERVE`；原协议的 final selection 未运行。终止报告 SHA-256 为
`93260683…2c01`。修正案次数已耗尽，禁止第三次重标、混轮、改分母或降门槛。

重复失败的只读拆解是：15 个分歧中 11 个保持 hallucinated path、但 onset 相差 3–43 个
unit；另外 4 个直接在 clean/hallucinated 之间翻转，没有一个只是相邻 unit 的 ±1 偏差。
因此不能用“exact 太严”解释，也不应把原数据改成 ±1 后追认通过。若另开新 H 协议，应
优先机械化坏主张的证据与不可挽回边界，或把监督改成预注册的候选集合/区间目标。

用户之后另行授权从现存数据中挑可用子集。v7.4 把这个决定明确标成 post-hoc exploratory，
不改变、也不覆盖原 `FAIL_H0_V7_RESERVE`：smoke 只留 A/B 精确非低置信共识；reserve
要求 retry A、retry B、原始 B 三路对 path/onset 完全相同，并排除任一 retry 自重复失败
对应的自然行；attempt-1 A 完全不用。严格 eligible 有 642 条，按原 proposal priority
取满 600 条：train 200 positive + 200 clean，dev 100 + 100，均为 distinct query，两个
split 无 query 重叠。选择和独立复算报告均已通过。它们的正确名字是
`silver_posthoc_triple_consensus_h0_v7_4`；没有人工复核，不能称 Gold 或“v7 标注通过”。

执行协议冻结在 `configs/ranking_expansion_v7/h0_experiment_v7_4/protocol.json`。四格
`C0/C1/H0/CH0` 将共享相同 5,168 行训练数据（v6.1 的 4,768 行 + H train 400 行），
只切换 Consistency 和 onset BCE 两个 loss。H1、Path MIL、pseudo-tail、Prior、Full 均
关闭。H dev 200 行只做机制验证。排序侧原始 24,000 行全部保留；为避免 null correctness
进入 BoN，特征清单只取 16/16 candidate 都有明确二值 checker 标签的 892 query、14,272
trajectory，不看 CLIR score 或正负比例。其中 347 query 同时有正负 candidate。H + ranking
selected-only 全层 BF16 原始 tensor 预算是 1,063,973,154,816 bytes（约 990.9 GiB）。

该 v7.4 路线现已完整执行。600 条 H 与 14,272 条 ranking candidate 的全宽 feature
抽取、8-worker 完整性复核、四格 full-width 梯度 preflight 均通过；12 个
`4 cells × 3 seeds × 3 epochs` checkpoint 全部完成并通过 load/finite/epoch/data-hash
检查。训练授权冻结在
`configs/ranking_expansion_v7/h0_experiment_v7_4/training_authorization.json`，没有根据
中途或最终结果改 subset、loss weight、epoch 或 seed。

892-query Best-of-N@16 的三 seed 结果为：

| Cell | mean | seed 42/43/44 | paired delta vs C0 | hierarchical 95% interval |
|---|---:|---:|---:|---:|
| C0 | `.8408` | `.8643/.8632/.7948` | — | — |
| C1 | `.8535` | `.8576/.8543/.8487` | `+.0127` | `[-.0168,+.0523]` |
| H0 | **`.8606`** | `.8576/.8655/.8587` | `+.0198` | `[-.0112,+.0613]` |
| CH0 | `.8558` | `.8520/.8554/.8599` | `+.0149` | `[-.0194,+.0639]` |

random expected/oracle 为 `.8254/.9552`。13,028 个同题 correct-vs-wrong pair 上，
C0/C1/H0/CH0 平均区分率分别为 `.6232/.6612/.6697/.6642`；H0 在三个 seed 都约
`.667–.674`，是最稳定的点估计。另一方面，C0 seed 44 明显回退，所有 C0-relative
hierarchical interval 都跨 0，不能把均值提升写成已确认的 Best-of-N 增益。

`CH0-C1-H0+C0` 的 BoN@16 交互均值为 `-.0176`，逐 seed
`+.0011/-.0011/-.0527`，区间 `[-.0538,+.0105]`。这次仍未看到 C 与 H0 叠加，CH0
也没有超过 H0；但符号并非三 seed 一致，不能升级为“二者天然冲突”的结论。

H dev 的机制结果给出了更直接的“可用范围”：H0 token AUROC/AP/BCE 为
`.8782/.8481/.5488`，path AUROC `.8415`；正路径检出率 `.8267`、clean-no-onset
`.7867`、平衡路径决策 `.8067`。可是首错 start exact 为 `0`，±5 token 只有 `.04`，
条件 onset MAE 约 124–151 token。故这批标签可训练出 tail/path 风险区分器，却没有训练出
可信的精确首错边界；对外应称 `first-bad-unit-derived tail supervision`，不能称精确
first-bad-token detector。CH0 的 token/path AUROC 为 `.8630/.8413`，略低于 H0。

最终状态 `COMPLETE_H0_V7_4_POSTHOC_EXPLORATORY_EVALUATION`；本地汇总 SHA-256
`d80fff82adeeaf84a72e9c867811e90519d52c241da12840213731dd104d291e`。原始
`FAIL_H0_V7_RESERVE` 与全部淘汰行仍保留，结果不覆盖它。

### Dual Prior v8--v12 均终止

Prior 历史可训练数据仍只有 48 条。v8 的依赖图双标和 raw gate 已经完成：eligibility
`60/60`、path agreement `.95`，但 Key/Complete F1=`.7667/.8040`，非低置信 exact derived
target 只有 `8/60`，最低裁决比例 `.8667`，first-flaw exact `.6563`，controls A/B=`4/6,5/6`。
最终状态为 `STOP_PRIOR_DEPENDENCY_SMOKE_V8_RAW_GATE_FAILURE`；raw report SHA-256
`e7e14002…0693`。按冻结协议不发第三模型、不抽 feature、不训练，也不挑 v8 子集救场。

只读诊断解释了为什么另开 v9。v8 的 52 条 Complete mismatch 中 44 条只由闭包边界造成，
37 条还是 A 为 B 的严格子集；两边对自然语言“主线”接近，却很难给出完全相同的边集合。
相反，历史 v3 直接集合标注的 Key exact 为 55/60、unit agreement `.9909`；Complete exact
26/60、unit agreement `.9341`、ambiguous fraction `.0659`、positive intersection/union
`.8503`，且 60/60 行都有非空 Complete 交集。这些只用于冻结 v9 设计，不能回头训练 v3。

v9 在提交 `3331edf542dfd7e836281b290118a88de3a67c5b` 上回到 direct Key/Complete：Key 只有
A/B 双方 usable、非低置信且 exact nonempty set 时整行训练；Complete 双方都选为正、都不选
为负、只一边选的 unit 显式 mask。Key/Complete attention 仍在完整 trajectory 上归一化，
loss 只看 coverage；没有修改 Prior 网络、mutual 或 main 固定 `.25` gate coupling。

本轮从 v6 已 materialize 的 16,000 rows 重新选 60 个全新 query/cluster，排除 v6.1 C 和 v8
全部 query/cluster，并确认与 v7 H/ranking 零重合。四格仍各 15。natural ordered hash 为
`ba0133d0…5852`；A/B 包 78/66 行，ordered hash=`e816f353…71d1` / `fa131ce2…7847`。
状态 `PASS_PRIOR_PARTIAL_SMOKE_V9_PACKAGES_READY`，独立复算
`PASS_PRIOR_PARTIAL_SMOKE_V9_RECOMPUTATION`。

GPT-5.6-sol xhigh 与 Claude Opus 5 high 已完成全部 v9 标签。A/B 文件分别为 78/66 行，
SHA-256=`5733cde9…1873` / `31f4082b…8158`；schema、population、item ID 和包绑定全部通过。
raw evaluator 在提交 `e1f2ea73ae228303164fa6e1427632ee135576fe` 上两次确定性重算得到
同一 report SHA-256 `30695a1a…1f3`，最终状态
`STOP_PRIOR_PARTIAL_SMOKE_V9_RAW_GATE_FAILURE`。

通过项是 eligibility 60/60、common usable 60、common non-low 53、Complete 非空交集 53、
A self-repeat 12/12 和非全集退化。失败项是 Key/Complete F1 `.7778/.7280`、exact non-low
Key 39/60、Key+Complete 同时可训练 39/60、Complete unit agreement `.7891`、ambiguous
`.2109`、positive IoU `.5665`、平均 mask coverage `.7999`，以及 controls A/B=`4/6,5/6`。
完整 joint exact 诊断只有 3/60。

只读诊断显示这不是均匀随机误差。全部 60 条中，B Complete 是 A 严格子集 53 条；在 raw
metric 的 53 条 non-low 行中是 47 条，equal 只有 5 条。A/B 平均 Complete 大小分别约
`10.87/6.03`。Key 全 60 行 exact 42，但 B 有 7 条 low，冻结可训练 exact 只剩 39。
两边还同时把错误链控制题的 Key 从预注册“最早致命错误”选成了后续结论步骤；A 另把一条
冗余输入复述放入 Complete。故 v9 不能用局部 mask 掩盖成成功：协议明确禁止裁决、第三模型、
post-hoc 子集、scale、feature 和训练。远端保留的无标签结果摘要为
`configs/data_expansion_prior_v9/completion.json`。

用户随后批准先优化提示词；如果新口径的语义/控制/稳定性门通过而只差可训练行数，则允许
再冻结一份“大池双标后按预注册规则取严格共识子集”的 scale 协议。v10 不能重标或翻转 v9，
也不能使用已检查的 39 行。提交 `61aeab664ce701d5e4c007b684db9a171480f423` 冻结了
`prepare_clir_prior_v10.py`、统一回溯提示词、纯 evaluator、8 个控制题和双边 self-repeat。

v10 保持 main 的 Key 窄、Complete 宽，但把标注自由度收紧：usable Key 恰好一个；错误主线
取最早致命错误，没有致命错误时取首次完成候选答案的最后非包装步骤。Complete 从最终实质
结论向前做同一套 backward slice；拆开的代入式和求值结果均保留，自包含计算不重复收结果，
题面复述、计划、通用公式、未用旁枝、重复和 final wrapper 排除。模型、direct/mutual loss、
显式 mask 与固定 `.25` main-style gate route 均未修改。

从 v6 的已 materialize 池中选出的 60 条全新自然样本按 GSM8K/MATH × match/mismatch 各 15，
query/cluster 与 v6.1 C、v7 H、v7 ranking、v8 和 v9 均零重合。A/B 各 80 行：60 natural、
8 controls、12 self-repeats。准备与独立复算状态是
`PASS_PRIOR_CANONICAL_SMOKE_V10_PACKAGES_READY` / `PASS_PRIOR_CANONICAL_SMOKE_V10_RECOMPUTATION`；
natural ordered hash=`37ae27fa…ecd9`，A/B package ordered hash=`01642639…4f3` /
`d6c7a5e5…2ee5`。

两份 v10 标签随后完成并通过 schema、population、ID/order、盲包绑定和 singleton-Key 校验。
自然 60 条 eligibility=`60/60`，Key/Complete F1=`.9000/.9253`，exact non-low Key 和 paired
support 都是 `54/60`，Complete nonempty consensus=`60/60`、unit agreement=`.9291`、ambiguous
fraction=`.0709`、positive IoU=`.8462`、mean coverage=`.9350`；A/B self-repeat 都是 `12/12`，
且 Complete 无全集退化。所有自然语义、数量、coverage、repeat 与反退化门因此通过。

唯一失败是 A 在隐藏 `earliest_fatal_error` 控制中漏看 `7+5=13`，选了下游 unit 1 而不是
unit 0。A/B controls=`7/8,8/8`，冻结要求各 `8/8`，所以不是 yield-only，而是
`STOP_PRIOR_CANONICAL_SMOKE_V10_DEFINITION_FAILURE`。raw report 两次重算逐字节相同，
SHA-256=`2ecc2e80…3025`。按原协议不得重标控制、降门、挑 54 行、oversample、抽 feature 或
训练；v10 永久保持终止。

用户随后明确授权“再试一次”。v11 保持 v10 的 Key/Complete 定义、partial-consensus mask、
网络、loss、mutual 与固定 `.25` main-style gate route，仅增加一个 prompt-first verification：
标注者必须先逐 unit 独立验算算术/代数、单位、对象、数量关系和所求量，并判断下游是否只传播
更早错误；错误 rationale 要写出实际反证。自然样本与 8 个 controls 全部换新。v11 选择仍为
四格各 15，但预先再排除 v10 query/cluster；A/B 仍各 80 行并保持 `8/8` controls、`.95`
self-repeat 与全部 v10 raw 门。协议为 `docs/data_expansion_prior_protocol_v11.md`，入口为
`prepare_clir_prior_v11.py`。该授权只解锁新包和双标，不解锁 feature、训练或旧数据救场。

冻结 commit `1b0e35db0e6f2023083241582527b29a2322df9a` 随后正式发布 v11 包并连续两次独立
重算通过。60 条 natural 来自 60 个不同 query 和 60 个不同 cluster，四格各 15；与 v6.1 C、
v7 H、v7 ranking、v8、v9、v10 的 query/cluster overlap 均为 0。natural ordered hash=
`26aec3c9…28e4`；A/B package ordered hash=`6b826261…80fc` / `e042b8e6…468c`，文件 hash=
`5a7899e1…d043` / `7d6d0a1a…26aa`。每包 80 行且 80 个唯一 ID，公开字段仅为 item ID、题目、
回复和 units；无 source/checker/reference/expected-signature 泄露。状态为
`PASS_PRIOR_VERIFIED_SMOKE_V11_PACKAGES_READY` /
`PASS_PRIOR_VERIFIED_SMOKE_V11_RECOMPUTATION`。

GPT-5.6-sol xhigh 与 Claude Opus 5 high 随后完成两份 v11 标签。schema、population、ID、
singleton-Key、controls 与 self-repeat 契约全部通过，纯 evaluator 连续两次产生同一个 report
SHA-256 `0729d9825bafc187a7937d4eddf01eeeae32d5dfea3565908cfdcd3f28a4fa8d`。60 条自然样本里，
eligibility、common usable、common non-low 都是 `60/60`；A/B controls=`8/8,8/8`，
self-repeat=`12/12,12/12`，Complete unit agreement=`.9062`、mask coverage=`.9215`。失败项是
Key macro F1 `.8333 < .90` 与 Complete positive IoU `.7957 < .80`，最终状态固定为
`STOP_PRIOR_VERIFIED_SMOKE_V11_DEFINITION_FAILURE`。v11 的 50 条 exact-Key 共识行、任何长度
子集和全部标签都不可训练。

v11 的只读诊断显示，分歧主要来自错误链中“最早致命错误”与下游错误答案之间的 Key 选择，
以及长 MATH 链里等价文字/等式与重复枚举的 Complete 边界。限制到短题虽然能把点指标抬高，
但这是查看标签后的容易题筛选，不能采用。用户此前批准的 fallback 因此成为 v12：全新大池
双标，再按预注册严格共识取子集；它不重标、不救场、不复用 v8--v11 的任何标签。

v12 的权威规格为 `docs/data_expansion_prior_protocol_v12.md` 与
`configs/data_expansion_prior_v12/protocol.json`，入口为 `prepare_clir_prior_scale_v12.py`。
先在未用过的 GSM8K/MATH train 上冻结 2,000 个 query/cluster（GSM 1,200、MATH 800；
train/dev 1,600/400），每题计划 8 个 Phi rollout，共 16,000 trajectories。只读容量审计已经
得到 2,647 个 fresh cluster representative，query/cluster exclusion overlap 均为 0，且未读
test。materialization 后按题源 × checker × split 冻结选 800 条自然样本交给两模型双标；最终
每格只按生成前 priority 取 exact singleton-Key、非空 Complete 交集的严格共识行，目标 400
train +100 dev。Complete 交集为正、并集外为负、对称差 mask，不改变 Prior 网络、direct/mutual
loss 或 main 固定 `.25` gate。该协议会引入明确的 easy-sample bias，所以即使数据门通过也只
能称 `silver_dual_ai_strict_consensus_prior_v12_no_human_verification`。

pre-rollout 在 clean commit `609f0cb` 冻结并连续两次独立复算通过；rollout 执行器 commit
`686e531` 随后完成全部 40 shard。每 shard 50 query ×8 candidate，合计 2,000 query、16,000
trajectory、5,642,715 output tokens；15,822 条 finish=`stop`，178 条 finish=`length`，后者
只能进审计不能送标。所有 shard 的 prompt token IDs、候选索引、query 顺序、模型/tokenizer
revision、授权和代码 commit 绑定均通过；combined raw SHA-256=
`ce18c0f7cd0222d20450391e47f43fda9b5368a7189ed21b76303729a74e7552`。

CPU-only checker/unitizer 随后完成并独立复算：16,000/16,000 unitization 通过，15,520 条
supervision-eligible；materialized 文件 SHA-256=`3a6c5a88…0058e`。冻结的 800 条自然
proposal 来自 800 个不同 query/cluster，八个 GSM8K/MATH × match/mismatch × train/dev 格
精确满足 `128/32/80/20/192/48/240/60`，文件 SHA-256=`334c7dbf…65bdc`，ordered hash=
`aca74fb1…8090`。proposal 独立复算无差异。

commit `a669e9a4c3a4a1f1410bedfb8ef14a71725a758e` 构造了 A/B 各 16 个公开盲包。每 shard 固定
50 natural +1 hidden control +5 self-repeat=`56` 行；每边共 896 行，所有 repeat 与其 natural
parent 位于不同 shard。公开字段严格只有 schema、item ID、题目、回复和 units；PRIVATE
expected signature 未泄露。package report 状态为 `PASS_PRIOR_V12_BLIND_ANNOTATION_SHARDS_READY`，
独立复算状态为 `PASS_PRIOR_V12_PACKAGE_INDEPENDENT_RECOMPUTE`，report/verification SHA-256=
`ffe02fb1…f452` / `37837dc0…c27e`。冻结时 A/B label 文件均为 0；随后才让 GPT-5.6-sol
xhigh 与 Claude Opus 5 high 在互盲上下文中完成各自 16 shard。

两边随后各完成 16 个 label shard、每 shard 56 行，总计每边 896 行。为了避免看结果后改尺子，
v12 evaluator、CLI 与测试先在 commit `f7e3a9e6d2d09acee84c1e9931437b884660de20` 锁定，
再首次读取标签；完整测试为 `175 passed`。评估器逐 shard 绑定 public package，保持全部 800 条
natural 原始分母，检查 16 controls、80 repeats、八格配额，并只按预冻结 priority 计算候选 500 条
的 Complete IoU、mask coverage 与全 material 退化率。

首次评估与独立复算的 canonical SHA-256 完全相同；raw report SHA-256=
`0dce22ba0f3e60483a66de0e882d9084a8f3aba908e7c85a046fde2b9817d0a1`，verification
SHA-256=`cf68ba49fcda0aad8c742e02a3b3a867d94ba39ad3038528b94c1ab7a4d7f20a`。
最终状态是 `STOP_PRIOR_V12_STRICT_CONSENSUS_DATA_GATE_FAILURE`，失败门如下：

| 门 | 实测 | 冻结要求 |
|---|---:|---:|
| A controls | `11/16=.6875` | `>=15/16` |
| B controls | `16/16=1.0` | `>=15/16` |
| A self-repeat | `51/80=.6375` | `>=.95` |
| B self-repeat | `51/80=.6375` | `>=.95` |
| 固定 500 条 Complete positive IoU | `.7065` | `>=.80` |
| 固定 500 条 Complete mask coverage | `.8709` | `>=.90` |

yield 和反退化门其实通过：800 条自然样本全部双方 usable/non-low；687/800 条满足 exact
singleton-Key 和非空 Complete intersection；八格可用数分别为 `123/31/58/17/172/43/194/49`，
均高于目标 `80/20/40/10/120/30/160/40`；两边在固定 500 条上 Complete=全 material 的比例
都是 0。自然 Complete macro F1=`.7669`，固定 500 条的 IoU/coverage 仍低于门，所以不能通过
“数量够”替代标签稳定性。

只读失败诊断进一步区分了 Key 与 Complete：A/B repeat 的 eligibility 都是 `80/80`，Key exact
分别 `76/80`、`78/80`，但 Complete exact 都只有 `51/80`；A 的 29 个重复分歧里 25 个仅改变
Complete，B 的 29 个里 27 个仅改变 Complete。A 漏掉的 5 个 controls 中，多数也是把题面复述、
拆分算式或错误传播链纳入/排除 Complete 的范围问题；B 虽 controls 全过，Complete repeat 仍同样
失稳。因而当前 blocker 是 canonical Complete 在长自然链上的可重复边界，而不是 query 数或 Key
候选 yield。按冻结协议不发布 500 条 target、不事后挑 687 条、273 条 exact-Complete 或其他子集，
不重标、不加 rollout、不降门、不抽 feature、不训练。

### Prior v13 机械局部审核烟测：schema 门终止

用户授权“试一下，不然再考虑拿 v12 训练”，但这不改写 v12 的终局。只读回放显示，单纯把
unit 投影到较大 block 几乎没有帮助：A Complete repeat `51/80→51/80`，B 仅
`51/80→53/80`。因此 v13 前瞻性改了标注对象：机器安全合并碎片、提示角色、每个 child 最多
提出两个 parent；AI 审核 `main_step/premise/formula/duplicate/wrapper/unused_branch`、局部边、
最终 block 和 raw-unit singleton Key，程序机械回溯得到 Complete。AI 不再直接输出 Complete。

冻结协议是 `configs/data_expansion_prior_v13/protocol.json`，说明在
`docs/data_expansion_prior_protocol_v13.md`，入口为 `prepare_clir_prior_mechanical_v13.py`，
实现 commit 为 `83775b769b550a23efa3b35ff5773c479fafd230`。从 v12 从未送标的 acquisition train
轨迹中确定性抽了 48 条，排除 v12 800 proposal 的全部 query/cluster；八个
GSM8K/MATH × match/mismatch × medium/long 格各 6 条，48 个 query/cluster 全不重复。

A/B 各 4 shard，每 shard 12 natural +2 controls +4 repeats=`18`，每边 72 行；parent/repeat
不共 shard。package/verify 状态分别是 `PASS_PRIOR_V13_FRESH_BLIND_PACKAGES_READY` 与
`PASS_PRIOR_V13_PACKAGE_INDEPENDENT_RECOMPUTE`，report hashes 为 `8de2a666…1422` 与
`abfaad27…fdd`。A 用 GPT-5.6-sol xhigh，B 用 Claude Opus 5 high；两边随后完成全部 144 个
package rows，八个 label 文件的 JSON、18 行计数、唯一 ID 与 package ID 均通过预检。

冻结 evaluator 在第一层 schema 门返回 `FAIL_PRIOR_V13_SCHEMA`。唯一错误为
`b:prior-v13-control-b-07:ineligible audit must leave all structure fields empty`：B 将该控制项标成
`no_auditable_reasoning`，`path/final/key` 也均为空，却保留了 block 0 的 `answer_wrapper` role。
因此 evaluator 没有继续计算 controls、self-repeat、自然样本 Key/final-block、机械 Complete
IoU/coverage、role 或 edge 指标；不能把“只有一个 schema 错误”解释成其余语义门通过。
终止报告 SHA-256 为 `179d10060bbe192b6e0411f0d57523b2a3212abd00881917a7551b1b6add577f`，
`trainable_labels_published=false`。

协议在看标签前已冻结 `no_adjudication_or_relabel_rescue=true` 与“失败后不改 prompt/自适应重标”。
所以即使这一行看起来像可机械删除 role 的格式错误，也不能修改 label 后重跑来把 v13 包装成
通过。v13 到此终止，不启动 v14、不抽 feature、不训练 smoke。以后若用 v12 子集，只能另立
post-hoc exploratory 版本，并明确不构成 v12 或 v13 翻盘。

### Prior v13 max bridge 与候选边 v14-dev：旧结论不变，机械召回问题已缓解

v13 终止后，用户澄清旧 B 实际使用了错误模型，并明确授权两边提高推理强度。GPT-5.6-sol/max
与新版 Opus/max 在两个隔离上下文中重标原公开 4+4 shard，分别写入
`labels_a_max_bridge`/`labels_b_max_bridge`，没有覆盖冻结标签。新版 Opus 精确型号/revision
尚未机器验证，因此本轮只能叫 post-hoc bridge。八个文件各 18 行，JSON、schema、唯一 ID 和
package binding 全过；两边 controls=`8/8,8/8`，self-repeat target/Complete/Key 均为
`16/16`。自然 48 条上 eligibility/path=`1/1`，final-block=`.9375`、Key=`.875`、Complete
F1/IoU/coverage=`.887958/.811093/.909000`，role/原候选 edge agreement=`.853222/.863436`。

剩余主 blocker 已转成旧机械候选召回：A/B 分别在 `.625/.6667` 的自然行使用 `missing_edges`；
43/50 条补边中各有 30 条对应 child 已占满 v13 的两个候选。双方补边的交集为 26、并集为 67，
所以既不能降低 missing-edge 门，也不能把单边自由补边直接并入 Complete。

用户随后授权改进机械候选。为保持 v13 可复现，新增的
`src/clir_prior_edge_candidates_v14.py` 与 `diagnose_clir_prior_edge_candidates_v14.py` 完全旁路
旧 compiler/evaluator：它修正逗号数和简单 LaTeX fraction，区分实际数值生产步骤与后续复述，
按显式 operand/变量保留最近生产者，跳过抢占邻接槽位的标题/计划，并把每个 child 的候选动态
限制为 2--6。post-hoc bridge 回放状态为 `PASS_POSTHOC_EDGE_CANDIDATE_V14_DEV_REPLAY`：共同
补边召回 `26/26=1`，单边召回 A=`38/43=.8837`、B=`48/50=.96`，剩余漏边行率
`.1042/.0417`；候选均值 `23.65→33.46`、总量 `1135→1606`（`1.415x`），最大 6 parent/child。

这一结果只支持“新候选规则值得进入前瞻性 fresh smoke”。它不修改
`FAIL_PRIOR_V13_SCHEMA`，bridge 不是训练数据，也没有重新测新候选上的 A/B `keep/drop`，所以
不能从旧标签机械推导新 Complete 后训练。正式下一步是先冻结独立新协议、排除 v12 proposal
与 v13 的全部 query/cluster、记录 GPT-5.6-sol/max 和新版 Opus/max 的精确身份，然后才生成
新包；在 fresh gate 通过前不抽 feature、不训练 Prior。

### Prior v14 fresh mechanical-recall smoke：Complete 稳定，但依赖召回/Key 失败并终止

上述前瞻性版本现已在干净 commit `5fb35a38f6ba30dccf4863af851f276b1a7136ea`
冻结。代码入口为 `prepare_clir_prior_mechanical_v14.py`，协议为
`configs/data_expansion_prior_v14/protocol.json`，人类可读协议为
`docs/data_expansion_prior_v14_smoke.md`。冻结 commit 同时包含新控制题、两侧 launch prompt、
纯 evaluator 与回归测试；提交前整仓 `248 passed`。

v14 只复用 v13 的安全 block 投影，不复用任何 v13 标签。它将候选边规则冻结为
`clir-prior-mechanical-edge-candidates-v14-frozen-v1`，并从 v12 已 materialize 的 train pool
按新 namespace 取 48 条轨迹。排除集合是 v12 800 proposal + v13 48 proposal，共 848 个
query/cluster；新样本有 48 个不同 query 和 48 个不同 cluster，GSM8K/MATH × numeric
match/mismatch × medium/long 八格各 6 条。proposal file SHA-256=`843c9e06…eba6`。

CPU prepare/independent verify 已分别返回
`PASS_PRIOR_V14_FRESH_BLIND_PACKAGES_READY` 和
`PASS_PRIOR_V14_PACKAGE_INDEPENDENT_RECOMPUTE`。package report SHA-256=`75af81e4…075b6`，
verification SHA-256=`282c51d5…6fc0`，protocol SHA-256=`fc2fa5c1…9028f`。每个 annotator
有 4×18=72 行：48 natural、8 个全新 hidden controls、16 个不与 parent 同 shard 的 repeats；
A/B 总计 144 个 public rows；private index 仍为私有文件，按协议不得发送。

自然样本候选负担为 1,627 edges：每行 min/mean/max=`6/33.8958/81`，每个 child 最多 6 个
parent。预冻结门要求两边 controls 都 8/8、target repeats 至少 15/16、common usable non-low
至少 40、final/Key 至少 `.90/.85`、Complete F1/IoU/coverage 至少 `.90/.80/.90`、role/edge
agreement 至少 `.85`，每侧 missing-edge row rate 至多 `.15`。任何失败都终止 v14，不能重标、
裁决、改 prompt/阈值或挑子集。

用户报告的 GPT-5.6-sol/max 与升级后的 Claude Opus/max 随后在隔离上下文完成全部 4+4 shard。
两边各 72 行均通过 JSON、schema、唯一 ID 与 package binding 预检；冻结 evaluator 只运行一次，
得到 `STOP_PRIOR_V14_MECHANICAL_RECALL_SMOKE`。终止报告 SHA-256=
`5d978b4f8085b110081174fed3f487ccbed605fef7be257c58cb1d7b3ecad23c`。

通过项：A/B self-repeat target/Complete/Key 都是 `16/16`；48 条 natural 全部 common usable
non-low，eligibility/path exact=`1/1`；final-block=`.9167`；Complete F1/IoU/coverage=
`.917764/.857262/.925823`；role/edge decision agreement=`.888262/.899816`；
all-material-union=`0`。失败项：两边都把 `variable_rewrite` hidden control 标错，controls 仅
`7/8,7/8`；Key exact=`.770833 < .85`；missing-edge row rate A/B=
`.4375/.291667`，均高于 `.15`。

这说明 v13 式机械闭包配合较高推理强度，已经把 Complete 边界稳定性推进到冻结门以上；但
v14-dev 在旧 bridge 上看到的候选召回没有在 fresh 样本上复现，当前候选器仍大量漏掉 AI 认为
必要的直接依赖，并连带使 singleton Key 不稳定。`trainable_labels_published=false`，v14 到此
终止：不修控制题、不改 prompt/阈值、不裁决/重标、不挑子集，不启动 scale-v15，不抽 feature，
也不训练。后续若继续 Prior，应先把依赖图更大比例机械化或改变 Key 定义，再用全新
query/cluster 另冻版本；不能再消费 v14 标签来调同一门。

### Prior v12-posthoc exact 子集：direct target 可学，ranking 不增益

用户随后明确选择“V12吧”，授权的不是修复原 v12，而是单独命名的事后探索路线。
实现/协议 commit `e8878061ca65ca43d4be8c35e6d489d4a4ab772a` 保留
`STOP_PRIOR_V12_STRICT_CONSENSUS_DATA_GATE_FAILURE` 与 `FAIL_PRIOR_V13_SCHEMA`，并冻结：
A/B usable、非 low、singleton Key exact、nonempty Complete exact；任一已观测 self-repeat
target 漂移则排除其 natural parent；不补固定 quota、不按 score/训练结果选行。

800 条 natural 中 266 条满足 exact Key+Complete，13 条因已观察 repeat 漂移被排除，得到
253 条（202 train/51 dev）。八个 source × checker × split 格均保留，但这是容易共识样本，
不代表 800 条总体。selected ID hash=`18103bb5…c0515`。Key/Complete 分别有 5,000/27,249
个正 token；完整输出轴其他 token 都是负例，两个 coverage mask 均为全 1，attention 仍对完整
trajectory 归一化。253 trajectory +253 condition 的 exact 33×3072 BF16 feature 全量复验通过，
raw bytes=`21,209,075,712`；finalization report SHA-256=`2f2aed6d…8cd8`。

训练授权 commit `1a75b4c37f3a436a5a969ea03f7fe66f441c2b0d` 只允许 matched R0/P0，
seeds 42/43/44、3 epochs。两边共享 4,170 行：3,968 条历史训练行 +202 条新 Prior train；
历史行本身已有 48 条 Prior target，因此 R0 忽略全部 250 条现存 target，P0 对旧 48 +新 202
使用 direct Key/Complete。两 cell 的唯一配置差是 `prior_weight=0/1`；Consistency、H0/H1、
mutual、gate、MIL、pseudo-tail、progress、reconstruction、Full 全关。六个 checkpoint 均完成、
可加载且 state tensor finite；completion report SHA-256=`b22f303e…7a2`。

51-row Prior dev 的三 seed 均值如下：

| Head/metric | R0 | P0 |
|---|---:|---:|
| Key AUROC | `.4883` | **`.9038`** |
| Key AP / BCE | `.0625 / .6160` | **`.5956 / .1453`** |
| Complete AUROC | `.5274` | **`.9612`** |
| Complete AP / BCE | `.3805 / .6826` | **`.9352 / .2702`** |
| correctness AUROC / BCE | `.8976 / .4909` | `.8829 / .5003` |

这建立了很强的 direct-Prior held-out 可学习性。它没有建立标签客观准确性：51 条同样来自
post-hoc exact 双 AI Silver 子集，且无人工复核。

排序评估协议在任何 scored output 完成前由 commit `18a525e` 冻结，汇总器 commit
`40e5dc5` 又在看到结果前锁定逐行原始字段、checkpoint hash、16-candidate population、finite
score、stable tie、K=1/2/4/8/16 与 10,000 次 paired bootstrap。六份 scalar-only score
全部通过 14,272 rows/892 queries 的逐行验收；summary SHA-256=
`dd8cc22b11e3ff201b316c0c4c1b4a268f710fb9c680106476511335c9ff5bf2`。

| K | R0 mean | P0 mean | P0−R0 |
|---:|---:|---:|---:|
| 1 | `.82735` | `.82735` | `.00000` |
| 2 | `.84865` | `.85127` | `+.00262` |
| 4 | `.857997` | `.85762` | `-.00037` |
| 8 | **`.86173`** | `.84604` | **`-.01570`** |
| 16 | **`.85725`** | `.85538` | `-.00187` |

BoN@16 逐 seed delta=`-.00673/+.00224/-.00112`，fixed-seed query interval
`[-.01308,+.00897]`，hierarchical seed+query interval `[-.01570,+.01121]`，因此主指标
没有增益，也不能确认稳定损害。BoN@8 三 seed 全负，逐 seed为
`-.01794/-.01906/-.01009`；两个区间 `[-.02653,-.00486]` 与
`[-.02952,-.00224]` 均低于 0。within-query pairwise 也从 `.66820` 降到 `.65937`。

后看结果后的描述性 selection audit（不作为预注册 gate）发现，K=16 的 P0/R0 每 seed 有
744/708/691 个 query 选择不同候选，即 `83.4%/79.4%/77.5%`；三 seed 总计 78 次错→对、
83 次对→错、1,982 次换了候选但 correctness 不变。也就是说 direct Prior 通过 shared encoder
确实间接改变了 score，而不是完全没接上；问题是这种变化没有提高顶部排序，K=8 还稳定回退。

最终状态是
`COMPLETE_PRIOR_V12_POSTHOC_EXPLORATORY_R0_P0_RANKING_EVALUATION`。允许声称 direct
Key/Complete 对这个 exact Silver 子集可学；不允许声称 v12/v13 通过、Prior 已提高
Best-of-N、结果是新 protected/confirmatory test，或 mutual/gate/Full 已解锁。也不能继续在
同一 51-row dev 或 892-query ranking 上调 epoch、direct weight、subset 或 gate weight。

### Prior v12-posthoc 固定 `.25` Gate：机制通过，K=16 排名门失败

用户随后明确要求先做 Gate、再另做三模块组合。commit `7afb2f5` 在结果前冻结 P0 对 PG0
的单因素实验：P0 复用上述 checkpoint；PG0 仅把 `gate_prior_weight=0` 改为 `.25`，仍用
4,170 条训练行、250 条 direct Prior 监督、seeds 42/43/44、3 epochs。commit `f0f98cf`
又在三个 checkpoint 和 51-row dev 机制结果完成后、任何 PG0 ranking score 产生前，冻结
892×16 排名验收与汇总规则。

51-row dev 上，Gate-to-fused-Prior squared L2 三 seed 均值从 `.031139` 降到 `.025836`，
逐 seed delta 为 `-.00164/-.01567/+.00140`，满足 mean 降低且 2/3 seed 改善。Key AP 仅降
`.01071`，Complete AP 仅降 `.00018`；PG0 normalized entropy `.87191`、effective-token
fraction `.41419`。因此 alignment learned、Prior protection、anti-collapse 均通过。

排序结果如下：

| K | P0 | PG0 | PG0−P0 |
|---:|---:|---:|---:|
| 1 | `.82735` | `.82735` | `.00000` |
| 2 | `.85127` | `.84903` | `-.00224` |
| 4 | `.85762` | `.85725` | `-.00037` |
| 8 | `.84604` | `.85052` | `+.00448` |
| 16 | `.85538` | `.85015` | `-.00523` |

K=8 三 seed 全正，但 fixed/hierarchical 区间均跨 0。主指标 K=16 三 seed 全负
(`-.00336/-.00785/-.00448`)；fixed-seed query interval=`[-.01196,+.00112]`，
hierarchical interval=`[-.01420,+.00448]`。pairwise `.65937 -> .65907`。K=16 每 seed
分别换了 510/271/351 个候选（`57.2%/30.4%/39.3%`）；错→对为 22/6/11，对→错为
25/13/15，说明 Gate 直接改动了最终 score，但净改动略差。

冻结状态是
`COMPLETE_PRIOR_V12_POSTHOC_FIXED_025_GATE_EXPLORATORY_SCREEN`：允许说 Gate 学到
Prior 对齐；不允许说固定 `.25` 提高了排名。按预注册判据，固定 `.25` 的 standalone
screen 被拒绝。用户仍要求三模块阶段保留 main-style `.25` 路径，因此后续 Full 可以把它
作为固定方法身份测试交互，但不能把本结果写成 Gate 通过，也不能在同一 dev/ranking 上调
Gate 权重。tracked completion 在
`configs/data_expansion_prior_v12/posthoc_v1/gate_v1/completion.json`；ignored dev/ranking
summary SHA-256 分别为 `ca09b9a9…b117` / `4e33d529…c4f9`。

### 三模块扩量 v1：完整八格已完成，Full 未通过收益门

统一 5,370-row train、198-row H dev、49-row Prior dev 已通过独立复核；完整
`C×H×P` 八格使用 seeds 42/43/44、3 epochs，共 24 个 run/72 个 cell-epoch，所有
checkpoint 与 optimizer state 均可加载且有限。这里 H 只表示 H0 onset BCE；P 同时包含
direct Key/Complete 和固定 `.25` 的 main-style Gate。H1、mutual、MIL、pseudo-tail、
progress、reconstruction 均关闭。

query-disjoint Silver 机制集确认三个 target 都能学：C 的 positive-minus-hard-negative
representation separation 主效应 `+.18110`；H token AP 从 U0 `.46482` 到 H `.83822`，
path AUROC `.83764`，但 Full 的 onset ±5-token 命中率仅 `.03333`；P 的 Key/Complete AP
从 U0 `.07211/.37005` 到 P `.59514/.92824`。raw Gate L2 会被 P-off 近 uniform prior 的
量纲混淆；保留原值的同时，另报不改参数的 scale-aware 诊断：对同一个 learned fused prior，
learned Gate 在所有 P-on cell/seed 均优于 uniform（`12/12`）。这只证明内部对齐，不证明排序。

复用 892×16 ranking 的 K=16 cell mean：

| U0 | C | H | P | CH | CP | HP | Full |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `.84155` | `.85800` | `.85538` | `.85725` | **`.86136`** | `.85202` | `.84043` | `.84679` |

主对比 Full−U0=`+.00523`，逐 seed `+.00112/+.01457/.00000`；fixed-seed query 95%
interval=`[-.00710,+.01756]`，hierarchical interval=`[-.01158,+.02242]`。因此冻结裁决为
`EXPLORATORY_INCONCLUSIVE`：Full 没通过收益门，也没触发整体伤害门。Full 每 seed 改了
`807/801/773` 个 query 的候选（约 87%--90%），但错→对与对→错基本抵消，说明模块确实
进入最终 score，只是没有稳定净收益。

最清楚的冲突是 H×P=`-.01962`，三 seed 全负，fixed interval
`[-.02971,-.00972]`、hierarchical interval `[-.03288,-.00635]`；Full−CH=`-.01457`，
两个 interval 也都低于 0。C×P 同样三 seed 全负，但 hierarchical interval 跨 0。CH 是
当前最好的描述性 cell，不过这些单 cell 差值没有各自预注册成确认性检验，不能挑 CH 当作
已经证明的 winner。tracked completion 是
`configs/three_module_expansion_v1/completion.json`；ignored mechanism/ranking summary hash
为 `f7b9b6d…982e2` / `eea7b3a…20ef`。不允许继续在同一 dev/ranking 上调 weight、epoch、
subset、threshold 或 `.25` Gate。

### Prior/Gate 新题归因与单轴调权 v1：一次性确认完成，Full 相对 CH 为 harm

协议 `configs/prior_gate_tuning_v1/protocol.json` 已完整执行。调参与确认各从 rollout 前冻结的
1,300 个 query 中按 checker-only hash 顺序选出 800 个（720 GSM8K +80 MATH），每题 16 个候选；
两边 query/cluster overlap 均为 0。最终 full-width selected-only feature 共 25,600 条 trajectory，
约 1.408 TB；八个独立 verifier 对 1,600 个 query marker、tensor SHA/shape/BF16/finiteness 全部通过。
feature completion hash=`34c9dd05…de4`，tuning input hash=`94be8c01…7fd`，confirmation sealed
input hash=`71a1dbe1…681`。

Stage A 使用 CH、direct-P/Gate0、Full(.25) 三格×三 seed。K=16 上 CH=`.95958`、direct-Gate0=`.95458`、
Full=`.95958`；direct effect=`-.00500`，Gate-given-direct=`+.00500`，Full−CH=`.00000`。所以 Gate
没有拖累结果，反而补偿了 direct Key/Complete 监督的负项；冻结规则只开放 direct 权重，固定 Gate
`.25`。Stage-A attribution hash=`bc015bb6…dcc3`。

direct `{.25,.5,1}` 的开放调参 K=16 均值分别为 `.95250/.95375/.95958`。`.25−1` 三 seed 全负且
fixed interval=`[-.01292,-.00167]`；`.5−1` 均值 `-.00583`，fixed interval=`[-.01125,-.00083]`。
减弱 direct 监督没有修复组合，按预注册最大均值规则锁定 `direct=1、Gate=.25`；其调参均值与 CH
完全相同，没有 Prior 候选超过 CH。selection hash=`3d773862…fc9`；12-checkpoint Git 锁文件为
`configs/prior_gate_tuning_v1/confirmation_lock.json`，hash=`3fbd59a9…273`。

锁定后只打开一次 800-query confirmation。K=16：Full=`.93917`、CH=`.94458`、U0=`.92708`。
主对比 Full−CH=`-.00542`，逐 seed `+.00375/-.01125/-.00875`，fixed interval
`[-.01375,+.00250]`，hierarchical interval `[-.01750,+.00625]`。区间跨 0，效应大小仍不精确；但
预注册 harm 规则只要求均值负且至少两个 seed 负，因此裁决为 `CONFIRMATION_HARM`。次要对比
Full−U0=`+.01208`，三 seed 全正，fixed interval `[+.00083,+.02333]`；说明 Full 强于纯 U0，
但 Prior 加到 CH 上会拉低排名。候选和 `full_025` 是同一 checkpoint 别名，三 seed 输出文件逐字节
一致，机械零对比也通过。summary hash=`5fdaa598…a77`，tracked terminal record 为
`configs/prior_gate_tuning_v1/completion.json`。

当前训练数据的 ranking 推荐是 CH；`.25` Gate 仍按用户要求保留为 main-style 工程默认，但不能写成
Full efficacy 结论。确认后禁止在 tuning/confirmation 1,600 题上再改 direct、Gate、epoch、subset
或阈值。若继续 Prior，先扩充/改善 Prior Silver 监督并诊断 H0×Prior 交互，再预注册全新训练与排名
population；不要继续在当前网格上加小数点权重。

## 已知限制

- smoke-v2 因 checker 假阴性、H positive yield 与 Prior stability 失败；v3 readiness 虽通过，但双标后因
  C raw agreement/裁决比例和 Prior 裁决比例失败。v3 的 H raw 门全部通过，只支持“当前定义可稳定操作”
  的诊断，不可单独绕过 all-task finalizer。两版都没有人工/专家复核或可训练 final manifest。
- extractor 现在会把 `--revision` 传给模型加载器，并写 model/revision/dtype 与 feature checksum；但本地模型若无 config commit 且未显式传 revision，`feature_revision` 仍可为 null。resume data contract 会使用 checksum 字段，不会每次重新 hash 巨大 feature 来验证 manifest 中的声明。
- extraction 虽原子发布单个 tensor 和最终 manifest，`--overwrite` 中途失败仍可在旧 manifest 下留下部分新 feature；正式运行应使用新目录而不是就地覆盖。
- 只支持预抽取 feature 训练，全层 payload 存储昂贵；online extraction 尚未进入 clean trainer。
- 当前模型用 pointwise correctness BCE，没有 pairwise/listwise ranking objective。
- clean 已完成历史 7-cell/CH0 以及扩量三模块完整八格三-seed matched evaluation；这没有
  使 `best_current` 成为 efficacy winner，也没有提供新的 protected ranking test。
- consistency 已有400个训练正对、150个 held-out 正对和150个 held-out hard negative 的
  三 seed 机制复测；均值 separation 与 score-gap 结构改善，但正对 cosine 下降、AUROC
  seed 方向混合。三模块排序仍复用同一 892-query exploratory population，不是新 population。
- 历史 H 证据来自很少的 Silver trajectory；v7 双 AI 的盲重复稳定性未过原门。v7.4 的
  600 条是用户授权的事后严格共识探索子集，不是原协议成功，也没有人工准确率保证。
  新结果支持 tail/path 风险可学和正向排序点信号，但 exact onset 为 0、±5 只有 4%，
  Best-of-N 增益区间跨 0；恢复的 main gold-tail 也仍未通过 locality/ranking 门。
- dual prior 的原协议可训练账本仍只有历史 48 条，v8--v13 都保持 frozen failure；但用户
  另行授权的 v12-posthoc exact 路线现在新增 202 train +51 dev。它证明 direct target 可学，
  不证明标签总体稳定或客观准确；复用 ranking 上 BoN@16 无增益、BoN@8 三 seed 一致回退。
  固定 `.25` Gate 在 standalone 路线上学到 Prior 对齐，但 K=16 三 seed 全负。扩量 Full
  已执行，却只比 U0 高 `.52` point 且区间跨 0；H×P 是当前最稳定的负交互。mutual 仍未建立。
  v13 仍因 B 的一个不可用控制项保留非空 block role 而在 schema 门终止，没有自然样本机制指标。
  新的 800-query 调参显示 direct `.25/.5` 都差于 `1`；独立 800-query confirmation 上 Full−CH
  均值 `-.54` point、两 seed 负，触发冻结 harm 规则。这个结果确认的是当前训练清单下的组合问题，
  不能倒推全部 Prior 构念无效，也不能替代更多高质量 Prior 训练标签。
- gate-prior 现在默认 `.25`，只在 row 同时具有 key/complete coverage 时计算；progress、reconstruction 等权重为 0 时，对应 loss/value 路由会直接跳过，不通过 `0×NaN` 污染 score 或 total。
- score 中始终输出 pseudo onset 和 path probability；这不表示 MIL/pseudo-tail 训练已经打开。
- resume 的相同设备 CPU 测试为 bit-exact；不要假设跨设备、跨 PyTorch/CUDA 版本也逐 bit 相同。
- checkpoint 已写 code commit/branch/dirty-worktree、完整命令和 Python/PyTorch/CUDA/device；上游标签/checker一致性、protected test 和 baseline completeness 仍不满足正式论文级协议。
- trainer 的 feature reference 会绑定 path、size、mtime 与 manifest 内 checksum 声明，但不会在每次训练前重 hash 数百 GiB payload；必须先确认 durable exhaustive mirror-verification report。
- clean evaluator 与新增 summarizer 已能完成跨 variants/seeds 的 parity 检查和 paired contrast；本地大 scored/checkpoint artifact 默认不进 Git，正式发布仍需独立 artifact manifest/存储。

## 下一步

1. 保留 v7 的 24,000 条排序候选、800 条 proposal、两轮原始标签和终止报告，不覆盖或
   重标；v7.4 子集必须始终与原失败证据并存。
2. v7.4 的 feature、preflight、12 次训练、H dev、892-query BoN、347-query 配对区分力和
   交互汇总均已完成；保留其 authorization、manifest/report hash 与本地大 artifact，不再在
   这 200 条 dev 或 892 题 ranking 上调阈值、subset、loss、epoch 或权重。
3. 下一轮若要确认 H0，应重新预注册并采一批独立 H train/dev 与 ranking population，优先
   把目标表述为 tail/path 风险；若研究目标仍要求精确 onset，先修改/验证 label target 与
   解码方法，而不是把本轮 0% exact 用调阈值包装成成功。
4. Prior v8--v14 都必须保持终止状态。保存 v12 的 32 个标签 shard、raw report 和独立复算，
   不发布其 prospective 400 train +100 dev、不重标 controls/repeats、不改原门；v13 不删除错误
   control role、不覆盖原标签或原报告。后来另行授权的 max bridge 与 v14-dev 候选回放只能作
   post-hoc 开发证据；v14 fresh smoke 的 Complete 门虽通过，但 controls、Key 与 missing-edge
   门失败，不能补标签、挑子集或启动 scale-v15。若继续，必须先改变依赖图/Key 的机械定义并
   另冻全新 query/cluster 协议，不能把 bridge 写成 v13 pass 或把 v14 写成可训练。
   v12-posthoc 253-row manifest、506 个 feature、六个 checkpoint、dev/ranking summary 与原失败
   证据并存，不能重命名为原 v12 pass。
5. 保留已完成的 P0/固定 `.25` PG0 三 seed 实验、机制报告、ranking scores 和 completion hash；
   它通过 Gate/Prior 对齐门，却在 K=16 三 seed 全负。不得在这 51 条 Prior dev 或 892 题复用
   ranking 上再调 direct weight、Gate 权重、epoch 或 subset。
6. 三模块扩量 v1 已完成 24 个训练、三类机制评估和 892-query ranking；保存全部 authorization、
   checkpoint、shard、merge、summary 和 completion hash，不覆盖、不重跑、不在同一数据上调参。
   Prior/Gate 新题协议 v1 也已完整结束；保留 1,600-query tuning/confirmation、1.408 TB feature、
   九个调权 checkpoint、锁文件、score merge 和 terminal completion，不得再次打开确认或二次选权。
   下一轮默认以 CH 为当前 ranking 推荐；若仍研究 Prior，先扩充/改善 Silver supervision，并把
   H0×Prior 交互诊断和新的 query/cluster-disjoint ranking population 一起预注册。

当前裁决是：C、H0、direct Prior/Gate 都建立了各自 Silver 机制可学习性；H0 更像 tail/path
风险头而不是精确首错定位器。完整组合没有把机制收益相加：新的独立 confirmation 上 CH
`94.458%`、Full `93.917%`，Full−CH 按冻结规则为 harm；Full 仍比 U0 高 `1.208` points。
因此当前训练数据的 ranking 推荐是 CH，`.25` Gate 只是保留的 main-style 工程默认，不是 Full
efficacy 结论。H1 与 mutual 没有因本轮获得新证据，原 v7/v12/v13 也仍不能写成通过。

任何后续结果都应把三件事分开报告：工程闭环是否运行、auxiliary target 是否可学、是否真正改善 held-out Best-of-N。三者不能互相替代。
