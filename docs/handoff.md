# CLIR clean integration 交接说明

本分支从 `main` 的十余文件结构重新出发，只整合 `panzhixin` 分支中可复用的工程进展和通过相应门的模块部分。它不是对 `panzhixin` 的压缩复制，也没有继承其大量标注流水账、版本化 runner 或失败实现。当前远端顶端只保留训练/打分/评测代码、测试、可运行配置、README、本 handoff、核心方法说明、关键 smoke 协议和当前 v6 扩容主协议；阶段报告、PDF、v1 协议及审查过程材料的最后完整快照是提交 `596a5e4`，需要时可从 Git 历史恢复。Consistency-v5 的 rollout、双盲包和标签继续只留在被忽略的 `run_artifacts/`；2026-08-26 重新生成的阶段 PDF 也只作为本地产物交付，不进入精简远端。

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

## 2026-08-26/29 数据扩容主协议 v6/v6.1：关系清单 PASS，抽特征仍待授权

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
推进到“已发布并复核关系清单”，不是可学习性或 Best-of-N 证据。plan 和 verifier 都明确
`feature_extraction_allowed=false`、`training_allowed=false`；下一步需单独授权只抽 inventory，禁止抽全16,000 条。

### Consistency v6.1 selected-inventory feature extraction：已授权，待正式执行

用户在关系/inventory 独立复核完成后明确回复“ok，接着做”，授权只推进 exact feature extraction。机器记录
[`../configs/data_expansion_scale_v6/feature_extraction_authorization_v6_1.json`](../configs/data_expansion_scale_v6/feature_extraction_authorization_v6_1.json)
绑定 commit `a96b8441` 上的 v6.1 plan/verifier、三份关系、1,357-row inventory 与16,000-row materialized file；
范围只含 selected rows join、一个全宽 preflight、inventory-only 抽取、逐文件独立复核和最终 feature manifest。
完整16,000条抽取、重 rollout、改标签/关系/阈值以及训练都是 false。

新入口 `extract_clir_scale_features.py` 不改变通用 exact-ID 数学定义，只补正式规模所需的恢复与审计层：按 query
的总 feature token 做固定 largest-first 八路均衡；同题所有 view 在同一 GPU worker；tensor 与 query marker
分别原子发布；无 marker 的现有 payload 必须 reload 验证后才续用；8个 writer 全通过后，再由8个 CPU verifier
逐个重读1,969个 payload，检查 shape、BF16、contiguous、finiteness 和 SHA-256。最终预期 raw tensor bytes 必须
精确等于 `105451923456`，且报告继续 `training_allowed=false`。完整执行规则见
[`feature_extraction_protocol_v6_1.md`](feature_extraction_protocol_v6_1.md)。本段在执行前只表示“授权与实现冻结”，
不表示 feature 已经完成；只有 finalize PASS 才能改状态。

用户报告全部 shard 完成后，
[`../configs/data_expansion_scale_v6/post_annotation_authorization.json`](../configs/data_expansion_scale_v6/post_annotation_authorization.json)
（SHA-256 `7dcd096a…ab34`）只解锁 deterministic label audit、条件式 400/150 选择和 frozen hard-negative
feasibility；第三模型、裁决、修标签/阈值、feature extraction 和训练仍全部关闭。

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
| evaluator | 工程兼容，已补多 seed 配对 | frozen prefix、stable tie、common population、random/oracle/pairwise aggregate、机制诊断及 parity-checked multi-seed paired summary 已有；formal 仍缺 protected test、held-out consistency evaluator 和完整 baseline 矩阵 |

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

## 已知限制

- smoke-v2 因 checker 假阴性、H positive yield 与 Prior stability 失败；v3 readiness 虽通过，但双标后因
  C raw agreement/裁决比例和 Prior 裁决比例失败。v3 的 H raw 门全部通过，只支持“当前定义可稳定操作”
  的诊断，不可单独绕过 all-task finalizer。两版都没有人工/专家复核或可训练 final manifest。
- extractor 现在会把 `--revision` 传给模型加载器，并写 model/revision/dtype 与 feature checksum；但本地模型若无 config commit 且未显式传 revision，`feature_revision` 仍可为 null。resume data contract 会使用 checksum 字段，不会每次重新 hash 巨大 feature 来验证 manifest 中的声明。
- extraction 虽原子发布单个 tensor 和最终 manifest，`--overwrite` 中途失败仍可在旧 manifest 下留下部分新 feature；正式运行应使用新目录而不是就地覆盖。
- 只支持预抽取 feature 训练，全层 payload 存储昂贵；online extraction 尚未进入 clean trainer。
- 当前模型用 pointwise correctness BCE，没有 pairwise/listwise ranking objective。
- clean 已在历史 3968-row 数据上完成 7-cell 主矩阵和 CH0 二因子补测、三 seed matched evaluation；它没有扩大独立机制样本，也没有使 `best_current` 成为 efficacy winner。
- consistency 证据只有 27 对且没有 held-out relations。
- H 证据来自很少的 Silver trajectory，首错边界一致性弱；clean onset ±5 仍为 0，恢复的 main gold-tail 再次未通过 locality/ranking 门。
- dual prior 只有 48 条历史 Key/Complete 标注 trajectory；clean direct target 可学，但 mutual 增量与 ranking improvement 都未建立。
- gate-prior 现在默认 `.25`，只在 row 同时具有 key/complete coverage 时计算；progress、reconstruction 等权重为 0 时，对应 loss/value 路由会直接跳过，不通过 `0×NaN` 污染 score 或 total。
- score 中始终输出 pseudo onset 和 path probability；这不表示 MIL/pseudo-tail 训练已经打开。
- resume 的相同设备 CPU 测试为 bit-exact；不要假设跨设备、跨 PyTorch/CUDA 版本也逐 bit 相同。
- checkpoint 已写 code commit/branch/dirty-worktree、完整命令和 Python/PyTorch/CUDA/device；上游标签/checker一致性、protected test 和 baseline completeness 仍不满足正式论文级协议。
- trainer 的 feature reference 会绑定 path、size、mtime 与 manifest 内 checksum 声明，但不会在每次训练前重 hash 数百 GiB payload；必须先确认 durable exhaustive mirror-verification report。
- clean evaluator 与新增 summarizer 已能完成跨 variants/seeds 的 parity 检查和 paired contrast；本地大 scored/checkpoint artifact 默认不进 Git，正式发布仍需独立 artifact manifest/存储。

## 下一步

1. 关闭 v3 后续消费：不发送第三模型包、不裁决、不 finalize、不抽特征、不训练；保留 H 通过和 C/Prior
   失败作为 pipeline-smoke 诊断。
2. C prompt-v4 已失败；v5 机械筛选/双盲事实审计通过；v6 raw gate 通过但旧 hard-negative 仅8/150；用户批准的
   v6.1 已在不改阈值和正关系的前提下发布并独立复核400 train positives、150 heldout positives、150 heldout
   hard negatives 及1,357-view inventory。下一步不是继续改负例或直接训练，而是另发 exact selected-view
   feature-extraction 授权；只抽 inventory，完成逐文件 shape/BF16/finiteness/hash 核验后再申请 C-only 训练。
3. H 若要进入训练，先用新 query 做独立 H-only confirmation 与第三模型稳定性审计；不得把已看过结果的
   v3 H 标签重新包装成确认性通过。新协议全部 raw/final 门过后才发布 `pre_extraction.jsonl`。
4. Prior 先用 40–60 条全新轨迹标显式 dependency edges，由确定性传递闭包得到 Complete、再选 Key；只有
   exact-set 分歧和裁决负担降下来才扩为 train 300–500/dev 100–200。H 通过后目标仍是 train
   200+200/dev 100+100；统一 checker 的 1500–2000 independent queries ×16 ranking validation 另发协议。
5. 用新数据完整重跑 `C0/C1/H0/CH0`；H 先过 boundary/calibration，当前 gold-tail、MIL、pseudo-tail 不进
   核心矩阵。prior 先复核 direct target，再在新 ranking population 上固定 `.25` 做 gate off/on，不再调权重。
   若要归因 encoder，补 strict/encoded SWIFT 等预算 baseline。

任何后续结果都应把三件事分开报告：工程闭环是否运行、auxiliary target 是否可学、是否真正改善 held-out Best-of-N。三者不能互相替代。
