# CLIR：Consistency-Localized Intrinsic Rewards

CLIR 是一个自包含的 hidden-state reward model 研究实现。它参考 SWIFT 的 token reward / gate 聚合方式，在 frozen LLM 的生成 token hidden states 上加入三类监督：语义一致性、幻觉起点后的局部负奖励，以及 key/complete 双先验定位。仓库不依赖 SWIFT 源码。

当前分支的目标是提供一个小而可运行的研究主干：保留 `main` 的方法语义，吸收 `panzhixin` 分支中已经证明有工程价值的 exact-token 数据链路、全层特征压缩、严格 mask、可恢复训练和查询级评估；历史协议、标注流水账和失败实验实现不进入核心目录。

需要先明确证据边界：[`configs/best_current.json`](configs/best_current.json) 是当前唯一的**整合配置**，不是已经证明优于 correctness-only baseline 的“最优效果配置”。三模块联合训练的历史结果没有通过扩展门，详见 [`docs/handoff.md`](docs/handoff.md)。

## 2026-08-23 clean integration 审计与训练试跑

本轮在 `clir-clean-integration` 的 `8b116c4` 上完成了代码、旧 artifact 兼容和真实训练审计，并修复了一个 CUDA 续训回归：full-state checkpoint 现在先加载到 CPU，再恢复 model/optimizer/RNG；若把整个 checkpoint 映射到 CUDA，CPU RNG state 会被搬到 GPU，`torch.set_rng_state` 会直接失败。修复后固定 SWIFT Python 环境的完整测试为 `37 passed`，CUDA 上的 interrupted→resume 回归也通过。

工程与数据门结果：

- toy generate→epoch 1→resume 到 epoch 2→score→evaluate 闭环完成；toy total loss 从 `4.5132` 降到 `4.2594`。这是随机数据的代码路径证据。
- clean reader 成功接入旧 3968-row train manifest（496 queries × 8 candidates）与 8000-row validation manifest（500 × 16），train/validation `query_id` 无交集；全部引用路径存在，代表性 trajectory/condition 为 BF16 `[221,101376]` / `[105,101376]`。
- 默认配置的真实模型有 `5,347,593` 个训练参数；全宽 consistency batch 和 hallucination/prior batch 的前向、反向与梯度均 finite，实测峰值显存约 `2.94 GiB`。
- seed 42、默认整合配置、3968-row train、query-disjoint 16-row mechanism dev 的 1 epoch 真实试跑完成：train total `.5374`，mechanism-dev total `3.6609`。checkpoint 位于 `run_artifacts/clean_integration_audit_20260823/real_full_seed42_epoch1.pt`，SHA-256 为 `e1dba08f91d6529213db1acadfc274a422a76c7c8d096a74da2576290c7c891f`。

同一 checkpoint 在 checker `clir_gsm8k_numeric_v4` 的 500×16 validation pool 上得到：

| K | reward BoN | random expected | oracle |
|---:|---:|---:|---:|
| 1 | `.884` | `.884` | `.884` |
| 2 | `.898` | `.892` | `.936` |
| 4 | `.910` | `.888` | `.956` |
| 8 | `.910` | `.89075` | `.970` |
| 16 | `.906` | `.8925` | `.976` |

query 内 correct-vs-wrong pairwise accuracy 为 `.6241`（5076 comparisons）。BoN@16 相对 random expected 的 paired query delta 为 `+.0135`，10,000 次 query bootstrap 95% 区间 `[-.0045,+.03175]`，跨 0。正确结论是：clean integration 已建立 `small-scale real pipeline pilot` 和弱排序信号；它只有 1 epoch/1 seed，且没有 matched clean correctness-only baseline，不能把结果归因给 consistency、hallucination 或 dual prior，也不能声称三模块联合有效。

## 2026-08-24 clean ablation v1

[`configs/clean_ablation_v1`](configs/clean_ablation_v1) 的 7-cell × 3-seed × 3-epoch
matched matrix 已全部完成：correctness-only、`+C`、H BCE only、H BCE+gold tail、
direct prior、direct+mutual prior 和 full 共 21 个 run。所有 run 使用同一 496-query
train、16-row mechanism dev 和 500×16 ranking population；checkpoint code/environment
provenance、候选 parity、checkpoint hash 与 scored-input hash 均通过。

BoN@16 三 seed 均值为：C0 `.9173`、C1 `.9220`、H0 `.9267`、H1 `.9187`、P0
`.9180`、P1 `.9180`、full `.9160`。相对 C0 的 paired delta 中，C1 为 `+.47`
points，H0 为 `+.93` points，P1 为 `+.07` points，full 为 `-.13` points；所有
query-bootstrap 区间都跨 0。H0→H1 的 gold-tail 增量在三个 seed 都回退，均值
`-.80` points；机制诊断同时显示 tail 没有形成 onset-localized value drop，而是把全局
token value 推到约 `-.62`。direct priors 在 16-row dev 上可学，但 mutual 没有额外
机制或 ranking 收益；consistency 因没有 held-out relation set 仍无法验证关系泛化。

3→5 epoch 的预注册扩展门未通过：多个 auxiliary cell 的 train loss 下降时 mechanism-dev
反而恶化，继续训练只会重复 27 个 consistency pair、17 个正 onset + 31 个 clean 和 48
条 prior trajectory。关键数字、置信区间和裁决已收敛在本 README 与
[`docs/handoff.md`](docs/handoff.md)；完整历史结果可从归档提交 `596a5e4` 恢复。这组结果仍是
`small-scale real screening`；不能声称三模块联合有效。

### CH0：consistency 与 onset BCE 的二因子交互补测

为避免把 H tail 和 dual prior 混入交互判断，本轮在查看新指标前冻结并运行了
`CH0 = correctness + consistency + onset BCE`，仍用同一 train/dev/ranking population、
3 epochs 和 seeds 42/43/44。CH0 的 BoN@16 为 `.9153 ± .0042`，逐 seed 为
`.920/.912/.914`；相对 C0、C1、H0 分别为 `-.20/-.67/-1.13` points。H0→CH0 三个
seed 都下降，fixed-seed query bootstrap 为 `[-2.20,-.13]` points；但只有三个训练 seed，
seed+query hierarchical interval 仍为 `[-2.73,+.47]` points，不能升级为正式稳定负效应。

二因子交互 `CH0 - C1 - H0 + C0` 为 `-1.60` points，三个 seed 都为负；当前数据支持的
筛选结论是 consistency 和 onset BCE **没有叠加，且有负交互信号**。这不等于二者天然
不兼容：consistency 只有 27 个正 pair、H 只有 17 个正 onset + 31 个 clean，机制 dev
只有 6+10 条，ranking 只有 500 queries。下一轮应扩数据后按同一 `C0/C1/H0/CH0` 2×2
矩阵复测，而不是直接把本轮写成最终论文结论；`best_current` 也不因这次筛选而更换。

三个模块的实现路径、最终分数耦合、历史数据生产流程、单模块效果和组合现象已合并到本
README 与 [`docs/handoff.md`](docs/handoff.md)。为保持远端分支精简，阶段报告、图解版和 PDF
不再放在当前分支顶端；需要追溯时可从提交 `596a5e4` 读取原文件。Gold/Silver/Pseudo、
clean/未标/pseudo onset、Key/Complete 和 gate `.25` 的含义与证据边界仍由 handoff 保留。

### Prior→reward gate 独立消融

[`configs/clean_gate_ablation_v1`](configs/clean_gate_ablation_v1) 已完成 P0 direct-prior 与
PG0 direct-prior+gate 的 2-cell × 3-seed × 3-epoch 对比。PG0 使用 `.0625`，等于
`origin/main` 的总有效系数 `.25×.25`；没有加入 mutual、C、H 或 Full 混杂项。

工程路径和 prior protection 均通过，但机制与 ranking 未通过：gate↔fused-prior
squared-L2 从 `.01195` 恶化到 `.01335`，只有 1/3 seed 改善；BoN@16 从 `.9180`
变为 `.9167`，paired delta `-.13` points，fixed-seed query interval
`[-.87,+.60]` points，seed+query interval `[-1.20,+1.00]` points。gate 虽使
`53%–76%` 的 query 更换最终候选，但三 seed 合计净少选对 2/1500 次。这仍是 `.0625`
这个单点的有效反证；关键历史结果保留在 [`docs/handoff.md`](docs/handoff.md)，完整旧文档可从
提交 `596a5e4` 恢复。

用户随后明确将“保留 `main` 原始 shared-gradient coupling 且默认开启”定为方法身份约束，
并允许在当前开发 population 上选择一个保守强度。为此在查看新结果前冻结
[`configs/clean_gate_tuning_v2`](configs/clean_gate_tuning_v2) 的 `.25/1/4/10` 网格，并
复用 `0/.0625` anchors，完成 6-weight × 3-seed × 3-epoch 严格配对评测。所有正权重通过
机制健康门；`10` 的 BoN@16 点估计最高 `.9207`，`.25` 为 `.9187`，差值恰好是冻结的
near-tie 边界 `.002`，因此按“近似时选更小权重”规则固定 `.25`。

`best_current` 和 `RewardConfig` 现已默认启用 `.25`。它保持 `origin/main` 的内部系数、
MSE、detach、mask 与 shared-gradient 路径，但 clean 的外层 `prior_weight=1`，所以总 loss
中的绝对 coupling 系数是 `.25`。`.25−P0` 的 BoN@16 只有 `+.07` point，两个配对区间都
跨 0；这是使用同一 dev 选出的 **dev-tuned engineering default**，不是 gate efficacy
结论。选择规则、结果摘要和证据边界保留在对应训练配置与
[`docs/handoff.md`](docs/handoff.md)。扩大数据后将
固定 `.25` 重做独立 `off/on` 诊断，而不在当前 dev 上继续调参。

## 2026-08-25 多题源扩量 smoke v2：双标完成，按预注册规则停止

v1 收到两份互盲外部 AI 审查后被共同判为 block，且从未执行，状态是
`superseded_before_execution`。v1 原文、审查提示词和逐项裁决已从当前分支顶端移出；最后一份
完整快照是提交 `596a5e4`。当前执行只认 v2，不应再从 v1 衍生实现。

修订后的可执行规格是
[`docs/data_expansion_smoke_protocol_v2.md`](docs/data_expansion_smoke_protocol_v2.md)，机器可读配置为
[`configs/data_expansion_smoke_v2/protocol.json`](configs/data_expansion_smoke_v2/protocol.json)。流水线、checker
v2、unitizer v2、C/H/P proposal、隐藏控制项、自一致性、第三模型先独立后匿名裁决和 fail-closed
finalizer 已实现；8-query/64-row 确定性 fixture 已通过。

真实 acquisition 与自然 proposal 冻结已经完成。本地按固定 revision 导出 `7473` 条 GSM8K train 和
`1218` 条 ASDiv-A，共 `8691` 条；从旧 outcome/ranking/mechanism population 汇总并规范化出 `1108` 个
不可复用的 GSM8K query ID。冻结检索器找到 29 对 near-duplicate candidate，其中 2 对的两端都已在历史
排除表内；实际送标的 27 对由用户报告的 OpenAI `gpt-5.5-sol`（xhigh）与 Anthropic
`claude-opus-5`（high）互盲判断。两者是不同、非 Phi 模型系列，target `27/27` 一致：`7` 对
duplicate、`20` 对 distinct，Cohen κ=`1.0`，无需第三模型。产品界面没有暴露精确 revision 与
temperature，因此本地 roster 明确记录为 unknown/unverified；这不能冒充预注册的 temperature=`0`。

去重决定随后冻结出 100 个永久 train-only query（60 GSM8K +40 ASDiv-A），与 1108 个历史排除项交集为
0；选中 query ID 集合 SHA-256 为 `08fc850d…df78b`。在一张 L20Z 上用固定 Phi-3.5 revision、
`vllm==0.5.3.post1`、TP=1、BF16、`max_num_seqs=32` 完成 100×8=800 条 rollout：800 条均为正常 stop，
没有空输出或长度截断，原始有序行 hash 为 `3b47bd39…101f`。

冻结的 v2 checker/unitizer 当时得到 680 条 numeric match、39 条明确 numeric mismatch、78 条
candidate-not-numeric mismatch 和 3 条 conflicting-boxed-answer ineligible；799/800 条通过 exact-token
unitization。由此冻结了 40 个 Consistency pairs（hash `6a97b2bf…9145`）与 60 条 H/P rows（hash
`36a2b380…80cb`）。A=`gpt-5.5-sol`/xhigh、B=`claude-opus-5`/high 已分别完成 C/H/P 六个隔离上下文，
所有文件通过 population、schema、index 与 A-self-repeat 校验。原始结果是：

| 任务 | 原始结果 | 预注册裁决 |
|---|---|---|
| Consistency | decision 36/40 一致，A/B accept 为 34/38，κ=.459；隐藏控制各 4/4，A 自重复 8/8 | 90% agreement 与 10% 分歧率达到工程门；因 B 仅 2 个 reject，κ 门按协议不适用 |
| Hallucination | path 60/60 一致；54 clean、仅 6 个共同 positive，onset exact 与 ±1 均为 5/6；隐藏控制各 6/6，A 自重复 12/12 | positive 支持数 6<15、±1=.833<.85，且不可能凑出最终 20 positive，失败 |
| Key/Complete | eligibility 60/60；Key F1=.928，Complete F1=.784；exact set 仅 27/60，33/60 需裁决；隐藏控制各 3/6，A 自重复 12/12 | control<100%、Complete F1<.82、裁决率 .55>.40，三项失败 |

H 的异常不是“Phi 几乎没有错误”这么简单。30 条冻结 numeric-mismatch proposal 中，24 条实际推理和
答案都正确，只是写成 `\boxed{38 cents}`、`\boxed{9 glasses}` 等“数字+单位/文字”；v2 checker 把整个
boxed 内容当纯数字，解析失败后误记为 0。两位盲标者都把这 24 条判为 clean，因而准确暴露了 checker
系统性假阴性。按协议，checker 改动必须升版、重建 proposal 并重做所有依赖标注，所以 v2 在此正式记为
`FAIL_PIPELINE`：不送第三模型粉饰原始门、不 finalize、不抽 hidden state，也不训练。

代码已加入协议钉住的 backward-compatible checker：v2 仍可复现旧结果；新的
`clir_numeric_multisource_v3` 会从 boxed prose 提取受支配数值，并覆盖单位短语、答案句、等式、金额和
复合时长回归。用 v3 对同一批 800 rollout 只读重算为 754 numeric match、42 parsed mismatch、1 个
non-numeric refusal 和 3 个冲突 box；可用于机制 proposal 的真实 mismatch 只覆盖 13 个 GSM8K query 和
1 个 ASDiv-A query。现有池因此仍不足以支持 20 个 query-distinct positive onset，下一版必须先扩大/加难
acquisition，而不是再次标同一份包。Prior 另暴露了定义边界：55/60 条 Key 完全一致；Complete 的 33 条
分歧中有 29 条是 B 取 A 的严格超集，说明“实际依赖链”与“可压缩的最短证明”需要在新版指南和控制题中
明确区分。

v2 把 train-only query 池扩为 `60 GSM8K +40 ASDiv-A`，每题 8 条 Phi rollout，共 800 raw rows；
自然标注预算仍只送 40 个 Consistency proposals 和同一批 60 个 H/P proposals。所有 proposal 的机械
过滤、每 query 上限、source/numeric strata、hash tie-break、固定分母与 joint H/P 入选顺序都必须在
A/B 看标签前发布。最终目标仍是 30 个 C accepts，以及同一批 prior-usable 的
`20 first-bad-unit positive +20 explicit clean`。

两个主标注者现在必须来自不同模型系列，且都不得与 Phi generator/backbone 同系列；分歧只能交给第三个
不同系列模型先独立判断再匿名盲裁，没有合格第三模型就丢弃分歧行。unitizer v2 必须在保存的
`output_token_ids` 上给出连续、无重叠、完整覆盖的半开 token ranges；H 的兼容 onset 字段只表示
`first-bad-unit start token`，不再冒充精确首错 token。所有自然比例都以预先 hash 的 proposal manifest
为分母，格式失败、low 或 uncertain 不能静默移除。

SVAMP 原论文由 100 个 ASDiv-A seed 生成变化题，因此 v2 不再把它称为与 ASDiv 训练来源独立的外部
holdout。它继续保持不训练、不调参的 protected 状态，但准确角色是 **ASDiv-derived contrast/challenge
set**；独立来源泛化还需另选 holdout，或以后做密封 seed-family 排除。

无人类复核的最终标签统一叫 `silver_dual_ai_v2`，不能叫 Gold。smoke 通过只证明多题源生成、
numeric checker、dual-AI Silver、盲裁和 exact-token materialization 流水线达到预注册门，不证明三模块
提高 Best-of-N。当前 `.25` gate 继续作为默认方法身份保留，但本轮不训练、不调参。

离线验收入口：

```bash
python prepare_clir_smoke.py fixture \
  --output-dir run_artifacts/smoke_v2_fixture
```

正式执行的完整逐步命令及 A/B/第三模型分发顺序在
[`docs/data_expansion_smoke_protocol_v2.md`](docs/data_expansion_smoke_protocol_v2.md#16-2026-08-25-实现状态与唯一执行入口)；
可复制提示词在
[`configs/data_expansion_smoke_v2/annotation_prompts.md`](configs/data_expansion_smoke_v2/annotation_prompts.md)。
v2 不再继续裁决或训练；它只保留为失败审计与 backward-compatible checker 回归。v3 acquisition 已按
下面的新协议完成，旧标签不跨 manifest 搬运。

## 2026-08-25 多题源扩量 smoke v3：H 通过，C/Prior raw gate 停止

canonical 规格是
[`docs/data_expansion_smoke_protocol_v3.md`](docs/data_expansion_smoke_protocol_v3.md)，机器契约与标注语义在
[`configs/data_expansion_smoke_v3`](configs/data_expansion_smoke_v3)。v3 完整保留 v2 的 60 GSM8K +40
ASDiv-A train-only incumbents，并从 MATH train 的 algebra、counting/probability、number theory、
prealgebra 中按 level 3/4/5 分层预选 60 个 primary +48 个 reserve。只纳入 final boxed answer 能由同一
checker 唯一解析为标量数值、官方解至少 45 词且不含 Asymptote 的题。production loader 只请求 train
parquet，MATH test 不用于读取、选样或调参；冻结前一次开发性 loader 预检曾把 test 写入本机 cache，
但没有查看 test row/答案/表现，因此不能声称该工作区从未下载过 test。

primary 新增 60 题×8=480 条 Phi rollout，与绑定 hash 的旧 800 条合并为 160 题、1,280 条。v3 checker
得到 995 match、240 个明确 parsed mismatch、10 个冲突多答案、14 个 parse failure、21 个长度截断；
1,280/1,280 exact-token partition 通过。截断占 1.64%，低于冻结 2% 上限且全部排除。可用于机制提议的
query-distinct parsed mismatch 有 57 个，超过预注册最低 30；40 个 C 与 60 个 H/P 固定 strata 也都能
一次凑齐，因此状态为 `READY_FOR_FROZEN_V3_PROPOSAL_AND_ANNOTATION`，48 题 reserve 没有生成。

H/P 的 60 条固定为 GSM8K match/mismatch `10/8`、ASDiv-A match `12`、MATH match/mismatch `8/22`；
parse failure 不得冒充错误富集。Complete 现明确指候选实际依赖链里所有被后续使用的唯一非冗余中间
步骤，不得压缩成另一条更短证明。盲包已生成：A=`52/78/78`，B=`44/66/66`（C/H/P，含隐藏控制和
A-only self-repeat）。六份 A/B 标签已全部通过 population/schema/index 校验；每个任务两边的 hidden
controls 都是 100%，A 的 self-repeat 也都是 100%，说明输出不是随机或格式性失败。

raw triage 随后按冻结门停止。C 只有 `26/40=.65` 决策一致，低于 `.90`；至少 `14/40=.35` 必须裁决，
高于 `.20`。A/B accept 数为 `25/39`，且 B 只有 1 个 reject，表明两者对“近乎照抄”和“关键中间量是否
改变”的边界并不一致。H 是这一轮唯一完整通过所有 raw 门的任务：path agreement=`59/60=.9833`，
共同 positive=30，五个以上 material units 上 exact/±1 onset 都是 `26/30=.8667`，最低需裁决
`5/60=.0833`。Prior 的 eligibility=`60/60`、Key F1=`.9167`、Complete F1=`.9267` 都通过，且没有
Complete=全部 material units 的退化；但 exact target 仅 `25/60`，最低需裁决 `35/60=.5833`，超过 `.40`。

因此终态是 `STOP_RAW_GATE_FAILURE`，`third_model_send_allowed=false`。第三模型包虽由旧 triage 路径机械
生成在本地，但不得发送；本轮不 finalize、不抽 hidden state、不训练，也不产生模块 efficacy 或 Best-of-N
结论。代码现会在 triage 时显式执行这个停止规则，避免把裁决误当成能救活 raw failure。

## Consistency 提示词修复 v4：先做 14 条开发回放

v2/v3 有 36 个完全相同的 C pair；B 的 36 个 decision 全部复现，A 却把其中 9 个从 accept 改为 reject，
8 个理由是“近乎照抄”。因此先按
[`docs/data_expansion_smoke_protocol_v4.md`](docs/data_expansion_smoke_protocol_v4.md) 做一个不改 schema、
不加相似度算法的提示词修复：先判断数学路径，再判断表达差异，并明确相同公式、数字和必然顺序不能单独
作为照抄证据。

当前状态是 `READY_FOR_DEVELOPMENT_REPLAY`。先用 14 条已知争议做 A/B prompt regression，至少13/14
一致且通过反退化门后，才冻结提示词并制作 30 条不复用旧 C query 的新确认样本。14 条回放只用于开发，
不训练、不产生新可靠性证据，也不改变 v3 的失败终态。

## 目录

```text
configs/best_current.json             唯一默认模型与训练配置
configs/clean_ablation_v1/            可复现的 C/H/P/full 消融训练配置
configs/clean_gate_ablation_v1/       P0→PG0 prior-to-gate 冻结消融
configs/clean_gate_tuning_v2/         gate 工程默认值选择训练配置
configs/data_expansion_smoke_v2/      双审查后多题源扩量 smoke 的机器可读协议
configs/data_expansion_smoke_v3/      当前 MATH 扩量 smoke 的机器协议与标注语义
configs/data_expansion_smoke_v4/      Consistency 提示词修复、14-ID 回放清单与机器门
prepare_clir_smoke.py                  v2/v3 数据管线与 v4 Consistency prompt gate 入口
src/clir_smoke.py                      checker/unitizer/proposal/label 核心契约
src/clir_features.py                  identity/layer-axis 特征编码器
src/clir_data.py                      JSONL 数据、严格 token 对齐、collate、sampler
src/consistency_localized_reward.py   reward model、三模块和 loss
extract_hidden_states.py              exact-ID teacher-forced 全层特征抽取
train_clir.py                         训练、验证、原子 checkpoint、精确续训
score_clir.py                         打分、定位诊断、Best-of-N 选择
evaluate_clir.py                      query-level Best-of-N 与 pairwise 评估
evaluate_clir_mechanisms.py           H/onset/value 与 dual-prior 机制诊断
summarize_clir_ablation.py            多 seed 候选 parity 与 paired contrast
examples/create_toy_clir_data.py      仅供管线 smoke test 的合成数据
tests/                                模型、数据、续训与评估测试
docs/proposal.md                      与当前实现一致的方法说明
docs/handoff.md                       迁移裁决、历史证据和下一步
docs/data_expansion_smoke_protocol_v2.md  双审查整合后的 100-query smoke 协议
docs/data_expansion_smoke_protocol_v3.md  当前 160-query primary acquisition 与双标协议
docs/data_expansion_smoke_protocol_v4.md  Consistency 提示词修复与新样本确认门
```

当前分支顶端只保留训练/打分/评测代码、测试、可运行配置、README、handoff、核心方法说明和
当前 canonical 扩量协议。历史报告与被 supersede 的协议仍可从 Git 历史恢复，但不再占用当前
远端目录视图。

## 环境

```bash
pip install -r requirements.txt
```

`torch`、`numpy` 和 `pytest` 是训练与测试依赖；`transformers`、`huggingface_hub` 与 `pyarrow` 用于
exact-ID materialization、MATH train parquet 和抽取。v2/v3 source export 固定 `datasets==3.6.0`，rollout 固定
`vllm==0.5.3.post1` 与 `numpy==1.26.4`；训练和打分本身仍不会导入 task LLM 或 vLLM。

## 唯一默认配置

`train_clir.py` 默认读取 `configs/best_current.json`。loss 权重只在这个 JSON 中定义，CLI 仅允许覆盖 epoch、batch size、学习率、设备等运行参数，避免形成第二套隐含方法。

当前真实数据配置为：

- 输入是 Phi-3.5-mini 风格的 embedding + 32 blocks，共 33 层；每层宽 3072，原始 token 特征宽度为 `33 × 3072 = 101376`。
- layer-axis encoder 使用 2 层 Transformer、宽度 256、8 heads 和 4 个 learned pooling queries，把每个 token 压到 `model_dim=768`。
- condition attention 再通过独立的 256 维瓶颈计算，避免在原始全层宽度上建立二次参数层。
- 训练默认 5 epochs、batch size 4、learning rate `1e-4`、BF16、seed 42、gradient clipping 1.0。
- dual-prior 使用 `joint` phase；semantic grouping sampler 默认开启。

`--hidden_dim` 是 toy/开发用覆盖项：指定后会把 encoder 切为 identity。真实全层训练应直接使用默认配置，不要传这个参数。

## 三模块的默认开关

模型始终输出各诊断 head，但只有同时满足“权重非零”和“batch 中存在对应监督”时才产生训练 loss。缺失标签通过 mask 跳过，不会被填成负样本。

| 模块/分量 | 默认权重 | 当前行为 |
|---|---:|---|
| final correctness BCE | `1.0` | 训练 scalar reward |
| consistency | `1.0` | 同 semantic、异 style 拉近；异 semantic、同 style 分离；score consistency 为 `0.1` |
| hallucination onset BCE | `1.0` | 使用 `main` 的 onset→tail 二值目标 |
| token reward target | `0.5` | onset 后 target 改为 `-0.5` |
| negative-tail margin | `0.5` | onset 后要求 `token_values <= -0.5` |
| path-level MIL | `0.0` | 保留稳定 log-space 实现，默认关闭 |
| pseudo-onset tail | `0.0` | 保留实现，默认关闭 |
| progress loss | `0.0` | head 仍输出；且 `progress_score_weight=0.0`，不进入 scalar score |
| key / complete direct prior | `1.0 / 1.0` | 有对应 token target 时启用 |
| bidirectional prior distillation | `0.25` | 双向 stop-gradient mutual MSE |
| gate-prior alignment | `0.25` | 默认开启；保持 `main` 原始 shared-gradient 公式，当前是 dev-tuned 工程值 |
| complete reconstruction | `0.0` | 只接受外部 target，默认关闭 |

当前 hallucination 默认不是 `panzhixin` 的 sparse-span diagnostic：它回到 `main` 的定义——若 `hallucination_onset=k`，则第 `k` 个生成 token 起都属于受污染 tail，H head 做 token BCE，reward value path 同时受到负 tail 约束。这是方法身份选择，不是已经建立的效果结论。

## 数据格式

每个 JSONL row 表示同一个 query 下的一条候选 trajectory。建议所有真实数据都保留 exact token IDs，即使训练阶段实际读取的是预抽取 feature。

### 抽取前的最小 row

```json
{
  "id": "q000-cand00",
  "query_id": "q000",
  "candidate_index": 0,
  "prompt_token_ids": [1, 2, 3],
  "output_token_ids": [4, 5, 6],
  "correctness": 1
}
```

- `id`：trajectory 唯一标识。
- `query_id`：必填；Best-of-N 分组和 train/validation query-disjoint 检查使用，不要拿 `semantic_id` 代替。
- `candidate_index`：可选；评估时决定冻结的候选前缀顺序。也支持 `completion_index` 或 `vllm_completion_output_index`。
- `prompt_token_ids`、`output_token_ids`：生成时保存的原始 token IDs。两个序列都必须非空、非负，且元素必须是真正的整数；字符串、浮点数和布尔值即使可强转也会被拒绝。抽取脚本不从 response 文本重新 tokenize。
- `correctness`：建议为数值 `0/1`。允许部分 row 缺失，此时 final BCE 对该 row 跳过。

### 抽取后增加的字段

`extract_hidden_states.py` 会原样保留输入 row，并增加：

```json
{
  "hidden_states_path": "features/00000000.hidden.pt",
  "condition_states_path": "features/condition-<prompt-sha>.pt",
  "hidden_states_sha256": "...",
  "condition_states_sha256": "...",
  "feature_dim": 101376,
  "num_feature_layers": 33,
  "per_layer_dim": 3072,
  "feature_model": "your/model-or-local-path",
  "feature_revision": "pinned-or-resolved-commit",
  "feature_dtype": "bfloat16"
}
```

路径相对于输出 JSONL 所在目录。也可以用 inline `hidden_states` / `condition_states`，或 `.pt`、`.pth`、`.npy`、`.json` feature 文件。数据/续训兼容层同时支持 `panzhixin` 旧 manifest 的嵌套 `feature_metadata` 维度字段，以及 `feature_sha256` / `condition_sha256` checksum 别名；新旧 metadata 同时存在时必须一致。

### 可选监督

| 字段 | 形状/语义 |
|---|---|
| `semantic_id` | consistency 语义组 |
| `style_id` | consistency style/domain 属性 |
| `hallucination_onset` | 生成 token 的首错索引；`-1` 表示已知 clean；字段缺失表示未标注 |
| `path_hallucinated` | path-level `0/1`，只供默认关闭的 MIL/pseudo-tail 使用 |
| `token_advantage` | 长度严格等于输出 token 数；token value 的外部 target |
| `progress_targets` | 等长 token target；当前默认关闭 |
| `key_prior_target` | 等长的 key evidence `0/1` target |
| `complete_prior_target` | 等长的 complete support `0/1` target |
| `complete_reconstruction_target` | 外部生成的固定宽度向量；当前默认关闭 |

`token_advantage`、`progress_targets`、`key_prior_target` 和 `complete_prior_target` 必须与 trajectory feature 的 token 长度完全一致；不一致直接报错，不做截断或补零。`correctness`、onset 和各 auxiliary target 都有独立 mask，因此一份 manifest 可以混合不同监督覆盖的 row。

## Exact-ID 全层特征抽取

输入必须已经包含 `query_id`、`prompt_token_ids` 和 `output_token_ids`。脚本对后两者拼接后做一次无 padding、teacher-forced causal forward，读取模型返回的全部 hidden-state layers；随后按保存的 prompt 长度精确切分：

```text
condition_states = all_layer_states[:len(prompt_token_ids)]
hidden_states    = all_layer_states[len(prompt_token_ids):]
```

整个过程不 decode response，也不调用 tokenizer 重建 output IDs，因此 token target、onset 和 hidden-state 位置使用同一身份坐标。

```bash
python extract_hidden_states.py \
  --input_jsonl data/rollouts.jsonl \
  --output_jsonl data/extracted.jsonl \
  --feature_dir data/features \
  --model your/model-or-local-path \
  --revision pinned-model-commit \
  --dtype bfloat16 \
  --expected_num_feature_layers 33 \
  --expected_per_layer_dim 3072
```

同一 `query_id` 的所有候选必须使用完全相同的 prompt IDs；相同 prompt 序列只保存一份 condition tensor，各候选复用同一相对路径和 checksum。`--revision` 会传给模型加载器，manifest 记录 `feature_model`、解析到的 commit/所给 revision、dtype 和两个 feature 文件的 SHA-256。

每个 tensor 和最终 manifest 都原子发布；脚本还会拒绝不等长 token target、非法 onset 和跨 row 不一致的 layer contract。若输出 manifest 或 feature 已存在，默认直接失败；`--overwrite` 只应用于明确可整体替换的重跑。正式数据优先写入新目录，避免中途失败后在旧 manifest 下留下部分已替换的 feature。脚本不负责 rollout 生成、correctness 判断或标注。

兼容性 smoke 已使用现有 3968-row 旧 manifest 完成 schema 解析和首条真实 feature 读取：trajectory 为 `[221,101376]` BF16，condition 为 `[105,101376]` BF16，`33×3072` metadata 一致。这只是 reader/schema smoke，不是新 extractor 的全量复制，也不是联合训练效果。

## 训练

推荐使用显式、query-disjoint 的 train/validation manifest：

```bash
python train_clir.py \
  --train_jsonl data/train_extracted.jsonl \
  --val_jsonl data/validation_extracted.jsonl \
  --config configs/best_current.json \
  --output_model outputs/best_current.pt
```

若只有一个 manifest，可按 `query_id` 而不是按 row 切分：

```bash
python train_clir.py \
  --train_jsonl data/extracted.jsonl \
  --val_fraction 0.1 \
  --output_model outputs/best_current.pt
```

训练前会检查 feature width；显式 validation 若与 train 有重复 `query_id` 会直接失败。每个 epoch 都会检查 total loss 和梯度是否 finite，执行 gradient clipping，然后原子发布 full-state checkpoint 和 `<checkpoint>.metrics.jsonl`。checkpoint 包含 model、optimizer、完成 epoch、RNG、配置、数据 hash 和 metrics。

### 精确续训

`epochs` 表示续训后的总目标 epoch 数。下例从已完成 epoch 5 的 checkpoint 继续到 epoch 10：

```bash
python train_clir.py \
  --train_jsonl data/train_extracted.jsonl \
  --val_jsonl data/validation_extracted.jsonl \
  --config configs/best_current.json \
  --output_model outputs/best_current.pt \
  --resume_from outputs/best_current.pt \
  --epochs 10
```

除 `epochs` 外，模型配置、训练设置和数据必须与 checkpoint 一致，否则拒绝续训。sampler 使用显式 `(seed, epoch)` 顺序，optimizer 和 RNG 也会恢复；CPU 测试覆盖了 interrupted 与 uninterrupted 训练最终状态完全一致。

## 打分与 Best-of-N 标记

```bash
python score_clir.py \
  --input_jsonl data/validation_extracted.jsonl \
  --model outputs/best_current.pt \
  --output_jsonl outputs/validation_scored.jsonl
```

打分默认 `batch_size=2` 和 BF16 autocast，是针对 101376 维全层 feature 的保守设置；需要 FP32 时显式传 `--amp_dtype none`。

输出保留原 row，并增加：

- `clir_score` 和 `clir_checkpoint_sha256`；
- `clir_path_hallucination_prob`、`clir_path_no_hallucination_log_prob` 和 `clir_pseudo_onset`；
- 逐 token `clir_hallucination_prob`、`clir_token_reward` 和 `clir_token_value`；
- `clir_mean_gate`、逐 token `clir_gate_attention` 和 `clir_condition_relevance`；
- 归一化的 `clir_key_prior` / `clir_complete_prior`，独立 sigmoid membership `clir_key_prior_membership` / `clir_complete_prior_membership`，以及 overlap 诊断 `clir_prior_gate_alignment` 和与训练目标同定义的 `clir_prior_gate_squared_l2`；
- `clir_selected_best_of_n`，每个 `query_id` 恰有一个最高分候选被标记。

`clir_pseudo_onset` 始终作为诊断输出；默认配置中的 pseudo-tail 训练仍为关闭状态。输出 JSONL 原子写入，且不得覆盖输入 manifest 或 checkpoint；已存在的其他输出需显式 `--overwrite`。

## 查询级评估

```bash
python evaluate_clir.py \
  --input_jsonl outputs/validation_scored.jsonl \
  --output_json outputs/validation_metrics.json \
  --k 1,2,4,8,16 \
  --bootstrap_replicates 2000
```

评估器按 `query_id` 分组，按冻结的 candidate index 排序，对每个 `k` 只使用前 `k` 个候选，报告：

- reward Best-of-N accuracy 和 query bootstrap 95% 区间；
- random expected accuracy；
- oracle accuracy；
- query 内 correct-vs-wrong pairwise accuracy 和 tie 数。

score tie 使用最早 candidate，保证结果稳定。默认要求每个 query 都至少有 `max(k)` 个候选，因此所有 K 使用同一 query population；任一 query 不足都会直接失败。只有探索性报告才应传 `--allow_incomplete_queries`，此时改为每个 K 单独过滤候选不足的 query，各 K 的 population 可能不同。

评估前会检查 score 全部 finite、correctness 严格为 finite `0/1`，candidate index 不重复且显式 index 从 0 连续。报告原子写入，记录 `input_jsonl_sha256` 以绑定输入 scored manifest；已存在的输出需显式 `--overwrite`。

机制标签存在时，可把 H 排序、onset 阈值、pre/tail value shift 和 key/complete prior
learnability 与 task ranking 分开评估：

```bash
python evaluate_clir_mechanisms.py \
  --input_jsonl outputs/mechanism_dev_scored.jsonl \
  --output_json outputs/mechanism_metrics.json
```

matched 多 seed 比较使用 `summarize_clir_ablation.py`。它要求目录为
`seed_<seed>/<cell>/validation_{scored,metrics}.*`，逐行核对所有 run 的候选身份、顺序、
correctness、scored-input hash 和 checkpoint hash，再对同 query outcome 做 paired
bootstrap；预声明 cells 见 `configs/clean_ablation_v1/`，已完成结果和结论见
[`docs/handoff.md`](docs/handoff.md)。

## Toy smoke test

Toy 数据只验证代码路径，不能证明方法有效：

```bash
python examples/create_toy_clir_data.py \
  --output_jsonl examples/toy_clir.jsonl \
  --feature_dir examples/features \
  --hidden_dim 8

python train_clir.py \
  --train_jsonl examples/toy_clir.jsonl \
  --feature_root . \
  --output_model outputs/clir_toy.pt \
  --hidden_dim 8 \
  --epochs 2 \
  --amp_dtype none \
  --num_workers 0

python score_clir.py \
  --input_jsonl examples/toy_clir.jsonl \
  --feature_root . \
  --model outputs/clir_toy.pt \
  --output_jsonl outputs/clir_toy_scored.jsonl

python evaluate_clir.py \
  --input_jsonl outputs/clir_toy_scored.jsonl \
  --output_json outputs/clir_toy_metrics.json \
  --k 1,2
```

完整测试：

```bash
pytest -q
```

## 当前限制

- 仓库已有 v2 rollout、numeric checker、first-bad-unit/dual-prior Silver 标注包与盲裁管线，但没有人工
  标注或人工复核；它不生成 Gold。
- 多题源管线不仅通过了 8-query/64-row deterministic fixture，也完成了真实 100-query/800-row Phi
  rollout、checker/unitizer audit 和 40 C/60 H/P proposal 冻结；双 AI C/H/P 标签、triage/finalize 与全部
  annotation-quality 硬门仍未完成。不得把 rollout 或 proposal yield 当作标签质量、训练效果证据。
- 默认仍使用预抽取全层 feature，真实数据的磁盘开销很大；没有集成 batch-local online extraction。
- 当前 objective 是 pointwise correctness BCE 加可用 auxiliary supervision，尚无 pairwise/listwise reward objective。
- clean integration 已在现有小数据上完成三 seed matched matrix，但没有扩充独立机制标签或 protected test；full 没有优于 correctness-only。
- 历史 consistency、hallucination 和 prior 标签规模都很小，不能支持跨域或正式机制结论。
- clean checkpoint 已记录配置、数据/split hash、feature reference、optimizer/RNG、metrics、code commit/branch/dirty state、完整命令与运行环境；这不替代缺失的数据 provenance 上游与 protected-test protocol。
- clean 已有 frozen-prefix evaluator、机制诊断和 parity-checked multi-seed paired summarizer；尚未重建 strict/encoded SWIFT 等预算 baseline，也没有单独的 held-out consistency evaluator。

研究假设、已有证据与未验证部分见 [`docs/proposal.md`](docs/proposal.md)；迁移依据和历史负结果见 [`docs/handoff.md`](docs/handoff.md)。
