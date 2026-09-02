# Prior v14 fresh smoke：扩大候选依赖边后重新双标

状态：`FROZEN_FRESH_SMOKE_PACKAGES_NOT_YET_BUILT`

这轮只回答一个问题：v13 的 block/机械回溯思路不变时，把“每步最多两个候选父依赖”改成更完整但有上限的 2--6 个候选，两个 max 推理模型能否在全新样本上稳定得到 Key 和 Complete。

它不是 v13 修复，也不是训练扩量。v12、v13 的失败结论和旧标签保持不变。

## 数据

- 父池：已经冻结并完成 exact-token materialization 的 Prior-v12 acquisition train split。
- 永久排除：v12 的 800 个 proposal query/cluster，以及 v13 的 48 个 natural query/cluster。
- 新取 48 条，每条来自不同 query 和不同模板簇。
- GSM8K/MATH × numeric match/mismatch × medium/long 共八层，每层固定 6 条。
- 选择只使用冻结的题源、checker 状态、长度和 SHA-256 优先级，不看任何 AI 标签。

Smoke 行不是训练数据。即使所有门都通过，也只能另行冻结 scale-v15。

## 机械表示

原始 material unit 和 `unit_index` 继续作为唯一 token 映射真相。v13 的安全 block 合并保持不变；变化仅限候选依赖边：

- 识别带逗号数字和简单数值 LaTeX 分数；
- 优先保留真正算出某个数值的步骤，而非后面的复述；
- 多输入计算尽量保留每个输入的最近生产者；
- 代数改写保留最近的同变量方程；
- 计划句和题面复述不能占掉唯一候选槽；
- 每个 child 最多 6 个候选父节点。

AI 必须逐条 `keep/drop/uncertain`，仍可补最多两条真正漏掉的直接边。Complete 不由 AI 直接圈选，而是程序从 final block 沿保留边向前回溯；Key 仍是一个原始 unit。

## 双 AI 包

每侧共 72 行，拆成四个独立 shard：

- 12 条自然样本；
- 2 条隐藏控制；
- 4 条跨 shard 自重复。

A 使用用户报告的 GPT-5.6-sol/max；B 使用升级后的 Claude Opus/max。两边必须在相互隔离的新上下文中运行，不得读取 `PRIVATE_*`、另一侧目录、checker、参考答案、历史标签、代码或测试。产品若显示精确 revision，应另记在会话记录中；没有机器证据时不得声称已验证 revision 或 temperature。

## 预冻结门槛

- 控制题：A/B 都必须 `8/8`；
- 16 条自重复目标精确率：每侧至少 `15/16=.9375`；
- 双方共同 usable、非 low：至少 40/48；
- final block 精确一致率至少 `.90`；
- Key 精确一致率至少 `.85`；
- Complete macro F1/IoU/coverage 至少 `.90/.80/.90`；
- block role、候选边决定一致率都至少 `.85`；
- Complete 并集等于全部 material units 的比例至多 `.25`；
- 每侧仍需补 missing edge 的行比例至多 `.15`。

任一门失败即停止 v14。不得改 prompt、改阈值、选择容易行、裁决、局部重标或混合多次尝试。

## 执行顺序

代码、协议、prompt 和评价器必须先处于一个干净 Git commit：

```bash
P=/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python
cd /prodcpfs/user/panzhixin/ICLR_2027
$P prepare_clir_prior_mechanical_v14.py prepare
$P prepare_clir_prior_mechanical_v14.py verify
```

只有 `PASS_PRIOR_V14_PACKAGE_INDEPENDENT_RECOMPUTE` 后，才可分别把
`configs/data_expansion_prior_v14/launch_prompt_a.txt` 和
`configs/data_expansion_prior_v14/launch_prompt_b.txt` 交给两个模型。

两侧四个标签文件全部完成后，只运行一次冻结评价器：

```bash
$P prepare_clir_prior_mechanical_v14.py evaluate
```

通过只表示“双 AI 在这套机械定义下达到预注册的操作稳定性”，不表示自然标签准确、Prior 有效、Gate 有效或 Best-of-N 提升。
