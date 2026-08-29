# CLIR Consistency v6.1 精确特征抽取协议

状态：`PASS_SELECTED_FEATURE_EXTRACTION_AND_VERIFICATION_V6_1`

授权日期：2026-08-29。机器授权文件：
`configs/data_expansion_scale_v6/feature_extraction_authorization_v6_1.json`。

## 1. 本阶段做什么，不做什么

本阶段只把已经发布并独立复核的 v6.1 关系清单转换成 CLIR 可读取的全层 hidden states：

```text
400 对训练正关系
150 对 heldout 正关系
150 对 heldout hard negative（只评估）
        ↓ 对 relation endpoint 去重
1,357 条唯一 trajectory + 612 份唯一 prompt condition
        ↓ Phi-3.5-mini 固定 revision，exact saved IDs teacher forcing
1,357 个 [T, 101376] BF16 trajectory tensor
  612 个 [C, 101376] BF16 condition tensor
```

明确不做：不抽完整 16,000 条 rollout，不重新生成，不改 A/B 标签、关系、阈值或 split，不调用第三个 AI，
不训练模型。抽取全部通过后，训练仍需用户另行授权。

## 2. 冻结输入

授权文件逐一固定以下父级文件及 SHA-256：v6 基础协议、v6.1 hard-negative 修订、正式 plan、独立关系
verifier、16,000 条 materialized rows、三份关系 manifest 和 1,357 条 selected inventory。

关键约束是：三份关系里所有 `left_id/right_id` 的并集必须与 inventory 的 1,357 个 trajectory ID 完全相等。
程序随后只凭这些 ID 去 materialized rows 取回原始 `prompt_token_ids/output_token_ids`。任何缺行、多行、ID
重复、token 数不符、同 query prompt IDs 不同或每题不是恰好一个 condition owner，都会在使用 GPU 前停止。

## 3. 特征定义

- 模型：`microsoft/Phi-3.5-mini-instruct`；
- revision：`2fe192450127e6a83f7441aef6e3ca586c338b77`；
- attention implementation：`sdpa`；
- 存储 dtype：BF16；
- 层：embedding output +32 transformer blocks，共33层；
- 每层宽度3072，拼接后每个 token 为 `33×3072=101376` 个 BF16 数；
- 唯一 token 轴是真实 rollout 保存的 IDs，禁止 decode 后重新 tokenize。

每条 trajectory 都把保存的 prompt IDs 与 output IDs 拼起来做无 padding、teacher-forced forward，再严格按保存
的 prompt 长度切出生成 token 部分。每个 query 只保存 inventory 指定 owner 那次 forward 的 prompt 部分，其他
候选复用这一个 condition 文件。

精确预算为输出 token 460,151、唯一 prompt token 59,952，共520,103 个 feature token；纯张量
`520103×101376×2=105,451,923,456` bytes，即98.210 GiB（105.452 GB）。序列化封装、manifest 和一次最大样本
preflight 会额外占少量空间；启动前要求至少130 GB 可用空间。

## 4. 为什么另做分片抽取器

通用 `extract_hidden_states.py` 的 exact-ID 逻辑正确，但它是单进程整表脚本，不适合约105 GB 的正式任务。
本阶段使用 `extract_clir_scale_features.py`：

1. 按 query 计算“全部输出 token +一份 prompt token”的成本；
2. 采用固定 largest-first 规则均衡分给8个 worker，同题所有 view 永远同 worker；
3. 每个 tensor 先写临时文件，再原子替换正式路径；
4. 一个 query 的 condition 与所有 trajectory 都完成后，才原子发布 query marker；
5. 中断后，有完整 marker 的题只核对路径和大小后跳过；无 marker 的现有 tensor 必须重新加载、检查后才能复用；
6. 8个 GPU worker 都通过后，再由8个 CPU verifier 逐文件重读，检查 shape、BF16、连续性、有限值和 SHA-256；
7. 只有全部 verifier 与 writer 的 payload digest 一致，才发布最终 extracted manifest。

这套恢复规则不会覆盖已经发布的 rollout、materialized rows、标签、关系或 inventory。前两次 plan 通过后，
preflight 都在模型加载前被 PyTorch 2.3 的 CUDA 显存统计 API 停止：第一次参数不是整数设备号，第二次虽改成
整数但还没初始化 CUDA context；两次都没有写 tensor，且已逐次记录在机器授权。修复后不复用旧 plan，正式
新文件放在全新的 Git 忽略目录 `run_artifacts/data_expansion_scale_v6/features_v6_1_run3/`。

## 5. 固定执行顺序

使用项目固定 Python：

```bash
P=/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python

$P extract_clir_scale_features.py prepare
$P extract_clir_scale_features.py verify-plan
CUDA_VISIBLE_DEVICES=0 $P extract_clir_scale_features.py preflight

# 0..7 各占一张 GPU，可并行；进程内都使用 cuda:0。
CUDA_VISIBLE_DEVICES=0 $P extract_clir_scale_features.py extract-worker --worker-index 0

# 0..7 均为 CPU 独立重读，可并行。
$P extract_clir_scale_features.py verify-worker --worker-index 0

$P extract_clir_scale_features.py finalize
```

正式执行前，授权、代码、测试和本协议必须先形成 clean commit。plan 会绑定该执行 commit；中途修改代码或
tracked 文件会使后续命令 fail closed。

## 6. 通过门与证据边界

最终 PASS 必须同时满足：

- 精确1,357 个 trajectory payload 和612个 condition payload；
- 每个形状分别严格为 `[保存的 output token 数,101376]` 和 `[保存的 prompt token 数,101376]`；
- 全部 BF16、连续、有限；
- 每个文件 SHA-256 与原子 query marker 一致；
- 独立 verifier 重算的 payload digest 与 writer 一致；
- raw tensor bytes 精确为105,451,923,456；
- 最终 manifest 的 endpoint population 仍等于三份已发布关系的并集；
- 报告继续写明 `training_allowed=false`。

通过只证明“选定关系的 exact-token 全层特征已完整、可读取、可复核”，属于数据/工程 pipeline 证据。它不证明
Consistency 可学，更不证明 Best-of-N 提升。下一个独立决策门才是 C-only 训练与 heldout 正负关系机制评估。

## 7. 正式执行结果

正式执行绑定 clean code commit `64470c8c76ffdbeee3c1f810e8d0ea9d86752b95`，写入全新的 Git-ignored
目录 `run_artifacts/data_expansion_scale_v6/features_v6_1_run3/`。plan、全宽 preflight、8 个 GPU writer、
8 个 CPU independent verifier 和 finalize 依次通过：

- plan SHA-256：`1a0961753e0babcd572a311fb134f97315bd7f29ff8994d0d41a5152b68f67cf`；
- preflight SHA-256：`9080bd88bca3523c214ea73f1dd050244752cabd9d431a7c93786c4ddad285d3`；
- 最终报告 SHA-256：`a1ce2d9b2023f97497714977a2faac3e926f5d9a34a397bb63c249c837848daa`；
- extracted manifest SHA-256：`ac1f35ff124f05caebd1676fdb29cb299f6f169a6bb2007b0133c124d8a47a8b`，
  有序行 hash `1901e88c214dff5b3755620776b527a2df3a2ed7306812048effb2909ed98c50`；
- payload 为1,357个 trajectory +612个 condition，共1,969个文件；raw tensor bytes
  `105451923456`，PyTorch 序列化后 `105455351485` bytes；
- verifier 逐文件重读并核验全部 shape、BF16、contiguous、finiteness 与 SHA-256；8组 writer/verifier 的
  payload digest 各自完全一致；
- `CLIRTrajectoryDataset` 额外只读加载普通行和最长行成功；最长行 trajectory/condition 为
  `[980,101376]` / `[85,101376]`，dtype 均为 BF16；
- 两个早期 preflight 目录都没有写入 tensor，正式目录没有临时/partial 文件；完整16,000条 rollout 从未抽取。

机器可提交的结果摘要见
`configs/data_expansion_scale_v6/feature_extraction_completion_v6_1.json`。大 tensor、worker report 和最终 manifest
继续只保存在 Git-ignored artifact 目录。终态仍是 `training_allowed=false`；下一门是
`SEPARATE_C_ONLY_TRAINING_AUTHORIZATION`，本阶段没有启动任何训练。
