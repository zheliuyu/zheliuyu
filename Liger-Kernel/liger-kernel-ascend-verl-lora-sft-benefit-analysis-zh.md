<h1 align="center">从 Kernel 到 verl LoRA SFT：Liger-Kernel 在昇腾训练中的收益分析</h1>

<p align="center">
<a href="https://github.com/zheliuyu">zheliuyu</a><sup>1</sup>,
<a href="https://github.com/Tcc0403">Tcc0403</a><sup>2</sup>,
<a href="https://github.com/TianHao324">TianHao324</a>,
<a href="https://github.com/lowdy1">lowdy1</a>,
<a href="https://github.com/noemotiovon">noemotiovon</a>,
<a href="https://github.com/sunyi0505">sunyi0505</a><br>
<a href="https://github.com/orangeH25">orangeH25</a>,
<a href="https://github.com/UserChen666">UserChen666</a>,
<a href="https://github.com/Ginray">Ginray</a>,
<a href="https://github.com/Hailey-Zh">Hailey-Zh</a>,
<a href="https://github.com/ji-huazhong">ji-huazhong</a><br>
<a href="https://github.com/jiaqiw09">jiaqiw09</a>,
<a href="https://github.com/kiritorl">kiritorl</a>,
<a href="https://github.com/pillumina">pillumina</a>, and
<a href="https://github.com/xuedinge233">xuedinge233</a>
<br><br>
<sup>1</sup> <em>通讯作者。</em><br>
<sup>2</sup> <em>Liger-Kernel 维护者；合入 NPU 相关 PR 并在检视过程中提供大量建议。</em><br>
<em>其余作者按已合入 NPU 相关 PR 数量排序（并列时按字母序）。</em>
</p>

## 摘要

在 **Qwen3-8B + verl LoRA SFT（4 卡 FSDP）+ Atlas 800T A3(x86)** 配置下，本报告采用「算子 micro-benchmark → verl 整网训练」两阶段方法，评估 **Liger-Kernel 在昇腾训练栈中的接入收益**。算子层以 Qwen3 SFT 路径上高频的 **rms_norm、rope、cross_entropy** 为代表性样例，在 T=8192、full 模式下测得加速比 **1.69× / 1.25× / 1.90×**，峰值 NPU 显存节省 **28.8%～68.4%**；整网层（GSM8K、580 step、global_tokens 100% 对齐）在启用 Liger-Kernel 后 MFU 中位数相对提升 **4.14%**，Host CPU 内存末步下降 **21.29%**，train/val loss 与对照组整体重合且末步略优。Amdahl 估算的理论 MFU 提升（约 **4.3%**）与实测一致。结果表明：Liger-Kernel 可在 verl LoRA SFT 中低侵入接入，未引入可观测精度风险，并在吞吐与 Host 内存方面带来稳定收益；上述三项 kernel 的单点表现可作为 Ascend 后端能力与整网收益的可复现验证样例。

**关键词：** Liger-Kernel；Ascend NPU；verl；LoRA SFT；Micro-Benchmark；MFU

## 1 引言

### 1.1 Liger-Kernel 概述

[Liger-Kernel](https://github.com/linkedin/Liger-Kernel) 是面向大语言模型训练的融合算子库，基于 Triton 在 GPU 侧实现 RMSNorm、RoPE、CrossEntropy 等高频算子的融合计算与 in-place 优化，目标是在保持计算语义不变的前提下降低显存占用并提升训练吞吐。主要接入方式包括：

- **高层 API（Patching）：** 对 HuggingFace 模型调用 `apply_liger_kernel_to_*`（如 Qwen3 对应 `apply_liger_kernel_to_qwen3`），按模块替换原生实现；
- **低层 API：** 直接调用 `LigerRMSNorm`、`LigerCrossEntropyLoss` 等算子类。

在 verl 等训练框架中，可通过 `use_liger=True` 一键触发 monkey patch，在不修改模型结构的前提下将模型中的部分算子切换为 Liger 实现；具体启用哪些模块由 patch 配置决定（如 Qwen3 的 `apply_liger_kernel_to_qwen3`）。

### 1.2 Ascend NPU 支持

Liger-Kernel 在 [v0.8.0 release](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.8.0) 中正式提供 NPU 原生支持。运行时，NPU 侧采用独立 Ascend 后端（`liger_kernel.ops.backends._ascend`），在检测到 NPU 设备后自动注册并替换默认 CUDA/Triton 实现；依赖栈为 **torch 2.6 + torch_npu 2.6.0 + triton-ascend 3.2.0**（见项目 `setup.py`）。

**表 1** Ascend 后端低层算子支持列表（节选）

| 类别 | 已支持算子（节选） |
|------|-------------------|
| **Norm / 激活** | RMSNorm、FusedAddRMSNorm、LayerNorm、GroupNorm、DyT、PolyNorm |
| **Attention 相关** | RoPE、Llama4 RoPE、Qwen2-VL MRoPE、Softmax、Sparsemax |
| **MLP / MoE** | GeGLU、FusedMoE |
| **Loss / 对齐** | CrossEntropy、FusedLinearCrossEntropy、GRPO Loss、JSD、KL Div、TVD |
| **其他** | Embedding、AttnRes、mHC 等 |

**高层 Patching：** Qwen3 等架构可通过 `apply_liger_kernel_to_qwen3` 按需启用 `rms_norm`、`rope`、`cross_entropy`、`fused_linear_cross_entropy` 等模块。

### 1.3 目标与主要结果

在 **Qwen3-8B + verl LoRA SFT（4 卡 FSDP）+ Atlas 800T A3(x86)** 场景下，本报告关注 **Liger-Kernel 从单算子能力到 verl 整网训练的综合收益**，而非某一类算子的孤立优化。Ascend 后端已覆盖 Norm、Attention、Loss 等多类算子（见表 1）。本次评估选取 Qwen3 SFT 路径上调用频次高、且与 verl 默认 patch 对齐的 **rms_norm、rope、cross_entropy** 作为代表性验证样例，用于回答 kernel 层是否值得接入，以及接入后整网表现如何。报告按两条主线组织：**单算子能力评估 → verl LoRA 整网验证**。

1. **算子层：** 以代表性 kernel 为样例，评估 Liger Ascend 实现相对基线的性能与显存收益，为整网接入提供依据；
2. **整网层：** 在真实 LoRA SFT 链路中启用 Liger-Kernel，观测吞吐（MFU）、显存占用及 loss 收敛等端到端指标。

**表 2** 主要结果摘要（GSM8K SFT，4 卡，580 step；算子层为代表性 kernel 样例）

| 层次 | 主要收益 | 关键数据 |
|------|----------|----------|
| **算子层** | Liger 单 kernel 加速与显存节省 | RMSNorm **1.69×** / **68.4%**；RoPE **1.25×** / **28.8%**；CE **1.90×** / **40.0%**（T=8192） |
| **verl LoRA SFT** | 吞吐、内存与精度 | MFU **+4.14%**；Host CPU 内存 **-21.29%**（末步）；val/loss **-3.27%**；token 对齐 **100%** |

## 2 实验设置

本节列出算子 micro-benchmark 与 verl LoRA SFT 整网实验共用的硬件、软件及模型配置；分节设计见第 3、4 节。

**表 3** 实验环境与配置

| 项目 | 配置 |
|------|------|
| 硬件 | Atlas 800T A3(x86) |
| 整网并行 | 1 node × **4 NPU**（FSDP，见表 7） |
| Liger-Kernel | [`8020e69`](https://github.com/linkedin/Liger-Kernel/commit/8020e691d4b78be6cc4868b96e5c73ca3c1058ea) |
| verl | [`c131c70`](https://github.com/verl-project/verl/commit/c131c704db5b2e2dadc7576edcad0e6f4a22c669) |
| 精度 | bfloat16 |
| 模型 | Qwen3-8B（hidden=4096，GQA 32 heads / 8 kv heads，vocab≈128256） |
| 最大序列长度 | **8192** tokens（SFT 与 benchmark 关键对齐点） |
| 验证样例 | `rms_norm`、`rope`、`cross_entropy`（Qwen3 SFT 高频路径，见表 8） |

## 3 算子层评估

**目标：** 通过代表性 kernel 的单点测试，说明 Liger Ascend 后端相对基线实现的性能与显存收益，支持整网 Liger-Kernel 接入。本节并不穷尽 Ascend 后端全部算子（见表 1），而以与第 4 节整网实验一致的 **rms_norm、rope、cross_entropy** 为样例展开分析。

### 3.1 实验设计

**表 4** 算子 micro-benchmark 实验设计

| 项目 | 配置 |
|------|------|
| 测试框架 | Liger-Kernel 官方 benchmark（`benchmark/data/all_benchmark_data.csv`） |
| 运行设备 | 单 NPU（算子 isolated 测试，与整网 4 卡配置无关） |
| 样例选取 | 与整网 patch 对齐：`rms_norm`、`rope`、`cross_entropy` |
| 序列长度 sweep | T = 1024 / 2048 / 4096 / **8192** |
| 测试模式 | forward / backward / **full**（报告以 full 模式为主） |
| RMSNorm 基线 | HuggingFace Qwen3RMSNorm |
| RoPE 基线 | HuggingFace `apply_rotary_pos_emb` |
| CrossEntropy 基线 | PyTorch `CrossEntropyLoss` |
| 对比方式 | 相同 shape、相同 T 下，Liger Ascend 实现 vs 基线实现 |

### 3.2 T=8192 汇总结果

**表 5** 代表性 kernel 在 T=8192 的 micro-benchmark 结果

| Kernel | 基线 | Liger | **加速比** | **显存节省** |
|--------|------|-------|------------|--------------|
| RMSNorm | 7.40 ms | 4.39 ms | **1.69×** | **68.4%** |
| RoPE | 5.19 ms | 4.14 ms | **1.25×** | **28.8%** |
| CrossEntropy | 22.52 ms | 11.88 ms | **1.90×** | **40.0%** |

### 3.3 逐算子分析

以下三节以 Qwen3 SFT 计算图中的三类高频算子为例，展示 Liger Ascend 后端在 **耗时、显存与序列长度扩展性** 上的单点收益；同类算子（如 LayerNorm、FusedMoE 等）的评估方法可复用本节流程。

#### 3.3.1 RMSNorm

Qwen3 每个 decoder 层包含 2 次 RMSNorm 调用，36 层合计 **72 次/step**，为本次样例中调用频次最高者。T=8192、full 模式下加速比为 **1.69×**，峰值显存节省 **68.4%**。LoRA 不改变各层 Norm 的执行次数，该路径的算子级收益较易传递至整网层面。

图 1–4 分别给出 RMSNorm forward / backward / full 耗时及 full 模式峰值显存随序列长度的变化：Liger 实现在全 T 区间内均低于 HuggingFace 基线。T=8192 时 forward 加速 **2.07×**（0.68 ms → 0.33 ms），backward 加速 **5.46×**（3.72 ms → 0.68 ms），full 模式加速 **1.69×**；T 由 1024 增至 8192 时 full 模式加速比由 1.58× 升至 1.69×，与表 3 所列序列长度上限 **8192** 相对应。LoRA SFT 仍须对冻结的 base 权重执行完整前向与反向计算，Norm 路径优化可直接反映为 step 耗时下降。峰值显存方面，四个 T 测试点的节省比例均为 **68.4%**（8192 时 1216 MB → 384 MB），为本次三个样例中最高。

<table align="center">
<tr>
<td align="center" width="49%"><img src="../assets/Liger-Kernel/benchmark/rms_norm_speed_forward_token_length.png" alt="RMSNorm forward-mode latency" width="100%"/><br/><strong>图 1</strong> RMSNorm forward 模式耗时（随序列长度）</td>
<td align="center" width="49%"><img src="../assets/Liger-Kernel/benchmark/rms_norm_speed_backward_token_length.png" alt="RMSNorm backward-mode latency" width="100%"/><br/><strong>图 2</strong> RMSNorm backward 模式耗时（随序列长度）</td>
</tr>
<tr>
<td align="center" width="49%"><img src="../assets/Liger-Kernel/benchmark/rms_norm_speed_full_token_length.png" alt="RMSNorm full-mode latency" width="100%"/><br/><strong>图 3</strong> RMSNorm full 模式耗时（随序列长度）</td>
<td align="center" width="49%"><img src="../assets/Liger-Kernel/benchmark/rms_norm_memory_full_token_length.png" alt="RMSNorm full-mode peak memory" width="100%"/><br/><strong>图 4</strong> RMSNorm full 模式峰值显存（随序列长度）</td>
</tr>
</table>

#### 3.3.2 RoPE

每层 attention 前调用 1 次，合计 **36 次/step**。T=8192、full 模式加速比为 **1.25×**。图 5–8 显示，随 T 增大 full 模式加速比由 1024 的 1.40× 略降至 8192 的 1.25×，全测试区间均保持正向收益；T=8192 时 forward 加速 **2.87×**（0.81 ms → 0.28 ms），backward 加速 **1.84×**（1.03 ms → 0.56 ms）；Forward 路径在短序列上优势更为明显，T=1024 时加速可达 **6.73×**。峰值显存节省 **28.8%**（8192 时 500 MB → 356 MB），低于 RMSNorm 与 CrossEntropy，与 RoPE 中间激活规模较小相符。在 dynamic batch 场景下，单步 token 数越高，相对增益略有上升（与后文图 13、图 14 中 MFU 按 token 四分位统计结果一致）。

<table align="center">
<tr>
<td align="center" width="49%"><img src="../assets/Liger-Kernel/benchmark/rope_speed_forward_token_length.png" alt="RoPE forward-mode latency" width="100%"/><br/><strong>图 5</strong> RoPE forward 模式耗时（随序列长度）</td>
<td align="center" width="49%"><img src="../assets/Liger-Kernel/benchmark/rope_speed_backward_token_length.png" alt="RoPE backward-mode latency" width="100%"/><br/><strong>图 6</strong> RoPE backward 模式耗时（随序列长度）</td>
</tr>
<tr>
<td align="center" width="49%"><img src="../assets/Liger-Kernel/benchmark/rope_speed_full_token_length.png" alt="RoPE full-mode latency" width="100%"/><br/><strong>图 7</strong> RoPE full 模式耗时（随序列长度）</td>
<td align="center" width="49%"><img src="../assets/Liger-Kernel/benchmark/rope_memory_full_token_length.png" alt="RoPE full-mode peak memory" width="100%"/><br/><strong>图 8</strong> RoPE full 模式峰值显存（随序列长度）</td>
</tr>
</table>

#### 3.3.3 CrossEntropy

每 step 调用 1 次（LM head loss），词表规模 vocab=128256。图 9–12 显示，full 模式加速比随 T **单调递增**：1024 时为 1.70×，8192 时为 **1.90×**（22.52 ms → 11.88 ms），与 GSM8K dynamic batch 及较大单步 token 量的训练特征相符。T=8192 时 forward 加速 **2.37×**（8.98 ms → 3.79 ms），backward 加速 **1.67×**（13.58 ms → 8.11 ms）；峰值显存节省稳定在 **40%**（基线绝对值约 20 GB，受词表维度影响）。本次整网实验未启用 `fused_linear_cross_entropy`；该算子及其他 Loss 类 kernel 虽已在 NPU 后端实现，其整网收益可沿用本报告两阶段方法另行验证。

<table align="center">
<tr>
<td align="center" width="49%"><img src="../assets/Liger-Kernel/benchmark/cross_entropy_speed_forward_token_length.png" alt="CrossEntropy forward-mode latency" width="100%"/><br/><strong>图 9</strong> CrossEntropy forward 模式耗时（随序列长度）</td>
<td align="center" width="49%"><img src="../assets/Liger-Kernel/benchmark/cross_entropy_speed_backward_token_length.png" alt="CrossEntropy backward-mode latency" width="100%"/><br/><strong>图 10</strong> CrossEntropy backward 模式耗时（随序列长度）</td>
</tr>
<tr>
<td align="center" width="49%"><img src="../assets/Liger-Kernel/benchmark/cross_entropy_speed_full_token_length.png" alt="CrossEntropy full-mode latency" width="100%"/><br/><strong>图 11</strong> CrossEntropy full 模式耗时（随序列长度）</td>
<td align="center" width="49%"><img src="../assets/Liger-Kernel/benchmark/cross_entropy_memory_full_token_length.png" alt="CrossEntropy full-mode peak memory" width="100%"/><br/><strong>图 12</strong> CrossEntropy full 模式峰值显存（随序列长度）</td>
</tr>
</table>

### 3.4 算子层小结

**表 6** 算子层评估小结

| 维度 | 结论 |
|------|------|
| 加速 | 样例 kernel 在 full 模式下加速比为 **1.25×～1.90×**，表明 Ascend 后端具备接入价值 |
| 显存 | 样例中 RMSNorm **68.4%** > CrossEntropy **40%** > RoPE **28.8%** |
| 与整网衔接 | 以上三项作为本次 Liger-Kernel 整网 patch 配置，进入第 4 节 verl 验证 |

## 4 整网评估

**目标：** 在完整 LoRA SFT 训练链路中，对比启用与关闭 Liger-Kernel 的端到端差异，观测其在吞吐、显存及精度上的综合收益。本节关注框架级接入效果；patch 算子清单见表 8。

### 4.1 verl 与 Liger-Kernel 接入

[verl](https://github.com/verl-project/verl)（Volcano Engine Reinforcement Learning for LLMs，HybridFlow 开源实现）是面向大语言模型后训练的开源框架，统一支持 SFT、RLHF、DPO 等任务。与第 3 节独立的算子 micro-benchmark 不同，整网实验需在 **真实训练栈** 中验证 Liger-Kernel 接入后，收益能否稳定传递至 step 级指标。

本次实验采用 verl 的 **SFTTrainer** 路径：在 Qwen3-8B 基座模型上执行 GSM8K 监督微调，在 **4 卡 NPU** 上通过 **FSDP** 进行参数分片、梯度同步与显存管理，通过 **LoRA（rank=32）** 仅更新低秩 adapter，其余 base 权重保持冻结。该配置代表当前 NPU 上较常见的多卡参数高效微调场景。

Liger-Kernel 的整网接入由 `use_liger=True` 触发：verl 在模型初始化阶段调用 `apply_liger_kernel_to_qwen3`，将模型前向/反向中的部分原生算子替换为 Ascend 后端实现（见表 8）。替换发生在 **模型计算图内部**，不改变 verl 的数据加载、优化器调度及 FSDP 通信逻辑；因此，整网对比的变量为 **Liger-Kernel 是否启用**，其余训练链路保持一致。

LoRA 冻结 base 权重后，MatMul、Attention 等大算子仍占 step 主体耗时；本次 patch 覆盖的 Norm、RoPE、CE 等路径虽非可训练参数主体，但在每 step 的前向与反向中 **必经且高频执行**。第 3 节以代表性 kernel 说明了单点收益；本节进一步检验 **Liger-Kernel 整体接入** 在 verl 全链路下的吞吐、显存、loss 与实验可重复性。

### 4.2 实验设计

**表 7** verl LoRA SFT 训练配置

| 项目 | 配置 |
|------|------|
| 框架 | verl SFTTrainer + **FSDP + LoRA（rank=32）** |
| 并行 | 1 node × **4 NPU**（FSDP，见表 3） |
| 数据集 | GSM8K，dynamic batch |
| 训练量 | 20 epoch，**580 step** |
| 日志来源 | `use_liger_rms_rope_ce.log`（实验组）、`no_liger.log`（对照组） |

**表 8** 本次整网实验的 Liger-Kernel patch 配置

| Kernel | 启用 | NPU 后端 | 替换对象 |
|--------|----------|----------|----------|
| **rms_norm** | 是 | `LigerRMSNormFunction` | HF Qwen3RMSNorm |
| **rope** | 是 | `LigerRopeFunction` | HF apply_rotary_pos_emb |
| **cross_entropy** | 是 | `LigerCrossEntropyFunction` | torch CrossEntropyLoss |
| **fused_linear_cross_entropy** | 否 | — | LM Head + CE 融合；verl 自有实现，本次未替换 |
| **其他 Ascend 算子** | 否 | — | 表 1 所列算子可按需单独评估，未纳入本次整网对比 |

**表 9** 整网实验对比方案

| 组别 | 配置 |
|------|------|
| **实验组** | `use_liger=True`，启用表 8 所列 patch |
| **对照组** | 相同 verl / LoRA 配置，`use_liger=False` |
| **有效性校验** | 580/580 step 的 `global_tokens` 完全一致 |

整网层面的收益幅度受 LoRA 计算图组成及本次 patch 范围影响，算子级加速向 MFU 的传递机制见第 5 节。

### 4.3 评估指标

整网对比使用 verl 训练日志中的逐步原始记录，未做平滑或截断。

**表 10** 整网评估指标

| 指标 | 含义 | 期望方向 |
|------|------|------|
| `train/mfu` | NPU 算力利用率（Model FLOPs Utilization） | 越高表示吞吐越好 |
| `perf/max_memory_allocated_gb` | NPU 实际分配显存峰值 | 越低越好 |
| `perf/max_memory_reserved_gb` | NPU 预留显存峰值 | 越低越好 |
| `perf/cpu_memory_used_gb` | Host 侧 CPU 内存占用 | 越低越好 |
| `train/loss`、`val/loss` | 训练/验证损失 | 与对照组等价或更优 |
| `train/global_tokens` | 每 step 参与训练的 token 数 | 两组应对齐，以保证对比因果有效 |

### 4.4 详细分析

**吞吐（MFU）：** 启用 Liger-Kernel 后，实验组 MFU 相对对照组提升约 **4%**，与第 3 节样例 kernel 的加速方向一致。图 13、图 14 显示，两条曲线自 step 2 起分离，实验组稳定运行于较高区间；step 1 受编译与预热影响，不宜作为稳态对比依据。剔除 step 1 后，MFU 中位数为 **0.7514 vs 0.7216（+4.14%）**，579 步中 574 步实验组更高。按 `global_tokens` 四分位统计，MFU 相对提升由低 token 步的 **+3.88%** 增至高 token 步的 **+4.48%**，与样例中 CrossEntropy、RoPE 在长序列下的单点收益特征一致。

<table align="center">
<tr>
<td align="center" width="49%"><img src="../assets/Liger-Kernel/verl-sft/train_mfu.png" alt="End-to-end MFU" width="100%"/><br/><strong>图 13</strong> 整网 MFU 逐步曲线</td>
<td align="center" width="49%"><img src="../assets/Liger-Kernel/verl-sft/train_mfu_skip_step1.png" alt="MFU excluding step 1" width="100%"/><br/><strong>图 14</strong> 整网 MFU 逐步曲线（剔除 step 1）</td>
</tr>
</table>

**NPU 显存：** allocated 峰值 **12.421 GB vs 12.476 GB（-0.44%）**，逐步曲线近乎重合（图 15）；reserved 峰值均为 **46.93 GB**（图 16）。整网 NPU 显存仍主要由 Attention、MatMul 等 **未纳入本次 Liger patch** 的路径主导，故节省幅度低于第 3 节 micro-benchmark 中的单点结果，属预期现象。

<table align="center">
<tr>
<td align="center" width="49%"><img src="../assets/Liger-Kernel/verl-sft/perf_max_memory_allocated_gb.png" alt="NPU allocated memory" width="100%"/><br/><strong>图 15</strong> NPU allocated 显存逐步曲线</td>
<td align="center" width="49%"><img src="../assets/Liger-Kernel/verl-sft/perf_max_memory_reserved_gb.png" alt="NPU reserved memory" width="100%"/><br/><strong>图 16</strong> NPU reserved 显存逐步曲线</td>
</tr>
</table>

**Host 内存：** 实验组自训练初期即低于对照组，末步为 **96.26 GB vs 122.29 GB（-21.29%）**，绝对差约 26 GB（图 17）。该指标无法由 isolated micro-benchmark 直接解释，属于 **Liger-Kernel 接入后** 训练栈层面的观测现象，对长周期 LoRA SFT 的资源规划具有实际意义。

<p align="center">
<img src="../assets/Liger-Kernel/verl-sft/perf_cpu_memory_used_gb.png" alt="Host CPU memory" width="66%"/><br/>
<strong>图 17</strong> Host CPU 内存逐步曲线
</p>

**精度与收敛：** 训练 loss 曲线整体重合，实验组末段略低（图 18）；step 580 处 **2.479 vs 2.564（-3.32%）**，逐步平均绝对差（MAD）为 **0.038**；验证 loss **2.558 vs 2.644（-3.27%）**。梯度范数曲线形态一致，未见异常尖峰（图 19），表明 **Liger-Kernel 接入** 未引入可观测的数值不稳定或收敛劣化。

<table align="center">
<tr>
<td align="center" width="49%"><img src="../assets/Liger-Kernel/verl-sft/train_loss.png" alt="Training loss" width="100%"/><br/><strong>图 18</strong> 训练 loss 逐步曲线</td>
<td align="center" width="49%"><img src="../assets/Liger-Kernel/verl-sft/train_grad_norm.png" alt="Gradient norm" width="100%"/><br/><strong>图 19</strong> 梯度范数逐步曲线</td>
</tr>
</table>

**实验有效性：** 580/580 step 的 `global_tokens` 完全一致（图 20），累计 token 均为 0.03065 B，可排除 batch 配置差异对对比结果的干扰。

<p align="center">
<img src="../assets/Liger-Kernel/verl-sft/train_global_tokens.png" alt="global_tokens alignment" width="66%"/><br/>
<strong>图 20</strong> `global_tokens` 逐步对齐曲线
</p>

### 4.5 结果汇总

**表 11** 启用 Liger-Kernel 后的 verl LoRA SFT 整网指标

| 指标 | 实验组 | 对照组 | **相对变化** | 备注 |
|------|--------|--------|--------------|------|
| **MFU 中位数** | 0.7514 | 0.7216 | **+4.14%** | step 2 起持续高于对照组 |
| **MFU 均值** | 0.7120 | 0.6835 | **+4.16%** | 579 步中 574 步实验组更高 |
| **NPU allocated 峰值** | 12.421 GB | 12.476 GB | **-0.44%** | 逐步中位差约 -56 MB |
| **NPU reserved 峰值** | 46.93 GB | 46.93 GB | 持平 | — |
| **Host CPU 内存（末步）** | 96.26 GB | 122.29 GB | **-21.29%** | 整网实验方可观测 |
| **train/loss（step 580）** | 2.479 | 2.564 | **-3.32%** | 曲线整体重合 |
| **val/loss（step 580）** | 2.558 | 2.644 | **-3.27%** | 未出现精度劣化 |
| **global_tokens 对齐** | 580/580 | 580/580 | **100%** | 对比具有因果有效性 |

### 4.6 整网小结

**表 12** 整网评估小结

| 维度 | 结论 |
|------|------|
| 吞吐 | MFU 相对提升 **4.14%** |
| NPU 显存 | allocated 略优，reserved 持平 |
| Host 内存 | 末步相对下降 **21.29%** |
| 精度 | loss 与基线等价，部分指标略优 |
| 对比有效性 | token 逐步对齐率 **100%** |

## 5 算子层与整网层收益关联

第 3 节样例 kernel 的 **1.69×～1.90×** 单点加速，与整网 **+4% MFU** 之间的数量关系，有助于理解 **Liger-Kernel 在 LoRA SFT 场景中的收益边界**：整网收益不仅取决于单个算子多快，还取决于 patch 范围与计算图占比。

### 5.1 Patch 范围与 LoRA 计算图

**本次 patch 覆盖路径：** RMSNorm（72 次/step）、RoPE（36 次/step）、CrossEntropy（1 次/step）。

**未纳入本次 patch 的路径：** QKV/O 投影 MatMul、FFN MLP、Attention、LM Head 等；LoRA adapter 主要挂载于 Linear 层，上述算子仍构成单 step 耗时的主体部分。Ascend 后端支持的更多算子（表 1）若后续纳入 patch，整网 MFU 上限可能随之变化。

### 5.2 收益衰减机制

- 以本次 patch 涉及的样例 kernel 估算，T=8192 下相关子系统累计可节省约 **35.7%** 耗时，但该子系统约占整 step 耗时的 **12%**（Amdahl 定律）。
- 据此推算整 step 理论提升约 **4.3%**，与实测 MFU **+4.14%** 及图 13、图 14 所示曲线相符。
- LoRA 进一步降低可训练参数路径在计算图中的占比；扩大 Liger patch 范围或采用全参数 SFT 时，MFU 提升上限可能高于本次观测值。

### 5.3 整网实验的增量价值

**表 13** 算子层与整网层可观测项对照

| 观测项 | Micro-Benchmark | verl LoRA SFT |
|--------|-----------------|---------------|
| 单 kernel 加速比 | 可测 | — |
| 整网 MFU 幅度 | 可估算 | **+4.14%**（实测） |
| Host CPU 内存 | 不可测 | **-21.29%**（实测） |
| 训练 loss 等价性 | 不可测 | MAD **0.038**（实测） |

算子层评估回答 **Liger Ascend 实现是否具备单点替换价值**；整网实验回答 **在 verl 中启用 Liger-Kernel 是否稳定、综合收益如何**。二者相互补充，构成从 Kernel 到训练框架的完整收益链条。

## 6 结论

本报告在 **Qwen3-8B + verl LoRA SFT（4 卡 FSDP）+ Atlas 800T A3(x86)** 配置下，采用「算子 micro-benchmark → verl 整网训练」两阶段方法，评估 **Liger-Kernel 在昇腾训练中的接入收益**。主要结论如下。

### 6.1 Liger-Kernel 与 NPU 接入

Liger-Kernel [v0.8.0](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.8.0) 的 Ascend 后端已提供较完整的低层算子能力（26 个模块）及 Qwen3 高层 patch 接口；在 verl 中通过 `use_liger=True` 即可在不改动训练调度逻辑的前提下完成接入。本次整网实验采用 Qwen3 上常见的 **rms_norm、rope、cross_entropy** patch 作为验证配置；**fused_linear_cross_entropy** 因 verl 自有实现而未纳入，表 1 中其余算子亦未在本次实验中一并开启。

### 6.2 算子层结论

以 **rms_norm、rope、cross_entropy** 为样例，T=8192、full 模式下相对基线的加速比分别为 **1.69× / 1.25× / 1.90×**，峰值 NPU 显存节省 **28.8%～68.4%**（见表 5）。其中 RMSNorm 反向路径收益最为突出（backward **5.46×**），与 LoRA SFT 需对冻结 base 权重执行完整反向的计算特征相符。样例结果表明，Liger Ascend 后端 **具备单点替换价值**，支持进入 verl 整网验证。

### 6.3 整网层结论

在 580 step、4 卡 FSDP、global_tokens 100% 对齐的对照实验中，**启用 Liger-Kernel** 后：

- **吞吐：** MFU 中位数相对提升 **4.14%**（0.7514 vs 0.7216），579 步中 574 步实验组更高，稳态曲线自 step 2 起持续分离（见表 11、图 13–14）；
- **NPU 显存：** allocated 峰值略降 **0.44%**，reserved 持平，整网显存仍由未 patch 的大算子路径主导；
- **Host 内存：** 末步 CPU 内存 **96.26 GB vs 122.29 GB（-21.29%）**，为整网实验方可观测的显著收益（见表 13）；
- **精度：** train/loss、val/loss 曲线与对照组整体重合，末步略优（-3.3% 量级），梯度范数形态一致，未见数值不稳定或收敛劣化。

综合而言，在 verl LoRA SFT 整网链路中，**Liger-Kernel 接入未引入可观测的精度风险**，并在吞吐与 Host 内存方面带来稳定收益。

### 6.4 两层结论的一致性

第 3 节样例 kernel 的 **1.69×～1.90×** 单点加速，经 Amdahl 定律折算后，理论整 step MFU 提升约 **4.3%**，与实测 **+4.14%** 一致（见 5.2 节）。这表明：整网 MFU 增益虽小于单 kernel 加速比，但数量关系合理，并非实现缺陷或测量异常所致。算子层验证 **Ascend 后端能力**，整网层验证 **verl 接入效果**；两层结论相互支撑，支持在本场景下采用 Liger-Kernel。

## 7 讨论

### 7.1 实践建议

- 在 **Qwen3-8B + verl + Atlas 800T A3(x86) + 4 卡 LoRA SFT** 配置下，建议通过 `use_liger=True` **接入 Liger-Kernel**；本次验证采用的 **rms_norm + rope + cross_entropy** patch 即可在较低成本下获得约 **4% MFU 提升** 及显著的 Host 内存节省。
- 对于 **FusedLinearCrossEntropy、GRPO、FusedMoE** 等已在 NPU 后端实现、尚未完成整网验证的算子，建议沿用本报告 **「micro-benchmark → verl LoRA SFT」** 两阶段流程：先确认单点性能与显存收益，再在真实训练栈中验证 MFU、内存与 loss 稳定性，再决定是否扩大 patch 范围。
- **全参数 SFT、多节点 / 更大规模集群、RLHF / DPO** 等场景尚未覆盖；可复用本报告方法评估 Liger-Kernel 收益，但 MFU 幅度可能因 patch 范围与计算图组成不同而与本次 4 卡 LoRA 场景存在差异（见 5.2 节）。

### 7.2 Host 内存下降的可能机制

整网实验中 Host CPU 内存相对下降 **21.29%**，无法由第 3 节 isolated micro-benchmark 直接解释，属于 **Liger-Kernel 接入后** 训练栈层面的观测现象。可能原因包括：融合 kernel 减少了中间 tensor 在 Host 侧的暂存与拷贝、verl/FSDP 数据管线与 in-place 算子路径的协同，以及 Python 侧对象生命周期差异。该现象对长周期 LoRA SFT 的资源规划具有实际意义，但具体归因有待结合 profiler 与内存快照进一步分析；本次报告将其作为 Liger-Kernel 整网接入的 **附加观测收益** 记录，不作为算子层结论的推导前提。

### 7.3 局限性与后续工作

- **实验范围：** 限于 **4 卡单节点** LoRA SFT 及表 8 所列 patch 配置；MFU 为算力利用率代理指标，未报告 wall-clock step 耗时，未单独量化 HCCL 通信开销对收益的上限影响。
- **任务指标：** 未包含 GSM8K exact match 等下游任务指标；loss 等价性基于训练/验证曲线对比，不能替代任务级精度验收。
- **算子覆盖：** 第 3、4 节以三类代表性 kernel 为例，**未覆盖表 1 全部 Ascend 算子**；FusedLinearCrossEntropy、GRPO 等与 verl 的集成取舍需单独评估。
- **后续方向：** 可在全参数 SFT、多节点集群及 RLHF 场景复现两阶段评估；逐步扩大 Liger patch 范围并量化边际收益；对 Host 内存收益做定向 profiling。

---

*数据可用性.* Liger 0.8.0、commit [`8020e69`](https://github.com/linkedin/Liger-Kernel/commit/8020e691d4b78be6cc4868b96e5c73ca3c1058ea)；verl commit [`c131c70`](https://github.com/verl-project/verl/commit/c131c704db5b2e2dadc7576edcad0e6f4a22c669)；Atlas 800T A3(x86) micro-benchmark 与 **4 卡** verl LoRA SFT 原始训练日志。整网指标均基于逐步原始记录统计，未做平滑或截断。

## 参考文献

[1] LinkedIn. *Liger-Kernel*. https://github.com/linkedin/Liger-Kernel

[2] verl-project. *verl*. https://github.com/verl-project/verl
