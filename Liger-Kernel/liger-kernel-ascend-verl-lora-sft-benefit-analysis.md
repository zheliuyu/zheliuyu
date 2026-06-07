<h1 align="center">从 Kernel 到 verl LoRA SFT：Liger-Kernel 在昇腾训练中的收益分析</h1>

<p align="center">
<a href="https://github.com/zheliuyu">zheliuyu</a><sup>1</sup>,
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
<a href="https://github.com/pillumina">pillumina</a>,
<a href="https://github.com/xuedinge233">xuedinge233</a>, and
<a href="https://github.com/Tcc0403">Tcc0403</a>
<br><br>
<sup>1</sup> <em>Corresponding author.</em><br>
<em>Authors are GitHub contributors to NPU-related work in Liger-Kernel.<br>
Corresponding author listed first; all other authors ranked by merged NPU-related PR count (ties broken alphabetically).</em>
</p>

---

## 目录

1. [一、概述](#一概述)
2. [二、实验设置](#二实验设置)
3. [三、算子层评估（Micro-Benchmark）](#三算子层评估micro-benchmark)
4. [四、整网层评估（verl LoRA SFT）](#四整网层评估verl-lora-sft)
5. [五、算子层与整网层收益的关联分析](#五算子层与整网层收益的关联分析)
6. [六、结论](#六结论)
7. [七、讨论](#七讨论)

---

## 一、概述

### 1.1 Liger-Kernel 简介

[Liger-Kernel](https://github.com/linkedin/Liger-Kernel) 是面向大语言模型训练的融合算子库，基于 Triton 在 GPU 侧实现 RMSNorm、RoPE、CrossEntropy 等高频算子的融合计算与 in-place 优化，目标是在保持计算语义不变的前提下降低显存占用并提升训练吞吐。主要接入方式包括：

- **高层 API（Patching）：** 对 HuggingFace 模型调用 `apply_liger_kernel_to_*`（如 Qwen3 对应 `apply_liger_kernel_to_qwen3`），按模块替换原生实现；
- **低层 API：** 直接调用 `LigerRMSNorm`、`LigerCrossEntropyLoss` 等算子类。

在 verl 等训练框架中，可通过 `use_liger=True` 触发 monkey patch，在不修改模型结构的前提下完成接入。

### 1.2 昇腾 NPU 支持情况

Liger-Kernel 在 [v0.8.0 release](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.8.0) 中正式提供 NPU 原生支持。运行时，NPU 侧采用独立 Ascend 后端（`liger_kernel.ops.backends._ascend`），在检测到 NPU 设备后自动注册并替换默认 CUDA/Triton 实现；依赖栈为 **torch 2.6 + torch_npu 2.6.0 + triton-ascend 3.2.0**（见项目 `setup.py`）。

**表 1  Ascend 后端低层算子支持列表（节选）**

| 类别 | 已支持算子（节选） |
|------|-------------------|
| **Norm / 激活** | RMSNorm、FusedAddRMSNorm、LayerNorm、GroupNorm、DyT、PolyNorm |
| **Attention 相关** | RoPE、Llama4 RoPE、Qwen2-VL MRoPE、Softmax、Sparsemax |
| **MLP / MoE** | GeGLU、FusedMoE |
| **Loss / 对齐** | CrossEntropy、FusedLinearCrossEntropy、GRPO Loss、JSD、KL Div、TVD |
| **其他** | Embedding、AttnRes、mHC 等 |

**高层 Patching：** Qwen3 等架构可通过 `apply_liger_kernel_to_qwen3` 按需启用 `rms_norm`、`rope`、`cross_entropy`、`fused_linear_cross_entropy` 等模块。

### 1.3 研究目标

在 **Qwen3-8B + verl LoRA SFT（4 卡 FSDP）+ Atlas 800T A3(x86)** 场景下，对 Liger-Kernel 的收益评估需同时覆盖算子级基准测试与整网训练观测，不宜仅依据单一层次的指标下结论。本报告按 **「单 Kernel 能力评估 → verl LoRA 整网验证」** 两条主线组织：

1. **算子层（Micro-Benchmark）：** 评估本次启用的 RMSNorm、RoPE、CrossEntropy 在 NPU 上相对基线实现的 **性能与显存收益是否满足接入条件**；
2. **整网层（verl End-to-End）：** 评估上述算子接入真实 LoRA SFT 链路后，在 **吞吐（MFU）、显存占用及 loss 收敛** 等方面的端到端表现。

**表 2  主要结果摘要（GSM8K SFT，4 卡，580 step）**

| 层次 | 主要收益 | 关键数据 |
|------|----------|----------|
| **算子层** | 单 kernel 加速与显存节省 | RMSNorm **1.69×** / 节省 **68.4%**；RoPE **1.25×** / 节省 **28.8%**；CE **1.90×** / 节省 **40.0%**（T=8192） |
| **verl LoRA SFT** | 吞吐、内存与精度 | MFU **+4.14%**；Host CPU 内存 **-21.29%**（末步）；val/loss **-3.27%**；token 对齐 **100%** |

---

## 二、实验设置

列出算子 micro-benchmark 与 verl LoRA SFT 整网实验 **共用** 的硬件、软件及模型相关配置；各实验的具体方案分别见第三节与第四节。

**表 3  实验环境与配置**

| 项目 | 配置 |
|------|------|
| 硬件平台 | Atlas 800T A3(x86) |
| 整网训练并行 | 1 node × **4 NPU**（FSDP，详见表 7） |
| Liger-Kernel | [`8020e69`](https://github.com/linkedin/Liger-Kernel/commit/8020e691d4b78be6cc4868b96e5c73ca3c1058ea) |
| verl | [`c131c70`](https://github.com/volcengine/verl/commit/c131c704db5b2e2dadc7576edcad0e6f4a22c669) |
| 计算精度 | bfloat16 |
| 对齐模型 | Qwen3-8B（hidden=4096，GQA 32 heads / 8 kv heads，vocab≈128256） |
| 序列长度上限 | **8192** tokens（SFT 与 benchmark 关键对齐点） |
| 评估算子 | `rms_norm`、`rope`、`cross_entropy` |

---

## 三、算子层评估（Micro-Benchmark）

**分析目标：** 在 NPU 后端上，分别评估各 kernel 相对基线实现的单点性能与显存收益，为整网接入提供依据。

### 3.1 实验设计

**表 4  算子 micro-benchmark 实验设计**

| 项目 | 配置 |
|------|------|
| 测试框架 | Liger-Kernel 官方 benchmark（`benchmark/data/all_benchmark_data.csv`） |
| 运行设备 | 单 NPU（算子 isolated 测试，与整网 4 卡配置无关） |
| 序列长度 sweep | T = 1024 / 2048 / 4096 / **8192** |
| 测试模式 | forward / backward / **full**（报告以 full 模式为主） |
| RMSNorm 基线 | HuggingFace Qwen3RMSNorm |
| RoPE 基线 | HuggingFace `apply_rotary_pos_emb` |
| CrossEntropy 基线 | PyTorch `CrossEntropyLoss` |
| 对比方式 | 相同 shape、相同 T 下，Liger Ascend 实现 vs 基线实现 |

### 3.2 T=8192 综合结果（full 模式）

**表 5  T=8192 算子 micro-benchmark 综合结果（full 模式）**

| Kernel | 基线 | Liger | **加速比** | **显存节省** |
|--------|------|-------|------------|--------------|
| RMSNorm | 7.40 ms | 4.39 ms | **1.69×** | **68.4%** |
| RoPE | 5.19 ms | 4.14 ms | **1.25×** | **28.8%** |
| CrossEntropy | 22.52 ms | 11.88 ms | **1.90×** | **40.0%** |

### 3.3 单算子分析

#### RMSNorm

Qwen3 每个 decoder 层包含 2 次 RMSNorm 调用，36 层合计 **72 次/step**，为三项 kernel 中调用频次最高者。T=8192、full 模式下加速比为 **1.69×**，峰值显存节省 **68.4%**。LoRA 不改变各层 Norm 的执行次数，该路径的算子级收益较易传递至整网层面。

图 1 给出 full 模式耗时随序列长度的变化：Liger 实现在全 T 区间内均低于 HuggingFace 基线；T 由 1024 增至 8192 时，加速比由 1.58× 升至 1.69×，与表 3 所列序列长度上限 **8192** 相对应。

![rms_norm_speed_full_token_length.png](../assets/Liger-Kernel/benchmark/rms_norm_speed_full_token_length.png)

**图 1  rms_norm_speed_full_token_length.png**

反向路径收益更为显著：T=8192 时 backward 加速 **5.46×**（3.72 ms → 0.68 ms）。LoRA SFT 仍须对冻结的 base 权重执行完整前向与反向计算，Norm 反向路径的优化可直接反映为 step 耗时下降（见图 2）。

![rms_norm_speed_backward_token_length.png](../assets/Liger-Kernel/benchmark/rms_norm_speed_backward_token_length.png)

**图 2  rms_norm_speed_backward_token_length.png**

峰值显存方面，四个 T 测试点的节省比例均为 **68.4%**（8192 时 1216 MB → 384 MB），为三项 kernel 中最高（见图 3）。

![rms_norm_memory_full_token_length.png](../assets/Liger-Kernel/benchmark/rms_norm_memory_full_token_length.png)

**图 3  rms_norm_memory_full_token_length.png**

#### RoPE

每层 attention 前调用 1 次，合计 **36 次/step**。T=8192、full 模式加速比为 **1.25×**。图 4 显示，随 T 增大，加速比由 1024 的 1.40× 略降至 8192 的 1.25×，全测试区间均保持正向收益；在 dynamic batch 场景下，单步 token 数越高，相对增益略有上升（与后文图 10、图 11 中 MFU 按 token 四分位统计结果一致）。

![rope_speed_full_token_length.png](../assets/Liger-Kernel/benchmark/rope_speed_full_token_length.png)

**图 4  rope_speed_full_token_length.png**

Forward 路径在短序列上优势更为明显：T=1024 时加速可达 **6.73×**（图 5）。峰值显存节省 **28.8%**（8192 时 500 MB → 356 MB），低于 RMSNorm 与 CrossEntropy，与 RoPE 中间激活规模较小相符（图 6）。

![rope_speed_forward_token_length.png](../assets/Liger-Kernel/benchmark/rope_speed_forward_token_length.png)

**图 5  rope_speed_forward_token_length.png**

![rope_memory_full_token_length.png](../assets/Liger-Kernel/benchmark/rope_memory_full_token_length.png)

**图 6  rope_memory_full_token_length.png**

#### CrossEntropy

每 step 调用 1 次（LM head loss），词表规模 vocab=128256。图 7 显示，full 模式加速比随 T **单调递增**：1024 时为 1.70×，8192 时为 **1.90×**（22.52 ms → 11.88 ms），与 GSM8K dynamic batch 及较大单步 token 量的训练特征相符。

![cross_entropy_speed_full_token_length.png](../assets/Liger-Kernel/benchmark/cross_entropy_speed_full_token_length.png)

**图 7  cross_entropy_speed_full_token_length.png**

T=8192 时 forward 加速 **2.37×**（图 8）；峰值显存节省稳定在 **40%**（基线绝对值约 20 GB，受词表维度影响，见图 9）。本次实验未启用 `fused_linear_cross_entropy`；该算子虽已在 NPU 后端实现，其整网收益有待后续 LoRA SFT 实验验证。

![cross_entropy_speed_forward_token_length.png](../assets/Liger-Kernel/benchmark/cross_entropy_speed_forward_token_length.png)

**图 8  cross_entropy_speed_forward_token_length.png**

![cross_entropy_memory_full_token_length.png](../assets/Liger-Kernel/benchmark/cross_entropy_memory_full_token_length.png)

**图 9  cross_entropy_memory_full_token_length.png**

### 3.4 算子层小结

**表 6  算子层评估小结**

| 维度 | 结论 |
|------|------|
| 加速 | 三项 kernel full 模式加速比为 **1.25×～1.90×**，满足接入条件 |
| 显存 | RMSNorm **68.4%** > CrossEntropy **40%** > RoPE **28.8%** |
| 整网实验选型 | 启用 **rms_norm + rope + cross_entropy** 进入 verl 整网验证 |

---

## 四、整网层评估（verl LoRA SFT）

**分析目标：** 在完整 LoRA SFT 训练链路中，观测算子级收益在吞吐、显存及精度指标上的端到端表现。

### 4.1 verl 训练框架与 Liger 接入

[verl](https://github.com/volcengine/verl)（Volcano Engine Reinforcement Learning）是面向大语言模型后训练的开源框架，统一支持 SFT、RLHF、DPO 等任务。与第三节独立的算子 micro-benchmark 不同，整网实验需在 **真实训练栈** 中验证 Liger-Kernel 的收益能否稳定传递至 step 级指标。

本次实验采用 verl 的 **SFTTrainer** 路径：在 Qwen3-8B 基座模型上执行 GSM8K 监督微调，在 **4 卡 NPU** 上通过 **FSDP** 进行参数分片、梯度同步与显存管理，通过 **LoRA（rank=32）** 仅更新低秩 adapter，其余 base 权重保持冻结。该配置代表当前 NPU 上较常见的多卡参数高效微调场景。

Liger-Kernel 的整网接入由 `use_liger=True` 触发：verl 在模型初始化阶段调用 `apply_liger_kernel_to_qwen3`，按配置将 HuggingFace 原生的 RMSNorm、RoPE 与 CrossEntropy 路径替换为 Ascend 后端实现。算子替换发生在 **模型前向/反向图内部**，不改变 verl 的数据加载、优化器调度及 FSDP 通信逻辑；因此，整网对比的变量仅为上述三项 kernel 的启用与否，其余训练链路保持一致。

LoRA 冻结 base 权重后，MatMul、Attention 等大算子仍占 step 主体耗时；Norm、RoPE、CE 虽非可训练参数路径，但在每 step 的前向与反向中 **必经且高频执行**。第三节已确认三项 kernel 的单点收益，本节进一步检验其在 verl 全链路下的 **吞吐、显存、loss 与实验可重复性**。

### 4.2 实验设计

**表 7  verl LoRA SFT 训练配置**

| 项目 | 配置 |
|------|------|
| 框架 | verl SFTTrainer + **FSDP + LoRA（rank=32）** |
| 并行 | 1 node × **4 NPU**（FSDP，见表 3） |
| 数据集 | GSM8K，dynamic batch |
| 训练量 | 20 epoch，**580 step** |
| 日志来源 | `use_liger_rms_rope_ce.log`（实验组）、`no_liger.log`（对照组） |

**表 8  整网实验启用的 Liger Kernel**

| Kernel | 是否启用 | NPU 后端 | 替换对象 |
|--------|----------|----------|----------|
| **rms_norm** | 是 | `LigerRMSNormFunction` | HF Qwen3RMSNorm |
| **rope** | 是 | `LigerRopeFunction` | HF apply_rotary_pos_emb |
| **cross_entropy** | 是 | `LigerCrossEntropyFunction` | torch CrossEntropyLoss |
| **fused_linear_cross_entropy** | 否 | — | LM Head + CE 融合；verl 自有实现，本次未替换 |

**表 9  整网实验对比方案**

| 组别 | 配置 |
|------|------|
| **实验组（Liger）** | 启用 `rms_norm + rope + cross_entropy` |
| **对照组（Baseline）** | 相同 verl / LoRA 配置，Liger kernel 全部关闭 |
| **有效性校验** | 580/580 step 的 `global_tokens` 完全一致 |

整网层面的收益幅度受 LoRA 计算图组成影响，算子级加速向 MFU 的传递机制见第五节。

### 4.3 评估指标

整网对比从 verl 训练日志中提取逐步（step-wise）原始记录，未做平滑或截断。主要观测指标如下。

**表 10  整网评估指标说明**

| 指标 | 含义 | 期望 |
|------|------|------|
| `train/mfu` | NPU 算力利用率（Model FLOPs Utilization） | 越高表示吞吐越好 |
| `perf/max_memory_allocated_gb` | NPU 实际分配显存峰值 | 越低越好 |
| `perf/max_memory_reserved_gb` | NPU 预留显存峰值 | 越低越好 |
| `perf/cpu_memory_used_gb` | Host 侧 CPU 内存占用 | 越低越好 |
| `train/loss`、`val/loss` | 训练/验证损失 | 与对照组等价或更优 |
| `train/global_tokens` | 每 step 参与训练的 token 数 | 两组应对齐，以保证对比因果有效 |

### 4.4 分项分析

**吞吐（MFU）：** 实验组 MFU 相对对照组提升约 **4%**，与算子层加速方向一致。图 10、图 11 显示，两条曲线自 step 2 起分离，实验组稳定运行于较高区间；step 1 受编译与预热影响，不宜作为稳态对比依据。剔除 step 1 后，MFU 中位数为 **0.7514 vs 0.7216（+4.14%）**，579 步中 574 步实验组更高。按 `global_tokens` 四分位统计，MFU 相对提升由低 token 步的 **+3.88%** 增至高 token 步的 **+4.48%**，与 CrossEntropy、RoPE 在长序列下的算子级收益特征一致。

![train_mfu.png](../assets/Liger-Kernel/verl-sft/train_mfu.png)

**图 10  train_mfu.png**

![train_mfu_skip_step1.png](../assets/Liger-Kernel/verl-sft/train_mfu_skip_step1.png)

**图 11  train_mfu_skip_step1.png**

**NPU 显存：** allocated 峰值 **12.421 GB vs 12.476 GB（-0.44%）**，逐步曲线近乎重合（图 12）；reserved 峰值均为 **46.93 GB**（图 13）。整网 NPU 显存仍主要由 Attention、MatMul 等未替换算子主导，故节省幅度低于算子级 micro-benchmark 中的单点结果，属预期现象。

![perf_max_memory_allocated_gb.png](../assets/Liger-Kernel/verl-sft/perf_max_memory_allocated_gb.png)

**图 12  perf_max_memory_allocated_gb.png**

![perf_max_memory_reserved_gb.png](../assets/Liger-Kernel/verl-sft/perf_max_memory_reserved_gb.png)

**图 13  perf_max_memory_reserved_gb.png**

**Host 内存：** 实验组自训练初期即低于对照组，末步为 **96.26 GB vs 122.29 GB（-21.29%）**，绝对差约 26 GB（图 14）。该指标无法由 NPU 单算子 micro-benchmark 直接推导，属于融合 kernel 与 verl/FSDP 数据管线协同作用下的整网效应，对长周期 LoRA SFT 的资源占用具有实际意义。

![perf_cpu_memory_used_gb.png](../assets/Liger-Kernel/verl-sft/perf_cpu_memory_used_gb.png)

**图 14  perf_cpu_memory_used_gb.png**

**精度与收敛：** 训练 loss 曲线整体重合，实验组末段略低（图 15）；step 580 处 **2.479 vs 2.564（-3.32%）**，逐步平均绝对差（MAD）为 **0.038**；验证 loss **2.558 vs 2.644（-3.27%）**。梯度范数曲线形态一致，未见异常尖峰（图 16），表明算子替换未引入可观测的数值不稳定或收敛劣化。

![train_loss.png](../assets/Liger-Kernel/verl-sft/train_loss.png)

**图 15  train_loss.png**

![train_grad_norm.png](../assets/Liger-Kernel/verl-sft/train_grad_norm.png)

**图 16  train_grad_norm.png**

**实验有效性：** 580/580 step 的 `global_tokens` 完全一致（图 17），累计 token 均为 0.03065 B，可排除 batch 配置差异对对比结果的干扰。

![train_global_tokens.png](../assets/Liger-Kernel/verl-sft/train_global_tokens.png)

**图 17  train_global_tokens.png**

### 4.5 结果汇总

上述分项分析的关键数值汇总如下。

**表 11  verl LoRA SFT 整网指标汇总**

| 指标 | 实验组 | 对照组 | **相对变化** | 说明 |
|------|--------|--------|--------------|------|
| **MFU 中位数** | 0.7514 | 0.7216 | **+4.14%** | step 2 起持续高于对照组 |
| **MFU 均值** | 0.7120 | 0.6835 | **+4.16%** | 579 步中 574 步实验组更高 |
| **NPU allocated 峰值** | 12.421 GB | 12.476 GB | **-0.44%** | 逐步中位差约 -56 MB |
| **NPU reserved 峰值** | 46.93 GB | 46.93 GB | 持平 | — |
| **Host CPU 内存（末步）** | 96.26 GB | 122.29 GB | **-21.29%** | 整网实验方可观测 |
| **train/loss（step 580）** | 2.479 | 2.564 | **-3.32%** | 曲线整体重合 |
| **val/loss（step 580）** | 2.558 | 2.644 | **-3.27%** | 未出现精度劣化 |
| **global_tokens 对齐** | 580/580 | 580/580 | **100%** | 对比具有因果有效性 |

### 4.6 整网层小结

**表 12  整网层评估小结**

| 维度 | 结论 |
|------|------|
| 吞吐 | MFU 相对提升 **4.14%** |
| NPU 显存 | allocated 略优，reserved 持平 |
| Host 内存 | 末步相对下降 **21.29%** |
| 精度 | loss 与基线等价，部分指标略优 |
| 对比有效性 | token 逐步对齐率 **100%** |

---

## 五、算子层与整网层收益的关联分析

算子层 **1.69×～1.90×** 的加速与整网 **+4% MFU** 之间的数量关系，是界定 Liger-Kernel 在 LoRA SFT 场景中收益边界的关键。

### 5.1 替换范围与 LoRA 计算图

**已替换路径：** RMSNorm（72 次/step）、RoPE（36 次/step）、CrossEntropy（1 次/step）。

**未替换路径：** QKV/O 投影 MatMul、FFN MLP、Attention、LM Head 等；LoRA adapter 主要挂载于 Linear 层，上述算子仍构成单 step 耗时的主体部分。

### 5.2 收益幅度的衰减机制

- 三项 kernel 在 T=8192 下累计可节省约 **35.7%** 的子系统耗时，但该子系统约占整 step 耗时的 **12%**（Amdahl 定律估算）。
- 据此推算整 step 理论提升约 **4.3%**，与实测 MFU **+4.14%** 及图 10、图 11 所示 MFU 曲线相符。
- LoRA 进一步降低可训练参数路径在计算图中的占比；在全参数 SFT 场景下，MFU 提升上限可能高于本次观测值。

### 5.3 整网实验的增量观测价值

**表 13  算子层与整网层可观测项对照**

| 观测项 | Micro-Benchmark | verl LoRA SFT |
|--------|-----------------|---------------|
| 单 kernel 加速比 | 可测 | — |
| 整网 MFU 幅度 | 可估算 | **+4.14%**（实测） |
| Host CPU 内存 | 不可测 | **-21.29%**（实测） |
| 训练 loss 等价性 | 不可测 | MAD **0.038**（实测） |

算子层评估回答 **实现是否具备替换价值**；整网实验回答 **替换后在真实训练链路中的稳定性与综合收益**。二者相互补充，缺一不可。

---

## 六、结论

本报告在 **Qwen3-8B + verl LoRA SFT（4 卡 FSDP）+ Atlas 800T A3(x86)** 配置下，采用「算子 micro-benchmark → verl 整网训练」两阶段方法，对 Liger-Kernel Ascend 后端的 **rms_norm、rope、cross_entropy** 三项 kernel 进行了收益评估。主要结论如下。

### 6.1 NPU 后端与接入范围

Liger-Kernel v0.8.0 的 Ascend 后端已提供较完整的低层算子能力（26 个模块）及 Qwen3 高层 patch 接口；在 verl 中通过 `use_liger=True` 即可在不改动训练调度逻辑的前提下完成算子替换。本次 LoRA SFT 实验仅启用 **rms_norm、rope、cross_entropy** 三项，**fused_linear_cross_entropy** 因 verl 自有实现而未纳入对比；其余 MatMul、Attention、FFN 等路径仍使用框架原生实现。

### 6.2 算子层结论

在 T=8192、full 模式下，三项 kernel 相对基线实现的加速比分别为 **1.69× / 1.25× / 1.90×**，峰值 NPU 显存节省 **28.8%～68.4%**（见表 5）。其中 RMSNorm 反向路径收益最为突出（backward **5.46×**），与 LoRA SFT 需对冻结 base 权重执行完整反向的计算特征相符。算子层结果表明，上述三项实现在 NPU 上 **具备替换价值**，满足接入 verl 整网训练的前置条件。

### 6.3 整网层结论

在 580 step、4 卡 FSDP、global_tokens 100% 对齐的对照实验中，启用 Liger kernel 后：

- **吞吐：** MFU 中位数相对提升 **4.14%**（0.7514 vs 0.7216），579 步中 574 步实验组更高，稳态曲线自 step 2 起持续分离（见表 11、图 10–11）；
- **NPU 显存：** allocated 峰值略降 **0.44%**，reserved 持平，整网显存仍由 Attention、MatMul 等未替换算子主导；
- **Host 内存：** 末步 CPU 内存 **96.26 GB vs 122.29 GB（-21.29%）**，为整网实验方可观测的显著收益（见表 13）；
- **精度：** train/loss、val/loss 曲线与对照组整体重合，末步略优（-3.3% 量级），梯度范数形态一致，未见数值不稳定或收敛劣化。

综合而言，在 verl LoRA SFT 整网链路中，Liger kernel 替换 **未引入可观测的精度风险**，并在吞吐与 Host 内存方面带来稳定收益。

### 6.4 两层结论的一致性

算子层 **1.69×～1.90×** 的单点加速，经 Amdahl 定律折算后，理论整 step MFU 提升约 **4.3%**，与实测 **+4.14%** 一致（见 5.2 节）。这表明：整网 MFU 增益虽小于单 kernel 加速比，但数量关系合理，并非实现缺陷或测量异常所致。算子层评估验证 **「能否替换」**，整网实验验证 **「替换后是否稳定、综合收益如何」**；两层结论相互支撑，支持在本场景下启用 Liger-Kernel。

---

## 七、讨论

### 7.1 实践建议

- 在 **Qwen3-8B + verl + Atlas 800T A3(x86) + 4 卡 LoRA SFT** 配置下，建议默认启用 **`rms_norm + rope + cross_entropy`**，以在可接受的接入成本下获得约 **4% MFU 提升** 及显著的 Host 内存节省。
- 对于 **FusedLinearCrossEntropy、GRPO** 等已在 NPU 后端实现、尚未完成整网验证的算子，建议沿用本报告采用的 **「micro-benchmark → verl LoRA SFT」** 分阶段流程：先确认单点性能与显存收益，再在真实训练栈中验证 MFU、内存与 loss 稳定性，再决定是否默认开启。
- **全参数 SFT、多节点 / 更大规模集群、RLHF / DPO** 等场景尚未覆盖；可复用本报告的两阶段评估方法，但 MFU 提升幅度可能因计算图组成不同而与 4 卡 LoRA 场景存在差异（见 5.2 节关于 LoRA 计算图占比的讨论）。

### 7.2 Host 内存下降的可能机制

整网实验中 Host CPU 内存相对下降 **21.29%**，无法由第三节 NPU 单算子 micro-benchmark 直接解释。可能原因包括：融合 kernel 减少了中间 tensor 在 Host 侧的暂存与拷贝、verl/FSDP 数据管线与 in-place 算子路径的协同，以及 Python 侧对象生命周期差异。该现象对长周期 LoRA SFT 的资源规划具有实际意义，但具体归因有待结合 profiler 与内存快照进一步分析；本次报告将其作为整网实验的 **附加观测收益** 记录，不作为算子层结论的推导前提。

### 7.3 局限性与后续工作

- **实验范围：** 限于 **4 卡单节点** LoRA SFT；MFU 为算力利用率代理指标，未报告 wall-clock step 耗时，未单独量化 HCCL 通信开销对收益的上限影响。
- **任务指标：** 未包含 GSM8K exact match 等下游任务指标；loss 等价性基于训练/验证曲线对比，不能替代任务级精度验收。
- **算子覆盖：** FusedLinearCrossEntropy、GRPO 等算子不在本次结论覆盖范围内；verl 侧 fused CE 与 Liger 实现的取舍需单独评估。
- **后续方向：** 可在全参数 SFT、多节点集群及 RLHF 场景复现两阶段评估；对 Host 内存收益做定向 profiling；在确认整网稳定后，逐步扩大默认启用的 kernel 范围。

---

*数据来源：Liger 0.8.0、commit [`8020e69`](https://github.com/linkedin/Liger-Kernel/commit/8020e691d4b78be6cc4868b96e5c73ca3c1058ea)；verl commit [`c131c70`](https://github.com/volcengine/verl/commit/c131c704db5b2e2dadc7576edcad0e6f4a22c669)；Atlas 800T A3(x86) micro-benchmark 与 **4 卡** verl LoRA SFT 原始训练日志。整网指标均基于逐步原始记录统计，未做平滑或截断。*
