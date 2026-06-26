# 从单 Kernel 到 verl LoRA SFT：Liger-Kernel 在昇腾训练中的收益分析

---

## 导读

[Liger-Kernel](https://github.com/linkedin/Liger-Kernel) 面向大语言模型训练，基于 Triton 实现 RMSNorm、RoPE、CrossEntropy 等高频算子的融合计算与 in-place 优化，目标是在保持计算语义不变的前提下降低显存占用并提升训练吞吐。自 v0.8.0 起，项目在 **昇腾 NPU** 上提供独立 Ascend 后端；在 verl 中可通过 `use_liger=True` 触发 monkey patch，在不修改模型结构的前提下替换部分原生算子。

单算子 micro-benchmark 与整网训练表现之间往往存在数量级差异：前者反映 kernel 层能力，后者还受 patch 范围、计算图占比及框架调度影响。本文在 **Qwen3-8B + verl LoRA SFT + Atlas 800T A3** 配置下，采用 **「算子 micro-benchmark → verl 端到端训练」** 两段式方法开展验证；profiling 细节从略，正文聚焦 benchmark 结论与整网对照结果。

**主要结论：**

- **算子层：** 与 verl patch 对齐的三项 kernel（RMSNorm / RoPE / CrossEntropy），T=8192、full 模式下加速比 **1.25×～2.06×**，峰值 NPU 显存节省 **28.8%～68.4%**
- **整网层：** 4 卡 FSDP、580 step、`global_tokens` 100% 对齐 —— MFU 中位数 **+4.14%**（579 步中 574 步更高），Host CPU 内存均值 **-11.35%**、末步 **-21.29%**，train/val loss 与对照组整体重合
- **方法论：** 算子层评估接入可行性；整网实验验证收益能否稳定传递至 step 级指标

---

## 一、Liger-Kernel 在昇腾上的接入方式

### 1.1 Ascend 后端能力

检测到 NPU 设备后，Liger-Kernel 使用 `liger_kernel.ops.backends._ascend` 后端，依赖 **torch 2.6.0 + torch_npu 2.6.0 + triton-ascend 3.2.0**。低层算子已覆盖 Norm、RoPE、CrossEntropy、FusedMoE、GRPO 等 **26 个模块**；Qwen3 可通过 `apply_liger_kernel_to_qwen3` 按需启用 `rms_norm`、`rope`、`cross_entropy` 等模块。

本次整网实验仅启用上述前三项；FusedLinearCrossEntropy、GRPO 等虽已在 NPU 后端实现，尚未纳入 verl 默认 patch，整网收益有待后续验证。

### 1.2 实验配置

| 项目 | 配置 |
|------|------|
| 硬件 | Atlas 800T A3(x86) |
| 算子 benchmark | 单 NPU isolated micro-benchmark |
| 整网训练 | 1 node × 4 NPU（FSDP） |
| 模型 | Qwen3-8B（max_seq=8192，bf16） |
| 框架 | verl SFTTrainer + LoRA（rank=32） |
| 数据 | GSM8K，dynamic batch，20 epoch / 580 step |
| Liger-Kernel | [`3bb3b3f`](https://github.com/linkedin/Liger-Kernel/commit/3bb3b3fae6d0b2356116034a7f0ee1dde0ea71ea) |

对比方案：实验组 `use_liger=True`（rms_norm + rope + cross_entropy），对照组相同 verl / LoRA 配置下关闭 Liger；580/580 step 的 `global_tokens` 完全一致，以保证对比因果有效。

---

## 二、算子层：micro-benchmark 结果

**评估目标：** 与 verl patch 对齐的算子，在 Ascend NPU 上是否具备单点替换价值。

测试框架为 Liger-Kernel 官方 `benchmark/scripts/benchmark_*.py`；序列长度 sweep 1024～8192（与 SFT 上限对齐），指标为 full 模式延迟（p50）与峰值 NPU 显存。

### 2.1 三项核心算子（T=8192，full 模式）

| Kernel | 基线 | Liger | 加速比 | 显存节省 |
|--------|------|-------|--------|----------|
| RMSNorm | 7.40 ms | 4.08 ms | 1.82× | 68.4% |
| CrossEntropy | 22.53 ms | 10.91 ms | 2.06× | 40.0% |
| RoPE | 5.18 ms | 4.14 ms | 1.25× | 28.8% |

### 2.2 分算子分析

**RMSNorm**

Qwen3-8B 共 36 层，每层 2 次 RMSNorm，合计 72 次/step。T=8192 下 full 加速 **1.82×**，峰值显存节省 **68.4%**。

<p align="center">
<img src="../assets/Liger-Kernel/benchmark/rms_norm_speed_full_token_length.png" width="48%"/>
<img src="../assets/Liger-Kernel/benchmark/rms_norm_memory_full_token_length.png" width="48%"/><br/>
<em>图 1  RMSNorm full 延迟与峰值显存</em>
</p>

**CrossEntropy**

每 step 执行 1 次；BT=8192 时 full 加速 **2.06×**，显存节省 **40%**。

<p align="center">
<img src="../assets/Liger-Kernel/benchmark/cross_entropy_speed_full_token_length.png" width="48%"/>
<img src="../assets/Liger-Kernel/benchmark/cross_entropy_memory_full_token_length.png" width="48%"/><br/>
<em>图 2  CrossEntropy full 延迟与峰值显存</em>
</p>

**RoPE**

每 step 36 次；T=8192 下 full 加速 **1.25×**，峰值显存节省 **28.8%**。

<p align="center">
<img src="../assets/Liger-Kernel/benchmark/rope_speed_full_token_length.png" width="48%"/>
<img src="../assets/Liger-Kernel/benchmark/rope_memory_full_token_length.png" width="48%"/><br/>
<em>图 3  RoPE full 延迟与峰值显存</em>
</p>


---

## 三、verl LoRA SFT 对照结果

**评估目标：** Liger-Kernel 接入 4 卡 FSDP + LoRA 全链路后，吞吐、内存与精度指标的变化。

对照数据来自使用 verl 训练日志中的逐步原始记录，未做平滑或截断。评估指标包括 `train/mfu`、`perf/max_memory_*`、`perf/cpu_memory_used_gb`、`train/loss`、`val/loss`、`train/grad_norm` 与 `train/global_tokens`。

### 3.1 指标汇总

| 指标 | 实验组 | 对照组 | 相对变化 |
|------|--------|--------|----------|
| MFU 中位数 | 0.7514 | 0.7216 | +4.14% |
| MFU 均值 | 0.7129 | 0.6843 | +4.17% |
| MFU 更高步数（剔除 step 1） | 574 / 579 | — | 99.1% |
| NPU allocated 峰值 | 12.421 GB | 12.476 GB | -0.44% |
| NPU allocated 差值中位数 | — | — | -56 MB |
| NPU reserved 峰值 | 46.93 GB | 46.93 GB | 持平 |
| Host CPU 内存（均值） | 95.70 GB | 107.95 GB | -11.35% |
| Host CPU 内存（末步） | 96.26 GB | 122.29 GB | -21.29% |
| train/loss（step 580） | 2.479 | 2.564 | -3.32% |
| val/loss（step 580） | 2.558 | 2.644 | -3.27% |
| train/loss 逐步最大绝对差 | — | — | 0.097 |
| global_tokens 对齐 | 580/580 | 580/580 | 100% |
| 累计 tokens | 0.03065 B | 0.03065 B | 一致 |

### 3.2 分项解读

**吞吐（MFU）**

step 1 受编译与预热影响，两组 MFU 均偏低（0.203 vs 0.229），不宜作为稳态对比依据。自 step 2 起两条曲线稳定分离，实验组持续运行于较高区间（见图 4）。剔除 step 1 后，579 步中 **574 步**实验组 MFU 更高；其余 5 步劣势均小于 0.005，属 dynamic batch 与调度噪声。

按 `global_tokens` 四分位统计，MFU 相对提升由短 batch 的 **+3.88%** 增至长 batch 的 **+4.48%**，与 CrossEntropy、RoPE 在长序列下的单点收益特征一致：

| 分位 | token 范围 | MFU 相对提升 |
|------|------------|--------------|
| Q1 | 48,472 – 52,102 | +3.88% |
| Q2 | 52,102 – 52,871 | +4.15% |
| Q3 | 52,873 – 53,592 | +4.46% |
| Q4 | 53,602 – 56,388 | +4.48% |

<p align="center">
<img src="../assets/Liger-Kernel/verl-sft/train_mfu.png" width="48%"/>
<img src="../assets/Liger-Kernel/verl-sft/train_mfu_skip_step1.png" width="48%"/><br/>
<em>图 4  整网 MFU 逐步曲线（左：含 step 1；右：剔除 step 1）</em>
</p>

**NPU 显存**

allocated 峰值略降 0.44%，逐步曲线近乎重合，中位差约 **-56 MB**（图 5 左）；reserved 峰值均为 46.93 GB，全程无差异（图 5 右）。整网 NPU 显存仍由 Attention、MatMul、LM Head 等未纳入 patch 的路径主导；算子 micro-benchmark 中的 workspace 节省在整 step 内时间复用，峰值取 max 而非 sum，故整网 allocated 降幅远低于单 kernel 的 28.8%～68.4%，属预期现象。

<p align="center">
<img src="../assets/Liger-Kernel/verl-sft/perf_max_memory_allocated_gb.png" width="48%"/>
<img src="../assets/Liger-Kernel/verl-sft/perf_max_memory_reserved_gb.png" width="48%"/><br/>
<em>图 5  NPU allocated / reserved 显存逐步曲线</em>
</p>

**Host 内存**

实验组自训练初期即低于对照组：step 1 即低 8.4%，全程均值低 11.35%，末步低 21.29%（绝对差约 26 GB）。对照组曲线在训练中后期持续抬升，实验组更平（见图 6）。该指标无法由 isolated micro-benchmark 直接解释，属于整网训练栈层面的观测结果，对长周期 LoRA SFT 的资源规划具有实际意义。

<p align="center">
<img src="../assets/Liger-Kernel/verl-sft/perf_cpu_memory_used_gb.png" width="72%"/><br/>
<em>图 6  Host CPU 内存逐步曲线</em>
</p>

**精度与收敛**

train/val loss 曲线整体重合（图 7 左），逐步平均绝对差（MAD）0.038，最大绝对差 0.097；step 580 处 train/val loss 略优于对照组。梯度范数自约 1.0 逐步上升至末期 2.3～2.6，两组形态平行，未见异常尖峰（图 7 右），表明 Liger-Kernel 接入未引入可观测的数值不稳定或收敛劣化。

<p align="center">
<img src="../assets/Liger-Kernel/verl-sft/train_loss.png" width="48%"/>
<img src="../assets/Liger-Kernel/verl-sft/train_grad_norm.png" width="48%"/><br/>
<em>图 7  训练 loss 与梯度范数逐步曲线</em>
</p>

**实验有效性**

580/580 step 的 `global_tokens` 完全一致，累计 token 均为 **0.03065 B**（图 8），可排除 batch 配置、数据顺序或 dynamic batch 切分差异对 MFU、loss、内存对比的干扰，保证对照具有因果有效性。

<p align="center">
<img src="../assets/Liger-Kernel/verl-sft/train_global_tokens.png" width="72%"/><br/>
<em>图 8  `global_tokens` 逐步对齐曲线</em>
</p>

---

## 四、算子层与整网层收益关联

算子层 1.25×～2.06× 的单点加速，与整网 +4% MFU 之间的数量关系，可用 patch 范围与 Amdahl 定律加以解释。

**已 patch 路径：** RMSNorm（72 次/step）、RoPE（36 次/step）、CrossEntropy（1 次/step）。

**未 patch 路径：** QKV/O 投影、FFN、Attention、LM Head 等；LoRA 仅更新 adapter，上述大算子仍构成单 step 耗时的主体部分。

按 T=8192 估算，三项 kernel 相关子系统累计可节省约 35.7% 耗时，但该子系统约占整 step 耗时的 12%。据此推算整 step 理论 MFU 提升约 4.3%，与实测 +4.14% 一致。

| 观测项 | micro-benchmark | verl LoRA SFT |
|--------|-----------------|---------------|
| 单 kernel 加速比 | 可测 | — |
| 整网 MFU 幅度 | 可估算 | +4.14%（实测） |
| NPU allocated 峰值 | 单 kernel 28.8%～68.4% | -0.44%（实测） |
| Host CPU 内存 | 不可测 | -21.29% 末步（实测） |
| 训练 loss 等价性 | 不可测 | MAD 0.038（实测） |
| 梯度范数稳定性 | 不可测 | 形态平行（实测） |
| global_tokens 对齐 | 不可测 | 580/580（实测） |

算子层验证 Ascend 后端能力；整网层验证 verl 接入效果。二者相互补充，构成从 kernel 到训练框架的完整评估链条。

---

## 五、结论

1. Liger-Kernel Ascend 后端已提供较完整的低层算子能力，可提供给与 verl LoRA SFT 类似任务开启 rms_norm、rope、cross_entropy 等使用。
2. 算子层：T=8192 下三项分别实现 1.82× / 1.25× / 2.06× 加速，峰值 NPU 显存节省 28.8%～68.4%。
3. 整网层：580 step、4 卡 FSDP、token 100% 对齐条件下，MFU +4.14%，Host 内存均值 -11.35%、末步 -21.29%，loss 与 grad_norm 无劣化。
4. 整网 MFU 增益与算子层数据在 Amdahl 框架下自洽；Host 内存下降为整网实验方可观测的附加收益。
5. 全参数 SFT、多节点集群、RLHF/DPO 等场景尚未覆盖；MFU 幅度可能因 patch 范围与计算图组成不同而与本次 LoRA 场景存在差异。


**— END —**
