# Profiling 图表与复现脚本

本目录存放中文报告第 3.6 节 profiling 图（图 28–48）及第 3.4 节评测算子集总览图（图 25–27）。Benchmark 逐算子扫点曲线（图 1–24）位于 `../benchmark/`。

## 目录结构

```
profiling/
├── *.png                 # 已生成的 profiling / 总览图（22 张）
├── data/
│   ├── all_benchmark_data.csv          # 图 25–27 数据源
│   └── npu_compare/                    # profiling 原始 trace（需自行采集或软链）
│       ├── rms_norm_liger/
│       ├── rms_norm_huggingface/
│       └── …
├── plot_profiling_figures.py           # 图 01–21、07、13 及 RoPE/MoE 等专项图
├── plot_benchmark_six_kernel.py        # 图 25–27（10/11/12_suite_*.png）
├── run_npu_prof_compare.py             # 在 NPU 上采集 profiling 原始数据
└── regenerate_figures.sh               # 一键重绘（需已有 npu_compare 数据）
```

## 环境要求

- Ascend NPU + CANN + `torch_npu`
- Python：`matplotlib`、`numpy`、`pandas`
- 采集脚本 additionally 需要完整的 **Liger-Kernel 源码树**（含 `benchmark/scripts/prof_kernel_compare.py`）

## 1. 准备 profiling 原始数据

若已在 Liger-Kernel 仓库中采集过数据，可软链到本目录：

```bash
cd /home/weichunyu/zheliuyu/assets/Liger-Kernel/profiling
ln -sfn /path/to/Liger-Kernel/benchmark/profiling/npu_compare data/npu_compare
```

或在 NPU 上重新采集（耗时较长，12 组 case）：

```bash
export LIGER_KERNEL_ROOT=/path/to/Liger-Kernel
export ASCEND_RT_VISIBLE_DEVICES=15

cd /home/weichunyu/zheliuyu/assets/Liger-Kernel/profiling
python run_npu_prof_compare.py --device 15 --skip-existing
```

输出写入 `data/npu_compare/`，并生成 `data/npu_compare/profiling_summary.md`。

## 2. 重绘 profiling 图

```bash
cd /home/weichunyu/zheliuyu/assets/Liger-Kernel/profiling

# 可选：若数据不在默认路径
# export LIGER_PROFILING_DATA=/path/to/npu_compare

python plot_profiling_figures.py
python plot_benchmark_six_kernel.py
```

PNG 直接写入本目录（与报告 `../assets/Liger-Kernel/profiling/*.png` 引用一致）。

## 3. Benchmark 扫点曲线（图 1–24）

图 1–24 由 Liger-Kernel 仓库的 `benchmark/benchmarks_visualizer.py` 生成，输出到 `../benchmark/`。示例：

```bash
export LIGER_KERNEL_ROOT=/path/to/Liger-Kernel
cd "$LIGER_KERNEL_ROOT/benchmark"

for op in forward backward full; do
  for metric in speed memory; do
    python benchmarks_visualizer.py \
      --kernel-name rms_norm --metric-name "$metric" \
      --kernel-operation-mode "$op" --sweep-mode token_length --overwrite
  done
done
# 对 rope、cross_entropy、swiglu、fused_moe、fused_linear_cross_entropy 重复上述命令
```

生成后复制或软链 PNG 到 `assets/Liger-Kernel/benchmark/`。

## 图号对照（报告第 3 节）

| 报告图号 | 本目录文件 |
|---------|-----------|
| 图 25 | `10_suite_speedup_memory.png` |
| 图 26 | `11_suite_fwd_bwd_speedup.png` |
| 图 27 | `12_suite_absolute_latency.png` |
| 图 28–32 | `02/03/05/06/08_*.png`（RMSNorm） |
| 图 33–34 | `04/09_*.png`（CrossEntropy） |
| 图 38–39 | `14/15_*.png`（RoPE） |
| 图 40–41 | `16/17_*.png`（SwiGLU） |
| 图 42–43 | `18/19_*.png`（Fused MoE） |
| 图 44–45 | `20/21_*.png`（Fused Linear CE） |
| 图 46 | `13_suite_launch_pairs.png` |
| 图 47 | `07_peak_memory_profiling.png` |
| 图 48 | `01_kernel_launch_count.png` |
