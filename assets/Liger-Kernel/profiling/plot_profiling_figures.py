#!/usr/bin/env python3
"""Generate profiling-based figures from torch_npu.profiler ASCEND_PROFILER_OUTPUT."""

from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("LIGER_PROFILING_DATA", ASSETS_DIR / "data" / "npu_compare"))
OUT = ASSETS_DIR
ACTIVE_STEPS = {4, 5, 6}  # warmup=2, active steps in profiler schedule

# Kernels that are benchmark harness noise (grad tensor generation), not RMS/CE compute
NOISE_PATTERNS = (
    "InplaceNormal",
    "DSARandomNormal",
    "DSARandomUniform",
    "InplaceRandom",
)

LIGER_PATTERNS = (
    "_rms_norm_",
    "_triton_rope",
    "_swiglu_",
    "_fused_up_proj",
    "_fused_down_proj",
    "_moe_",
    "_token_gather",
    "liger_cross_entropy",
    "liger_rotary",
    "liger_swiglu",
    "liger_fused_moe",
    "liger_fused_linear",
    "liger_moe",
    "liger_",
)

ALL_COMPARE_CASES = [
    ("rms_norm", "liger", "RMSNorm Liger"),
    ("rms_norm", "huggingface", "RMSNorm HF"),
    ("cross_entropy", "liger", "CE Liger"),
    ("cross_entropy", "torch", "CE Torch"),
    ("rope", "liger", "RoPE Liger"),
    ("rope", "huggingface", "RoPE HF"),
    ("swiglu", "liger", "SwiGLU Liger"),
    ("swiglu", "huggingface", "SwiGLU HF"),
    ("fused_moe", "liger", "MoE Liger"),
    ("fused_moe", "huggingface", "MoE HF"),
    ("fused_linear_cross_entropy", "liger", "FLCE Liger"),
    ("fused_linear_cross_entropy", "torch", "FLCE Torch"),
]

# Color palette
C_LIGER = "#2e7d32"
C_BASELINE = "#c62828"
C_NOISE = "#bdbdbd"
C_ACLNN = "#ef6c00"
C_CAST = "#5c6bc0"
C_REDUCE = "#00838f"
C_ELEM = "#f9a825"
C_OTHER = "#90a4ae"


def find_ascend_out(case: str) -> Path:
    case_dir = ROOT / case
    matches = list(case_dir.glob("localhost*/ASCEND_PROFILER_OUTPUT"))
    if not matches:
        raise FileNotFoundError(case)
    return matches[0]


def find_ascend_out_optional(case: str) -> Path | None:
    case_dir = ROOT / case
    matches = list(case_dir.glob("localhost*/ASCEND_PROFILER_OUTPUT"))
    return matches[0] if matches else None


def parse_float(s: str) -> float:
    return float(str(s).strip().replace("\t", ""))


def load_kernels(ascend_out: Path) -> list[dict]:
    rows = []
    with (ascend_out / "kernel_details.csv").open() as f:
        for row in csv.DictReader(f):
            try:
                step = int(row["Step Id"])
            except (KeyError, ValueError):
                continue
            name = row.get("Name", "").strip()
            if not name:
                continue
            rows.append(
                {
                    "step": step,
                    "name": name,
                    "duration_us": parse_float(row.get("Duration(us)", 0)),
                    "wait_us": parse_float(row.get("Wait Time(us)", 0)),
                    "start_us": parse_float(row.get("Start Time(us)", 0)),
                }
            )
    return rows


def is_noise(name: str) -> bool:
    return any(p in name for p in NOISE_PATTERNS)


def is_liger_kernel(name: str) -> bool:
    return any(p in name for p in LIGER_PATTERNS)


def classify_kernel(name: str) -> str:
    if is_noise(name):
        return "benchmark_noise"
    if is_liger_kernel(name):
        if "forward" in name.lower():
            return "liger_forward"
        if "backward" in name.lower():
            return "liger_backward"
        return "liger_fused"
    n = name.lower()
    if "cast" in n:
        return "aclnn_cast"
    if "matmul" in n:
        return "aclnn_matmul"
    if any(x in n for x in ("indexput", "indexadd", "indexselect", "nonzero", "gather")):
        return "aclnn_index"
    if any(x in n for x in ("reduce", "mean", "sum")):
        return "aclnn_reduce"
    if any(x in n for x in ("pow", "mul", "div", "add", "rsqrt", "silu", "gelu")):
        return "aclnn_elementwise"
    if "logsoftmax" in n or "nllloss" in n:
        return "aclnn_ce_chain"
    return "aclnn_other"


def kernels_in_active(rows: list[dict], exclude_noise: bool = True) -> list[dict]:
    out = [r for r in rows if r["step"] in ACTIVE_STEPS]
    if exclude_noise:
        out = [r for r in out if not is_noise(r["name"])]
    return out


def aggregate_by_name(rows: list[dict]) -> dict[str, dict]:
    agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "duration_us": 0.0, "wait_us": 0.0})
    for r in rows:
        agg[r["name"]]["count"] += 1
        agg[r["name"]]["duration_us"] += r["duration_us"]
        agg[r["name"]]["wait_us"] += r["wait_us"]
    return dict(agg)


def per_step_compute_launch_count(rows: list[dict]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for r in rows:
        if r["step"] in ACTIVE_STEPS and not is_noise(r["name"]):
            counts[r["step"]] += 1
    return dict(counts)


def pick_representative_step(rows: list[dict]) -> int:
    """Step with median compute-kernel count among active steps."""
    counts = per_step_compute_launch_count(rows)
    if not counts:
        return 4
    return sorted(counts.items(), key=lambda x: x[1])[len(counts) // 2][0]


def split_fwd_bwd_iterations(rows: list[dict], step: int, forward_marker: str) -> list[list[dict]]:
    """Split kernels in one profiler step into individual fwd+bwd iterations."""
    step_rows = [r for r in rows if r["step"] == step and not is_noise(r["name"])]
    step_rows.sort(key=lambda r: r["start_us"])
    iterations: list[list[dict]] = []
    current: list[dict] = []
    for r in step_rows:
        if forward_marker in r["name"]:
            if current:
                iterations.append(current)
            current = [r]
        elif current:
            current.append(r)
    if current:
        iterations.append(current)
    return iterations


def count_rms_norm_ops(iteration: list[dict]) -> dict:
    """Classify one RMSNorm fwd+bwd iteration."""
    liger_fwd = sum(1 for r in iteration if "_rms_norm_forward" in r["name"])
    liger_bwd = sum(1 for r in iteration if "_rms_norm_backward" in r["name"])
    aclnn_rms = 0
    for r in iteration:
        n = r["name"].lower()
        if is_liger_kernel(r["name"]) or is_noise(r["name"]):
            continue
        if any(k in n for k in ("pow", "mean", "rsqrt", "cast", "div", "mul")):
            aclnn_rms += 1
    helpers = len(iteration) - liger_fwd - liger_bwd - aclnn_rms
    return {
        "total": len(iteration),
        "liger_fwd": liger_fwd,
        "liger_bwd": liger_bwd,
        "aclnn_rms": aclnn_rms,
        "helpers": helpers,
        "liger_main": liger_fwd + liger_bwd,
    }


def short_kernel_label(name: str, max_len: int = 42) -> str:
    if is_liger_kernel(name):
        m = re.search(
            r"(_rms_norm_\w+|liger_cross_entropy_\w+|_triton_rope\w*|_swiglu_\w+|_fused_\w+|_moe_\w+|_token_gather\w*)",
            name,
        )
        return m.group(1) if m else name[:max_len]
    # strip aclnn prefix for readability
    label = re.sub(r"^aclnn", "", name)
    label = label.split("_")[0:3]
    label = "_".join(label)
    return label[:max_len]


def setup_style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def fig_kernel_launch_totals():
    """Fig 1: total device kernel launches (with/without noise)."""
    cases = []
    for _kernel, provider, label in ALL_COMPARE_CASES:
        tag = f"{_kernel}_{provider}"
        if find_ascend_out_optional(tag):
            cases.append((tag, label))
    if not cases:
        return
    labels, all_tot, compute_tot = [], [], []
    for case, label in cases:
        rows = load_kernels(find_ascend_out(case))
        active = [r for r in rows if r["step"] in ACTIVE_STEPS]
        labels.append(label)
        all_tot.append(len(active))
        compute_tot.append(len(kernels_in_active(rows, exclude_noise=True)))

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(max(11, len(labels) * 0.85), 5.2))
    b1 = ax.bar(x - w / 2, all_tot, w, label="All device kernels (incl. harness)", color="#90caf9", edgecolor="white")
    b2 = ax.bar(x + w / 2, compute_tot, w, label="Compute kernels (excl. randn harness)", color=C_LIGER, edgecolor="white")
    use_log = max(compute_tot) / max(min(compute_tot), 1) > 50
    if use_log:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Kernel launch count (log scale)" if use_log else "Kernel launch count")
    ax.set_title("Device kernel launches · 3 active profiler steps · NPU device 15")
    ymax = max(max(all_tot), max(compute_tot))
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * (1.08 if use_log else 1.0) + (0 if use_log else ymax * 0.02),
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(OUT / "01_kernel_launch_count.png")
    plt.close(fig)


def fig_rms_single_step_gantt():
    """Fig 2: ONE fwd+bwd iteration timeline — 2 fused vs ~27 aclnn kernels."""
    cases = [
        ("rms_norm_liger", "Liger (1× fwd+bwd)"),
        ("rms_norm_huggingface", "HF (1× fwd+bwd)"),
    ]
    markers = ["_rms_norm_forward", "PowTensorScalar_PowAiCore_Pow"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)

    for ax, (case, title), marker in zip(axes, cases, markers):
        rows = load_kernels(find_ascend_out(case))
        step = pick_representative_step(rows)
        iters = split_fwd_bwd_iterations(rows, step, marker)
        step_rows = iters[0] if iters else []
        if not step_rows:
            ax.set_title(f"{title} — step {step} (no rows)")
            continue

        t0 = min(r["start_us"] for r in step_rows)
        colors = []
        for r in step_rows:
            if is_liger_kernel(r["name"]):
                colors.append(C_LIGER)
            elif "Cast" in r["name"]:
                colors.append(C_CAST)
            elif classify_kernel(r["name"]) == "aclnn_reduce":
                colors.append(C_REDUCE)
            else:
                colors.append(C_ELEM)

        y = 0
        for i, (r, c) in enumerate(zip(step_rows, colors)):
            start = (r["start_us"] - t0) / 1000.0  # ms
            dur = r["duration_us"] / 1000.0
            ax.barh(y, dur, left=start, height=0.72, color=c, edgecolor="white", linewidth=0.5)
            y += 1

        ax.set_yticks([])
        ax.set_ylabel("")
        stats = count_rms_norm_ops(step_rows)
        if is_liger_kernel(step_rows[0]["name"]) or any(is_liger_kernel(r["name"]) for r in step_rows):
            subtitle = f"{stats['liger_main']:.0f} fused + {stats['helpers']:.1f} grad helpers = {stats['total']:.0f} kernels"
        else:
            subtitle = f"{stats['aclnn_rms']:.1f} aclnn RMS ops + {stats['helpers']:.1f} helpers = {stats['total']:.0f} kernels"
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.grid(axis="x", alpha=0.25, linestyle="--")

    axes[-1].set_xlabel("Relative time within one fwd+bwd iteration (ms)")
    legend = [
        mpatches.Patch(color=C_LIGER, label="Liger fused Triton kernel"),
        mpatches.Patch(color=C_CAST, label="aclnn Cast"),
        mpatches.Patch(color=C_REDUCE, label="aclnn Reduce / Mean"),
        mpatches.Patch(color=C_ELEM, label="aclnn Pow / Mul / Add / …"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06), framealpha=0.95)
    fig.suptitle("RMSNorm T=8192 · single fwd+bwd iteration timeline (profiler)", y=1.02, fontsize=11)
    fig.subplots_adjust(bottom=0.14, hspace=0.45)
    fig.savefig(OUT / "02_rms_norm_single_step_timeline.png")
    plt.close(fig)


def fig_rms_duration_stacked():
    """Fig 3: stacked duration by kernel category."""
    data = {}
    for case, label in [("rms_norm_liger", "Liger"), ("rms_norm_huggingface", "HF")]:
        rows = kernels_in_active(load_kernels(find_ascend_out(case)))
        cat_dur: dict[str, float] = defaultdict(float)
        cat_cnt: dict[str, int] = defaultdict(int)
        for r in rows:
            cat = classify_kernel(r["name"])
            if cat == "benchmark_noise":
                continue
            cat_dur[cat] += r["duration_us"]
            cat_cnt[cat] += 1
        data[label] = (cat_dur, cat_cnt)

    categories = ["liger_forward", "liger_backward", "aclnn_cast", "aclnn_elementwise", "aclnn_reduce", "aclnn_other"]
    cat_labels = {
        "liger_forward": "Liger fwd",
        "liger_backward": "Liger bwd",
        "aclnn_cast": "Cast",
        "aclnn_elementwise": "Pow/Mul/Add",
        "aclnn_reduce": "Reduce/Mean",
        "aclnn_other": "Other",
    }
    cat_colors = {
        "liger_forward": "#1b5e20",
        "liger_backward": "#66bb6a",
        "aclnn_cast": C_CAST,
        "aclnn_elementwise": C_ELEM,
        "aclnn_reduce": C_REDUCE,
        "aclnn_other": C_OTHER,
    }

    labels = list(data.keys())
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(labels))
    for cat in categories:
        vals = np.array([data[l][0].get(cat, 0) / 1000.0 for l in labels])  # ms
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bottom=bottom, label=cat_labels[cat], color=cat_colors[cat], edgecolor="white")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Summed device kernel time (ms)\n3 active steps, excl. randn")
    ax.set_title("RMSNorm T=8192 · device time by operator category")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    for i, label in enumerate(labels):
        total_cnt = sum(data[label][1].get(c, 0) for c in categories)
        ax.text(i, bottom[i] + max(bottom) * 0.03 + 0.5, f"{total_cnt} launches", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "03_rms_norm_duration_breakdown.png")
    plt.close(fig)


def fig_ce_duration_stacked():
    """Fig 4: CE kernel duration — LogSoftmax chain vs 2 liger kernels."""
    data = {}
    for case, label in [("cross_entropy_liger", "Liger"), ("cross_entropy_torch", "Torch")]:
        rows = kernels_in_active(load_kernels(find_ascend_out(case)))
        cat_dur: dict[str, float] = defaultdict(float)
        cat_cnt: dict[str, int] = defaultdict(int)
        for r in rows:
            cat = classify_kernel(r["name"])
            if cat == "benchmark_noise":
                continue
            cat_dur[cat] += r["duration_us"]
            cat_cnt[cat] += 1
        data[label] = (cat_dur, cat_cnt)

    categories = ["liger_forward", "liger_backward", "aclnn_ce_chain", "aclnn_elementwise", "aclnn_other"]
    cat_labels = {
        "liger_forward": "Liger CE fwd",
        "liger_backward": "Liger CE bwd",
        "aclnn_ce_chain": "LogSoftmax+NLL",
        "aclnn_elementwise": "Aux ops",
        "aclnn_other": "Other",
    }
    cat_colors = {
        "liger_forward": "#1b5e20",
        "liger_backward": "#66bb6a",
        "aclnn_ce_chain": C_BASELINE,
        "aclnn_elementwise": C_ELEM,
        "aclnn_other": C_OTHER,
    }

    labels = list(data.keys())
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(labels))
    for cat in categories:
        vals = np.array([data[l][0].get(cat, 0) / 1000.0 for l in labels])
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bottom=bottom, label=cat_labels[cat], color=cat_colors[cat], edgecolor="white")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Summed device kernel time (ms)\n3 active steps, excl. randn")
    ax.set_title("CrossEntropy BT=4096 · fused kernel vs LogSoftmax+NLL chain")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for i, label in enumerate(labels):
        total_cnt = sum(data[label][1].get(c, 0) for c in categories)
        ax.text(i, bottom[i] + max(bottom) * 0.03 + 1, f"{total_cnt} launches", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "04_cross_entropy_duration_breakdown.png")
    plt.close(fig)


def fig_top_kernels_horizontal():
    """Fig 5: top kernels by duration — side by side RMSNorm."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (case, title) in zip(
        axes,
        [("rms_norm_liger", "RMSNorm Liger"), ("rms_norm_huggingface", "RMSNorm HF")],
    ):
        agg = aggregate_by_name(kernels_in_active(load_kernels(find_ascend_out(case))))
        ranked = sorted(agg.items(), key=lambda x: x[1]["duration_us"], reverse=True)[:8]
        names = [short_kernel_label(n) for n, _ in ranked][::-1]
        durs = [st["duration_us"] / 1000.0 for _, st in ranked][::-1]
        cnts = [st["count"] for _, st in ranked][::-1]
        colors = [C_LIGER if is_liger_kernel(n) else C_ACLNN for n, _ in ranked][::-1]
        bars = ax.barh(names, durs, color=colors, edgecolor="white")
        ax.set_xlabel("Total duration (ms)")
        ax.set_title(title)
        for bar, c in zip(bars, cnts):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"×{c}", va="center", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Top device kernels by accumulated time (3 active steps)")
    fig.savefig(OUT / "05_rms_norm_top_kernels.png")
    plt.close(fig)


def fig_per_iteration_kernel_bars():
    """Fig 6: kernels per single fwd+bwd iteration (the fair comparison)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    groups = ["Liger\nfused", "Liger\nhelpers", "HF\nRMS ops", "HF\nhelpers"]
    liger_rows = load_kernels(find_ascend_out("rms_norm_liger"))
    hf_rows = load_kernels(find_ascend_out("rms_norm_huggingface"))
    step = pick_representative_step(liger_rows)
    l_stats = [count_rms_norm_ops(it) for it in split_fwd_bwd_iterations(liger_rows, step, "_rms_norm_forward")]
    h_stats = [count_rms_norm_ops(it) for it in split_fwd_bwd_iterations(hf_rows, step, "PowTensorScalar_PowAiCore_Pow")]
    # average over iterations in one profiler step
    def avg(stats, key):
        return np.mean([s[key] for s in stats]) if stats else 0

    vals = [
        avg(l_stats, "liger_main"),
        avg(l_stats, "helpers"),
        avg(h_stats, "aclnn_rms"),
        avg(h_stats, "helpers"),
    ]
    colors = [C_LIGER, "#a5d6a7", C_BASELINE, "#ef9a9a"]
    legend_labels = ["Liger fused (fwd+bwd)", "Liger grad helpers", "HF RMS aclnn ops", "HF grad helpers"]
    x = np.arange(len(groups))
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Avg kernel launches per fwd+bwd iteration")
    ax.set_title(f"RMSNorm T=8192 · profiler step {step} (per-iteration average)")
    ymax = max(vals) * 1.25
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.02, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, ymax)
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, legend_labels)]
    ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(OUT / "06_rms_norm_kernels_per_iteration.png")
    plt.close(fig)


def fig_ce_single_iteration():
    """Fig 9: CE one fwd+bwd — 2 liger vs 4 heavy torch kernels."""
    leg_handles = [
        mpatches.Patch(color=C_LIGER, label="Liger fused kernel"),
        mpatches.Patch(color=C_BASELINE, label="Baseline aclnn kernel"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (case, title, marker) in zip(
        axes,
        [
            ("cross_entropy_liger", "Liger CE", "liger_cross_entropy_forward"),
            ("cross_entropy_torch", "Torch CE", "LogSoftmax_LogSoftmax"),
        ],
    ):
        rows = load_kernels(find_ascend_out(case))
        step = pick_representative_step(rows)
        iters = split_fwd_bwd_iterations(rows, step, marker)
        iteration = iters[0] if iters else []
        # only show kernels with duration > 50us OR liger
        shown = [r for r in iteration if r["duration_us"] > 50 or is_liger_kernel(r["name"])]
        shown.sort(key=lambda r: r["start_us"])
        t0 = shown[0]["start_us"] if shown else 0
        labels, durs, colors = [], [], []
        for r in shown:
            labels.append(short_kernel_label(r["name"], 28))
            durs.append(r["duration_us"] / 1000.0)
            colors.append(C_LIGER if is_liger_kernel(r["name"]) else C_BASELINE)
        y = np.arange(len(labels))
        ax.barh(y, durs, color=colors, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Duration (ms)")
        ax.set_title(f"{title}\n1× fwd+bwd · kernels with duration > 50 µs")
        ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.legend(handles=leg_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), framealpha=0.95)
    fig.suptitle("CrossEntropy BT=4096 · heavy compute kernels per iteration", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "09_ce_single_iteration.png")
    plt.close(fig)


def _stacked_duration_by_category(
    case_pairs: list[tuple[str, str]],
    categories: list[str],
    cat_labels: dict[str, str],
    cat_colors: dict[str, str],
    title: str,
    outfile: str,
):
    data = {}
    for case, label in case_pairs:
        rows = kernels_in_active(load_kernels(find_ascend_out(case)))
        cat_dur: dict[str, float] = defaultdict(float)
        cat_cnt: dict[str, int] = defaultdict(int)
        for r in rows:
            cat = classify_kernel(r["name"])
            if cat == "benchmark_noise":
                continue
            cat_dur[cat] += r["duration_us"]
            cat_cnt[cat] += 1
        data[label] = (cat_dur, cat_cnt)

    labels = [p[1] for p in case_pairs]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bottom = np.zeros(len(labels))
    for cat in categories:
        vals = np.array([data[l][0].get(cat, 0) / 1000.0 for l in labels])
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bottom=bottom, label=cat_labels[cat], color=cat_colors[cat], edgecolor="white")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Summed device kernel time (ms)\n3 active steps, excl. randn")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    pad = max(bottom) * 0.04 + 0.3 if len(bottom) else 0.5
    for i, label in enumerate(labels):
        total_cnt = sum(data[label][1].get(c, 0) for c in categories)
        ax.text(i, bottom[i] + pad, f"{total_cnt} launches", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / outfile)
    plt.close(fig)


def _single_iteration_bar_pair(
    liger_case: str,
    base_case: str,
    liger_marker: str,
    base_marker: str,
    liger_title: str,
    base_title: str,
    suptitle: str,
    outfile: str,
    min_us: float = 50,
):
    leg_handles = [
        mpatches.Patch(color=C_LIGER, label="Liger fused kernel"),
        mpatches.Patch(color=C_BASELINE, label="Baseline aclnn kernel"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (case, title, marker) in zip(
        axes,
        [(liger_case, liger_title, liger_marker), (base_case, base_title, base_marker)],
    ):
        rows = load_kernels(find_ascend_out(case))
        step = pick_representative_step(rows)
        iters = split_fwd_bwd_iterations(rows, step, marker)
        iteration = iters[0] if iters else []
        shown = [r for r in iteration if r["duration_us"] > min_us or is_liger_kernel(r["name"])]
        shown.sort(key=lambda r: r["start_us"])
        labels = [short_kernel_label(r["name"], 30) for r in shown]
        durs = [r["duration_us"] / 1000.0 for r in shown]
        colors = [C_LIGER if is_liger_kernel(r["name"]) else C_BASELINE for r in shown]
        y = np.arange(len(labels))
        ax.barh(y, durs, color=colors, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Duration (ms)")
        ax.set_title(f"{title}\n1× fwd+bwd · kernels > {min_us:.0f} µs or Liger")
        ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.legend(handles=leg_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), framealpha=0.95)
    fig.suptitle(suptitle, y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / outfile)
    plt.close(fig)


def _top_kernels_side_by_side(liger_case: str, base_case: str, liger_title: str, base_title: str, suptitle: str, outfile: str, top_n: int = 8):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (case, title) in zip(axes, [(liger_case, liger_title), (base_case, base_title)]):
        agg = aggregate_by_name(kernels_in_active(load_kernels(find_ascend_out(case))))
        ranked = sorted(agg.items(), key=lambda x: x[1]["duration_us"], reverse=True)[:top_n]
        names = [short_kernel_label(n, 32) for n, _ in ranked][::-1]
        durs = [st["duration_us"] / 1000.0 for _, st in ranked][::-1]
        cnts = [st["count"] for _, st in ranked][::-1]
        colors = [C_LIGER if is_liger_kernel(n) else C_ACLNN for n, _ in ranked][::-1]
        bars = ax.barh(names, durs, color=colors, edgecolor="white")
        ax.set_xlabel("Total duration (ms)")
        ax.set_title(title)
        xmax = max(durs) if durs else 1
        for bar, c in zip(bars, cnts):
            ax.text(bar.get_width() + xmax * 0.02, bar.get_y() + bar.get_height() / 2, f"×{c}", va="center", fontsize=7)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(suptitle, y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / outfile)
    plt.close(fig)


def fig_rope_profiling():
    cats = ["liger_fused", "liger_forward", "liger_backward", "aclnn_elementwise", "aclnn_cast", "aclnn_other"]
    labels = {
        "liger_fused": "Liger RoPE fused",
        "liger_forward": "Liger fwd",
        "liger_backward": "Liger bwd",
        "aclnn_elementwise": "Mul/Add/Transpose",
        "aclnn_cast": "Cast",
        "aclnn_other": "Other",
    }
    colors = {
        "liger_fused": "#1b5e20",
        "liger_forward": "#388e3c",
        "liger_backward": "#66bb6a",
        "aclnn_elementwise": C_ELEM,
        "aclnn_cast": C_CAST,
        "aclnn_other": C_OTHER,
    }
    _stacked_duration_by_category(
        [("rope_liger", "Liger"), ("rope_huggingface", "HF")],
        cats,
        labels,
        colors,
        "RoPE T=8192 · device time by operator category",
        "14_rope_duration_breakdown.png",
    )
    _single_iteration_bar_pair(
        "rope_liger",
        "rope_huggingface",
        "_triton_rope_npu",
        "Mul_MulAiCore_Mul",
        "Liger RoPE",
        "HF RoPE",
        "RoPE T=8192 · heavy compute kernels per iteration",
        "15_rope_single_iteration.png",
        min_us=20,
    )


def fig_swiglu_profiling():
    cats = ["aclnn_matmul", "liger_fused", "liger_forward", "liger_backward", "aclnn_elementwise", "aclnn_cast", "aclnn_other"]
    labels = {
        "aclnn_matmul": "MatMul (dominant)",
        "liger_fused": "Liger SwiGLU fused",
        "liger_forward": "Liger fwd",
        "liger_backward": "Liger bwd",
        "aclnn_elementwise": "Silu/Mul/Add",
        "aclnn_cast": "Cast",
        "aclnn_other": "Other",
    }
    colors = {
        "aclnn_matmul": "#1565c0",
        "liger_fused": "#1b5e20",
        "liger_forward": "#388e3c",
        "liger_backward": "#66bb6a",
        "aclnn_elementwise": C_ELEM,
        "aclnn_cast": C_CAST,
        "aclnn_other": C_OTHER,
    }
    _stacked_duration_by_category(
        [("swiglu_liger", "Liger"), ("swiglu_huggingface", "HF")],
        cats,
        labels,
        colors,
        "SwiGLU T=8192 · device time by operator category",
        "16_swiglu_duration_breakdown.png",
    )
    _single_iteration_bar_pair(
        "swiglu_liger",
        "swiglu_huggingface",
        "_swiglu_forward_kernel",
        "Matmul_MatMulV3Common_MatMulV3",
        "Liger SwiGLU",
        "HF SwiGLU",
        "SwiGLU T=8192 · MatMul vs fused elementwise per iteration",
        "17_swiglu_single_iteration.png",
        min_us=100,
    )


def fig_fused_moe_profiling():
    cats = ["liger_fused", "liger_forward", "liger_backward", "aclnn_matmul", "aclnn_index", "aclnn_elementwise", "aclnn_other"]
    labels = {
        "liger_fused": "Liger MoE fused",
        "liger_forward": "Liger fwd",
        "liger_backward": "Liger bwd",
        "aclnn_matmul": "MatMul",
        "aclnn_index": "Index/Nonzero",
        "aclnn_elementwise": "Add/Mul",
        "aclnn_other": "Other",
    }
    colors = {
        "liger_fused": "#1b5e20",
        "liger_forward": "#388e3c",
        "liger_backward": "#66bb6a",
        "aclnn_matmul": "#1565c0",
        "aclnn_index": "#d84315",
        "aclnn_elementwise": C_ELEM,
        "aclnn_other": C_OTHER,
    }
    _stacked_duration_by_category(
        [("fused_moe_liger", "Liger"), ("fused_moe_huggingface", "HF")],
        cats,
        labels,
        colors,
        "Fused MoE T=8192 · device time by operator category",
        "18_fused_moe_duration_breakdown.png",
    )
    _top_kernels_side_by_side(
        "fused_moe_liger",
        "fused_moe_huggingface",
        "Fused MoE Liger",
        "Fused MoE HF (Python loop)",
        "Fused MoE T=8192 · top device kernels by accumulated time (3 active steps)",
        "19_fused_moe_top_kernels.png",
        top_n=8,
    )


def fig_flce_profiling():
    cats = ["aclnn_matmul", "liger_forward", "liger_backward", "liger_fused", "aclnn_ce_chain", "aclnn_elementwise", "aclnn_other"]
    labels = {
        "aclnn_matmul": "LM Head MatMul",
        "liger_forward": "Liger CE fwd",
        "liger_backward": "Liger CE bwd",
        "liger_fused": "Liger fused",
        "aclnn_ce_chain": "LogSoftmax+NLL",
        "aclnn_elementwise": "Aux ops",
        "aclnn_other": "Other",
    }
    colors = {
        "aclnn_matmul": "#1565c0",
        "liger_forward": "#1b5e20",
        "liger_backward": "#66bb6a",
        "liger_fused": "#388e3c",
        "aclnn_ce_chain": C_BASELINE,
        "aclnn_elementwise": C_ELEM,
        "aclnn_other": C_OTHER,
    }
    _stacked_duration_by_category(
        [("fused_linear_cross_entropy_liger", "Liger"), ("fused_linear_cross_entropy_torch", "Torch")],
        cats,
        labels,
        colors,
        "Fused Linear CE BT=4096 · device time by operator category",
        "20_flce_duration_breakdown.png",
    )
    _single_iteration_bar_pair(
        "fused_linear_cross_entropy_liger",
        "fused_linear_cross_entropy_torch",
        "liger_cross_entropy_forward_kernel_plain",
        "LogSoftmax_LogSoftmaxAiCore_LogSoftmaxV2",
        "Liger FLCE",
        "Torch Linear+CE",
        "Fused Linear CE BT=4096 · MatMul + CE kernels per iteration",
        "21_flce_single_iteration.png",
        min_us=50,
    )


def _read_peak_memory_from_summary() -> dict[str, float]:
    """Parse peak_MB table from profiling_summary.md if present."""
    summary = ROOT / "profiling_summary.md"
    peaks: dict[str, float] = {}
    if not summary.exists():
        return peaks
    in_table = False
    for line in summary.read_text().splitlines():
        if line.startswith("| case |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) == 2 and parts[0] != "case" and parts[0] != "------":
                try:
                    peaks[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return peaks


def fig_peak_memory_and_benchmark():
    """Fig 7: profiling peak memory for all profiled cases."""
    peaks = _read_peak_memory_from_summary()
    prof_cases = []
    for _kernel, provider, label in ALL_COMPARE_CASES:
        tag = f"{_kernel}_{provider}"
        if tag in peaks:
            color = C_LIGER if provider == "liger" else C_BASELINE
            prof_cases.append((label, peaks[tag], color))
    if not prof_cases:
        return

    labels = [p[0] for p in prof_cases]
    vals = [p[1] for p in prof_cases]
    colors = [p[2] for p in prof_cases]

    fig, ax = plt.subplots(figsize=(max(11, len(labels) * 0.85), 5.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, edgecolor="white", alpha=0.9, linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Peak HBM per fwd+bwd step (MB)")
    ax.set_title("Profiling peak memory · benchmark operator suite · NPU device 15")
    ymax = max(vals) if vals else 1
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.015, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, ymax * 1.12)
    leg = [mpatches.Patch(color=C_LIGER, label="Liger"), mpatches.Patch(color=C_BASELINE, label="Baseline")]
    ax.legend(handles=leg, loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(OUT / "07_peak_memory_profiling.png")
    plt.close(fig)


def fig_suite_launch_pairs():
    """Fig 13: per-kernel liger vs baseline compute launch count (benchmark operator suite)."""
    pairs = [
        ("rms_norm", "huggingface", "RMSNorm"),
        ("cross_entropy", "torch", "CrossEntropy"),
        ("rope", "huggingface", "RoPE"),
        ("swiglu", "huggingface", "SwiGLU"),
        ("fused_moe", "huggingface", "Fused MoE"),
        ("fused_linear_cross_entropy", "torch", "Fused Linear CE"),
    ]
    labels, liger_cnt, base_cnt = [], [], []
    for kernel, baseline, label in pairs:
        l_tag, b_tag = f"{kernel}_liger", f"{kernel}_{baseline}"
        l_out, b_out = find_ascend_out_optional(l_tag), find_ascend_out_optional(b_tag)
        if not l_out or not b_out:
            continue
        labels.append(label)
        liger_cnt.append(len(kernels_in_active(load_kernels(l_out))))
        base_cnt.append(len(kernels_in_active(load_kernels(b_out))))

    if not labels:
        return

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    b1 = ax.bar(x - w / 2, liger_cnt, w, label="Liger", color=C_LIGER, edgecolor="white")
    b2 = ax.bar(x + w / 2, base_cnt, w, label="Baseline", color=C_BASELINE, edgecolor="white")
    use_log = max(base_cnt) / max(min(base_cnt), 1) > 30
    if use_log:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Compute kernel launches (log)" if use_log else "Compute kernel launches")
    ax.set_title("Benchmark operator suite · compute launches (3 active profiler steps, excl. randn)")
    ymax = max(max(liger_cnt), max(base_cnt))
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * (1.1 if use_log else 1.0) + (0 if use_log else ymax * 0.02),
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(OUT / "13_suite_launch_pairs.png")
    plt.close(fig)


def fig_avg_kernel_duration_per_launch():
    """Fig 8: avg duration per launch for fused vs scattered ops."""
    cases = [
        ("rms_norm_liger", "RMSNorm Liger"),
        ("rms_norm_huggingface", "RMSNorm HF"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    labels, avgs, counts = [], [], []
    for case, label in cases:
        agg = aggregate_by_name(kernels_in_active(load_kernels(find_ascend_out(case))))
        total_d = sum(v["duration_us"] for v in agg.values())
        total_c = sum(v["count"] for v in agg.values())
        labels.append(label)
        avgs.append(total_d / total_c / 1000.0 if total_c else 0)
        counts.append(total_c)

    x = np.arange(2)
    bars = ax.bar(x, avgs, color=[C_LIGER, C_BASELINE], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Avg duration per compute kernel (ms)")
    ax.set_title("RMSNorm: fused kernels are fewer but each does more work")
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{c} launches",
            ha="center",
            fontsize=9,
        )
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(OUT / "08_rms_avg_kernel_duration.png")
    plt.close(fig)


def write_figure_index():
    lines = [
        "# Profiling Figures Index",
        "",
        "数据来源：`benchmark/profiling/npu_compare/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv`",
        "环境：NPU device 15 · torch_npu.profiler · active steps 4–6",
        "",
        "| 图 | 文件 | 说明 |",
        "|----|------|------|",
        "| 1 | 01_kernel_launch_count.png | 总 launch 次数：全部 vs 排除 randn 噪声 |",
        "| 2 | 02_rms_norm_single_step_timeline.png | **单步** fwd+bwd 时间线：Liger 2 kernel vs HF 10+ |",
        "| 3 | 03_rms_norm_duration_breakdown.png | RMSNorm 累计 device 时间按算子类别堆叠 |",
        "| 4 | 04_cross_entropy_duration_breakdown.png | CE：2 个 Liger kernel vs LogSoftmax+NLL 链 |",
        "| 5 | 05_rms_norm_top_kernels.png | Top kernel 耗时横向对比 |",
        "| 6 | 06_rms_norm_kernels_per_iteration.png | **单次 fwd+bwd** launch 个数（公平对比） |",
        "| 7 | 07_peak_memory_profiling.png | Profiling 实测 peak HBM |",
        "| 8 | 08_rms_avg_kernel_duration.png | 平均每次 launch 耗时 |",
        "| 9 | 09_ce_single_iteration.png | CE 单次迭代重 kernel 对比 |",
        "| 14 | 14_rope_duration_breakdown.png | RoPE device 时间构成 |",
        "| 15 | 15_rope_single_iteration.png | RoPE 单次迭代重 kernel |",
        "| 16 | 16_swiglu_duration_breakdown.png | SwiGLU device 时间构成 |",
        "| 17 | 17_swiglu_single_iteration.png | SwiGLU 单次迭代（MatMul vs 融合） |",
        "| 18 | 18_fused_moe_duration_breakdown.png | Fused MoE device 时间构成 |",
        "| 19 | 19_fused_moe_top_kernels.png | Fused MoE Top kernel 对比 |",
        "| 20 | 20_flce_duration_breakdown.png | Fused Linear CE device 时间构成 |",
        "| 21 | 21_flce_single_iteration.png | Fused Linear CE 单次迭代 |",
        "| 10 | 10_suite_speedup_memory.png | 评测算子集：加速比与显存节省 |",
        "| 11 | 11_suite_fwd_bwd_speedup.png | 评测算子集：forward/backward 分项加速 |",
        "| 12 | 12_suite_absolute_latency.png | 评测算子集：绝对延迟对比 |",
        "| 13 | 13_suite_launch_pairs.png | 评测算子集：launch 数 Liger vs Baseline |",
        "",
        "## 「8–10 个小算子 vs 1–2 融合 kernel」怎么看？",
        "",
        "1. **06_rms_norm_kernels_per_iteration.png**：单次 fwd+bwd，Liger **2** 次融合 launch；HF **~24** 次 RMS 相关 aclnn launch。",
        "2. **02_rms_norm_single_step_timeline.png**：截取 1 次迭代的时间条，HF 满屏细条，Liger 仅 2 条绿色主 kernel。",
        "3. **01_kernel_launch_count.png**：3 个 profiler step 累计（含每 step 内多次迭代），HF **498** vs Liger **102**。",
        "4. **09_ce_single_iteration.png**：Torch 4 个大 kernel（LogSoftmax×2 + NLL×2），Liger 2 个融合 kernel。",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    fig_kernel_launch_totals()
    fig_rms_single_step_gantt()
    fig_rms_duration_stacked()
    fig_ce_duration_stacked()
    fig_top_kernels_horizontal()
    fig_per_iteration_kernel_bars()
    fig_peak_memory_and_benchmark()
    fig_avg_kernel_duration_per_launch()
    fig_ce_single_iteration()
    fig_rope_profiling()
    fig_swiglu_profiling()
    fig_fused_moe_profiling()
    fig_flce_profiling()
    fig_suite_launch_pairs()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
