#!/usr/bin/env python3
"""Generate benchmark operator suite overview figures from all_benchmark_data.csv."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ASSETS_DIR = Path(__file__).resolve().parent
CSV = Path(os.environ.get("LIGER_BENCHMARK_CSV", ASSETS_DIR / "data" / "all_benchmark_data.csv"))
OUT = ASSETS_DIR

KERNEL_META = {
    "rms_norm": {"baseline": "huggingface", "x_at": 8192, "x_label": "T=8192"},
    "cross_entropy": {"baseline": "torch", "x_at": 8192, "x_label": "BT=8192"},
    "rope": {"baseline": "huggingface", "x_at": 8192, "x_label": "T=8192"},
    "swiglu": {"baseline": "huggingface", "x_at": 8192, "x_label": "T=8192"},
    "fused_moe": {"baseline": "huggingface", "x_at": 32768, "x_label": "T=32768"},
    "fused_linear_cross_entropy": {"baseline": "torch", "x_at": 8192, "x_label": "BT=8192"},
}

DISPLAY = {
    "rms_norm": "RMSNorm",
    "cross_entropy": "CrossEntropy",
    "rope": "RoPE",
    "swiglu": "SwiGLU",
    "fused_moe": "Fused MoE",
    "fused_linear_cross_entropy": "Fused Linear CE",
}

C_LIGER = "#2e7d32"
C_BASE = "#c62828"


def setup_style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def annotate_bars(ax, bars, labels, y_pad=0.08):
    ymax = max(b.get_height() for b in bars)
    ymin = min(b.get_height() for b in bars)
    for bar, txt in zip(bars, labels):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + ymax * y_pad,
            txt,
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False,
        )
    span = ymax - ymin if ymax != ymin else ymax
    ax.set_ylim(bottom=min(0, ymin - span * 0.05), top=ymax + span * 0.22)


def load_max_scale_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for kernel, meta in KERNEL_META.items():
        x_at = meta["x_at"]
        baseline = meta["baseline"]
        speed = df[
            (df.kernel_name == kernel)
            & (df.metric_name == "speed")
            & (df.kernel_operation_mode == "full")
            & (df.x_value == x_at)
        ]
        mem = df[
            (df.kernel_name == kernel)
            & (df.metric_name == "memory")
            & (df.kernel_operation_mode == "full")
            & (df.x_value == x_at)
        ]
        liger_s = speed[speed.kernel_provider == "liger"]
        base_s = speed[speed.kernel_provider == baseline]
        liger_m = mem[mem.kernel_provider == "liger"]
        base_m = mem[mem.kernel_provider == baseline]
        if liger_s.empty or base_s.empty or liger_m.empty or base_m.empty:
            continue
        rows.append(
            {
                "kernel": kernel,
                "label": DISPLAY[kernel],
                "x_label": meta["x_label"],
                "liger_ms": float(liger_s.iloc[0].y_value_50),
                "base_ms": float(base_s.iloc[0].y_value_50),
                "liger_mb": float(liger_m.iloc[0].y_value_50),
                "base_mb": float(base_m.iloc[0].y_value_50),
                "baseline": baseline,
            }
        )
    return rows


def fig_speedup_and_memory(rows: list[dict]):
    labels = [r["label"] for r in rows]
    speedups = [r["base_ms"] / r["liger_ms"] for r in rows]
    mem_save = [(r["base_mb"] - r["liger_mb"]) / r["base_mb"] * 100 for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(labels))

    ax = axes[0]
    colors = [C_LIGER if s >= 1.0 else C_BASE for s in speedups]
    bars = ax.bar(x, speedups, color=colors, edgecolor="white", linewidth=0.6)
    ax.axhline(1.0, color="#555", linestyle="--", linewidth=1, label="Baseline parity (1.0×)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Speedup (baseline latency / Liger latency)")
    ax.set_title("(a) Full fwd+bwd speedup at max benchmark scale")
    annotate_bars(ax, bars, [f"{v:.2f}×" for v in speedups])
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    ax = axes[1]
    colors = [C_LIGER if m >= 0 else C_BASE for m in mem_save]
    bars = ax.bar(x, mem_save, color=colors, edgecolor="white", linewidth=0.6)
    ax.axhline(0, color="#555", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Peak HBM savings vs baseline (%)")
    ax.set_title("(b) Peak memory savings at max benchmark scale")
    annotate_bars(
        ax,
        bars,
        [f"{v:+.0f}%" for v in mem_save],
        y_pad=0.06 if min(mem_save) >= 0 else 0.12,
    )
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle("Benchmark operator suite · Ascend910 · source: all_benchmark_data.csv", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "10_suite_speedup_memory.png")
    plt.close(fig)


def fig_forward_backward_breakdown(df: pd.DataFrame):
    kernels = list(KERNEL_META.keys())
    labels = [DISPLAY[k] for k in kernels]
    fwd_ratio, bwd_ratio = [], []
    for kernel, meta in KERNEL_META.items():
        x_at = meta["x_at"]
        baseline = meta["baseline"]
        sub = df[(df.kernel_name == kernel) & (df.metric_name == "speed") & (df.x_value == x_at)]
        l_fwd = sub[(sub.kernel_provider == "liger") & (sub.kernel_operation_mode == "forward")].y_value_50.iloc[0]
        l_bwd = sub[(sub.kernel_provider == "liger") & (sub.kernel_operation_mode == "backward")].y_value_50.iloc[0]
        b_fwd = sub[(sub.kernel_provider == baseline) & (sub.kernel_operation_mode == "forward")].y_value_50.iloc[0]
        b_bwd = sub[(sub.kernel_provider == baseline) & (sub.kernel_operation_mode == "backward")].y_value_50.iloc[0]
        fwd_ratio.append(b_fwd / l_fwd)
        bwd_ratio.append(b_bwd / l_bwd)

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    b1 = ax.bar(x - w / 2, fwd_ratio, w, label="Forward speedup (baseline/Liger)", color="#66bb6a", edgecolor="white")
    b2 = ax.bar(x + w / 2, bwd_ratio, w, label="Backward speedup (baseline/Liger)", color="#1b5e20", edgecolor="white")
    ax.axhline(1.0, color="#555", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Forward vs backward speedup · benchmark operator suite")
    ymax = max(max(fwd_ratio), max(bwd_ratio))
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + ymax * 0.02, f"{h:.2f}×", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, ymax * 1.18)
    ax.legend(loc="upper right", ncol=1, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(OUT / "11_suite_fwd_bwd_speedup.png")
    plt.close(fig)


def fig_absolute_latency(rows: list[dict]):
    labels = [f"{r['label']}\n({r['x_label']})" for r in rows]
    liger = [r["liger_ms"] for r in rows]
    base = [r["base_ms"] for r in rows]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11, 4.8))
    b1 = ax.bar(x - w / 2, liger, w, label="Liger", color=C_LIGER, edgecolor="white")
    b2 = ax.bar(x + w / 2, base, w, label="Baseline", color=C_BASE, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Full fwd+bwd latency (ms)")
    ax.set_title("Absolute latency · benchmark operator suite")
    ymax = max(max(liger), max(base))
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + ymax * 0.015,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_ylim(0, ymax * 1.15)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(OUT / "12_suite_absolute_latency.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    df = pd.read_csv(CSV)
    rows = load_max_scale_rows(df)
    fig_speedup_and_memory(rows)
    fig_forward_backward_breakdown(df)
    fig_absolute_latency(rows)
    print(f"Wrote benchmark operator suite overview figures to {OUT}")


if __name__ == "__main__":
    main()
