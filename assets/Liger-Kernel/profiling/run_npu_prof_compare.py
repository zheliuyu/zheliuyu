#!/usr/bin/env python3
"""Batch NPU profiling: liger vs baseline for all six benchmark kernels."""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch_npu.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

ASSETS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = ASSETS_DIR / "data" / "npu_compare"


def _resolve_liger_kernel_root() -> Path:
    if env := os.environ.get("LIGER_KERNEL_ROOT"):
        return Path(env).resolve()
    for candidate in (
        ASSETS_DIR.parents[1] / "Liger-Kernel",
        Path("/home/weichunyu/dev_lk/Liger-Kernel"),
    ):
        if (candidate / "benchmark" / "scripts" / "prof_kernel_compare.py").exists():
            return candidate.resolve()
    raise RuntimeError(
        "Set LIGER_KERNEL_ROOT to a Liger-Kernel checkout that contains benchmark/scripts/prof_kernel_compare.py"
    )


sys.path.insert(0, str(_resolve_liger_kernel_root()))
from benchmark.scripts.prof_kernel_compare import (  # noqa: E402
    run_cross_entropy,
    run_fused_linear_cross_entropy,
    run_fused_moe,
    run_rms_norm,
    run_rope,
    run_swiglu,
)

# (kernel, provider, run_once_fn, profiler_steps)
CASES = [
    ("rms_norm", "liger", lambda: run_rms_norm("liger", iters=1), 7),
    ("rms_norm", "huggingface", lambda: run_rms_norm("huggingface", iters=1), 7),
    ("cross_entropy", "liger", lambda: run_cross_entropy("liger", iters=1), 7),
    ("cross_entropy", "torch", lambda: run_cross_entropy("torch", iters=1), 7),
    ("rope", "liger", lambda: run_rope("liger", iters=1), 7),
    ("rope", "huggingface", lambda: run_rope("huggingface", iters=1), 7),
    ("swiglu", "liger", lambda: run_swiglu("liger", iters=1), 7),
    ("swiglu", "huggingface", lambda: run_swiglu("huggingface", iters=1), 7),
    ("fused_moe", "liger", lambda: run_fused_moe("liger", iters=1), 5),
    ("fused_moe", "huggingface", lambda: run_fused_moe("huggingface", iters=1), 5),
    ("fused_linear_cross_entropy", "liger", lambda: run_fused_linear_cross_entropy("liger", iters=1), 5),
    ("fused_linear_cross_entropy", "torch", lambda: run_fused_linear_cross_entropy("torch", iters=1), 5),
]


def measure_peak_mem(run_once):
    getattr(torch, "npu").memory.reset_peak_memory_stats()
    run_once()
    getattr(torch, "npu").synchronize()
    return getattr(torch, "npu").max_memory_allocated() / 2**20


def profile_case(kernel: str, provider: str, run_once, out_root: Path, profiler_steps: int):
    tag = f"{kernel}_{provider}"
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    peak_mb = measure_peak_mem(run_once)

    with profile(
        activities=[ProfilerActivity.NPU, ProfilerActivity.CPU],
        schedule=schedule(wait=0, warmup=2, active=min(5, profiler_steps - 2), repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(out_dir)),
        record_shapes=False,
        profile_memory=True,
    ) as prof:
        for _ in range(profiler_steps):
            run_once()
            prof.step()

    prof_dirs = list(out_dir.glob("localhost*"))
    if not prof_dirs:
        raise RuntimeError(f"No profiler output under {out_dir}")
    ascend_out = prof_dirs[0] / "ASCEND_PROFILER_OUTPUT"
    return tag, peak_mb, ascend_out


def summarize_kernels(ascend_out: Path):
    kd = ascend_out / "kernel_details.csv"
    if not kd.exists():
        return {}

    agg = defaultdict(lambda: {"count": 0, "duration_us": 0.0, "wait_us": 0.0})
    with kd.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "").strip()
            if not name or name == "Name":
                continue
            try:
                dur = float(row.get("Duration(us)", "0").strip().replace("\t", ""))
                wait = float(row.get("Wait Time(us)", "0").strip().replace("\t", ""))
            except ValueError:
                continue
            agg[name]["count"] += 1
            agg[name]["duration_us"] += dur
            agg[name]["wait_us"] += wait
    return dict(agg)


def summarize_operators(ascend_out: Path, top_n=25):
    od = ascend_out / "operator_details.csv"
    if not od.exists():
        return []

    rows = []
    with od.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "")
            try:
                dev = float(row.get("Device Total Duration(us)", 0) or 0)
                host = float(row.get("Host Total Duration(us)", 0) or 0)
            except ValueError:
                dev, host = 0.0, 0.0
            rows.append((name, dev, host))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_n]


def case_already_done(out_root: Path, tag: str) -> bool:
    case_dir = out_root / tag
    return any(case_dir.glob("localhost*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--device", default="15")
    parser.add_argument("--kernels", nargs="*", help="Subset of kernels to profile (default: all)")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.device
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    selected = set(args.kernels) if args.kernels else None

    report_lines = [
        f"# NPU Profiling Compare (device {args.device})",
        "",
        "Kernels: rms_norm, cross_entropy, rope, swiglu, fused_moe, fused_linear_cross_entropy",
        "",
        "## Peak HBM (MB) per fwd+bwd step",
        "| case | peak_MB |",
        "|------|---------|",
    ]

    kernel_reports = {}

    for kernel, provider, fn, steps in CASES:
        if selected and kernel not in selected:
            continue
        tag = f"{kernel}_{provider}"
        if args.skip_existing and case_already_done(out_root, tag):
            print(f"skip existing {tag}", flush=True)
            prof_dirs = list((out_root / tag).glob("localhost*/ASCEND_PROFILER_OUTPUT"))
            ascend_out = prof_dirs[0]
            peak_mb = measure_peak_mem(fn)
            kernel_reports[tag] = {
                "peak_mb": peak_mb,
                "kernels": summarize_kernels(ascend_out),
                "operators": summarize_operators(ascend_out),
                "ascend_out": str(ascend_out),
            }
            report_lines.append(f"| {tag} | {peak_mb:.1f} |")
            continue

        tag, peak_mb, ascend_out = profile_case(kernel, provider, fn, out_root, steps)
        report_lines.append(f"| {tag} | {peak_mb:.1f} |")
        kernel_reports[tag] = {
            "peak_mb": peak_mb,
            "kernels": summarize_kernels(ascend_out),
            "operators": summarize_operators(ascend_out),
            "ascend_out": str(ascend_out),
        }
        print(f"done {tag} peak={peak_mb:.1f}MB", flush=True)

    for tag, data in kernel_reports.items():
        kernels = data["kernels"]
        total_dur = sum(v["duration_us"] for v in kernels.values())
        total_cnt = sum(v["count"] for v in kernels.values())
        report_lines += [
            "",
            f"## {tag}",
            f"- ascend_out: `{data['ascend_out']}`",
            f"- device kernel launches (all steps): **{total_cnt}**",
            f"- summed kernel duration (us): **{total_dur:.0f}**",
            "",
            "### Top device kernels by total duration",
            "| kernel | count | total_us | avg_us | total_wait_us |",
            "|--------|-------|----------|--------|---------------|",
        ]
        ranked = sorted(kernels.items(), key=lambda x: x[1]["duration_us"], reverse=True)
        for name, st in ranked[:20]:
            avg = st["duration_us"] / st["count"] if st["count"] else 0
            report_lines.append(
                f"| {name} | {st['count']} | {st['duration_us']:.1f} | {avg:.1f} | {st['wait_us']:.1f} |"
            )

        report_lines += [
            "",
            "### Top operators by device time",
            "| op | device_us | host_us |",
            "|----|-----------|---------|",
        ]
        for name, dev, host in data["operators"][:15]:
            report_lines.append(f"| {name} | {dev:.1f} | {host:.1f} |")

    report_path = out_root / "profiling_summary.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
