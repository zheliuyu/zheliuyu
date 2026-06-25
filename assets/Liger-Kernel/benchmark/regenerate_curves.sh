#!/usr/bin/env bash
# Regenerate token-length benchmark curves (report figures 1–24) into this directory.
# Requires LIGER_KERNEL_ROOT pointing to a full Liger-Kernel checkout.
set -euo pipefail

LK_ROOT="${LIGER_KERNEL_ROOT:?Set LIGER_KERNEL_ROOT to Liger-Kernel checkout}"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIS="$LK_ROOT/benchmark/benchmarks_visualizer.py"

kernels=(rms_norm rope cross_entropy swiglu fused_moe fused_linear_cross_entropy)
modes=(forward backward full)
metrics=(speed memory)

for k in "${kernels[@]}"; do
  for m in "${metrics[@]}"; do
    for op in "${modes[@]}"; do
      python3 "$VIS" \
        --kernel-name "$k" \
        --metric-name "$m" \
        --kernel-operation-mode "$op" \
        --sweep-mode token_length \
        --overwrite
    done
  done
done

cp -f "$LK_ROOT/benchmark/visualizations/"*.png "$OUT_DIR/"
echo "Copied benchmark curves to $OUT_DIR"
