#!/usr/bin/env bash
# Regenerate profiling PNGs in this directory (requires data/npu_compare).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [[ ! -d "${LIGER_PROFILING_DATA:-data/npu_compare}" ]] && [[ ! -d data/npu_compare ]]; then
  echo "Missing profiling data. Symlink or run run_npu_prof_compare.py first." >&2
  echo "  ln -sfn /path/to/Liger-Kernel/benchmark/profiling/npu_compare data/npu_compare" >&2
  exit 1
fi

python3 plot_profiling_figures.py
python3 plot_benchmark_six_kernel.py
echo "Done. PNGs written to $DIR"
