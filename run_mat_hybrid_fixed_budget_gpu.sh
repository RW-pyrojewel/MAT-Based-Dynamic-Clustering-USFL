#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"
DATA_DIR="${1:-${DATA_DIR:-../Data}}"
LOG_DIR="${2:-${LOG_DIR:-logs}}"
RUN_CHECKS="${RUN_CHECKS:-1}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Data directory not found: $DATA_DIR" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable" >&2
  exit 2
fi
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
mkdir -p "$LOG_DIR"

if [[ "$RUN_CHECKS" == "1" ]]; then
  "$PYTHON_BIN" -m compileall -q baselines models utils mat_hybrid_fixed_budget_runner.py scenario_a_runner.py
  "$PYTHON_BIN" -m unittest discover -s tests -q
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
STDOUT_LOG="$LOG_DIR/mat_hybrid_fixed_budget_${STAMP}.log"
echo "Starting fixed-budget hybrid MAT at $(date --iso-8601=seconds)"
"$PYTHON_BIN" mat_hybrid_fixed_budget_runner.py \
  --data-dir "$DATA_DIR" --log-dir "$LOG_DIR" --device cuda 2>&1 | tee "$STDOUT_LOG"
