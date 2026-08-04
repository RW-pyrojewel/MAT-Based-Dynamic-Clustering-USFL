#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PROBE_REPORT="${1:-${PROBE_REPORT:-}}"
DATA_DIR="${2:-${DATA_DIR:-../Data}}"
LOG_DIR="${3:-${LOG_DIR:-logs}}"
RESUME_CHECKPOINT="${4:-${RESUME_CHECKPOINT:-}}"
START_CYCLE="${START_CYCLE:-2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"
RUN_CHECKS="${RUN_CHECKS:-1}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LOCAL_STEPS="${LOCAL_STEPS:-4}"
EVALUATION_BATCHES="${EVALUATION_BATCHES:-10}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [[ -z "$PROBE_REPORT" ]]; then
  echo "Usage: bash $0 PROBE_REPORT [DATA_DIR] [LOG_DIR] [RESUME_CHECKPOINT]" >&2
  exit 2
fi
if [[ ! -f "$PROBE_REPORT" ]]; then
  echo "Probe report not found: $PROBE_REPORT" >&2
  exit 2
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "Data directory not found: $DATA_DIR" >&2
  exit 2
fi
if [[ -n "$RESUME_CHECKPOINT" && ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "Resume checkpoint not found: $RESUME_CHECKPOINT" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; a CUDA server is required." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

"$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); assert p.get("passed") is True, "probe report did not pass"; assert "legacy" in p, "probe report lacks legacy baseline"' "$PROBE_REPORT"
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "torch.cuda.is_available() is false"; print(f"CUDA ready: {torch.cuda.get_device_name(0)}; torch={torch.__version__}")'

mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
STDOUT_LOG="$LOG_DIR/mat_channel_scenario_server_${STAMP}.log"

if [[ "$RUN_CHECKS" == "1" ]]; then
  "$PYTHON_BIN" -m compileall -q models utils mat_channel_probe_runner.py mat_channel_scenario_runner.py scenario_a_runner.py
  "$PYTHON_BIN" -m unittest discover -s tests -q
fi

echo "Starting Scenario A at $(date --iso-8601=seconds)"
echo "GPU_ID=$GPU_ID DATA_DIR=$DATA_DIR LOG_DIR=$LOG_DIR"
echo "EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE LOCAL_STEPS=$LOCAL_STEPS EVALUATION_BATCHES=$EVALUATION_BATCHES"
echo "stdout log: $STDOUT_LOG"

SCENARIO_ARGS=(
  --probe-report "$PROBE_REPORT"
  --data-dir "$DATA_DIR"
  --log-dir "$LOG_DIR"
  --device cuda
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --local-steps "$LOCAL_STEPS"
  --evaluation-batches "$EVALUATION_BATCHES"
)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  SCENARIO_ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT" --start-cycle "$START_CYCLE")
  echo "Resuming from $RESUME_CHECKPOINT at cycle $START_CYCLE"
fi

"$PYTHON_BIN" mat_channel_scenario_runner.py "${SCENARIO_ARGS[@]}" 2>&1 | tee "$STDOUT_LOG"
