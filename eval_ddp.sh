#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/scratch/shahils/futurePredVLA/checkpoints_future_pred_vla/ckpt_step_10000.pt}"
if [[ -z "$CHECKPOINT_PATH" ]]; then
  echo "Set CHECKPOINT_PATH to a saved .pt checkpoint." >&2
  exit 1
fi

accelerate launch --num_processes "${NUM_PROCESSES:-1}" eval.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --eval_mode "${EVAL_MODE:-val}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --num_workers "${NUM_WORKERS:-0}" \
  --annotation_path "${ANNOTATION_PATH:-}" \
  --media_root "${MEDIA_ROOT:-}" \
  --data_root "${DATA_ROOT:-/scratch/shahils/driod_dataset/droid}" \
  --save_dir "${SAVE_DIR:-}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --metrics_path "${METRICS_PATH:-}" \
  --max_eval_batches "${MAX_EVAL_BATCHES:-0}" \
  --log_every "${LOG_EVERY:-100}" \
  "$@"
