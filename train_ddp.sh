#!/usr/bin/env bash
set -euo pipefail

VL_MODEL_NAME="${VL_MODEL_NAME:-/scratch/shahils/hf_models/InternVL3_5-2B-HF}"
ANNOTATION_PATH="${ANNOTATION_PATH:-}"
MEDIA_ROOT="${MEDIA_ROOT:-}"
SAVE_DIR="${SAVE_DIR:-checkpoints_future_pred_vla}"

if [[ -z "$ANNOTATION_PATH" ]]; then
  echo "Set ANNOTATION_PATH to a DROID manifest (.json/.jsonl/.csv)." >&2
  exit 1
fi

accelerate launch --num_processes 8 train.py \
  --dataset_type droid_manifest \
  --annotation_path "$ANNOTATION_PATH" \
  --media_root "$MEDIA_ROOT" \
  --vl_model_name "$VL_MODEL_NAME" \
  --batch_size "${BATCH_SIZE:-1}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --video_frames "${VIDEO_FRAMES:-4}" \
  --future_video_frames "${FUTURE_VIDEO_FRAMES:-4}" \
  --chunk_horizon "${CHUNK_HORIZON:-16}" \
  --action_dim "${ACTION_DIM:-7}" \
  --predictor_hidden_dim "${PREDICTOR_HIDDEN_DIM:-2048}" \
  --num_future_tokens "${NUM_FUTURE_TOKENS:-2}" \
  --num_future_samples "${NUM_FUTURE_SAMPLES:-4}" \
  --flow_sampling_steps "${FLOW_SAMPLING_STEPS:-16}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS:-1}" \
  --epochs "${EPOCHS:-3}" \
  --lr "${LR:-2e-5}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --save_dir "$SAVE_DIR" \
  "$@"
