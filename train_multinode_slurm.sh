#!/usr/bin/env bash
set -euo pipefail

VL_MODEL_NAME="${VL_MODEL_NAME:-/scratch/shahils/hf_models/InternVL3_5-2B-HF}"
DATASET_TYPE="${DATASET_TYPE:-droid_manifest}"
ANNOTATION_PATH="${ANNOTATION_PATH:-}"
MEDIA_ROOT="${MEDIA_ROOT:-}"
DATA_ROOT="${DATA_ROOT:-}"
SAVE_DIR="${SAVE_DIR:-checkpoints_future_pred_vla}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NNODES="${NNODES:-${SLURM_NNODES:-5}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-8}}"
MASTER_PORT="${MASTER_PORT:-29500}"
TORCHRUN_RDZV_BACKEND="${TORCHRUN_RDZV_BACKEND:-c10d}"

if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
else
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi

if [[ "$DATASET_TYPE" == "droid_manifest" && -z "$ANNOTATION_PATH" ]]; then
  echo "Set ANNOTATION_PATH to a DROID manifest (.json/.jsonl/.csv)." >&2
  exit 1
fi

if [[ "$DATASET_TYPE" == "droid_rlds" && -z "$DATA_ROOT" ]]; then
  echo "Set DATA_ROOT to a local TFDS/RLDS DROID directory." >&2
  exit 1
fi

TRAIN_CMD=$(cat <<EOF
cd ${SCRIPT_DIR}
torchrun \
  --nnodes=${NNODES} \
  --nproc_per_node=${GPUS_PER_NODE} \
  --node_rank=\${SLURM_NODEID} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  --rdzv_backend=${TORCHRUN_RDZV_BACKEND} \
  train.py \
    --dataset_type "$DATASET_TYPE" \
    --annotation_path "$ANNOTATION_PATH" \
    --media_root "$MEDIA_ROOT" \
    --data_root "$DATA_ROOT" \
    --vl_model_name "$VL_MODEL_NAME" \
    --batch_size "${BATCH_SIZE:-1}" \
    --num_workers "${NUM_WORKERS:-0}" \
    --video_frames "${VIDEO_FRAMES:-4}" \
    --future_video_frames "${FUTURE_VIDEO_FRAMES:-2}" \
    --current_history "${CURRENT_HISTORY:-4}" \
    --future_offset "${FUTURE_OFFSET:-8}" \
    --future_span "${FUTURE_SPAN:-4}" \
    --image_key "${IMAGE_KEY:-wrist_image_left}" \
    --future_image_key "${FUTURE_IMAGE_KEY:-wrist_image_left}" \
    --image_keys "${IMAGE_KEYS:-}" \
    --future_image_keys "${FUTURE_IMAGE_KEYS:-}" \
    --default_prompt "${DEFAULT_PROMPT:-You are controlling a robot from visual observations and task instructions.}" \
    --state_conditioning "${STATE_CONDITIONING:-token}" \
    --robot_state_keys "${ROBOT_STATE_KEYS:-cartesian_position,gripper_position,joint_position}" \
    --robot_state_dim "${ROBOT_STATE_DIM:-14}" \
    --num_state_tokens "${NUM_STATE_TOKENS:-2}" \
    --robot_state_precision "${ROBOT_STATE_PRECISION:-4}" \
    --rlds_episode_shuffle_buffer "${RLDS_EPISODE_SHUFFLE_BUFFER:-100}" \
    --rlds_max_samples_per_episode "${RLDS_MAX_SAMPLES_PER_EPISODE:-64}" \
    --normalize_actions \
    --action_stats_path "${ACTION_STATS_PATH:-}" \
    --action_stats_max_episodes "${ACTION_STATS_MAX_EPISODES:-0}" \
    --action_stats_max_steps "${ACTION_STATS_MAX_STEPS:-500000}" \
    --chunk_horizon "${CHUNK_HORIZON:-16}" \
    --action_dim "${ACTION_DIM:-7}" \
    --predictor_hidden_dim "${PREDICTOR_HIDDEN_DIM:-1024}" \
    --num_future_tokens "${NUM_FUTURE_TOKENS:-4}" \
    --future_model_num_layers "${FUTURE_MODEL_NUM_LAYERS:-4}" \
    --future_model_num_heads "${FUTURE_MODEL_NUM_HEADS:-8}" \
    --future_model_dropout "${FUTURE_MODEL_DROPOUT:-0.1}" \
    --num_future_samples "${NUM_FUTURE_SAMPLES:-4}" \
    --flow_sampling_steps "${FLOW_SAMPLING_STEPS:-16}" \
    --action_head_type "${ACTION_HEAD_TYPE:-flow}" \
    --action_flow_hidden_dim "${ACTION_FLOW_HIDDEN_DIM:-1024}" \
    --action_flow_num_layers "${ACTION_FLOW_NUM_LAYERS:-4}" \
    --action_flow_num_heads "${ACTION_FLOW_NUM_HEADS:-8}" \
    --action_flow_dropout "${ACTION_FLOW_DROPOUT:-0.1}" \
    --policy_conditioning "${POLICY_CONDITIONING:-token}" \
    --policy_num_queries "${POLICY_NUM_QUERIES:-4}" \
    --policy_num_heads "${POLICY_NUM_HEADS:-8}" \
    "${USE_FUTURE_PREDICTION_FLAG:---use_future_prediction}" \
    --grad_accum_steps "${GRAD_ACCUM_STEPS:-1}" \
    --epochs "${EPOCHS:-10}" \
    --lr "${LR:-1e-5}" \
    --weight_decay "${WEIGHT_DECAY:-0.01}" \
    --mixed_precision "${MIXED_PRECISION:-bf16}" \
    --save_every_steps "${SAVE_EVERY_STEPS:-5000}" \
    --max_step_checkpoints "${MAX_STEP_CHECKPOINTS:-3}" \
    --save_latest \
    --save_dir "$SAVE_DIR" \
    ${GRADIENT_CHECKPOINTING_FLAG:---gradient_checkpointing} \
    ${DDP_UNUSED_FLAG:---ddp_find_unused_parameters} \
    "$@"
EOF
)

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun \
    --nodes="${NNODES}" \
    --ntasks="${NNODES}" \
    --ntasks-per-node=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    bash -lc "$TRAIN_CMD"
else
  echo "SLURM_JOB_ID is not set. Run this script inside sbatch/salloc." >&2
  exit 1
fi
