# futurePredVLA

DDP-friendly codebase for future-distribution-conditioned VLA training with an InternVL backbone.

This project reuses the stable InternVL and `accelerate` patterns from `Belief-VLM`, but removes vector memory entirely. The default training path targets DROID-style robot trajectories through a local manifest.

## Implemented method

- InternVL encoder for current observation clips and language instruction
- Conditional flow-matching future latent predictor
- Sample-based predictive distribution summary
- Projection of the predictive summary into the language token space
- Continuous action-chunk regression head
- `accelerate`-based multi-GPU training

## Expected dataset format

The current implementation uses a local `json/jsonl/csv` manifest. Each row should define one training sample with:

- `id`: sample id
- `instruction`: language instruction
- `actions`: nested list with shape `[chunk_horizon, action_dim]`

And either:

- `current_frame_paths`: list of image paths
- `future_frame_paths`: list of image paths

Or:

- `video_path`: path to a video
- `current_start_sec`, `current_end_sec`
- `future_start_sec`, `future_end_sec`

Optional fields:

- `task_name`
- `robot_state`
- `gripper_state`
- `camera_name`

Example JSONL row:

```json
{
  "id": "episode_001_step_0042",
  "task_name": "open_drawer",
  "instruction": "open the drawer",
  "current_frame_paths": ["frames/ep1/40.png", "frames/ep1/41.png", "frames/ep1/42.png", "frames/ep1/43.png"],
  "future_frame_paths": ["frames/ep1/52.png", "frames/ep1/53.png", "frames/ep1/54.png", "frames/ep1/55.png"],
  "actions": [[0.01, -0.02, 0.00, 0.10, 0.00, 0.03, 1.0], [0.02, -0.03, 0.01, 0.08, 0.00, 0.02, 1.0]]
}
```

## Launch

```bash
accelerate launch --num_processes 8 train.py \
  --dataset_type droid_manifest \
  --annotation_path /path/to/droid_manifest.jsonl \
  --vl_model_name /path/to/InternVL3_5-2B-HF \
  --batch_size 1 \
  --grad_accum_steps 4 \
  --video_frames 4 \
  --future_video_frames 4 \
  --action_dim 7 \
  --chunk_horizon 16 \
  --save_dir checkpoints_future_pred_vla
```

## Notes

- The DROID loader is intentionally schema-flexible because local dataset exports vary.
- The code is ready for DDP/FSDP-style `accelerate` training, but you may need to adapt the manifest builder to your exact DROID export.
- The current action head is continuous regression. If your action parameterization differs, update `action_dim` and normalization accordingly.
