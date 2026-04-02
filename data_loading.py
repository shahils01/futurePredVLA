import csv
import hashlib
import json
import os
from glob import glob

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler


def _normalize_media_size(image_size):
    if isinstance(image_size, (tuple, list)):
        if len(image_size) >= 2:
            return {"height": int(image_size[0]), "width": int(image_size[1])}
        if len(image_size) == 1:
            size = int(image_size[0])
            return {"height": size, "width": size}
    if image_size is None:
        return None
    size = int(image_size)
    return {"height": size, "width": size}


def build_vlm_processor(args):
    from transformers import AutoConfig, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.vl_model_name, trust_remote_code=True)
    cfg_hf = AutoConfig.from_pretrained(args.vl_model_name, trust_remote_code=True)
    vision_cfg = getattr(cfg_hf, "vision_config", None)
    media_size = _normalize_media_size(getattr(vision_cfg, "image_size", None) if vision_cfg is not None else None)
    if media_size is not None:
        for proc_name in ("image_processor", "video_processor"):
            proc = getattr(processor, proc_name, None)
            if proc is None:
                continue
            if hasattr(proc, "size"):
                proc.size = dict(media_size)
            if hasattr(proc, "crop_size"):
                proc.crop_size = dict(media_size)
    return processor


def _sample_frame_indices(num_frames: int, num_samples: int):
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if num_frames <= num_samples:
        return list(range(num_frames))
    return np.linspace(0, num_frames - 1, num=num_samples, dtype=np.int64).tolist()


def _parse_timecode_seconds(value):
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        return float(text)
    except ValueError:
        return None


def _expand_annotation_paths(path_spec: str):
    raw_parts = [part.strip() for part in str(path_spec).split(",") if part.strip()]
    expanded = []
    for part in raw_parts:
        if os.path.isdir(part):
            expanded.extend(sorted(glob(os.path.join(part, "*.json"))))
            expanded.extend(sorted(glob(os.path.join(part, "*.jsonl"))))
            expanded.extend(sorted(glob(os.path.join(part, "*.csv"))))
        else:
            expanded.append(part)
    deduped = []
    seen = set()
    for path in expanded:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def _load_single_records(path: str):
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            obj = json.load(handle)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ("data", "samples", "items", "annotations"):
                if isinstance(obj.get(key), list):
                    return obj[key]
        raise RuntimeError("JSON manifest must contain a list or a data/items/samples/annotations field.")
    if path.endswith(".csv"):
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise RuntimeError(f"Unsupported annotation file format: {path}")


def _load_records(args):
    if not args.annotation_path:
        raise RuntimeError("DROID training requires --annotation_path pointing to a local manifest.")
    records = []
    for path in _expand_annotation_paths(args.annotation_path):
        records.extend(_load_single_records(path))
    if not records:
        raise RuntimeError(f"No records loaded from {args.annotation_path}")
    return records


def _stable_fold(value: str, seed: int) -> float:
    digest = hashlib.md5(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _get_first(record, keys, default=None):
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def _load_image(path: str) -> Image.Image:
    with Image.open(path) as handle:
        return handle.convert("RGB")


def _resolve_frame_paths(root: str, value):
    if value is None:
        return None
    if isinstance(value, str):
        if value.endswith(".json"):
            with open(value, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            value = loaded
        elif "," in value:
            value = [part.strip() for part in value.split(",") if part.strip()]
        else:
            value = [value]
    if not isinstance(value, (list, tuple)):
        return None
    paths = []
    for item in value:
        path = str(item)
        if root and not os.path.isabs(path):
            path = os.path.join(root, path)
        paths.append(path)
    return paths


def _frames_from_paths(frame_paths, num_frames):
    if not frame_paths:
        raise RuntimeError("No frame paths provided.")
    chosen = [frame_paths[idx] for idx in _sample_frame_indices(len(frame_paths), num_frames)]
    return [_load_image(path) for path in chosen]


def decode_mp4_frames(video_path: str, num_frames: int, start_time_sec=None, end_time_sec=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")
    start_frame = 0
    end_frame = frame_count - 1
    if fps > 0.0:
        if start_time_sec is not None:
            start_frame = max(0, min(frame_count - 1, int(start_time_sec * fps)))
        if end_time_sec is not None:
            end_frame = max(start_frame, min(frame_count - 1, int(end_time_sec * fps)))
    clip_frame_count = max(1, end_frame - start_frame + 1)
    target_indices = [start_frame + idx for idx in _sample_frame_indices(clip_frame_count, num_frames)]
    frames = []
    for frame_idx in target_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    if not frames:
        raise RuntimeError(f"Failed to decode frames from {video_path}")
    if len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    return frames


def _add_media_token(tokenizer, text: str) -> str:
    vocab = tokenizer.get_vocab()
    if any(token in text for token in ("<video>", "<image>", "<img>")):
        return text
    for token in ("<video>", "<image>", "<img>"):
        if token in vocab:
            return f"{token}\n{text}"
    return text


def build_prompt_only_example(processor, frames, prompt):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Processor does not expose a tokenizer.")

    if hasattr(processor, "apply_chat_template"):
        prompt_with_media = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_with_media = _add_media_token(tokenizer, f"User: {prompt}\nAssistant:")
    try:
        inputs = processor(text=[prompt_with_media], videos=[frames], return_tensors="pt", padding="longest", truncation=False)
    except TypeError:
        inputs = processor(text=[prompt_with_media], images=[frames], return_tensors="pt", padding="longest", truncation=False)
    packed = {}
    for key, value in dict(inputs).items():
        if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == 1:
            value = value.squeeze(0)
        packed[key] = value
    return packed


def _build_distributed_rank_info():
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank()), int(dist.get_world_size())
    rank = os.environ.get("RANK")
    world_size = os.environ.get("WORLD_SIZE")
    if rank is not None and world_size is not None:
        return int(rank), int(world_size)
    return 0, 1


class DroidManifestDataset(Dataset):
    def __init__(self, records, processor, args, split_name: str, is_train: bool):
        self.records = []
        self.processor = processor
        self.args = args
        self.split_name = split_name
        self.is_train = is_train
        max_samples = int(args.max_samples_per_split)
        for idx, record in enumerate(records):
            sample_id = str(_get_first(record, ["id", "sample_id", "uid"], default=idx))
            fold = _stable_fold(sample_id, args.seed)
            in_val = fold < max(0.0, min(0.5, float(args.val_ratio)))
            keep = (not in_val) if is_train else in_val
            if args.val_ratio == 0:
                keep = True if is_train else False
            if not keep:
                continue
            self.records.append(record)
            if max_samples > 0 and len(self.records) >= max_samples:
                break

    def __len__(self):
        return len(self.records)

    def _load_clip(self, record, frame_key, start_key, end_key, fallback_num_frames):
        frame_paths = _resolve_frame_paths(
            self.args.media_root,
            _get_first(record, [frame_key]),
        )
        if frame_paths:
            return _frames_from_paths(frame_paths, fallback_num_frames)
        video_path = _get_first(record, ["video_path", "clip_path", "path"])
        if video_path is None:
            raise RuntimeError(f"Record is missing {frame_key} and video_path.")
        if self.args.media_root and not os.path.isabs(video_path):
            video_path = os.path.join(self.args.media_root, str(video_path))
        return decode_mp4_frames(
            str(video_path),
            fallback_num_frames,
            start_time_sec=_parse_timecode_seconds(_get_first(record, [start_key])),
            end_time_sec=_parse_timecode_seconds(_get_first(record, [end_key])),
        )

    def __getitem__(self, index):
        record = dict(self.records[int(index)])
        sample_id = str(_get_first(record, ["id", "sample_id", "uid"], default=index))
        instruction = str(_get_first(record, ["instruction", "prompt", "language_instruction", "task"], default="What action should the robot take next?"))
        task_name = str(_get_first(record, ["task_name", "task", "skill"], default="droid"))
        current_frames = self._load_clip(record, "current_frame_paths", "current_start_sec", "current_end_sec", self.args.video_frames)
        future_frames = self._load_clip(record, "future_frame_paths", "future_start_sec", "future_end_sec", self.args.future_video_frames)

        current_inputs = build_prompt_only_example(self.processor, current_frames, instruction)
        future_inputs = build_prompt_only_example(self.processor, future_frames, instruction)

        actions = _get_first(record, ["actions", "action_chunk", "future_actions"])
        if isinstance(actions, str):
            actions = json.loads(actions)
        actions = torch.tensor(actions, dtype=torch.float32)
        if actions.dim() != 2:
            raise RuntimeError(f"Expected actions to have shape [H, A], got {tuple(actions.shape)}")

        if actions.size(0) != int(self.args.chunk_horizon):
            if actions.size(0) > int(self.args.chunk_horizon):
                actions = actions[: int(self.args.chunk_horizon)]
            else:
                pad = torch.zeros(int(self.args.chunk_horizon) - actions.size(0), actions.size(1), dtype=actions.dtype)
                actions = torch.cat([actions, pad], dim=0)
        if actions.size(1) != int(self.args.action_dim):
            raise RuntimeError(f"Expected action_dim={self.args.action_dim}, got {actions.size(1)} for sample {sample_id}")

        return {
            "id": sample_id,
            "task_name": task_name,
            "instruction": instruction,
            "current_inputs": current_inputs,
            "future_inputs": future_inputs,
            "actions": actions,
        }


def _stack_inputs(items):
    output = {}
    for key in items[0].keys():
        values = [item[key] for item in items]
        if torch.is_tensor(values[0]):
            if key == "pixel_values" and values[0].dim() == 4:
                output[key] = torch.cat(values, dim=0)
            elif key in {"input_ids", "attention_mask"}:
                max_len = max(int(v.shape[0]) for v in values)
                padded = [F.pad(v, (0, max_len - int(v.shape[0])), value=0) for v in values]
                output[key] = torch.stack(padded, dim=0)
            else:
                output[key] = torch.stack(values, dim=0)
        else:
            output[key] = values
    return output


def collate_droid_batch(batch):
    return {
        "ids": [item["id"] for item in batch],
        "task_names": [item["task_name"] for item in batch],
        "instructions": [item["instruction"] for item in batch],
        "current_inputs": _stack_inputs([item["current_inputs"] for item in batch]),
        "future_inputs": _stack_inputs([item["future_inputs"] for item in batch]),
        "actions": torch.stack([item["actions"] for item in batch], dim=0),
    }


def build_train_loader(args, split: str, batch_size: int, num_workers: int, is_train: bool):
    processor = build_vlm_processor(args)
    if args.dataset_type != "droid_manifest":
        raise RuntimeError(f"Unsupported dataset_type={args.dataset_type}. Expected droid_manifest.")
    dataset = DroidManifestDataset(_load_records(args), processor, args, split_name=split, is_train=is_train)
    sampler = None
    shuffle = False
    rank, world_size = _build_distributed_rank_info()
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=is_train, seed=args.seed, drop_last=False)
    else:
        shuffle = is_train
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_droid_batch,
        "pin_memory": torch.cuda.is_available(),
        "sampler": sampler,
        "shuffle": shuffle,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)
