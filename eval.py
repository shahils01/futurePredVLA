import argparse
import json
import os
from types import SimpleNamespace

import torch
from accelerate import Accelerator

try:
    from accelerate import DataLoaderConfiguration
except Exception:
    DataLoaderConfiguration = None

try:
    from accelerate.utils import DistributedDataParallelKwargs
except Exception:
    DistributedDataParallelKwargs = None

from data_loading import build_train_loader, resolve_action_stats
from train import _apply_peft, _configure_model_optimizations, _load_checkpoint_state, _set_loader_epoch, build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--eval_mode", type=str, default="val", choices=["val", "train"])
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--annotation_path", type=str, default="")
    parser.add_argument("--media_root", type=str, default="")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--mixed_precision", type=str, default="")
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--metrics_path", type=str, default="")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--val_ratio", type=float, default=-1.0)
    parser.add_argument("--max_samples_per_split", type=int, default=-1)
    parser.add_argument("--ddp_find_unused_parameters", action="store_true")
    return parser.parse_args()


def _checkpoint_args_namespace(checkpoint_path: str) -> tuple[SimpleNamespace, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    raw_args = dict(checkpoint.get("args", {}))
    if not raw_args:
        raise RuntimeError(f"Checkpoint does not contain saved args: {checkpoint_path}")
    return SimpleNamespace(**raw_args), checkpoint


def _apply_overrides(args: SimpleNamespace, cli_args):
    if cli_args.annotation_path:
        args.annotation_path = cli_args.annotation_path
    if cli_args.media_root:
        args.media_root = cli_args.media_root
    if cli_args.data_root:
        args.data_root = cli_args.data_root
    if cli_args.save_dir:
        args.save_dir = cli_args.save_dir
    if cli_args.batch_size > 0:
        args.batch_size = cli_args.batch_size
    if cli_args.num_workers >= 0:
        args.num_workers = cli_args.num_workers
    if cli_args.mixed_precision:
        args.mixed_precision = cli_args.mixed_precision
    if cli_args.log_every >= 0:
        args.log_every = cli_args.log_every
    if cli_args.val_ratio >= 0.0:
        args.val_ratio = cli_args.val_ratio
    if cli_args.max_samples_per_split >= 0:
        args.max_samples_per_split = cli_args.max_samples_per_split

    if not hasattr(args, "ddp_find_unused_parameters"):
        args.ddp_find_unused_parameters = False
    if cli_args.ddp_find_unused_parameters:
        args.ddp_find_unused_parameters = True

    if not hasattr(args, "image_keys"):
        args.image_keys = [args.image_key]
    elif isinstance(args.image_keys, str):
        args.image_keys = [item.strip() for item in str(args.image_keys).split(",") if item.strip()] or [args.image_key]
    if not hasattr(args, "future_image_keys"):
        args.future_image_keys = [args.future_image_key]
    elif isinstance(args.future_image_keys, str):
        args.future_image_keys = [item.strip() for item in str(args.future_image_keys).split(",") if item.strip()] or [args.future_image_key]
    if not hasattr(args, "robot_state_keys"):
        args.robot_state_keys = []
    elif isinstance(args.robot_state_keys, str):
        args.robot_state_keys = [item.strip() for item in str(args.robot_state_keys).split(",") if item.strip()]
    if not hasattr(args, "state_conditioning"):
        args.state_conditioning = "text"
    if not hasattr(args, "include_robot_state"):
        args.include_robot_state = args.state_conditioning != "off"
    else:
        args.include_robot_state = bool(args.include_robot_state or args.state_conditioning != "off")
    if not hasattr(args, "disable_vl_cache"):
        args.disable_vl_cache = True
    if not hasattr(args, "gradient_checkpointing"):
        args.gradient_checkpointing = False
    if not hasattr(args, "allow_tf32"):
        args.allow_tf32 = False
    if not hasattr(args, "wandb"):
        args.wandb = False
    return args


def _build_accelerator(args):
    ddp_kwargs = None
    if DistributedDataParallelKwargs is not None:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=bool(args.ddp_find_unused_parameters))
    dataloader_config = None
    if DataLoaderConfiguration is not None:
        dataloader_config = DataLoaderConfiguration(split_batches=False, dispatch_batches=False)
    return Accelerator(
        mixed_precision=getattr(args, "mixed_precision", "bf16"),
        kwargs_handlers=[ddp_kwargs] if ddp_kwargs is not None else None,
        dataloader_config=dataloader_config,
    )


def _denormalize(actions: torch.Tensor, action_stats):
    if action_stats is None:
        return actions
    mean = action_stats["mean"].to(actions.device).view(1, 1, -1)
    std = action_stats["std"].to(actions.device).view(1, 1, -1)
    return actions * std + mean


def evaluate(model, loader, accelerator, args, action_stats):
    model.eval()
    metrics = {
        "loss_sum": torch.zeros(1, device=accelerator.device),
        "action_loss_sum": torch.zeros(1, device=accelerator.device),
        "future_loss_sum": torch.zeros(1, device=accelerator.device),
        "action_mae_sum": torch.zeros(1, device=accelerator.device),
        "action_mse_sum": torch.zeros(1, device=accelerator.device),
        "action_mae_denorm_sum": torch.zeros(1, device=accelerator.device),
        "action_mse_denorm_sum": torch.zeros(1, device=accelerator.device),
        "examples": torch.zeros(1, device=accelerator.device),
    }

    max_eval_batches = int(getattr(args, "max_eval_batches", 0))
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            if max_eval_batches > 0 and step > max_eval_batches:
                break
            current_inputs = {
                key: value.to(accelerator.device) if torch.is_tensor(value) else value
                for key, value in batch["current_inputs"].items()
            }
            future_inputs = {
                key: value.to(accelerator.device) if torch.is_tensor(value) else value
                for key, value in batch["future_inputs"].items()
            }
            current_robot_state = batch["current_robot_state"].to(accelerator.device)
            future_robot_state = batch["future_robot_state"].to(accelerator.device)
            actions = batch["actions"].to(accelerator.device)

            outputs = model(
                current_inputs=current_inputs,
                future_inputs=future_inputs,
                robot_state=current_robot_state,
                future_robot_state=future_robot_state,
                actions=actions,
                inject_layer_idx=args.inject_layer_idx,
                num_future_samples=args.num_future_samples,
                flow_sampling_steps=args.flow_sampling_steps,
            )

            batch_size = actions.size(0)
            pred_actions = outputs["pred_actions"]
            action_mae = (pred_actions - actions).abs().mean()
            action_mse = ((pred_actions - actions) ** 2).mean()
            pred_actions_denorm = _denormalize(pred_actions, action_stats)
            actions_denorm = _denormalize(actions, action_stats)
            action_mae_denorm = (pred_actions_denorm - actions_denorm).abs().mean()
            action_mse_denorm = ((pred_actions_denorm - actions_denorm) ** 2).mean()

            loss = outputs["loss"] if outputs["loss"] is not None else torch.zeros((), device=accelerator.device)
            action_loss = outputs["action_loss"] if outputs["action_loss"] is not None else torch.zeros((), device=accelerator.device)
            future_loss = outputs["future_loss"] if outputs["future_loss"] is not None else torch.zeros((), device=accelerator.device)

            local = torch.tensor(
                [
                    float(loss.item()) * batch_size,
                    float(action_loss.item()) * batch_size,
                    float(future_loss.item()) * batch_size,
                    float(action_mae.item()) * batch_size,
                    float(action_mse.item()) * batch_size,
                    float(action_mae_denorm.item()) * batch_size,
                    float(action_mse_denorm.item()) * batch_size,
                    float(batch_size),
                ],
                device=accelerator.device,
            )
            gathered = accelerator.gather_for_metrics(local[None, :]).sum(dim=0)
            if accelerator.is_main_process:
                metrics["loss_sum"] += gathered[0:1]
                metrics["action_loss_sum"] += gathered[1:2]
                metrics["future_loss_sum"] += gathered[2:3]
                metrics["action_mae_sum"] += gathered[3:4]
                metrics["action_mse_sum"] += gathered[4:5]
                metrics["action_mae_denorm_sum"] += gathered[5:6]
                metrics["action_mse_denorm_sum"] += gathered[6:7]
                metrics["examples"] += gathered[7:8]

            if args.log_every > 0 and step % args.log_every == 0:
                accelerator.print(f"eval step={step}")

    if not accelerator.is_main_process:
        return None

    examples = max(float(metrics["examples"].item()), 1.0)
    return {
        "eval_mode": args.eval_mode,
        "examples": int(metrics["examples"].item()),
        "loss": float(metrics["loss_sum"].item() / examples),
        "action_loss": float(metrics["action_loss_sum"].item() / examples),
        "future_loss": float(metrics["future_loss_sum"].item() / examples),
        "action_mae": float(metrics["action_mae_sum"].item() / examples),
        "action_mse": float(metrics["action_mse_sum"].item() / examples),
        "action_mae_denorm": float(metrics["action_mae_denorm_sum"].item() / examples),
        "action_mse_denorm": float(metrics["action_mse_denorm_sum"].item() / examples),
        "checkpoint_path": args.checkpoint_path,
    }


def main():
    cli_args = parse_args()
    args, checkpoint = _checkpoint_args_namespace(cli_args.checkpoint_path)
    args = _apply_overrides(args, cli_args)
    args.checkpoint_path = cli_args.checkpoint_path
    args.max_eval_batches = cli_args.max_eval_batches

    accelerator = _build_accelerator(args)
    action_stats = resolve_action_stats(args) if getattr(args, "normalize_actions", False) else None

    split_name = args.val_split if cli_args.eval_mode == "val" else args.train_split
    is_train = cli_args.eval_mode == "train"
    loader = build_train_loader(args, split_name, args.batch_size, args.num_workers, is_train=is_train)

    model = build_model(args, accelerator.device)
    model = _apply_peft(model, args)
    _configure_model_optimizations(model, args)
    model, loader = accelerator.prepare(model, loader)
    _load_checkpoint_state(model, checkpoint["model"], accelerator)
    _set_loader_epoch(loader, 0)

    accelerator.print(f"evaluating checkpoint={cli_args.checkpoint_path} mode={cli_args.eval_mode}")
    metrics = evaluate(model, loader, accelerator, args, action_stats)
    if not accelerator.is_main_process:
        return

    metrics_path = cli_args.metrics_path or os.path.join(
        args.save_dir,
        f"eval_{cli_args.eval_mode}_{os.path.splitext(os.path.basename(cli_args.checkpoint_path))[0]}.json",
    )
    metrics_dir = os.path.dirname(metrics_path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    accelerator.print(json.dumps(metrics, indent=2, sort_keys=True))
    accelerator.print(f"saved metrics={metrics_path}")


if __name__ == "__main__":
    main()
