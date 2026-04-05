import argparse
import functools
import inspect
import os
import shutil

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

try:
    from accelerate.utils import FullyShardedDataParallelPlugin
except Exception:
    try:
        from accelerate.utils import FSDPPlugin as FullyShardedDataParallelPlugin
    except Exception:
        FullyShardedDataParallelPlugin = None

try:
    from torch.distributed.fsdp import CPUOffload
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
except Exception:
    CPUOffload = None
    size_based_auto_wrap_policy = None

from data_loading import build_train_loader
from model import FuturePredVLA, ModelConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="droid_manifest", choices=["droid_manifest", "droid_rlds"])
    parser.add_argument("--annotation_path", type=str, default="")
    parser.add_argument("--media_root", type=str, default="")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="validation")
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--max_samples_per_split", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--video_frames", type=int, default=4)
    parser.add_argument("--future_video_frames", type=int, default=4)
    parser.add_argument("--chunk_horizon", type=int, default=16)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--current_history", type=int, default=4)
    parser.add_argument("--future_offset", type=int, default=8)
    parser.add_argument("--future_span", type=int, default=4)
    parser.add_argument("--rlds_split", type=str, default="train")
    parser.add_argument("--rlds_dataset_name", type=str, default="r2d2_faceblur")
    parser.add_argument("--image_key", type=str, default="wrist_image_left")
    parser.add_argument("--future_image_key", type=str, default="wrist_image_left")
    parser.add_argument("--rlds_episode_shuffle_buffer", type=int, default=500000)
    parser.add_argument("--rlds_shuffle_steps", action="store_true", default=True)
    parser.add_argument("--no_rlds_shuffle_steps", dest="rlds_shuffle_steps", action="store_false")
    parser.add_argument("--rlds_max_samples_per_episode", type=int, default=64)
    parser.add_argument("--normalize_actions", action="store_true")
    parser.add_argument("--action_stats_path", type=str, default="")
    parser.add_argument("--action_stats_max_episodes", type=int, default=0)
    parser.add_argument("--action_stats_max_steps", type=int, default=500000)

    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--fsdp_min_num_params", type=int, default=1_000_000)
    parser.add_argument("--fsdp_cpu_offload", action="store_true")
    parser.add_argument("--fsdp_use_orig_params", action="store_true")
    parser.add_argument("--ddp_find_unused_parameters", action="store_true")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--disable_vl_cache", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")

    parser.add_argument("--vl_backend", type=str, default="internvl", choices=["internvl"])
    parser.add_argument("--vl_model_name", type=str, default="OpenGVLab/InternVL3_5-2B-HF")
    parser.add_argument("--vl_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--freeze_vl", action="store_true")
    parser.add_argument("--peft", type=str, default="none", choices=["none", "lora", "qlora"])
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="")
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])

    parser.add_argument("--predictor_hidden_dim", type=int, default=2048)
    parser.add_argument("--num_future_tokens", type=int, default=2)
    parser.add_argument("--inject_layer_idx", type=int, default=1)
    parser.add_argument("--num_future_samples", type=int, default=4)
    parser.add_argument("--flow_sampling_steps", type=int, default=16)
    parser.add_argument("--use_future_prediction", action="store_true", default=True)
    parser.add_argument("--disable_future_prediction", dest="use_future_prediction", action="store_false")
    parser.add_argument("--action_head_type", type=str, default="regression", choices=["regression", "flow"])
    parser.add_argument("--action_flow_hidden_dim", type=int, default=2048)
    parser.add_argument("--future_loss_weight", type=float, default=1.0)
    parser.add_argument("--action_loss_weight", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="checkpoints_future_pred_vla")
    parser.add_argument("--save_every_steps", type=int, default=5000)
    parser.add_argument("--max_step_checkpoints", type=int, default=3)
    parser.add_argument("--save_latest", action="store_true")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--load_model_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="future-pred-vla")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    return parser.parse_args()


def _parse_lora_targets(args):
    if args.lora_target_modules:
        return [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _apply_peft(model, args):
    if args.peft == "none":
        return model
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as exc:
        raise RuntimeError("PEFT requested but `peft` is not installed.") from exc

    for param in model.backbone.model.parameters():
        param.requires_grad = False

    if args.peft == "qlora":
        model.backbone.model = prepare_model_for_kbit_training(model.backbone.model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=_parse_lora_targets(args),
        task_type="CAUSAL_LM",
    )
    model.backbone.model = get_peft_model(model.backbone.model, lora_cfg)
    return model


def _configure_model_optimizations(model, args):
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(model.backbone.model, "config") and hasattr(model.backbone.model.config, "use_cache"):
        model.backbone.model.config.use_cache = False if args.disable_vl_cache or args.gradient_checkpointing else model.backbone.model.config.use_cache
    if args.gradient_checkpointing:
        fn = getattr(model.backbone.model, "gradient_checkpointing_enable", None)
        if callable(fn):
            try:
                fn(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                fn()
        enable_inputs = getattr(model.backbone.model, "enable_input_require_grads", None)
        if callable(enable_inputs):
            try:
                enable_inputs()
            except Exception:
                pass


def _count_parameters(model):
    total = 0
    trainable = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return total, trainable


def _load_checkpoint_state(model, ckpt_state, accelerator):
    unwrapped = accelerator.unwrap_model(model)
    try:
        unwrapped.load_state_dict(ckpt_state)
    except RuntimeError:
        incompatible = unwrapped.load_state_dict(ckpt_state, strict=False)
        accelerator.print(
            "Non-strict checkpoint load complete: "
            f"missing_keys={len(getattr(incompatible, 'missing_keys', []))} "
            f"unexpected_keys={len(getattr(incompatible, 'unexpected_keys', []))}"
        )


def build_model(args, device):
    cfg = ModelConfig(
        vl_backend=args.vl_backend,
        vl_model_name=args.vl_model_name,
        vl_dtype=args.vl_dtype,
        freeze_vl=args.freeze_vl,
        quantization_config=getattr(args, "quantization_config", None),
        use_cache=not args.disable_vl_cache,
    )
    return FuturePredVLA(
        cfg=cfg,
        device=device,
        action_dim=args.action_dim,
        chunk_horizon=args.chunk_horizon,
        predictor_hidden_dim=args.predictor_hidden_dim,
        num_future_tokens=args.num_future_tokens,
        use_future_prediction=args.use_future_prediction,
        action_head_type=args.action_head_type,
        action_flow_hidden_dim=args.action_flow_hidden_dim,
    )


def _checkpoint_state(model, optimizer, args, epoch: int, global_step: int):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
    }


def _rotate_step_checkpoints(save_dir: str, max_to_keep: int):
    if int(max_to_keep) <= 0:
        return
    candidates = []
    for filename in os.listdir(save_dir):
        if not (filename.startswith("ckpt_step_") and filename.endswith(".pt")):
            continue
        path = os.path.join(save_dir, filename)
        try:
            step = int(filename[len("ckpt_step_") : -len(".pt")])
        except ValueError:
            continue
        candidates.append((step, path))
    candidates.sort()
    while len(candidates) > int(max_to_keep):
        _, path = candidates.pop(0)
        if os.path.exists(path):
            os.remove(path)


def _save_checkpoint(accelerator, model, optimizer, args, epoch: int, global_step: int, tag: str, rotate_steps: bool = False):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(model)
    ckpt_path = os.path.join(args.save_dir, f"{tag}.pt")
    torch.save(
        _checkpoint_state(unwrapped, optimizer, args, epoch=epoch, global_step=global_step),
        ckpt_path,
    )
    if args.save_latest:
        latest_path = os.path.join(args.save_dir, "latest.pt")
        shutil.copy2(ckpt_path, latest_path)
    if rotate_steps and tag.startswith("ckpt_step_"):
        _rotate_step_checkpoints(args.save_dir, args.max_step_checkpoints)
    accelerator.print(f"saved checkpoint={ckpt_path}")


def run_epoch(model, loader, optimizer, accelerator, args, train: bool, global_step: int, epoch: int):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_action_loss = 0.0
    total_future_loss = 0.0
    total_examples = 0
    step = 0

    for batch in loader:
        step += 1
        current_inputs = {
            key: value.to(accelerator.device) if torch.is_tensor(value) else value
            for key, value in batch["current_inputs"].items()
        }
        future_inputs = {
            key: value.to(accelerator.device) if torch.is_tensor(value) else value
            for key, value in batch["future_inputs"].items()
        }
        actions = batch["actions"].to(accelerator.device)

        with accelerator.accumulate(model):
            with torch.set_grad_enabled(train):
                outputs = model(
                    current_inputs=current_inputs,
                    future_inputs=future_inputs,
                    actions=actions,
                    inject_layer_idx=args.inject_layer_idx,
                    num_future_samples=args.num_future_samples,
                    flow_sampling_steps=args.flow_sampling_steps,
                )
                loss = 0.0
                if outputs["future_loss"] is not None:
                    loss = loss + args.future_loss_weight * outputs["future_loss"]
                if outputs["action_loss"] is not None:
                    loss = loss + args.action_loss_weight * outputs["action_loss"]
                if train:
                    accelerator.backward(loss)
                    if args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        batch_size = actions.size(0)
        total_examples += batch_size
        total_loss += float(loss.detach().item()) * batch_size
        total_action_loss += float((outputs["action_loss"].detach().item() if outputs["action_loss"] is not None else 0.0)) * batch_size
        total_future_loss += float((outputs["future_loss"].detach().item() if outputs["future_loss"] is not None else 0.0)) * batch_size

        if args.log_every > 0 and step % args.log_every == 0:
            phase = "train" if train else "val"
            avg_loss = total_loss / max(total_examples, 1)
            avg_action = total_action_loss / max(total_examples, 1)
            avg_future = total_future_loss / max(total_examples, 1)
            accelerator.print(f"{phase} step={step} loss={avg_loss:.4f} action={avg_action:.4f} future={avg_future:.4f}")
            if args.wandb:
                metrics = {
                    f"{phase}/loss": avg_loss,
                    f"{phase}/action_loss": avg_action,
                    f"{phase}/future_loss": avg_future,
                }
                if train:
                    metrics["train/lr"] = optimizer.param_groups[0]["lr"]
                    accelerator.log(metrics, step=global_step + step)
                else:
                    accelerator.log(metrics, step=global_step)

        if train and int(args.save_every_steps) > 0:
            absolute_step = global_step + step
            if absolute_step % int(args.save_every_steps) == 0:
                _save_checkpoint(
                    accelerator=accelerator,
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    epoch=epoch,
                    global_step=absolute_step,
                    tag=f"ckpt_step_{absolute_step}",
                    rotate_steps=True,
                )
    avg_total = total_loss / max(total_examples, 1)
    avg_action = total_action_loss / max(total_examples, 1)
    avg_future = total_future_loss / max(total_examples, 1)
    return {"loss": avg_total, "action_loss": avg_action, "future_loss": avg_future}, (global_step + step if train else global_step)


def _set_loader_epoch(loader, epoch: int):
    dataset = getattr(loader, "dataset", None)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.peft == "qlora" and args.fsdp:
        raise RuntimeError("FSDP + QLoRA is not supported.")
    if args.peft == "qlora":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError("QLoRA requested but bitsandbytes/transformers are not available.") from exc
        compute_dtype = torch.bfloat16 if args.vl_dtype == "bfloat16" else (torch.float16 if args.vl_dtype == "float16" else torch.float32)
        args.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        args.quantization_config = None

    fsdp_plugin = None
    if args.fsdp:
        if FullyShardedDataParallelPlugin is None:
            raise RuntimeError("FSDP requested but accelerate FSDP plugin is unavailable.")
        fsdp_kwargs = {}
        use_orig_params = args.fsdp_use_orig_params or (args.peft != "none")
        if size_based_auto_wrap_policy is not None:
            fsdp_kwargs["auto_wrap_policy"] = functools.partial(size_based_auto_wrap_policy, min_num_params=args.fsdp_min_num_params)
        else:
            try:
                params = inspect.signature(FullyShardedDataParallelPlugin).parameters
                if "min_num_params" in params:
                    fsdp_kwargs["min_num_params"] = args.fsdp_min_num_params
            except Exception:
                pass
        try:
            params = inspect.signature(FullyShardedDataParallelPlugin).parameters
            if "use_orig_params" in params:
                fsdp_kwargs["use_orig_params"] = use_orig_params
        except Exception:
            pass
        if args.fsdp_cpu_offload:
            if CPUOffload is None:
                raise RuntimeError("FSDP CPU offload requested but torch.distributed.fsdp is unavailable.")
            fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)
        fsdp_plugin = FullyShardedDataParallelPlugin(**fsdp_kwargs)

    ddp_kwargs = None
    if not args.fsdp and DistributedDataParallelKwargs is not None:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=args.ddp_find_unused_parameters)
    dataloader_config = None
    if DataLoaderConfiguration is not None:
        dataloader_config = DataLoaderConfiguration(split_batches=False, dispatch_batches=False)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum_steps,
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=[ddp_kwargs] if ddp_kwargs is not None else None,
        dataloader_config=dataloader_config,
        log_with="wandb" if args.wandb else None,
    )

    if args.wandb:
        init_kwargs = {"wandb": {}}
        if args.wandb_entity:
            init_kwargs["wandb"]["entity"] = args.wandb_entity
        if args.wandb_run_name:
            init_kwargs["wandb"]["name"] = args.wandb_run_name
        accelerator.init_trackers(project_name=args.wandb_project, init_kwargs=init_kwargs)

    train_loader = build_train_loader(args, args.train_split, args.batch_size, args.num_workers, is_train=True)
    val_loader = build_train_loader(args, args.val_split, args.batch_size, args.num_workers, is_train=False) if args.val_ratio > 0 else None

    model = build_model(args, accelerator.device)
    model = _apply_peft(model, args)
    _configure_model_optimizations(model, args)

    total_params, trainable_params = _count_parameters(model)
    accelerator.print(f"parameters total={total_params:,} trainable={trainable_params:,}")

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    resume_ckpt = torch.load(args.resume_checkpoint, map_location="cpu") if args.resume_checkpoint else None
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    start_epoch = 0
    global_step = 0
    if resume_ckpt is not None:
        _load_checkpoint_state(model, resume_ckpt["model"], accelerator)
        if not args.load_model_only:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
            global_step = int(resume_ckpt.get("global_step", 0))
            accelerator.print(f"resumed checkpoint={args.resume_checkpoint} start_epoch={start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        _set_loader_epoch(train_loader, epoch)
        train_metrics, global_step = run_epoch(model, train_loader, optimizer, accelerator, args, True, global_step, epoch)
        accelerator.print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"train_action={train_metrics['action_loss']:.4f} train_future={train_metrics['future_loss']:.4f}"
        )
        if args.wandb:
            accelerator.log({f"train/epoch_{key}": value for key, value in train_metrics.items()}, step=global_step)

        if val_loader is not None:
            _set_loader_epoch(val_loader, epoch)
            with torch.no_grad():
                val_metrics, _ = run_epoch(model, val_loader, optimizer, accelerator, args, False, global_step, epoch)
            accelerator.print(
                f"epoch={epoch} val_loss={val_metrics['loss']:.4f} "
                f"val_action={val_metrics['action_loss']:.4f} val_future={val_metrics['future_loss']:.4f}"
            )
            if args.wandb:
                accelerator.log({f"val/epoch_{key}": value for key, value in val_metrics.items()}, step=global_step)

        _save_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            args=args,
            epoch=epoch,
            global_step=global_step,
            tag=f"ckpt_epoch_{epoch}",
            rotate_steps=False,
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
