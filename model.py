from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vl_backend: str = "internvl"
    vl_model_name: str = "OpenGVLab/InternVL3_5-1B-HF"
    vl_dtype: str = "bfloat16"
    freeze_vl: bool = False
    quantization_config: Optional[Any] = None
    use_cache: bool = False


class InternVLBackbone(nn.Module):
    @staticmethod
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

    def _configure_processor_media_size(self, cfg_hf):
        vision_cfg = getattr(cfg_hf, "vision_config", None)
        image_size = getattr(vision_cfg, "image_size", None) if vision_cfg is not None else None
        media_size = self._normalize_media_size(image_size)
        if media_size is None:
            return

        for proc_name in ("image_processor", "video_processor"):
            proc = getattr(self.processor, proc_name, None)
            if proc is None:
                continue
            if hasattr(proc, "size"):
                proc.size = dict(media_size)
            if hasattr(proc, "crop_size"):
                proc.crop_size = dict(media_size)

    def __init__(self, cfg: ModelConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        if cfg.vl_dtype == "float16":
            dtype = torch.float16
        elif cfg.vl_dtype == "float32":
            dtype = torch.float32
        else:
            dtype = torch.bfloat16

        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
            try:
                from transformers.models.auto.modeling_auto import AutoModelForImageTextToText
            except Exception:
                AutoModelForImageTextToText = None
        except Exception as exc:
            raise ImportError("HF multimodal backbones require transformers installed.") from exc

        if cfg.vl_backend != "internvl":
            raise RuntimeError(f"Unsupported vl_backend={cfg.vl_backend}. This project supports InternVL only.")

        trust_remote_code = True
        cfg_hf = AutoConfig.from_pretrained(cfg.vl_model_name, trust_remote_code=trust_remote_code)
        self.processor = AutoProcessor.from_pretrained(cfg.vl_model_name, trust_remote_code=trust_remote_code)
        self.tokenizer = getattr(self.processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
            cfg.vl_model_name,
            trust_remote_code=trust_remote_code,
        )
        self._configure_processor_media_size(cfg_hf)

        model_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
        if cfg.quantization_config is not None:
            model_kwargs["quantization_config"] = cfg.quantization_config

        if AutoModelForImageTextToText is not None:
            self.model = AutoModelForImageTextToText.from_pretrained(cfg.vl_model_name, **model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.vl_model_name, **model_kwargs)

        self.model.to(device)
        if cfg.freeze_vl:
            for param in self.model.parameters():
                param.requires_grad = False
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = bool(cfg.use_cache)
        self._dtype = dtype

    def _move_inputs_to_device(self, inputs):
        moved = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                if key in ("pixel_values", "pixel_values_videos", "video_values", "video", "videos"):
                    moved[key] = value.to(self.device, dtype=self._dtype)
                else:
                    moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _get_core_model(self):
        model = self.model
        if hasattr(model, "get_base_model"):
            try:
                model = model.get_base_model()
            except Exception:
                pass
        if hasattr(model, "model") and hasattr(model.model, "get_image_features"):
            model = model.model
        return model


class FutureTokenProjector(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int, num_future_tokens: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.num_future_tokens = int(num_future_tokens)
        self.mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * num_future_tokens),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        tokens = self.mlp(stats).view(stats.size(0), self.num_future_tokens, self.hidden_dim)
        return self.norm(tokens)


class ConditionalFlowMatchingPredictor(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim + condition_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        time_feat = self.time_mlp(t)
        return self.net(torch.cat([xt, cond, time_feat], dim=-1))

    def flow_matching_loss(self, cond: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x0 = torch.randn_like(target)
        t = torch.rand(target.size(0), 1, device=target.device, dtype=target.dtype)
        xt = (1.0 - t) * x0 + t * target
        velocity_target = target - x0
        velocity_pred = self.forward(xt, t, cond)
        return F.mse_loss(velocity_pred, velocity_target)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, num_steps: int, num_samples: int) -> torch.Tensor:
        batch, dim = cond.shape
        device = cond.device
        dtype = cond.dtype
        xt = torch.randn(batch * num_samples, dim, device=device, dtype=dtype)
        cond_rep = cond[:, None, :].expand(batch, num_samples, cond.size(-1)).reshape(batch * num_samples, cond.size(-1))
        dt = 1.0 / max(int(num_steps), 1)
        for step in range(int(num_steps)):
            t_scalar = (step + 0.5) * dt
            t = torch.full((xt.size(0), 1), fill_value=t_scalar, device=device, dtype=dtype)
            xt = xt + dt * self.forward(xt, t, cond_rep)
        return xt.view(batch, num_samples, dim)


class ActionChunkHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, chunk_horizon: int, dropout: float = 0.1):
        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.action_dim * self.chunk_horizon),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).view(hidden.size(0), self.chunk_horizon, self.action_dim)


class ConditionalActionFlowMatchingHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, chunk_horizon: int, flow_hidden_dim: int):
        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.flat_action_dim = self.action_dim * self.chunk_horizon
        self.time_mlp = nn.Sequential(
            nn.Linear(1, flow_hidden_dim),
            nn.SiLU(),
            nn.Linear(flow_hidden_dim, flow_hidden_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, flow_hidden_dim),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Linear(self.flat_action_dim + 2 * flow_hidden_dim, flow_hidden_dim),
            nn.GELU(),
            nn.Linear(flow_hidden_dim, flow_hidden_dim),
            nn.GELU(),
            nn.Linear(flow_hidden_dim, self.flat_action_dim),
        )

    def forward(self, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        batch = xt.size(0)
        flat_xt = xt.view(batch, -1)
        time_feat = self.time_mlp(t)
        cond_feat = self.cond_mlp(cond)
        velocity = self.net(torch.cat([flat_xt, cond_feat, time_feat], dim=-1))
        return velocity.view(batch, self.chunk_horizon, self.action_dim)

    def flow_matching_loss(self, cond: torch.Tensor, target_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(target_actions)
        t = torch.rand(target_actions.size(0), 1, device=target_actions.device, dtype=target_actions.dtype)
        xt = (1.0 - t[:, None, :]) * noise + t[:, None, :] * target_actions
        velocity_target = target_actions - noise
        velocity_pred = self.forward(xt, t, cond)
        loss = F.mse_loss(velocity_pred, velocity_target)
        return loss, velocity_pred

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, num_steps: int, noise_scale: float = 1.0) -> torch.Tensor:
        batch = cond.size(0)
        device = cond.device
        dtype = cond.dtype
        xt = noise_scale * torch.randn(batch, self.chunk_horizon, self.action_dim, device=device, dtype=dtype)
        dt = 1.0 / max(int(num_steps), 1)
        for step in range(int(num_steps)):
            t_scalar = (step + 0.5) * dt
            t = torch.full((batch, 1), fill_value=t_scalar, device=device, dtype=dtype)
            xt = xt + dt * self.forward(xt, t, cond)
        return xt


class FuturePredVLA(nn.Module):
    @staticmethod
    def _resolve_hidden_dim(backbone_model) -> int:
        cfg = getattr(backbone_model, "config", None)
        candidates = []
        if cfg is not None:
            candidates.extend(
                [
                    getattr(cfg, "hidden_size", None),
                    getattr(getattr(cfg, "text_config", None), "hidden_size", None),
                    getattr(getattr(cfg, "llm_config", None), "hidden_size", None),
                    getattr(getattr(cfg, "language_config", None), "hidden_size", None),
                ]
            )
        candidates.extend(
            [
                getattr(getattr(backbone_model, "language_model", None), "config", None),
                getattr(getattr(backbone_model, "model", None), "config", None),
            ]
        )
        for candidate in list(candidates):
            if candidate is None:
                continue
            if isinstance(candidate, int):
                return int(candidate)
            value = getattr(candidate, "hidden_size", None)
            if value is not None:
                return int(value)
            for attr in ("text_config", "llm_config", "language_config"):
                nested = getattr(candidate, attr, None)
                value = getattr(nested, "hidden_size", None)
                if value is not None:
                    return int(value)

        for module in (
            getattr(backbone_model, "lm_head", None),
            getattr(getattr(backbone_model, "language_model", None), "lm_head", None),
            getattr(getattr(backbone_model, "model", None), "lm_head", None),
        ):
            if module is not None and hasattr(module, "in_features"):
                return int(module.in_features)

        raise RuntimeError("Could not resolve language hidden size from the InternVL model/config.")

    def __init__(
        self,
        cfg: ModelConfig,
        device: torch.device,
        action_dim: int,
        chunk_horizon: int,
        predictor_hidden_dim: int,
        num_future_tokens: int,
        use_future_prediction: bool = True,
        action_head_type: str = "regression",
        action_flow_hidden_dim: int = 2048,
    ):
        super().__init__()
        self.backbone = InternVLBackbone(cfg, device=device)
        self.hidden_dim = self._resolve_hidden_dim(self.backbone.model)
        self.use_future_prediction = bool(use_future_prediction)
        self.action_head_type = str(action_head_type)
        if self.use_future_prediction:
            self.predictor = ConditionalFlowMatchingPredictor(
                latent_dim=self.hidden_dim,
                hidden_dim=int(predictor_hidden_dim),
                condition_dim=self.hidden_dim,
            )
            self.projector = FutureTokenProjector(
                hidden_dim=self.hidden_dim,
                latent_dim=self.hidden_dim,
                num_future_tokens=num_future_tokens,
            )
        else:
            self.predictor = None
            self.projector = None
        if self.action_head_type == "flow":
            self.action_head = ConditionalActionFlowMatchingHead(
                hidden_dim=self.hidden_dim,
                action_dim=action_dim,
                chunk_horizon=chunk_horizon,
                flow_hidden_dim=int(action_flow_hidden_dim),
            )
        elif self.action_head_type == "regression":
            self.action_head = ActionChunkHead(
                hidden_dim=self.hidden_dim,
                action_dim=action_dim,
                chunk_horizon=chunk_horizon,
            )
        else:
            raise RuntimeError(f"Unsupported action_head_type={self.action_head_type}. Expected regression or flow.")

    def get_language_layers(self):
        queue = [
            self.backbone.model,
            getattr(self.backbone.model, "language_model", None),
            getattr(self.backbone.model, "model", None),
            getattr(getattr(self.backbone.model, "language_model", None), "model", None),
            getattr(getattr(self.backbone.model, "model", None), "model", None),
        ]
        seen = set()
        while queue:
            candidate = queue.pop(0)
            if candidate is None:
                continue
            ident = id(candidate)
            if ident in seen:
                continue
            seen.add(ident)
            layers = getattr(candidate, "layers", None)
            if layers is not None:
                return layers
            for attr in ("model", "language_model", "base_model", "transformer"):
                child = getattr(candidate, attr, None)
                if child is not None:
                    queue.append(child)
        return None

    def inject_future_tokens(self, future_tokens: torch.Tensor, inject_layer_idx: int):
        layers = self.get_language_layers()
        if layers is None:
            raise RuntimeError("Could not locate language layers for future-token injection.")
        inject_layer_idx = max(0, min(int(inject_layer_idx), len(layers) - 1))
        future_tokens = future_tokens.to(self.backbone.device)

        def _hook(_module, args, output):
            if isinstance(output, tuple):
                hidden = output[0]
                remainder = output[1:]
                tokens = future_tokens.to(hidden.device, dtype=hidden.dtype)
                if tokens.size(1) == 1:
                    hidden = hidden + tokens
                else:
                    prefix = hidden[:, : tokens.size(1), :] + tokens
                    hidden = torch.cat([prefix, hidden[:, tokens.size(1) :, :]], dim=1)
                return (hidden, *remainder)
            tokens = future_tokens.to(output.device, dtype=output.dtype)
            if tokens.size(1) == 1:
                return output + tokens
            prefix = output[:, : tokens.size(1), :] + tokens
            return torch.cat([prefix, output[:, tokens.size(1) :, :]], dim=1)

        return layers[inject_layer_idx].register_forward_hook(_hook)

    def _pool_hidden_states(self, hidden_states, attention_mask, layer_idx: int = -1):
        if not hidden_states:
            raise RuntimeError("Backbone did not return hidden states.")
        num_states = len(hidden_states)
        if layer_idx < 0:
            layer_idx = num_states + layer_idx
        layer_idx = max(0, min(int(layer_idx), num_states - 1))
        sequence_hidden = hidden_states[layer_idx]
        if attention_mask is None:
            pooled = sequence_hidden[:, -1, :]
        else:
            last_indices = attention_mask.sum(dim=1).long().clamp_min(1) - 1
            pooled = sequence_hidden[
                torch.arange(sequence_hidden.size(0), device=sequence_hidden.device),
                last_indices,
            ]
        return pooled, sequence_hidden

    def encode_inputs(self, inputs, layer_idx: int = -1):
        model_inputs = self.backbone._move_inputs_to_device(inputs)
        outputs = self.backbone.model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )
        pooled, seq = self._pool_hidden_states(outputs.hidden_states, model_inputs.get("attention_mask"), layer_idx=layer_idx)
        return {
            "pooled_state": pooled,
            "sequence_hidden": seq,
            "attention_mask": model_inputs.get("attention_mask"),
        }

    def summarize_distribution(self, future_samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.projector is None:
            raise RuntimeError("Future distribution summarization requested but future prediction is disabled.")
        mean = future_samples.mean(dim=1)
        std = future_samples.std(dim=1, unbiased=False)
        stats = torch.cat([mean, std], dim=-1)
        tokens = self.projector(stats)
        return stats, tokens

    def forward(
        self,
        current_inputs,
        future_inputs=None,
        actions=None,
        inject_layer_idx: int = 1,
        num_future_samples: int = 4,
        flow_sampling_steps: int = 16,
    ):
        current_encoded = self.encode_inputs(current_inputs)
        current_state = current_encoded["pooled_state"]

        if not self.use_future_prediction:
            if self.action_head_type == "flow":
                pred_actions = self.action_head.sample(current_state.float(), num_steps=flow_sampling_steps)
                action_loss, _ = self.action_head.flow_matching_loss(current_state.float(), actions.float()) if actions is not None else (None, None)
            else:
                pred_actions = self.action_head(current_state.float())
                action_loss = F.smooth_l1_loss(pred_actions, actions.float()) if actions is not None else None
            return {
                "loss": action_loss,
                "future_loss": None,
                "action_loss": action_loss,
                "pred_actions": pred_actions,
                "current_state": current_state,
                "future_samples": None,
                "future_tokens": None,
            }

        future_target = None
        if future_inputs is not None:
            with torch.no_grad():
                future_target = self.encode_inputs(future_inputs)["pooled_state"].detach()

        future_loss = None
        if future_target is not None:
            future_loss = self.predictor.flow_matching_loss(current_state.float(), future_target.float())

        future_samples = self.predictor.sample(
            cond=current_state.float(),
            num_steps=flow_sampling_steps,
            num_samples=num_future_samples,
        ).to(current_state.dtype)
        _, future_tokens = self.summarize_distribution(future_samples)

        hook = self.inject_future_tokens(future_tokens, inject_layer_idx=inject_layer_idx)
        try:
            conditioned = self.encode_inputs(current_inputs)
        finally:
            hook.remove()
        conditioned_state = conditioned["pooled_state"]

        if self.action_head_type == "flow":
            pred_actions = self.action_head.sample(conditioned_state.float(), num_steps=flow_sampling_steps)
            action_loss, _ = self.action_head.flow_matching_loss(conditioned_state.float(), actions.float()) if actions is not None else (None, None)
        else:
            pred_actions = self.action_head(conditioned_state.float())
            action_loss = F.smooth_l1_loss(pred_actions, actions.float()) if actions is not None else None

        total_loss = None
        if future_loss is not None and action_loss is not None:
            total_loss = future_loss + action_loss
        elif future_loss is not None:
            total_loss = future_loss
        elif action_loss is not None:
            total_loss = action_loss

        return {
            "loss": total_loss,
            "future_loss": future_loss,
            "action_loss": action_loss,
            "pred_actions": pred_actions,
            "current_state": current_state,
            "future_samples": future_samples,
            "future_tokens": future_tokens,
        }
