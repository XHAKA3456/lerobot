#!/usr/bin/env python

import math
from collections import deque
from itertools import chain
from pathlib import Path

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import save_file as save_safetensors_file
from torch import Tensor, nn
from transformers import AutoModel

from lerobot.policies.flowmatching_tc_v2_co.configuration_flowmatching_tc_v2_co import (
    FlowMatchingTCV2CoConfig,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

_DINOV2_DIM = 768
_DINOV2_IMAGE_SIZE = 224
_DINOV2_GRID_SIZE = 16
_POOLED_NUM_PATCHES = (_DINOV2_GRID_SIZE // 2) ** 2


def _sample_beta(alpha: float, beta: float, n: int, device) -> Tensor:
    dist = torch.distributions.Beta(
        torch.tensor(alpha, dtype=torch.float32),
        torch.tensor(beta, dtype=torch.float32),
    )
    return dist.sample((n,)).to(device=device, dtype=torch.float32)


def _sinusoidal_time_embedding(time: Tensor, dim: int, min_period: float, max_period: float) -> Tensor:
    assert dim % 2 == 0
    device = time.device
    frac = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** frac
    angles = (2.0 * math.pi / period)[None, :] * time[:, None]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class FlowMatchingTCV2CoPolicy(PreTrainedPolicy):
    config_class = FlowMatchingTCV2CoConfig
    name = "flowmatching_tc_v2_co"

    def __init__(self, config: FlowMatchingTCV2CoConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = FlowMatchingTCV2CoModel(config)
        self.reset()

    def get_optim_params(self) -> list:
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]

    def reset(self):
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            actions = actions[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        batch = self._collect_images(batch)
        return self.model.sample_actions(batch)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = self._collect_images(batch)
        losses = self.model.compute_loss(batch)
        main_losses = losses.pop("action_loss")

        if "action_is_pad" in batch:
            mask = ~batch["action_is_pad"].unsqueeze(-1)
            masked_action_loss = main_losses * mask
            loss = masked_action_loss.sum() / mask.sum()
        else:
            loss = main_losses.mean()

        for aux_value in losses.values():
            loss = loss + aux_value

        loss_dict = {"loss": float(loss.detach().cpu())}
        for key, value in losses.items():
            loss_dict[key] = float(value.detach().cpu())
        loss_dict["loss_per_dim"] = main_losses.mean(dim=[0, 1]).detach().cpu().tolist()
        return loss, loss_dict

    def _collect_images(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)
        if self.config.image_features:
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        if self.config.use_roi_branch and self.config.roi_feature_names:
            missing = [key for key in self.config.roi_feature_names if key not in batch]
            if not missing:
                batch["observation.roi_images"] = [batch[key] for key in self.config.roi_feature_names]
        return batch

    def _save_pretrained(self, save_directory: Path) -> None:
        self.config._save_pretrained(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self
        safe_state_dict = {
            key: value.detach().clone().contiguous().cpu()
            for key, value in model_to_save.state_dict().items()
        }
        save_safetensors_file(safe_state_dict, str(save_directory / SAFETENSORS_SINGLE_FILE))


class FlowMatchingTCV2CoModel(nn.Module):
    def __init__(self, config: FlowMatchingTCV2CoConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim_model
        action_dim = config.action_feature.shape[0]

        self.task_embed = nn.Embedding(config.num_tasks, self.dim)

        if config.image_features:
            self.global_backbone = AutoModel.from_pretrained(config.dinov2_model_name)
            self.global_backbone.eval()
            for param in self.global_backbone.parameters():
                param.requires_grad_(False)
            self.global_img_proj = nn.Linear(_DINOV2_DIM, self.dim)
            self.camera_embed = nn.Embedding(len(config.image_features), self.dim)
            self.global_patch_pos_embed = nn.Embedding(_POOLED_NUM_PATCHES, self.dim)

        if config.use_roi_branch and config.roi_feature_names:
            if config.use_roi_encoder_shared_backbone and hasattr(self, "global_backbone"):
                self.roi_backbone = self.global_backbone
            else:
                self.roi_backbone = AutoModel.from_pretrained(config.dinov2_model_name)
                self.roi_backbone.eval()
                for param in self.roi_backbone.parameters():
                    param.requires_grad_(False)
            self.roi_img_proj = nn.Linear(_DINOV2_DIM, self.dim)
            self.roi_camera_embed = nn.Embedding(len(config.roi_feature_names), self.dim)
            self.roi_patch_pos_embed = nn.Embedding(_POOLED_NUM_PATCHES, self.dim)

        self.state_proj = None
        if config.robot_state_feature:
            self.state_proj = nn.Linear(config.robot_state_feature.shape[0], self.dim)

        self.env_state_proj = None
        if config.env_state_feature:
            self.env_state_proj = nn.Linear(config.env_state_feature.shape[0], self.dim)

        self.ft_proj = nn.Sequential(
            nn.Linear(config.ft_feature_dim * (2 if config.use_force_derivative else 1), self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )

        self.temporal_state_mixer = nn.GRU(
            input_size=self.dim,
            hidden_size=self.dim,
            num_layers=config.temporal_history_layers,
            batch_first=True,
        )

        self.obs_encoder = _TransformerEncoder(config, n_layers=config.n_obs_encoder_layers)

        self.action_in_proj = nn.Linear(action_dim, self.dim)
        self.action_out_proj = nn.Linear(self.dim, action_dim)
        self.action_pos_embed = nn.Embedding(config.chunk_size, self.dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(self.dim * 2, self.dim))
        self.velocity_decoder = _TransformerDecoder(config, n_layers=config.n_velocity_layers)
        self.output_norm = nn.LayerNorm(self.dim, elementwise_affine=False)
        self.output_time_proj = nn.Sequential(nn.SiLU(), nn.Linear(self.dim, 2 * self.dim))

        self.phase_head = nn.Linear(self.dim, config.num_phase_classes)
        self.contact_head = nn.Linear(self.dim, config.num_contact_classes)
        self.relative_head = nn.Linear(self.dim, 6)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in chain(
            self.obs_encoder.parameters(),
            self.velocity_decoder.parameters(),
            self.temporal_state_mixer.parameters(),
        ):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def train(self, mode: bool = True):
        super().train(mode)
        for backbone_name in ("global_backbone", "roi_backbone"):
            if hasattr(self, backbone_name):
                getattr(self, backbone_name).eval()
        return self

    def _pool_patch_tokens(self, patch_tokens: Tensor, proj: nn.Linear) -> Tensor:
        batch_size = patch_tokens.shape[0]
        feat = proj(patch_tokens)
        feat = feat.view(batch_size, _DINOV2_GRID_SIZE, _DINOV2_GRID_SIZE, self.dim)
        feat = feat.permute(0, 3, 1, 2)
        feat = F.avg_pool2d(feat, kernel_size=2)
        feat = feat.permute(0, 2, 3, 1).reshape(batch_size, _POOLED_NUM_PATCHES, self.dim)
        return feat

    def _encode_image_sequence(
        self,
        images: list[Tensor],
        backbone: nn.Module,
        proj: nn.Linear,
        camera_embed: nn.Embedding,
        patch_pos_embed: nn.Embedding,
    ) -> tuple[list[Tensor], list[Tensor]]:
        tokens: list[Tensor] = []
        pos_embeds: list[Tensor] = []
        for cam_idx, img in enumerate(images):
            if img.ndim == 5:
                bsz, steps, _, _, _ = img.shape
                img = img.reshape(bsz * steps, *img.shape[2:])
            else:
                bsz = img.shape[0]
                steps = 1

            img_resized = F.interpolate(
                img,
                size=(_DINOV2_IMAGE_SIZE, _DINOV2_IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            with torch.no_grad():
                dino_out = backbone(pixel_values=img_resized)
            patch_tokens = dino_out.last_hidden_state[:, 1:]
            feat = self._pool_patch_tokens(patch_tokens, proj)
            feat = feat.view(bsz, steps, _POOLED_NUM_PATCHES, self.dim)
            feat = feat.mean(dim=1)
            feat = feat + camera_embed.weight[cam_idx]
            feat = feat + patch_pos_embed.weight.unsqueeze(0)
            feat = einops.rearrange(feat, "b n d -> n b d")
            pos = torch.zeros(_POOLED_NUM_PATCHES, 1, self.dim, device=feat.device, dtype=feat.dtype)
            tokens.extend(list(feat))
            pos_embeds.extend(list(pos))
        return tokens, pos_embeds

    def _encode_history_token(self, tensor: Tensor | None, projector: nn.Module | None) -> Tensor | None:
        if tensor is None or projector is None:
            return None
        if tensor.ndim == 2:
            tensor = tensor[:, None, :]
        projected = projector(tensor)
        mixed, hidden = self.temporal_state_mixer(projected)
        del mixed
        return hidden[-1]

    def _extract_force_features(self, state: Tensor | None) -> Tensor | None:
        if state is None:
            return None
        ft_dim = self.config.ft_feature_dim
        if state.shape[-1] < ft_dim:
            return None
        ft = state[..., -ft_dim:]
        if ft.ndim == 2:
            ft = ft[:, None, :]
        if self.config.use_force_derivative:
            diff = torch.zeros_like(ft)
            if ft.shape[1] > 1:
                diff[:, 1:] = ft[:, 1:] - ft[:, :-1]
            ft = torch.cat([ft, diff], dim=-1)
        return self._encode_history_token(ft, self.ft_proj)

    def _encode_obs(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        tokens: list[Tensor] = []
        pos_embeds: list[Tensor] = []
        aux: dict[str, Tensor] = {}

        task_index = batch["task_index"]
        if task_index.ndim == 2:
            task_index = task_index.squeeze(-1)
        task_tok = self.task_embed(task_index)
        tokens.append(task_tok)
        pos_embeds.append(torch.zeros(1, self.dim, device=task_tok.device, dtype=task_tok.dtype))

        state = batch.get(OBS_STATE)
        state_tok = self._encode_history_token(state, self.state_proj)
        if state_tok is not None:
            tokens.append(state_tok)
            pos_embeds.append(torch.zeros(1, self.dim, device=state_tok.device, dtype=state_tok.dtype))

        env_state = batch.get(OBS_ENV_STATE)
        env_tok = self._encode_history_token(env_state, self.env_state_proj)
        if env_tok is not None:
            tokens.append(env_tok)
            pos_embeds.append(torch.zeros(1, self.dim, device=env_tok.device, dtype=env_tok.dtype))

        ft_tok = self._extract_force_features(state)
        if ft_tok is not None:
            tokens.append(ft_tok)
            pos_embeds.append(torch.zeros(1, self.dim, device=ft_tok.device, dtype=ft_tok.dtype))

        if self.config.image_features:
            image_tokens, image_pos = self._encode_image_sequence(
                batch[OBS_IMAGES],
                self.global_backbone,
                self.global_img_proj,
                self.camera_embed,
                self.global_patch_pos_embed,
            )
            tokens.extend(image_tokens)
            pos_embeds.extend(image_pos)

        if self.config.use_roi_branch and "observation.roi_images" in batch:
            roi_tokens, roi_pos = self._encode_image_sequence(
                batch["observation.roi_images"],
                self.roi_backbone,
                self.roi_img_proj,
                self.roi_camera_embed,
                self.roi_patch_pos_embed,
            )
            tokens.extend(roi_tokens)
            pos_embeds.extend(roi_pos)

        tokens_tensor = torch.stack(tokens, dim=0)
        pos_tensor = torch.stack(pos_embeds, dim=0)
        obs_features = self.obs_encoder(tokens_tensor, pos_embed=pos_tensor)

        pooled_obs = obs_features.mean(dim=0)
        aux["phase_logits"] = self.phase_head(pooled_obs)
        aux["contact_logits"] = self.contact_head(pooled_obs)
        aux["relative_pred"] = self.relative_head(pooled_obs)
        return obs_features, pooled_obs, pos_tensor, aux

    def _decode_action_velocity(self, noisy_action: Tensor, obs_features: Tensor, time: Tensor) -> Tensor:
        bsz = noisy_action.shape[0]
        chunk = noisy_action.shape[1]

        action_tok = self.action_in_proj(noisy_action)
        action_tok = action_tok + self.action_pos_embed.weight[:chunk].unsqueeze(0)
        action_tok = einops.rearrange(action_tok, "b t d -> t b d")

        time_embed = _sinusoidal_time_embedding(
            time=time,
            dim=self.dim,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
        )
        time_embed = self.time_mlp(time_embed)
        cond = torch.cat([obs_features.mean(dim=0), time_embed], dim=-1)
        cond = self.cond_proj(cond)
        cond = cond.unsqueeze(0).expand(chunk, bsz, self.dim)

        dec_out = self.velocity_decoder(action_tok + cond, memory=obs_features)
        dec_out = einops.rearrange(dec_out, "t b d -> b t d")
        shift, scale = self.output_time_proj(time_embed).chunk(2, dim=-1)
        dec_out = self.output_norm(dec_out)
        dec_out = dec_out * (1 + scale[:, None, :]) + shift[:, None, :]
        return self.action_out_proj(dec_out)

    def compute_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        target_action = batch[ACTION]
        obs_features, pooled_obs, _, aux = self._encode_obs(batch)

        bsz = target_action.shape[0]
        device = target_action.device
        time = _sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            bsz,
            device,
        )
        time = time * self.config.time_sampling_scale + self.config.time_sampling_offset

        noise = torch.randn_like(target_action)
        time_expanded = time[:, None, None]
        x_t = (1 - time_expanded) * noise + time_expanded * target_action
        target_velocity = target_action - noise

        pred_velocity = self._decode_action_velocity(x_t, obs_features, time)
        action_loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
        losses: dict[str, Tensor] = {"action_loss": action_loss}

        if "phase_label" in batch:
            losses["phase_loss"] = (
                F.cross_entropy(aux["phase_logits"], batch["phase_label"].long())
                * self.config.aux_phase_loss_weight
            )
        if "contact_label" in batch:
            losses["contact_loss"] = (
                F.cross_entropy(aux["contact_logits"], batch["contact_label"].long())
                * self.config.aux_contact_loss_weight
            )
        if "relative_target" in batch:
            losses["relative_loss"] = (
                F.mse_loss(aux["relative_pred"], batch["relative_target"], reduction="mean")
                * self.config.aux_relative_loss_weight
            )

        return losses

    @torch.no_grad()
    def sample_actions(self, batch: dict[str, Tensor]) -> Tensor:
        obs_features, _, _, _ = self._encode_obs(batch)
        bsz = batch["task_index"].shape[0]
        action_dim = self.action_out_proj.out_features
        sample = torch.randn(
            bsz,
            self.config.chunk_size,
            action_dim,
            device=obs_features.device,
            dtype=obs_features.dtype,
        )
        time_grid = torch.linspace(0.0, 1.0, self.config.num_inference_steps + 1, device=sample.device)
        for start, end in zip(time_grid[:-1], time_grid[1:], strict=False):
            t = torch.full((bsz,), start, device=sample.device, dtype=sample.dtype)
            velocity = self._decode_action_velocity(sample, obs_features, t)
            sample = sample + (end - start) * velocity
        return sample


class _TransformerEncoder(nn.Module):
    def __init__(self, config: FlowMatchingTCV2CoConfig, n_layers: int):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=False,
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: Tensor, pos_embed: Tensor | None = None) -> Tensor:
        if pos_embed is not None:
            x = x + pos_embed
        return self.net(x)


class _TransformerDecoder(nn.Module):
    def __init__(self, config: FlowMatchingTCV2CoConfig, n_layers: int):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=False,
            norm_first=True,
        )
        self.net = nn.TransformerDecoder(layer, num_layers=n_layers)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        return self.net(tgt, memory)
