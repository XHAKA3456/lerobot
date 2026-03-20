#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flow Matching Task-Conditioned Policy.

Extends FlowMatching with a learnable task embedding prepended to the observation
encoder input, enabling multi-task conditioning (e.g., SFP vs SC).
"""

import math
from collections import deque
from itertools import chain

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoModel

from lerobot.policies.flowmatching_tc.configuration_flowmatching_tc import FlowMatchingTCConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

# DINOv2 ViT-B/14 patch token dimension
_DINOV2_DIM = 768
_DINOV2_IMAGE_SIZE = 224
_DINOV2_NUM_PATCHES = 256
_DINOV2_GRID_SIZE = 16
_POOL_KERNEL = 2
_POOLED_NUM_PATCHES = (_DINOV2_GRID_SIZE // _POOL_KERNEL) ** 2  # 64


# ---------------------------------------------------------------------------
# Flow matching helpers (pi0-style)
# ---------------------------------------------------------------------------

def _sample_beta(alpha: float, beta: float, n: int, device) -> Tensor:
    dist = torch.distributions.Beta(
        torch.tensor(alpha, dtype=torch.float32),
        torch.tensor(beta, dtype=torch.float32),
    )
    return dist.sample((n,)).to(device=device, dtype=torch.float32)


def _sinusoidal_time_embedding(
    time: Tensor, dim: int, min_period: float, max_period: float,
) -> Tensor:
    assert dim % 2 == 0
    assert time.ndim == 1
    device = time.device
    frac = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** frac
    angles = (2.0 * math.pi / period)[None, :] * time[:, None]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


# ---------------------------------------------------------------------------
# Policy (outer shell, LeRobot interface)
# ---------------------------------------------------------------------------

class FlowMatchingTCPolicy(PreTrainedPolicy):
    """Flow Matching Task-Conditioned Policy.

    Adds a task embedding token to the observation encoder for multi-task support.
    """

    config_class = FlowMatchingTCConfig
    name = "flowmatching_tc"

    def __init__(self, config: FlowMatchingTCConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = FlowMatchingTCModel(config)
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

        if "action_is_pad" in batch:
            mask = ~batch["action_is_pad"].unsqueeze(-1)
            losses = losses * mask
            loss = losses.sum() / mask.sum()
        else:
            loss = losses.mean()
        loss_dict = {
            "loss": loss.item(),
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().tolist(),
        }
        return loss, loss_dict

    def _collect_images(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return batch


# ---------------------------------------------------------------------------
# Model (inner neural network)
# ---------------------------------------------------------------------------

class FlowMatchingTCModel(nn.Module):
    """Core neural network for flow matching with task conditioning.

    Adds a learnable task embedding token that is prepended to the observation
    encoder input sequence, allowing the model to distinguish between tasks.
    """

    def __init__(self, config: FlowMatchingTCConfig):
        super().__init__()
        self.config = config
        D = config.dim_model
        A = config.action_feature.shape[0]

        # ---- Task embedding ----
        self.task_embed = nn.Embedding(config.num_tasks, D)

        # ---- Vision backbone: DINOv2 ViT-B/14 (frozen) ----
        if config.image_features:
            self.dinov2 = AutoModel.from_pretrained(config.dinov2_model_name)
            self.dinov2.eval()
            for param in self.dinov2.parameters():
                param.requires_grad_(False)
            self.img_feat_proj = nn.Linear(_DINOV2_DIM, D)
            num_cameras = len(config.image_features)
            self.camera_embed = nn.Embedding(num_cameras, D)
            self.patch_pos_embed = nn.Embedding(_POOLED_NUM_PATCHES, D)

        # ---- 1-D tokens (task, state, env_state) ----
        # task token is always first, then state, then env_state
        n_1d = 1  # task token always present
        if config.robot_state_feature:
            self.state_proj = nn.Linear(config.robot_state_feature.shape[0], D)
            n_1d += 1
        if config.env_state_feature:
            self.env_state_proj = nn.Linear(config.env_state_feature.shape[0], D)
            n_1d += 1
        self.obs_1d_pos_embed = nn.Embedding(n_1d, D)

        # ---- Observation encoder (Transformer encoder) ----
        self.obs_encoder = _TransformerEncoder(config, n_layers=config.n_obs_encoder_layers)

        # ---- Velocity network (Transformer decoder with AdaLN) ----
        self.action_in_proj = nn.Linear(A, D)
        self.action_out_proj = nn.Linear(D, A)

        self.time_mlp = nn.Sequential(
            nn.Linear(D, D),
            nn.SiLU(),
            nn.Linear(D, D),
        )

        self.action_time_mlp = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.SiLU(),
            nn.Linear(D, D),
        )

        self.action_pos_embed = nn.Embedding(config.chunk_size, D)

        self.velocity_decoder = _TransformerDecoder(config, n_layers=config.n_velocity_layers)

        self.task_cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(D, D))

        self.output_norm = nn.LayerNorm(D, elementwise_affine=False)
        self.output_time_proj = nn.Sequential(nn.SiLU(), nn.Linear(D, 2 * D))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in chain(self.obs_encoder.parameters(), self.velocity_decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "dinov2"):
            self.dinov2.eval()
        return self

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def _encode_obs(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Encode observations into context tokens.

        Task token is prepended as the first 1-D token.

        Returns:
            obs_features:   (S, B, D)
            obs_pos_embeds: (S, 1, D)
            task_embed:     (B, D)
        """
        tokens = []
        pos_embeds = []
        one_d_idx = 0

        # Task token (always first)
        task_index = batch["task_index"]  # (B,) or (B, 1)
        if task_index.ndim == 2:
            task_index = task_index.squeeze(-1)  # (B,)
        task_tok = self.task_embed(task_index)  # (B, D)
        pos = self.obs_1d_pos_embed.weight[one_d_idx].unsqueeze(0)  # (1, D)
        tokens.append(task_tok)
        pos_embeds.append(pos)
        one_d_idx += 1

        if self.config.robot_state_feature:
            state = batch[OBS_STATE]
            if state.ndim == 3:
                state = state[:, 0]
            tok = self.state_proj(state)
            pos = self.obs_1d_pos_embed.weight[one_d_idx].unsqueeze(0)
            tokens.append(tok)
            pos_embeds.append(pos)
            one_d_idx += 1

        if self.config.env_state_feature:
            env_state = batch[OBS_ENV_STATE]
            if env_state.ndim == 3:
                env_state = env_state[:, 0]
            tok = self.env_state_proj(env_state)
            pos = self.obs_1d_pos_embed.weight[one_d_idx].unsqueeze(0)
            tokens.append(tok)
            pos_embeds.append(pos)
            one_d_idx += 1

        if self.config.image_features:
            D = self.config.dim_model
            for cam_idx, img in enumerate(batch[OBS_IMAGES]):
                if img.ndim == 5:
                    img = img[:, 0]
                B_img = img.shape[0]
                img_resized = F.interpolate(
                    img, size=(_DINOV2_IMAGE_SIZE, _DINOV2_IMAGE_SIZE),
                    mode="bilinear", align_corners=False,
                )
                with torch.no_grad():
                    dino_out = self.dinov2(pixel_values=img_resized)
                patch_tokens = dino_out.last_hidden_state[:, 1:]

                feat = self.img_feat_proj(patch_tokens)
                feat = feat.view(B_img, _DINOV2_GRID_SIZE, _DINOV2_GRID_SIZE, D)
                feat = feat.permute(0, 3, 1, 2)
                feat = F.avg_pool2d(feat, kernel_size=_POOL_KERNEL)
                feat = feat.permute(0, 2, 3, 1).reshape(B_img, _POOLED_NUM_PATCHES, D)

                feat = feat + self.camera_embed.weight[cam_idx]
                feat = feat + self.patch_pos_embed.weight.unsqueeze(0)

                feat = einops.rearrange(feat, "b n d -> n b d")

                pos = torch.zeros(_POOLED_NUM_PATCHES, 1, D, device=feat.device, dtype=feat.dtype)

                tokens.extend(list(feat))
                pos_embeds.extend(list(pos))

        tokens = torch.stack(tokens, dim=0)
        pos_embeds = torch.stack(pos_embeds, dim=0)

        obs_features = self.obs_encoder(tokens, pos_embed=pos_embeds)
        return obs_features, pos_embeds, task_tok

    # ------------------------------------------------------------------
    # Velocity prediction
    # ------------------------------------------------------------------

    def _predict_velocity(
        self,
        obs_features: Tensor,
        obs_pos_embeds: Tensor,
        x_t: Tensor,
        time: Tensor,
        task_embed: Tensor,
    ) -> Tensor:
        B, T, _ = x_t.shape

        t_sinusoidal = _sinusoidal_time_embedding(
            time, self.config.dim_model,
            self.config.min_period, self.config.max_period,
        )
        temb = self.time_mlp(t_sinusoidal)
        temb = temb + self.task_cond_proj(task_embed)

        a_emb = self.action_in_proj(x_t)
        t_broadcast = temb[:, None, :].expand(B, T, -1)
        fused = self.action_time_mlp(torch.cat([a_emb, t_broadcast], dim=-1))

        action_pos = self.action_pos_embed.weight[:T].unsqueeze(1)

        fused = einops.rearrange(fused, "b t d -> t b d")

        out = self.velocity_decoder(
            x=fused,
            encoder_out=obs_features,
            temb=temb,
            decoder_pos_embed=action_pos,
            encoder_pos_embed=obs_pos_embeds,
        )

        out = einops.rearrange(out, "t b d -> b t d")

        scale, shift = self.output_time_proj(temb).chunk(2, dim=-1)
        out = self.output_norm(out) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        return self.action_out_proj(out)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        actions = batch[ACTION]
        B = actions.shape[0]
        device = actions.device

        noise = torch.randn_like(actions)

        time = _sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            B, device,
        ) * self.config.time_sampling_scale + self.config.time_sampling_offset

        t = time[:, None, None]
        x_t = t * noise + (1.0 - t) * actions
        u_t = noise - actions

        obs_features, obs_pos_embeds, task_embed = self._encode_obs(batch)

        v_pred = self._predict_velocity(obs_features, obs_pos_embeds, x_t, time, task_embed)

        return F.mse_loss(v_pred, u_t, reduction="none")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_actions(self, batch: dict[str, Tensor]) -> Tensor:
        if self.config.robot_state_feature:
            B = batch[OBS_STATE].shape[0]
            device = batch[OBS_STATE].device
        elif self.config.env_state_feature:
            B = batch[OBS_ENV_STATE].shape[0]
            device = batch[OBS_ENV_STATE].device
        else:
            B = batch[OBS_IMAGES][0].shape[0]
            device = batch[OBS_IMAGES][0].device

        A = self.config.action_feature.shape[0]
        T = self.config.chunk_size

        x_t = torch.randn(B, T, A, dtype=torch.float32, device=device)

        obs_features, obs_pos_embeds, task_embed = self._encode_obs(batch)

        dt = -1.0 / self.config.num_inference_steps
        for step in range(self.config.num_inference_steps):
            t_val = 1.0 + step * dt
            time = torch.full((B,), t_val, dtype=torch.float32, device=device)
            v_t = self._predict_velocity(obs_features, obs_pos_embeds, x_t, time, task_embed)
            x_t = x_t + dt * v_t

        return x_t


# ---------------------------------------------------------------------------
# Adaptive Layer Normalization (DiT-style, time-conditioned)
# ---------------------------------------------------------------------------

class _AdaLayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        scale, shift = self.proj(temb).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class _TransformerEncoder(nn.Module):
    def __init__(self, config: FlowMatchingTCConfig, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_EncoderLayer(config) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(self, x: Tensor, pos_embed: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed)
        return self.norm(x)


class _EncoderLayer(nn.Module):
    def __init__(self, config: FlowMatchingTCConfig):
        super().__init__()
        D = config.dim_model
        self.self_attn = nn.MultiheadAttention(D, config.n_heads, dropout=config.dropout)
        self.ff = nn.Sequential(
            nn.Linear(D, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, D),
        )
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        self.drop1 = nn.Dropout(config.dropout)
        self.drop2 = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, pos_embed: Tensor | None = None) -> Tensor:
        q = k = x if pos_embed is None else x + pos_embed
        x = self.norm1(x + self.drop1(self.self_attn(q, k, value=x)[0]))
        return self.norm2(x + self.drop2(self.ff(x)))


class _TransformerDecoder(nn.Module):
    def __init__(self, config: FlowMatchingTCConfig, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_DecoderLayer(config) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        temb: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, encoder_out, temb=temb,
                      decoder_pos_embed=decoder_pos_embed,
                      encoder_pos_embed=encoder_pos_embed)
        return self.norm(x)


class _DecoderLayer(nn.Module):
    def __init__(self, config: FlowMatchingTCConfig):
        super().__init__()
        D = config.dim_model
        self.self_attn = nn.MultiheadAttention(D, config.n_heads, dropout=config.dropout)
        self.cross_attn = nn.MultiheadAttention(D, config.n_heads, dropout=config.dropout)
        self.ff = nn.Sequential(
            nn.Linear(D, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, D),
        )
        self.norm1 = _AdaLayerNorm(D)
        self.norm2 = _AdaLayerNorm(D)
        self.norm3 = _AdaLayerNorm(D)
        self.drop1 = nn.Dropout(config.dropout)
        self.drop2 = nn.Dropout(config.dropout)
        self.drop3 = nn.Dropout(config.dropout)

    @staticmethod
    def _add_pos(t: Tensor, pos: Tensor | None) -> Tensor:
        return t if pos is None else t + pos

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        temb: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        q = k = self._add_pos(x, decoder_pos_embed)
        x = self.norm1(x + self.drop1(self.self_attn(q, k, value=x)[0]), temb)

        x = self.norm2(x + self.drop2(self.cross_attn(
            query=self._add_pos(x, decoder_pos_embed),
            key=self._add_pos(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]), temb)

        return self.norm3(x + self.drop3(self.ff(x)), temb)
