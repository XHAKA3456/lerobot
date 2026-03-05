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

"""Flow Matching Policy for robot learning.

Architecture:
  - Vision backbone: DINOv2 ViT-B/14 (frozen) → 256 patch tokens × 768-dim per camera
  - Obs encoder: Transformer encoder over [state_token, *patch_tokens_cam0, *patch_tokens_cam1, ...]
  - Velocity network: Transformer decoder
      queries = noisy_action_tokens + time_embedding
      keys/values = obs_features  (cross-attention)
  - Output: predicted velocity → action via backward Euler ODE

Flow matching math (pi0-style):
  Training:
      t ~ Beta(1.5, 1.0) * 0.999 + 0.001          in [0.001, 0.999]
      x_t = t * noise + (1 - t) * action            forward process
      u_t = noise - action                           velocity target
      loss = MSE(network(x_t, t, obs), u_t)

  Inference (backward Euler, t: 1 → 0, num_inference_steps=10):
      x_1 = noise ~ N(0, I)
      dt = -1 / num_inference_steps
      for step in range(num_inference_steps):
          t = 1.0 + step * dt          # 1.0, 0.9, 0.8, ..., 0.1
          v_t = network(x_t, t, obs)
          x_t = x_t + dt * v_t         # dt < 0 → moves toward action
      return x_0
"""

import math
from collections import deque
from itertools import chain

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoModel

from lerobot.policies.flowmatching.configuration_flowmatching import FlowMatchingConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

# DINOv2 ViT-B/14 patch token dimension
_DINOV2_DIM = 768
# Input image size for DINOv2
_DINOV2_IMAGE_SIZE = 224
# Number of patch tokens for 224×224 with patch_size=14: (224/14)^2 = 256
_DINOV2_NUM_PATCHES = 256


# ---------------------------------------------------------------------------
# Flow matching helpers (pi0-style)
# ---------------------------------------------------------------------------

def _sample_beta(alpha: float, beta: float, n: int, device) -> Tensor:
    """Draw n samples from Beta(alpha, beta)."""
    dist = torch.distributions.Beta(
        torch.tensor(alpha, dtype=torch.float32),
        torch.tensor(beta, dtype=torch.float32),
    )
    return dist.sample((n,)).to(device=device, dtype=torch.float32)


def _sinusoidal_time_embedding(
    time: Tensor,           # (B,)
    dim: int,
    min_period: float,
    max_period: float,
) -> Tensor:
    """Geometric-spacing sinusoidal embedding for scalar time values (pi0-style).

    Returns: (B, dim)
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    assert time.ndim == 1, f"time must be 1-D, got {time.shape}"

    device = time.device
    frac = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** frac   # (dim/2,)
    angles = (2.0 * math.pi / period)[None, :] * time[:, None]  # (B, dim/2)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)   # (B, dim)


# ---------------------------------------------------------------------------
# Policy (outer shell, LeRobot interface)
# ---------------------------------------------------------------------------

class FlowMatchingPolicy(PreTrainedPolicy):
    """Flow Matching Policy with frozen DINOv2 ViT-B/14 vision backbone.

    Learns a velocity field v(x_t, t, obs) that transports Gaussian noise to
    the action distribution. Inference runs 10 Euler steps from t=1 to t=0.
    """

    config_class = FlowMatchingConfig
    name = "flowmatching"

    def __init__(self, config: FlowMatchingConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = FlowMatchingModel(config)
        self.reset()

    def get_optim_params(self) -> list:
        """DINOv2 is frozen; return all trainable params with a single lr."""
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]

    def reset(self):
        """Clear action queue. Must be called on every episode reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Return one action, regenerating the chunk when the queue is empty."""
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)           # (B, chunk_size, A)
            actions = actions[:, : self.config.n_action_steps]   # (B, n, A)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Run ODE integration and return (B, chunk_size, action_dim)."""
        self.eval()
        batch = self._collect_images(batch)
        return self.model.sample_actions(batch)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass. Returns (scalar_loss, loss_dict)."""
        batch = self._collect_images(batch)
        losses = self.model.compute_loss(batch)   # (B, T, A)

        if "action_is_pad" in batch:
            mask = ~batch["action_is_pad"].unsqueeze(-1)  # (B, T, 1)
            losses = losses * mask

        loss = losses.mean()
        loss_dict = {
            "loss": loss.item(),
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().tolist(),
        }
        return loss, loss_dict

    def _collect_images(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Collect image features into batch[OBS_IMAGES] list."""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return batch


# ---------------------------------------------------------------------------
# Model (inner neural network)
# ---------------------------------------------------------------------------

class FlowMatchingModel(nn.Module):
    """Core neural network for flow matching.

    Obs encoding (runs once per chunk):
        DINOv2(each_image) → patch tokens (B, 256, 768) → Linear proj (B, 256, D)
        state → (B, D)
        TransformerEncoder([state_token, *img_tokens_cam0, ..., *img_tokens_camN]) → obs_features

    Velocity prediction (runs num_inference_steps times at inference):
        noisy_action (B, T, A) → project → (B, T, D)
        time (B,) → sinusoidal emb (B, D) → broadcast + fuse with action_emb
        TransformerDecoder(queries=action_time_tokens, kv=obs_features) → (T, B, D)
        project → v_pred (B, T, A)
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config
        D = config.dim_model
        A = config.action_feature.shape[0]

        # ---- Vision backbone: DINOv2 ViT-B/14 (frozen) ----
        if config.image_features:
            self.dinov2 = AutoModel.from_pretrained(config.dinov2_model_name)
            self.dinov2.eval()
            for param in self.dinov2.parameters():
                param.requires_grad_(False)
            # Project DINOv2 patch tokens (768-dim) → dim_model
            self.img_feat_proj = nn.Linear(_DINOV2_DIM, D)

        # ---- 1-D tokens (state, env_state) ----
        n_1d = 0
        if config.robot_state_feature:
            self.state_proj = nn.Linear(config.robot_state_feature.shape[0], D)
            n_1d += 1
        if config.env_state_feature:
            self.env_state_proj = nn.Linear(config.env_state_feature.shape[0], D)
            n_1d += 1
        if n_1d > 0:
            self.obs_1d_pos_embed = nn.Embedding(n_1d, D)

        # ---- Observation encoder (Transformer encoder) ----
        self.obs_encoder = _TransformerEncoder(config, n_layers=config.n_obs_encoder_layers)

        # ---- Velocity network (Transformer decoder) ----
        self.action_in_proj = nn.Linear(A, D)
        self.action_out_proj = nn.Linear(D, A)

        self.action_time_mlp = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.SiLU(),
            nn.Linear(D, D),
        )

        self.action_pos_embed = nn.Embedding(config.chunk_size, D)

        self.velocity_decoder = _TransformerDecoder(config, n_layers=config.n_velocity_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in chain(self.obs_encoder.parameters(), self.velocity_decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def train(self, mode: bool = True):
        """Keep DINOv2 in eval mode regardless of training mode (it's frozen)."""
        super().train(mode)
        if hasattr(self, "dinov2"):
            self.dinov2.eval()
        return self

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def _encode_obs(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Encode observations into context tokens.

        Returns:
            obs_features:   (S, B, D)
            obs_pos_embeds: (S, 1, D)   — broadcasted over batch
        """
        tokens = []       # (B, D) items accumulated
        pos_embeds = []   # (1, D) items accumulated
        one_d_idx = 0

        if self.config.robot_state_feature:
            state = batch[OBS_STATE]
            if state.ndim == 3:
                state = state[:, 0]                                        # (B, state_dim)
            tok = self.state_proj(state)                                   # (B, D)
            pos = self.obs_1d_pos_embed.weight[one_d_idx].unsqueeze(0)    # (1, D)
            tokens.append(tok)
            pos_embeds.append(pos)
            one_d_idx += 1

        if self.config.env_state_feature:
            env_state = batch[OBS_ENV_STATE]
            if env_state.ndim == 3:
                env_state = env_state[:, 0]                                # (B, env_dim)
            tok = self.env_state_proj(env_state)                           # (B, D)
            pos = self.obs_1d_pos_embed.weight[one_d_idx].unsqueeze(0)    # (1, D)
            tokens.append(tok)
            pos_embeds.append(pos)
            one_d_idx += 1

        if self.config.image_features:
            D = self.config.dim_model
            for img in batch[OBS_IMAGES]:
                # img: (B, n_obs_steps, 3, H, W) or (B, 3, H, W) — squeeze time dim
                if img.ndim == 5:
                    img = img[:, 0]  # (B, 3, H, W)
                # img: (B, 3, H, W) — resize to 224×224 for DINOv2
                img_resized = F.interpolate(
                    img, size=(_DINOV2_IMAGE_SIZE, _DINOV2_IMAGE_SIZE),
                    mode="bilinear", align_corners=False,
                )
                # DINOv2 forward (frozen — no grad needed through backbone)
                with torch.no_grad():
                    dino_out = self.dinov2(pixel_values=img_resized)
                # Skip CLS token, use patch tokens: (B, 256, 768)
                patch_tokens = dino_out.last_hidden_state[:, 1:]

                feat = self.img_feat_proj(patch_tokens)                # (B, 256, D)
                feat = einops.rearrange(feat, "b n d -> n b d")        # (256, B, D)

                # DINOv2 encodes position internally; use zero pos embeds here
                pos = torch.zeros(_DINOV2_NUM_PATCHES, 1, D, device=feat.device, dtype=feat.dtype)

                tokens.extend(list(feat))
                pos_embeds.extend(list(pos))

        tokens = torch.stack(tokens, dim=0)          # (S, B, D)
        pos_embeds = torch.stack(pos_embeds, dim=0)  # (S, 1, D)

        obs_features = self.obs_encoder(tokens, pos_embed=pos_embeds)  # (S, B, D)
        return obs_features, pos_embeds

    # ------------------------------------------------------------------
    # Velocity prediction
    # ------------------------------------------------------------------

    def _predict_velocity(
        self,
        obs_features: Tensor,   # (S, B, D)
        obs_pos_embeds: Tensor, # (S, 1, D)
        x_t: Tensor,            # (B, T, A)  noisy action
        time: Tensor,           # (B,)
    ) -> Tensor:                # (B, T, A)  predicted velocity
        B, T, _ = x_t.shape

        a_emb = self.action_in_proj(x_t)                                  # (B, T, D)

        t_emb = _sinusoidal_time_embedding(
            time, self.config.dim_model,
            self.config.min_period, self.config.max_period,
        )                                                                   # (B, D)
        t_emb = t_emb[:, None, :].expand(B, T, -1)                        # (B, T, D)

        fused = self.action_time_mlp(torch.cat([a_emb, t_emb], dim=-1))   # (B, T, D)

        action_pos = self.action_pos_embed.weight[:T].unsqueeze(1)         # (T, 1, D)

        fused = einops.rearrange(fused, "b t d -> t b d")                  # (T, B, D)

        out = self.velocity_decoder(
            x=fused,
            encoder_out=obs_features,
            decoder_pos_embed=action_pos,
            encoder_pos_embed=obs_pos_embeds,
        )                                                                   # (T, B, D)

        out = einops.rearrange(out, "t b d -> b t d")                      # (B, T, D)
        return self.action_out_proj(out)                                    # (B, T, A)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute per-element MSE flow matching loss.

        Returns: (B, chunk_size, action_dim)
        """
        actions = batch[ACTION]                    # (B, T, A)
        B = actions.shape[0]
        device = actions.device

        noise = torch.randn_like(actions)

        time = _sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            B, device,
        ) * self.config.time_sampling_scale + self.config.time_sampling_offset  # (B,)

        t = time[:, None, None]
        x_t = t * noise + (1.0 - t) * actions   # (B, T, A)
        u_t = noise - actions                    # velocity target (B, T, A)

        obs_features, obs_pos_embeds = self._encode_obs(batch)

        v_pred = self._predict_velocity(obs_features, obs_pos_embeds, x_t, time)

        return F.mse_loss(v_pred, u_t, reduction="none")  # (B, T, A)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """Generate actions via backward Euler ODE (t: 1 → 0).

        Returns: (B, chunk_size, action_dim)
        """
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

        obs_features, obs_pos_embeds = self._encode_obs(batch)

        dt = -1.0 / self.config.num_inference_steps
        for step in range(self.config.num_inference_steps):
            t_val = 1.0 + step * dt
            time = torch.full((B,), t_val, dtype=torch.float32, device=device)
            v_t = self._predict_velocity(obs_features, obs_pos_embeds, x_t, time)
            x_t = x_t + dt * v_t

        return x_t    # (B, chunk_size, A)


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class _TransformerEncoder(nn.Module):
    def __init__(self, config: FlowMatchingConfig, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_EncoderLayer(config) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(self, x: Tensor, pos_embed: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed)
        return self.norm(x)


class _EncoderLayer(nn.Module):
    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        D = config.dim_model
        self.self_attn = nn.MultiheadAttention(D, config.n_heads, dropout=config.dropout)
        self.ff = nn.Sequential(
            nn.Linear(D, config.dim_feedforward),
            nn.ReLU(),
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
    def __init__(self, config: FlowMatchingConfig, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_DecoderLayer(config) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, encoder_out,
                      decoder_pos_embed=decoder_pos_embed,
                      encoder_pos_embed=encoder_pos_embed)
        return self.norm(x)


class _DecoderLayer(nn.Module):
    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        D = config.dim_model
        self.self_attn = nn.MultiheadAttention(D, config.n_heads, dropout=config.dropout)
        self.cross_attn = nn.MultiheadAttention(D, config.n_heads, dropout=config.dropout)
        self.ff = nn.Sequential(
            nn.Linear(D, config.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, D),
        )
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        self.norm3 = nn.LayerNorm(D)
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
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        q = k = self._add_pos(x, decoder_pos_embed)
        x = self.norm1(x + self.drop1(self.self_attn(q, k, value=x)[0]))

        x = self.norm2(x + self.drop2(self.cross_attn(
            query=self._add_pos(x, decoder_pos_embed),
            key=self._add_pos(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]))

        return self.norm3(x + self.drop3(self.ff(x)))
