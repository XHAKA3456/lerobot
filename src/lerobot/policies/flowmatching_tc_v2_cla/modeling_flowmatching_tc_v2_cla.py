#!/usr/bin/env python

"""FlowMatchingTC V2 — end-to-end policy for AIC cable insertion.

Key improvements over V1:
- DINOv2 ViT-L/14 + LoRA on Q/V projections
- Multi-scale features (layers 6, 12, 18, 24)
- No spatial pooling: 256 tokens/cam
- State & F/T temporal history (4 steps) + force derivative
- Delta action 6D (dx, dy, dz, axis-angle)
- 12-layer DiT decoder with AdaLN-Zero (gated residuals)
- Classifier-Free Guidance (w=1.1)
- 4-step midpoint ODE solver
- Temporal ensemble (exponential weighting)
- Auxiliary phase/contact heads with phase-weighted loss

See /home/sst/aic/flowmatching_v2_cla.md for full design.
"""

import math
from collections import deque
from itertools import chain

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoModel

from lerobot.policies.flowmatching_tc_v2_cla.configuration_flowmatching_tc_v2_cla import FlowMatchingTCV2ClaConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for a frozen nn.Linear."""

    def __init__(self, original: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.original = original
        original.weight.requires_grad_(False)
        if original.bias is not None:
            original.bias.requires_grad_(False)
        in_dim, out_dim = original.in_features, original.out_features
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank) * (1.0 / rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scale = alpha / rank

    def forward(self, x: Tensor) -> Tensor:
        base = self.original(x)
        lora = (x @ self.lora_A @ self.lora_B) * self.scale
        return base + lora


def _apply_lora_to_dinov2(model: nn.Module, rank: int, alpha: float) -> list[nn.Parameter]:
    """Inject LoRA into Q and V projections of all attention layers. Returns trainable params."""
    trainable = []
    for name, module in model.named_modules():
        # DINOv2 attention Q/V are: encoder.layer.N.attention.attention.query / .value
        if hasattr(module, "query") and isinstance(module.query, nn.Linear):
            lora_q = LoRALinear(module.query, rank, alpha)
            module.query = lora_q
            trainable.extend([lora_q.lora_A, lora_q.lora_B])
        if hasattr(module, "value") and isinstance(module.value, nn.Linear):
            lora_v = LoRALinear(module.value, rank, alpha)
            module.value = lora_v
            trainable.extend([lora_v.lora_A, lora_v.lora_B])
    return trainable


# ---------------------------------------------------------------------------
# Flow matching helpers
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
    device = time.device
    frac = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** frac
    angles = (2.0 * math.pi / period)[None, :] * time[:, None]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


# ---------------------------------------------------------------------------
# AdaLN-Zero (DiT-style with gate)
# ---------------------------------------------------------------------------

class _AdaLNZero(nn.Module):
    """Adaptive LayerNorm-Zero: produces scale, shift, gate from conditioning."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
        # Initialize gate to zero so residual starts as identity
        nn.init.zeros_(self.proj[1].weight[-dim:])
        nn.init.zeros_(self.proj[1].bias[-dim:])

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (normalized_x_with_scale_shift, gate)."""
        scale, shift, gate = self.proj(cond).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        return h, gate


# ---------------------------------------------------------------------------
# DiT Decoder Layer (AdaLN-Zero)
# ---------------------------------------------------------------------------

class _DiTDecoderLayer(nn.Module):
    """Single DiT decoder layer with self-attn, cross-attn, FFN, all gated via AdaLN-Zero."""

    def __init__(self, dim: int, n_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        # Self-attention block
        self.adaln_sa = _AdaLNZero(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=False)

        # Cross-attention block
        self.adaln_ca = _AdaLNZero(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=False)

        # FFN block
        self.adaln_ff = _AdaLNZero(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim),
        )

    def forward(
        self,
        x: Tensor,          # (T, B, D) action tokens
        memory: Tensor,      # (S, B, D) encoder output
        temb: Tensor,        # (B, D) conditioning
    ) -> Tensor:
        # Self-attention with AdaLN-Zero gate
        h, gate = self.adaln_sa(x, temb)
        sa_out = self.self_attn(h, h, h)[0]
        x = x + gate * sa_out

        # Cross-attention with AdaLN-Zero gate
        h, gate = self.adaln_ca(x, temb)
        ca_out = self.cross_attn(h, memory, memory)[0]
        x = x + gate * ca_out

        # FFN with AdaLN-Zero gate
        h, gate = self.adaln_ff(x, temb)
        ff_out = self.ff(h)
        x = x + gate * ff_out

        return x


# ---------------------------------------------------------------------------
# Transformer Encoder (standard pre-norm)
# ---------------------------------------------------------------------------

class _EncoderLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=False)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, pos_embed: Tensor | None = None) -> Tensor:
        q = k = x if pos_embed is None else x + pos_embed
        x = x + self.drop1(self.self_attn(q, k, value=x)[0])
        x = self.norm1(x)
        x = x + self.drop2(self.ff(x))
        x = self.norm2(x)
        return x


class _TransformerEncoder(nn.Module):
    def __init__(self, dim: int, n_heads: int, dim_ff: int, dropout: float, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            _EncoderLayer(dim, n_heads, dim_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, pos_embed: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed)
        return self.norm(x)


# ---------------------------------------------------------------------------
# DiT Decoder (stack of AdaLN-Zero layers)
# ---------------------------------------------------------------------------

class _DiTDecoder(nn.Module):
    def __init__(self, dim: int, n_heads: int, dim_ff: int, dropout: float, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            _DiTDecoderLayer(dim, n_heads, dim_ff, dropout) for _ in range(n_layers)
        ])

    def forward(self, x: Tensor, memory: Tensor, temb: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, temb)
        return x


# ---------------------------------------------------------------------------
# Policy (outer shell, LeRobot interface)
# ---------------------------------------------------------------------------

class FlowMatchingTCV2ClaPolicy(PreTrainedPolicy):
    """FlowMatchingTC V2 Policy with temporal ensemble support."""

    config_class = FlowMatchingTCV2ClaConfig
    name = "flowmatching_tc_v2_cla"

    def __init__(self, config: FlowMatchingTCV2ClaConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = FlowMatchingTCV2ClaModel(config)
        self.reset()

    def get_optim_params(self) -> list:
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]

    def reset(self):
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self._prev_chunk: Tensor | None = None

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            new_chunk = self.predict_action_chunk(batch)  # (B, T, A)

            # Temporal ensemble: blend with previous chunk
            if self._prev_chunk is not None:
                lam = self.config.temporal_ensemble_lambda
                n_exec = self.config.n_action_steps
                overlap = min(new_chunk.shape[1], self._prev_chunk.shape[1] - n_exec)
                if overlap > 0:
                    prev_remaining = self._prev_chunk[:, n_exec: n_exec + overlap]
                    new_chunk[:, :overlap] = (
                        lam * new_chunk[:, :overlap]
                        + (1 - lam) * prev_remaining
                    )

            self._prev_chunk = new_chunk
            actions = new_chunk[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        batch = self._collect_images(batch)
        return self.model.sample_actions(batch)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = self._collect_images(batch)
        loss_result = self.model.compute_loss(batch)
        action_losses = loss_result["action_loss"]
        aux_losses = {k: v for k, v in loss_result.items() if k != "action_loss"}

        if "action_is_pad" in batch:
            mask = ~batch["action_is_pad"].unsqueeze(-1)
            action_losses = action_losses * mask
            loss = action_losses.sum() / mask.sum()
        else:
            loss = action_losses.mean()

        for v in aux_losses.values():
            loss = loss + v

        loss_dict = {"loss": float(loss.detach().cpu())}
        for k, v in aux_losses.items():
            loss_dict[k] = float(v.detach().cpu())
        loss_dict["loss_per_dim"] = action_losses.mean(dim=[0, 1]).detach().cpu().tolist()
        return loss, loss_dict

    def _collect_images(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return batch


# ---------------------------------------------------------------------------
# Model (core neural network)
# ---------------------------------------------------------------------------

class FlowMatchingTCV2ClaModel(nn.Module):
    def __init__(self, config: FlowMatchingTCV2ClaConfig):
        super().__init__()
        self.config = config
        D = config.dim_model

        # ---- Task embedding (num_tasks + 1 for CFG null token) ----
        self.task_embed = nn.Embedding(config.num_tasks + 1, D)
        self.null_task_index = config.num_tasks  # index for null/unconditional

        # ---- Vision backbone: DINOv2 ViT-L/14 + LoRA ----
        self.dinov2 = None
        self.lora_params: list[nn.Parameter] = []
        if config.image_features:
            self.dinov2 = AutoModel.from_pretrained(
                config.dinov2_model_name, output_hidden_states=True,
            )
            self.dinov2.eval()
            for param in self.dinov2.parameters():
                param.requires_grad_(False)

            # Apply LoRA to Q/V
            self.lora_params = _apply_lora_to_dinov2(
                self.dinov2, config.lora_rank, config.lora_alpha,
            )

            # Multi-scale projection: 4 layers × dinov2_dim → dim_model
            ms_in_dim = config.dinov2_dim * len(config.multiscale_layers)
            self.multiscale_proj = nn.Linear(ms_in_dim, D)

            num_cameras = len(config.image_features)
            self.camera_embed = nn.Embedding(num_cameras, D)
            self.patch_pos_embed = nn.Embedding(config.dinov2_num_patches, D)

        # ---- State temporal encoder ----
        # 4 steps × 20D = 80D → 768
        self.state_temporal_encoder = None
        if config.robot_state_feature:
            state_input_dim = config.state_dim * config.n_obs_steps
            self.state_temporal_encoder = nn.Sequential(
                nn.Linear(state_input_dim, D),
                nn.SiLU(),
                nn.Linear(D, D),
            )

        # ---- F/T temporal encoder ----
        # 4 steps × 6D = 24D + 6D derivative = 30D → 768
        ft_input_dim = config.ft_dim * config.n_obs_steps
        if config.use_force_derivative:
            ft_input_dim += config.ft_dim  # derivative
        self.ft_temporal_encoder = nn.Sequential(
            nn.Linear(ft_input_dim, D // 2),
            nn.SiLU(),
            nn.Linear(D // 2, D),
        )

        # ---- Observation encoder ----
        self.obs_encoder = _TransformerEncoder(
            D, config.n_heads, config.dim_feedforward, config.dropout,
            config.n_obs_encoder_layers,
        )

        # ---- DiT Action Decoder ----
        self.action_in_proj = nn.Linear(config.action_dim, D)
        self.action_out_proj = nn.Linear(D, config.action_dim)
        self.action_pos_embed = nn.Embedding(config.chunk_size, D)

        # temb construction
        self.time_mlp = nn.Sequential(
            nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D),
        )
        self.task_cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(D, D))
        self.ft_cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(D, D))

        self.decoder = _DiTDecoder(
            D, config.n_heads, config.dim_feedforward, config.dropout,
            config.n_decoder_layers,
        )

        # Output projection with time-adaptive affine
        self.output_norm = nn.LayerNorm(D, elementwise_affine=False)
        self.output_time_proj = nn.Sequential(nn.SiLU(), nn.Linear(D, 2 * D))

        # ---- Auxiliary heads ----
        self.phase_head = nn.Linear(D, config.num_phase_classes)
        self.contact_head = nn.Linear(D, config.num_contact_classes)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in chain(self.obs_encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Re-zero the AdaLN-Zero gates (already done in _AdaLNZero.__init__)
        nn.init.zeros_(self.action_out_proj.weight)
        nn.init.zeros_(self.action_out_proj.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.dinov2 is not None:
            self.dinov2.eval()
        return self

    # ------------------------------------------------------------------
    # Vision: multi-scale DINOv2 + LoRA
    # ------------------------------------------------------------------

    def _encode_images(self, images: list[Tensor]) -> Tensor:
        """Encode camera images → (S_img, B, D) image tokens."""
        D = self.config.dim_model
        all_tokens = []

        for cam_idx, img in enumerate(images):
            if img.ndim == 5:
                img = img[:, -1]  # take current (most recent) frame
            B = img.shape[0]

            img_resized = F.interpolate(
                img,
                size=(self.config.dinov2_image_size, self.config.dinov2_image_size),
                mode="bilinear",
                align_corners=False,
            )

            # Forward through DINOv2 with hidden states
            dino_out = self.dinov2(pixel_values=img_resized)
            hidden_states = dino_out.hidden_states  # tuple of (B, 1+N_patches, dinov2_dim)

            # Extract multi-scale features (skip CLS token at index 0)
            ms_features = []
            for layer_idx in self.config.multiscale_layers:
                # hidden_states is 0-indexed; layer 6 output = hidden_states[6]
                feat = hidden_states[layer_idx][:, 1:]  # (B, 256, 1024)
                ms_features.append(feat)

            # Concat along feature dim: (B, 256, 4096)
            ms_concat = torch.cat(ms_features, dim=-1)
            # Project: (B, 256, D)
            projected = self.multiscale_proj(ms_concat)

            # Add camera and patch position embeddings
            projected = projected + self.camera_embed.weight[cam_idx]
            projected = projected + self.patch_pos_embed.weight.unsqueeze(0)

            # (B, 256, D) → (256, B, D)
            projected = einops.rearrange(projected, "b n d -> n b d")
            all_tokens.append(projected)

        # Concat all cameras: (768, B, D)
        return torch.cat(all_tokens, dim=0)

    # ------------------------------------------------------------------
    # Temporal encoders
    # ------------------------------------------------------------------

    def _encode_state_history(self, batch: dict[str, Tensor]) -> Tensor | None:
        """Encode state history → single token (B, D)."""
        if self.state_temporal_encoder is None:
            return None
        state = batch.get(OBS_STATE)
        if state is None:
            return None
        # state shape: (B, n_obs_steps, state_dim) or (B, state_dim)
        if state.ndim == 2:
            state = state.unsqueeze(1)
        B = state.shape[0]
        # Flatten history: (B, n_obs_steps * state_dim)
        # Only use first state_dim dimensions (exclude F/T which we handle separately)
        state_only = state[..., : self.config.state_dim]
        flat = state_only.reshape(B, -1)
        return self.state_temporal_encoder(flat)

    def _encode_ft_history(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode F/T history + derivative → single token (B, D)."""
        state = batch.get(OBS_STATE)
        if state is None:
            # Fallback: zeros
            B = batch["task_index"].shape[0]
            device = batch["task_index"].device
            return torch.zeros(B, self.config.dim_model, device=device)

        if state.ndim == 2:
            state = state.unsqueeze(1)

        # Extract F/T from end of state vector
        ft = state[..., -self.config.ft_dim:]  # (B, n_obs_steps, 6)
        B = ft.shape[0]

        # Flatten history
        ft_flat = ft.reshape(B, -1)  # (B, n_obs_steps * 6)

        if self.config.use_force_derivative:
            # Derivative: last step - second to last step (or zeros if single step)
            if ft.shape[1] >= 2:
                derivative = ft[:, -1] - ft[:, -2]  # (B, 6)
            else:
                derivative = torch.zeros(B, self.config.ft_dim, device=ft.device)
            ft_flat = torch.cat([ft_flat, derivative], dim=-1)  # (B, 30)

        return self.ft_temporal_encoder(ft_flat)

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def _encode_obs(
        self, batch: dict[str, Tensor], null_task: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Encode all observations.

        Returns:
            obs_features: (S, B, D)
            obs_pos_embeds: (S, 1, D)
            task_embed: (B, D)
            aux: dict with phase_logits, contact_logits
        """
        tokens = []
        pos_embeds = []
        D = self.config.dim_model

        # -- Task token --
        task_index = batch["task_index"]
        if task_index.ndim == 2:
            task_index = task_index.squeeze(-1)
        if null_task:
            task_index = torch.full_like(task_index, self.null_task_index)
        task_tok = self.task_embed(task_index)  # (B, D)
        tokens.append(task_tok)
        pos_embeds.append(torch.zeros(1, D, device=task_tok.device, dtype=task_tok.dtype))

        # -- State token --
        state_tok = self._encode_state_history(batch)
        if state_tok is not None:
            tokens.append(state_tok)
            pos_embeds.append(torch.zeros(1, D, device=state_tok.device, dtype=state_tok.dtype))

        # -- F/T token --
        ft_tok = self._encode_ft_history(batch)
        tokens.append(ft_tok)
        pos_embeds.append(torch.zeros(1, D, device=ft_tok.device, dtype=ft_tok.dtype))

        # -- Image tokens --
        if self.config.image_features and OBS_IMAGES in batch:
            img_tokens = self._encode_images(batch[OBS_IMAGES])  # (768, B, D)
            S_img = img_tokens.shape[0]
            tokens_1d = torch.stack(tokens, dim=0)  # (n_1d, B, D)
            pos_1d = torch.stack(pos_embeds, dim=0)  # (n_1d, 1, D)

            # Image pos embeds are already baked into the tokens (camera_embed + patch_pos)
            img_pos = torch.zeros(S_img, 1, D, device=img_tokens.device, dtype=img_tokens.dtype)

            all_tokens = torch.cat([tokens_1d, img_tokens], dim=0)
            all_pos = torch.cat([pos_1d, img_pos], dim=0)
        else:
            all_tokens = torch.stack(tokens, dim=0)
            all_pos = torch.stack(pos_embeds, dim=0)

        # Encode
        obs_features = self.obs_encoder(all_tokens, pos_embed=all_pos)

        # Auxiliary heads: use state and ft token positions
        # Token order: [task, state, ft, img...]
        # state_tok is at index 1, ft_tok is at index 2 (if state exists)
        aux = {}
        state_idx = 1 if state_tok is not None else None
        ft_idx = 2 if state_tok is not None else 1

        if state_idx is not None:
            aux["phase_logits"] = self.phase_head(obs_features[state_idx])  # (B, 4)
        else:
            aux["phase_logits"] = self.phase_head(obs_features[0])

        aux["contact_logits"] = self.contact_head(obs_features[ft_idx])  # (B, 3)

        return obs_features, all_pos, task_tok, ft_tok, aux

    # ------------------------------------------------------------------
    # Velocity prediction (DiT decoder)
    # ------------------------------------------------------------------

    def _predict_velocity(
        self,
        obs_features: Tensor,
        x_t: Tensor,
        time: Tensor,
        task_embed: Tensor,
        ft_embed: Tensor,
    ) -> Tensor:
        B, T, _ = x_t.shape
        D = self.config.dim_model

        # Build temb: time + task + F/T
        t_sinusoidal = _sinusoidal_time_embedding(
            time, D, self.config.min_period, self.config.max_period,
        )
        temb = self.time_mlp(t_sinusoidal)
        temb = temb + self.task_cond_proj(task_embed)
        temb = temb + self.ft_cond_proj(ft_embed)

        # Action tokens
        a_emb = self.action_in_proj(x_t)  # (B, T, D)
        a_emb = a_emb + self.action_pos_embed.weight[:T].unsqueeze(0)
        a_emb = einops.rearrange(a_emb, "b t d -> t b d")

        # DiT decoder
        out = self.decoder(a_emb, obs_features, temb)  # (T, B, D)
        out = einops.rearrange(out, "t b d -> b t d")

        # Output affine with time conditioning
        scale, shift = self.output_time_proj(temb).chunk(2, dim=-1)
        out = self.output_norm(out) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        return self.action_out_proj(out)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        actions = batch[ACTION]  # (B, T, 6)
        B = actions.shape[0]
        device = actions.device

        # CFG: randomly drop task conditioning
        drop_mask = torch.rand(B, device=device) < self.config.cfg_dropout_prob
        task_index = batch["task_index"]
        if task_index.ndim == 2:
            task_index = task_index.squeeze(-1)

        # Replace dropped task indices with null
        task_index_cfg = task_index.clone()
        task_index_cfg[drop_mask] = self.null_task_index

        # Temporarily override batch task_index for encoding
        batch_cfg = dict(batch)
        batch_cfg["task_index"] = task_index_cfg

        obs_features, _, task_embed, ft_embed, aux = self._encode_obs(batch_cfg)

        # Flow matching
        noise = torch.randn_like(actions)
        time = _sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            B, device,
        ) * self.config.time_sampling_scale + self.config.time_sampling_offset

        t = time[:, None, None]
        x_t = t * noise + (1.0 - t) * actions
        u_t = noise - actions

        v_pred = self._predict_velocity(obs_features, x_t, time, task_embed, ft_embed)

        action_loss = F.mse_loss(v_pred, u_t, reduction="none")  # (B, T, 6)

        # Phase-weighted loss
        if "phase_label" in batch:
            phase = batch["phase_label"].long()  # (B,)
            weights = torch.ones(B, device=device)
            weights[phase == 0] = self.config.phase_weight_approach
            weights[phase == 1] = self.config.phase_weight_align
            weights[phase == 2] = self.config.phase_weight_insert
            weights[phase == 3] = self.config.phase_weight_recover
            action_loss = action_loss * weights[:, None, None]

        losses: dict[str, Tensor] = {"action_loss": action_loss}

        # Auxiliary losses
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

        return losses

    # ------------------------------------------------------------------
    # Inference: midpoint ODE solver with CFG
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_actions(self, batch: dict[str, Tensor]) -> Tensor:
        B = batch["task_index"].shape[0]
        device = batch["task_index"].device
        A = self.config.action_dim
        T = self.config.chunk_size

        # Encode observations (conditional and unconditional)
        obs_cond, _, task_embed_cond, ft_embed_cond, _ = self._encode_obs(batch, null_task=False)
        obs_uncond, _, task_embed_uncond, ft_embed_uncond, _ = self._encode_obs(batch, null_task=True)

        x_t = torch.randn(B, T, A, dtype=torch.float32, device=device)

        n_steps = self.config.num_inference_steps
        dt = -1.0 / n_steps
        w = self.config.cfg_guidance_weight

        for step in range(n_steps):
            t_val = 1.0 + step * dt
            time_curr = torch.full((B,), t_val, dtype=torch.float32, device=device)

            # Midpoint method (2nd order)
            # Step 1: evaluate at current point
            v_cond_1 = self._predict_velocity(obs_cond, x_t, time_curr, task_embed_cond, ft_embed_cond)
            v_uncond_1 = self._predict_velocity(obs_uncond, x_t, time_curr, task_embed_uncond, ft_embed_uncond)
            v1 = v_uncond_1 + w * (v_cond_1 - v_uncond_1)

            # Step 2: midpoint
            x_mid = x_t + (dt / 2) * v1
            time_mid = torch.full((B,), t_val + dt / 2, dtype=torch.float32, device=device)

            v_cond_2 = self._predict_velocity(obs_cond, x_mid, time_mid, task_embed_cond, ft_embed_cond)
            v_uncond_2 = self._predict_velocity(obs_uncond, x_mid, time_mid, task_embed_uncond, ft_embed_uncond)
            v2 = v_uncond_2 + w * (v_cond_2 - v_uncond_2)

            # Full step with midpoint velocity
            x_t = x_t + dt * v2

        return x_t
