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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("flowmatching")
@dataclass
class FlowMatchingConfig(PreTrainedConfig):
    """Configuration for the Flow Matching policy.

    Uses conditional flow matching (pi0-style math) with a frozen DINOv2 ViT-B/14 vision backbone
    and Transformer velocity network.

    Flow matching math (from pi0):
        Training:
            t ~ Beta(1.5, 1.0) * 0.999 + 0.001
            x_t = t * noise + (1 - t) * actions   (forward process)
            u_t = noise - actions                  (velocity target)
            loss = MSE(network(x_t, t, obs), u_t)

        Inference (backward Euler ODE, t: 1→0):
            x_1 = noise ~ N(0, I)
            for step in range(num_inference_steps):
                t = 1.0 + step * (-1/num_inference_steps)
                v_t = network(x_t, t, obs)
                x_t = x_t + (-1/num_inference_steps) * v_t
            return x_0  (denoised action)

    Architecture:
        - Vision: DINOv2 ViT-B/14 (frozen) → 256 patch tokens per image, 768-dim → projected to dim_model
        - Obs encoder: Transformer encoder over (state_token + image_tokens)
        - Velocity net: Transformer decoder, queries=action+time, keys/values=obs_features
    """

    # Input/output
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Vision backbone: DINOv2 (frozen, HuggingFace)
    dinov2_model_name: str = "facebook/dinov2-base"  # outputs 768-dim patch tokens

    # Transformer dimensions
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    dropout: float = 0.1

    # Obs encoder: Transformer encoder over vision + state tokens
    n_obs_encoder_layers: int = 6

    # Velocity network: Transformer decoder (cross-attn to obs features)
    n_velocity_layers: int = 6

    # Flow matching hyperparameters (pi0-style)
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5   # Beta(α=1.5, β=1.0) biases toward mid-range t
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999      # Avoids t=1.0 (numerical stability)
    time_sampling_offset: float = 0.001     # Avoids t=0.0 (numerical stability)
    min_period: float = 4e-3                # Sinusoidal time emb: shortest period
    max_period: float = 4.0                 # Sinusoidal time emb: longest period

    # Optimizer (DINOv2 is frozen, so single lr)
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})."
            )
        if self.n_obs_steps != 1:
            raise ValueError("Multiple observation steps not supported. Set n_obs_steps=1.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature and not self.robot_state_feature:
            raise ValueError("At least one image, environment state, or robot state must be provided as input.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
