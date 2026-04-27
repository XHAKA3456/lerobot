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
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("flowmatching_tc_v0")
@dataclass
class FlowMatchingTCV0Config(PreTrainedConfig):
    """Configuration for the Flow Matching Task-Conditioned V0 policy.

    Extends FlowMatching with a task embedding to support multi-task conditioning.
    A learnable task embedding token is prepended to the observation encoder input.
    """

    # Input/output
    n_obs_steps: int = 4
    chunk_size: int = 20
    n_action_steps: int = 20
    use_latest_image_only: bool = True

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
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Task conditioning
    num_tasks: int = 3

    # Final-phase weighting (V1.0)
    enable_final_phase_weighting: bool = True
    base_loss_weight: float = 1.0
    near_port_loss_weight: float = 2.0
    contact_onset_loss_weight: float = 1.5
    near_port_descent_ratio_center: float = 1.5
    near_port_descent_ratio_scale: float = 4.0
    near_port_speed_center: float = 1.0
    near_port_speed_scale: float = 3.0
    contact_force_level_center: float = 1.0
    contact_force_level_scale: float = 2.0
    contact_force_delta_center: float = 0.5
    contact_force_delta_scale: float = 4.0

    # Optimizer
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    # LR scheduler
    scheduler_warmup_steps: int = 5000
    scheduler_decay_steps: int = 100000
    scheduler_decay_lr: float = 1e-6

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})."
            )
        if self.n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be positive.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
        )

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature and not self.robot_state_feature:
            raise ValueError("At least one image, environment state, or robot state must be provided as input.")

    @property
    def observation_delta_indices_map(self) -> dict[str, list[int]]:
        history = list(range(-(self.n_obs_steps - 1), 1))
        delta_map = {}

        for key, feature in self.input_features.items():
            feature_type = getattr(feature, "type", None)
            feature_type_name = getattr(feature_type, "value", str(feature_type))
            if feature_type_name == "VISUAL":
                # V0 design goal: current image only.
                delta_map[key] = [0]
            elif feature_type_name == "STATE":
                # V0 design goal: temporal proprio / wrench history.
                delta_map[key] = history

        return delta_map

    @property
    def observation_delta_indices(self) -> list[int]:
        # History: [-3, -2, -1, 0] for n_obs_steps=4 at dataset fps.
        return list(range(-(self.n_obs_steps - 1), 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
