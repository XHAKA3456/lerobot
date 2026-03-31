#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("flowmatching_tc_v2_co")
@dataclass
class FlowMatchingTCV2CoConfig(PreTrainedConfig):
    """Configuration for the Co V2.1 end-to-end flow matching policy."""

    n_obs_steps: int = 4
    chunk_size: int = 12
    n_action_steps: int = 4

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    dinov2_model_name: str = "facebook/dinov2-base"
    image_size: int = 224
    roi_image_size: int = 160
    roi_feature_names: list[str] = field(
        default_factory=lambda: [
            "observation.images.center_camera_roi",
            "observation.images.right_camera_roi",
        ]
    )
    use_roi_branch: bool = True
    use_roi_encoder_shared_backbone: bool = True
    roi_patch_pool_kernel: int = 2
    roi_crop_size: int = 160

    ft_feature_dim: int = 6
    use_force_history: bool = True
    use_force_derivative: bool = True
    use_action_history: bool = False

    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    n_obs_encoder_layers: int = 8
    n_velocity_layers: int = 8
    temporal_history_layers: int = 2

    num_inference_steps: int = 4
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    num_tasks: int = 3
    num_phase_classes: int = 4
    num_contact_classes: int = 3
    aux_phase_loss_weight: float = 0.5
    aux_contact_loss_weight: float = 0.5
    aux_relative_loss_weight: float = 1.0

    action_mode: str = "delta_pose"
    near_port_loss_weight: float = 2.0
    contact_loss_weight: float = 2.0

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 100000
    scheduler_decay_lr: float = 1e-6

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})."
            )
        if self.n_obs_steps < 1:
            raise ValueError("n_obs_steps must be >= 1.")

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
        if not self.image_features and not self.robot_state_feature and not self.env_state_feature:
            raise ValueError("At least one image, environment state, or robot state must be provided as input.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
