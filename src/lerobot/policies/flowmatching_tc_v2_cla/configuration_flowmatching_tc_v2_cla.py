#!/usr/bin/env python

# FlowMatchingTC V2 Cla Configuration
# See /home/sst/aic/flowmatching_v2_cla.md for full design document.

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("flowmatching_tc_v2_cla")
@dataclass
class FlowMatchingTCV2ClaConfig(PreTrainedConfig):
    """Configuration for FlowMatchingTC V2.

    Key changes from V1:
    - DINOv2 ViT-L/14 + LoRA (rank 32)
    - Multi-scale features from layers [6, 12, 18, 24]
    - No pool: 256 tokens/cam = 768 image tokens
    - State/F/T temporal history (4 steps)
    - Force derivative input
    - Delta action (6D: dx, dy, dz, axis-angle)
    - 12-layer DiT decoder with AdaLN-Zero (gate)
    - CFG w=1.1 with null task dropout
    - 4-step midpoint ODE solver
    - Temporal ensemble
    - Auxiliary phase/contact heads
    - Phase-weighted loss
    """

    # --- Input / Output ---
    n_obs_steps: int = 4
    chunk_size: int = 10
    n_action_steps: int = 3

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # --- Vision backbone ---
    dinov2_model_name: str = "facebook/dinov2-large"
    dinov2_dim: int = 1024  # ViT-L output dim
    dinov2_image_size: int = 224
    dinov2_num_patches: int = 256  # (224/14)^2 = 16^2 = 256
    multiscale_layers: list[int] = field(default_factory=lambda: [6, 12, 18, 24])

    # LoRA
    lora_rank: int = 32
    lora_alpha: float = 32.0

    # --- Temporal history ---
    state_dim: int = 20  # tcp_pose(7) + velocity(6) + joints(7)
    ft_dim: int = 6  # force(3) + torque(3)
    use_force_derivative: bool = True

    # --- Model dimensions ---
    dim_model: int = 768
    n_heads: int = 12
    dim_feedforward: int = 3072
    dropout: float = 0.1

    # Encoder
    n_obs_encoder_layers: int = 6

    # Decoder (DiT with AdaLN-Zero)
    n_decoder_layers: int = 12

    # --- Action ---
    action_dim: int = 6  # delta: dx, dy, dz, ax, ay, az

    # --- Task conditioning ---
    num_tasks: int = 3

    # --- CFG ---
    cfg_dropout_prob: float = 0.1
    cfg_guidance_weight: float = 1.1

    # --- Auxiliary heads ---
    num_phase_classes: int = 4   # approach, align, insert, recover
    num_contact_classes: int = 3  # free, touch, jam
    aux_phase_loss_weight: float = 0.3
    aux_contact_loss_weight: float = 0.3

    # --- Phase-weighted loss ---
    phase_weight_approach: float = 1.0
    phase_weight_align: float = 3.0
    phase_weight_insert: float = 3.0
    phase_weight_recover: float = 2.0

    # --- Flow matching ---
    num_inference_steps: int = 4
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # --- Temporal ensemble ---
    temporal_ensemble_lambda: float = 0.01

    # --- Optimizer ---
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    # --- LR scheduler ---
    scheduler_warmup_steps: int = 5000
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
            raise ValueError(
                "At least one image, environment state, or robot state must be provided."
            )

    @property
    def observation_delta_indices(self) -> list[int]:
        # History: [-3, -2, -1, 0] for n_obs_steps=4 at dataset fps
        return list(range(-(self.n_obs_steps - 1), 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
