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

"""
Flow Matching Policy for Robot Learning.

Implements conditional flow matching for action generation from visual and proprioceptive observations.
"""

import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn

from lerobot.policies.flowmatching.configuration_flowmatching import FlowMatchingConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_output_shape
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.utils import get_safe_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float = 4e-3, max_period: float = 4.0, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].float()
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def resize_image(img, size):
    """Resize image to target size."""
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")
    return F.interpolate(img, size=size, mode="bilinear", align_corners=False)


def pad_vector(vector, new_dim):
    """Pad vector to new dimension."""
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


class VisionEncoder(nn.Module):
    """Vision encoder using Vision Transformer (ViT)."""

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config

        # Load pretrained ViT
        if config.pretrained_backbone_weights:
            weights_class = getattr(torchvision.models, config.vision_backbone_weights)
            weights = getattr(weights_class, config.pretrained_backbone_weights)
            vit_model = getattr(torchvision.models, config.vision_backbone)(weights=weights)
        else:
            vit_model = getattr(torchvision.models, config.vision_backbone)()

        # Extract encoder (everything except the classification head)
        self.vit_encoder = vit_model
        # Remove the classification head
        self.vit_encoder.heads = nn.Identity()

        # ViT outputs [CLS] token embedding
        self.feature_dim = config.vit_embed_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode images to feature vectors.
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1]
        Returns:
            (B, vit_embed_dim) feature vector
        """
        # ViT expects images normalized with ImageNet statistics
        # Mean and std for ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # Forward through ViT
        features = self.vit_encoder(x)
        return features


class FlowMatchingPolicy(PreTrainedPolicy):
    """
    Flow Matching Policy for robot control.

    Uses conditional flow matching to generate action sequences from visual observations
    and robot state. The policy learns to model the velocity field of actions through
    continuous normalizing flows.
    """

    config_class = FlowMatchingConfig
    name = "flowmatching"

    def __init__(self, config: FlowMatchingConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = FlowMatchingModel(config)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("Currently not implemented for FlowMatching")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        if len(self._action_queue) == 0:
            images, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)

            actions = self.model.sample_actions(images, img_masks, state, noise=noise)

            # Unpad actions
            original_action_dim = self.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]

            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict[str, Tensor]]:
        """Do a full training forward pass to compute the loss."""
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        loss_dict = {}
        losses = self.model.forward(images, img_masks, state, actions, noise, time)
        loss_dict["losses_after_forward"] = losses.clone()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()

        # For backward pass
        loss = losses.mean()
        # For logging
        loss_dict["l2_loss"] = loss.item()

        return loss, loss_dict

    def prepare_images(self, batch):
        """Apply preprocessing to the images."""
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # Resize to expected input size for ViT
            img = resize_image(img, self.config.image_size)

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.zeros_like(img)
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_state(self, batch):
        """Pad state."""
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action."""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions


class FlowMatchingModel(nn.Module):
    """
    Core Flow Matching model implementing conditional flow matching.

    Architecture:
    - Vision encoder: ViT for image feature extraction
    - State encoder: MLP for proprioceptive state
    - Conditioning network: Fuses vision and state features
    - Denoising network: Transformer layers for action prediction
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config

        # Vision encoder
        if config.image_features:
            self.vision_encoder = VisionEncoder(config)
            num_images = len(config.image_features)
            vision_dim = self.vision_encoder.feature_dim * num_images
        else:
            self.vision_encoder = None
            vision_dim = 0

        # Projections
        self.state_proj = nn.Linear(config.max_state_dim, config.hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, config.hidden_dim) if vision_dim > 0 else None

        # Action input/output projections
        self.action_in_proj = nn.Linear(config.max_action_dim, config.hidden_dim)
        self.action_out_proj = nn.Linear(config.hidden_dim, config.max_action_dim)

        # Time embedding + action MLP
        self.action_time_mlp_in = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.action_time_mlp_out = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Conditioning network - processes observations
        self.cond_network = nn.Sequential(
            nn.Linear(config.hidden_dim * 2 if vision_dim > 0 else config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Denoising network - transformer-like architecture
        self.denoising_network = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=8,
                dim_feedforward=config.hidden_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(6)
        ])

    def sample_noise(self, shape, device):
        """Sample noise from standard normal distribution."""
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        """Sample time from Beta distribution."""
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def encode_observations(self, images, img_masks, state):
        """Encode observations (images + state) into conditioning vector."""
        # Encode state
        state_emb = self.state_proj(state)

        # Encode images if available
        if self.vision_encoder is not None and len(images) > 0:
            img_features = []
            for img, mask in zip(images, img_masks, strict=False):
                # Only encode valid images
                img_feat = self.vision_encoder(img)
                # Apply mask
                img_feat = img_feat * mask.unsqueeze(-1).float()
                img_features.append(img_feat)

            # Concatenate all image features
            vision_emb = torch.cat(img_features, dim=-1)
            vision_emb = self.vision_proj(vision_emb)

            # Combine state and vision
            combined = torch.cat([state_emb, vision_emb], dim=-1)
        else:
            combined = state_emb

        # Process through conditioning network
        cond = self.cond_network(combined)
        return cond

    def forward(
        self, images, img_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Training forward pass - compute flow matching loss."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # Flow matching: x_t = t * noise + (1 - t) * actions
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions  # Target velocity field

        # Encode observations
        cond = self.encode_observations(images, img_masks, state)

        # Embed time using sinusoidal encoding
        time_emb = create_sinusoidal_pos_embedding(
            time, self.config.hidden_dim, min_period=4e-3, max_period=4.0, device=actions.device
        )

        # Embed noisy actions
        action_emb = self.action_in_proj(x_t)  # (B, n_action_steps, hidden_dim)

        # Fuse time with action embeddings
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add conditioning to each action step
        cond_expanded = cond[:, None, :].expand_as(action_time_emb)
        x = action_time_emb + cond_expanded

        # Process through denoising network
        for layer in self.denoising_network:
            x = layer(x)

        # Project back to action space
        v_t = self.action_out_proj(x)

        # Compute loss
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(self, images, img_masks, state, noise=None) -> Tensor:
        """Inference forward pass - sample actions using Euler method."""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        # Encode observations once
        cond = self.encode_observations(images, img_masks, state)

        # Euler integration
        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(cond, x_t, expanded_time)

            # Euler step
            x_t += dt * v_t
            time += dt

        return x_t

    def denoise_step(self, cond, x_t, timestep):
        """Apply one denoising step."""
        device = x_t.device

        # Embed time
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.hidden_dim, min_period=4e-3, max_period=4.0, device=device
        )

        # Embed noisy actions
        action_emb = self.action_in_proj(x_t)

        # Fuse time with action embeddings
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add conditioning
        cond_expanded = cond[:, None, :].expand_as(action_time_emb)
        x = action_time_emb + cond_expanded

        # Process through denoising network
        for layer in self.denoising_network:
            x = layer(x)

        # Project back to action space
        v_t = self.action_out_proj(x)
        return v_t