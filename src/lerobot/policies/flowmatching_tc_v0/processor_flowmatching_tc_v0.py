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

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.policies.flowmatching_tc_v0.configuration_flowmatching_tc_v0 import FlowMatchingTCV0Config
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import OBS_STATE, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


LOSS_WEIGHT = "loss_weight"


@ProcessorStepRegistry.register(name="flowmatching_tc_v0_final_phase_loss_weight")
@dataclass
class FlowMatchingTCV0FinalPhaseLossWeightStep(ProcessorStep):
    """Compute V1 final-phase loss weights before action/state normalization."""

    enabled: bool = True
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
    robot_state_feature: bool = True
    proprio_dim: int = 20

    def __call__(self, transition):
        if not self.enabled:
            return transition

        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        if action.ndim == 2:
            actions = action.unsqueeze(1)
        elif action.ndim == 3:
            actions = action
        else:
            return transition

        B, T, A = actions.shape
        device = actions.device
        dtype = actions.dtype
        weight = torch.full((B, T, 1), self.base_loss_weight, device=device, dtype=dtype)

        pos_action = actions[..., :3]
        lateral_mag = torch.linalg.vector_norm(pos_action[..., :2], dim=-1)
        downward_mag = torch.relu(-pos_action[..., 2])
        descent_ratio = downward_mag / (lateral_mag + 1e-6)
        descent_gate = torch.sigmoid(
            (descent_ratio - self.near_port_descent_ratio_center) * self.near_port_descent_ratio_scale
        )

        speed_gate = torch.ones((B,), device=device, dtype=dtype)
        force_level_gate = torch.zeros((B,), device=device, dtype=dtype)
        force_delta_gate = torch.zeros((B,), device=device, dtype=dtype)

        observation = transition.get(TransitionKey.OBSERVATION) or {}
        state = observation.get(OBS_STATE)
        if self.robot_state_feature and state is not None:
            if state.ndim == 2:
                state = state.unsqueeze(1)

            current_state = state[:, -1]
            tcp_velocity = current_state[:, 7:13]
            speed_norm = torch.linalg.vector_norm(tcp_velocity, dim=-1)
            speed_gate = torch.sigmoid((self.near_port_speed_center - speed_norm) * self.near_port_speed_scale)

            current_wrench = current_state[:, self.proprio_dim :]
            force_norm = torch.linalg.vector_norm(current_wrench, dim=-1)
            force_level_gate = torch.sigmoid(
                (force_norm - self.contact_force_level_center) * self.contact_force_level_scale
            )

            if state.shape[1] >= 2:
                prev_wrench = state[:, -2, self.proprio_dim :]
                force_delta = torch.linalg.vector_norm(current_wrench - prev_wrench, dim=-1)
                force_delta_gate = torch.sigmoid(
                    (force_delta - self.contact_force_delta_center) * self.contact_force_delta_scale
                )

        near_port_gate = descent_gate * speed_gate[:, None]
        contact_onset_gate = near_port_gate * torch.maximum(force_level_gate, force_delta_gate)[:, None]
        weight = (
            weight
            + self.near_port_loss_weight * near_port_gate.unsqueeze(-1)
            + self.contact_onset_loss_weight * contact_onset_gate.unsqueeze(-1)
        )

        new_transition = transition.copy()
        complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        complementary_data[LOSS_WEIGHT] = weight.expand(-1, -1, A)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "base_loss_weight": self.base_loss_weight,
            "near_port_loss_weight": self.near_port_loss_weight,
            "contact_onset_loss_weight": self.contact_onset_loss_weight,
            "near_port_descent_ratio_center": self.near_port_descent_ratio_center,
            "near_port_descent_ratio_scale": self.near_port_descent_ratio_scale,
            "near_port_speed_center": self.near_port_speed_center,
            "near_port_speed_scale": self.near_port_speed_scale,
            "contact_force_level_center": self.contact_force_level_center,
            "contact_force_level_scale": self.contact_force_level_scale,
            "contact_force_delta_center": self.contact_force_delta_center,
            "contact_force_delta_scale": self.contact_force_delta_scale,
            "robot_state_feature": self.robot_state_feature,
            "proprio_dim": self.proprio_dim,
        }

    def transform_features(self, features):
        return features


def make_flowmatching_tc_v0_pre_post_processors(
    config: FlowMatchingTCV0Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the FlowMatchingTCV0 policy.
    """

    # Pre-processing steps
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        FlowMatchingTCV0FinalPhaseLossWeightStep(
            enabled=config.enable_final_phase_weighting,
            base_loss_weight=config.base_loss_weight,
            near_port_loss_weight=config.near_port_loss_weight,
            contact_onset_loss_weight=config.contact_onset_loss_weight,
            near_port_descent_ratio_center=config.near_port_descent_ratio_center,
            near_port_descent_ratio_scale=config.near_port_descent_ratio_scale,
            near_port_speed_center=config.near_port_speed_center,
            near_port_speed_scale=config.near_port_speed_scale,
            contact_force_level_center=config.contact_force_level_center,
            contact_force_level_scale=config.contact_force_level_scale,
            contact_force_delta_center=config.contact_force_delta_center,
            contact_force_delta_scale=config.contact_force_delta_scale,
            robot_state_feature=bool(config.robot_state_feature),
        ),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]

    # Post-processing steps
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
