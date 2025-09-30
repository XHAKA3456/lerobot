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
Runs inference with a trained policy on a robot (no data collection).

Example:

```shell
lerobot-infer \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --robot.id=black \
    --policy.path=${HF_USER}/my_policy \
    --policy.device=cuda \
    --task="Pick and place the cube" \
    --fps=30 \
    --display_data=true
```

Example with bimanual robot:
```shell
lerobot-infer \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
    right: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}
  }' \
  --policy.path=${HF_USER}/my_bimanual_policy \
  --policy.device=cuda \
  --task="Pick and handover the cube" \
  --fps=30
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Any

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class InferConfig:
    robot: RobotConfig
    # Pretrained policy to load
    policy: PreTrainedConfig
    # Task description (used for VLM policies)
    task: str = "Complete the task"
    # Control frequency (Hz)
    fps: int = 30
    # Maximum inference time in seconds (0 = infinite)
    max_time_s: int = 0
    # Display cameras and action visualization
    display_data: bool = False
    # Use vocal synthesis to read events
    play_sounds: bool = True
    # Dataset repo_id to get stats for normalization (optional, will use policy's dataset if None)
    dataset_repo_id: str | None = None

    def __post_init__(self):
        # Parse policy path from CLI
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        elif self.policy is None:
            raise ValueError("You must provide a policy path using --policy.path=...")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


def inference_loop(
    robot: Robot,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    dataset_features: dict,
    events: dict,
    fps: int,
    task: str,
    max_time_s: int = 0,
    display_data: bool = False,
):
    """
    Main inference loop that continuously runs policy inference on the robot.

    Args:
        robot: Robot instance
        policy: Trained policy
        preprocessor: Policy input preprocessor
        postprocessor: Policy output postprocessor
        robot_action_processor: Processes actions before sending to robot
        robot_observation_processor: Processes observations from robot
        dataset_features: Feature definitions for building observation frames
        events: Dictionary of keyboard events (exit_early, stop_inference, etc.)
        fps: Control frequency
        task: Task description
        max_time_s: Maximum inference time (0 = infinite)
        display_data: Whether to visualize data in rerun
    """
    # Reset policy and processors
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    timestamp = 0
    start_time = time.perf_counter()

    logging.info(f"Starting inference loop at {fps} Hz")
    logging.info(f"Task: {task}")
    logging.info(f"Press 'q' to stop inference")

    while True:
        loop_start = time.perf_counter()

        # Check for exit events
        if events.get("exit_early") or events.get("stop_inference"):
            logging.info("Stopping inference...")
            break

        # Check max time
        if max_time_s > 0 and timestamp >= max_time_s:
            logging.info(f"Reached max inference time: {max_time_s}s")
            break

        # Get robot observation
        obs = robot.get_observation()

        # Process observation
        obs_processed = robot_observation_processor(obs)

        # Build observation frame for policy
        observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

        # Predict action using policy
        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=task,
            robot_type=robot.robot_type,
        )

        # Convert action values to robot action dictionary
        action_names = dataset_features[ACTION]["names"]
        act_processed_policy: RobotAction = {
            f"{name}": float(action_values[i]) for i, name in enumerate(action_names)
        }

        # Process action before sending to robot
        robot_action_to_send = robot_action_processor((act_processed_policy, obs))

        # Send action to robot
        robot.send_action(robot_action_to_send)

        # Visualize if requested
        if display_data:
            log_rerun_data(observation=obs_processed, action=act_processed_policy)

        # Maintain control frequency
        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_time

    logging.info(f"Inference completed. Total time: {timestamp:.2f}s")


@parser.wrap()
def infer(cfg: InferConfig):
    """Main inference function."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="inference")

    # Create robot
    robot = make_robot_from_config(cfg.robot)

    # Create default processors for robot actions/observations
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    # Load dataset to get features and stats for normalization
    # If dataset_repo_id is not provided, try to use the one from policy training
    dataset_repo_id = cfg.dataset_repo_id
    if dataset_repo_id is None and hasattr(cfg.policy, "dataset_repo_id"):
        dataset_repo_id = cfg.policy.dataset_repo_id
        logging.info(f"Using dataset from policy training: {dataset_repo_id}")

    if dataset_repo_id is None:
        raise ValueError(
            "Could not determine dataset for normalization stats. "
            "Please provide --dataset_repo_id or ensure the policy was trained with dataset metadata."
        )

    # Load dataset metadata (we only need features and stats, no actual data)
    dataset = LeRobotDataset(dataset_repo_id, download=False)

    # Load policy
    logging.info(f"Loading policy from: {cfg.policy.pretrained_path}")
    policy = make_policy(cfg.policy, ds_meta=dataset.meta)

    # Create preprocessor and postprocessor
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
        },
    )

    # Connect to robot
    log_say("Connecting to robot", cfg.play_sounds)
    robot.connect()

    # Initialize keyboard listener for exit control
    listener, events = init_keyboard_listener()

    try:
        log_say("Starting inference", cfg.play_sounds)
        inference_loop(
            robot=robot,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            dataset_features=dataset.features,
            events=events,
            fps=cfg.fps,
            task=cfg.task,
            max_time_s=cfg.max_time_s,
            display_data=cfg.display_data,
        )
    finally:
        log_say("Disconnecting from robot", cfg.play_sounds)
        robot.disconnect()

        if not is_headless() and listener is not None:
            listener.stop()

    log_say("Inference complete", cfg.play_sounds)


def main():
    infer()


if __name__ == "__main__":
    main()