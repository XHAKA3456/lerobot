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

from __future__ import annotations

import logging
import time
from functools import cached_property
from itertools import chain
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_xlerobot_arms import XLerobotArmsConfig

logger = logging.getLogger(__name__)


class XLerobotArms(Robot):
    """Arms-only variant of XLeRobot (two 6-DOF arms, no head/base)."""

    config_class = XLerobotArmsConfig
    name = "xlerobot_arms"

    def __init__(self, config: XLerobotArmsConfig):
        super().__init__(config)
        self.config = config

        norm_mode = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        left_calib = (
            {k: v for k, v in self.calibration.items() if k.startswith("left_arm")}
            if self.calibration
            else {}
        )
        right_calib = (
            {k: v for k, v in self.calibration.items() if k.startswith("right_arm")}
            if self.calibration
            else {}
        )

        self.bus_left = FeetechMotorsBus(
            port=self.config.port_left,
            motors={
                "left_arm_shoulder_pan": Motor(1, "sts3215", norm_mode),
                "left_arm_shoulder_lift": Motor(2, "sts3215", norm_mode),
                "left_arm_elbow_flex": Motor(3, "sts3215", norm_mode),
                "left_arm_wrist_flex": Motor(4, "sts3215", norm_mode),
                "left_arm_wrist_roll": Motor(5, "sts3215", norm_mode),
                "left_arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=left_calib,
        )
        self.bus_right = FeetechMotorsBus(
            port=self.config.port_right,
            motors={
                "right_arm_shoulder_pan": Motor(1, "sts3215", norm_mode),
                "right_arm_shoulder_lift": Motor(2, "sts3215", norm_mode),
                "right_arm_elbow_flex": Motor(3, "sts3215", norm_mode),
                "right_arm_wrist_flex": Motor(4, "sts3215", norm_mode),
                "right_arm_wrist_roll": Motor(5, "sts3215", norm_mode),
                "right_arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=right_calib,
        )

        self.left_arm_motors = list(self.bus_left.motors.keys())
        self.right_arm_motors = list(self.bus_right.motors.keys())
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                "left_arm_shoulder_pan.pos",
                "left_arm_shoulder_lift.pos",
                "left_arm_elbow_flex.pos",
                "left_arm_wrist_flex.pos",
                "left_arm_wrist_roll.pos",
                "left_arm_gripper.pos",
                "right_arm_shoulder_pan.pos",
                "right_arm_shoulder_lift.pos",
                "right_arm_elbow_flex.pos",
                "right_arm_wrist_flex.pos",
                "right_arm_wrist_roll.pos",
                "right_arm_gripper.pos",
            ),
            float,
        )

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self.bus_left.is_connected and self.bus_right.is_connected and all(
            cam.is_connected for cam in self.cameras.values()
        )

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus_left.connect()
        self.bus_right.connect()

        if self.calibration_fpath.is_file():
            logger.info(f"Calibration file found at {self.calibration_fpath}")
            user_input = input(
                "Press ENTER to restore calibration from file, or type 'c' to run manual calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info("Restoring calibration from file...")
                try:
                    self.bus_left.calibration = {
                        k: v for k, v in self.calibration.items() if k in self.bus_left.motors
                    }
                    self.bus_right.calibration = {
                        k: v for k, v in self.calibration.items() if k in self.bus_right.motors
                    }
                    self.bus_left.write_calibration(self.bus_left.calibration)
                    self.bus_right.write_calibration(self.bus_right.calibration)
                    logger.info("Calibration restored successfully!")
                except Exception as exc:
                    logger.warning(f"Failed to restore calibration: {exc}")
                    if calibrate:
                        self.calibrate()
            elif calibrate:
                self.calibrate()
        elif calibrate:
            logger.info("No calibration file found, starting manual calibration...")
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus_left.is_calibrated and self.bus_right.is_calibrated

    def calibrate(self) -> None:
        logger.info(f"Running calibration for {self}")

        # Left arm
        self.bus_left.disable_torque(self.left_arm_motors)
        for name in self.left_arm_motors:
            self.bus_left.write("Operating_Mode", name, OperatingMode.POSITION.value)
        input("Move LEFT arm joints to mid-range and press ENTER...")
        homing_offsets = self.bus_left.set_half_turn_homings(self.left_arm_motors)
        print("Move LEFT arm joints through full range, press ENTER to stop recording...")
        range_mins, range_maxes = self.bus_left.record_ranges_of_motion(self.left_arm_motors)
        calibration_left = {
            name: MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets[name],
                range_min=range_mins[name],
                range_max=range_maxes[name],
            )
            for name, motor in self.bus_left.motors.items()
        }
        self.bus_left.write_calibration(calibration_left)

        # Right arm
        self.bus_right.disable_torque(self.right_arm_motors)
        for name in self.right_arm_motors:
            self.bus_right.write("Operating_Mode", name, OperatingMode.POSITION.value)
        input("Move RIGHT arm joints to mid-range and press ENTER...")
        homing_offsets = self.bus_right.set_half_turn_homings(self.right_arm_motors)
        print("Move RIGHT arm joints through full range, press ENTER to stop recording...")
        range_mins, range_maxes = self.bus_right.record_ranges_of_motion(self.right_arm_motors)
        calibration_right = {
            name: MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets[name],
                range_min=range_mins[name],
                range_max=range_maxes[name],
            )
            for name, motor in self.bus_right.motors.items()
        }
        self.bus_right.write_calibration(calibration_right)

        self.calibration = {**calibration_left, **calibration_right}
        self._save_calibration()
        logger.info("Calibration saved.")

    def configure(self) -> None:
        self.bus_left.disable_torque(self.left_arm_motors)
        self.bus_right.disable_torque(self.right_arm_motors)

        for bus, motors in ((self.bus_left, self.left_arm_motors), (self.bus_right, self.right_arm_motors)):
            for name in motors:
                bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
                bus.write("P_Coefficient", name, 16)
                bus.write("I_Coefficient", name, 0)
                bus.write("D_Coefficient", name, 43)

        self._enable_torque_safely(self.bus_left, self.left_arm_motors)
        self._enable_torque_safely(self.bus_right, self.right_arm_motors)

    def _enable_torque_safely(self, bus: FeetechMotorsBus, motors: list[str]) -> None:
        for name in motors:
            try:
                bus.enable_torque(name)
            except ConnectionError as exc:
                logger.warning("Skipping torque enable for %s: %s", name, exc)

    def setup_motors(self) -> None:
        for motor in chain(reversed(self.left_arm_motors), reversed(self.right_arm_motors)):
            bus = self.bus_left if motor.startswith("left") else self.bus_right
            input(f"Connect controller to '{motor}' only, then press ENTER...")
            bus.setup_motor(motor)
            logger.info("'%s' motor id set to %s", motor, bus.motors[motor].id)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        left_arm_pos = self.bus_left.sync_read("Present_Position", self.left_arm_motors)
        right_arm_pos = self.bus_right.sync_read("Present_Position", self.right_arm_motors)
        obs = {
            **{f"{k}.pos": v for k, v in left_arm_pos.items()},
            **{f"{k}.pos": v for k, v in right_arm_pos.items()},
        }

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read arm state in {dt_ms:.1f} ms")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f} ms")

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        left_arm_pos = {
            k: v for k, v in action.items() if k.startswith("left_arm_") and k.endswith(".pos")
        }
        right_arm_pos = {
            k: v for k, v in action.items() if k.startswith("right_arm_") and k.endswith(".pos")
        }

        if self.config.max_relative_target is not None:
            present_pos_left = {
                f"{k}.pos": v
                for k, v in self.bus_left.sync_read("Present_Position", self.left_arm_motors).items()
            }
            present_pos_right = {
                f"{k}.pos": v
                for k, v in self.bus_right.sync_read("Present_Position", self.right_arm_motors).items()
            }
            present_pos = {**present_pos_left, **present_pos_right}
            goal_present = {
                key: (goal, present_pos[key]) for key, goal in chain(left_arm_pos.items(), right_arm_pos.items())
            }
            safe_goals = ensure_safe_goal_position(goal_present, self.config.max_relative_target)
            left_arm_pos = {k: v for k, v in safe_goals.items() if k in left_arm_pos}
            right_arm_pos = {k: v for k, v in safe_goals.items() if k in right_arm_pos}

        left_raw = {k.replace(".pos", ""): v for k, v in left_arm_pos.items()}
        right_raw = {k.replace(".pos", ""): v for k, v in right_arm_pos.items()}

        if left_raw:
            self.bus_left.sync_write("Goal_Position", left_raw)
        if right_raw:
            self.bus_right.sync_write("Goal_Position", right_raw)

        return {**left_arm_pos, **right_arm_pos}

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.disable_torque_on_disconnect:
            self.bus_left.disable_torque()
            self.bus_right.disable_torque()

        self.bus_left.disconnect()
        self.bus_right.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
