"""
단일 모터 ID 설정 스크립트.

사용법:
    python bi_so101_configs/setup_single_motor.py --port /dev/ttyACM0 --motor wrist_flex

사용 가능한 모터 이름과 할당 ID:
    shoulder_pan  → 1
    shoulder_lift → 2
    elbow_flex    → 3
    wrist_flex    → 4
    wrist_roll    → 5
    gripper       → 6
"""

import argparse

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

MOTOR_TABLE = {
    "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex":    Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex":    Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll":    Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}


def main():
    parser = argparse.ArgumentParser(description="단일 모터 ID 설정")
    parser.add_argument("--port", required=True, help="시리얼 포트 (예: /dev/ttyACM0)")
    parser.add_argument(
        "--motor",
        required=True,
        choices=list(MOTOR_TABLE.keys()),
        help="설정할 모터 이름",
    )
    args = parser.parse_args()

    motor_def = MOTOR_TABLE[args.motor]

    bus = FeetechMotorsBus(
        port=args.port,
        motors={args.motor: motor_def},
    )

    print(f"'{args.motor}' 모터를 찾는 중... (해당 모터만 보드에 연결되어 있어야 합니다)")
    bus.setup_motor(args.motor)
    print(f"완료: '{args.motor}' 모터 ID가 {motor_def.id}로 설정되었습니다.")


if __name__ == "__main__":
    main()
