"""
포트에 연결된 모터 스캔 스크립트.

사용법:
    python bi_so101_configs/scan_motor.py --port /dev/ttyACM0
"""

import argparse

from lerobot.motors.feetech import FeetechMotorsBus

ID_TO_NAME = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper",
}


def main():
    parser = argparse.ArgumentParser(description="포트에 연결된 모터 스캔")
    parser.add_argument("--port", required=True, help="시리얼 포트 (예: /dev/ttyACM0)")
    args = parser.parse_args()

    print(f"포트 '{args.port}' 스캔 중...")
    result = FeetechMotorsBus.scan_port(args.port)

    if not result:
        print("모터를 찾지 못했습니다.")
        return

    print("\n=== 스캔 결과 ===")
    for baudrate, ids in result.items():
        print(f"\nbaudrate {baudrate}:")
        for motor_id in ids:
            name = ID_TO_NAME.get(motor_id, "알 수 없음")
            print(f"  ID {motor_id} → {name}")


if __name__ == "__main__":
    main()
