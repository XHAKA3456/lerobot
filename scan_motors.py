#!/usr/bin/env python3
"""Scan for all motors on a given port."""

import sys
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

def scan_motors(port: str):
    """Scan all motor IDs on the given port."""
    print(f"Scanning for motors on {port}...")

    # Create a temporary bus with a dummy motor just to initialize
    dummy_motors = {"dummy": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100)}
    bus = FeetechMotorsBus(port=port, motors=dummy_motors, protocol_version=0)

    try:
        # Open port
        if not bus.port_handler.openPort():
            print(f"Failed to open port {port}")
            return

        # Set baudrate
        bus.set_baudrate(1_000_000)

        # Broadcast ping to find all motors
        print("Broadcasting ping...")
        result = bus.broadcast_ping(num_retry=2)

        if result:
            print(f"\n✓ Found {len(result)} motor(s):")
            for motor_id, model_number in result.items():
                print(f"  - ID: {motor_id}, Model: {model_number}")
        else:
            print("\n✗ No motors found!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if bus.port_handler.is_open:
            bus.port_handler.closePort()

if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB1"
    scan_motors(port)
