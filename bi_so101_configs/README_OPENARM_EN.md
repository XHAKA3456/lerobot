# OpenArm Teleoperation Guide (Meta Quest)

A guide for teleoperating the OpenArm bimanual robot using Meta Quest controllers.

---

## 1. Set Up the Robot

Stand the OpenArm upright and connect the power cables to the correct arms (left and right).

## 2. Connect CAN Communication

Connect the CAN communication cables to the appropriate ports.

| Arm | CAN Port | Wire Color |
|-----|----------|------------|
| Right arm | `can0` | Red = CAN_H, Black = CAN_L |
| Left arm | `can1` | Red = CAN_H, Black = CAN_L |

## 3. Connect Meta Quest

Connect the Meta Quest to the Rubik Pi via USB cable.

## 4. Navigate to Workspace

▶️ **Run**
```bash
cd ros2_ws
```

## 5. Set Up CAN Interface

▶️ **Run**
```bash
./setup_can.sh up
```

## 6. Set Up Quest Connection

▶️ **Run**
```bash
./set_quest.sh
```

## 7. Launch the Robot

▶️ **Run**
```bash
ros2 launch openarm_quest_teleop quest_teleop_bimanual_no_rviz.launch.py use_fake_hardware:=false
```

## 8. Wait for Robot Initialization

The robot will move to its default pose (arms extended). **Stand next to the robot** while it initializes.

## 9. Turn On Meta Quest

Once the robot is ready:

1. Stand next to the robot
2. Turn on the Meta Quest

> **⚠️ IMPORTANT**:
> - Do NOT change your position after turning on the Meta Quest
> - If the Meta Quest is already on, **turn it off and on again**

## 10. Connect Meta Quest to Rubik Pi

Wait **3 seconds** after turning on the Meta Quest, then press the **A button**.

> The A button establishes the connection between Rubik Pi and Meta Quest.

## 11. Start Teleoperation

After the connection is established, press the **gripper button** on the controller to:
- Start sending data
- Make the robot follow your movements

## 12. Stop the Demo

When finished, press `Ctrl+C` in the terminal to stop the robot.

> **Note**: You don't need to press the emergency stop button. Just use `Ctrl+C`.
