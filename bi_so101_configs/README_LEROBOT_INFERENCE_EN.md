# BI-SO101 Inference Guide

> ## ⚠️ Run this on Rubik Pi

A guide for running inference with trained models using the BI-SO101 bimanual robot.

---

## 1. Activate Environment

Open a terminal and activate the xlerobot environment.

▶️ **Run**
```bash
conda activate xlerobot
```

## 2. Navigate to Working Directory

▶️ **Run**
```bash
cd lerobot
```

## 3. Connect the Robot

Use two USB hubs. **⚠️ The connection order is CRITICAL! You MUST follow the exact order below.**

### Connection Order

> **⚠️ IMPORTANT**: Connect cables in the exact order listed below (1 → 2 → 3 → 4 → 5 → 6 → 7). Do NOT skip or change the order!

| Order | USB Hub | Device | Port |
|:-----:|---------|--------|------|
| 1 | First USB Hub | Follower left arm | `/dev/ttyACM0` |
| 2 | First USB Hub | Follower right arm | `/dev/ttyACM1` |
| 3 | First USB Hub | Left camera | `/dev/video2` |
| 4 | Second USB Hub | Leader left arm | `/dev/ttyACM2` |
| 5 | Second USB Hub | Leader right arm | `/dev/ttyACM3` |
| 6 | Second USB Hub | Right camera | `/dev/video4` |
| 7 | First USB Hub | Front camera | `/dev/video6` |

> **⚠️ Note**: The Front camera is connected to the **First USB Hub**, but must be plugged in **LAST** after all other connections are complete.

### USB Permission Setup

Grant permissions after connecting all cables.

▶️ **Run**
```bash
chacm
```

> **Note**:
> - You don't need to run this after each cable connection. Just run it once after all cables are connected.
> - If any cable gets disconnected, **reconnect all cables in order** and grant permissions again with `chacm`.

## 4. Maintain Data Collection Environment

Keep the same environment setup that was used during data collection. This includes:
- Camera positions and angles
- Lighting conditions
- Object placement area

## 5. Download the Model

Download the trained model from Hugging Face Hub using the CLI.

▶️ **Run**
```bash
huggingface-cli download <repo_id> --local-dir /home/ubuntu/lerobot/bi_so101_configs/models/<model_name>
```

📝 **Example**
```bash
huggingface-cli download xhaka3456/univ_model --local-dir /home/ubuntu/lerobot/bi_so101_configs/models/univ_model
```

> **Note**: Harvey will provide the `repo_id` of the trained model after training is complete. Just replace the `<repo_id>` with the one provided.

## 6. Configure Policy Path

📁 **Script Path**
```
/home/ubuntu/lerobot/bi_so101_configs/run_record_policy.sh
```

Open the script and modify the `policy` path to point to your downloaded model.

📝 **Example**
```bash
policy="/home/ubuntu/lerobot/bi_so101_configs/models/univ_model"
```

## 7. Run Inference

▶️ **Run**
```bash
cd lerobot && ./bi_so101_configs/run_record_policy.sh
```

The robot will now execute the learned policy autonomously.

---

## Troubleshooting

### Port Permission Error

▶️ **Run**
```bash
chacm
```

### Model Not Found Error

Verify that:
1. The model was downloaded correctly
2. The policy path in `run_record_policy.sh` matches the downloaded model location
