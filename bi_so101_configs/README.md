# BI-SO101 Bimanual Robot — Full Pipeline Guide

End-to-end guide for the **BI-SO101** dual-arm robot: teleoperation, dataset recording, training, and inference using [LeRobot](https://github.com/huggingface/lerobot).

All configuration files and calibration data are in [`bi_so101_configs/`](https://github.com/XHAKA3456/lerobot/tree/dual-arm-xlerobot/bi_so101_configs).

---

## Table of Contents

1. [Installation](#installation)
2. [Hardware Setup](#hardware-setup)
3. [Teleoperation](#teleoperation)
4. [Dataset Recording](#dataset-recording)
5. [Training](#training)
6. [Inference](#inference)
7. [Tips for Good Data](#tips-for-good-data)
8. [Troubleshooting](#troubleshooting)

---

## Installation

```bash
# Clone the dual-arm-xlerobot branch directly
git clone -b dual-arm-xlerobot https://github.com/XHAKA3456/lerobot.git
cd lerobot

# Create conda environment
conda create -n lerobot python=3.10 -y
conda activate lerobot

# Install with feetech motor support
pip install -e ".[feetech]"
```

> **Hugging Face login** (required for dataset upload and model download):
> ```bash
> huggingface-cli login
> ```

---

## Hardware Setup

### Robot Overview

| Role | Device | Ports |
|------|--------|-------|
| **Follower** (performs the task) | BI-SO101 left arm | `/dev/ttyACM0` |
| **Follower** | BI-SO101 right arm | `/dev/ttyACM1` |
| **Leader** (human-operated) | BI-SO101 left arm | `/dev/ttyACM2` |
| **Leader** | BI-SO101 right arm | `/dev/ttyACM3` |

### Camera Setup

| Camera | Device | Connection |
|--------|--------|------------|
| Left | `/dev/video2` | First USB hub |
| Right | `/dev/video4` | Second USB hub |
| Front | `/dev/video6` | First USB hub (last to connect) |

### Connection Order — Example

> **This is an example configuration** using 2 leader arms, 2 follower arms, and 3 cameras. Adapt the port assignments to match your own setup.

The OS assigns `/dev/ttyACM*` and `/dev/video*` sequentially based on connection order. Connect devices in a consistent order each time so port assignments stay predictable.

```
Example setup:

First USB Hub:
  1. Follower left arm   → /dev/ttyACM0
  2. Follower right arm  → /dev/ttyACM1
  3. Left camera         → /dev/video2

Second USB Hub:
  4. Leader left arm     → /dev/ttyACM2
  5. Leader right arm    → /dev/ttyACM3
  6. Right camera        → /dev/video4

First USB Hub (last):
  7. Front camera        → /dev/video6
```

Update the `port` and `index_or_path` fields in the config files to match whatever your system assigns.

### USB Permissions

After connecting all devices, grant read/write access to each port:

```bash
# Grant permission to all serial and video devices at once
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1 /dev/ttyACM2 /dev/ttyACM3
sudo chmod 666 /dev/video2 /dev/video4 /dev/video6
```

Run `ls /dev/ttyACM*` and `ls /dev/video*` first to confirm which devices are actually present, then chmod only those. No need to replug anything — just run chmod against the correct device nodes.

### Calibration

Calibration files are already included in this repository:

```
bi_so101_configs/calibration/
├── bi_so101_follower/
│   ├── black_left.json
│   └── black_right.json
└── bi_so101_leader/
    ├── black_left.json
    └── black_right.json
```

No re-calibration needed unless you replace motors.

---

## Teleoperation

Verify hardware is connected correctly and adjust camera angles before recording.

```bash
cd lerobot
./bi_so101_configs/run_teleoperate.sh
```

The follower arms will mirror the leader arms in real time. Camera feeds will be displayed on screen. Adjust camera positions until all three views are satisfactory.

**Config:** `bi_so101_configs/scripts/bi_so101_teleoperate.yaml`

---

## Dataset Recording

### 1. Edit the config file

**`bi_so101_configs/scripts/bi_so101_record.yaml`**

The minimum required changes are `repo_id` and `root`:

```yaml
dataset:
  repo_id: your_hf_username/your_dataset_name   # ← change this
  root: /path/to/your/datasets/your_dataset_name # ← change this
  single_task: "Describe the task here"
  episode_time_s: 20    # seconds per episode
  reset_time_s: 5       # reset window between episodes
  num_episodes: 30      # total episodes to collect

resume: false           # set true to continue an existing dataset
```

| Parameter | Description |
|-----------|-------------|
| `repo_id` | Hugging Face dataset ID (`username/name`) |
| `root` | Local directory to save the dataset |
| `episode_time_s` | Duration of each episode in seconds |
| `reset_time_s` | Time between episodes to reset the environment |
| `num_episodes` | Total number of episodes to record |
| `resume` | `true` to append to an existing dataset, `false` to start fresh |

### 2. Run

```bash
cd lerobot
./bi_so101_configs/run_record.sh
```

The dataset will automatically upload to Hugging Face Hub once all episodes are collected (~10 min for upload).

### Interrupted by Ctrl+C?

No problem — data is saved **locally** even if upload doesn't complete. Re-run the script and finish the remaining episodes; all local data will be uploaded together at the end.

### Manual upload

```bash
python3 bi_so101_configs/push_dataset.py \
    --repo_id "your_hf_username/your_dataset_name" \
    --root "/path/to/your/datasets/your_dataset_name"
```

---

## Training

> **⚠️ Training cannot be run on Rubik Pi or other edge devices.** A GPU server is required. Upload your dataset to Hugging Face Hub first, then run training on a machine with a CUDA-capable GPU.

Training uses the standard LeRobot training pipeline. Refer to the official documentation:

👉 **[LeRobot Training Guide](https://github.com/huggingface/lerobot?tab=readme-ov-file#train-your-own-policy)**

Example (ACT policy on a GPU server):

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=your_hf_username/your_dataset_name \
  --output_dir=outputs/train/your_run_name
```

The trained model will be saved to `output_dir` and can be pushed to Hugging Face Hub with `--hub_id` for later download during inference.

---

## Inference

### 1. Download the trained model

```bash
huggingface-cli download your_hf_username/your_model_name \
    --local-dir bi_so101_configs/models/your_model_name
```

### 2. Edit the config file

**`bi_so101_configs/scripts/bi_so101_infer.yaml`**

```yaml
policy:
  type: act
  pretrained_path: /path/to/bi_so101_configs/models/your_model_name  # ← change this
  device: cuda   # or "cpu"

dataset_repo_id: your_hf_username/your_dataset_name  # ← change this
```

### 3. Run

```bash
cd lerobot
./bi_so101_configs/run_infer.sh
```

The follower robot will execute the learned policy autonomously.

> **On Rubik Pi:** activate the environment first with `source ~/miniconda3/bin/activate lerobot`

### Record inference episodes (optional)

To record the robot's autonomous behavior as a dataset for evaluation:

```bash
./bi_so101_configs/run_record_policy.sh
```

Or with a custom policy:

```bash
./bi_so101_configs/run_record_policy.sh \
    --policy.path=your_hf_username/your_model_name \
    --policy.n_action_steps=50
```

---

## Tips for Good Data

### Set up a clean background

Point at least one camera toward a **plain wall** with no moving objects. Remove anything from the frame that might shift between episodes — it introduces noise the model can't reason about.

### Break motions into deliberate steps

Avoid rushing through the task. Decompose the movement into clear stages:

1. Move the gripper **above** the target object
2. Rotate to a comfortable grasping angle
3. Open the gripper fully
4. Lower and close around the object
5. Transport and release

Deliberate, step-by-step motions produce cleaner action trajectories and improve policy learning.

### Control lighting

Vision-based policies are sensitive to lighting. Shadows shifting between episodes, or the difference between morning and afternoon light, can degrade performance significantly. Record in a **consistent lighting environment**, and if possible avoid direct sunlight through windows.

### Vary object position every episode

Placing the object in the exact same spot every time leads to a policy that only works at that exact position. Shuffle the object's location and orientation each episode to build generalization.

---

## Troubleshooting

### Port permission denied
```bash
sudo chmod 666 /dev/ttyACM* /dev/video*
```

### Wrong device assigned to port
Disconnect all cables and reconnect **in order** from step 1 of [Hardware Setup](#hardware-setup).

### Dataset directory already exists
Either delete the local dataset directory, or set `resume: true` in the config to append to it.

### Model not found during inference
Verify the `pretrained_path` in `bi_so101_infer.yaml` matches the directory where the model was downloaded.

### Upload stuck / slow
Large datasets can take 10–30 minutes to upload. If it fails, use `push_dataset.py` for a manual retry.
