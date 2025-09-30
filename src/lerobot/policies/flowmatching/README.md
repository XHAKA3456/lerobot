# Flow Matching Policy

A robot learning policy that uses conditional flow matching for action generation from visual and proprioceptive observations.

## Overview

Flow matching provides a principled approach to generating action sequences by learning the velocity field of a continuous normalizing flow. Unlike diffusion-based methods, flow matching directly learns deterministic paths from noise to actions, enabling efficient sampling with few integration steps.

### Key Features

- **Conditional Flow Matching**: Learns velocity field conditioned on observations
- **Vision Transformer**: Pretrained ViT for robust visual feature extraction
- **Efficient Sampling**: 10 Euler integration steps (vs 100+ for diffusion)
- **Multi-camera Support**: Processes multiple camera views simultaneously

## Architecture

```
Observations (Images + State)
         ↓
    ┌────────────────────────────┐
    │   Vision Encoder (ViT)     │  Images → Visual features
    └────────────────────────────┘
         ↓
    ┌────────────────────────────┐
    │   State Encoder (MLP)      │  Robot state → State features
    └────────────────────────────┘
         ↓
    ┌────────────────────────────┐
    │  Conditioning Network      │  Fuse vision + state
    └────────────────────────────┘
         ↓
    ┌────────────────────────────┐
    │  Denoising Network         │  6-layer Transformer
    │  (Transformer Encoder)     │  + Time embedding
    └────────────────────────────┘
         ↓
    Action Sequence (chunk_size steps)
```

## Configuration

Key hyperparameters:

```python
# Vision
vision_backbone: str = "vit_b_16"      # ViT-B/16 (86M params)
vit_embed_dim: int = 768               # ViT feature dimension
image_size: tuple = (224, 224)         # Input resolution

# Model architecture
hidden_dim: int = 1024                 # Hidden feature dimension
max_state_dim: int = 32                # Max state dimension (with padding)
max_action_dim: int = 32               # Max action dimension (with padding)

# Flow matching
num_steps: int = 10                    # Euler integration steps
n_action_steps: int = 50               # Action chunk size

# Training
optimizer_lr: float = 1e-4
batch_size: int = 32
```

## Usage

### Training

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/your_dataset \
  --policy.type=flowmatching \
  --output_dir=outputs/train/flowmatching \
  --job_name=flowmatching_experiment \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/my_flowmatching_policy
```

### Inference

```python
from lerobot.policies.flowmatching import FlowMatchingPolicy

# Load trained policy
policy = FlowMatchingPolicy.from_pretrained("path/to/checkpoint")

# Generate actions
observation = {
    "observation.state": state_tensor,
    "observation.images.cam_1": image_tensor,
}
action = policy.select_action(observation)
```

### Custom Configuration

```python
from lerobot.policies.flowmatching import FlowMatchingConfig, FlowMatchingPolicy

config = FlowMatchingConfig(
    n_action_steps=50,
    num_steps=10,  # Integration steps
    hidden_dim=1024,
    vision_backbone="vit_b_16",
)
policy = FlowMatchingPolicy(config)
```

## Flow Matching Details

### Training

The policy learns a velocity field `v_θ(x_t, t, c)` where:
- `x_t = t·ε + (1-t)·x_1`: Interpolated trajectory
- `ε ~ N(0, I)`: Noise
- `x_1`: Target actions
- `c`: Conditioning (vision + state)
- `t ~ Beta(1.5, 1.0)`: Time

**Loss**: `L = E[||v_θ(x_t, t, c) - (ε - x_1)||²]`

### Inference

Generate actions via Euler integration:

```python
x_t = ε  # Start from noise
t = 1.0
dt = -1.0 / num_steps

while t >= 0:
    v_t = model(x_t, t, observations)
    x_t = x_t + dt * v_t
    t = t + dt

actions = x_t  # Final denoised actions
```

## Input/Output Interface

### Input
- `observation.state`: Robot joint positions/velocities (proprioception)
- `observation.images.*`: Camera images, RGB format, [0,1] range

### Output
- Action tensor: `(batch_size, n_action_steps, action_dim)`
- Actions are automatically queued and executed step-by-step

## Requirements

```bash
pip install torch torchvision
# LeRobot will be installed with all dependencies
```

## Performance

- **Model size**: ~100M parameters (ViT-B: 86M + heads: ~14M)
- **Inference speed**: ~50ms per action chunk (on RTX 3090)
- **Memory**: ~2GB GPU memory during inference

## Citation

If you use this policy, please cite:

```bibtex
@article{lipman2022flow,
  title={Flow Matching for Generative Modeling},
  author={Lipman, Yaron and Chen, Ricky TQ and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matthew},
  journal={arXiv preprint arXiv:2210.02747},
  year={2022}
}
```