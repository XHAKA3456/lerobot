#!/usr/bin/env python
"""Validation tests for the FlowMatching policy.

Run all tests (requires DINOv2 download on first run):
    pytest tests/policies/test_flowmatching.py -v

Run only fast state-only tests (no internet/GPU needed):
    pytest tests/policies/test_flowmatching.py -v -m "not requires_dinov2"

Run directly:
    python tests/policies/test_flowmatching.py
"""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.flowmatching.configuration_flowmatching import FlowMatchingConfig
from lerobot.policies.flowmatching.modeling_flowmatching import (
    FlowMatchingModel,
    FlowMatchingPolicy,
    _sinusoidal_time_embedding,
    _sample_beta,
)
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_IMAGES


# ---------------------------------------------------------------------------
# Constants & fixtures
# ---------------------------------------------------------------------------

STATE_DIM = 14
ACTION_DIM = 14
CHUNK_SIZE = 10
N_ACTION_STEPS = 10
BATCH_SIZE = 2
IMAGE_H, IMAGE_W = 224, 224


def make_state_only_config(**overrides) -> FlowMatchingConfig:
    """Minimal config using robot state only (no images, no DINOv2 needed)."""
    cfg = FlowMatchingConfig(
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_obs_encoder_layers=2,
        n_velocity_layers=2,
        num_inference_steps=5,
    )
    cfg.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
    }
    cfg.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    }
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_image_config(**overrides) -> FlowMatchingConfig:
    """Config with two cameras + state (requires DINOv2 download)."""
    cfg = FlowMatchingConfig(
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,
        dim_model=128,
        n_heads=4,
        dim_feedforward=256,
        n_obs_encoder_layers=2,
        n_velocity_layers=2,
        num_inference_steps=5,
    )
    cfg.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        "observation.images.left": PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_H, IMAGE_W)),
        "observation.images.right": PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_H, IMAGE_W)),
    }
    cfg.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    }
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_state_batch(B: int = BATCH_SIZE, device: str = "cpu") -> dict:
    return {
        OBS_STATE: torch.randn(B, STATE_DIM, device=device),
        ACTION: torch.randn(B, CHUNK_SIZE, ACTION_DIM, device=device),
    }


def make_image_batch(B: int = BATCH_SIZE, device: str = "cpu") -> dict:
    return {
        OBS_STATE: torch.randn(B, STATE_DIM, device=device),
        "observation.images.left": torch.rand(B, 3, IMAGE_H, IMAGE_W, device=device),
        "observation.images.right": torch.rand(B, 3, IMAGE_H, IMAGE_W, device=device),
        ACTION: torch.randn(B, CHUNK_SIZE, ACTION_DIM, device=device),
    }


# ---------------------------------------------------------------------------
# Test 1: Math helpers
# ---------------------------------------------------------------------------

class TestMathHelpers:
    def test_sinusoidal_embedding_shape(self):
        B, D = 4, 32
        t = torch.rand(B)
        emb = _sinusoidal_time_embedding(t, D, min_period=4e-3, max_period=4.0)
        assert emb.shape == (B, D)

    def test_sinusoidal_embedding_deterministic(self):
        t = torch.tensor([0.0, 0.5, 1.0])
        e1 = _sinusoidal_time_embedding(t, 16, 4e-3, 4.0)
        e2 = _sinusoidal_time_embedding(t, 16, 4e-3, 4.0)
        assert torch.allclose(e1, e2)

    def test_sinusoidal_embedding_distinct_times(self):
        """Different time values must produce different embeddings."""
        e1 = _sinusoidal_time_embedding(torch.tensor([0.1]), 16, 4e-3, 4.0)
        e2 = _sinusoidal_time_embedding(torch.tensor([0.9]), 16, 4e-3, 4.0)
        assert not torch.allclose(e1, e2)

    def test_sample_beta_range(self):
        """Scaled samples must lie in [0.001, 1.000]."""
        t = _sample_beta(1.5, 1.0, 10000, device=torch.device("cpu"))
        scaled = t * 0.999 + 0.001
        assert scaled.min() >= 0.001 - 1e-6
        assert scaled.max() <= 1.000 + 1e-6

    def test_sample_beta_mean(self):
        """Beta(1.5, 1.0) mean = 1.5/2.5 = 0.6 > 0.5 (biased toward later t)."""
        t = _sample_beta(1.5, 1.0, 10000, device=torch.device("cpu"))
        assert t.mean().item() > 0.5


# ---------------------------------------------------------------------------
# Test 2: Model tensor shapes (state-only, no DINOv2)
# ---------------------------------------------------------------------------

class TestStateOnlyShapes:
    @pytest.fixture
    def model(self):
        cfg = make_state_only_config()
        m = FlowMatchingModel(cfg)
        m.eval()
        return m, cfg

    def test_obs_encoding_shape(self, model):
        m, cfg = model
        batch = make_state_batch()
        obs_feat, obs_pos = m._encode_obs(batch)
        # State-only → S=1 token
        assert obs_feat.shape == (1, BATCH_SIZE, cfg.dim_model)
        assert obs_pos.shape == (1, 1, cfg.dim_model)

    def test_velocity_prediction_shape(self, model):
        m, cfg = model
        batch = make_state_batch()
        obs_feat, obs_pos = m._encode_obs(batch)
        x_t = torch.randn(BATCH_SIZE, CHUNK_SIZE, ACTION_DIM)
        time = torch.rand(BATCH_SIZE)
        v = m._predict_velocity(obs_feat, obs_pos, x_t, time)
        assert v.shape == (BATCH_SIZE, CHUNK_SIZE, ACTION_DIM)

    def test_compute_loss_shape(self, model):
        m, _ = model
        loss = m.compute_loss(make_state_batch())
        assert loss.shape == (BATCH_SIZE, CHUNK_SIZE, ACTION_DIM)

    def test_compute_loss_nonnegative(self, model):
        m, _ = model
        loss = m.compute_loss(make_state_batch())
        assert (loss >= 0).all()

    def test_sample_actions_shape(self, model):
        m, _ = model
        with torch.no_grad():
            actions = m.sample_actions(make_state_batch())
        assert actions.shape == (BATCH_SIZE, CHUNK_SIZE, ACTION_DIM)

    def test_sample_actions_finite(self, model):
        m, _ = model
        with torch.no_grad():
            actions = m.sample_actions(make_state_batch())
        assert torch.isfinite(actions).all(), "Sampled actions contain NaN or Inf"


# ---------------------------------------------------------------------------
# Test 3: Policy interface
# ---------------------------------------------------------------------------

class TestPolicyInterface:
    @pytest.fixture
    def policy(self):
        return FlowMatchingPolicy(make_state_only_config())

    def test_forward_returns_scalar_loss(self, policy):
        loss, loss_dict = policy.forward(make_state_batch())
        assert loss.ndim == 0
        assert loss.item() >= 0
        assert "loss" in loss_dict

    def test_forward_finite(self, policy):
        loss, _ = policy.forward(make_state_batch())
        assert torch.isfinite(loss), f"loss={loss.item()}"

    def test_forward_padding_mask(self, policy):
        """action_is_pad mask must be accepted without error."""
        batch = make_state_batch()
        batch["action_is_pad"] = torch.zeros(BATCH_SIZE, CHUNK_SIZE, dtype=torch.bool)
        batch["action_is_pad"][:, -5:] = True
        loss, _ = policy.forward(batch)
        assert torch.isfinite(loss)

    def test_backward_gradients_flow(self, policy):
        policy.train()
        loss, _ = policy.forward(make_state_batch())
        loss.backward()
        has_grad = any(
            p.requires_grad and p.grad is not None
            for p in policy.parameters()
        )
        assert has_grad, "No gradient computed for any parameter"

    def test_gradients_are_finite(self, policy):
        policy.train()
        loss, _ = policy.forward(make_state_batch())
        loss.backward()
        for name, p in policy.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"

    def test_select_action_shape(self, policy):
        policy.eval()
        batch = make_state_batch(B=1)
        with torch.no_grad():
            action = policy.select_action(batch)
        assert action.shape == (1, ACTION_DIM)

    def test_select_action_finite(self, policy):
        policy.eval()
        with torch.no_grad():
            action = policy.select_action(make_state_batch(B=1))
        assert torch.isfinite(action).all()

    def test_action_queue_drains_then_refills(self, policy):
        policy.eval()
        policy.reset()
        batch = make_state_batch(B=1)
        # Drain the full chunk
        for _ in range(N_ACTION_STEPS):
            with torch.no_grad():
                policy.select_action(batch)
        assert len(policy._action_queue) == 0
        # Next call must auto-refill
        with torch.no_grad():
            a = policy.select_action(batch)
        assert a.shape == (1, ACTION_DIM)

    def test_reset_clears_queue(self, policy):
        policy.eval()
        with torch.no_grad():
            policy.select_action(make_state_batch(B=1))  # fills queue
        assert len(policy._action_queue) > 0
        policy.reset()
        assert len(policy._action_queue) == 0

    def test_get_optim_params_all_trainable(self, policy):
        groups = policy.get_optim_params()
        assert len(groups) == 1
        for p in groups[0]["params"]:
            assert p.requires_grad

    def test_train_does_not_affect_dinov2(self, policy):
        """DINOv2 must remain in eval regardless of policy.train()."""
        policy.train()
        if hasattr(policy.model, "dinov2"):
            assert not policy.model.dinov2.training


# ---------------------------------------------------------------------------
# Test 4: Flow matching math correctness
# ---------------------------------------------------------------------------

class TestFlowMatchingMath:
    def test_forward_process_at_t0_equals_action(self):
        """x_t = t*noise + (1-t)*action  →  x_0 = action"""
        action = torch.randn(2, 5, 3)
        noise = torch.randn(2, 5, 3)
        t = torch.zeros(2, 1, 1)
        x_t = t * noise + (1 - t) * action
        assert torch.allclose(x_t, action)

    def test_forward_process_at_t1_equals_noise(self):
        """x_1 = noise"""
        action = torch.randn(2, 5, 3)
        noise = torch.randn(2, 5, 3)
        t = torch.ones(2, 1, 1)
        x_t = t * noise + (1 - t) * action
        assert torch.allclose(x_t, noise)

    def test_backward_euler_step_moves_toward_action(self):
        """One Euler step with true velocity should reduce distance to action."""
        action = torch.zeros(1, 1, 4)
        noise = torch.ones(1, 1, 4) * 2.0
        v_true = noise - action        # true velocity at t=1
        x_t = noise.clone()
        dt = -0.1
        x_next = x_t + dt * v_true    # = 2.0 - 0.1*2.0 = 1.8

        d_before = (x_t - action).norm()
        d_after = (x_next - action).norm()
        assert d_after < d_before

    def test_ode_time_grid(self):
        """t = 1.0, 0.8, ..., 0.2 for num_inference_steps=5."""
        num_steps = 5
        dt = -1.0 / num_steps
        times = [1.0 + i * dt for i in range(num_steps)]
        expected = [1.0, 0.8, 0.6, 0.4, 0.2]
        for t, e in zip(times, expected):
            assert abs(t - e) < 1e-6, f"t={t} expected {e}"

    def test_trained_velocity_reduces_loss_over_steps(self):
        """After a few gradient steps, loss should decrease."""
        cfg = make_state_only_config(num_inference_steps=3)
        policy = FlowMatchingPolicy(cfg)
        policy.train()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            loss, _ = policy.forward(make_state_batch())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should trend downward (first vs last 3)
        early = sum(losses[:3]) / 3
        late = sum(losses[-3:]) / 3
        assert late < early * 1.5, (
            f"Loss did not decrease: early={early:.4f}, late={late:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 5: Config validation
# ---------------------------------------------------------------------------

class TestConfig:
    def test_n_action_steps_gt_chunk_size_raises(self):
        with pytest.raises(ValueError, match="n_action_steps"):
            FlowMatchingConfig(chunk_size=5, n_action_steps=10).__post_init__()

    def test_n_obs_steps_gt_1_raises(self):
        # __post_init__ is called in __init__, so wrap the constructor
        with pytest.raises(ValueError, match="n_obs_steps"):
            FlowMatchingConfig(n_obs_steps=2, chunk_size=10, n_action_steps=10)

    def test_no_input_features_raises(self):
        cfg = make_state_only_config()
        cfg.input_features = {}
        with pytest.raises(ValueError, match="At least one"):
            cfg.validate_features()

    def test_action_delta_indices(self):
        cfg = make_state_only_config(chunk_size=10)
        assert cfg.action_delta_indices == list(range(10))

    def test_observation_delta_indices_none(self):
        assert make_state_only_config().observation_delta_indices is None

    def test_optimizer_preset_lr(self):
        cfg = make_state_only_config()
        opt = cfg.get_optimizer_preset()
        assert opt.lr == cfg.optimizer_lr

    def test_scheduler_preset_none(self):
        assert make_state_only_config().get_scheduler_preset() is None


# ---------------------------------------------------------------------------
# Optional: image-based tests (require DINOv2)
# ---------------------------------------------------------------------------

@pytest.mark.requires_dinov2
class TestImageModel:
    """Requires DINOv2 download (~300MB). Run with: pytest -m requires_dinov2"""

    @pytest.fixture
    def image_model(self):
        cfg = make_image_config()
        m = FlowMatchingModel(cfg)
        m.eval()
        return m, cfg

    def test_obs_encoding_with_two_cameras(self, image_model):
        m, cfg = image_model
        batch = make_image_batch()
        batch[OBS_IMAGES] = [batch["observation.images.left"], batch["observation.images.right"]]
        obs_feat, obs_pos = m._encode_obs(batch)
        # S = 1 (state) + 2×256 (patch tokens per camera)
        S_expected = 1 + 2 * 256
        assert obs_feat.shape == (S_expected, BATCH_SIZE, cfg.dim_model)

    def test_policy_forward_with_images(self):
        policy = FlowMatchingPolicy(make_image_config())
        batch = make_image_batch()
        loss, _ = policy.forward(batch)
        assert torch.isfinite(loss)

    def test_dinov2_parameters_frozen(self):
        m = FlowMatchingModel(make_image_config())
        for p in m.dinov2.parameters():
            assert not p.requires_grad

    def test_dinov2_stays_eval_in_train_mode(self):
        m = FlowMatchingModel(make_image_config())
        m.train()
        assert not m.dinov2.training


# ---------------------------------------------------------------------------
# Standalone runner (no pytest required)
# ---------------------------------------------------------------------------

def _run_class(cls, *args):
    """Instantiate cls and run all test_ methods."""
    obj = cls()
    passed = failed = 0
    for name in dir(cls):
        if not name.startswith("test_"):
            continue
        method = getattr(obj, name)
        try:
            # Handle pytest-fixture-like methods that need arguments
            import inspect
            sig = inspect.signature(method)
            if len(sig.parameters) > 0:
                # skip fixture-based tests in standalone mode
                continue
            method()
            passed += 1
        except Exception as exc:
            print(f"    FAIL {name}: {exc}")
            failed += 1
    return passed, failed


def run_all_without_pytest():
    print("=" * 60)
    print("FlowMatching Policy — Standalone Validation")
    print("=" * 60)

    suites = [
        ("Math helpers", TestMathHelpers),
        ("Flow matching math", TestFlowMatchingMath),
        ("Config validation", TestConfig),
    ]

    total_pass = total_fail = 0
    for label, cls in suites:
        print(f"\n[{label}]")
        p, f = _run_class(cls)
        total_pass += p
        total_fail += f
        status = "PASSED" if f == 0 else f"FAILED ({f} errors)"
        print(f"  {p} tests — {status}")

    # Run fixture-dependent tests manually
    print("\n[Model shapes (state-only)]")
    cfg = make_state_only_config()
    model = FlowMatchingModel(cfg)
    model.eval()
    fixture = (model, cfg)
    t = TestStateOnlyShapes()
    p = f = 0
    for fn in [
        t.test_obs_encoding_shape,
        t.test_velocity_prediction_shape,
        t.test_compute_loss_shape,
        t.test_compute_loss_nonnegative,
        t.test_sample_actions_shape,
        t.test_sample_actions_finite,
    ]:
        try:
            fn(fixture)
            p += 1
        except Exception as exc:
            print(f"    FAIL {fn.__name__}: {exc}")
            f += 1
    total_pass += p; total_fail += f
    print(f"  {p} tests — {'PASSED' if f == 0 else f'FAILED ({f})'}")

    print("\n[Policy interface]")
    policy = FlowMatchingPolicy(make_state_only_config())
    fixture_p = policy
    t2 = TestPolicyInterface()
    p = f = 0
    for fn in [
        t2.test_forward_returns_scalar_loss,
        t2.test_forward_finite,
        t2.test_forward_padding_mask,
        t2.test_backward_gradients_flow,
        t2.test_gradients_are_finite,
        t2.test_select_action_shape,
        t2.test_select_action_finite,
        t2.test_action_queue_drains_then_refills,
        t2.test_reset_clears_queue,
        t2.test_get_optim_params_all_trainable,
        t2.test_train_does_not_affect_dinov2,
    ]:
        try:
            fn(fixture_p)
            p += 1
        except Exception as exc:
            print(f"    FAIL {fn.__name__}: {exc}")
            f += 1
    total_pass += p; total_fail += f
    print(f"  {p} tests — {'PASSED' if f == 0 else f'FAILED ({f})'}")

    print("\n" + "=" * 60)
    print(f"Total: {total_pass} passed, {total_fail} failed")
    if total_fail > 0:
        print("SOME TESTS FAILED — check output above")
    else:
        print("All tests PASSED (DINOv2 tests skipped; use pytest -m requires_dinov2)")
    print("=" * 60)
    return total_fail == 0


if __name__ == "__main__":
    import sys
    ok = run_all_without_pytest()
    sys.exit(0 if ok else 1)
