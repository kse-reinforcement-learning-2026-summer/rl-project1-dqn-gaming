"""
Automated tests for Project 1: DQN — Playing Flappy Bird from Pixels.

Tests:
  1. test_make_env      — Environment produces correct observation shape/dtype
  2. test_make_dqn      — Network has correct architecture (params, I/O shapes)
  3. test_get_epsilon   — Linear decay schedule is correct
  4. test_model         — Saved model achieves reward >= threshold

Run:  cd <project_root> && pytest tests/test.py -v
"""

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
import flappy_bird_gymnasium  # noqa: F401 — registers FlappyBird-v0
import numpy as np
import pytest
import torch

from grader_utils import load_notebook_functions


# ---------------------------------------------------------------------------
# Reference environment (independent of student code)
# ---------------------------------------------------------------------------

class _RenderAsObsWrapper(gym.ObservationWrapper):
    """Replace numeric obs with rendered RGB frame."""
    def __init__(self, env):
        super().__init__(env)
        env.reset()
        frame = env.render()
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=frame.shape, dtype=np.uint8
        )

    def observation(self, obs):
        return self.env.render()


class _FrameSkipWrapper(gym.Wrapper):
    """Execute agent's action once, then idle (action=0) for (skip-1) frames."""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action if i == 0 else 0)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def _make_reference_env():
    """Ground-truth environment for model evaluation (not student code)."""
    env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
    env = _FrameSkipWrapper(env, skip=4)
    env = _RenderAsObsWrapper(env)
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.ResizeObservation(env, shape=(84, 84))
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NOTEBOOK_PATH = "project.ipynb"  # student's submitted notebook
MODEL_PATH = "model.pt"
REWARD_THRESHOLD = 30.0
EXPECTED_PARAMS = 676_658
N_ACTIONS = 2

# ---------------------------------------------------------------------------
# Load student code from notebook
# ---------------------------------------------------------------------------

hw = load_notebook_functions(NOTEBOOK_PATH)


# ===========================================================================
# Test 1: make_env()
# ===========================================================================
class TestMakeEnv:
    """Verify the preprocessing pipeline produces correct observations."""

    def test_function_exists(self):
        assert hasattr(hw, "make_env"), "Function make_env() not found in notebook"

    def test_observation_shape(self):
        env = hw.make_env()
        obs, _ = env.reset(seed=42)
        assert obs.shape == (4, 84, 84), (
            f"Expected observation shape (4, 84, 84), got {obs.shape}"
        )
        env.close()

    def test_observation_dtype(self):
        env = hw.make_env()
        obs, _ = env.reset(seed=42)
        assert obs.dtype == np.uint8, (
            f"Expected dtype uint8, got {obs.dtype}"
        )
        env.close()

    def test_action_space(self):
        env = hw.make_env()
        assert env.action_space.n == 2, (
            f"Expected 2 actions, got {env.action_space.n}"
        )
        env.close()

    def test_step_consistency(self):
        env = hw.make_env()
        obs, _ = env.reset(seed=42)
        obs2, reward, terminated, truncated, info = env.step(0)
        assert obs2.shape == (4, 84, 84), (
            f"Shape changed after step: {obs2.shape}"
        )
        env.close()


# ===========================================================================
# Test 2: make_dqn()
# ===========================================================================
class TestMakeDQN:
    """Verify DQN architecture matches the 2013 paper."""

    def test_function_exists(self):
        assert hasattr(hw, "make_dqn"), "Function make_dqn() not found in notebook"

    def test_parameter_count(self):
        net = hw.make_dqn(N_ACTIONS)
        total_params = sum(p.numel() for p in net.parameters())
        assert total_params == EXPECTED_PARAMS, (
            f"Expected {EXPECTED_PARAMS:,} params, got {total_params:,}"
        )

    def test_output_shape(self):
        net = hw.make_dqn(N_ACTIONS)
        dummy = torch.zeros(1, 4, 84, 84)
        out = net(dummy)
        assert out.shape == (1, N_ACTIONS), (
            f"Expected output shape (1, {N_ACTIONS}), got {out.shape}"
        )

    def test_batch_forward(self):
        net = hw.make_dqn(N_ACTIONS)
        dummy = torch.zeros(8, 4, 84, 84)
        out = net(dummy)
        assert out.shape == (8, N_ACTIONS), (
            f"Expected output shape (8, {N_ACTIONS}), got {out.shape}"
        )


# ===========================================================================
# Test 3: get_epsilon()
# ===========================================================================
class TestGetEpsilon:
    """Verify epsilon decay schedule."""

    def test_function_exists(self):
        assert hasattr(hw, "get_epsilon"), "Function get_epsilon() not found in notebook"

    def test_start_value(self):
        eps = hw.get_epsilon(0, 0.8, 0.05, 50_000)
        assert abs(eps - 0.8) < 1e-9, f"Expected eps=eps_start at step 0, got {eps}"

    def test_end_value(self):
        eps = hw.get_epsilon(50_000, 0.8, 0.05, 50_000)
        assert abs(eps - 0.05) < 1e-9, f"Expected eps=eps_end at decay_steps, got {eps}"

    def test_clamped_after_decay(self):
        eps = hw.get_epsilon(100_000, 0.8, 0.05, 50_000)
        assert abs(eps - 0.05) < 1e-9, (
            f"Expected eps=eps_end after decay_steps, got {eps}"
        )

    def test_midpoint(self):
        eps = hw.get_epsilon(25_000, 0.8, 0.05, 50_000)
        expected = 0.425
        assert abs(eps - expected) < 1e-6, (
            f"Expected eps≈{expected} at midpoint, got {eps}"
        )

    def test_monotonic_decrease(self):
        prev = hw.get_epsilon(0, 0.8, 0.05, 50_000)
        for step in range(1000, 50_000, 1000):
            curr = hw.get_epsilon(step, 0.8, 0.05, 50_000)
            assert curr <= prev, (
                f"Epsilon increased at step {step}: {prev} → {curr}"
            )
            prev = curr


# ===========================================================================
# Test 4: Hyperparameters (from paper)
# ===========================================================================
class TestHyperparameters:
    """Verify student selected correct hyperparameters from the DQN paper.

    We check arithmetic combinations of three hyperparameters per test
    to avoid students simply reading expected values from test code.
    """

    def test_hyperparams_exist(self):
        for name in ("BATCH_SIZE", "EPS_START", "EPS_END",
                     "TRAINING_STEPS", "REPLAY_MEMORY_SIZE", "EPS_DECAY_STEPS"):
            assert hasattr(hw, name), f"{name} not found in notebook"

    def test_combo_batch_eps(self):
        """Three-way check: BATCH_SIZE, EPS_START, EPS_END."""
        val = hw.BATCH_SIZE * (hw.EPS_START + hw.EPS_END)
        assert abs(val - 35.2) < 1e-6, (
            f"BATCH_SIZE * (EPS_START + EPS_END) = {val}. "
            f"Check the paper for correct values."
        )

    def test_combo_batch_training_replay(self):
        """Three-way check: BATCH_SIZE, TRAINING_STEPS, REPLAY_MEMORY_SIZE."""
        val = hw.BATCH_SIZE * hw.TRAINING_STEPS / hw.REPLAY_MEMORY_SIZE
        assert abs(val - 160.0) < 1e-6, (
            f"BATCH_SIZE * TRAINING_STEPS / REPLAY_MEMORY_SIZE = {val}. "
            f"Check the paper for correct values."
        )

    def test_combo_eps_start_training_decay(self):
        """Three-way check: EPS_START, TRAINING_STEPS, EPS_DECAY_STEPS."""
        val = hw.EPS_START * hw.TRAINING_STEPS / hw.EPS_DECAY_STEPS
        assert abs(val - 5.0) < 1e-6, (
            f"EPS_START * TRAINING_STEPS / EPS_DECAY_STEPS = {val}. "
            f"Check the paper for correct values."
        )

    def test_combo_batch_replay_decay(self):
        """Three-way check: BATCH_SIZE, REPLAY_MEMORY_SIZE, EPS_DECAY_STEPS."""
        val = hw.BATCH_SIZE + hw.REPLAY_MEMORY_SIZE / hw.EPS_DECAY_STEPS
        assert abs(val - 33.0) < 1e-6, (
            f"BATCH_SIZE + REPLAY_MEMORY_SIZE / EPS_DECAY_STEPS = {val}. "
            f"Check the paper for correct values."
        )

    def test_combo_eps_end_training_decay(self):
        """Three-way check: EPS_END, TRAINING_STEPS, EPS_DECAY_STEPS."""
        val = hw.EPS_END + hw.TRAINING_STEPS / hw.EPS_DECAY_STEPS
        assert abs(val - 5.1) < 1e-6, (
            f"EPS_END + TRAINING_STEPS / EPS_DECAY_STEPS = {val}. "
            f"Check the paper for correct values."
        )

    def test_combo_training_replay_decay(self):
        """Three-way check: TRAINING_STEPS, REPLAY_MEMORY_SIZE, EPS_DECAY_STEPS."""
        val = hw.TRAINING_STEPS / hw.REPLAY_MEMORY_SIZE + hw.TRAINING_STEPS / hw.EPS_DECAY_STEPS
        assert abs(val - 10.0) < 1e-6, (
            f"TRAINING_STEPS / REPLAY_MEMORY_SIZE + TRAINING_STEPS / EPS_DECAY_STEPS = {val}. "
            f"Check the paper for correct values."
        )

    def test_decay_before_training_ends(self):
        """Epsilon decay must complete before training ends."""
        assert hw.EPS_DECAY_STEPS <= hw.TRAINING_STEPS, (
            f"EPS_DECAY_STEPS ({hw.EPS_DECAY_STEPS}) > TRAINING_STEPS ({hw.TRAINING_STEPS}). "
            f"Epsilon decay should complete before training ends."
        )

    def test_optimizer_type(self):
        """The paper uses a specific optimizer."""
        import nbformat
        with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        source = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
        rmsprop_lines = [
            line for line in source.split("\n")
            if "RMSprop" in line and not line.strip().startswith("#")
        ]
        assert len(rmsprop_lines) > 0, (
            "Could not find the correct optimizer in notebook code. "
            "Check the paper for the optimizer used."
        )

    def test_loss_function(self):
        """DQN uses a specific loss on TD error."""
        import nbformat
        with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        source = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
        mse_lines = [
            line for line in source.split("\n")
            if "MSELoss" in line and not line.strip().startswith("#")
        ]
        assert len(mse_lines) > 0, (
            "Could not find the correct loss function in notebook code. "
            "Check the paper for the loss used on TD error."
        )


# ===========================================================================
# Test 5: Model performance
# ===========================================================================
class TestModelPerformance:
    """Load saved model and evaluate in the environment."""

    def test_model_file_exists(self):
        assert os.path.isfile(MODEL_PATH), (
            f"Model file not found at {MODEL_PATH}"
        )

    def test_model_loads_into_architecture(self):
        """Verify model.pt is compatible with student's make_dqn()."""
        net = hw.make_dqn(N_ACTIONS)
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        net.load_state_dict(state_dict)  # will raise if architecture mismatches

    def test_reward_threshold(self):
        """Run 30 deterministic episodes and check mean reward >= threshold."""
        import gymnasium as gym
        import flappy_bird_gymnasium  # noqa: F401

        # Load model
        net = hw.make_dqn(N_ACTIONS)
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        net.load_state_dict(state_dict)
        net.eval()

        device = torch.device("cpu")
        net = net.to(device)

        # Evaluate over 30 episodes with different seeds
        n_episodes = 30
        episode_rewards = []

        for ep in range(n_episodes):
            env = _make_reference_env()
            obs, _ = env.reset(seed=ep)

            episode_reward = 0.0
            done = False

            while not done:
                with torch.no_grad():
                    state_t = torch.tensor(
                        np.array(obs), dtype=torch.float32
                    ).unsqueeze(0).to(device)
                    action = net(state_t).argmax(dim=1).item()

                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            env.close()
            episode_rewards.append(episode_reward)

        median_reward = np.median(episode_rewards)
        assert median_reward >= REWARD_THRESHOLD, (
            f"Model median reward {median_reward:.2f} over {n_episodes} episodes "
            f"< threshold {REWARD_THRESHOLD}. "
            f"(mean={np.mean(episode_rewards):.2f}, "
            f"std={np.std(episode_rewards):.2f}, "
            f"min={np.min(episode_rewards):.2f}, "
            f"max={np.max(episode_rewards):.2f})"
        )
