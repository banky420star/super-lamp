"""
End-to-end integration test: RegimeRoutedPPO.learn(10000).

Validates that the full training pipeline works end-to-end:
- Synthetic OHLCV data generation with regime structure
- Synthetic environment instantiation with regime observation
- RegimeRoutedPPO with RegimeRoutedActorCriticPolicy
- model.learn(10000) completes without errors
- Regime probabilities are non-uniform (meaningful decomposition)
- Actor and critic have independent regime decompositions
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest
from typing import Callable

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def make_synthetic_ohlcv(n_bars=1000, seed=42, regimes=3):
    """Generate synthetic OHLCV data with known regime structure."""
    np.random.seed(seed)
    idx = pd.date_range("2026-01-01", periods=n_bars, freq="5min", tz="UTC")
    price = np.zeros(n_bars)
    base = 100.0
    chunk = n_bars // max(regimes, 1)

    for r in range(regimes):
        start = r * chunk
        end = min(start + chunk, n_bars)
        sz = end - start
        for j in range(start, end):
            i = j - start
            if r == 0:  # Trending up
                price[j] = base * (1 + 0.0003 * i + 0.0015 * np.random.randn())
            elif r == 1:  # Ranging
                price[j] = base * (1 + 0.001 * chunk) + 0.003 * base * np.random.randn()
            else:  # Volatile
                price[j] = base * (1 + 0.001 * chunk) + 0.008 * base * np.sin(i * 0.05) + 0.005 * base * np.random.randn()
        if r == 0:
            base = price[end - 1] if end < n_bars else price[-1]

    df = pd.DataFrame({
        "open": price * (1 - 0.0004 * abs(np.random.randn(n_bars))),
        "high": price * (1 + 0.003 * abs(np.random.randn(n_bars))),
        "low":  price * (1 - 0.003 * abs(np.random.randn(n_bars))),
        "close": price,
        "volume": 100 + 50 * np.random.rand(n_bars),
        "tick_volume": (100 + 50 * np.random.rand(n_bars)).astype(int),
    }, index=idx)
    df.index.name = "time"
    return df


# ── Synthetic Environment ──────────────────────────────────────────────

import gymnasium as gym
from gymnasium import spaces


class SyntheticTradingEnv(gym.Env):
    """
    Minimal synthetic trading environment for integration testing.

    Uses a RegimeDetector to produce regime observations, simulates
    OHLCV-like observation vectors, and rewards based on next-bar
    price movement (proxy for trading).

    Performance: Regime observations are pre-computed once in __init__
    to avoid O(n^2) slowdown during training.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 100,
        num_regimes: int = 5,
    ):
        super().__init__()
        self.df = df
        self.window_size = window_size
        self.num_regimes = num_regimes
        self._step_idx = window_size
        self.regime_dim = 6  # 5 one-hot + confidence

        # Build per-bar feature vectors
        self._build_features()

        # Regime detector
        from drl.regime_detector import RegimeDetector
        self.regime_detector = RegimeDetector(use_patterns=False)
        self.regime_detector.fit_heuristic(df)

        # ── Pre-compute regime observations ──
        # This prevents O(n^2) slowdown from repeatedly calling
        # get_regime_observation on growing DataFrame slices
        n = len(df)
        self._regime_obs_cache = np.zeros((n, 1 + num_regimes), dtype=np.float32)
        for i in range(window_size, n):
            obs = self.regime_detector.get_regime_observation(df.iloc[:i])
            self._regime_obs_cache[i] = obs

        # Total observation dim: window * features_per_bar + portfolio_dim + regime_dim
        # NOTE: portfolio_state MUST come before regime_obs in the tail to match
        # AdaptiveLSTMFeatureExtractor's expectation:
        #   tail = observations[:, -portfolio_dim:]  (last portfolio_dim elements)
        #   regime = tail[:, -regime_dim:]  (last regime_dim of the tail)
        #   portfolio_state = tail[:, :-regime_dim]  (rest of tail before regime)
        self.n_features_per_bar = 6
        self.portfolio_dim = 1
        obs_dim = window_size * self.n_features_per_bar + self.portfolio_dim + self.regime_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        # Episode tracking
        self._position = 0.0
        self._entry_price = 0.0
        self._cumulative_pnl = 0.0

    def _build_features(self):
        """Compute per-bar feature vectors."""
        df = self.df
        close = df["close"].values
        volume = df["volume"].values
        n = len(close)

        features = []
        for i in range(n):
            o = df["open"].iloc[i]
            h = df["high"].iloc[i]
            l = df["low"].iloc[i]
            c = close[i]
            v = volume[i]

            norm_close = c / (o + 1e-8) - 1.0
            norm_high = h / (o + 1e-8) - 1.0
            norm_low = l / (o + 1e-8) - 1.0
            range_pct = (h - l) / (l + 1e-8)
            vol_ratio = v / (np.mean(volume[max(0, i - 20): i + 1]) + 1e-8)

            features.append([norm_close, norm_high, norm_low, range_pct, vol_ratio, c / 100.0])

        self._features = np.array(features, dtype=np.float32)

    def _get_obs(self):
        """Build the observation vector for the current step (O(1) array lookups).

        Structure: [feat_window(window * features), port_state(1), regime_obs(6)]
        Portfolio state comes BEFORE regime to match AdaptiveLSTMFeatureExtractor's
        tail handling: tail = [portfolio_state, regime_features].
        """
        end = self._step_idx
        start = end - self.window_size
        feat_window = self._features[start:end].reshape(-1)
        regime_obs = self._regime_obs_cache[end]  # pre-computed at index i for df.iloc[:i]
        port_state = np.array([self._cumulative_pnl / 1000.0], dtype=np.float32)
        # portfolio_state BEFORE regime_obs so extractor's tail handling is correct
        return np.concatenate([feat_window, port_state, regime_obs]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_idx = self.window_size
        self._position = 0.0
        self._entry_price = 0.0
        self._cumulative_pnl = 0.0
        return self._get_obs(), {}

    def step(self, action):
        self._step_idx += 1
        done = self._step_idx >= len(self.df) - 1
        truncated = False

        if done:
            return self._get_obs(), 0.0, True, False, {}

        current_close = self.df["close"].iloc[self._step_idx]
        next_close = self.df["close"].iloc[min(self._step_idx + 1, len(self.df) - 1)]
        price_change = (next_close - current_close) / (current_close + 1e-8)

        action_dir = float(np.clip(action[0], -1, 1))
        reward = action_dir * price_change * 10.0
        reward -= abs(action_dir) * 0.001

        self._cumulative_pnl += reward

        return self._get_obs(), reward, done, truncated, {}


# ── Tests ──────────────────────────────────────────────────────────────


class TestEnvCreation:
    """Validate the synthetic environment can be created and reset."""

    def test_env_creation(self):
        """Environment should have the right spaces."""
        df = make_synthetic_ohlcv(500)
        env = SyntheticTradingEnv(df)
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.Box)
        assert env.observation_space.shape[0] > 0
        assert env.action_space.shape[0] == 6

    def test_env_reset_and_step(self):
        """Environment should reset and step without errors."""
        df = make_synthetic_ohlcv(500)
        env = SyntheticTradingEnv(df)
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs.shape == obs.shape
        assert isinstance(reward, (int, float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))


def _assert_actor_critic_regime_independence(
    policy,
    make_env: Callable,
    *,
    n_samples: int = 20,
    min_weight_diff: float = 1e-4,
) -> tuple[float, float]:
    """
    Structural-first check that actor and critic regime heads are independent.

    Prefers direct weight inspection (different params after different gradient paths)
    over behavioral sampling. Falls back to sampling only if weights are nearly identical
    (possible on very short training runs).

    Returns (agreement, weight_diff) for reporting.
    """
    assert policy.regime_classifier is not policy.value_classifier, "Shared classifier module (not independent)"

    rc_w = policy.regime_classifier.weight.detach().cpu().numpy()
    vc_w = policy.value_classifier.weight.detach().cpu().numpy()
    weight_diff = float(np.abs(rc_w - vc_w).max())

    agreement = 1.0
    if weight_diff < min_weight_diff:
        disagreed = False
        for _ in range(n_samples):
            o, _ = make_env().reset()
            ot = th.as_tensor(o[None, :], dtype=th.float32, device=policy.device)
            rp = policy.get_regime_probs(ot)
            vp = policy.get_value_regime_probs(ot)
            agreement = float(rp[0].argmax() == vp[0].argmax())
            if agreement < 1.0:
                disagreed = True
                break
        assert disagreed or weight_diff >= min_weight_diff, (
            f"Actor and critic regime classifiers appear identical "
            f"(max weight diff={weight_diff:.2e}; no argmax disagreement in {n_samples} samples)"
        )
    else:
        # Compute one representative agreement for logging
        o, _ = make_env().reset()
        ot = th.as_tensor(o[None, :], dtype=th.float32, device=policy.device)
        rp = policy.get_regime_probs(ot)
        vp = policy.get_value_regime_probs(ot)
        agreement = float(rp[0].argmax() == vp[0].argmax())

    return agreement, weight_diff


@pytest.mark.slow
class TestRegimeRoutedPPOTraining:
    """Full training pipeline integration test."""

    def test_regime_routed_ppo_train_10k(self):
        """
        Train RegimeRoutedPPO for 10000 steps and verify regime decomposition.

        This is the main integration test covering:
        - Synthetic data generation with regime structure
        - Synthetic environment with RegimeDetector
        - Model instantiation with regime-routed policy
        - Training loop completion
        - Regime probability structure (non-uniform)
        - Actor-critic independence
        """
        import torch as th
        from stable_baselines3.common.vec_env import DummyVecEnv

        from drl.regime_routed_policy import (
            RegimeRoutedPPO,
            RegimeRoutedActorCriticPolicy,
        )

        # ── 1. Create synthetic data ──
        df = make_synthetic_ohlcv(1500, seed=42, regimes=3)
        window_size = 100
        total_timesteps = 10000

        # ── 2. Create DummyVecEnv wrapper ──
        def make_env():
            return SyntheticTradingEnv(df, window_size=window_size)

        env = DummyVecEnv([make_env])

        # ── 3. Determine regime info from env ──
        test_env = make_env()
        obs_dim = test_env.observation_space.shape[0]
        regime_dim = test_env.regime_dim

        # ── 4. Create RegimeRoutedPPO ──
        policy_kwargs = {
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "num_regimes": 5,
            "regime_dim": regime_dim,
        }

        model = RegimeRoutedPPO(
            RegimeRoutedActorCriticPolicy,
            env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            regime_loss_coef=0.05,
            verbose=0,
        )

        # ── 5. Train ──
        model.learn(total_timesteps=total_timesteps)

        # ── 6. Verify policy architecture ──
        policy = model.policy
        assert hasattr(policy, "regime_classifier"), "Missing regime_classifier"
        assert hasattr(policy, "value_classifier"), "Missing value_classifier"
        assert hasattr(policy, "regime_action_nets"), "Missing regime_action_nets"
        assert hasattr(policy, "regime_value_nets"), "Missing regime_value_nets"
        assert len(policy.regime_action_nets) == 5
        assert len(policy.regime_value_nets) == 5

        # ── 7. Verify regime probabilities are non-uniform ──
        obs, _ = make_env().reset()
        obs_tensor = th.as_tensor(obs[None, :], dtype=th.float32, device=policy.device)
        regime_probs = policy.get_regime_probs(obs_tensor)
        assert regime_probs.shape == (1, 5), f"Expected (1, 5), got {regime_probs.shape}"

        # Check that probabilities are non-uniform (meaningful decomposition)
        uniform = 1.0 / 5
        assert (
            regime_probs[0] > uniform * 1.05
        ).any(), f"Regime probs too uniform: {regime_probs[0].detach().cpu().numpy()}"

        # ── 8. Verify critic regime classifier is independent ──
        value_probs = policy.get_value_regime_probs(obs_tensor)
        assert value_probs.shape == (1, 5)

        agreement, weight_diff = _assert_actor_critic_regime_independence(
            policy, make_env, n_samples=20, min_weight_diff=1e-4
        )

        print(f"\nIntegration test results:")
        print(f"  Actor regime probs:    {regime_probs[0].detach().cpu().numpy()}")
        print(f"  Value regime probs:    {value_probs[0].detach().cpu().numpy()}")
        print(f"  Actor-Value agreement: {agreement:.0%} (weight_diff={weight_diff:.2e})")
        print(f"  Training: OK ({total_timesteps} steps completed)")
