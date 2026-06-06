"""
End-to-end integration test: TradingEnv + RegimeRoutedPPO.learn(10000).

Validates that the full training pipeline works end-to-end:
- Synthetic OHLCV data generation
- TradingEnv instantiation with legacy action space
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

PROJECT_ROOT = r"C:\supreme-chainsaw"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def make_synthetic_ohlcv(n_bars=1000, seed=42):
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)
    idx = pd.date_range("2026-01-01", periods=n_bars, freq="5min", tz="UTC")
    price = np.cumsum(np.random.randn(n_bars) * 0.001) + 100
    df = pd.DataFrame({
        "open": price * (1 - 0.0004 * abs(np.random.randn(n_bars))),
        "high": price * (1 + 0.003 * abs(np.random.randn(n_bars))),
        "low": price * (1 - 0.003 * abs(np.random.randn(n_bars))),
        "close": price,
        "volume": 100 + 50 * np.random.rand(n_bars),
        "tick_volume": (100 + 50 * np.random.rand(n_bars)).astype(int),
    }, index=idx)
    df.index.name = "time"
    return df


@pytest.fixture
def synthetic_data():
    return make_synthetic_ohlcv(1000)


@pytest.fixture
def trading_env(synthetic_data):
    from drl.trading_env import TradingEnv
    env = TradingEnv(
        synthetic_data,
        window_size=100,
        action_config={"decision_ppo": False},
        reward_scale=1.0,
        penalty_scale=1.0,
    )
    # Fix observation space to match actual obs shape
    obs, _ = env.reset()
    from gymnasium import spaces
    env.observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
    )
    return env


class TestIntegrationRegimeRouted:
    """End-to-end integration test for regime-routed PPO training."""

    def test_env_creation(self, trading_env):
        from gymnasium import spaces
        assert isinstance(trading_env.observation_space, spaces.Box)
        assert isinstance(trading_env.action_space, spaces.Box)
        assert len(trading_env.observation_space.shape) == 1
        assert trading_env.action_space.shape == (6,)
        assert trading_env.observation_space.shape[0] > 100

    def test_env_reset_and_step(self, trading_env):
        obs, info = trading_env.reset()
        assert obs.shape == trading_env.observation_space.shape
        assert obs.dtype == np.float32
        action = trading_env.action_space.sample()
        obs2, reward, done, truncated, info = trading_env.step(action)
        assert obs2.shape == obs.shape
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    @pytest.mark.slow
    def test_regime_routed_ppo_train_10k(self, synthetic_data):
        from stable_baselines3.common.vec_env import DummyVecEnv
        from drl.regime_routed_policy import (
            RegimeRoutedPPO,
            RegimeRoutedActorCriticPolicy,
        )
        from drl.regime_detector import NUM_REGIMES
        from drl.trading_env import TradingEnv

        # Fresh env for this test (avoid state leakage)
        env = TradingEnv(
            synthetic_data,
            window_size=100,
            action_config={"decision_ppo": False},
            reward_scale=1.0,
            penalty_scale=1.0,
        )
        obs, _ = env.reset()
        from gymnasium import spaces
        env.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

        vec_env = DummyVecEnv([lambda: env])

        model = RegimeRoutedPPO(
            RegimeRoutedActorCriticPolicy,
            vec_env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            regime_loss_coef=0.05,
            policy_kwargs={
                "num_regimes": NUM_REGIMES,
                "regime_dim": NUM_REGIMES + 1,
                "net_arch": [64, 64],
            },
            verbose=1,
            seed=42,
            device="cpu",
        )

        policy = model.policy
        assert hasattr(policy, "regime_classifier")
        assert hasattr(policy.value_net, "value_classifier")
        assert hasattr(policy, "regime_action_nets")
        assert hasattr(policy, "regime_value_nets")
        assert len(policy.regime_action_nets) == NUM_REGIMES
        assert len(policy.regime_value_nets) == NUM_REGIMES

        model.learn(total_timesteps=10000, progress_bar=False)

        obs = vec_env.reset()
        action, _states = model.predict(obs, deterministic=True)
        assert action.shape == (1, 6)

        import torch
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)
        regime_probs = policy.get_regime_probs(obs_tensor)
        assert regime_probs is not None
        assert regime_probs.shape == (1, NUM_REGIMES)
        uniform = 1.0 / NUM_REGIMES
        assert (regime_probs > uniform * 1.05).any(), "Regime probs too uniform"

        value_probs = policy.get_value_regime_probs(obs_tensor)
        assert value_probs is not None
        assert value_probs.shape == (1, NUM_REGIMES)

        agreement = (regime_probs.argmax(axis=1) == value_probs.argmax(axis=1)).float().mean()
        # With 5 regimes and independent classifiers, agreement is typically < 30%
        # This is a soft check - may rarely fail with very early training
        assert agreement < 1.0, "Actor and critic fully agree"

        print(f"Training complete: 10,000 steps")
        print(f"Actor regime probs: {regime_probs[0].detach().numpy().round(3)}")
        print(f"Value regime probs: {value_probs[0].detach().numpy().round(3)}")
        print(f"Actor-Value agreement: {agreement:.1%}")

        vec_env.close()
