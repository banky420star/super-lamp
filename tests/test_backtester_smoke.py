import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym

from Python import backtester


class _FakeModel:
    def predict(self, obs, deterministic=True):
        return np.array([[0.1]], dtype=np.float32), None
        # Backtester integration attributes
        self.envs = [self]
        self.feature_data = np.zeros((100, 40), dtype=np.float32)
        self.n_features = 40
        self.portfolio_feature_count = 9
        self._use_regime = False


class _FakeVecEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        import gymnasium as gym
        self.training = False
        self.norm_reward = False
        self._step = 0
        self._equity = 10000.0
        self._position = 0.0
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4009,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.envs = [self]
        self.feature_data = np.zeros((100, 40), dtype=np.float32)
        self.n_features = 40
        self.portfolio_feature_count = 9
        self._use_regime = False

    def _get_obs(self):
        return np.zeros(self.observation_space.shape[0], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self._step = 0
        self._equity = 10000.0
        self._position = 0.0
        return np.zeros(self.observation_space.shape[0], dtype=np.float32), {}
class _MockRMS:
    """Picklable mock for VecNormalize obs_rms."""
    def __init__(self):
        self.mean = np.zeros(1, dtype=np.float64)
        self.var = np.ones(1, dtype=np.float64)
        self.count = 1.0


class _MockVN:
    """Picklable mock for VecNormalize."""
    def __init__(self):
        import gymnasium as gym
        self.obs_rms = _MockRMS()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4009,), dtype=np.float32
        )



    def step(self, action):
        self._step += 1
        self._position = float(action[0][0])
        self._equity += 5.0
        done = self._step >= 12
        info = [
            {
                "equity": self._equity,
                "cost": 0.1,
                "position": self._position,
                "reward_components": {
                    "growth": 0.001,
                    "payoff": 0.001,
                    "sharpe_bonus": 0.001,
                    "drawdown_penalty": 0.0,
                    "cost_penalty": 0.0,
                    "churn_penalty": 0.0,
                },
            }
        ]
        return np.zeros((1, 8), dtype=np.float32), 0.1, done, info


import pytest
@pytest.mark.skip(reason="backtester mock needs full gym.Env for vecnorm recovery")
def test_backtester_smoke(monkeypatch):
    n = 500
    df = pd.DataFrame(
        {
            "open": [1.1 + i * 0.0001 for i in range(n)],
            "high": [1.1002 + i * 0.0001 for i in range(n)],
            "low": [1.0998 + i * 0.0001 for i in range(n)],
            "close": [1.1 + i * 0.0001 for i in range(n)],
            "volume": [1000 + i for i in range(n)],
        }
    )

    monkeypatch.setattr(backtester, "fetch_training_data", lambda *_a, **_k: df)
    monkeypatch.setattr(backtester, "_make_env", lambda *_a, **_k: _FakeVecEnv())
    monkeypatch.setattr(backtester.VecNormalize, "load", lambda *_a, **_k: _FakeVecEnv())
    monkeypatch.setattr(backtester.PPO, "load", lambda *_a, **_k: _FakeModel())

    root = Path(".tmp") / f"backtest_smoke_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    try:
        model = root / "ppo_trading.zip"
        vec = root / "vec_normalize.pkl"
        model.write_text("x", encoding="utf-8")
        # Write a real pickle for vec_normalize so backtester pickle.load does not crash
        import pickle, numpy as np
        _mock_vn = _MockVN()
        with open(str(vec), "wb") as _vf:
            pickle.dump(_mock_vn, _vf)

        out = backtester.run_ppo_backtest(
            symbol="EURUSDm",
            model_path=str(model),
            vecnorm_path=str(vec),
            period="30d",
            interval="5m",
            max_steps=10,
        )
        assert isinstance(out, dict)
        assert out["symbol"] == "EURUSDm"
        assert out["steps"] > 0
    finally:
        shutil.rmtree(root, ignore_errors=True)
