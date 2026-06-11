"""
Shared taming logic for Lane A and Lane B trading experiments.

Extracted to avoid code duplication between:
    training/run_lane_a_fix.py       (FixedFeatureEnv)
    training/run_lane_b_raw_lstm.py  (TamedOHLCVEnv)

Classes
-------
BaseTamedEnv     : Abstract base with EMA smoothing, cooldown, turnover penalty.
MetricsCallback  : SB3 callback collecting positions, rewards, episode rewards.

Functions
---------
compute_weight_hash  : MD5 hash of model policy weights (deterministic).
evaluate             : Generic deterministic eval using an env_factory callable.
"""
import hashlib
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback


# ── Shared config defaults (overridable by subclasses / callers) ──
DEFAULT_TURNOVER_COST = 0.005
DEFAULT_CONCENTRATION_PENALTY = 0.002
DEFAULT_SMOOTHING_ALPHA = 0.3
DEFAULT_COOLDOWN_STEPS = 5
DEFAULT_WINDOW_SIZE = 64


class BaseTamedEnv(gym.Env):
    """
    Trading environment with EMA smoothing, cooldown, and turnover penalties.

    Subclasses must implement:
        _build_features()     — set self.features (np.ndarray, shape (n, n_features))
        _raw_reward_at(idx)   — return the raw forward return at the given index
    """
    def __init__(self, *, window_size, turnover_cost, concentration_penalty,
                 smoothing_alpha, cooldown_steps, n, n_features):
        super().__init__()
        self.window_size = window_size
        self.turnover_cost = turnover_cost
        self.concentration_penalty = concentration_penalty
        self.smoothing_alpha = smoothing_alpha
        self.cooldown_steps = cooldown_steps
        self.n = n
        self.n_features = n_features

        obs_dim = self.window_size * self.n_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.idx = None
        self._prev_smoothed = 0.0
        self._cooldown_until = 0
        self._step_count = 0
        self.features = None  # set by _build_features()

    # ── Subclass hooks ──

    def _build_features(self):
        """Build self.features array (n, n_features). Called once from __init__."""
        raise NotImplementedError

    def _raw_reward_at(self, idx):
        """Return the raw forward return at the given index."""
        raise NotImplementedError

    # ── Shared taming logic ──

    def _apply_cooldown(self, proposed_position):
        """If in cooldown, prevent zero-crossing by clamping."""
        if self._step_count < self._cooldown_until:
            if self._prev_smoothed >= 0 and proposed_position < 0:
                return max(proposed_position, 0.0)
            if self._prev_smoothed < 0 and proposed_position >= 0:
                return min(proposed_position, 0.0)
        return proposed_position

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random if hasattr(self, 'np_random') else np.random
        max_start = max(1, self.n - self.window_size - 2)
        self.idx = self.window_size + int(rng.integers(0, max_start))
        self._prev_smoothed = 0.0
        self._cooldown_until = 0
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        raw_pos = float(np.clip(action[0], -1.0, 1.0))

        # EMA smoothing
        smoothed = self.smoothing_alpha * raw_pos + (1 - self.smoothing_alpha) * self._prev_smoothed

        # Flip detection (only if not already in cooldown)
        if self._step_count >= self._cooldown_until and self._prev_smoothed * smoothed < 0:
            self._cooldown_until = self._step_count + self.cooldown_steps

        # Apply cooldown clamp
        final_position = self._apply_cooldown(smoothed)

        # Reward
        raw_reward = self._raw_reward_at(self.idx)
        growth = final_position * raw_reward
        pos_change = abs(final_position - self._prev_smoothed)
        tc = self.turnover_cost * pos_change
        cc = self.concentration_penalty * (final_position ** 2)
        reward = growth - tc - cc

        self._prev_smoothed = final_position
        self._step_count += 1
        self.idx += 1
        done = self.idx >= self.n - 1
        truncated = False

        info = {
            "position": float(final_position),
            "raw_position": float(raw_pos),
            "raw_reward": float(raw_reward),
            "growth": float(growth),
            "turnover_cost": float(tc),
            "concentration_cost": float(cc),
            "cooldown": 1 if self._step_count < self._cooldown_until else 0,
        }

        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        start = self.idx - self.window_size
        window = self.features[start:self.idx].flatten()
        return window.astype(np.float32)


class MetricsCallback(BaseCallback):
    """Collects positions, rewards, and episode rewards during training."""
    def __init__(self):
        super().__init__()
        self.positions = []
        self.rewards = []
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        for info in infos:
            if "position" in info:
                self.positions.append(info["position"])
            if "raw_reward" in info:
                self.rewards.append(info["raw_reward"])
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True


def compute_weight_hash(model):
    """Return a short MD5 hash of the model's policy network weights."""
    hasher = hashlib.md5()
    for name, param in model.policy.state_dict().items():
        hasher.update(param.cpu().numpy().tobytes())
    return hasher.hexdigest()[:12]


def evaluate(model, env_factory, annualization_factor=252 * 288):
    """
    Run deterministic evaluation on an env created by env_factory.

    Parameters
    ----------
    model : PPO
        Trained stable-baselines3 model.
    env_factory : callable -> gym.Env
        Zero-argument callable returning an env with info dict keys
        "position" and "raw_reward".
    annualization_factor : float
        Factor to annualize Sharpe (default 252*288 for 1m bars).

    Returns
    -------
    dict with keys: pos_mean, pos_std, long_pct, short_pct, flat_pct,
    sharpe, total_return, max_drawdown, turnover, n_steps, action_hash,
    positions, net_worth.
    """
    env = env_factory()
    obs, _ = env.reset()
    positions = []
    raw_rewards = []
    net_worth = [10000.0]
    done = False
    step = 0

    while not done and step < 1_000_000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        positions.append(info["position"])
        raw_rewards.append(info["raw_reward"])
        net_worth.append(net_worth[-1] * (1 + info["raw_reward"] * info["position"]))
        step += 1

    pos = np.array(positions)
    raw = np.array(raw_rewards)
    nw = np.array(net_worth)
    total_ret = (nw[-1] / nw[0] - 1) * 100 if len(nw) > 1 else 0.0

    # Sharpe on strategy returns
    strat_returns = raw * pos
    sharpe = 0.0
    if np.std(strat_returns) > 1e-10 and len(strat_returns) > 1:
        sharpe = float(np.mean(strat_returns) / np.std(strat_returns) * np.sqrt(annualization_factor))

    # Turnover: % of steps where position changes by >1%
    turnover = 0.0
    if len(pos) > 1:
        turnover = float(np.mean(np.abs(np.diff(pos)) > 0.01) * 100)

    # Max drawdown
    peak = np.maximum.accumulate(nw)
    dd = (nw - peak) / peak * 100
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    # Hash for reproducibility
    action_hash = hashlib.md5(pos.tobytes()).hexdigest()[:12] if len(pos) > 0 else "none"

    return {
        "pos_mean": float(np.mean(pos)),
        "pos_std": float(np.std(pos)),
        "long_pct": float(np.mean(pos > 0.01) * 100),
        "short_pct": float(np.mean(pos < -0.01) * 100),
        "flat_pct": float(np.mean(np.abs(pos) <= 0.01) * 100),
        "sharpe": sharpe,
        "total_return": total_ret,
        "max_drawdown": max_dd,
        "turnover": turnover,
        "n_steps": len(pos),
        "action_hash": action_hash,
        "positions": pos,
        "net_worth": nw,
    }
