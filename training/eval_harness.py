"""
Shared eval harness for plotting scripts.

Provides reusable components extracted from plot_regime_actions.py and
plot_position_timeseries.py:

  - fit_regime_detector: fit RegimeDetector on OHLCV data
  - build_regime_observations: append regime one-hot + confidence to feature array
  - make_eval_env: DummyVecEnv with action-contingent reward
  - collect_positions: deterministic eval returning positions array
  - collect_metrics: deterministic eval returning (positions, rewards) tuple
  - plot_regime_action_distribution: 5-panel per-regime action bar chart
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from drl.regime_detector import RegimeDetector, NUM_REGIMES

# One-hot (5) + confidence = 6
REGIME_DIM = NUM_REGIMES + 1

# Public API
__all__ = [
    "fit_regime_detector",
    "build_regime_observations",
    "make_eval_env",
    "collect_positions",
    "collect_metrics",
    "plot_regime_action_distribution",
    "REGIME_DIM",
]

# ── RegimeDetector fitting ─────────────────────────────────────────────


def fit_regime_detector(df: pd.DataFrame, verbose: bool = True) -> RegimeDetector:
    """Fit RegimeDetector on OHLCV data and return the fitted detector.

    Args:
        df: OHLCV DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        verbose: If True, print a status line.

    Returns:
        Fitted RegimeDetector instance (calls ``fit_heuristic`` internally).
    """
    if verbose:
        print("Fitting RegimeDetector...")
    detector = RegimeDetector()
    detector.fit_heuristic(df)
    return detector


# ── Observation construction ───────────────────────────────────────────


def build_regime_observations(
    features: np.ndarray,
    df: pd.DataFrame,
    detector: RegimeDetector,
    window_size: int = 100,
    regime_dim: int = REGIME_DIM,
) -> np.ndarray:
    """Build observation array with real regime features appended to the tail.

    For each window index ``i``, appends a 6-element regime observation
    (one-hot of the detected regime + confidence) to the feature vector.

    Args:
        features: ``(n_windows, n_feature_dims)`` feature array (first return
            element from ``make_synthetic_features``).
        df: OHLCV DataFrame whose length equals the number of raw bars.
        detector: A fitted ``RegimeDetector``.
        window_size: Number of bars per observation window.
        regime_dim: Size of the regime feature tail (default 6).

    Returns:
        ``(n_windows, n_feature_dims + regime_dim)`` observation array.
    """
    n_windows = len(features)
    obs_dim = features.shape[1] + regime_dim
    observations = np.zeros((n_windows, obs_dim), dtype=np.float32)

    for i in range(n_windows):
        observations[i, : features.shape[1]] = features[i]
        bar_idx = i + window_size - 1
        lookback = df.iloc[max(0, bar_idx - 70) : bar_idx + 1]
        fv = detector.compute_features(lookback)
        r_idx, conf = detector.classify(fv)
        oh = np.zeros(regime_dim, dtype=np.float32)
        oh[r_idx] = 1.0
        oh[-1] = conf
        observations[i, features.shape[1] :] = oh

    return observations


# ── Environment factory ────────────────────────────────────────────────


def make_eval_env(
    obs: np.ndarray,
    df: Optional[pd.DataFrame] = None,
    window_size: int = 100,
) -> DummyVecEnv:
    """Create a DummyVecEnv for evaluation with an action-contingent reward.

    Reward formula when *df* is provided::

        reward = sign(position) * bar_return - 0.0001 * |position_change|

    When *df* is ``None``, falls back to a synthetic reward computed from
    the first five elements of the observation vector.
    """
    _df = df  # capture for closure

    def _init():
        obs_dim = obs.shape[1]

        class _EvalEnv(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Box(
                    -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
                )
                self.action_space = spaces.Box(-1, 1, shape=(6,), dtype=np.float32)
                self._obs = obs
                self._df = _df
                self._step = window_size
                self._last_pos = 0.0
                self._close_idx = 3

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self._step = window_size
                self._last_pos = 0.0
                return self._obs[self._step - window_size].copy(), {}

            def step(self, action):
                act = np.array(action).flatten()
                pos = float(np.sign(np.mean(act)))
                idx = self._step
                if idx >= len(self._obs):
                    return self._obs[-1].copy(), 0.0, True, False, {}
                out = self._obs[idx].copy()
                if self._df is not None and idx < len(self._df):
                    cp = float(self._df.iloc[idx, self._close_idx])
                    cp_prev = float(self._df.iloc[idx - 1, self._close_idx])
                    ret = (cp - cp_prev) / (abs(cp_prev) + 1e-8)
                    change = pos - self._last_pos
                    r = np.sign(pos) * ret - 0.0001 * abs(change)
                    self._last_pos = pos
                else:
                    r = float(np.mean(out[:5])) * 0.01
                self._step += 1
                done = self._step >= len(self._obs)
                if done:
                    self._last_pos = 0.0
                return out, r, done, False, {}

        return _EvalEnv()

    return DummyVecEnv([_init])


# ── Collectors ─────────────────────────────────────────────────────────


def collect_positions(
    model,
    obs: np.ndarray,
    df: Optional[pd.DataFrame],
    window_size: int,
) -> np.ndarray:
    """Run deterministic evaluation and return the position time series.

    Args:
        model: A trained SB3 model (PPO or RegimeRoutedPPO).
        obs: Evaluation observation array.
        df: OHLCV DataFrame for reward computation.
        window_size: Observation window size.

    Returns:
        ``(n_steps,)`` array with values in {-1, 0, +1}.
    """
    env = make_eval_env(obs, df=df, window_size=window_size)
    cur_obs = env.reset()
    pos_list: list[float] = []
    while True:
        act, _ = model.predict(cur_obs, deterministic=True)
        pos_list.append(float(np.sign(np.mean(act))))
        cur_obs, _, done, _ = env.step(act)
        if done[0]:
            break
    return np.array(pos_list)


def collect_metrics(
    model,
    obs: np.ndarray,
    df: Optional[pd.DataFrame],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run deterministic evaluation and return (positions, rewards).

    Args:
        model: A trained SB3 model.
        obs: Evaluation observation array.
        df: OHLCV DataFrame for reward computation.
        window_size: Observation window size.

    Returns:
        Tuple of ``(positions, rewards)`` arrays, each ``(n_steps,)``.
    """
    env = make_eval_env(obs, df=df, window_size=window_size)
    cur_obs = env.reset()
    pos_list: list[float] = []
    rew_list: list[float] = []
    while True:
        act, _ = model.predict(cur_obs, deterministic=True)
        pos_list.append(float(np.sign(np.mean(act))))
        cur_obs, r, done, _ = env.step(act)
        rew_list.append(r.item())
        if done[0]:
            break
    return np.array(pos_list), np.array(rew_list)
# ── Plotting ──────────────────────────────────────────────────────────────


def plot_regime_action_distribution(
    positions: np.ndarray,
    regime_indices: np.ndarray,
    regime_labels: list[str] | None = None,
    num_regimes: int = NUM_REGIMES,
    title: str = "Per-Regime Action Distribution",
    save_path: str | None = "logs/regime_action_distribution.png",
    dpi: int = 150,
    verbose: bool = True,
):
    """Plot a 5-panel bar chart showing position distribution per regime.

    Creates one bar panel per regime with three bars (Short, Flat, Long)
    showing the count and percentage of steps in each position state.

    Args:
        positions: ``(n_steps,)`` array with values in {-1, 0, +1}.
        regime_indices: ``(n_steps,)`` array of regime index (0--4).
        regime_labels: List of ``num_regimes`` label strings.
            Defaults to ``REGIME_LABELS`` from ``drl.regime_detector``.
        num_regimes: Number of regime classes (default 5).
        title: Figure super-title.
        save_path: Path to save the PNG.  ``None`` disables saving.
        dpi: Figure DPI.
        verbose: If ``True``, print a textual summary table to stdout.

    Returns:
        The matplotlib ``Figure`` object (already saved to *save_path* if set).
    """
    import matplotlib.pyplot as plt

    if regime_labels is None:
        regime_labels = REGIME_LABELS

    fig, axes = plt.subplots(1, num_regimes, figsize=(18, 4), sharey=True)

    colors_short = "#e74c3c"
    colors_flat = "#95a5a6"
    colors_long = "#2ecc71"

    for r in range(num_regimes):
        ax = axes[r]
        mask = regime_indices == r
        pos_r = positions[mask]
        n_r = len(pos_r)

        if n_r == 0:
            ax.text(
                0.5, 0.5, "No steps",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray",
            )
            ax.set_title(f"{regime_labels[r]}\n(0 steps)", fontsize=10)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["Short", "Flat", "Long"])
            continue

        n_short = int((pos_r == -1).sum())
        n_flat = int((pos_r == 0).sum())
        n_long = int((pos_r == 1).sum())

        bars = ax.bar(
            ["Short (-1)", "Flat (0)", "Long (+1)"],
            [n_short, n_flat, n_long],
            color=[colors_short, colors_flat, colors_long],
            edgecolor="white",
            linewidth=0.8,
        )

        for bar, count in zip(bars, [n_short, n_flat, n_long]):
            pct = count / n_r * 100
            if pct > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(1, n_r * 0.01),
                    f"{pct:.1f}%",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                )

        ax.set_title(
            f"{regime_labels[r]}\n({n_r} steps)",
            fontsize=10, fontweight="bold",
        )
        ax.set_ylim(0, max(n_short, n_flat, n_long) * 1.25)
        ax.tick_params(axis="x", labelsize=8)

    axes[0].set_ylabel("Step count", fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        import os as _os
        _os.makedirs(_os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if verbose:
            print(f"Saved: {save_path}")

    if verbose:
        print()
        print("=" * 60)
        print("  Per-Regime Action Distribution Summary")
        print("=" * 60)
        print(f"  {'Regime':<16} {'Steps':>6} {'Short%':>8} {'Flat%':>8} {'Long%':>8}")
        print("  " + "-" * 46)
        for r in range(num_regimes):
            mask = regime_indices == r
            n = mask.sum()
            if n > 0:
                short_pct = (positions[mask] == -1).mean() * 100
                flat_pct = (positions[mask] == 0).mean() * 100
                long_pct = (positions[mask] == 1).mean() * 100
            else:
                short_pct = flat_pct = long_pct = 0.0
            print(
                f"  {regime_labels[r]:<16} {n:>6} "
                f"{short_pct:>7.1f}% {flat_pct:>7.1f}% {long_pct:>7.1f}%"
            )
        print()

    return fig
