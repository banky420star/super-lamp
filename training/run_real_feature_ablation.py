"""
Real Feature Ablation Test Harness (ENGINEERED_V2)

Trains multiple RegimeRoutedPPO agents on real XAUUSDm data with different
feature subsets ablated from the ENGINEERED_V2 env feature matrix.

Feature groups tested (59-column env matrix):
  - ALL: all features (baseline)
  - NO_TREND: zero out trend features (htf_trend, vol_bucket)
  - NO_MOMENTUM: zero out momentum features (log_ret1, log_ret5, log_ret20)
  - NO_VOLATILITY: zero out realized volatility (rv_20)
  - NO_VOLUME: zero out volume features (rel_volume, spread_est_bps)
  - NO_CROSS_ASSET: zero out 18 cross-asset features
  - NO_ML_SIGNAL: zero out XGBoost signal feature
  - NO_REGIME: disable regime detector (regime_dim=0)
  - NO_PATTERN: zero out 11 classical pattern features

Each group is trained for a configurable number of timesteps on real data.
Results are logged to a CSV for comparison.

Usage:
    python training/run_real_feature_ablation.py --symbol XAUUSDm --steps 30000 --trials 1
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gymnasium as gym
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from drl.adaptive_feature_extractor import AdaptiveLSTMFeatureExtractor
from drl.regime_routed_policy import RegimeRoutedPPO, RegimeRoutedActorCriticPolicy

# Disable SB3 warnings
os.environ["SB3_VERBOSE"] = "0"


# ── Feature Group Definitions ──────────────────────────────────────────

# The # ENGINEERED_V2 env feature matrix (59 columns for XAUUSDm) columns are
# built by _build_engineered_env_matrix in this order:
#
#   open_rel, high_rel, low_rel, close_rel      -> 0-3
#   log_vol                                       -> 4
#   log_ret1, log_ret5, log_ret20                 -> 5-7
#   body_ratio, upper_wick, lower_wick, range_ratio  -> 8-11
#   rv_20                                         -> 12
#   rel_volume                                    -> 13
#   spread_est_bps                                -> 14
#   htf_trend                                     -> 15
#   vol_bucket                                    -> 16
#   hour_sin, hour_cos, dow_sin, dow_cos          -> 17-20
#   session_london, session_ny, major_open        -> 21-23
#   news_prox, news_soon, session_overlap,
#     mins_since_london, news_avoid               -> 24-28
#   11 patterns                                   -> 29-39
#   18 cross-asset features                       -> 40-57
#   1 ml_signal probability                       -> 58

FEATURE_GROUPS = {
    "trend": {
        "indices": [15, 16],
        "description": "Trend strength indicator, volatility bucket",
    },
    "momentum": {
        "indices": [5, 6, 7],
        "description": "Log returns at 1, 5, 20 bars",
    },
    "volatility": {
        "indices": [12],
        "description": "20-bar realized volatility",
    },
    "volume": {
        "indices": [13, 14],
        "description": "Relative volume, spread estimate (bps)",
    },
    "cross_asset": {
        "indices": list(range(40, 58)),
        "description": "Cross-asset correlation features (18 cols)",
    },
    "ml_signal": {
        "indices": [58],
        "description": "XGBoost next-bar direction probability",
    },
    "pattern": {
        "indices": list(range(29, 40)),
        "description": "Classical candlestick patterns (11 cols)",
    },
}

# Groups tested in the ablation study
ABLATION_GROUPS = [
    "ALL",
    "NO_TREND",
    "NO_MOMENTUM",
    "NO_VOLATILITY",
    "NO_VOLUME",
    "NO_CROSS_ASSET",
    "NO_ML_SIGNAL",
    "NO_REGIME",
    "NO_PATTERN",
]

# Feature set cardinality (for 59-col matrix or 40-col base)
FULL_FEATURE_COUNT = 59


# ── Real Data & Features ──────────────────────────────────────────────

def load_real_data(
    symbol: str = "XAUUSDm",
    n_bars: int = 5000,
) -> pd.DataFrame:
    """Load real OHLCV data for the given symbol.

    Falls back to fetching from the data_feed pipeline; if that fails,
    downloads from a Python data source or generates synthetic data as
    a last resort so the harness is always runnable.
    """
    try:
        from Python.data_feed import fetch_training_data

        df = fetch_training_data(symbol, bars=n_bars)
        if df is not None and len(df) >= 1000:
            print(f"  Loaded {len(df)} bars of {symbol} from data_feed")
            return df
    except Exception as exc:
        print(f"  data_feed unavailable ({exc}), trying MT5 download...")

    # Fallback: try downloading via MT5
    try:
        import MetaTrader5 as mt5

        if mt5.initialize():
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n_bars)
            mt5.shutdown()
            if rates is not None and len(rates) >= 1000:
                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                df.set_index("time", inplace=True)
                print(f"  Loaded {len(df)} bars of {symbol} from MT5")
                return df
    except Exception as exc:
        print(f"  MT5 download failed ({exc}), using synthetic data...")

    # Last resort: synthetic data spanning a realistic price range
    print("  WARNING: Using synthetic data — results will not reflect real market structure")
    np.random.seed(42)
    idx = pd.date_range("2026-01-01", periods=n_bars, freq="5min", tz="UTC")
    price = 100.0 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.0005))
    df = pd.DataFrame({
        "open": price * (1 - 0.0003 * np.abs(np.random.randn(n_bars))),
        "high": price * (1 + 0.002 * np.abs(np.random.randn(n_bars))),
        "low": price * (1 - 0.002 * np.abs(np.random.randn(n_bars))),
        "close": price,
        "volume": 100 + 50 * np.random.rand(n_bars),
        "tick_volume": (100 + 50 * np.random.rand(n_bars)).astype(int),
    }, index=idx)
    df.index.name = "time"
    return df


def build_real_features(
    df: pd.DataFrame,
    symbol: str = "XAUUSDm",
    ablation_group: Optional[str] = None,
) -> tuple[np.ndarray, int]:
    """Build the ENGINEERED_V2 env feature matrix from real OHLCV data.

    Optionally ablates (zeros out) a feature group to measure its impact.

    Args:
        df: OHLCV DataFrame from load_real_data().
        symbol: Trading symbol (for cross-asset features).
        ablation_group: Feature group to ablate, or "ALL" / None for full set.

    Returns:
        (observations, n_features_per_bar) tuple.
        observations: (n_windows, window_size * n_features_per_bar) array
                      where each row is a flattened window of bars.
    """
    try:
        from Python.feature_pipeline import build_env_feature_matrix
        env_matrix = build_env_feature_matrix(df, symbol=symbol)
    except Exception as exc:
        print(f"  Feature pipeline failed ({exc}), building inline...")
        env_matrix = _build_features_fallback(df, symbol)

    n_features = env_matrix.shape[1]
    print(f"  Feature matrix: {n_features} columns x {len(env_matrix)} bars")

    # ── Apply ablation: zero out the specified feature group ──
    if ablation_group and ablation_group in FEATURE_GROUPS:
        group = FEATURE_GROUPS[ablation_group]
        idx = [i for i in group["indices"] if 0 <= i < n_features]
        if idx:
            env_matrix[:, idx] = 0.0
            print(f"  Ablated '{ablation_group}': zeroed {len(idx)} cols — {group['description']}")
    elif ablation_group and ablation_group != "ALL":
        print(f"  Unknown ablation group '{ablation_group}', using ALL features")

    return env_matrix, n_features


def _build_features_fallback(df: pd.DataFrame, symbol: str = "") -> np.ndarray:
    """Simple fallback feature builder when the real pipeline is unavailable."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values
    n = len(close)

    features = []
    for i in range(n):
        feats = []
        # Returns
        r1 = close[i] / close[max(0, i - 1)] - 1 if i >= 1 else 0.0
        r5 = close[i] / close[max(0, i - 5)] - 1 if i >= 5 else 0.0
        r20 = close[i] / close[max(0, i - 20)] - 1 if i >= 20 else 0.0
        # Volatility
        if i >= 20:
            rv = np.std(close[i - 20:i + 1]) / close[i]
        else:
            rv = 0.0
        # Volume ratio
        vol_ma10 = np.mean(volume[max(0, i - 10):i + 1])
        rel_vol = volume[i] / max(vol_ma10, 1e-8)
        # Trend
        ma50 = np.mean(close[max(0, i - 50):i + 1])
        htf_trend = (close[i] / ma50 - 1) if ma50 > 0 else 0.0

        feats.extend([close[i] / df["open"].iloc[i] - 1 if i > 0 else 0.0,  # open_rel
                      high[i] / close[i] - 1,  # high_rel
                      low[i] / close[i] - 1,  # low_rel
                      0.0,  # placeholders ...
                      0.0, r1, r5, r20,
                      0.0, 0.0, 0.0, 0.0,  # candle geometry
                      rv,
                      rel_vol,
                      0.0,  # spread
                      htf_trend,
                      0.0,  # vol_bucket
                      ])
        # Pad to 40 base cols + 18 cross + 1 ml = 59
        while len(feats) < 59:
            feats.append(0.0)
        features.append(feats)

    return np.array(features, dtype=np.float32)


def compute_reward_signal(close_prices: np.ndarray, lookahead: int = 3) -> np.ndarray:
    """Compute a realistic trading reward based on forward returns.

    Uses a clipped forward return scaled by volatility as the reward signal.
    This gives higher reward to configurations that can predict directional moves.
    """
    n = len(close_prices)
    rewards = np.zeros(n, dtype=np.float32)
    for i in range(n - lookahead):
        fwd_ret = (close_prices[i + lookahead] / close_prices[i]) - 1.0
        # Scale by recent volatility for risk-adjusted signal
        if i >= 20:
            local_vol = np.std(close_prices[i - 20:i + 1] / close_prices[i - 20] - 1) + 1e-8
        else:
            local_vol = 0.001
        rewards[i] = np.clip(fwd_ret / local_vol, -1.0, 1.0)
    return rewards


# ── Environment ────────────────────────────────────────────────────────

def make_env(
    feature_matrix: np.ndarray,
    rewards: np.ndarray,
    window_size: int = 100,
    regime_dim: int = 5,
) -> DummyVecEnv:
    """Create a DummyVecEnv that feeds real feature observations.

    The environment exposes pre-computed feature windows as observations
    and uses the pre-computed reward signal. This isolates the feature
    utilisation question from trading environment complexity.
    """
    n_bars = feature_matrix.shape[0]
    n_features = feature_matrix.shape[1]
    obs_dim = window_size * n_features + regime_dim

    def _init() -> gym.Env:
        class _FeatureEnv(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
                )
                self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
                self._step = window_size
                self._max_step = n_bars - 1

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self._step = window_size
                # Return first observation
                obs = self._get_obs(self._step)
                return obs, {}

            def step(self, action):
                self._step += 1
                if self._step >= self._max_step:
                    # Terminal
                    obs = self._get_obs(self._max_step - 1)
                    return obs, 0.0, True, False, {}

                obs = self._get_obs(self._step)
                reward = float(rewards[self._step])
                return obs, reward, False, False, {}

            def _get_obs(self, idx: int) -> np.ndarray:
                """Build observation: windowed features + regime features."""
                start = idx - window_size
                window = feature_matrix[start:idx]  # (window_size, n_features)
                flat = window.reshape(-1)  # flatten

                if regime_dim > 0:
                    # Simple regime heuristic based on recent vol and price position
                    if idx >= 20:
                        recent_vol = np.std(feature_matrix[idx - 20:idx, 0])  # open_rel vol
                    else:
                        recent_vol = 0.0
                    regime_feat = np.zeros(regime_dim, dtype=np.float32)
                    regime_feat[0] = 1.0 if recent_vol > 0.5 else 0.0  # high vol regime
                    regime_feat[1] = 1.0 if feature_matrix[idx - 1, 0] > 0 else 0.0  # up bias
                    regime_feat[2] = 0.5  # confidence
                    regime_feat[3] = float(feature_matrix[idx - 1, 5])  # recent return
                    regime_feat[4] = float(feature_matrix[idx - 1, 12])  # recent vol
                    obs = np.concatenate([flat, regime_feat])
                else:
                    obs = flat

                return obs.astype(np.float32)

        return _FeatureEnv()

    return DummyVecEnv([_init])


# ── Training Runner ────────────────────────────────────────────────────

def run_trial(
    ablation_group: str,
    feature_matrix: np.ndarray,
    rewards: np.ndarray,
    window_size: int,
    regime_dim: int,
    total_timesteps: int,
    close_prices: np.ndarray,
    trial_id: int = 0,
    verbose: bool = False,
) -> dict:
    """Run a single training trial with a given feature ablation.

    Args:
        ablation_group: Feature group name to ablate, or "ALL" for baseline.
        feature_matrix: (n_bars, n_features) pre-computed feature matrix.
        rewards: (n_bars,) pre-computed reward signal.
        window_size: Number of bars per observation window.
        regime_dim: Regime feature dimension (0 to disable).
        total_timesteps: Number of training timesteps.
        close_prices: (n_bars,) close prices for metric computation.
        trial_id: Trial index (for logging).
        verbose: If True, print progress.

    Returns:
        Dict with training results and metrics.
    """
    n_features = feature_matrix.shape[1]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Trial {trial_id}: Ablation '{ablation_group}'")
        print(f"  Features per bar: {n_features}")
        print(f"  Regime dim: {regime_dim}")
        print(f"  Total timesteps: {total_timesteps}")
        print(f"{'='*60}")

    # Create environment
    env = make_env(
        feature_matrix, rewards,
        window_size=window_size,
        regime_dim=regime_dim,
    )

    # Build policy kwargs — use AdaptiveLSTMFeatureExtractor to match real training pipeline
    use_regime = regime_dim > 0
    policy_kwargs = {
        "features_extractor_class": AdaptiveLSTMFeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "window_size": window_size,
            "num_heads": 4,
            "regime_dim": regime_dim,
        },
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        "num_regimes": 5 if use_regime else 1,
        "regime_dim": regime_dim,
    }

    start_time = time.time()
    result = {
        "ablation_group": ablation_group,
        "trial_id": trial_id,
        "total_timesteps": total_timesteps,
    }

    try:
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
            regime_loss_coef=0.05 if use_regime else 0.0,
            verbose=0,
        )

        model.learn(total_timesteps=total_timesteps)
        elapsed = time.time() - start_time

        # ── Evaluate on validation split ──
        val_metrics = _evaluate(model, feature_matrix, close_prices, window_size, regime_dim)

        result.update({
            "elapsed_seconds": round(elapsed, 1),
            "completed": True,
            "status": "ok",
            **val_metrics,
        })

        if verbose:
            sharpe = val_metrics.get("sharpe_ratio", 0)
            win_rate = val_metrics.get("win_rate", 0)
            profit_factor = val_metrics.get("profit_factor", 0)
            print(f"  [OK] Completed in {elapsed:.1f}s | Sharpe: {sharpe:.3f} | WinRate: {win_rate:.1%} | PF: {profit_factor:.2f}")

    except Exception as exc:
        elapsed = time.time() - start_time
        result.update({
            "elapsed_seconds": round(elapsed, 1),
            "completed": False,
            "status": str(exc)[:200],
        })
        if verbose:
            print(f"  [FAIL] {exc}")

    return result


def _evaluate(
    model: RegimeRoutedPPO,
    feature_matrix: np.ndarray,
    close_prices: np.ndarray,
    window_size: int,
    regime_dim: int,
    val_split: float = 0.7,
) -> dict:
    """Run a validation forward pass and compute trading metrics.

    Uses the last 30% of data as validation. Simulates a simple position
    strategy using the model's action (position size) and computes
    standard trading metrics.
    """
    n = len(feature_matrix)
    val_start = int(n * val_split)

    if val_start + window_size + 50 >= n:
        return {
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "net_return_pct": 0.0,
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "validation_samples": 0,
        }

    n_features = feature_matrix.shape[1]

    # Run model over validation window
    positions = []
    with th.no_grad():
        for i in range(val_start, n - 1):
            # Build observation
            start = i - window_size
            window = feature_matrix[start:i]
            flat = window.reshape(-1)

            if regime_dim > 0:
                regime_feat = np.zeros(regime_dim, dtype=np.float32)
                if i >= 20:
                    recent_vol = np.std(feature_matrix[i - 20:i, 0])
                else:
                    recent_vol = 0.0
                regime_feat[0] = 1.0 if recent_vol > 0.5 else 0.0
                regime_feat[1] = 1.0 if feature_matrix[i - 1, 0] > 0 else 0.0
                regime_feat[2] = 0.5
                regime_feat[3] = float(feature_matrix[i - 1, 5])
                regime_feat[4] = float(feature_matrix[i - 1, 12])
                obs = np.concatenate([flat, regime_feat]).astype(np.float32)
            else:
                obs = flat.astype(np.float32)

            action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
            positions.append(float(action[0, 0]))  # position size from first action dim

    # Compute PnL from positions
    positions = np.array(positions, dtype=np.float32)
    val_returns = close_prices[val_start + 1:n] / close_prices[val_start:n - 1] - 1.0

    # Align lengths
    min_len = min(len(positions), len(val_returns))
    positions = positions[:min_len]
    val_returns = val_returns[:min_len]

    # Strategy returns = position * market return - transaction cost
    tc = 0.0002  # 2 bps per trade
    position_changes = np.abs(np.diff(positions, prepend=positions[0]))
    strategy_returns = positions * val_returns - tc * position_changes

    # ── Compute metrics ──
    total_return = np.sum(strategy_returns)
    avg_return = np.mean(strategy_returns)
    std_return = np.std(strategy_returns) + 1e-8

    sharpe = avg_return / std_return * np.sqrt(288 * 252)  # 5-min bars → annualised (24h, 252 days)

    # Trade analysis
    direction = np.sign(positions)
    trade_returns = direction[:-1] * val_returns[:-1]  # return when position is active
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    trade_count = len(trade_returns)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / max(trade_count, 1)

    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    profit_factor = abs(np.sum(wins) / max(abs(np.sum(losses)), 1e-8))

    # Max drawdown
    cumulative = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(np.min(drawdown))

    return {
        "sharpe_ratio": round(float(sharpe), 4),
        "profit_factor": round(float(profit_factor), 4),
        "max_drawdown": round(float(max_dd), 6),
        "net_return_pct": round(float(total_return * 100), 4),
        "trade_count": int(trade_count),
        "win_rate": round(float(win_rate), 6),
        "avg_win": round(float(avg_win), 6),
        "avg_loss": round(float(avg_loss), 6),
        "validation_samples": int(min_len),
    }


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real feature ablation test harness for RegimeRoutedPPO (ENGINEERED_V2)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="XAUUSDm",
        help="Trading symbol (default: XAUUSDm)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30000,
        help="Total training timesteps per trial (default: 30000)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials per ablation group (default: 1)",
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Feature groups to test (default: all). "
            f"Options: {', '.join(ABLATION_GROUPS)}"
        ),
    )
    parser.add_argument(
        "--n-bars",
        type=int,
        default=5000,
        help="Number of OHLCV bars to load (default: 5000)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Observation window size in bars (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/real_feature_ablation_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    args = parser.parse_args()

    groups = args.groups or ABLATION_GROUPS

    print(f"{'='*60}")
    print(f"REAL FEATURE ABLATION TEST HARNESS")
    print(f"{'='*60}")
    print(f"Symbol: {args.symbol}")
    print(f"Steps per trial: {args.steps}")
    print(f"Trials per group: {args.trials}")
    print(f"Groups: {', '.join(groups)}")
    print(f"Bars: {args.n_bars}")
    print(f"Window: {args.window}")
    print(f"{'='*60}")

    # ── Load data ──
    print(f"\nLoading real data for {args.symbol}...")
    df = load_real_data(symbol=args.symbol, n_bars=args.n_bars)
    close_prices = df["close"].values.astype(np.float32)
    print(f"  Got {len(df)} bars of data")

    # ── Compute reward signal ──
    print("\nComputing reward signal...")
    rewards = compute_reward_signal(close_prices)

    # ── Build feature matrices ──
    # Base features (ALL: no ablation)
    print("\nBuilding base feature matrix (ALL)...")
    all_features, n_features = build_real_features(df, symbol=args.symbol)
    print(f"  Base feature count: {n_features}")

    # Track which groups disable regime (not a feature-column ablation)
    regime_off_groups = {"NO_REGIME"}

    # ── Train each group ──
    all_results = []
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for group in groups:
        # Determine regime dimension
        if group in regime_off_groups:
            regime_dim = 0
        else:
            regime_dim = 5  # default regime features

        # Build ablated features (unless NO_REGIME which just disables regime)
        if group == "ALL" or group in regime_off_groups:
            features = all_features.copy()
        else:
            ablation = group.replace("NO_", "").lower()
            if ablation in FEATURE_GROUPS:
                features = all_features.copy()
                idx = FEATURE_GROUPS[ablation]["indices"]
                idx = [i for i in idx if 0 <= i < features.shape[1]]
                if idx:
                    features[:, idx] = 0.0
                    print(f"\nAblating '{group}': zeroed {len(idx)} columns")
            else:
                print(f"\nUnknown group '{group}', using ALL features")
                features = all_features.copy()

        for trial in range(args.trials):
            result = run_trial(
                ablation_group=group,
                feature_matrix=features,
                rewards=rewards,
                window_size=args.window,
                regime_dim=regime_dim,
                total_timesteps=args.steps,
                close_prices=close_prices,
                trial_id=trial,
                verbose=args.verbose,
            )
            all_results.append(result)

            # Save incremental results
            _save_results(all_results, args.output)

    # ── Final summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    completed = [r for r in all_results if r.get("completed")]
    failed = [r for r in all_results if not r.get("completed")]
    print(f"Total trials: {len(all_results)}")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")

    if completed:
        print(f"\n{'Group':<20} {'Sharpe':<10} {'WinRate':<10} {'PF':<10} {'Drawdown':<10} {'Trades':<8} {'Time':<8}")
        print(f"{'-'*70}")
        for r in completed:
            sharpe = r.get("sharpe_ratio", 0)
            wr = r.get("win_rate", 0)
            pf = r.get("profit_factor", 0)
            dd = r.get("max_drawdown", 0)
            tc = r.get("trade_count", 0)
            et = r.get("elapsed_seconds", 0)
            status = "OK" if r.get("completed") else "FAIL"
            print(f"{r['ablation_group']:<20} {sharpe:<10.3f} {wr:<10.2%} {pf:<10.2f} {dd:<10.4f} {tc:<8} {et:<8.1f}s {status}")

        # Best performer by Sharpe
        best = max(completed, key=lambda r: r.get("sharpe_ratio", -999))
        print(f"\nBest Sharpe: {best['ablation_group']} ({best['sharpe_ratio']:.3f})")

    print(f"\nResults saved to: {args.output}")


def _save_results(results: list[dict], path: str):
    """Save results to CSV."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    # Ensure status and key metrics are always first columns
    priority = ["ablation_group", "trial_id", "completed", "status", "sharpe_ratio",
                "profit_factor", "win_rate", "max_drawdown", "net_return_pct",
                "trade_count", "avg_win", "avg_loss"]
    for p in reversed(priority):
        if p in fieldnames:
            fieldnames.remove(p)
            fieldnames.insert(0, p)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
