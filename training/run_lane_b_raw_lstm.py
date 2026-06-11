"""
Lane B: Tamed Raw OHLCV + LSTM PPO — 3-seed walk-forward validation.

Tier 2 fixes (Jun 2026):
  - Demeaned reward: excess over 100-bar rolling mean (forces alpha, not trend-riding)
  - Stationary features: RSI(14) + MACD histogram added to OHLCV + session
  - Zero turnover cost: removed the trap that locked model into one direction
  - Holding penalty: small penalty for consecutive same-direction bars

Usage:
    python training/run_lane_b_raw_lstm.py
    python training/run_lane_b_raw_lstm.py --steps 100 --n-bars 1000 --seed 42

CLI args:
    --symbol      Trading symbol (default: XAUUSDm)
    --n-bars      Number of bars to load (default: 100000)
    --steps       Total training timesteps (default: 50000)
    --seed        Single seed to run (optional, default: run all 3 seeds)
    --timeframe   MT5 timeframe (default: M5)

Output: runtime/lane_b_raw_all_seeds.csv
"""
import sys, os, time, warnings, argparse
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "C:/supreme-chainsaw")

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

from training.run_real_feature_ablation import load_real_data
from training.taming_shared import BaseTamedEnv, MetricsCallback, compute_weight_hash, evaluate


# ── Defaults (overridden by CLI when run directly) ──
SYMBOL = "XAUUSDm"
N_BARS = 100000
N_STEPS = 50000
SEEDS = [42, 123, 456]
TIMEFRAME = "M5"


# ── Fixed config (not CLI-overridable) ──
N_FEATURES = 11         # 5 OHLCV + 4 session + 1 RSI + 1 MACD histogram
WINDOW_SIZE = 64        # 64 x 5min = 320min ~ 5.3 hours of context
HIDDEN_SIZE = 128
N_LSTM_LAYERS = 2
FEATURES_DIM = 64

# Taming parameters
TURNOVER_COST = 0.0           # ZERO — removed the trap that locked model into one direction
CONCENTRATION_PENALTY = 0.002
POSITION_SMOOTHING_ALPHA = 0.3
COOLDOWN_STEPS = 5
REWARD_HORIZON = 5            # predict 5-bar forward return
REWARD_SCALE = 1000.0         # scale raw returns (~0.001) to PPO-friendly range (~1.0)
DEMEAN_WINDOW = 100           # rolling window for reward demeaning (excess over trend)
HOLDING_PENALTY = 0.00005     # small penalty per step when holding > 0.3 position unchanged


# ── Tamed environment ──
class TamedOHLCVEnv(BaseTamedEnv):
    """Raw OHLCV + session + RSI + MACD environment with demeaned reward."""
    def __init__(self, df, window_size=WINDOW_SIZE,
                 turnover_cost=TURNOVER_COST,
                 concentration_penalty=CONCENTRATION_PENALTY,
                 smoothing_alpha=POSITION_SMOOTHING_ALPHA,
                 cooldown_steps=COOLDOWN_STEPS):
        # Store timestamps before reset_index destroys them
        self._timestamps = df.index.copy()
        self.df = df.reset_index(drop=True)
        n = len(self.df)
        super().__init__(
            window_size=window_size,
            turnover_cost=turnover_cost,
            concentration_penalty=concentration_penalty,
            smoothing_alpha=smoothing_alpha,
            cooldown_steps=cooldown_steps,
            n=n,
            n_features=N_FEATURES,
            reward_scale=REWARD_SCALE,
            holding_penalty=HOLDING_PENALTY,
        )
        self._build_features()

    def _build_features(self):
        """Build features: OHLCV log-returns, session encoding, RSI, MACD histogram, demeaned H-bar signal."""
        o = self.df["open"].values.astype(np.float64)
        h = self.df["high"].values.astype(np.float64)
        l = self.df["low"].values.astype(np.float64)
        c = self.df["close"].values.astype(np.float64)
        v = self.df["volume"].values.astype(np.float64)

        n = len(self.df)
        H = REWARD_HORIZON

        # ── OHLCV log-return features (columns 0-4) ──
        ohlcv = np.zeros((n, 5), dtype=np.float32)
        for i in range(n):
            if i == 0:
                continue
            ohlcv[i, 0] = np.log(o[i] / o[i-1]) if o[i-1] > 0 else 0.0
            ohlcv[i, 1] = np.log(h[i] / h[i-1]) if h[i-1] > 0 else 0.0
            ohlcv[i, 2] = np.log(l[i] / l[i-1]) if l[i-1] > 0 else 0.0
            ohlcv[i, 3] = np.log(c[i] / c[i-1]) if c[i-1] > 0 else 0.0
            ohlcv[i, 4] = np.log(v[i] / v[i-1]) if v[i-1] > 0 else 0.0

        # ── Session encoding (columns 5-8, NOT z-scored — already in [-1,1]) ──
        session = np.zeros((n, 4), dtype=np.float32)
        try:
            timestamps = pd.to_datetime(self._timestamps)
            for i in range(n):
                ts = timestamps[i]
                hour = ts.hour + ts.minute / 60.0
                dow = ts.dayofweek
                session[i, 0] = np.sin(2 * np.pi * hour / 24)
                session[i, 1] = np.cos(2 * np.pi * hour / 24)
                session[i, 2] = np.sin(2 * np.pi * dow / 5)
                session[i, 3] = np.cos(2 * np.pi * dow / 5)
        except Exception:
            pass  # no datetime index -> session features stay at 0

        # ── RSI(14) — column 9 ──
        rsi = np.zeros(n, dtype=np.float32)
        delta = np.diff(c, prepend=c[0])
        gains = np.maximum(delta, 0.0)
        losses = np.maximum(-delta, 0.0)
        rsi_period = 14
        for i in range(rsi_period, n):
            avg_gain = float(np.mean(gains[i - rsi_period + 1 : i + 1]))
            avg_loss = float(np.mean(losses[i - rsi_period + 1 : i + 1]))
            if avg_loss < 1e-10:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - 100.0 / (1.0 + rs)

        # ── MACD histogram — column 10 ──
        alpha12 = 2.0 / 13.0
        alpha26 = 2.0 / 27.0
        ema12 = np.zeros(n, dtype=np.float64)
        ema26 = np.zeros(n, dtype=np.float64)
        ema12[0] = c[0]
        ema26[0] = c[0]
        for i in range(1, n):
            ema12[i] = alpha12 * c[i] + (1.0 - alpha12) * ema12[i-1]
            ema26[i] = alpha26 * c[i] + (1.0 - alpha26) * ema26[i-1]
        macd_line = ema12 - ema26
        signal_line = np.zeros(n, dtype=np.float64)
        signal_line[0] = macd_line[0]
        alpha_sig = 2.0 / 10.0
        for i in range(1, n):
            signal_line[i] = alpha_sig * macd_line[i] + (1.0 - alpha_sig) * signal_line[i-1]
        macd_hist = macd_line - signal_line

        # Normalize MACD histogram by price level for comparability across regimes
        macd_hist_norm = np.zeros(n, dtype=np.float32)
        for i in range(n):
            denom = max(c[i], 1.0)
            macd_hist_norm[i] = float(macd_hist[i] / denom)

        # ── Combine all features ──
        raw_features = np.zeros((n, N_FEATURES), dtype=np.float32)
        raw_features[:, 0:5] = ohlcv       # OHLCV log returns
        raw_features[:, 5:9] = session     # session sin/cos
        raw_features[:, 9] = rsi           # RSI(14)
        raw_features[:, 10] = macd_hist_norm  # MACD histogram / price

        # Z-score normalize non-session columns (0-4, 9-10)
        self.features = raw_features.copy()
        zscore_cols = [0, 1, 2, 3, 4, 9, 10]
        for col in zscore_cols:
            col_data = raw_features[:, col]
            mean = np.mean(col_data)
            std = np.std(col_data)
            self.features[:, col] = (col_data - mean) / max(std, 1e-10)

        # ── H-bar forward return (raw) ──
        self._forward_ret = np.zeros(n, dtype=np.float32)
        for i in range(n - H):
            self._forward_ret[i] = float((c[i+H] / c[i]) - 1.0)

        # ── Rolling mean for demeaned reward (lookback only, no future leak) ──
        self._forward_mean = np.zeros(n, dtype=np.float32)
        for i in range(DEMEAN_WINDOW, n):
            self._forward_mean[i] = float(np.mean(self._forward_ret[i - DEMEAN_WINDOW : i]))

    def _raw_reward_at(self, idx):
        """Excess return over rolling mean — forces the model to find alpha, not ride the trend."""
        return self._forward_ret[idx] - self._forward_mean[idx]

    def _eval_reward_at(self, idx):
        """ACTUAL forward return for honest P&L evaluation (not demeaned)."""
        return self._forward_ret[idx]


# ── LSTM feature extractor ──
class LSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = FEATURES_DIM):
        super().__init__(observation_space, features_dim=features_dim)
        self.window_size = WINDOW_SIZE
        self.n_features = N_FEATURES

        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=HIDDEN_SIZE,
            num_layers=N_LSTM_LAYERS,
            batch_first=True,
            bidirectional=False,
            dropout=0.1 if N_LSTM_LAYERS > 1 else 0,
        )
        self.projection = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, features_dim),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.window_size, self.n_features)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        return self.projection(last)


# ── Train one seed ──
def train_seed(seed, train_df):
    """Train a PPO model with the given seed, return (model, callback)."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    def make_env():
        return TamedOHLCVEnv(train_df)

    train_env = DummyVecEnv([make_env])
    train_env = VecMonitor(train_env)

    policy_kwargs = {
        "features_extractor_class": LSTMFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": FEATURES_DIM},
        "net_arch": {"pi": [64], "vf": [64]},
    }

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,       # lower LR for financial data
        n_steps=2048,             # longer rollout for M5 bars
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,            # high entropy to encourage dynamic trading
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )

    cb = MetricsCallback()
    model.learn(total_timesteps=N_STEPS, callback=cb)
    return model, cb


# ── CLI parsing + main (only runs when executed directly) ──
def main():
    parser = argparse.ArgumentParser(description="Lane B: Tamed Raw OHLCV + LSTM (Tier 2)")
    parser.add_argument("--symbol", default=SYMBOL, help=f"MT5 symbol (default: {SYMBOL})")
    parser.add_argument("--n-bars", type=int, default=N_BARS, help=f"Bars to load (default: {N_BARS})")
    parser.add_argument("--steps", type=int, default=N_STEPS, help=f"Timesteps (default: {N_STEPS})")
    parser.add_argument("--seed", type=int, default=None, help="Single seed (default: run all 3)")
    parser.add_argument("--timeframe", default=TIMEFRAME, help=f"MT5 timeframe (default: {TIMEFRAME})")
    args = parser.parse_args()

    # Override globals with CLI values
    globals()['SYMBOL'] = args.symbol
    globals()['N_BARS'] = args.n_bars
    globals()['N_STEPS'] = args.steps
    globals()['TIMEFRAME'] = args.timeframe
    if args.seed is not None:
        globals()['SEEDS'] = [args.seed]

    # ── Begin experiment ──
    print("=" * 64)
    print("LANE B — TAMED RAW OHLCV + LSTM (Tier 2: demeaned + RSI/MACD)")
    print("=" * 64)
    print()
    print(f"  Config:")
    print(f"    timeframe:           {TIMEFRAME}")
    print(f"    n_features:          {N_FEATURES} (5 OHLCV + 4 session + RSI + MACD hist)")
    print(f"    window_size:         {WINDOW_SIZE} bars ({WINDOW_SIZE*5} min)")
    print(f"    turnover_cost:       {TURNOVER_COST} (ZERO — removed direction trap)")
    print(f"    concentration_penalty: {CONCENTRATION_PENALTY}")
    print(f"    holding_penalty:     {HOLDING_PENALTY}")
    print(f"    smoothing_alpha:     {POSITION_SMOOTHING_ALPHA}")
    print(f"    cooldown_steps:      {COOLDOWN_STEPS}")
    print(f"    reward_horizon:      {REWARD_HORIZON} bars ({REWARD_HORIZON*5} min)")
    print(f"    demean_window:       {DEMEAN_WINDOW} bars (excess over rolling mean)")
    print(f"    reward_scale:        {REWARD_SCALE}")
    print(f"    total_timesteps:     {N_STEPS}")
    print(f"    ppo n_steps:         2048")
    print(f"    ppo ent_coef:        0.03")
    print(f"    ppo learning_rate:   1e-4")
    print(f"    seeds:               {SEEDS}")
    print(f"    symbol:              {SYMBOL}")
    print(f"    n_bars:              {N_BARS}")
    print()

    # Load data
    print(f"  Loading {SYMBOL} data ({TIMEFRAME}, {N_BARS} bars)...")
    df = load_real_data(symbol=SYMBOL, n_bars=N_BARS)
    print(f"  Loaded {len(df)} bars of {SYMBOL}")
    split = int(len(df) * 0.7)
    train_df = df.iloc[:split].reset_index(drop=True)
    val_df = df.iloc[split:].reset_index(drop=True)
    print(f"  Train: {len(train_df)} bars | Val: {len(val_df)} bars")
    print()

    # Run seeds
    all_val_metrics = []
    all_train_metrics = []
    all_positions = []
    all_nw = []

    for seed in SEEDS:
        print(f"  {'='*56}")
        print(f"  Seed {seed}")
        print(f"  {'='*56}")
        t0 = time.time()

        model, cb = train_seed(seed, train_df)
        train_time = time.time() - t0

        # Training metrics
        trn_pos = np.array(cb.positions)
        train_metrics = {
            "pos_mean": float(np.mean(trn_pos)) if len(trn_pos) > 0 else 0,
            "pos_std": float(np.std(trn_pos)) if len(trn_pos) > 0 else 0,
            "long_pct": float(np.mean(trn_pos > 0.01) * 100) if len(trn_pos) > 0 else 0,
            "short_pct": float(np.mean(trn_pos < -0.01) * 100) if len(trn_pos) > 0 else 0,
            "flat_pct": float(np.mean(np.abs(trn_pos) <= 0.01) * 100) if len(trn_pos) > 0 else 0,
            "ep_reward_mean": float(np.mean(cb.episode_rewards)) if cb.episode_rewards else 0,
            "time": train_time,
        }
        all_train_metrics.append(train_metrics)

        print(f"    Training: {train_time:.1f}s ({N_STEPS/train_time:.0f} steps/s)")
        print(f"    Train long%={train_metrics['long_pct']:.1f} short%={train_metrics['short_pct']:.1f} "
              f"flat%={train_metrics['flat_pct']:.1f}")

        # Validation
        val_metrics = evaluate(model, lambda: TamedOHLCVEnv(val_df, window_size=WINDOW_SIZE),
                               turnover_cost=TURNOVER_COST)
        val_metrics["weight_hash"] = compute_weight_hash(model)
        all_val_metrics.append(val_metrics)
        all_positions.append(val_metrics["positions"])
        all_nw.append(val_metrics["net_worth"])

        print(f"    Val long%={val_metrics['long_pct']:.1f} short%={val_metrics['short_pct']:.1f} "
              f"flat%={val_metrics['flat_pct']:.1f}")
        print(f"    Val Sharpe={val_metrics['sharpe']:.2f} Return={val_metrics['total_return']:.2f}% "
              f"DD={val_metrics['max_drawdown']:.2f}%")
        print(f"    Turnover={val_metrics['turnover']:.1f}% PosStd={val_metrics['pos_std']:.4f}")
        print(f"    action_hash={val_metrics['action_hash']}  weight_hash={val_metrics['weight_hash']}")
        print()

    # ── Aggregate ──
    print()
    print("=" * 64)
    print("AGGREGATED RESULTS (mean +/- std across 3 seeds)")
    print("=" * 64)

    metrics_names = ["long_pct", "short_pct", "flat_pct", "pos_mean", "pos_std",
                     "sharpe", "total_return", "max_drawdown", "turnover", "n_steps"]

    print(f"\n  {'Metric':<20} {'Mean':>10} {'Std':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")

    for name in metrics_names:
        vals = [m[name] for m in all_val_metrics]
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        if name == "sharpe":
            print(f"  {name:<20} {mean_v:>10.2f} {std_v:>10.2f}")
        elif name in ("n_steps",):
            print(f"  {name:<20} {mean_v:>10.0f} {std_v:>10.0f}")
        else:
            print(f"  {name:<20} {mean_v:>10.2f} {std_v:>10.2f}")

    # Training metrics
    print()
    print("  TRAINING METRICS (mean across seeds):")
    mean_ep_ret = np.mean([m["ep_reward_mean"] for m in all_train_metrics])
    mean_long = np.mean([m["long_pct"] for m in all_train_metrics])
    mean_short = np.mean([m["short_pct"] for m in all_train_metrics])
    mean_flat = np.mean([m["flat_pct"] for m in all_train_metrics])
    print(f"    Ep reward mean: {mean_ep_ret:.2f}")
    print(f"    Long/Short/Flat: {mean_long:.1f}% / {mean_short:.1f}% / {mean_flat:.1f}%")

    # Save CSV
    os.makedirs("runtime", exist_ok=True)
    all_data = []
    for i, seed in enumerate(SEEDS):
        pos = all_positions[i]
        nw = all_nw[i]
        for j in range(len(pos)):
            all_data.append({"seed": seed, "step": j, "position": pos[j],
                             "net_worth": nw[min(j+1, len(nw)-1)]})
    pd.DataFrame(all_data).to_csv("runtime/lane_b_raw_all_seeds.csv", index=False)
    print(f"\n  CSV: runtime/lane_b_raw_all_seeds.csv")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        colors = ["steelblue", "darkorange", "seagreen"]
        for i, seed in enumerate(SEEDS):
            pos = all_positions[i]
            nw = all_nw[i]
            ax = axes[i, 0]
            ax.plot(pos, color=colors[i], linewidth=0.6, alpha=0.8)
            ax.axhline(0, color="gray", ls="--", lw=0.4)
            ax.set_ylabel(f"Pos (seed {seed})")
            ax.grid(alpha=0.3)
            ax2 = axes[i, 1]
            ax2.plot(nw, color=colors[i], linewidth=0.6, alpha=0.8)
            ax2.set_ylabel(f"NW (seed {seed})")
            ax2.grid(alpha=0.3)
        axes[-1, 0].set_xlabel("Validation step")
        axes[-1, 1].set_xlabel("Validation step")
        plt.suptitle("Tamed Raw OHLCV + LSTM — Tier 2: Demeaned + RSI/MACD (M5)", fontsize=13)
        plt.tight_layout()
        plt.savefig("runtime/lane_b_raw_all_seeds.png", dpi=150)
        plt.close()
        print(f"  Plot: runtime/lane_b_raw_all_seeds.png")
    except Exception as e:
        print(f"  Plot failed: {e}")

    # Conclusion
    print()
    print("=" * 64)
    print("CONCLUSION")
    print("=" * 64)
    avg_sharpe = np.mean([m["sharpe"] for m in all_val_metrics])
    avg_turnover = np.mean([m["turnover"] for m in all_val_metrics])
    avg_return = np.mean([m["total_return"] for m in all_val_metrics])
    avg_long = np.mean([m["long_pct"] for m in all_val_metrics])
    avg_short = np.mean([m["short_pct"] for m in all_val_metrics])

    print(f"  Avg Sharpe:  {avg_sharpe:.2f}")
    print(f"  Avg Return:  {avg_return:.2f}%")
    print(f"  Avg Long:    {avg_long:.1f}%")
    print(f"  Avg Short:   {avg_short:.1f}%")
    print(f"  Avg Turnover: {avg_turnover:.1f}%")

    if avg_sharpe > 0:
        print("\n  PROFITABLE! The demeaned reward + RSI/MACD features found an edge.")
    elif abs(avg_long - avg_short) < 30:
        print("\n  Model trades both directions — the direction trap is broken.")
    else:
        print("\n  Model still biased to one direction. Consider cross-asset features (DXY, VIX).")
    print()
    print("  Done.")


if __name__ == "__main__":
    main()
