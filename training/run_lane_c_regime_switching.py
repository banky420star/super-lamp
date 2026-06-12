"""
Lane C: Regime-Switching Architecture - separate models for trending vs ranging markets,
with a meta-controller to switch between them.

Architecture
------------
1. Regime classifier (ADX-based): detects ranging (0) vs trending (1) for each bar
2. Two specialized PPO-LSTM models:
   - Ranging model: trained on ranging-regime data segments
   - Trend model: trained on trending-regime data segments
3. Meta-controller: at each validation bar, classify regime, select appropriate model
4. Comparison against single-model baseline (trained on all data)

Usage
-----
    python training/run_lane_c_regime_switching.py
    python training/run_lane_c_regime_switching.py --steps 2048 --n-bars 5000 --seed 42

CLI args:
    --symbol      MT5 symbol (default: XAUUSDm)
    --n-bars      Number of bars to load (default: 100000)
    --steps       Total training timesteps (default: 50000)
    --seed        Single seed for reproducibility (optional, default: run 3 seeds)
    --timeframe   MT5 timeframe (default: 1m)

Output: runtime/lane_c_results.csv
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
from training.regime_classifier import classify_regime, get_regime_summary

# -- Defaults --
SYMBOL = "XAUUSDm"
N_BARS = 100000
N_STEPS = 50000
SEEDS = [42, 123, 456]

WINDOW_SIZE = 64
HIDDEN_SIZE = 128
N_LSTM_LAYERS = 2
FEATURES_DIM = 64
REWARD_HORIZON = 5

TURNOVER_COST = 0.0003
CONCENTRATION_PENALTY = 0.0
SMOOTHING_ALPHA = 0.3
COOLDOWN_STEPS = 5
DISCRETE = True
ENT_COEF = 0.05
REWARD_SCALE = 1000.0
INACTIVITY_PENALTY = 0.0003
HOLDING_PENALTY = 0.0

ADX_PERIOD = 14
BB_PERIOD = 20
TREND_THRESHOLD = 25
RANGING_THRESHOLD = 20
N_FEATURES = 8

# Annualization factor by timeframe (for honest Sharpe)
_TF_ANNUAL = {"M1": 252*1440, "M5": 252*288, "M15": 252*96, "M30": 252*48,
              "H1": 252*24, "H4": 252*6, "D1": 252}


# -- Environment helpers (mirrored from Lane B) --

def make_inverted_df(df):
    """Create mirror-image dataframe for market symmetry."""
    inverted = df.copy()
    mean_price = float(df["close"].mean())
    for col in ["open", "high", "low", "close"]:
        inverted[col] = 2.0 * mean_price - df[col].values
    h = inverted["high"].values.copy()
    l = inverted["low"].values.copy()
    inverted["high"] = np.maximum(h, l)
    inverted["low"] = np.minimum(h, l)
    return inverted


def compute_rsi(prices, period=14):
    """Relative Strength Index with Wilder smoothing."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.full(len(prices), np.nan)
    avg_loss = np.full(len(prices), np.nan)
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    return 100.0 - 100.0 / (1.0 + rs)


def compute_macd(prices, fast=12, slow=26, signal=9):
    """MACD histogram normalized by price."""
    ema_fast = pd.Series(prices).ewm(span=fast).mean().values
    ema_slow = pd.Series(prices).ewm(span=slow).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal).mean().values
    histogram = macd_line - signal_line
    return histogram / np.maximum(prices, 1e-10), macd_line, signal_line


class RegimeAwareEnv(BaseTamedEnv):
    """
    OHLCV environment with regime features (regime_score) added to observation.
    Observation = 5 OHLCV log-returns + RSI(14) + MACD histogram + regime_score = 8 features
    """
    def __init__(self, df, *, regimes=None, regime_scores=None,
                 window_size=WINDOW_SIZE, reward_horizon=REWARD_HORIZON,
                 turnover_cost=TURNOVER_COST, concentration_penalty=CONCENTRATION_PENALTY,
                 smoothing_alpha=SMOOTHING_ALPHA, cooldown_steps=COOLDOWN_STEPS,
                 reward_scale=REWARD_SCALE, holding_penalty=HOLDING_PENALTY,
                 inactivity_penalty=INACTIVITY_PENALTY, discrete=DISCRETE):

        self.df = df.reset_index(drop=True)
        self.reward_horizon = reward_horizon
        n = len(self.df)
        self._regimes = regimes if regimes is not None else np.zeros(n, dtype=np.int32)
        self._regime_scores = regime_scores if regime_scores is not None else np.zeros(n, dtype=np.float32)

        super().__init__(
            window_size=window_size,
            turnover_cost=turnover_cost,
            concentration_penalty=concentration_penalty,
            smoothing_alpha=smoothing_alpha,
            cooldown_steps=cooldown_steps,
            n=n,
            n_features=N_FEATURES,
            reward_scale=reward_scale,
            holding_penalty=holding_penalty,
            inactivity_penalty=inactivity_penalty,
            discrete=discrete,
        )
        self._build_features()

    def _build_features(self):
        close = self.df["close"].values.astype(np.float64)
        n = len(self.df)

        # 1. OHLCV log-returns
        raw_features = np.zeros((n, N_FEATURES), dtype=np.float64)
        cols = ["open", "high", "low", "close", "volume"]
        for j, col in enumerate(cols):
            raw = self.df[col].values.astype(np.float64)
            safe = np.maximum(raw, 1e-10)
            lr = np.full(n, 0.0)
            lr[1:] = np.log(safe[1:] / safe[:-1])
            raw_features[:, j] = lr

        # 2. RSI(14) normalized
        rsi = compute_rsi(close, 14)
        raw_features[:, 5] = (rsi - 50.0) / 50.0

        # 3. MACD histogram
        macd_hist_norm, _, _ = compute_macd(close, 12, 26, 9)
        raw_features[:, 6] = macd_hist_norm

        # 4. Regime score
        raw_features[:, 7] = self._regime_scores

        # Z-score normalize
        self.features = np.zeros_like(raw_features, dtype=np.float32)
        for col in range(N_FEATURES):
            col_data = raw_features[:, col]
            finite = np.isfinite(col_data)
            if np.any(finite):
                mean = np.mean(col_data[finite])
                std = np.std(col_data[finite])
                self.features[:, col] = np.where(finite, (col_data - mean) / max(std, 1e-10), 0.0)

        # Forward returns
        close_safe = np.maximum(close, 1e-10)
        self._forward_ret = np.zeros(n, dtype=np.float64)
        self._forward_ret[:-self.reward_horizon] = (
            close_safe[self.reward_horizon:] / close_safe[:-self.reward_horizon] - 1.0
        )

    def _raw_reward_at(self, idx):
        return float(self._forward_ret[idx]) if idx < len(self._forward_ret) else 0.0

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        idx = self.idx - 1
        info["regime"] = int(self._regimes[idx]) if 0 <= idx < len(self._regimes) else 0
        info["regime_score"] = float(self._regime_scores[idx]) if 0 <= idx < len(self._regime_scores) else 0.0
        return obs, reward, done, truncated, info



# -- LSTM Feature Extractor (same architecture as Lane B) --

class LSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = FEATURES_DIM):
        super().__init__(observation_space, features_dim=features_dim)
        self.window_size = WINDOW_SIZE
        total_dim = observation_space.shape[0]
        self.n_features = total_dim // self.window_size
        self.lstm = nn.LSTM(
            input_size=self.n_features, hidden_size=HIDDEN_SIZE,
            num_layers=N_LSTM_LAYERS, batch_first=True, bidirectional=False,
            dropout=0.1 if N_LSTM_LAYERS > 1 else 0,
        )
        self.projection = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, features_dim), nn.LayerNorm(features_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.window_size, self.n_features)
        lstm_out, _ = self.lstm(x)
        return self.projection(lstm_out[:, -1, :])


# -- Regime segment extraction --

def extract_regime_segments(features, forward_ret, regimes, regime_id, min_segment_len=256):
    """
    Extract contiguous training segments where regime == regime_id.
    Returns concatenated features and forward_ret arrays, or (None, None).
    """
    seg_feats, seg_rets = [], []
    in_seg, start = False, 0
    for i in range(len(regimes)):
        if regimes[i] == regime_id and not in_seg:
            start, in_seg = i, True
        elif regimes[i] != regime_id and in_seg:
            if i - start >= min_segment_len:
                seg_feats.append(features[start:i].copy())
                seg_rets.append(forward_ret[start:i].copy())
            in_seg = False
    if in_seg and len(regimes) - start >= min_segment_len:
        seg_feats.append(features[start:].copy())
        seg_rets.append(forward_ret[start:].copy())
    if not seg_feats:
        return None, None
    return np.concatenate(seg_feats), np.concatenate(seg_rets)


# -- Meta-controller evaluation --

def meta_controller_evaluate(trend_model, ranging_model, env_factory,
                             val_regimes, turnover_cost=0.0, annualization_factor=252 * 288):
    """Evaluate with regime-switching meta-controller."""
    import hashlib
    env = env_factory()
    obs, _ = env.reset()
    positions, net_worth, regime_choices = [], [10000.0], []
    done, step, prev_position = False, 0, 0.0

    while not done and step < 1_000_000:
        current_idx = env.idx
        regime = int(val_regimes[current_idx]) if current_idx < len(val_regimes) else 0
        if regime == 1:
            action, _ = trend_model.predict(obs, deterministic=True)
        else:
            action, _ = ranging_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        pos_now = info["position"]
        tc = turnover_cost * abs(pos_now - prev_position)
        actual_return = info.get("eval_reward", info["raw_reward"])
        nw_growth = 1.0 + actual_return * pos_now - tc
        net_worth.append(net_worth[-1] * max(nw_growth, 0.0))
        prev_position = pos_now
        positions.append(pos_now)
        regime_choices.append(regime)
        step += 1

    pos, nw = np.array(positions), np.array(net_worth)
    total_ret = (nw[-1] / nw[0] - 1) * 100 if len(nw) > 1 else 0.0
    nw_returns = np.diff(nw) / nw[:-1]
    sharpe = 0.0
    if len(nw_returns) > 1 and np.std(nw_returns) > 1e-10:
        sharpe = float(np.mean(nw_returns) / np.std(nw_returns) * np.sqrt(annualization_factor))
    turnover = float(np.mean(np.abs(np.diff(pos)) > 0.01) * 100) if len(pos) > 1 else 0.0
    peak = np.maximum.accumulate(nw)
    max_dd = float(np.min((nw - peak) / peak * 100)) if len(nw) > 0 else 0.0
    ah = hashlib.md5(pos.tobytes()).hexdigest()[:12] if len(pos) > 0 else "none"
    rc = np.array(regime_choices)
    return {
        "pos_mean": float(np.mean(pos)), "pos_std": float(np.std(pos)),
        "long_pct": float(np.mean(pos > 0.01) * 100),
        "short_pct": float(np.mean(pos < -0.01) * 100),
        "flat_pct": float(np.mean(np.abs(pos) <= 0.01) * 100),
        "sharpe": sharpe, "total_return": total_ret, "max_drawdown": max_dd,
        "turnover": turnover, "n_steps": len(pos), "action_hash": ah,
        "positions": pos, "net_worth": nw,
        "regime_choices_pct_trend": float(np.mean(rc == 1) * 100),
    }



# -- Config display --

def print_config():
    print(f"  Window size  : {WINDOW_SIZE}")
    print(f"  N features   : {N_FEATURES} (5 OHLCV + RSI + MACD + regime_score)")
    print(f"  Hidden size  : {HIDDEN_SIZE}")
    print(f"  LSTM layers  : {N_LSTM_LAYERS}")
    print(f"  Actions      : Discrete (Long/Flat/Short)")
    print(f"  Ent coef     : {ENT_COEF}")
    print(f"  Turnover cost: {TURNOVER_COST}")
    print(f"  Inactivity   : {INACTIVITY_PENALTY}")
    print(f"  ADX period   : {ADX_PERIOD} (trend>={TREND_THRESHOLD}, range<={RANGING_THRESHOLD})")
    print(f"  Market sym   : True (inverted dataset)")
    print()



# -- Main --

def main():
    parser = argparse.ArgumentParser(description="Lane C: Regime-Switching")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--n-bars", type=int, default=N_BARS)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeframe", default="M5")
    args = parser.parse_args()
    globals()['SYMBOL'] = args.symbol
    globals()['N_BARS'] = args.n_bars
    globals()['N_STEPS'] = args.steps
    if args.seed is not None:
        globals()['SEEDS'] = [args.seed]

    print("=" * 64)
    print("LANE C - REGIME-SWITCHING ARCHITECTURE")
    print("=" * 64)
    print()
    print_config()

    print("[1] Loading data...")
    df = load_real_data(symbol=SYMBOL, n_bars=N_BARS, timeframe=args.timeframe)
    print(f"  Loaded {len(df)} bars")
    print()

    print("[2] Classifying regimes (ADX-based)...")
    regimes, regime_scores = classify_regime(
        df, adx_period=ADX_PERIOD, bb_period=BB_PERIOD,
        trend_threshold=TREND_THRESHOLD, ranging_threshold=RANGING_THRESHOLD
    )
    regime_dist = get_regime_summary(regimes)
    print()

    split = int(len(df) * 0.7)
    print(f"[3] Split: {split} train / {len(df) - split} val")
    train_df = df.iloc[:split].copy().reset_index(drop=True)
    val_df = df.iloc[split:].copy().reset_index(drop=True)
    train_regimes = regimes[:split]
    val_regimes = regimes[split:]
    train_scores = regime_scores[:split]
    val_scores = regime_scores[split:]

    tr_rng = np.sum(train_regimes == 0)
    tr_trd = np.sum(train_regimes == 1)
    print(f"  Train: {tr_rng} ranging ({tr_rng / len(train_regimes) * 100:.1f}%), "
          f"{tr_trd} trending ({tr_trd / len(train_regimes) * 100:.1f}%)")
    print()

    print("[4] Building features with market symmetry...")
    inverted_df = make_inverted_df(train_df)
    combined_df = pd.concat([train_df, inverted_df], ignore_index=True)
    combined_regimes = np.concatenate([train_regimes, train_regimes])
    combined_scores = np.concatenate([train_scores, -train_scores])  # negate for inverted half
    print(f"  Combined: {len(combined_df)} bars")
    print()


    all_results = []

    for seed in SEEDS:
        print(f"  {'='*56}")
        print(f"  Seed {seed}")
        print(f"  {'='*56}")
        t0 = time.time()
        np.random.seed(seed)
        torch.manual_seed(seed)

        # --- Train ALL-DATA model (baseline) ---
        print(f"  [A] Training ALL-DATA model...")
        
        # Create env to get features, then extract regime segments
        all_env = RegimeAwareEnv(combined_df, regimes=combined_regimes, regime_scores=combined_scores)
        all_features = all_env.features
        all_forward_ret = all_env._forward_ret
        
        model_all = None
        model_ranging = None
        model_trending = None
        
        # Train all-data model
        if len(all_features) >= WINDOW_SIZE + 10:
            all_dummy = pd.DataFrame({
                "open": np.zeros(len(all_features)), "high": np.zeros(len(all_features)),
                "low": np.zeros(len(all_features)), "close": np.zeros(len(all_features)),
                "volume": np.ones(len(all_features)),
            })
            
            def make_all_env():
                e = RegimeAwareEnv(all_dummy, regimes=combined_regimes, regime_scores=combined_scores)
                e.features = all_features.astype(np.float32)
                e._forward_ret = all_forward_ret
                return e
            
            all_train_env = VecMonitor(DummyVecEnv([make_all_env]))
            ppo_kwargs = {
                "policy_kwargs": {
                    "features_extractor_class": LSTMFeatureExtractor,
                    "features_extractor_kwargs": {"features_dim": FEATURES_DIM},
                    "net_arch": {"pi": [64], "vf": [64]},
                },
                "learning_rate": 3e-4, "n_steps": 1024, "batch_size": 64,
                "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95,
                "clip_range": 0.2, "ent_coef": ENT_COEF, "vf_coef": 0.5,
                "max_grad_norm": 0.5, "seed": seed, "verbose": 0,
            }
            model_all = PPO("MlpPolicy", all_train_env, **ppo_kwargs)
            t_a = time.time()
            cb_a = MetricsCallback()
            all_steps = min(N_STEPS, max(1024, len(all_features) * 2))
            model_all.learn(total_timesteps=all_steps, callback=cb_a)
            a_pos = np.array(cb_a.positions)
            print(f"    All-data: {time.time()-t_a:.1f}s ({all_steps/(time.time()-t_a):.0f} sps)")
            if len(a_pos) > 0:
                print(f"    Train L/S/F: {np.mean(a_pos>0.01)*100:.1f}/{np.mean(a_pos<-0.01)*100:.1f}/{np.mean(np.abs(a_pos)<=0.01)*100:.1f}%")

        # --- Train RANGING model ---
        print(f"  [B] Training RANGING model...")
        ranging_features, ranging_ret = extract_regime_segments(
            all_features, all_forward_ret, combined_regimes, 0
        )
        print(f"    Ranging segments: {len(ranging_features) if ranging_features is not None else 0} bars")

        if ranging_features is not None and len(ranging_features) >= WINDOW_SIZE + 10:
            rng_dummy = pd.DataFrame({
                "open": np.zeros(len(ranging_features)), "high": np.zeros(len(ranging_features)),
                "low": np.zeros(len(ranging_features)), "close": np.zeros(len(ranging_features)),
                "volume": np.ones(len(ranging_features)),
            })
            rng_regimes = np.zeros(len(ranging_features), dtype=np.int32)
            rng_scores = np.zeros(len(ranging_features), dtype=np.float32)
            
            def make_rng_env():
                e = RegimeAwareEnv(rng_dummy, regimes=rng_regimes, regime_scores=rng_scores)
                e.features = ranging_features.astype(np.float32)
                e._forward_ret = ranging_ret
                return e
            
            rng_train_env = VecMonitor(DummyVecEnv([make_rng_env]))
            rng_kwargs = dict(ppo_kwargs)
            rng_kwargs["seed"] = seed + 1000
            model_ranging = PPO("MlpPolicy", rng_train_env, **rng_kwargs)
            t_r = time.time()
            cb_r = MetricsCallback()
            rng_steps = min(N_STEPS // 2, max(1024, len(ranging_features) * 2))
            model_ranging.learn(total_timesteps=rng_steps, callback=cb_r)
            r_pos = np.array(cb_r.positions)
            print(f"    Ranging: {time.time()-t_r:.1f}s ({rng_steps/(time.time()-t_r):.0f} sps)")
            if len(r_pos) > 0:
                print(f"    Train L/S/F: {np.mean(r_pos>0.01)*100:.1f}/{np.mean(r_pos<-0.01)*100:.1f}/{np.mean(np.abs(r_pos)<=0.01)*100:.1f}%")

        # --- Train TRENDING model ---
        print(f"  [C] Training TRENDING model...")
        trending_features, trending_ret = extract_regime_segments(
            all_features, all_forward_ret, combined_regimes, 1
        )
        print(f"    Trending segments: {len(trending_features) if trending_features is not None else 0} bars")

        if trending_features is not None and len(trending_features) >= WINDOW_SIZE + 10:
            trd_dummy = pd.DataFrame({
                "open": np.zeros(len(trending_features)), "high": np.zeros(len(trending_features)),
                "low": np.zeros(len(trending_features)), "close": np.zeros(len(trending_features)),
                "volume": np.ones(len(trending_features)),
            })
            trd_regimes = np.ones(len(trending_features), dtype=np.int32)
            trd_scores = np.zeros(len(trending_features), dtype=np.float32)
            
            def make_trd_env():
                e = RegimeAwareEnv(trd_dummy, regimes=trd_regimes, regime_scores=trd_scores)
                e.features = trending_features.astype(np.float32)
                e._forward_ret = trending_ret
                return e
            
            trd_train_env = VecMonitor(DummyVecEnv([make_trd_env]))
            trd_kwargs = dict(ppo_kwargs)
            trd_kwargs["seed"] = seed + 2000
            model_trending = PPO("MlpPolicy", trd_train_env, **trd_kwargs)
            t_t = time.time()
            cb_t = MetricsCallback()
            trd_steps = min(N_STEPS // 2, max(1024, len(trending_features) * 2))
            model_trending.learn(total_timesteps=trd_steps, callback=cb_t)
            t_pos = np.array(cb_t.positions)
            print(f"    Trending: {time.time()-t_t:.1f}s ({trd_steps/(time.time()-t_t):.0f} sps)")
            if len(t_pos) > 0:
                print(f"    Train L/S/F: {np.mean(t_pos>0.01)*100:.1f}/{np.mean(t_pos<-0.01)*100:.1f}/{np.mean(np.abs(t_pos)<=0.01)*100:.1f}%")


        # --- Evaluate ---
        print(f"  [D] Evaluating...")

        def make_val_env():
            return RegimeAwareEnv(val_df, regimes=val_regimes, regime_scores=val_scores)

        # Evaluate ALL-DATA model
        val_all = evaluate(model_all, make_val_env, turnover_cost=TURNOVER_COST, annualization_factor=_TF_ANNUAL.get(args.timeframe.upper(), 252*288)) if model_all else None
        if val_all:
            print(f"    All-data: L={val_all['long_pct']:.1f}/S={val_all['short_pct']:.1f}/F={val_all['flat_pct']:.1f}%  "
                  f"SR={val_all['sharpe']:.2f}  Ret={val_all['total_return']:.2f}%  "
                  f"DD={val_all['max_drawdown']:.2f}%  TO={val_all['turnover']:.1f}%")

        # Evaluate META-CONTROLLER
        val_meta = None
        if model_ranging is not None and model_trending is not None:
            val_meta = meta_controller_evaluate(
                model_trending, model_ranging, make_val_env, val_regimes, turnover_cost=TURNOVER_COST, annualization_factor=_TF_ANNUAL.get(args.timeframe.upper(), 252*288)
            )
            print(f"    Meta-ctrl: L={val_meta['long_pct']:.1f}/S={val_meta['short_pct']:.1f}/F={val_meta['flat_pct']:.1f}%  "
                  f"SR={val_meta['sharpe']:.2f}  Ret={val_meta['total_return']:.2f}%  "
                  f"DD={val_meta['max_drawdown']:.2f}%  TO={val_meta['turnover']:.1f}%  "
                  f"TrendUsed={val_meta['regime_choices_pct_trend']:.1f}%")

        # Evaluate RANGING-ONLY model on all val data
        val_ranging = evaluate(model_ranging, make_val_env, turnover_cost=TURNOVER_COST, annualization_factor=_TF_ANNUAL.get(args.timeframe.upper(), 252*288)) if model_ranging else None
        if val_ranging:
            print(f"    Ranging-o: L={val_ranging['long_pct']:.1f}/S={val_ranging['short_pct']:.1f}/F={val_ranging['flat_pct']:.1f}%  "
                  f"SR={val_ranging['sharpe']:.2f}  Ret={val_ranging['total_return']:.2f}%")

        # Evaluate TRENDING-ONLY model on all val data
        val_trending = evaluate(model_trending, make_val_env, turnover_cost=TURNOVER_COST, annualization_factor=_TF_ANNUAL.get(args.timeframe.upper(), 252*288)) if model_trending else None
        if val_trending:
            print(f"    Trending-o: L={val_trending['long_pct']:.1f}/S={val_trending['short_pct']:.1f}/F={val_trending['flat_pct']:.1f}%  "
                  f"SR={val_trending['sharpe']:.2f}  Ret={val_trending['total_return']:.2f}%")


        # Store results
        result = {
            "seed": seed,
            "all_sharpe": val_all["sharpe"] if val_all else None,
            "all_return": val_all["total_return"] if val_all else None,
            "all_long_pct": val_all["long_pct"] if val_all else None,
            "all_short_pct": val_all["short_pct"] if val_all else None,
            "all_flat_pct": val_all["flat_pct"] if val_all else None,
            "all_turnover": val_all["turnover"] if val_all else None,
            "all_maxdd": val_all["max_drawdown"] if val_all else None,
            "ranging_sharpe": val_ranging["sharpe"] if val_ranging else None,
            "ranging_return": val_ranging["total_return"] if val_ranging else None,
            "ranging_long_pct": val_ranging["long_pct"] if val_ranging else None,
            "ranging_short_pct": val_ranging["short_pct"] if val_ranging else None,
            "ranging_flat_pct": val_ranging["flat_pct"] if val_ranging else None,
            "trending_sharpe": val_trending["sharpe"] if val_trending else None,
            "trending_return": val_trending["total_return"] if val_trending else None,
            "trending_long_pct": val_trending["long_pct"] if val_trending else None,
            "trending_short_pct": val_trending["short_pct"] if val_trending else None,
            "trending_flat_pct": val_trending["flat_pct"] if val_trending else None,
            "meta_sharpe": val_meta["sharpe"] if val_meta else None,
            "meta_return": val_meta["total_return"] if val_meta else None,
            "meta_long_pct": val_meta["long_pct"] if val_meta else None,
            "meta_short_pct": val_meta["short_pct"] if val_meta else None,
            "meta_flat_pct": val_meta["flat_pct"] if val_meta else None,
            "meta_turnover": val_meta["turnover"] if val_meta else None,
            "meta_trend_pct": val_meta["regime_choices_pct_trend"] if val_meta else None,
        }
        all_results.append(result)
        total_time = time.time() - t0
        print(f"  Seed {seed} done in {total_time:.1f}s")
        print()

    # Aggregate results
    print()
    print("=" * 64)
    print("AGGREGATED - LANE C (Regime-Switching)")
    print("=" * 64)

    results_df = pd.DataFrame(all_results)

    models_to_show = [
        ("ALL-DATA", "all"),
        ("RANGING-ONLY", "ranging"),
        ("TRENDING-ONLY", "trending"),
        ("META-CONTROLLER", "meta"),
    ]

    for model_name, prefix in models_to_show:
        cols = [f"{prefix}_sharpe", f"{prefix}_return", f"{prefix}_maxdd",
                f"{prefix}_long_pct", f"{prefix}_short_pct", f"{prefix}_flat_pct",
                f"{prefix}_turnover"]
        avail = [c for c in cols if c in results_df.columns]
        vals = results_df[avail].dropna()
        if len(vals) == 0:
            continue
        print(f"\n  {model_name}:")
        print(f"    {'Metric':<20} {'Mean':>10} {'Std':>10}  {'N':>4}")
        print(f"    {'-'*20} {'-'*10} {'-'*10} {'-'*4}")
        for col in avail:
            v = vals[col].values
            short = col.replace(f"{prefix}_", "")
            print(f"    {short:<20} {np.nanmean(v):>10.2f} {np.nanstd(v):>10.2f}  {len(v):>4d}")

    # Head-to-head
    print(f"\n  {'='*40}")
    print(f"  HEAD-TO-HEAD: META vs ALL-DATA")
    print(f"  {'='*40}")
    print(f"\n    {'Metric':<20} {'Meta':>10} {'All':>10} {'Diff':>10} {'Better':>10}")
    print(f"    {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for metric, label in [("sharpe", "Sharpe"), ("return", "Return%"),
                           ("maxdd", "MaxDD%"), ("long_pct", "Long%"),
                           ("short_pct", "Short%"), ("flat_pct", "Flat%"),
                           ("turnover", "Turnover%")]:
        mc, ac = f"meta_{metric}", f"all_{metric}"
        if mc in results_df.columns and ac in results_df.columns:
            mv = results_df[mc].dropna().values
            av = results_df[ac].dropna().values
            if len(mv) > 0 and len(av) > 0:
                mm, am = np.nanmean(mv), np.nanmean(av)
                diff = mm - am
                better = "META" if diff > 0 and metric in ("sharpe", "return", "maxdd") else ("ALL" if diff < 0 and metric in ("sharpe", "return", "maxdd") else "--")
                print(f"    {label:<20} {mm:>10.2f} {am:>10.2f} {diff:>+10.2f} {better:>10}")

    # Conclusion
    print(f"\n  CONCLUSION:")
    if "meta_sharpe" in results_df.columns and "all_sharpe" in results_df.columns:
        ms = results_df["meta_sharpe"].dropna().values
        ma = results_df["all_sharpe"].dropna().values
        if len(ms) > 0 and len(ma) > 0:
            meta_avg = np.nanmean(ms)
            all_avg = np.nanmean(ma)
            if meta_avg > all_avg:
                print(f"    OK: Regime-switching outperforms (Meta SR={meta_avg:.2f} vs All SR={all_avg:.2f})")
            else:
                print(f"    X: Single model still better (All SR={all_avg:.2f} vs Meta SR={meta_avg:.2f})")

    # Save aggregate CSV
    os.makedirs("runtime", exist_ok=True)
    results_df.to_csv("runtime/lane_c_results.csv", index=False)
    print(f"\n  CSV: runtime/lane_c_results.csv")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
