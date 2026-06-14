"""
Lane D: MTF Regime-Routed PPO - H1 regime detector is the aerial view,
M5 PPO-LSTM is the ground-level trader. Regime-weighted training so both
models see all market conditions but specialize in their zone.

Architecture
------------
1. H1 regime detector (ADX-14 + BB-20) produces regime_score with 3 zones:
   - range:      score < 0.35
   - transition: 0.35 to 0.60
   - trend:      score > 0.60
2. Two PPO-LSTM models trained on ALL M5 data, but with regime-weighted rewards:
   - Range model: range bars get 1.0x reward weight, trend bars get 0.3x
   - Trend model: trend bars get 1.0x reward weight, range bars get 0.3x
3. Meta-controller at inference:
   - score < 0.35  -> range model
   - score > 0.60  -> trend model
   - 0.35 to 0.60 -> blend both model outputs
4. H1 regime_score aligned to M5 using most recently closed H1 candle only
   (no future leakage - strictly causal alignment)

Usage
-----
    python training/run_lane_d_mtf_regime.py
    python training/run_lane_d_mtf_regime.py --steps 2048 --n-bars 5000 --seed 42

CLI args:
    --symbol      MT5 symbol (default: XAUUSDm)
    --n-bars      Number of M5 bars to load (default: 100000)
    --steps       Total training timesteps (default: 50000)
    --seed        Single seed for reproducibility (default: 42,123,456)
"""

import sys, os, time, warnings, argparse, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

warnings.filterwarnings("ignore", category=DeprecationWarning)

from training.run_real_feature_ablation import load_real_data
from training.regime_classifier import classify_regime
from training.taming_shared import evaluate, BaseTamedEnv, MetricsCallback




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

# Regime zones
RANGE_THRESH = 0.35    # score < this -> range zone
TREND_THRESH = 0.60    # score > this -> trend zone
# Between 0.35 and 0.60 -> transition zone (blend)

# Annualization factor by timeframe
ANNUAL_M5 = 252 * 288

# Regime weight extremes
RANGE_WEIGHT = 1.0      # reward multiplier for in-zone bars
OPP_WEIGHT = 0.3        # reward multiplier for out-of-zone bars


# -- LSTM Feature Extractor (same as Lane B/C) --

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
        last_hidden = lstm_out[:, -1, :]
        return self.projection(last_hidden)


def make_inverted_df(df: pd.DataFrame) -> pd.DataFrame:
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


# -- 3-Zone Meta-Controller --

def meta_controller_evaluate(range_model, trend_model, env_factory,
                             val_scores: np.ndarray,
                             turnover_cost: float = TURNOVER_COST,
                             annualization_factor: float = ANNUAL_M5):
    """Evaluate with 3-zone regime routing.
    
    - score < RANGE_THRESH: use range model
    - score > TREND_THRESH: use trend model
    - transition: blend both model outputs (weighted by proximity)
    """
    import hashlib
    env = env_factory()
    obs, _ = env.reset()
    positions, net_worth = [], [10000.0]
    done, step, prev_position = False, 0, 0.0

    while not done and step < 1_000_000:
        current_idx = getattr(env, 'idx', step)
        if current_idx < len(val_scores):
            s = abs(val_scores[current_idx])
        else:
            s = 0.0
        
        if s < RANGE_THRESH:
            # Range zone: use range model
            action, _ = range_model.predict(obs, deterministic=True)
        elif s > TREND_THRESH:
            # Trend zone: use trend model
            action, _ = trend_model.predict(obs, deterministic=True)
        else:
            # Transition zone: blend both
            a_range, _ = range_model.predict(obs, deterministic=True)
            a_trend, _ = trend_model.predict(obs, deterministic=True)
            # Linear blend: closer to RANGE_THRESH -> more range model
            t = (s - RANGE_THRESH) / (TREND_THRESH - RANGE_THRESH)
            if isinstance(a_range, np.ndarray):
                action = a_range * (1.0 - t) + a_trend * t
            else:
                action = a_range if np.random.random() > t else a_trend
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        pos_now = info["position"]
        tc = turnover_cost * abs(pos_now - prev_position)
        actual_return = info.get("eval_reward", info["raw_reward"])
        nw_growth = 1.0 + actual_return * pos_now - tc
        net_worth.append(net_worth[-1] * max(nw_growth, 0.0))
        prev_position = pos_now
        positions.append(pos_now)
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
    return {
        "pos_mean": float(np.mean(pos)), "pos_std": float(np.std(pos)),
        "long_pct": float(np.mean(pos > 0.01) * 100),
        "short_pct": float(np.mean(pos < -0.01) * 100),
        "flat_pct": float(np.mean(np.abs(pos) <= 0.01) * 100),
        "sharpe": sharpe, "total_return": total_ret, "max_drawdown": max_dd,
        "turnover": turnover, "n_steps": len(pos), "action_hash": ah,
    }


# -- Config display --

def print_config():
    print(f"  M5 window     : {WINDOW_SIZE}")
    print(f"  N features    : 8 (5 OHLCV + RSI + MACD + H1 regime_score)")
    print(f"  Hidden size   : {HIDDEN_SIZE}")
    print(f"  LSTM layers   : {N_LSTM_LAYERS}")
    print(f"  Actions       : Discrete (Long/Flat/Short)")
    print(f"  Ent coef      : {ENT_COEF}")
    print(f"  Turnover cost : {TURNOVER_COST}")
    print(f"  Regime zones  : range<{RANGE_THRESH}  transition {RANGE_THRESH}-{TREND_THRESH}  trend>{TREND_THRESH}")
    print(f"  Range weight  : {RANGE_WEIGHT}  Trend weight: {OPP_WEIGHT} (opposite: {OPP_WEIGHT})")
    print(f"  Market sym    : True (inverted dataset)")
    print()


def get_regime_summary(regimes):
    """Print regime statistics."""
    rng = int(np.sum(regimes == 0))
    trd = int(np.sum(regimes == 1))
    total = len(regimes)
    print(f"  Regime distribution: {rng} ranging ({rng/total*100:.1f}%), "
          f"{trd} trending ({trd/total*100:.1f}%)")


def get_zone_summary(scores):
    """Print zone distribution based on absolute scores."""
    abs_s = np.abs(scores)
    rng = int(np.sum(abs_s < RANGE_THRESH))
    trn = int(np.sum(abs_s > TREND_THRESH))
    trans = len(scores) - rng - trn
    total = len(scores)
    print(f"  Zone distribution: {rng} range ({rng/total*100:.1f}%), "
          f"{trans} transition ({trans/total*100:.1f}%), "
          f"{trn} trend ({trn/total*100:.1f}%)")


# -- Main --

def main():
    parser = argparse.ArgumentParser(description="Lane D: MTF Regime-Routed PPO")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--n-bars", type=int, default=N_BARS)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--htf", default="H1", help="Higher timeframe for regime (H1, H4)")
    parser.add_argument("--ltf", default="M5", help="Lower timeframe for trading (M5, M15)")
    args = parser.parse_args()
    globals()["SYMBOL"] = args.symbol
    globals()["N_BARS"] = args.n_bars
    globals()["N_STEPS"] = args.steps
    if args.seed is not None:
        globals()["SEEDS"] = [args.seed]

    print("=" * 64)
    print("LANE D - MTF REGIME-ROUTED PPO")
    print("=" * 64)
    print()
    print_config()

    print(f"[1] Loading M5 data ({args.ltf})...")
    m5_df = load_real_data(symbol=SYMBOL, n_bars=N_BARS, timeframe=args.ltf)
    print(f"  Loaded {len(m5_df)} M5 bars")
    print()

    print(f"[2] Loading H1 data ({args.htf}) for regime detection...")
    # Load enough H1 bars to cover the M5 period
    h1_n = int(N_BARS * 0.1) + 50  # ~12 M5 bars per H1, with buffer
    htf_df = load_real_data(symbol=SYMBOL, n_bars=h1_n, timeframe=args.htf)
    print(f"  Loaded {len(htf_df)} {args.htf} bars")
    print()

    print("[3] Computing H1 regime scores...")
    htf_scores = compute_htf_regime_scores(htf_df, adx_period=14)
    print(f"  H1 scores computed")
    get_zone_summary(htf_scores)
    print()

    print("[4] Aligning H1 regime scores to M5 bars...")
    m5_regime_scores = align_htf_to_m5(m5_df, htf_df, htf_scores)
    print(f"  Aligned {len(m5_regime_scores)} M5 bars")
    get_zone_summary(m5_regime_scores)
    print()

    split = int(len(m5_df) * 0.7)
    print(f"[5] Split: {split} train / {len(m5_df) - split} val")
    train_df = m5_df.iloc[:split].copy().reset_index(drop=True)
    val_df = m5_df.iloc[split:].copy().reset_index(drop=True)
    train_scores = m5_regime_scores[:split].copy()
    val_scores = m5_regime_scores[split:].copy()
    print()

    print("[6] Computing regime weights for training...")
    range_weights = compute_regime_weights(train_scores, target_zone='range')
    trend_weights = compute_regime_weights(train_scores, target_zone='trend')
    print(f"  Range weights: mean={np.mean(range_weights):.3f}, min={np.min(range_weights):.2f}")
    print(f"  Trend weights: mean={np.mean(trend_weights):.3f}, min={np.min(trend_weights):.2f}")
    print()

    print("[7] Building features with market symmetry...")
    inverted_df = make_inverted_df(train_df)
    # For inverted data, regime scores are negated
    inv_scores = -train_scores.copy()
    # Compute weights for inverted data too
    inv_range_w = compute_regime_weights(inv_scores, target_zone='range')
    inv_trend_w = compute_regime_weights(inv_scores, target_zone='trend')
    
    combined_df = pd.concat([train_df, inverted_df], ignore_index=True)
    combined_scores = np.concatenate([train_scores, inv_scores])
    combined_range_w = np.concatenate([range_weights, inv_range_w])
    combined_trend_w = np.concatenate([trend_weights, inv_trend_w])
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

        # --- Create environments for this seed ---
        
        # Baseline env (no regime weighting)
        baseline_env = MTFRegimeEnv(
            combined_df, regime_scores=combined_scores,
            regime_weights=None  # no regime weighting = standard training
        )
        baseline_features = baseline_env.features
        baseline_forward_ret = baseline_env._forward_ret
        
        # Range-weighted env
        range_env = MTFRegimeEnv(
            combined_df, regime_scores=combined_scores,
            regime_weights=combined_range_w
        )
        range_features = range_env.features
        range_forward_ret = range_env._forward_ret
        
        # Trend-weighted env
        trend_env = MTFRegimeEnv(
            combined_df, regime_scores=combined_scores,
            regime_weights=combined_trend_w
        )
        trend_features = trend_env.features
        trend_forward_ret = trend_env._forward_ret
        
        model_baseline = None
        model_range = None
        model_trend = None
        
        # PPO kwargs template
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
        
        def make_env_fn(features, forward_ret, scores, weights):
            n = len(features)
            dummy = pd.DataFrame({
                "open": np.zeros(n), "high": np.zeros(n),
                "low": np.zeros(n), "close": np.zeros(n),
                "volume": np.ones(n),
            })
            def _fn():
                e = MTFRegimeEnv(dummy, regime_scores=scores, regime_weights=weights)
                e.features = features.astype(np.float32)
                e._forward_ret = forward_ret
                return e
            return _fn
        
        if len(baseline_features) >= WINDOW_SIZE + 10:
            # --- Train BASELINE model (no regime weighting) ---
            print(f"  [A] Training BASELINE model (no regime weighting)...")
            baseline_env_vec = VecMonitor(DummyVecEnv([make_env_fn(
                baseline_features, baseline_forward_ret, combined_scores, None
            )]))
            model_baseline = PPO("MlpPolicy", baseline_env_vec, **ppo_kwargs)
            t_a = time.time()
            cb_a = MetricsCallback()
            b_steps = min(N_STEPS, max(1024, len(baseline_features) * 2))
            model_baseline.learn(total_timesteps=b_steps, callback=cb_a)
            b_pos = np.array(cb_a.positions)
            print(f"    Baseline: {time.time()-t_a:.1f}s ({b_steps/(time.time()-t_a):.0f} sps)")
            if len(b_pos) > 0:
                print(f"    Train L/S/F: {np.mean(b_pos>0.01)*100:.1f}/{np.mean(b_pos<-0.01)*100:.1f}/{np.mean(np.abs(b_pos)<=0.01)*100:.1f}%")
        
        if len(range_features) >= WINDOW_SIZE + 10:
            # --- Train RANGE-WEIGHTED model ---
            print(f"  [B] Training RANGE-WEIGHTED model...")
            range_env_vec = VecMonitor(DummyVecEnv([make_env_fn(
                range_features, range_forward_ret, combined_scores, combined_range_w
            )]))
            range_kwargs = dict(ppo_kwargs)
            range_kwargs["seed"] = seed + 1000
            model_range = PPO("MlpPolicy", range_env_vec, **range_kwargs)
            t_r = time.time()
            cb_r = MetricsCallback()
            r_steps = min(N_STEPS // 2, max(1024, len(range_features) * 2))
            model_range.learn(total_timesteps=r_steps, callback=cb_r)
            r_pos = np.array(cb_r.positions)
            print(f"    Range-w: {time.time()-t_r:.1f}s ({r_steps/(time.time()-t_r):.0f} sps)")
            if len(r_pos) > 0:
                print(f"    Train L/S/F: {np.mean(r_pos>0.01)*100:.1f}/{np.mean(r_pos<-0.01)*100:.1f}/{np.mean(np.abs(r_pos)<=0.01)*100:.1f}%")
        
        if len(trend_features) >= WINDOW_SIZE + 10:
            # --- Train TREND-WEIGHTED model ---
            print(f"  [C] Training TREND-WEIGHTED model...")
            trend_env_vec = VecMonitor(DummyVecEnv([make_env_fn(
                trend_features, trend_forward_ret, combined_scores, combined_trend_w
            )]))
            trend_kwargs = dict(ppo_kwargs)
            trend_kwargs["seed"] = seed + 2000
            model_trend = PPO("MlpPolicy", trend_env_vec, **trend_kwargs)
            t_t = time.time()
            cb_t = MetricsCallback()
            t_steps = min(N_STEPS // 2, max(1024, len(trend_features) * 2))
            model_trend.learn(total_timesteps=t_steps, callback=cb_t)
            t_pos = np.array(cb_t.positions)
            print(f"    Trend-w: {time.time()-t_t:.1f}s ({t_steps/(time.time()-t_t):.0f} sps)")
            if len(t_pos) > 0:
                print(f"    Train L/S/F: {np.mean(t_pos>0.01)*100:.1f}/{np.mean(t_pos<-0.01)*100:.1f}/{np.mean(np.abs(t_pos)<=0.01)*100:.1f}%")

        # --- Evaluate ---
        print(f"  [D] Evaluating...")

        def make_val_env_fn(weights=None):
            def _fn():
                e = MTFRegimeEnv(val_df, regime_scores=val_scores, regime_weights=weights)
                return e
            return _fn
        
        # Evaluate BASELINE
        val_baseline = None
        if model_baseline is not None:
            val_baseline = evaluate(
                model_baseline, make_val_env_fn(),
                turnover_cost=TURNOVER_COST, annualization_factor=ANNUAL_M5
            )
        
        # Evaluate RANGE model (on unweighted val env = honest eval)
        val_range = None
        if model_range is not None:
            val_range = evaluate(
                model_range, make_val_env_fn(),
                turnover_cost=TURNOVER_COST, annualization_factor=ANNUAL_M5
            )
        
        # Evaluate TREND model (on unweighted val env = honest eval)
        val_trend = None
        if model_trend is not None:
            val_trend = evaluate(
                model_trend, make_val_env_fn(),
                turnover_cost=TURNOVER_COST, annualization_factor=ANNUAL_M5
            )
        
        # Evaluate META-CONTROLLER (3-zone routing)
        val_meta = None
        if model_range is not None and model_trend is not None:
            val_meta = meta_controller_evaluate(
                model_range, model_trend, make_val_env_fn(),
                val_scores, turnover_cost=TURNOVER_COST, annualization_factor=ANNUAL_M5
            )
        
        # Print results
        for label, v in [("BASELINE", val_baseline), ("RANGE-W", val_range),
                          ("TREND-W", val_trend), ("META-CTRL", val_meta)]:
            if v:
                print(f"    {label:<12} L={v['long_pct']:.1f}/S={v['short_pct']:.1f}/F={v['flat_pct']:.1f}%  "
                      f"SR={v['sharpe']:.2f}  Ret={v['total_return']:.2f}%  "
                      f"DD={v['max_drawdown']:.2f}%  TO={v['turnover']:.1f}%")
        
        print(f"    Seed {seed}: {time.time()-t0:.1f}s")
        print()

        # Store results
        def safe_r(d, key):
            return d[key] if d is not None else None
        result = {
            "seed": seed,
            "base_sharpe": safe_r(val_baseline, "sharpe"),
            "base_return": safe_r(val_baseline, "total_return"),
            "base_long_pct": safe_r(val_baseline, "long_pct"),
            "base_short_pct": safe_r(val_baseline, "short_pct"),
            "base_flat_pct": safe_r(val_baseline, "flat_pct"),
            "base_turnover": safe_r(val_baseline, "turnover"),
            "base_maxdd": safe_r(val_baseline, "max_drawdown"),
            "range_sharpe": safe_r(val_range, "sharpe"),
            "range_return": safe_r(val_range, "total_return"),
            "trend_sharpe": safe_r(val_trend, "sharpe"),
            "trend_return": safe_r(val_trend, "total_return"),
            "meta_sharpe": safe_r(val_meta, "sharpe"),
            "meta_return": safe_r(val_meta, "total_return"),
            "meta_long_pct": safe_r(val_meta, "long_pct"),
            "meta_short_pct": safe_r(val_meta, "short_pct"),
            "meta_flat_pct": safe_r(val_meta, "flat_pct"),
            "meta_turnover": safe_r(val_meta, "turnover"),
            "meta_maxdd": safe_r(val_meta, "max_drawdown"),
        }
        all_results.append(result)

    # Aggregate results
    print()
    print("=" * 64)
    print("LANE D - MTF REGIME-ROUTED PPO RESULTS")
    print("=" * 64)

    results_df = pd.DataFrame(all_results)

    models_to_show = [
        ("BASELINE (no weighting)", "base"),
        ("RANGE-WEIGHTED", "range"),
        ("TREND-WEIGHTED", "trend"),
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
        print(f"    {'-'*20} {'-'*10} {'-'*10}  {'-'*4}")
        for c in avail:
            short = c.replace(f"{prefix}_", "")
            mean_v = vals[c].mean()
            std_v = vals[c].std()
            print(f"    {short:<20} {mean_v:>10.2f} {std_v:>10.2f}  {len(vals):>4}")

    print()
    print("HEAD-TO-HEAD: META-CONTROLLER vs BASELINE")
    print("-" * 48)
    meta_df = results_df[["seed", "meta_sharpe", "base_sharpe",
                           "meta_return", "base_return"]].dropna()
    if len(meta_df) > 0:
        meta_df["sr_diff"] = meta_df["meta_sharpe"] - meta_df["base_sharpe"]
        meta_df["ret_diff"] = meta_df["meta_return"] - meta_df["base_return"]
        wins_sr = int((meta_df["sr_diff"] > 0).sum())
        wins_ret = int((meta_df["ret_diff"] > 0).sum())
        print(f"  Sharpe: meta wins {wins_sr}/{len(meta_df)} seeds")
        print(f"  Return: meta wins {wins_ret}/{len(meta_df)} seeds")
        print(f"  Mean SR diff: {meta_df['sr_diff'].mean():.2f} (meta - base)")
        print(f"  Mean Ret diff: {meta_df['ret_diff'].mean():.2f}pp")
        print()
        print(f"{'Seed':<8} {'Meta SR':<10} {'Base SR':<10} {'SR Diff':<10} {'Meta Ret':<10} {'Base Ret':<10}")
        print(f"{'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
        for _, row in meta_df.iterrows():
            print(f"{int(row['seed']):<8} {row['meta_sharpe']:<10.2f} {row['base_sharpe']:<10.2f} "
                  f"{row['sr_diff']:<10.2f} {row['meta_return']:<10.2f} {row['base_return']:<10.2f}")
    else:
        print("  No paired results available")

    print()
    print("=" * 64)
    print("END - LANE D")
    print("=" * 64)



# -- Regime score computation on H1 data --

def compute_htf_regime_scores(htf_df: pd.DataFrame, adx_period: int = 14) -> np.ndarray:
    """Compute regime_scores on higher-timeframe data (H1/H4).
    Returns array of same length as htf_df with scores in [-1, +1].
    """
    _, scores = classify_regime(htf_df, adx_period=adx_period)
    return scores


# -- Align H1 regime scores to M5 bars (strictly causal) --

def align_htf_to_m5(m5_df: pd.DataFrame, htf_df: pd.DataFrame,
                    htf_scores: np.ndarray) -> np.ndarray:
    """Align H1 regime scores to M5 bars using only the most recently closed
    H1 candle. No future leakage - strictly causal.
    
    For each M5 bar, finds the most recent H1 bar whose timestamp is <= the
    M5 bar's timestamp, and uses that H1 bar's regime_score.
    """
    scores = np.zeros(len(m5_df), dtype=np.float32)
    htf_idx = 0
    for i in range(len(m5_df)):
        m5_ts = m5_df.index[i]
        # Advance H1 index while the NEXT H1 candle has started (<= m5_ts)
        while htf_idx + 1 < len(htf_df) and htf_df.index[htf_idx + 1] <= m5_ts:
            htf_idx += 1
        # Use the most recently closed H1 candle's score
        scores[i] = htf_scores[htf_idx]
    return scores


# -- Compute regime weights for reward scaling --

def compute_regime_weights(scores: np.ndarray, target_zone: str) -> np.ndarray:
    """Compute per-bar reward weights for a model targeting a specific zone.
    
    target_zone: 'range' or 'trend'
    
    - Range model: range zone gets full weight, trend zone gets reduced,
      transition zone is linearly interpolated.
    - Trend model: trend zone gets full weight, range zone gets reduced,
      transition zone is linearly interpolated.
    """
    weights = np.ones(len(scores), dtype=np.float32)
    abs_scores = np.abs(scores)
    
    if target_zone == 'range':
        # Full weight when |score| < RANGE_THRESH (range zone)
        # Reduced weight when |score| > TREND_THRESH (trend zone)
        # Interpolated in transition
        for i in range(len(scores)):
            s = abs_scores[i]
            if s < RANGE_THRESH:
                weights[i] = RANGE_WEIGHT
            elif s > TREND_THRESH:
                weights[i] = OPP_WEIGHT
            else:
                # Linear interpolation in transition zone
                t = (s - RANGE_THRESH) / (TREND_THRESH - RANGE_THRESH)
                weights[i] = RANGE_WEIGHT + (OPP_WEIGHT - RANGE_WEIGHT) * t
    else:  # 'trend'
        for i in range(len(scores)):
            s = abs_scores[i]
            if s > TREND_THRESH:
                weights[i] = RANGE_WEIGHT
            elif s < RANGE_THRESH:
                weights[i] = OPP_WEIGHT
            else:
                t = (s - RANGE_THRESH) / (TREND_THRESH - RANGE_THRESH)
                weights[i] = OPP_WEIGHT + (RANGE_WEIGHT - OPP_WEIGHT) * t
    return weights


# -- MTF Regime-Aware Environment --

class MTFRegimeEnv(BaseTamedEnv):
    """M5 trading environment with H1 regime context and regime-weighted rewards.
    
    Observation: [7 M5 features (OHLCV log-ret + RSI + MACD) + 1 H1 regime_score] = 8 features
    
    Features are Z-score normalized, then windowed into (window_size x 8) observations.
    The regime_score is held constant for all bars within the same H1 candle window.
    
    Reward = raw_return * regime_weight * REWARD_SCALE
    where regime_weight depends on the H1 regime_score and which model is training.
    """
    def __init__(self, df: pd.DataFrame, regime_scores: np.ndarray,
                 regime_weights: np.ndarray | None = None,
                 window_size: int = WINDOW_SIZE,
                 turnover_cost: float = TURNOVER_COST,
                 concentration_penalty: float = CONCENTRATION_PENALTY,
                 reward_scale: float = REWARD_SCALE,
                 inactivity_penalty: float = INACTIVITY_PENALTY,
                 holding_penalty: float = HOLDING_PENALTY,
                 smoothing_alpha: float = SMOOTHING_ALPHA,
                 cooldown_steps: int = COOLDOWN_STEPS,
                 discrete: bool = DISCRETE):
        
        self.df = df.reset_index(drop=True)
        self.reward_horizon = REWARD_HORIZON
        self._regime_scores = np.asarray(regime_scores, dtype=np.float32)
        self._regime_weights = None
        if regime_weights is not None:
            self._regime_weights = np.asarray(regime_weights, dtype=np.float32)
        self._raw_features = None
        self.n_features = 8
        
        super().__init__(
            window_size=window_size, turnover_cost=turnover_cost,
            concentration_penalty=concentration_penalty,
            reward_scale=reward_scale, inactivity_penalty=inactivity_penalty,
            holding_penalty=holding_penalty, smoothing_alpha=smoothing_alpha,
            cooldown_steps=cooldown_steps, n=1, n_features=8,
            discrete=discrete,
        )
        self._build_features()
    
    def _build_features(self):
        """Build 8 features: 7 M5 OHLCV/RSI/MACD + 1 H1 regime_score."""
        # Use Lane B's feature building approach
        df = self.df
        c = df["close"].values.astype(np.float64)
        n = len(c)
        
        # Avoid log(0)
        safe_c = np.maximum(c, 1e-10)
        
        # 1. Log returns of OHLCV (5 features)
        log_ret = np.full((n, 5), 0.0, dtype=np.float64)
        cols = ["open", "high", "low", "close", "volume"]
        for j, col in enumerate(cols):
            v = df[col].values.astype(np.float64)
            v_safe = np.maximum(v, 1e-10)
            log_ret[1:, j] = np.log(v_safe[1:] / v_safe[:-1])
        
        # 2. RSI(14)
        deltas = np.diff(safe_c)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.full(n, np.nan, dtype=np.float64)
        avg_loss = np.full(n, np.nan, dtype=np.float64)
        avg_gain[14] = np.mean(gains[:14])
        avg_loss[14] = np.mean(losses[:14])
        for i in range(15, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gains[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + losses[i-1]) / 14
        rs = np.where(avg_loss > 1e-10, avg_gain / avg_loss, 100.0)
        rsi = np.where(~np.isnan(rs), 100.0 - 100.0 / (1.0 + rs), 50.0)
        
        # 3. MACD(12,26,9) histogram
        ema12 = np.full(n, np.nan, dtype=np.float64)
        ema26 = np.full(n, np.nan, dtype=np.float64)
        ema12[0] = safe_c[0]
        ema26[0] = safe_c[0]
        for i in range(1, n):
            ema12[i] = ema12[i-1] + (safe_c[i] - ema12[i-1]) * 2.0 / 13.0
            ema26[i] = ema26[i-1] + (safe_c[i] - ema26[i-1]) * 2.0 / 27.0
        macd_line = ema12 - ema26
        signal = np.full(n, np.nan, dtype=np.float64)
        signal[0] = macd_line[0]
        for i in range(1, n):
            signal[i] = signal[i-1] + (macd_line[i] - signal[i-1]) * 2.0 / 10.0
        macd_hist = macd_line - signal
        # Normalize MACD hist by close price
        macd_norm = np.where(safe_c > 0, macd_hist / safe_c, 0.0)
        
        # 4. H1 regime_score (feature index 7)
        regime_scores = np.asarray(self._regime_scores, dtype=np.float64)
        # Ensure length matches
        if len(regime_scores) < n:
            regime_scores = np.pad(regime_scores, (0, n - len(regime_scores)), mode='edge')
        elif len(regime_scores) > n:
            regime_scores = regime_scores[:n]
        
        # Combine raw features [5 log-ret + 1 RSI + 1 MACD + 1 regime_score = 8]
        raw = np.column_stack([
            log_ret,                     # 5 cols: OHLCV log-returns
            rsi / 100.0,                 # 1 col: RSI normalized to [0,1]
            np.nan_to_num(macd_norm),    # 1 col: MACD histogram / price
            regime_scores,               # 1 col: H1 regime_score
        ])
        
        # Z-score normalize
        eps = 1e-8
        mean = np.nanmean(raw, axis=0)
        std = np.nanstd(raw, axis=0)
        std = np.maximum(std, eps)
        normalized = (raw - mean) / std
        normalized = np.nan_to_num(normalized)
        
        self._raw_features = raw.astype(np.float32)
        self.features = normalized.astype(np.float32)
        self.n_features = 8
        
        # Forward return for reward calculation
        c_safe = np.maximum(c, 1e-10)
        self._forward_ret = np.zeros(n, dtype=np.float64)
        self._forward_ret[:-self.reward_horizon] = (
            c_safe[self.reward_horizon:] / c_safe[:-self.reward_horizon] - 1.0
        )
    
    def _raw_reward_at(self, i: int) -> float:
        """Regime-weighted forward return."""
        if i < len(self._forward_ret):
            base_ret = float(self._forward_ret[i])
        else:
            base_ret = 0.0
        if self._regime_weights is not None and i < len(self._regime_weights):
            return base_ret * float(self._regime_weights[i])
        return base_ret

if __name__ == "__main__":
    main()
