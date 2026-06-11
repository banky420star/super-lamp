"""
Tier 0 Profitability Gate: Reward Sanity Check

Runs fixed directional policies (Long/Flat/Short) and a simple trend rule
against the *current* reward landscape (now using full TradingReward primary
after the drl/trading_env.py change).

Used as a pre-flight gate before big training runs or promotion.
Requires at least one non-flat policy to meaningfully beat Flat on multiple
slices, or the run is considered high-risk for flat collapse.

Usage (as gate):
  python tools/reward_sanity_check.py --gate --slices 3 --min-margin 50

Staying on feature/profitability-tier0-reward branch.
"""
import sys, os, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, "C:/supreme-chainsaw")

from training.run_real_feature_ablation import load_real_data

# Try the main modern env first (respects the 100% TradingReward primary fix)
try:
    from drl.trading_env import TradingEnv as MainTradingEnv
    HAS_MAIN_ENV = True
except Exception:
    HAS_MAIN_ENV = False

# Fallback to the old Lane B env for compatibility with existing experiments
try:
    from training.run_lane_b_raw_lstm import TamedOHLCVEnv, REWARD_SCALE
    HAS_LANE_ENV = True
except Exception:
    HAS_LANE_ENV = False


def run_fixed_main_env(df, direction, max_steps=800, window=100):
    """Test using the real drl.TradingEnv (Decision PPO / rich action path)."""
    if not HAS_MAIN_ENV:
        return None
    env = MainTradingEnv(df, window_size=window, feature_version="engineered_v2")
    obs, _ = env.reset()
    total_r = 0.0
    steps = 0
    done = False
    while not done and steps < max_steps:
        # Simple fixed mapping for rich action head (direction, confidence-ish, size, tp/sl etc.)
        # direction >0 = long bias, <0 = short bias, 0 = flat
        if direction > 0.1:
            action = np.array([0.7, 0.1, 0.0, 0.4, 0.6, 0.0], dtype=np.float32)  # long-ish
        elif direction < -0.1:
            action = np.array([-0.7, 0.0, 0.1, 0.4, 0.0, 0.6], dtype=np.float32)  # short-ish
        else:
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # flat
        obs, r, term, trunc, info = env.step(action)
        total_r += float(r)
        done = term or trunc
        steps += 1
    return total_r


def run_fixed_lane_env(df, action_idx, max_steps=1200):
    """Fallback using the simplified Lane B discrete env."""
    if not HAS_LANE_ENV:
        return None
    env = TamedOHLCVEnv(df)
    obs, _ = env.reset()
    total_r = 0.0
    steps = 0
    done = False
    while not done and steps < max_steps:
        obs, r, term, trunc, info = env.step(action_idx)
        total_r += r
        done = term or trunc
        steps += 1
    return total_r


def simple_trend_rule_main(df, max_steps=800):
    if not HAS_MAIN_ENV:
        return None
    env = MainTradingEnv(df, window_size=100, feature_version="engineered_v2")
    obs, _ = env.reset()
    total_r = 0.0
    steps = 0
    done = False
    nfeat = 7  # rough for engineered_v2 per-bar
    while not done and steps < max_steps:
        last_ret = float(obs[-nfeat + 3]) if len(obs) > nfeat else 0.0
        direction = 1.0 if last_ret > 0 else -1.0
        action = np.array([direction * 0.7, 0.0, 0.0, 0.4, 0.5 if direction > 0 else 0.0, 0.5 if direction < 0 else 0.0], dtype=np.float32)
        obs, r, term, trunc, info = env.step(action)
        total_r += float(r)
        done = term or trunc
        steps += 1
    return total_r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true", help="Exit non-zero if no non-flat policy beats flat by margin (for CI/gates)")
    parser.add_argument("--slices", type=int, default=2, help="Number of different data slices to test")
    parser.add_argument("--min-margin", type=float, default=30.0, help="Minimum scaled reward advantage non-flat must have over flat on a slice")
    parser.add_argument("--n-bars", type=int, default=2500)
    args = parser.parse_args()

    print("=== Tier 0 Profitability Reward Sanity Gate ===")
    print(f"Using main drl.TradingEnv primary reward path: {HAS_MAIN_ENV}")
    print(f"Using legacy Lane B env fallback: {HAS_LANE_ENV}")
    print()

    passed_slices = 0
    for i in range(args.slices):
        seed = 42 + i * 7
        df = load_real_data(symbol="XAUUSDm", n_bars=args.n_bars, seed=seed)
        print(f"Slice {i+1}/{args.slices} (seed={seed}, bars={len(df)}):")

        scores = {}
        if HAS_MAIN_ENV:
            for name, d in [("Long", 1.0), ("Flat", 0.0), ("Short", -1.0)]:
                scores[name] = run_fixed_main_env(df, d)
        elif HAS_LANE_ENV:
            # Map to discrete indices for legacy env
            for name, idx in [("Long", 0), ("Flat", 1), ("Short", 2)]:
                scores[name] = run_fixed_lane_env(df, idx)

        if not scores:
            print("  No usable env found.")
            continue

        flat_r = scores.get("Flat", 0.0) or 0.0
        best_nonflat = max(scores.get("Long", -1e9) or -1e9, scores.get("Short", -1e9) or -1e9)
        print(f"  Flat={flat_r:+8.1f}  BestNonFlat={best_nonflat:+8.1f}  (Long={scores.get('Long',0):+7.1f} Short={scores.get('Short',0):+7.1f})")

        if best_nonflat > flat_r + args.min_margin:
            passed_slices += 1
            print(f"  PASS slice (non-flat beats flat by >{args.min_margin})")
        else:
            print(f"  WEAK slice (margin < {args.min_margin})")

        # Also try simple rule on main env if available
        if HAS_MAIN_ENV:
            rule = simple_trend_rule_main(df)
            if rule is not None:
                print(f"  SimpleTrend total={rule:+8.1f}")

        print()

    overall_pass = passed_slices >= max(1, args.slices - 1)  # allow 1 weak slice

    print(f"Summary: {passed_slices}/{args.slices} slices passed margin requirement.")
    if overall_pass:
        print("VERDICT: GOOD - at least one non-flat policy shows clear advantage on most slices. Safe to train.")
        if args.gate:
            sys.exit(0)
    else:
        print("VERDICT: RISK - flat may still be rational on these slices under current reward. Consider more tuning or different data.")
        if args.gate:
            sys.exit(2)

    print("Note: repeat on fresh data / different seeds before big runs. This is a Tier 0 gate from the profitability review.")


if __name__ == "__main__":
    main()
