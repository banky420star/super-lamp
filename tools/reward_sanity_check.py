"""
Quick sanity test for Lane B env.
Runs fixed Long / Flat / Short policies and a couple of simple rules.
Prints whether any non-flat policy can beat flat on the actual reward signal.
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, "C:/supreme-chainsaw")

from training.run_real_feature_ablation import load_real_data
from training.run_lane_b_raw_lstm import TamedOHLCVEnv, REWARD_SCALE

def run_fixed(env, action, max_steps=1200):
    """action: 0=Long, 1=Flat, 2=Short"""
    obs, _ = env.reset()
    total_r = 0.0
    rews = []
    poss = []
    raw_rews = []
    steps = 0
    done = False
    while not done and steps < max_steps:
        obs, r, term, trunc, info = env.step(action)
        total_r += r
        rews.append(r)
        poss.append(info["position"])
        raw_rews.append(info.get("raw_reward", 0.0))
        done = term or trunc
        steps += 1
    return {
        "steps": steps,
        "total_scaled_reward": total_r,
        "mean_scaled_r": np.mean(rews) if rews else 0,
        "mean_pos": np.mean(poss) if poss else 0,
        "mean_raw_ret": np.mean(raw_rews) if raw_rews else 0,
        "sum_raw_ret_when_positioned": np.sum([rr * pp for rr, pp in zip(raw_rews, poss)]),
    }

def simple_trend_rule(env, max_steps=1200):
    """Very naive: if last return > 0 go long else short. No filtering."""
    obs, _ = env.reset()
    total_r = 0.0
    poss = []
    steps = 0
    done = False
    while not done and steps < max_steps:
        # crude: look at last close logret in the window (feature 3 in last bar)
        # obs is flattened window*features
        nfeat = env.n_features
        last_ret = obs[-nfeat + 3]  # rough
        if last_ret > 0:
            a = 0  # Long
        else:
            a = 2  # Short
        obs, r, term, trunc, info = env.step(a)
        total_r += r
        poss.append(info["position"])
        done = term or trunc
        steps += 1
    return {"steps": steps, "total_scaled_reward": total_r, "mean_pos": np.mean(poss) if poss else 0}

def main():
    print("Loading data (small slice for speed)...")
    df = load_real_data(symbol="XAUUSDm", n_bars=2500)
    print(f"  {len(df)} bars")

    env = TamedOHLCVEnv(df)
    print(f"  Features: {env.n_features}, window={env.window_size}, discrete={env.discrete}")
    print(f"  Inactivity penalty (raw): {env.inactivity_penalty}")
    print(f"  Reward scale: {env.reward_scale}")
    print()

    results = {}
    for name, a in [("Long(+1)", 0), ("Flat(0)", 1), ("Short(-1)", 2)]:
        res = run_fixed(env, a)
        results[name] = res
        print(f"{name:12s} | scaled_total={res['total_scaled_reward']:8.1f}  mean_r={res['mean_scaled_r']:+.4f}  mean_pos={res['mean_pos']:+.3f}  sum(pos*raw)={res['sum_raw_ret_when_positioned']:+.6f}")

    print()
    rule_res = simple_trend_rule(env)
    print(f"SimpleTrend  | scaled_total={rule_res['total_scaled_reward']:8.1f}  mean_pos={rule_res['mean_pos']:+.3f}")

    # Verdict
    flat_r = results["Flat(0)"]["total_scaled_reward"]
    best_nonflat = max(results["Long(+1)"]["total_scaled_reward"], results["Short(-1)"]["total_scaled_reward"], rule_res["total_scaled_reward"])
    print()
    if best_nonflat > flat_r + 1.0:
        print("VERDICT: At least one directional policy beats flat on this slice. Signal exists in the env.")
    else:
        print("VERDICT: No tested policy beats flat (or margin is tiny). Flat is rational. The reward landscape does not yet reward taking positions.")
    print("Note: this is one random validation-like walk; repeat on different slices/seeds for robustness.")

if __name__ == "__main__":
    main()
