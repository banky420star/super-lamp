"""
Champion Selection — Tier 3 Gatekeeper.

Reads Lane B per-seed results, applies strict rejection rules,
and promotes the best candidate model to champion.

Rejection rules:
  - Return < 0%
  - Max drawdown < -20%
  - Long% > 98% (one-sided trap)
  - Short% > 98% (one-sided trap)

Champion = highest return / max(abs(drawdown), 0.1) among candidates.

Usage:
    python training/select_champion.py
    python training/select_champion.py --csv runtime/lane_b_raw_all_seeds.csv
"""
import sys, os, argparse, json, shutil
import numpy as np
import pandas as pd

DEFAULT_CSV = "runtime/lane_b_raw_all_seeds.csv"
MODEL_DIR = "runtime"
CHAMPION_MODEL = "runtime/champion_lane_b_model.zip"
SCORECARD_PATH = "runtime/champion_lane_b_scorecard.json"


def compute_metrics(df, seed):
    """Compute validation metrics for a single seed from positions DataFrame."""
    seed_df = df[df["seed"] == seed].copy()
    if len(seed_df) == 0:
        return None

    nw = seed_df["net_worth"].values
    positions = seed_df["position"].values

    # Return from start to end
    total_return = (nw[-1] / nw[0] - 1.0) * 100.0

    # Max drawdown
    peak = np.maximum.accumulate(nw)
    dd = (nw - peak) / peak
    max_drawdown = float(np.min(dd) * 100.0)

    # Direction breakdown
    long_pct = float(np.mean(positions > 0.01) * 100)
    short_pct = float(np.mean(positions < -0.01) * 100)
    flat_pct = float(np.mean(np.abs(positions) <= 0.01) * 100)

    # Sharpe (approximate from nw changes)
    returns = np.diff(nw) / nw[:-1]
    sharpe = float(np.mean(returns) / max(np.std(returns), 1e-10)) * np.sqrt(252 * 288)  # 288 M5 bars/day

    # Turnover
    changes = np.sum(np.abs(np.diff(np.sign(positions)))) / 2
    turnover = changes / len(positions) * 100.0

    return {
        "seed": int(seed),
        "total_return": round(total_return, 2),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe": round(sharpe, 2),
        "long_pct": round(long_pct, 1),
        "short_pct": round(short_pct, 1),
        "flat_pct": round(flat_pct, 1),
        "turnover": round(turnover, 1),
        "n_bars": len(seed_df),
    }


def select_champion(metrics_list):
    """Apply rejection rules and select champion.

    Returns (champion_seed, scorecard_rows, best_score).
    """
    champion_seed = None
    best_score = -float("inf")
    scorecard_rows = []

    for m in metrics_list:
        ret = m["total_return"]
        dd = m["max_drawdown"]
        long_pct = m["long_pct"]
        short_pct = m["short_pct"]

        reject_reasons = []
        if ret < 0:
            reject_reasons.append(f"return={ret:.2f}% < 0%")
        if dd < -20:
            reject_reasons.append(f"drawdown={dd:.2f}% < -20%")
        if long_pct > 98:
            reject_reasons.append(f"long_pct={long_pct:.1f}% > 98% (one-sided trap)")
        if short_pct > 98:
            reject_reasons.append(f"short_pct={short_pct:.1f}% > 98% (one-sided trap)")

        status = "REJECTED" if reject_reasons else "CANDIDATE"
        score = ret / max(abs(dd), 0.1)

        scorecard_rows.append({
            **m,
            "score": round(score, 4),
            "status": status,
            "reject_reasons": reject_reasons,
        })

        if status == "CANDIDATE" and score > best_score:
            best_score = score
            champion_seed = m["seed"]

    return champion_seed, scorecard_rows, best_score


def promote_champion(champion_seed):
    """Copy champion model to champion_lane_b_model.zip."""
    src = os.path.join(MODEL_DIR, f"lane_b_seed_{champion_seed}_model.zip")
    dst = CHAMPION_MODEL

    if not os.path.exists(src):
        print(f"  [WARN] Champion model not found: {src}")
        print(f"         Run Lane B training first to produce per-seed model files.")
        return False

    shutil.copy2(src, dst)
    print(f"  [OK] Champion model promoted: {src} -> {dst}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Lane B Champion Selection")
    parser.add_argument("--csv", default=DEFAULT_CSV, help=f"Results CSV (default: {DEFAULT_CSV})")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV not found: {args.csv}")
        print("Run Lane B training first: python training/run_lane_b_raw_lstm.py")
        return

    print("=" * 64)
    print("LANE B — CHAMPION SELECTION")
    print("=" * 64)
    print(f"  CSV: {args.csv}")
    print()

    # Load data
    df = pd.read_csv(args.csv)
    seeds = sorted(df["seed"].unique())
    print(f"  Seeds found: {seeds}")
    print(f"  Total rows:  {len(df)}")
    print()

    # Compute per-seed metrics
    all_metrics = []
    for seed in seeds:
        m = compute_metrics(df, seed)
        if m:
            all_metrics.append(m)
            print(f"  Seed {seed:>4}: Return={m['total_return']:>8.2f}%  "
                  f"DD={m['max_drawdown']:>7.2f}%  "
                  f"Long={m['long_pct']:>5.1f}%  Short={m['short_pct']:>5.1f}%  "
                  f"Sharpe={m['sharpe']:>7.2f}")

    if not all_metrics:
        print("  No valid seeds found.")
        return

    # Select champion
    print()
    print("  --- Rejection Rules ---")
    champion, scorecard, best_score = select_champion(all_metrics)

    # Print scorecard
    print()
    print(f"  {'Seed':>6} {'Return%':>9} {'DD%':>8} {'Sharpe':>8} {'Long%':>7} {'Short%':>7} {'Score':>8} {'Status':>10}")
    print(f"  {'-'*6} {'-'*9} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*10}")
    for row in scorecard:
        print(f"  {row['seed']:>6} {row['total_return']:>9.2f} {row['max_drawdown']:>8.2f} "
              f"{row['sharpe']:>8.2f} {row['long_pct']:>7.1f} {row['short_pct']:>7.1f} "
              f"{row['score']:>8.2f} {row['status']:>10}")

    # Print rejections
    rejected = [r for r in scorecard if r["status"] == "REJECTED"]
    if rejected:
        print(f"\n  Rejections:")
        for r in rejected:
            for reason in r["reject_reasons"]:
                print(f"    Seed {r['seed']}: {reason}")

    # Promote champion
    if champion is not None:
        print(f"\n  [OK] CHAMPION: Seed {champion} (score={best_score:.2f})")
        promote_champion(champion)
    else:
        print(f"\n  [NO] No champion — all seeds rejected.")

    # Save scorecard
    os.makedirs(os.path.dirname(SCORECARD_PATH) or ".", exist_ok=True)
    scorecard_data = {
        "champion_seed": champion,
        "champion_score": best_score if champion else None,
        "scorecard": scorecard,
        "source_csv": args.csv,
    }
    with open(SCORECARD_PATH, "w") as f:
        json.dump(scorecard_data, f, indent=2)
    print(f"  Scorecard saved: {SCORECARD_PATH}")

    print()
    print("  Done.")


if __name__ == "__main__":
    main()
