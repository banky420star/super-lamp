"""
Re-evaluate champion selection from an existing scorecard JSON
with a configurable max-drawdown threshold.

Symbol-specific paths enforced (Phase 5):
  Per-seed: runtime/lane_b_seed_{SEED}_{SYMBOL}_model.zip
  Champion: runtime/champion_lane_b_{SYMBOL}_model.zip

Usage:
    python training/re_evaluate_champion.py runtime/champion_lane_b_scorecard.json --max-dd -40
"""
import sys, json, argparse, shutil, os, hashlib, time
from datetime import datetime, timezone


def re_evaluate(scorecard_path, max_dd=-20, promote=True):
    """Re-evaluate champion selection with a different max_dd threshold."""
    with open(scorecard_path) as f:
        data = json.load(f)

    scorecard = data["scorecard"]
    config = data.get("config", {})
    symbol = config.get("symbol")
    if not symbol:
        # Infer from filename e.g. champion_lane_b_XAUUSDm_scorecard.json or use first data
        base = os.path.basename(scorecard_path)
        for token in ["XAUUSDm", "EURUSDm", "GBPUSDm", "BTCUSDm", "ETHUSDm"]:
            if token in base:
                symbol = token
                break
    if not symbol:
        # look in first row if any
        if scorecard and isinstance(scorecard, list) and "symbol" in scorecard[0]:
            symbol = scorecard[0].get("symbol")
    symbol = symbol or "XAUUSDm"  # safe default, but Phase5 requires explicit per-symbol
    old_champion = data.get("champion_seed")

    print("=" * 64)
    print(f"CHAMPION RE-EVALUATION  —  {symbol}")
    print(f"Max drawdown threshold: {max_dd:.0f}%  (was {config.get('max_dd', -20):.0f}%)")
    print("=" * 64)

    champion_seed = None
    best_score = -float('inf')
    updated_rows = []

    for row in scorecard:
        # Support both legacy (return_pct) and current scorecard keys (total_return etc)
        ret = row.get("return_pct", row.get("total_return", 0.0))
        dd = row.get("drawdown_pct", row.get("max_drawdown", row.get("drawdown", 0.0)))
        long_pct = row.get("long_pct", row.get("long", 0.0))
        short_pct = row.get("short_pct", row.get("short", 0.0))
        sharpe = row.get("sharpe", 0.0)

        reject_reasons = []
        if ret < 0:
            reject_reasons.append(f"return={ret:.2f}% < 0%")
        if dd < max_dd:
            reject_reasons.append(f"drawdown={dd:.2f}% < {max_dd:.0f}%")
        if long_pct > 98:
            reject_reasons.append(f"long_pct={long_pct:.1f}% > 98%")
        if short_pct > 98:
            reject_reasons.append(f"short_pct={short_pct:.1f}% > 98%")

        status = "REJECTED" if reject_reasons else "CANDIDATE"
        score = ret / max(abs(dd), 0.1)

        updated_rows.append({
            "seed": row["seed"],
            "return_pct": round(ret, 2),
            "drawdown_pct": round(dd, 2),
            "sharpe": round(sharpe, 2),
            "long_pct": round(long_pct, 1),
            "short_pct": round(short_pct, 1),
            "score": round(score, 4),
            "status": status,
            "reject_reasons": reject_reasons,
        })

        if status == "CANDIDATE" and score > best_score:
            best_score = score
            champion_seed = row["seed"]

    # ── Print scorecard ──
    print(f"\n  {'Seed':>6} {'Return%':>9} {'DD%':>8} {'Sharpe':>8} {'Long%':>7} {'Short%':>7} {'Score':>8} {'Status':>10}")
    print(f"  {'-'*6} {'-'*9} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*10}")
    for row in updated_rows:
        print(f"  {row['seed']:>6} {row['return_pct']:>9.2f} {row['drawdown_pct']:>8.2f} "
              f"{row['sharpe']:>8.2f} {row['long_pct']:>7.1f} {row['short_pct']:>7.1f} "
              f"{row['score']:>8.4f} {row['status']:>10}")

    rejected = [r for r in updated_rows if r['status'] == 'REJECTED']
    if rejected:
        print(f"\n  Rejections:")
        for r in rejected:
            for reason in r['reject_reasons']:
                print(f"    Seed {r['seed']}: {reason}")

    # ── Result ──
    print()
    if champion_seed is not None:
        print(f"  [OK] NEW CHAMPION: Seed {champion_seed} (score={best_score:.4f})")
        if promote:
            src = f"runtime/lane_b_seed_{champion_seed}_{symbol}_model.zip"
            dst = f"runtime/champion_lane_b_{symbol}_model.zip"
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"       Model promoted: {src} -> {dst}")
            else:
                print(f"       [WARN] {src} not found — cannot promote.")

            # Phase 7/13/14: Register to model_registry with full fields (model_id, lane, symbol, seed, hashes, status)
            try:
                from Python.model_registry import ModelRegistry
                reg = ModelRegistry()
                model_id = f"lane_b_{symbol}_seed{champion_seed}_{int(time.time())}"
                hashes = {}
                if os.path.exists(dst):
                    h = hashlib.sha256()
                    with open(dst, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            h.update(chunk)
                    hashes["model_sha256"] = h.hexdigest()
                # Use register + set per-symbol champion with metadata
                cand_dir = reg.new_candidate_dir(tag=f"laneb_{symbol}")
                # copy model into cand bundle for registry integrity
                shutil.copy2(dst, os.path.join(cand_dir, "ppo_trading.zip"))
                # minimal metadata for Lane B
                meta = {
                    "lane": "b",
                    "symbol": symbol,
                    "seed": champion_seed,
                    "model_id": model_id,
                    "status": "champion",
                    "hashes": hashes,
                    "source": "run_lane_b_raw_lstm + re_evaluate",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "metrics": {"score": best_score},
                }
                with open(os.path.join(cand_dir, "metadata.json"), "w") as mf:
                    json.dump(meta, mf, indent=2)
                reg.register_candidate(cand_dir, meta)
                reg.set_champion(symbol, cand_dir)
                print(f"       Registry: model_id={model_id} status=champion registered for {symbol}")
            except Exception as reg_err:
                print(f"       [WARN] Registry register failed (non-fatal for file promote): {reg_err}")
    else:
        print(f"  [NO] No champion — all seeds rejected with max_dd={max_dd:.0f}%.")

    # ── Compare with previous champion ──
    if old_champion is not None and old_champion != champion_seed:
        print(f"\n  Note: previous champion was seed {old_champion} (under old threshold).")
    elif old_champion is not None and old_champion == champion_seed:
        print(f"\n  Champion unchanged: seed {champion_seed}.")
    elif old_champion is None and champion_seed is not None:
        print(f"\n  A champion now exists where none did before (threshold relaxed).")

    print()
    return champion_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-evaluate champion selection with different DD threshold")
    parser.add_argument("scorecard", nargs="?", default="runtime/champion_lane_b_scorecard.json",
                        help="Path to scorecard JSON (default: runtime/champion_lane_b_scorecard.json). Must contain symbol in config or filename for per-symbol champion paths.")
    parser.add_argument("--max-dd", type=float, default=-20,
                        help="Max drawdown threshold in percent (default: -20)")
    parser.add_argument("--no-promote", action="store_true",
                        help="Don't copy the champion model to runtime/champion_lane_b_{SYMBOL}_model.zip")
    args = parser.parse_args()

    champion = re_evaluate(args.scorecard, max_dd=args.max_dd, promote=not args.no_promote)
    sys.exit(0 if champion is not None else 1)
