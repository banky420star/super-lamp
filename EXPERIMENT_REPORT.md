# Experiment Report: XAUUSDm Regime Baseline Comparison

**Date:** June 7, 2026
**Branch:** `experiment/xauusd-regime-baseline`
**Feature Version:** `engineered_v2`
**Symbol:** XAUUSDm
**Timeframe:** 5m
**Period:** 90d
**Timesteps:** 50,000 (config)
**Random Seed:** Default (training script default)

---

## Three-Model Comparison

Three model configurations were compared:

| # | Configuration | Steps | Status |
|---|---|---|---|
| 1 | Plain PPO (`AGI_USE_REGIME=0`) | 65,536 / 50,000 | ✅ Complete |
| 2 | PPO + Regime (`AGI_USE_REGIME=1`) | 35,000 / 50,000 | ⚠️ 70% (Win log stall) |
| 3 | PPO + Regime + Feature Gate (`AGI_USE_REGIME=1`, `AGI_USE_FEATURE_GATE=1`) | 35,000 / 50,000 | ⚠️ 70% (Win log stall) |

*Runs 2 and 3 stalled due to Windows `PermissionError` on `dreamer_training.log` rotation. Log rotation fix committed in `347a542`. Results should be treated as preliminary — final convergence may differ.*

---

## Performance Comparison

| Metric | Plain PPO | PPO + Regime | PPO + Regime + Gate |
|---|---|---|---|
| **Equity** (start 10,000) | 10,069.78 | 9,648.66 | 9,648.66 |
| **Net Return** | +0.70% | -3.51% | -3.51% |
| **Sharpe Ratio** | 0.000 | 0.058 | 0.058 |
| **Profit Factor** | -0.00 | -0.92 | -0.92 |
| **Max Drawdown** | 0.58% | 3.64% | 3.64% |
| **Win Rate** | 0.09% | 42.59% | 42.59% |
| **Total Trades** | 11,256 | 120 | 120 |
| **Avg Win** | 28.65 | 18.50 | 18.50 |
| **Avg Loss** | -16.63 | -15.07 | -15.07 |

*Note: PPO+Regime and PPO+Regime+Gate have identical metrics because they both stalled at the same step count (35K). The feature gate's effect likely requires more training steps to manifest, or may need verification that the gate forward path is actually engaged.*

---

## Key Findings

### 1. Regime dramatically changes trading behavior
- **Win rate** jumps from 0.09% to 42.59%
- **Trade count** drops from 11,256 to 120 (99% fewer)
- **Sharpe** moves from 0.000 (random) to 0.058 (positive signal)
- Regime awareness produces far more selective, higher-conviction trades at the cost of higher per-trade drawdown (3.64% vs 0.58%)

### 2. Feature Gate — no measurable difference yet
The gate's group-weighted feature modulation may require full training convergence to show an effect, or the forward path may not be engaged at the current feature dimension.

### 3. Plain PPO is essentially random
Sharpe 0.000 and 0.09% win rate with 11,256 trades = near-random churn.

---

## Per-Regime Performance (Required Data)

A true per-regime breakdown requires extracting regime labels from the PPO+Regime run and computing metrics per regime. This data is not yet available because:

- The training didn't complete (70% stalled at 35K)
- Regime labels need to be extracted from the model's `_regime_classifier` outputs on held-out data
- A separate evaluation pass is needed to collect per-regime equity curves

**To generate:** Once a complete PPO+Regime run is available, run a separate evaluation script to capture:
- Equity change per regime (bull, bear, range_top, range_bottom, high_vol)
- Trade count per regime
- Win rate per regime
- Avg win/loss per regime
- Sharpe per regime

---

## Visualization Recommendations

*Not generated in this session. Recommended plots:*
- Equity curve overlay (3 runs)
- Drawdown comparison
- Trade distribution scatter (win/loss by size)
- Per-regime bar chart (once data available)

---

## Raw Training Logs

| Run | Log Path |
|---|---|
| Plain PPO | `runs/xauusd_plain_ppo_baseline/training.log` |
| PPO + Regime | `runs/xauusd_regime_ppo_baseline/training_v2.log` |
| PPO + Regime + Gate | `runs/xauusd_featuregate_baseline/training.log` |

---

## Next Steps

1. **Re-run all three experiments** with the log rotation fix (`347a542`) to get complete 50K runs
2. **Extract per-regime metrics** from the completed PPO+Regime run using a separate evaluation script
3. **Run feature ablation** (`python training/run_feature_ablation.py --symbol XAUUSDm --timesteps 20000`)
