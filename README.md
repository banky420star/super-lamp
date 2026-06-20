# supreme-chainsaw

## Quick Start

### 1. Start the API Server

```bash
.venv312/Scripts/python.exe Python/api_server.py
```

This starts the backend API on **port 5050** (required by the trading dashboard).

### 2. Start the Trading Dashboard

```bash
cd frontend
npm install   # first time only
npm run dev
```

Open **http://localhost:5173** in your browser. The dashboard auto-refreshes every 15 seconds.

### 3. Run the Autonomous Pipeline

```bash
# Full autonomous run (trains models, validates, promotes to champion)
.venv312/Scripts/python.exe Python/autonomous/run_cycle.py     --symbol XAUUSDm --timeframe M5 --timesteps 500000     --feature-set-id prod_v1 --dataset-id prod_v1

# Force champion promotion (bypass canary approval)
.venv312/Scripts/python.exe Python/autonomous/run_cycle.py     --symbol XAUUSDm --timeframe M5 --timesteps 500000     --feature-set-id prod_v1 --dataset-id prod_v1 --force-champion
```


**Regime-routed PPO trading agent with real feature ablation testing for XAUUSDm (Gold).**

---

## Overview

This project trains reinforcement learning trading agents (PPO-based) on real market data and systematically ablates feature groups to identify which engineered features actually help — and which are noise.

The core question: *Which of our 59 engineered features drive performance, and which should be removed?*

---

## Prerequisites

**Python 3.14.5** — pinned in [`.python-version`](.python-version) for reproducible builds.

If you use `pyenv`, the correct version is auto-selected when entering the project directory.

### Setup

```bash
python -m venv venv
venv/Scripts/activate   # Windows: venv\Scripts\activate; Unix: source venv/bin/activate
pip install -r requirements.txt
```

Verify:

```bash
python --version   # should print Python 3.14.5
```

---

## Architecture

```
Market Data (XAUUSDm M5)
       ↓
ENGINEERED_V2 Feature Pipeline (59 columns)
       ↓
Windowed Observations (100 bars × 59 features + regime)
       ↓
AdaptiveLSTMFeatureExtractor
  ├── Bidirectional LSTM (2-layer, hidden=160)
  ├── Multi-Head Attention Pooling (4 heads)
  ├── Projection (→ 256-dim)
  ├── [opt] TrendMomentumBiasLayer (6 bias features)
  └── [opt] FeatureGroupGate (regime-conditional)
       ↓
RegimeRoutedPPO Policy
  ├── Regime classifier (5 regimes)
  ├── Actor heads (per-regime)
  └── Critic head
       ↓
Action: position size, direction, exits
```

### Key Components

| Component | File | Purpose |
|---|---|---|
| **Real Feature Ablation Harness** | `training/run_real_feature_ablation.py` | Trains PPO with feature groups ablated, measures impact |
| **AdaptiveLSTMFeatureExtractor** | `drl/adaptive_feature_extractor.py` | Bidirectional LSTM + attention + projection |
| **TrendMomentumBiasLayer** | `drl/trend_momentum_bias.py` | Soft directional prior (6 bias features) — currently parked |
| **FeatureGroupGate** | `drl/adaptive_feature_extractor.py` | Regime-conditional feature weighting |
| **RegimeRoutedPPO** | `drl/regime_routed_policy.py` | Per-regime actor heads + shared critic |
| **ENGINEERED_V2 Pipeline** | `Python/feature_pipeline.py` | 59-column feature matrix builder |

---

## Feature Ablation Groups

The harness tests which feature groups help vs hurt by zeroing out specific columns:

| Group | Indices | What it removes | Columns |
|---|---|---|---|
| `ALL` | — | Nothing (baseline) | All 59 |
| `NO_TREND` | 19-20 | Trend indicator | `htf_trend`, `vol_bucket` |
| `NO_MOMENTUM` | 5-7 | Price momentum | `log_ret1`, `log_ret5`, `log_ret20` |
| `NO_VOLATILITY` | 12 | Realized volatility | `rv_20` |
| `NO_VOLUME` | 13-14 | Volume + spread | `rel_volume`, `spread_est_bps` |
| `NO_CROSS_ASSET` | 40-57 | Cross-asset correlations | 18 columns |
| `NO_ML_SIGNAL` | 58 | XGBoost direction | `ml_signal_prob` |
| `NO_PATTERN` | 29-39 | Candlestick patterns | 11 columns |
| `NO_REGIME` | — | Regime routing | Disables regime detector |
| `TREND_MOMENTUM_FIRST` | — | Bias layer ON | All features + bias layer |
| `NO_BIAS_SATURATION` | — | Fixed temp=0.1 | Bias layer with clamped temperature |

### Fingerprint Verification

Every group's feature matrix is fingerprinted (MD5 hash) before training and compared against the `ALL` baseline:

```python
def matrix_fingerprint(x: np.ndarray) -> str:
    arr = np.nan_to_num(x).astype(np.float32)
    return hashlib.md5(arr.tobytes()).hexdigest()[:12]
```

If a group that *should* differ produces an identical fingerprint, the harness raises `AssertionError` — the ablation mask is broken.

---

## Key Findings

### Full 30K-step Ablation (with LSTM extractor)

| Rank | Group | Sharpe | WinRate | PF | MaxDD | Verdict |
|---|---|---|---|---|---|---|
| 1 | NO_TREND* | +10.94 | 51.2% | 1.12 | -2.0% | Trend features harmful |
| 2 | NO_MOMENTUM | +9.54 | 51.2% | 1.12 | -2.0% | Momentum harmful |
| 3 | NO_VOLUME | -2.76 | 49.9% | 1.08 | -4.1% | Volume slightly helpful |
| 4 | NO_REGIME | -10.94 | 48.8% | 0.89 | -6.0% | Regime routing helps |
| 5 | NO_ML_SIGNAL | -12.76 | 50.2% | 1.05 | -6.8% | ML signal helps |
| 6 | NO_VOLATILITY | -21.10 | 48.8% | 0.90 | -8.9% | Volatility helps |
| 7 | NO_PATTERN | -23.48 | 49.7% | 0.94 | -10.2% | Patterns help |
| 8 | NO_CROSS_ASSET | -30.62 | 49.0% | 0.94 | -9.9% | Cross-asset very important |
| 9 | ALL (baseline) | -37.12 | 49.3% | 0.91 | -11.0% | Full set worst — too noisy |

> **\*Important**: The original NO_TREND result was invalid — see Bug Fix section below.

### TrendMomentumBiasLayer — Parked

The bias layer (soft directional prior) was tested extensively and found to add noise rather than signal:

| Configuration | Sharpe | Bias behavior |
|---|---|---|
| `NO_BIAS_SATURATION` (fixed temp=0.1) | Best | Bias stays neutral → closest to ALL |
| `ALL` (no bias) | Baseline | — |
| `TREND_MOMENTUM_FIRST` (learnable temp) | Worst | Bias recalibrates but degrades performance |

The bias layer is currently parked — it will be rebuilt as an isolated risk-sizing encoder later.

---

## Critical Bug Fix (June 2026)

**`trend` group indices were wrong:**

| Before (bug) | After (fix) |
|---|---|
| `[15, 16]` → `hour_sin`, `hour_cos` (time-of-day) | `[19, 20]` → `htf_trend`, `vol_bucket` (trend indicator) |

The original `NO_TREND` ablation was actually removing **time-of-day features**, not trend features. All six other group indices were verified correct against the pipeline column order.

Fixed in commit `30040b2`. The `NO_TREND` results above are from the **original** (buggy) run and need re-validation.

---

## Running the Harness

### Quick smoke test (fingerprint verification)

```bash
python training/run_real_feature_ablation.py \
  --symbol XAUUSDm \
  --steps 1000 \
  --groups ALL NO_TREND NO_MOMENTUM \
  --n-bars 1000 \
  --verbose
```

### Full ablation study

```bash
python training/run_real_feature_ablation.py \
  --symbol XAUUSDm \
  --steps 30000 \
  --groups ALL NO_TREND NO_MOMENTUM NO_VOLATILITY NO_VOLUME \
           NO_CROSS_ASSET NO_ML_SIGNAL NO_PATTERN NO_REGIME \
  --n-bars 5000 \
  --verbose
```

### Bias layer diagnostics

```bash
python training/run_real_feature_ablation.py \
  --symbol XAUUSDm \
  --steps 5000 \
  --groups NO_BIAS_SATURATION TREND_MOMENTUM_FIRST ALL \
  --n-bars 3000 \
  --verbose
```

---


## Autonomous Pipeline

The autonomous pipeline (`Python/autonomous/run_cycle.py`) runs the full training-to-deployment lifecycle:

1. **Data Intake** - MT5 market data ingestion
2. **Feature Factory** - 59 engineered features with ablation testing
3. **Model Training** - LSTM, Rainforest, Dreamer, PPO models
4. **Model Bundle** - Version-locked ensemble bundle with all model IDs
5. **Validation** - Backtest court, walk-forward, baseline comparison
6. **Promotion Gates** - 8 gate categories (data, training, performance, stability, baseline, canary, safety, execution)
7. **Demo Canary** - Real backtester evaluation via FastBacktester
8. **Champion Promotion** - Automated canary-to-champion promotion

### CLI Usage

```bash
# Full autonomous run with force-champion (bypasses canary approval)
python Python/autonomous/run_cycle.py     --symbol XAUUSDm --timeframe M5     --timesteps 500000     --feature-set-id prod_v1 --dataset-id prod_v1     --force-champion

# Standard run (requires canary approval for champion promotion)
python Python/autonomous/run_cycle.py     --symbol XAUUSDm --timeframe M5     --timesteps 500000     --feature-set-id prod_v1 --dataset-id prod_v1
```

### Champion Cycle

`tools/champion_cycle.py` runs on a 30-minute loop (`tools/champion_cycle_loop.py`) to:
- Train fresh candidate models
- Evaluate candidates vs current champion
- Promote winning candidates to canary status
- Promote approved canaries to champion

### Model Lifecycle

```
Train -> Bundle -> Validate -> Promotion Gates -> Demo Canary -> Champion
```

Bundle statuses: `candidate` | `validation_pending` | `rejected` | `demo_canary` | `champion` | `retired` | `quarantined`

## Trading Dashboard

The Vite + React frontend (`frontend/`) provides a real-time trading dashboard:

- **Portfolio Overview** - Equity, daily P&L, floating P&L, drawdown
- **Model Status** - Champion and canary models per symbol
- **Pipeline Status** - All 11 pipeline stages
- **Trade Summary** - Win rate, profit factor, total P&L
- **Risk Limits** - Max daily loss, drawdown limits, trading halt status
- **Recent Trades** - Trade history with entry/exit prices

### Running the Dashboard

```bash
cd frontend
npm install
npm run dev    # Development server on port 5173
npm run build  # Production build
```

The dashboard proxies API requests to `http://localhost:5050` (the AGI API server).

## Promotion Gates

The promotion gate system (`Python/registry/promotion_gates.py`) evaluates bundles across 8 categories:

| Gate | Key Thresholds |
|------|----------------|
| Data | source=mt5, no leakage, feature audit passed |
| Training | timesteps>=10K, seed logged, dataset/feature IDs |
| Performance | OOS return>=2%, profit factor>=1.15, Sharpe>=0.50, max DD<=8% |
| Stability | >=3 walk-forward windows, stress test passed |
| Baseline | Beat random, buy-and-hold, previous champion |
| Canary | >=50 trades, >=7 days, positive PnL |
| Safety | No corruption, no outlier behavior |
| Execution | Quality>=0.60, trailing success>=0.35, risk adherence>=0.80 |

## Branch Strategy

**Trading core (do not mix platform changes here):** `experiment/xauusd-regime-baseline` (and safe descendants only for core DRL/regime/reward/training/execution changes).

**Platform / UI / Ops layer:** `feature/dashboard-mission-control`, `feature/safety-registry-core`, `feature/ci-compile-gates`, `feature/runtime-hygiene`, `refactor/profitability-tier0-reward-structure`, `feature/*` (off `main`).

See `SETUP.md` section 7 for full rules, separation of concerns, and "new branch first" hygiene. Never edit `main` or `experiment/xauusd-regime-baseline` directly.

## Status

- ✅ Real feature ablation harness working end-to-end
- ✅ LSTM feature extractor integrated
- ✅ Fingerprint verification prevents silent mask failures
- ✅ Bias layer diagnostics in place
- ✅ Trend column index bug fixed
- 🔄 Re-run full ablation with corrected indices
- 🔄 Clean feature pipeline based on ablation results
- 🔄 Rebuild bias layer as isolated risk-sizing encoder

---

## Key Commits

| Hash | Description |
|---|---|
| `30040b2` | Fix trend group indices [15,16]→[19,20] |
| `286d045` | Add NO_BIAS_SATURATION ablation group |
| `4bb7fa9` | Fix bias layer saturation (temperature-scaled activations) |
| `22f334d` | Add bias layer diagnostics to harness |
| `d69b800` | Upgrade harness to use AdaptiveLSTMFeatureExtractor |
| `84d2fd4` | Real feature ablation harness created |

## License

Code under `02_Core_Python/` is derived from
[`banky420star/upgraded-spoon`](https://github.com/banky420star/upgraded-spoon)
and distributed under the MIT License as documented there. All other code in this
repository is rights-reserved to the author(s) until a top-level
`LICENSE` is added.
