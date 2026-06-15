# Super-Lamp — Full Training & Evolution Pipeline

> A **trading research and promotion factory**. Lane B is one worker inside the factory.
> The pipeline goes from MT5 ingestion all the way to champion promotion and continuous retraining.

---

## Master Pipeline Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                     SUPER-LAMP PIPELINE                            │
└────────────────────────────────────────────────────────────────────┘

[MT5 / Broker / Cache]
        │
        ▼
[Data Ingestion]
        │
        ├─ candles
        ├─ ticks
        ├─ spread
        ├─ volume
        ├─ account/equity snapshots
        └─ symbol metadata
        │
        ▼
[Data Provenance + Data Foundation]
        │
        ├─ dataset_hash (SHA-256)
        ├─ missing candle checks
        ├─ duplicate timestamp checks
        ├─ timezone check
        ├─ leakage check
        ├─ approved_for_training
        └─ approved_for_champion_training
        │
        ▼
[Feature Builder + Feature Registry]
        │
        ├─ Engineered V2 features
        ├─ Raw OHLCV/LSTM features
        ├─ MTF/regime features
        ├─ Cross-asset features
        ├─ ML signal features
        └─ Feature groups / ablation groups
        │
        ▼
[Feature Audit]
        │
        ├─ Dead columns
        ├─ NaN / inf checks
        ├─ Leakage checks
        ├─ Correlation checks
        ├─ Predictive importance
        ├─ Stability over time
        ├─ Stability across regimes
        └─ Live availability
        │
        ▼
[Training Lanes]
        │
        ├─ Lane A: Engineered features
        ├─ Lane B: Raw OHLCV + LSTM PPO
        ├─ Lane C: MTF regime-weighted PPO
        ├─ Lane D / future: Expanded regime/MTF variants
        ├─ Rainforest regime classifier
        ├─ Dreamer/world model
        └─ PPO/DRL variants
        │
        ▼
[Evaluation]
        │
        ├─ In-sample metrics
        ├─ Out-of-sample validation
        ├─ Walk-forward windows
        ├─ Baseline comparisons
        ├─ Stress tests
        ├─ Action distribution checks
        └─ Drawdown/trade/equity metrics
        │
        ▼
[Promotion Gates]
        │
        ├─ Data gates
        ├─ Training gates
        ├─ Performance gates
        ├─ Stability gates
        ├─ Baseline gates
        ├─ Demo canary gates
        └─ Safety gates
        │
        ▼
[Model Registry]
        │
        ├─ Candidate
        ├─ Rejected
        ├─ Quarantined
        ├─ Canary
        └─ Champion
        │
        ▼
[Paper / Demo Execution]
        │
        ├─ Model decision
        ├─ Safety gate
        ├─ Paper order / Demo order
        ├─ Decision log
        ├─ Trade journal
        └─ Equity curve
        │
        ▼
[Evolution Loop]
        │
        ├─ Trade coroner
        ├─ Replay builder
        ├─ Failure clustering
        ├─ New data ingestion
        ├─ Retraining trigger
        ├─ New candidate training
        └─ Promotion attempt
```

---

## 1. Data Ingestion Layer

The first stage is not "train model." It is **prove the data is usable.**

The provenance object (`Python/data/provenance.py`) tracks:

| Field | Purpose |
|-------|---------|
| `dataset_id` | Unique identifier |
| `symbol` | Trading symbol (XAUUSDm, BTCUSDm, etc.) |
| `timeframe` | Bar timeframe |
| `source` | Data origin (mt5, cache, file) |
| `broker` | Broker name |
| `start` / `end` | Date range |
| `rows` | Bar count |
| `missing_candles` | Gap count |
| `duplicate_timestamps` | Duplicate count |
| `spread_included` | Whether spread data is present |
| `commission_model` | Commission structure |
| `slippage_model` | Slippage assumptions |
| `timezone_checked` | UTC alignment verified |
| `leakage_checked` | No future data contamination |
| `dataset_hash` | SHA-256 hash of data content |
| `approved_for_training` | Meets training quality bar |
| `approved_for_champion_training` | Meets stricter champion bar |

### Approval Gates

**Training approval** requires: enough rows, spread data if required, timezone check, leakage check, low missing/duplicate ratios.

**Champion training approval** is stricter: requires the dataset to already be approved for training AND additional quality thresholds.

### Data Foundation Tables (CSV-based)

```
market_bars      → Raw OHLCV with provenance
decisions         → Every model decision logged
trades            → Every executed trade
equity            → Equity curve snapshots
training_runs     → All training experiments
feature_audits    → Feature audit results
regime_logs       → Regime detection events
```

> Later this can move to Postgres/Timescale/TigerData. CSV is enough for current paper/demo phase.

---

## 2. Feature Registry & Grouping

**File:** `Python/feature_registry.py`

All feature-column code must import from this single source of truth.

### Engineered V2 Feature Families

```
Price-relative OHLC:
  open_rel, high_rel, low_rel, close_rel

Volume:
  log_vol

Momentum:
  log_ret1, log_ret5, log_ret20

Candle Geometry:
  body_ratio, upper_wick, lower_wick, range_ratio

Volatility:
  rv_20

Volume/Spread:
  rel_volume, spread_est_bps

Trend/Regime:
  htf_trend, vol_bucket

Time/Calendar:
  hour_sin, hour_cos, dow_sin, dow_cos

Session/News:
  session_london, session_ny, major_open
  news_prox, news_soon, session_overlap
  mins_since_london, news_avoid

Pattern Slots:
  pattern_0 … pattern_10

Cross-Asset Slots:
  cross_asset_0 … cross_asset_17

ML Signal:
  ml_signal
```

> `N_FEATURES = len(ENGINEERED_V2_COLUMNS)` is the source of truth. Audit this every time.

### Active Feature Groups

| Group | Columns |
|-------|---------|
| `trend` | `htf_trend`, `vol_bucket` |
| `momentum` | `log_ret1`, `log_ret5`, `log_ret20` |
| `volatility` | `rv_20` |
| `volume` | `rel_volume`, `spread_est_bps` |
| `cross_asset` | `cross_asset_0` to `cross_asset_5` |
| `ml_signal` | `ml_signal` |

The registry derives feature indices from **names**, not hand-coded index guesses.

### Paused Feature Groups

| Group | Reason |
|-------|--------|
| `pattern` | Pattern detector missing |
| `trend_momentum_first` | Adaptive LSTM bias experiment not validated |
| `bias_saturation` | Bias fixed-temperature experiment not validated |

### Ablation Groups

```
ALL
NO_TREND
NO_MOMENTUM
NO_TREND_MOMENTUM
NO_VOLATILITY
NO_VOLUME
NO_CROSS_ASSET
NO_ML_SIGNAL
NO_REGIME
```

Ablation indices are derived from feature names.

---

## 3. Feature Audit Layer

**File:** `FeatureAuditor.run_full_audit()`

Checks every feature before training:

| Check | What It Catches |
|-------|-----------------|
| **Leakage** | Target-like columns, overlap between features and labels |
| **Correlation** | Highly correlated feature pairs |
| **Predictive importance** | Mutual information or random forest importance |
| **Time stability** | Feature mean/std drift over time splits |
| **Regime stability** | Feature behavior across market regimes |
| **Live availability** | Production-expected features actually present |
| **Missing rate** | NaN proportion |
| **Outlier rate** | Extreme value proportion |

> This layer prevents "mystery soup features" from polluting training.

---

## 4. Training Lanes (A, B, C)

### Lane A: Engineered-Feature Lane

```
Input:    Engineered V2 feature matrix
Purpose:  Use named technical/session/cross-asset/ML features
Strength: Explainable feature groups and ablations
Risk:     Feature pipeline can go flat, include dead columns,
          leak, or overcompress validation data
```

Lane A is the **"feature science" lane.** It should use the feature registry and feature audit heavily.

---

### Lane B: Raw OHLCV + LSTM PPO Lane

**File:** `training/run_lane_b_raw_lstm.py` (524 lines)

This is a raw sequence lane, not the full system.

#### Lane B Features (7 only)

```
0: open log return
1: high log return
2: low log return
3: close log return
4: volume log return
5: RSI(14)
6: MACD histogram normalized by price

N_FEATURES = 7
```

#### Lane B Architecture

```
INPUT: 7 features, window_size=64 bars (320 min)
┌─────────────────────────────────────────────────────────


---

## 5. The Direction Trap

### Root Cause (5 Factors)

| # | Cause | Detail |
|---|-------|--------|
| 1 | TURNOVER_COST = 0.0 | Model can hold position forever with zero friction |
| 2 | CONCENTRATION_PENALTY = 0.0 | Model can sit at position=-1.0 with no cost |
| 3 | Noisy reward signal | 5-bar M5 returns are mostly noise |
| 4 | Symmetry gap | Inverted data helps training, but eval is real-only |
| 5 | Only 7 features | No regime context, no ADX, no multi-TF signal |

### Net Effect

PPO converges to:  action = always SHORT, position = -1.0 (frozen), turnover = 0.0%, pos_std = 0.0000

### Detection (lines 510-515)

if avg_long > 10 and avg_short > 10 and avg_turnover > 5: TRAP BROKEN else: Direction trap persists.

> TRAP BROKEN only means it trades both directions. It does NOT mean profitable.

### Evidence

| Run | Symbol | Long% | Short% | Turnover | Return | Verdict |
|-----|--------|-------|--------|----------|--------|---------|
| seed 42 | XAUUSDm | 0.0 | 100.0 | 0.0% | -59.42% | DIRECTION TRAP |
| seed 456 | XAUUSDm | 35.6 | 46.1 | 14.3% | -53.93% | TRAP BROKEN (unprofitable) |

> The problem moved from action collapse to bad trading economics.

---

## 6. Evaluation & Walk-Forward

One validation split is not enough. The full pipeline uses walk-forward validation.

WalkForwardValidator builds overlapping train/validate windows: Train A -> Validate B, Train A+B -> Validate C, etc.

Returns: windows_total, windows_passed, windows_failed, mean_return_after_costs, worst_window_return, max_drawdown, per-window results.

---

## 7. Promotion Gates

### Default Thresholds

| Gate | Threshold |
|------|-----------|
| min_oos_return | 0.02 (2%) |
| min_profit_factor | 1.15 |
| min_sharpe | 0.50 |
| max_drawdown | 0.08 (8%) |
| min_trade_count | 50 |
| min_walk_forward_windows_passed | 3 |
| min_demo_canary_trades | 50 |
| min_demo_canary_days | 7 |
| min_timesteps | 10000 |

### Gate Categories

- Data Gates: data_source != mt5, missing spread, leakage, audit failed
- Training Gates: timesteps low, seed missing, dataset_id/feature_set_id missing
- Performance Gates: return, profit factor, Sharpe, drawdown, trade count
- Stability/Baseline/Canary/Safety Gates: walk-forward, regime breakdown, stress, baseline beats, demo canary, tests, telemetry, money locked

### Lane B Quick Rejection

Reject if: total_return <= 0, sharpe <= 0, max_drawdown < -30%, long_pct > 95%, short_pct > 95%, flat_pct > 95%, pos_std < 0.05, turnover < 0.5%

---

## 8. Evolution Training Loop

The full pipeline loops: ingest -> provenance -> feature build -> feature audit -> train -> OOS -> walk-forward -> gates -> reject/candidate -> paper canary -> demo canary -> champion -> trade/equity logs -> coroner -> retrain.

16 steps per cycle.

---

## 9. Lane B vs Full Super-Lamp Comparison

| Layer | Lane B (narrow) | Full Super-Lamp (platform) |
|-------|-----------------|---------------------------|
| Data source | MT5 -> load_real_data() | MT5/cache/ticks/deals/account snapshots |
| Data proof | Assumes data loaded | Dataset hash, approval gates |
| Features | 7 raw OHLCV/RSI/MACD | Engineered V2 + Lane A/B/C + audits |
| Training | Lane B PPO-LSTM | Lane A/B/C/D, Rainforest, Dreamer, PPO |
| Validation | One real split | OOS, walk-forward, baselines, stress, canary |
| Promotion | Printed conclusion | PromotionGates: 7 gate categories |
| Execution | Not main focus | Paper/demo, safety gate, kill switch, logs |
| Evolution | Manual rerun | Retrain -> validate -> promote/reject loop |
| Dashboard | Not core | Truth dashboard: all layers visible |

---

## 10. Correct Training Standard (Future)

For every future model: real data, dataset hash, feature audit, seed logged, symbol-specific path, no generic champion, metrics saved, walk-forward, baselines, paper canary, promotion gates, real money locked.

---

## 11. The Clean Final Target

MT5 data -> provenance + data foundation -> feature registry + audit -> multi-lane training -> walk-forward validation -> promotion gates -> symbol-specific champion -> paper/demo execution -> decision/trade/equity logs -> replay/coroner -> retrain better candidates.

**Not:** train one model -> save zip -> call it champion -> start live bot. That is how the goblin drives.

---

## 12. Key Source Files

| File | Purpose |
|------|---------|
| Python/feature_registry.py | Canonical feature names & groups |
| Python/data/provenance.py | Dataset hashing & approval |
| training/run_lane_b_raw_lstm.py | Lane B training (524 lines) |
| training/taming_shared.py | Gym env + reward + evaluate() |
| training/run_lane_c_mtf_regime.py | Lane C MTF regime training |
| training/run_lane_a_fix.py | Lane A engineered features |
| training/select_champion.py | Champion selection rules |
| training/eval_harness.py | Evaluation harness |
| training/live_trade_lane_b.py | Live trading from model |
| training/live_trade_lane_c.py | Lane C live trading |
| training/re_evaluate_champion.py | Champion re-validation |
| training/dashboard_backend.py | FastAPI dashboard backend |
| Python/api_server.py | API server host |
| frontend/src/App.tsx | React dashboard root |
