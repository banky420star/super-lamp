# Super-Lamp Architecture Map

> *Last updated: June 2026*

## Mental Model

```
Rainforest / Random Forest  =  scout (regime + signal + importance)
PPO / LSTM                  =  driver (Long / Flat / Short)
Dreamer                     =  simulator + strategy laboratory
Feature Auditor             =  mechanic inspector
Promotion Gates             =  judge
Paper / Demo                =  test track
Champion                    =  approved vehicle
```

---

## Full Pipeline

```
MT5 / Data Feed
  │
  ▼
Provenance + Hash
  │
  ▼
Feature Builder (ENGINEERED_V2 / ULTIMATE_150)
  │
  ▼
Feature Audit (leakage, correlation, importance, stability)
  │
  ▼
┌─────────────────────────────────────────────┐
│         Rainforest Detector                  │
│  ─────────────────────────────────────────── │
│  Outputs:                                    │
│  • regime / confidence (routing)             │
│  • ml_signal (directional probability)       │
│  • feature_importances (audit integration)   │
└─────────────────┬───────────────────────────┘
  │
  ├──▶ RegimeController (ensemble weights, risk adaptation)
  │
  ├──▶ FeatureAuditor (dead-column detection via integrate_rf_importances)
  │
  ├──▶ ml_signal column → Lane A (ENGINEERED_V2 feature matrix)
  │
  ▼
┌─────────────────────────────────────────────┐
│              Dreamer World Model             │
│  ─────────────────────────────────────────── │
│  Learns: features + action → next state     │
│  Imagines: future rollouts for strategy     │
│  Outputs: candidate policies, stress tests  │
│  Does NOT trade live                        │
└─────────────────┬───────────────────────────┘
  │
  ├──▶ Candidate policy ideas → PPO training
  ├──▶ Synthetic scenarios → training curriculum
  ├──▶ Regime stress testing → PromotionGates
  └──▶ Reward design diagnosis
  │
  ▼
┌─────────────────────────────────────────────┐
│              Training Lanes                  │
│  ─────────────────────────────────────────── │
│                                              │
│  Lane A: Engineered PPO                      │
│    • Features: ENGINEERED_V2                 │
│    • Architecture: SB3 PPO + DecisionHead    │
│    • Obs dim: 60 (includes rf_ml_signal)     │
│                                              │
│  Lane B: Raw LSTM-PPO                        │
│    • Features: 7 raw (OHLCV + RSI + MACD)    │
│    • Architecture: LSTM extractor → PPO      │
│    • N_FEATURES = 7                          │
│                                              │
│  Lane C: MTF Regime-Routed PPO               │
│    • Features: Multi-timeframe + Rainforest  │
│    • Routing: RegimeController selects       │
│      regime-specific champion                │
│                                              │
│  Dreamer: World Model                        │
│    • DreamerV3: RSSM + Actor + Critic        │
│    • Imagined rollouts for discovery         │
└─────────────────┬───────────────────────────┘
  │
  ▼
Validation (Walk-Forward + OOS)
  │
  ▼
Promotion Gates (8 categories, strict thresholds)
  │
  ▼
Candidate Registry → Paper / Demo Canary
  │
  ▼
Champion (per-symbol, gated, paper-tested)
```

---

## Component Map: Code to Architecture

### 1. Data Ingestion

| Piece | File | Status |
|---|---|---|
| MT5 connector | `Python/data/ingest_mt5.py` | ✅ Fetches OHLCV via `mt5.copy_rates_from_pos` |
| Provenance | `Python/data/provenance.py` | ✅ Tracks `dataset_id`, `symbol`, `timeframe`, `source`, `broker`, `rows`, `missing_candles`, `leakage_checked`, `approved_for_training`, `dataset_hash` |
| Symbol metadata | `Python/data/symbol_metadata.py` | ✅ `ContractSpec` + `SYMBOL_REGISTRY` |
| Data validation | `Python/data/validate_data.py` | ✅ OHLC integrity, missing/duplicate timestamps, zero volume |
| Leakage detection | `Python/features/leakage_detector.py` | ✅ Forbidden prefixes, overlap, `_future`/`_lead` heuristics |
| Data feed | `Python/data_feed.py` | ⚠️ `load_real_data()` import fails (MT5 absent on non-broker machines) — synthetic fallback active |

### 2. Data Foundation (Black Box Recorder)

Located in `Python/runtime/data_foundation/`.

| CSV Table | Schema |
|---|---|
| `market_bars.csv` | `ts, symbol, timeframe, open, high, low, close, volume, dataset_hash, source` |
| `decisions.csv` | `decision_id, ts, symbol, model_id, dataset_hash, regime, action, confidence, side, lots, linked_run_id, meta_json` |
| `trades.csv` | `trade_id, decision_id, symbol, entry_ts, exit_ts, entry_price, exit_price, pnl, pnl_pct, bars_held, regime, model_id, dataset_hash, outcome` |
| `equity.csv` | `ts, run_id, equity, balance, drawdown_pct, open_positions, dataset_hash` |
| `training_runs.csv` | `run_id, ts, symbol, lane, steps, model_id, dataset_hash, feature_version, train_start, train_end, metrics_json, passed_gates` |
| `feature_audits.csv` | `audit_id, ts, symbol, feature_version, n_features, dead_cols, leakage_issues, passed, details_json` |
| `regime_logs.csv` | `ts, symbol, regime_score, zone, h1_bar_time, model_used, decision_id, dataset_hash` |

### 3. Feature Engineering

| Piece | File | Status |
|---|---|---|
| Feature registry | `Python/features/feature_registry.py` | ✅ `FeatureMeta` tracks family, status, leakage_risk. Auto-registers by prefix. `get_enabled()`, `get_live_allowed()` |
| Feature builder | `Python/features/build_features.py` | ✅ Builds ENGINEERED_V2 (trend, momentum, volatility, volume, cross-asset, ml_signal) |
| Feature pipeline | `Python/feature_pipeline.py` | ✅ Orchestrates ENGINEERED_V2 + ULTIMATE_150. Integrates PatternDetector, cross-asset, ml_signal, **now includes rf_ml_signal** |
| Multi-TF builder | `Python/features/multitimeframe_builder.py` | ✅ Builds 1m/5m/15m/1h features with symbol-specific params |
| Feature auditor | `Python/features/audit_features.py` | ✅ 8-check audit: leakage, correlation, importance (mutual_info/RF), stability (time + regime), live availability, missing rate, outlier rate, **now includes rf_importance_audit** |
| Cross-asset | `Python/cross_asset.py` | ✅ Computes DXY, US10Y, USDJPY correlations |
| ml_signal (XGBoost) | `Python/ml_signal.py` | ✅ Trains XGBoost/RF on feature matrix → next-bar direction probability |

**Feature registry — active groups:**

| Group | Columns | Status |
|---|---|---|
| `trend` | `htf_trend`, `vol_bucket` | ✅ Active |
| `momentum` | `log_ret1`, `log_ret5`, `log_ret20` | ✅ Active |
| `volatility` | `rv_20` | ✅ Active |
| `volume` | `rel_volume`, `spread_est_bps` | ✅ Active |
| `cross_asset` | `cross_asset_0 … cross_asset_5` | ✅ Active |
| `ml_signal` | `ml_signal`, `rf_ml_signal` | ✅ Active |
| `pattern` | — | 🔧 Paused (pattern detector missing) |
| `trend_momentum_first` | — | 🔧 Paused (experiment not validated) |
| `bias_saturation` | — | 🔧 Paused (experiment not validated) |

### 4. Rainforest Detector

| Piece | File | Status |
|---|---|---|
| Detector class | `Python/rainforest_detector.py` | ✅ RandomForestClassifier (200 trees, max_depth=12), 31 features, 7 regimes |
| Training script | `Python/training/train_rainforest.py` | ✅ Trains in isolation, maps regimes to policy configs |
| Pre-trained model | `models/rainforest_XAUUSDm.pkl` | ✅ Trained (97.6% confidence on synthetic) |

**Canonical outputs:**

| Output | Method | Consumer | Status |
|---|---|---|---|
| `regime` + `confidence` | `predict_regime(df)` | RegimeController, Dashboard | ✅ |
| `ml_signal` | `predict_ml_signal(df)` → `(n, 1) array` | Lane A feature matrix via `feature_pipeline.py` | ✅ |
| `feature_importances` | `export_feature_importances()` → `dict` | FeatureAuditor via `integrate_rf_importances()` | ✅ |

**Wiring diagram:**

```
RainforestDetector
  │
  ├── predict_regime(df)
  │     → regime / confidence → RegimeController.adapt_trade_decision()
  │     → regime / confidence → Dashboard endpoint /api/patterns/rainforest
  │     → regime / confidence → data_foundation/regime_logs.csv
  │
  ├── predict_ml_signal(df) [NEW]
  │     → (n, 1) directional probability [0, 1]
  │     → compute_rainforest_ml_signal(symbol, df) in ml_signal.py
  │     → _build_engineered_env_matrix() in feature_pipeline.py
  │     → Lane A PPO observation space (obs_dim increased from 59 → 60)
  │     → Feature registry: rf_ml_signal in ml_signal family
  │
  └── export_feature_importances() [NEW]
        → {feature_name: importance} dict
        → FeatureAuditor.integrate_rf_importances() in:
            • run_cycle.py stage_feature_audit() (autonomous cycle)
            • dashboard_backend.py _lazy_imports() (startup audit)
        → Dead-column detection in feature_audit report
```

### 5. Training Lanes

| Lane | File | Architecture | Status |
|---|---|---|---|
| **Lane A** | `training/train_drl.py` + `drl/trading_env.py` | SB3 PPO, ENGINEERED_V2 (40+ features), DecisionHead (18-dim action), `TradingReward`, MTF support | ✅ |
| **Lane B** | `training/run_lane_b_raw_lstm.py` | 7 raw features (OHLCV log returns + RSI + MACD), LSTMFeatureExtractor (2-layer, hidden=128), discrete Long/Flat/Short, 3-seed walk-forward | ✅ |
| **Lane C** | Implicit (via RegimeController) | MTF data available + RegimeController routing, but no dedicated training script for regime-specific champions | 🟡 Partially built |
| **Dreamer** | `training/train_dreamer.py` + `drl/dreamer_agent.py` | DreamerV3: Encoder, RSSM, Decoder, RewardPredictor, Actor, Critic. Separate optimizers. Imagined rollouts. | ✅ |

### 6. Dreamer (World Model)

| Piece | File | Status |
|---|---|---|
| Dreamer agent | `drl/dreamer_agent.py` | ✅ DreamerV3 with full world model architecture |
| Training script | `training/train_dreamer.py` | ✅ |
| Saved models | `models/dreamer/` | ✅ Multiple checkpoints |

**What Dreamer does:**
- Learns: `current features + action → predicted next state → predicted reward → predicted risk`
- "Dreams" possible futures for strategy discovery, NOT live trading
- Outputs: candidate policy ideas, synthetic scenario performance, risk warnings, reward-shaping suggestions

**Where Dreamer fits:**
```
Features → Dreamer world model → imagined rollouts
  → candidate behaviors discovered
  → PPO trains / is evaluated on those rollouts
  → Real-data validation + walk-forward
  → Promotion gates decide
  → Paper/Demo proves
  → Champion
```

### 7. Promotion Gates → Champion Pipeline

| Piece | File | Status |
|---|---|---|
| Promotion gates | `Python/registry/promotion_gates.py` | ✅ 8 categories: Data, Training, Performance, Rich Exec, Stability, Baseline, Canary, Safety |
| Promote script | `Python/registry/promote.py` | ✅ Evaluates bundle → `demo_canary` or `rejected` |
| Model registry | `Python/model_registry.py` | ✅ File-based, `active.json` tracks champion/canary per symbol, file-lock safe |
| Paper executor | `Python/execution/paper_executor.py` | ✅ Simulated fills with SpeedSimulator, auto ticket numbering (900001+) |
| Executor router | `Python/execution/executor_router.py` | ✅ Routes: `paper_sim` → PaperExecutor, `demo_live`/`real_live` → MT5DemoExecutor |
| Gate engine | `Python/execution/gate_engine.py` | ✅ Pre-flight: kill switch, mode check. Intent: symbol allowlist, model file exists, confidence ≥ 0.6, data freshness < 300s |

**Promotion gate thresholds:**

| Category | Criteria | Threshold |
|---|---|---|
| Performance | OOS return | ≥ 0.02 |
| Performance | Profit factor | ≥ 1.15 |
| Performance | Sharpe | ≥ 0.50 |
| Performance | Max drawdown | ≤ 0.08 |
| Performance | Trade count | ≥ 50 |
| Performance | Single trade profit share | ≤ 0.20 |
| Stability | Walk-forward windows passed | ≥ 3 |
| Training | Timesteps | ≥ 10,000 |
| Canary | Demo trades | ≥ 50 |
| Canary | Demo days | ≥ 7 |
| Rich Exec | Execution quality | ≥ 0.60 |
| Rich Exec | Trailing success rate | ≥ 0.35 |
| Rich Exec | Risk sizing adherence | ≥ 0.80 |
| Rich Exec | Decision success rate | ≥ 0.85 |

### 8. Orchestration & API

| Piece | File | Status |
|---|---|---|
| SmartAGI (inference) | `Python/agi_brain.py` | ✅ LSTM-based, 3-class (HOLD/BUY/SELL), integrates RainforestDetector for risk gating |
| Autonomy loop | `Python/autonomy_loop.py` | ✅ Orchestrates training, evaluation, promotion, perpetual improvement logging |
| Run cycle | `Python/autonomous/run_cycle.py` | ✅ Stage-based pipeline: safety, data, features, audit, labels, rainforest training, PPO, backtest, walk-forward, promotion, canary |
| Hybrid brain | `Python/hybrid_brain.py` | ✅ Multi-signal aggregation |
| Safety gates | `Python/safety_gates.py`, `Python/live_safety.py` | ✅ Kill switch, account telemetry, pytest validation |
| Dashboard backend | `Python/dashboard_backend.py` | ✅ FastAPI bridge, 20+ endpoints, lazy imports |
| API server | `Python/api_server.py` | ✅ Bottle-based lightweight bridge on :5050, WebSocket support |

---

## Status Summary

### ✅ Fully implemented and wired
- MT5 ingestion with provenance tracking
- Data foundation (7 CSV tables)
- Feature builder (ENGINEERED_V2 + ULTIMATE_150) with multi-TF support
- Feature registry (active/paused groups, auto-registration)
- Feature auditor (8 checks + RF importance audit integration)
- Rainforest regime classifier (3 canonical outputs: regime, ml_signal, feature_importances)
- Regime adaptive controller (ensemble weights per regime)
- Training lanes A (engineered PPO), B (raw LSTM-PPO), Dreamer (world model)
- Promotion gates (8 categories, strict thresholds)
- Model registry (champion/canary per symbol with file locking)
- Paper trader with SpeedSimulator
- Executor router (paper → demo → live)
- AGI inference engine (LSTM, 3-class)
- Autonomy loop + run cycle
- Dashboard backend + React frontend

### ⚠️ Partially wired / needs attention
- **Lane C (MTF Regime-Routed)** — RegimeController can route but no dedicated training script
- **`data_feed.load_real_data()`** — Import fails (MT5 absent), synthetic fallback works but live data path is broken
- **Dreamer → PPO curriculum** — Dreamer produces rollouts but they're not explicitly piped as training curriculum

### ❌ Missing
- **Lane C training script** — No script that trains regime-specific champions and routes via RegimeController
- **Feature importance → registry feedback loop** — Feature importances from Rainforest should automatically update the registry (dead columns → `disabled_noise`, unstable → `disabled_unstable`)
