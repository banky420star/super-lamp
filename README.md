# CHAIN GAMBLER — Autonomous Trading Stack

> MT5-native AI ensemble (LSTM + PPO + Dreamer) with a self-evolving champion/canary model pipeline, live risk supervision, and a 13-tab React operator console.

## Table of Contents

- [Quick Start (Development)](#quick-start-development)
- [Quick Start (Production)](#quick-start-production)
- [Architecture Overview](#architecture-overview)
- [Configuration](#configuration)
- [Dashboard Tabs](#dashboard-tabs)
- [Risk Controls](#risk-controls)
- [Model Pipeline](#model-pipeline)
- [Troubleshooting](#troubleshooting)

---

## Quick Start (Development)

**Prerequisites (Windows for live):** Windows + MT5 + Python 3.12+.  
**macOS / Linux:** Dry-run / simulation mode only (real MT5 orders require Windows VPS or complex Wine bridge). Use `start_dryrun_mac.sh`.

```powershell
# 1. Create and activate the virtual environment
python -m venv .venv312
.venv312\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and configure secrets
Copy-Item config.yaml.example config.yaml
# Edit config.yaml: set mt5.login, mt5.password, mt5.server, telegram.token, telegram.chat_id

# 4. Start the server (dry-run by default everywhere; live needs Windows MT5)
#   macOS convenience: bash start_dryrun_mac.sh
python -m Python.Server_AGI

# 5. Start the dashboard (separate terminal)
python tools/project_status_ui.py
# Open: http://127.0.0.1:8088 (or the port your UI uses)

# Recommended one-command full production monitoring stack (React UI + api data layer + supervisor + TUI):
#   .\launch_full_project.ps1                 # Full observable stack (dev UI + proxy to 5050) -- NEW MTF STANDARD DEFAULT (1m+5m+15m+1h + best feats per symbol; TUI parity + React)
#   .\launch_full_project.ps1 -Preview        # Production preview build mode
#   .\launch_full_project.ps1 -DryRun         # Validate plan + detections only
#   (Legacy single-TF: AGI_USE_LEGACY_SINGLE_TF=1 ; see PRODUCTION.md 5.6)
# See script header for all flags, prerequisites, and Windows VPS notes.
```

Run the test suite to verify the installation:

```powershell
.venv312\Scripts\python.exe -m pytest
```

---

## Quick Start (Production)

See [PRODUCTION.md](PRODUCTION.md) for the full step-by-step deployment guide, including:

- MT5 bridge setup (Windows-native vs. Wine/Mac)
- Docker stack configuration
- Process supervision
- Health validation
- Backup and recovery

---

## Architecture Overview

```
                      ┌─────────────────────────────────────────┐
                      │         MetaTrader 5 Broker              │
                      │    (MT5 candles + order execution)       │
                      └──────────────────┬──────────────────────┘
                                         │
                      ┌──────────────────▼──────────────────────┐
                      │  data_feed.py → feature_pipeline.py      │
                      │  (150-feature ultimate_150 vector)        │
                      └──────────────────┬──────────────────────┘
                                         │
                      ┌──────────────────▼──────────────────────┐
                      │             HybridBrain                   │
                      │  (blends model signals into exposure)     │
                      └──┬──────────┬───────────┬───────────────┘
                         │          │           │
              ┌──────────▼─┐  ┌────▼────┐  ┌──▼────────┐
              │    LSTM     │  │   PPO   │  │  Dreamer  │
              │  (regime &  │  │ (policy │  │ (optional │
              │  context)   │  │ agent)  │  │  blend)   │
              └─────────────┘  └─────────┘  └───────────┘
                                         │
                      ┌──────────────────▼──────────────────────┐
                      │          Risk Engine + Supervisor         │
                      │  (halt gates, drawdown cap, spread cap)  │
                      └──────────────────┬──────────────────────┘
                                         │
                      ┌──────────────────▼──────────────────────┐
                      │             MT5 Executor                  │
                      │    (symbol-scoped orders + SL/TP)        │
                      └─────────────────────────────────────────┘

  Champion/Canary Pipeline:
  training/ → models/registry/candidates/ → model_evaluator.py
            → canary (shadow) → promote to champion → hot-swap in server

  Operator Surface:
  React Dashboard (port 8088) ←→ API Server (port 5000)
  Telegram Alerts ← audit_events.jsonl + trade_events.jsonl
```

**Key modules:**

| Module | Role |
|---|---|
| `Python/Server_AGI.py` | Main trading loop, risk supervision, MT5 execution |
| `Python/hybrid_brain.py` | Signal blending from LSTM + PPO + Dreamer |
| `Python/model_registry.py` | Champion/canary state, promotion policy, integrity |
| `Python/feature_pipeline.py` | Centralized 150-feature vector construction |
| `Python/data_feed.py` | MT5 candle acquisition and caching |
| `tools/champion_cycle.py` | Full retrain → evaluate → stage automation |
| `Python/autonomy_loop.py` | Scheduled canary evaluation and promotion |
| `tools/project_status_ui.py` | Dashboard + control API (port 8088) |

---

## Configuration

Copy `config.yaml.example` to `config.yaml`. The file is gitignored and is machine-local.

**Required fields:**

```yaml
mt5:
  login: 123456789
  password: "your_password"
  server: "Exness-MT5Trial9"

telegram:
  token: "YOUR_BOT_TOKEN"
  chat_id: "YOUR_CHAT_ID"

trading:
  symbols:
    - BTCUSDm
    - XAUUSDm
```

**Key configuration sections:**

| Section | Purpose |
|---|---|
| `trading` | Symbols, timeframe, confidence threshold, magic number |
| `risk` | Daily loss cap, max lots, per-symbol limits |
| `risk.supervisor` | Hard pre-trade gate: drawdown halt, spread cap, cooldown |
| `drl` | PPO training: timesteps, feature version, Dreamer settings |
| `training` | LSTM: epochs, period, candles, feature version |
| `evaluation` | Candidate pass criteria: Sharpe, drawdown, return margins |
| `registry.canary_policy` | Canary promotion thresholds per symbol |
| `event_intel` | News/economic calendar hold-out windows |

See `config.yaml.example` for all defaults and inline comments.

---

## Dashboard Tabs

The React dashboard runs at `http://127.0.0.1:8088` and exposes 13 tabs:

| Tab | What it shows |
|---|---|
| **Dashboard** | System health, account balance/equity, risk gauge, open positions, incident feed |
| **Trading** | Per-symbol lane status — live action, exposure, confidence, champion/canary indicator |
| **History** | Closed trade log with symbol filter, bot-lane filter, PnL, and hold time |
| **Training** | Active training progress for LSTM, PPO, and Dreamer with per-symbol timestep bars |
| **Models** | Registry state — champion path, canary path, candidate list with scorecards |
| **PPO Brain** | PPO model diagnostics: obs shape, device, loaded state, bias correction, last actions |
| **HFT Health** | Latency, spread, and execution quality metrics per symbol |
| **Scenarios** | Scenario memory — best/worst scenario clusters, avoid-list, session review |
| **Perpetual** | Continuous learning stats from the trade-memory feedback loop |
| **LR Timeline** | Learning rate and loss curve timeline across training runs |
| **Patterns** | Candlestick pattern log and regime-aware pattern library |
| **Agents** | Agent team status (n8n workflow integration and autonomy sub-agents) |
| **Settings** | Control panel: start/stop bot, emergency stop, unblock, canary promotion/rollback |

---

## Risk Controls

The system has two layers of protection:

**Risk Engine** (`Python/risk_engine.py`) — soft limits:
- `max_daily_loss`: halt when realized PnL crosses this threshold (default $1000)
- `max_daily_trades`: trade count ceiling per day
- `max_lots`: single-order size cap
- `max_drawdown`: portfolio drawdown ceiling

**Supervisor** (`risk.supervisor` in config) — hard pre-trade gate:
- `max_drawdown_pct`: halt entire trading if equity drawdown exceeds this (default 8%)
- `max_spread_bps`: skip order if spread too wide (default 25 bps)
- `min_trade_interval_sec`: per-symbol cooldown between trades (default 45s)
- `max_open_positions`: hard cap on concurrent open positions (default 6)
- `max_positions_per_symbol`: per-symbol position cap (default 3)
- `max_symbol_exposure`: fraction of equity for one symbol (default 35%)
- `max_total_exposure`: total portfolio exposure cap (default 1.2x)

When halted, the bot will still close positions (risk reduction allowed while halted). Use the **Settings** tab or `POST /api/control` with `{"action": "unblock"}` to resume.

---

## Model Pipeline

```
1. TRAIN       training/train_lstm.py  → models/per_symbol/<sym>/
               training/train_drl.py   → models/registry/candidates/<timestamp>/
               training/train_dreamer.py → models/dreamer/<sym>/

2. EVALUATE    Python/model_evaluator.py
               Criteria: Sharpe ≥ threshold, return ≥ threshold, drawdown ≤ cap
               Forward windows: 60d, 90d, 120d (must win ≥ 1/3)

3. STAGE       Passing candidates become canaries in models/registry/active.json
               Canary runs shadow alongside champion, accumulates live stats

4. PROMOTE     registry.canary_policy criteria: min_trades, min_realized_pnl,
               max_drawdown, min_runtime_minutes (per-symbol overrides supported)

5. HOT-SWAP    Server_AGI detects champion change in active.json and reloads
               the model bundle without restarting the process
```

Run a full automated cycle:

```powershell
python tools/champion_cycle.py
```

Run individual training steps:

```powershell
python training/train_lstm.py
python training/train_drl.py
python training/train_dreamer.py
```

Check for feature drift between backtest and live:

```powershell
python tools/backtest_vs_live_drift.py
```


## Evaluation & Plotting

Standalone analysis scripts in `training/` evaluate model behavior and regime
classifier performance on both synthetic and real XAUUSD data:

| Script | Purpose |
|--------|---------|
| `training/eval_harness.py` | Shared module: env factory, regime observation builder, position/reward collectors, regime bar-chart plotter |
| `training/plot_regime_actions.py` | Per-regime action distribution (5-panel bar chart) — synthetic data |
| `training/plot_position_timeseries.py` | ALL vs no_regime position time series + CSV metrics — XAUUSD |
| `training/plot_regime_time_series.py` | Regime detector colour bands overlaid on price chart — XAUUSD |
| `training/run_feature_ablation.py` | Feature group ablation harness — compares ALL, no_volume, no_momentum, no_volatility, no_trend, no_regime |

### Quick start

```bash
# Per-regime action distribution (synthetic)
python training/plot_regime_actions.py

# Position time series + metrics CSV (XAUUSD)
python training/plot_position_timeseries.py

# Regime bands on price (XAUUSD)
python training/plot_regime_time_series.py

# Feature ablation comparison (synthetic)
python training/run_feature_ablation.py --steps 20000 --verbose
```

Outputs land in `logs/`: PNG figures and `position_metrics.csv` / `feature_ablation_results.csv`.

The shared module `training/eval_harness.py` provides:

- `fit_regime_detector(df)` — fit RegimeDetector on OHLCV data
- `build_regime_observations(features, df, detector)` — append regime one-hot + confidence
- `make_eval_env(obs, df)` — DummyVecEnv with action-contingent reward
- `collect_positions(model, obs, df)` — deterministic eval returning positions
- `collect_metrics(model, obs, df)` — eval returning (positions, rewards)
- `plot_regime_action_distribution(positions, regime_indices)` — 5-panel regime bar chart

---

## Troubleshooting

**Bot stays flat after startup**
The code can be healthy while the bot stays flat if no champion model is promoted. Run `tools/champion_cycle.py` to train and promote a champion, then check the Models tab.

**MT5 connection fails**
Verify credentials in `config.yaml`. On non-Windows systems, MT5 is unavailable — the server falls back to dry-run mode automatically.

**Risk engine halted**
Check the halt reason in the Dashboard tab or via `GET /api/emergency_status`. Use the Settings tab "Unblock" button or `POST /api/control {"action": "unblock"}` after resolving the cause.

**Training stuck**
Check `logs/ppo_progress.json` and `logs/lstm_progress.json` for the last update timestamp. If stale, kill the training process and restart `tools/champion_cycle.py`.

**Dashboard not loading**
Confirm `tools/project_status_ui.py` is running and check `http://127.0.0.1:8088/api/health` for component status.

---

For the full production deployment guide, see [PRODUCTION.md](PRODUCTION.md).

For the API reference, see [docs/API.md](docs/API.md).

For the system architecture, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
