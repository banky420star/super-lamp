# System Architecture

Chain Gambler is a layered autonomous trading stack designed so data acquisition, model training, live execution, and the control surface can each evolve independently. This document describes how those layers connect.

---

## Table of Contents

- [Full System Diagram](#full-system-diagram)
- [Data Layer](#data-layer)
- [Model Stack](#model-stack)
- [Champion/Canary Pipeline](#championcanary-pipeline)
- [Live Trading Loop](#live-trading-loop)
- [Risk Engine and Halt Conditions](#risk-engine-and-halt-conditions)
- [Parallel Training Lanes](#parallel-training-lanes)
- [Rainforest Regime Gating](#rainforest-regime-gating)
- [WebSocket and Real-Time Data Flow](#websocket-and-real-time-data-flow)
- [Control Surface](#control-surface)
- [File and Directory Map](#file-and-directory-map)

---

## Full System Diagram

```
┌─────────────────────────────── CHAIN GAMBLER ───────────────────────────────┐
│                                                                               │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │                      MetaTrader 5 Broker                             │   │
│   │            Candles (copy_rates_from_pos)  │  Order Execution         │   │
│   └───────────────────────────┬──────────────────────────────────────────┘   │
│                               │                                               │
│   ┌───────────────────────────▼──────────────────────────────────────────┐   │
│   │               Data Feed  →  Feature Pipeline                         │   │
│   │   data_feed.py            feature_pipeline.py (ultimate_150 vector)  │   │
│   └───────────────────────────┬──────────────────────────────────────────┘   │
│                               │                                               │
│   ┌───────────────────────────▼──────────────────────────────────────────┐   │
│   │                         HybridBrain                                  │   │
│   │              (blends model signals into final exposure)              │   │
│   └──────┬──────────────┬──────────────────┬────────────────────────────┘   │
│          │              │                  │                                  │
│   ┌──────▼──────┐ ┌────▼────┐  ┌──────────▼──────┐  ┌──────────────────┐   │
│   │    LSTM     │ │   PPO   │  │     Dreamer      │  │  Scenario Memory │   │
│   │ (regime +   │ │(policy  │  │  (optional blend │  │  (trade-memory   │   │
│   │  context)   │ │ agent)  │  │   per symbol)    │  │   expectancy)    │   │
│   │ models/     │ │ models/ │  │  models/dreamer/ │  │  logs/learning/  │   │
│   │ per_symbol/ │ │registry/│  │                  │  │                  │   │
│   └─────────────┘ └─────────┘  └──────────────────┘  └──────────────────┘   │
│                               │                                               │
│   ┌───────────────────────────▼──────────────────────────────────────────┐   │
│   │                  Risk Engine + Supervisor                            │   │
│   │   Soft limits: daily loss, max lots, drawdown cap                    │   │
│   │   Hard gate:   spread cap, position count, cooldown, exposure cap    │   │
│   └───────────────────────────┬──────────────────────────────────────────┘   │
│                               │                                               │
│   ┌───────────────────────────▼──────────────────────────────────────────┐   │
│   │                       MT5 Executor                                   │   │
│   │   mt5_executor.py — symbol-scoped orders, SL/TP, magic numbers       │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│   ─────────────────────── TRAINING PIPELINE ────────────────────────────     │
│                                                                               │
│   train_lstm.py   ─→  models/per_symbol/<sym>/                               │
│   train_drl.py    ─→  models/registry/candidates/<timestamp>/                │
│   train_dreamer.py─→  models/dreamer/<sym>/                                  │
│          │                                                                    │
│          ▼                                                                    │
│   model_evaluator.py  (Sharpe, return, drawdown, forward windows)            │
│          │                                                                    │
│          ▼                                                                    │
│   model_registry.py   ─→  models/registry/active.json                        │
│          │                                                                    │
│      Canary (shadow)  ──promote──▶  Champion (live)                          │
│      live stats gate                hot-swap in Server_AGI                   │
│                                                                               │
│   ─────────────────────── OPERATOR SURFACE ────────────────────────────      │
│                                                                               │
│   React Dashboard (port 8088 / 4180)                                         │
│         │                                                                     │
│         ▼                                                                     │
│   API Server — api_server.py (port 5000, Bottle/wsgiref)                     │
│         │                                                                     │
│         ├── reads: live_state.json, live_incidents.json                       │
│         ├── reads: models/registry/active.json                                │
│         └── reads: logs/*.jsonl, logs/*_progress.json                        │
│                                                                               │
│   Telegram Alerts ←── audit_events.jsonl + trade_events.jsonl                │
│   n8n Workflows   ←── /api/control (authenticated)                           │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Layer

**`Python/data_feed.py`** is the single source of truth for candle data. It calls `mt5.copy_rates_from_pos()` for each configured symbol and caches the result. Both training and live trading read from this module to prevent feature drift between training-time and inference-time data.

**`Python/feature_pipeline.py`** converts raw OHLCV candles into the `ultimate_150` feature vector — 150 engineered features covering:

- Price-action: returns, log returns, candle body ratios
- Momentum: RSI, MACD, Stochastic variants
- Volatility: ATR (multiple windows), Bollinger width, historical vol
- Volume: OBV, volume z-scores
- Regime: rolling regime label, volatility regime flags
- Calendar: hour-of-day, day-of-week, session flags (Asian/London/NY)

The feature version is pinned in config (`training.feature_version` and `drl.feature_version`). Mismatching versions between trained models and the live pipeline causes silent prediction degradation.

---

## Model Stack

### LSTM (Context Models)

Trained per symbol by `training/train_lstm.py`. The LSTM:
- Consumes a sequence of feature vectors to produce a regime state and confidence score
- Outputs `top_indicators` attribution for each decision
- Artifacts saved to `models/per_symbol/<symbol>/`

The LSTM does not place trades directly — it feeds regime context into HybridBrain.

### PPO (Policy Agent)

Trained per symbol by `training/train_drl.py` against `drl/trading_env.py`. The PPO:
- Takes the full `ultimate_150` feature vector as observation
- Outputs a continuous action (exposure target) in `[-1, 1]`
- Is trained with the `v2_risk_adjusted` reward, which penalizes drawdown, spread costs, and churn
- Artifacts staged to `models/registry/candidates/<timestamp>/`

PPO is the primary signal in HybridBrain. Its weight is controlled by `drl.ppo_blend` in config.

### Dreamer (Optional World Model)

Trained per symbol by `training/train_dreamer.py`. Dreamer:
- Learns a world model of market dynamics (latent state transitions)
- Is blended into HybridBrain only when `drl.dreamer.enabled: true`
- Blend weight is controlled by `drl.dreamer.blend`
- Symbols included are controlled by `drl.dreamer.symbols`

Dreamer adds a planning layer on top of PPO but can be disabled with zero code changes.

### HybridBrain

`Python/hybrid_brain.py` is the signal aggregator. It:
1. Receives LSTM regime context
2. Loads the PPO model for the current champion/canary
3. Optionally loads the Dreamer policy
4. Blends outputs into a final exposure target and confidence score
5. Applies the Rainforest regime gate (see below)
6. Passes the decision to the Risk Engine

---

## Champion/Canary Pipeline

The pipeline ensures only statistically validated models reach live trading.

```
train_lstm.py / train_drl.py / train_dreamer.py
        │
        ▼ (writes artifacts)
models/registry/candidates/<timestamp>/
        │
        ▼ model_evaluator.py
        │   Pass criteria (config.evaluation):
        │   ├── max_drawdown ≤ threshold (default 0.25, relaxed)
        │   ├── min_sharpe ≥ threshold (default -0.5, relaxed)
        │   ├── min_return ≥ threshold (default -0.10, relaxed)
        │   └── forward_windows: must win ≥ 1/3 of [60d, 90d, 120d]
        │
        ▼ (pass → stage as canary)
models/registry/active.json  { "canary": "<path>" }
        │
        │  canary runs in shadow mode alongside champion
        │  accumulating live stats via registry.canary_policy
        │
        ▼ canary promotion criteria (per-symbol overrides allowed):
        │   ├── min_trades: 30 (BTCUSDm: 60)
        │   ├── min_realized_pnl: 0.0
        │   ├── max_drawdown: 0.12 (BTCUSDm: 0.08)
        │   └── min_runtime_minutes: 45
        │
        ▼ (promote)
models/registry/active.json  { "champion": "<path>" }
        │
        ▼ Server_AGI hot-swaps the model bundle without restart
```

**Rollback:** If a canary underperforms, `rollback_canary` clears it from `active.json` and the server reverts to the previous champion. `rollback_champion` steps the champion back to the prior version in the history list.

The champion history limit is controlled by `registry.ensemble.history_limit` (default 3).

---

## Live Trading Loop

`Python/Server_AGI.py` runs the main loop:

```
Every AGI_TRADE_INTERVAL_SEC (default 300s):
  1. Poll MT5 candles via data_feed.py
  2. Build feature vector via feature_pipeline.py
  3. Ask HybridBrain for a decision {action, exposure, confidence, reason}
  4. Cache decision for API and Telegram
  5. Pass to Risk Engine → Supervisor pre-trade gate
  6. If approved: send order via mt5_executor.py
  7. Log to trade_events.jsonl and audit_events.jsonl

Every AGI_REVIEW_INTERVAL_SEC (default 120s):
  1. Review open positions for trailing stop updates
  2. Check model registry for champion changes (hot-swap if needed)
  3. Update live_state.json for dashboard

Every AGI_EQUITY_POLL_SEC (default 15s):
  1. Poll MT5 account info (balance, equity, margin, profit)
  2. Update risk engine equity state
  3. Check drawdown limits
```

---

## Risk Engine and Halt Conditions

The risk engine has two tiers:

**Tier 1 — Risk Engine soft limits:**

| Condition | Default | Action |
|---|---|---|
| `realized_pnl_today < -max_daily_loss` | -$1000 | Set `halt = True` |
| 3 consecutive execution errors | — | Set `halt = True` |
| `daily_trades >= max_daily_trades` | 500 | Reject new orders |

**Tier 2 — Supervisor hard gate (checked per trade):**

| Condition | Default | Action |
|---|---|---|
| Drawdown `> max_drawdown_pct` | 8% | Block order (halt) |
| Spread `> max_spread_bps` | 25 bps | Skip order (cooldown) |
| Position count `> max_open_positions` | 6 | Block order |
| Symbol positions `> max_positions_per_symbol` | 3 | Block order |
| Time since last trade `< min_trade_interval_sec` | 45s | Block order |
| Symbol exposure `> max_symbol_exposure` | 35% equity | Block order |
| Total exposure `> max_total_exposure` | 1.2x equity | Block order |

**Risk reduction is allowed while halted.** The engine will close or reduce positions even when `halt = True`.

**Unblocking:** Use `POST /api/control {"action": "unblock"}` or the Settings tab. This resets the halt flag, error counter, and executor cooldowns simultaneously.

---

## Parallel Training Lanes

When `cycle.parallel_symbols: true` is set, `tools/champion_cycle.py` trains up to `cycle.max_parallel_symbols` (default 2) symbols concurrently using Python `multiprocessing`.

Each symbol gets its own isolated training lane:
- Separate LSTM artifact directory (`models/per_symbol/<sym>/`)
- Separate PPO candidate directory (`models/registry/candidates/<timestamp>_<sym>/`)
- Separate per-symbol progress file (`logs/ppo_<sym>_progress.json`)

The API merges per-symbol progress files and reports the most recently active one as the primary PPO progress. Both lanes share the same `config.yaml` and `feature_pipeline.py`.

---

## Rainforest Regime Gating

The Rainforest gate is a volatility-regime filter applied inside HybridBrain before signals reach the executor. It:

1. Reads the current regime from the LSTM context output (`HIGH`, `MEDIUM`, `LOW`)
2. Applies a confidence multiplier based on the regime
3. Dampens or amplifies the PPO exposure signal accordingly

This prevents the PPO from over-trading during low-volatility regimes where its training distribution is sparse. The regime stats are exposed via `GET /api/regimes`.

---

## WebSocket and Real-Time Data Flow

The API server uses Bottle with wsgiref (single-threaded). True WebSocket is not supported. Instead, real-time updates are delivered through two mechanisms:

**SSE stream** (`GET /api/status/stream`): emits one lightweight event per connection containing halt state, PnL, trade count, and mode. The client reconnects immediately to achieve polling.

**Dashboard polling**: the React app polls `GET /api/status` every 2-5 seconds. This is the primary real-time mechanism. The full status payload includes positions, training progress, lane decisions, and registry state.

**Telegram push**: `alerts/telegram_alerts.py` subscribes to the same audit log stream and pushes events (trade opens/closes, halt triggers, daily summaries) to a configured chat.

---

## Control Surface

```
React Dashboard (port 8088)
        │
        │  GET /api/status (every 2-5s)
        │  GET /api/trades, /api/lanes, /api/learning, etc.
        │  POST /api/control (with X-Control-Token)
        ▼
API Server — api_server.py (port 5000)
        │
        ├── embedded in Server_AGI.py (preferred mode)
        │     reads _server_ref for live risk/brain data
        │
        └── standalone mode (python -m Python.api_server)
              reads live_state.json and registry files as fallback
```

**n8n integration:** The n8n container (port 5678) can call `POST /api/control` on a schedule or in response to external triggers (e.g., a webhook from a news feed). Workflows are stored in `n8n-workflow/`.

**CORS security:** The API only allows origins `http://localhost:4180` and `http://127.0.0.1:4180` by default. In production with `AGI_IS_LIVE=1`, additional origins can be added via `AGI_ALLOWED_ORIGINS`.

---

## File and Directory Map

```
chain_gambler-main/
├── Python/
│   ├── Server_AGI.py         Main trading loop and server
│   ├── api_server.py         HTTP/SSE API (Bottle, port 5000)
│   ├── hybrid_brain.py       Signal blending (LSTM + PPO + Dreamer)
│   ├── feature_pipeline.py   ultimate_150 feature construction
│   ├── data_feed.py          MT5 candle acquisition
│   ├── risk_engine.py        Soft risk limits + halt logic
│   ├── model_registry.py     Champion/canary state + promotion
│   ├── model_evaluator.py    Candidate evaluation vs. thresholds
│   ├── mt5_executor.py       MT5 order placement
│   ├── autonomy_loop.py      Scheduled canary evaluation
│   ├── scenario_memory.py    Trade scenario clustering + memory
│   ├── trade_review.py       Post-trade analysis + economic calendar
│   ├── ollama_advisor.py     Optional local LLM advisor
│   └── backup_manager.py     Backup creation and rotation
│
├── training/
│   ├── train_lstm.py         Per-symbol LSTM training
│   ├── train_drl.py          Per-symbol PPO training
│   ├── train_dreamer.py      Per-symbol Dreamer training
│   └── build_trade_memory.py Trade memory extraction
│
├── drl/
│   ├── trading_env.py        Gymnasium trading environment for PPO
│   └── feature_extractor.py  Policy network feature extractor
│
├── tools/
│   ├── champion_cycle.py     Full retrain → evaluate → stage automation
│   ├── project_status_ui.py  Dashboard server (port 8088)
│   ├── backtest_vs_live_drift.py  Feature drift analysis
│   ├── release_summary.py    Release evidence builder
│   └── create_migration_backup.py  VPS migration snapshot
│
├── alerts/
│   └── telegram_alerts.py   Telegram alert transport
│
├── models/
│   ├── per_symbol/           LSTM artifacts per symbol
│   ├── registry/
│   │   ├── active.json       Live champion/canary paths
│   │   └── candidates/       PPO candidate bundles
│   └── dreamer/              Dreamer artifacts per symbol
│
├── logs/                     Runtime, training, and audit logs
├── frontend/                 React dashboard (Vite + TypeScript)
├── docs/                     Architecture, API, and operations docs
├── config.yaml               Machine-local config (gitignored)
├── config.yaml.example       Config template
└── docker-compose.yml        Redis + n8n sidecar stack
```

## Branching Strategy and Trading Core vs Platform Layer Separation

This architecture is deliberately split across two concerns to prevent accidental mixing during development (per project-completion roadmap Phases 1-4):

**Trading Core (isolated to `experiment/xauusd-regime-baseline` lineage only):**
- Pure trading intelligence and execution: `drl/` (trading_env.py, ppo_agent, regime_detector, feature extractors, etc.), core `training/train_*.py` (DRL, LSTM), `Python/` modules directly tied to decisions (`data_feed.py`, `feature_pipeline.py`, `hybrid_brain.py`, `risk_engine.py`, `mt5_executor.py`, reward functions, `model_registry.py` core + evaluator for promotion).
- Changes here affect live PnL, model behavior, risk math — must be validated on experiment branch, never mixed with UI refactors.
- CI: full (incl. training paths if tagged), but Linux CI excludes MT5/broker.

**Platform Layer (safe on `main`-based `feature/` branches e.g. `feature/dashboard-mission-control`, `feature/safety-registry-core`, `feature/ci-compile-gates`, `feature/runtime-hygiene`):**
- User/control surface, observability, orchestration: `frontend/`, `dashboard/`, `ui_*`, `alerts/`, most `tools/` (non-training), launchers (DesktopLaunchers/, *.ps1, *.sh), `Server_AGI.py` (top-level server glue, but trading logic stays core), api_server.py (serving), config loaders, desktop tools, runtime hygiene scripts.
- Dashboard mission control, safety registry wrappers (high-level), CI workflow enforcement, artifact ignoring.
- These can evolve independently; no impact on core trading math when properly separated.

**Enforcement:**
- New work: always `git checkout -b feature/...` (or worktree) first. Reference this doc + SETUP.md#7.
- PR descriptions must state "trading core" or "platform layer only".
- `git status` / pre-commit should flag cross-layer file touches.
- Do not stage/commit runtime artifacts, models, node_modules, pycache, etc. (see Phase 4 .gitignore).

This separation was locked in during Phase 1-2 of the roadmap to keep the super-lamp AGI trading platform maintainable. Core trading evidence stays on experiment; platform polish and CI/safety tooling on feature branches.
```
