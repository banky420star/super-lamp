# API Reference

The Chain Gambler API server runs on port 5000 (embedded in the AGI engine) and is proxied by the dashboard on port 8088. All endpoints are read-only except `POST /api/control`.

**Base URL:** `http://127.0.0.1:5000`

**Authentication:** Most endpoints are unauthenticated. `POST /api/control` actions listed under [Protected Actions](#protected-actions) require the `X-Control-Token` header.

**CORS:** Allowed origins are `http://localhost:4180` and `http://127.0.0.1:4180`. Additional origins can be added via `AGI_ALLOWED_ORIGINS` environment variable in production.

> **Port reference:** Port **4180** is the `ui_lab_app/` served from the desktop `SupremeChainsaw_Clean/` tree (the canonical runtime). Port **5173** is the `frontend/` dev server launched by `launch_full_project.ps1` in the `C:\supreme-chainsaw\` source tree. API is **5050** in both. The examples below use `http://localhost:5050` to be portable across both trees.

---

## Table of Contents

- [System Status](#system-status)
- [Health](#health)
- [Trades](#trades)
- [Models and Learning](#models-and-learning)
- [Diagnostics](#diagnostics)
- [Control](#control)
- [Patterns and Regimes](#patterns-and-regimes)
- [Performance](#performance)
- [Calendar and Review](#calendar-and-review)
- [Strategies and Scenarios](#strategies-and-scenarios)
- [Training](#training)
- [Backup](#backup)
- [Ollama Advisor](#ollama-advisor)
- [Real-Time Stream](#real-time-stream)

---

## System Status

### GET /api/status

Full system snapshot. The primary endpoint polled by the dashboard every 2-5 seconds.

**Response shape (abbreviated):**

```json
{
  "state": "online",
  "status": "online",
  "mode": "LIVE",
  "uptime_sec": 3600,
  "server": {
    "running": true,
    "pids": [1234],
    "bot_pid": 5678
  },
  "account": {
    "balance": 10000.00,
    "equity": 10142.50,
    "free_margin": 9800.00,
    "profit": 142.50,
    "open_positions": 2,
    "positions": [
      {
        "ticket": 123456,
        "symbol": "BTCUSDm",
        "type": "BUY",
        "volume": 0.01,
        "open_price": 67500.00,
        "current_price": 67850.00,
        "profit": 35.00,
        "sl": 67000.00,
        "tp": 68500.00,
        "open_time": "2026-05-10T08:30:00+00:00"
      }
    ],
    "realized_today": 107.50,
    "drawdown_pct": 0.014,
    "connected": true
  },
  "risk": {
    "halt": false,
    "halt_reason": "",
    "daily_trades": 14,
    "max_daily_trades": 500,
    "realized_pnl": 107.50,
    "max_daily_loss": 1000,
    "current_dd": 0.014,
    "max_drawdown_pct": 8.0,
    "can_trade": true
  },
  "training": {
    "lstm_running": false,
    "drl_running": true,
    "dreamer_running": false,
    "symbol_lane_rows": [ ... ],
    "pipeline_summary": { ... }
  },
  "registry_summary": {
    "champion": "20260501_120000",
    "canary": "20260510_080000",
    "per_symbol_models": { ... }
  },
  "incidents": [ ... ],
  "timestamp": 1746864000.0
}
```

**Example:**

```bash
curl http://127.0.0.1:5000/api/status
```

---

## Health

### GET /api/health

Component health check for monitoring and load balancers.

**Response shape:**

```json
{
  "status": "ok",
  "pid": 1234,
  "timestamp": "2026-05-10T08:00:00+00:00",
  "uptime_seconds": 3600,
  "checks": {
    "server_running": true,
    "risk_engine": true,
    "brain_initialized": true,
    "model_registry": true,
    "config_loaded": true
  }
}
```

`status` is `"ok"` when `server_running` and `risk_engine` both pass. Any other failed check yields `"degraded"`.

**Example:**

```bash
curl http://127.0.0.1:5000/api/health
```

---

### GET /api/health/ready

Strict readiness probe. Returns HTTP 200 only when all components are loaded and a champion model is promoted.

**Success response:**

```json
{ "ready": true, "timestamp": "2026-05-10T08:00:00+00:00" }
```

**Failure response (HTTP 503):**

```json
{ "ready": false, "reason": "no_champion_model" }
```

Possible `reason` values: `server_not_initialized`, `risk_engine_not_loaded`, `brain_not_loaded`, `no_champion_model`, `registry_error: <detail>`.

**Example:**

```bash
curl -f http://127.0.0.1:5000/api/health/ready && echo "READY"
```

---

### GET /api/emergency_status

Returns the current halt state and reason without the full status payload.

**Response shape:**

```json
{
  "halted": false,
  "reason": "",
  "daily_trades": 14,
  "realized_pnl": 107.50
}
```

When halted, `reason` contains the trigger: `daily_loss_exceeded`, `3_consecutive_errors`, or `manual_or_emergency_stop`.

---

## Trades

### GET /api/trades

Paginated closed trade history. Sourced from MT5 deal history (Windows/live) or the in-memory decision cache (dry-run).

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `limit` | integer | 50 | Number of trades to return |
| `offset` | integer | 0 | Pagination offset |
| `symbol` | string | `""` | Filter by symbol (e.g., `BTCUSDm`) |
| `bot_lane` | string | `""` | Filter by lane: `champion`, `canary`, `history`, `unknown` |

**Response shape:**

```json
{
  "trades": [
    {
      "ticket": 123456,
      "symbol": "BTCUSDm",
      "side": "BUY",
      "volume": 0.01,
      "open_time": "2026-05-10T07:00:00+00:00",
      "close_time": "2026-05-10T08:30:00+00:00",
      "open_price": 67500.00,
      "close_price": 67850.00,
      "profit": 35.00,
      "comment": "AGI|BTCUSDm|CH|...",
      "hold_minutes": 90,
      "bot_lane": "champion",
      "outcome": "win"
    }
  ],
  "total": 142,
  "limit": 50,
  "offset": 0
}
```

**Example:**

```bash
curl "http://127.0.0.1:5000/api/trades?limit=10&symbol=BTCUSDm"
```

---

### GET /api/trades/summary

Aggregate trade statistics across all closed trades.

**Query parameters:** same `symbol` and `bot_lane` filters as `/api/trades`.

**Response shape:**

```json
{
  "overall": {
    "total_trades": 142,
    "wins": 89,
    "losses": 53,
    "win_rate": 0.627,
    "total_pnl": 312.40,
    "avg_profit": 18.50,
    "avg_loss": -12.30,
    "profit_factor": 2.14,
    "avg_hold_minutes": 47.2,
    "max_loss_streak": 4
  },
  "by_symbol": {
    "BTCUSDm": { "trades": 80, "wins": 52, "pnl": 220.00, "win_rate": 0.65 },
    "XAUUSDm": { "trades": 62, "wins": 37, "pnl": 92.40, "win_rate": 0.597 }
  }
}
```

**Example:**

```bash
curl http://127.0.0.1:5000/api/trades/summary
```

---

## Models and Learning

### GET /api/learning

Champion and canary model metadata, candidate list, training schedule, and trade-learning log.

**Response shape:**

```json
{
  "champion": {
    "path": "models/registry/candidates/20260501_120000",
    "version": "20260501_120000",
    "scorecard": { "win_rate": 0.64, "loss": 0.021, "saved_at": "..." }
  },
  "canary": {
    "active": true,
    "path": "models/registry/candidates/20260510_080000",
    "version": "20260510_080000",
    "scorecard": { ... }
  },
  "candidates": [
    {
      "version": "20260510_080000",
      "win_rate": 0.66,
      "loss": 0.018,
      "saved_at": "2026-05-10T08:00:00",
      "type": "ppo"
    }
  ],
  "training_schedule": {
    "enabled": false,
    "interval_sec": 3600,
    "auto_canary": true
  },
  "learning_log": { ... }
}
```

**Example:**

```bash
curl http://127.0.0.1:5000/api/learning
```

---

### GET /api/lanes

Per-symbol trading lane status including current model assignment and last decision.

**Response shape:**

```json
{
  "lanes": [
    {
      "symbol": "BTCUSDm",
      "champion": "20260501_120000",
      "canary": "20260510_080000",
      "has_per_symbol_champion": true,
      "action": "BUY",
      "exposure": 0.45,
      "confidence": 0.91,
      "volatility": "HIGH",
      "can_trade": true,
      "is_canary": false,
      "last_decision_at": 1746864000.0,
      "recent_decisions": 12
    }
  ]
}
```

**Example:**

```bash
curl http://127.0.0.1:5000/api/lanes
```

---

### GET /api/per_symbol_models

Per-symbol champion and canary paths from the registry, including global fallback status.

**Response shape:**

```json
{
  "global_champion": "models/registry/candidates/20260501_120000",
  "global_canary": null,
  "symbols": {
    "BTCUSDm": {
      "champion": "models/registry/candidates/20260501_120000",
      "champion_basename": "20260501_120000",
      "canary": null,
      "uses_global_champion": false,
      "uses_global_canary": true,
      "canary_policy": { "min_trades": 60, "max_drawdown": 0.08 },
      "champion_history_count": 3
    }
  }
}
```

**Example:**

```bash
curl http://127.0.0.1:5000/api/per_symbol_models
```

---

## Diagnostics

### GET /api/ppo_diagnostics

PPO model diagnostics: loaded state, observation shape, device, bias correction, and last decision per symbol.

**Response shape:**

```json
{
  "ppo_loaded": true,
  "obs_shape": [150],
  "action_shape": [1],
  "is_canary": false,
  "device": "cpu",
  "champion_path": "models/registry/candidates/20260501_120000",
  "canary_path": "",
  "model_version": "20260501_120000",
  "last_actions": {
    "BTCUSDm": {
      "action": "BUY",
      "exposure": 0.45,
      "confidence": 0.91,
      "volatility": "HIGH",
      "cached_at": 1746864000.0
    }
  },
  "ppo_biases": {}
}
```

**Example:**

```bash
curl http://127.0.0.1:5000/api/ppo_diagnostics
```

---

### GET /api/lstm_explanations

Per-symbol LSTM decision with top-indicator attribution from the most recent cached decision.

**Response shape:**

```json
{
  "symbols": {
    "BTCUSDm": {
      "regime": "HIGH",
      "confidence": 0.91,
      "top_indicators": [
        { "name": "rsi_14", "weight": 0.18 },
        { "name": "atr_14", "weight": 0.14 }
      ],
      "cached_at": 1746864000.0
    }
  }
}
```

Returns `{"symbols": {}, "message": "No LSTM decisions cached yet."}` before the first decision cycle completes.

**Example:**

```bash
curl http://127.0.0.1:5000/api/lstm_explanations
```

---

### GET /api/regimes

Volatility regime breakdown from the decision cache: BUY/SELL/HOLD counts and average confidence per regime.

**Response shape:**

```json
{
  "regimes": {
    "HIGH": {
      "total_decisions": 42,
      "buy_count": 18,
      "sell_count": 14,
      "hold_count": 10,
      "avg_confidence": 0.872,
      "avg_exposure": 0.38,
      "symbols": ["BTCUSDm", "XAUUSDm"]
    },
    "LOW": { ... }
  }
}
```

**Example:**

```bash
curl http://127.0.0.1:5000/api/regimes
```

---

## Control

### POST /api/control

Accept control commands from the dashboard or automation scripts.

**Request body:**

```json
{ "action": "<action_name>", "symbol": "<optional_symbol>" }
```

**Headers for protected actions:**

```
X-Control-Token: <AGI_CONTROL_TOKEN value>
Content-Type: application/json
```

**Unprotected actions** (no token required):

| Action | Description |
|---|---|
| `reset_peak_equity` | Reset the drawdown peak equity reference |
| `force_ingest` | Trigger an immediate data ingest |
| `rebuild_trade_memory` | Queue a trade memory rebuild |
| `hft_start` | Enable HFT mode (stub — not active in current build) |
| `hft_stop` | Disable HFT mode |

**Protected actions** (require `X-Control-Token`):

| Action | Description |
|---|---|
| `emergency_stop` | Halt all trading immediately and close open positions |
| `clear_emergency_stop` | Clear the halt flag and resume trading |
| `unblock` | Reset risk engine halt + clear executor cooldowns |
| `arm_live` | Enable live order execution |
| `start_bot` | Spawn the Server_AGI process (Windows only) |
| `stop_bot` | Terminate the Server_AGI process |
| `promote_canary` | Promote canary to champion (add `"symbol"` for per-symbol) |
| `rollback_canary` | Roll back canary (add `"symbol"` for per-symbol) |
| `rollback_champion` | Roll back champion to previous version |
| `start_training_cycle` | Start the autonomy training loop |
| `stop_training_cycle` | Stop the autonomy training loop |
| `restart_server` | Restart the server (returns advisory — use system-level restart) |

**Response shape:**

```json
{ "ok": true, "action": "emergency_stop", "message": "Emergency stop activated.", "halted": true }
```

Error response (HTTP 403 for missing/invalid token):

```json
{ "ok": false, "action": "emergency_stop", "error": "control token required" }
```

**Example — emergency stop:**

```bash
curl -X POST http://127.0.0.1:5000/api/control \
  -H "Content-Type: application/json" \
  -H "X-Control-Token: your-token-here" \
  -d '{"action": "emergency_stop"}'
```

**Example — promote canary for one symbol:**

```bash
curl -X POST http://127.0.0.1:5000/api/control \
  -H "Content-Type: application/json" \
  -H "X-Control-Token: your-token-here" \
  -d '{"action": "promote_canary", "symbol": "BTCUSDm"}'
```

---

## Patterns and Regimes

### GET /api/patterns

Pattern library sourced from `logs/patterns.jsonl`, live MT5 candlestick detection, or the incident feed as a fallback.

**Response shape:**

```json
[
  {
    "type": "pattern",
    "pattern": "hammer",
    "symbol": "BTCUSDm",
    "timestamp": "2026-05-10T07:45:00+00:00",
    "open": 67500.00,
    "high": 67900.00,
    "low": 67100.00,
    "close": 67800.00
  }
]
```

Returns up to 50 patterns from the log or 10 from live detection.

**Example:**

```bash
curl http://127.0.0.1:5000/api/patterns
```

---

## Performance

### GET /api/perf

Equity curve, PnL curve, confidence curve, and LSTM loss curve from the risk engine's in-memory history. Falls back to `live_state.json` when the server reference is unavailable.

**Response shape:**

```json
{
  "equity_curve": [10000.0, 10050.0, 10142.5, ...],
  "pnl_curve": [0.0, 50.0, 142.5, ...],
  "confidence_curve": [0.85, 0.91, 0.88, ...],
  "lstm_loss_curve": [],
  "adaptation_history": []
}
```

Arrays contain raw numeric values in chronological order. No timestamps are attached; the frontend plots these as indexed series.

**Example:**

```bash
curl http://127.0.0.1:5000/api/perf
```

---

## Calendar and Review

### GET /api/economic_calendar

Upcoming economic events from the MT5 calendar API.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `days_ahead` | integer | 7 | Number of days ahead to fetch |

**Response shape:**

```json
{
  "events": [
    {
      "time": "2026-05-12T12:30:00+00:00",
      "country": "US",
      "event": "CPI m/m",
      "impact": "high"
    }
  ],
  "count": 18
}
```

Results are cached for 30 minutes.

**Example:**

```bash
curl "http://127.0.0.1:5000/api/economic_calendar?days_ahead=3"
```

---

### GET /api/trade_review

Latest post-trade review with annotations, tags, and per-symbol breakdown. Runs a fresh review if no cached review exists.

**Response shape:**

```json
{
  "total_trades": 142,
  "wins": 89,
  "losses": 53,
  "win_rate": 0.627,
  "total_pnl": 312.40,
  "by_symbol": { ... },
  "top_wins": [ ... ],
  "top_losses": [ ... ]
}
```

**Example:**

```bash
curl http://127.0.0.1:5000/api/trade_review
```

---

### GET /api/trade_review/enriched

Full enriched trade list (up to 50) with decision context and tags.

**Response shape:**

```json
{
  "trades": [
    {
      "ticket": 123456,
      "symbol": "BTCUSDm",
      "profit": 35.00,
      "tags": ["high_confidence", "london_session"],
      "decision_context": { ... }
    }
  ],
  "summary": { ... }
}
```

**Example:**

```bash
curl http://127.0.0.1:5000/api/trade_review/enriched
```

---

### POST /api/trade_review/refresh

Force a fresh 7-day trade review cycle, bypassing the 5-minute cache.

**Response:** Same shape as `GET /api/trade_review`.

**Example:**

```bash
curl -X POST http://127.0.0.1:5000/api/trade_review/refresh
```

---

## Strategies and Scenarios

### GET /api/strategies

Analyze closed trades into strategy buckets (symbol × session × side) ranked by expectancy and Sharpe score.

**Response shape:**

```json
{
  "strategies": [
    {
      "id": "BTCUSDm_london_BUY",
      "symbol": "BTCUSDm",
      "session": "london",
      "side": "BUY",
      "trades": 32,
      "wins": 21,
      "win_rate": 0.656,
      "total_pnl": 180.50,
      "expectancy": 5.64,
      "profit_factor": 2.8,
      "sharpe": 1.24,
      "score": 14.8
    }
  ],
  "patterns": [
    { "type": "symbol", "name": "BTCUSDm", "trades": 80, "pnl": 220.00, "win_rate": 0.65 },
    { "type": "session", "name": "london", "trades": 68, "pnl": 195.00, "win_rate": 0.661 }
  ],
  "meta": { "total_trades": 142, "analysis_window": "30d" }
}
```

**Example:**

```bash
curl http://127.0.0.1:5000/api/strategies
```

---

### GET /api/scenarios

Scenario memory statistics: best and worst scenario clusters, avoid-list, and session review.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `symbol` | string | `""` | Filter session review by symbol |

**Response shape:**

```json
{
  "ok": true,
  "total_scenarios": 48,
  "total_records": 284,
  "best_scenarios": [ { "scenario_key": "...", "avg_pnl_pct": 0.018, "trades": 12 } ],
  "worst_scenarios": [ ... ],
  "should_avoid": ["scenario_key_a", "scenario_key_b"],
  "session_review": { ... }
}
```

**Example:**

```bash
curl "http://127.0.0.1:5000/api/scenarios?symbol=BTCUSDm"
```

---

### POST /api/scenarios/record_outcome

Manually record a trade outcome against a decision ID stored in scenario memory.

**Request body:**

```json
{
  "decision_id": "abc123",
  "exit_price": 67850.00,
  "pnl": 35.00,
  "pnl_pct": 0.0052,
  "hold_minutes": 90,
  "close_reason": "tp_hit",
  "max_drawup": 0.008,
  "max_drawdown": 0.002
}
```

**Response shape:**

```json
{ "ok": true, "decision_id": "abc123", "outcome": "win" }
```

---

## Training

### GET /api/training/metrics

Per-symbol training metrics from the most recent enhanced training run.

**Response shape:**

```json
{
  "symbols": ["BTCUSDm", "XAUUSDm"],
  "average_return": 0.042,
  "max_drawdown": 0.08,
  "best_symbol": "BTCUSDm",
  "worst_symbol": "XAUUSDm",
  "per_symbol_metrics": {
    "BTCUSDm": { "return_pct": 0.061, "max_drawdown_pct": 0.05, "win_rate": 0.64 }
  },
  "timeframe_selections": {
    "BTCUSDm": { "best_timeframe": "M5", "sharpe": 1.42 }
  },
  "training_active": false
}
```

**Example:**

```bash
curl http://127.0.0.1:5000/api/training/metrics
```

---

### GET /api/training/analysis

AI-generated natural-language analysis of what the model is currently learning.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `symbol` | string | `""` | Symbol to analyze; omit for global insights |

**Response shape:**

```json
{
  "ok": true,
  "description": "BTCUSDm PPO is currently in exploitation phase...",
  "trajectory": { "direction": "improving", "loss_trend": -0.003 },
  "insights": ["High confidence on BUY signals during London session"]
}
```

**Example:**

```bash
curl "http://127.0.0.1:5000/api/training/analysis?symbol=BTCUSDm"
```

---

### POST /api/training/enhanced

Start an enhanced DRL training run with multi-timeframe optimization.

**Request body:**

```json
{
  "symbols": ["BTCUSDm", "XAUUSDm"],
  "timeframe_opt": true,
  "per_symbol_metrics": true
}
```

All fields are optional. Defaults to configured symbols with both optimizations enabled.

**Response shape:**

```json
{
  "ok": true,
  "message": "Enhanced training started",
  "symbols": ["BTCUSDm", "XAUUSDm"],
  "timeframe_opt": true,
  "per_symbol_metrics": true
}
```

Training runs as a background process. Track progress via `GET /api/training/metrics`.

**Example:**

```bash
curl -X POST http://127.0.0.1:5000/api/training/enhanced \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["BTCUSDm"], "timeframe_opt": true}'
```

---

### POST /api/training/analyze

Analyze the connection between a training run and live trading performance.

**Request body:**

```json
{
  "training_symbol": "BTCUSDm",
  "trading_symbol": "BTCUSDm"
}
```

**Response shape:**

```json
{
  "ok": true,
  "analysis": { "alignment": "good", "notes": "..." },
  "training_metrics": { "epoch": 480, "win_rate": 0.64 },
  "trading_metrics": { "pnl": 35.0, "avg_confidence": 0.89 }
}
```

---

## Backup

### POST /api/backup/create

Trigger an immediate backup of logs and state (models excluded by default).

**Response shape:**

```json
{
  "ok": true,
  "path": "backups/20260510_120000-backup.zip",
  "name": "20260510_120000-backup.zip"
}
```

**Example:**

```bash
curl -X POST http://127.0.0.1:5000/api/backup/create
```

---

### GET /api/backup/status

Backup manager status and list of recent backups.

**Response shape:**

```json
{
  "count": 7,
  "latest": "2026-05-10T08:00:00",
  "latest_size_mb": 12.4,
  "auto_enabled": false,
  "max_backups": 7,
  "backups": [
    { "name": "20260510_080000-backup.zip", "created_at": "2026-05-10T08:00:00", "size_mb": 12.4 }
  ]
}
```

---

## Ollama Advisor

Optional local LLM advisor. All Ollama endpoints return `{"enabled": false}` when Ollama is not installed.

### GET /api/ollama

Ollama advisor status.

**Response shape:**

```json
{ "enabled": true, "available": true, "model": "llama3" }
```

---

### POST /api/ollama/analyze_trade

Analyze a single trade with the local LLM.

**Request body:** Any trade object from `/api/trades`.

**Response shape:**

```json
{
  "analysis": "This was a momentum trade on a breakout...",
  "trade": { ... }
}
```

---

### POST /api/ollama/review_risk

Review the current risk state with the local LLM.

**Request body:** Empty `{}` — risk data is read from the live server.

**Response shape:**

```json
{ "review": "Current drawdown is within acceptable range...", "risk_data": { ... } }
```

---

### POST /api/ollama/daily_summary

Generate a daily trading summary narrative.

**Request body:** Empty `{}` — today's trades and decisions are gathered automatically.

**Response shape:**

```json
{ "summary": "Today's session produced 14 trades...", "data": { ... } }
```

---

## Real-Time Stream

### GET /api/status/stream

Server-Sent Events stream for lightweight real-time updates. Emits one event per connection and closes (the client reconnects to achieve polling-like behavior, as wsgiref is single-threaded).

**Event format:**

```
data: {"timestamp": 1746864000.0, "halt": false, "daily_trades": 14, "realized_pnl": 107.5, "mode": "LIVE", "live_armed": true}

```

**Example:**

```bash
curl -N http://127.0.0.1:5000/api/status/stream
```

For full status polling, use `GET /api/status` at a 2-5 second interval instead.

---

## Telegram Mini App

### GET /mini

Serves the Telegram Mini App HTML page for mobile monitoring.

Returns `text/html`. No parameters. Used by the Telegram bot's inline web app button.

---

## Eval Harness (training/eval_harness.py)

Shared module for model evaluation, regime-aware observation construction,
and plotting.  All scripts in `training/` import from this module.

### Public API

#### `fit_regime_detector(df: pd.DataFrame, verbose: bool = True) -> RegimeDetector`

Fit a Random Forest market regime classifier on historical OHLCV data.

- **df**: OHLCV DataFrame with columns ``open``, ``high``, ``low``, ``close``.
- **verbose**: Print status line when ``True``.
- **Returns**: Fitted ``RegimeDetector`` (calls ``fit_heuristic`` internally).

---

#### `build_regime_observations(features, df, detector, window_size=100, regime_dim=REGIME_DIM) -> np.ndarray`

Append regime one-hot + confidence to each feature window.

- **features**: ``(n_windows, n_feature_dims)`` array from ``make_synthetic_features``.
- **df**: OHLCV DataFrame (length == raw bar count).
- **detector**: Fitted ``RegimeDetector``.
- **window_size**: Bars per observation (default 100).
- **regime_dim**: Regime feature width 6 = onehot(5) + confidence(1).
- **Returns**: ``(n_windows, n_feature_dims + 6)`` observation array.

---

#### `make_eval_env(obs, df=None, window_size=100) -> DummyVecEnv`

Create a ``DummyVecEnv`` for evaluation with action-contingent reward.

- **obs**: Observation array ``(n_windows, obs_dim)``.
- **df**: Optional OHLCV DataFrame for real price rewards. When ``None`` uses
  a synthetic reward ``mean(feature[:5]) * 0.01``.
- **window_size**: Bars per observation (default 100).
- **Returns**: ``DummyVecEnv`` wrapping a single ``gym.Env``.

Reward formula (when *df* is provided)::

    reward = sign(position) * bar_return - 0.0001 * |position_change|

---

#### `collect_positions(model, obs, df, window_size) -> np.ndarray`

Run deterministic evaluation and return the position time series.

- **model**: Trained SB3 model (PPO or RegimeRoutedPPO).
- **obs**: Evaluation observation array.
- **df**: OHLCV DataFrame for reward computation.
- **window_size**: Observation window size.
- **Returns**: ``(n_steps,)`` array with values in {-1, 0, +1}.

---

#### `collect_metrics(model, obs, df, window_size) -> tuple[np.ndarray, np.ndarray]`

Run deterministic evaluation and return both positions and rewards.

- **model**: Trained SB3 model.
- **obs**: Evaluation observation array.
- **df**: OHLCV DataFrame for reward computation.
- **window_size**: Observation window size.
- **Returns**: ``(positions, rewards)`` tuple, each ``(n_steps,)``.

---

#### `plot_regime_action_distribution(positions, regime_indices, regime_labels=None, num_regimes=5, title="Per-Regime Action Distribution", save_path="logs/regime_action_distribution.png", dpi=150, verbose=True) -> plt.Figure`

Plot a 5-panel bar chart showing Short/Flat/Long distribution per regime.

- **positions**: ``(n_steps,)`` array in {-1, 0, +1}.
- **regime_indices**: ``(n_steps,)`` array of regime index (0--4).
- **regime_labels**: 5 label strings; defaults to ``REGIME_LABELS``.
- **num_regimes**: Number of regime classes (default 5).
- **title**: Figure super-title.
- **save_path**: PNG save path (``None`` disables saving).
- **dpi**: Figure DPI (default 150).
- **verbose**: Print summary table to stdout when ``True``.
- **Returns**: matplotlib ``Figure`` (saved to *save_path* if set).

---

#### `REGIME_DIM` (constant)

``REGIME_DIM = NUM_REGIMES + 1 = 6`` — one-hot vector for 5 regime classes
plus a confidence scalar appended to observations.

---

### Quick start

```bash
# Per-regime action distribution (synthetic data)
python training/plot_regime_actions.py

# Position time series + metrics CSV (XAUUSD)
python training/plot_position_timeseries.py

# Regime bands on price chart (XAUUSD)
python training/plot_regime_time_series.py

# Feature ablation comparison (synthetic data)
python training/run_feature_ablation.py --steps 20000 --verbose
```

Outputs (PNGs + CSVs) land in ``logs/``.
