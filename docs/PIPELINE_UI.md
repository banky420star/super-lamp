# Chain Gambler — Mission Control Dashboard (UI Pipeline)

> **URL:** `localhost:5173` (Vite) → proxies to `localhost:5050` (FastAPI)
> **Stack:** React 18 + TypeScript + Vite 4  |  Python FastAPI
> **Theme:** Dark military-grade terminal (scanline overlay, neon cyan accents)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  MT5 Terminal                                                   │
│  ┌──────────┐                                                   │
│  │ candles  │──fetch──→                                        │
│  │ ticks    │                                                  │
│  │ account  │                                                  │
│  └──────────┘                                                   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Python Backend (:5050)                                          │
│  ┌────────────────────┐     ┌────────────────────┐               │
│  │ dashboard_backend.py│     │  api_server.py     │               │
│  │ (FastAPI routes)    │     │  (FastAPI host)    │               │
│  └────────┬───────────┘     └────────┬───────────┘               │
│           └──────────┬───────────────┘                           │
│                      ▼                                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  ~30 GET endpoints  │  4 POST endpoints  │  /ws/status   │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  React Frontend (:5173) — Vite dev server                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  SystemCommandBar (always visible, 11 status pills)      │   │
│  │  MODE | TRANSPORT | LOCKED | API | MT5 | ACCT | CHAMPION │   │
│  │  TELEMETRY | TESTS | FAILURES | BUNDLE                   │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  Navigation: 16 tabs                                      │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  Active Panel (one of 16, rendered by activeTab)         │   │
│  │  Data: WebSocket push + 10s polling                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Vite proxy config:                                              │
│    /api → http://localhost:5050                                  │
│    /ws  → ws://localhost:5050                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Tab Panels (16 total)

| # | Tab | Component | What It Shows | API Endpoint |
|---|-----|-----------|---------------|-------------|
| 1 | **Overview** (System Truth) | `OverviewPanel.tsx` | Mode, Safety, Account status cards | `/api/system_header` |
| 2 | **Trades** | `TradesPanel.tsx` + `EquityChart.tsx` | Trade history table, summary, equity curve | `/api/trades`, `/api/trades/summary`, `/api/equity_curve` |
| 3 | **Model Brains** | `ModelBrainsPanel.tsx` | LSTM/PPO/Rainforest/Dreamer brain state & metrics | `/api/model_brains`, `/api/lstm_explanations`, `/api/ppo_diagnostics` |
| 4 | **Pipeline** | `PipelinePanel.tsx` | Stage cards: Data→Features→Train→Validate→Promote | `/api/pipeline/stages`, `/api/learning` |
| 5 | **Training** | `TrainingPanel.tsx` + `TrainingLaneCard.tsx` | Lane A/B/C status, progress bars, epochs | `/api/training/lanes`, `/api/status` |
| 6 | **Registry** | `RegistryPanel.tsx` | Model IDs, versions, champion, per-symbol tracking | `/api/registry` |
| 7 | **Promotion Gates** | `PromotionGatesPanel.tsx` | Gate checklist: X/Y PASSED, blocking gates | `/api/promotion_gates` |
| 8 | **Demo Canary** | `DemoCanaryPanel.tsx` | Demo account metrics, timeline, win rate | `/api/demo_canary` |
| 9 | **Trade Coroner** | `TradeCoronerPanel.tsx` | Mistake autopsy, clusters, root cause | `/api/trades/coroner` |
| 10 | **Patterns** | `PatternsPanel.tsx` + `PatternLibraryPanel.tsx` | Pattern records, verified, Rainforest detector | `/api/patterns`, `/api/patterns/rainforest`, `/api/patterns/verified` |
| 11 | **Perpetual Improvement** | `PerpetualPanel.tsx` | Learning events, candidate experiments, evolution | `/api/perpetual_improvement` |
| 12 | **Agents** | `AgentsPanel.tsx` + `AgentTeamPanel.tsx` | Agent status, lifecycle, metrics | `/api/agents/status` |
| 13 | **Safety Lock** | `SafetyPanel.tsx` | 🔒 Real money lock, lock reasons, safety gates | `/api/safety` |
| 14 | **Evidence Locker** | `EvidenceLockerPanel.tsx` | Artifacts, log files, model hashes, audit trail | `/api/evidence` |
| 15 | **Settings** | `SettingsPanel.tsx` | Mode toggle, MT5 login, paper reset, controls | `/api/mode`, `/api/mt5_login`, `/api/paper_reset`, `/api/control` |
| 16 | **Legacy Dashboard** | `DashboardPanel.tsx` | Original multi-panel grid layout | (multiple endpoints) |

---

## API Endpoints (Backend on :5050)

### WebSocket (real-time)

| Endpoint | Purpose |
|----------|---------|
| `/ws/status` | Real-time StatusPayload stream (agent & system state) |

### GET Endpoints (polled every 10s)

| Category | Endpoints |
|----------|-----------|
| **Core State** | `/api/status`, `/api/system_header`, `/api/lanes`, `/api/regimes`, `/api/patterns`, `/api/perf` |
| **Training/Models** | `/api/training/lanes`, `/api/model_brains`, `/api/registry`, `/api/pipeline/stages`, `/api/learning` |
| **Safety/Gates** | `/api/safety`, `/api/promotion_gates`, `/api/demo_canary`, `/api/evidence` |
| **Analytics** | `/api/trades`, `/api/trades/summary`, `/api/equity_curve`, `/api/ppo_diagnostics`, `/api/lstm_explanations` |
| **Diagnostics** | `/api/trades/coroner`, `/api/patterns/rainforest`, `/api/patterns/verified`, `/api/perpetual_improvement`, `/api/agents/status`, `/api/economic_calendar` |

### POST Endpoints (user actions)

| Endpoint | Purpose |
|----------|---------|
| `/api/control` | Generic system control actions |
| `/api/mode` | Set trading mode (`paper` / `live`) |
| `/api/mt5_login` | Submit MT5 credentials |
| `/api/paper_reset` | Reset paper trading balance |

---

## TypeScript Data Model (types.ts)

```
SystemHeaderState     →  MODE, TRANSPORT, LOCKED, API, MT5, ACCT, CHAMPION badges
StatusPayload         →  AccountInfo + TrainingState + SafetyState + SystemTruth
TrainingState         →  LSTM/PPO/Dreamer progress, epochs, loss, queue
PipelineStage         →  passed/running/failed/blocked status per stage
ModelBundle           →  model ID, data source, backtest/walk-forward metrics
ModelBrains           →  LSTM/Rainforest/Dreamer/PPO brain state & heuristics
SafetyState           →  real_money_locked, SafetyGate checklist
PromotionGateItem     →  individual gate pass/fail with reason
TradeCoronerState     →  mistake clusters, root cause groups
DemoCanaryState       →  metrics, timeline, win rate, profit factor
PerpetualImprovementState → learning events, candidate experiments
AgentOperationalStatus    → agent lifecycle & metadata
EvidenceArtifact     →  log files, model hashes, audit records
PatternRecord        →  chart pattern detection data
```

---

## Design System

| Token | Value |
|-------|-------|
| **Background** | `#04080f` (base), `#0a1628` (panels) |
| **Text** | `#e8f4ff` (primary), `#7a94b0` (muted) |
| **Cyan (brand)** | `#00f0ff` |
| **Green** | `#00ff88` |
| **Amber** | `#ffd700` |
| **Red** | `#ff3366` |
| **Purple** | `#b967ff` |
| **Fonts** | Inter (body), IBM Plex Mono (code), Orbitron (display) |
| **Radius** | 8px / 14px / 20px |
| **Spacing** | pad: 24px, gap: 16px, nav-h: 60px |
| **Effects** | Animated bg pulse, radial gradients, glowing neon |
| **Critical states** | Pulse animation
