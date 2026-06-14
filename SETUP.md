# Setup Guide

This repo is designed to run locally on Windows against MetaTrader 5. Docker files exist, but the active production path in this repo is the Windows/MT5 runtime.

## 1. Python environment

```powershell
python -m venv .venv312
.venv312\Scripts\activate
pip install -r requirements.txt
```

## 2. Configuration

Copy:

```powershell
Copy-Item config.yaml.example config.yaml
```

Then set at minimum:

- `mt5.login`
- `mt5.password`
- `mt5.server`
- `telegram.token`
- `telegram.chat_id`
- `trading.symbols`

Recommended current defaults:

- `training.feature_version: ultimate_150`
- `drl.feature_version: ultimate_150`
- `drl.dreamer.enabled: true`
- `drl.dreamer.train_in_cycle: true`

`config.yaml` is local-only and ignored by git.

## 3. Start the stack

Live server:

```powershell
python -m Python.Server_AGI --live
```

Dashboard:

```powershell
python tools/project_status_ui.py
```

Full cycle:

```powershell
python tools/champion_cycle.py
```

## 4. Useful one-shot helpers

Build trade memory:

```powershell
python training/build_trade_memory.py
```

Run release summary:

```powershell
python tools/release_summary.py
```

Run drift check:

```powershell
python tools/backtest_vs_live_drift.py
```

## 5. Health checks

Tests:

```powershell
.venv312\Scripts\python.exe -m pytest
```

Compile checks:

```powershell
.venv312\Scripts\python.exe -m compileall Python training tools drl
```

Dashboard API:

```powershell
Invoke-RestMethod http://127.0.0.1:8088/api/status
```

## 6. Important directories

- `logs/` runtime, training, and audit logs
- `models/per_symbol/` LSTM artifacts
- `models/registry/candidates/` PPO candidates
- `models/dreamer/` Dreamer artifacts
- `docs/results/` release and evidence outputs

## 7. GitHub + Branch Hygiene (Critical for Safety)

Always start from the canonical checkout of https://github.com/banky420star/super-lamp.git:

```powershell
cd C:\supreme-chainsaw
git fetch origin
git checkout -b feature/your-safe-work  # NEW BRANCH FIRST (per review feedback — never edit main or long-lived experiment branches directly)
git push -u origin feature/your-safe-work
```

**Recommended clean branch strategy (Phases 1-4 project completion):**
- `main`: integration point for releases; treat as protected (no direct edits).
- `experiment/xauusd-regime-baseline`: **trading core only** — keep this as the long-lived home for DRL trading logic, regime experiments, reward functions, training core, feature engineering for models, execution, risk. This is the "super-lamp" AGI trading engine core. **Do not mix platform/UI changes here.**
- `feature/dashboard-mission-control`: platform layer for React/frontend, dashboard UIs (mission control views, status, controls), launchers, top-level monitoring.
- `feature/safety-registry-core`: platform layer for model registry, safety gates, promotion harnesses, champion/canary tools (non-core trading), audit, supervisor wrappers.
- `feature/ci-compile-gates`, `feature/runtime-hygiene`: supporting platform/ops branches for CI rules, compile/test enforcement, runtime/ logs/ models/ artifact hygiene.
- Other `feature/*` or `refactor/*` for safe work (e.g. current `refactor/profitability-tier0-reward-structure`).

- **Separation rule (do not mix):** Trading core (dirs: drl/, training/ (core train scripts), Python/{data_feed.py,feature_pipeline.py,hybrid_brain.py,risk_engine.py,mt5_executor.py,model_registry.py (core),rewards/,...}, drl/trading_env.py etc.) lives exclusively on experiment lineage branches. Platform layer (frontend/, dashboard/, ui_*, alerts/, most tools/ UI scripts, launchers, api_server wrappers) on feature/ branches off main. Crossing the boundary requires explicit PR cross-review.
- Push experiment/feature branches freely (CI triggers on `experiment/**` and will run compile + tests; feature/** too).
- Create PRs on GitHub when ready (reference docs/FULL_STACK_PROFITABILITY_REVIEW.md for context).
- Pull updates: `git pull origin main`.
- Use `git worktree` if safe/needed for parallel branch work without polluting main tree (e.g. `git worktree add ../supreme-wt-dashboard feature/dashboard-mission-control`).
- Desktop convenience copies: After any pull or edits, run `python tools/update_desktop_clean_copy.py` (regenerates SupremeChainsaw_Clean with the bucket layout: 01_Launchers, 02_Core_Python/* etc.). Do not commit inside the Desktop copies.
- Never work from C:\windows\system32 or other system paths (they have been cleaned of stray copies).

See also PRODUCTION.md (Upgrading section) and the profitability review in docs/.

## 8. Operational note

The code can be healthy while the live bot still stays flat if the currently promoted champion is weak. Training/promoting updated artifacts is a separate step from code hygiene.

## 9. UI / Dashboard (profitability-tier0-reward focus)

- Lane B real-time dashboard (status, trades, equity, control for the live lanes):
  Start: `python training/dashboard_backend.py` (or via launch_full_project / DesktopLaunchers)
  URL: http://localhost:5051/
  Port override: `$env:DASHBOARD_PORT=5052 ; python training/dashboard_backend.py`

- The resolver for feature_version now lives in Python/config_utils.resolve_drl_feature_version (central, testable, with clear provenance: env / cfg / default). This was extracted as part of the structural cleanup on the `refactor/profitability-tier0-reward-structure` branch.

- Independence checks for regime heads (actor vs critic) are now in a reusable `_assert_actor_critic_regime_independence` helper in the integration test.
