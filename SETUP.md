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

- Push experiment/feature branches freely (CI triggers on `experiment/**` and will run compile + tests).
- Create PRs on GitHub when ready (reference docs/FULL_STACK_PROFITABILITY_REVIEW.md for context).
- Pull updates: `git pull origin main`.
- Desktop convenience copies: After any pull or edits, run `python tools/update_desktop_clean_copy.py` (regenerates SupremeChainsaw_Clean with the bucket layout: 01_Launchers, 02_Core_Python/* etc.). Do not commit inside the Desktop copies.
- Never work from C:\windows\system32 or other system paths (they have been cleaned of stray copies).

See also PRODUCTION.md (Upgrading section) and the profitability review in docs/.

## 8. Operational note

The code can be healthy while the live bot still stays flat if the currently promoted champion is weak. Training/promoting updated artifacts is a separate step from code hygiene.
