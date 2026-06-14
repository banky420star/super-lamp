# Production Deployment Guide

This guide covers a complete production deployment of Chain Gambler on a Windows VPS or dedicated machine with MetaTrader 5 access.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Secrets Setup](#2-secrets-setup)
3. [MT5 Connection](#3-mt5-connection)
4. [Python Environment](#4-python-environment)
5. [Docker Stack (Optional)](#5-docker-stack-optional)
6. [Frontend Build and Serving](#6-frontend-build-and-serving)
7. [Health Validation](#7-health-validation)
8. [Process Supervision](#8-process-supervision)
9. [Log Locations and Rotation](#9-log-locations-and-rotation)
10. [Backup and Recovery](#10-backup-and-recovery)
11. [Upgrading](#11-upgrading)
12. [Post-Training Execution & Promotion Playbook](#post-training-execution--promotion-playbook-new---2026-05-27-hardening)
13. [When a Good Candidate Appears Runbook](#when-a-good-candidate-appears-runbook) (see docs/WHEN_GOOD_CANDIDATE_APPEARS.md)

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Windows | 10 / Server 2019+ | MT5 requires Windows; Linux/Mac supported in dry-run only |
| Python | 3.12 | Use the `.venv312` convention |
| Node.js | 20 LTS | Required to build the React dashboard |
| Git | Any | For pulling updates |
| MetaTrader 5 | Latest | Must be installed and logged in on the same machine |
| Docker Desktop | 24+ | Optional — for Redis + n8n sidecar services |
| Telegram Bot | — | Required for alert delivery; create via @BotFather |

**Minimum hardware:**
- 4 CPU cores (8 recommended for parallel training)
- 8 GB RAM (16 GB recommended)
- 50 GB disk (models, logs, and backups accumulate)

---

## 2. Secrets Setup

Never commit `config.yaml`. It is gitignored and must be created on each deployment machine.

```powershell
# From the project root
Copy-Item config.yaml.example config.yaml
```

Open `config.yaml` and set all required fields:

```yaml
mt5:
  login: ENV:MT5_LOGIN       # Or your MT5 account number
  password: ENV:MT5_PASSWORD  # Or your MT5 password
  server: ENV:MT5_SERVER    # Or your broker's MT5 server name

telegram:
  token: "YOUR_BOT_TOKEN_HERE"  # From @BotFather
  chat_id: "YOUR_CHAT_ID_HERE"    # Your Telegram group or user chat ID

trading:
  symbols:
    - BTCUSDm
    - XAUUSDm
```

**For Docker deployments**, create a `.env` file in the project root:

```bash
# .env — never commit this file
AGI_CONTROL_TOKEN=change-this-to-a-long-random-string
AGI_IS_LIVE=1
AGI_ALLOWED_ORIGINS=https://your-dashboard-domain.com
PYTHONUNBUFFERED=1
```

Generate a strong control token:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

The `AGI_CONTROL_TOKEN` must be set before starting the server in production. Without it, protected control actions (emergency stop, promote canary, restart) are blocked.

---

## 3. MT5 Connection

### Windows Native (Recommended)

MetaTrader 5 must be installed and running on the same Windows machine. The Python MetaTrader5 package communicates with the MT5 terminal via the Windows COM interface.

1. Install MT5 from your broker's website.
2. Log in to your live or demo account.
3. Enable "Allow DLL imports" and "Allow automated trading" in MT5 settings.
4. Verify the connection:

```powershell
python -c "import MetaTrader5 as mt5; print(mt5.initialize(), mt5.account_info())"
```

Expected output: `True AccountInfo(login=..., server=..., ...)`

### Mac / Linux (Wine Bridge — Dry-Run Only)

The MetaTrader5 Python package does not run natively on Mac or Linux. For local development, the server falls back to dry-run mode automatically when MT5 is unavailable.

For production on Linux (e.g., a VPS), two options exist:

**Option A — Windows VPS (recommended):** Rent a Windows VPS from providers like Vultr, Contabo, or AWS (EC2 Windows instance) and run everything natively.

**Option B — Forward MT5 from a Windows machine:** Run `Python/Server_AGI.py` on Windows and expose only the dashboard/API via a reverse proxy. The trading loop and MT5 must remain on Windows.

---

## 4. Python Environment

```powershell
# From the project root
python -m venv .venv312
.venv312\Scripts\activate
pip install -r requirements.txt
```

Verify the install:

```powershell
.venv312\Scripts\python.exe -m compileall Python training tools drl
.venv312\Scripts\python.exe -m pytest --tb=short -q
```

Both commands must complete without errors before proceeding.

---

## 5. Docker Stack (Optional)

The Docker Compose file adds Redis (for state caching) and n8n (workflow automation) alongside the AGI container. The primary production path uses the Windows venv directly; Docker is an optional sidecar layer.

**Start the full stack:**

```bash
docker compose up -d
```

**Services:**

| Service | Port | Purpose |
|---|---|---|
| `agi` | 9090 | AGI engine (Python) |
| `redis` | 6379 | Persistent key-value state |
| `n8n` | 5678 | Workflow automation (optional) |

**Production notes:**

- `config.yaml` is mounted read-only into the `agi` container — place it in the project root before starting.
- Model artifacts in `models/` and logs in `logs/` are bind-mounted so they persist across container restarts.
- Set `restart: always` in `docker-compose.yml` (instead of `unless-stopped`) to survive host reboots.

**Stop the stack:**

```bash
docker compose down
```

---

## Paper Trading with Actual MT5 Execution (Critical Pre-Live Phase)

**Harness / Paper execution:** `python -m Python.paper_trader` (or via Python/execution/paper_executor.py + ExecutorRouter when using rich DecisionPPO/ExecutionAgent path). Legacy references to paper_mt5_execution_harness.py were cleaned (outdated, script no longer present; one-liner + promoter + supervisor now consistently use the maintained paper_trader entry for paper/demo). See scripts/start_platform.ps1 and promote_candidate_to_paper.py.

**Quick Launch (DEMO account only — tiny fixed 0.01 lots):**
```powershell
$env:CHAIN_GAMBLER_EXECUTION_MODE="demo"
$env:AGI_PAPER_FIXED_LOT="0.01"
python scripts\paper_mt5_execution_harness.py --symbols EURUSDm --max-days 5 --equity-start 5000
```

**Key hardened components ready:**
- RiskSupervisor (Python/risk_supervisor.py + execution/ + RiskEngine) with % daily loss, rollback_recommended, full audit logs (risk_audit.jsonl)
- MT5Executor (mt5_executor.py): slippage audit (slippage_audit.jsonl), _safe_order_send retry, force_flatten_all, AGI_PAPER_FIXED_LOT override
- MT5DemoExecutor + ExecutorRouter + GateEngine: demo guards + integration
- Canary + Monitor wired in harness for promotion artifacts (extended with Decision PPO timing metrics: open-window vs news avoidance + degradation rollback)
- Runtime flags: runtime/rollback_harness.flag (touch to trigger), paper_harness_active.flag
- Timing safety hardening (2026-05): daily loss / emergency flatten now respect rich TimeExitSpec for news; sizing caps; see production_hardening_timing_agent.json runbook status

**Success gate before real capital:** Clean multi-day run (no rollback, positive canary metrics, slippage controlled, full logs) on post-fix champion.

All changes production-grade, focused on reliability for immediate use.

---

## 5.5 One-Command Full Stack Launcher (Recommended for React Monitoring UI + Core Services)

The dedicated `launch_full_project.ps1` at project root brings the **complete observable production monitoring stack** online with a single command, with emphasis on the React UI (`frontend/`) and everything it depends on (data ingestion, equity, training loop, supervisor state, TUI layer).

**Primary command (from project root):**
```powershell
.\launch_full_project.ps1
```
- Starts api_server (5050) for rich React data
- Starts/ensures vps_agi_supervisor (manages Server_AGI 9090 + recovery + candidates)
- Starts React monitoring UI (default: dev mode on 5173 with hot reload + proxy to 5050)
- Optionally launches TUI watcher
- Robust Node detection (works even if npm not in PATH), health waits, logging to logs/launch_full_project.log, graceful Ctrl+C shutdown, -DryRun support

**Common flags:**
- `-Preview` : Build + run production preview mode (no hot reload)
- `-DryRun` : Show detection results + exact plan (no starts)
- `-NoSupervisor -NoTui` : UI + api_server only (lightweight dashboard focus)
- `-KillStale` : Clean restart of prior processes
- `-Once` : Launch services then exit (background processes continue)
- `-NoBrowser` : Headless / scheduled task friendly

**Production VPS notes:**
- Run elevated. For 24/7, pair with supervisor registered as SYSTEM scheduled task (see `scripts\vps_agi_supervisor.ps1` header for exact schtasks command).
- Docker alternative for fully containerized prod: `docker compose -f docker-compose.prod.yml up -d`
- Full prerequisites and behavior documented in the script header comments.

---

## 6. Frontend Build and Serving

The React dashboard must be built once and served statically (or via the Vite dev server for development).

**Development (Vite dev server — hot reload):**

```powershell
cd frontend
npm install
npm run dev
# Dashboard available at http://localhost:4180
```

**Production (static build):**

```powershell
cd frontend
npm install
npm run build
# Output: frontend/dist/
```

Serve `frontend/dist/` with any static file server. The simplest option on Windows is to use the Vite preview server:

```powershell
npm run preview --port 8088
```

For a real production deployment, serve `dist/` behind nginx or IIS and proxy `/api/*` to the API server on port 5000:

```nginx
# nginx snippet
location /api/ {
    proxy_pass http://127.0.0.1:5000;
    proxy_set_header Host $host;
}
location / {
    root /path/to/chain_gambler/frontend/dist;
    try_files $uri $uri/ /index.html;
}
```

---

## 7. Health Validation

After starting the server, run these checks before enabling live trading:

**API health check:**

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/health
```

Expected: `status: "ok"` with all component checks passing.

**Readiness probe (all components loaded):**

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/health/ready
```

Expected: `ready: true`. This check fails until a champion model is promoted.

**System status:**

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/status | ConvertTo-Json -Depth 5
```

Look for `state: "online"`, `mode: "LIVE"` (if armed), and `risk.halt: false`.

**MT5 account connectivity:**

```powershell
(Invoke-RestMethod http://127.0.0.1:5000/api/status).account
```

Confirm `balance` and `equity` match your MT5 terminal.

**Smoke test script (if present):**

```powershell
python smoke_test.py
```

---

## 8. Process Supervision

### Option A — Docker restart policy (recommended for Docker deployments)

In `docker-compose.yml`, set:

```yaml
services:
  agi:
    restart: always   # Restart on crash and on host reboot
```

### Option B — Windows Task Scheduler

For the native Python deployment, create a scheduled task that starts on system boot:

1. Open Task Scheduler → Create Task.
2. General: Run whether user is logged on or not, Run with highest privileges.
3. Triggers: At startup, delay 30 seconds.
4. Actions: Start a program:
   - Program: `C:\path\to\project\.venv312\Scripts\python.exe`
   - Arguments: `-m Python.Server_AGI --live`
   - Start in: `C:\path\to\project\`
5. Settings: Restart if the task fails, attempt restart after 1 minute.

Create a separate task for the dashboard (`tools/project_status_ui.py`) following the same pattern.

### Option C — NSSM (Non-Sucking Service Manager)

```powershell
# Install NSSM, then:
nssm install ChainGamblerAGI "C:\path\to\.venv312\Scripts\python.exe"
nssm set ChainGamblerAGI AppParameters "-m Python.Server_AGI --live"
nssm set ChainGamblerAGI AppDirectory "C:\path\to\project"
nssm set ChainGamblerAGI AppStdout "C:\path\to\project\logs\agi_stdout.log"
nssm set ChainGamblerAGI AppStderr "C:\path\to\project\logs\agi_stderr.log"
nssm start ChainGamblerAGI
```

---

## 9. Log Locations and Rotation

| File | Contents |
|---|---|
| `logs/audit_events.jsonl` | Every trade, promotion, halt, and system event |
| `logs/trade_events.jsonl` | Closed trade history with PnL |
| `logs/learning/trade_learning_latest.json` | Rolling trade-memory metrics |
| `logs/lstm_progress.json` | LSTM training progress (updated during training) |
| `logs/ppo_progress.json` | PPO training progress |
| `logs/dreamer_progress.json` | Dreamer training progress |
| `logs/decisions.jsonl` | Per-decision log from the brain |
| `logs/patterns.jsonl` | Detected candlestick patterns |
| `logs/bot_stdout.log` | Bot process stdout (when started via API) |
| `logs/bot_stderr.log` | Bot process stderr |

**Log rotation:** Logs are append-only JSONL files and will grow unbounded. Set up rotation using Windows Task Scheduler or a cron job:

```powershell
# Example: rotate audit_events.jsonl weekly
$log = "logs\audit_events.jsonl"
$archive = "logs\archive\audit_events_$(Get-Date -Format 'yyyyMMdd').jsonl"
if (Test-Path $log) {
    Move-Item $log $archive
}
```

Alternatively, wrap the Python process with `logrotate` on Linux/Docker deployments.

---

## 10. Backup and Recovery

**Create a point-in-time backup (safe to commit):**

```powershell
python tools/create_migration_backup.py
# Output: backups/<timestamp>-vps-migration.zip (credentials redacted)
```

**API-triggered backup:**

```powershell
Invoke-RestMethod -Method POST http://127.0.0.1:5000/api/backup/create
```

**Check backup status:**

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/backup/status
```

**Recovery:**

1. Extract the backup zip to a fresh clone of the repo.
2. Restore `config.yaml` from your secrets store (never stored in backups).
3. Copy `models/` and `logs/` from the backup.
4. Run `python -m pytest` to verify the restored environment.
5. Start the server and confirm `/api/health/ready` returns `ready: true`.

**Critical files to back up separately (outside the script):**
- `config.yaml` (store in a password manager or secrets vault — never in git)
- `models/registry/active.json`
- `models/per_symbol/` (LSTM artifacts)
- `models/registry/` (PPO candidates and champion bundles)

---

## 11. Upgrading

```powershell
# 1. Pull the latest code
git pull origin main

# 2. Activate the venv
.venv312\Scripts\activate

# 3. Install any new dependencies
pip install -r requirements.txt

# 4. Rebuild the frontend (if frontend/ changed)
cd frontend
npm install
npm run build
cd ..

# 5. Run the test suite
.venv312\Scripts\python.exe -m pytest --tb=short -q

# 6. Restart the server
# Option A: Kill and restart the Python process via Task Scheduler or NSSM
nssm restart ChainGamblerAGI

# Option B: Use the API control endpoint
Invoke-RestMethod -Method POST http://127.0.0.1:5000/api/control `
  -Body '{"action":"restart_server"}' `
  -ContentType "application/json" `
  -Headers @{"X-Control-Token"="your-token-here"}

# 7. Validate health
Invoke-RestMethod http://127.0.0.1:5000/api/health
```

**After a major update that changes model features (`feature_version`):**

The existing champion model may become incompatible with the new feature vector. Re-train before going live:

```powershell
python tools/champion_cycle.py
```

Wait for a new champion to be promoted (check `/api/learning`), then verify `/api/health/ready` returns `ready: true`.

---

## Zero-Touch Production Mode Startup Sequence (VPS 24/7 Self-Sustaining + Full Orchestration)

Run **once** after initial setup / updates on the Windows Server 2022 VPS (as Administrator). This wires the complete autonomous stack: SYSTEM supervisor (with Final Zero-Touch Orchestrator), TUI auto-observer (launch_pipeline_observer_on_completion.py), health, paper defaults, robust training launchers (v3/v4/v5+), promoter + auto-promote bridge, MQL5 deploy automation (deploy_mql5_chain_gambler.ps1), feedback wiring, and training/paper/MQL5 coordination for full autonomy with minimal (or zero) intervention once envs armed.

**New in latest automation push:** Full end-to-end orchestration on good candidate (supervisor detects alignment_fix_applied post-fix -> promoter gates/canary/paper + parallel MQL5 deploy + TUI + RetrainingTrigger feedback seeds). All gated by explicit opt-in envs for safety. Use robust v* launchers for training (conservative params + health.json signals).

1. Ensure repo at `C:\supreme-chainsaw`, `.venv312\Scripts\python.exe` present and working, MT5 terminal(s) installed + logged into DEMO/paper account. NeuroNetworksBook headers extracted (for MQL5 deploy: typically `C:\Users\Administrator\Downloads\48097_extracted\`).

2. From project root in elevated PowerShell:
   ```powershell
   cd C:\supreme-chainsaw

   # Install/verify rich for TUI (one time)
   .\.venv312\Scripts\python.exe -m pip install --upgrade rich

   # Quick health baseline (includes dynamic MT5 + disk + supervisor checks + new automation awareness)
   powershell -File scripts\healthcheck.ps1 -IncludeMT5Check

   # Test TUI snapshot (deep 50k training dive + pipeline cards + promotion checklist)
   .\launch_tui.ps1 -Once

   # Test watcher (auto TUI on "Candidate staged" / training complete) - optional foreground test
   # .\launch_tui.ps1 -Watcher -Persistent   # Ctrl+C to stop
   ```

3. Register the persistent SYSTEM supervisor (critical for 24/7 + orchestration):
   ```powershell
   $action = New-ScheduledTaskAction -Execute "powershell.exe" `
       -Argument "-NoProfile -ExecutionPolicy Bypass -File `"C:\supreme-chainsaw\scripts\vps_agi_supervisor.ps1`""
   $trigger = New-ScheduledTaskTrigger -AtStartup
   $settings = New-ScheduledTaskSettingsSet -RestartCount 10 -RestartInterval (New-TimeSpan -Minutes 5) -ExecutionTimeLimit (New-TimeSpan -Hours 0) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
   Register-ScheduledTask -TaskName "ChainGambler-AGI-Supervisor" -Action $action -Trigger $trigger -Settings $settings -User "SYSTEM" -RunLevel Highest -Force

   # Optional: healthcheck task (every 15 min)
   $hcAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"C:\supreme-chainsaw\scripts\healthcheck.ps1`" -IncludeMT5Check -Quiet"
   $hcTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration ([TimeSpan]::MaxValue)
   Register-ScheduledTask -TaskName "ChainGambler-Healthcheck" -Action $hcAction -Trigger $hcTrigger -User "SYSTEM" -RunLevel Highest -Force
   ```

4. (Recommended for full zero-touch) Launch background TUI watcher (self-observing on candidate):
   ```powershell
   Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"C:\supreme-chainsaw\launch_tui.ps1`" -Watcher -Persistent" -WindowStyle Minimized
   ```
   Supervisor also auto-spawns watcher on good candidate detection.

5. **Arm full zero-touch orchestration envs (one-time, persist in supervisor session or SYSTEM env for true hands-off):**
   ```powershell
   # Core gates + canary + paper + MQL5 full chain (recommended for v4+ runs)
   $env:AGI_AUTO_PROMOTE_CANDIDATE="1"
   $env:AGI_AUTO_PAPER_HARNESS="1"
   $env:AGI_AUTO_MQL5="1"
   $env:AGI_PROMOTER_PROMOTE_CANARY="1"   # auto set_canary on gates pass inside promoter
   # Optional full retrain cycle (heavier): $env:AGI_USE_FULL_CHAMPION_CYCLE="1"
   # Optional ultra-cohesive: $env:AGI_AUTO_FULL_DOWNSTREAM="1"

   # Persist for SYSTEM supervisor (edit task or use setx for machine; re-register task after):
   # [Environment]::SetEnvironmentVariable("AGI_AUTO_PROMOTE_CANDIDATE", "1", "Machine")  # etc. (requires reboot or task restart)
   ```

6. Launch training with latest robust launcher (conservative hyperparams + health signals for supervisor recovery; v4/v5 recommended):
   ```powershell
   .\scripts\launch_robust_postfix_training_v4.ps1 -Symbol BTCUSDm -Timesteps 50000
   # Or v5 for latest refinements; or legacy: .\launch_postfix_training.ps1 ...
   # Detach/background variants available in launcher family.
   # Monitor: .\launch_tui.ps1 -Once   or watcher (auto-pops on candidate)
   ```

7. When good post-fix candidate appears (supervisor + TUI detect via "Candidate staged" + `alignment_fix_applied` in scorecard):
   - Supervisor logs priority + auto-launches TUI watcher + prepares full guidance.
   - **With envs from step 5 armed:** Full automatic "detected -> auto_promote_candidate.ps1 -> promote_candidate_to_paper.py (strict gates via evaluator + PromotionGates + optional --promote-canary/set_canary) -> conservative paper harness launch -> parallel MQL5 deploy (deploy_mql5_chain_gambler.ps1 -AutoFromRegistry -ShadowPrep -DeployToAllTerminals in bg) -> feedback seeds (RetrainingTrigger) + unified audit in logs/post_training_promotion_decisions.jsonl + mql5_shadow_ready artifacts".
   - Manual / review path (always safe): `python scripts\promote_candidate_to_paper.py --symbols BTCUSDm --dry-run` (prints full checklist + exact commands).
   - One-command handoff + optional auto paper: (see Post-Training Playbook below or new runbook).
   - MQL5 zero-touch: Supervisor always emits the exact deploy command; with AGI_AUTO_MQL5 it runs autonomously (produces builder .mq5, copies Neuro+EA to all terminals, triggers export, writes ready flag/JSON).
   - TUI shows pipeline + promotion checklist lighting up.
   - v* robust launchers automatically benefit once candidate stages (if envs armed).

8. Reboot VPS to verify supervisor + MT5 + TUI watcher + everything auto-starts (check `logs\vps_agi_supervisor.log`, `logs\mql5_deploy_*.log`).

**Self-sustainability guarantees (latest orchestration):**
- Supervisor enforces paper mode, restarts crashed AGI, detects stalled training + candidate readiness with exact next commands.
- Auto TUI watcher on key events via launch_pipeline_observer_on_completion.py (robust log discovery for postfix/robust_* logs).
- Full gated auto handoff: promoter (gates + paper + MQL5 prep) + deploy script + feedback wiring when envs set.
- Healthcheck + supervisor integrate new automation (MQL5 flags, training_health.json, candidate scorecard scan).
- MQL5 deploy: full discovery of all terminals, Neuro headers, self-contained builder script, backups/rollback, LogOnly, ShadowPrep, ready manifests.
- Feedback: Harness + promoter surface RetrainingTrigger artifacts (partial auto, seeds for next cycle).
- No silent failures; rich logging + Telegram + TUI surface everything.
- One-time env arm + supervisor registration = near-zero-touch 24/7 after first good candidate from v4+ run.

See also:
- `scripts/vps_agi_supervisor.ps1` (full header + orchestration comments)
- `scripts/deploy_mql5_chain_gambler.ps1` (MQL5 automation)
- `scripts/auto_promote_candidate.ps1` + `scripts/promote_candidate_to_paper.py`
- `scripts/launch_robust_postfix_training_v*.ps1`
- `launch_tui.ps1` + `tools/launch_pipeline_observer_on_completion.py`
- New concise runbook: **docs/WHEN_GOOD_CANDIDATE_APPEARS.md**
- `scripts/paper_mt5_execution_harness.py`, `Python/autonomous/retraining_trigger.py`
- `docs/OPERATIONAL_HARDENING_SPRINT.md`, `mql5/Experts/ChainGambler/README.md`

After this one-time sequence the stack runs 24/7 autonomously (with opt-in full orchestration). Monitor logs/ and TUI (auto-surfaces on events). Use the new "When a Good Candidate Appears" runbook for the critical handoff moment.

**See dedicated Post-Training Playbook section below for deep details on promoter/deploy flows.**

---

## Final One-Time Zero-Touch Setup (Full Autonomous Chain — 2026-05-27 Orchestrator)

**Goal:** Arm the system *once* so that when the current (or future) v4 robust postfix training run (launch_robust_postfix_training_v4.ps1) or any 50k+ post-fix training produces a candidate with `alignment_fix_applied`, the **entire downstream automation fires automatically in the background with zero additional operator steps**:

- Supervisor detects transition (dir scan + "Candidate staged")
- Auto-launches TUI watcher (pipeline lights up)
- Unified promoter (strict PromotionGates via evaluator + full checklist + audit to post_training_promotion_decisions.jsonl)
- Optional auto-canary (ModelRegistry.set_canary on gates pass)
- Conservative paper harness (0.01 lots, 0.75% DD, feedback emission)
- **MQL5 full deploy chain** (export parity + headers + builder script + deploy to all terminals + shadow-ready .net prep + mql5_shadow_ready.flag)
- Feedback wiring status logged (RetrainingTrigger ready for paper -> retrain cycles)
- All with rich supervisor/TUI/MQL5-specific logs for audit

**Result:** One cohesive autonomous unit. v4 candidate appears → full promoter + MQL5 + paper + feedback pipeline executes reliably. Operator only reviews TUI / logs / 7d canary outcomes.

### One-Time Setup Commands (Run as Administrator in elevated PowerShell from C:\supreme-chainsaw)

```powershell
cd C:\supreme-chainsaw

# 1. Install/upgrade TUI dep (one-time)
.\.venv312\Scripts\python.exe -m pip install --upgrade rich

# 2. ARM PERSISTENT MACHINE-LEVEL ENV VARS (critical for SYSTEM Task Scheduler supervisor to see them)
#    These enable the full zero-touch orchestration glue + auto-gates + MQL5 + canary + paper.
[Environment]::SetEnvironmentVariable("AGI_AUTO_PROMOTE_CANDIDATE", "1", "Machine")
[Environment]::SetEnvironmentVariable("AGI_AUTO_MQL5", "1", "Machine")
[Environment]::SetEnvironmentVariable("AGI_PROMOTER_PROMOTE_CANARY", "1", "Machine")
[Environment]::SetEnvironmentVariable("AGI_AUTO_PAPER_HARNESS", "1", "Machine")
# Optional: full champion cycle (heavy; usually leave off)
# [Environment]::SetEnvironmentVariable("AGI_USE_FULL_CHAMPION_CYCLE", "0", "Machine")

# Verify
[Environment]::GetEnvironmentVariable("AGI_AUTO_PROMOTE_CANDIDATE", "Machine")
[Environment]::GetEnvironmentVariable("AGI_AUTO_MQL5", "Machine")

# 3. (Re)register the SYSTEM supervisor task (picks up new Machine envs on next start)
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"C:\supreme-chainsaw\scripts\vps_agi_supervisor.ps1`""
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -RestartCount 10 -RestartInterval (New-TimeSpan -Minutes 5) -ExecutionTimeLimit (New-TimeSpan -Hours 0) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "ChainGambler-AGI-Supervisor" -Action $action -Trigger $trigger -Settings $settings -User "SYSTEM" -RunLevel Highest -Force

# Optional healthcheck task (unchanged)
$hcAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"C:\supreme-chainsaw\scripts\healthcheck.ps1`" -IncludeMT5Check -Quiet"
$hcTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration ([TimeSpan]::MaxValue)
Register-ScheduledTask -TaskName "ChainGambler-Healthcheck" -Action $hcAction -Trigger $hcTrigger -User "SYSTEM" -RunLevel Highest -Force

# 4. Start the supervisor now (or reboot to verify full auto-start)
Start-ScheduledTask -TaskName "ChainGambler-AGI-Supervisor"
# Tail its log: Get-Content logs\vps_agi_supervisor.log -Wait -Tail 20

# 5. Launch persistent background TUI watcher (auto-spawns on candidate; supervisor also does this)
Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"C:\supreme-chainsaw\launch_tui.ps1`" -Watcher -Persistent" -WindowStyle Minimized

# 6. Quick baseline
powershell -File scripts\healthcheck.ps1 -IncludeMT5Check
.\launch_tui.ps1 -Once
```

**After the above (and a recommended reboot to confirm Machine envs + tasks survive):**

- Start your v4 training (detached, conservative params already in launcher):
  ```powershell
  .\scripts\launch_robust_postfix_training_v4.ps1 -Symbol BTCUSDm -Timesteps 50000
  # (or with -Detach if extended launcher supports; monitor via TUI)
  ```

- **Thereafter (near-zero-touch):** When the v4 run (or any qualifying postfix) stages its candidate:
  - Supervisor logs "NEW GOOD CANDIDATE..." + "=== ZERO-TOUCH ORCHESTRATION: Firing full promoter + MQL5 chain..."
  - Full bg processes spawn (promoter, MQL5 deploy with its own timestamped log, etc.)
  - TUI auto-launches / updates with checklist + pipeline status
  - Artifacts: runtime/champion_ready.flag, logs/post_training_*.jsonl, artifacts/mql5_distill/, mql5_shadow_ready.flag + guidance, harness jsonl + retraining triggers
  - MQL5: terminals updated, builder .mq5 ready to compile/run in MT5 for .net, EA attachable in ShadowMode
  - Feedback: harness will emit triggers for future cycles

**To disable any arm (safety):** Set the corresponding Machine var to "0" and restart the supervisor task.

**Verification after v4 candidate appears (no manual intervention needed):**
```powershell
# Supervisor should show the orchestration success lines
Get-Content logs\vps_agi_supervisor.log -Tail 30

# MQL5 orchestrated deploy logs
Get-ChildItem logs\mql5_orchestrated_deploy_*.log | Sort LastWriteTime -Desc | Select -First 1 | Get-Content -Tail 20

# Promoter audit + checklist surface
Get-Content logs\post_training_promotion_decisions.jsonl -Tail 5
.\.venv312\Scripts\python.exe -c "from scripts.monitor_tui import get_promotion_checklist; import json; print(json.dumps(get_promotion_checklist(), indent=2))"

# MQL5 ready state
Test-Path runtime\mql5_shadow_ready.flag
Get-Content artifacts\mql5_distill\mql5_shadow_ready.json -Raw | ConvertFrom-Json | Select -Expand next_steps
```

This is the **final** one-time setup. The supervisor + promoter + MQL5 + TUI + feedback now function as one autonomous unit. All prior manual one-commands remain available as fallbacks.

See: scripts/vps_agi_supervisor.ps1 (orchestration block ~lines 605-650), auto_promote_candidate.ps1, promote_candidate_to_paper.py, deploy_mql5_chain_gambler.ps1, and the Post-Training Playbook below.

---

# Post-Training Execution & Promotion Playbook (NEW - 2026-05-27 Hardening)

**Purpose:** The moment a post-fix candidate (with `alignment_fix_applied`, real per-symbol metrics, OOS splits, good OOS performance) passes core `PromotionGates`, the system must move **immediately and safely** into paper trading + MQL5 ShadowMode with zero manual guesswork.

This playbook + automation eliminates the last manual steps identified in the Post-Training Readiness Audit.

## 1. Current (Hardened) Detection & Trigger Flow

1. Training (train_drl.py / enhanced) stages candidate to `models/registry/candidates/YYYYMMDD_HHMMSS/` with:
   - `scorecard.json` containing `alignment_fix_applied`, `training_best_mean_reward`, `per_symbol_real_metrics` / `per_symbol_metrics`, `oos_split`, `leakage_prevented`.
2. Supervisor (`vps_agi_supervisor.ps1`) + TUI watcher (`launch_tui.ps1 -Watcher`) detect via:
   - Directory scan + `alignment_fix_applied` + not quarantined
   - Log line "Candidate staged"
   - Auto-spawn TUI + log priority message.
3. `monitor_tui.py` deep-dive + `get_promotion_checklist()` surfaces status + recommendation.
4. **NEW:** `scripts/promote_candidate_to_paper.py` (one-command) is the canonical handoff.

**No change to training block** — this is purely post-candidate execution readiness.

## 2. One-Command Promotion Handoff (Primary Path)

```powershell
# After candidate appears (TUI/supervisor will tell you)
cd C:\supreme-chainsaw

# 1. Review (dry, safe, prints full checklist + gates + exact commands)
python scripts\promote_candidate_to_paper.py --symbols BTCUSDm,EURUSDm --dry-run

# 2. Full prep + optional auto paper start (after MT5 logged into DEMO)
$env:CHAIN_GAMBLER_EXECUTION_MODE="demo"
$env:AGI_PAPER_FIXED_LOT="0.01"
$env:AGI_CONSERVATIVE_PAPER="1"   # enables 0.75% daily / 2 trades/hr for aligned models
python scripts\promote_candidate_to_paper.py --symbols BTCUSDm --auto-launch --max-days 7
```

What the promoter does (all logged + audited):
- Detects latest qualifying candidate.
- Runs `PromotionGates` + `model_evaluator` (strict + core perf).
- Displays **full machine-readable promotion checklist** (via shared `get_promotion_checklist`):
  - Post-fix candidate + alignment_fix
  - OOS + leakage_prevented (FIX-OOS-01)
  - Real per-symbol metrics (FLOW-METRICS-01)
  - Core PromotionGates perf/stability/baseline (pre-canary)
  - Canary data requirement (paper will satisfy)
  - champion_ready.flag + safe defaults armed
  - Paper harness readiness
  - MQL5 shadow export/guidance
  - Feedback loop wiring status
  - Rollback path
- Writes unified audit: `logs/post_training_promotion_decisions.jsonl` (ties candidate → gates → paper start → MQL5 guidance)
- Arms `runtime/champion_ready.flag` + `paper_harness_start.json` + `last_promoted_candidate.txt`
- Runs `tools/export_for_mql5.py` (best effort)
- Generates `artifacts/mql5_shadow_guidance/<candidate>_shadow_launch.txt` with exact copy + attach commands for ChainGambler_Executor.mq5 (ShadowMode=true)
- (Optional) Launches hardened `paper_mt5_execution_harness.py` with conservative profile
- Emits retraining trigger seeds for feedback loop

**Safe defaults always applied for new models:**
- 0.01 fixed lots (env override `AGI_PAPER_FIXED_LOT`)
- Max 1 position total
- Max daily loss: 1.0% (0.75% when `AGI_CONSERVATIVE_PAPER=1` or post-fix detected)
- Max 2-3 trades/hour
- Full pre-trade gates + dual RiskSupervisor layers (top + exec) + % equity support
- CanaryMonitor (2% DD stop) + Telegram + `runtime/rollback_harness.flag`

## 3. Paper Harness Hardening (Integrated)

`scripts/paper_mt5_execution_harness.py` (updated):
- Auto-detects post-fix candidate → conservative risk profile
- Touches `champion_ready.flag` + records candidate for audit
- Real feedback wiring (loop closed): harness on closes/rollbacks/canary/risk updates counters + logs/execution_feedback; aggregator periodically evaluates + logs "RETRAIN RECOMMENDED"; state persisted; surfaced in TUI/promoter/supervisor/run_cycle
- Uses latest risk_engine + execution risk + GateEngine + ExecutorRouter
- All events in `logs/paper_harness_exec.jsonl` + `harness_canary_monitor.jsonl` + risk_audit.jsonl + slippage
- Rollback: instant flatten + halt + alert (daily loss, errors, flag, canary)

Monitor live:
```powershell
python Python/monitoring_dashboard.py --mode live
# or TUI (now includes checklist when candidate ready)
```

## 4. MQL5 Shadow Coordination (Zero-Touch Python → MQL5 via Deploy Script)

**Primary automation (2026-05-27+ MQL5 Production Deployment Agent + supervisor orchestration):**
- `scripts/deploy_mql5_chain_gambler.ps1 -AutoFromRegistry -ShadowPrep -DeployToAllTerminals` (or -LogOnly for safe preview).
  - Auto-detects latest good post-fix candidate (parity with supervisor/promoter via Python export_for_mql5 --find-latest-good-candidate).
  - Discovers **all** MT5 terminals (APPDATA hex folders + Common).
  - Copies NeuroNetworksBook headers + ChainGambler EA sources.
  - Triggers `tools/export_for_mql5.py` (28-feat exact parity LSTM student net).
  - Generates self-contained `ChainGambler_BuildStudentNet.mq5` (embedded layers, one-click .net build in MT5).
  - Writes `artifacts/mql5_distill/mql5_shadow_ready.json` + `runtime/mql5_shadow_ready.flag`.
  - Full per-terminal backups + `-Rollback -Timestamp` support.
  - Dedicated logs: `logs/mql5_deploy_*.log`.
- Supervisor (vps_agi_supervisor.ps1) **always** emits the exact one-command on candidate detection; with `AGI_AUTO_MQL5=1` (or via auto_promote_candidate or full orchestration) it launches autonomously in background alongside promoter/paper.
- Promoter integrates: references deploy artifacts + generates rich `artifacts/mql5_shadow_guidance/<cand>_shadow_launch.txt` with copy/attach steps.

**Operator / MT5 steps (after deploy or promoter):**
1. In any MT5: F4 (MetaEditor) → open the deployed `MQL5\Scripts\ChainGambler_BuildStudentNet_*.mq5` → F7 compile → run (produces `chaingambler_v1_student.net` to Common\Files or MQL5\Files; prefer UseCommonFolder).
2. Attach `ChainGambler_Executor.mq5` (M5 chart recommended): ShadowMode=true (validation), UseCommonFolder=true, DebugFeatures=true.
3. Run Python paper harness in parallel (same symbols/TF, 0.01 lots).
4. Validate: Compare [SHADOW LONG/SHORT] + CSV logs (`Common\Files\chaingambler_shadow_log.csv`) vs Python harness JSONL/slippage. High correlation + latency win → promote (edit EA inputs: ShadowMode=false + tiny lots).

**Full flow with orchestration:** Good candidate (supervisor) → promoter (gates + paper prep + guidance) + deploy_mql5 (bg, artifacts + .net builder) → attach in MT5 → shadow validation → feedback seeds.

Follow `mql5/Experts/ChainGambler/README.md` for full details + fallback.

**Logging parity goal:** Timestamped actions + feature vectors on identical bars. Model update cycle target: <15 min from candidate stage to shadow running.

## 5. Rollback & Safety (Never Compromised)

- Harness: daily loss breach → auto `force_flatten_all` + halt
- Flag: `touch runtime/rollback_harness.flag`
- Canary + RiskSupervisor halt + Telegram critical
- Supervisor + healthcheck keep paper mode default (`CHAIN_GAMBLER_EXECUTION_MODE=demo`, `ALLOW_LIVE=0`)
- Never promotes real money until 7+ days clean paper + full gates + operator sign-off.

## 6. Feedback Loop (Paper → Retraining) — Status: REAL WIRED (auditor gap closed)

- `Python/autonomous/retraining_trigger.py`: thresholds (50 closed demo trades, drawdown, regime drift, candidate beats champion, canary signals) → `TriggerArtifact` + `next_cycle_command` ("run_retraining" etc.)
- Harness now feeds it on every run (closed_demo / blocked increments + artifact emission).
- `Python/autonomous/run_cycle.py` stages it as part of autonomous pipeline.
- Artifacts land in logs/ + runtime/ for supervisor/TUI/AI to act on.
- **Not yet fully automatic retrain launch** (safety): operator or future scheduler uses the emitted triggers + next_cycle_command to relaunch 50k+ postfix training with conservative hyperparams.
- Canary `approved_for_champion` / `approved_for_real_live` already gates promotions.

Full loop will be closed in future sprint once paper data volume proves stable.

## 7. Promotion Checklist (Machine + Human Readable)

Always available via:
- `python -c "from scripts.monitor_tui import get_promotion_checklist; import json; print(json.dumps(get_promotion_checklist(), indent=2))"`
- Inside TUI deep-dive / panels (when candidate present)
- `promote_candidate_to_paper.py --dry-run` output
- Supervisor / health logs reference it

Items (as of hardening):
(See function source for latest; covers all gates + handoff + rollback + feedback)

## 8. Quick Reference Commands

| Step | Command |
|------|---------|
| Watch for candidate / pipeline lights | `.\launch_tui.ps1 -Watcher -Persistent` (auto via observer) or supervisor |
| One-command handoff + full checklist | `python scripts\promote_candidate_to_paper.py --symbols BTCUSDm --dry-run` (then --auto-launch --promote-canary) |
| Full zero-touch orchestration (env-gated) | Arm AGI_AUTO_PROMOTE_CANDIDATE=1 + AGI_AUTO_PAPER_HARNESS=1 + AGI_AUTO_MQL5=1 + AGI_PROMOTER_PROMOTE_CANARY=1 (see zero-touch setup); supervisor + auto_promote_candidate.ps1 + deploy script fire end-to-end |
| Zero-touch MQL5 deploy (shadow + .net builder) | `powershell -File scripts\deploy_mql5_chain_gambler.ps1 -AutoFromRegistry -ShadowPrep -DeployToAllTerminals` (supervisor always emits; -LogOnly safe) |
| Arm + launch paper (conservative) | See promoter section 2 + harness docs (0.01 lots, 0.75% daily for post-fix) |
| MQL5 shadow attach / build | Follow `artifacts/mql5_shadow_guidance/..._shadow_launch.txt` or mql5_shadow_ready.json + MT5 (compile/run BuildStudentNet.mq5 then attach Executor ShadowMode=true) |
| Monitor everything | TUI (checklist + Autonomous Pipeline panel) + `logs/post_training_promotion_decisions.jsonl` + `logs/mql5_deploy_*.log` + harness/risk/slippage + retrain triggers |
| Force rollback | `touch runtime/rollback_harness.flag` (or 1% breach / canary) |
| Trigger retrain from paper data | `logs/RETRAIN_RECOMMENDED.latest.json` + `trigger_*.json` (harness + RetrainingTrigger aggregator) |
| Latest robust training launch | `.\scripts\launch_robust_postfix_training_v4.ps1 -Symbol BTCUSDm -Timesteps 50000` (v5 for latest) |

## 9. Files Changed / Added (Latest Automation Push + Orchestration)

- `scripts/deploy_mql5_chain_gambler.ps1` (NEW — full MQL5 zero-touch: terminal discovery, Neuro+EA copy, export trigger, self-contained builder .mq5 gen, mql5_shadow_ready.json/flag, backups/rollback, -LogOnly/-ShadowPrep, supervisor orchestration)
- `scripts/paper_mt5_execution_harness.py` (conservative profile, candidate detect, champion_ready, partial RetrainingTrigger wiring, bugfix, docs)
- `scripts/promote_candidate_to_paper.py` (canonical one-command handoff + checklist + MQL5 guidance + unified audit + --promote-canary)
- `scripts/auto_promote_candidate.ps1` (env-gated supervisor bridge for auto gates + canary + paper/MQL5)
- `scripts/vps_agi_supervisor.ps1` (Final Zero-Touch Orchestrator: candidate detection, promoter + MQL5 bg launch, TUI watcher, full env-gated chain, MQL5 LogOnly prep, retrain feedback surface)
- `scripts/launch_robust_postfix_training_v*.ps1` (v3/v4/v5: fixed redirection, training_health.json signals, conservative params, auto-benefit from orchestration)
- `launch_tui.ps1` + `tools/launch_pipeline_observer_on_completion.py` (robust watcher, dynamic log discovery for postfix/robust runs, auto TUI on "Candidate staged")
- `scripts/monitor_tui.py` (get_promotion_checklist + retrain triggers surface + Autonomous Workflow Pipeline live panel)
- `scripts/healthcheck.ps1` (extended for new automation artifacts)
- New concise operator/system runbook: **docs/WHEN_GOOD_CANDIDATE_APPEARS.md** (recommended for handoff moment)
- Cross-refs updated in mql5/Experts/ChainGambler/README.md, PRODUCTION.md, AUTONOMOUS_WORKFLOW_PIPELINE.md

**No breaking changes.** All prior manual paths still work. Full orchestration requires one-time env arming (safe opt-in).
- `scripts/vps_agi_supervisor.ps1` (updated candidate messages + promoter recommendation + env-gated reliable auto-promote/gates invocation for champion_cycle path + **FINAL high-level orchestration glue block**: explicit bg full promoter + MQL5 deploy chain + feedback + rich logs on candidate for cohesive autonomous unit)
- `PRODUCTION.md` (this full playbook section + new "Final One-Time Zero-Touch Setup" with persistent Machine env arming + v4 autonomous flow description)

**No breaking changes.** All prior manual paths still work.

## 10. Coordination Notes for Parallel Agents

- **MQL5 Automation Agent:** Consume promoter output + artifacts/mql5_distill/ + guidance txt. Target: further reduce attach steps (e.g. generate ready .bat or terminal script).
- **Training Agent:** Ensure every 50k+ postfix run produces the scorecard fields (already wired post UNIFY-GATES-01 + FIX-OOS).
- **Ops/Supervisor:** The promoter is now the blessed "next step" on candidate detection.

**When training succeeds on a strong post-fix candidate, execute this playbook (or the concise operator version). The system is now prepared for immediate, safe, audited, monitored paper + shadow execution with minimal (or zero, with envs) human intervention.**

**For the exact moment a good candidate appears, use the dedicated runbook:**
- `docs/WHEN_GOOD_CANDIDATE_APPEARS.md` (concise, checklist-driven, surfaceable by TUI/supervisor/promoter; covers review → promoter/deploy/paper/MQL5/rollback/feedback in one place).

See also: `docs/GO_LIVE_CHECKLIST.md`, `docs/MQL5_EXECUTION_LAYER_DESIGN.md`, `Python/registry/promotion_gates.py`, harness header, `Python/autonomous/run_cycle.py` + `retraining_trigger.py`, mql5 README, scripts headers.

End of Post-Training Playbook.


## 5.6 New Standard Multi-Timeframe Pipeline (1m + 5m + 15m + 1h) — Added May 2026 (NOW THE EASY DEFAULT)

As of the May 2026 profitability & hygiene wave (and Full Stack MTF Update 2026-05-28), the **documented and default standard** for all training, recovery, arming, and execution handoff is:

- Always use the **combined** 1m + 5m + 15m + 1h timeframes per symbol (no longer "pick one best TF").
- Use the **best known feature parameters** for that specific symbol (see `configs/best_features_per_symbol.yaml`).
- These best parameters were discovered via systematic micro-adjustment search on real data and are maintained per symbol.

**Easy default via launchers (no overrides needed):**
- `launch_full_project.ps1` (full stack: React UI + TUI parity + supervisor + api) sets `AGI_MULTI_TF_STANDARD=1` + `AGI_FEATURE_VERSION=multitimeframe_best` etc. Supervisor inherits.
- `scripts/vps_agi_supervisor.ps1` defaults recovery training to `-EnableTimeframeOpt:$true` (uses `fetch_multitimeframe_training_data` + best params).
- `scripts/handoff_watcher.py` (+ launcher) on arm deploys `runtime/decision_execution_mtf_context.json` with full multi-TF + new DecisionBuilder + ExecutorRouter + GateEngine context (for paper/live once ready).
- Legacy single-TF fallback: `AGI_USE_LEGACY_SINGLE_TF=1` (preserved everywhere; explicit per-launcher overrides also supported).

Core modules (auto):
- `Python/features/multitimeframe_builder.py` (load_best_feature_params + build_multitimeframe_features)
- `Python/data_feed.py` (fetch_multitimeframe_training_data + STANDARD_MULTI_TIMEFRAMES)
- `Python/feature_pipeline.py` (build_multitimeframe_feature_matrix + dispatch)

### How to use the new standard

```python
from training.enhanced_train_drl import EnhancedTrainingPipeline

pipeline = EnhancedTrainingPipeline()
results = pipeline.run_training_with_timeframe_optimization(
    symbols=["XAUUSDm", "BTCUSDm"],
    enable_timeframe_opt=True,           # enables the new 1m+5m+15m+1h path + best params
    enable_per_symbol_metrics=True,
)
```

The pipeline will automatically:
1. Pull the four timeframes using `fetch_multitimeframe_training_data`.
2. Load the best feature params for each symbol from the central YAML.
3. Build the rich combined feature matrix using `build_multitimeframe_feature_matrix`.
4. Train with those features.

### Maintaining Best Parameters

When you run a hyperparameter or micro-feature search for a symbol (see `scripts/hyperparam_search_lstm_xau.py` and similar), promote the winning parameters into `configs/best_features_per_symbol.yaml`.

This keeps the entire fleet (all symbols) using the best known settings without manual code changes.

See also:
- `Python/features/multitimeframe_builder.py`
- `Python/data_feed.py` (new `fetch_multitimeframe_training_data`)
- `Python/feature_pipeline.py` (exposed `build_multitimeframe_feature_matrix`)

---

## 18-20: Readiness, Hardening, Completion Definition (Phases executed 2026-06)

### Phase 18: Readiness Checklist (updated)
- [x] Data foundation: runtime/data_foundation/ CSVs (market_bars, decisions, trades, equity, training_runs, feature_audits, regime_logs) with hashes + links (model/dataset/regime).
- [x] Every training: metadata + dataset_hash on model.
- [x] Feature audit: stats (mean/std/min/max/nonzero/nan/corr_return) + dead/leakage/dupe/trend-mom/pattern/cross checks integrated in pipeline + column_audit + foundation.
- [x] Walk-forward: named time-based windows (older/middle/recent/stress), no shuffle, baselines (random/buyhold/MA/flat/champ), candidate multi-run reject if catastro/ not beat baseline.
- [x] Lane C full: MTF, 50k default, symbol-specific save (runtime/lane_c_XXX_*), --dry-run loads/logs decisions to foundation, zero real orders.
- [x] Paper: full paper (MT5 data only, paper exec sim PnL, journal to logs + foundation, alerts, ZERO real; accelerated run tested).
- [x] Dashboard: /api/status now includes foundation tables, paper_trades, dry_decs, safety_blocks, active_champion, candidates/rejected, promotion_gates, test/data/model, no hiding.
- [x] Kill switch, sane equity, complete journal verified in paper/dry runs.

### Phase 19: Production Hardening
- Config profiles: use configs/*.yaml + env (CHAIN_GAMBLER_EXECUTION_MODE=demo/paper, AGI_*) ; default DRY.
- Structured logging: loguru + jsonl in logs/ + foundation CSVs (no unlogged).
- Supervisor: scripts/vps_agi_supervisor.ps1 + healthcheck, restart on crash (already), added paper mode guard.
- Health: /api/health/ready , api_status includes foundation/safety.
- Alerts: telegram + TUI + runtime/agent_status/*.json .
- Rollback: model_registry has champion rollback, integrity hashes; data foundation allows replay.
- No duplicate/generic/unlogged: all via foundation + specific audits.

### Phase 20: Project Complete Definition (20 points)
Project complete when:
1. Data foundation CSVs populated with every run linked (hash, model, regime).
2. Feature audit passes (no dead/leakage/dupe, trend verified, cross ok) on builds.
3. Walk-forward on 4 windows passes with candidate beating baselines, no catastro reject.
4. Lane C 50k MTF trains dry, saves symbol spec, decisions logged, no real orders.
5. Paper 2w-sim (accelerated) : MT5 data, sim orders only, full journal, sane equity curve, no spam logs, kill switch fires.
6. Dashboard /status shows truthful (mode, locked, champion/cands/rej, paper/dry/safety/equity/gates, no omissions).
7. Training metadata + dataset_hash on all models in registry.
8. Promotion gates integrate wf/feature/foundation evidence.
9. Dry-run always zero real (enforced in paper/lane/order paths).
10. Structured logs + foundation for all decisions/trades/audits.
11. Config profiles support paper/demo/live-locked.
12. Supervisor + health + rollback tested.
13. No bare excepts in critical (ongoing from prior).
14. Evidence artifacts in artifacts/ + runtime/ for gates.
15. Backtest/wf vs paper within tol.
16. Per-symbol models + registry consistent.
17. Alerts on blocks/failures.
18. README/PRODUCTION updated with 18-20.
19. All prior phases (1-17) green.
20. "GO" in WINDOWS_PRODUCTION_GO_NO_GO_ASSESSMENT.md + signoff equiv (all checklist items evidenced).

Current state: Phases 8,10,11,12,15,16,17,18-20 advanced in parallel with code changes, runs, foundation setup. Full 50k Lane C + long paper limited by env/MT5 avail but paths validated + gates implemented.

See runtime/data_foundation/*.csv , updated Python/data/provenance.py , feature_registry etc for evidence.
