"""
Lane B Dashboard Backend — FastAPI server serving React dashboard + live bot data.

Connects to MT5, reads per-symbol Lane B bot status files, and exposes
REST + WebSocket endpoints for the React dashboard.

Usage:
    python training/dashboard_backend.py

    Then open: http://localhost:5051/

Endpoints:
    GET  /api/lane_b/bots               — List all known bots (symbol + status)
    GET  /api/lane_b/{symbol}/status    — Per-symbol bot status
    GET  /api/lane_b/{symbol}/trades    — Trade history for symbol
    GET  /api/lane_b/{symbol}/equity    — Equity curve for symbol
    POST /api/lane_b/{symbol}/control   — Bot control (start/stop)
    GET  /api/lane_b/status             — Combined account-level view (legacy)
    WS   /ws/lane_b                     — Real-time combined status updates
"""
import sys, os, json, time, asyncio, subprocess, logging, glob
from datetime import datetime, timezone, timedelta
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# MT5 is optional — degrades gracefully if not available
try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False
    mt5 = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dashboard")

# ─── Config ───
DASHBOARD_DIR = Path("C:/supreme-chainsaw/dashboard/dist")
RUNTIME_DIR = Path(__file__).resolve().parent.parent / "runtime"
BOT_PID_FILE = RUNTIME_DIR / "lane_b_pid.txt"
PORT = 5051

# Known symbols (populated from status files)
KNOWN_SYMBOLS = ["XAUUSDm", "BTCUSDm"]

# Cache for discover_bots to limit wmic subprocess spawns
_bots_cache: tuple[float, list[dict]] | None = None
_BOTS_CACHE_TTL = 5  # seconds


def discover_bots(force_refresh: bool = False) -> list[dict]:
    """Scan runtime/ for lane_b_*_status.json files and return active bot list.

    Results are cached for _BOTS_CACHE_TTL seconds to avoid spawning wmic
    subprocess on every WebSocket tick."""
    global _bots_cache
    if not force_refresh and _bots_cache is not None:
        ts, cached = _bots_cache
        if time.time() - ts < _BOTS_CACHE_TTL:
            return cached

    bots = []
    pattern = str(RUNTIME_DIR / "lane_b_*_status.json")
    for fpath in glob.glob(pattern):
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        symbol = data.get("symbol") or os.path.basename(fpath).replace("lane_b_", "").replace("_status.json", "")
        # Check if process is still alive (cached via wmic)
        pid = data.get("pid", 0)
        alive = False
        if pid:
            try:
                r = subprocess.run(
                    f'wmic process where ProcessId={pid} get processid',
                    shell=True, capture_output=True, text=True, timeout=5
                )
                alive = str(pid) in r.stdout
            except Exception:
                pass
        bots.append({
            "symbol": symbol,
            "pid": pid,
            "alive": alive,
            "current_position": data.get("current_position", "FLAT"),
            "bar_count": data.get("bar_count", 0),
            "trades_taken": data.get("trades_taken", 0),
            "model_name": data.get("model_name", ""),
            "drawdown_pct": data.get("drawdown_pct", 0),
            "status_file": fpath,
        })

    _bots_cache = (time.time(), bots)
    return bots


# ─── App lifespan ───
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize MT5 on startup, shutdown on exit."""
    mt5_ok = False
    if HAS_MT5:
        try:
            mt5_ok = mt5.initialize()
            log.info(f"MT5 initialized: {mt5_ok}")
        except Exception as e:
            log.warning(f"MT5 init failed: {e}")
    if not mt5_ok:
        log.warning("MT5 unavailable — will return placeholder data")
    yield
    if HAS_MT5:
        try:
            mt5.shutdown()
            log.info("MT5 shut down")
        except Exception:
            pass


app = FastAPI(lifespan=lifespan)

# ─── WebSocket connections ───
ws_clients: set[WebSocket] = set()


# ─── Data helpers ───
def get_mt5_account():
    """Get MT5 account info or placeholder."""
    if not HAS_MT5 or not mt5.account_info():
        return {
            "login": None, "server": None, "balance": 0, "equity": 0,
            "free_margin": 0, "currency": "USD", "trade_mode": "unknown"
        }
    acct = mt5.account_info()
    mode_map = {0: "demo", 1: "contest", 2: "real", 3: "unknown"}
    return {
        "login": acct.login,
        "server": acct.server,
        "balance": round(acct.balance, 2),
        "equity": round(acct.equity, 2),
        "free_margin": round(acct.margin_free, 2),
        "currency": acct.currency,
        "trade_mode": mode_map.get(acct.trade_mode, "unknown"),
    }


def get_mt5_position(symbol: str):
    """Get current position for a given symbol from MT5."""
    if not HAS_MT5:
        return {"active": False, "type": None, "volume": 0, "open_price": 0, "sl": 0, "profit": 0, "ticket": 0}
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return {"active": False, "type": None, "volume": 0, "open_price": 0, "sl": 0, "profit": 0, "ticket": 0}
    p = positions[0]
    return {
        "active": True,
        "type": "BUY" if p.type == 0 else "SELL",
        "volume": p.volume,
        "open_price": p.price_open,
        "sl": p.sl,
        "profit": round(p.profit, 2),
        "ticket": p.ticket,
    }


def get_mt5_trades(symbol: str, limit=50, offset=0):
    """Get trade history from MT5 for a given symbol."""
    if not HAS_MT5:
        return {"trades": [], "total": 0}

    now = datetime.now()
    from_time = now - timedelta(days=90)
    deals = mt5.history_deals_get(from_time, now)

    if deals is None:
        return {"trades": [], "total": 0}

    records = []
    for d in deals:
        if d.symbol != symbol:
            continue
        records.append({
            "ticket": d.ticket,
            "symbol": d.symbol,
            "side": "BUY" if d.type == 0 else "SELL",
            "volume": d.volume,
            "time": datetime.fromtimestamp(d.time).isoformat() if d.time else None,
            "open_price": d.price,
            "close_price": d.price,
            "profit": round(d.profit, 2),
            "comment": d.comment or "",
            "hold_minutes": None,
        })

    records.reverse()
    total = len(records)
    sliced = records[offset:offset + limit]
    return {"trades": sliced, "total": total}


def get_equity_curve(symbol: str, window="all"):
    """Build equity curve from MT5 deal history for a given symbol."""
    if not HAS_MT5:
        return {"points": [], "summary": {"start_equity": 0, "current_equity": 0, "peak_equity": 0, "max_drawdown_pct": 0, "total_trades": 0}}

    now = datetime.now()
    if window == "30d":
        from_time = now - timedelta(days=30)
    elif window == "90d":
        from_time = now - timedelta(days=90)
    else:
        from_time = now - timedelta(days=365)

    deals = mt5.history_deals_get(from_time, now)
    if deals is None:
        return {"points": [], "summary": {"start_equity": 0, "current_equity": 0, "peak_equity": 0, "max_drawdown_pct": 0, "total_trades": 0}}

    acct = mt5.account_info()
    start_balance = acct.balance if acct else 0
    running_pnl = 0.0
    eq_history = []

    for d in deals:
        if d.symbol != symbol:
            continue
        running_pnl += d.profit
        equity = start_balance + running_pnl
        eq_history.append({
            "ts": datetime.fromtimestamp(d.time).isoformat() if d.time else now.isoformat(),
            "equity": round(equity, 2),
            "balance": round(start_balance, 2),
            "drawdown_pct": 0.0,
        })

    peak = 0
    for p in eq_history:
        peak = max(peak, p["equity"])
        p["drawdown_pct"] = round((peak - p["equity"]) / peak * 100, 2) if peak > 0 else 0

    summary = {
        "start_equity": eq_history[0]["equity"] if eq_history else 0,
        "current_equity": eq_history[-1]["equity"] if eq_history else 0,
        "peak_equity": peak,
        "max_drawdown_pct": round(max((p["drawdown_pct"] for p in eq_history), default=0), 2),
        "total_trades": len(eq_history),
    }

    return {"points": eq_history, "summary": summary}


def read_bot_status_file(symbol: str) -> dict | None:
    """Read the per-symbol status file written by the bot."""
    status_file = RUNTIME_DIR / f"lane_b_{symbol}_status.json"
    if not status_file.exists():
        return None
    try:
        with open(status_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def build_status(symbol: str | None = None):
    """Build status response for a specific symbol or combined account view."""
    if symbol:
        return build_symbol_status(symbol)
    return build_combined_status()


def build_symbol_status(symbol: str):
    """Build per-symbol bot status."""
    saved = read_bot_status_file(symbol) or {}
    account = get_mt5_account()
    position = get_mt5_position(symbol)

    trading = {
        "symbol": symbol,
        "dry_run": saved.get("dry_run", True),
        "risk_pct": saved.get("risk_pct", 0.02),
        "max_dd_pct": saved.get("max_dd_pct", 5.0),
        "sl_atr": saved.get("sl_atr", 2.0),
        "interval_sec": saved.get("interval_sec", 60),
        "current_position": saved.get("current_position", "FLAT"),
        "last_action": saved.get("last_action"),
        "bar_count": saved.get("bar_count", 0),
        "trades_taken": saved.get("trades_taken", 0),
        "peak_equity": saved.get("peak_equity", account.get("equity", 0)),
        "start_equity": saved.get("start_equity", account.get("balance", 0)),
        "drawdown_pct": saved.get("drawdown_pct", 0),
    }

    peak = trading["peak_equity"]
    equity = account.get("equity", 0)
    trading["drawdown_pct"] = round((peak - equity) / peak * 100, 2) if peak > 0 else 0

    pid = saved.get("pid", 0)
    alive = False
    if pid:
        try:
            r = subprocess.run(
                f'wmic process where ProcessId={pid} get processid',
                shell=True, capture_output=True, text=True, timeout=5
            )
            alive = str(pid) in r.stdout
        except Exception:
            pass

    model_info = {
        "loaded": alive,
        "path": f"runtime/{saved.get('model_name', '')}",
        "name": saved.get("model_name", "").replace(".zip", ""),
    }

    return {
        "symbol": symbol,
        "running": alive,
        "pid": pid,
        "account": account,
        "position": position,
        "model": model_info,
        "trading": trading,
        "diag": {"action": saved.get("current_position", "FLAT"), "spread_pts": 0},
        "recent_decisions": [],
    }


def build_combined_status():
    """Build legacy combined account-level status.

    Preserves the old response shape for backward compatibility with the React
    frontend. Shows the first active bot's details or aggregated account view."""
    account = get_mt5_account()
    bots = discover_bots()
    running_bots = [b for b in bots if b["alive"]]

    # Pick the first active bot for the primary position/model display
    primary = running_bots[0] if running_bots else (bots[0] if bots else None)
    primary_symbol = primary["symbol"] if primary else KNOWN_SYMBOLS[0]

    position = get_mt5_position(primary_symbol) if primary else \
        {"active": False, "type": None, "volume": 0, "open_price": 0, "sl": 0, "profit": 0, "ticket": 0}

    trading = {
        "dry_run": False,
        "risk_pct": 0.02,
        "max_dd_pct": 5.0,
        "sl_atr": 2.0,
        "interval_sec": 60,
        "current_position": position["type"] if position["active"] else "FLAT",
        "last_action": None,
        "bar_count": 0,
        "trades_taken": 0,
        "peak_equity": account.get("equity", 0),
        "start_equity": account.get("balance", 0),
        "drawdown_pct": 0.0,
    }

    # Blend with saved status from the primary bot if available
    if primary_symbol:
        saved = read_bot_status_file(primary_symbol)
        if saved:
            trading.update({k: v for k, v in saved.items() if k in trading})

    peak = trading["peak_equity"]
    equity = account.get("equity", 0)
    trading["drawdown_pct"] = round((peak - equity) / peak * 100, 2) if peak > 0 else 0

    model_info = {
        "loaded": len(running_bots) > 0,
        "path": f"runtime/{primary['model_name']}" if primary else "",
        "name": primary["model_name"].replace(".zip", "") if primary else "",
        "n_features": 7,
        "window_size": 64,
        "action_space": "Discrete(3)",
    }

    tg_enabled = bool(os.environ.get("TG_BOT_TOKEN") and os.environ.get("TG_CHAT_ID"))

    return {
        "running": len(running_bots) > 0,
        "pid": running_bots[0]["pid"] if running_bots else None,
        "start_time": None,
        "uptime_seconds": None,
        "account": account,
        "position": position,
        "model": model_info,
        "trading": trading,
        "telegram": {"enabled": tg_enabled, "available": tg_enabled},
        "diag": {"action": trading["current_position"], "spread_pts": 0},
        "recent_decisions": [],
    }


# ─── REST endpoints ───

_start_time = time.time()


@app.get("/health")
async def health_check():
    """Lightweight health check — no MT5 calls, no subprocess."""
    bots = discover_bots()
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "mt5_available": HAS_MT5,
        "active_bots": sum(1 for b in bots if b["alive"]),
        "total_bots": len(bots),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/lane_b/bots")
async def api_list_bots():
    """List all known Lane B bots."""
    return {"bots": discover_bots()}


@app.get("/api/lane_b/{symbol}/status")
async def api_symbol_status(symbol: str):
    """Get per-symbol bot status."""
    return build_symbol_status(symbol)


@app.get("/api/lane_b/{symbol}/trades")
async def api_symbol_trades(symbol: str, limit: int = 50, offset: int = 0):
    """Get trade history for a specific symbol."""
    return get_mt5_trades(symbol, limit, offset)


@app.get("/api/lane_b/{symbol}/equity")
async def api_symbol_equity(symbol: str, window: str = "all"):
    """Get equity curve for a specific symbol."""
    return get_equity_curve(symbol, window)


class ControlRequest(BaseModel):
    action: str


@app.post("/api/lane_b/{symbol}/control")
async def api_symbol_control(symbol: str, req: ControlRequest):
    """Start/stop a Lane B bot for a specific symbol."""
    if req.action == "start":
        try:
            model_path = f"runtime/champion_lane_b_model.zip" if symbol == "BTCUSDm" else f"runtime/lane_b_seed_456_model.zip"
            p = subprocess.Popen(
                [sys.executable, "-u", "training/live_trade_lane_b.py",
                 "--symbol", symbol,
                 "--model", model_path,
                 "--risk", "0.02",
                 "--interval", "60",
                 "--sl-atr", "4.0"],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                stdout=open(f"runtime/lane_b_{symbol}_output.log", "w"),
                stderr=subprocess.STDOUT,
            )
            BOT_PID_FILE.write_text(str(p.pid))
            return {"success": True, "message": f"Bot started PID {p.pid}", "symbol": symbol, "sl_atr": 4.0, "risk": 0.02}
        except Exception as e:
            return {"success": False, "message": str(e)}
    elif req.action == "stop":
        # Find bot by symbol and kill it
        bots = discover_bots()
        target = next((b for b in bots if b["symbol"] == symbol and b["alive"]), None)
        if target and target["pid"]:
            try:
                subprocess.run(f"taskkill /F /PID {target['pid']}", shell=True, check=True, timeout=10)
                return {"success": True, "message": f"Bot PID {target['pid']} stopped", "symbol": symbol}
            except Exception as e:
                return {"success": False, "message": str(e)}
        return {"success": False, "message": f"No running bot found for {symbol}"}
    return {"success": False, "message": f"Unknown action: {req.action}"}


@app.get("/api/lane_b/status")
async def api_status():
    """Legacy combined account-level status."""
    return build_combined_status()


@app.get("/api/lane_b/trades")
async def api_trades(limit: int = 50, offset: int = 0):
    """Legacy trades — returns XAUUSDm by default."""
    return get_mt5_trades("XAUUSDm", limit, offset)


@app.get("/api/lane_b/equity")
async def api_equity(window: str = "all"):
    """Legacy equity — returns XAUUSDm by default."""
    return get_equity_curve("XAUUSDm", window)


@app.post("/api/lane_b/control")
async def api_control(req: ControlRequest):
    """Legacy control endpoint — delegates to first known symbol."""
    bots = discover_bots()
    symbol = bots[0]["symbol"] if bots else KNOWN_SYMBOLS[0]
    return await api_symbol_control(symbol, req)


# ─── WebSocket endpoint ───
@app.websocket("/ws/lane_b")
async def ws_status(websocket: WebSocket):
    await websocket.accept()
    ws_clients.add(websocket)
    log.info(f"WS client connected ({len(ws_clients)} total)")
    try:
        while True:
            try:
                data = build_status()
                await websocket.send_json(data)
                await asyncio.sleep(2)
            except WebSocketDisconnect:
                break
            except Exception as e:
                log.warning(f"WS send error: {e}")
                break
    finally:
        ws_clients.discard(websocket)
        log.info(f"WS client disconnected ({len(ws_clients)} total)")


# ─── Static file serving ───
if DASHBOARD_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(DASHBOARD_DIR / "assets")), name="assets")

    @app.get("/")
    @app.get("/{full_path:path}")
    async def serve_react(full_path: str = ""):
        index_path = DASHBOARD_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"error": "Dashboard not built. Run: cd dashboard && npm run build"}


if __name__ == "__main__":
    import uvicorn
    print(f"Lane B Dashboard Backend")
    print(f"  Dashboard: {DASHBOARD_DIR}")
    print(f"  Runtime:   {RUNTIME_DIR}")
    print(f"  MT5: {'AVAILABLE' if HAS_MT5 else 'NOT AVAILABLE'}")
    print(f"  Port: {PORT}")
    print(f"  Open: http://localhost:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
