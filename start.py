"""MT5 Quant OS — single entry point. Starts supervisor, all services, and dashboard.

Usage:
  python start.py
  START_AGENT.bat
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.pipeline import run_pipeline
from core.supervisor import ManagedService, Supervisor
from core.utils import ensure_dirs, load_config, setup_logger

_shutdown = False

BANNER = r"""
 ███╗   ███╗████████╗███████╗     ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗     ██████╗ ███████╗
 ████╗ ████║╚══██╔══╝██╔════╝    ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝    ██╔═══██╗██╔════╝
 ██╔████╔██║   ██║   █████╗      ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║       ██║   ██║███████╗
 ██║╚██╔╝██║   ██║   ██╔══╝      ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║       ██║   ██║╚════██║
 ██║ ╚═╝ ██║   ██║   ███████╗    ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║       ╚██████╔╝███████║
 ╚═╝     ╚═╝   ╚═╝   ╚══════╝     ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝        ╚═════╝ ╚══════╝
"""


def _print_banner(version: str, dashboard_url: str | None) -> None:
    print(BANNER)
    print(f"  MT5 QUANT OS v{version}")
    print("  ─────────────────────────────────────────")
    services = [
        ("MT5 Connection", "pending"),
        ("Dashboard", dashboard_url or "starting…"),
        ("History Engine", "scheduled"),
        ("Trading Pipeline", "scheduled"),
        ("Research Engine", "scheduled"),
        ("Health Monitor", "active"),
    ]
    for label, status in services:
        print(f"  ✓ {label:<22} {status}")
    print("  ─────────────────────────────────────────")
    print("  Listening…  (Ctrl+C to stop)\n")


def _handle_signal(signum, frame) -> None:
    global _shutdown
    _shutdown = True


def _start_dashboard(config: dict, logger) -> str | None:
    dash_cfg = config.get("app", {}).get("dashboard", {})
    if not dash_cfg.get("enabled", True):
        return None

    host = dash_cfg.get("host", "127.0.0.1")
    port = int(dash_cfg.get("port", 8080))
    url = f"http://{host}:{port}"

    def _serve() -> None:
        from dashboard.server import run
        run(host=host, port=port)

    threading.Thread(target=_serve, name="dashboard", daemon=True).start()
    time.sleep(0.5)

    if dash_cfg.get("auto_open_browser", True):
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()

    logger.info("Dashboard at %s", url)
    return url


def start(once: bool = False) -> None:
    global _shutdown
    ensure_dirs()
    config = load_config()
    logger = setup_logger("quant_os", "system.log")
    app_cfg = config.get("app", {})
    sup_cfg = app_cfg.get("supervisor", {})

    interval = float(app_cfg.get("loop_interval_seconds", 60))
    history_interval = float(sup_cfg.get("history_interval_seconds", 300))
    research_interval = float(sup_cfg.get("research_interval_seconds", 1800))

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    dashboard_url = _start_dashboard(config, logger)
    _print_banner(app_cfg.get("version", "1.0"), dashboard_url)

    supervisor = Supervisor(config, logger)

    def _pipeline() -> dict:
        return run_pipeline(config, logger)

    def _history() -> dict:
        from loops import history_loop
        history_loop.run()
        return {"history_loop": "OK"}

    def _research() -> dict:
        from loops import research_loop
        research_loop.run()
        return {"research_loop": "OK"}

    pipeline_svc = ManagedService(
        "trading_pipeline",
        "Trading Pipeline",
        _pipeline,
        interval,
        logger,
        run_once=once,
    )
    pipeline_svc._on_result = supervisor._record_loop_results
    supervisor.register(pipeline_svc)

    if not once:
        supervisor.register(ManagedService(
            "history_engine",
            "History Engine",
            _history,
            history_interval,
            logger,
        ))
        supervisor.register(ManagedService(
            "research_engine",
            "Research Engine",
            _research,
            research_interval,
            logger,
        ))

    supervisor.start_all()

    try:
        while not _shutdown:
            time.sleep(0.5)
            if once and pipeline_svc.state.status in ("ok", "error", "stopped"):
                break
    finally:
        logger.info("Shutting down MT5 Quant OS…")
        supervisor.stop_all()
        print("\n  MT5 Quant OS stopped.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MT5 Quant OS")
    parser.add_argument("--once", action="store_true", help="Run one pipeline cycle then exit")
    args = parser.parse_args()
    start(once=args.once)