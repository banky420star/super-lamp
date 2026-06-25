"""MT5 connection test with session diagnostics — writes result to logs/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import MetaTrader5 as mt5
from core.mt5_client import detect_session_alignment, format_mt5_connection_error

OUT = ROOT / "logs" / "mt5_connect_test_result.json"
TERMINAL = r"C:\Users\Administrator\MT5Agent\terminal64.exe"

result: dict = {
    "session": detect_session_alignment(),
    "attempts": [],
    "success": False,
}

probes = [
    ("no_args", {}),
    ("path_only", {"path": TERMINAL, "timeout": 30000}),
    (
        "explicit_login",
        {
            "path": TERMINAL,
            "login": 435656990,
            "password": "Fuckyou2/",
            "server": "Exness-MT5Trial9",
            "timeout": 30000,
        },
    ),
]

for name, kwargs in probes:
    mt5.shutdown()
    ok = mt5.initialize(**kwargs)
    err = mt5.last_error()
    attempt = {"name": name, "initialize": ok, "last_error": str(err)}
    if ok:
        acc = mt5.account_info()
        attempt["account"] = {
            "login": acc.login if acc else None,
            "server": acc.server if acc else None,
            "balance": float(acc.balance) if acc else None,
        }
        attempt["symbols_total"] = mt5.symbols_total()
        result["attempts"].append(attempt)
        result["success"] = True
        result["account_info"] = attempt["account"]
        mt5.shutdown()
        break
    attempt["detail"] = format_mt5_connection_error(err, result["session"])
    result["attempts"].append(attempt)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
sys.exit(0 if result["success"] else 1)