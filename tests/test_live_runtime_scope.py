import pytest
from pathlib import Path


def test_recent_live_runtime_decisions_are_btc_and_xau_only():
    log_path = Path("logs") / "server.log"
    if not log_path.exists():
        pytest.skip("No live runtime log present in CI")

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    recent_decisions = [line for line in lines if "DECISION " in line][-20:]

    if not recent_decisions:
        pytest.skip("No recent DECISION lines present in CI fixture")

    for line in recent_decisions:
        assert "BTCUSDm" in line or "XAUUSDm" in line,             f"Unexpected symbol in DECISION line: {line}"
