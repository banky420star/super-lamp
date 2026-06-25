"""Trade quantity limits — config helpers."""

from __future__ import annotations

from typing import Any


def unlimited_trades(config: dict[str, Any]) -> bool:
    """When true, no cap on candidates, exposure, or duplicate positions."""
    return bool(config.get("risk", {}).get("unlimited_trades", False))


def max_candidates_per_run(config: dict[str, Any]) -> int | None:
    """Return max candidates, or None for unlimited (0 or unlimited_trades)."""
    if unlimited_trades(config):
        return None
    n = int(config.get("signals", {}).get("max_candidates_per_run", 10))
    return None if n <= 0 else n