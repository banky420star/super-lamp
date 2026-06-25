"""Shared utilities for config, logging, and state I/O."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = PROJECT_ROOT / "state"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"


def ensure_dirs() -> None:
    """Create required runtime directories."""
    for path in (STATE_DIR, LOGS_DIR, DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load YAML configuration."""
    config_path = path or (PROJECT_ROOT / "config.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Configure a file + console logger."""
    ensure_dirs()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOGS_DIR / log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def read_json_state(filename: str, default: Any = None) -> Any:
    """Read JSON state file; return default if missing."""
    path = STATE_DIR / filename
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_state(filename: str, data: Any) -> Path:
    """Write JSON state file atomically."""
    ensure_dirs()
    path = STATE_DIR / filename
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
    tmp_path.replace(path)
    return path


def fail_safe_missing(filename: str, logger: logging.Logger) -> bool:
    """Log and return True if required input file is missing."""
    path = STATE_DIR / filename
    if not path.exists():
        logger.error("Required input missing: %s", path)
        return True
    return False