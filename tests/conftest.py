import warnings
import threading

import pytest

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Disable loguru file output during pytest ──────────────────────────────
# Module-level loguru sinks (e.g. train_ppo.py's ppo_training.log)
# create file locks that interfere with test collection on Windows.
# Remove all handlers and redirect to stderr-only. Also monkey-patch
# logger.add to prevent any subsequently-imported module from creating
# file-path sinks during the test run.
# ───────────────────────────────────────────────────────────────────────────
from loguru import logger

logger.remove()

_original_loguru_add = logger.add


def _test_safe_add(sink=None, *args, **kwargs):
    """Wrap logger.add to redirect file-path sinks to stderr during tests."""
    if sink is None:
        sink = sys.stderr
    if isinstance(sink, str):
        # Drop file-specific kwargs that don't apply to stderr
        kwargs.pop("rotation", None)
        kwargs.pop("retention", None)
        kwargs.pop("compression", None)
        kwargs.setdefault("level", "WARNING")
        return _original_loguru_add(sys.stderr, *args, **kwargs)
    return _original_loguru_add(sink, *args, **kwargs)


logger.add = _test_safe_add

# Add a stderr sink so test output is visible
logger.add(sys.stderr, level="WARNING")


# ── Prevent background threads during pytest ────────────────────────────────
# HybridBrain._start_autonomy_if_enabled creates a daemon thread that
# survives across test modules and causes segfaults when stale state
# is accessed. Disable it globally before any test module is imported.
# ───────────────────────────────────────────────────────────────────────────
def pytest_configure(config):
    """Disable autonomy loop and other background threads for all tests."""
    os.environ["AGI_AUTONOMY_ENABLED"] = "false"

    # Belt-and-suspenders: also patch HybridBrain to no-op the method
    # (guards against codepaths that might bypass the env-var check)
    try:
        from Python.hybrid_brain import HybridBrain
        HybridBrain._start_autonomy_if_enabled = lambda self: None
    except (ImportError, AttributeError):
        pass


# ── Thread leak detection ──────────────────────────────────────────────────
# Background threads started by imported modules can survive across test
# modules and access stale state, causing segfaults.  This fixture warns
# when a test module starts threads that are still alive after it finishes.
# ───────────────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True, scope="module")
def _detect_thread_leaks():
    """Warn about and join background threads surviving past the test module."""
    main = threading.main_thread()
    before = {t.ident: t for t in threading.enumerate() if t is not main}
    yield
    after = {t.ident: t for t in threading.enumerate() if t is not main}
    leaked = {}
    for tid, t in after.items():
        if tid not in before and tid is not None:
            leaked[tid] = t
    if leaked:
        names = ", ".join(t.name for t in leaked.values())
        warnings.warn(f"Thread leak between test modules: {names}")
        # Attempt to join non-daemon threads; daemon threads cannot be
        # safely joined (they run infinite loops) — just report them.
        for t in leaked.values():
            if not t.daemon:
                t.join(timeout=1.0)
                if t.is_alive():
                    warnings.warn(f"Thread still alive after join: {t.name}")
