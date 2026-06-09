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
