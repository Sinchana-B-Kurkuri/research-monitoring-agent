"""
access.py
~~~~~~~~~
Access control and audit logging for the frozen test split (Phase 4).

Every programmatic or manual access to test labels or the test split must be
recorded in `experiments/results/test_access.log` to prevent test-set leakage
or unrecorded tuning iterations.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Optional

from research_monitor.config import config

logger = logging.getLogger(__name__)


def get_default_log_path() -> Path:
    """Return Path to test_access.log from config or default."""
    path = Path(config.TEST_ACCESS_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def log_test_access(
    caller: str,
    purpose: str,
    sample_count: int,
    log_path: Optional[str | Path] = None,
) -> Path:
    """
    Log an authorized access event to the frozen test split.

    Args:
        caller: Name of module, script, or function accessing test data.
        purpose: Stated reason for access (e.g. 'Final baseline test reporting').
        sample_count: Number of test samples accessed.
        log_path: Optional custom path to log file.

    Returns:
        Path to the updated log file.
    """
    target_path = Path(log_path) if log_path else get_default_log_path()
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        log_entry = (
            f"{timestamp} | CALLER={caller} | PURPOSE={purpose} | "
            f"SAMPLES={sample_count}\n"
        )
        with open(target_path, "a", encoding="utf-8") as f:
            f.write(log_entry)
            f.flush()
    except Exception as exc:
        logger.critical(
            "Audit log write failure to %s: %s. Aborting test set access.",
            target_path,
            exc,
        )
        raise RuntimeError(
            f"CRITICAL: Test-set audit log write failed for '{target_path}'. "
            "Aborting operation to protect test set integrity (fail-closed)."
        ) from exc

    logger.info(
        "Logged test split access: caller=%s, purpose=%s, samples=%d",
        caller,
        purpose,
        sample_count,
    )
    return target_path


def get_test_access_history(log_path: Optional[str | Path] = None) -> list[str]:
    """
    Return all recorded access entries from the test access log.

    Args:
        log_path: Optional custom path to log file.

    Returns:
        List of log entry strings.
    """
    target_path = Path(log_path) if log_path else get_default_log_path()
    if not target_path.exists():
        return []
    with open(target_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
