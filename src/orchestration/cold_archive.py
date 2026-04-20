"""Cold tier — Phase 3 parquet archive (Plan §1.3).

Monthly rollup from the warm SQLite into data/replay/YYYY-MM.parquet. Used
for offline RL pretraining + batch credit-assignment re-calibration (Phase
4.5a Shapley fallback).

Idempotent: running the archiver twice for the same month overwrites the
parquet from the latest SQLite state — no duplicates, no append.

Dependencies are soft: if pandas+pyarrow (via lightgbm's transitive install
or a direct pin) aren't available, archive_month returns False and logs a
warning. The warm tier keeps everything regardless.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ARCHIVE_DIR = Path(os.getenv(
    "ACT_COLD_ARCHIVE_DIR",
    str(Path(__file__).resolve().parents[2] / "data" / "replay"),
))


def _month_bounds(year: int, month: int) -> tuple[float, float]:
    """Return (start_ts, end_ts_exclusive) for the UTC calendar month."""
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    return start.timestamp(), end.timestamp()


def archive_month(year: int, month: int) -> Optional[Path]:
    """Export all outcomes in [year-month) to a parquet file.

    Returns the written Path on success, None on failure / no rows.
    """
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        logger.warning("pandas not installed — cold archive disabled.")
        return None

    from src.orchestration.warm_store import get_store

    start_ts, end_ts = _month_bounds(year, month)
    store = get_store()
    store.flush()
    conn = store._get_conn()  # intentionally use the internal handle for a read-only select
    try:
        import pandas as pd
        df = pd.read_sql_query(
            "SELECT * FROM outcomes WHERE exit_ts >= ? AND exit_ts < ? ORDER BY exit_ts",
            conn,
            params=(start_ts, end_ts),
        )
    except Exception as e:
        logger.warning("cold archive read failed: %s", e)
        return None

    if df.empty:
        logger.info("cold archive: no outcomes for %04d-%02d", year, month)
        return None

    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    out = _ARCHIVE_DIR / f"{year:04d}-{month:02d}.parquet"
    try:
        df.to_parquet(out, index=False)
        logger.info("cold archive wrote %d rows → %s", len(df), out)
        return out
    except Exception as e:
        logger.warning("cold archive write failed (%s). Is pyarrow installed?", e)
        return None


def archive_prior_month() -> Optional[Path]:
    """Convenience: archive the month that ended most recently.

    Intended to run as a monthly PeriodicJob (Phase 4 scheduler).
    """
    now = datetime.now(timezone.utc)
    if now.month == 1:
        return archive_month(now.year - 1, 12)
    return archive_month(now.year, now.month - 1)
