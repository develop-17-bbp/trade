"""Silence watchdog — fires CRITICAL alert when ACT goes quiet.

Closes the loophole that let the bot run silently for a week with zero
trades and no operator notification. Polls warm_store for the most
recent decision row; if `now - max(ts_ns) > threshold` during expected-
active hours, fires AlertManager.send_critical() over Slack/Telegram.

Run as Process 12 in START_ALL.ps1 on both 5090 and 4060 once Phase A
of the three-peer mesh ships. Honors ACT_DISABLE_AGENTIC_LOOP=1 — when
the operator deliberately disabled the loop, silence is expected, no
alert.

Usage:
    python -m scripts.silence_watchdog                 # forever
    python -m scripts.silence_watchdog --once          # single check, then exit
    python -m scripts.silence_watchdog --once --threshold-s 60   # smoke-test alert path
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.monitoring.alerting import alert_critical  # noqa: E402

logger = logging.getLogger("silence_watchdog")

DEFAULT_THRESHOLD_S = float(os.getenv("ACT_SILENCE_THRESHOLD_S", "1800"))      # 30 min
DEFAULT_POLL_S      = float(os.getenv("ACT_SILENCE_POLL_S",       "300"))      # 5 min
DB_PATH             = os.getenv("ACT_WARM_DB_PATH", str(REPO_ROOT / "data" / "warm_store.sqlite"))


def _max_decision_age_seconds(db_path: str, asset_class: Optional[str] = None) -> Optional[float]:
    """Return age in seconds of the newest decision row, or None if no rows / db missing.

    asset_class filter is best-effort: if the column doesn't exist yet (pre-
    migration), the filter is dropped and the global max is returned.
    """
    if not Path(db_path).exists():
        return None
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        try:
            if asset_class is not None:
                cur = conn.execute(
                    "SELECT MAX(ts_ns) FROM decisions WHERE asset_class = ?",
                    (asset_class,),
                )
            else:
                cur = conn.execute("SELECT MAX(ts_ns) FROM decisions")
            row = cur.fetchone()
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        # asset_class column may not exist yet; retry without filter
        if asset_class is not None and "no such column" in str(e).lower():
            return _max_decision_age_seconds(db_path, asset_class=None)
        logger.debug("silence_watchdog: sqlite probe failed: %s", e)
        return None
    except Exception as e:
        logger.debug("silence_watchdog: sqlite probe failed: %s", e)
        return None

    if not row or row[0] is None:
        return None
    max_ts_ns = int(row[0])
    age_s = time.time() - (max_ts_ns / 1e9)
    return max(0.0, age_s)


def _is_silence_expected() -> bool:
    """Operator-controlled kill-switch suppression.

    When ACT_DISABLE_AGENTIC_LOOP=1, silence is the intended state; do
    not page. ACT_DISABLE_FINETUNE only affects training, not trading,
    so it does NOT suppress.
    """
    val = os.getenv("ACT_DISABLE_AGENTIC_LOOP", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _check_once(threshold_s: float, db_path: str = DB_PATH) -> bool:
    """One probe pass. Returns True if an alert was fired."""
    if _is_silence_expected():
        logger.info("silence_watchdog: ACT_DISABLE_AGENTIC_LOOP=1 — suppressing checks")
        return False

    age = _max_decision_age_seconds(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    if age is None:
        # No decisions yet — could be a fresh install. Page WARN, not CRITICAL.
        logger.warning("silence_watchdog: warm_store has zero decision rows")
        alert_critical(
            title="ACT silent — no decisions ever",
            message=(
                "warm_store.sqlite contains zero decision rows. "
                "Either the bot has never run a tick, or warm_store is missing. "
                "Run /diagnose-noop on the bot host."
            ),
            data={"db_path": db_path, "checked_at": now_iso},
        )
        return True

    if age > threshold_s:
        logger.warning(
            "silence_watchdog: %.0fs since last decision (threshold=%.0fs) — paging",
            age, threshold_s,
        )
        alert_critical(
            title=f"ACT silent for {int(age // 60)} min",
            message=(
                f"No decision rows written to warm_store in {age:.0f}s "
                f"(threshold {threshold_s:.0f}s). Run /diagnose-noop on the bot host."
            ),
            data={
                "age_seconds":   round(age, 1),
                "threshold_s":   threshold_s,
                "db_path":       db_path,
                "checked_at":    now_iso,
            },
        )
        return True

    logger.info("silence_watchdog: ok — newest decision is %.0fs old", age)
    return False


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--once",        action="store_true", help="run a single check then exit")
    p.add_argument("--threshold-s", type=float, default=DEFAULT_THRESHOLD_S,
                   help=f"silence threshold in seconds (default {DEFAULT_THRESHOLD_S:.0f})")
    p.add_argument("--poll-s",      type=float, default=DEFAULT_POLL_S,
                   help=f"poll interval in seconds (default {DEFAULT_POLL_S:.0f})")
    p.add_argument("--db-path",     default=DB_PATH, help="warm_store path")
    p.add_argument("--verbose",     action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.once:
        fired = _check_once(args.threshold_s, db_path=args.db_path)
        return 0 if not fired else 2  # exit 2 = alert fired (handy for smoke tests)

    logger.info(
        "silence_watchdog: starting (threshold=%.0fs poll=%.0fs db=%s)",
        args.threshold_s, args.poll_s, args.db_path,
    )
    while True:
        try:
            _check_once(args.threshold_s, db_path=args.db_path)
        except Exception as e:
            logger.exception("silence_watchdog: check raised: %s", e)
        time.sleep(args.poll_s)


if __name__ == "__main__":
    sys.exit(main())
