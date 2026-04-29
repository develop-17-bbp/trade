"""Analyst-only nightly QLoRA fine-tune scheduler for the RTX 5090 box.

Runs as Process 9 in START_ALL.ps1. Polls the local clock every 60s; at
03:00 ± 30 min UTC daily, fires one analyst-only cycle if all gates
pass:

  1. Bot uptime > 1h (skip during startup window)
  2. ≥ 100 new positive samples since last cycle
  3. 7-day Sharpe > 0.7 (don't train on pure-loss data)
  4. Quarantine list empty
  5. Hard cap: 1 cycle / 24h regardless of trigger

On promotion, dual_brain_trainer._hot_swap sets ACT_ANALYST_MODEL env;
the agentic loop reads at next tick. No Ollama daemon restart needed.

Logs JSON to logs/fine_tune/<YYYY-MM-DD>_analyst.json.

CLI:
    python -m scripts.nightly_analyst_finetune              # forever loop
    python -m scripts.nightly_analyst_finetune --once       # force one cycle now
    python -m scripts.nightly_analyst_finetune --dry-run    # stub backend

Honors ACT_DISABLE_FINETUNE=1 global kill switch.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.dual_brain_trainer import (  # noqa: E402
    StubBackend, run_cycle, persist_report,
)


logger = logging.getLogger("analyst_5090")

LOG_DIR = REPO_ROOT / "logs" / "fine_tune"
LAST_RUN_MARKER = REPO_ROOT / "scripts" / ".last_analyst_run.json"

ANALYST_BASE = os.getenv("ACT_ANALYST_BASE_MODEL", "qwen3-coder:30b")
SCHEDULE_HOUR_UTC = int(os.getenv("ACT_ANALYST_NIGHTLY_HOUR_UTC", "3"))
SCHEDULE_WINDOW_MIN = int(os.getenv("ACT_ANALYST_NIGHTLY_WINDOW_MIN", "30"))
MIN_NEW_SAMPLES = int(os.getenv("ACT_ANALYST_MIN_NEW_SAMPLES", "100"))
MIN_SHARPE_7D = float(os.getenv("ACT_ANALYST_MIN_SHARPE_7D", "0.7"))
MIN_BOT_UPTIME_S = float(os.getenv("ACT_ANALYST_MIN_BOT_UPTIME_S", "3600"))
HARD_CAP_S = 23 * 3600  # 1 / 24h


def _setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "analyst_5090.log", encoding="utf-8"),
        ],
    )


def _kill_switch_set() -> bool:
    return os.getenv("ACT_DISABLE_FINETUNE", "0") == "1"


def _last_run_ts() -> float:
    if not LAST_RUN_MARKER.exists():
        return 0.0
    try:
        d = json.loads(LAST_RUN_MARKER.read_text(encoding="utf-8"))
        return float(d.get("ts", 0.0))
    except Exception:
        return 0.0


def _write_last_run(summary: Dict[str, Any]) -> None:
    try:
        LAST_RUN_MARKER.write_text(
            json.dumps({"ts": time.time(), "summary": summary}, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.debug("last_run write failed: %s", e)


def _in_schedule_window() -> bool:
    """True if local UTC clock is within SCHEDULE_HOUR_UTC ± window."""
    now = datetime.now(tz=timezone.utc)
    hour_min = now.hour * 60 + now.minute
    target = SCHEDULE_HOUR_UTC * 60
    return abs(hour_min - target) <= SCHEDULE_WINDOW_MIN


def _sharpe_7d_ok() -> bool:
    """Rolling Sharpe over last 7d. If unreadable, assume OK (fail-open
    on the metric — we'd rather skip on hard counts than block on a
    metric-fetch glitch)."""
    try:
        from src.orchestration.warm_store import get_store
        import sqlite3
        store = get_store()
        cutoff_ns = int((time.time() - 7 * 86400) * 1e9)
        conn = sqlite3.connect(store.db_path, timeout=2.0)
        try:
            rows = conn.execute(
                "SELECT json_extract(self_critique, '$.realized_pnl_pct') "
                "FROM decisions WHERE ts_ns >= ? "
                "AND self_critique IS NOT NULL AND self_critique != '{}'",
                (cutoff_ns,),
            ).fetchall()
        finally:
            conn.close()
        pnls = [float(r[0]) for r in rows if r and r[0] is not None]
        if len(pnls) < 10:
            logger.info("sharpe_7d skip: only %d trades", len(pnls))
            return True  # fail-open
        import statistics
        mean = statistics.mean(pnls)
        sd = statistics.stdev(pnls) or 1e-9
        sharpe = mean / sd * (365.0 ** 0.5)
        logger.info("sharpe_7d = %.2f over %d trades", sharpe, len(pnls))
        return sharpe >= MIN_SHARPE_7D
    except Exception as e:
        logger.warning("sharpe_7d check failed (%s) — fail-open", e)
        return True


def _quarantine_empty() -> bool:
    """Reads credit_assigner quarantine list; OK if empty or unreadable."""
    try:
        from src.learning.credit_assigner import quarantine_list  # type: ignore
        q = list(quarantine_list() or [])
        if q:
            logger.info("quarantine non-empty: %s — skipping", q)
            return False
        return True
    except Exception:
        return True  # module not present or readable; fail-open


def _bot_uptime_ok() -> bool:
    """Skip if the trading bot started less than MIN_BOT_UPTIME_S ago.
    Detected via the executor PID file or the most-recent decision row."""
    try:
        from src.orchestration.warm_store import get_store
        import sqlite3
        store = get_store()
        conn = sqlite3.connect(store.db_path, timeout=2.0)
        try:
            row = conn.execute(
                "SELECT MIN(ts_ns) FROM decisions WHERE ts_ns >= ?",
                (int((time.time() - 86400) * 1e9),),
            ).fetchone()
        finally:
            conn.close()
        if not row or not row[0]:
            return True  # fail-open — no recent decisions, can't tell
        oldest_today = float(row[0]) / 1e9
        uptime_s = time.time() - oldest_today
        ok = uptime_s >= MIN_BOT_UPTIME_S
        logger.info("bot uptime ≈ %.0fs (>= %.0fs ? %s)", uptime_s, MIN_BOT_UPTIME_S, ok)
        return ok
    except Exception as e:
        logger.warning("uptime check failed (%s) — fail-open", e)
        return True


def _count_new_samples() -> int:
    try:
        from src.ai.training_data_filter import load_experience_samples
        samples, _ = load_experience_samples(
            asset=None, max_age_days=30.0, min_pnl_abs_pct=0.3,
        )
        return len(samples)
    except Exception as e:
        logger.warning("sample count failed: %s", e)
        return 0


def _run_cycle(dry_run: bool) -> Dict[str, Any]:
    started = time.time()
    summary: Dict[str, Any] = {
        "started_at": started, "dry_run": dry_run, "ok": False,
        "skipped_reason": None,
    }

    if _kill_switch_set():
        summary["skipped_reason"] = "ACT_DISABLE_FINETUNE=1"
        return summary

    # Hard cap: 1 cycle / 24h
    elapsed = time.time() - _last_run_ts()
    if elapsed < HARD_CAP_S:
        summary["skipped_reason"] = f"24h_cap (last run {elapsed:.0f}s ago)"
        return summary

    # Gate stack
    if not _bot_uptime_ok():
        summary["skipped_reason"] = "bot_uptime_too_short"
        return summary
    if not _quarantine_empty():
        summary["skipped_reason"] = "quarantine_non_empty"
        return summary
    if not _sharpe_7d_ok():
        summary["skipped_reason"] = f"sharpe_7d < {MIN_SHARPE_7D}"
        return summary

    n = _count_new_samples()
    summary["n_filtered_samples"] = n
    if n < MIN_NEW_SAMPLES:
        summary["skipped_reason"] = f"only {n} samples (< {MIN_NEW_SAMPLES})"
        return summary

    # ── Backend ────────────────────────────────────────────────────────
    if dry_run:
        backend = StubBackend()
        logger.info("DRY RUN — using StubBackend")
    else:
        try:
            from src.ai.unsloth_backend import UnslothQLoRABackend
            backend = UnslothQLoRABackend(export="gguf", lora_r=16)
        except Exception as e:
            summary["skipped_reason"] = f"unsloth_import_failed: {e}"
            return summary

    logger.info("starting analyst-only cycle (base=%s)", ANALYST_BASE)
    report = run_cycle(
        backend, brains=["analyst"], pause_agentic=True,
        analyst_incumbent=ANALYST_BASE,
    )
    summary["report"] = report.to_dict()
    persist_report(report)

    if report.analyst and report.analyst.training_ok:
        summary["ok"] = True
        summary["promoted"] = bool(report.analyst.promoted)
        summary["challenger_tag"] = report.analyst.challenger_tag
    else:
        summary["skipped_reason"] = "training_failed"

    summary["duration_s"] = round(time.time() - started, 1)
    _write_last_run(summary)
    return summary


def _persist_summary(summary: Dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    path = LOG_DIR / f"{date}_analyst.json"
    try:
        prev = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        prev = []
    prev.append(summary)
    try:
        path.write_text(json.dumps(prev, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        logger.debug("summary write failed: %s", e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="bypass schedule window, fire one cycle now")
    ap.add_argument("--dry-run", action="store_true", help="stub backend")
    args = ap.parse_args()

    _setup_logging()
    logger.info("analyst_5090 boot — base=%s schedule=%02d:00 UTC ± %dmin once=%s dry_run=%s",
                ANALYST_BASE, SCHEDULE_HOUR_UTC, SCHEDULE_WINDOW_MIN,
                args.once, args.dry_run)

    if args.once:
        # Bypass schedule + 24h cap by clearing last_run marker first.
        try:
            LAST_RUN_MARKER.unlink(missing_ok=True)
        except Exception:
            pass
        summary = _run_cycle(args.dry_run)
        _persist_summary(summary)
        logger.info("once-mode complete: ok=%s reason=%s",
                    summary.get("ok"), summary.get("skipped_reason"))
        return 0 if summary.get("ok") or summary.get("skipped_reason") else 1

    # ── Forever loop: poll clock every 60s ─────────────────────────────
    while True:
        try:
            if _in_schedule_window() and (time.time() - _last_run_ts()) >= HARD_CAP_S:
                logger.info("schedule window hit — firing analyst cycle")
                summary = _run_cycle(args.dry_run)
                _persist_summary(summary)
                logger.info("cycle complete: ok=%s reason=%s",
                            summary.get("ok"), summary.get("skipped_reason"))
        except Exception as e:
            logger.exception("loop iteration exception: %s", e)
        time.sleep(60.0)


if __name__ == "__main__":
    sys.exit(main())
