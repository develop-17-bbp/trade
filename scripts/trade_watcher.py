"""Continuous trade-firing watcher. Run alongside the bot in a
separate cmd window:

    python scripts\\trade_watcher.py

Polls warm_store + brain_memory + ollama every 30s and prints ONE
line per cycle showing whether the LLM is producing plans, whether
trades are firing, and whether the frontend will see them. Designed
so the operator can paste a single screen of output to me and I can
see exactly what's happening without scrolling through huge bot
logs.

Output line shape:
    [HH:MM:SS] llm=ok|empty  plans/min=N  shadow=N  agentic=N  paper_open=N  paper_closed=N  last=<asset/dir/tier>

Stop with Ctrl+C.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib import error, request

REPO_ROOT = Path(__file__).resolve().parent.parent
WARM_STORE = REPO_ROOT / "data" / "warm_store.sqlite"
BRAIN_MEMORY = REPO_ROOT / "data" / "brain_memory.sqlite"
PAPER_LOG = REPO_ROOT / "logs" / "robinhood_paper.jsonl"

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
POLL_INTERVAL_S = int(os.environ.get("ACT_WATCH_INTERVAL_S", "30"))


def _check_ollama() -> str:
    """Returns 'ok' if Ollama responds + has resident models, else
    'down' / 'no-models'."""
    try:
        with request.urlopen(f"{OLLAMA_URL}/api/ps", timeout=3.0) as r:
            data = json.loads(r.read().decode("utf-8"))
        models = data.get("models") or []
        if not models:
            return "no-models"
        return f"ok({len(models)})"
    except (error.URLError, OSError):
        return "down"
    except (json.JSONDecodeError, ValueError):
        return "bad-json"


def _scan_freshness_s() -> float:
    """Seconds since the most recent scan_report. Smaller = scanner
    actively producing. inf = nothing yet."""
    try:
        if not BRAIN_MEMORY.exists():
            return float("inf")
        c = sqlite3.connect(str(BRAIN_MEMORY), timeout=2.0)
        row = c.execute(
            "SELECT MAX(ts) FROM scan_reports"
        ).fetchone()
        c.close()
        if not row or row[0] is None:
            return float("inf")
        return max(0.0, time.time() - float(row[0]))
    except (sqlite3.OperationalError, OSError):
        return float("inf")


def _decision_counts(window_s: int) -> dict:
    """Count decisions in the recent window: total, shadow, agentic
    (real), and the most recent row's asset/direction/tier."""
    out = {"total": 0, "shadow": 0, "agentic": 0, "last": "-"}
    try:
        if not WARM_STORE.exists():
            return out
        cutoff_ns = int((time.time() - window_s) * 1_000_000_000)
        c = sqlite3.connect(str(WARM_STORE), timeout=2.0)
        out["total"] = c.execute(
            "SELECT COUNT(*) FROM decisions WHERE ts_ns >= ?", (cutoff_ns,),
        ).fetchone()[0]
        out["shadow"] = c.execute(
            "SELECT COUNT(*) FROM decisions WHERE ts_ns >= ? "
            "AND decision_id LIKE 'shadow-%'",
            (cutoff_ns,),
        ).fetchone()[0]
        out["agentic"] = out["total"] - out["shadow"]
        last = c.execute(
            "SELECT decision_id, symbol, direction, final_action "
            "FROM decisions ORDER BY ts_ns DESC LIMIT 1"
        ).fetchone()
        c.close()
        if last:
            d_id = (last[0] or "")[:7]
            out["last"] = (
                f"{last[1]}/{last[2]:+}" + (f"/{last[3]}" if last[3] else "")
                + f"#{d_id}"
            )
    except sqlite3.OperationalError:
        pass
    return out


def _paper_counts() -> dict:
    """Open + closed paper-trade counts from robinhood_paper.jsonl."""
    out = {"open": 0, "closed": 0, "last_entry": "-"}
    if not PAPER_LOG.exists():
        return out
    try:
        opens = 0
        closes = 0
        last_entry = "-"
        with PAPER_LOG.open(encoding="utf-8", errors="replace") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ev = rec.get("event", "")
                if ev == "ENTRY":
                    opens += 1
                    last_entry = (
                        f"{rec.get('asset', '?')}/"
                        f"{rec.get('direction', '?')}@"
                        f"${rec.get('fill_price', 0):.0f}"
                    )
                elif ev == "EXIT":
                    closes += 1
        out["open"] = max(0, opens - closes)
        out["closed"] = closes
        out["last_entry"] = last_entry
    except OSError:
        pass
    return out


def main() -> int:
    print(f" trade_watcher polling every {POLL_INTERVAL_S}s. Ctrl+C to stop.")
    print(f" warm_store: {WARM_STORE}")
    print(f" brain_memory: {BRAIN_MEMORY}")
    print(f" paper_log: {PAPER_LOG}")
    print()

    while True:
        try:
            ollama = _check_ollama()
            scan_age = _scan_freshness_s()
            scan_age_str = (
                f"{scan_age:.0f}s" if scan_age != float("inf") else "never"
            )
            d5 = _decision_counts(window_s=300)
            d60 = _decision_counts(window_s=3600)
            paper = _paper_counts()
            now = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{now}] ollama={ollama}  scan={scan_age_str}  "
                f"5min(s/a)={d5['shadow']}/{d5['agentic']}  "
                f"60min(s/a)={d60['shadow']}/{d60['agentic']}  "
                f"paper(o/c)={paper['open']}/{paper['closed']}  "
                f"last={d5['last']}  last_entry={paper['last_entry']}"
            )
            sys.stdout.flush()
        except Exception as e:  # pragma: no cover
            print(f"[{datetime.now():%H:%M:%S}] watcher error: {e}")

        try:
            time.sleep(POLL_INTERVAL_S)
        except KeyboardInterrupt:
            print("\n  trade_watcher stopped.")
            return 0


if __name__ == "__main__":
    sys.exit(main())
