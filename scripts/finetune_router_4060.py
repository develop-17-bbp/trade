"""4060 cross-class finetune router.

Wraps `scripts/finetune_scanner_4060.py` with a market-hours-aware
scheduler so the 4060 box trades US stocks live during RTH and uses
the ~17.5 h/day outside RTH for finetune cycles — alternating
crypto-scanner and stocks-scanner training.

Per-loop decision tree:

    if is_us_market_open():
        # Box is busy live-trading stocks. No finetune.
        sleep until next_close OR next 30 minutes (whichever first).
    else:
        # Pick what to train next.
        if hours_since_stocks_train >= 3 and stocks_new_samples >= 30:
            run_cycle(asset_class='STOCK')
        elif hours_since_crypto_train >= 3 and crypto_new_samples >= 30:
            run_cycle(asset_class='CRYPTO')
        else:
            sleep_for(15min)

Operator overrides:
    ACT_FINETUNE_PRIORITY=stocks   stocks-first when both classes ready
    ACT_FINETUNE_PRIORITY=crypto   crypto-first
    ACT_FINETUNE_PRIORITY=alternate strict round-robin (default = stocks-first)
    ACT_DISABLE_FINETUNE=1         halts the whole router
    ACT_FINETUNE_MIN_INTERVAL_H=3.0 per-class minimum gap between cycles
    ACT_FINETUNE_MAX_PER_DAY=4     per-class hard cap per UTC day
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("finetune_router")

MIN_INTERVAL_H = float(os.getenv("ACT_FINETUNE_MIN_INTERVAL_H", "3.0"))
MAX_PER_DAY    = int(os.getenv("ACT_FINETUNE_MAX_PER_DAY", "4"))
PRIORITY       = (os.getenv("ACT_FINETUNE_PRIORITY") or "stocks").lower()
SLEEP_OPEN_S   = float(os.getenv("ACT_FINETUNE_SLEEP_OPEN_S",  "1800"))   # 30 min
SLEEP_IDLE_S   = float(os.getenv("ACT_FINETUNE_SLEEP_IDLE_S",  "900"))    # 15 min

ROUTER_STATE = REPO_ROOT / "scripts" / ".finetune_router_state.json"


def _kill_switch_set() -> bool:
    return os.getenv("ACT_DISABLE_FINETUNE", "0") == "1"


def _load_state() -> Dict:
    if not ROUTER_STATE.exists():
        return {}
    try:
        return json.loads(ROUTER_STATE.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _save_state(state: Dict) -> None:
    try:
        ROUTER_STATE.write_text(json.dumps(state), encoding="utf-8")
    except Exception:
        pass


def _hours_since(ts: Optional[float]) -> float:
    if not ts:
        return float("inf")
    return max(0.0, (time.time() - float(ts)) / 3600.0)


def _todays_count(state: Dict, asset_class: str) -> int:
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    counts = state.get("counts_by_day", {}).get(today, {})
    return int(counts.get(asset_class, 0))


def _bump_count(state: Dict, asset_class: str) -> None:
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    state.setdefault("counts_by_day", {}).setdefault(today, {})
    state["counts_by_day"][today][asset_class] = (
        int(state["counts_by_day"][today].get(asset_class, 0)) + 1
    )
    state[f"last_{asset_class.lower()}_train_ts"] = time.time()


def _can_train(state: Dict, asset_class: str) -> Tuple[bool, str]:
    """Per-class gate: hours-since + daily cap + new-sample threshold.

    Returns (ok, reason). If ok=False, reason is human-readable for logs.
    """
    if _todays_count(state, asset_class) >= MAX_PER_DAY:
        return False, f"daily_cap_{MAX_PER_DAY}_hit"
    last_ts = state.get(f"last_{asset_class.lower()}_train_ts")
    h = _hours_since(last_ts)
    if h < MIN_INTERVAL_H:
        return False, f"too_soon ({h:.1f}h < {MIN_INTERVAL_H}h)"

    # New-sample check — pulls warm_store via the trainer's helper so we
    # don't duplicate the filter logic.
    try:
        from scripts.finetune_scanner_4060 import _count_new_samples_for_class, MIN_NEW_SAMPLES
        n_new, _ = _count_new_samples_for_class(asset_class)
        if n_new < MIN_NEW_SAMPLES:
            return False, f"only_{n_new}_new_samples (< {MIN_NEW_SAMPLES})"
    except Exception as e:
        logger.debug("router: sample-count probe failed for %s: %s", asset_class, e)

    return True, "ready"


def _pick_next(state: Dict) -> Optional[str]:
    """Decide which class to train next, applying ACT_FINETUNE_PRIORITY."""
    stocks_ok,  stocks_reason  = _can_train(state, "STOCK")
    crypto_ok, crypto_reason = _can_train(state, "CRYPTO")

    if PRIORITY == "crypto":
        order = ["CRYPTO", "STOCK"]
    elif PRIORITY == "alternate":
        last = state.get("last_picked")
        order = ["STOCK", "CRYPTO"] if last == "CRYPTO" else ["CRYPTO", "STOCK"]
    else:
        # Default: stocks-first (operator basket grows fresher than crypto
        # post-launch since the live stocks bot is the new arrival).
        order = ["STOCK", "CRYPTO"]

    for cls in order:
        ok = stocks_ok if cls == "STOCK" else crypto_ok
        if ok:
            return cls
    logger.info("router: nothing ready (stocks=%s crypto=%s)", stocks_reason, crypto_reason)
    return None


def _run_cycle(asset_class: str, dry_run: bool) -> Dict:
    """Invoke the trainer. Imported lazily so import errors land in summary."""
    try:
        from scripts.finetune_scanner_4060 import _run_one_cycle, _persist_summary
    except Exception as e:
        return {"asset_class": asset_class, "ok": False,
                "skipped_reason": f"trainer import failed: {e}"}
    summary = _run_one_cycle(dry_run=dry_run, asset_class=asset_class)
    try:
        _persist_summary(summary)
    except Exception:
        pass
    return summary


def _is_market_open_safe() -> bool:
    try:
        from src.utils.market_hours import is_us_market_open
        return is_us_market_open()
    except Exception as e:
        logger.debug("router: market_hours import failed (%s); assuming open", e)
        return True


def _next_sleep_seconds() -> float:
    """When market is open, sleep until next_close OR 30 min. When closed,
    sleep idle window (15 min)."""
    try:
        from src.utils.market_hours import next_close
        nc = next_close().timestamp() - time.time()
        return max(60.0, min(nc, SLEEP_OPEN_S))
    except Exception:
        return SLEEP_OPEN_S


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--once",     action="store_true", help="single decision then exit")
    p.add_argument("--dry-run",  action="store_true", help="StubBackend, no GPU")
    p.add_argument("--force",    choices=["CRYPTO", "STOCK"],
                   help="override scheduler — train this class once regardless of gates")
    p.add_argument("--verbose",  action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if _kill_switch_set():
        logger.info("router: ACT_DISABLE_FINETUNE=1 — exiting")
        return 0

    if args.force:
        logger.info("router: --force %s — running one cycle regardless of gates", args.force)
        summary = _run_cycle(args.force, args.dry_run)
        logger.info("router: forced cycle: %s", summary.get("skipped_reason") or "ok")
        return 0

    while True:
        try:
            if _is_market_open_safe():
                sleep_s = _next_sleep_seconds()
                logger.info("router: NYSE in session; sleeping %.0fs", sleep_s)
                if args.once:
                    return 0
                time.sleep(sleep_s)
                continue

            state = _load_state()
            cls = _pick_next(state)
            if cls is None:
                logger.info("router: nothing to do; sleeping %.0fs", SLEEP_IDLE_S)
                if args.once:
                    return 0
                time.sleep(SLEEP_IDLE_S)
                continue

            logger.info("router: launching %s cycle", cls)
            summary = _run_cycle(cls, args.dry_run)
            if summary.get("ok"):
                _bump_count(state, cls)
            state["last_picked"] = cls
            _save_state(state)
            logger.info("router: %s cycle: ok=%s reason=%s",
                        cls, summary.get("ok"), summary.get("skipped_reason"))
            if args.once:
                return 0
            time.sleep(SLEEP_IDLE_S)
        except KeyboardInterrupt:
            logger.info("router: interrupted")
            return 130
        except Exception as e:
            logger.exception("router cycle: %s", e)
            time.sleep(SLEEP_IDLE_S)


if __name__ == "__main__":
    sys.exit(main())
