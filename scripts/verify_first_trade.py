"""One-shot post-restart sanity check. Run on the GPU box after
`git pull && STOP_ALL && START_ALL`. Tells you whether everything
needed for the first paper trade is wired correctly, and if not,
exactly which check failed.

Usage:
    cd C:\\Users\\admin\\trade
    python scripts/verify_first_trade.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib import error, request

REPO_ROOT = Path(__file__).resolve().parent.parent
WARM_STORE = REPO_ROOT / "data" / "warm_store.sqlite"
BRAIN_MEMORY = REPO_ROOT / "data" / "brain_memory.sqlite"
SOAK_OVERLAY = REPO_ROOT / "data" / "paper_soak_loose.json"

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
EXPECTED_PROFILE = os.environ.get("ACT_BRAIN_PROFILE", "dense_r1")
EXPECTED_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def _http_json(url: str, timeout: float = 5.0) -> Optional[dict]:
    try:
        with request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (error.URLError, json.JSONDecodeError, OSError):
        return None


def check_ollama_models() -> Tuple[bool, str]:
    print("\n[1/6] Ollama: are BOTH brain models resident?")
    data = _http_json(f"{OLLAMA_URL}/api/ps")
    if data is None:
        _fail(f"could not reach {OLLAMA_URL}/api/ps -- Ollama service down?")
        return False, "ollama_down"
    models = data.get("models") or []
    if not models:
        _fail("Ollama reports zero resident models. START_ALL pre-load failed.")
        return False, "no_models"
    names = [m.get("name", "") for m in models]
    ctxs = [m.get("context_length") or m.get("size_vram") or "?" for m in models]
    print(f"    resident: {list(zip(names, ctxs))}")
    if len(models) >= 2:
        _ok(f"{len(models)} models resident")
        return True, "ok"
    _fail("only 1 model resident -- the other was evicted.")
    _warn("Most likely: a python path is still passing num_ctx that exceeds VRAM.")
    return False, "evicted"


def check_brain_memory() -> Tuple[bool, str]:
    print("\n[2/6] Brain memory: scanner publishing ScanReports?")
    if not BRAIN_MEMORY.exists():
        _fail(f"{BRAIN_MEMORY} does not exist -- no scans yet.")
        return False, "missing"
    try:
        c = sqlite3.connect(str(BRAIN_MEMORY))
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT asset, ts, opportunity_score FROM scan_reports "
            "ORDER BY ts DESC LIMIT 5"
        ).fetchall()
        c.close()
    except sqlite3.OperationalError as e:
        _fail(f"could not query scan_reports: {e}")
        return False, "schema"
    if not rows:
        _fail("no rows in scan_reports -- scanner is silent or unparseable.")
        return False, "empty"
    most_recent = rows[0]["ts"]
    age = time.time() - float(most_recent)
    print(f"    most recent scan: {age:.0f}s ago, asset={rows[0]['asset']}, "
          f"score={rows[0]['opportunity_score']}")
    if age > 600:
        _warn(f"most recent scan is {age:.0f}s old (>10 min). Scanner may be stalled.")
        return False, "stale"
    _ok(f"{len(rows)} recent scans, freshest {age:.0f}s ago")
    return True, "ok"


def check_paper_soak_overlay() -> Tuple[bool, str]:
    print("\n[3/6] Paper-soak-loose: overlay present?")
    if not SOAK_OVERLAY.exists():
        _fail(f"{SOAK_OVERLAY} not found -- enable with: "
              "python -m src.skills.cli run paper-soak-loose enable=true")
        return False, "missing"
    try:
        overlay = json.loads(SOAK_OVERLAY.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        _fail(f"overlay file unreadable: {e}")
        return False, "corrupt"
    if not overlay.get("requires_paper_mode"):
        _fail("overlay missing requires_paper_mode=true safety flag")
        return False, "unsafe"
    sniper = overlay.get("sniper", {})
    print(f"    sniper.min_score={sniper.get('min_score')}, "
          f"min_move_pct={sniper.get('min_expected_move_pct')}, "
          f"min_confluence={sniper.get('min_confluence')}")
    _ok("paper-soak-loose active")
    return True, "ok"


def check_warm_store_decisions() -> Tuple[bool, str]:
    print("\n[4/6] Warm store: any decisions yet?")
    if not WARM_STORE.exists():
        _fail(f"{WARM_STORE} does not exist -- bot may not have started.")
        return False, "missing"
    try:
        c = sqlite3.connect(str(WARM_STORE))
        total = c.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        non_shadow = c.execute(
            "SELECT COUNT(*) FROM decisions "
            "WHERE decision_id NOT LIKE 'shadow-%'"
        ).fetchone()[0]
        recent = c.execute(
            "SELECT decision_id, asset, ts FROM decisions "
            "ORDER BY ts DESC LIMIT 5"
        ).fetchall()
        c.close()
    except sqlite3.OperationalError as e:
        _fail(f"warm_store query failed: {e}")
        return False, "schema"
    print(f"    total decisions: {total}")
    print(f"    non-shadow (real agentic): {non_shadow}")
    if recent:
        print("    last 5:")
        for r in recent:
            print(f"      - {r[0]} {r[1]} {r[2]}")
    if non_shadow == 0:
        _warn("zero non-shadow decisions yet -- the agentic-pipeline "
              "submit step has not fired. Wait 5-10 ticks then re-run.")
        return False, "no_trades"
    _ok(f"{non_shadow} non-shadow decision(s) recorded -- pipeline alive")
    return True, "ok"


def check_readiness_gate() -> Tuple[bool, str]:
    print("\n[5/6] Readiness gate: real-capital criteria status (paper soak progress)")
    try:
        from src.orchestration.readiness_gate import evaluate as _eval  # type: ignore
    except ImportError:
        _warn("readiness_gate module unimportable (paper-soak details skipped).")
        return True, "unimportable"
    try:
        res = _eval()
        crit = getattr(res, "criteria", {}) or {}
        for k, v in crit.items():
            print(f"    {k}: {v}")
        if getattr(res, "open_", False):
            _ok("readiness gate OPEN -- real capital can be enabled if you set ACT_REAL_CAPITAL_ENABLED=1")
        else:
            _warn(f"readiness gate CLOSED (expected during 14-day soak). reason={getattr(res, 'reason', '?')}")
        return True, "ok"
    except Exception as e:  # pragma: no cover
        _warn(f"readiness gate evaluation raised {type(e).__name__}: {e}")
        return True, "skipped"


def check_diagnose_noop() -> Tuple[bool, str]:
    print("\n[6/6] /diagnose-noop: top blocker (if any)")
    try:
        from src.skills.cli import _resolve_skill, _normalize_args  # type: ignore
        skill = _resolve_skill("diagnose-noop")
        if skill is None:
            _warn("/diagnose-noop skill not registered -- skipping.")
            return True, "missing"
        result = skill.run({})
        msg = getattr(result, "message", "") or getattr(result, "error", "") or ""
        for line in str(msg).splitlines()[:8]:
            print(f"    {line}")
        return True, "ok"
    except Exception as e:  # pragma: no cover
        _warn(f"diagnose-noop call failed: {type(e).__name__}: {e}")
        return True, "skipped"


def main() -> int:
    print("=" * 60)
    print(" ACT first-paper-trade verification")
    print("=" * 60)
    print(f" repo:    {REPO_ROOT}")
    print(f" profile: {EXPECTED_PROFILE}, expected ctx={EXPECTED_CTX}")
    print(f" ollama:  {OLLAMA_URL}")
    print()

    checks = [
        check_ollama_models,
        check_brain_memory,
        check_paper_soak_overlay,
        check_warm_store_decisions,
        check_readiness_gate,
        check_diagnose_noop,
    ]
    failures = []
    for fn in checks:
        try:
            ok, code = fn()
        except Exception as e:  # pragma: no cover
            print(f"  [ERROR] {fn.__name__} raised {type(e).__name__}: {e}")
            ok, code = False, "exception"
        if not ok:
            failures.append((fn.__name__, code))

    print()
    print("=" * 60)
    if not failures:
        print(" RESULT: all critical checks PASSED.")
        print(" Next: watch warm_store for first non-shadow decision_id.")
        print("=" * 60)
        return 0
    print(f" RESULT: {len(failures)} check(s) failed:")
    for name, code in failures:
        print(f"   - {name}: {code}")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())
