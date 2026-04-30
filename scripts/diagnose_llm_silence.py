"""Disk-side proof of what the LLM lane is doing right now.

Answers, from warm_store.sqlite alone (no MCP needed):
  1. Are there ANY non-shadow LONG/SHORT decisions in the last 24h?
     (those are the ones submit_trade_plan writes on a successful LLM submit)
  2. If only SHADOW_SKIP / SHADOW_FLAT ? group the skip reasons so we
     know WHY the LLM is silent (parse_failure, max_steps, low_vol,
     macro_crisis, etc).
  3. Is paper_exploration_tick.py firing?
     (paper_explore conviction_tier rows in the last 24h)
  4. Is the technical lane being gated correctly?
     (ACT_LLM_SOLE_AUTHOR=1 -> no records from _evaluate_entry -> zero
     non-shadow rows authored by source != agentic_brain_submit)

Run on the 4060 OR 5090:
    python scripts/diagnose_llm_silence.py
    python scripts/diagnose_llm_silence.py --hours 6
    python scripts/diagnose_llm_silence.py --tail 20

Tail mode prints the latest 20 decisions with skip_reason, so you can
see at a glance whether parse_failure is dominating.
"""
from __future__ import annotations
import argparse
import collections
import datetime as _dt
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hours", type=int, default=24)
    p.add_argument("--tail", type=int, default=10,
                   help="Show this many most-recent decisions with skip_reason.")
    args = p.parse_args()

    from src.orchestration.warm_store import get_store
    store = get_store()
    store.flush()
    conn = store._get_conn()

    cutoff_ns = (_dt.datetime.now(_dt.timezone.utc) -
                 _dt.timedelta(hours=args.hours)).timestamp() * 1e9

    # -- 1. Action histogram --
    rows = conn.execute(
        "SELECT final_action, COUNT(*) FROM decisions "
        "WHERE ts_ns >= ? GROUP BY final_action ORDER BY COUNT(*) DESC",
        (int(cutoff_ns),),
    ).fetchall()
    print(f"\n=== Decisions in the last {args.hours}h (by final_action) ===")
    if not rows:
        print(f"  [EMPTY] zero decisions in warm_store since "
              f"{_dt.datetime.fromtimestamp(cutoff_ns/1e9, _dt.timezone.utc).isoformat()}")
        print(f"  -> the agentic loop probably isn't running; check ACT_AGENTIC_LOOP=1")
        return 1
    for action, n in rows:
        print(f"  {n:>6} {action}")

    # -- 2. Non-shadow LONG/SHORT (the ones we want to see) ---------
    longs = conn.execute(
        "SELECT COUNT(*) FROM decisions WHERE ts_ns >= ? "
        "AND decision_id NOT LIKE 'shadow-%' "
        "AND (final_action='LONG' OR final_action='SHORT')",
        (int(cutoff_ns),),
    ).fetchone()[0]
    print(f"\n=== LLM-authored (NON-shadow LONG/SHORT) in last {args.hours}h ===")
    print(f"  {longs} successful LLM submissions")
    if longs == 0:
        print(f"  -> LLM has not produced a single actionable plan in {args.hours}h")

    # -- 3. paper_exploration trace ---------------------------------
    explore = conn.execute(
        "SELECT COUNT(*), MIN(ts_ns), MAX(ts_ns) FROM decisions "
        "WHERE ts_ns >= ? AND decision_id LIKE 'paper_explore_%'",
        (int(cutoff_ns),),
    ).fetchone()
    print(f"\n=== paper_exploration_tick.py firings in last {args.hours}h ===")
    if explore and explore[0] > 0:
        print(f"  {explore[0]} explore trades")
    else:
        print(f"  0 explore trades ? paper_exploration loop hasn't fired since cutoff.")
        print(f"  (after the 2026-04-30 fix, this should fire when no real-trade in 4h)")

    # -- 4. Skip-reason histogram -----------------------------------
    print(f"\n=== Why agentic ticks ended (skip reasons / terminated_reason) ===")
    skip_rows = conn.execute(
        "SELECT component_signals FROM decisions "
        "WHERE ts_ns >= ? AND decision_id LIKE 'shadow-%' "
        "ORDER BY ts_ns DESC LIMIT 500",
        (int(cutoff_ns),),
    ).fetchall()
    reason_counts = collections.Counter()
    for (cs_raw,) in skip_rows:
        if not cs_raw:
            continue
        try:
            cs = json.loads(cs_raw) if isinstance(cs_raw, str) else cs_raw
        except Exception:
            continue
        tr = cs.get("terminated_reason") or cs.get("skip_reason") or "(no_reason)"
        reason_counts[str(tr)[:60]] += 1
    if not reason_counts:
        print("  no shadow rows with terminated_reason ? agentic loop not writing telemetry")
    else:
        for reason, n in reason_counts.most_common(15):
            print(f"  {n:>4}  {reason}")

    # -- 5. Tail latest decisions -----------------------------------
    print(f"\n=== Latest {args.tail} decisions (any kind) ===")
    tail = conn.execute(
        "SELECT ts_ns, decision_id, symbol, final_action, component_signals "
        "FROM decisions WHERE ts_ns >= ? ORDER BY ts_ns DESC LIMIT ?",
        (int(cutoff_ns), int(args.tail)),
    ).fetchall()
    for ts, did, sym, fa, cs_raw in tail:
        ts_str = _dt.datetime.fromtimestamp(ts/1e9, _dt.timezone.utc).strftime("%H:%M:%S")
        tr = ""
        if cs_raw:
            try:
                cs = json.loads(cs_raw) if isinstance(cs_raw, str) else cs_raw
                tr = cs.get("terminated_reason") or cs.get("skip_reason") or ""
                tr = (str(tr)[:50])
            except Exception:
                pass
        kind = "SHADOW" if did.startswith("shadow-") else \
               "EXPLORE" if did.startswith("paper_explore") else \
               "AGENTIC" if did.startswith("agentic-") else "OTHER"
        print(f"  {ts_str} {kind:>7} {sym:>6} {fa:>14}  {tr}")

    return 0 if longs > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
