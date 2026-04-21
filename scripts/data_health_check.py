"""
Cold-boot data health probe — runs the 12-layer EconomicIntelligence aggregator
and reports which layers are healthy / stale / stubbed.

Exit code 0 if no critical layer is broken, 1 otherwise. Wire into cron or run
manually before starting the paper-soak clock.

Usage:
    python scripts/data_health_check.py
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Dict

# Quiet the noisy per-layer warnings during init — we'll summarize them ourselves.
logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(name)s: %(message)s")


CRITICAL = {"social_sentiment", "derivatives", "onchain"}


def main() -> int:
    from src.data.economic_intelligence import EconomicIntelligence

    ei = EconomicIntelligence()
    results = ei.fetch_all_now()

    print("\n=== 12-layer data health ===")
    print(f"{'LAYER':<22} {'STATUS':<10} {'SIGNAL':<10} {'CONF':<6} {'NOTES'}")
    print("-" * 80)

    healthy: list[str] = []
    stubbed: list[str] = []
    failed: list[str] = []
    crit_broken: list[str] = []

    for name, r in sorted(results.items()):
        if r is None:
            status = "FAIL"
            notes = "layer returned None"
            failed.append(name)
            if name in CRITICAL:
                crit_broken.append(name)
            print(f"{name:<22} {status:<10} {'-':<10} {'-':<6} {notes}")
            continue
        if r.get("stub"):
            status = "STUB"
            stubbed.append(name)
            if name in CRITICAL:
                crit_broken.append(name)
            notes = f"reason={r.get('reason','?')[:40]}"
            print(f"{name:<22} {status:<10} {str(r.get('signal','?')):<10} {float(r.get('confidence',0)):<6.2f} {notes}")
            continue
        status = "OK"
        healthy.append(name)
        notes = r.get("source", "")
        print(f"{name:<22} {status:<10} {str(r.get('signal','?')):<10} {float(r.get('confidence',0)):<6.2f} {notes}")

    print("-" * 80)
    print(f"Healthy: {len(healthy)}/{len(results)}   Stubbed: {len(stubbed)}   Failed: {len(failed)}")

    if crit_broken:
        print(f"\nCRITICAL layers broken: {crit_broken}")
        print("Readiness: DEGRADED — fix these before starting the 14-day soak")
        return 1

    if stubbed or failed:
        print(f"\nNon-critical layers degraded: {stubbed + failed}")
        print("Readiness: OK — system can run; fix at leisure")
        return 0

    print("\nReadiness: ALL GREEN — safe to start paper soak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
