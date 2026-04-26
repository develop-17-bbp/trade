"""
Shadow-tick orchestrator — one unified per-asset tick.

The executor calls this from `_run_agentic_shadow` on every cycle. It
wires together the pieces that were built in isolation but needed a
single call-site:

  1. Fetch a web-context bundle (news / sentiment / institutional /
     polymarket / macro / fear_greed).
  2. Ingest the bundle into the real-time knowledge graph (C12).
  3. Run the Scanner brain on a compact prompt seeded with the
     bundle digests, publish a ScanReport into brain_memory (C7b).
     This is the corpus-callosum push — the Analyst's next compile
     will see it.
  4. Call `compile_agentic_plan` (which reads that scan + last
     analyst traces) to produce a TradePlan.
  5. Refresh the PersonaManager (throttled — once per N ticks).
  6. Return a compact dict summarizing what happened for the
     executor's warm_store row.

Design:
  * Every step is best-effort — a failure in ingest or scanner does
    not block the analyst path. The analyst still runs with whatever
    brain_memory has.
  * Scanner is SKIPPED when brain_memory already has a fresh scan
    (< FRESH_SCAN_S) to avoid burning LLM quota every tick on
    trivially-repeated state. The analyst always uses the latest.
  * Persona refresh is throttled via a process-local counter
    (PERSONA_REFRESH_EVERY_TICKS).
  * The hook is FULLY SAFE to call from the shadow executor — it
    never raises.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


FRESH_SCAN_S = float(os.getenv("ACT_SCAN_FRESH_S", "120"))
PERSONA_REFRESH_EVERY_TICKS = int(os.getenv("ACT_PERSONA_REFRESH_EVERY", "5"))
INGEST_EVERY_TICKS = int(os.getenv("ACT_GRAPH_INGEST_EVERY", "3"))
BODY_CONTROLLER_REFRESH_EVERY_TICKS = int(os.getenv("ACT_BODY_CONTROLLER_EVERY", "5"))

# Per-asset tick counters (process-local).
_tick_counters: Dict[str, int] = {}
_tick_lock = threading.Lock()


def _bump_tick(asset: str) -> int:
    with _tick_lock:
        _tick_counters[asset] = _tick_counters.get(asset, 0) + 1
        return _tick_counters[asset]


# ── Scanner pass ───────────────────────────────────────────────────────


def _extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    """Delegate to the agentic_trade_loop's JSON extractor — same logic
    (greedy regex + progressive shrinking), keep it in one place."""
    try:
        from src.ai.agentic_trade_loop import _extract_json
        return _extract_json(text)
    except Exception:
        return None


def _run_scanner(asset: str, quant_digest: str, bundle_block: str) -> bool:
    """Invoke dual_brain.scan() and publish a ScanReport to brain_memory.
    Returns True on publish, False on any failure."""
    try:
        from src.ai.dual_brain import scan
        from src.ai.brain_memory import ScanReport, publish_scan
    except Exception as e:
        logger.debug("shadow_tick: scan imports unavailable: %s", e)
        return False

    # Defensive: when quant_digest is empty AND bundle_block is empty,
    # the model receives a prompt with no substantive content and
    # produces empty/garbage output. Audit finding (2026-04-27): this
    # was a real failure mode -- on cold-boot the executor passed an
    # empty quant_digest before the first quant computation completed.
    quant_digest = (quant_digest or "").strip()
    bundle_block = (bundle_block or "").strip()
    if not quant_digest and not bundle_block:
        # Fall back to whatever the brain_memory has (even if stale)
        # so the model has at least price + regime context to
        # rationalize over rather than emitting empty JSON.
        try:
            from src.ai.brain_memory import get_brain_memory
            mem = get_brain_memory()
            latest = mem.read_latest_scan(asset, max_age_s=86400.0)
            if latest is not None and latest.rationale:
                quant_digest = (
                    f"[stale scan ts={latest.ts:.0f}] "
                    f"prior_score={latest.opportunity_score:.0f} "
                    f"prior_direction={latest.proposed_direction} "
                    f"prior_rationale={latest.rationale[:200]}"
                )
        except Exception:
            pass
        if not quant_digest:
            quant_digest = (
                f"[no quant digest yet -- first tick of session] "
                f"asset={asset} -- emit FLAT proposal until next tick "
                f"populates the digest."
            )

    prompt = (
        f"Asset: {asset}\n\n"
        f"Quant digest:\n{quant_digest}\n\n"
        f"Web context:\n{bundle_block or '(no web context this tick)'}\n\n"
        f"Respond with ONE JSON object only: "
        f'{{"opportunity_score": 0-100, "proposed_direction": "LONG"|"SHORT"|"FLAT", '
        f'"top_signals": ["s1","s2","s3"], "rationale": "<=200 chars"}}. '
        f"Be terse -- NO prose, NO <think> trace, ONLY the JSON."
    )
    try:
        resp = scan(prompt)
    except Exception as e:
        logger.debug("shadow_tick: scan() raised: %s", e)
        return False
    if not resp or not resp.ok:
        return False

    parsed = _extract_first_json(resp.text or "")
    if not isinstance(parsed, dict):
        return False

    try:
        report = ScanReport(
            asset=asset.upper(),
            ts=time.time(),
            opportunity_score=float(parsed.get("opportunity_score", 0.0) or 0.0),
            proposed_direction=str(parsed.get("proposed_direction", "FLAT") or "FLAT").upper(),
            top_signals=[str(s)[:60] for s in (parsed.get("top_signals") or [])][:8],
            rationale=str(parsed.get("rationale", "") or "")[:300],
            raw={
                "opportunity_score": parsed.get("opportunity_score"),
                "proposed_direction": parsed.get("proposed_direction"),
                "model": resp.model,
                "fallback_used": resp.fallback_used,
            },
        )
        publish_scan(report)
        return True
    except Exception as e:
        logger.debug("shadow_tick: publish_scan failed: %s", e)
        return False


# ── Ingest helpers ─────────────────────────────────────────────────────


def _ingest_bundle(asset: str, bundle: Dict[str, Any]) -> Dict[str, int]:
    """Push the web-context bundle into the knowledge graph."""
    from src.ai.graph_rag import (
        ingest_correlation,
        ingest_institutional,
        ingest_news,
        ingest_polymarket,
        ingest_sentiment,
    )
    counts = {"news": 0, "sentiment": 0, "institutional": 0, "polymarket": 0}

    # News — the fetcher returns a list of NewsItem objects OR dicts.
    try:
        news_items = (bundle.get("news") or {}).get("_raw_items", [])
        if news_items:
            counts["news"] = ingest_news(asset, news_items)
    except Exception as e:
        logger.debug("shadow_tick: ingest_news failed: %s", e)

    # Sentiment — take the latest digest value.
    try:
        sent = bundle.get("sentiment")
        if sent:
            # WebDigest-shaped: use the digest.confidence + tags heuristic.
            direction = 0
            tags = getattr(sent, "tags", None) or []
            if "bullish" in tags:
                direction = 1
            elif "bearish" in tags:
                direction = -1
            fake_vote = type("_V", (), {
                "direction": direction,
                "confidence": float(getattr(sent, "confidence", 0.5)),
            })()
            if ingest_sentiment(asset, fake_vote):
                counts["sentiment"] = 1
    except Exception as e:
        logger.debug("shadow_tick: ingest_sentiment failed: %s", e)

    # Institutional — re-fetch on demand since the web_context digest
    # already collapsed the numbers.
    try:
        from src.data.institutional_fetcher import InstitutionalFetcher
        inst = InstitutionalFetcher().get_all_institutional(asset.upper()) or {}
        if inst:
            counts["institutional"] = ingest_institutional(asset, inst)
    except Exception as e:
        logger.debug("shadow_tick: ingest_institutional failed: %s", e)

    # Polymarket — re-fetch for fresh markets.
    try:
        from src.data.polymarket_fetcher import PolymarketFetcher
        markets = PolymarketFetcher().fetch_crypto_markets() or []
        if markets:
            counts["polymarket"] = ingest_polymarket(asset, markets[:10])
    except Exception as e:
        logger.debug("shadow_tick: ingest_polymarket failed: %s", e)

    return counts


# ── Main entrypoint ────────────────────────────────────────────────────


def run_tick(asset: str, quant_digest: str = "") -> Dict[str, Any]:
    """One unified shadow tick. Returns a summary dict for audit.

    Never raises — every sub-step swallows its own errors.
    """
    out: Dict[str, Any] = {
        "asset": asset.upper(),
        "ts": time.time(),
        "scanner_published": False,
        "ingest_counts": {},
        "personas_refreshed": False,
        "plan": None,
    }
    tick_n = _bump_tick(asset.upper())

    # ── 1. Web bundle ──────────────────────────────────────────────────
    bundle: Dict[str, Any] = {}
    try:
        from src.ai.web_context import bundle_to_prompt_block, fetch_bundle
        bundle = fetch_bundle(asset.upper())
        bundle_block = bundle_to_prompt_block(bundle)
    except Exception as e:
        logger.debug("shadow_tick: web bundle fetch failed: %s", e)
        bundle_block = ""

    # ── 2. Ingest into graph (throttled) ──────────────────────────────
    if bundle and tick_n % max(1, INGEST_EVERY_TICKS) == 0:
        out["ingest_counts"] = _ingest_bundle(asset, bundle)

    # ── 3. Scanner (skip when fresh report exists) ────────────────────
    try:
        from src.ai.brain_memory import get_scan_for_analyst
        existing = get_scan_for_analyst(asset.upper(), max_age_s=FRESH_SCAN_S)
    except Exception:
        existing = None
    if existing is None:
        out["scanner_published"] = _run_scanner(asset.upper(), quant_digest, bundle_block)

    # ── 4. Compile the analyst plan (reads the scan we just published) ──
    try:
        from src.ai.agentic_bridge import compile_agentic_plan
        result = compile_agentic_plan(
            asset=asset.upper(),
            quant_data=quant_digest,
        )
        out["plan"] = {
            "direction": result.plan.direction,
            "entry_tier": result.plan.entry_tier,
            "size_pct": result.plan.size_pct,
            "terminated_reason": result.terminated_reason,
            "steps_taken": result.steps_taken,
        }
        out["_loop_result"] = result         # not serialized; for caller
    except Exception as e:
        logger.debug("shadow_tick: compile_agentic_plan failed: %s", e)

    # ── 5. Persona refresh (throttled) ────────────────────────────────
    if tick_n % max(1, PERSONA_REFRESH_EVERY_TICKS) == 0:
        try:
            from src.agents.persona_from_graph import get_manager
            report = get_manager().refresh(asset=asset.upper())
            out["personas_refreshed"] = True
            out["persona_report"] = {
                "spawned": len(report.get("spawned") or []),
                "dissolved": len(report.get("dissolved") or []),
                "kept": len(report.get("kept") or []),
            }
        except Exception as e:
            logger.debug("shadow_tick: persona refresh failed: %s", e)

    # ── 6. Brain-to-body controller refresh (C9, throttled) ───────────
    if tick_n % max(1, BODY_CONTROLLER_REFRESH_EVERY_TICKS) == 0:
        try:
            from src.learning.brain_to_body import get_controller
            controls = get_controller().refresh(asset=asset.upper())
            out["body_controls"] = {
                "exploration_bias": controls.exploration_bias,
                "emergency_level": controls.emergency_level,
                "priority_agents": controls.priority_agents[:5],
                "reason": controls.reason,
            }
        except Exception as e:
            logger.debug("shadow_tick: body controller refresh failed: %s", e)

    return out
