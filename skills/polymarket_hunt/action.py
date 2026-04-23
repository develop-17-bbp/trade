"""
Skill: /polymarket-hunt — scan + evaluate + submit top Polymarket candidate.

Always safe by default: without ACT_POLYMARKET_LIVE=1 the executor
runs in shadow mode (no real orders placed, everything logged).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.skills.registry import SkillResult

logger = logging.getLogger(__name__)


def _load_config() -> Dict[str, Any]:
    try:
        import yaml
        from pathlib import Path
        cfg_path = Path(__file__).resolve().parents[3] / "config.yaml"
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


def _fetch_crypto_markets(limit: int) -> List[Dict[str, Any]]:
    """Pull live Polymarket crypto markets via existing fetcher."""
    try:
        from src.data.polymarket_fetcher import PolymarketFetcher
        f = PolymarketFetcher()
        markets = f.fetch_crypto_markets() or []
        # Prefer markets with highest 24h volume so we hit liquid ones first.
        markets.sort(key=lambda m: float(m.get("volume_24h") or 0.0), reverse=True)
        return markets[: max(1, int(limit))]
    except Exception as e:
        logger.debug("polymarket-hunt: fetch failed: %s", e)
        return []


def _estimate_probability(market: Dict[str, Any]) -> float:
    """Back-compat wrapper — kept because tests may import it. New
    code uses `_estimate_via_analyst` which returns a full
    PolymarketProbabilityEstimate with rationale + confidence."""
    est = _estimate_via_analyst(market)
    return est.estimated_yes_probability


def _estimate_via_analyst(market: Dict[str, Any]):
    """Delegate to the Analyst brain (C13b). Returns a
    PolymarketProbabilityEstimate. Never raises — the estimator has
    its own fallback path returning zero-edge."""
    try:
        from src.trading.polymarket_analyst import estimate_probability
        return estimate_probability(market)
    except Exception as e:
        logger.debug("polymarket-hunt: estimator failed: %s", e)
        # Final fallback — manual zero-edge estimate.
        from src.trading.polymarket_analyst import PolymarketProbabilityEstimate
        try:
            implied = float(market.get("yes_price") or 0.5)
        except Exception:
            implied = 0.5
        implied = max(0.01, min(0.99, implied))
        return PolymarketProbabilityEstimate(
            market_id=str(market.get("market_id") or "?"),
            question=str(market.get("question") or ""),
            implied_yes_probability=implied,
            estimated_yes_probability=implied,
            edge=0.0, confidence=0.0,
            rationale=f"hunt estimator exception: {type(e).__name__}",
            fallback_used=True,
        )


def run(args: Dict[str, Any]) -> SkillResult:
    if not args.get("confirm", False):
        return SkillResult(
            ok=False,
            error=(
                "pass confirm=true to run polymarket-hunt. "
                "Default mode is SHADOW — no real orders fire unless "
                "ACT_POLYMARKET_LIVE=1 AND config.polymarket.enabled=true "
                "AND the readiness gate is open."
            ),
        )

    limit = int(args.get("scan_limit") or 20)
    equity_usd = float(args.get("equity_usd") or 10_000.0)
    existing_exposure = float(args.get("existing_polymarket_exposure_usd") or 0.0)
    dry_run = bool(args.get("dry_run", True))  # default dry-run even with confirm

    markets = _fetch_crypto_markets(limit)
    if not markets:
        return SkillResult(
            ok=False,
            error="no Polymarket crypto markets fetched (network or fetcher unavailable)",
        )

    # Score each market through the Polymarket conviction gate.
    from src.trading.polymarket_conviction import evaluate as eval_pm

    scored: List[Dict[str, Any]] = []
    for m in markets:
        # C13b: delegate to the Analyst brain for a real probability
        # estimate. If the Analyst has no edge, it returns implied ==
        # estimated and the conviction gate auto-rejects.
        estimate = _estimate_via_analyst(m)
        est_p = estimate.estimated_yes_probability
        for side in ("YES", "NO"):
            r = eval_pm(
                market=m, proposed_side=side,
                estimated_probability=est_p,
                equity_usd=equity_usd,
                existing_polymarket_exposure_usd=existing_exposure,
            )
            if r.passed:
                scored.append({
                    "market": m,
                    "conviction": r.to_dict(),
                    "estimate": estimate.to_dict(),
                })
                break   # don't take both sides of the same market

    # Rank by tier then EV.
    tier_rank = {"sniper": 0, "normal": 1}
    scored.sort(key=lambda x: (
        tier_rank.get(x["conviction"]["tier"], 9),
        -float(x["conviction"]["expected_value_usd"] or 0.0),
    ))

    # Submit top candidate (if any).
    from src.exchanges.polymarket_executor import PolymarketExecutor
    executor = PolymarketExecutor(config=_load_config())
    submitted: Optional[Dict[str, Any]] = None

    if scored and not dry_run:
        top = scored[0]
        m = top["market"]
        c = top["conviction"]
        side = c["side"]
        buy_price = float(m.get("yes_price") or 0.5) if side == "YES" else \
                    float(m.get("no_price") or (1.0 - float(m.get("yes_price") or 0.5)))
        res = executor.place_order(
            market_id=str(m.get("market_id") or m.get("id") or ""),
            side=side,
            shares=int(c["shares"]),
            price=buy_price,
            plan_digest={"conviction": c, "market_question": m.get("question")},
        )
        submitted = res.to_dict()

    # Compact message.
    msg_lines = [
        f"Polymarket hunt — mode={executor.mode()} dry_run={dry_run}",
        f"  Scanned {len(markets)} markets, {len(scored)} passed conviction",
    ]
    for rank, s in enumerate(scored[:3], 1):
        c = s["conviction"]
        e = s.get("estimate") or {}
        q = (s["market"].get("question") or "")[:80]
        msg_lines.append(
            f"  #{rank} {c['tier']:<6} {c['side']:<3} "
            f"shares={c['shares']:<4} EV=${c['expected_value_usd']:,.2f} "
            f"edge={c['edge']:+.3f} conf={e.get('confidence', '-'):<5}  [{q}]"
        )
        if e.get("rationale"):
            msg_lines.append(f"           why: {e['rationale'][:120]}")
    if submitted:
        msg_lines.append(
            f"  Submitted: market={submitted['market_id']} "
            f"side={submitted['side']} shares={submitted['shares']} "
            f"mode={submitted['mode']} ok={submitted['ok']}"
        )
    elif scored:
        msg_lines.append("  (dry_run=True — no order submitted)")

    return SkillResult(
        ok=True,
        message="\n".join(msg_lines),
        data={
            "mode": executor.mode(),
            "scanned": len(markets),
            "passed_conviction": len(scored),
            "top3": scored[:3],
            "submitted": submitted,
        },
    )
