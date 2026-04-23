"""
Polymarket-specific conviction gate — binary-option sizing math.

Polymarket markets are YES/NO binary outcomes with discrete 0-or-1
payouts, fundamentally different from Robinhood spot positions. The
existing conviction_gate.py (sniper/normal/reject tiers) was built for
continuous price moves; we need a parallel gate with binary-option
arithmetic before the agentic loop can safely submit Polymarket orders.

Input (from agentic loop + polymarket_fetcher):
  * market: {market_id, question, yes_price, no_price, end_ts, volume_24h}
  * thesis: LLM's direction (YES or NO) + estimated true probability
  * portfolio: current equity, existing Polymarket exposure
  * authority: still enforced — we don't bypass authority_rules
Output:
  * tier: 'sniper' | 'normal' | 'reject'
  * side: 'YES' | 'NO'
  * shares: integer (Polymarket trades in whole shares, $1 payout each)
  * cost_usd: cash outlay (= shares × price)
  * expected_value_usd: EV = shares × (estimated_prob - price) if YES,
    or shares × ((1 - estimated_prob) - (1 - price)) if NO
  * reasons: list of why this tier was chosen

Rules:
  * sniper: EV >= 8% of cost, volume_24h >= $10k, market expires within
    24h (short-horizon = less headline risk), |estimated_prob - price| >= 0.12
  * normal: EV >= 3% of cost, volume_24h >= $2k, expires within 7d,
    |edge| >= 0.05
  * reject: anything below normal thresholds OR market illiquid OR expired

Position sizing:
  * sniper: up to 3% of equity per market
  * normal: up to 1% of equity per market
  * Per-portfolio: total Polymarket exposure capped at 15% of equity
    (leaving 85% for spot BTC/ETH).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Tunables — env-overridable so operators can pin conservatively at first.
SNIPER_MIN_EV_PCT = float(os.getenv("ACT_PM_SNIPER_MIN_EV_PCT", "8.0"))
SNIPER_MIN_EDGE = float(os.getenv("ACT_PM_SNIPER_MIN_EDGE", "0.12"))
SNIPER_MAX_HOURS = float(os.getenv("ACT_PM_SNIPER_MAX_HOURS", "24.0"))
SNIPER_MIN_VOL_USD = float(os.getenv("ACT_PM_SNIPER_MIN_VOL_USD", "10000"))
SNIPER_MAX_EQUITY_PCT = float(os.getenv("ACT_PM_SNIPER_MAX_EQUITY_PCT", "3.0"))

NORMAL_MIN_EV_PCT = float(os.getenv("ACT_PM_NORMAL_MIN_EV_PCT", "3.0"))
NORMAL_MIN_EDGE = float(os.getenv("ACT_PM_NORMAL_MIN_EDGE", "0.05"))
NORMAL_MAX_HOURS = float(os.getenv("ACT_PM_NORMAL_MAX_HOURS", "168.0"))   # 7d
NORMAL_MIN_VOL_USD = float(os.getenv("ACT_PM_NORMAL_MIN_VOL_USD", "2000"))
NORMAL_MAX_EQUITY_PCT = float(os.getenv("ACT_PM_NORMAL_MAX_EQUITY_PCT", "1.0"))

PORTFOLIO_MAX_POLYMARKET_EQUITY_PCT = float(
    os.getenv("ACT_PM_PORTFOLIO_MAX_PCT", "15.0")
)

# Minimum order size — Polymarket's minimum is typically $1 (1 share).
MIN_SHARES = int(os.getenv("ACT_PM_MIN_SHARES", "10"))


@dataclass
class PolymarketConvictionResult:
    tier: str                                # 'sniper' | 'normal' | 'reject'
    passed: bool                             # True iff tier != 'reject'
    side: str                                # 'YES' | 'NO' | ''
    shares: int
    cost_usd: float
    expected_value_usd: float
    edge: float                              # |estimated_prob - price|
    hours_to_expiry: float
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier,
            "passed": bool(self.passed),
            "side": self.side,
            "shares": int(self.shares),
            "cost_usd": round(self.cost_usd, 2),
            "expected_value_usd": round(self.expected_value_usd, 2),
            "edge": round(self.edge, 4),
            "hours_to_expiry": round(self.hours_to_expiry, 1),
            "reasons": list(self.reasons),
        }


def _reject(reason: str, side: str = "", edge: float = 0.0,
            hours: float = 0.0) -> PolymarketConvictionResult:
    return PolymarketConvictionResult(
        tier="reject", passed=False, side=side, shares=0,
        cost_usd=0.0, expected_value_usd=0.0, edge=edge,
        hours_to_expiry=hours, reasons=[reason],
    )


def evaluate(
    *,
    market: Dict[str, Any],
    proposed_side: str,
    estimated_probability: float,
    equity_usd: float,
    existing_polymarket_exposure_usd: float = 0.0,
) -> PolymarketConvictionResult:
    """Decide tier + position size for a proposed Polymarket trade.

    Never raises — returns a reject-result with a reason on any
    malformed input.
    """
    # ── Input validation ───────────────────────────────────────────────
    try:
        yes_price = float(market.get("yes_price") or 0.0)
        no_price = float(market.get("no_price") or (1.0 - yes_price))
        end_ts = float(market.get("end_ts") or 0.0)
        volume_24h = float(market.get("volume_24h") or 0.0)
    except Exception as e:
        return _reject(f"malformed market data: {type(e).__name__}")

    side = (proposed_side or "").strip().upper()
    if side not in ("YES", "NO"):
        return _reject(f"invalid side {proposed_side!r}; expected YES or NO")

    est_p = float(estimated_probability or 0.0)
    if not (0.0 < est_p < 1.0):
        return _reject(f"estimated_probability {est_p} out of (0,1)")

    if yes_price <= 0.0 or yes_price >= 1.0:
        return _reject(f"yes_price {yes_price} out of (0,1)")
    if no_price <= 0.0 or no_price >= 1.0:
        return _reject(f"no_price {no_price} out of (0,1)")

    # Pick the relevant side's price.
    buy_price = yes_price if side == "YES" else no_price

    # Signed edge = (our probability that the chosen side wins)
    #             − (market's implied probability the chosen side wins).
    # `est_p` is the estimated probability of the YES side winning.
    # For a NO trade we win when YES LOSES, so our p(side) = 1 - est_p
    # and market's p(side) = 1 - yes_price.
    if side == "YES":
        signed_edge = est_p - yes_price
    else:
        signed_edge = (1.0 - est_p) - (1.0 - yes_price)    # = yes_price - est_p

    # Expiry check.
    import time as _t
    now = _t.time()
    hours_to_expiry = (end_ts - now) / 3600.0 if end_ts > 0 else 0.0
    if end_ts > 0 and now >= end_ts:
        return _reject("market expired", side=side, edge=signed_edge,
                       hours=hours_to_expiry)

    # ── Sizing helper: compute EV + shares for a given tier ───────────
    def _size_tier(tier_name: str, max_equity_pct: float) -> Dict[str, float]:
        max_cost = (max_equity_pct / 100.0) * max(0.0, equity_usd)
        # Portfolio cap — never let Polymarket exposure exceed the
        # portfolio-wide cap.
        portfolio_room = (
            (PORTFOLIO_MAX_POLYMARKET_EQUITY_PCT / 100.0) * equity_usd
            - existing_polymarket_exposure_usd
        )
        max_cost = max(0.0, min(max_cost, portfolio_room))
        shares = int(max_cost / buy_price) if buy_price > 0 else 0
        shares = max(0, shares)
        cost = shares * buy_price
        # EV per share bought: (1 × est_p) − price if we win with
        # probability est_p; 0 × (1 - est_p) − price otherwise.
        # Simplified: EV_per_share = est_p − buy_price (on YES side).
        if side == "YES":
            ev_per_share = est_p - buy_price
        else:
            ev_per_share = (1.0 - est_p) - buy_price
        ev_usd = shares * ev_per_share
        ev_pct = (ev_usd / cost * 100.0) if cost > 1e-9 else 0.0
        return {
            "shares": float(shares), "cost": cost,
            "ev_usd": ev_usd, "ev_pct": ev_pct,
        }

    # ── Tier checks ────────────────────────────────────────────────────
    # Sniper first; fall through to normal; else reject.
    reasons: List[str] = []

    sniper_ok = (
        signed_edge >= SNIPER_MIN_EDGE
        and volume_24h >= SNIPER_MIN_VOL_USD
        and (hours_to_expiry <= SNIPER_MAX_HOURS or end_ts == 0)
    )
    if sniper_ok:
        s = _size_tier("sniper", SNIPER_MAX_EQUITY_PCT)
        if s["shares"] >= MIN_SHARES and s["ev_pct"] >= SNIPER_MIN_EV_PCT:
            return PolymarketConvictionResult(
                tier="sniper", passed=True, side=side,
                shares=int(s["shares"]), cost_usd=s["cost"],
                expected_value_usd=s["ev_usd"], edge=signed_edge,
                hours_to_expiry=hours_to_expiry,
                reasons=[
                    f"edge {signed_edge:+.3f} ≥ {SNIPER_MIN_EDGE:.2f}",
                    f"vol24h ${volume_24h:,.0f} ≥ ${SNIPER_MIN_VOL_USD:,.0f}",
                    f"expires in {hours_to_expiry:.1f}h",
                    f"EV {s['ev_pct']:.1f}% ≥ {SNIPER_MIN_EV_PCT:.1f}%",
                ],
            )
        reasons.append(
            f"sniper math short: shares={int(s['shares'])} "
            f"EV%={s['ev_pct']:.1f} (need {SNIPER_MIN_EV_PCT:.1f})"
        )
    else:
        if signed_edge < SNIPER_MIN_EDGE:
            reasons.append(f"edge {signed_edge:+.3f} < sniper {SNIPER_MIN_EDGE:.2f}")
        if volume_24h < SNIPER_MIN_VOL_USD:
            reasons.append(f"vol24h ${volume_24h:,.0f} < sniper ${SNIPER_MIN_VOL_USD:,.0f}")
        if hours_to_expiry > SNIPER_MAX_HOURS and end_ts > 0:
            reasons.append(f"expires {hours_to_expiry:.1f}h > sniper {SNIPER_MAX_HOURS:.1f}h")

    normal_ok = (
        signed_edge >= NORMAL_MIN_EDGE
        and volume_24h >= NORMAL_MIN_VOL_USD
        and (hours_to_expiry <= NORMAL_MAX_HOURS or end_ts == 0)
    )
    if normal_ok:
        s = _size_tier("normal", NORMAL_MAX_EQUITY_PCT)
        if s["shares"] >= MIN_SHARES and s["ev_pct"] >= NORMAL_MIN_EV_PCT:
            return PolymarketConvictionResult(
                tier="normal", passed=True, side=side,
                shares=int(s["shares"]), cost_usd=s["cost"],
                expected_value_usd=s["ev_usd"], edge=signed_edge,
                hours_to_expiry=hours_to_expiry,
                reasons=[
                    f"edge {signed_edge:+.3f} ≥ {NORMAL_MIN_EDGE:.2f}",
                    f"vol24h ${volume_24h:,.0f} ≥ ${NORMAL_MIN_VOL_USD:,.0f}",
                    f"expires in {hours_to_expiry:.1f}h",
                    f"EV {s['ev_pct']:.1f}% ≥ {NORMAL_MIN_EV_PCT:.1f}%",
                ],
            )
        reasons.append(
            f"normal math short: shares={int(s['shares'])} "
            f"EV%={s['ev_pct']:.1f} (need {NORMAL_MIN_EV_PCT:.1f})"
        )
    else:
        if signed_edge < NORMAL_MIN_EDGE:
            reasons.append(f"edge < normal {NORMAL_MIN_EDGE:.2f}")
        if volume_24h < NORMAL_MIN_VOL_USD:
            reasons.append(f"vol24h < normal ${NORMAL_MIN_VOL_USD:,.0f}")

    return PolymarketConvictionResult(
        tier="reject", passed=False, side=side, shares=0,
        cost_usd=0.0, expected_value_usd=0.0, edge=signed_edge,
        hours_to_expiry=hours_to_expiry,
        reasons=reasons or ["no qualifying tier"],
    )
