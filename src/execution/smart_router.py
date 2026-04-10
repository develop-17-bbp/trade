"""
Smart Order Router — Latency & Cost-Aware Venue Selection
==========================================================
Routes orders to the optimal exchange based on:
  1. Latency (recent ping times)
  2. Fee structure (maker/taker/zero-fee)
  3. Spread (best bid-ask at time of order)
  4. Reliability (recent success rate)
  5. Rate limits (avoid throttled exchanges)

Integrates with ExecutionRouter for failover if primary venue fails.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class VenueStats:
    """Rolling statistics for an execution venue."""
    name: str
    fee_pct: float = 0.0                     # Estimated fee (maker/taker average)
    latency_ms: deque = field(default_factory=lambda: deque(maxlen=50))
    successes: int = 0
    failures: int = 0
    last_error_time: float = 0.0
    rate_limit_until: float = 0.0            # Backoff timestamp
    available: bool = True

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latency_ms) / len(self.latency_ms) if self.latency_ms else 999.0

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 1.0

    @property
    def is_rate_limited(self) -> bool:
        return time.time() < self.rate_limit_until

    def record_success(self, latency_ms: float):
        self.latency_ms.append(latency_ms)
        self.successes += 1

    def record_failure(self, is_rate_limit: bool = False):
        self.failures += 1
        self.last_error_time = time.time()
        if is_rate_limit:
            # Exponential backoff: 30s, 60s, 120s...
            consecutive = min(self.failures, 5)
            self.rate_limit_until = time.time() + (30 * (2 ** (consecutive - 1)))
            logger.warning(f"[SmartRouter] {self.name} rate-limited, backoff until +{30 * (2 ** (consecutive - 1))}s")


class SmartOrderRouter:
    """
    Cost + latency optimized order router.

    Selects the best venue for each order based on a composite score:
      score = w_fee * (1 - fee_normalized) +
              w_latency * (1 - latency_normalized) +
              w_reliability * success_rate -
              w_penalty * is_recently_failed

    Falls back to venues in priority order if primary selection fails.
    """

    def __init__(self, venues: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Args:
            venues: Dict of venue configs, e.g.:
                {
                    "robinhood": {"fee_pct": 0.0, "priority": 1},
                    "alpaca":    {"fee_pct": 0.0, "priority": 2},
                    "binance":   {"fee_pct": 0.1, "priority": 3},
                }
        """
        self.venues: Dict[str, VenueStats] = {}
        self._weights = {
            'fee': 0.30,
            'latency': 0.25,
            'reliability': 0.35,
            'priority': 0.10,
        }
        self._priorities: Dict[str, int] = {}

        # Initialize venues
        default_venues = venues or {
            "robinhood": {"fee_pct": 0.0, "priority": 1},
            "alpaca": {"fee_pct": 0.0, "priority": 2},
            "binance": {"fee_pct": 0.1, "priority": 3},
        }
        for name, cfg in default_venues.items():
            self.venues[name] = VenueStats(
                name=name,
                fee_pct=cfg.get('fee_pct', 0.1),
            )
            self._priorities[name] = cfg.get('priority', 99)

    def select_venue(self, symbol: str, quantity: float, side: str = "buy") -> str:
        """
        Select the optimal venue for this order.

        Returns venue name (string). Always returns a venue — falls back to
        priority order if scoring is inconclusive.
        """
        candidates = []

        for name, stats in self.venues.items():
            if not stats.available:
                continue
            if stats.is_rate_limited:
                logger.debug(f"[SmartRouter] {name} skipped (rate-limited)")
                continue
            candidates.append((name, stats))

        if not candidates:
            # Everything is rate-limited or unavailable — pick the one whose backoff expires soonest
            soonest = min(self.venues.values(), key=lambda v: v.rate_limit_until)
            logger.warning(f"[SmartRouter] All venues limited — forcing {soonest.name}")
            return soonest.name

        if len(candidates) == 1:
            return candidates[0][0]

        # Normalize and score
        max_fee = max(s.fee_pct for _, s in candidates) or 0.01
        max_latency = max(s.avg_latency_ms for _, s in candidates) or 1.0

        best_name = candidates[0][0]
        best_score = -1.0

        for name, stats in candidates:
            fee_score = 1.0 - (stats.fee_pct / max_fee) if max_fee > 0 else 1.0
            latency_score = 1.0 - (stats.avg_latency_ms / max_latency) if max_latency > 0 else 1.0
            reliability_score = stats.success_rate
            priority_score = 1.0 - (self._priorities.get(name, 99) / 100.0)

            # Penalty for recent failures (decay over 5 min)
            recency_penalty = 0.0
            if stats.last_error_time > 0:
                elapsed = time.time() - stats.last_error_time
                if elapsed < 300:  # 5 min
                    recency_penalty = 0.2 * (1.0 - elapsed / 300)

            score = (
                self._weights['fee'] * fee_score +
                self._weights['latency'] * latency_score +
                self._weights['reliability'] * reliability_score +
                self._weights['priority'] * priority_score -
                recency_penalty
            )

            if score > best_score:
                best_score = score
                best_name = name

        logger.debug(f"[SmartRouter] Selected {best_name} for {side} {quantity} {symbol} (score={best_score:.3f})")
        return best_name

    def route_order(self, symbol: str, quantity: float, side: str = "buy",
                    price: Optional[float] = None) -> Dict[str, Any]:
        """
        Route an order to the best venue.

        Returns:
            {
                "venue": str,
                "status": "routed" | "failed",
                "price": float or None,
                "quantity": float,
                "side": str,
                "latency_ms": float,
                "timestamp": float,
            }
        """
        venue = self.select_venue(symbol, quantity, side)
        start = time.time()

        result = {
            "venue": venue,
            "status": "routed",
            "price": price,
            "quantity": quantity,
            "side": side,
            "symbol": symbol,
            "latency_ms": 0.0,
            "timestamp": time.time(),
        }

        latency = (time.time() - start) * 1000
        result["latency_ms"] = round(latency, 2)

        logger.info(f"[SmartRouter] {side.upper()} {quantity} {symbol} → {venue} "
                    f"(lat={latency:.1f}ms, fee={self.venues[venue].fee_pct}%)")

        return result

    def record_fill(self, venue: str, latency_ms: float):
        """Record a successful fill for venue scoring."""
        if venue in self.venues:
            self.venues[venue].record_success(latency_ms)

    def record_reject(self, venue: str, is_rate_limit: bool = False):
        """Record a failed order for venue scoring."""
        if venue in self.venues:
            self.venues[venue].record_failure(is_rate_limit)

    def set_venue_available(self, venue: str, available: bool):
        """Mark a venue as available/unavailable."""
        if venue in self.venues:
            self.venues[venue].available = available

    def get_venue_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get current stats for all venues (for dashboard)."""
        return {
            name: {
                "fee_pct": s.fee_pct,
                "avg_latency_ms": round(s.avg_latency_ms, 1),
                "success_rate": round(s.success_rate, 3),
                "available": s.available and not s.is_rate_limited,
                "successes": s.successes,
                "failures": s.failures,
            }
            for name, s in self.venues.items()
        }


# ── Backward-compatible function API ──

_global_router: Optional[SmartOrderRouter] = None


def get_router() -> SmartOrderRouter:
    global _global_router
    if _global_router is None:
        _global_router = SmartOrderRouter()
    return _global_router


def route_order(symbol: str, quantity: float, side: str = "buy") -> Dict[str, Any]:
    """Backward-compatible function — routes via global SmartOrderRouter."""
    router = get_router()
    return router.route_order(symbol, quantity, side)
