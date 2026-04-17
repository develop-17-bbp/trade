"""ACT v8.0 Memory System — Sentiment Memory Layer."""

import json
import time
import logging

from src.memory.base_memory import LayerMemory

logger = logging.getLogger(__name__)


class SentimentMemory(LayerMemory):
    """Single-instance memory for all sentiment signals."""

    VALID_SOURCES = ("news", "geopolitical", "macro", "social", "currency")

    def __init__(self, db_dir: str = "memory/"):
        super().__init__(layer_id="sentiment", db_dir=db_dir)

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------
    def record_sentiment(
        self,
        headline: str,
        sentiment_score: float,
        source_type: str,
        market_impact_pct: float | None = None,
    ):
        """Store a sentiment event."""
        if source_type not in self.VALID_SOURCES:
            logger.warning("Invalid source_type '%s' — expected one of %s", source_type, self.VALID_SOURCES)

        direction = "positive" if sentiment_score >= 0 else "negative"
        outcome_label = None
        if market_impact_pct is not None:
            # Direction correct if sentiment sign matches impact sign
            outcome_label = "WIN" if (sentiment_score * market_impact_pct) > 0 else "LOSS"

        extra = {
            "headline": headline,
            "sentiment_score": sentiment_score,
            "source_type": source_type,
            "market_impact_pct": market_impact_pct,
        }

        self.record({
            "timestamp": time.time(),
            "market_regime": source_type,
            "signal_context": {"headline": headline, "sentiment_score": sentiment_score},
            "action_taken": direction,
            "outcome_pnl": market_impact_pct if market_impact_pct is not None else 0.0,
            "outcome_label": outcome_label,
            "confidence_at_entry": abs(sentiment_score),
            "layer_id": self.layer_id,
            "extra_data": extra,
        })

    # ------------------------------------------------------------------
    # High-impact patterns
    # ------------------------------------------------------------------
    def get_high_impact_patterns(self) -> list[dict]:
        """Sentiment events where abs(market_impact_pct) > 2%, grouped by source_type."""
        sql = """
            SELECT market_regime AS source_type,
                   signal_context,
                   outcome_pnl AS market_impact_pct,
                   extra_data
            FROM events
            WHERE ABS(outcome_pnl) > 2.0
            ORDER BY ABS(outcome_pnl) DESC
        """
        try:
            with self._lock:
                rows = self._conn.execute(sql).fetchall()
            results = []
            for r in rows:
                extra = json.loads(r["extra_data"]) if r["extra_data"] else {}
                results.append({
                    "source_type": r["source_type"],
                    "headline": extra.get("headline", ""),
                    "sentiment_score": extra.get("sentiment_score"),
                    "market_impact_pct": r["market_impact_pct"],
                })
            return results
        except Exception:
            logger.exception("get_high_impact_patterns failed")
            return []

    # ------------------------------------------------------------------
    # Source reliability
    # ------------------------------------------------------------------
    def get_source_reliability(self, source_type: str) -> dict:
        """Return avg_impact, accuracy, trade_count for a given source_type."""
        sql = """
            SELECT COUNT(*) AS trade_count,
                   AVG(ABS(outcome_pnl)) AS avg_impact,
                   SUM(CASE WHEN outcome_label='WIN' THEN 1 ELSE 0 END) AS correct
            FROM events
            WHERE market_regime = ?
        """
        try:
            with self._lock:
                row = self._conn.execute(sql, (source_type,)).fetchone()
            count = row["trade_count"] or 0
            correct = row["correct"] or 0
            return {
                "avg_impact": row["avg_impact"] if row["avg_impact"] is not None else 0.0,
                "accuracy": (correct / count) if count > 0 else 0.0,
                "trade_count": count,
            }
        except Exception:
            logger.exception("get_source_reliability failed for %s", source_type)
            return {"avg_impact": 0.0, "accuracy": 0.0, "trade_count": 0}

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------
    def decay_old_sentiment(self, max_age_hours: int = 24):
        """Delete events older than max_age_hours."""
        cutoff = time.time() - (max_age_hours * 3600)
        try:
            with self._lock:
                cur = self._conn.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
                self._conn.commit()
            logger.info("Decayed %d old sentiment events (older than %dh)", cur.rowcount, max_age_hours)
        except Exception:
            logger.exception("decay_old_sentiment failed")
