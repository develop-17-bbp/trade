"""ACT v8.0 Memory System — Quant model memory layer."""

import time
import logging

from src.memory.base_memory import LayerMemory

logger = logging.getLogger(__name__)


class QuantMemory(LayerMemory):
    """One instance per quantitative model (lgbm, patchtst, rl)."""

    def __init__(self, model_name: str, db_dir: str = "memory/"):
        self.model_name = model_name
        super().__init__(layer_id=f"quant_{model_name}", db_dir=db_dir)

    # ------------------------------------------------------------------
    def record_prediction(
        self,
        asset: str,
        direction: str,
        confidence: float,
        features_top5: list,
        regime: str,
        hurst: float,
        volatility: float,
        session: str,
        outcome_pnl: float,
        outcome_label: str,
    ):
        self.record({
            "timestamp": time.time(),
            "market_regime": regime,
            "signal_context": {
                "asset": asset,
                "direction": direction,
                "features_top5": features_top5,
                "hurst": hurst,
                "volatility": volatility,
                "session": session,
            },
            "action_taken": direction,
            "outcome_pnl": outcome_pnl,
            "outcome_label": outcome_label,
            "confidence_at_entry": confidence,
            "layer_id": self.layer_id,
            "extra_data": {"model": self.model_name, "asset": asset},
        })

    # ------------------------------------------------------------------
    def extract_winning_patterns(self) -> dict:
        """Return patterns with win_rate > 0.6."""
        try:
            self.consolidate()
            with self._lock:
                rows = self._conn.execute(
                    "SELECT pattern_signature, frequency, avg_pnl, win_rate, best_regime "
                    "FROM pattern_index WHERE win_rate > 0.6"
                ).fetchall()
            return {
                r["pattern_signature"]: {
                    "frequency": r["frequency"],
                    "avg_pnl": r["avg_pnl"],
                    "win_rate": r["win_rate"],
                    "best_regime": r["best_regime"],
                }
                for r in rows
            }
        except Exception:
            logger.exception("extract_winning_patterns failed for %s", self.model_name)
            return {}

    # ------------------------------------------------------------------
    def get_regime_performance(self) -> dict:
        """Performance keyed by regime (BULL/BEAR/SIDEWAYS/CRISIS)."""
        result = {}
        try:
            with self._lock:
                rows = self._conn.execute("""
                    SELECT market_regime,
                           COUNT(*)  AS trade_count,
                           AVG(outcome_pnl) AS avg_pnl,
                           SUM(CASE WHEN outcome_label='WIN' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS accuracy
                    FROM events
                    GROUP BY market_regime
                """).fetchall()
            for r in rows:
                regime = r["market_regime"] or "UNKNOWN"
                result[regime] = {
                    "accuracy": r["accuracy"],
                    "avg_pnl": r["avg_pnl"],
                    "trade_count": r["trade_count"],
                }
        except Exception:
            logger.exception("get_regime_performance failed for %s", self.model_name)
        return result

    # ------------------------------------------------------------------
    def should_trust_model(self, regime: str, volatility: float) -> bool:
        """False if model win_rate in the given regime < 0.55."""
        perf = self.get_regime_performance()
        regime_data = perf.get(regime)
        if regime_data is None:
            return True  # no data yet — give benefit of the doubt
        return regime_data["accuracy"] >= 0.55
