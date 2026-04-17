"""ACT v8.0 Memory System — Agent voting memory layer."""

import time
import logging

from src.memory.base_memory import LayerMemory

logger = logging.getLogger(__name__)


class AgentMemory(LayerMemory):
    """One instance per voting agent (12 total)."""

    def __init__(self, agent_name: str, db_dir: str = "memory/"):
        self.agent_name = agent_name
        super().__init__(layer_id=f"agent_{agent_name}", db_dir=db_dir)

    # ------------------------------------------------------------------
    def record_vote(
        self,
        direction: str,
        confidence: float,
        reasoning_summary: str,
        was_overruled: bool,
        final_outcome_pnl: float,
        final_outcome_label: str,
    ):
        self.record({
            "timestamp": time.time(),
            "market_regime": None,
            "signal_context": {
                "reasoning_summary": reasoning_summary,
                "was_overruled": was_overruled,
            },
            "action_taken": direction,
            "outcome_pnl": final_outcome_pnl,
            "outcome_label": final_outcome_label,
            "confidence_at_entry": confidence,
            "layer_id": self.layer_id,
            "extra_data": {"agent": self.agent_name, "was_overruled": was_overruled},
        })

    # ------------------------------------------------------------------
    def get_agent_accuracy(self, last_n: int = 100) -> dict:
        """Accuracy, veto accuracy, and confidence calibration over last_n votes."""
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT outcome_label, confidence_at_entry, signal_context, extra_data "
                    "FROM events ORDER BY timestamp DESC LIMIT ?",
                    (last_n,),
                ).fetchall()

            if not rows:
                return {"accuracy": 0.0, "veto_accuracy": 0.0, "confidence_calibration": {}}

            import json

            total = len(rows)
            wins = sum(1 for r in rows if r["outcome_label"] == "WIN")

            # Veto accuracy: when agent was overruled, how often was the final outcome a LOSS
            overruled = [r for r in rows if json.loads(r["extra_data"]).get("was_overruled")]
            veto_correct = sum(1 for r in overruled if r["outcome_label"] == "LOSS") if overruled else 0

            # Confidence calibration
            win_confs = [r["confidence_at_entry"] for r in rows if r["outcome_label"] == "WIN"]
            loss_confs = [r["confidence_at_entry"] for r in rows if r["outcome_label"] == "LOSS"]

            return {
                "accuracy": wins / total if total else 0.0,
                "veto_accuracy": veto_correct / len(overruled) if overruled else 0.0,
                "confidence_calibration": {
                    "avg_confidence_wins": sum(win_confs) / len(win_confs) if win_confs else 0.0,
                    "avg_confidence_losses": sum(loss_confs) / len(loss_confs) if loss_confs else 0.0,
                },
            }
        except Exception:
            logger.exception("get_agent_accuracy failed for %s", self.agent_name)
            return {"accuracy": 0.0, "veto_accuracy": 0.0, "confidence_calibration": {}}

    # ------------------------------------------------------------------
    def get_regime_specialization(self) -> str:
        """Return the regime where this agent is most accurate."""
        try:
            with self._lock:
                rows = self._conn.execute("""
                    SELECT market_regime,
                           SUM(CASE WHEN outcome_label='WIN' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS accuracy
                    FROM events
                    WHERE market_regime IS NOT NULL
                    GROUP BY market_regime
                    HAVING COUNT(*) >= 5
                    ORDER BY accuracy DESC
                    LIMIT 1
                """).fetchall()
            if rows:
                return rows[0]["market_regime"]
            return "UNKNOWN"
        except Exception:
            logger.exception("get_regime_specialization failed for %s", self.agent_name)
            return "UNKNOWN"

    # ------------------------------------------------------------------
    def recalibrate_weight(self, last_n: int = 100) -> float:
        """Suggested Bayesian weight 0.0–2.0 based on recent accuracy."""
        try:
            stats = self.get_agent_accuracy(last_n)
            acc = stats["accuracy"]

            if acc >= 0.75:
                return 2.0
            elif acc >= 0.65:
                return 1.5 + (acc - 0.65) * (0.5 / 0.10)  # 1.5 → 2.0
            elif acc >= 0.55:
                return 1.0 + (acc - 0.55) * (0.5 / 0.10)  # 1.0 → 1.5
            elif acc >= 0.45:
                return 0.5 + (acc - 0.45) * (0.5 / 0.10)  # 0.5 → 1.0
            else:
                return max(0.0, 0.5 * (acc / 0.45))        # 0.0 → 0.5
        except Exception:
            logger.exception("recalibrate_weight failed for %s", self.agent_name)
            return 1.0
