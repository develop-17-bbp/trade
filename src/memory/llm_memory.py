"""ACT v8.0 Memory System — LLM Decision Memory Layer."""

import json
import time
import logging

from src.memory.base_memory import LayerMemory

logger = logging.getLogger(__name__)


class LLMMemory(LayerMemory):
    """Per-model memory for LLM decision tracking (mistral_scanner / llama_analyst)."""

    def __init__(self, model_name: str, db_dir: str = "memory/"):
        super().__init__(layer_id=f"llm_{model_name}", db_dir=db_dir)
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------
    def record_decision(
        self,
        prompt_hash: str,
        parsed_output: dict,
        trade_outcome_pnl: float,
        trade_outcome_label: str,
        bear_veto_fired: bool,
        actual_move_pct: float,
        predicted_move_pct: float,
    ):
        """Store a full LLM decision context as an event."""
        direction = parsed_output.get("direction", parsed_output.get("action"))
        confidence = parsed_output.get("confidence", 0.0)
        market_regime = parsed_output.get("market_regime", "unknown")

        extra = {
            "prompt_hash": prompt_hash,
            "parsed_output": parsed_output,
            "bear_veto_fired": bear_veto_fired,
            "actual_move_pct": actual_move_pct,
            "predicted_move_pct": predicted_move_pct,
            "model_name": self.model_name,
        }

        self.record({
            "timestamp": time.time(),
            "market_regime": market_regime,
            "signal_context": parsed_output,
            "action_taken": direction,
            "outcome_pnl": trade_outcome_pnl,
            "outcome_label": trade_outcome_label,
            "confidence_at_entry": confidence,
            "layer_id": self.layer_id,
            "extra_data": extra,
        })

    # ------------------------------------------------------------------
    # Pattern / Failure libraries
    # ------------------------------------------------------------------
    def get_pattern_library(self) -> list[dict]:
        """Top-20 prompt contexts that led to profitable trades, ordered by avg_pnl desc."""
        sql = """
            SELECT signal_context, AVG(outcome_pnl) AS avg_pnl, COUNT(*) AS frequency
            FROM events
            WHERE outcome_label = 'WIN'
            GROUP BY signal_context
            ORDER BY avg_pnl DESC
            LIMIT 20
        """
        try:
            with self._lock:
                rows = self._conn.execute(sql).fetchall()
            return [
                {"signal_context": json.loads(r["signal_context"]), "avg_pnl": r["avg_pnl"], "frequency": r["frequency"]}
                for r in rows
            ]
        except Exception:
            logger.exception("get_pattern_library failed for %s", self.layer_id)
            return []

    def get_failure_library(self) -> list[dict]:
        """Top-20 patterns that led to losses, ordered by avg_pnl asc."""
        sql = """
            SELECT signal_context, AVG(outcome_pnl) AS avg_pnl, COUNT(*) AS frequency
            FROM events
            WHERE outcome_label = 'LOSS'
            GROUP BY signal_context
            ORDER BY avg_pnl ASC
            LIMIT 20
        """
        try:
            with self._lock:
                rows = self._conn.execute(sql).fetchall()
            return [
                {"signal_context": json.loads(r["signal_context"]), "avg_pnl": r["avg_pnl"], "frequency": r["frequency"]}
                for r in rows
            ]
        except Exception:
            logger.exception("get_failure_library failed for %s", self.layer_id)
            return []

    # ------------------------------------------------------------------
    # Dynamic prompt context (few-shot)
    # ------------------------------------------------------------------
    def build_dynamic_prompt_context(self, current_signal: dict) -> str:
        """Return few-shot examples: 3 similar wins + 2 similar losses.

        Similarity = match on market_regime + direction.
        """
        regime = current_signal.get("market_regime", "unknown")
        direction = current_signal.get("direction", current_signal.get("action", "unknown"))

        def _fetch(label: str, limit: int) -> list[dict]:
            sql = """
                SELECT signal_context, outcome_pnl, extra_data
                FROM events
                WHERE market_regime = ? AND action_taken = ? AND outcome_label = ?
                ORDER BY ABS(outcome_pnl) DESC
                LIMIT ?
            """
            try:
                with self._lock:
                    rows = self._conn.execute(sql, (regime, direction, label, limit)).fetchall()
                return [dict(r) for r in rows]
            except Exception:
                logger.exception("_fetch failed")
                return []

        wins = _fetch("WIN", 3)
        losses = _fetch("LOSS", 2)

        parts = []
        if wins:
            parts.append("=== SIMILAR WINNING TRADES ===")
            for i, w in enumerate(wins, 1):
                ctx = json.loads(w["signal_context"])
                extra = json.loads(w["extra_data"]) if w["extra_data"] else {}
                parts.append(
                    f"Win #{i}  |  PnL: {w['outcome_pnl']:.2f}%  |  "
                    f"Regime: {regime}  |  Direction: {direction}\n"
                    f"  Signal: {json.dumps(ctx, indent=2)}\n"
                    f"  Actual move: {extra.get('actual_move_pct', 'N/A')}%  |  "
                    f"Predicted move: {extra.get('predicted_move_pct', 'N/A')}%"
                )

        if losses:
            parts.append("\n=== SIMILAR LOSING TRADES ===")
            for i, lo in enumerate(losses, 1):
                ctx = json.loads(lo["signal_context"])
                extra = json.loads(lo["extra_data"]) if lo["extra_data"] else {}
                parts.append(
                    f"Loss #{i}  |  PnL: {lo['outcome_pnl']:.2f}%  |  "
                    f"Regime: {regime}  |  Direction: {direction}\n"
                    f"  Signal: {json.dumps(ctx, indent=2)}\n"
                    f"  Actual move: {extra.get('actual_move_pct', 'N/A')}%  |  "
                    f"Predicted move: {extra.get('predicted_move_pct', 'N/A')}%"
                )

        if not parts:
            return "No similar historical trades found for this signal context."

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Confidence calibration
    # ------------------------------------------------------------------
    def calibrate_confidence_threshold(self, min_trades: int = 20) -> float:
        """Find lowest confidence bucket where win_rate > 55%.

        Buckets: 0.0-0.5, 0.5-0.7, 0.7-0.85, 0.85-1.0.
        Returns 0.60 if insufficient data.
        """
        buckets = [
            (0.0, 0.5),
            (0.5, 0.7),
            (0.7, 0.85),
            (0.85, 1.0),
        ]
        # Map bucket lower-bound to its midpoint for the threshold value
        bucket_thresholds = {0.0: 0.0, 0.5: 0.5, 0.7: 0.7, 0.85: 0.85}

        try:
            with self._lock:
                total = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            if total < min_trades:
                return 0.60

            for lo, hi in buckets:
                with self._lock:
                    row = self._conn.execute(
                        """
                        SELECT COUNT(*) AS cnt,
                               SUM(CASE WHEN outcome_label='WIN' THEN 1 ELSE 0 END) AS wins
                        FROM events
                        WHERE confidence_at_entry >= ? AND confidence_at_entry < ?
                        """,
                        (lo, hi),
                    ).fetchone()

                cnt, wins = row["cnt"], row["wins"] or 0
                if cnt > 0 and (wins / cnt) > 0.55:
                    return bucket_thresholds[lo]

            return 0.60
        except Exception:
            logger.exception("calibrate_confidence_threshold failed for %s", self.layer_id)
            return 0.60
