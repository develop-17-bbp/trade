"""
PHASE 5: Trading Journal
========================
Maintains a rich, persistent log of all trades for post-trade analysis.
Tracks entry/exit reasons, market regime at time of trade, and alpha decay.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JOURNAL_FILE = "logs/trading_journal.json"

class TradingJournal:
    """
    Persistent repository for trade outcomes and metadata.
    Enables 'Alpha Review' and strategy refinement.
    """
    def __init__(self):
        self._ensure_log_dir()
        self.trades = self._load_journal()

    def _ensure_log_dir(self):
        os.makedirs("logs", exist_ok=True)

    def _load_journal(self) -> List[Dict]:
        if os.path.exists(JOURNAL_FILE):
            try:
                with open(JOURNAL_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load journal: {e}")
                return []
        return []

    def log_trade(self, 
                  asset: str, 
                  side: str, 
                  quantity: float, 
                  price: float, 
                  regime: str, 
                  strategy_name: str,
                  confidence: float,
                  reasoning: str,
                  order_id: str,
                  feature_vector: Optional[Dict] = None,
                  model_signal: int = 0):
        """Record trade with professional Audit/Compliance logging."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "asset": asset,
            "side": side,
            "quantity": quantity,
            "price": price,
            "regime": regime,
            "strategy": strategy_name,
            "confidence": confidence,
            "reasoning": reasoning,
            "order_id": order_id,
            "model_decision": model_signal,
            "audit_trail": {
                "feature_snapshot": feature_vector or {}, # Full vector for post-mortem
                "execution_mode": "LIVE" if "live" in reasoning.lower() else "SHADOW"
            },
            "status": "OPEN",
            "pnl": 0.0,
            "exit_price": None,
            "exit_time": None
        }
        self.trades.append(entry)
        self._save_journal()
        logger.info(f"Journaled {side} trade for {asset} (Order: {order_id}) - Audit Trail Saved.")

    def close_trade(self, order_id: str, exit_price: float, pnl: float):
        """Update existing trade with exit data."""
        for entry in self.trades:
            if entry["order_id"] == order_id:
                entry["status"] = "CLOSED"
                entry["exit_price"] = exit_price
                entry["exit_time"] = datetime.now().isoformat()
                entry["pnl"] = pnl
                # Calculate R-Multiple
                try:
                    entry["return_pct"] = (exit_price - entry["price"]) / entry["price"] if entry["side"] == "buy" else (entry["price"] - exit_price) / entry["price"]
                except ZeroDivisionError:
                    entry["return_pct"] = 0.0
                break
        self._save_journal()

    def _save_journal(self):
        try:
            with open(JOURNAL_FILE, 'w') as f:
                json.dump(self.trades, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save journal: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Generate high-level stats for the dashboard."""
        closed_trades = [t for t in self.trades if t["status"] == "CLOSED"]
        if not closed_trades:
            return {"total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0}
        
        wins = [t for t in closed_trades if t["pnl"] > 0]
        return {
            "total_trades": len(self.trades),
            "closed_trades": len(closed_trades),
            "win_rate": len(wins) / len(closed_trades),
            "total_pnl": sum(t["pnl"] for t in closed_trades),
            "avg_return": sum(t.get("return_pct", 0) for t in closed_trades) / len(closed_trades)
        }

    def get_recent_trades(self, limit: int = 5) -> List[Dict]:
        """Get the most recent trades for context in LLM analysis."""
        return self.trades[-limit:] if self.trades else []
