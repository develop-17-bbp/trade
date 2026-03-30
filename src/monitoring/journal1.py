"""Trade Journal - logs every trade to JSONL file."""

import json
import os
import time
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

JOURNAL_PATH = os.path.join("logs", "trading_journal.jsonl")


class TradeJournal:
    def __init__(self, path: str = JOURNAL_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    def log_trade(self, trade: dict, llm_reasoning: str = "", confidence: float = 0.0,
                  order_type: str = "market"):
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "asset": trade.get("asset", ""),
            "action": trade.get("side", "").upper(),
            "entry_price": trade.get("entry_price", 0),
            "exit_price": trade.get("exit_price", 0),
            "qty": trade.get("qty", 0),
            "pnl_usd": trade.get("pnl_usd", 0),
            "pnl_pct": trade.get("pnl_pct", 0),
            "sl_progression": trade.get("sl_progression", ""),
            "exit_reason": trade.get("exit_reason", ""),
            "llm_reasoning": llm_reasoning[:200] if llm_reasoning else "",
            "confidence": confidence,
            "order_type": order_type,
            "duration_minutes": trade.get("duration_minutes", 0),
            "order_id": trade.get("order_id", ""),
        }
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            logger.info(f"Journaled {entry['action']} trade for {entry['asset']} "
                        f"(Order: {entry['order_id']}) - Audit Trail Saved.")
        except Exception as e:
            logger.error(f"Journal write failed: {e}")

    def load_trades(self, asset: Optional[str] = None) -> List[dict]:
        trades = []
        if not os.path.exists(self.path):
            return trades
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        t = json.loads(line)
                        if asset is None or t.get("asset") == asset:
                            trades.append(t)
                    except json.JSONDecodeError:
                        pass
        return trades

    def print_summary(self):
        trades = self.load_trades()
        if not trades:
            print("No trades recorded yet.")
            return
        pnls = [t.get("pnl_usd", 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total = sum(pnls)
        wr = len(wins) / len(pnls) * 100 if pnls else 0
        pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")

        print(f"Total trades: {len(pnls)}")
        print(f"Win rate: {wr:.0f}%")
        print(f"Total P&L: ${total:+,.2f}")
        if wins:
            print(f"Avg win: ${sum(wins)/len(wins):,.2f}")
        if losses:
            print(f"Avg loss: ${sum(losses)/len(losses):,.2f}")
        print(f"Profit factor: {pf:.2f}")
        if pnls:
            print(f"Best trade: ${max(pnls):+,.2f}")
            print(f"Worst trade: ${min(pnls):+,.2f}")
