"""
Trade Journal
=============
Logs every trade to a JSON Lines file for analysis and auditing.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Default journal path relative to project root
DEFAULT_JOURNAL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs", "trading_journal.jsonl"
)


class TradeJournal:
    """Append-only trade journal backed by a JSONL file."""

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath or DEFAULT_JOURNAL_PATH
        self._ensure_directory()

    def _ensure_directory(self):
        """Create the parent directory if it does not exist."""
        parent = os.path.dirname(os.path.abspath(self.filepath))
        os.makedirs(parent, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log_trade(
        self,
        asset: str,
        action: str,
        entry_price: float,
        exit_price: float,
        qty: float,
        pnl_usd: float,
        pnl_pct: float,
        sl_progression: str = "",
        exit_reason: str = "",
        llm_reasoning: str = "",
        confidence: float = 0.0,
        order_type: str = "market",
        duration_minutes: float = 0.0,
        order_id: str = "",
        exchange: str = "",
        extra: Optional[Dict] = None,
    ) -> dict:
        """Append a single trade record as one JSON line."""
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "asset": asset,
            "exchange": exchange,
            "action": action,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "qty": qty,
            "pnl_usd": round(pnl_usd, 2),
            "pnl_pct": round(pnl_pct, 4),
            "sl_progression": sl_progression,
            "exit_reason": exit_reason,
            "llm_reasoning": llm_reasoning,
            "confidence": round(confidence, 4),
            "order_type": order_type,
            "duration_minutes": round(duration_minutes, 2),
            "order_id": order_id,
        }

        if extra:
            record.update(extra)

        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            logger.info("Journal: logged %s trade for %s (PnL $%.2f)", action, asset, pnl_usd)
        except Exception as e:
            logger.error("Journal write failed: %s", e)

        return record

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_trades(self, asset: Optional[str] = None, exchange: Optional[str] = None) -> List[dict]:
        """
        Read all trade records from the JSONL file.
        Optionally filter by asset and/or exchange.
        """
        trades: List[dict] = []

        if not os.path.exists(self.filepath):
            logger.info("Journal file does not exist yet: %s", self.filepath)
            return trades

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if asset is not None and record.get("asset") != asset:
                            continue
                        if exchange is not None and record.get("exchange", "") != "":
                            # Only filter if record HAS exchange tag (backward compat with old entries)
                            if record.get("exchange") != exchange:
                                continue
                        trades.append(record)
                    except json.JSONDecodeError as e:
                        logger.warning("Skipping malformed line %d: %s", lineno, e)
        except Exception as e:
            logger.error("Failed to read journal: %s", e)

        logger.info("Loaded %d trades from journal", len(trades))
        return trades

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_summary(self, asset: Optional[str] = None):
        """Print performance statistics to console."""
        trades = self.load_trades(asset=asset)
        if not trades:
            print("No trades recorded yet.")
            return

        total = len(trades)
        wins = [t for t in trades if t.get("pnl_usd", 0) > 0]
        losses = [t for t in trades if t.get("pnl_usd", 0) <= 0]

        total_pnl = sum(t.get("pnl_usd", 0) for t in trades)
        total_win_pnl = sum(t.get("pnl_usd", 0) for t in wins) if wins else 0
        total_loss_pnl = abs(sum(t.get("pnl_usd", 0) for t in losses)) if losses else 0

        win_rate = len(wins) / total * 100 if total > 0 else 0
        profit_factor = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float("inf")
        avg_win = total_win_pnl / len(wins) if wins else 0
        avg_loss = total_loss_pnl / len(losses) if losses else 0
        avg_duration = (
            sum(t.get("duration_minutes", 0) for t in trades) / total if total > 0 else 0
        )

        header = "Trade Journal Summary" + (f" ({asset})" if asset else "")
        print(f"\n{'=' * 50}")
        print(f"  {header}")
        print(f"{'=' * 50}")
        print(f"  Total trades:    {total}")
        print(f"  Wins / Losses:   {len(wins)} / {len(losses)}")
        print(f"  Win rate:        {win_rate:.1f}%")
        print(f"  Profit factor:   {profit_factor:.2f}")
        print(f"  Total PnL:       ${total_pnl:+.2f}")
        print(f"  Avg win:         ${avg_win:.2f}")
        print(f"  Avg loss:        ${avg_loss:.2f}")
        print(f"  Avg duration:    {avg_duration:.1f} min")
        print(f"{'=' * 50}\n")
