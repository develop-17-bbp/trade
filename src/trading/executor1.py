"""Trade Executor with Dynamic Trailing Stop-Loss (L1->L2->L3->L4)."""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Position:
    asset: str
    side: str  # "long" or "short"
    entry_price: float
    qty: float
    sl_levels: List[float] = field(default_factory=list)
    current_sl: float = 0.0
    peak_price: float = 0.0
    bars_held: int = 0
    order_id: str = ""
    entry_time: float = field(default_factory=time.time)

    def sl_str(self) -> str:
        parts = [f"L{i+1}=${s:,.2f}" for i, s in enumerate(self.sl_levels)]
        return " -> ".join(parts) if parts else "none"


class TradeExecutor:
    def __init__(self, fetcher, equity: float = 10000.0):
        self.fetcher = fetcher
        self.equity = equity
        self.initial_equity = equity
        self.peak_equity = equity
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[dict] = []
        self.daily_pnl = 0.0
        self.last_trade_time: float = 0.0
        self.last_close_times: Dict[str, float] = {}

    def passes_quality_gates(self, confidence, atr, price, asset) -> Tuple[bool, str]:
        now = time.time()
        if confidence < 0.70:
            return False, f"Confidence {confidence:.2f} < 0.70"
        if price > 0 and atr / price < 0.0003:
            return False, f"ATR/price too low"
        if now - self.last_trade_time < 60:
            return False, f"Trade cooldown: {60 - (now - self.last_trade_time):.0f}s"
        last_close = self.last_close_times.get(asset, 0)
        if now - last_close < 120:
            return False, f"Post-close cooldown: {120 - (now - last_close):.0f}s"
        return True, "OK"

    def check_risk_limits(self) -> Tuple[bool, str]:
        if self.daily_pnl < 0 and abs(self.daily_pnl) / max(self.initial_equity, 1) >= 0.03:
            return False, "Daily loss 3% hit"
        dd = (self.peak_equity - self.equity) / max(self.peak_equity, 1)
        if dd >= 0.10:
            return False, "Max drawdown 10% hit"
        return True, "OK"

    def enter_trade(self, asset, symbol, side, price, stop_loss,
                    order_type="market", limit_price=None, size_pct=5.0,
                    confidence=0.0, atr=0.0) -> Optional[Position]:
        if asset in self.positions:
            return None
        ok, reason = self.check_risk_limits()
        if not ok:
            logger.warning(f"Risk: {reason}")
            return None
        ok, reason = self.passes_quality_gates(confidence, atr, price, asset)
        if not ok:
            logger.info(f"Gate: {reason}")
            return None

        size_pct = max(2.0, min(10.0, size_pct))
        notional = self.equity * (size_pct / 100)
        qty = notional / price if price > 0 else 0
        if qty <= 0:
            return None

        exchange_side = "buy" if side == "long" else "sell"
        if order_type == "limit" and limit_price:
            order = self.fetcher.create_order(symbol, "limit", exchange_side, qty, limit_price)
        else:
            order = self.fetcher.create_order(symbol, "market", exchange_side, qty)

        if isinstance(order, dict) and "error" in order:
            logger.error(f"Order failed: {order['error']}")
            return None

        oid = order.get("id", "unknown") if isinstance(order, dict) else "unknown"
        # Bybit market orders may return None for average/filled — use fallbacks
        _avg = order.get("average") or order.get("price") or price if isinstance(order, dict) else price
        filled_price = float(_avg) if _avg is not None else price
        _filled = order.get("filled") or order.get("amount") or qty if isinstance(order, dict) else qty
        filled_qty = float(_filled) if _filled is not None else qty

        pos = Position(asset=asset, side=side, entry_price=filled_price,
                       qty=filled_qty, sl_levels=[stop_loss], current_sl=stop_loss,
                       peak_price=filled_price, order_id=oid)
        self.positions[asset] = pos
        self.last_trade_time = time.time()
        logger.info(f"ENTER {side.upper()} {asset} @ ${filled_price:,.2f} qty={filled_qty:.6f} SL L1=${stop_loss:,.2f}")
        return pos

    def update_trailing_sl(self, asset, current_price, swing_points=None):
        pos = self.positions.get(asset)
        if not pos:
            return
        pos.bars_held += 1

        if pos.side == "long":
            if current_price > pos.peak_price:
                pos.peak_price = current_price
            profit = pos.peak_price - pos.entry_price
            giveback_sl = pos.peak_price - profit * 0.30 if profit > 0 else pos.current_sl
            swing_sl = pos.current_sl
            if swing_points:
                valid = [s for s in swing_points if pos.current_sl < s < current_price]
                if valid:
                    swing_sl = max(valid)
            new_sl = max(giveback_sl, swing_sl)
            if new_sl > pos.current_sl:
                pos.current_sl = new_sl
                pos.sl_levels.append(new_sl)
                logger.info(f"SL {asset} LONG: L{len(pos.sl_levels)}=${new_sl:,.2f}")

        elif pos.side == "short":
            if current_price < pos.peak_price:
                pos.peak_price = current_price
            profit = pos.entry_price - pos.peak_price
            giveback_sl = pos.peak_price + profit * 0.30 if profit > 0 else pos.current_sl
            swing_sl = pos.current_sl
            if swing_points:
                valid = [s for s in swing_points if current_price < s < pos.current_sl]
                if valid:
                    swing_sl = min(valid)
            new_sl = min(giveback_sl, swing_sl)
            if new_sl < pos.current_sl:
                pos.current_sl = new_sl
                pos.sl_levels.append(new_sl)
                logger.info(f"SL {asset} SHORT: L{len(pos.sl_levels)}=${new_sl:,.2f}")

    def check_sl_hit(self, asset, price) -> bool:
        pos = self.positions.get(asset)
        if not pos:
            return False
        if pos.side == "long":
            return price <= pos.current_sl
        return price >= pos.current_sl

    def close_position(self, asset, symbol, price, reason="manual"):
        pos = self.positions.get(asset)
        if not pos:
            return None
        close_side = "sell" if pos.side == "long" else "buy"
        self.fetcher.create_order(symbol, "market", close_side, pos.qty, params={"reduceOnly": True})

        if pos.side == "long":
            pnl = (price - pos.entry_price) * pos.qty
        else:
            pnl = (pos.entry_price - price) * pos.qty

        pnl_pct = pnl / (pos.entry_price * pos.qty) * 100 if pos.entry_price * pos.qty > 0 else 0
        duration = (time.time() - pos.entry_time) / 60

        record = {
            "asset": asset, "side": pos.side,
            "entry_price": pos.entry_price, "exit_price": price,
            "qty": pos.qty, "pnl_usd": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
            "sl_progression": pos.sl_str(), "exit_reason": reason,
            "bars_held": pos.bars_held, "duration_minutes": round(duration, 1),
            "order_id": pos.order_id,
        }
        self.closed_trades.append(record)
        self.equity += pnl
        self.daily_pnl += pnl
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        self.last_close_times[asset] = time.time()
        del self.positions[asset]

        win = "WIN" if pnl > 0 else "LOSS"
        logger.info(f"EXIT {pos.side.upper()} {asset} @ ${price:,.2f} PnL=${pnl:+,.2f} ({win}) reason={reason} {pos.sl_str()}")
        return record

    def get_summary(self) -> dict:
        if not self.closed_trades:
            return {"total": 0, "equity": f"${self.equity:,.2f}"}
        pnls = [t["pnl_usd"] for t in self.closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        return {
            "total": len(pnls), "wins": len(wins), "losses": len(losses),
            "win_rate": f"{len(wins)/len(pnls)*100:.0f}%",
            "total_pnl": f"${sum(pnls):+,.2f}",
            "avg_win": f"${sum(wins)/len(wins):,.2f}" if wins else "$0",
            "avg_loss": f"${sum(losses)/len(losses):,.2f}" if losses else "$0",
            "profit_factor": f"{abs(sum(wins)/sum(losses)):.2f}" if losses and sum(losses) != 0 else "inf",
            "equity": f"${self.equity:,.2f}",
        }
