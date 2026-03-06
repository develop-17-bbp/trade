import time
import random
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from src.execution.router import ExecutionRouter, ExecutionMode

logger = logging.getLogger(__name__)

class VWAPEngine:
    """
    Volume-Weighted Average Price (VWAP) Execution Engine.
    Executes in proportion to market volume profile.
    """
    def __init__(self, router: ExecutionRouter, 
                 participation_rate: float = 0.1, 
                 max_window_mins: int = 120):
        self.router = router
        self.participation_rate = participation_rate
        self.max_window = max_window_mins
        self.active_orders = {}

    def schedule_vwap(self, symbol: str, side: str, total_qty: float) -> str:
        """
        Creates a VWAP execution schedule.
        """
        order_id = f"VWAP_{int(time.time())}_{symbol.replace('/', '_')}"
        
        self.active_orders[order_id] = {
            "symbol": symbol,
            "side": side,
            "total_qty": total_qty,
            "completed_qty": 0.0,
            "start_time": time.time(),
            "status": "active"
        }
        
        logger.info(f"[VWAP] Active for {total_qty} {symbol} (Participation: {self.participation_rate*100}%).")
        return order_id

    def process_tick(self, order_id: str, current_market_volume: float):
        """
        Execute based on actual volume recently observed since last tick.
        """
        order = self.active_orders.get(order_id)
        if not order or order['status'] != 'active': return

        # Target size = market_volume * participation_rate
        # Participation target: Execute 10% of what the market is doing
        target_qty = current_market_volume * self.participation_rate
        remaining = order['total_qty'] - order['completed_qty']
        
        if target_qty > 0:
            exec_qty = min(target_qty, remaining)
            
            # Slippage control: Don't execute if target is too small for exchange mins
            if exec_qty < 0.0001: return

            res = self.router.execute_order(
                symbol=order['symbol'],
                side=order['side'],
                quantity=exec_qty,
                order_type="market"
            )
            
            if res.success:
                order['completed_qty'] += exec_qty
                logger.debug(f"[VWAP-TICK] Executed {exec_qty:.4f}")
                
                if order['completed_qty'] >= order['total_qty']:
                    order['status'] = 'completed'
                    logger.info(f"[VWAP-COMPLETE] {order['symbol']} filled.")
            else:
                logger.error(f"[VWAP-ERROR] Slice fail: {res.error_message}")
