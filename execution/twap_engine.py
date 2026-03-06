import time
import random
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.execution.router import ExecutionRouter, ExecutionMode

logger = logging.getLogger(__name__)

class TWAPEngine:
    """
    Time-Weighted Average Price (TWAP) Execution Engine.
    Splits large BTC/ETH orders into sub-orders to minimize market impact.
    """
    def __init__(self, router: ExecutionRouter, 
                 window_minutes: int = 15, 
                 num_slices: int = 10):
        self.router = router
        self.window_mins = window_minutes
        self.num_slices = num_slices
        self.active_schedules = {}

    def schedule_twap(self, symbol: str, side: str, total_qty: float, 
                      max_params: Optional[Dict] = None) -> str:
        """
        Creates an execution schedule for a large order.
        """
        schedule_id = f"TWAP_{int(time.time())}_{symbol.replace('/', '_')}"
        
        # Determine randomized slices
        slice_qty = total_qty / self.num_slices
        interval_sec = (self.window_mins * 60) / self.num_slices
        
        self.active_schedules[schedule_id] = {
            "symbol": symbol,
            "side": side,
            "total_qty": total_qty,
            "completed_qty": 0.0,
            "slices_remaining": self.num_slices,
            "slice_size": slice_qty,
            "interval_sec": interval_sec,
            "last_slice_time": 0,
            "status": "active"
        }
        
        logger.info(f"[TWAP] Scheduled {total_qty} {symbol} over {self.window_mins} mins.")
        return schedule_id

    def process_tick(self, schedule_id: str):
        """
        Executes the next sub-order if interval is reached.
        """
        sched = self.active_schedules.get(schedule_id)
        if not sched or sched['status'] != 'active': return

        now = time.time()
        if now - sched['last_slice_time'] > sched['interval_sec']:
            # Time to execute!
            # Randomize qty ± 10% to avoid detectable execution patterns
            random_qty = sched['slice_size'] * random.uniform(0.9, 1.1)
            remaining = sched['total_qty'] - sched['completed_qty']
            exec_qty = min(random_qty, remaining)
            
            # Execute through router
            res = self.router.execute_order(
                symbol=sched['symbol'],
                side=sched['side'],
                quantity=exec_qty,
                order_type="market" # Institutional TWAP often uses markets for slices
            )
            
            if res.success:
                sched['completed_qty'] += exec_qty
                sched['slices_remaining'] -= 1
                sched['last_slice_time'] = now
                logger.info(f"  [TWAP-SLICE] Executed {exec_qty:.4f} of {sched['symbol']} ({sched['slices_remaining']} left)")
                
                if sched['slices_remaining'] <= 0 or sched['completed_qty'] >= sched['total_qty']:
                    sched['status'] = 'completed'
                    logger.info(f"  [TWAP-COMPLETE] {sched['symbol']} order fulfilled.")
            else:
                logger.error(f"  [TWAP-ERROR] Slice failed: {res.error_message}")
                # Optional: Handle retry logic or pause
