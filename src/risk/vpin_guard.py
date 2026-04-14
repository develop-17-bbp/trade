import numpy as np
import logging
from typing import List, Dict, Any, Optional
# Note: scipy.stats removed — norm was unused and hangs on Python 3.14

logger = logging.getLogger(__name__)

class VPINGuard:
    """
    Adverse Selection Protection: Volume-synchronous Probability of Informed Trading.
    Detects 'toxic' order flow that precedes predatory HFT aggression.
    
    If VPIN crosses threshold, it triggers MetaController.reduce_position_size().
    """
    def __init__(self, bucket_size: float = 1.0, 
                 window_buckets: int = 50, 
                 threshold: float = 0.7):
        self.bucket_size = bucket_size # Volume per bucket (e.g. 1.0 BTC)
        self.window = window_buckets
        self.threshold = threshold
        
        # State: Volume buckets
        self.buckets = [] # Each element: abs(BuyVol - SellVol) in that bucket
        self.current_buy_vol = 0.0
        self.current_sell_vol = 0.0

    def add_trade(self, price: float, volume: float, side: str):
        """
        Record a trade and updates the VPIN calculation.
        """
        if side == 'buy':
            self.current_buy_vol += volume
        else:
            self.current_sell_vol += volume
            
        # Check if bucket is full
        if (self.current_buy_vol + self.current_sell_vol) >= self.bucket_size:
            # Bucket is complete
            imbalance = abs(self.current_buy_vol - self.current_sell_vol)
            self.buckets.append(imbalance)
            
            # Keep window size
            if len(self.buckets) > self.window:
                self.buckets.pop(0)
                
            # Reset counters
            self.current_buy_vol = 0.0
            self.current_sell_vol = 0.0

    def calculate_vpin(self) -> float:
        """
        VPIN = sum(abs(V_buy - V_sell)) / (n * V_bucket)
        """
        if len(self.buckets) < self.window:
            return 0.5 # Neutral fallback

        vpin = sum(self.buckets) / (self.window * self.bucket_size)
        return min(1.0, vpin)

    def is_flow_toxic(self) -> Dict[str, Any]:
        """
        Institutional Risk Guard: Logic for blocking trades during adverse flow.
        """
        vpin = self.calculate_vpin()
        is_toxic = vpin > self.threshold
        
        # Dynamic threshold based on volatility (Optional extension)
        # toxicity_limit = norm.ppf(0.95, loc=np.mean(self.buckets), scale=np.std(self.buckets))
        
        status = {
            "vpin": vpin,
            "threshold": self.threshold,
            "is_toxic": is_toxic,
            "risk_action": "REDUCE" if vpin > (self.threshold * 0.8) else ("BLOCK" if is_toxic else "ALLOW"),
            "engine": "VPIN_v2"
        }
        
        if is_toxic:
            logger.warning(f"[VPIN-TOXIC] Detected informed flow aggression: {vpin:.2f} (Threshold: {self.threshold})")
            
        return status
