"""
PHASE 5: Hedging Mechanisms
===========================
Provides cross-market hedging, delta-neutral adjustments, and tail risk protection
during extreme market volatility or drawdown events.
"""

from typing import Dict, List, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class HedgingAction:
    asset: str
    action_type: str  # e.g., 'HEDGE_SHORT', 'REDUCE_EXPOSURE', 'TAIL_RISK_PROTECTION'
    hedge_ratio: float
    reason: str

class PortfolioHedger:
    """
    Manages portfolio hedging and tail-risk protection.
    """
    def __init__(self, risk_manager: Any = None):
        self.risk_manager = risk_manager
        self.hedge_correlation_threshold = 0.8  # Assets correlated > 0.8 trigger hedging
        self.tail_risk_volatility_threshold = 0.08  # 8% intraday vol implies tail risk
        
    def calculate_hedges(self, 
                       allocations: Dict[str, Any], 
                       volatilities: Dict[str, float], 
                       regimes: Dict[str, Any],
                       onchain_metrics: Dict[str, Any]) -> List[HedgingAction]:
        """
        Evaluate current allocations and market conditions to propose hedging actions.
        """
        hedges = []
        
        # 1. Check for Tail Risk (Extreme Volatility or Liquidation Cascades)
        global_tail_risk = False
        avg_vol = sum(volatilities.values()) / max(len(volatilities), 1)
        
        if avg_vol > self.tail_risk_volatility_threshold:
            global_tail_risk = True
            logger.warning(f"[HEDGER] Global Tail Risk Detected (Avg Volatility: {avg_vol:.2%})")
            
        # Check onchain liquidation risks
        high_liquidation_risk = False
        for asset, metrics in onchain_metrics.items():
            if metrics.get('liquidation_risk_score', 0) > 0.8:
                high_liquidation_risk = True
                logger.warning(f"[HEDGER] High On-Chain Liquidation Risk Detected for {asset}")
                
        # 2. Determine Action per Asset
        for asset, allocation in allocations.items():
            regime = regimes.get(asset, None)
            regime_type = regime.regime_type if regime else "UNKNOWN"
            
            # Action logic
            if global_tail_risk or high_liquidation_risk:
                # Delta-neutral or heavy reduction
                hedges.append(HedgingAction(
                    asset=asset,
                    action_type='TAIL_RISK_PROTECTION',
                    hedge_ratio=0.8,  # Hedge out 80% of exposure
                    reason="Extreme volatility or liquidation cascade imminent."
                ))
            elif regime_type == "VOLATILE" and allocation.position_size_pct > 0.1:
                # Partial hedge in volatile markets if heavily allocated
                hedges.append(HedgingAction(
                    asset=asset,
                    action_type='VOLATILITY_HEDGE',
                    hedge_ratio=0.5,  # Hedge out 50%
                    reason="Asset regime is VOLATILE, reducing large localized exposure."
                ))
            elif regime_type == "BEARISH" and allocation.position_size_pct > 0.0:
                 # Fully hedge or short against long exposure
                 hedges.append(HedgingAction(
                    asset=asset,
                    action_type='BEAR_MARKET_HEDGE',
                    hedge_ratio=1.0,  # Fully hedge (100%)
                    reason="Asset regime is BEARISH, neutralizing long exposure."
                ))
                
        return hedges
