"""
PHASE 5: Portfolio Allocation Engine
=====================================
Automated capital allocation with Kelly sizing and on-chain confidence weighting.
Implements risk-parity allocation across multiple assets.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AllocationResult:
    """Result of portfolio allocation calculation."""
    asset: str
    position_size_pct: float  # % of total capital
    position_size_usd: float
    kelly_fraction: float
    onchain_confidence: float
    risk_adjusted_weight: float
    dca_progress: float = 1.0  # 1.0 = full size, < 1.0 = partial DCA leg
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class PortfolioAllocator:
    """
    Advanced portfolio allocation engine with Kelly sizing and on-chain integration.
    """

    def __init__(self, total_capital: float = 100_000.0, max_allocation_pct: float = 0.05):
        """
        Initialize portfolio allocator.

        Args:
            total_capital: Total portfolio capital in USD
            max_allocation_pct: Maximum allocation per asset (5% = 0.05)
        """
        self.total_capital = total_capital
        self.max_allocation_pct = max_allocation_pct
        self.min_allocation_pct = 0.005  # 0.5% minimum per asset

        # Kelly sizing parameters
        self.kelly_fraction = 0.5  # Conservative Kelly (half-Kelly)
        self.max_kelly_leverage = 2.0  # Maximum leverage multiplier

        # Risk parameters
        self.max_portfolio_risk_pct = 0.02  # 2% max portfolio risk per trade
        self.correlation_penalty = 0.3  # Reduce allocation for correlated assets
        
        # DCA (Dollar Cost Averaging) settings
        self.dca_enabled = True
        self.dca_legs = 3  # Scale in over 3 separate orders
        self.dca_interval_multiplier = 1.1  # Increase size slightly as we go deeper

    def calculate_dca_size(self, base_size_pct: float, current_leg: int) -> float:
        """
        Calculate size for a specific DCA leg.
        
        Args:
            base_size_pct: Total target position size
            current_leg: The current entry number (1, 2, 3...)
        """
        if not self.dca_enabled or self.dca_legs <= 1:
            return base_size_pct
            
        # Distribute size across legs (e.g., 33%, 33%, 33%)
        leg_size = base_size_pct / self.dca_legs
        
        # Apply a multiplier if we want to "load up" as the trade progresses
        return leg_size * (self.dca_interval_multiplier ** (current_leg - 1))

    def calculate_kelly_size(self, win_rate: float, win_loss_ratio: float,
                           onchain_confidence: float = 0.5) -> float:
        """
        Calculate Kelly criterion position size with on-chain confidence adjustment.

        Kelly Formula: f = (bp - q) / b
        Where: b = odds (win_loss_ratio), p = win probability, q = loss probability

        Args:
            win_rate: Historical win rate (0.0-1.0)
            win_loss_ratio: Average win/loss ratio
            onchain_confidence: On-chain confidence boost (0.0-1.0)

        Returns:
            Kelly fraction (0.0-1.0)
        """
        if win_rate <= 0 or win_rate >= 1 or win_loss_ratio <= 0:
            return 0.0

        # Standard Kelly formula
        b = win_loss_ratio
        p = win_rate
        q = 1 - p

        kelly_f = (b * p - q) / b

        # Adjust for on-chain confidence (boost positive Kelly, reduce negative)
        if kelly_f > 0:
            confidence_boost = 1.0 + (onchain_confidence - 0.5) * 0.5  # ±25% adjustment
            kelly_f *= confidence_boost
        else:
            # Reduce negative Kelly (losses) with high confidence
            confidence_reduction = 1.0 - (onchain_confidence - 0.5) * 0.3  # ±15% adjustment
            kelly_f *= confidence_reduction

        # Apply conservative fraction and clamp
        kelly_f *= self.kelly_fraction
        kelly_f = max(-0.1, min(kelly_f, self.max_kelly_leverage))  # Clamp between -10% and max leverage

        return kelly_f

    def calculate_risk_parity_weights(self, assets: List[str],
                                    volatilities: Dict[str, float],
                                    correlations: Optional[Dict[Tuple[str, str], float]] = None) -> Dict[str, float]:
        """
        Calculate risk-parity weights across assets.

        Args:
            assets: List of asset symbols
            volatilities: Asset volatilities
            correlations: Asset correlation matrix

        Returns:
            Risk-parity weights summing to 1.0
        """
        if not assets:
            return {}

        # Risk parity: equal risk contribution from each asset
        weights = {}

        for asset in assets:
            vol = volatilities.get(asset, 0.03)  # Default 3% volatility
            if vol > 0:
                # Inverse volatility weighting (higher vol = lower weight)
                weights[asset] = 1.0 / vol
            else:
                weights[asset] = 0.0

        # Apply correlation penalty
        if correlations:
            for i, asset1 in enumerate(assets):
                for asset2 in assets[i+1:]:
                    corr = correlations.get((asset1, asset2), 0.0)
                    if abs(corr) > 0.7:  # High correlation
                        # Reduce weights for highly correlated pairs
                        weights[asset1] *= (1.0 - self.correlation_penalty)
                        weights[asset2] *= (1.0 - self.correlation_penalty)

        # Normalize to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {asset: w / total_weight for asset, w in weights.items()}

        return weights

    def allocate_strategies(self, strategies: List[str],
                            alpha_scores: Dict[str, float],
                            regime: str,
                            performance_history: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Dynamically allocates capital BETWEEN different alpha strategies.
        
        Args:
            strategies: List of strategy names (e.g. 'trend', 'mean_reversion')
            alpha_scores: Current predictive edge for each strategy
            regime: Current market regime (e.g. 'TRENDING', 'RANGING')
            performance_history: Last 20 trades' returns for each strategy
        """
        weights = {}
        
        for s in strategies:
            score = alpha_scores.get(s, 0.5)
            
            # 1. Regime Alignment (Multiplier)
            regime_mult = 1.0
            if regime == 'TRENDING' and 'trend' in s.lower(): regime_mult = 1.3
            if regime == 'RANGING' and 'reversion' in s.lower(): regime_mult = 1.3
            if regime == 'PANIC': regime_mult = 0.5 # De-risk all
            
            # 2. Performance Decay Adjustment
            history = performance_history.get(s, [0.0])
            recent_perf = np.mean(history[-10:]) if len(history) >= 10 else 0.0
            decay_adj = 1.0 + (recent_perf * 2.0) # Boost if winning, cut if losing
            
            weights[s] = score * regime_mult * max(0.2, decay_adj)
            
        # Normalize weights
        total = sum(weights.values()) or 1.0
        return {s: w / total for s, w in weights.items()}

    def allocate_portfolio(self, assets: List[str],
                          performance_metrics: Dict[str, Dict[str, float]],
                          onchain_data: Optional[Dict[str, Dict[str, Any]]] = None,
                          current_positions: Optional[Dict[str, float]] = None,
                          regime: str = 'NORMAL') -> Dict[str, AllocationResult]:
        """
        Advanced allocation using Strategy Weighting + Asset Kelly sizing.
        """
        # (This would be calling the new allocate_strategies internally in production)
        # For now, we enhance the existing loop with regime awareness.
        allocations = {}
        
        regime_risk_limit = self.max_allocation_pct
        if regime == 'VOLATILE': regime_risk_limit *= 0.7
        elif regime == 'ILLIQUID': regime_risk_limit *= 0.5

        # Calculate base risk-parity weights
        volatilities = {asset: max(performance_metrics.get(asset, {}).get('volatility', 0.03), 0.005) for asset in assets}
        risk_parity_weights = self.calculate_risk_parity_weights(assets, volatilities)

        for asset in assets:
            metrics = performance_metrics.get(asset, {})
            win_rate = metrics.get('win_rate', 0.5)
            win_loss_ratio = metrics.get('avg_win_loss_ratio', 1.0)

            # Get on-chain confidence
            onchain_confidence = 0.5
            if onchain_data and asset in onchain_data:
                whale_score = onchain_data[asset].get('whale_score', 0.0)
                momentum = onchain_data[asset].get('on_chain_momentum', 0.0)
                confidence = onchain_data[asset].get('confidence', 60.0) / 100.0
                onchain_confidence = (whale_score + momentum + confidence) / 3.0

            kelly_fraction = self.calculate_kelly_size(win_rate, win_loss_ratio, onchain_confidence)
            combined_weight = kelly_fraction * risk_parity_weights.get(asset, 1.0 / len(assets))

            # Apply regime-adjusted limits
            final_weight = max(self.min_allocation_pct, min(combined_weight, regime_risk_limit))

            allocations[asset] = AllocationResult(
                asset=asset,
                position_size_pct=round(final_weight, 4),
                position_size_usd=round(final_weight * self.total_capital, 2),
                kelly_fraction=round(kelly_fraction, 4),
                onchain_confidence=round(onchain_confidence, 3),
                risk_adjusted_weight=round(final_weight, 4)
            )

        return allocations

    def get_portfolio_summary(self, allocations: Dict[str, AllocationResult]) -> Dict[str, Any]:
        """Generate portfolio summary statistics."""
        total_p = sum(a.position_size_pct for a in allocations.values())
        return {
            "total_assets": len(allocations),
            "total_allocated_pct": round(total_p, 4),
            "total_allocated_usd": round(total_p * self.total_capital, 2),
            "avg_kelly": round(np.mean([a.kelly_fraction for a in allocations.values()]), 4),
            "timestamp": datetime.now().isoformat()
        }
