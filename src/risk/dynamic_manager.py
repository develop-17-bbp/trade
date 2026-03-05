"""
PHASE 5: Dynamic Risk Manager
==============================
Circuit breakers, drawdown halts, and dynamic risk limits for autonomous trading.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RiskLimits:
    """Dynamic risk limits that adjust based on market conditions."""
    max_daily_loss_pct: float = 0.02  # 2% max daily loss
    max_weekly_loss_pct: float = 0.05  # 5% max weekly loss
    max_monthly_loss_pct: float = 0.10  # 10% max monthly loss
    max_single_trade_pct: float = 0.01  # 1% max per trade
    max_portfolio_heat_pct: float = 0.15  # 15% max portfolio concentration
    volatility_multiplier: float = 1.0  # Adjusts based on market vol

@dataclass
class CircuitBreaker:
    """Circuit breaker state and thresholds."""
    name: str
    threshold: float
    current_value: float
    is_triggered: bool
    last_triggered: Optional[str] = None
    cooldown_minutes: int = 60

class DynamicRiskManager:
    """
    Advanced risk management with circuit breakers and dynamic limits.
    Implements drawdown halts and volatility-adjusted position sizing.
    """

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # Risk state tracking
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        self.portfolio_heat = 0.0  # Current portfolio risk exposure

        # Circuit breakers
        self.circuit_breakers = self._initialize_circuit_breakers()

        # Risk limits (dynamic)
        self.risk_limits = RiskLimits()

        # Historical tracking
        self.pnl_history = []
        self.trade_history = []

        # Recovery mode
        self.recovery_mode = False
        self.recovery_multiplier = 0.5  # Reduce risk by 50% in recovery

    def _initialize_circuit_breakers(self) -> Dict[str, CircuitBreaker]:
        """Initialize circuit breaker thresholds."""
        return {
            "daily_loss": CircuitBreaker(
                name="Daily Loss Limit",
                threshold=0.02,  # 2%
                current_value=0.0,
                is_triggered=False,
                cooldown_minutes=1440  # 24 hours
            ),
            "weekly_loss": CircuitBreaker(
                name="Weekly Loss Limit",
                threshold=0.05,  # 5%
                current_value=0.0,
                is_triggered=False,
                cooldown_minutes=10080  # 7 days
            ),
            "volatility_spike": CircuitBreaker(
                name="Volatility Spike",
                threshold=0.08,  # 8% daily vol
                current_value=0.0,
                is_triggered=False,
                cooldown_minutes=60
            ),
            "correlation_breakdown": CircuitBreaker(
                name="Correlation Breakdown",
                threshold=0.9,  # 90% correlation
                current_value=0.0,
                is_triggered=False,
                cooldown_minutes=120
            ),
            "liquidation_risk": CircuitBreaker(
                name="High Liquidation Risk",
                threshold=0.7,  # 70% liquidation risk
                current_value=0.0,
                is_triggered=False,
                cooldown_minutes=30
            )
        }

    def update_pnl(self, pnl_change: float, timestamp: Optional[str] = None):
        """
        Update P&L tracking and check circuit breakers.

        Args:
            pnl_change: P&L change in USD
            timestamp: ISO timestamp string
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        # Update P&L
        self.daily_pnl += pnl_change
        self.weekly_pnl += pnl_change
        self.monthly_pnl += pnl_change

        # Store in history
        self.pnl_history.append({
            "pnl": pnl_change,
            "timestamp": timestamp,
            "daily_total": self.daily_pnl,
            "weekly_total": self.weekly_pnl,
            "monthly_total": self.monthly_pnl
        })

        # Update circuit breakers
        self._update_circuit_breakers()

        # Check for recovery mode
        self._check_recovery_mode()

    def _update_circuit_breakers(self):
        """Update circuit breaker states based on current conditions."""
        # Daily loss breaker
        daily_loss_pct = abs(self.daily_pnl) / self.current_capital
        self.circuit_breakers["daily_loss"].current_value = daily_loss_pct

        # Weekly loss breaker
        weekly_loss_pct = abs(self.weekly_pnl) / self.current_capital
        self.circuit_breakers["weekly_loss"].current_value = weekly_loss_pct

        # Check if any breakers are triggered
        for breaker in self.circuit_breakers.values():
            was_triggered = breaker.is_triggered

            if breaker.name in ["daily_loss", "weekly_loss"]:
                # Loss-based breakers trigger when threshold exceeded
                breaker.is_triggered = breaker.current_value >= breaker.threshold
            elif breaker.name == "volatility_spike":
                # This would be updated from market data
                pass
            elif breaker.name == "correlation_breakdown":
                # This would be updated from correlation analysis
                pass
            elif breaker.name == "liquidation_risk":
                # This would be updated from on-chain data
                pass

            # Log if newly triggered
            if breaker.is_triggered and not was_triggered:
                breaker.last_triggered = datetime.now().isoformat()
                logger.warning(f"Circuit breaker triggered: {breaker.name} "
                             f"(value: {breaker.current_value:.3f}, threshold: {breaker.threshold:.3f})")

    def _check_recovery_mode(self):
        """Check if system should enter recovery mode."""
        # Enter recovery if recent losses exceed thresholds
        recent_pnl = sum(p["pnl"] for p in self.pnl_history[-10:])  # Last 10 trades
        recent_loss_pct = abs(recent_pnl) / self.current_capital

        # Exit recovery if recent performance improves
        if self.recovery_mode and recent_pnl > 0:
            self.recovery_mode = False
            logger.info("Exiting recovery mode - performance improved")

        # Enter recovery if recent losses are significant
        elif not self.recovery_mode and recent_loss_pct > 0.03:  # 3% recent loss
            self.recovery_mode = True
            logger.warning("Entering recovery mode - reducing risk exposure")

    def update_market_conditions(self, volatility: float,
                               correlations: Dict[Tuple[str, str], float],
                               liquidation_risk: float):
        """
        Update risk limits based on current market conditions.

        Args:
            volatility: Current market volatility
            correlations: Asset correlation matrix
            liquidation_risk: Average liquidation risk across positions
        """
        # Update volatility-based circuit breaker
        self.circuit_breakers["volatility_spike"].current_value = volatility

        # Update correlation breaker (use max correlation)
        max_corr = max(correlations.values()) if correlations else 0.0
        self.circuit_breakers["correlation_breakdown"].current_value = max_corr

        # Update liquidation risk breaker
        self.circuit_breakers["liquidation_risk"].current_value = liquidation_risk

        # Adjust risk limits based on conditions
        self._adjust_risk_limits(volatility, max_corr, liquidation_risk)

    def _adjust_risk_limits(self, volatility: float, max_correlation: float, liquidation_risk: float):
        """Dynamically adjust risk limits based on market conditions."""

        # Base multiplier from volatility
        vol_multiplier = 1.0
        if volatility > 0.05:  # High volatility
            vol_multiplier = 0.7  # Reduce risk by 30%
        elif volatility < 0.02:  # Low volatility
            vol_multiplier = 1.2  # Increase risk by 20%

        # Correlation adjustment
        corr_multiplier = 1.0
        if max_correlation > 0.8:  # High correlation
            corr_multiplier = 0.8  # Reduce risk by 20%

        # Liquidation risk adjustment
        liq_multiplier = 1.0
        if liquidation_risk > 0.6:  # High liquidation risk
            liq_multiplier = 0.6  # Reduce risk by 40%

        # Recovery mode adjustment
        recovery_multiplier = self.recovery_multiplier if self.recovery_mode else 1.0

        # Combined adjustment
        total_multiplier = vol_multiplier * corr_multiplier * liq_multiplier * recovery_multiplier

        # Apply to risk limits
        self.risk_limits.max_daily_loss_pct = 0.02 * total_multiplier
        self.risk_limits.max_single_trade_pct = 0.01 * total_multiplier
        self.risk_limits.volatility_multiplier = total_multiplier

    def check_trade_allowed(self, asset: str, proposed_size_pct: float,
                          current_portfolio_heat: float) -> Tuple[bool, str]:
        """
        Check if a proposed trade is allowed based on current risk limits.

        Args:
            asset: Asset symbol
            proposed_size_pct: Proposed position size as % of capital
            current_portfolio_heat: Current total portfolio exposure

        Returns:
            (allowed: bool, reason: str)
        """
        # Check circuit breakers
        active_breakers = [b for b in self.circuit_breakers.values() if b.is_triggered]
        if active_breakers:
            breaker_names = [b.name for b in active_breakers]
            return False, f"Circuit breaker(s) triggered: {', '.join(breaker_names)}"

        # Check position size limits
        if proposed_size_pct > self.risk_limits.max_single_trade_pct:
            return False, f"Position size {proposed_size_pct:.3f} exceeds max single trade limit {self.risk_limits.max_single_trade_pct:.3f}"

        # Check portfolio heat
        new_heat = current_portfolio_heat + proposed_size_pct
        if new_heat > self.risk_limits.max_portfolio_heat_pct:
            return False, f"Portfolio heat {new_heat:.3f} would exceed max limit {self.risk_limits.max_portfolio_heat_pct:.3f}"

        # Check daily loss limit
        if self.daily_pnl < -self.risk_limits.max_daily_loss_pct * self.current_capital:
            return False, f"Daily loss limit exceeded: {self.daily_pnl:.2f}"

        return True, "Trade allowed"

    def get_risk_status(self) -> Dict[str, Any]:
        """Get comprehensive risk status report."""
        active_breakers = [b for b in self.circuit_breakers.values() if b.is_triggered]

        return {
            "recovery_mode": self.recovery_mode,
            "circuit_breakers": {
                name: {
                    "triggered": breaker.is_triggered,
                    "current_value": breaker.current_value,
                    "threshold": breaker.threshold,
                    "last_triggered": breaker.last_triggered
                }
                for name, breaker in self.circuit_breakers.items()
            },
            "risk_limits": {
                "max_daily_loss_pct": self.risk_limits.max_daily_loss_pct,
                "max_single_trade_pct": self.risk_limits.max_single_trade_pct,
                "max_portfolio_heat_pct": self.risk_limits.max_portfolio_heat_pct,
                "volatility_multiplier": self.risk_limits.volatility_multiplier
            },
            "pnl_summary": {
                "daily_pnl": self.daily_pnl,
                "weekly_pnl": self.weekly_pnl,
                "monthly_pnl": self.monthly_pnl,
                "daily_pnl_pct": self.daily_pnl / self.current_capital,
                "weekly_pnl_pct": self.weekly_pnl / self.current_capital,
                "monthly_pnl_pct": self.monthly_pnl / self.current_capital
            },
            "active_breakers": len(active_breakers),
            "timestamp": datetime.now().isoformat()
        }

    def reset_daily_pnl(self):
        """Reset daily P&L tracking (call at market open)."""
        self.daily_pnl = 0.0
        logger.info("Daily P&L reset to 0.0")

    def reset_weekly_pnl(self):
        """Reset weekly P&L tracking (call at week start)."""
        self.weekly_pnl = 0.0
        logger.info("Weekly P&L reset to 0.0")

    def reset_monthly_pnl(self):
        """Reset monthly P&L tracking (call at month start)."""
        self.monthly_pnl = 0.0
        logger.info("Monthly P&L reset to 0.0")