"""
Dynamic Position Limits — ACT v8.0
All position sizes, leverage, and limits are LEARNED from memory.

- Max position size adjusts based on Monte Carlo P(ruin) + Sharpe
- Leverage scales with regime confidence + win streak
- Per-asset limits learned from historical performance
- Everything self-adjusts — no hardcoded caps

Memory-driven sizing hierarchy:
  1. Monte Carlo P(ruin) sets absolute max risk budget
  2. Sharpe mode scales within that budget
  3. Regime confidence scales further
  4. Per-asset win rate adjusts per-asset allocation
  5. Leverage = f(regime_confidence, streak, volatility)
"""
import time
import math
import logging
from typing import Dict, Optional
from collections import deque

logger = logging.getLogger(__name__)


class DynamicPositionLimits:
    """Learns optimal position limits from trade history and market conditions."""

    def __init__(self, config: dict = None, accuracy_engine=None,
                 sharpe_optimizer=None, monte_carlo=None):
        cfg = config or {}
        self._accuracy = accuracy_engine
        self._sharpe = sharpe_optimizer
        self._monte_carlo = monte_carlo

        # Safety floor/ceiling (absolute bounds — never exceeded)
        self._absolute_min_size_pct = cfg.get('absolute_min_size_pct', 0.5)
        self._absolute_max_size_pct = cfg.get('absolute_max_size_pct', 25.0)
        self._absolute_max_leverage = cfg.get('absolute_max_leverage', 5.0)
        self._absolute_min_leverage = 1.0  # never below 1x

        # Per-asset tracking
        self._asset_outcomes: Dict[str, deque] = {}  # asset → [pnl_pct, ...]
        self._asset_max_size: Dict[str, float] = {}  # learned max per asset

        # Regime-based leverage memory
        self._regime_leverage_outcomes: Dict[str, deque] = {}  # regime → [(leverage, pnl), ...]

        # Current equity for notional calcs
        self._equity = cfg.get('initial_equity', 16000)

        # Monte Carlo recommended max risk
        self._mc_max_risk_pct = cfg.get('default_risk_per_trade_pct', 1.0)

    def update_equity(self, equity: float):
        self._equity = equity

    def record_trade(self, asset: str, size_pct: float, leverage: float,
                     pnl_pct: float, regime: str = 'unknown'):
        """Record outcome for dynamic limit learning."""
        self._asset_outcomes.setdefault(asset, deque(maxlen=200)).append(pnl_pct)
        self._regime_leverage_outcomes.setdefault(regime, deque(maxlen=200)).append(
            (leverage, pnl_pct)
        )

    def update_from_monte_carlo(self, mc_result: dict):
        """Update max risk from Monte Carlo P(ruin) analysis."""
        if not mc_result or 'error' in mc_result:
            return
        p_ruin = mc_result.get('probability_of_ruin', 0.5)
        # Low P(ruin) → can take more risk. High → reduce.
        if p_ruin < 0.01:
            self._mc_max_risk_pct = 2.0
        elif p_ruin < 0.05:
            self._mc_max_risk_pct = 1.5
        elif p_ruin < 0.10:
            self._mc_max_risk_pct = 1.0
        elif p_ruin < 0.20:
            self._mc_max_risk_pct = 0.75
        else:
            self._mc_max_risk_pct = 0.5
        logger.info(f"[LIMITS] Monte Carlo P(ruin)={p_ruin:.1%} → max_risk={self._mc_max_risk_pct}%")

    def get_max_position_pct(self, asset: str, regime: str = 'unknown',
                              regime_confidence: float = 0.5) -> float:
        """Dynamic max position size as % of equity. Learned from:
        - Monte Carlo risk budget
        - Sharpe mode (recovery → small, momentum → larger)
        - Per-asset historical win rate
        - Regime confidence
        """
        # Start with Monte Carlo budget
        base = self._mc_max_risk_pct

        # Sharpe mode multiplier
        sharpe_mult = 1.0
        if self._sharpe:
            mode = self._sharpe.mode
            if mode == 'RECOVERY':
                sharpe_mult = 0.5
            elif mode == 'MOMENTUM':
                sharpe_mult = 1.3

        # Per-asset win rate multiplier
        asset_mult = 1.0
        outcomes = list(self._asset_outcomes.get(asset, deque()))
        if len(outcomes) >= 10:
            win_rate = sum(1 for p in outcomes[-30:] if p > 0) / len(outcomes[-30:])
            # Win rate 0.3 → 0.6x, 0.5 → 1.0x, 0.7 → 1.4x
            asset_mult = max(0.4, min(1.5, win_rate * 2.0))

        # Regime confidence multiplier
        regime_mult = max(0.5, min(1.3, regime_confidence * 1.5))

        # Accuracy engine streak
        streak_mult = 1.0
        if self._accuracy:
            streak_mult = self._accuracy.get_position_size_multiplier()

        raw_pct = base * sharpe_mult * asset_mult * regime_mult * streak_mult

        # Scale up for larger accounts (more room for diversification)
        if self._equity > 50000:
            equity_scale = min(3.0, self._equity / 50000)
        elif self._equity > 10000:
            equity_scale = 1.0
        else:
            equity_scale = max(2.0, 10000 / max(self._equity, 1000))  # small accounts need bigger %

        final_pct = raw_pct * equity_scale

        # Clamp to absolute bounds
        final_pct = max(self._absolute_min_size_pct, min(self._absolute_max_size_pct, final_pct))

        return round(final_pct, 2)

    def get_max_position_usd(self, asset: str, regime: str = 'unknown',
                              regime_confidence: float = 0.5) -> float:
        """Max position in USD (notional)."""
        pct = self.get_max_position_pct(asset, regime, regime_confidence)
        return round(self._equity * (pct / 100.0), 2)

    def get_optimal_leverage(self, asset: str, regime: str = 'unknown',
                              regime_confidence: float = 0.5,
                              volatility: float = 0.15,
                              exchange: str = 'robinhood') -> float:
        """Dynamic leverage based on regime, volatility, and track record.

        Robinhood: always 1x (no leverage on crypto)
        Bybit/Delta: 1x-5x based on conditions
        """
        # Robinhood = no leverage ever
        if exchange.lower() in ('robinhood', 'alpaca'):
            return 1.0

        # Base leverage from regime confidence
        # High confidence BULL → more leverage, BEAR/CRISIS → reduce
        if regime.upper() in ('CRISIS', 'BEAR'):
            base_lev = 1.0
        elif regime_confidence > 0.8:
            base_lev = 3.0
        elif regime_confidence > 0.6:
            base_lev = 2.0
        else:
            base_lev = 1.0

        # Volatility adjustment: high vol → less leverage
        vol_mult = max(0.5, min(1.5, 0.15 / max(volatility, 0.01)))

        # Learn from past leverage outcomes
        regime_data = list(self._regime_leverage_outcomes.get(regime, deque()))
        if len(regime_data) >= 10:
            # Find leverage level with best avg PnL
            lev_buckets = {}
            for lev, pnl in regime_data[-50:]:
                bucket = round(lev)
                lev_buckets.setdefault(bucket, []).append(pnl)
            best_lev = 1.0
            best_avg = -999
            for lev, pnls in lev_buckets.items():
                avg = sum(pnls) / len(pnls)
                if avg > best_avg:
                    best_avg = avg
                    best_lev = float(lev)
            # Blend: 70% calculated + 30% learned
            base_lev = base_lev * 0.7 + best_lev * 0.3

        # Sharpe adjustment
        if self._sharpe and self._sharpe.mode == 'RECOVERY':
            base_lev = min(base_lev, 1.0)  # no leverage in recovery

        # Win streak bonus (small)
        if self._accuracy:
            if (self._accuracy._streak_type == 'winning' and
                self._accuracy._streak_length >= 5):
                base_lev *= 1.1

        final_lev = base_lev * vol_mult
        final_lev = max(self._absolute_min_leverage,
                       min(self._absolute_max_leverage, final_lev))

        return round(final_lev, 1)

    def get_per_asset_allocation(self, assets: list, regime: str = 'unknown') -> Dict[str, float]:
        """Dynamic allocation across assets based on per-asset performance.
        Better-performing assets get more allocation."""
        if not assets:
            return {}

        raw_weights = {}
        for asset in assets:
            outcomes = list(self._asset_outcomes.get(asset, deque()))
            if len(outcomes) >= 5:
                win_rate = sum(1 for p in outcomes[-20:] if p > 0) / len(outcomes[-20:])
                avg_pnl = sum(outcomes[-20:]) / len(outcomes[-20:])
                # Weight = win_rate * (1 + avg_pnl/5)
                raw_weights[asset] = max(0.1, win_rate * (1 + avg_pnl / 5))
            else:
                raw_weights[asset] = 1.0  # equal weight until enough data

        total = sum(raw_weights.values())
        return {a: round(w / total, 3) for a, w in raw_weights.items()}

    def should_use_leverage(self, exchange: str, regime: str,
                             regime_confidence: float) -> bool:
        """Should we use leverage at all? Conservative gate."""
        if exchange.lower() in ('robinhood', 'alpaca'):
            return False
        if regime.upper() in ('CRISIS', 'BEAR'):
            return False
        if self._sharpe and self._sharpe.mode == 'RECOVERY':
            return False
        if regime_confidence < 0.6:
            return False
        # Check if leverage has been profitable historically
        data = list(self._regime_leverage_outcomes.get(regime, deque()))
        leveraged = [(lev, pnl) for lev, pnl in data if lev > 1.0]
        if len(leveraged) >= 10:
            avg = sum(pnl for _, pnl in leveraged) / len(leveraged)
            if avg < 0:
                return False  # leverage has been losing money → don't use it
        return True

    def get_stats(self) -> dict:
        asset_stats = {}
        for asset, outcomes in self._asset_outcomes.items():
            recent = list(outcomes)[-20:]
            if recent:
                asset_stats[asset] = {
                    'trades': len(recent),
                    'win_rate': round(sum(1 for p in recent if p > 0) / len(recent), 2),
                    'avg_pnl': round(sum(recent) / len(recent), 3),
                }
        return {
            'mc_max_risk_pct': self._mc_max_risk_pct,
            'equity': self._equity,
            'asset_performance': asset_stats,
            'absolute_max_leverage': self._absolute_max_leverage,
        }
