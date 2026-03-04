"""
L3 Risk Engine — Position Sizing
==================================
Mathematical position sizing models:

  Fixed Fraction:  size = E · f%
  Kelly Criterion: f* = (b·p − q) / b
  Half-Kelly:      f = f* / 2  (more conservative)
  ATR-based:       size = (E · risk%) / (k · ATR)
  Volatility-scaled: size = target_vol / realized_vol · base_size
"""

import math
from typing import Optional, Dict


def fixed_fraction(account_balance: float, risk_pct: float) -> float:
    """Fixed fractional position sizing: size = balance * (risk% / 100)"""
    if risk_pct <= 0:
        return 0.0
    return account_balance * (risk_pct / 100.0)


def kelly_criterion(win_prob: float, win_loss_ratio: float,
                     fraction: float = 1.0) -> float:
    """
    Kelly Criterion optimal fraction:
      f* = (b·p − q) / b
    where p = win probability, q = 1-p, b = win/loss ratio.

    Args:
        fraction: Kelly fraction (0.5 = half-Kelly, recommended for safety)
    Returns:
        Optimal fraction of bankroll to risk [0, 1]
    """
    if win_loss_ratio <= 0 or win_prob <= 0 or win_prob >= 1:
        return 0.0
    b = win_loss_ratio
    p = win_prob
    q = 1 - p
    k = (b * p - q) / b
    return max(0.0, k) * fraction


def half_kelly(win_prob: float, win_loss_ratio: float) -> float:
    """Half-Kelly: more conservative, reduces drawdowns by ~50%."""
    return kelly_criterion(win_prob, win_loss_ratio, fraction=0.5)


def atr_position_size(account_balance: float, atr_value: float,
                       risk_pct: float = 1.0, atr_multiplier: float = 2.0
                       ) -> float:
    """
    ATR-based position sizing:
      size = (balance * risk%) / (k * ATR)
    where k = ATR multiplier (stop distance in ATR units)

    This determines how many units to buy such that a k*ATR move against
    the position loses exactly risk_pct of the account.
    """
    if atr_value <= 0 or risk_pct <= 0:
        return 0.0
    risk_amount = account_balance * (risk_pct / 100.0)
    stop_distance = atr_multiplier * atr_value
    if stop_distance <= 0:
        return 0.0
    return risk_amount / stop_distance


def volatility_scaled_size(account_balance: float,
                            realized_vol: float,
                            target_vol: float = 0.15,
                            base_alloc_pct: float = 2.0
                            ) -> float:
    """
    Volatility-targeting position sizing:
      size = (target_vol / realized_vol) * base_size

    When vol is high → reduce position. When vol is low → increase position.
    Keeps realized portfolio volatility around target_vol.
    """
    if realized_vol <= 0:
        return account_balance * (base_alloc_pct / 100.0)
    vol_ratio = target_vol / realized_vol
    # Cap the scaling factor to avoid excessive leverage in low-vol regimes
    vol_ratio = max(0.1, min(3.0, vol_ratio))
    base_size = account_balance * (base_alloc_pct / 100.0)
    return base_size * vol_ratio


def optimal_position_size(account_balance: float,
                           win_prob: float,
                           win_loss_ratio: float,
                           atr_value: float,
                           realized_vol: float,
                           risk_pct: float = 1.0,
                           method: str = 'conservative'
                           ) -> Dict:
    """
    Compute position size using multiple methods and return the most
    conservative (smallest) for safety.

    Returns dict with each method's result and the final chosen size.
    """
    kelly_size = kelly_criterion(win_prob, win_loss_ratio, fraction=0.5)
    kelly_amount = account_balance * kelly_size

    ff_amount = fixed_fraction(account_balance, risk_pct)
    atr_amount = atr_position_size(account_balance, atr_value, risk_pct)
    vol_amount = volatility_scaled_size(account_balance, realized_vol)

    if method == 'conservative':
        # Take the minimum of all methods
        final = min(kelly_amount, ff_amount, atr_amount, vol_amount)
    elif method == 'moderate':
        # Average of Kelly and ATR
        final = (kelly_amount + atr_amount) / 2.0
    else:  # aggressive
        final = max(kelly_amount, atr_amount)

    # Hard cap at max_position_pct of account
    max_cap = account_balance * (risk_pct / 100.0) * 2  # never more than 2x risk
    final = min(final, max_cap)

    return {
        'final_size': max(0.0, final),
        'kelly_amount': kelly_amount,
        'kelly_fraction': kelly_size,
        'fixed_fraction': ff_amount,
        'atr_based': atr_amount,
        'vol_scaled': vol_amount,
        'method': method,
    }
