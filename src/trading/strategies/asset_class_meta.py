"""Asset-class metadata for strategies.

Each strategy declares which instrument types it applies to. The LLM's
seed context filters strategy votes by venue compatibility, and
submit_trade_plan rejects mismatches.

InstrumentClass taxonomy:
    SPOT   - cash market (RH crypto, Alpaca stocks, Alpaca crypto)
    PERP   - perpetual futures (Bybit perp, future Alpaca futures)
    FUTURE - dated futures
    OPTION - listed options (Alpaca options L3 today)
    BINARY - binary options (Polymarket)

Most directional/structural strategies (Fibonacci, Gann, Elliott Wave,
Harmonic, Wyckoff, Heikin-Ashi, Ichimoku, MACD, RSI...) work on ANY
underlying chart. They produce a directional bias - how that bias
translates to a position is the venue's responsibility.

Some strategies are venue-specific:
    - Iron condor / credit spread: OPTION only (volatility selling)
    - Long straddle / strangle: OPTION only (volatility buying)
    - Funding-arbitrage: PERP + SPOT pair only
    - Delta-neutral basket: SPOT + PERP pair
"""
from __future__ import annotations

from enum import Enum
from typing import Set


class InstrumentClass(str, Enum):
    SPOT = "spot"
    PERP = "perp"
    FUTURE = "future"
    OPTION = "option"
    BINARY = "binary"


# Default - any directional strategy works on these underlying chart types
ANY_DIRECTIONAL: Set[InstrumentClass] = {
    InstrumentClass.SPOT,
    InstrumentClass.PERP,
    InstrumentClass.FUTURE,
    InstrumentClass.OPTION,  # options ALSO have an underlying we can read
}

# Spot venues only (no derivatives short-leg available)
SPOT_LONG_ONLY: Set[InstrumentClass] = {InstrumentClass.SPOT}

# Derivatives only (short available)
DERIVATIVES_ONLY: Set[InstrumentClass] = {
    InstrumentClass.PERP,
    InstrumentClass.FUTURE,
    InstrumentClass.OPTION,
}

# Options-specific (vol-selling structures)
OPTIONS_ONLY: Set[InstrumentClass] = {InstrumentClass.OPTION}

# Pairs-trading (need two legs)
PAIR_TRADING: Set[InstrumentClass] = {
    InstrumentClass.SPOT,
    InstrumentClass.PERP,
}


# Strategy-name to compatible-instrument mapping
# Strategies that don't appear here default to ANY_DIRECTIONAL
STRATEGY_INSTRUMENT_MAP: dict = {
    # ── Directional / structural — work on any underlying ──
    "fibonacci": ANY_DIRECTIONAL,
    "gann_angles": ANY_DIRECTIONAL,
    "gann_squares": ANY_DIRECTIONAL,
    "elliott_wave": ANY_DIRECTIONAL,
    "harmonic_patterns": ANY_DIRECTIONAL,
    "volume_profile": ANY_DIRECTIONAL,
    "wyckoff": ANY_DIRECTIONAL,
    "heikin_ashi": ANY_DIRECTIONAL,
    "ichimoku": ANY_DIRECTIONAL,
    "ict": ANY_DIRECTIONAL,
    "vwap_bounce": ANY_DIRECTIONAL,
    "order_block": ANY_DIRECTIONAL,
    "divergence": ANY_DIRECTIONAL,
    "break_retest": ANY_DIRECTIONAL,
    "ma_cross": ANY_DIRECTIONAL,
    "keltner_squeeze": ANY_DIRECTIONAL,

    # ── Pine ports — most are directional ──
    "pine_macd_hist": ANY_DIRECTIONAL,
    "pine_stochrsi": ANY_DIRECTIONAL,
    "pine_ichimoku": ANY_DIRECTIONAL,
    "pine_psar": ANY_DIRECTIONAL,
    "pine_adx_dmi": ANY_DIRECTIONAL,

    # ── Options-specific (vol structures) ──
    "iron_condor": OPTIONS_ONLY,
    "credit_spread_call": OPTIONS_ONLY,
    "credit_spread_put": OPTIONS_ONLY,
    "long_straddle": OPTIONS_ONLY,
    "long_strangle": OPTIONS_ONLY,
    "covered_call": OPTIONS_ONLY,
    "cash_secured_put": OPTIONS_ONLY,

    # ── Funding/basis (pair trading) ──
    "funding_arbitrage": PAIR_TRADING,
    "basis_arbitrage": PAIR_TRADING,

    # ── Universe-generated (U_*) inherit ANY_DIRECTIONAL ──
}


def is_strategy_compatible(
    strategy_name: str,
    instrument: InstrumentClass,
) -> bool:
    """Check if a strategy applies to the target instrument class."""
    compatible = STRATEGY_INSTRUMENT_MAP.get(strategy_name, ANY_DIRECTIONAL)
    return instrument in compatible


def venue_default_instrument(venue: str) -> InstrumentClass:
    """Map a venue name to its default instrument class.

    Used by submit_trade_plan + LLM seed context to filter strategies.
    """
    venue_lower = (venue or "").lower()
    if venue_lower == "robinhood":
        return InstrumentClass.SPOT  # RH crypto = spot long-only
    if venue_lower == "alpaca":
        return InstrumentClass.SPOT  # Alpaca stocks = spot (margin not modeled here)
    if venue_lower == "alpaca_crypto":
        return InstrumentClass.SPOT  # Alpaca crypto = spot 24/7
    if venue_lower == "alpaca_options":
        return InstrumentClass.OPTION
    if venue_lower in ("bybit_perp", "bybit"):
        return InstrumentClass.PERP
    if venue_lower in ("bybit_spot",):
        return InstrumentClass.SPOT
    if venue_lower == "polymarket":
        return InstrumentClass.BINARY
    return InstrumentClass.SPOT  # safe default


def filter_strategies_for_venue(
    strategy_names: list,
    venue: str,
) -> list:
    """Return subset of strategy_names compatible with the venue's
    instrument class. Used by LLM seed context to avoid showing
    options strategies on spot venues."""
    instr = venue_default_instrument(venue)
    return [s for s in strategy_names if is_strategy_compatible(s, instr)]
