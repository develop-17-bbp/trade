"""Asset-class enum + symbol-aware routing helpers.

Single source of truth for "is this BTC, AAPL, or a polymarket ID?" so
the executor + fetcher + conviction gate can dispatch off one shared
classifier instead of grepping the symbol string at every callsite.

Why a dedicated module:
    * The symbol string alone (`BTCUSDT`, `BTC`, `polymarket-…`, `SPY`) is
      ambiguous and venue-coupled. Robinhood writes `BTC`; some scripts
      write `BTCUSDT`. Without a normalizer we'd silently mis-route.
    * Leveraged ETFs (TQQQ, SOXL) need different overnight + size rules
      than 1× ETFs (SPY, QQQ). One flag here keeps the logic centralized.
    * Future polymarket / equity / options expansion lands cleanly: the
      enum grows; downstream code only checks `cls.is_stock()` etc.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional


# Hardcoded leveraged-ETF list. Keep small; expand only when operator
# explicitly approves. Source: 2026-04-29 operator basket (SPY/QQQ +
# TQQQ/SOXL "with caution").
_LEVERAGED_ETFS = frozenset({
    "TQQQ",   # 3× Nasdaq-100
    "SOXL",   # 3× Semiconductors
    "UPRO",   # 3× S&P 500 (allowed but not in active basket)
    "SQQQ",   # -3× Nasdaq-100 (inverse — short via long ETF)
    "SPXU",   # -3× S&P 500 (inverse)
    "SOXS",   # -3× Semiconductors (inverse)
})

# 1× ETFs in the active stocks basket. Anything outside this set lands
# in the SINGLE_NAME bucket which gets even tighter risk caps.
_INDEX_ETFS = frozenset({
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI",
})

# Polymarket markets are addressed by hex-ish IDs or `polymarket-<slug>`.
_POLYMARKET_PREFIXES = ("POLYMARKET-", "PM-")


class AssetClass(str, enum.Enum):
    """The five classes ACT can route. CRYPTO is the existing default."""
    CRYPTO       = "CRYPTO"
    STOCK        = "STOCK"            # any equity — 1× ETF, leveraged ETF, single name
    POLYMARKET   = "POLYMARKET"
    UNKNOWN      = "UNKNOWN"

    def is_stock(self) -> bool:
        return self is AssetClass.STOCK

    def is_crypto(self) -> bool:
        return self is AssetClass.CRYPTO


@dataclass(frozen=True)
class SymbolMeta:
    """Resolved metadata for one symbol — what the conviction gate +
    executor + finetune corpus filter all agree on."""
    symbol:           str            # canonical uppercase string ('BTC', 'SPY', …)
    asset_class:      AssetClass
    is_leveraged_etf: bool = False   # TQQQ/SOXL → tighter overnight + size rules
    is_index_etf:     bool = False   # SPY/QQQ — relax sniper threshold
    venue:            Optional[str] = None     # 'robinhood' | 'alpaca' | 'polymarket'

    def overnight_pct_max(self, default: float = 5.0) -> float:
        """Leveraged ETFs cannot be held overnight (daily-reset decay)."""
        if self.is_leveraged_etf:
            return 0.0
        return default

    def intraday_pct_max(self, default: float = 15.0) -> float:
        """Leveraged ETFs get a 5% cap regardless of conviction tier."""
        if self.is_leveraged_etf:
            return 5.0
        return default


def normalize_symbol(raw: str) -> str:
    """Canonicalize a user/api-provided symbol to upper-case, no spaces.

    Strips trailing 'USDT'/'USD' for crypto so 'BTCUSDT' and 'BTC' map
    to the same class. Leaves polymarket prefixes intact.
    """
    s = (raw or "").strip().upper().replace(" ", "")
    if not s:
        return s
    if s.startswith(_POLYMARKET_PREFIXES):
        return s
    # Strip stablecoin pair suffix for crypto-style symbols.
    for suffix in ("USDT", "USDC", "BUSD", "USD"):
        if s.endswith(suffix) and len(s) > len(suffix) + 1:
            base = s[: -len(suffix)]
            # Don't strip 'USD' from 3-letter stocks like 'AUS' — only
            # strip when leftover is ≥3 chars and recognizable as crypto.
            if len(base) >= 3 and base in _CRYPTO_BASES:
                return base
    return s


# Canonical crypto base symbols ACT trades. Extend as the universe grows.
_CRYPTO_BASES = frozenset({
    "BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "MATIC", "XRP",
    "LTC", "BCH", "DOT", "ADA", "TRX", "ATOM", "FIL", "NEAR",
    "APT", "ARB", "OP", "INJ", "TIA", "SEI",
})


def classify(raw: str, venue_hint: Optional[str] = None) -> SymbolMeta:
    """Resolve a raw symbol to its asset class + flags.

    `venue_hint` lets the caller bias classification when the symbol
    string alone is ambiguous (e.g. 'AAVE' is both a crypto token and
    a NYSE ticker; venue=robinhood → CRYPTO, venue=alpaca → STOCK).
    """
    sym = normalize_symbol(raw)
    if not sym:
        return SymbolMeta(symbol=sym, asset_class=AssetClass.UNKNOWN)

    if sym.startswith(_POLYMARKET_PREFIXES):
        return SymbolMeta(symbol=sym, asset_class=AssetClass.POLYMARKET, venue="polymarket")

    if sym in _LEVERAGED_ETFS:
        return SymbolMeta(
            symbol=sym, asset_class=AssetClass.STOCK,
            is_leveraged_etf=True, is_index_etf=False,
            venue=venue_hint or "alpaca",
        )

    if sym in _INDEX_ETFS:
        return SymbolMeta(
            symbol=sym, asset_class=AssetClass.STOCK,
            is_leveraged_etf=False, is_index_etf=True,
            venue=venue_hint or "alpaca",
        )

    if sym in _CRYPTO_BASES:
        return SymbolMeta(
            symbol=sym, asset_class=AssetClass.CRYPTO,
            venue=venue_hint or "robinhood",
        )

    # Disambiguate single-name equities (e.g. AAPL, NVDA) only if the
    # caller hints they're stocks — otherwise UNKNOWN keeps us from
    # silently routing 'XYZ' to whatever venue happens to load first.
    if venue_hint == "alpaca":
        return SymbolMeta(symbol=sym, asset_class=AssetClass.STOCK, venue="alpaca")
    if venue_hint == "robinhood":
        return SymbolMeta(symbol=sym, asset_class=AssetClass.CRYPTO, venue="robinhood")

    return SymbolMeta(symbol=sym, asset_class=AssetClass.UNKNOWN)


def is_leveraged_etf(raw: str) -> bool:
    """Convenience wrapper — True iff `raw` resolves to a leveraged ETF."""
    return classify(raw).is_leveraged_etf
