"""
Data Validation Layer
=====================
Validates API responses from Binance, news feeds, and on-chain sources
before they enter the trading pipeline. Prevents garbage-in-garbage-out
from malformed, stale, or manipulated API responses.
"""

import logging
import math
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class OHLCVValidator:
    """Validates OHLCV candle data from exchange APIs."""

    # Absolute sanity bounds per asset (updated dynamically)
    PRICE_BOUNDS = {
        'BTC': (1_000, 500_000),
        'ETH': (50, 50_000),
        'AAVE': (1, 5_000),
    }

    @classmethod
    def validate(cls, ohlcv: Dict, asset: str = "BTC") -> Tuple[bool, List[str]]:
        """
        Validate OHLCV data dict with keys: opens, highs, lows, closes, volumes.
        Returns (is_valid, list_of_warnings).
        """
        warnings = []

        if not ohlcv or not isinstance(ohlcv, dict):
            return False, ["OHLCV data is empty or not a dict"]

        required_keys = ['opens', 'highs', 'lows', 'closes', 'volumes']
        for k in required_keys:
            if k not in ohlcv or not ohlcv[k]:
                return False, [f"Missing or empty OHLCV key: {k}"]

        closes = ohlcv['closes']
        highs = ohlcv['highs']
        lows = ohlcv['lows']
        opens = ohlcv['opens']
        volumes = ohlcv['volumes']

        # Length consistency
        lengths = [len(opens), len(highs), len(lows), len(closes), len(volumes)]
        if len(set(lengths)) > 1:
            warnings.append(f"OHLCV array length mismatch: {lengths}")

        # Minimum data points
        if len(closes) < 10:
            return False, [f"Insufficient data: {len(closes)} candles (need >= 10)"]

        # Price sanity
        lo, hi = cls.PRICE_BOUNDS.get(asset, (0.001, 10_000_000))
        latest = closes[-1]
        if latest <= 0:
            return False, [f"Invalid price: {latest} <= 0"]
        if latest < lo or latest > hi:
            warnings.append(f"Price {latest} outside expected range [{lo}, {hi}] for {asset}")

        # NaN / Inf check
        for name, arr in [('closes', closes), ('volumes', volumes)]:
            for i, v in enumerate(arr[-20:]):  # Check last 20 only for performance
                if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                    return False, [f"NaN/Inf found in {name}[{len(arr)-20+i}]"]

        # High >= Low check
        for i in range(max(0, len(highs) - 20), len(highs)):
            if highs[i] < lows[i]:
                warnings.append(f"Candle {i}: high ({highs[i]}) < low ({lows[i]})")

        # Volume non-negative
        if any(v < 0 for v in volumes[-20:]):
            warnings.append("Negative volume detected")

        # Staleness: check if last N candles have identical close
        if len(closes) >= 5 and len(set(closes[-5:])) == 1:
            warnings.append("Last 5 candles have identical close — possible stale data")

        is_valid = True
        if warnings:
            logger.warning(f"[DataValidation] {asset} OHLCV warnings: {warnings}")

        return is_valid, warnings


class SentimentValidator:
    """Validates news/sentiment API responses."""

    @classmethod
    def validate_headlines(cls, headlines: List[str], timestamps: List,
                           sources: List[str]) -> Tuple[bool, List[str]]:
        """Validate news headline data."""
        warnings = []

        if not headlines:
            warnings.append("No headlines returned")
            return True, warnings  # Empty is valid, just noted

        # Type check
        if not all(isinstance(h, str) for h in headlines):
            return False, ["Non-string headline detected"]

        # Reasonable length
        if any(len(h) > 2000 for h in headlines):
            warnings.append("Headline exceeds 2000 chars — possible data corruption")

        # Length consistency
        if len(headlines) != len(timestamps) or len(headlines) != len(sources):
            warnings.append(
                f"Array length mismatch: headlines={len(headlines)}, "
                f"timestamps={len(timestamps)}, sources={len(sources)}"
            )

        return True, warnings

    @classmethod
    def validate_sentiment_score(cls, score: float) -> float:
        """Clamp sentiment score to valid range [-1.0, 1.0]."""
        if score is None or (isinstance(score, float) and math.isnan(score)):
            return 0.0
        return max(-1.0, min(1.0, float(score)))


class OnChainValidator:
    """Validates on-chain data."""

    @classmethod
    def validate(cls, data: Dict) -> Tuple[bool, List[str]]:
        """Validate on-chain signal data."""
        warnings = []

        if not data or not isinstance(data, dict):
            return True, ["Empty on-chain data"]  # Optional source, empty is OK

        # Check for NaN/Inf in numeric fields
        for key, val in data.items():
            if isinstance(val, float):
                if math.isnan(val) or math.isinf(val):
                    data[key] = 0.0
                    warnings.append(f"NaN/Inf in on-chain field '{key}', replaced with 0.0")

        return True, warnings
