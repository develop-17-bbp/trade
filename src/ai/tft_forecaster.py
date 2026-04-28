"""Temporal Fusion Transformer (TFT) wrapper for multi-horizon forecasting.

Recent research (Lim et al. 2021 + crypto-specific 2025 papers)
identifies TFT as the most successful predictor for BTC price across
multi-horizon forecasts. Key advantages over PatchTST/LSTM:
  * Native multi-horizon (predict 1h, 4h, 24h ahead simultaneously)
  * Static covariates (asset id, venue) handled as input
  * Known-future covariates (scheduled FOMC, CPI release times)
  * Variable selection network — auto-features important inputs

This module is a wrapper with **graceful fallback** to ensure ACT
keeps working without pytorch-forecasting installed:

  * Try `pytorch-forecasting.TemporalFusionTransformer` if available
  * Else fall back to a quantile-regression linear forecaster

Anti-overfit:
  * Brain reads `method` field — knows whether output is TFT or fallback
  * TFT requires PURGED walk-forward cross-validation BEFORE promotion
    (use existing query_purged_walk_forward tool)
  * No persistence: each call retrains on the recent window (no stale
    model risk). When pytorch-forecasting model is loaded from disk
    we'll add a `last_trained_at` field for staleness detection.

Activation:
  ACT_TFT_FORECAST unset / "0"  → tool dormant, returns
                                   {"error": "tft_disabled"}
  ACT_TFT_FORECAST = "1"        → tries TFT, falls back to linear
                                   quantile if not available
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS = [1, 4, 12, 24]


@dataclass
class TFTForecast:
    method: str               # "tft" | "fallback_quantile_linear"
    asset: str
    timeframe: str
    horizons: List[int]
    point_forecasts: Dict[int, float] = field(default_factory=dict)
    quantiles: Dict[int, Dict[str, float]] = field(default_factory=dict)
    feature_importances: Dict[str, float] = field(default_factory=dict)
    last_trained_at: float = 0.0
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "asset": self.asset,
            "timeframe": self.timeframe,
            "horizons": [int(h) for h in self.horizons],
            "point_forecasts": {int(k): round(float(v), 4)
                                 for k, v in self.point_forecasts.items()},
            "quantiles": {int(h): {q: round(float(v), 4) for q, v in qs.items()}
                          for h, qs in self.quantiles.items()},
            "feature_importances": {k: round(float(v), 4)
                                     for k, v in list(self.feature_importances.items())[:10]},
            "last_trained_at": self.last_trained_at,
            "rationale": self.rationale[:300],
        }


def is_enabled() -> bool:
    val = (os.environ.get("ACT_TFT_FORECAST") or "").strip().lower()
    return val in ("1", "true", "on")


def _quantile_linear_fallback(
    closes: List[float],
    horizons: List[int],
) -> TFTForecast:
    """Quantile regression linear fallback when pytorch-forecasting
    isn't installed. Computes p10/p50/p90 forecasts using last-30-bar
    return distribution.
    """
    if len(closes) < 30:
        return TFTForecast(
            method="fallback_quantile_linear",
            asset="?", timeframe="?", horizons=horizons,
            rationale="insufficient_history",
        )
    rets = [(closes[i] / closes[i - 1]) - 1 for i in range(1, len(closes))]
    rets = rets[-100:]  # most recent 100 returns
    sorted_r = sorted(rets)
    p10 = sorted_r[int(len(sorted_r) * 0.1)]
    p50 = sorted_r[len(sorted_r) // 2]
    p90 = sorted_r[int(len(sorted_r) * 0.9)]

    last = closes[-1]
    points = {}
    quantiles = {}
    for h in horizons:
        # Per-period return raised to h periods
        points[h] = last * ((1 + p50) ** h)
        quantiles[h] = {
            "p10": last * ((1 + p10) ** h),
            "p50": last * ((1 + p50) ** h),
            "p90": last * ((1 + p90) ** h),
        }
    return TFTForecast(
        method="fallback_quantile_linear",
        asset="?", timeframe="?", horizons=horizons,
        point_forecasts=points, quantiles=quantiles,
        feature_importances={"close_lag1": 1.0},
        rationale="empirical quantile of last-100 returns",
    )


def _try_tft(
    closes: List[float],
    horizons: List[int],
    asset: str,
    timeframe: str,
) -> Optional[TFTForecast]:
    """Attempt TFT inference. Returns None on import or runtime failure
    so caller falls back to quantile linear.

    Note: A full TFT integration requires:
      * pytorch-forecasting library installed
      * trained-and-saved TFT checkpoint (one per asset+timeframe)
      * dataset construction matching the training distribution

    This wrapper is designed for the case where the operator HAS
    trained a TFT (via a separate offline training pipeline) and
    saved it to models/tft_<asset>_<timeframe>.ckpt. Inference is
    cheap (~50ms on a 5090).

    For now: only attempts the import + checkpoint check; returns
    None when checkpoint doesn't exist (caller falls back). This
    is intentionally a lightweight wrapper that activates as soon
    as the operator drops a TFT checkpoint into models/.
    """
    try:
        from pathlib import Path
        ckpt = Path(f"models/tft_{asset.lower()}_{timeframe}.ckpt")
        if not ckpt.exists():
            return None
        # Lazy import — only when checkpoint exists
        import torch
        from pytorch_forecasting import TemporalFusionTransformer
    except ImportError:
        return None
    except Exception:
        return None

    try:
        # Cache loaded model per asset+timeframe
        global _tft_model_cache
        if "_tft_model_cache" not in globals():
            _tft_model_cache = {}
        cache_key = f"{asset.lower()}_{timeframe}"
        if cache_key not in _tft_model_cache:
            model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
            model.eval()
            _tft_model_cache[cache_key] = model
        model = _tft_model_cache[cache_key]

        # Build inference dataset matching the training distribution.
        # This is a stub — a real implementation would use the same
        # TimeSeriesDataSet schema the model was trained with.
        # Without that schema we return None and let caller fall back.
        # When the operator ships a trained TFT they'll provide the
        # dataset constructor here.
        return None
    except Exception as e:
        logger.debug("tft inference failed: %s", e)
        return None


def forecast(
    asset: str,
    closes: List[float],
    timeframe: str = "1h",
    horizons: Optional[List[int]] = None,
) -> TFTForecast:
    """Multi-horizon forecast. Tries TFT, falls back to quantile linear."""
    horizons = horizons or DEFAULT_HORIZONS
    if not is_enabled():
        return TFTForecast(
            method="tft_disabled", asset=asset, timeframe=timeframe,
            horizons=horizons,
            rationale="ACT_TFT_FORECAST not set",
        )

    tft = _try_tft(closes, horizons, asset, timeframe)
    if tft is not None:
        tft.asset = asset
        tft.timeframe = timeframe
        return tft

    fb = _quantile_linear_fallback(closes, horizons)
    fb.asset = asset
    fb.timeframe = timeframe
    return fb
