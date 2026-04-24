"""Chart-image vision adapter — stub for FinAgent-style multimodal.

Reference: arXiv:2402.18485 — "A Multimodal Foundation Agent for
Financial Trading" (FinAgent, Feb 2024). They feed Kline (candlestick)
chart *images* to the brain alongside numerical + textual data,
reporting a 36% average profit improvement vs text-only baselines.

ACT today feeds text-only context (price numbers, news, sentiment) to
the analyst. This module is the hook for a future chart-image modality:

  1. `render_chart_png(asset, timeframe, bars)` — produces a PNG
     screenshot of the latest chart via matplotlib.
  2. `summarize_chart(png_bytes)` — sends the image to a local
     vision-LLM (via Ollama's vision endpoint) and returns a short
     prose summary, ready to inject into the EvidenceDocument.

**Current status: STUB.** The code is here, but the vision-model call
falls back to an empty summary unless a vision model is pulled and
`ACT_CHART_VISION_MODEL` is set. This keeps the import path safe
without forcing the operator to pull an extra 8-12GB model.

When the operator is ready:
  1. `ollama pull llama3.2-vision:11b` (or similar)
  2. `setx ACT_CHART_VISION_MODEL llama3.2-vision:11b`
  3. Restart ACT. The analyst's seed context will now include a
     "CHART_VISION" section.

Kill switch: `ACT_DISABLE_CHART_VISION=1`.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


DISABLE_ENV = "ACT_DISABLE_CHART_VISION"
MODEL_ENV = "ACT_CHART_VISION_MODEL"


def is_enabled() -> bool:
    """Vision is off unless a model is explicitly set. Kill switch also
    wins."""
    if (os.environ.get(DISABLE_ENV) or "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    return bool((os.environ.get(MODEL_ENV) or "").strip())


def render_chart_png(asset: str, timeframe: str = "1h", bars: int = 100) -> Optional[bytes]:
    """Render a PNG chart for `asset` as bytes.

    Best-effort: missing matplotlib or missing data → None. Safe to
    call even when matplotlib isn't installed — just returns None.
    """
    try:
        import io
        import matplotlib
        matplotlib.use("Agg")   # headless
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.debug("chart_vision: matplotlib unavailable: %s", e)
        return None

    try:
        from src.data.fetcher import get_recent_bars
        data = get_recent_bars(asset.upper(), timeframe=timeframe, limit=bars)
    except Exception as e:
        logger.debug("chart_vision: data fetch failed: %s", e)
        return None

    if not data:
        return None

    try:
        closes = [float(b.get("close") or 0.0) for b in data]
        highs = [float(b.get("high") or 0.0) for b in data]
        lows = [float(b.get("low") or 0.0) for b in data]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(closes, label=f"{asset} close", linewidth=1.2)
        ax.fill_between(range(len(closes)), lows, highs, alpha=0.2)
        ax.set_title(f"{asset} {timeframe} — last {len(closes)} bars")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=96)
        plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        logger.debug("chart_vision: render failed: %s", e)
        return None


def summarize_chart(png_bytes: Optional[bytes], asset: str = "") -> str:
    """Send the PNG to the configured vision model, return a short
    prose summary. Empty string if disabled / no model / failure."""
    if not is_enabled() or not png_bytes:
        return ""
    model = (os.environ.get(MODEL_ENV) or "").strip()
    if not model:
        return ""

    prompt = (
        f"Describe the {asset or 'crypto'} chart in 2-3 sentences: "
        "recent trend direction, any obvious pattern "
        "(breakout/consolidation/reversal), and where the price "
        "currently sits relative to recent highs/lows. Be specific, "
        "avoid hedging language."
    )

    try:
        import requests
        b64 = base64.b64encode(png_bytes).decode("ascii")
        base = os.environ.get("OLLAMA_BASE_URL") or "http://127.0.0.1:11434"
        resp = requests.post(
            f"{base.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [b64],
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=60,
        )
        if resp.status_code != 200:
            return ""
        data = resp.json() or {}
        return str(data.get("response") or "").strip()
    except Exception as e:
        logger.debug("chart_vision: inference failed: %s", e)
        return ""


def chart_summary_section(asset: str, timeframe: str = "1h") -> Dict[str, Any]:
    """One-call helper for context_builders — renders chart, gets
    vision summary, returns a dict ready for EvidenceSection."""
    if not is_enabled():
        return {"content": "", "confidence": 0.0, "source": "chart_vision(disabled)"}
    png = render_chart_png(asset, timeframe=timeframe)
    if png is None:
        return {"content": "", "confidence": 0.0, "source": "chart_vision(render_failed)"}
    summary = summarize_chart(png, asset=asset)
    if not summary:
        return {"content": "", "confidence": 0.0, "source": "chart_vision(no_response)"}
    return {
        "content": summary,
        "confidence": 0.55,   # vision-model verdicts — medium trust
        "source": f"chart_vision({os.environ.get(MODEL_ENV, 'unknown')})",
    }
