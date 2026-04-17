"""ACT v8.0 — Geopolitical risk economic data layer."""

import logging
import time

import requests

logger = logging.getLogger(__name__)


class Geopolitical:
    """Tracks geopolitical risk via GDELT tone/sentiment data."""

    def __init__(self):
        self._last_result: dict | None = None
        self._last_fetch_time: float = 0.0

    @staticmethod
    def _compute_risk_score(tone: float, event_count: int) -> float:
        """Map GDELT tone + event volume to a 0-100 risk score.

        Lower (more negative) tone and higher event counts raise the score.
        """
        # tone typically ranges roughly -10 to +10
        tone_component = max(0.0, min(50.0, (5 - tone) * 5))
        volume_component = max(0.0, min(50.0, event_count / 200.0 * 50))
        return round(min(100.0, tone_component + volume_component), 2)

    def fetch(self) -> dict:
        try:
            tone = 0.0
            event_count = 0
            source = "gdelt"

            try:
                resp = requests.get(
                    "https://api.gdeltproject.org/api/v2/summary/summary",
                    timeout=15,
                )
                if resp.status_code == 200:
                    data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                    # GDELT summary may return various shapes; extract what we can
                    if isinstance(data, dict):
                        tone = float(data.get("tone", data.get("averageTone", 0)))
                        event_count = int(data.get("eventCount", data.get("totalEvents", 0)))
                    elif isinstance(data, list) and data:
                        tone = float(data[0].get("tone", 0))
                        event_count = len(data)
            except Exception as e:
                logger.debug("GDELT fetch error: %s", e)
                source = "gdelt:fallback"

            risk_score = self._compute_risk_score(tone, event_count)

            if risk_score > 70:
                signal = "BEARISH"
                confidence = min(0.9, 0.5 + (risk_score - 70) * 0.01)
            elif risk_score < 20:
                signal = "NEUTRAL"
                confidence = 0.5
            else:
                # Weighted / cautious
                signal = "BEARISH" if risk_score > 50 else "NEUTRAL"
                confidence = 0.3 + (risk_score / 100) * 0.3

            result = {
                "value": risk_score,
                "change_pct": None,
                "signal": signal,
                "confidence": round(confidence, 3),
                "source": source,
                "stale": False,
                "detail": {"tone": tone, "event_count": event_count},
            }
            self._last_result = result
            self._last_fetch_time = time.time()
            return result

        except Exception as e:
            logger.error("Geopolitical.fetch failed: %s", e)
            if self._last_result is not None:
                return {**self._last_result, "stale": True}
            return {
                "value": None,
                "change_pct": None,
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "source": "error",
                "stale": True,
            }

    def get_cached(self) -> dict | None:
        return self._last_result

    def get_safe_haven_flow(self) -> bool:
        """Return True if a risk-off / safe-haven flow pattern is detected."""
        r = self._last_result
        if r is None:
            return False
        score = r.get("value")
        if score is not None and score > 60:
            return True
        return False
