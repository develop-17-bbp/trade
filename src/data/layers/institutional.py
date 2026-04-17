"""Layer 8 — Bank & Institutional Announcements"""
import time, logging, requests
from datetime import datetime, timezone
logger = logging.getLogger(__name__)

class Institutional:
    def __init__(self):
        self._last_result = None
        self._last_fetch = 0
        self._events = []

    def fetch(self) -> dict:
        try:
            r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
            events = r.json() if r.status_code == 200 else []
            high_impact = [e for e in events if e.get('impact', '').lower() == 'high']
            self._events = high_impact
            upcoming = len(high_impact)
            signal = 'BEARISH' if upcoming > 3 else 'NEUTRAL'
            conf = min(1.0, upcoming / 5)
            self._last_result = {'value': upcoming, 'change_pct': 0, 'signal': signal,
                                 'confidence': round(conf, 2), 'source': 'forexfactory', 'stale': False,
                                 'high_impact_events': upcoming}
            self._last_fetch = time.time()
        except Exception as e:
            logger.warning(f"[INSTITUTIONAL] fetch failed: {e}")
            if self._last_result:
                self._last_result['stale'] = True
            else:
                self._last_result = {'value': 0, 'change_pct': 0, 'signal': 'NEUTRAL', 'confidence': 0, 'source': 'forexfactory', 'stale': True}
        return self._last_result

    def get_cached(self):
        return self._last_result or {'signal': 'NEUTRAL', 'confidence': 0, 'stale': True}

    def get_event_buffer_window(self) -> bool:
        """Returns True if high-impact event in next 2 hours."""
        now = datetime.now(timezone.utc)
        for ev in self._events:
            try:
                date_str = ev.get('date', '')
                ev_time = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                diff_hours = (ev_time - now).total_seconds() / 3600
                if 0 < diff_hours < 2:
                    return True
            except Exception:
                continue
        return False

    def get_post_event_drift_signal(self):
        return 'NEUTRAL'
