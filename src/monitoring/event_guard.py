"""
Market Event Guard — Dynamic Calendar-Based Trading Pause
==========================================================
Guards the bot against high-risk periods using:
  1. Manual event list (user-configurable)
  2. Auto-detected recurring events (FOMC, CPI, NFP first Fridays)
  3. Exchange status monitoring

Pauses or adapts trading during HIGH risk windows (±2 hours of event).
"""

import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Recurring US macro events (auto-generated each month)
RECURRING_EVENTS = [
    {'name': 'FOMC Rate Decision', 'day_of_month': None, 'weekday': 2, 'week': 3, 'risk': 'HIGH'},  # 3rd Wednesday
    {'name': 'CPI Inflation Data', 'day_of_month': None, 'weekday': 2, 'week': 2, 'risk': 'HIGH'},   # 2nd Wednesday
    {'name': 'NFP Jobs Report', 'day_of_month': None, 'weekday': 4, 'week': 1, 'risk': 'HIGH'},      # 1st Friday
]

# Known upcoming events (update periodically)
STATIC_EVENTS = [
    {'date': '2025-04-16', 'name': 'FOMC Minutes Release', 'risk': 'MEDIUM'},
    {'date': '2025-04-30', 'name': 'FOMC Rate Decision', 'risk': 'HIGH'},
    {'date': '2028-03-15', 'name': 'Bitcoin Halving (est.)', 'risk': 'HIGH'},
]


class MarketEventGuard:
    """
    Guards the bot against extreme market events.
    Auto-generates recurring macro event dates + accepts manual events.
    """

    def __init__(self, custom_events: Optional[List[Dict]] = None,
                 risk_window_hours: float = 2.0):
        self.risk_window_hours = risk_window_hours
        self.paused = False

        # Build event calendar: static + recurring + custom
        self.events: List[Dict] = []
        self.events.extend(STATIC_EVENTS)
        if custom_events:
            self.events.extend(custom_events)

        # Auto-generate recurring events for current + next 2 months
        self._generate_recurring_events()
        logger.info(f"[EVENT-GUARD] Loaded {len(self.events)} events (including auto-generated)")

    def _generate_recurring_events(self):
        """Auto-generate recurring macro event dates for current + next 2 months."""
        now = datetime.now()
        for month_offset in range(3):
            year = now.year
            month = now.month + month_offset
            if month > 12:
                month -= 12
                year += 1

            for event_template in RECURRING_EVENTS:
                try:
                    target_weekday = event_template['weekday']  # 0=Mon, 4=Fri
                    target_week = event_template['week']  # 1st, 2nd, 3rd occurrence

                    # Find Nth weekday of month
                    first_day = datetime(year, month, 1)
                    count = 0
                    for day in range(1, 32):
                        try:
                            d = datetime(year, month, day)
                        except ValueError:
                            break
                        if d.weekday() == target_weekday:
                            count += 1
                            if count == target_week:
                                self.events.append({
                                    'date': d.strftime('%Y-%m-%d'),
                                    'name': event_template['name'],
                                    'risk': event_template['risk'],
                                    'auto_generated': True,
                                })
                                break
                except Exception:
                    continue

    def is_risk_high(self) -> bool:
        """Check if current time is within risk window of a HIGH risk event."""
        now = datetime.now()
        for event in self.events:
            if event.get('risk', 'LOW') not in ('HIGH', 'MEDIUM'):
                continue
            try:
                event_date = datetime.strptime(event['date'], '%Y-%m-%d')
                # Check if within risk window (hours before/after event date)
                diff_hours = abs((now - event_date).total_seconds()) / 3600
                if diff_hours <= self.risk_window_hours:
                    logger.warning(f"[EVENT-GUARD] HIGH RISK: {event['name']} on {event['date']} "
                                   f"({diff_hours:.1f}h away)")
                    return True
            except Exception:
                continue
        return False

    def upcoming_events(self, days_ahead: int = 7) -> List[Dict]:
        """Return events in the next N days."""
        now = datetime.now()
        cutoff = now + timedelta(days=days_ahead)
        upcoming = []
        for event in self.events:
            try:
                event_date = datetime.strptime(event['date'], '%Y-%m-%d')
                if now <= event_date <= cutoff:
                    upcoming.append(event)
            except Exception:
                continue
        return sorted(upcoming, key=lambda e: e['date'])

    def check_exchange_status(self, exchange_response: Dict) -> bool:
        """Detect exchange outages or maintenance."""
        if not exchange_response:
            return False
        if exchange_response.get('status') in ('maintenance', 'outage'):
            return False
        return True

    def add_event(self, date: str, name: str, risk: str = 'HIGH'):
        """Add a custom event at runtime."""
        self.events.append({'date': date, 'name': name, 'risk': risk})
        logger.info(f"[EVENT-GUARD] Added event: {name} on {date} ({risk})")

    def toggle_pause(self, manual: Optional[bool] = None):
        """Manually or automatically pause trading."""
        if manual is not None:
            self.paused = manual
        else:
            self.paused = not self.paused
        logger.info(f"[EVENT-GUARD] Bot Status: {'PAUSED' if self.paused else 'ACTIVE'}")
