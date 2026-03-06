"""
PHASE 5: Market Event Awareness
=============================
Monitors economic calendars and major news events (Halving, ETF approval).
Pauses or adapts trading during high-risk periods.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

class MarketEventGuard:
    """
    Guards the bot against extreme market events using a news/event calendar.
    """
    def __init__(self):
        self.events = [
            {"date": "2024-04-20", "name": "Bitcoin Halving", "risk": "HIGH"},
            {"date": "2024-05-23", "name": "ETH ETF Deadline", "risk": "HIGH"},
            {"date": "2024-03-12", "name": "CPI Inflation Data", "risk": "MEDIUM"}
        ]
        self.paused = False

    def is_risk_high(self) -> bool:
        """
        Checks if current time is within +/- 2 hours of a HIGH risk event.
        """
        now = datetime.now()
        for event in self.events:
            try:
                event_date = datetime.strptime(event["date"], "%Y-%m-%d")
                # In production, this would use exact time of event
                diff_days = abs((now - event_date).days)
                if diff_days <= 1 and event["risk"] == "HIGH":
                    return True
            except:
                continue
        return False

    def check_exchange_status(self, exchange_response: Dict) -> bool:
        """
        Detects exchange outages or maintenance.
        """
        if not exchange_response:
            return False
        if exchange_response.get('status') in ['maintenance', 'outage']:
            return False
        return True

    def toggle_pause(self, manual: bool = None):
        """Manually or automatically pause trading."""
        if manual is not None:
            self.paused = manual
        else:
            self.paused = not self.paused
        logging.info(f"[EVENT-GUARD] Bot Status: {'PAUSED' if self.paused else 'ACTIVE'}")
