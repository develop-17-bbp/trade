"""
Health monitor for external APIs
================================
Checks third-party services and returns status codes.  Used by executor to
throttle or disable layers based on availability.
"""

from typing import Dict


def check_services() -> Dict[str, bool]:
    """Return a map of service name to up/down status."""
    # placeholder assumes everything is up
    return {
        'binance': True,
        'coinapi': True,
        'newsapi': True,
        'cryptopanic': True,
    }
