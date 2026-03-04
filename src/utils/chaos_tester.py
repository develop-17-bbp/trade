"""
Chaos tester
============
Utility to randomly disable components (e.g., Robinhood connector) to verify
failover logic.  This file defines helper functions used in tests or debug
scripts.
"""

import random


def maybe_disable(component: str, probability: float = 0.1) -> bool:
    """Return True if the named component should be disabled during this run."""
    return random.random() < probability
