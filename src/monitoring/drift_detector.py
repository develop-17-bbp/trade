"""
Drift detector service
======================
Monitors feature distributions and model outputs; alerts when statistics drift
beyond configured bounds.  This stub has a no-op implementation.
"""

from typing import Dict, Any


def check_drift(stats: Dict[str, Any]) -> bool:
    """Return True if drift detected, False otherwise."""
    # placeholder always returns False
    return False
