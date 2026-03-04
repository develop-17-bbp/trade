"""
On-chain data fetcher
======================
This module provides a simple interface for retrieving on-chain metrics such as
MVRV, SOPR, whale flows, etc.  In this scaffold the methods return empty
placeholders; integration with services like Glassnode or Nansen can be added
later.
"""

from typing import Dict, Any


def fetch_metrics(symbol: str) -> Dict[str, Any]:
    """Retrieve on-chain metrics for the given symbol.

    Returns a dictionary of metric name -> value.
    """
    # placeholder - real implementation would call API clients
    return {}
