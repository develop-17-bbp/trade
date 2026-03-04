"""
Portfolio optimizer module
==========================
Provides functions for calculating optimal asset weights using Markowitz,
risk-parity or Kelly-based algorithms.  This is an extensible scaffolding;
clients can call `optimize_weights` with a dataframe of returns and receive a
weight vector.
"""

from typing import List, Dict
import numpy as np


def optimize_weights(returns: np.ndarray, method: str = "markowitz") -> np.ndarray:
    """Return portfolio weights for given return series.

    Args:
        returns: 2D array of shape (n_periods, n_assets)
        method: one of 'markowitz', 'risk_parity', 'kelly'

    Returns:
        1D weight array of length n_assets that sums to 1.0
    """
    n_assets = returns.shape[1]
    if method == "markowitz":
        # simple equal weights placeholder
        return np.ones(n_assets) / n_assets
    elif method == "risk_parity":
        return np.ones(n_assets) / n_assets
    elif method == "kelly":
        return np.ones(n_assets) / n_assets
    else:
        raise ValueError(f"Unknown optimization method: {method}")
