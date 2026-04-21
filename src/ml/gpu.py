"""
Shared GPU device selection for LightGBM trainers.

The operator runs on an RTX 5090; LightGBM's OpenCL GPU backend works on the standard
`pip install lightgbm` build. The CUDA tree learner needs a recompile and is NOT
available on most prebuilt wheels, so we target `device='gpu'` (OpenCL) only.

Env overrides (highest to lowest precedence):
  ACT_LGBM_DEVICE=cpu  → force CPU (useful for CI, smoke tests)
  ACT_LGBM_DEVICE=gpu  → force GPU (will raise at train time if unavailable)
  (unset)              → auto-detect once and cache

Pass the result into any `lgb.train()` params dict:
    params = {'objective': 'binary', **lgbm_device_params(), ...}
"""
from __future__ import annotations

import os
from typing import Dict, Optional


_cached: Optional[Dict[str, str]] = None


def _probe_gpu() -> bool:
    """Run a tiny lgb.train() with device='gpu' to confirm the backend works."""
    try:
        import lightgbm as lgb
        import numpy as np
    except Exception:
        return False
    try:
        X = np.random.rand(64, 4).astype(np.float32)
        y = (X[:, 0] > 0.5).astype(int)
        ds = lgb.Dataset(X, label=y)
        lgb.train(
            {"objective": "binary", "device": "gpu", "verbose": -1},
            ds,
            num_boost_round=3,
        )
        return True
    except Exception:
        return False


def lgbm_device_params(force_refresh: bool = False) -> Dict[str, str]:
    """Return LightGBM params dict selecting GPU or CPU.

    Cached after the first call; `force_refresh=True` re-probes. The cache keeps the
    probe cost off the hot training loop — Optuna will call this for every trial.
    """
    global _cached
    if _cached is not None and not force_refresh:
        return dict(_cached)

    env = (os.environ.get("ACT_LGBM_DEVICE") or "").strip().lower()
    if env == "cpu":
        _cached = {}
        return dict(_cached)
    if env == "gpu":
        _cached = {"device": "gpu"}
        return dict(_cached)

    # Auto-detect
    if _probe_gpu():
        _cached = {"device": "gpu"}
    else:
        _cached = {}
    return dict(_cached)


def describe_device() -> str:
    """One-liner for startup logging."""
    p = lgbm_device_params()
    return p.get("device", "cpu")
