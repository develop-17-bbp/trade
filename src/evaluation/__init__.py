"""Evaluation + visualization utilities for the ACT trading system.

Pure analysis layer — reads runtime artifacts (paper journal, shadow log,
safe-entries state, retrain history, config, env flags) and returns
structured dicts. No side effects, no writes. Consumers: CLI report
script + Streamlit dashboard page.
"""
from .act_evaluator import (
    build_report,
    load_component_state,
    load_paper_trades,
    bucket_attribution,
    rolling_sharpe_series,
    recommendations,
)

__all__ = [
    "build_report",
    "load_component_state",
    "load_paper_trades",
    "bucket_attribution",
    "rolling_sharpe_series",
    "recommendations",
]
