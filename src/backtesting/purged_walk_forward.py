"""Purged Walk-Forward Cross-Validation (López de Prado 2017).

Standard walk-forward: train on [0, T/2], test on (T/2, T]. The
problem in financial time-series: a sample at time t can have
features computed from windows that overlap with samples in the
test set, leaking future info backward (or test info forward).

Purged WF fixes this by:
  1. PURGE — drop training samples whose feature-computation window
     overlaps with the test fold.
  2. EMBARGO — leave a buffer period after each test fold before
     resuming training, so the next training window doesn't draw
     on data that bleeds from the test.

ACT's existing walk-forward (`src/trading/backtest.py`) does the
80/15/5 split but does NOT purge or embargo. This module provides
the purged variant for the brain to call when it wants to validate
a strategy hypothesis with leak-free out-of-sample folds.

Anti-overfit design:
  * Returns *per-fold* metrics, not just an aggregate, so the
    operator sees variance across folds (overfit detector)
  * Embargo period is mandatory (default 5% of fold length)
  * Sample-size warnings when fewer than 5 folds or < 30
    samples per fold
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class PurgedFold:
    fold_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    purged_indices: int
    embargo_size: int
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    train_n_trades: int = 0
    test_n_trades: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_index": int(self.fold_index),
            "train_range": [int(self.train_start), int(self.train_end)],
            "test_range": [int(self.test_start), int(self.test_end)],
            "purged_indices": int(self.purged_indices),
            "embargo_size": int(self.embargo_size),
            "train_sharpe": round(float(self.train_sharpe), 4),
            "test_sharpe": round(float(self.test_sharpe), 4),
            "train_n_trades": int(self.train_n_trades),
            "test_n_trades": int(self.test_n_trades),
        }


@dataclass
class PurgedWalkForwardResult:
    n_folds: int
    n_samples: int
    embargo_pct: float
    feature_window: int
    folds: List[PurgedFold] = field(default_factory=list)
    aggregate_train_sharpe: float = 0.0
    aggregate_test_sharpe: float = 0.0
    fold_test_sharpe_stdev: float = 0.0
    overfit_indicator: float = 0.0   # train_sharpe - test_sharpe (large = overfit)
    sample_warning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_folds": int(self.n_folds),
            "n_samples": int(self.n_samples),
            "embargo_pct": round(float(self.embargo_pct), 4),
            "feature_window": int(self.feature_window),
            "aggregate_train_sharpe": round(float(self.aggregate_train_sharpe), 4),
            "aggregate_test_sharpe": round(float(self.aggregate_test_sharpe), 4),
            "fold_test_sharpe_stdev": round(float(self.fold_test_sharpe_stdev), 4),
            "overfit_indicator": round(float(self.overfit_indicator), 4),
            "folds": [f.to_dict() for f in self.folds],
            "sample_warning": self.sample_warning,
            "advisory": (
                "overfit_indicator > 1.0 = train Sharpe much higher than test → "
                "overfit. fold_test_sharpe_stdev > 1.5 × aggregate_test_sharpe "
                "= unstable across folds (also overfit signal)."
            ),
        }


def _sharpe(returns: List[float]) -> float:
    if not returns:
        return 0.0
    n = len(returns)
    m = sum(returns) / n
    var = sum((r - m) ** 2 for r in returns) / max(1, n)
    sd = math.sqrt(var)
    return (m / sd * math.sqrt(252)) if sd > 0 else 0.0


def purged_walk_forward(
    returns: List[float],
    strategy_fn: Callable[[List[float]], List[float]],
    n_folds: int = 5,
    embargo_pct: float = 0.05,
    feature_window: int = 20,
) -> PurgedWalkForwardResult:
    """Run purged walk-forward over a return series.

    Args:
      returns: per-period returns to walk over.
      strategy_fn: callable that takes a slice of returns and emits
        a per-period strategy-return series of equal length. The
        strategy is parameter-free (parameters baked in) so each
        fold replays the same strategy on different windows.
      n_folds: number of test folds.
      embargo_pct: fraction of fold length to embargo after each
        test fold.
      feature_window: length of the feature-computation window;
        purging removes that many indices on each side of a test
        fold from training.

    Returns:
      PurgedWalkForwardResult with per-fold + aggregate metrics +
      overfit indicator.
    """
    n = len(returns)
    if n < n_folds * (feature_window + 10):
        return PurgedWalkForwardResult(
            n_folds=n_folds, n_samples=n,
            embargo_pct=embargo_pct,
            feature_window=feature_window,
            sample_warning="insufficient_samples_for_n_folds_with_purge",
        )

    fold_size = n // n_folds
    embargo_size = max(1, int(fold_size * embargo_pct))
    folds: List[PurgedFold] = []

    for fold_i in range(n_folds):
        test_start = fold_i * fold_size
        test_end = test_start + fold_size if fold_i < n_folds - 1 else n

        # Purge feature_window indices on both sides of the test fold
        # from training.
        purge_start = max(0, test_start - feature_window)
        purge_end = min(n, test_end + feature_window)
        # Embargo: extend purge by embargo_size after the test fold.
        embargo_end = min(n, test_end + feature_window + embargo_size)

        # Build training set: everything outside [purge_start, embargo_end].
        train_indices = list(range(0, purge_start)) + list(range(embargo_end, n))
        train_returns = [returns[i] for i in train_indices]
        test_returns = returns[test_start:test_end]

        if len(train_returns) < 30 or len(test_returns) < 5:
            continue

        try:
            train_strategy = strategy_fn(train_returns)
            test_strategy = strategy_fn(test_returns)
        except Exception:
            continue

        purged_count = (purge_end - purge_start) - (test_end - test_start)
        folds.append(PurgedFold(
            fold_index=fold_i,
            train_start=0, train_end=purge_start,
            test_start=test_start, test_end=test_end,
            purged_indices=purged_count,
            embargo_size=embargo_size,
            train_sharpe=_sharpe(train_strategy),
            test_sharpe=_sharpe(test_strategy),
            train_n_trades=sum(1 for r in train_strategy if r != 0),
            test_n_trades=sum(1 for r in test_strategy if r != 0),
        ))

    if not folds:
        return PurgedWalkForwardResult(
            n_folds=n_folds, n_samples=n,
            embargo_pct=embargo_pct,
            feature_window=feature_window,
            sample_warning="no_valid_folds",
        )

    train_sharpes = [f.train_sharpe for f in folds]
    test_sharpes = [f.test_sharpe for f in folds]
    agg_train = sum(train_sharpes) / len(train_sharpes)
    agg_test = sum(test_sharpes) / len(test_sharpes)
    test_var = sum((s - agg_test) ** 2 for s in test_sharpes) / max(1, len(test_sharpes))
    test_stdev = math.sqrt(test_var)
    overfit_indicator = agg_train - agg_test

    sample_warning = ""
    if len(folds) < 3:
        sample_warning = "fewer_than_3_folds_indicators_unstable"
    elif n < 200:
        sample_warning = "short_history_below_200_samples"

    return PurgedWalkForwardResult(
        n_folds=len(folds), n_samples=n,
        embargo_pct=embargo_pct,
        feature_window=feature_window,
        folds=folds,
        aggregate_train_sharpe=agg_train,
        aggregate_test_sharpe=agg_test,
        fold_test_sharpe_stdev=test_stdev,
        overfit_indicator=overfit_indicator,
        sample_warning=sample_warning,
    )
