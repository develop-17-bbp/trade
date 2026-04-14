import os
import tempfile

import pytest

from src.trading.executor import TradingExecutor
from src.models.lightgbm_classifier import LightGBMClassifier


def test_feedback_logging_and_retraining(tmp_path):
    """Test that executor initializes with feedback + ML components and classifier works."""
    # create a minimal executor with synthetic data
    cfg = {'mode': 'paper', 'assets': ['BTC'], 'risk': {'risk_per_trade_pct': 1.0}}
    execu = TradingExecutor(cfg)

    # Executor should initialize without crashing
    assert execu is not None
    assert execu.config.get('mode') == 'paper'

    # LightGBM classifier should be available on the executor
    assert hasattr(execu, '_lgbm')
    clf = execu._lgbm
    assert isinstance(clf, LightGBMClassifier)

    # Classifier should support trade logging
    assert hasattr(clf, '_trade_log')
    assert isinstance(clf._trade_log, list)

    # retrain_from_log should run without error even if no feature columns
    clf.retrain_from_log(max_examples=10)

    # Adaptive feedback loop should be initialized
    assert hasattr(execu, '_adaptive')
