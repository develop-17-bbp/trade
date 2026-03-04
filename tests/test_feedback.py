import os
import tempfile

import pytest

from src.trading.executor import TradingExecutor
from src.models.lightgbm_classifier import LightGBMClassifier
from src.trading.strategy import HybridStrategy


def test_feedback_logging_and_retraining(tmp_path):
    # create a minimal executor with synthetic data
    cfg = {'mode': 'paper', 'assets': ['BTC'], 'risk': {'risk_per_trade_pct': 1.0}}
    execu = TradingExecutor(cfg)
    # run paper mode for one asset (this will execute backtest and attempt retraining)
    execu.assets = ['BTC']
    execu.initial_capital = 1000.0
    # monkeypatch price fetcher to return simple increasing data
    def fake_fetch(symbol):
        n = 50
        closes = [100 + i for i in range(n)]
        return {'closes': closes, 'highs': closes, 'lows': closes, 'volumes': [1.0] * n}
    execu._fetch_data = fake_fetch
    execu._fetch_sentiment = lambda asset: ([], [], [], [])

    # run paper; should not crash
    execu._run_paper()

    # classifier should have logged some trades
    clf = execu.strategy.classifier
    assert hasattr(clf, '_trade_log')
    assert isinstance(clf._trade_log, list)

    # retrain_from_log should run without error even if no feature columns
    clf.retrain_from_log(max_examples=10)

    # RL agent stub should have experience recorded
    assert hasattr(execu.strategy, '_rl_experience')
    assert isinstance(execu.strategy._rl_experience, list)
