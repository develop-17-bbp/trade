import pytest
from src.models.forecasting import ForecastingEngine


def test_forecaster_defaults():
    # when no optional models installed, engine returns zero signals
    eng = ForecastingEngine({})
    series = [100.0, 101.0, 102.0, 103.0]
    sig = eng.generate_signal(series)
    assert isinstance(sig, list)
    assert len(sig) == len(series)
    assert all(s == 0 for s in sig)


def test_forecaster_lgbm_override():
    # if lightgbm is installed we expect non-error behaviour (may still be zeros)
    eng = ForecastingEngine({'use_lgbm': True})
    series = [i for i in range(50, 100)]
    sig = eng.generate_signal(series)
    assert isinstance(sig, list)
    assert len(sig) == len(series)


def test_l1_engine_integration():
    from src.models.numerical_models import L1SignalEngine
    cfg = {'forecast': {'use_lgbm': False, 'use_fingpt': False}}
    engine = L1SignalEngine(cfg)
    series = [float(i) for i in range(100, 150)]
    result = engine.generate_signals(series)
    assert 'composite' in result
    assert 'signals' in result
    assert 'forecast_signal' in result
    # forecast values default to 0 since weight is zero
    assert all(v == 0 for v in result['forecast_signal'])
    assert isinstance(result['composite'], list)
