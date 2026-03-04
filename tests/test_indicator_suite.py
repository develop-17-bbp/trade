import pytest
from src.indicators.indicators import bulk_indicators


def test_bulk_indicators_length():
    # generate a simple price series
    closes = [float(i) for i in range(100)]
    highs = [c + 1 for c in closes]
    lows = [c - 1 for c in closes]
    volumes = [100.0] * len(closes)
    features = bulk_indicators(closes, highs, lows, volumes)
    # we expect at least 120 distinct feature series
    assert isinstance(features, dict)
    assert len(features) >= 120
    # each series length should match input length
    for name, series in features.items():
        assert len(series) == len(closes), f"{name} length mismatch"
    # new features should be present
    assert any(k.startswith('kama_') for k in features)
    assert any(k.startswith('ou_') for k in features)
    assert 'wavelet_strength' in features
