from src.indicators.indicators import sma, rsi, kama, ou_signal, wavelet_cycle_strength

def test_sma_basic():
    vals = [1,2,3,4,5]
    out = sma(vals, 3)
    assert len(out) == 5
    assert out[2] == (1+2+3)/3

def test_rsi_length():
    vals = [1,2,1,2,1,2,1,2,1,2,1,2,1,2,1]
    out = rsi(vals, period=5)
    assert len(out) == len(vals)


def test_kama_behaviour():
    vals = [10.0] * 50
    out = kama(vals, period=5)
    # after warmup, should remain at flat value
    assert all(x == 10.0 for x in out[4:])


def test_ou_signal_basic():
    vals = list(range(30))
    sig = ou_signal(vals, window=5)
    assert len(sig) == len(vals)
    assert sig[0] == 0.0  # no data to compute
    # the first non-zero entry should occur at index window-1
    assert sig[4] != 0.0
    assert any(abs(x) > 0 for x in sig[5:])


def test_wavelet_cycle_strength_exists():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = wavelet_cycle_strength(vals)
    assert isinstance(out, list)
    assert len(out) == len(vals)
