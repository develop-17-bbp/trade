from src.risk.position_sizing import fixed_fraction, kelly_criterion

def test_fixed_fraction():
    assert fixed_fraction(1000, 2) == 20

def test_kelly_nonnegative():
    k = kelly_criterion(0.6, 1.5)
    assert k >= 0
