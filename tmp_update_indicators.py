import sys

file_path = r'c:\Users\convo\trade\src\indicators\indicators.py'

new_code = r'''
# ---------------------------------------------------------------------------
# Volume Delta (Aggressor Volume)
# ---------------------------------------------------------------------------
def volume_delta(opens, closes, volumes):
    """Simplified Volume Delta: Positive = Buy Bias, Negative = Sell Bias."""
    n = len(closes)
    out = []
    for i in range(n):
        try:
            dr = abs(closes[i] - opens[i]) or 1e-10
            bias = (closes[i] - opens[i]) / dr
            out.append(float(bias * volumes[i]))
        except:
            out.append(0.0)
    return out

# ---------------------------------------------------------------------------
# Liquidity Sweep Detection
# ---------------------------------------------------------------------------
def liquidity_sweep(highs, lows, closes, lookback=20):
    """Detects stop-hunting sweeps below Low/above High."""
    n = len(closes)
    out = [0.0] * n
    for i in range(lookback, n):
        rh, rl = max(highs[i-lookback:i]), min(lows[i-lookback:i])
        if lows[i] < rl and closes[i] > rl: out[i] = 1.0 # Bullish
        elif highs[i] > rh and closes[i] < rh: out[i] = -1.0 # Bearish
    return out

# ---------------------------------------------------------------------------
# VWAP Deviation
# ---------------------------------------------------------------------------
def vwap_deviation(closes, vwap_vals, period=20):
    """Z-score of Price relative to VWAP."""
    n = len(closes)
    out = [0.0] * n
    for i in range(period, n):
        try:
            import numpy as np
            devs = [c - v for c, v in zip(closes[i+1-period:i+1], vwap_vals[i+1-period:i+1])]
            std = float(np.std(devs)) or 1e-10
            out[i] = float((closes[i] - vwap_vals[i]) / std)
        except:
            out[i] = 0.0
    return out

# Optional: expose in __all__
__all__ = ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands',
           'true_range', 'atr', 'stochastic', 'vwap', 'obv', 'adx',
           'bb_width', 'roc', 'williams_r', 'bulk_indicators',
           'kama', 'ou_signal', 'wavelet_cycle_strength',
           'volume_delta', 'liquidity_sweep', 'vwap_deviation']
'''

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

if "__all__ =" in content:
    # Find start of __all__ and remove it to the end
    idx = content.find("# Optional: expose in __all__")
    if idx == -1: idx = content.find("__all__ =")
    content = content[:idx]

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content.strip() + "\n" + new_code)

print("Successfully updated indicators.py")
