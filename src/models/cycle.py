from typing import List, Optional
import numpy as np

def rolling_fft_period(closes: List[float], window: int = 128, top_k: int = 1) -> List[Optional[float]]:
    """Return estimated dominant cycle period per window (in bars)."""
    out = [None] * len(closes)
    n = len(closes)
    for i in range(window - 1, n):
        window_vals = np.array(closes[i - window + 1:i + 1])
        # detrend via linear detrend
        x = np.arange(window)
        coeffs = np.polyfit(x, window_vals, 1)
        detrended = window_vals - (coeffs[0] * x + coeffs[1])
        fft = np.fft.rfft(detrended)
        ps = np.abs(fft)**2
        # ignore zero frequency
        ps[0] = 0
        idx = np.argmax(ps)
        freq = idx / float(window)
        if freq == 0:
            out[i] = None
        else:
            period = 1.0 / freq
            out[i] = float(period)
    return out
