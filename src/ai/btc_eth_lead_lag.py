"""BTC ↔ ETH lead-lag analyzer.

In bullish phases, ETH often LEADS BTC (alts run first, BTC catches
up). In panic/bearish phases, BTC LEADS ETH (BTC drops first, alts
follow + amplify). Detecting which is leading right now is a real
predictive signal for the trailing asset.

Algorithm — cross-correlation across lags:
  For lags k = -10 to +10 bars:
    compute correlation(BTC_returns_t, ETH_returns_{t-k})
  The lag with peak |correlation| indicates which is leading.
    Positive optimal lag → BTC leads ETH by k bars
    Negative optimal lag → ETH leads BTC by |k| bars
  Strength = absolute value of peak correlation.

Anti-noise:
  * Requires N=30+ bars minimum (low-sample warning below 50)
  * Lag range capped at ±10 bars (no spurious long-range correlations)
  * Strength threshold (default 0.3) to filter weak relationships
  * Pure compute — no external API, no learned weights
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

DEFAULT_MAX_LAG = 10
DEFAULT_MIN_STRENGTH = 0.3


@dataclass
class LeadLagSignals:
    method: str
    optimal_lag_bars: int = 0          # +ve = BTC leads ETH; -ve = ETH leads BTC
    correlation_strength: float = 0.0  # absolute value of peak corr
    relationship: str = "unclear"       # "btc_leads_eth" / "eth_leads_btc" / "synchronous" / "unclear"
    btc_etn_corr_at_zero_lag: float = 0.0
    sample_warning: str = ""
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "optimal_lag_bars": int(self.optimal_lag_bars),
            "correlation_strength": round(float(self.correlation_strength), 4),
            "relationship": self.relationship,
            "btc_etn_corr_at_zero_lag": round(float(self.btc_etn_corr_at_zero_lag), 4),
            "sample_warning": self.sample_warning,
            "rationale": self.rationale[:300],
        }


def _returns(closes: List[float]) -> List[float]:
    """Per-bar pct returns."""
    if len(closes) < 2:
        return []
    return [(closes[i] / closes[i - 1] - 1.0) for i in range(1, len(closes))]


def _correlation(xs: List[float], ys: List[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 5:
        return 0.0
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n
    var_x = sum((xs[i] - mx) ** 2 for i in range(n)) / n
    var_y = sum((ys[i] - my) ** 2 for i in range(n)) / n
    if var_x <= 0 or var_y <= 0:
        return 0.0
    return cov / ((var_x * var_y) ** 0.5)


def analyze_lead_lag(
    btc_closes: List[float],
    eth_closes: List[float],
    max_lag: int = DEFAULT_MAX_LAG,
    min_strength: float = DEFAULT_MIN_STRENGTH,
) -> LeadLagSignals:
    """Find the lag at which BTC↔ETH correlation peaks. Returns
    relationship classification."""
    n = min(len(btc_closes), len(eth_closes))
    if n < 30:
        return LeadLagSignals(method="lead_lag",
                                sample_warning="insufficient_bars",
                                rationale="need >= 30 aligned bars")

    btc_returns = _returns(btc_closes[-n:])
    eth_returns = _returns(eth_closes[-n:])

    if len(btc_returns) < 20 or len(eth_returns) < 20:
        return LeadLagSignals(method="lead_lag",
                                sample_warning="insufficient_returns",
                                rationale="too_few_returns_after_diff")

    # Lag = +k means BTC at t correlates with ETH at t-k, i.e. BTC leads ETH by k bars
    # We compute correlation for lag in [-max_lag, +max_lag]
    best_lag = 0
    best_corr = 0.0
    zero_lag_corr = _correlation(btc_returns, eth_returns)
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            c = zero_lag_corr
        elif lag > 0:
            # BTC[t] vs ETH[t-lag] → trim
            c = _correlation(btc_returns[lag:], eth_returns[:-lag])
        else:
            c = _correlation(btc_returns[:lag], eth_returns[-lag:])
        if abs(c) > abs(best_corr):
            best_corr = c
            best_lag = lag

    strength = abs(best_corr)
    if strength < min_strength:
        relationship = "unclear"
    elif best_lag == 0:
        relationship = "synchronous"
    elif best_lag > 0:
        relationship = "btc_leads_eth"
    else:
        relationship = "eth_leads_btc"

    sample_warning = ""
    if n < 50:
        sample_warning = "low_sample_under_50_bars"

    rationale = (
        f"relationship={relationship} optimal_lag={best_lag:+d} "
        f"strength={strength:.3f} zero_lag_corr={zero_lag_corr:+.3f}"
    )

    return LeadLagSignals(
        method="lead_lag",
        optimal_lag_bars=best_lag,
        correlation_strength=strength,
        relationship=relationship,
        btc_etn_corr_at_zero_lag=zero_lag_corr,
        sample_warning=sample_warning,
        rationale=rationale[:300],
    )
