Numerical Models, Mathematical Equations, Indicators, Market Cycles,
News & Sentiment Analysis, and a Realistic Risk Proposal

Overview
- Purpose: define the core quantitative building blocks, the math behind them, and a risk management proposal that explicitly addresses the 1% daily benchmark realism.
- Architecture mapping: adopt the provided 3‑layer hybrid architecture (L1 quantitative engine, L2 sentiment layer, L3 risk engine).

1) Core mathematical primitives
- Price returns (log return):
  - $r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$
- Simple return (for discrete P&L):
  - $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$

2) Volatility & regime detection
- EWMA (Exponential weighted moving variance) volatility estimate:
  - Let $\lambda \in (0,1)$ be the decay. For squared returns $x_t = r_t^2$:
  - $\sigma_t^2 = (1-\lambda) x_t + \lambda \sigma_{t-1}^2$
  - Instantaneous vol: $\sigma_t = \sqrt{\sigma_t^2}$
- GARCH(1,1) (model spec):
  - $\sigma_t^2 = \omega + \alpha\, r_{t-1}^2 + \beta\, \sigma_{t-1}^2$
  - Constraint: $\omega>0,\ \alpha,\beta\ge 0,\ \alpha+\beta<1$ for stationarity.
- ATR (Average True Range) useful for stop levels:
  - $TR_t = \max( H_t-L_t, |H_t-C_{t-1}|, |L_t-C_{t-1}|)$
  - $ATR_t = SMA(TR, n)$ (or EMA for faster reaction)
- Volatility regime filter: classify as low/medium/high using EWMA or GARCH; e.g. high-vol if $\sigma_t > k\cdot\sigma_{median}$.

3) Numerical models used for signals
- Z-score normalization for mean-reversion:
  - Given window of prices $\{P_{t-w+1},...,P_t\}$ compute mean $\mu$ and sd $s$ then $z_t=(P_t-\mu)/s$.
- Momentum/Trend signals: MA crossover, slope, normalized momentum.
  - SMA_t(n) = $\frac{1}{n}\sum_{i=0}^{n-1}P_{t-i}$
  - EMA recursive: $EMA_t = \alpha P_t + (1-\alpha) EMA_{t-1}$ with $\alpha=2/(n+1)$
- MA Crossover Signal (discrete):
  - signal_t = 1 if SMA_short_t > SMA_long_t and prior diff <=0
  - signal_t = -1 if SMA_short_t < SMA_long_t and prior diff >=0
- Volatility-adjusted returns: $R_t^{adj} = \frac{R_t}{\sigma_t}$ to compare across regimes.
**Forecasting Sub-models (optional L1 extension)**
- The framework can plug in machine learning forecasters which output short-term price predictions. The following classes are supported via `src/models/forecasting.py`:
  * **FinGPTForecaster** – a GPT-based financial forecasting model (requires `fingpt` package).
  * **LightGBMForecaster** – gradient-boosted regression trained autoregressively on past prices.
  * **NBeatsForecaster** / **TFTForecaster** – deep sequence models from `pytorch-forecasting`.
- Output of the chosen forecaster is converted to a directional signal (buy if predicted > current price, sell if lower). Weighting is configurable under the L1 `weights.forecast` parameter.
4) Technical indicators (equations)
- SMA: $SMA_t(n)=\frac{1}{n}\sum_{i=0}^{n-1} P_{t-i}$
- EMA: recursive as above
- RSI (14):
  - $RSI = 100 - \frac{100}{1+RS}$ where $RS = \frac{AvgGain}{AvgLoss}$ using smoothing
- MACD: difference of EMAs: $MACD = EMA_{12}-EMA_{26}$ and signal = EMA(MACD,9)
- ATR for volatility-aware stops (see ATR above)

5) Market cycles detection
- Spectral / FFT peak detection (pseudo): find dominant frequencies in log-return series
  - compute periodogram of $r_t$ (or detrended series), find peaks at frequency $f^*$ → cycle period $T = 1/f^*$
- Band-pass filter (e.g., Butterworth) or cycle extraction via Hodrick–Prescott (HP) filter for low-freq trends and cyclical component.
- Practical approach: compute FFT on rolling windows (e.g., 128/256 points), detect peaks above threshold and tag cycle regime. Use cycle period to adapt holding horizon.

6) News analysis & event scoring pipeline
- Stages:
  1. Ingest: RSS, NewsAPI, Reddit, Twitter (careful with rate limits and ToS).
  2. Normalize & dedupe headlines; extract timestamp and tickers (entity extraction).
  3. Classify events (binary/score): event types = {regulatory, hack, ETF news, macro, exchange outage, listing}.
  4. Score: produce $s_i \in [-1,1]$ and confidence $c_i \in [0,1]$.
  5. Aggregate with time-decay and source weights: see equation below.

- Weighted, time-decayed aggregation (per asset):
  - Let events in last window be $(s_i,c_i,t_i)$ with now = $t$ and decay kernel $w(t_i)=e^{-\gamma (t-t_i)}$.
  - Aggregate sentiment:
    $$S_t = \frac{\sum_i c_i \, w(t_i) \, s_i}{\sum_i c_i \, w(t_i)}$$
  - Use $S_t$ as L2 input; threshold to generate bullish/bearish event flags.

7) Sentiment analysis (text → numeric)
- Two-tier approach:
  - Fast rule-based scoring (keyword lexicon) for low latency.
  - Transformer-based classifier for higher accuracy (queued, L2). Use sentence-transformers or fine-tuned FinBERT/CryptoBERT. Keep transformer inference queued or batched.
- Sentence aggregation: average embeddings + classifier or direct sentiment pipeline.
- Example scoring: transform label {POS,NEU,NEG} into numeric s in {-1,0,1} and attach confidence from classifier probability.

8) Hybrid signal architecture mapping (attachment)
- L1 Quantitative Engine (50% weight): indicators, volatility regime, numeric models (z-score, MA cross, momentum, GARCH features). Latency target <100ms.
- L2 Sentiment Layer (30% weight): news scoring, event classification, narrative changes. Latency target <10s (queued/batched).
- L3 Risk Engine (20%+ veto): position sizing, drawdown gating, volatility filter, circuit breaker. Latency <50ms for enforcement.

9) Risk management & 1% daily realism assessment
- Growth math: daily target $g_d=1\%$ gives annual compounded growth (using 365 days for crypto):
  - $G_{yr}=(1+g_d)^{365}-1$; numerically $G_{yr}\approx (1.01)^{365}-1 \approx 36.7$ → ~3670% annual.
  - This is extremely high and implies very large tail risk.
- Trade-level required edge (simplified): For N trades/day with expected per-trade return $\mu$ and per-trade volatility $\sigma$, to get average daily return $g_d$, we need $N\mu \approx g_d$ on average (ignoring compounding). If typical per-trade edge is O(0.1%) then N would need to be large and costs/slippage dominate.
- Kelly-style sizing (for binary edge): $f^* = \frac{bp - q}{b}$ with win probability $p$, loss prob $q=1-p$, payoff ratio $b=\frac{E[win]}{E[loss]}$; unconstrained Kelly often too aggressive.

Realistic operational risk controls (must be implemented before any live run):
1. Max position size cap (configurable): e.g., max 1-2% account per position (we use 2% default in repo).
2. Daily loss limit: e.g., 3-5% stop-out for all trading. Trigger: stop all trading for remainder of day.
3. Volatility gating: if EWMA volatility > threshold (e.g., 2x median), reduce position sizing by factor or pause intraday trading.
4. ATR-based stop-loss: set stop at entry ± k * ATR (k typically 1.5–3 depending on horizon).
5. Circuit breaker: if drawdown over X% in Y days exceed threshold, disable strategy until human review.
6. Conservative leverage: avoid leverage initially; if used, cap to low multiples (<=2x) and require additional risk capital.
7. Model monitoring: track prediction distributions, concept drift, and trigger retraining or rollback.

10) Practical calibration and a sample risk plan
- Assume initial equity $E_0 = $100,000. Max position (2%) → $2,000 per position.
- If average trade ATR-based stop is 2%, risk per trade = $2,000 * 2% = $40.
- To make $1,000/day (1% of $100k) you would need 25 such independent trades all profitable after costs — unrealistic.
- Therefore alternatives:
  - Lower daily target: e.g., 0.05–0.2% daily target is more achievable depending on edge and frequency.
  - Focus on risk-adjusted returns (Sharpe, Sortino) rather than raw daily %.

11) Implementation checklist & code pointers (in repo)
- L1: `src/indicators/indicators.py`, `src/models/numerical_models.py`, `src/trading/backtest.py` (already present)
- L2: `src/data/news_fetcher.py`, `src/ai/sentiment.py` (rule-based + transformer optional)
- L3: `src/risk/manager.py`, `src/risk/position_sizing.py` (position caps, daily loss)

12) Next steps (recommended immediate actions)
- Add volatility module (EWMA & ATR) and integrate gating in `TradingExecutor`.
- Add cycle detector (rolling FFT peak) to adapt holding periods.
- Replace Reddit-only news with a paid NewsAPI during development for reliability.
- Add slippage/fees modeling to `run_backtest` and walk-forward testing.
- Run long backtests and Monte Carlo to estimate drawdowns for target daily returns.

Appendix: Key equations (KaTeX-ready)
- Log return: $r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$
- EWMA volatility: $\sigma_t^2 = (1-\lambda) r_t^2 + \lambda \sigma_{t-1}^2$
- GARCH(1,1): $\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2$
- Kelly fraction: $f^* = \frac{bp - q}{b}$
- Weighted sentiment aggregate: $S_t = \dfrac{\sum_i c_i e^{-\gamma (t-t_i)} s_i}{\sum_i c_i e^{-\gamma (t-t_i)}}$

Notes on compliance & safety
- Robinhood and other broker automation may have ToS limits. Use paper trading or official broker APIs when possible.
- Always include manual kill-switch and audit logs for every live order.



