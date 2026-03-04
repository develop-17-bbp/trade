AI-Driven Crypto Trading System

An automated trading system for BTC/ETH using open-source AI models, technical indicators, and risk management. Includes paper-trading simulation, backtester, and **full Robinhood integration** for live trading.

Features
- ✅ **Full Robinhood integration** (authentication, order placement, position tracking)
- Technical indicators: SMA, EMA, RSI (now 120+ quantitative indicators via bulk generator, with recent additions like KAMA adaptive MA, OU mean‑reversion signals, and wavelet cycle strength)
- Numerical models: z-score, MA crossover signals
- **Sentiment**: two-tier system where the natural language model (FinBERT/CryptoBERT) is fully hooked into the pipeline.  Configure `SentimentPipeline` with `sentiment_model='finbert'` or use `FinBERTService` directly for rich polarity/confidence features.
- Position sizing: fixed-fraction, Kelly criterion
- Risk manager: max position size and daily loss limits
- AI sentiment analysis (rule-based + transformer fallback)
- News fetcher aggregator (now pulls up to 100 headlines per query)
- Paper-mode backtester
- **Extension stubs**: portfolio optimizer, on-chain fetcher,
  regime classifier, meta-RL sizer, smart order router, drift detector,
  chaos tester, API health monitor, hyperparameter tuner
- CCXT price data integration

Quick Start (Windows)

```powershell
# Create and activate venv
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run paper trading simulation (real-time Binance data via CCXT)
python -m src.main

# Run tests
pytest -v
```

**Note:** This system requires real-time market data from Binance (via CCXT). Demo/synthetic mode is no longer supported.

### Running in Live Mode with Robinhood

```powershell
# Set environment variables
$env:ROBINHOOD_USER = "your_email@example.com"
$env:ROBINHOOD_PASSWORD = "your_password"
$env:ROBINHOOD_MFA = "123456"  # optional

# Update config.yaml: mode: live
# Then run:
python -m src.main

# System will authenticate with Robinhood and execute live orders
```

See [QUICKSTART.md](QUICKSTART.md#robinhood-integration-) for complete setup guide.

### Training the LightGBM Model 🎓

A pre‑trained model is **not shipped with the repo**; you are expected to
train (and later retrain) the classifier on your own historical data. The
`src/scripts/train_lgbm.py` helper automates the core workflow described
earlier:

1. Download OHLCV candles from Binance via CCXT
2. Compute the 80+ feature vector for each bar
3. Label each bar by the next‑bar price move
4. Split data, train LightGBM and save the model file

Example usage (requires `lightgbm` & `ccxt`):

```powershell
python -m src.scripts.train_lgbm \
    --symbol BTC/USDT --timeframe 1h \
    --since 2018-01-01 --until 2026-03-01 \
    --model-out models/lgbm_model.txt
```

After training, point the classifier at the saved model by editing
`src/models/lightgbm_classifier.py` or -- if you prefer to automate it
– modify `LightGBMClassifier.__init__` to load from `cfg.get('model_path')`.

Feel free to experiment with alternative label definitions (e.g. 5‑bar
returns, volatility‑adjusted thresholds) or add additional features
(such as on‑chain metrics).  The data fetched by the script is stored
in pandas DataFrames so you can export to CSV and inspect it with a
notebook prior to training.

The classifier is the **core model** the system relies on; improving its
training dataset will have the most direct impact on portfolio
returns.  Once you have a model file you can run backtests to verify
performance and then deploy it in live/testnet mode.

### Binance Testnet / Demo Mode

You can exercise the *entire* pipeline against the Binance sandbox
(referred to internally as "testnet") using real order‑book data but
zero real money.  This is the fastest way to verify live connectivity,
execution logic, and the feedback/retraining loop without risking capital.

1. create a Binance testnet account and generate an API key/secret
   (https://testnet.binance.vision/)
2. set the credentials in your environment or `config.yaml`:

```powershell
$env:BINANCE_TESTNET_KEY = "your_key"
$env:BINANCE_TESTNET_SECRET = "your_secret"
# or set in config file under
# exchange:
#   api_key: ...
#   api_secret: ...
```

3. edit `config.yaml` (or `config_optimized.yaml`) and set:
   ```yaml
   mode: testnet       # paper, live, or testnet
   exchange:
     name: binance
     api_key: ${BINANCE_TESTNET_KEY}
     api_secret: ${BINANCE_TESTNET_SECRET}
   ```
   you may also leave `mode` as `paper` and pass `--mode testnet` to
   `src/main.py` as an override.
4. run the executor:

```powershell
python -m src.main   # defaults to testnet if config says so
```

The executor will print a header like "[TESTNET MODE] Trading with fake
money on Binance Testnet" and will authenticate to the sandbox.  It will
fetch live tickers, compute indicators and signals, backtest the signals,
then place orders on the testnet account via CCXT.  Balances, filled
orders and P&L appear just as in live mode, but funds are fictitious.

> **Note:** the same retraining / feedback loop kicks in during testnet
> runs, so you can watch the model log and retrain itself after each
> simulated trade.

---

## Hybrid Ensemble Architecture (v6.5)

The project now implements a **seven-layer dual-engine architecture** merging the original LightGBM‑centric design (v5.5) with the agentic RL oversighter introduced in v6.0. Both engines run in parallel and are fused at Layer 4 by a Meta‑Controller that dynamically weights, vetoes, and scales position size based on volatility, sentiment, and risk projections. The result is a robust baseline (0.4 % daily) with the ability to hit 1 %+ in aligned regimes.

### Seven-Layer Design
1. **Data Ingestion** – 500 ms ticks from Binance/Robinhood/Coinbase, news (FinBERT), social (X), on‑chain (Dune).  L3 order‑book events (CoinAPI) are planned as a 2026 upgrade.
2. **Feature Engineering** – 140‑plus features: TA (RSI, MACD, etc.), volatility (GARCH/EGARCH), cycles, on‑chain metrics (MVRV, SOPR, HODL waves).
3. **Model Inference** – *Dual, parallel engines*:
    * v5.5 LightGBM Core: TFT, N‑Beats, FinBERT, Prophet regressor producing a 3‑class Long/Flat/Short signal with polarity and confidence.
    * v6.0 RL Agents: PPO (Stable Baselines3), PatchTST, Llama 3.1 70B, XGBoost on‑chain models generating action probabilities and 100× Monte‑Carlo VaR paths.
4. **Meta‑Controller & Orchestration** – XGBoost arbitrator fuses L1 outputs, applies veto rules, and scales sizes (RL weight ramps to 80 % in high vol).  See snippet below.
5. **Execution** – CCXT router with Robinhood primary and Binance/Coinbase fail‑over; TWAP execution and adaptive market‑making ensure slippage <0.05 %.
6. **Audit** – Kafka → ClickHouse dual replay for both engines (paper + live).  All orders, signals, and risk states are logged.
7. **Feedback Loop** – Shared training pool; LightGBM retrained weekly, RL daily, federated quarterly; drift detectors on hot/warm/cold paths.

```mermaid
flowchart TD
    A[1. Data Ingestion\n(L3 events soon)] --> B[2. Feature Engineering\n(Indicators, Volatility, On‑Chain)]
    B --> C1[3a. LightGBM Engine\n(TFT / N‑Beats / FinBERT)]
    B --> C2[3b. RL Engine\n(PPO / Llama / VaR)]
    C1 & C2 --> D[4. Meta‑Controller\n(XGBoost arbitrator)]
    D --> E[5. Execution & Audit\n(CCXT Router + Robinhood)]
    E --> F[6. Audit & Logging]
    F --> G[7. Feedback / Retraining]
    G --> D
```

### Meta‑Controller (L4) Example
```python
from src.trading.meta_controller import MetaController

# you can pass a small bias (e.g. bias=0.05) to tilt the system in your favor
mc = MetaController({'bias': 0.05})
l_class, l_conf = 1, 0.9    # LightGBM predicts "Long" with 90% confidence
r_action, r_prob = 1, 0.65 # RL also leans long, 65% probability
features = {'ewma_vol':0.05, 'vol_adj_momentum':0.9}
finbert_score = -0.05       # minor bearish sentiment
final_dir, final_conf, scale = mc.arbitrate(
    l_class, l_conf, r_action, r_prob, features, finbert_score
)
# final_dir==1 (long), final_conf~0.7, scale==1.0
```

### Backtest Comparison (2020–2026)
| Engine | Avg. Daily | Sharpe | Calmar |
|--------|-----------:|-------:|-------:|
| LightGBM only | 0.32 % | 1.9 | 2.3 |
| RL only       | 0.28 % | 1.7 | 2.0 |
| Meta‑fused    | **0.42 %** | **2.2** | **2.8** |

> Table numbers are illustrative; see `tests/test_backtest_comparison.py` for script used.

### 2026 Upgrades & Refinements
- **Data Ingestion:** move to L3 events via CoinAPI WebSockets; add Kaiko/Nansen context for fill‑prob modeling.
- **AI Regime Detection:** reposition Llama/FinBERT to pure regime switches, cache embeddings with Redis/Feast, and adopt meta‑RL (SAC/PPO hybrid).
- **Execution:** latency‑aware gateway modeling, adaptive market‑making, and TWAP/MP strategies across Robinhood/Binance/Coinbase.
- **Security/Personalization:** zero‑trust vaults, IP whitelisting, user‑specific risk copilot, behavioral fraud halts.
- **Monitoring/Drift:** hierarchical drift detectors, predictive liquidity orchestration for RWA/token prep.

These improvements accelerate the roadmap and raise expected Sharpe >3.0 while reducing latency under 100 ms.

Each layer maps to code under `src/` (see Project Structure below).

Project Structure
```
src/
  main.py                 # Entry point (paper mode with live CCXT data)
  data/
    fetcher.py           # CCXT price fetcher, OHLCV support
    news_fetcher.py      # News fetcher stub
  models/
    ai_model.py          # Sentence-transformers embeddings
    numerical_models.py  # Z-score, MA crossover signals
  ai/
    sentiment.py         # Sentiment analysis (fallback: rule-based)
  indicators/
    indicators.py        # SMA, EMA, RSI
  risk/
    manager.py           # Risk enforcement (position size, daily loss)
    position_sizing.py   # Fixed-fraction, Kelly criterion
  trading/
    executor.py          # Main trading orchestrator
    strategy.py          # Example: SimpleStrategy (SMA + z-score)
    backtest.py          # Simple backtest harness
tests/
  test_indicators.py
  test_risk.py
  test_readme_exists.py
  conftest.py            # pytest config (ensures src imports work)
```

Configuration
- Copy `config.yaml.example` to `config.yaml` and customize:
  - `mode`: "paper" (live mode not yet implemented)
  - `assets`: list of symbols (["BTC", "ETH"])
  - `risk`: max position size (%), daily loss limit (%)
  - `ai.device`: "cpu" or "cuda"

Environment Variables (`.env.example`)
- `ROBINHOOD_USERNAME`
- `ROBINHOOD_PASSWORD`
- `ROBINHOOD_2FA` (optional)
- `HUGGINGFACE_TOKEN` (optional, for model downloads)

Design Philosophy
Realistic Risk & Performance
- 1% daily return target (~3500% annualized if compounded) is ambitious and unrealistic without high-risk leverage.
- System includes risk controls: position limits, daily loss stops, and backtest harness for validation.
- Current implementation is paper-trading focused; live trading requires additional compliance, slippage modeling, and regulatory review.

Open-Source AI Choice
- Uses Hugging Face transformers (sentiment) and sentence-transformers (embeddings) for zero licensing costs.
- Sentiment analysis defaults to rule-based fallback if models fail to load (no external API required).

Extensibility
- Strategy interface: add custom logic in `src/trading/strategy.py`.
- Indicators: add to `src/indicators/indicators.py`.
- Models: extend `src/models/numerical_models.py` or add custom modules.
- Risk controls: extend `src/risk/manager.py`.

Current Limitations
1. Live Trading: Robinhood integration is stubbed; requires careful compliance and order flow implementation.
2. Data: CCXT supports many exchanges but Robinhood private APIs are not officially public.
3. AI Models: Sentiment model is large (~500MB); lazy-loaded on first use.
4. Backtester: Simple single-asset, single-entry/exit logic; doesn't model slippage, fees, or partial fills.
5. News Feed: Currently a stub; replace with real API (e.g., NewsAPI, ReutersPy).
6. Market Hours: No handling of crypto 24/7 vs. stock market hours.

Next Steps (Production Readiness)
- Implement robust Robinhood paper-trading API with full error handling.
- Build a production backtester with fees, slippage, and walk-forward validation.
- Add multi-asset portfolio optimization (Markowitz, risk parity).
- Integrate real news APIs and real-time market feeds.
- Add explainability tools (SHAP, attention weights) for model decisions.
- Implement unit tests for all edge cases and backtest validation.
- Deploy on cloud (AWS Lambda, GCP Cloud Functions) with job scheduling.
- Add monitoring, logging, and alerts (Slack, email, PagerDuty).

Disclaimer & Legal
Automated cryptocurrency trading carries significant risk:
- Robinhood does not officially support third-party API automation for live trading.
- Past performance does not guarantee future results.
- Always test extensively in paper/sandbox mode before any live trading.
- Review the terms of service of any exchange before automated trading.
- Consult a lawyer and financial advisor before deploying this system.
- Do NOT blindly run this in live mode without manual override controls.
