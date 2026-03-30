# AUTONOMOUS LLM-DRIVEN CRYPTO TRADING SYSTEM — COMPLETE BUILD PROMPT

## ROLE
You are an expert quant developer + ML engineer + trading systems architect. Build a complete, production-ready, autonomous crypto trading system that uses a locally-hosted LLM to make ALL trading decisions based on a specific EMA Reversal Strategy.

---

## SYSTEM OVERVIEW

Build a Python trading system that:
1. Connects to **Bybit Testnet** (USDT-margined linear perpetual futures) via `ccxt`
2. Fetches live 1-minute OHLCV candle data continuously
3. Computes EMA(8) and ATR(14) indicators on every candle
4. Sends candle data + indicators to a **remote LLM** (Ollama on GPU server)
5. The LLM decides: **LONG (CALL)** / **SHORT (PUT)** / **FLAT (HOLD)**
6. The LLM also decides: **order type**, **position size**, **limit price**
7. Executes trades on Bybit via API (both LONG and SHORT supported)
8. Manages positions with a **dynamic trailing stop-loss** system (L1→L2→L3→L4...)
9. Exits on **EMA reversal** (E1 pattern) or **stop-loss hit**
10. Logs every trade decision with full audit trail

---

## EXCHANGE CONFIGURATION

```
Exchange: Bybit Testnet
API: ccxt.bybit with sandbox=True
Market type: linear (USDT-margined perpetual futures)
recvWindow: 60000 (60s time sync tolerance)
enableRateLimit: True
Symbols: BTC/USDT:USDT, ETH/USDT:USDT
Auth: BYBIT_TESTNET_KEY and BYBIT_TESTNET_SECRET from environment variables
Time sync: Call exchange.load_time_difference() before any authenticated request
```

**Why Bybit:** Alpaca does not support crypto short-selling. Bybit futures support both LONG and SHORT positions from USD balance, enabling the full CALL+PUT strategy.

---

## LLM CONFIGURATION

```
Provider: Ollama (remote GPU server)
Model: llama3.2:latest
Endpoint: Configurable via OLLAMA_REMOTE_URL env var (or config.yaml)
Format: Send prompt, receive JSON response
Timeout: 30 seconds per inference
Rate limit: 15 calls per minute, 60-second window
Fallback: If remote fails, try local Ollama at http://localhost:11434
```

---

## STRATEGY LOGIC — EMA REVERSAL CROSSOVER (CRITICAL — IMPLEMENT EXACTLY)

### Core Concept
The strategy detects **EMA(8) crossover patterns** on 1-minute candles and trades in the direction of the confirmed crossover. It works in BOTH directions (LONG and SHORT).

### INDICATORS
| Indicator | Period | Formula | Purpose |
|-----------|--------|---------|---------|
| EMA | 8 | Exponential Moving Average of close prices | Trend direction + crossover detection |
| SMA | 20 | Simple Moving Average of close prices | Trend confirmation (secondary) |
| ATR | 14 | Average True Range | Volatility measurement, stop-loss sizing |

### DOWNTREND ENTRY — PUT / SHORT (P1)
**ALL 3 conditions must be true simultaneously:**
1. **Previous candle**: EMA(8) line crosses THROUGH it (EMA value is between candle's Low and High)
   - `candle[i-1].low <= ema[i-1] <= candle[i-1].high`
2. **Current candle**: Forms ENTIRELY BELOW the EMA line
   - `candle[i].high < ema[i]`
3. **EMA direction**: Descending (falling)
   - `ema[i] < ema[i-1]`

**Action:** SELL SHORT at current price → This is the P1 entry point
**Mark this candle with `*CROSS*` in the data sent to LLM**

### UPTREND ENTRY — CALL / LONG (P1)
**Mirror logic — ALL 3 conditions must be true:**
1. **Previous candle**: EMA(8) line crosses THROUGH it
   - `candle[i-1].low <= ema[i-1] <= candle[i-1].high`
2. **Current candle**: Forms ENTIRELY ABOVE the EMA line
   - `candle[i].low > ema[i]`
3. **EMA direction**: Ascending (rising)
   - `ema[i] > ema[i-1]`

**Action:** BUY LONG at current price → This is the P1 entry point

### EXIT LOGIC — E1 (Reversal Confirmed)
Exit when the EXACT REVERSE of the entry pattern occurs:

**Exit LONG position:**
- EMA crosses through current candle (bullish-to-bearish transition)
- Next candle forms entirely BELOW EMA
- EMA is now FALLING
→ Close LONG position at market

**Exit SHORT position:**
- EMA crosses through current candle (bearish-to-bullish transition)
- Next candle forms entirely ABOVE EMA
- EMA is now RISING
→ Close SHORT position at market

---

## DYNAMIC TRAILING STOP-LOSS SYSTEM (CORE INNOVATION)

This is the most important part of the strategy. The stop-loss progressively locks in profits.

### Definitions
| Level | Meaning |
|-------|---------|
| L1 | Initial stop-loss at first structure point (recent swing high for SHORT, swing low for LONG) |
| L2 | Updated stop-loss after price moves favorably past L1 |
| L3, L4, L5... | Progressively tighter stops as profits accumulate |
| E1 | Exit point — where the trade is closed (at last SL level or EMA reversal) |

### Rules for LONG (BUY) Trades
```
1. L1 = Recent swing LOW before entry (structure lookback = 15 candles)
   - Safety margin: L1 = swing_low * 0.999

2. As price rises above entry:
   - When unrealized profit > 0.2% of entry:
     → Start trailing the stop-loss upward
   - Find new structure LOW (higher than current SL)
   - Move SL up to new structure point → becomes L2

3. Max giveback rule:
   - Track peak profit (highest price reached since entry)
   - max_giveback = peak_profit * 0.30 (30% of max favorable excursion)
   - profit_trail_SL = peak_price - max_giveback
   - If profit_trail_SL > current_SL → move SL up

4. Repeat: L2 → L3 → L4 → L5...
   - Each new SL level MUST be higher than the previous
   - SL can ONLY move in the direction of profit (upward for LONG)

5. Exit conditions:
   - Price drops below last SL level → EXIT at market
   - EMA reversal confirmed (E1 pattern) → EXIT at market
   - Whichever comes first
```

### Rules for SHORT (SELL) Trades
```
1. L1 = Recent swing HIGH before entry (structure lookback = 15 candles)
   - Safety margin: L1 = swing_high * 1.001

2. As price falls below entry:
   - When unrealized profit > 0.2%:
     → Start trailing the stop-loss downward
   - Find new structure HIGH (lower than current SL)
   - Move SL down → becomes L2

3. Max giveback rule:
   - Track peak profit (lowest price reached since entry)
   - max_giveback = peak_favorable * 0.30
   - profit_trail_SL = peak_favorable + max_giveback
   - If profit_trail_SL < current_SL → move SL down

4. Repeat: L2 → L3 → L4 → L5...
   - Each new SL level MUST be lower than the previous (for SHORT)

5. Exit conditions:
   - Price rises above last SL level → EXIT at market
   - EMA reversal confirmed → EXIT at market
```

### CRITICAL BEHAVIORAL CONSTRAINT
```
The system must ALWAYS prioritize:
"Maximize profit by aggressively trailing stop-loss while protecting accumulated gains."

- NEVER exit too early (only on confirmed reversal or SL hit)
- ALWAYS push stop-loss in direction of profit
- ONLY exit on confirmed EMA reversal OR SL breach
- Losses from any trade must be coverable by profits locked in via trailing SL
- The buffer of profit must always exceed potential loss
```

---

## LLM PROMPT TEMPLATE

Send this exact prompt structure to the LLM on every candle:

```
You are a crypto trading AI. Analyze this {SYMBOL} 1-minute data and decide: LONG, SHORT, or FLAT.

=== MARKET DATA (Last 20 candles) ===
 # |    Open |    High |     Low |   Close |   EMA(8) | Signal
 1 | 68900.0 | 68950.0 | 68880.0 | 68920.0 | 68915.00 |
 2 | 68920.0 | 68960.0 | 68890.0 | 68940.0 | 68918.12 |
...
19 | 69100.0 | 69150.0 | 69050.0 | 69120.0 | 69045.00 | *CROSS*
20 | 69120.0 | 69180.0 | 69100.0 | 69160.0 | 69059.00 |

=== CURRENT STATE ===
Price: $69,160.00 | EMA(8): $69,059.00 | ATR(14): $45.20
EMA direction: RISING | Price vs EMA: ABOVE
Resistance: $69,200, $69,350, $69,500
Support: $68,900, $68,750, $68,600
Volume: 1.2x average (20-bar)
Consecutive green candles: 4

=== EMA CROSSOVER RULES ===
ENTRY RULES:
- LONG (CALL): Previous candle has *CROSS* marker AND current candle is ENTIRELY ABOVE EMA AND EMA is RISING
- SHORT (PUT): Previous candle has *CROSS* marker AND current candle is ENTIRELY BELOW EMA AND EMA is FALLING
- If NO *CROSS* marker in recent candles → FLAT (no trade)

EXIT RULES:
- Exit LONG when: EMA reversal — candle crosses below EMA + next candle entirely below + EMA falling
- Exit SHORT when: EMA reversal — candle crosses above EMA + next candle entirely above + EMA rising

STOP-LOSS RULES:
- Use dynamic trailing SL (L1→L2→L3→L4)
- Aggressively push SL in profit direction
- Max giveback: 30% of peak profit
- NEVER widen SL — only tighten

=== RESPOND WITH VALID JSON ONLY ===
{
  "action": "LONG" or "SHORT" or "FLAT",
  "order_type": "market" or "limit" or "stop" or "stop_limit" or "trailing_stop",
  "confidence": 0.0 to 1.0,
  "position_size_pct": 2.0 to 10.0,
  "limit_price": <price or 0 for market>,
  "stop_loss_price": <initial SL price>,
  "reasoning": "<cite specific candle numbers and *CROSS* markers>"
}
```

### `*CROSS*` Marker Logic
For each candle in the table, mark it with `*CROSS*` if:
```python
if candle.low <= ema_value <= candle.high:
    signal = "*CROSS*"
```
This tells the LLM exactly which candles have EMA crossover events.

---

## POSITION SIZING

| Parameter | Value | Description |
|-----------|-------|-------------|
| Default size | 5% of equity | Standard position |
| Minimum size | 2% of equity | Floor |
| Maximum size | 10% of equity | Ceiling |
| LLM can adjust | Yes | Within 2-10% range based on confidence |

```python
position_value = equity * (position_size_pct / 100)
qty = position_value / current_price
```

---

## ORDER TYPES (ALL 5 SUPPORTED)

| Order Type | When to Use | Implementation |
|------------|------------|----------------|
| **Market** | Urgent entries/exits, SL hits | `exchange.create_order(symbol, 'market', side, qty)` |
| **Limit** | Better entry price, BUY below market / SELL above market | `exchange.create_order(symbol, 'limit', side, qty, limit_price)` |
| **Stop** | Stop-loss execution on exchange | `exchange.create_order(symbol, 'stop', side, qty, None, {'stopPrice': price})` |
| **Stop Limit** | Controlled SL with max slippage | `exchange.create_order(symbol, 'stop_limit', side, qty, limit_price, {'stopPrice': trigger})` |
| **Trailing Stop** | Dynamic exchange-side trailing | `exchange.create_order(symbol, 'trailing_stop', side, qty, None, {'trailingValue': trail_pct})` |

### Limit Price Logic
```python
# For LONG/BUY entries: bid slightly below market for better fill
limit_price = current_price * 0.9995  # 0.05% below

# For SHORT/SELL entries: offer slightly above market
limit_price = current_price * 1.0005  # 0.05% above

# For closing positions: use MARKET orders (speed > price)
```

---

## QUALITY GATES (PREVENT BAD TRADES)

| Gate | Threshold | Purpose |
|------|-----------|---------|
| Confidence minimum | 0.70 | Don't trade on weak signals |
| ATR/Price ratio | > 0.03% | Enough volatility to cover spread |
| Trade cooldown | 60 seconds | Prevent overtrading per asset |
| Post-close cooldown | 120 seconds | Wait after closing before new entry |
| EMA crossover required | Must have `*CROSS*` | No entry without confirmed pattern |

```python
# Reject trade if ANY gate fails:
if confidence < 0.70: SKIP
if atr / price < 0.0003: SKIP
if time_since_last_trade < 60: SKIP
if time_since_last_close < 120: SKIP
if no_cross_marker_in_recent_candles: SKIP
```

---

## RISK MANAGEMENT

| Parameter | Value | Description |
|-----------|-------|-------------|
| Daily loss limit | 3.0% of equity | Halt trading for the day |
| Max drawdown | 10.0% of initial capital | Emergency shutdown |
| Risk per trade | 0.5% of equity | Max loss per single trade |
| ATR stop multiplier | 1.5x | SL distance = 1.5 * ATR |
| ATR take-profit multiplier | 4.5x | TP distance = 4.5 * ATR (only if no EMA exit) |

```python
# Daily loss check
if daily_pnl <= -(equity * 0.03):
    halt_trading("Daily loss limit hit")

# Max drawdown check
if equity <= initial_capital * 0.90:
    shutdown("Max drawdown exceeded")
```

---

## DATA PIPELINE

### Live Data (Every 1-minute candle)
```python
# Fetch latest OHLCV from Bybit
ohlcv = exchange.fetch_ohlcv('BTC/USDT:USDT', timeframe='1m', limit=200)

# Convert to pandas DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Compute indicators
df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
df['sma20'] = df['close'].rolling(20).mean()
df['atr14'] = compute_atr(df, period=14)
```

### Fast Price Data (Optional secondary source)
```
LiveCoinWatch API:
  URL: https://api.livecoinwatch.com
  Auth: x-api-key header
  Endpoint: /coins/single (current price)
  Endpoint: /coins/single/history (historical)
  Timeout: 5 seconds
```

### OHLCV Cache
```python
# Cache OHLCV data for 30 seconds to avoid redundant API calls
ohlcv_cache_ttl = 30  # seconds
```

---

## POSITION STATE MACHINE

```
States:
  FLAT     → No position, waiting for entry signal
  LONG     → Holding a BUY position, trailing SL upward
  SHORT    → Holding a SELL position, trailing SL downward

Transitions:
  FLAT  → LONG:  EMA crossover UP confirmed + LLM says LONG + confidence >= 0.70
  FLAT  → SHORT: EMA crossover DOWN confirmed + LLM says SHORT + confidence >= 0.70
  LONG  → FLAT:  EMA reversal (E1) OR stop-loss hit OR daily limit
  SHORT → FLAT:  EMA reversal (E1) OR stop-loss hit OR daily limit
  LONG  → SHORT: Close LONG first, then enter SHORT (never simultaneous)
  SHORT → LONG:  Close SHORT first, then enter LONG
```

### State Tracking Per Asset
```python
active_trades = {
    'BTC': {
        'side': 'long',           # 'long', 'short', or None
        'entry_price': 69160.0,
        'entry_time': '2026-03-27T10:15:00',
        'qty': 0.072,
        'sl_levels': [68900.0, 69050.0, 69120.0],  # L1, L2, L3
        'current_sl': 69120.0,    # Latest SL
        'peak_price': 69500.0,    # Max favorable excursion
        'order_id': 'abc123',
    },
    'ETH': None  # No position
}
```

---

## LOGGING & EVALUATION

### Trade Journal (JSON Lines format)
Every trade must log to `logs/trading_journal.jsonl`:
```json
{
  "timestamp": "2026-03-27T10:15:00Z",
  "asset": "BTC",
  "action": "LONG",
  "entry_price": 69160.0,
  "exit_price": 69500.0,
  "qty": 0.072,
  "pnl_usd": 24.48,
  "pnl_pct": 0.49,
  "sl_progression": ["L1:68900", "L2:69050", "L3:69120"],
  "exit_reason": "EMA_REVERSAL",
  "llm_reasoning": "Previous candle [-2] has *CROSS* at EMA 69045...",
  "confidence": 0.85,
  "order_type": "limit",
  "duration_minutes": 12,
  "order_id": "abc123"
}
```

### Performance Summary (generated after each trade closes)
```
Total trades: 15
Win rate: 60%
Total P&L: +$342.50
Avg profit per winning trade: +$45.20
Avg loss per losing trade: -$18.30
Max drawdown: -$85.00
Profit factor: 2.47
Longest winning streak: 4
Best trade: +$120.00 (BTC LONG)
Worst trade: -$35.00 (ETH SHORT)
```

### Console Dashboard (printed every candle)
```
[BAR 42] Equity: $100,342.50 | Cash: $95,342.50 | Return: +0.34% | Positions: 1
  [BTC] $69,500.00 | EMA: $69,350.00 | RISING | Signal: NONE | ATR: $45.20
  [BTC] HOLD LONG @ $69,160.00 | Now: $69,500.00 | SL: L1=$68,900 → L2=$69,050 → L3=$69,120 | P&L: +0.49%
  [ETH] $2,085.00 | EMA: $2,070.00 | FALLING | Signal: SELL | ATR: $5.20
  [SLEEP] 55s
```

---

## COMPLETE TRADING LOOP (PSEUDOCODE)

```python
def main_loop():
    while True:
        for asset in ['BTC', 'ETH']:
            symbol = f"{asset}/USDT:USDT"

            # 1. Fetch latest candles
            ohlcv = fetch_ohlcv(symbol, '1m', limit=200)

            # 2. Compute indicators
            ema8 = compute_ema(ohlcv.close, period=8)
            atr14 = compute_atr(ohlcv, period=14)

            # 3. Detect EMA crossover pattern
            cross_detected = check_ema_crossover(ohlcv, ema8)

            # 4. Check if we have an active position
            if has_position(asset):
                # Manage existing position
                check_ema_reversal_exit(asset, ohlcv, ema8)
                update_trailing_sl(asset, ohlcv, ema8, atr14)
                check_sl_hit(asset, current_price)
            else:
                # Look for new entry
                if cross_detected:
                    # Build prompt with last 20 candles + indicators
                    prompt = build_llm_prompt(asset, ohlcv, ema8, atr14)

                    # Ask LLM for decision
                    decision = query_llm(prompt)

                    # Validate quality gates
                    if passes_quality_gates(decision, atr14, asset):
                        # Execute trade
                        execute_trade(asset, symbol, decision)

        # Sleep until next candle
        sleep_until_next_minute()
```

---

## TECHNICAL REQUIREMENTS

### Dependencies
```
pandas >= 2.0
numpy >= 1.24
ccxt >= 4.0
requests >= 2.28
python-dotenv >= 1.0
```

### File Structure
```
trade/
├── .env                    # API keys (gitignored)
├── config.yaml             # All configurable parameters
├── src/
│   ├── main.py             # Entry point + main loop
│   ├── data/
│   │   └── fetcher.py      # Bybit + LiveCoinWatch data
│   ├── ai/
│   │   ├── llm_provider.py # Ollama LLM interface
│   │   └── agentic_strategist.py  # LLM prompt + decision parsing
│   ├── trading/
│   │   ├── executor.py     # Trade execution + position management
│   │   ├── strategy.py     # EMA crossover detection
│   │   └── sub_strategies.py  # Entry/exit/SL logic
│   └── monitoring/
│       └── journal.py      # Trade logging
├── logs/
│   ├── trading_journal.jsonl
│   └── performance_summary.json
└── memory/                 # LLM experience vault
```

### Environment Variables
```bash
# Required
BYBIT_TESTNET_KEY=<your-key>
BYBIT_TESTNET_SECRET=<your-secret>

# Optional
LIVECOINWATCH_API_KEY=<your-key>
OLLAMA_REMOTE_URL=https://your-gpu-server.com
OLLAMA_REMOTE_MODEL=llama3.2:latest
```

---

## BEHAVIORAL RULES FOR THE LLM

The LLM must follow these rules when making decisions:

1. **Only cite what you see**: Reference specific candle numbers and `*CROSS*` markers
2. **No hallucinated signals**: If no `*CROSS*` marker exists in recent candles → output FLAT
3. **Respect the pattern**: LONG only when price is ABOVE EMA and EMA is RISING
4. **Respect the pattern**: SHORT only when price is BELOW EMA and EMA is FALLING
5. **Never fight the trend**: Don't go LONG in a falling EMA, don't go SHORT in a rising EMA
6. **Confidence calibration**: High confidence (0.8+) only for textbook crossover patterns
7. **Position sizing**: Scale position size with confidence (low conf = small size)
8. **Prefer limit orders**: For entries, use limit orders 0.05% better than market
9. **Use market orders**: For exits and SL hits (speed matters)
10. **Always provide reasoning**: Explain which candle numbers and indicators drove the decision

---

## VALIDATION CHECKLIST

Before deploying, verify:
- [ ] Bybit testnet connection works (both LONG and SHORT)
- [ ] EMA(8) computation matches TradingView EMA(8)
- [ ] `*CROSS*` markers appear at correct candles
- [ ] Entry signals fire only on confirmed crossover (all 3 conditions)
- [ ] Exit signals fire only on confirmed reversal
- [ ] Stop-loss trails in profit direction only
- [ ] Stop-loss never widens (only tightens)
- [ ] L1→L2→L3→L4 progression logged correctly
- [ ] Position size within 2-10% bounds
- [ ] Quality gates block low-confidence trades
- [ ] Daily loss limit halts trading
- [ ] Trade journal captures all required fields
- [ ] Performance summary computes correctly

---

## REFERENCE CHART EXPLANATION

Refer to the annotated TradingView chart (ETH/USD 1-min on Bitstamp):

- **P1**: Entry point — EMA crosses through previous candle, next candle forms below EMA → SHORT entry
- **L1**: Initial stop-loss at recent swing high above entry
- **L2**: First SL update — moved down as trade becomes profitable
- **L3**: Second SL update — locked in more profit
- **L4**: Third SL update — price reached maximum favorable excursion
- **E1**: Exit point — price reversed back to L3 level, trade closed at profit
- **P2**: New entry after E1 — fresh crossover pattern detected

The entire L1→L2→L3→L4→E1 cycle ensures that profits are progressively locked in and losses are always covered by previously secured gains.
