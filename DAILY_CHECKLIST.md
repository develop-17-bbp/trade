# QUICK REFERENCE CHECKLIST
## Daily, Weekly, and Monthly Checks

---

## DAILY CHECKLIST (Start of Each Trading Day)

**First Thing Checklist (5 min)**:
- [ ] System running? `ps aux | grep python` → should show `run_training.py`
- [ ] P&L checked? Go to Binance dashboard → Portfolio value
- [ ] No overnight crashes? Check logs: `tail -100 logs/trading_journal.json`
- [ ] Ready to trade? Yes → Proceed to "During Trading"

**During Trading (Every 2-4 hours, 2 min each)**:
- [ ] System still running?
- [ ] Trades executing normally?
- [ ] No error messages in logs?
- [ ] P&L trending positive/expected?

**End of Day Checklist (10 min)**:
- [ ] Record day's P&L: `echo "$____" > daily_pnl/2026_03_12.txt"`
- [ ] Count trades executed: `grep "Trade" logs/trading_journal.json | wc -l`
- [ ] Any unusual trades? Review questionable entries
- [ ] System health good? Uptime, crashes, API latency normal?

---

## WEEKLY CHECKLIST (Every Friday, 1 Hour)

**1. Performance Analysis (20 min)**:
```
Week's P&L: $____
Target P&L: $____ (0.5% of capital)
Win rate: __% (target: 55%+)
Best trade: +$____ (asset: ____)
Worst trade: -$____ (asset: ____)
```

**2. Top 3 Winners Analysis (10 min)**:
- What do they have in common?
  - Similar asset? (BTC/ETH/AAVE)
  - Similar time of day?
  - Similar market condition?
  - Similar confidence level?
- Can we trade MORE of these?

**3. Top 3 Losers Analysis (10 min)**:
- What went wrong?
  - Bad entry signal?
  - Stopped out by noise?
  - Market regime change?
  - API issue?
- How to avoid in future?

**4. System Health (10 min)**:
```
Uptime: _% (target: 99%+)
Crashes: _ (target: 0)
API errors: _ (target: 0)
API latency: __ ms (target: <500ms)
```

**5. Weekly Sign-Off**:
Document in: `weekly_metrics/week_[#]_review.txt`

**Decision**:
- Continue trading? YES / NO
- Change any settings? YES / NO / WHAT
- Try new asset? YES / NO / WHICH

---

## MONTHLY CHECKLIST (First Friday of Month, 2 Hours)

**1. Capital & P&L (15 min)**:
```
Starting capital: $____
Current capital: $____
Month return: __% (target: +1-3%)
Cumulative return: __% (since start)
```

**2. Model Performance (15 min)**:
```
L1 accuracy: __% (target: 70%+)
L2 sentiment correlation: __% (target: 60%+)
L3 risk detection: __% (target: 80%+)
Overall model: STRONG / OK / NEEDS WORK
```

**3. Risk Management (15 min)**:
```
Max single trade loss: -$____ (acceptable: <-$500)
Max daily loss: -$____ (acceptable: <-2%)
Max drawdown: __% (acceptable: <-5%)
Circuit breakers: ACTIVE / INACTIVE
Any violations?: YES / NO
```

**4. Best/Worst Assets (15 min)**:
```
Most profitable: ____ (avg +$____ per trade)
Least profitable: ____ (avg -$____ per trade)
Most traded: ____ (__ trades)
Highest win rate: ____ (_%)
Lowest win rate: ____ (_%

Recommendation:
- Increase trading: ____ (why: _____
- Decrease trading: ____ (why: _____
- Stop trading: ____ (why: _____
```

**5. Model Update Decision (30 min)**:
```
Last retraining: [DATE]
Days since retrain: __
New data available?: YES / NO
Expected accuracy improvement: ___%

DECISION: RETRAIN / SKIP / WAIT
If retrain: Run `python implement_premium_training.py`
```

**6. Scaling Decision (20 min)**:
```
Current capital deployed: $____
Cumulative return: __% (target: +3-5% to scale)
Win rate stable?: YES / NO
Ready to scale?: YES / NO / WAIT

Next capital level:
$10K → $25K: Ready if +5% after 4 weeks
$25K → $50K: Ready if +3% after 2 weeks  
$50K → $100K: Ready if +2% after 1 week
```

**7. System Improvements (30 min)**:
- [ ] Any bugs found? Document in: `bugs_found.md`
- [ ] Any improvements needed? Document in: `improvements.md`
- [ ] Any new data sources? Research in: `new_data_sources.md`
- [ ] Performance optimization ideas? Test in: `optimization_tests.md`

---

## RED FLAG TRIGGERS - IMMEDIATE STOP

**If ANY of these happen, STOP TRADING IMMEDIATELY**:

```
PROFIT PROTECTION:
✗ Daily loss > 3% → STOP TRADING FOR DAY
✗ Weekly loss > 7% → STOP, INVESTIGATE

PERFORMANCE ISSUES:
✗ Win rate < 40% on last 10 trades → PAUSE, DEBUG
✗ 3 consecutive losses > $500 each → PAUSE, CHECK SIGNALS

SYSTEM ISSUES:
✗ Crash/downtime > 10 min → STOP, DIAGNOSE
✗ API latency > 2 seconds → PAUSE, CHECK CONNECTION
✗ 3 rejected trades at exchange → STOP, CHECK API KEY

EXECUTION ISSUES:
✗ Slippage > 1% on 3 trades → PAUSE, CHECK MARKET CONDITIONS
✗ Positions not closing as planned → STOP, MANUAL CLOSE
✗ Unauthorized trades appearing → STOP, CHANGE API KEY

THEN:
1. Close all open positions
2. Pause automated trading
3. Contact if you need help
4. Review what went wrong
5. Get approval before restarting
```

---

## SCALING RULES

**Never skip these steps**:

```
Start: $10,000
├─ Run 1 week
├─ If +5% → SCALE TO $25,000
└─ If -5% → HOLD AT $10K, RUN 1 MORE WEEK

$25,000
├─ Run 2 weeks
├─ If +3% each week → SCALE TO $50,000
└─ If any -3% → HOLD, INVESTIGATE

$50,000
├─ Run 2 weeks
├─ If +2% each week → SCALE TO $100,000
└─ If any -2% → HOLD, REBALANCE

$100,000+
├─ Run indefinitely
├─ Monitor monthly
└─ Can compound profits or withdraw
```

**CRITICAL**: Stick to % targets, don't scale prematurely!

---

## EMERGENCY PROCEDURES

### If System Crashes

**Immediate**:
1. [ ] Check if trades are still open in Binance
2. [ ] If YES, close any losing positions manually
3. [ ] If NO, note the loss and restart

**Recovery**:
```bash
# 1. Check logs for error
tail -50 logs/*.log

# 2. Restart system
python run_training.py

# 3. Verify it's trading again
# (Should open new trades within 5 min)
```

**Wait Before Restarting**:
- If crashed in last 5 min: WAIT 5 min (might just be bouncing back)
- If crashed >1x in 1 hour: WAIT 1 hour (something's wrong)
- If crashed >2x in 1 day: PAUSE TRADING, INVESTIGATE

### If Losing Money Fast

**Red Alert Procedure** (Activate when):
- Daily loss > 2% actual (e.g., -$2000 on $100K)
- 3+ consecutive losses in a row
- Win rate dropped to <40% in last 10 trades

**Steps**:
1. [ ] CLOSE ALL POSITIONS (at market, don't wait)
2. [ ] SET TO PAPER MODE (stop using real capital)
3. [ ] ANALYZE: What went wrong?
4. [ ] ONLY restart after:
   - [ ] Found root cause
   - [ ] Made fixes
   - [ ] Tested on paper for 1 week
   - [ ] Confirmed fixes worked

### If Market Crashes

**If Market Down >10% in 1 Hour**:
- [ ] NOT time to trade during chaos (spreads widen)
- [ ] Activate defensive mode:
  ```yaml
  risk:
    min_confidence: 0.80  # Only ultra-high confidence
    max_position_size_pct: 0.2  # Tiny positions
    risk_per_trade_pct: 0.1  # Minimal risk
  ```

**If Market Down >20% (True Crash)**:
- [ ] CLOSE ALL POSITIONS
- [ ] PAUSE TRADING
- [ ] WAIT 24 hours to see if market stabilizes

---

## PERFORMANCE TRACKING

### Daily Log Template

```
Date: 2026-03-12
Capital: $100,000
Starting P&L: +$5,000

Trades: 15
Wins: 8 (53%)
Losses: 7
Largest win: +$250
Largest loss: -$150

End P&L: +$5,100
Daily return: +0.1%

Notes:
- System stable all day
- One API lag spike (resolved)
- BTC trades performed best
- AAVE trades underperformed

Status: GOOD
```

### Weekly Log Template

```
Week: W#
Capital: $____
Starting P&L: +$____

Total Trades: __
Win Rate: _% (target: 55%+)
Total P&L: +$____ (target: +$500)

Best Trade: _____ @ ____ = +$____
Worst Trade: ____ @ ____ = -$____

Assets Performance:
- BTC: __% win rate
- ETH: __% win rate
- AAVE: __% win rate

System Health:
- Uptime: __% (target: 99%+)
- Crashes: _
- API issues: _

Next Week Plan:
- ______
- ______
- ______

Status: READY TO CONTINUE / PAUSE / SCALE
```

---

## IMPORTANT NUMBERS TO TRACK

**Keep these in a spreadsheet**:

```
Date        Capital   P&L    Daily%   Cumulative%   Win%   Notes
2026-03-15  $100,000  +$500  +0.5%    +0.5%         57%    Scaled from $50K
2026-03-16  $100,000  +$300  +0.3%    +0.8%         55%    Good day
2026-03-17  $100,000  -$200  -0.2%    +0.6%         52%    One bad trade
...
```

**Monthly Summary Template**:
```
MARCH 2026
Start Capital: $100,000
End Capital: $101,200
Return: +1.2%

Trades: 200+
Win Rate: 56%
Best Day: +$500 (March 15)
Worst Day: -$300 (March 22)

Ready for next month?: YES
Any changes needed?: NO
Scale capital?: NO
```

---

## DECISION TREE: WHEN TO PAUSE TRADING

```
P&L negative?
├─ YES, < -1% → Reduce position sizes, keep trading
├─ YES, -1% to -3% → Increase stops, higher confidence
└─ YES, > -3% → STOP: Something's broken

Win rate < 50%?
├─ YES, on 5 trades → Keep watching (-noise)
├─ YES, on 20 trades → Check confidence thresholds
└─ YES, >30 trades → Pause, debug signals

API errors?
├─ YES, 1-2 per day → Acceptable (crypto volatility)
├─ YES, 5+ per day → Check network
└─ YES, continuous → Emergency stop!

Uptime < 99%?
├─ YES, 95-99% → Acceptable (monitor)
├─ YES, 90-95% → Find root cause
└─ YES, <90% → Use reserve capital / pause

Decision:
├─ All metrics good? → CONTINUE TRADING
├─ 1 metric bad? → Investigate
├─ 2+ metrics bad? → PAUSE & FIX
└─ Emergency signal? → IMMEDIATE STOP
```

---

## CONTACTS & RESOURCES

**Documentation Files**:
- `PRODUCTION_READINESS_GUIDE.md` - Full guide (this week)
- `WEEK_BY_WEEK_IMPLEMENTATION_PLAN.md` - 12-week calendar
- `L6_PER_TRADE_REASONING_GUIDE.md` - LLM explanations
- `ARCHITECTURE_9_LAYERS_WITH_L6_REASONING.md` - System architecture

**Configuration**:
- `config.yaml` - Main settings
- `.env` (in root) - API keys (NEVER commit!)

**Code Files**:
- `src/trading/executor.py` - Main trading logic
- `src/risk/profit_protector.py` - Risk management
- `src/monitoring/journal.py` - Trade logging
- `src/monitoring/health_checker.py` - System health

**Logs**:
- `logs/trading_journal.json` - All trades
- `logs/trading_errors.log` - Any errors

---

## SUMMARY

**Key Rules**:
1. ✓ Check daily (5 min) - System running?
2. ✓ Review weekly (1 hr) - Performance OK?
3. ✓ Analyze monthly (2 hr) - Improvements needed?
4. ✓ Start with $10K, scale slowly
5. ✓ Stop at red flags immediately
6. ✓ Document everything
7. ✓ Never skip risk checks

**Target Goals**:
- Week 1-3: Get 55%+ win rate
- Week 4-5: Paper trade successfully
- Week 6-12: Real money > +8-12%

**Success Metrics**:
- P&L: +3-5% per month
- Win Rate: 55-65%
- Sharpe Ratio: 0.3-0.5
- Uptime: 99%+

---

*Print this page or bookmark it!*  
*Check it daily during trading*
