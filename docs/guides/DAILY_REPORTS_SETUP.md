# 📊 DAILY TRADING REPORT SYSTEM
**Automated Daily Logging for Easy Review & Analysis**

---

## Overview

This system automatically generates three types of daily reports at the end of each trading day:

1. **📄 Text Summary** (.txt) - Quick 5-minute read with key metrics
2. **📋 Detailed Report** (.md) - Full analysis with charts and insights  
3. **📊 Data Export** (.csv) - Raw data for Excel/Google Sheets analysis

All reports are saved to `logs/daily_reports/` with date stamps (e.g., `report_2026-03-12.md`)

---

## Quick Start (3 Steps)

### Step 1: Run Setup Once (Administrator)

```bash
# Right-click setup_daily_reports.bat and select "Run as Administrator"
# Or from PowerShell as admin:
.\setup_daily_reports.bat
```

This creates a Windows Task Scheduler job that runs automatically every day at 21:00 UTC.

### Step 2: Generate First Report

```bash
# Option A: Let scheduler run at 21:00 UTC (automatic)
# Option B: Generate manually anytime:
.\generate_daily_report.bat
```

### Step 3: View Reports

Open `logs/daily_reports/INDEX.md` to see all available reports:
- `report_2026-03-12.txt` - Read this first (2 min overview)
- `report_2026-03-12.md` - Read this for details (10 min analysis)
- `report_2026-03-12.csv` - Import into Excel (custom analysis)

---

## What Each Report Shows

### 1. Text Summary (report_2026-03-12.txt)

**Time to Read:** 2 minutes

**Contains:**
```
✓ Total trades executed today
✓ Buy vs Sell breakdown
✓ Trades per asset (BTC/ETH/AAVE)
✓ Performance metrics (Win %, P&L, Profit Factor)
✓ Achievement vs 1% daily goal
✓ System status (all layers online?)
✓ Key insights & recommendations
```

**Best For:** Quick daily review while trading

**Example Output:**
```
OVERALL STATISTICS
==================
Total Trades:           42
Average Confidence:     58.3%

PERFORMANCE METRICS
===================
Total P&L:              $127.35
Win Rate:               52.4%
Profit Factor:          1.245

DAILY TARGET TRACKING
====================
Daily Goal:             +$820.71 (1% on $82,071)
Today's Result:         $127.35
Achievement:            15.5% of target
```

### 2. Detailed Report (report_2026-03-12.md)

**Time to Read:** 10 minutes

**Contains:**
- Executive summary
- Trading statistics table
- All trades from day (in table format)
- Asset breakdown with percentages
- Performance analysis graphs
- System status for all 9 layers
- Next steps recommendations

**Best For:** 
- Understanding why certain trades happened
- Identifying patterns in entry/exit logic
- Detailed debugging of system behavior
- Archive documentation

**Structure:**
```md
# Daily Trading Report

## Executive Summary (Key Metrics)
## Asset Breakdown (Pie Chart Style)
## All Trades Today (Full Table)
## Performance Analysis (Detailed Stats)
## System Status (Layer Verification)
## Next Steps (Recommendations)
```

### 3. Data Export (report_2026-03-12.csv)

**Time to Read:** Variable (custom analysis)

**Contains:**
- Every trade from the day
- Columns: Timestamp, Asset, Side, Quantity, Price, Status, Confidence, etc.
- RAW DATA (no aggregation/summarization)

**Best For:**
- Custom Excel analysis
- Google Sheets pivot tables
- Statistical analysis
- Drawing custom charts
- Data science exploration

**How to Use:**
```
1. Open Google Sheets
2. File → Import → Upload file
3. Select report_2026-03-12.csv
4. Create Pivot Tables for custom analysis
5. Example: Group by Asset, Sum P&L → see performance by coin
```

---

## Setup Instructions

### Option 1: Automatic Setup (Recommended)

```bash
# 1. Open PowerShell as Administrator
# 2. Navigate to project directory:
cd C:\Users\convo\trade

# 3. Run setup script:
.\setup_daily_reports.bat

# Done! Reports will generate at 21:00 UTC daily
```

### Option 2: Manual Schedule (Windows Task Scheduler)

```
1. Open Windows Task Scheduler
2. Right-click "Task Scheduler Library"
3. Select "Create Basic Task..."
4. Name: "Trading Daily Report"
5. Trigger: Daily at 21:00 UTC
6. Action: Run python src\reporting\daily_report_scheduler.py manual
7. Click OK
```

### Option 3: Manual Generation Anytime

```bash
# Generate immediately (don't wait for scheduler):
.\generate_daily_report.bat

# Or from Python:
python -c "from src.reporting.daily_report_generator import DailyReportGenerator; gen = DailyReportGenerator(); gen.save_reports(gen.get_todays_trades(gen.read_trades_from_journal()))"
```

---

## File Structure

```
logs/
├── daily_reports/
│   ├── INDEX.md                    ← START HERE (links to all reports)
│   ├── report_2026-03-12.txt       (today's summary)
│   ├── report_2026-03-12.md        (today's detailed)
│   ├── report_2026-03-12.csv       (today's raw data)
│   ├── report_2026-03-11.txt       (yesterday's summary)
│   ├── report_2026-03-11.md        (yesterday's detailed)
│   ├── report_2026-03-11.csv       (yesterday's raw data)
│   └── ... (30+ days of history)
│
├── monthly_reports/
│   ├── report_2026-03.md           (March summary)
│   ├── report_2026-03.csv          (March data)
│   └── ...
│
└── trading_journal.json            (raw trade data)
```

---

## Understanding the Reports

### Reading the Text Summary

```txt
PERFORMANCE METRICS
===================
Total P&L:              $127.35        ← Money made/lost
Win Rate:               52.4%          ← % of winning trades
Winning Trades:         22             ← # of trades with profit
Losing Trades:          20             ← # of trades with loss
Profit Factor:          1.245          ← Ratio (need > 2.0)
Average Win:            $45.23         ← Avg $ per winning trade
Average Loss:           $36.12         ← Avg $ per losing trade
```

**Interpreting Metrics:**

| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| Win Rate | >55% | 50-55% | <50% |
| Profit Factor | >2.0 | 1.0-2.0 | <1.0 |
| Avg Win:Loss Ratio | >1:1 | 1:1 | <1:1 |
| Daily P&L | >$820 | $400-820 | <$400 |

### Reading the Detailed Report

```md
### ASSET BREAKDOWN
Asset Distribution:
  • BTC    42 trades (33.3%)
  • ETH    40 trades (31.7%)
  • AAVE   36 trades (28.6%)

### TODAY'S TRADES
| # | Timestamp | Asset | Side | Qty | Price | Status | Confidence |
|---|-----------|-------|------|-----|-------|--------|------------|
| 1 | 21:34:38  | BTC   | SELL | ... | ...   | OPEN   | 43.3%      |
```

**Understanding Confidence:**
- 60%+ = Good signal (enter confidently)
- 50-60% = Medium signal (enter cautiously)
- <50% = Weak signal (skip or small position)

### Using the CSV Export

**In Google Sheets:**
```
1. Create Pivot Table
2. Rows: Asset
3. Values: Quantity (sum)
4. Result: See total quantity per asset traded

Alternative:
1. Rows: Side (BUY/SELL)
2. Values: Confidence (average)
3. Result: See confidence by direction
```

---

## Daily Review Checklist

**Time: 5 minutes/day**

```
Every morning, open logs/daily_reports/INDEX.md and:

☐ Read yesterday's .txt report (2 min)
  → Check: Win rate > 50%? Daily goal >80%?
  → Note: What worked? What failed?

☐ Check CSV in Google Sheets (2 min)
  → Chart: Confidence vs Win Rate
  → Pattern: Do higher confidence = more wins?

☐ Detailed .md report (1 min scan)
  → Look at "Recommendations" section
  → Check: Any red flags? Any improvements?

☐ Monthly summary (1x per week)
  → Week trending up or down?
  → Compare to 1% daily goal
```

---

## Customization

### Change Report Time

**Windows Task Scheduler:**
```
1. Open Task Scheduler
2. Find "Trading System - Daily Report"
3. Right-click → Edit
4. Switch to "Triggers" tab
5. Change time (UTC)
6. Click OK
```

**Alternative Times:**
- 21:00 UTC = 4:00 PM EST / 1:00 PM PST (default)
- 23:00 UTC = 6:00 PM EST / 3:00 PM PST
- 04:00 UTC = Previous day 11:00 PM EST

### Change Report Fields

Edit `src/reporting/daily_report_generator.py`:

```python
# Line ~150: Edit generate_markdown_report() to add/remove fields
# Line ~200: Edit generate_text_summary() to change formatting
# Add custom calculations as needed
```

### Send Reports by Email

Create `src/reporting/email_sender.py`:
```python
import smtplib
from email.mime.text import MIMEText

def send_daily_report(report_path, recipient_email):
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Send using Gmail/SMTP
    # (Add your email config here)
```

---

## Troubleshooting

### Reports Not Generating

**Problem:** No files in `logs/daily_reports/`

**Solutions:**
```
1. Check journal file exists:
   ls src/monitoring/journal.csv
   
2. Run manual report:
   .\generate_daily_report.bat
   
3. Check for errors:
   python src/reporting/daily_report_generator.py 2>&1 | tee error.log
   
4. Verify permissions:
   - logs/ directory writable?
   - src/reporting/ readable?
```

### Task Scheduler Not Running

**Problem:** Task created but doesn't run at scheduled time

**Solutions:**
```
1. Check if enabled:
   schtasks /query /tn "Trading System - Daily Report" /v
   
2. Re-create task:
   .\setup_daily_reports.bat
   
3. Check Python path:
   where python
   (Update full path in task if needed)
   
4. Verify network access (if pulling remote data)
```

### CSV Format Issues

**Problem:** Importing to Sheets shows garbled data

**Solutions:**
```
1. Try UTF-8 encoding:
   - In Google Sheets, select "More upload options" → UTF-8
   
2. Try different delimiter:
   - Sheets → File → Import → CSV DELIMITER dropdown
   
3. Check for commas in data:
   - May need to escape/quote fields
```

---

## Advanced Usage

### Generate Reports for Past Days

```python
from src.reporting.daily_report_generator import DailyReportGenerator
from datetime import datetime, timedelta

gen = DailyReportGenerator()

# Get trades for March 10
gen.today = datetime(2026, 3, 10)
trades = gen.read_trades_from_journal()
todays = gen.get_todays_trades(trades)
gen.save_reports(todays)
```

### Create Weekly Summaries

```python
# In src/reporting/weekly_report_generator.py (create this):
from daily_report_generator import DailyReportGenerator
from datetime import datetime, timedelta

def generate_weekly_summary():
    gen = DailyReportGenerator()
    
    # Get all trades this week
    week_trades = []
    for i in range(7):
        gen.today = datetime.now() - timedelta(days=i)
        week_trades.extend(gen.get_todays_trades(...))
    
    # Generate combined report
    return generate_weekly_report(week_trades)
```

### Connect to Email/Slack

```python
# In main(), after generating reports:
import smtplib
from slack import WebClient

# Email reports
send_email_report(md_path, recipients=[])

# Slack notification
slack_client = WebClient(token="xoxb-...")
slack_client.files_upload(
    channels=["#trading"],
    file=txt_path
)
```

---

## API Reference

### DailyReportGenerator Class

```python
from src.reporting.daily_report_generator import DailyReportGenerator

gen = DailyReportGenerator(journal_path="src/monitoring/journal.csv")

# Read trades
trades = gen.read_trades_from_journal()  # All trades ever

# Filter
todays_trades = gen.get_todays_trades(trades)  # Just today

# Analyze
stats = gen.calculate_statistics(todays_trades)
print(f"Win Rate: {stats['win_rate']}%")

# Generate reports
md, csv, txt = gen.save_reports(todays_trades)

# Create index
gen.generate_index()
```

---

## Support & Maintenance

### Daily Maintenance

- **Monitor:** Check .txt report every morning
- **Archive:** Reports auto-saved for 30+ days
- **Cleanup:** Older reports auto-purged (optional)

### Monthly Maintenance

- **Review:** Run weekly summaries
- **Audit:** Compare CSV to trading journal
- **Optimize:** Adjust report fields if needed

### Troubleshooting

For issues, check logs:
```
logs/daily_reports/               # Report files
logs/trading_journal.csv          # Raw data source
error.log                         # If manual generation fails
```

---

**Last Updated:** March 12, 2026  
**Version:** 1.0  
**Status:** Production Ready
