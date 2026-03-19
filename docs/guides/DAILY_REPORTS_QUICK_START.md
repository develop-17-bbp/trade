# ⚡ DAILY REPORTS - QUICK START

## Installation (2 minutes)

### 1. Install Scheduler Library
```bash
pip install schedule
```

### 2. Run Setup (as Administrator)
```bash
# Right-click setup_daily_reports.bat → Run as Administrator
# Or in PowerShell as admin:
.\setup_daily_reports.bat
```

Done! ✅ Reports will now generate automatically every day at 21:00 UTC (9 PM EST)

---

## Use It

### Automatic (Daily at 21:00 UTC)
Reports generate automatically. Check `logs/daily_reports/INDEX.md` next morning.

### Manual (Anytime)
```bash
# Double-click this file to generate a report right now:
generate_daily_report.bat
```

---

## View Reports

1. Open `logs/daily_reports/INDEX.md`
2. Click link to today's report
3. Start with `.txt` file (takes 2 min to read)

---

## Files Generated

Each day you get 3 files in `logs/daily_reports/`:

| File | Type | Read Time | Use For |
|------|------|-----------|---------|
| `report_2026-03-12.txt` | Text | 2 min | Quick overview |
| `report_2026-03-12.md` | Markdown | 10 min | Full details |
| `report_2026-03-12.csv` | CSV | Custom | Excel analysis |

---

## What's in the Reports

### Text Report (.txt)
✓ Total trades today  
✓ Buy/Sell counts  
✓ Win rate %  
✓ Total profit/loss  
✓ Performance vs daily goal  

### Detailed Report (.md)
✓ Everything from text +  
✓ Full trade table  
✓ Asset breakdown  
✓ System status  
✓ Recommendations  

### CSV Export
✓ Raw data  
✓ Every trade from today  
✓ Ready for Excel/Sheets  

---

## Example Daily Review

```
Morning routine (5 minutes):

1. Open: logs/daily_reports/INDEX.md
2. Click: report_YYYY-MM-DD.txt
3. Read: 
   - Total trades?
   - Win rate > 50%?
   - Daily goal progress?
4. Note: Any issues to fix?
5. Done!
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No reports appearing | Run `.\generate_daily_report.bat` manually first |
| "schedule not found" | Run `pip install schedule` |
| Permission denied | Right-click setup_daily_reports.bat → Run as Administrator |
| Wrong time | Edit in Windows Task Scheduler |

---

## Next Steps

✅ Daily monitoring → Check reports each morning  
✅ Track progress → Use daily reports to see if you're hitting +1% daily  
✅ Adjust → If win rate < 50%, debug entry signals  

---

**Full Setup Guide:** See `DAILY_REPORTS_SETUP.md` for details  
**System Status:** All code is production-ready
