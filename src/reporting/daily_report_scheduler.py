#!/usr/bin/env python3
"""
Daily Report Scheduler
Runs the daily report generator at end of each trading day (21:00 UTC)
Can run as a daemon or be called by Windows Task Scheduler / cron
"""

import schedule
import time
from datetime import datetime
from pathlib import Path
import sys

def run_daily_report():
    """Execute the daily report generator"""
    print(f"\n{'='*60}")
    print(f"📊 DAILY REPORT GENERATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    try:
        # Import the generator
        from src.reporting.daily_report_generator import DailyReportGenerator
        
        # Generate reports
        generator = DailyReportGenerator()
        
        # Read trades
        all_trades = generator.read_trades_from_journal()
        print(f"✅ Loaded {len(all_trades)} total trades")
        
        # Get today's trades
        todays_trades = generator.get_todays_trades(all_trades)
        print(f"✅ Found {len(todays_trades)} trades for today")
        
        if todays_trades:
            # Generate and save reports
            md_path, csv_path, txt_path = generator.save_reports(todays_trades)
            print(f"\n✅ All reports generated successfully!\n")
            
            # Print quick summary
            summary_path = txt_path.replace('.txt', '_summary.txt')
            print(f"📄 Quick Summary (saved to {txt_path}):")
            with open(txt_path, 'r') as f:
                lines = f.readlines()[:20]
                for line in lines:
                    print(f"   {line.rstrip()}")
        else:
            print(f"⚠️  No trades found for today - skipping report generation\n")
    
    except Exception as e:
        print(f"❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()


def schedule_reports():
    """Schedule daily report generation"""
    
    # Schedule at 21:00 UTC (end of trading day)
    schedule.every().day.at("21:00").do(run_daily_report)
    
    # Also schedule for backup times
    schedule.every().day.at("23:59").do(run_daily_report)  # End of day
    
    print("📅 Report Scheduler Started")
    print(f"⏰ Reports will be generated at:")
    print(f"   • 21:00 UTC (daily)")
    print(f"   • 23:59 UTC (end-of-day backup)")
    print(f"\n⏸️  Press CTRL+C to stop\n")
    
    # Run scheduler loop
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


def run_manual_report():
    """Run report immediately (for manual testing)"""
    print("🚀 Running manual daily report generation...\n")
    run_daily_report()
    print("\n✅ Manual report generation complete!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "manual":
        # Run immediately for testing
        run_manual_report()
    else:
        # Run scheduler
        try:
            schedule_reports()
        except KeyboardInterrupt:
            print("\n\n⏹️  Report scheduler stopped")
