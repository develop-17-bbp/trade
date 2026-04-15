"""
ACT Daily Operations — Autonomous System Health & Maintenance
================================================================
Runs daily to ensure the trading system stays healthy, profitable,
and self-correcting. Designed to run unattended via Task Scheduler.

Operations:
  1. Check/restart crashed processes (handles Windows file locks)
  2. Compute daily P&L with EMA and spread metrics
  3. Auto-trigger retraining if win rate < 45% or drawdown > threshold
  4. Validate Cloudflare tunnel connectivity to Ollama instances
  5. Check disk space and rotate logs older than 7 days
  6. Write structured daily report to reports/daily-YYYY-MM-DD.json
  7. Flag issues for human review only if a fix fails twice

Usage:
    python -m src.scripts.daily_ops                 # Single run
    python -m src.scripts.daily_ops --continuous     # Run every 4h
    python -m src.scripts.daily_ops --dry-run        # Report only, no actions
"""

import os
import sys
import json
import time
import glob
import shutil
import signal
import socket
import logging
import argparse
import subprocess
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('daily_ops')

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
REPORTS_DIR = PROJECT_ROOT / 'reports'
LOGS_DIR = PROJECT_ROOT / 'logs'
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'

# Thresholds
WIN_RATE_THRESHOLD = 0.45       # Auto-retrain if below
MAX_DRAWDOWN_THRESHOLD = 0.08   # 8% max drawdown trigger
LOG_RETENTION_DAYS = 7
DISK_SPACE_WARN_GB = 5          # Warn if less than 5GB free

# Process definitions — what should be running
# 'check' = substring to find in wmic command line output
# 'start_cmd' = how START_ALL.bat actually launches (must match window titles for STOP_ALL)
EXPECTED_PROCESSES = [
    {'name': 'Trading Bot', 'check': 'src.main', 'port': None,
     'window_title': 'ACTs - Trading Bot',
     'start_cmd': 'cmd /k "cd /d C:\\Users\\convo\\trade && set PYTHONUNBUFFERED=1 && python -m src.main"'},
    {'name': 'API Server', 'check': 'production_server', 'port': 11007,
     'window_title': 'ACTs - API Server',
     'start_cmd': 'cmd /k "cd /d C:\\Users\\convo\\trade && set TRADE_API_DEV_MODE=1 && python -m uvicorn src.api.production_server:app --host 0.0.0.0 --port 11007"'},
    {'name': 'Continuous Adapt', 'check': 'continuous_adapt', 'port': None,
     'window_title': 'ACTs - Continuous Adapt',
     'start_cmd': 'cmd /k "cd /d C:\\Users\\convo\\trade && python -m src.scripts.continuous_adapt --continuous --interval 0.5"'},
    {'name': 'Monitor', 'check': 'run_monitor', 'port': None,
     'window_title': 'ACTs - Monitor', 'process_name': 'cmd.exe',
     'start_cmd': 'cmd /k "cd /d C:\\Users\\convo\\trade && scripts\\windows\\run_monitor.bat"'},
]


class DailyOps:
    """Autonomous daily operations manager."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.report: Dict[str, Any] = {
            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'dry_run': dry_run,
            'steps': {},
            'actions_taken': [],
            'human_review_needed': [],
            'overall_status': 'GREEN',
        }
        self._retry_tracker_path = DATA_DIR / 'ops_retry_tracker.json'
        self._retry_tracker = self._load_retry_tracker()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════
    # Step 1: Process Health Check & Auto-Restart
    # ══════════════════════════════════════════════════════════════

    def step1_check_processes(self) -> Dict:
        """Check all expected processes and restart crashed ones."""
        logger.info("Step 1: Checking running processes...")
        result = {'status': 'GREEN', 'processes': {}, 'restarts': []}

        for proc_def in EXPECTED_PROCESSES:
            name = proc_def['name']
            status = self._check_process(proc_def)
            result['processes'][name] = status

            if not status['running']:
                logger.warning(f"  {name}: NOT RUNNING")
                result['status'] = 'YELLOW'

                if not self.dry_run:
                    success = self._restart_process(proc_def)
                    if success:
                        result['restarts'].append(name)
                        self.report['actions_taken'].append(f"Restarted {name}")
                        logger.info(f"  {name}: RESTARTED")
                    else:
                        # Check retry tracker
                        if self._should_escalate(f'restart_{name}'):
                            result['status'] = 'RED'
                            self.report['human_review_needed'].append(
                                f"{name} failed to restart twice — manual intervention needed"
                            )
                        else:
                            self.report['actions_taken'].append(
                                f"{name} restart failed (attempt 1) — will retry next cycle"
                            )
            else:
                logger.info(f"  {name}: OK (PID {status.get('pid', '?')})")

        self.report['steps']['processes'] = result
        return result

    def _check_process(self, proc_def: Dict) -> Dict:
        """Check if a process is running.

        Detection methods (in order):
        1. Port check (API server — most reliable)
        2. Window title check via tasklist (matches START_ALL.bat window titles)
        3. Command line check via wmic/powershell (fallback)
        """
        status = {'running': False, 'pid': None, 'details': ''}

        # Method 1: Check by port if applicable (most reliable for API server)
        if proc_def.get('port'):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(('127.0.0.1', proc_def['port']))
                sock.close()
                if result == 0:
                    status['running'] = True
                    status['details'] = f"Port {proc_def['port']} responding"
                    return status
            except Exception:
                pass

        # Method 2: Check by window title (matches how START_ALL.bat names windows)
        title = proc_def.get('window_title', '')
        if title:
            try:
                output = subprocess.check_output(
                    ['tasklist', '/FI', f'WINDOWTITLE eq {title}*', '/FO', 'CSV', '/NH'],
                    text=True, timeout=10,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                )
                # tasklist returns "INFO: No tasks..." when nothing found
                if output.strip() and 'No tasks' not in output and 'INFO' not in output:
                    # Parse PID from CSV: "cmd.exe","1234","Console","1","..."
                    for line in output.strip().split('\n'):
                        parts = line.strip('"').split('","')
                        if len(parts) >= 2:
                            pid = parts[1].strip('"')
                            status['running'] = True
                            status['pid'] = pid
                            status['details'] = f"Window '{title}' found (PID {pid})"
                            return status
            except Exception:
                pass

        # Method 3: Check command line via PowerShell (works on modern Windows)
        check_str = proc_def.get('check', '')
        if check_str:
            try:
                proc_name = proc_def.get('process_name', 'python.exe')
                ps_cmd = (
                    f'Get-CimInstance Win32_Process -Filter "name=\'{proc_name}\'" | '
                    f'Where-Object {{ $_.CommandLine -match \'{check_str}\' }} | '
                    f'Select-Object ProcessId -First 1 | '
                    f'ForEach-Object {{ $_.ProcessId }}'
                )
                output = subprocess.check_output(
                    ['powershell', '-NoProfile', '-Command', ps_cmd],
                    text=True, timeout=15,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                )
                pid = output.strip()
                if pid and pid.isdigit():
                    status['running'] = True
                    status['pid'] = pid
                    status['details'] = f"Found '{check_str}' in process list (PID {pid})"
                    return status
            except Exception:
                pass

            # Method 3b: Legacy wmic fallback
            try:
                wmic_out = subprocess.check_output(
                    ['wmic', 'process', 'where', "name='python.exe'", 'get',
                     'processid,commandline', '/format:csv'],
                    text=True, timeout=10,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                )
                for line in wmic_out.strip().split('\n'):
                    if check_str in line.lower():
                        parts = line.strip().split(',')
                        pid = parts[-1].strip() if parts else '?'
                        status['running'] = True
                        status['pid'] = pid
                        status['details'] = f"Found via wmic (PID {pid})"
                        return status
            except Exception:
                pass

        return status

    def _restart_process(self, proc_def: Dict) -> bool:
        """Attempt to restart a process using the same method as START_ALL.bat.

        Uses 'start "WindowTitle" /MIN cmd /k ...' so that:
        - STOP_ALL.bat can find processes by window title
        - Process runs in its own minimized window
        - Matches exactly how START_ALL.bat launches components
        """
        try:
            title = proc_def.get('window_title', f'ACTs - {proc_def["name"]}')
            cmd = proc_def['start_cmd']
            # Use START /MIN with the correct window title (matches STOP_ALL.bat taskkill)
            full_cmd = f'start "{title}" /MIN {cmd}'
            subprocess.Popen(
                full_cmd, shell=True,
                cwd=str(PROJECT_ROOT),
            )
            # Wait longer for startup (bot takes ~8s to init all 25 subsystems)
            time.sleep(10)
            status = self._check_process(proc_def)
            return status['running']
        except Exception as e:
            logger.error(f"Restart failed for {proc_def['name']}: {e}")
            return False

    # ══════════════════════════════════════════════════════════════
    # Step 2: Daily P&L Computation
    # ══════════════════════════════════════════════════════════════

    def step2_compute_daily_pnl(self) -> Dict:
        """Pull trade data and compute daily P&L with EMA and spread metrics."""
        logger.info("Step 2: Computing daily P&L...")
        result = {
            'status': 'GREEN', 'total_pnl_pct': 0, 'total_pnl_usd': 0,
            'trades_today': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
            'spread_cost_pct': 1.69, 'pnl_after_spread': 0,
            'ema_20_pnl': 0, 'max_drawdown': 0,
        }

        journal_path = LOGS_DIR / 'trading_journal.jsonl'
        if not journal_path.exists():
            result['status'] = 'YELLOW'
            result['details'] = 'No trading journal found'
            self.report['steps']['daily_pnl'] = result
            return result

        # Load all trades
        all_trades = []
        today = datetime.now().strftime('%Y-%m-%d')
        try:
            with open(journal_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        trade = json.loads(line)
                        all_trades.append(trade)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Journal read failed: {e}")

        if not all_trades:
            result['details'] = 'Journal empty'
            self.report['steps']['daily_pnl'] = result
            return result

        # Filter today's trades
        today_trades = [t for t in all_trades if t.get('close_time', t.get('timestamp', '')).startswith(today)]

        # Compute P&L
        pnls = [t.get('pnl_pct', 0) for t in all_trades if t.get('pnl_pct') is not None]
        today_pnls = [t.get('pnl_pct', 0) for t in today_trades if t.get('pnl_pct') is not None]

        result['trades_today'] = len(today_trades)
        result['total_pnl_pct'] = round(sum(today_pnls), 4)
        result['total_pnl_usd'] = round(sum(t.get('pnl_usd', 0) for t in today_trades), 2)
        result['wins'] = sum(1 for p in today_pnls if p > 0)
        result['losses'] = sum(1 for p in today_pnls if p <= 0)
        result['win_rate'] = round(result['wins'] / max(len(today_pnls), 1), 4)

        # P&L after spread
        result['pnl_after_spread'] = round(
            sum(max(0, p - 1.69) if p > 0 else p for p in today_pnls), 4
        )

        # EMA-20 of recent daily PnLs
        if len(pnls) >= 20:
            import numpy as np
            alpha = 2 / (20 + 1)
            ema = pnls[0]
            for p in pnls[1:]:
                ema = alpha * p + (1 - alpha) * ema
            result['ema_20_pnl'] = round(ema, 4)

        # Max drawdown (cumulative)
        if pnls:
            import numpy as np
            cum = np.cumsum(pnls)
            peak = np.maximum.accumulate(cum)
            dd = (peak - cum)
            result['max_drawdown'] = round(float(np.max(dd)) if len(dd) > 0 else 0, 4)

        # Rolling win rate (last 50 trades)
        recent = all_trades[-50:]
        recent_pnls = [t.get('pnl_pct', 0) for t in recent if t.get('pnl_pct') is not None]
        result['rolling_win_rate'] = round(
            sum(1 for p in recent_pnls if p > 0) / max(len(recent_pnls), 1), 4
        )
        result['rolling_trades'] = len(recent_pnls)

        # Status based on performance
        if result['rolling_win_rate'] < WIN_RATE_THRESHOLD:
            result['status'] = 'YELLOW'
        if result['max_drawdown'] > MAX_DRAWDOWN_THRESHOLD:
            result['status'] = 'RED'

        self.report['steps']['daily_pnl'] = result
        return result

    # ══════════════════════════════════════════════════════════════
    # Step 3: Auto-Retrain if Performance Degrades
    # ══════════════════════════════════════════════════════════════

    def step3_auto_retrain(self, pnl_result: Dict) -> Dict:
        """Trigger model retraining if win rate or drawdown exceed thresholds."""
        logger.info("Step 3: Checking if retraining needed...")
        result = {'status': 'GREEN', 'retrained': False, 'reason': None}

        wr = pnl_result.get('rolling_win_rate', 0.5)
        dd = pnl_result.get('max_drawdown', 0)
        needs_retrain = False
        reasons = []

        if wr < WIN_RATE_THRESHOLD and pnl_result.get('rolling_trades', 0) >= 10:
            needs_retrain = True
            reasons.append(f"Win rate {wr:.0%} < {WIN_RATE_THRESHOLD:.0%}")

        if dd > MAX_DRAWDOWN_THRESHOLD:
            needs_retrain = True
            reasons.append(f"Max drawdown {dd:.1%} > {MAX_DRAWDOWN_THRESHOLD:.1%}")

        if not needs_retrain:
            logger.info("  Performance within thresholds — no retrain needed")
            self.report['steps']['auto_retrain'] = result
            return result

        result['reason'] = '; '.join(reasons)
        logger.warning(f"  RETRAINING TRIGGERED: {result['reason']}")

        if self.dry_run:
            result['details'] = 'Dry run — would have retrained'
            self.report['steps']['auto_retrain'] = result
            return result

        # Run continuous_adapt which handles data refresh + backtest + retrain
        try:
            logger.info("  Running continuous_adapt cycle...")
            proc = subprocess.run(
                [sys.executable, '-m', 'src.scripts.continuous_adapt'],
                cwd=str(PROJECT_ROOT),
                capture_output=True, text=True, timeout=600,
            )
            if proc.returncode == 0:
                result['retrained'] = True
                result['status'] = 'GREEN'
                self.report['actions_taken'].append(f"Auto-retrained models: {result['reason']}")
                logger.info("  Retraining COMPLETE")
            else:
                result['status'] = 'YELLOW'
                result['details'] = f"Retrain exited with code {proc.returncode}"
                if self._should_escalate('auto_retrain'):
                    result['status'] = 'RED'
                    self.report['human_review_needed'].append(
                        f"Auto-retrain failed twice: {result['reason']}"
                    )
        except subprocess.TimeoutExpired:
            result['status'] = 'YELLOW'
            result['details'] = 'Retrain timed out (600s)'
        except Exception as e:
            result['status'] = 'YELLOW'
            result['details'] = f"Retrain error: {e}"

        self.report['steps']['auto_retrain'] = result
        return result

    # ══════════════════════════════════════════════════════════════
    # Step 4: Cloudflare Tunnel Connectivity
    # ══════════════════════════════════════════════════════════════

    def step4_check_tunnels(self) -> Dict:
        """Validate Cloudflare tunnel connectivity to Ollama instances."""
        logger.info("Step 4: Checking Cloudflare tunnel connectivity...")
        result = {'status': 'GREEN', 'endpoints': {}}

        # Find Ollama URLs from codebase/env
        ollama_urls = self._find_ollama_urls()
        if not ollama_urls:
            result['details'] = 'No Ollama URLs found in config'
            result['status'] = 'YELLOW'
            self.report['steps']['tunnels'] = result
            return result

        import urllib.request
        for name, url in ollama_urls.items():
            ep_result = {'url': url, 'status': 'RED', 'latency_ms': None}
            try:
                start = time.time()
                req = urllib.request.Request(url, method='GET')
                req.add_header('User-Agent', 'ACT-HealthCheck/1.0')
                resp = urllib.request.urlopen(req, timeout=10)
                latency = (time.time() - start) * 1000
                ep_result['status'] = 'GREEN'
                ep_result['latency_ms'] = round(latency, 1)
                ep_result['http_status'] = resp.status
                logger.info(f"  {name}: OK ({latency:.0f}ms)")
            except Exception as e:
                ep_result['error'] = str(e)
                result['status'] = 'YELLOW'
                logger.warning(f"  {name}: FAILED — {e}")

                # Try to restart cloudflared if down
                if not self.dry_run:
                    self._try_restart_tunnel(name)
                    self.report['actions_taken'].append(f"Attempted tunnel restart: {name}")

                    if self._should_escalate(f'tunnel_{name}'):
                        result['status'] = 'RED'
                        self.report['human_review_needed'].append(
                            f"Tunnel {name} ({url}) failed twice"
                        )

            result['endpoints'][name] = ep_result

        self.report['steps']['tunnels'] = result
        return result

    def _find_ollama_urls(self) -> Dict[str, str]:
        """Search codebase for Ollama/tunnel URLs."""
        urls = {}
        # Check common locations
        search_files = [
            PROJECT_ROOT / '.env',
            PROJECT_ROOT / 'src' / 'ai' / 'agentic_strategist.py',
            PROJECT_ROOT / 'src' / 'ai' / 'llm_provider.py',
            PROJECT_ROOT / 'data' / 'llm_config.json',
        ]
        import re
        url_pattern = re.compile(r'https?://[a-zA-Z0-9\-]+\.trycloudflare\.com')

        for fpath in search_files:
            if fpath.exists():
                try:
                    content = fpath.read_text(errors='ignore')
                    matches = url_pattern.findall(content)
                    for i, m in enumerate(matches):
                        key = f'ollama_{"primary" if i == 0 else "secondary_" + str(i)}'
                        if m not in urls.values():
                            urls[key] = m
                except Exception:
                    pass

        # Also check for localhost Ollama
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            if sock.connect_ex(('127.0.0.1', 11434)) == 0:
                urls['ollama_local'] = 'http://127.0.0.1:11434'
            sock.close()
        except Exception:
            pass

        return urls

    def _try_restart_tunnel(self, name: str):
        """Attempt to restart cloudflared tunnel."""
        try:
            # Check if cloudflared is installed
            subprocess.run(
                ['cloudflared', '--version'],
                capture_output=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            logger.info(f"  Cloudflared found — restart would be manual (tunnel URLs are dynamic)")
        except FileNotFoundError:
            logger.info(f"  Cloudflared not installed locally — tunnel is remote")
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════
    # Step 5: Disk Space & Log Rotation
    # ══════════════════════════════════════════════════════════════

    def step5_disk_and_logs(self) -> Dict:
        """Check disk space and rotate old logs."""
        logger.info("Step 5: Disk space and log rotation...")
        result = {'status': 'GREEN', 'disk_free_gb': 0, 'logs_rotated': 0, 'space_freed_mb': 0}

        # Disk space
        try:
            usage = shutil.disk_usage(str(PROJECT_ROOT))
            free_gb = usage.free / (1024 ** 3)
            result['disk_free_gb'] = round(free_gb, 2)
            result['disk_total_gb'] = round(usage.total / (1024 ** 3), 2)
            result['disk_used_pct'] = round((usage.used / usage.total) * 100, 1)

            if free_gb < DISK_SPACE_WARN_GB:
                result['status'] = 'YELLOW'
                logger.warning(f"  Disk space low: {free_gb:.1f} GB free")
                if free_gb < 1:
                    result['status'] = 'RED'
                    self.report['human_review_needed'].append(
                        f"Critical: Only {free_gb:.1f} GB disk space remaining"
                    )
            else:
                logger.info(f"  Disk: {free_gb:.1f} GB free ({result['disk_used_pct']}% used)")
        except Exception as e:
            result['disk_error'] = str(e)

        # Log rotation — remove logs older than 7 days
        cutoff = datetime.now() - timedelta(days=LOG_RETENTION_DAYS)
        space_freed = 0
        rotated = 0

        log_patterns = [
            str(LOGS_DIR / '*.log'),
            str(LOGS_DIR / '*.jsonl'),
            str(REPORTS_DIR / 'daily-*.json'),
        ]
        # Don't rotate essential files
        protected = {
            'trading_journal.jsonl', 'genetic_evolution_results.json',
            'strategy_backtest_results.json', 'performance_metrics.json',
        }

        for pattern in log_patterns:
            for fpath in glob.glob(pattern):
                fname = os.path.basename(fpath)
                if fname in protected:
                    continue
                try:
                    mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
                    if mtime < cutoff:
                        size = os.path.getsize(fpath)
                        if not self.dry_run:
                            os.remove(fpath)
                        space_freed += size
                        rotated += 1
                except Exception:
                    pass

        result['logs_rotated'] = rotated
        result['space_freed_mb'] = round(space_freed / (1024 * 1024), 2)
        if rotated > 0:
            logger.info(f"  Rotated {rotated} old logs, freed {result['space_freed_mb']} MB")
            self.report['actions_taken'].append(
                f"Rotated {rotated} logs older than {LOG_RETENTION_DAYS} days"
            )

        self.report['steps']['disk_and_logs'] = result
        return result

    # ══════════════════════════════════════════════════════════════
    # Step 6: Write Daily Report
    # ══════════════════════════════════════════════════════════════

    def step6_write_report(self) -> str:
        """Write structured daily report to reports/daily-YYYY-MM-DD.json."""
        logger.info("Step 6: Writing daily report...")

        # Determine overall status (worst of all steps)
        statuses = []
        for step_name, step_data in self.report['steps'].items():
            if isinstance(step_data, dict):
                statuses.append(step_data.get('status', 'GREEN'))

        if 'RED' in statuses:
            self.report['overall_status'] = 'RED'
        elif 'YELLOW' in statuses:
            self.report['overall_status'] = 'YELLOW'
        else:
            self.report['overall_status'] = 'GREEN'

        # Write report
        date_str = datetime.now().strftime('%Y-%m-%d')
        report_path = REPORTS_DIR / f'daily-{date_str}.json'

        try:
            with open(report_path, 'w') as f:
                json.dump(self.report, f, indent=2, default=str)
            logger.info(f"  Report saved: {report_path}")
            return str(report_path)
        except Exception as e:
            logger.error(f"  Report write failed: {e}")
            return ''

    # ══════════════════════════════════════════════════════════════
    # Retry Tracker (only escalate if fix fails twice)
    # ══════════════════════════════════════════════════════════════

    def _load_retry_tracker(self) -> Dict:
        try:
            if self._retry_tracker_path.exists():
                with open(self._retry_tracker_path) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_retry_tracker(self):
        try:
            self._retry_tracker_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._retry_tracker_path, 'w') as f:
                json.dump(self._retry_tracker, f, indent=2)
        except Exception:
            pass

    def _should_escalate(self, key: str) -> bool:
        """Returns True if this issue has already failed once (escalate to human review)."""
        today = datetime.now().strftime('%Y-%m-%d')
        entry = self._retry_tracker.get(key, {})
        if entry.get('date') == today:
            count = entry.get('count', 0)
            if count >= 1:
                return True  # Failed twice today — escalate
            entry['count'] = count + 1
        else:
            entry = {'date': today, 'count': 1}
        self._retry_tracker[key] = entry
        self._save_retry_tracker()
        return False

    # ══════════════════════════════════════════════════════════════
    # Main Runner
    # ══════════════════════════════════════════════════════════════

    def run_all(self) -> Dict:
        """Execute all daily operations steps."""
        start = time.time()
        logger.info("=" * 60)
        logger.info(f"  ACT DAILY OPS — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"  Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        logger.info("=" * 60)

        # Step 1: Process health
        self.step1_check_processes()

        # Step 2: Daily P&L
        pnl_result = self.step2_compute_daily_pnl()

        # Step 3: Auto-retrain if needed
        self.step3_auto_retrain(pnl_result)

        # Step 4: Tunnel connectivity
        self.step4_check_tunnels()

        # Step 5: Disk & logs
        self.step5_disk_and_logs()

        # Step 6: Write report
        elapsed = time.time() - start
        self.report['duration_seconds'] = round(elapsed, 1)
        report_path = self.step6_write_report()

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  DAILY OPS COMPLETE — {self.report['overall_status']}")
        logger.info(f"  Duration: {elapsed:.1f}s")
        logger.info(f"  Actions: {len(self.report['actions_taken'])}")
        if self.report['human_review_needed']:
            logger.warning(f"  HUMAN REVIEW NEEDED: {len(self.report['human_review_needed'])} issues")
            for issue in self.report['human_review_needed']:
                logger.warning(f"    - {issue}")
        logger.info("=" * 60)

        return self.report


def main():
    parser = argparse.ArgumentParser(description='ACT Daily Operations')
    parser.add_argument('--continuous', action='store_true', help='Run every 4 hours')
    parser.add_argument('--interval', type=float, default=4.0, help='Hours between runs')
    parser.add_argument('--dry-run', action='store_true', help='Report only, no actions')
    args = parser.parse_args()

    if args.continuous:
        logger.info(f"Starting continuous daily ops (every {args.interval}h)...")
        while True:
            try:
                ops = DailyOps(dry_run=args.dry_run)
                ops.run_all()
            except Exception as e:
                logger.error(f"Daily ops cycle failed: {e}")
                traceback.print_exc()
            sleep_sec = args.interval * 3600
            logger.info(f"Sleeping {args.interval}h until next cycle...")
            time.sleep(sleep_sec)
    else:
        ops = DailyOps(dry_run=args.dry_run)
        report = ops.run_all()
        print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
