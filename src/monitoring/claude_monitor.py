"""
Claude-Powered Trading Monitor
================================
Uses Claude API to analyze system status, positions, and recent trades.
Run: python -m src.monitoring.claude_monitor [command]

Commands:
  status    - Overall system health check
  positions - Analyze open positions with risk assessment
  journal   - Review recent trades and suggest improvements
  ask       - Free-form question about the system
"""

import os
import sys
import json
import glob
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=True)

try:
    import anthropic
except ImportError:
    print("ERROR: pip install anthropic")
    sys.exit(1)


def get_client():
    key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not key:
        print("ERROR: Set ANTHROPIC_API_KEY in .env")
        sys.exit(1)
    return anthropic.Anthropic(api_key=key)


def load_recent_journal(n=20):
    """Load last N trade journal entries."""
    journal_dir = os.path.join(PROJECT_ROOT, 'logs')
    files = sorted(glob.glob(os.path.join(journal_dir, 'journal_*.json')))
    if not files:
        return "No trade journal files found."

    entries = []
    for f in files[-3:]:  # Last 3 journal files
        try:
            with open(f, 'r') as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    entries.extend(data)
                elif isinstance(data, dict):
                    entries.append(data)
        except Exception:
            pass

    if not entries:
        return "Journal files exist but no valid entries."

    # Return last N entries
    recent = entries[-n:]
    return json.dumps(recent, indent=2, default=str)


def load_system_log(n=50):
    """Load last N lines from system output log."""
    log_path = os.path.join(PROJECT_ROOT, 'logs', 'system_output.log')
    if not os.path.exists(log_path):
        return "No system log found."

    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        return ''.join(lines[-n:])
    except Exception as e:
        return f"Error reading log: {e}"


def load_config():
    """Load current config."""
    import yaml
    config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def ask_claude(client, system_prompt, user_prompt):
    """Query Claude with system context."""
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.3,
    )
    return message.content[0].text


def cmd_status(client):
    """System health check."""
    config = load_config()
    recent_log = load_system_log(80)
    journal = load_recent_journal(10)

    system = """You are a crypto trading system monitor. Analyze the system logs and config to provide a concise health report.
Focus on: equity changes, open positions, recent errors, LLM status, exchange connectivity."""

    prompt = f"""CURRENT CONFIG:
{json.dumps(config, indent=2, default=str)}

RECENT SYSTEM LOG (last 80 lines):
{recent_log}

RECENT TRADES:
{journal}

Provide a brief health report:
1. System status (running/stopped/error)
2. Current equity and unrealized P&L
3. Open positions (if any)
4. Recent trade performance
5. Any warnings or issues
6. Recommendation"""

    print("\n" + "=" * 60)
    print("  CLAUDE SYSTEM MONITOR")
    print("=" * 60)
    result = ask_claude(client, system, prompt)
    print(result)
    print("=" * 60)


def cmd_positions(client):
    """Analyze open positions."""
    recent_log = load_system_log(100)

    system = """You are a crypto risk analyst. Analyze open positions and assess risk.
Be specific about entry prices, current P&L, stop-loss levels, and time held."""

    prompt = f"""RECENT SYSTEM LOG:
{recent_log}

Analyze all open positions:
1. List each position (asset, direction, entry, current price, P&L%)
2. Risk assessment for each (SL distance, time held, trend alignment)
3. Recommended action (hold, tighten SL, close)
4. Overall portfolio risk level"""

    print("\n" + "=" * 60)
    print("  POSITION ANALYSIS")
    print("=" * 60)
    result = ask_claude(client, system, prompt)
    print(result)
    print("=" * 60)


def cmd_journal(client):
    """Review recent trades."""
    journal = load_recent_journal(30)

    system = """You are a trading performance analyst. Review trade history and identify patterns.
Focus on: win rate, average P&L, common exit reasons, and improvement suggestions."""

    prompt = f"""RECENT TRADES (last 30):
{journal}

Analyze trading performance:
1. Win rate and average P&L per trade
2. Best and worst trades (what happened)
3. Common patterns in losses
4. Common patterns in wins
5. Top 3 actionable improvements"""

    print("\n" + "=" * 60)
    print("  TRADE JOURNAL ANALYSIS")
    print("=" * 60)
    result = ask_claude(client, system, prompt)
    print(result)
    print("=" * 60)


def cmd_ask(client, question):
    """Free-form question about the system."""
    config = load_config()
    recent_log = load_system_log(60)
    journal = load_recent_journal(15)

    system = """You are an expert crypto trading system assistant. Answer questions about this EMA(8) crossover + LLM confirmation trading system.
The system trades BTC and ETH on Robinhood Crypto (real account, read-only API)."""

    prompt = f"""SYSTEM CONFIG:
{json.dumps(config, indent=2, default=str)}

RECENT LOG:
{recent_log}

RECENT TRADES:
{journal}

USER QUESTION: {question}"""

    print("\n" + "-" * 60)
    result = ask_claude(client, system, prompt)
    print(result)
    print("-" * 60)


def main():
    client = get_client()

    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()

    if cmd == 'status':
        cmd_status(client)
    elif cmd == 'positions':
        cmd_positions(client)
    elif cmd == 'journal':
        cmd_journal(client)
    elif cmd == 'ask':
        question = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else input("Question: ")
        cmd_ask(client, question)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == '__main__':
    main()
