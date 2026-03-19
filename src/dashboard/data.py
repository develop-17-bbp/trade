"""
Shared data helpers for dashboard pages.
Centralises journal loading, trade filtering, and P&L computation
so every page uses the same logic and avoids duplicate work.
"""
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# JOURNAL LOADING (cached — one decryption per 8s across all pages)
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=8)
def load_journal_trades() -> List[Dict]:
    """Load all trades from the encrypted journal. Cached for 8s."""
    try:
        from src.monitoring.journal import TradingJournal
        j = TradingJournal()
        return j.trades or []
    except Exception as e:
        logger.warning(f"[DASHBOARD] Journal load failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════
# DASHBOARD STATE (cached — one disk read per 5s)
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=5)
def load_dashboard_state() -> Dict[str, Any]:
    """Load full system state via DashboardState. Cached for 5s."""
    try:
        from src.api.state import DashboardState
        return DashboardState().get_full_state()
    except Exception as e:
        logger.warning(f"[DASHBOARD] State load failed: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════
# TRADE FILTERS
# ═══════════════════════════════════════════════════════════════════

def is_closed_trade(t: Dict) -> bool:
    """True if trade is closed (has exit_price or status=CLOSED)."""
    if not isinstance(t, dict):
        return False
    if t.get('status') == 'CLOSED':
        return True
    if 'exit_price' in t and t.get('exit_price') and t.get('status') != 'OPEN':
        return True
    return False


def filter_closed(trades: List[Dict]) -> List[Dict]:
    """Return only closed trades."""
    return [t for t in trades if is_closed_trade(t)]


def filter_today(trades: List[Dict], date_str: Optional[str] = None) -> List[Dict]:
    """Return trades from today (by timestamp or exit_time)."""
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    result = []
    for t in trades:
        if not isinstance(t, dict):
            continue
        ts = str(t.get('exit_time') or t.get('timestamp', ''))
        # Handle Unix float timestamps
        try:
            if ts.replace('.', '', 1).isdigit():
                ts = datetime.fromtimestamp(float(ts)).isoformat()
        except (ValueError, OSError):
            pass
        if ts[:10] == date_str:
            result.append(t)
    return result


# ═══════════════════════════════════════════════════════════════════
# TODAY P&L — journal-authoritative for true today-only realized P&L
# ═══════════════════════════════════════════════════════════════════

def compute_today_pnl(portfolio: Dict, journal_trades: List[Dict] = None,
                      date_str: Optional[str] = None) -> Tuple[float, str]:
    """
    Three-tier P&L priority:
      1. Journal realized P&L from trades closed today
      2. Equity curve intraday diff (fallback only)
      3. Exchange/state today_pnl (fallback only)

    Returns: (pnl_value, source_label)
    """
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')

    # Priority 1: Journal realized P&L.
    # If there are no closed trades today, the correct day P&L is 0.0.
    if journal_trades is not None:
        closed_today = filter_today(filter_closed(journal_trades), date_str)
        realized = sum(float(t.get('pnl', 0) or 0) for t in closed_today)
        return realized, "Journal (today realized)"

    # Priority 2: Equity curve intraday diff
    equity_curve = portfolio.get('equity_curve', [])
    eq_today = [e for e in equity_curve if str(e.get('t', ''))[:10] == date_str]
    if len(eq_today) >= 2:
        diff = float(eq_today[-1].get('v', 0)) - float(eq_today[0].get('v', 0))
        return diff, "Equity Curve (fallback)"

    # Priority 3: Exchange/state fallback
    sod_balance = portfolio.get('sod_balance')
    sod_date = portfolio.get('sod_date', '')
    state_today_pnl = portfolio.get('today_pnl')
    if sod_balance is not None and sod_date == date_str and state_today_pnl is not None:
        return float(state_today_pnl), f"Exchange (fallback, SOD ${sod_balance:,.0f})"

    return float(portfolio.get('pnl', 0.0)), "Cumulative (fallback)"


# ═══════════════════════════════════════════════════════════════════
# TRADE STATS (single-pass computation)
# ═══════════════════════════════════════════════════════════════════

def compute_trade_summary(trades: List[Dict], date_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Single-pass computation of all trade stats.
    Returns dict with keys: total, closed, open, wins, losses,
    win_rate, total_pnl, today_count, today_wins, today_closed.
    """
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')

    total = len(trades)
    closed = []
    open_trades = []
    today_all = []
    today_closed = []

    for t in trades:
        if not isinstance(t, dict):
            continue
        if is_closed_trade(t):
            closed.append(t)
        elif t.get('status') == 'OPEN':
            open_trades.append(t)

        # Check if today
        ts = str(t.get('exit_time') or t.get('timestamp', ''))
        try:
            if ts.replace('.', '', 1).isdigit():
                ts = datetime.fromtimestamp(float(ts)).isoformat()
        except (ValueError, OSError):
            pass
        if ts[:10] == date_str:
            today_all.append(t)
            if is_closed_trade(t):
                today_closed.append(t)

    wins = [t for t in closed if (t.get('pnl') or 0) > 0]
    losses = [t for t in closed if (t.get('pnl') or 0) < 0]
    total_pnl = sum(t.get('pnl', 0) for t in closed)
    today_wins = sum(1 for t in today_closed if (t.get('pnl') or 0) > 0)

    return {
        'total': total,
        'closed': len(closed),
        'open': len(open_trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(closed) if closed else 0.0,
        'total_pnl': total_pnl,
        'today_count': len(today_all),
        'today_closed': len(today_closed),
        'today_wins': today_wins,
    }
