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
        out = j.trades or []
        if out:
            return out
    except Exception as e:
        logger.warning(f"[DASHBOARD] Journal load failed: {e}")
    try:
        import json
        p = os.path.join("logs", "trading_journal.json")
        if os.path.isfile(p):
            with open(p, "r") as f:
                data = json.load(f)
            return data if isinstance(data, list) else data.get("trades", [])
    except Exception:
        pass
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
# TODAY P&L — exchange NAV vs SOD when comparable; else journal / equity
# ═══════════════════════════════════════════════════════════════════

def _legacy_usdt_only_sod(portfolio: Dict) -> bool:
    """
    Detect old bug: sod_balance was USDT-only while current_total_value is full NAV.
    In that case exchange today_pnl is meaningless — fall back to journal.
    """
    sod = portfolio.get('sod_balance')
    cur = portfolio.get('current_total_value')
    try:
        if sod is None or cur is None:
            return False
        sod_f, cur_f = float(sod), float(cur)
        if sod_f <= 0 or cur_f <= 0:
            return False
        if sod_f < cur_f * 0.35:
            return True
    except (TypeError, ValueError):
        return False
    return False


def compute_today_pnl(portfolio: Dict, journal_trades: List[Dict] = None,
                      date_str: Optional[str] = None) -> Tuple[float, str]:
    """
    Priority:
      1. Exchange: today_pnl from state when SOD is same basis as current_total_value
         (full NAV anchor) and sod_date matches today.
      2. Equity curve intraday diff (v = session cumulative P&L — rough intraday).
      3. Journal closed trades realized today.

    Returns: (pnl_value, source_label)
    """
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')

    sod_balance = portfolio.get('sod_balance')
    sod_date = portfolio.get('sod_date', '')
    state_today_pnl = portfolio.get('today_pnl')

    use_exchange = (
        sod_balance is not None
        and sod_date == date_str
        and state_today_pnl is not None
        and not _legacy_usdt_only_sod(portfolio)
    )
    if use_exchange:
        return float(state_today_pnl), f"Exchange (SOD NAV ${float(sod_balance):,.0f})"

    equity_curve = portfolio.get('equity_curve', [])
    eq_today = [e for e in equity_curve if str(e.get('t', ''))[:10] == date_str]
    if len(eq_today) >= 2:
        diff = float(eq_today[-1].get('v', 0)) - float(eq_today[0].get('v', 0))
        return diff, "Equity curve (session)"

    if journal_trades is not None:
        closed_today = filter_today(filter_closed(journal_trades), date_str)
        realized = sum(float(t.get('pnl', 0) or 0) for t in closed_today)
        return realized, "Journal (realized today)"

    return float(portfolio.get('pnl', 0.0) or 0.0), "State (fallback)"


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
    today_closed_n = len(today_closed)
    today_win_rate = (today_wins / today_closed_n) if today_closed_n else None

    return {
        'total': total,
        'closed': len(closed),
        'open': len(open_trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(closed) if closed else 0.0,
        'total_pnl': total_pnl,
        'today_count': len(today_all),
        'today_closed': today_closed_n,
        'today_wins': today_wins,
        'today_win_rate': today_win_rate,
    }
