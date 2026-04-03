"""
Unified Trading Dashboard — ProJournX Style
=============================================
Launch: streamlit run src/dashboard/app.py --server.port 8501
"""
import sys
import logging
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Suppress noisy Tornado WebSocket warnings (tab close during auto-refresh)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)
logging.getLogger("tornado.general").setLevel(logging.CRITICAL)


def _suppress_websocket_closed_errors():
    """Avoid noisy WebSocketClosedError / StreamClosedError when browser tab closes or refreshes."""
    def _async_handler(loop, context):
        exc = context.get("exception")
        if exc is not None and type(exc).__name__ in ("WebSocketClosedError", "StreamClosedError"):
            return
        asyncio.default_exception_handler(context)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.call_soon_threadsafe(lambda: loop.set_exception_handler(_async_handler))
        else:
            loop.set_exception_handler(_async_handler)
    except Exception:
        pass

    class _WebSocketFilter(logging.Filter):
        def filter(self, record):
            msg = (record.msg % record.args) if record.args else str(record.msg)
            if "Task exception was never retrieved" in msg or "WebSocketClosedError" in msg or "StreamClosedError" in msg:
                return False
            return True

    logging.getLogger("asyncio").addFilter(_WebSocketFilter())
    logging.getLogger("tornado.general").addFilter(_WebSocketFilter())
    logging.getLogger("tornado.application").addFilter(_WebSocketFilter())


_suppress_websocket_closed_errors()

import streamlit as st
from datetime import datetime, timedelta
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

# Load .env so JOURNAL_ENCRYPTION_KEY and other secrets are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(
    page_title="Autonomous Trading Desk",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Authentication Gate ──
from src.dashboard.auth import check_auth, show_dev_mode_warning
if not check_auth():
    st.stop()

# ── Theme ──
from src.dashboard.theme import MARKETEDGE_CSS, GREEN, RED, MUTED
st.markdown(MARKETEDGE_CSS, unsafe_allow_html=True)

# ── Shared data (cached — no duplicate disk reads) ──
from src.dashboard.data import (
    load_dashboard_state,
    load_journal_trades,
    compute_today_pnl,
    compute_trade_summary,
    filter_closed,
    trade_realized_pnl_usd,
)

state = load_dashboard_state()
journal = load_journal_trades()
portfolio = state.get('portfolio', {})

# Auto-refresh: periodic fragment runs must trigger a full rerun, but Streamlit also
# invokes the fragment once synchronously during every full script run; st.rerun()
# there would loop forever and the page never loads (stuck on "Connecting").
if getattr(st, "fragment", None):
    @st.fragment(run_every=timedelta(seconds=10))
    def _auto_refresh():
        ctx = get_script_run_ctx()
        if ctx and ctx.fragment_ids_this_run:
            st.rerun()
else:
    def _auto_refresh():
        pass
_auto_refresh()

now = datetime.now()
today_str = now.strftime('%Y-%m-%d')

# ── Greeting ──
hour = now.hour
greeting = "Good morning" if hour < 12 else ("Good afternoon" if hour < 17 else "Good evening")
st.sidebar.markdown(f"""
<div style="padding:16px 0 12px 0; border-bottom: 1px solid #1e2330; margin-bottom:16px">
    <div style="font-size:1.1rem; font-weight:600; color:#e2e8f0">{greeting}, Trader</div>
    <div style="font-size:0.72rem; color:#64748b; margin-top:2px">{now.strftime('%A, %B %d, %Y')}</div>
</div>
""", unsafe_allow_html=True)

# ── Today P&L (exchange-anchored) ──
pnl, pnl_source = compute_today_pnl(portfolio, journal, today_str)
pnl_color = GREEN if pnl >= 0 else RED

# ── Trade stats (single-pass) ──
stats = compute_trade_summary(journal, today_str)
_today_wr = stats.get("today_win_rate")
_today_wr_display = f"{_today_wr:.0%}" if _today_wr is not None else "—"

# ── Account Balance ──
_current_total = portfolio.get('current_total_value')
_sod = portfolio.get('sod_balance')
if _current_total and _current_total > 0:
    balance_str = f"${_current_total:,.2f}"
elif _sod and _sod > 0:
    balance_str = f"${_sod:,.2f}"
else:
    balance_str = "---"

# ── Sidebar: Quick Stats Grid ──
st.sidebar.markdown(f"""
<div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:8px">
    <div class="pj-card" style="padding:10px; text-align:center; margin:0">
        <div style="font-size:0.6rem; color:#64748b; text-transform:uppercase">Account Balance</div>
        <div style="font-size:1rem; font-weight:700; color:#e2e8f0">{balance_str}</div>
    </div>
    <div class="pj-card" style="padding:10px; text-align:center; margin:0">
        <div style="font-size:0.6rem; color:#64748b; text-transform:uppercase">Today's P&L</div>
        <div style="font-size:1rem; font-weight:700; color:{pnl_color}">${pnl:+,.2f}</div>
    </div>
</div>
<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; margin-bottom:12px">
    <div class="pj-card" style="padding:8px; text-align:center; margin:0">
        <div style="font-size:0.55rem; color:#64748b; text-transform:uppercase">Trades (today)</div>
        <div style="font-size:0.9rem; font-weight:700; color:#e2e8f0">{stats['today_count']}</div>
    </div>
    <div class="pj-card" style="padding:8px; text-align:center; margin:0">
        <div style="font-size:0.55rem; color:#64748b; text-transform:uppercase">Wins (today)</div>
        <div style="font-size:0.9rem; font-weight:700; color:{GREEN}">{stats['today_wins']}</div>
    </div>
    <div class="pj-card" style="padding:8px; text-align:center; margin:0">
        <div style="font-size:0.55rem; color:#64748b; text-transform:uppercase">Win rate (today)</div>
        <div style="font-size:0.9rem; font-weight:700; color:{GREEN if (_today_wr or 0) > 0.5 else MUTED}">{_today_wr_display}</div>
    </div>
</div>
""", unsafe_allow_html=True)

show_dev_mode_warning()

# ── Recent Trades (last 5 closed) ──
_recent_closed = filter_closed(journal)[-5:]
if _recent_closed:
    st.sidebar.markdown('<div style="font-size:0.65rem; color:#64748b; text-transform:uppercase; margin:8px 0 4px">Recent Trades</div>', unsafe_allow_html=True)
    for t in reversed(_recent_closed):
        _asset = t.get('asset', '?')
        _side = t.get('side', '?').upper()
        _t_pnl = trade_realized_pnl_usd(t)
        _t_color = GREEN if _t_pnl > 0 else RED
        _ts = str(t.get('exit_time') or t.get('timestamp', ''))[:16]
        st.sidebar.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; padding:4px 8px; margin:2px 0; background:rgba(255,255,255,0.03); border-radius:6px; font-size:0.72rem">
            <span style="color:#94a3b8">{_asset} {_side}</span>
            <span style="color:{_t_color}; font-weight:600">${_t_pnl:+,.2f}</span>
        </div>
        """, unsafe_allow_html=True)

# ── Data source + refresh ──
last_update = state.get("last_update") or "---"
st.sidebar.caption(f"Data: **{last_update[:19] if isinstance(last_update, str) and len(last_update) > 19 else last_update}** | Source: {pnl_source}")
if st.sidebar.button("Refresh data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# ── Agent Overlay ──
agent_overlay = state.get('agent_overlay', {})
if agent_overlay.get('enabled'):
    dec = agent_overlay.get('last_decision', {})
    d = dec.get('direction', 0)
    dir_text = 'LONG' if d > 0 else ('SHORT' if d < 0 else 'FLAT')
    dir_color = GREEN if d > 0 else (RED if d < 0 else MUTED)
    st.sidebar.markdown(f"""
    <div class="pj-card" style="padding:10px; margin-top:8px">
        <div style="font-size:0.6rem; color:#64748b; text-transform:uppercase; margin-bottom:4px">Agent Overlay</div>
        <div style="font-size:1rem; font-weight:700; color:{dir_color}">{dir_text}</div>
        <div style="font-size:0.7rem; color:#64748b">
            Conf: {dec.get('confidence', 0):.0%} | {agent_overlay.get('consensus_level', 'N/A')}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Navigation ──
pages = {
    "Trading": [
        st.Page("pages/1_Core_HUD.py", title="Dashboard", icon="📊"),
        st.Page("pages/3_Agent_Intelligence.py", title="Agent Intelligence", icon="🤖"),
    ],
    "Analytics": [
        st.Page("pages/2_Performance.py", title="Performance", icon="📈"),
    ],
    "System": [
        st.Page("pages/4_System_Control.py", title="System Control", icon="⚙️"),
    ],
}

pg = st.navigation(pages)
pg.run()
