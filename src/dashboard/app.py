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

# ── Sidebar: Greeting + Quick Stats ──
try:
    from src.api.state import DashboardState
    state = DashboardState().get_full_state()
except Exception:
    state = {}

# Auto-refresh every 30s so P&L and trades update when trading system is running
if getattr(st, "fragment", None):
    @st.fragment(run_every=timedelta(seconds=10))
    def _auto_refresh():
        st.rerun()
else:
    def _auto_refresh():
        pass
_auto_refresh()

now = datetime.now()
hour = now.hour
greeting = "Good morning" if hour < 12 else ("Good afternoon" if hour < 17 else "Good evening")

st.sidebar.markdown(f"""
<div style="padding:16px 0 12px 0; border-bottom: 1px solid #1e2330; margin-bottom:16px">
    <div style="font-size:1.1rem; font-weight:600; color:#e2e8f0">{greeting}, Trader</div>
    <div style="font-size:0.72rem; color:#64748b; margin-top:2px">{now.strftime('%A, %B %d, %Y')}</div>
</div>
""", unsafe_allow_html=True)

portfolio = state.get('portfolio', {})
pnl = portfolio.get('pnl', 0.0)
pnl_color = GREEN if pnl >= 0 else RED
total_balance = portfolio.get('total')
if total_balance is None:
    total_balance = 0.0
else:
    try:
        total_balance = float(total_balance)
    except (TypeError, ValueError):
        total_balance = 0.0
trades = state.get('trade_history', [])
today_str = now.strftime('%Y-%m-%d')
today_trades = [t for t in trades if isinstance(t, dict) and str(t.get('timestamp', '')).startswith(today_str)]
today_wins = len([t for t in today_trades if isinstance(t, dict) and t.get('pnl', 0) > 0])

st.sidebar.markdown(f"""
<div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:8px">
    <div class="pj-card" style="padding:10px; text-align:center; margin:0">
        <div style="font-size:0.6rem; color:#64748b; text-transform:uppercase">Today's P&L</div>
        <div style="font-size:1rem; font-weight:700; color:{pnl_color}">${pnl:+,.2f}</div>
    </div>
    <div class="pj-card" style="padding:10px; text-align:center; margin:0">
        <div style="font-size:0.6rem; color:#64748b; text-transform:uppercase">Trades / Wins</div>
        <div style="font-size:1rem; font-weight:700; color:#e2e8f0">{len(today_trades)} / {today_wins}</div>
    </div>
</div>
<div class="pj-card" style="padding:10px; text-align:center; margin-bottom:16px">
    <div style="font-size:0.6rem; color:#64748b; text-transform:uppercase">Wallet balance</div>
    <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0">${total_balance:,.2f}</div>
</div>
""", unsafe_allow_html=True)

show_dev_mode_warning()

# Data source notice: P&L/trades only update when the trading process runs
last_update = state.get("last_update") or "—"
st.sidebar.caption(f"Data last updated: **{last_update[:19] if isinstance(last_update, str) and len(last_update) > 19 else last_update}**")
if st.sidebar.button("Refresh data", use_container_width=True):
    st.rerun()
st.sidebar.info(
    "**P&L and trades** update only when the **trading system** is running. "
    "In another terminal run: `python -m src.main --dashboard`",
    icon="ℹ️",
)

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
