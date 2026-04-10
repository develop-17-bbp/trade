"""
Autonomous Trading Desk — Paper Trading Dashboard
===================================================
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

# Load .env so secrets are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(
    page_title="Paper Trading Desk",
    page_icon="📄",
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

# Auto-refresh every 10s
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

# ── Sidebar: Greeting ──
hour = now.hour
greeting = "Good morning" if hour < 12 else ("Good afternoon" if hour < 17 else "Good evening")
st.sidebar.markdown(f"""
<div style="padding:16px 0 12px 0; border-bottom: 1px solid #1e2330; margin-bottom:16px">
    <div style="font-size:1.1rem; font-weight:600; color:#e2e8f0">{greeting}, Trader</div>
    <div style="font-size:0.72rem; color:#64748b; margin-top:2px">{now.strftime('%A, %B %d, %Y')}</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: Mode indicator ──
st.sidebar.markdown(f"""
<div class="pj-card" style="padding:10px; text-align:center; margin:0 0 12px 0">
    <div style="font-size:0.6rem; color:#64748b; text-transform:uppercase">Mode</div>
    <div style="font-size:1rem; font-weight:700; color:{GREEN}">Paper Trading</div>
    <div style="font-size:0.65rem; color:#64748b">Robinhood Real Prices</div>
</div>
""", unsafe_allow_html=True)

show_dev_mode_warning()

if st.sidebar.button("Refresh data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# ── Navigation (Paper Trading only) ──
pages = {
    "Trading": [
        st.Page("pages/5_Paper_Trading.py", title="Paper Trading", icon="📄"),
    ],
}

pg = st.navigation(pages)
pg.run()
