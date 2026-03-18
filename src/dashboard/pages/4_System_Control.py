"""
Page 4: System Control & Configuration
Consolidated from Flask dashboard_server.py features.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st
import time
import json
import os
import yaml
from datetime import datetime
import plotly.graph_objects as go

from src.api.state import DashboardState
from src.dashboard.theme import MARKETEDGE_CSS, metric_card, plotly_layout


def _parse_ts(ts) -> str:
    """Normalize Unix float or ISO timestamp to ISO string."""
    if not ts:
        return ''
    ts_str = str(ts)
    try:
        if ts_str.replace('.', '', 1).isdigit():
            return datetime.fromtimestamp(float(ts_str)).isoformat()
    except (ValueError, OSError):
        pass
    return ts_str

st.markdown(MARKETEDGE_CSS, unsafe_allow_html=True)

state_manager = DashboardState()
state = state_manager.get_full_state()


# ═══ HEADER ═══
st.markdown("""
<div class="main-header">
    <div>
        <div class="title">SYSTEM CONTROL</div>
        <div class="subtext">CONFIGURATION | LAYER STATUS | TRADE JOURNAL | LOGS</div>
    </div>
    <div style="text-align:right;">
        <div class="subtext">AUTO-REFRESH: 10s</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══ SECTION 1: SYSTEM OVERVIEW ═══
st.markdown('<div class="section-title">SYSTEM OVERVIEW</div>', unsafe_allow_html=True)

sources = state.get("sources", {})
status = state.get("status", "UNKNOWN")
last_update = state.get("last_update", "")
layers = state.get("layers", {})

c1, c2, c3, c4 = st.columns(4)
with c1:
    sys_color = "#22c55e" if status in ("TRADING", "RUNNING") else "#ffa500"
    st.markdown(f"""<div class="glass-card" style="text-align:center;">
        <div style="font-size:2rem;">⚙</div>
        <div class="metric-label">SYSTEM STATUS</div>
        <div class="metric-value" style="color:{sys_color};">{status}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    online_count = sum(1 for v in sources.values() if v == "ONLINE")
    total_sources = len(sources)
    st.markdown(f"""<div class="glass-card" style="text-align:center;">
        <div style="font-size:2rem;">📡</div>
        <div class="metric-label">DATA SOURCES</div>
        <div class="metric-value metric-value-cyan">{online_count}/{total_sources}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    layer_ok = sum(1 for v in layers.values() if v.get("status") == "OK")
    st.markdown(f"""<div class="glass-card" style="text-align:center;">
        <div style="font-size:2rem;">🧠</div>
        <div class="metric-label">LAYERS ONLINE</div>
        <div class="metric-value metric-value-green">{layer_ok}/{len(layers)}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="glass-card" style="text-align:center;">
        <div style="font-size:2rem;">🕐</div>
        <div class="metric-label">LAST UPDATE</div>
        <div style="color:#ddd; font-size:0.8rem; margin-top:8px;">{last_update[-8:] if last_update else 'N/A'}</div>
    </div>""", unsafe_allow_html=True)


# ═══ SECTION 2: 9-LAYER DETAILED STATUS ═══
st.markdown('<div class="section-title">9-LAYER DETAILED STATUS</div>', unsafe_allow_html=True)

if layers:
    for l_name, l_data in layers.items():
        l_status = l_data.get("status", "UNKNOWN")
        prog = l_data.get("progress", 0)
        metric = l_data.get("metric", "")
        color = "#22c55e" if l_status == "OK" else "#ffa500" if l_status == "WARN" else "#ef4444"
        prog_pct = int(prog * 100)

        st.markdown(f"""
        <div style="background: rgba(15,15,30,0.7); padding: 14px 20px; border-radius: 12px; margin-bottom: 8px; border-left: 4px solid {color}; display:flex; justify-content:space-between; align-items:center;">
            <div style="flex:1;">
                <div style="font-family:Inter; font-size:0.8rem; color:#fff; letter-spacing:1px;">{l_name}</div>
                <div style="height:4px; background:rgba(255,255,255,0.05); border-radius:2px; margin-top:6px; width:200px;">
                    <div style="width:{prog_pct}%; height:100%; background:{color}; border-radius:2px; box-shadow: 0 0 8px {color}66;"></div>
                </div>
            </div>
            <div style="text-align:right;">
                <span style="padding:4px 12px; border-radius:20px; font-size:0.65rem; font-weight:700; background:{color}22; color:{color}; border:1px solid {color}44;">{l_status}</span>
                <div style="font-size:0.65rem; color:#666; margin-top:4px;">{metric}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Layer logs
    with st.expander("Layer Activity Logs", expanded=False):
        for layer_key in ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]:
            logs = state_manager.get_layer_logs(layer_key, 5)
            if logs:
                st.markdown(f"**{layer_key}**")
                for log in logs:
                    level = log.get("level", "INFO")
                    lc = "#22c55e" if level == "INFO" else "#ffa500" if level == "WARN" else "#ef4444"
                    st.markdown(f"""<div style="font-size:0.7rem; padding:3px 8px; margin-bottom:2px; border-left:2px solid {lc}; color:#ccc;">
                        <span style="color:#555;">{log.get('timestamp','')}</span> {log.get('message','')}</div>""", unsafe_allow_html=True)
else:
    st.info("Layer status not yet available. Start the trading system.")


# ═══ SECTION 3: CONFIGURATION ═══
st.markdown('<div class="section-title">ACTIVE CONFIGURATION</div>', unsafe_allow_html=True)

try:
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f) or {}

        col_gen, col_risk, col_agent = st.columns(3)

        with col_gen:
            st.markdown("**General**")
            st.markdown(f"""<div class="glass-card">
                <div style="font-size:0.75rem; color:#888;">
                    <div style="margin-bottom:6px;">Mode: <span style="color:#3b82f6; font-weight:700;">{config.get('mode','?')}</span></div>
                    <div style="margin-bottom:6px;">Poll Interval: <span style="color:#fff;">{config.get('poll_interval','?')}s</span></div>
                    <div style="margin-bottom:6px;">Assets: <span style="color:#22c55e;">{', '.join(config.get('assets',[]))}</span></div>
                    <div style="margin-bottom:6px;">Capital: <span style="color:#fff;">${config.get('initial_capital',0):,.0f}</span></div>
                    <div>Force Trade: <span style="color:{'#22c55e' if config.get('force_trade') else '#ef4444'};">{'ON' if config.get('force_trade') else 'OFF'}</span></div>
                </div>
            </div>""", unsafe_allow_html=True)

        with col_risk:
            st.markdown("**Risk Parameters**")
            risk_cfg = config.get("risk", {})
            st.markdown(f"""<div class="glass-card">
                <div style="font-size:0.75rem; color:#888;">
                    <div style="margin-bottom:6px;">Max Position: <span style="color:#ffa500;">{risk_cfg.get('max_position_size_pct','?')}%</span></div>
                    <div style="margin-bottom:6px;">Daily Loss Limit: <span style="color:#ef4444;">{risk_cfg.get('daily_loss_limit_pct','?')}%</span></div>
                    <div style="margin-bottom:6px;">Max Drawdown: <span style="color:#ef4444;">{risk_cfg.get('max_drawdown_pct','?')}%</span></div>
                    <div style="margin-bottom:6px;">Risk/Trade: <span style="color:#fff;">{risk_cfg.get('risk_per_trade_pct','?')}%</span></div>
                    <div style="margin-bottom:6px;">ATR Stop: <span style="color:#fff;">{risk_cfg.get('atr_stop_mult','?')}x</span></div>
                    <div>ATR TP: <span style="color:#fff;">{risk_cfg.get('atr_tp_mult','?')}x</span></div>
                </div>
            </div>""", unsafe_allow_html=True)

        with col_agent:
            st.markdown("**Agent Overlay**")
            agent_cfg = config.get("agents", {})
            lp_cfg = agent_cfg.get("loss_prevention", {})
            st.markdown(f"""<div class="glass-card">
                <div style="font-size:0.75rem; color:#888;">
                    <div style="margin-bottom:6px;">Enabled: <span style="color:{'#22c55e' if agent_cfg.get('enabled') else '#ef4444'};">{'YES' if agent_cfg.get('enabled') else 'NO'}</span></div>
                    <div style="margin-bottom:6px;">Blend Weight: <span style="color:#3b82f6;">{agent_cfg.get('blend_weight','?')}</span></div>
                    <div style="margin-bottom:6px;">Consensus: <span style="color:#fff;">{agent_cfg.get('consensus_threshold','?')}</span></div>
                    <div style="margin-bottom:6px;">Daily Target: <span style="color:#22c55e;">{lp_cfg.get('daily_target_pct','?')}%</span></div>
                    <div style="margin-bottom:6px;">Halt At: <span style="color:#ef4444;">{lp_cfg.get('halt_threshold_pct','?')}%</span></div>
                    <div>Learning Rate: <span style="color:#fff;">{agent_cfg.get('weight_update_alpha','?')}</span></div>
                </div>
            </div>""", unsafe_allow_html=True)
except Exception as e:
    st.warning(f"Config load error: {e}")


# ═══ SECTION 4: TRADE JOURNAL ═══
st.markdown('<div class="section-title">RECENT TRADES</div>', unsafe_allow_html=True)

trade_history = state.get("trade_history", [])
if trade_history:
    # Show last 20 trades
    recent_trades = trade_history[-20:][::-1]
    for trade in recent_trades:
        pnl = trade.get("pnl", trade.get("net_pnl", 0))
        pnl_color = "#22c55e" if pnl > 0 else "#ef4444"
        direction = "LONG" if trade.get("direction", 0) > 0 else "SHORT" if trade.get("direction", 0) < 0 else trade.get("side", "?")
        asset = trade.get("asset", trade.get("symbol", "?"))
        conf = trade.get("confidence", 0)
        entry = trade.get("entry_price", 0)
        exit_p = trade.get("exit_price", 0)
        ts = _parse_ts(trade.get("timestamp", trade.get("entry_time", "")))

        st.markdown(f"""
        <div style="background: rgba(15,15,30,0.7); padding: 12px 16px; border-radius: 10px; margin-bottom: 6px; border-left: 4px solid {pnl_color}; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-family:Inter; font-size:0.75rem; color:#3b82f6;">{asset}</span>
                <span style="font-size:0.65rem; color:#888; margin-left:8px;">{direction}</span>
                <span style="font-size:0.55rem; color:#555; margin-left:8px;">{str(ts)[-8:]}</span>
            </div>
            <div style="text-align:right;">
                <span style="font-size:1rem; font-weight:700; color:{pnl_color};">${pnl:+,.2f}</span>
                <span style="font-size:0.6rem; color:#666; margin-left:8px;">{conf:.0%} conf</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No trades recorded yet.")


# ═══ SECTION 5: DATA SOURCES DETAIL ═══
st.markdown('<div class="section-title">DATA SOURCE STATUS</div>', unsafe_allow_html=True)

source_details = [
    ("📡 Exchange (Binance)", sources.get("exchange", "UNKNOWN"), "Real-time OHLCV, order book, derivatives"),
    ("📰 News Feed (NewsAPI)", sources.get("news", "UNKNOWN"), "Sentiment via FinBERT + CryptoPanic + Reddit"),
    ("⛓ On-Chain Analytics", sources.get("onchain", "UNKNOWN"), "Whale movements, exchange flows, liquidations"),
    ("🧠 LLM Engine", sources.get("llm", "UNKNOWN"), "Strategic reasoning via Ollama / Gemini / OpenAI"),
]

for name, s, desc in source_details:
    dot_class = "dot-green" if s == "ONLINE" else "dot-red" if s == "OFFLINE" else "dot-yellow"
    st.markdown(f"""
    <div style="background: rgba(15,15,30,0.7); padding: 12px 16px; border-radius: 10px; margin-bottom: 6px; display:flex; justify-content:space-between; align-items:center;">
        <div>
            <span style="font-size:0.85rem; color:#fff;">{name}</span>
            <div style="font-size:0.65rem; color:#555; margin-top:2px;">{desc}</div>
        </div>
        <div><span class="status-dot {dot_class}"></span><span style="color:#ddd; font-size:0.8rem;">{s}</span></div>
    </div>
    """, unsafe_allow_html=True)


# ═══ SECTION 6: ENVIRONMENT STATUS ═══
st.markdown('<div class="section-title">ENVIRONMENT</div>', unsafe_allow_html=True)

env_vars = [
    ("BINANCE_TESTNET_KEY", bool(os.environ.get("BINANCE_TESTNET_KEY"))),
    ("NEWSAPI_KEY", bool(os.environ.get("NEWSAPI_KEY"))),
    ("CRYPTOPANIC_TOKEN", bool(os.environ.get("CRYPTOPANIC_TOKEN"))),
    ("REASONING_LLM_KEY", bool(os.environ.get("REASONING_LLM_KEY"))),
]

cols = st.columns(4)
for i, (name, present) in enumerate(env_vars):
    with cols[i]:
        color = "#22c55e" if present else "#ef4444"
        label = "SET" if present else "MISSING"
        st.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div class="metric-label" style="font-size:0.6rem;">{name}</div>
            <div style="color:{color}; font-weight:700; margin-top:6px;">{label}</div>
        </div>""", unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div style="text-align:center; color:#333; font-size:0.65rem; margin-top:40px; padding:20px 0; border-top:1px solid rgba(255,255,255,0.03);">
    System Control Panel v1.0 | Autonomous Trading Desk
</div>
""", unsafe_allow_html=True)

time.sleep(10)
st.rerun()
