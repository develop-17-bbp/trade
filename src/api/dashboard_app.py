import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime
import plotly.graph_objects as go
from src.api.state import DashboardState

# Page config for 'Premium' look
st.set_page_config(
    page_title="Autonomous Trading Desk - Core 9-Layer HUD",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for masterclass aesthetic (MarketEdge Style)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
    radial-gradient(circle at 10% 20%, #0b0b1d 0%, transparent 40%),
    radial-gradient(circle at 80% 80%, #0d2436 0%, transparent 40%),
    linear-gradient(135deg, #04040c, #0b0b20) !important;
}

/* HEADER */
.main-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 0;
    margin-bottom: 20px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

.title {
    font-family: 'Orbitron';
    font-size: 1.9rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00eaff, #00ff9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtext {
    font-size: 0.65rem;
    color: #6e6e8f;
    letter-spacing: 1px;
}

/* GLASS CARD */
.layer-card {
    background: rgba(20,20,40,0.55) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    backdrop-filter: blur(18px) !important;
    border-radius: 14px !important;
    padding: 18px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 6px 30px rgba(0,0,0,0.4), inset 0 0 0 1px rgba(255,255,255,0.02) !important;
    transition: all .25s ease !important;
}

.layer-card:hover {
    transform: translateY(-3px) !important;
    border: 1px solid rgba(0,234,255,0.25) !important;
}

/* EXPANDABLE LAYER CARD */
.expandable-layer {
    background: rgba(15,15,30,0.7) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 16px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5) !important;
    transition: all .3s ease !important;
}

.expandable-layer:hover {
    transform: translateY(-2px) !important;
    border: 1px solid rgba(0,234,255,0.3) !important;
}

.layer-header {
    padding: 20px !important;
    cursor: pointer !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    transition: background 0.2s ease !important;
}

.layer-header:hover {
    background: rgba(255,255,255,0.03) !important;
}

.layer-content {
    padding: 0 20px 20px 20px !important;
    border-top: 1px solid rgba(255,255,255,0.05) !important;
}

.sub-metric {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 8px !important;
    padding: 12px !important;
    margin-bottom: 10px !important;
    border-left: 3px solid #00eaff !important;
}

.sub-metric-title {
    font-size: 0.7rem !important;
    color: #8b8ba7 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    margin-bottom: 4px !important;
}

.sub-metric-value {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #fff !important;
}

.status-badge {
    padding: 4px 10px !important;
    border-radius: 20px !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

.status-ok { background: rgba(0,255,157,0.15) !important; color: #00ff9d !important; border: 1px solid #00ff9d44 !important; }
.status-warn { background: rgba(255,165,0,0.15) !important; color: #ffa500 !important; border: 1px solid #ffa50044 !important; }
.status-error { background: rgba(255,77,77,0.15) !important; color: #ff4d4d !important; border: 1px solid #ff4d4d44 !important; }

.metric-small {
    font-size: 0.72rem;
    color: #8b8ba7;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-large {
    font-size: 1.7rem;
    font-weight: 600;
    margin-top: 4px;
    color: #fff;
}

/* LAYER TITLE */
.layer-title {
    font-family: 'Orbitron';
    font-size: .75rem;
    letter-spacing: 1.4px;
    color: #00eaff;
    margin-bottom: 15px;
    text-transform: uppercase;
    display: flex;
    align-items: center;
}

.layer-title:after {
    content: "";
    flex: 1;
    height: 1px;
    margin-left: 10px;
    background: linear-gradient(90deg, #00eaff, transparent);
}

.status-green { color: #00ff9d; font-weight: 700; }
.status-red { color: #ff4d6d; font-weight: 700; }

/* ATTRIBUTION BAR */
.attribution-bar {
    height: 7px;
    border-radius: 6px;
    background: #111326;
    display: flex;
    overflow: hidden;
    margin-top: 10px;
}

/* EXPANDER STYLING */
.stExpander {
    background: rgba(20,20,40,0.4) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
}

.stExpander > div:first-child {
    font-family: 'Orbitron' !important;
    color: #00eaff !important;
    font-size: 0.85rem !important;
}

.sub-metric-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 15px;
    margin: 10px 0;
}

.sub-metric-item {
    padding: 10px;
    background: rgba(255,255,255,0.02);
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.04);
}
</style>
""", unsafe_allow_html=True)

def metric_card(title, value, color="white"):
    return f"""
    <div class="layer-card" style="text-align:center">
        <div class="metric-small">{title}</div>
        <div class="metric-large" style="color:{color}">
        {value}
        </div>
    </div>
    """

def get_tv_widget(symbol="BINANCE:BTCUSDT"):
    return f"""
    <div class="tradingview-widget-container" style="height:500px;width:100%;">
      <div id="tradingview_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{symbol}",
        "interval": "5",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """

def get_tv_tech_analysis(symbol="BINANCE:BTCUSDT"):
    return f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
      {{
      "interval": "1m",
      "width": "100%",
      "isTransparent": true,
      "height": 450,
      "symbol": "{symbol}",
      "showIntervalTabs": true,
      "displayMode": "single",
      "locale": "en",
      "colorTheme": "dark"
    }}
      </script>
    </div>
    """

from dotenv import load_dotenv
load_dotenv()

state_manager = DashboardState()

def run_app():
    state = state_manager.get_full_state()
    asset = "BTC"
    asset_data = state.get("active_assets", {}).get(asset, {})
    l1_data = state.get("l1_features", {}).get(asset, {})
    sent = state.get("sentiment", {}).get(asset, {})
    sources = state.get("sources", {})

    
    # --- HEADER ---
    source_hud = ""
    for s_name, s_stat in sources.items():
        color = "#00ff9d" if s_stat == "ONLINE" else "#ff4d6d"
        source_hud += f'<span style="margin-left:15px; font-size: 0.65rem; color:#6e6e8f;">{s_name.upper()}: <span style="color:{color}; font-weight:700;">{s_stat}</span></span>'

    st.markdown(f"""
    <div class="main-header">
        <div>
            <div class="title">LAYER-9 CORE</div>
            <div class="subtext">Autonomous Multi-Layer Trading Intelligence {source_hud}</div>
        </div>
        <div style="text-align:right">
            <div class="subtext">SYSTEM STATUS <span class="status-green">{state['status']}</span></div>
            <div class="subtext">{state.get('last_update','')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- TOTALS (L9 Preview) ---
    p_perf = state["portfolio"]
    perf_edge = state.get("performance_edge", {})
    t1, t2, t3, t4 = st.columns(4)
    with t1: st.markdown(metric_card("PORTFOLIO P&L", f"${p_perf['pnl']:,.2f}", "#00ff9d"), unsafe_allow_html=True)
    with t2: st.markdown(metric_card("AGENT WINRATE", f"{perf_edge.get('agent_winrate', 0.5)*100:.1f}%", "#00eaff"), unsafe_allow_html=True)
    with t3: st.markdown(metric_card("BASELINE UPLIFT", f"{perf_edge.get('uplift_pct', 0):+.2f}%", "#00ff9d"), unsafe_allow_html=True)
    with t4: st.markdown(metric_card("ACTIVE LAYERS", "09 / 09", "#ffffff"), unsafe_allow_html=True)

    # --- SENTIMENT HEATMAP ---
    st.markdown('<div class="layer-title">Sentiment Heatmap & Impact Analysis</div>', unsafe_allow_html=True)
    s_score = sent.get('composite_score', 0.0)
    h_color = "#00ff9d" if s_score > 0.2 else "#ff4d6d" if s_score < -0.2 else "#00d4ff"
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #ff4d6d22 0%, #00d4ff22 50%, #00ff9d22 100%); height: 40px; border-radius: 20px; position: relative; margin: 10px 0; border: 1px solid rgba(255,255,255,0.05);">
        <div style="position: absolute; left: {(s_score + 1) * 50}%; top: -5px; width: 4px; height: 50px; background: {h_color}; box-shadow: 0 0 15px {h_color}; transition: all 0.5s ease;"></div>
        <div style="position: absolute; left: 0; width: 100%; height: 100%; display: flex; justify-content: space-between; align-items: center; padding: 0 20px; font-size: 0.6rem; color: #555; font-weight: 700;">
            <span>EXTREME FEAR</span>
            <span>NEUTRAL</span>
            <span>GREED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- 9-LAYER FULL REFLECTION GRID ---
    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
    
    # We add the 9-Layer Sidebar for process monitoring as requested
    with st.sidebar:
        st.markdown("""
        <h2 style="font-family:Orbitron; color:#00eaff; font-size:1.2rem; margin-bottom:20px;">
        SYSTEM CONTROL
        </h2>
        """, unsafe_allow_html=True)
        
        layers = state.get("layers", {})
        if not layers:
            # Placeholder if not yet populated
            for i in range(1, 10):
                st.markdown(f"""
                <div style="margin-bottom: 15px; opacity: 0.5;">
                    <div style="display:flex; justify-content:space-between; font-size: 0.7rem;">
                        <span style="color:#888;">L{i} Module</span>
                        <span style="color:#444;">SYNCING...</span>
                    </div>
                    <div style="height:4px; background:#111; border-radius:2px; margin-top:4px;"></div>
                </div>
                """, unsafe_allow_html=True)
        else:
            for l_name, l_data in layers.items():
                status = l_data.get("status", "UNKNOWN")
                prog = l_data.get("progress", 0) * 100
                metric = l_data.get("metric", "")
                color = "#00ff88" if status == "OK" else "#ffa500" if status == "WARN" else "#ff4d4d"
                
                st.markdown(f"""
                <div style="margin-bottom: 18px; padding: 10px; background: rgba(255,255,255,0.02); border-radius: 8px; border-left: 3px solid {color};">
                    <div style="display:flex; justify-content:space-between; font-size: 0.75rem; font-weight: 700;">
                        <span style="color:#fff;">{l_name}</span>
                        <span style="color:{color};">{status}</span>
                    </div>
                    <div style="height:3px; background: rgba(255,255,255,0.05); border-radius:2px; margin: 6px 0;">
                        <div style="width:{prog}%; height:100%; background:{color}; box-shadow: 0 0 10px {color}66;"></div>
                    </div>
                    <div style="font-size: 0.6rem; color: #666; text-transform: uppercase;">{metric}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<h3 style="font-family: Orbitron; font-size: 0.8rem; color: #888;">🎛️ Control Panel</h3>', unsafe_allow_html=True)
        st.selectbox("Focus Asset", ["BTC/USDT", "ETH/USDT"])
        st.checkbox("Force Sentiment Veto", value=False)
        if st.button("Emergency Halt"):
            st.error("SYSTEM HALTED")

    # --- TRADING VIEW REAL-TIME WIDGETS ---
    st.markdown('<div class="layer-title">Sentinel Intelligence: Real-Time Market Convergence</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.components.v1.html(get_tv_widget(f"BINANCE:{asset}USDT"), height=520)
    with c2:
        st.components.v1.html(get_tv_tech_analysis(f"BINANCE:{asset}USDT"), height=480)

    # --- DETAILED 9-LAYER EXPANDABLE HUD ---
    st.markdown('<div class="layer-title">Layer Evolution: Process Reflection Grid</div>', unsafe_allow_html=True)
    layers = state.get("layers", {})
    l1_data = state.get("l1_features", {}).get(asset, {})
    sent = state.get("sentiment", {}).get(asset, {})
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        with st.expander("🛡️ L1 Quant & Microstructure", expanded=False):
            st.markdown(f"""
            <div class="sub-metric-grid">
                <div class="sub-metric-item"><div class="metric-small">VPIN Toxicity</div><div class="status-green" style="font-size:1.2rem;">{l1_data.get('vpin', 0.0):.4f}</div></div>
                <div class="sub-metric-item"><div class="metric-small">Regime</div><div style="color:#00eaff; font-size:1.2rem;">{l1_data.get("liquidity_regime", "NORMAL")}</div></div>
                <div class="sub-metric-item"><div class="metric-small">Flow Imbalance</div><div>{l1_data.get('flow_imbalance', 0.0):.2f}</div></div>
                <div class="sub-metric-item"><div class="metric-small">Top Features</div><div>{len(l1_data.get('top_features', []))} active</div></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add L1 logs
            l1_logs = state_manager.get_layer_logs("L1", 3)
            if l1_logs:
                st.markdown("<div style='margin-top: 15px; font-size: 0.7rem; color: #888;'>RECENT ACTIVITY</div>", unsafe_allow_html=True)
                for log in l1_logs:
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.02); padding: 6px 10px; border-radius: 4px; margin-bottom: 4px; border-left: 2px solid #00eaff;">
                        <div style="font-size: 0.6rem; color: #666;">{log.get('timestamp', '')}</div>
                        <div style="font-size: 0.7rem; color: #ddd;">{log.get('message', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
        with st.expander("🧠 L2 Sentiment Intelligence", expanded=False):
            st.markdown(f"""
            <div class="sub-metric-grid">
                <div class="sub-metric-item"><div class="metric-small">Composite Score</div><div class="status-green" style="font-size:1.2rem;">{sent.get('composite_score', 0.5):.3f}</div></div>
                <div class="sub-metric-item"><div class="metric-small">Sentiment Bias</div><div style="color:#00eaff; font-size:1.2rem;">BULLISH</div></div>
                <div class="sub-metric-item"><div class="metric-small">Bullish %</div><div style="color:#00ff9d;">{sent.get('bull_pct', 50):.1f}%</div></div>
                <div class="sub-metric-item"><div class="metric-small">Bearish %</div><div style="color:#ff4d4d;">{sent.get('bear_pct', 50):.1f}%</div></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add L2 logs
            l2_logs = state_manager.get_layer_logs("L2", 3)
            if l2_logs:
                st.markdown("<div style='margin-top: 15px; font-size: 0.7rem; color: #888;'>RECENT ACTIVITY</div>", unsafe_allow_html=True)
                for log in l2_logs:
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.02); padding: 6px 10px; border-radius: 4px; margin-bottom: 4px; border-left: 2px solid #00ff88;">
                        <div style="font-size: 0.6rem; color: #666;">{log.get('timestamp', '')}</div>
                        <div style="font-size: 0.7rem; color: #ddd;">{log.get('message', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)

        with st.expander("⚔️ L3 On-Chain & Risk", expanded=False):
            oc = state.get("onchain_metrics", {})
            st.markdown(f"""
            <div class="sub-metric-grid">
                <div class="sub-metric-item"><div class="metric-small">Whale Bias</div><div style="color:#00ff9d;">{oc.get("whale_sentiment", "NEUTRAL")}</div></div>
                <div class="sub-metric-item"><div class="metric-small">Net Flow</div><div style="color:#ffa500;">{oc.get("whale_metrics", {}).get("net_exchange_flow", 0):+.2f}</div></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add L3 logs
            l3_logs = state_manager.get_layer_logs("L3", 3)
            if l3_logs:
                st.markdown("<div style='margin-top: 15px; font-size: 0.7rem; color: #888;'>RECENT ACTIVITY</div>", unsafe_allow_html=True)
                for log in l3_logs:
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.02); padding: 6px 10px; border-radius: 4px; margin-bottom: 4px; border-left: 2px solid #ffa500;">
                        <div style="font-size: 0.6rem; color: #666;">{log.get('timestamp', '')}</div>
                        <div style="font-size: 0.7rem; color: #ddd;">{log.get('message', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)

        with st.expander("🔀 L4 Signal Fusion", expanded=False):
            attr = asset_data.get('attribution', {'l1': 0.5, 'l2': 0.3, 'l3': 0.2})
            total_inf = sum(attr.values()) or 1
            l1_p, l2_p, l3_p = (attr.get('l1', 0)/total_inf)*100, (attr.get('l2', 0)/total_inf)*100, (attr.get('l3', 0)/total_inf)*100
            st.markdown(f"""
            <div class="sub-metric-grid">
                <div class="sub-metric-item"><div class="metric-small">L1 Weight</div><div style="color:#00d4ff;">{l1_p:.1f}%</div></div>
                <div class="sub-metric-item"><div class="metric-small">L2 Weight</div><div style="color:#00ff88;">{l2_p:.1f}%</div></div>
                <div class="sub-metric-item"><div class="metric-small">L3 Weight</div><div style="color:#ffa500;">{l3_p:.1f}%</div></div>
                <div class="sub-metric-item"><div class="metric-small">Final Signal</div><div style="color:#fff;">{"BULLISH" if l1_p > 40 else "NEUTRAL"}</div></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add L4 logs
            l4_logs = state_manager.get_layer_logs("L4", 3)
            if l4_logs:
                st.markdown("<div style='margin-top: 15px; font-size: 0.7rem; color: #888;'>RECENT ACTIVITY</div>", unsafe_allow_html=True)
                for log in l4_logs:
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.02); padding: 6px 10px; border-radius: 4px; margin-bottom: 4px; border-left: 2px solid #00d4ff;">
                        <div style="font-size: 0.6rem; color: #666;">{log.get('timestamp', '')}</div>
                        <div style="font-size: 0.7rem; color: #ddd;">{log.get('message', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        with st.expander("⚡ L5 Execution Engine", expanded=False):
            exec_data = state.get("execution", {})
            st.markdown(f"""
            <div class="sub-metric-grid">
                <div class="sub-metric-item"><div class="metric-small">Slippage</div><div style="color:#00ff9d;">{exec_data.get('slippage', 0.0):.3f}%</div></div>
                <div class="sub-metric-item"><div class="metric-small">Fill Rate</div><div style="color:#00eaff;">{exec_data.get('fill_rate', 100):.1f}%</div></div>
                <div class="sub-metric-item"><div class="metric-small">Latency</div><div style="color:#ffa500;">{exec_data.get('latency_ms', 0):.0f}ms</div></div>
                <div class="sub-metric-item"><div class="metric-small">Orders/min</div><div style="color:#fff;">{exec_data.get('orders_per_min', 0):.1f}</div></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add L5 logs
            l5_logs = state_manager.get_layer_logs("L5", 3)
            if l5_logs:
                st.markdown("<div style='margin-top: 15px; font-size: 0.7rem; color: #888;'>RECENT ACTIVITY</div>", unsafe_allow_html=True)
                for log in l5_logs:
                     st.markdown(f"""
                     <div style="background: rgba(255,255,255,0.02); padding: 6px 10px; border-radius: 4px; margin-bottom: 4px; border-left: 2px solid #00ff9d;">
                         <div style="font-size: 0.6rem; color: #666;">{log.get('timestamp', '')}</div>
                         <div style="font-size: 0.7rem; color: #ddd;">{log.get('message', '')}</div>
                     </div>
                     """, unsafe_allow_html=True)

    with col_b:
        with st.expander("🧩 L6-L9 High-Level Autonomy", expanded=False):
            thoughts = state.get("agentic_log", [])
            thought = thoughts[0]["thought"][:150] if thoughts else "Observing market dynamics..."
            st.markdown(f"""
            <div class="sub-metric-item">
                <div class="metric-small">Strategist reasoning</div>
                <div style="color:#ddd; font-style:italic; font-size:0.85rem; padding-top:5px;">"{thought}..."</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add L6-L9 logs
            st.markdown("<div style='margin-top: 15px; font-size: 0.7rem; color: #888;'>LAYER ACTIVITY</div>", unsafe_allow_html=True)
            for layer in ["L6", "L7", "L8", "L9"]:
                layer_logs = state_manager.get_layer_logs(layer, 2)
                if layer_logs:
                    st.markdown(f"<div style='font-size: 0.7rem; color: #00eaff; margin-top: 8px; font-weight: 600;'>{layer} Activity</div>", unsafe_allow_html=True)
                    for log in layer_logs:
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.02); padding: 4px 8px; border-radius: 3px; margin-bottom: 3px; border-left: 1px solid #00eaff;">
                            <div style="font-size: 0.55rem; color: #666;">{log.get('timestamp', '')}</div>
                            <div style="font-size: 0.65rem; color: #ddd;">{log.get('message', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)

    # --- MODEL PERFORMANCE & LEADERBOARD COMPARISON ---
    st.markdown('<div style="margin: 40px 0 20px 0;"><div class="layer-title">Model Performance & Leaderboard Comparison</div></div>', unsafe_allow_html=True)
    
    try:
        from src.models.benchmark import ModelBenchmark, INTERNAL_MODELS, GLOBAL_LEADERBOARD
        bench = ModelBenchmark(model_version="v6.5")
        
        # Get REAL data from state
        perf = state.get("performance", {})
        trades = state.get("trade_history", [])
        bench_data = state.get("benchmark", {})
        per_model = bench_data.get("per_model", {})
        
        # ── Per-Model Accuracy Cards ──
        model_cols = st.columns(4)
        model_keys = ["lightgbm", "patchtst", "rl_agent", "strategist"]
        model_icons = ["🌲", "📈", "🤖", "🧠"]
        model_colors = ["#00ff9d", "#00d4ff", "#ffa500", "#ff4d6d"]
        
        for i, mkey in enumerate(model_keys):
            with model_cols[i]:
                info = INTERNAL_MODELS.get(mkey, {})
                mdata = per_model.get(mkey, {})
                total = mdata.get("total", 0)
                correct = mdata.get("correct", 0)
                
                if total > 0:
                    acc = correct / total
                    acc_display = f"{acc*100:.1f}%"
                    acc_color = model_colors[i]
                    sample_text = f"n={total} real predictions"
                    
                    # Compare to leaderboard
                    metric_key = info.get("metric_key", "")
                    lb = GLOBAL_LEADERBOARD.get(metric_key, {})
                    models = lb.get("models", {})
                    beats = sum(1 for v in models.values() if acc > v)
                    total_lb = len(models)
                    rank = total_lb + 1 - beats
                    rank_text = f"Rank #{rank}/{total_lb+1}"
                    
                    top_model = max(models, key=models.get) if models else "N/A"
                    top_val = max(models.values()) if models else 0
                    vs_top = f"Top: {top_model} ({top_val*100:.0f}%)"
                    status_emoji = "✅" if acc > top_val else "📈"
                else:
                    acc_display = "--"
                    acc_color = "#333"
                    sample_text = "Awaiting predictions..."
                    rank_text = ""
                    vs_top = ""
                    status_emoji = "⏳"
                
                st.markdown(f"""
                <div style="background: rgba(20,20,40,0.6); padding: 18px; border-radius: 12px; border: 1px solid {model_colors[i]}22; text-align: center; min-height: 220px;">
                    <div style="font-size: 1.5rem;">{model_icons[i]}</div>
                    <div style="font-size: 0.75rem; color: {model_colors[i]}; font-family: Orbitron; margin-top: 5px;">{info.get('display_name', mkey)}</div>
                    <div style="font-size: 2rem; font-weight: 700; color: {acc_color}; font-family: Orbitron; margin: 8px 0;">{acc_display}</div>
                    <div style="font-size: 0.6rem; color: #888;">{sample_text}</div>
                    <div style="font-size: 0.55rem; color: #666; margin-top: 6px;">{rank_text}</div>
                    <div style="font-size: 0.5rem; color: #555; margin-top: 2px;">{vs_top}</div>
                    <div style="border-top: 1px solid rgba(255,255,255,0.05); margin-top: 10px; padding-top: 8px;">
                        <div style="font-size: 0.5rem; color: #555;">TASK</div>
                        <div style="font-size: 0.55rem; color: #888;">{info.get('task', '')}</div>
                        <div style="font-size: 0.5rem; color: #555; margin-top: 4px;">DATA SOURCE</div>
                        <div style="font-size: 0.55rem; color: #888;">{info.get('data_source', '')}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # ── Per-Model Leaderboard Charts (only for models with data) ──
        models_with_data = [mkey for mkey in model_keys if per_model.get(mkey, {}).get("total", 0) > 0]
        
        if models_with_data:
            chart_cols = st.columns(min(len(models_with_data), 2))
            for idx, mkey in enumerate(models_with_data[:2]):
                info = INTERNAL_MODELS.get(mkey, {})
                mdata = per_model.get(mkey, {})
                acc = mdata["correct"] / mdata["total"] if mdata["total"] > 0 else 0
                metric_key = info.get("metric_key", "")
                our_label = f"{model_icons[model_keys.index(mkey)]} {info.get('display_name', mkey)}"
                
                with chart_cols[idx]:
                    fig = bench.generate_leaderboard_chart(metric_key, acc, our_label)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=f"lb_{mkey}")
        
        # ── Ensemble Summary Row ──
        has_trades = bool(trades) and len(trades) > 0
        closed_trades = [t for t in trades if 'pnl' in t] if has_trades else []
        
        if closed_trades or models_with_data:
            st.markdown('<div style="margin: 20px 0 10px 0;"><div style="font-size: 0.8rem; color: #00eaff; font-family: Orbitron;">ENSEMBLE PERFORMANCE (All Models Combined)</div></div>', unsafe_allow_html=True)
            ens_cols = st.columns(4)
            
            # Win Rate
            with ens_cols[0]:
                if closed_trades:
                    wr = sum(1 for t in closed_trades if t['pnl'] > 0) / len(closed_trades)
                    wr_display = f"{wr*100:.1f}%"
                    wr_color = "#00ff9d" if wr > 0.5 else "#ff4d6d"
                else:
                    wr_display = "--"
                    wr_color = "#333"
                st.markdown(f'<div style="background:rgba(20,20,40,0.6);padding:12px;border-radius:10px;text-align:center;"><div style="font-size:0.55rem;color:#888;">ENSEMBLE WIN RATE</div><div style="font-size:1.5rem;font-weight:700;color:{wr_color};font-family:Orbitron;">{wr_display}</div><div style="font-size:0.5rem;color:#555;">Closed trades: {len(closed_trades)}</div></div>', unsafe_allow_html=True)
            
            # Sharpe
            with ens_cols[1]:
                rets = [t['return_pct'] / 100 for t in closed_trades if 'return_pct' in t]
                if len(rets) >= 2:
                    import numpy as np
                    arr = np.array(rets)
                    sharpe = (np.mean(arr) / np.std(arr)) * np.sqrt(252) if np.std(arr) > 0 else 0.0
                    sh_display = f"{sharpe:.2f}"
                    sh_color = "#00ff9d" if sharpe > 1.0 else "#ffa500" if sharpe > 0 else "#ff4d6d"
                else:
                    sh_display = "--"
                    sh_color = "#333"
                st.markdown(f'<div style="background:rgba(20,20,40,0.6);padding:12px;border-radius:10px;text-align:center;"><div style="font-size:0.55rem;color:#888;">SHARPE RATIO</div><div style="font-size:1.5rem;font-weight:700;color:{sh_color};font-family:Orbitron;">{sh_display}</div><div style="font-size:0.5rem;color:#555;">Annualized</div></div>', unsafe_allow_html=True)
            
            # Ensemble Direction Accuracy
            with ens_cols[2]:
                ens_preds = bench_data.get("predictions", [])
                ens_acts = bench_data.get("actuals", [])
                if ens_preds and ens_acts:
                    n = min(len(ens_preds), len(ens_acts))
                    dp = [(p,a) for p,a in zip(ens_preds[:n], ens_acts[:n]) if a != 0]
                    ens_acc = sum(1 for p,a in dp if (p>0)==(a>0)) / len(dp) if dp else 0
                    ea_display = f"{ens_acc*100:.1f}%"
                    ea_color = "#00ff9d" if ens_acc > 0.55 else "#ffa500" if ens_acc > 0.45 else "#ff4d6d"
                    ea_n = len(dp)
                else:
                    ea_display = "--"
                    ea_color = "#333"
                    ea_n = 0
                st.markdown(f'<div style="background:rgba(20,20,40,0.6);padding:12px;border-radius:10px;text-align:center;"><div style="font-size:0.55rem;color:#888;">ENSEMBLE ACCURACY</div><div style="font-size:1.5rem;font-weight:700;color:{ea_color};font-family:Orbitron;">{ea_display}</div><div style="font-size:0.5rem;color:#555;">n={ea_n} directional</div></div>', unsafe_allow_html=True)
            
            # Total P&L
            with ens_cols[3]:
                if closed_trades:
                    total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
                    pnl_display = f"${total_pnl:+,.2f}"
                    pnl_color = "#00ff9d" if total_pnl > 0 else "#ff4d6d"
                else:
                    pnl_display = "--"
                    pnl_color = "#333"
                st.markdown(f'<div style="background:rgba(20,20,40,0.6);padding:12px;border-radius:10px;text-align:center;"><div style="font-size:0.55rem;color:#888;">TOTAL P&L</div><div style="font-size:1.5rem;font-weight:700;color:{pnl_color};font-family:Orbitron;">{pnl_display}</div><div style="font-size:0.5rem;color:#555;">Real execution</div></div>', unsafe_allow_html=True)
            
            # Radar chart for models with data
            if len(models_with_data) >= 2:
                our_scores = {}
                for mkey in models_with_data:
                    info = INTERNAL_MODELS.get(mkey, {})
                    mdata = per_model[mkey]
                    acc = mdata["correct"] / mdata["total"]
                    our_scores[info["metric_key"]] = acc
                
                fig_radar = bench.generate_radar_chart(our_scores)
                if fig_radar:
                    st.plotly_chart(fig_radar, use_container_width=True, key="bench_radar")
                    
            # Timeline if enough history
            fig_timeline = bench.generate_performance_timeline("ensemble_win_rate")
            if fig_timeline:
                st.plotly_chart(fig_timeline, use_container_width=True, key="bench_timeline")
        
        elif not models_with_data and not closed_trades:
            # No data at all
            st.markdown("""
            <div style="background: rgba(20,20,40,0.6); padding: 40px; border-radius: 16px; border: 1px solid rgba(0,234,255,0.1); text-align: center;">
                <div style="font-size: 3rem;">🎯</div>
                <div style="color: #00eaff; font-size: 1.2rem; font-family: Orbitron; margin-top: 15px;">Awaiting Real Trade Data</div>
                <div style="color: #888; font-size: 0.85rem; margin-top: 10px; max-width: 600px; margin-left: auto; margin-right: auto;">
                    Each model (LightGBM, PatchTST, RL Agent, Strategist) will be benchmarked individually once the system starts trading.
                    All metrics are from <b style="color:#00ff9d">real execution data only</b> — zero mock scores.
                </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Benchmark module loading: {e}")

    # --- NEWS IMPACT STREAM ---
    st.markdown('<div style="margin: 40px 0 20px 0;"><div class="layer-title">Sentinel Intelligence: News & Event Impact Stream</div></div>', unsafe_allow_html=True)
    org_news = sent.get("organized_news", [])
    if org_news:
        for item in org_news[:10]:
            src = item.get('source', 'Unknown')
            ev = item.get('event', 'GENERAL')
            txt = item.get('text', '')
            tm = item.get('time', '--:--')
            impact = "HIGH" if "BINANCE" in src.upper() else "MED"
            i_col = "#00ff9d" if impact == "HIGH" else "#00d4ff"
            
            st.markdown(f"""
            <div style="background: rgba(20,20,40,0.6); padding: 15px; border-radius: 12px; border-left: 4px solid {i_col}; margin-bottom: 12px; display: flex; align-items: center; gap: 20px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-family: 'Courier New'; color: #444; font-size: 0.8rem;">{tm}</div>
                <div style="flex-grow: 1;">
                    <div style="display:flex; gap:10px; margin-bottom:4px;">
                        <span style="background:{i_col}22; color:{i_col}; font-size:0.6rem; padding:2px 6px; border-radius:4px; font-weight:700;">{src}</span>
                        <span style="background:rgba(255,255,255,0.05); color:#888; font-size:0.6rem; padding:2px 6px; border-radius:4px;">{ev}</span>
                    </div>
                    <div style="color: #ddd; font-size: 0.9rem;">{txt}</div>
                </div>
                <div style="text-align:right;">
                    <div class="metric-small">IMPACT</div>
                    <div style="color:{i_col}; font-weight:700; font-family:Orbitron;">{impact}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Awaiting live news stream impact synchronization...")

if __name__ == "__main__":
    run_app()
    time.sleep(2)
    st.rerun()
