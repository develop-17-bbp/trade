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
    page_title="Autonomous Trading Desk - Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for high-end aesthetic
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%) !important;
        color: #e0e0e0 !important;
    }
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%) !important;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        transition: all 0.3s ease !important;
    }
    .stMetric:hover {
        border-color: rgba(0, 212, 255, 0.3) !important;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1) !important;
    }
    .agent-thought {
        background: rgba(0, 0, 0, 0.3) !important;
        border-left: 5px solid #58a6ff !important;
        padding: 10px !important;
        margin-bottom: 10px !important;
        border-radius: 5px !important;
        font-family: 'Courier New', monospace !important;
        font-size: 0.9rem !important;
    }
    .memory-card {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
        padding: 12px !important;
        margin-bottom: 8px !important;
        font-size: 0.85rem !important;
    }
    .regime-indicator {
        font-size: 2rem !important;
        font-weight: 700 !important;
        padding: 20px !important;
        border-radius: 12px !important;
        margin: 16px 0 !important;
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 255, 136, 0.1) 100%) !important;
        border: 2px solid !important;
        text-align: center !important;
    }
    .regime-trending {
        border-color: #00ff88 !important;
        color: #00ff88 !important;
    }
    .regime-ranging {
        border-color: #ffa500 !important;
        color: #ffa500 !important;
    }
    .asset-signal {
        padding: 4px 12px !important;
        border-radius: 20px !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        display: inline-block !important;
    }
    .signal-long {
        background: rgba(0, 255, 136, 0.2) !important;
        color: #00ff88 !important;
        border: 1px solid rgba(0, 255, 136, 0.3) !important;
    }
    .signal-short {
        background: rgba(255, 77, 77, 0.2) !important;
        color: #ff4d4d !important;
        border: 1px solid rgba(255, 77, 77, 0.3) !important;
    }
    .signal-flat {
        background: rgba(255, 165, 0, 0.2) !important;
        color: #ffa500 !important;
        border: 1px solid rgba(255, 165, 0, 0.3) !important;
    }
    .signal-veto {
        background: rgba(128, 128, 128, 0.2) !important;
        color: #888 !important;
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
    }
    .header-title {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        margin-bottom: 8px !important;
    }
    .header-subtitle {
        color: #888 !important;
        font-size: 1.1rem !important;
        font-weight: 400 !important;
    }
    .widget-title {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        margin-bottom: 16px !important;
        color: #fff !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }
    .widget-title::before {
        content: '' !important;
        width: 4px !important;
        height: 20px !important;
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%) !important;
        border-radius: 2px !important;
    }
    .reasoning-feed {
        background: rgba(0, 0, 0, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        max-height: 300px !important;
        overflow-y: auto !important;
        position: relative !important;
    }
    .reasoning-line {
        margin-bottom: 8px !important;
        padding-left: 20px !important;
        border-left: 2px solid transparent !important;
        transition: all 0.3s ease !important;
    }
    .reasoning-line.new {
        border-left-color: #00d4ff !important;
        background: rgba(0, 212, 255, 0.05) !important;
    }
    .reasoning-line .timestamp {
        color: #666 !important;
        font-size: 0.8rem !important;
        margin-right: 8px !important;
    }
    .equity-stats {
        display: flex !important;
        justify-content: space-between !important;
        margin-top: 16px !important;
        padding: 16px !important;
        background: rgba(0, 255, 136, 0.05) !important;
        border: 1px solid rgba(0, 255, 136, 0.2) !important;
        border-radius: 8px !important;
    }
    .stat {
        text-align: center !important;
    }
    .stat-value {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #00ff88 !important;
    }
    .stat-label {
        color: #888 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.3); }
        50% { box-shadow: 0 0 30px rgba(0, 212, 255, 0.5); }
    }
    .glowing { animation: glow 2s ease-in-out infinite; }
</style>
""", unsafe_allow_html=True)

state_manager = DashboardState()

def run_app():
    # Header
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.02); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h1 class="header-title">🤖 Autonomous Trading Desk</h1>
        <div class="header-subtitle">AI-Driven Crypto Trading System | Layer 7: Full Autonomy</div>
    </div>
    """, unsafe_allow_html=True)

    # Top Row: Metrics
    col1, col2, col3, col4 = st.columns(4)
    state = state_manager.get_full_state()
    
    with col1:
        st.metric("Total Portfolio P&L", f"${state['portfolio']['pnl']:,.2f}", delta=f"{state['portfolio']['return']:.2f}%")
    with col2:
        regime = state.get("active_assets", {}).get("BTC", {}).get("regime", "RANGING")
        st.metric("Current Regime", regime)
    with col3:
        st.metric("System Condition", state["status"])
    with col4:
        st.metric("Historical Accuracy", f"{state.get('accuracy', 0.5)*100:.1f}%")

    # Main Layout
    main_col, sidebar_col = st.columns([3, 1])

    with main_col:
        # Strategist Hub
        st.markdown('<h3 class="widget-title">🤖 Strategist Hub</h3>', unsafe_allow_html=True)
        st.markdown('<div class="reasoning-feed">', unsafe_allow_html=True)
        thoughts = state.get("agentic_log", [])
        for t in thoughts[:10]:
            is_new = "new" if thoughts.index(t) < 2 else ""
            st.markdown(f"""
            <div class="reasoning-line {is_new}">
                <span class="timestamp">{t['timestamp']}</span>
                <span class="content">{t['asset']} - {t['regime']}: {t['thought']}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Equity Curve
        st.markdown('<h3 class="widget-title">📈 Real-Time Equity Curve</h3>', unsafe_allow_html=True)
        equity_data = state["portfolio"]["equity_curve"]
        if equity_data:
            df_equity = pd.DataFrame(equity_data)
            df_equity['t'] = pd.to_datetime(df_equity['t'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_equity['t'], y=df_equity['v'], mode='lines', 
                                    line=dict(color='#00d4ff', width=3),
                                    fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.1)'))
            fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20), height=350,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Equity Stats
            st.markdown("""
            <div class="equity-stats">
                <div class="stat"><div class="stat-value">+2.34%</div><div class="stat-label">Return</div></div>
                <div class="stat"><div class="stat-value">1.87</div><div class="stat-label">Sharpe</div></div>
                <div class="stat"><div class="stat-value">-1.2%</div><div class="stat-label">Drawdown</div></div>
                <div class="stat"><div class="stat-value">68.5%</div><div class="stat-label">Win Rate</div></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No equity data recorded yet.")
    
    # Phase 6: Advanced Learning Section
    st.markdown('<h3 class="widget-title">🧬 Phase 6: Advanced Learning</h3>', unsafe_allow_html=True)
    advanced_learning = state.get("advanced_learning", {})
    
    if advanced_learning.get("regimes"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<h4 style="color: #00d4ff; margin-top: 0;">📊 Market Regimes</h4>', unsafe_allow_html=True)
            for asset, regime in list(advanced_learning.get("regimes", {}).items())[:3]:
                if isinstance(regime, dict):
                    regime_type = regime.get("regime_type", "UNKNOWN")
                    confidence = regime.get("confidence", 0)
                    color = "#00ff88" if "TREND" in regime_type else "#ffa500" if "RANGE" in regime_type else "#ff4d4d"
                    st.markdown(f"""
                    <div class="memory-card" style="border-left: 3px solid {color};">
                        <div style="font-weight: 600; color: {color};">{asset}: {regime_type}</div>
                        <div style="font-size: 0.8rem; color: #888;">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<h4 style="color: #00d4ff; margin-top: 0;">🎯 Adaptive Strategies</h4>', unsafe_allow_html=True)
            for asset, strategy in list(advanced_learning.get("strategies", {}).items())[:3]:
                if isinstance(strategy, dict):
                    name = strategy.get("strategy_name", "Unknown")
                    perf = strategy.get("predicted_performance", 0)
                    perf_color = "#00ff88" if perf > 0.5 else "#ffa500" if perf > 0.3 else "#ff4d4d"
                    st.markdown(f"""
                    <div class="memory-card" style="border-left: 3px solid {perf_color};">
                        <div style="font-weight: 600;">{asset}: {name}</div>
                        <div style="font-size: 0.8rem; color: {perf_color};">Performance: {perf:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col3:
            st.markdown('<h4 style="color: #00d4ff; margin-top: 0;">🔍 Pattern Recognition</h4>', unsafe_allow_html=True)
            patterns = advanced_learning.get("patterns", {})
            pattern_counts = {
                "Momentum Breakouts": len(patterns.get("momentum_breakout", [])),
                "Mean Reversion": len(patterns.get("mean_reversion", [])),
                "Volatility Jumps": len(patterns.get("volatility_expansion", []))
            }
            for pattern_type, count in pattern_counts.items():
                if count > 0:
                    st.markdown(f"""
                    <div class="memory-card">
                        <div style="font-weight: 600; color: #00ff88;">{pattern_type}</div>
                        <div style="font-size: 0.8rem; color: #888;">{count} assets detected</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Phase 6 Advanced Learning awaiting backtest results...")

    with sidebar_col:
        # Memory Vault
        st.markdown('<h3 class="widget-title">🧠 Memory Vault</h3>', unsafe_allow_html=True)
        memory_hits = state.get("memory_hits", [])
        if memory_hits:
            for hit in memory_hits[:3]:
                pnl = hit['metadata'].get('pnl_pct', 0)
                color = "#00ff88" if pnl > 0 else "#ff4d4d"
                st.markdown(f"""
                <div class="memory-card">
                    <div style="color: {color}; font-weight: 600;">{hit['metadata'].get('similarity', 0):.0f}% Match</div>
                    <div>{hit['document'][:80]}...</div>
                    <div style="color: #888; font-size: 0.8rem;">Outcome: {pnl:+.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Awaiting setups...")

        # Market Regime
        st.markdown('<h3 class="widget-title">🎭 Market Regime</h3>', unsafe_allow_html=True)
        regime = state.get("active_assets", {}).get("BTC", {}).get("regime", "RANGING").lower()
        regime_class = "regime-trending" if regime == "trending" else "regime-ranging"
        st.markdown(f"""
        <div class="regime-indicator {regime_class} glowing">{regime.upper()}</div>
        """, unsafe_allow_html=True)

        # On-Chain Alpha (Layer 4)
        st.markdown('<h3 class="widget-title">⛓️ On-Chain Alpha</h3>', unsafe_allow_html=True)
        oc = state.get("onchain_metrics", {})
        if oc:
            whale = oc.get("whale_metrics", {})
            network = oc.get("network_metrics", {})
            w_sent = whale.get("whale_sentiment", "NEUTRAL")
            w_color = "#00ff88" if w_sent == "BULLISH" else "#ff4d4d" if w_sent == "BEARISH" else "#888"
            
            st.markdown(f"""
            <div class="memory-card" style="border-left: 3px solid {w_color};">
                <div style="font-weight: 600;">Whales: {w_sent}</div>
                <div style="font-size: 0.8rem; color: #888;">
                    Flow: {whale.get('net_exchange_flow', 0):+.2f} BTC<br/>
                    Growth: {network.get('address_growth_pct', 0):+.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Syncing L4 Telemetry...")

if __name__ == "__main__":
    run_app()
    time.sleep(2)
    st.rerun()