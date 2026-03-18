"""
Page 3: Agent Intelligence Overlay
NEW page — visualizes the 12-agent overlay status, votes, weights, consensus.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st
import time
import json
import os
from datetime import datetime
import plotly.graph_objects as go

import textwrap
from src.api.state import DashboardState
from src.dashboard.theme import MARKETEDGE_CSS, metric_card, plotly_layout

st.markdown(MARKETEDGE_CSS, unsafe_allow_html=True)


def _html(html_str: str):
    st.markdown(textwrap.dedent(html_str), unsafe_allow_html=True)

state_manager = DashboardState()
state = state_manager.get_full_state()
agent_overlay = state.get("agent_overlay", {})
last_decision = agent_overlay.get("last_decision", {})
agent_votes = agent_overlay.get("agent_votes", {})
agent_weights = agent_overlay.get("agent_weights", {})

# ═══ HEADER ═══
st.markdown("""
<div class="main-header">
    <div>
        <div class="title">AGENT INTELLIGENCE</div>
        <div class="subtext">12-AGENT OVERLAY | BAYESIAN CONSENSUS | LOSS PREVENTION</div>
    </div>
    <div style="text-align:right;">
        <div class="subtext">AUTO-REFRESH: 5s</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══ OVERLAY STATUS BAR ═══
enabled = agent_overlay.get("enabled", False)
cycle_count = agent_overlay.get("cycle_count", 0)
last_cycle = agent_overlay.get("last_cycle_time", "")
data_quality = agent_overlay.get("data_quality", 0.0)
pnl_mode = agent_overlay.get("daily_pnl_mode", "NORMAL")
consensus = agent_overlay.get("consensus_level", "N/A")

mode_colors = {"NORMAL": "#22c55e", "CAUTION": "#ffa500", "DEFENSIVE": "#ff8800",
               "HALT": "#ef4444", "PRESERVATION": "#3b82f6", "APPROACHING": "#00d4ff"}

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    color = "#22c55e" if enabled else "#ef4444"
    st.markdown(f"""<div class="glass-card" style="text-align:center;">
        <div class="metric-label">OVERLAY</div>
        <div class="metric-value" style="color:{color};">{"ACTIVE" if enabled else "OFF"}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="glass-card" style="text-align:center;">
        <div class="metric-label">CYCLES</div>
        <div class="metric-value metric-value-cyan">{cycle_count}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    dq_color = "#22c55e" if data_quality > 0.7 else "#ffa500" if data_quality > 0.3 else "#ef4444"
    st.markdown(f"""<div class="glass-card" style="text-align:center;">
        <div class="metric-label">DATA QUALITY</div>
        <div class="metric-value" style="color:{dq_color};">{data_quality:.1%}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    mc = mode_colors.get(pnl_mode, "#888")
    st.markdown(f"""<div class="glass-card" style="text-align:center;">
        <div class="metric-label">PNL MODE</div>
        <div class="metric-value" style="color:{mc}; font-size:1.2rem;">{pnl_mode}</div>
    </div>""", unsafe_allow_html=True)
with c5:
    cons_colors = {"STRONG": "#22c55e", "MODERATE": "#3b82f6", "WEAK": "#ffa500",
                   "CONFLICT": "#ef4444", "VETOED": "#ff0000"}
    cc = cons_colors.get(consensus, "#888")
    st.markdown(f"""<div class="glass-card" style="text-align:center;">
        <div class="metric-label">CONSENSUS</div>
        <div class="metric-value" style="color:{cc}; font-size:1.2rem;">{consensus}</div>
    </div>""", unsafe_allow_html=True)
with c6:
    direction = last_decision.get("direction", 0)
    dir_label = "LONG" if direction > 0 else "SHORT" if direction < 0 else "FLAT"
    dir_color = "#22c55e" if direction > 0 else "#ef4444" if direction < 0 else "#888"
    conf = last_decision.get("confidence", 0)
    st.markdown(f"""<div class="glass-card" style="text-align:center;">
        <div class="metric-label">DECISION</div>
        <div class="metric-value" style="color:{dir_color};">{dir_label}</div>
        <div style="font-size:0.7rem; color:#888;">{conf:.1%} conf</div>
    </div>""", unsafe_allow_html=True)


# ═══ ENHANCED DECISION DETAIL ═══
st.markdown('<div class="section-title">LATEST ENHANCED DECISION</div>', unsafe_allow_html=True)

if last_decision:
    dc1, dc2, dc3, dc4 = st.columns(4)
    with dc1:
        st.markdown(f"""<div class="glass-card">
            <div class="metric-label">ASSET</div>
            <div class="metric-value metric-value-cyan">{last_decision.get('asset', 'N/A')}</div>
        </div>""", unsafe_allow_html=True)
    with dc2:
        scale = last_decision.get("position_scale", 0)
        st.markdown(f"""<div class="glass-card">
            <div class="metric-label">POSITION SCALE</div>
            <div class="metric-value metric-value-green">{scale:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with dc3:
        veto = last_decision.get("veto", False)
        v_color = "#ef4444" if veto else "#22c55e"
        st.markdown(f"""<div class="glass-card">
            <div class="metric-label">VETO</div>
            <div class="metric-value" style="color:{v_color};">{"ACTIVE" if veto else "CLEAR"}</div>
        </div>""", unsafe_allow_html=True)
    with dc4:
        strat = last_decision.get("strategy", "N/A")
        st.markdown(f"""<div class="glass-card">
            <div class="metric-label">STRATEGY</div>
            <div class="metric-value" style="font-size:0.9rem; color:#ffa500;">{strat}</div>
        </div>""", unsafe_allow_html=True)
else:
    st.info("No agent decision yet. Start the trading system to see agent overlay results.")


# ═══ AGENT VOTES GRID ═══
st.markdown('<div class="section-title">12 AGENT VOTES</div>', unsafe_allow_html=True)

# Define agent display info
AGENT_DISPLAY = {
    "DataIntegrityValidator": {"icon": "🔍", "color": "#888", "role": "PRE-GATE"},
    "MarketStructure": {"icon": "🏗️", "color": "#3b82f6", "role": "ANALYSIS"},
    "RegimeIntelligence": {"icon": "🌊", "color": "#3498db", "role": "ANALYSIS"},
    "MeanReversion": {"icon": "🔄", "color": "#9b59b6", "role": "ANALYSIS"},
    "TrendMomentum": {"icon": "📈", "color": "#2ecc71", "role": "ANALYSIS"},
    "RiskGuardian": {"icon": "🛡️", "color": "#e74c3c", "role": "ANALYSIS"},
    "SentimentDecoder": {"icon": "💬", "color": "#f39c12", "role": "ANALYSIS"},
    "TradeTiming": {"icon": "⏱️", "color": "#1abc9c", "role": "ANALYSIS"},
    "PortfolioOptimizer": {"icon": "📊", "color": "#e91e63", "role": "ANALYSIS"},
    "PatternMatcher": {"icon": "🧩", "color": "#ff9800", "role": "ANALYSIS"},
    "LossPreventionGuard": {"icon": "🚨", "color": "#ff0000", "role": "VETO"},
    "DecisionAuditor": {"icon": "✅", "color": "#888", "role": "POST-AUDIT"},
}

if agent_votes:
    # Render in 3x4 grid
    agent_names = list(agent_votes.keys())
    for row_start in range(0, len(agent_names), 4):
        cols = st.columns(4)
        for idx, col in enumerate(cols):
            ai = row_start + idx
            if ai >= len(agent_names):
                break
            name = agent_names[ai]
            vote = agent_votes[name]
            display = AGENT_DISPLAY.get(name, {"icon": "🤖", "color": "#888", "role": "AGENT"})

            direction = vote.get("direction", 0)
            confidence = vote.get("confidence", 0)
            reasoning = vote.get("reasoning", "")[:80]
            weight = agent_weights.get(name, 1.0)

            dir_label = "LONG" if direction > 0 else "SHORT" if direction < 0 else "FLAT"
            dir_class = "dir-long" if direction > 0 else "dir-short" if direction < 0 else "dir-flat"
            conf_bar_width = int(confidence * 100)
            conf_color = "#22c55e" if confidence > 0.7 else "#ffa500" if confidence > 0.4 else "#ef4444"

            with col:
                _html(f"""
                <div class="agent-vote-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <div>
                            <span style="font-size:1.1rem;">{display['icon']}</span>
                            <span class="agent-name" style="margin-left:6px;">{name}</span>
                        </div>
                        <span style="font-size:0.55rem; padding:2px 6px; border-radius:8px; background:rgba(255,255,255,0.05); color:#666;">{display['role']}</span>
                    </div>
                    <div class="agent-direction {dir_class}">{dir_label}</div>
                    <div style="margin: 6px 0;">
                        <div style="display:flex; justify-content:space-between; font-size:0.65rem; color:#888;">
                            <span>Confidence</span><span style="color:{conf_color};">{confidence:.1%}</span>
                        </div>
                        <div style="height:4px; background:rgba(255,255,255,0.05); border-radius:2px; margin-top:3px;">
                            <div style="width:{conf_bar_width}%; height:100%; background:{conf_color}; border-radius:2px;"></div>
                        </div>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.6rem; color:#666; margin-top:4px;">
                        <span>Weight: {weight:.2f}</span>
                    </div>
                    <div style="font-size:0.6rem; color:#555; margin-top:6px; font-style:italic; border-top:1px solid rgba(255,255,255,0.03); padding-top:5px;">
                        {reasoning}
                    </div>
                </div>
                """)
else:
    _html("""
    <div style="background: rgba(20,20,40,0.6); padding: 40px; border-radius: 16px; border: 1px solid rgba(0,234,255,0.1); text-align: center;">
        <div style="font-size: 3rem;">🤖</div>
        <div style="color: #3b82f6; font-size: 1.2rem; font-family: Inter; margin-top: 15px;">Awaiting Agent Votes</div>
        <div style="color: #888; font-size: 0.85rem; margin-top: 10px;">
            The 12-agent intelligence overlay will populate once the trading system processes its first cycle.
        </div>
    </div>
    """)


# ═══ CONSENSUS VISUALIZATION ═══
if agent_votes:
    st.markdown('<div class="section-title">CONSENSUS ANALYSIS</div>', unsafe_allow_html=True)

    # Count votes
    long_votes = sum(1 for v in agent_votes.values() if v.get("direction", 0) > 0)
    short_votes = sum(1 for v in agent_votes.values() if v.get("direction", 0) < 0)
    flat_votes = sum(1 for v in agent_votes.values() if v.get("direction", 0) == 0)
    total_agents = len(agent_votes)

    col_pie, col_bar = st.columns(2)

    with col_pie:
        fig_pie = go.Figure(go.Pie(
            labels=["LONG", "SHORT", "FLAT"],
            values=[long_votes, short_votes, flat_votes],
            marker=dict(colors=["#22c55e", "#ef4444", "#555"]),
            textinfo="label+value", textfont=dict(color="#fff", size=14),
            hole=0.4,
        ))
        fig_pie.update_layout(**plotly_layout(
            title=f"Vote Distribution ({total_agents} agents)", height=300,
        ))
        st.plotly_chart(fig_pie, width="stretch")

    with col_bar:
        # Weighted confidence by direction
        long_conf = sum(v.get("confidence", 0) * agent_weights.get(n, 1.0) for n, v in agent_votes.items() if v.get("direction", 0) > 0)
        short_conf = sum(v.get("confidence", 0) * agent_weights.get(n, 1.0) for n, v in agent_votes.items() if v.get("direction", 0) < 0)
        flat_conf = sum(v.get("confidence", 0) * agent_weights.get(n, 1.0) for n, v in agent_votes.items() if v.get("direction", 0) == 0)
        total_conf = long_conf + short_conf + flat_conf + 1e-10

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=["LONG", "SHORT", "FLAT"],
            y=[long_conf / total_conf, short_conf / total_conf, flat_conf / total_conf],
            marker_color=["#22c55e", "#ef4444", "#555"],
            text=[f"{long_conf/total_conf:.1%}", f"{short_conf/total_conf:.1%}", f"{flat_conf/total_conf:.1%}"],
            textposition="auto", textfont=dict(color="#fff", size=14),
        ))
        fig_bar.add_hline(y=0.55, line_dash="dash", line_color="#3b82f6",
                          annotation_text="Consensus Threshold (55%)", annotation_font_color="#3b82f6")
        fig_bar.update_layout(**plotly_layout(
            title="Weighted Probability (Bayesian)", height=300,
            yaxis=dict(tickformat=".0%", range=[0, 1]),
        ))
        st.plotly_chart(fig_bar, width="stretch")


# ═══ AGENT WEIGHTS ═══
if agent_weights:
    st.markdown('<div class="section-title">BAYESIAN AGENT WEIGHTS</div>', unsafe_allow_html=True)

    names = list(agent_weights.keys())
    weights = [float(agent_weights[n]) for n in names]
    colors = [AGENT_DISPLAY.get(n, {}).get("color", "#888") for n in names]

    fig_weights = go.Figure(go.Bar(
        x=names, y=weights,
        marker_color=colors,
        text=[f"{w:.2f}" for w in weights],
        textposition="auto", textfont=dict(color="#fff"),
    ))
    fig_weights.add_hline(y=1.0, line_dash="dot", line_color="#555", annotation_text="Default (1.0)")
    fig_weights.update_layout(**plotly_layout(
        title="Dynamic Agent Weights (Bayesian EMA Updated)", height=350,
        xaxis=dict(tickangle=-45),
        yaxis=dict(range=[0, max(weights) * 1.3] if weights else [0, 3]),
    ))
    st.plotly_chart(fig_weights, width="stretch")


# ═══ LOSS PREVENTION DASHBOARD ═══
st.markdown('<div class="section-title">LOSS PREVENTION GUARDIAN</div>', unsafe_allow_html=True)

# PnL Mode Thresholds Visual
lp_config = {}
try:
    import yaml
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f) or {}
        lp_config = cfg.get("agents", {}).get("loss_prevention", {})
except Exception:
    pass

daily_target = lp_config.get("daily_target_pct", 1.0)
preservation = lp_config.get("preservation_threshold_pct", 0.8)
caution = lp_config.get("caution_threshold_pct", -0.5)
halt = lp_config.get("halt_threshold_pct", -1.0)

# Create gauge for current PnL mode
mode_values = {"HALT": -1.0, "DEFENSIVE": -0.7, "CAUTION": -0.3, "NORMAL": 0.3,
               "APPROACHING": 0.7, "PRESERVATION": 1.0}
mode_val = mode_values.get(pnl_mode, 0.3)

fig_mode = go.Figure(go.Indicator(
    mode="gauge+number",
    value=mode_val,
    number={"font": {"size": 1}},
    title={"text": f"PnL Mode: {pnl_mode}", "font": {"size": 16, "color": mode_colors.get(pnl_mode, "#888")}},
    gauge={
        "axis": {"range": [-1.5, 1.5], "tickvals": [-1.0, -0.5, 0, 0.8, 1.0],
                 "ticktext": ["HALT", "CAUTION", "NORMAL", "PRESERVE", "TARGET"],
                 "tickcolor": "#555", "tickfont": {"color": "#888"}},
        "bar": {"color": mode_colors.get(pnl_mode, "#888")},
        "bgcolor": "rgba(20,20,40,0.3)",
        "steps": [
            {"range": [-1.5, -1.0], "color": "rgba(255,0,0,0.15)"},
            {"range": [-1.0, -0.5], "color": "rgba(255,77,109,0.1)"},
            {"range": [-0.5, 0], "color": "rgba(255,165,0,0.1)"},
            {"range": [0, 0.8], "color": "rgba(0,255,157,0.05)"},
            {"range": [0.8, 1.0], "color": "rgba(0,234,255,0.1)"},
            {"range": [1.0, 1.5], "color": "rgba(0,255,157,0.15)"},
        ],
    },
))
fig_mode.update_layout(**plotly_layout(height=250))
st.plotly_chart(fig_mode, width="stretch")

# Mode description table
st.markdown("""
<div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; margin-top:10px;">
    <div class="glass-card" style="border-left: 3px solid #22c55e;">
        <div class="metric-label">NORMAL (0% to +0.8%)</div>
        <div style="color:#ccc; font-size:0.75rem;">Standard trading, full positions.</div>
    </div>
    <div class="glass-card" style="border-left: 3px solid #3b82f6;">
        <div class="metric-label">PRESERVATION (+0.8% to +1%)</div>
        <div style="color:#ccc; font-size:0.75rem;">Only high-confidence trades. Positions halved.</div>
    </div>
    <div class="glass-card" style="border-left: 3px solid #ffa500;">
        <div class="metric-label">CAUTION (-0.5% to 0%)</div>
        <div style="color:#ccc; font-size:0.75rem;">Tighter stops. Need 6+ agents agree.</div>
    </div>
    <div class="glass-card" style="border-left: 3px solid #ff8800;">
        <div class="metric-label">DEFENSIVE (-1% to -0.5%)</div>
        <div style="color:#ccc; font-size:0.75rem;">Positions halved. Need 7+ agents.</div>
    </div>
    <div class="glass-card" style="border-left: 3px solid #ef4444;">
        <div class="metric-label">HALT (below -1%)</div>
        <div style="color:#ccc; font-size:0.75rem;">ABSOLUTE VETO. Only exits allowed.</div>
    </div>
    <div class="glass-card" style="border-left: 3px solid #22c55e;">
        <div class="metric-label">TARGET (+1%)</div>
        <div style="color:#ccc; font-size:0.75rem;">Daily target achieved. Minimal risk.</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══ PIPELINE FLOW ═══
st.markdown('<div class="section-title">4-STEP AGENT PIPELINE</div>', unsafe_allow_html=True)

st.markdown("""
<div style="display:flex; gap:15px; align-items:center; justify-content:center; margin:20px 0;">
    <div class="glass-card" style="text-align:center; flex:1; border-top:3px solid #ffa500;">
        <div style="font-size:1.5rem;">🔍</div>
        <div style="font-family:Inter; font-size:0.7rem; color:#ffa500; margin-top:5px;">STEP 1</div>
        <div style="color:#ddd; font-size:0.8rem; margin-top:3px;">Data Validator</div>
        <div style="color:#666; font-size:0.65rem;">Sanitize + Quality Gate</div>
    </div>
    <div style="color:#555; font-size:1.5rem;">→</div>
    <div class="glass-card" style="text-align:center; flex:1; border-top:3px solid #3b82f6;">
        <div style="font-size:1.5rem;">🤖</div>
        <div style="font-family:Inter; font-size:0.7rem; color:#3b82f6; margin-top:5px;">STEP 2</div>
        <div style="color:#ddd; font-size:0.8rem; margin-top:3px;">10 Agents</div>
        <div style="color:#666; font-size:0.65rem;">Parallel Analysis</div>
    </div>
    <div style="color:#555; font-size:1.5rem;">→</div>
    <div class="glass-card" style="text-align:center; flex:1; border-top:3px solid #22c55e;">
        <div style="font-size:1.5rem;">⚖️</div>
        <div style="font-family:Inter; font-size:0.7rem; color:#22c55e; margin-top:5px;">STEP 3</div>
        <div style="color:#ddd; font-size:0.8rem; margin-top:3px;">Bayesian Combiner</div>
        <div style="color:#666; font-size:0.65rem;">Weighted Consensus</div>
    </div>
    <div style="color:#555; font-size:1.5rem;">→</div>
    <div class="glass-card" style="text-align:center; flex:1; border-top:3px solid #ef4444;">
        <div style="font-size:1.5rem;">✅</div>
        <div style="font-family:Inter; font-size:0.7rem; color:#ef4444; margin-top:5px;">STEP 4</div>
        <div style="color:#ddd; font-size:0.8rem; margin-top:3px;">Decision Auditor</div>
        <div style="color:#666; font-size:0.65rem;">Cross-check + Approve</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div style="text-align:center; color:#333; font-size:0.65rem; margin-top:40px; padding:20px 0; border-top:1px solid rgba(255,255,255,0.03);">
    Agent Intelligence Overlay v1.0 | 12 Specialized Agents | Bayesian Weight Updates | Last cycle: {last_cycle}
</div>
""", unsafe_allow_html=True)

time.sleep(5)
st.rerun()
