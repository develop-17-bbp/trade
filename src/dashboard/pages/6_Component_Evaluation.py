"""
Page 6: Component Evaluation & Control

Single-pane answer to "what's helping vs hurting — and what do I turn off?"
Powered by src.evaluation.act_evaluator which reads:
  * logs/robinhood_paper.jsonl (ENTRY + EXIT events)
  * logs/meta_shadow.jsonl (shadow predict + outcome)
  * logs/safe_entries_state.json (rolling Sharpe, consec losses)
  * models/retrain_history.json (per-asset retrain log)
  * env flags (ACT_*)

Renders:
  1. Component status grid with ON/OFF state + exact toggle commands
  2. Paper-journal totals
  3. Bucket attribution tables (score, LLM conf, ML conf, spread, direction, asset, exit reason)
  4. Rolling 30-trade Sharpe line chart
  5. Shadow-log stats (if populated)
  6. Recommendation panel

Auto-refreshes every 10 seconds.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.evaluation.act_evaluator import build_report
from src.dashboard.theme import MARKETEDGE_CSS, GREEN, RED, AMBER, CYAN, MUTED, CARD_BG, CARD_BORDER, plotly_layout


st.markdown(MARKETEDGE_CSS, unsafe_allow_html=True)


# ═══ HEADER ═══
st.markdown(
    """
<div class="main-header">
    <div>
        <div class="title">COMPONENT EVALUATION</div>
        <div class="subtext">ATTRIBUTION | CONTROL | RECOMMENDATIONS</div>
    </div>
    <div style="text-align:right;">
        <div class="subtext">AUTO-REFRESH: 10s</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

report = build_report()


# ═══ TOTALS STRIP ═══
t = report["totals"]
c1, c2, c3, c4, c5 = st.columns(5)
if t.get("n", 0) > 0:
    wr = t["wr"]
    wr_color = GREEN if wr >= 0.55 else (AMBER if wr >= 0.45 else RED)
    pnl_color = GREEN if t["total_pnl_pct"] >= 0 else RED
    with c1:
        st.markdown(
            f"""<div class="glass-card" style="text-align:center;">
                <div class="metric-label">TRADES</div>
                <div class="metric-value">{t['n']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="glass-card" style="text-align:center;">
                <div class="metric-label">WIN RATE</div>
                <div class="metric-value" style="color:{wr_color};">{wr:.1%}</div>
                <div class="subtext">{t['wins']}W / {t['losses']}L</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class="glass-card" style="text-align:center;">
                <div class="metric-label">TOTAL PnL</div>
                <div class="metric-value" style="color:{pnl_color};">{t['total_pnl_pct']:+.2f}%</div>
                <div class="subtext">${t['total_pnl_usd']:+.2f}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""<div class="glass-card" style="text-align:center;">
                <div class="metric-label">MEAN / TRADE</div>
                <div class="metric-value">{t['mean_pnl_pct']:+.3f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c5:
        rs = report.get("rolling_sharpe_30") or []
        sharpe = rs[-1]["sharpe"] if rs else 0.0
        sh_color = GREEN if sharpe >= 1.0 else (AMBER if sharpe >= 0.3 else RED)
        st.markdown(
            f"""<div class="glass-card" style="text-align:center;">
                <div class="metric-label">ROLLING SHARPE (30)</div>
                <div class="metric-value" style="color:{sh_color};">{sharpe:+.3f}</div>
            </div>""",
            unsafe_allow_html=True,
        )
else:
    st.info("No completed paper trades yet. Start the bot to begin collecting.")


# ═══ COMPONENT STATUS ═══
st.markdown('<div class="section-title">COMPONENT STATUS</div>', unsafe_allow_html=True)
comp_df = pd.DataFrame(report["components"]["components"])
if not comp_df.empty:
    comp_df["state"] = comp_df["is_on"].map({True: "ON", False: "OFF"})
    comp_df["toggle_cmd"] = comp_df.apply(
        lambda r: r["toggle_cmd_off"] if r["is_on"] else r["toggle_cmd_on"], axis=1
    )
    display = comp_df[["name", "env", "value", "state", "toggle_cmd"]]
    display.columns = ["Component", "Env var", "Current value", "State", "Toggle cmd"]
    st.dataframe(
        display, use_container_width=True, hide_index=True,
        column_config={
            "State": st.column_config.TextColumn("State", width="small"),
            "Toggle cmd": st.column_config.TextColumn("Toggle cmd (paste into cmd)", width="medium"),
        },
    )


# ═══ ATTRIBUTION TABLES ═══
st.markdown('<div class="section-title">ATTRIBUTION: WHICH BUCKETS ARE WINNING / LOSING</div>', unsafe_allow_html=True)


def _render_attr(rows, title):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if df.empty or (df["n"] == 0).all():
        return
    df = df[df["n"] > 0].copy()
    df["wr"] = df["wr"].apply(lambda v: f"{v:.1%}" if v is not None else "-")
    df["mean_pnl_pct"] = df["mean_pnl_pct"].apply(
        lambda v: f"{v:+.3f}%" if v is not None else "-"
    )
    df["total_pnl_pct"] = df["total_pnl_pct"].apply(lambda v: f"{v:+.2f}%")
    df = df[["bucket", "n", "wr", "mean_pnl_pct", "total_pnl_pct"]]
    df.columns = [title, "N", "WR", "Mean %", "Total %"]
    st.dataframe(df, use_container_width=True, hide_index=True)


left, right = st.columns(2)
with left:
    _render_attr(report["attribution"]["by_score"], "By entry score")
    _render_attr(report["attribution"]["by_llm_conf"], "By LLM confidence")
    _render_attr(report["attribution"]["by_direction"], "By direction")
    _render_attr(report["attribution"]["by_asset"], "By asset")
with right:
    _render_attr(report["attribution"]["by_ml_conf"], "By ML confidence")
    _render_attr(report["attribution"]["by_spread"], "By spread %")
    _render_attr(report["attribution"]["by_exit_reason"], "By exit reason")
    _render_attr(report["attribution"]["by_size"], "By size %")


# ═══ ROLLING SHARPE CHART ═══
st.markdown('<div class="section-title">ROLLING 30-TRADE SHARPE</div>', unsafe_allow_html=True)
rs = report.get("rolling_sharpe_30") or []
if rs:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[r["idx"] for r in rs],
            y=[r["sharpe"] for r in rs],
            mode="lines+markers",
            line=dict(color=CYAN, width=2),
            marker=dict(size=4),
            name="Sharpe(30)",
        )
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color=GREEN, annotation_text="promotion floor (1.0)")
    fig.add_hline(y=0.3, line_dash="dot", line_color=AMBER, annotation_text="warning (0.3)")
    fig.add_hline(y=0.0, line_dash="dot", line_color=MUTED)
    fig.update_layout(**plotly_layout(height=300, title=""))
    fig.update_xaxes(title="Trade #")
    fig.update_yaxes(title="Sharpe")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info(f"Need >= 30 trades for rolling Sharpe. Have {t.get('n', 0)}.")


# ═══ SHADOW LOG ═══
st.markdown('<div class="section-title">META SHADOW LOG</div>', unsafe_allow_html=True)
s = report.get("shadow", {})
if not s.get("available"):
    st.info("Shadow logging helper unavailable.")
elif s.get("joined_trades", 0) == 0:
    st.info(
        f"Shadow log has {s.get('total_records', 0)} predictions but 0 joined trades yet. "
        "Enable ACT_META_SHADOW_MODE=1 + ACT_DISABLE_ML=0 and let the bot run."
    )
else:
    combined = s.get("combined", {}) or {}
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("Joined trades", s.get("joined_trades", 0))
    with sc2:
        st.metric(
            "Meta WOULD VETO",
            f"{combined.get('meta_veto_count', 0)}",
            help="Count of predictions where meta model would have vetoed",
        )
    with sc3:
        st.metric(
            "Veto precision (loss)",
            f"{combined.get('veto_precision_loss', 0):.1%}",
            help="Of the trades meta would have vetoed, how many were actually losers",
        )
    with sc4:
        tot = combined.get("total_pnl_pct", 0.0)
        iv = combined.get("if_vetoed_pnl_pct", 0.0)
        delta = iv - tot
        st.metric(
            "If-vetoed PnL uplift",
            f"{delta:+.2f}pp",
            help="How much better (or worse) total PnL would be if meta vetoes had been applied",
        )


# ═══ RECOMMENDATIONS ═══
st.markdown('<div class="section-title">RECOMMENDATIONS</div>', unsafe_allow_html=True)
for r in report["recommendations"]:
    sev = r["severity"]
    color_map = {"high": RED, "medium": AMBER, "info": MUTED}
    color = color_map.get(sev, MUTED)
    st.markdown(
        f"""<div class="glass-card" style="border-left:4px solid {color}; margin-bottom:8px; padding:12px 16px;">
            <div style="color:{color}; font-weight:600; text-transform:uppercase; font-size:0.75rem;">
                [{sev}] {r['area']}
            </div>
            <div style="color:#e2e8f0; margin-top:4px;">{r['reason']}</div>
            <div class="subtext" style="margin-top:4px;">&rarr; {r['action']}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ═══ AUTO-REFRESH ═══
import time
time.sleep(10)
st.rerun()
