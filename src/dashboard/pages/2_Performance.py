"""
Page 2: Performance Metrics
Migrated from src/metrics/performance_dashboard.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import os
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any
import plotly.graph_objects as go

from src.dashboard.theme import MARKETEDGE_CSS, metric_card, plotly_layout

st.markdown(MARKETEDGE_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=5)
def load_config() -> dict:
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f) or {}
    return {}


@st.cache_data(ttl=3)
def load_state() -> dict:
    try:
        if os.path.exists("logs/dashboard_state.json"):
            with open("logs/dashboard_state.json", "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


@st.cache_data(ttl=10)
def load_journal() -> list:
    """Load trades from encrypted journal (authoritative source of truth)."""
    try:
        os.environ.setdefault('JOURNAL_ENCRYPTION_KEY',
                              'e2717e63c5babe3202ba02c93d900edb4d954b01be59462cc4734cc88f6ea1fe')
        from src.monitoring.journal import TradingJournal
        j = TradingJournal()
        return j.trades or []
    except Exception:
        pass
    # Fallback: plain JSON journal
    try:
        if os.path.exists("logs/trading_journal.json"):
            with open("logs/trading_journal.json", "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []


@st.cache_data(ttl=30)
def load_feature_importance() -> pd.DataFrame:
    try:
        if os.path.exists("models/lgbm_feature_importance.csv"):
            return pd.read_csv("models/lgbm_feature_importance.csv")
    except Exception:
        pass
    return pd.DataFrame(columns=["feature", "importance"])


@st.cache_data(ttl=10)
def load_retrain_history() -> list:
    try:
        if os.path.exists("models/retrain_history.json"):
            with open("models/retrain_history.json", "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []


@st.cache_data(ttl=5)
def load_training_state() -> dict:
    try:
        if os.path.exists("logs/training_state.json"):
            with open("logs/training_state.json", "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


@st.cache_data(ttl=30)
def load_model_inventory() -> list:
    model_dir = "models"
    inventory = []
    if not os.path.exists(model_dir):
        return inventory
    for fname in sorted(os.listdir(model_dir)):
        if fname.startswith("lgbm_") and fname.endswith(".txt"):
            fpath = os.path.join(model_dir, fname)
            stat = os.stat(fpath)
            parts = fname.replace("lgbm_", "").replace(".txt", "").replace("_optimized", " (opt)")
            inventory.append({
                'filename': fname, 'asset': parts.upper(),
                'size_kb': round(stat.st_size / 1024, 1),
                'modified': datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                'path': fpath,
            })
    return inventory


# ═══════════════════════════════════════════════════════════════════════
# COMPUTATION HELPERS
# ═══════════════════════════════════════════════════════════════════════

def compute_daily_pnl_series(equity_curve: list) -> Dict[str, float]:
    if not equity_curve:
        return {}
    daily = {}
    for entry in equity_curve:
        ts = entry.get("t", "")
        val = float(entry.get("v", 0.0))
        if not ts:
            continue
        date_str = ts[:10]
        if date_str not in daily:
            daily[date_str] = {"first": val, "last": val}
        daily[date_str]["last"] = val
    dates = sorted(daily.keys())
    result = {}
    for i, d in enumerate(dates):
        if i == 0:
            result[d] = daily[d]["last"] - daily[d]["first"]
        else:
            result[d] = daily[d]["last"] - daily[dates[i - 1]]["last"]
    return result


def compute_rolling_accuracy(predictions: list, actuals: list, window: int = 50) -> List[float]:
    if len(predictions) < window or len(actuals) < window:
        return []
    n = min(len(predictions), len(actuals))
    rolling = []
    for i in range(window, n + 1):
        p_win = predictions[i - window:i]
        a_win = actuals[i - window:i]
        correct = sum(1 for p, a in zip(p_win, a_win) if (p > 0 and a > 0) or (p < 0 and a < 0) or (p == 0 and a == 0))
        rolling.append(correct / window)
    return rolling


def compute_confusion_matrix(predictions: list, actuals: list) -> np.ndarray:
    tp = fp = fn = tn = 0
    for p, a in zip(predictions, actuals):
        if p > 0 and a > 0: tp += 1
        elif p > 0 and a <= 0: fp += 1
        elif p <= 0 and a > 0: fn += 1
        else: tn += 1
    return np.array([[tp, fp], [fn, tn]])


def compute_rolling_sharpe(equity_values: list, window: int = 50) -> List[float]:
    if len(equity_values) < window + 1:
        return []
    returns = np.diff(equity_values) / (np.array(equity_values[:-1]) + 1e-10)
    rolling = []
    for i in range(window, len(returns) + 1):
        r = returns[i - window:i]
        mu = np.mean(r)
        sigma = np.std(r)
        sharpe = (mu / sigma) * np.sqrt(252) if sigma > 0 else 0.0
        rolling.append(float(sharpe))
    return rolling


def compute_drawdown(equity_values: list) -> List[float]:
    if not equity_values:
        return []
    peak = equity_values[0]
    dd = []
    for v in equity_values:
        peak = max(peak, v)
        dd.append((v - peak) / peak * 100 if peak > 0 else 0.0)
    return dd


# ═══════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <div>
        <div class="title">PERFORMANCE METRICS</div>
        <div class="subtext">1% DAILY TARGET TRACKER | MODEL ACCURACY | SYSTEM HEALTH</div>
    </div>
    <div style="text-align:right;">
        <div class="subtext">AUTO-REFRESH: 10s</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Load data
config = load_config()
state = load_state()
journal = load_journal()
fi_df = load_feature_importance()


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: DAILY 1% TARGET PROGRESS
# ═══════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">DAILY 1% TARGET PROGRESS</div>', unsafe_allow_html=True)
initial_capital = config.get("initial_capital", 100000.0)
daily_target = initial_capital * 0.01
equity_curve = state.get("portfolio", {}).get("equity_curve", [])
daily_pnls = compute_daily_pnl_series(equity_curve)

today_str = datetime.now().strftime("%Y-%m-%d")
_equity_today_pnl = daily_pnls.get(today_str, 0.0)

# If equity curve has ≤1 entry today (no intraday change measurable),
# fall back to realized P&L from journal for today's gauge
_journal_closed = [t for t in journal if isinstance(t, dict) and (
    t.get('status') == 'CLOSED' or
    ('exit_price' in t and t.get('exit_price') and t.get('status') != 'OPEN')
)]
_journal_today_realized = sum(
    t.get('pnl', 0) for t in _journal_closed
    if str(t.get('exit_time') or t.get('timestamp', ''))[:10] == today_str
)
_eq_today_entries = [e for e in equity_curve if str(e.get('t', ''))[:10] == today_str]
# Use equity-based P&L only if we have 2+ data points (meaningful change); else use journal realized
today_pnl = _equity_today_pnl if len(_eq_today_entries) >= 2 else _journal_today_realized

col_gauge, col_cards = st.columns([1, 1])
with col_gauge:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=today_pnl,
        number={"prefix": "$", "font": {"size": 36, "color": "#fff"}},
        delta={"reference": daily_target, "relative": False, "prefix": "$",
               "increasing": {"color": "#22c55e"}, "decreasing": {"color": "#ef4444"}},
        title={"text": f"Today's P&L vs ${daily_target:,.0f} Target", "font": {"size": 14, "color": "#8b8ba7"}},
        gauge={
            "axis": {"range": [min(-daily_target, today_pnl * 1.2 if today_pnl < 0 else 0),
                               max(daily_target * 2, today_pnl * 1.2)],
                     "tickcolor": "#555", "tickfont": {"color": "#888"}},
            "bar": {"color": "#22c55e" if today_pnl >= daily_target else "#ffa500" if today_pnl > 0 else "#ef4444"},
            "bgcolor": "rgba(20,20,40,0.3)", "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, daily_target * 0.5], "color": "rgba(255,77,109,0.1)"},
                {"range": [daily_target * 0.5, daily_target], "color": "rgba(255,165,0,0.1)"},
                {"range": [daily_target, daily_target * 2], "color": "rgba(0,255,157,0.1)"},
            ],
            "threshold": {"line": {"color": "#3b82f6", "width": 3}, "thickness": 0.8, "value": daily_target},
        },
    ))
    gauge.update_layout(**plotly_layout(height=280))
    st.plotly_chart(gauge, width="stretch")

with col_cards:
    sorted_dates = sorted(daily_pnls.keys())
    last_30 = sorted_dates[-30:] if len(sorted_dates) >= 30 else sorted_dates
    streak = 0
    for d in reversed(sorted_dates):
        if daily_pnls[d] >= daily_target:
            streak += 1
        else:
            break
    days_hit = sum(1 for d in last_30 if daily_pnls.get(d, 0) >= daily_target)
    hit_rate = (days_hit / len(last_30) * 100) if last_30 else 0
    avg_daily = np.mean([daily_pnls[d] for d in last_30]) if last_30 else 0
    projected_monthly = avg_daily * 30

    c1, c2 = st.columns(2)
    with c1:
        streak_class = "streak-hot" if streak >= 3 else "streak-cold"
        st.markdown(f"""<div class="glass-card"><div class="metric-label">CONSECUTIVE DAYS AT TARGET</div><div class="metric-value">{streak} <span class="streak-badge {streak_class}">{'STREAK' if streak >= 3 else 'BUILDING'}</span></div></div>""", unsafe_allow_html=True)
    with c2:
        color = "metric-value-green" if hit_rate >= 50 else "metric-value-red"
        st.markdown(f"""<div class="glass-card"><div class="metric-label">TARGET HIT RATE (30D)</div><div class="metric-value {color}">{hit_rate:.0f}%</div></div>""", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        color = "metric-value-green" if avg_daily > 0 else "metric-value-red"
        st.markdown(f"""<div class="glass-card"><div class="metric-label">AVG DAILY P&L</div><div class="metric-value {color}">{"+" if avg_daily >= 0 else ""}${avg_daily:,.2f}</div></div>""", unsafe_allow_html=True)
    with c4:
        color = "metric-value-green" if projected_monthly > 0 else "metric-value-red"
        st.markdown(f"""<div class="glass-card"><div class="metric-label">PROJECTED MONTHLY</div><div class="metric-value {color}">${projected_monthly:,.0f}</div></div>""", unsafe_allow_html=True)

# 30-Day Chart
if last_30:
    pnl_vals = [daily_pnls.get(d, 0) for d in last_30]
    colors = ["#22c55e" if v >= daily_target else "#ffa500" if v > 0 else "#ef4444" for v in pnl_vals]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=last_30, y=pnl_vals, marker_color=colors, name="Daily P&L"))
    fig.add_hline(y=daily_target, line_dash="dash", line_color="#3b82f6", line_width=2,
                  annotation_text=f"1% Target (${daily_target:,.0f})", annotation_font_color="#3b82f6")
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
    fig.update_layout(**plotly_layout(title="30-Day P&L vs 1% Target", height=300))
    st.plotly_chart(fig, width="stretch")
else:
    st.info("Awaiting equity curve data to display daily P&L history.")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: MODEL ACCURACY
# ═══════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">MODEL ACCURACY VS TOP PUBLIC MODELS</div>', unsafe_allow_html=True)

benchmark = state.get("benchmark", {})
per_model = benchmark.get("per_model", {})

try:
    from src.models.benchmark import GLOBAL_LEADERBOARD, INTERNAL_MODELS
except Exception:
    GLOBAL_LEADERBOARD = {}
    INTERNAL_MODELS = {}

model_configs = [
    ("lightgbm", "LightGBM Classifier", "lgbm_direction_accuracy", "#3b82f6"),
    ("patchtst", "PatchTST Forecaster", "ptst_forecast_accuracy", "#22c55e"),
    ("rl_agent", "RL Policy Agent", "rl_action_accuracy", "#ffa500"),
    ("strategist", "Strategist LLM (L6)", "strategist_regime_accuracy", "#ef4444"),
]

for row_start in range(0, 4, 2):
    cols = st.columns(2)
    for idx, col in enumerate(cols):
        model_idx = row_start + idx
        if model_idx >= len(model_configs):
            break
        key, name, metric_key, color = model_configs[model_idx]
        data = per_model.get(key, {})
        correct = data.get("correct", 0)
        total = data.get("total", 0)
        accuracy = correct / total if total > 0 else 0.0
        preds = data.get("predictions", [])
        acts = data.get("actuals", [])

        with col:
            with st.expander(f"{name} | {accuracy:.1%} ({total} preds)", expanded=(total > 0)):
                acc_color = "#22c55e" if accuracy > 0.55 else "#ffa500" if accuracy > 0.45 else "#ef4444"
                st.markdown(f"""<div class="model-card">
                    <div class="model-name">{name}</div>
                    <div class="model-accuracy" style="color: {acc_color}">{accuracy:.1%}</div>
                    <div class="metric-label">{total} total | {correct} correct</div>
                </div>""", unsafe_allow_html=True)

                # Leaderboard comparison
                lb_data = GLOBAL_LEADERBOARD.get(metric_key, {})
                if lb_data:
                    models_dict = lb_data.get("models", {})
                    names_list = list(models_dict.keys()) + [f"Ours: {name}"]
                    values_list = list(models_dict.values()) + [accuracy]
                    colors_list = ["#555"] * len(models_dict) + [color]
                    combined = sorted(zip(values_list, names_list, colors_list), reverse=True)
                    fig_lb = go.Figure(go.Bar(
                        x=[c[0] for c in combined], y=[c[1] for c in combined], orientation="h",
                        marker_color=[c[2] for c in combined],
                        text=[f"{c[0]:.1%}" for c in combined], textposition="auto", textfont=dict(color="#fff"),
                    ))
                    our_rank = next((i + 1 for i, c in enumerate(combined) if c[1].startswith("Ours")), len(combined))
                    fig_lb.update_layout(**plotly_layout(
                        title=f"Rank #{our_rank}/{len(combined)}", height=max(200, len(combined) * 30),
                        margin=dict(l=180, r=20, t=40, b=20), xaxis=dict(tickformat=".0%", range=[0, 1]),
                    ))
                    st.plotly_chart(fig_lb, width="stretch")

                if len(preds) >= 50:
                    rolling = compute_rolling_accuracy(preds, acts, 50)
                    if rolling:
                        fig_roll = go.Figure()
                        fig_roll.add_trace(go.Scatter(y=rolling, mode="lines", line=dict(color=color, width=2), fill="tozeroy"))
                        fig_roll.add_hline(y=0.5, line_dash="dot", line_color="#555")
                        fig_roll.update_layout(**plotly_layout(title="Rolling 50-Prediction Accuracy", height=220, yaxis=dict(tickformat=".0%")))
                        st.plotly_chart(fig_roll, width="stretch")

                if total == 0:
                    st.info(f"No predictions yet for {name}.")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: ENSEMBLE PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">ENSEMBLE PERFORMANCE</div>', unsafe_allow_html=True)

ensemble_preds = benchmark.get("predictions", [])
ensemble_acts = benchmark.get("actuals", [])
performance = state.get("performance", {})
ensemble_total = min(len(ensemble_preds), len(ensemble_acts))
ensemble_correct = sum(1 for p, a in zip(ensemble_preds[:ensemble_total], ensemble_acts[:ensemble_total])
                       if (p > 0 and a > 0) or (p < 0 and a < 0) or (p == 0 and a == 0))
ensemble_acc = ensemble_correct / ensemble_total if ensemble_total > 0 else 0

# Compute trade stats from authoritative journal (already loaded above)
_closed_journal = [t for t in journal if isinstance(t, dict) and (
    t.get('status') == 'CLOSED' or
    ('exit_price' in t and t.get('exit_price') and t.get('status') != 'OPEN')
)]
_journal_win_rate = (sum(1 for t in _closed_journal if (t.get('pnl') or 0) > 0) / len(_closed_journal)) if _closed_journal else 0.0
_journal_total_pnl = sum(t.get('pnl', 0) for t in _closed_journal)

c1, c2, c3, c4 = st.columns(4)
with c1:
    color = "metric-value-green" if ensemble_acc > 0.55 else "metric-value-orange"
    st.markdown(f"""<div class="glass-card"><div class="metric-label">ENSEMBLE ACCURACY</div><div class="metric-value {color}">{ensemble_acc:.1%}</div></div>""", unsafe_allow_html=True)
with c2:
    win_rate = float(performance.get("win_rate") or _journal_win_rate)
    color = "metric-value-green" if win_rate > 0.55 else "metric-value-orange"
    st.markdown(f"""<div class="glass-card"><div class="metric-label">TRADE WIN RATE</div><div class="metric-value {color}">{win_rate:.1%} ({len(_closed_journal)} closed)</div></div>""", unsafe_allow_html=True)
with c3:
    sharpe = float(performance.get("sharpe_ratio", 0))
    color = "metric-value-green" if sharpe > 1 else "metric-value-orange" if sharpe > 0 else "metric-value-red"
    st.markdown(f"""<div class="glass-card"><div class="metric-label">SHARPE RATIO</div><div class="metric-value {color}">{sharpe:.2f}</div></div>""", unsafe_allow_html=True)
with c4:
    total_pnl = float(performance.get("total_pnl") or _journal_total_pnl)
    color = "metric-value-green" if total_pnl > 0 else "metric-value-red"
    st.markdown(f"""<div class="glass-card"><div class="metric-label">REALIZED P&L</div><div class="metric-value {color}">${total_pnl:,.2f}</div></div>""", unsafe_allow_html=True)

# Equity Curve
if equity_curve and len(equity_curve) > 10:
    eq_vals = [float(e.get("v", 0)) for e in equity_curve]
    eq_times = [e.get("t", "")[:16] for e in equity_curve]
    col_s, col_dd = st.columns(2)
    with col_s:
        rolling_s = compute_rolling_sharpe(eq_vals, window=min(50, len(eq_vals) // 2))
        if rolling_s:
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(y=rolling_s, mode="lines", line=dict(color="#3b82f6", width=2), fill="tozeroy", fillcolor="rgba(0,234,255,0.08)"))
            fig_s.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
            fig_s.add_hline(y=1, line_dash="dot", line_color="#22c55e", annotation_text="Good (1.0)")
            fig_s.update_layout(**plotly_layout(title="Rolling Sharpe Ratio", height=280))
            st.plotly_chart(fig_s, width="stretch")
    with col_dd:
        dd = compute_drawdown(eq_vals)
        if dd:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=eq_times[-len(dd):], y=dd, mode="lines", line=dict(color="#ef4444", width=1.5), fill="tozeroy", fillcolor="rgba(255,77,109,0.15)"))
            fig_dd.update_layout(**plotly_layout(title="Drawdown (%)", height=280, yaxis=dict(ticksuffix="%")))
            st.plotly_chart(fig_dd, width="stretch")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: TRAINING PROGRESS
# ═══════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">TRAINING PROGRESS</div>', unsafe_allow_html=True)

train_state = load_training_state()
retrain_hist = load_retrain_history()
model_inv = load_model_inventory()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    ct_status = train_state.get("status", state.get("training_status", "IDLE"))
    status_color = "metric-value-green" if ct_status in ("IDLE", "COMPLETED") else "metric-value-orange"
    st.markdown(f"""<div class="glass-card"><div class="metric-label">TRAINING STATUS</div><div class="metric-value {status_color}">{ct_status}</div></div>""", unsafe_allow_html=True)
with c2:
    total_models = train_state.get("total_models_trained", len(model_inv))
    st.markdown(f"""<div class="glass-card"><div class="metric-label">MODELS TRAINED</div><div class="metric-value metric-value-cyan">{total_models}</div></div>""", unsafe_allow_html=True)
with c3:
    best_acc = max((r.get("new_accuracy", 0) for r in retrain_hist), default=0) if retrain_hist else 0
    color = "metric-value-green" if best_acc > 0.55 else "metric-value-orange"
    st.markdown(f"""<div class="glass-card"><div class="metric-label">BEST ACCURACY</div><div class="metric-value {color}">{best_acc:.1%}</div></div>""", unsafe_allow_html=True)
with c4:
    cur_sym = train_state.get("current_symbol", "--")
    st.markdown(f"""<div class="glass-card"><div class="metric-label">CURRENT TARGET</div><div class="metric-value" style="font-size:1.1rem; color:#ffa500;">{cur_sym}</div></div>""", unsafe_allow_html=True)
with c5:
    version = state.get("model_version", "v6.0")
    st.markdown(f"""<div class="glass-card"><div class="metric-label">MODEL VERSION</div><div class="metric-value metric-value-cyan">{version}</div></div>""", unsafe_allow_html=True)

# Model Inventory + Feature Importance
col_inv, col_fi = st.columns([1.2, 1])
with col_inv:
    if model_inv:
        st.markdown("**Trained Model Inventory**")
        inv_data = []
        for m in model_inv:
            inv_data.append({'Model': m['asset'], 'Size (KB)': m['size_kb'], 'Last Trained': m['modified']})
        st.dataframe(pd.DataFrame(inv_data), width="stretch", hide_index=True, height=250)
    else:
        st.info("No trained models found.")

with col_fi:
    if not fi_df.empty and len(fi_df) > 0:
        top_10 = fi_df.nlargest(10, "importance")
        fig_fi = go.Figure(go.Bar(
            x=top_10["importance"].values[::-1], y=top_10["feature"].values[::-1], orientation="h",
            marker_color=["#3b82f6" if i < 3 else "#3498db" if i < 6 else "#555" for i in range(len(top_10))][::-1],
        ))
        fig_fi.update_layout(**plotly_layout(title="Feature Importance (Top 10)", height=280, margin=dict(l=140, r=20, t=40, b=40)))
        st.plotly_chart(fig_fi, width="stretch")
    else:
        st.info("Feature importance not available yet.")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: SYSTEM HEALTH
# ═══════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">SYSTEM HEALTH</div>', unsafe_allow_html=True)

sources = state.get("sources", {})
execution = state.get("execution", {})
risk = state.get("risk_metrics", {})
status = state.get("status", "UNKNOWN")

cols = st.columns(5)
source_list = [
    ("Exchange", sources.get("exchange", "UNKNOWN"), "📡"),
    ("News Feed", sources.get("news", "UNKNOWN"), "📰"),
    ("On-Chain", sources.get("onchain", "UNKNOWN"), "⛓"),
    ("LLM Engine", sources.get("llm", "UNKNOWN"), "🧠"),
]
for i, (name, s, icon) in enumerate(source_list):
    with cols[i]:
        dot_class = "dot-green" if s == "ONLINE" else "dot-red" if s == "OFFLINE" else "dot-yellow"
        st.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div style="font-size:1.5rem;">{icon}</div>
            <div class="metric-label">{name}</div>
            <div><span class="status-dot {dot_class}"></span>{s}</div>
        </div>""", unsafe_allow_html=True)

with cols[4]:
    sys_color = "dot-green" if status == "TRADING" else "dot-yellow"
    st.markdown(f"""<div class="glass-card" style="text-align:center;">
        <div style="font-size:1.5rem;">⚙</div>
        <div class="metric-label">SYSTEM</div>
        <div><span class="status-dot {sys_color}"></span>{status}</div>
    </div>""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    latency = float(execution.get("latency_ms", 0))
    lat_color = "metric-value-green" if latency < 100 else "metric-value-orange" if latency < 500 else "metric-value-red"
    st.markdown(f"""<div class="glass-card"><div class="metric-label">API LATENCY</div><div class="metric-value {lat_color}">{latency:.0f}ms</div></div>""", unsafe_allow_html=True)
with c2:
    slippage = float(execution.get("slippage", 0))
    st.markdown(f"""<div class="glass-card"><div class="metric-label">AVG SLIPPAGE</div><div class="metric-value metric-value-green">{slippage:.3f}%</div></div>""", unsafe_allow_html=True)
with c3:
    fill = float(execution.get("fill_rate", 100))
    st.markdown(f"""<div class="glass-card"><div class="metric-label">FILL RATE</div><div class="metric-value metric-value-green">{fill:.1f}%</div></div>""", unsafe_allow_html=True)
with c4:
    max_dd = float(risk.get("max_drawdown", 0))
    dd_color = "metric-value-green" if abs(max_dd) < 5 else "metric-value-orange" if abs(max_dd) < 10 else "metric-value-red"
    st.markdown(f"""<div class="glass-card"><div class="metric-label">MAX DRAWDOWN</div><div class="metric-value {dd_color}">{max_dd:.1f}%</div></div>""", unsafe_allow_html=True)

# Footer
st.markdown(f'<div style="text-align:center; color:#555; font-size:0.7rem; margin-top:10px;">Last Update: {state.get("last_update","")}</div>', unsafe_allow_html=True)

time.sleep(10)
st.rerun()
