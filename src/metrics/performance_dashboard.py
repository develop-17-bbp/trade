"""
Performance Metrics Dashboard
==============================
Visual metrics for daily 1% target progress, individual model accuracy
vs top public models, ensemble performance, training progress, and system health.

Launch:
    streamlit run src/metrics/performance_dashboard.py --server.port 8502
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import os
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go

# ── Page Config ──
st.set_page_config(
    page_title="Performance Metrics - 1% Daily Target",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS (Matching MarketEdge Theme) ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
    radial-gradient(circle at 10% 20%, #0b0b1d 0%, transparent 40%),
    radial-gradient(circle at 80% 80%, #0d2436 0%, transparent 40%),
    linear-gradient(135deg, #04040c, #0b0b20) !important;
}

.main-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 14px 0; margin-bottom: 20px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.title {
    font-family: 'Orbitron'; font-size: 1.9rem; font-weight: 700;
    background: linear-gradient(90deg, #00eaff, #00ff9d);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.subtext { font-size: 0.65rem; color: #6e6e8f; letter-spacing: 1px; }

.glass-card {
    background: rgba(20,20,40,0.55); border: 1px solid rgba(255,255,255,0.06);
    backdrop-filter: blur(18px); border-radius: 14px; padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 6px 30px rgba(0,0,0,0.4), inset 0 0 0 1px rgba(255,255,255,0.02);
}
.glass-card:hover { transform: translateY(-2px); border-color: rgba(0,234,255,0.25); }

.metric-label { font-size: 0.72rem; color: #8b8ba7; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { font-size: 1.7rem; font-weight: 600; margin-top: 4px; color: #fff; }
.metric-value-green { color: #00ff9d; }
.metric-value-red { color: #ff4d6d; }
.metric-value-cyan { color: #00eaff; }
.metric-value-orange { color: #ffa500; }

.section-title {
    font-family: 'Orbitron'; font-size: .8rem; letter-spacing: 1.4px;
    color: #00eaff; margin: 30px 0 15px 0; text-transform: uppercase;
    display: flex; align-items: center;
}
.section-title:after {
    content: ""; flex: 1; height: 1px; margin-left: 10px;
    background: linear-gradient(90deg, #00eaff, transparent);
}

.model-card {
    background: rgba(15,15,30,0.7); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 16px; margin-bottom: 12px;
}
.model-name { font-family: 'Orbitron'; font-size: 0.75rem; color: #00eaff; letter-spacing: 1px; }
.model-accuracy { font-size: 2rem; font-weight: 700; margin: 6px 0; }
.model-trend-up { color: #00ff9d; font-size: 0.8rem; }
.model-trend-down { color: #ff4d6d; font-size: 0.8rem; }
.model-trend-flat { color: #8b8ba7; font-size: 0.8rem; }

.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.dot-green { background: #00ff9d; box-shadow: 0 0 6px #00ff9d; }
.dot-red { background: #ff4d6d; box-shadow: 0 0 6px #ff4d6d; }
.dot-yellow { background: #ffa500; box-shadow: 0 0 6px #ffa500; }

.stExpander {
    background: rgba(20,20,40,0.4) !important; border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important; margin-bottom: 10px !important;
}
.stExpander > div:first-child { font-family: 'Orbitron' !important; color: #00eaff !important; font-size: 0.85rem !important; }

.streak-badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.5px;
}
.streak-hot { background: rgba(0,255,157,0.15); color: #00ff9d; border: 1px solid #00ff9d44; }
.streak-cold { background: rgba(255,77,77,0.15); color: #ff4d6d; border: 1px solid #ff4d4d44; }
</style>
""", unsafe_allow_html=True)


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
    state_file = "logs/dashboard_state.json"
    try:
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


@st.cache_data(ttl=10)
def load_journal() -> list:
    journal_file = "logs/trading_journal.json"
    try:
        if os.path.exists(journal_file):
            with open(journal_file, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []


@st.cache_data(ttl=30)
def load_feature_importance() -> pd.DataFrame:
    csv_path = "models/lgbm_feature_importance.csv"
    try:
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
    except Exception:
        pass
    return pd.DataFrame(columns=["feature", "importance"])


@st.cache_data(ttl=30)
def load_benchmark_history() -> list:
    hist_file = "logs/benchmark_history.json"
    try:
        if os.path.exists(hist_file):
            with open(hist_file, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def get_leaderboard() -> dict:
    try:
        from src.models.benchmark import GLOBAL_LEADERBOARD
        return GLOBAL_LEADERBOARD
    except Exception:
        return {}


def get_internal_models() -> dict:
    try:
        from src.models.benchmark import INTERNAL_MODELS
        return INTERNAL_MODELS
    except Exception:
        return {}


def _plotly_layout(**kwargs) -> dict:
    base = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ddd', family='Inter'),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)'),
    )
    base.update(kwargs)
    return base


# ═══════════════════════════════════════════════════════════════════════
# COMPUTATION HELPERS
# ═══════════════════════════════════════════════════════════════════════

def compute_daily_pnl_series(equity_curve: list) -> Dict[str, float]:
    """Group equity curve entries by date, compute daily P&L deltas."""
    if not equity_curve:
        return {}

    daily = {}
    for entry in equity_curve:
        ts = entry.get("t", "")
        val = float(entry.get("v", 0.0))
        if not ts:
            continue
        try:
            date_str = ts[:10]
        except Exception:
            continue
        if date_str not in daily:
            daily[date_str] = {"first": val, "last": val}
        daily[date_str]["last"] = val

    # Compute daily deltas
    dates = sorted(daily.keys())
    result = {}
    for i, d in enumerate(dates):
        if i == 0:
            result[d] = daily[d]["last"] - daily[d]["first"]
        else:
            prev_d = dates[i - 1]
            result[d] = daily[d]["last"] - daily[prev_d]["last"]
    return result


def compute_rolling_accuracy(predictions: list, actuals: list, window: int = 50) -> List[float]:
    """Compute rolling window directional accuracy."""
    if len(predictions) < window or len(actuals) < window:
        return []
    n = min(len(predictions), len(actuals))
    rolling = []
    for i in range(window, n + 1):
        p_win = predictions[i - window:i]
        a_win = actuals[i - window:i]
        correct = sum(1 for p, a in zip(p_win, a_win)
                      if (p > 0 and a > 0) or (p < 0 and a < 0) or (p == 0 and a == 0))
        rolling.append(correct / window)
    return rolling


def compute_confusion_matrix(predictions: list, actuals: list) -> np.ndarray:
    """2x2 confusion matrix: predicted positive/negative vs actual positive/negative."""
    tp = fp = fn = tn = 0
    for p, a in zip(predictions, actuals):
        if p > 0 and a > 0:
            tp += 1
        elif p > 0 and a <= 0:
            fp += 1
        elif p <= 0 and a > 0:
            fn += 1
        else:
            tn += 1
    return np.array([[tp, fp], [fn, tn]])


def compute_rolling_sharpe(equity_values: list, window: int = 50) -> List[float]:
    """Compute rolling Sharpe ratio from equity values."""
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
    """Compute drawdown series from equity values."""
    if not equity_values:
        return []
    peak = equity_values[0]
    dd = []
    for v in equity_values:
        peak = max(peak, v)
        dd.append((v - peak) / peak * 100 if peak > 0 else 0.0)
    return dd


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: DAILY 1% TARGET PROGRESS
# ═══════════════════════════════════════════════════════════════════════

def render_section_daily_target(state: dict, config: dict):
    st.markdown('<div class="section-title">DAILY 1% TARGET PROGRESS</div>', unsafe_allow_html=True)

    initial_capital = config.get("initial_capital", 100000.0)
    daily_target = initial_capital * 0.01

    equity_curve = state.get("portfolio", {}).get("equity_curve", [])
    daily_pnls = compute_daily_pnl_series(equity_curve)

    today_str = datetime.now().strftime("%Y-%m-%d")
    today_pnl = daily_pnls.get(today_str, 0.0)
    progress_pct = (today_pnl / daily_target * 100) if daily_target > 0 else 0.0

    # ── Row 1: Gauge + Summary Cards ──
    col_gauge, col_cards = st.columns([1, 1])

    with col_gauge:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=today_pnl,
            number={"prefix": "$", "font": {"size": 36, "color": "#fff"}},
            delta={"reference": daily_target, "relative": False, "prefix": "$",
                   "increasing": {"color": "#00ff9d"}, "decreasing": {"color": "#ff4d6d"}},
            title={"text": f"Today's P&L vs ${daily_target:,.0f} Target",
                   "font": {"size": 14, "color": "#8b8ba7"}},
            gauge={
                "axis": {"range": [min(-daily_target, today_pnl * 1.2 if today_pnl < 0 else 0),
                                   max(daily_target * 2, today_pnl * 1.2)],
                         "tickcolor": "#555", "tickfont": {"color": "#888"}},
                "bar": {"color": "#00ff9d" if today_pnl >= daily_target else "#ffa500" if today_pnl > 0 else "#ff4d6d"},
                "bgcolor": "rgba(20,20,40,0.3)",
                "bordercolor": "rgba(255,255,255,0.1)",
                "steps": [
                    {"range": [0, daily_target * 0.5], "color": "rgba(255,77,109,0.1)"},
                    {"range": [daily_target * 0.5, daily_target], "color": "rgba(255,165,0,0.1)"},
                    {"range": [daily_target, daily_target * 2], "color": "rgba(0,255,157,0.1)"},
                ],
                "threshold": {
                    "line": {"color": "#00eaff", "width": 3},
                    "thickness": 0.8,
                    "value": daily_target,
                },
            },
        ))
        gauge.update_layout(**_plotly_layout(height=280))
        st.plotly_chart(gauge, use_container_width=True)

    with col_cards:
        # Compute streak and stats
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

        # Cards in 2x2 grid
        c1, c2 = st.columns(2)
        with c1:
            streak_class = "streak-hot" if streak >= 3 else "streak-cold"
            st.markdown(f"""<div class="glass-card">
                <div class="metric-label">CONSECUTIVE DAYS AT TARGET</div>
                <div class="metric-value">{streak} <span class="streak-badge {streak_class}">{'STREAK' if streak >= 3 else 'BUILDING'}</span></div>
            </div>""", unsafe_allow_html=True)

        with c2:
            color = "metric-value-green" if hit_rate >= 50 else "metric-value-red"
            st.markdown(f"""<div class="glass-card">
                <div class="metric-label">TARGET HIT RATE (30D)</div>
                <div class="metric-value {color}">{hit_rate:.0f}%</div>
            </div>""", unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            sign = "+" if avg_daily >= 0 else ""
            color = "metric-value-green" if avg_daily > 0 else "metric-value-red"
            st.markdown(f"""<div class="glass-card">
                <div class="metric-label">AVG DAILY P&L</div>
                <div class="metric-value {color}">{sign}${avg_daily:,.2f}</div>
            </div>""", unsafe_allow_html=True)

        with c4:
            color = "metric-value-green" if projected_monthly > 0 else "metric-value-red"
            st.markdown(f"""<div class="glass-card">
                <div class="metric-label">PROJECTED MONTHLY</div>
                <div class="metric-value {color}">${projected_monthly:,.0f}</div>
            </div>""", unsafe_allow_html=True)

    # ── Row 2: 30-Day Achievement Chart ──
    if last_30:
        pnl_vals = [daily_pnls.get(d, 0) for d in last_30]
        colors = ["#00ff9d" if v >= daily_target else "#ffa500" if v > 0 else "#ff4d6d" for v in pnl_vals]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=last_30, y=pnl_vals,
            marker_color=colors, name="Daily P&L",
            hovertemplate="Date: %{x}<br>P&L: $%{y:,.2f}<extra></extra>"
        ))
        fig.add_hline(y=daily_target, line_dash="dash", line_color="#00eaff", line_width=2,
                      annotation_text=f"1% Target (${daily_target:,.0f})",
                      annotation_font_color="#00eaff", annotation_font_size=11)
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
        fig.update_layout(**_plotly_layout(
            title="30-Day P&L vs 1% Target", height=300,
            xaxis_title="Date", yaxis_title="P&L ($)",
        ))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Awaiting equity curve data to display daily P&L history.")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: INDIVIDUAL MODEL ACCURACY VS PUBLIC LEADERBOARDS
# ═══════════════════════════════════════════════════════════════════════

def render_section_model_accuracy(state: dict):
    st.markdown('<div class="section-title">MODEL ACCURACY VS TOP PUBLIC MODELS</div>', unsafe_allow_html=True)

    benchmark = state.get("benchmark", {})
    per_model = benchmark.get("per_model", {})
    leaderboard = get_leaderboard()
    internal = get_internal_models()

    model_configs = [
        ("lightgbm", "LightGBM Classifier", "lgbm_direction_accuracy", "#00eaff"),
        ("patchtst", "PatchTST Forecaster", "ptst_forecast_accuracy", "#00ff9d"),
        ("rl_agent", "RL Policy Agent", "rl_action_accuracy", "#ffa500"),
        ("strategist", "Strategist LLM (L6)", "strategist_regime_accuracy", "#ff4d6d"),
    ]

    # ── 2x2 Model Grid ──
    for row_start in range(0, 4, 2):
        cols = st.columns(2)
        for idx, col in enumerate(cols):
            model_idx = row_start + idx
            if model_idx >= len(model_configs):
                break

            key, name, metric_key, color = model_configs[model_idx]
            data = per_model.get(key, {})
            preds = data.get("predictions", [])
            acts = data.get("actuals", [])
            correct = data.get("correct", 0)
            total = data.get("total", 0)
            accuracy = correct / total if total > 0 else 0.0

            with col:
                with st.expander(f"{name} | {accuracy:.1%} Accuracy ({total} predictions)", expanded=(total > 0)):
                    # ── Accuracy Card with Trend ──
                    trend_html = ""
                    if len(preds) >= 100:
                        recent_50 = compute_rolling_accuracy(preds[-100:], acts[-100:], 50)
                        if len(recent_50) >= 2:
                            delta = recent_50[-1] - recent_50[0]
                            if delta > 0.01:
                                trend_html = f'<span class="model-trend-up">&#9650; +{delta:.1%}</span>'
                            elif delta < -0.01:
                                trend_html = f'<span class="model-trend-down">&#9660; {delta:.1%}</span>'
                            else:
                                trend_html = '<span class="model-trend-flat">&#9644; Stable</span>'

                    acc_color = "#00ff9d" if accuracy > 0.55 else "#ffa500" if accuracy > 0.45 else "#ff4d6d"
                    st.markdown(f"""<div class="model-card">
                        <div class="model-name">{name}</div>
                        <div class="model-accuracy" style="color: {acc_color}">{accuracy:.1%} {trend_html}</div>
                        <div class="metric-label">{total} total predictions | {correct} correct</div>
                    </div>""", unsafe_allow_html=True)

                    # ── Leaderboard Comparison ──
                    lb_data = leaderboard.get(metric_key, {})
                    if lb_data:
                        models_dict = lb_data.get("models", {})
                        names_list = list(models_dict.keys()) + [f"Ours: {name}"]
                        values_list = list(models_dict.values()) + [accuracy]
                        colors_list = ["#555"] * len(models_dict) + [color]

                        # Sort by value
                        combined = sorted(zip(values_list, names_list, colors_list), reverse=True)
                        s_vals = [c[0] for c in combined]
                        s_names = [c[1] for c in combined]
                        s_colors = [c[2] for c in combined]

                        fig_lb = go.Figure(go.Bar(
                            x=s_vals, y=s_names, orientation="h",
                            marker_color=s_colors,
                            text=[f"{v:.1%}" for v in s_vals],
                            textposition="auto",
                            textfont=dict(color="#fff", size=11),
                        ))
                        our_rank = next((i + 1 for i, n in enumerate(s_names) if n.startswith("Ours")), len(s_names))
                        fig_lb.update_layout(**_plotly_layout(
                            title=f"Leaderboard (Rank #{our_rank}/{len(s_names)})",
                            height=max(200, len(s_names) * 30),
                            margin=dict(l=180, r=20, t=40, b=20),
                            xaxis=dict(tickformat=".0%", range=[0, 1]),
                        ))
                        st.plotly_chart(fig_lb, use_container_width=True)

                    # ── Rolling Accuracy ──
                    if len(preds) >= 50:
                        rolling = compute_rolling_accuracy(preds, acts, 50)
                        if rolling:
                            fig_roll = go.Figure()
                            fig_roll.add_trace(go.Scatter(
                                y=rolling, mode="lines",
                                line=dict(color=color, width=2),
                                fill="tozeroy", fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}",
                                name="Rolling 50 Accuracy",
                            ))
                            fig_roll.add_hline(y=0.5, line_dash="dot", line_color="#555",
                                               annotation_text="Random (50%)", annotation_font_size=10)
                            fig_roll.update_layout(**_plotly_layout(
                                title="Rolling 50-Prediction Accuracy", height=220,
                                yaxis=dict(tickformat=".0%", range=[0.3, 0.8]),
                            ))
                            st.plotly_chart(fig_roll, use_container_width=True)

                    # ── Confusion Matrix ──
                    if len(preds) >= 10:
                        cm = compute_confusion_matrix(preds, acts)
                        fig_cm = go.Figure(go.Heatmap(
                            z=cm, x=["Actual UP", "Actual DOWN"],
                            y=["Predicted UP", "Predicted DOWN"],
                            text=cm, texttemplate="%{text}",
                            colorscale=[[0, "rgba(20,20,40,0.8)"], [1, color]],
                            showscale=False,
                            textfont=dict(size=16, color="#fff"),
                        ))
                        fig_cm.update_layout(**_plotly_layout(
                            title="Direction Confusion Matrix", height=220,
                            margin=dict(l=100, r=20, t=40, b=40),
                        ))
                        st.plotly_chart(fig_cm, use_container_width=True)

                    if total == 0:
                        st.info(f"No predictions recorded yet for {name}. Start the trading system to collect data.")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: ENSEMBLE PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════

def render_section_ensemble(state: dict):
    st.markdown('<div class="section-title">ENSEMBLE PERFORMANCE</div>', unsafe_allow_html=True)

    benchmark = state.get("benchmark", {})
    per_model = benchmark.get("per_model", {})
    ensemble_preds = benchmark.get("predictions", [])
    ensemble_acts = benchmark.get("actuals", [])
    equity_curve = state.get("portfolio", {}).get("equity_curve", [])
    performance = state.get("performance", {})

    # ── Row 1: Summary Cards ──
    ensemble_total = min(len(ensemble_preds), len(ensemble_acts))
    ensemble_correct = sum(1 for p, a in zip(ensemble_preds[:ensemble_total], ensemble_acts[:ensemble_total])
                          if (p > 0 and a > 0) or (p < 0 and a < 0) or (p == 0 and a == 0))
    ensemble_acc = ensemble_correct / ensemble_total if ensemble_total > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        color = "metric-value-green" if ensemble_acc > 0.55 else "metric-value-orange"
        st.markdown(f"""<div class="glass-card">
            <div class="metric-label">ENSEMBLE ACCURACY</div>
            <div class="metric-value {color}">{ensemble_acc:.1%}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        win_rate = float(performance.get("win_rate", 0))
        color = "metric-value-green" if win_rate > 0.55 else "metric-value-orange"
        st.markdown(f"""<div class="glass-card">
            <div class="metric-label">TRADE WIN RATE</div>
            <div class="metric-value {color}">{win_rate:.1%}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        sharpe = float(performance.get("sharpe_ratio", 0))
        color = "metric-value-green" if sharpe > 1 else "metric-value-orange" if sharpe > 0 else "metric-value-red"
        st.markdown(f"""<div class="glass-card">
            <div class="metric-label">SHARPE RATIO</div>
            <div class="metric-value {color}">{sharpe:.2f}</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        total_pnl = float(performance.get("total_pnl", state.get("portfolio", {}).get("pnl", 0)))
        color = "metric-value-green" if total_pnl > 0 else "metric-value-red"
        st.markdown(f"""<div class="glass-card">
            <div class="metric-label">TOTAL P&L</div>
            <div class="metric-value {color}">${total_pnl:,.2f}</div>
        </div>""", unsafe_allow_html=True)

    # ── Row 2: Ensemble vs Individual Models ──
    col_bar, col_agree = st.columns(2)

    with col_bar:
        model_names = ["Ensemble"]
        model_accs = [ensemble_acc]
        model_colors = ["#00eaff"]

        for key, display, _, color in [
            ("lightgbm", "LightGBM", "", "#3498db"),
            ("patchtst", "PatchTST", "", "#00ff9d"),
            ("rl_agent", "RL Agent", "", "#ffa500"),
            ("strategist", "Strategist", "", "#ff4d6d"),
        ]:
            d = per_model.get(key, {})
            t = d.get("total", 0)
            c = d.get("correct", 0)
            acc = c / t if t > 0 else 0
            model_names.append(display)
            model_accs.append(acc)
            model_colors.append(color)

        fig_compare = go.Figure(go.Bar(
            x=model_names, y=model_accs,
            marker_color=model_colors,
            text=[f"{a:.1%}" for a in model_accs],
            textposition="auto", textfont=dict(color="#fff"),
        ))
        fig_compare.add_hline(y=0.5, line_dash="dot", line_color="#555",
                              annotation_text="Random", annotation_font_size=10)
        fig_compare.update_layout(**_plotly_layout(
            title="Ensemble vs Individual Model Accuracy", height=320,
            yaxis=dict(tickformat=".0%", range=[0, 1]),
        ))
        st.plotly_chart(fig_compare, use_container_width=True)

    with col_agree:
        # Signal agreement analysis
        if ensemble_total >= 10:
            agree_data = {"2 agree": {"correct": 0, "total": 0},
                          "3 agree": {"correct": 0, "total": 0},
                          "4 agree": {"correct": 0, "total": 0}}

            models_preds = {k: per_model.get(k, {}).get("predictions", []) for k in ["lightgbm", "patchtst", "rl_agent", "strategist"]}
            min_len = min(len(v) for v in models_preds.values()) if models_preds else 0
            min_len = min(min_len, len(ensemble_acts))

            for i in range(min_len):
                directions = []
                for k in ["lightgbm", "patchtst", "rl_agent", "strategist"]:
                    p = models_preds[k][i] if i < len(models_preds[k]) else 0
                    directions.append(1 if p > 0 else -1 if p < 0 else 0)

                # Count most common direction
                from collections import Counter
                counts = Counter(directions)
                most_common_dir, most_common_count = counts.most_common(1)[0]
                actual = ensemble_acts[i]
                is_correct = (most_common_dir > 0 and actual > 0) or (most_common_dir < 0 and actual < 0)

                if most_common_count >= 4:
                    agree_data["4 agree"]["total"] += 1
                    if is_correct:
                        agree_data["4 agree"]["correct"] += 1
                elif most_common_count >= 3:
                    agree_data["3 agree"]["total"] += 1
                    if is_correct:
                        agree_data["3 agree"]["correct"] += 1
                else:
                    agree_data["2 agree"]["total"] += 1
                    if is_correct:
                        agree_data["2 agree"]["correct"] += 1

            labels = list(agree_data.keys())
            totals = [agree_data[l]["total"] for l in labels]
            accs = [agree_data[l]["correct"] / agree_data[l]["total"] if agree_data[l]["total"] > 0 else 0 for l in labels]

            fig_agree = go.Figure()
            fig_agree.add_trace(go.Bar(
                x=labels, y=accs, name="Accuracy",
                marker_color=["#ffa500", "#00eaff", "#00ff9d"],
                text=[f"{a:.0%}<br>({t} trades)" for a, t in zip(accs, totals)],
                textposition="auto", textfont=dict(color="#fff"),
            ))
            fig_agree.update_layout(**_plotly_layout(
                title="Accuracy by Model Agreement Level", height=320,
                yaxis=dict(tickformat=".0%", range=[0, 1]),
            ))
            st.plotly_chart(fig_agree, use_container_width=True)
        else:
            st.info("Need 10+ ensemble predictions for agreement analysis.")

    # ── Row 3: Sharpe & Drawdown ──
    if equity_curve and len(equity_curve) > 10:
        eq_vals = [float(e.get("v", 0)) for e in equity_curve]
        eq_times = [e.get("t", "")[:16] for e in equity_curve]

        col_sharpe, col_dd = st.columns(2)

        with col_sharpe:
            rolling_s = compute_rolling_sharpe(eq_vals, window=min(50, len(eq_vals) // 2))
            if rolling_s:
                fig_s = go.Figure()
                fig_s.add_trace(go.Scatter(
                    y=rolling_s, mode="lines",
                    line=dict(color="#00eaff", width=2),
                    fill="tozeroy", fillcolor="rgba(0,234,255,0.08)",
                ))
                fig_s.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
                fig_s.add_hline(y=1, line_dash="dot", line_color="#00ff9d",
                                annotation_text="Good (1.0)", annotation_font_size=10)
                fig_s.update_layout(**_plotly_layout(title="Rolling Sharpe Ratio", height=280))
                st.plotly_chart(fig_s, use_container_width=True)

        with col_dd:
            dd = compute_drawdown(eq_vals)
            if dd:
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=eq_times[-len(dd):], y=dd, mode="lines",
                    line=dict(color="#ff4d6d", width=1.5),
                    fill="tozeroy", fillcolor="rgba(255,77,109,0.15)",
                ))
                fig_dd.update_layout(**_plotly_layout(
                    title="Drawdown (%)", height=280,
                    yaxis=dict(ticksuffix="%"),
                ))
                st.plotly_chart(fig_dd, use_container_width=True)

    # ── Radar Chart ──
    try:
        from src.models.benchmark import ModelBenchmark
        bench = ModelBenchmark(model_version=state.get("model_version", "v6.0"))

        our_scores = {}
        for key, _, metric_key, _ in [
            ("lightgbm", "", "lgbm_direction_accuracy", ""),
            ("patchtst", "", "ptst_forecast_accuracy", ""),
            ("rl_agent", "", "rl_action_accuracy", ""),
        ]:
            d = per_model.get(key, {})
            t = d.get("total", 0)
            c = d.get("correct", 0)
            our_scores[metric_key] = c / t if t > 0 else 0

        our_scores["ensemble_win_rate"] = ensemble_acc
        our_scores["ensemble_sharpe"] = float(performance.get("sharpe_ratio", 0))

        fig_radar = bench.generate_radar_chart(our_scores)
        if fig_radar:
            fig_radar.update_layout(**_plotly_layout(title="Multi-Metric Performance Radar", height=400))
            st.plotly_chart(fig_radar, use_container_width=True)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: TRAINING PROGRESS
# ═══════════════════════════════════════════════════════════════════════

def render_section_training(state: dict, fi_df: pd.DataFrame, journal: list):
    st.markdown('<div class="section-title">TRAINING PROGRESS</div>', unsafe_allow_html=True)

    # ── Status Cards ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        version = state.get("model_version", "v6.0")
        st.markdown(f"""<div class="glass-card">
            <div class="metric-label">MODEL VERSION</div>
            <div class="metric-value metric-value-cyan">{version}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        status = state.get("training_status", "IDLE")
        status_color = "metric-value-green" if status == "IDLE" else "metric-value-orange"
        st.markdown(f"""<div class="glass-card">
            <div class="metric-label">TRAINING STATUS</div>
            <div class="metric-value {status_color}">{status}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        gain = float(state.get("performance_gain", 0))
        color = "metric-value-green" if gain > 0 else "metric-value-red"
        st.markdown(f"""<div class="glass-card">
            <div class="metric-label">IMPROVEMENT DELTA</div>
            <div class="metric-value {color}">{gain:+.1f}%</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        next_train = state.get("next_training", "Not scheduled")
        st.markdown(f"""<div class="glass-card">
            <div class="metric-label">NEXT TRAINING</div>
            <div class="metric-value" style="font-size:1.1rem; color:#8b8ba7;">{next_train or 'Not scheduled'}</div>
        </div>""", unsafe_allow_html=True)

    # ── Feature Importance + Calibration ──
    col_fi, col_cal = st.columns(2)

    with col_fi:
        if not fi_df.empty and len(fi_df) > 0:
            top_10 = fi_df.nlargest(10, "importance")
            fig_fi = go.Figure(go.Bar(
                x=top_10["importance"].values[::-1],
                y=top_10["feature"].values[::-1],
                orientation="h",
                marker_color=["#00eaff" if i < 3 else "#3498db" if i < 6 else "#555"
                              for i in range(len(top_10))][::-1],
                text=top_10["importance"].values[::-1],
                textposition="auto", textfont=dict(color="#fff"),
            ))
            fig_fi.update_layout(**_plotly_layout(
                title="LightGBM Feature Importance (Top 10)", height=350,
                margin=dict(l=160, r=20, t=40, b=40),
            ))
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance data not available. Run training to generate.")

    with col_cal:
        # Calibration Plot: bucket predictions by confidence, check actual accuracy
        # Use closed trades if available, otherwise use open trades with unrealized P&L
        closed_trades = [t for t in journal if t.get("status") == "CLOSED" and t.get("confidence")]
        if len(closed_trades) < 10:
            # Fall back to all trades with confidence (use direction match as proxy)
            closed_trades = [t for t in journal if t.get("confidence")]
        if len(closed_trades) >= 10:
            buckets = [(0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
            bucket_labels = []
            predicted_confs = []
            actual_rates = []

            for lo, hi in buckets:
                in_bucket = [t for t in closed_trades if lo <= float(t.get("confidence", 0)) < hi]
                if not in_bucket:
                    continue
                wins = sum(1 for t in in_bucket
                          if float(t.get("pnl", t.get("unrealized_pnl", t.get("net_pnl", 0)))) > 0)
                bucket_labels.append(f"{lo:.0%}-{hi:.0%}")
                predicted_confs.append((lo + hi) / 2)
                actual_rates.append(wins / len(in_bucket))

            if predicted_confs:
                fig_cal = go.Figure()
                fig_cal.add_trace(go.Scatter(
                    x=predicted_confs, y=actual_rates,
                    mode="markers+lines", name="Actual Win Rate",
                    marker=dict(size=10, color="#00eaff"),
                    line=dict(color="#00eaff", width=2),
                    text=bucket_labels, hovertemplate="Confidence: %{x:.0%}<br>Actual: %{y:.0%}<extra></extra>",
                ))
                fig_cal.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines", name="Perfect Calibration",
                    line=dict(color="#555", dash="dash"),
                ))
                fig_cal.update_layout(**_plotly_layout(
                    title="Model Calibration (Confidence vs Win Rate)", height=350,
                    xaxis=dict(tickformat=".0%", title="Predicted Confidence"),
                    yaxis=dict(tickformat=".0%", title="Actual Win Rate", range=[0, 1]),
                ))
                st.plotly_chart(fig_cal, use_container_width=True)
            else:
                st.info("Insufficient closed trades for calibration plot.")
        else:
            st.info("Need 10+ closed trades with confidence scores for calibration analysis.")

    # ── Benchmark History Timeline ──
    history = load_benchmark_history()
    if history and len(history) >= 2:
        with st.expander("Benchmark History Timeline"):
            timestamps = []
            metrics_over_time: Dict[str, list] = {}
            for snap in history[-50:]:
                ts = snap.get("timestamp", "")
                timestamps.append(ts[:16] if ts else "")
                results = snap.get("results", {})
                for metric_key, result in results.items():
                    if metric_key not in metrics_over_time:
                        metrics_over_time[metric_key] = []
                    metrics_over_time[metric_key].append(float(result.get("value", 0)))

            fig_hist = go.Figure()
            colors = ["#00eaff", "#00ff9d", "#ffa500", "#ff4d6d", "#9b59b6"]
            for i, (mk, vals) in enumerate(metrics_over_time.items()):
                fig_hist.add_trace(go.Scatter(
                    x=timestamps[:len(vals)], y=vals, mode="lines+markers",
                    name=mk.replace("_", " ").title(),
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=5),
                ))
            fig_hist.update_layout(**_plotly_layout(
                title="Model Performance Over Time", height=350,
                yaxis=dict(tickformat=".0%"),
            ))
            st.plotly_chart(fig_hist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: SYSTEM HEALTH
# ═══════════════════════════════════════════════════════════════════════

def render_section_health(state: dict):
    st.markdown('<div class="section-title">SYSTEM HEALTH</div>', unsafe_allow_html=True)

    sources = state.get("sources", {})
    execution = state.get("execution", {})
    risk = state.get("risk_metrics", {})
    status = state.get("status", "UNKNOWN")
    last_update = state.get("last_update", "")

    # ── Row 1: Source Status + System Status ──
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

    # ── Row 2: Execution Metrics + Risk ──
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        latency = float(execution.get("latency_ms", 0))
        lat_color = "metric-value-green" if latency < 100 else "metric-value-orange" if latency < 500 else "metric-value-red"
        st.markdown(f"""<div class="glass-card">
            <div class="metric-label">API LATENCY</div>
            <div class="metric-value {lat_color}">{latency:.0f}ms</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        slippage = float(execution.get("slippage", 0))
        slip_color = "metric-value-green" if slippage < 0.1 else "metric-value-orange"
        st.markdown(f"""<div class="glass-card">
            <div class="metric-label">AVG SLIPPAGE</div>
            <div class="metric-value {slip_color}">{slippage:.3f}%</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        fill = float(execution.get("fill_rate", 100))
        fill_color = "metric-value-green" if fill >= 99 else "metric-value-orange"
        st.markdown(f"""<div class="glass-card">
            <div class="metric-label">FILL RATE</div>
            <div class="metric-value {fill_color}">{fill:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        max_dd = float(risk.get("max_drawdown", 0))
        dd_color = "metric-value-green" if abs(max_dd) < 5 else "metric-value-orange" if abs(max_dd) < 10 else "metric-value-red"
        st.markdown(f"""<div class="glass-card">
            <div class="metric-label">MAX DRAWDOWN</div>
            <div class="metric-value {dd_color}">{max_dd:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    # ── Row 3: Latency Gauge + Risk Score ──
    col_lat, col_risk = st.columns(2)

    with col_lat:
        fig_lat = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latency,
            number={"suffix": "ms", "font": {"size": 28, "color": "#fff"}},
            title={"text": "Exchange API Latency", "font": {"size": 13, "color": "#8b8ba7"}},
            gauge={
                "axis": {"range": [0, 1000], "tickcolor": "#555"},
                "bar": {"color": "#00ff9d" if latency < 100 else "#ffa500" if latency < 500 else "#ff4d6d"},
                "bgcolor": "rgba(20,20,40,0.3)",
                "steps": [
                    {"range": [0, 100], "color": "rgba(0,255,157,0.1)"},
                    {"range": [100, 500], "color": "rgba(255,165,0,0.1)"},
                    {"range": [500, 1000], "color": "rgba(255,77,109,0.1)"},
                ],
            },
        ))
        fig_lat.update_layout(**_plotly_layout(height=250))
        st.plotly_chart(fig_lat, use_container_width=True)

    with col_risk:
        vpin = float(risk.get("vpin_threshold", 0.8))
        risk_score = float(risk.get("risk_score", 0))

        fig_risk = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={"font": {"size": 28, "color": "#fff"}},
            title={"text": f"Risk Score (VPIN threshold: {vpin:.2f})", "font": {"size": 13, "color": "#8b8ba7"}},
            gauge={
                "axis": {"range": [0, 1], "tickcolor": "#555"},
                "bar": {"color": "#00ff9d" if risk_score < 0.3 else "#ffa500" if risk_score < 0.6 else "#ff4d6d"},
                "bgcolor": "rgba(20,20,40,0.3)",
                "steps": [
                    {"range": [0, 0.3], "color": "rgba(0,255,157,0.1)"},
                    {"range": [0.3, 0.6], "color": "rgba(255,165,0,0.1)"},
                    {"range": [0.6, 1], "color": "rgba(255,77,109,0.1)"},
                ],
            },
        ))
        fig_risk.update_layout(**_plotly_layout(height=250))
        st.plotly_chart(fig_risk, use_container_width=True)

    # ── Last Update ──
    if last_update:
        st.markdown(f'<div style="text-align:center; color:#555; font-size:0.7rem; margin-top:10px;">Last State Update: {last_update}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    # ── Header ──
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

    # ── Load Data ──
    config = load_config()
    state = load_state()
    journal = load_journal()
    fi_df = load_feature_importance()

    # ── Render Sections ──
    try:
        render_section_daily_target(state, config)
    except Exception as e:
        st.error(f"Daily Target section error: {e}")

    try:
        render_section_model_accuracy(state)
    except Exception as e:
        st.error(f"Model Accuracy section error: {e}")

    try:
        render_section_ensemble(state)
    except Exception as e:
        st.error(f"Ensemble section error: {e}")

    try:
        render_section_training(state, fi_df, journal)
    except Exception as e:
        st.error(f"Training Progress section error: {e}")

    try:
        render_section_health(state)
    except Exception as e:
        st.error(f"System Health section error: {e}")

    # ── Footer ──
    st.markdown("""
    <div style="text-align:center; color:#333; font-size:0.65rem; margin-top:40px; padding:20px 0; border-top:1px solid rgba(255,255,255,0.03);">
        Performance Metrics Dashboard v1.0 | Autonomous Trading System v6.5
    </div>
    """, unsafe_allow_html=True)

    # ── Auto-refresh ──
    time.sleep(10)
    st.rerun()


if __name__ == "__main__":
    main()
