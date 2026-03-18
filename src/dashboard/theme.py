"""
ProJournX-Inspired Dashboard Theme
====================================
Clean dark navy aesthetic with muted colors and solid cards.
"""

import plotly.graph_objects as go
import calendar
from html import escape as _he

# ── Color Palette (ProJournX-inspired muted tones) ──
GREEN = "#22c55e"
RED = "#ef4444"
BLUE = "#3b82f6"
AMBER = "#f59e0b"
CYAN = "#06b6d4"
PURPLE = "#a855f7"
WHITE = "#e2e8f0"
MUTED = "#94a3b8"
CARD_BG = "#1a1e2e"
CARD_BORDER = "#2a2e3e"
BG_DEEP = "#0f1117"
BG_SIDEBAR = "#131722"

MARKETEDGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: #0f1117 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #131722 !important;
    border-right: 1px solid #1e2330 !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span {
    color: #94a3b8;
}

/* Cards */
.pj-card {
    background: #1a1e2e; border: 1px solid #2a2e3e;
    border-radius: 10px; padding: 18px;
    margin-bottom: 14px;
    transition: border-color 0.2s ease;
}
.pj-card:hover { border-color: #3b82f6; }

.glass-card, .layer-card {
    background: #1a1e2e; border: 1px solid #2a2e3e;
    border-radius: 10px; padding: 18px;
    margin-bottom: 14px;
}

/* Metric cards */
.metric-label, .metric-small {
    font-size: 0.7rem; color: #64748b; text-transform: uppercase;
    letter-spacing: 0.8px; font-weight: 500; margin-bottom: 4px;
}
.metric-value, .metric-large {
    font-size: 1.6rem; font-weight: 700; color: #e2e8f0;
    line-height: 1.2;
}
.metric-value-green { color: #22c55e; }
.metric-value-red { color: #ef4444; }
.metric-value-cyan { color: #06b6d4; }
.metric-value-orange { color: #f59e0b; }
.metric-sub {
    font-size: 0.75rem; color: #64748b; margin-top: 2px;
}

/* Section headers */
.section-title, .layer-title {
    font-size: 0.8rem; font-weight: 600; letter-spacing: 1.2px;
    color: #64748b; text-transform: uppercase;
    margin: 28px 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2330;
}

/* Greeting */
.greeting-bar {
    padding: 16px 0;
    margin-bottom: 20px;
    border-bottom: 1px solid #1e2330;
}
.greeting-text {
    font-size: 1.4rem; font-weight: 600; color: #e2e8f0;
}
.greeting-date {
    font-size: 0.78rem; color: #64748b;
}

/* Badges */
.badge, .status-badge, .streak-badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.5px;
}
.badge-green, .status-ok, .streak-hot { background: rgba(34,197,94,0.12); color: #22c55e; border: 1px solid rgba(34,197,94,0.25); }
.badge-red, .status-error, .streak-cold { background: rgba(239,68,68,0.12); color: #ef4444; border: 1px solid rgba(239,68,68,0.25); }
.badge-blue { background: rgba(59,130,246,0.12); color: #3b82f6; border: 1px solid rgba(59,130,246,0.25); }
.badge-amber, .status-warn { background: rgba(245,158,11,0.12); color: #f59e0b; border: 1px solid rgba(245,158,11,0.25); }
.badge-gray { background: rgba(100,116,139,0.12); color: #94a3b8; border: 1px solid rgba(100,116,139,0.25); }

.status-green { color: #22c55e; font-weight: 700; }
.status-red { color: #ef4444; font-weight: 700; }

.status-dot, .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 5px; }
.dot-green, .dot-green { background: #22c55e; box-shadow: 0 0 5px rgba(34,197,94,0.5); }
.dot-red { background: #ef4444; box-shadow: 0 0 5px rgba(239,68,68,0.5); }
.dot-yellow, .dot-amber { background: #f59e0b; box-shadow: 0 0 5px rgba(245,158,11,0.5); }

/* Agent cards */
.agent-vote-card, .agent-card {
    background: #1a1e2e; border: 1px solid #2a2e3e;
    border-radius: 10px; padding: 12px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.agent-vote-card:hover, .agent-card:hover { border-color: #3b82f6; }
.agent-name {
    font-size: 0.7rem; color: #94a3b8; text-transform: uppercase;
    letter-spacing: 0.5px; font-weight: 600;
}
.agent-direction { font-size: 1.3rem; font-weight: 700; }
.dir-long { color: #22c55e; }
.dir-short { color: #ef4444; }
.dir-flat { color: #64748b; }

/* Misc */
.sub-metric-grid {
    display: grid; grid-template-columns: repeat(2, 1fr);
    gap: 12px; margin: 10px 0;
}
.sub-metric-item {
    padding: 10px; background: rgba(255,255,255,0.02);
    border-radius: 8px; border: 1px solid #2a2e3e;
}

.attribution-bar {
    height: 7px; border-radius: 6px; background: #1e2330;
    display: flex; overflow: hidden; margin-top: 10px;
}

.model-card {
    background: #1a1e2e; border: 1px solid #2a2e3e;
    border-radius: 10px; padding: 16px; margin-bottom: 12px;
}
.model-name { font-size: 0.75rem; color: #3b82f6; letter-spacing: 1px; font-weight: 600; text-transform: uppercase; }
.model-accuracy { font-size: 2rem; font-weight: 700; margin: 6px 0; }
.model-trend-up { color: #22c55e; font-size: 0.8rem; }
.model-trend-down { color: #ef4444; font-size: 0.8rem; }
.model-trend-flat { color: #94a3b8; font-size: 0.8rem; }

/* Calendar cells */
.cal-cell {
    display: inline-flex; align-items: center; justify-content: center;
    width: 36px; height: 36px; border-radius: 6px;
    font-size: 0.75rem; font-weight: 500; margin: 2px;
}

/* Trade rows */
.trade-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px; border-radius: 8px;
    background: rgba(26,30,46,0.6);
    border-left: 3px solid transparent;
    margin-bottom: 4px;
}
.trade-win { border-left-color: #22c55e; }
.trade-loss { border-left-color: #ef4444; }

/* Layer progress bars */
.layer-bar-bg {
    background: #1e2330; border-radius: 4px; height: 6px; overflow: hidden;
}
.layer-bar-fill {
    height: 100%; border-radius: 4px; transition: width 0.3s ease;
}

/* Streamlit overrides */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: #131722; border-radius: 8px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px; font-size: 0.8rem; font-weight: 500;
    color: #94a3b8; padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: #1a1e2e !important; color: #e2e8f0 !important;
}

.stExpander {
    background: #1a1e2e !important; border: 1px solid #2a2e3e !important;
    border-radius: 10px !important; margin-bottom: 8px !important;
}
.stExpander > div:first-child {
    color: #94a3b8 !important; font-size: 0.85rem !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""


def metric_card(title: str, value: str, color: str = WHITE,
                subtitle: str = "", icon: str = "") -> str:
    """Render a ProJournX-style metric card."""
    icon_html = f'<span style="font-size:1.1rem;margin-right:6px">{icon}</span>' if icon else ''
    sub_html = f'<div class="metric-sub">{subtitle}</div>' if subtitle else ''
    return (
        f'<div class="pj-card" style="text-align:center">'
        f'{icon_html}'
        f'<div class="metric-label">{title}</div>'
        f'<div class="metric-value" style="color:{color}">{value}</div>'
        f'{sub_html}'
        f'</div>'
    )


def plotly_layout(**kwargs) -> dict:
    """Standard dark Plotly layout matching ProJournX theme."""
    base = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', family='Inter', size=11),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.08)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.08)'),
        legend=dict(font=dict(size=10, color='#94a3b8'), bgcolor='rgba(0,0,0,0)'),
    )
    base.update(kwargs)
    return base


def radar_chart(categories: list, values: list, title: str = "AI Performance Score") -> go.Figure:
    """Radar chart for AI performance metrics."""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(59,130,246,0.15)',
        line=dict(color=BLUE, width=2),
        marker=dict(size=5, color=BLUE),
    ))
    fig.update_layout(
        **plotly_layout(title=dict(text=title, font=dict(size=13))),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.06)',
                            tickfont=dict(size=9, color='#475569')),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.06)',
                             tickfont=dict(size=10, color='#94a3b8')),
            bgcolor='rgba(0,0,0,0)',
        ),
        showlegend=False, height=300,
    )
    return fig


def calendar_heatmap_html(daily_pnl: dict, year: int, month: int) -> str:
    """
    HTML calendar heatmap for daily P&L.
    daily_pnl: Dict mapping 'YYYY-MM-DD' to float P&L.
    """
    cal = calendar.Calendar(firstweekday=6)
    days = cal.monthdayscalendar(year, month)
    month_name = calendar.month_name[month]
    day_headers = ['S', 'M', 'T', 'W', 'T', 'F', 'S']

    html = f"""
    <div class="pj-card">
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px">
            <div class="metric-label" style="margin:0">{month_name} {year}</div>
        </div>
        <div style="display:grid; grid-template-columns:repeat(7, 1fr); gap:4px; text-align:center">
    """
    for d in day_headers:
        html += f'<div style="font-size:0.65rem; color:#475569; padding:4px">{d}</div>'

    for week in days:
        for day in week:
            if day == 0:
                html += '<div></div>'
                continue
            date_key = f"{year}-{month:02d}-{day:02d}"
            pnl = daily_pnl.get(date_key)
            if pnl is None:
                bg, tc = '#1e2330', '#475569'
            elif pnl > 0:
                intensity = min(1.0, abs(pnl) / 500)
                bg, tc = f'rgba(34,197,94,{0.15 + intensity * 0.5})', '#22c55e'
            else:
                intensity = min(1.0, abs(pnl) / 500)
                bg, tc = f'rgba(239,68,68,{0.15 + intensity * 0.5})', '#ef4444'
            tip = f'title="${pnl:+,.2f}"' if pnl is not None else ''
            html += f'<div class="cal-cell" style="background:{bg};color:{tc}" {tip}>{day}</div>'

    html += '</div></div>'
    return html


def source_badge(name: str, status: str) -> str:
    """Source status badge (ONLINE/OFFLINE)."""
    ok = status.upper() == 'ONLINE'
    dc = 'dot-green' if ok else 'dot-red'
    bc = 'badge-green' if ok else 'badge-red'
    return f'<span class="badge {bc}"><span class="dot {dc}"></span>{_he(name)}: {_he(status)}</span>'
