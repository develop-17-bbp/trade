"""
Page 1: Main Dashboard — ProJournX Style
==========================================
Greeting bar, 5-metric row, radar chart, P&L charts,
calendar heatmap, recent trades, Polymarket section.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st
import json
import plotly.graph_objects as go
from html import escape as h
from datetime import datetime, timedelta
from collections import defaultdict

from src.dashboard.data import (
    load_dashboard_state,
    load_journal_trades,
    compute_today_pnl,
)
from src.dashboard.theme import (
    MARKETEDGE_CSS, metric_card, plotly_layout, radar_chart, calendar_heatmap_html,
    source_badge, GREEN, RED, BLUE, AMBER, CYAN, PURPLE, WHITE, MUTED,
)

st.markdown(MARKETEDGE_CSS, unsafe_allow_html=True)


def _html(html_str: str):
    """Render HTML without Markdown code-block interpretation from indentation."""
    import textwrap
    st.markdown(textwrap.dedent(html_str), unsafe_allow_html=True)


def _parse_ts(ts) -> str:
    """Normalize a timestamp to ISO date string. Handles both ISO strings and Unix floats."""
    if not ts:
        return ''
    ts_str = str(ts)
    # If it looks like a Unix timestamp (all digits / float)
    try:
        if ts_str.replace('.', '', 1).isdigit():
            return datetime.fromtimestamp(float(ts_str)).isoformat()
    except (ValueError, OSError):
        pass
    return ts_str


def _safe_pnl(t: dict) -> float:
    """Coerce trade pnl to float; return 0 on missing or invalid."""
    try:
        v = t.get('pnl')
        if v is None:
            return 0.0
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _safe_float(x, default: float = 0.0) -> float:
    """Coerce to float for display; return default on invalid."""
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _compute_trade_stats(trades):
    """Compute win rate, profit factor, risk:reward from trade history (CLOSED trades only)."""
    if not trades:
        return {'win_rate': 0.0, 'profit_factor': 0.0, 'risk_reward': '1:0.00', 'total_pnl': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0}
    # Only CLOSED trades count for these metrics
    closed = [
        t for t in trades
        if isinstance(t, dict) and (
            t.get('status') == 'CLOSED' or
            ('exit_price' in t and t.get('status') != 'OPEN')
        )
    ]
    if not closed:
        return {'win_rate': 0.0, 'profit_factor': 0.0, 'risk_reward': '1:0.00', 'total_pnl': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0}

    pnls = [_safe_pnl(t) for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_pnl = sum(wins)
    loss_pnl = abs(sum(losses))
    total_pnl = sum(pnls)

    if loss_pnl > 0:
        pf = win_pnl / loss_pnl
        rr = f"1:{pf:.2f}"
    else:
        pf = float('inf') if win_pnl > 0 else 0.0
        rr = "1:∞" if win_pnl > 0 else "1:0.00"

    return {
        'win_rate': len(wins) / len(closed),
        'profit_factor': pf,
        'risk_reward': rr,
        'total_pnl': total_pnl,
        'avg_win': win_pnl / len(wins) if wins else 0.0,
        'avg_loss': loss_pnl / len(losses) if losses else 0.0,
    }


def _metrics_for_display(trade_stats: dict, portfolio: dict, today_str: str,
                         journal_trades: list) -> dict:
    """Metrics for cards; today's P&L uses same rules as sidebar (compute_today_pnl)."""
    realized_pnl = _safe_float(trade_stats.get('total_pnl'), 0.0)
    wr = max(0.0, min(1.0, _safe_float(trade_stats.get('win_rate'), 0.0)))
    pf = trade_stats.get('profit_factor', 0.0)
    try:
        pf = float(pf) if pf != float('inf') else pf
    except (TypeError, ValueError):
        pf = 0.0
    rr = str(trade_stats.get('risk_reward', '1:0.00') or '1:0.00')
    acc = wr
    today_pnl, _ = compute_today_pnl(portfolio, journal_trades, today_str)
    return {
        'today_pnl': float(today_pnl),
        'realized_pnl': realized_pnl,
        'win_rate': wr,
        'profit_factor': pf,
        'ai_accuracy': acc,
        'risk_reward': rr,
    }


def _compute_daily_pnl(trades):
    """Aggregate P&L by date for calendar and charts. Only closed trades count."""
    daily = defaultdict(float)
    for t in trades:
        if not isinstance(t, dict):
            continue
        if t.get('status') != 'CLOSED' and not t.get('exit_price'):
            continue
        ts = t.get('exit_time') or t.get('timestamp', '')
        if not ts:
            continue
        iso = _parse_ts(ts)
        day = iso[:10] if len(iso) >= 10 else ''
        if day:
            try:
                daily[day] += float(t.get('pnl', 0) or 0)
            except (TypeError, ValueError):
                pass
    return dict(daily)


# ══════════════════════════════════════════════
# MAIN RENDER — cached loaders; journal is authoritative for realized stats
# ══════════════════════════════════════════════
state = load_dashboard_state()
portfolio = state.get('portfolio', {})
_journal_trades = load_journal_trades()
_state_trades = state.get('trade_history', []) or []
# Journal first (persistent closed trades); state fills gaps if journal empty this session
trades = _journal_trades if _journal_trades else _state_trades
open_positions = state.get('open_positions', {})
trade_stats = _compute_trade_stats(trades)
daily_pnl = _compute_daily_pnl(trades)
sources = state.get('sources', {})
agent_overlay = state.get('agent_overlay', {})
polymarket = state.get('polymarket', {})

now = datetime.now()

# ── Greeting Bar ──
hour = now.hour
greeting = "Good morning" if hour < 12 else ("Good afternoon" if hour < 17 else "Good evening")

today_str = now.strftime('%Y-%m-%d')

def _trade_date(t):
    """Date of trade for 'today' filtering: use exit_time for closed, else timestamp."""
    ts = t.get('exit_time') or t.get('timestamp') or ''
    if not ts:
        return ''
    iso = _parse_ts(ts)
    return iso[:10] if len(iso) >= 10 else ''

today_trades = [
    t for t in trades
    if isinstance(t, dict) and _trade_date(t) == today_str
]
today_pnl, _today_pnl_src = compute_today_pnl(portfolio, _journal_trades, today_str)
_today_pnl_label = f"Today's P&L ({_today_pnl_src.split('(')[0].strip()})"

# All-time closed trades and win count (use _safe_pnl so bad data doesn't break display)
_all_closed = [t for t in trades if isinstance(t, dict) and (
    t.get('status') == 'CLOSED' or ('exit_price' in t and t.get('status') != 'OPEN')
)]
_total_realized = sum(_safe_pnl(t) for t in _all_closed)
_total_wins = sum(1 for t in _all_closed if _safe_pnl(t) > 0)
today_wins = len([t for t in today_trades if isinstance(t, dict)
                  and t.get('status') == 'CLOSED' and (float(t.get('pnl', 0) or 0) > 0)])

g1, g2 = st.columns([3, 2])
with g1:
    _html(f"""
    <div class="greeting-bar">
        <div class="greeting-text">{greeting}, Trader</div>
        <div class="greeting-date">{now.strftime('%A, %B %d, %Y')} &nbsp; | &nbsp;
        Stay focused. Every trade is a learning opportunity.</div>
    </div>
    """)
with g2:
    _html(f"""
    <div style="display:flex; gap:14px; padding:16px 0; justify-content:flex-end; align-items:center; flex-wrap:wrap">
        <div style="text-align:center">
            <div style="font-size:0.6rem; color:#64748b; text-transform:uppercase">{_today_pnl_label}</div>
            <div style="font-size:1.1rem; font-weight:700; color:{GREEN if today_pnl >= 0 else RED}">${today_pnl:+,.2f}</div>
        </div>
        <div style="text-align:center">
            <div style="font-size:0.6rem; color:#64748b; text-transform:uppercase">Realized P&L (all)</div>
            <div style="font-size:1.1rem; font-weight:700; color:{GREEN if _total_realized >= 0 else RED}">${_total_realized:+,.2f}</div>
        </div>
        <div style="text-align:center">
            <div style="font-size:0.6rem; color:#64748b; text-transform:uppercase">Today / Total</div>
            <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0">{len(today_trades)} / {len(trades)}</div>
        </div>
        <div style="text-align:center">
            <div style="font-size:0.6rem; color:#64748b; text-transform:uppercase">Wins (all)</div>
            <div style="font-size:1.1rem; font-weight:700; color:{GREEN}">{_total_wins}</div>
        </div>
    </div>
    """)

# ── Source Status Badges ──
badges_html = ' '.join([
    source_badge('EXCHANGE', sources.get('exchange', 'UNKNOWN')),
    source_badge('NEWS', sources.get('news', 'UNKNOWN')),
    source_badge('ON-CHAIN', sources.get('onchain', 'UNKNOWN')),
    source_badge('LLM', sources.get('llm', 'UNKNOWN')),
    source_badge('POLYMARKET', 'ONLINE' if polymarket.get('active_markets', 0) > 0 else 'OFFLINE'),
])
st.markdown(f'<div style="margin-bottom:16px">{badges_html}</div>', unsafe_allow_html=True)

# ── Tabs ──
tab_live, tab_agents, tab_overview = st.tabs(["Live Trading", "Agent Overlay", "Overview"])

# ══════════════════════════════════════════════
# TAB: LIVE TRADING
# ══════════════════════════════════════════════
with tab_live:
    state = load_dashboard_state()
    portfolio = state.get('portfolio', {})
    open_positions = state.get('open_positions', {})
    _j = load_journal_trades()
    _st = state.get('trade_history', []) or []
    trades = _j if _j else _st
    trade_stats = _compute_trade_stats(trades)
    daily_pnl_live = _compute_daily_pnl(trades)
    m = _metrics_for_display(trade_stats, portfolio, today_str, _j if _j else trades)

    _closed_count = len([t for t in trades if isinstance(t, dict) and t.get('status') == 'CLOSED'])
    _total_count = len([t for t in trades if isinstance(t, dict)])
    _wtv = portfolio.get("current_total_value")
    _wallet_display = f"${_safe_float(_wtv, 0.0):,.2f}" if _wtv is not None else "—"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(metric_card("Wallet balance", _wallet_display, "#e2e8f0",
                                subtitle="NAV from exchange (same as state)"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Realized P&L", f"${m['realized_pnl']:+,.2f}",
                                GREEN if m['realized_pnl'] >= 0 else RED,
                                subtitle=f"{_closed_count} closed / {_total_count} total"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Win Rate", f"{m['win_rate']:.1%}", GREEN if m['win_rate'] > 0.5 else RED,
                                subtitle=f"{_closed_count} closed trades"), unsafe_allow_html=True)
    with c4:
        pf_val = m['profit_factor']
        pf_color = GREEN if pf_val > 1 else RED
        pf_display = f"{pf_val:.2f}" if pf_val < 100 else "∞"
        st.markdown(metric_card("Profit Factor", pf_display, pf_color,
                                subtitle="BREAKEVEN" if pf_val == 0 else ""), unsafe_allow_html=True)
    with c5:
        st.markdown(metric_card("AI Accuracy", f"{m['ai_accuracy']:.1%}", BLUE,
                                subtitle=f"Model {state.get('model_version', '?')}"), unsafe_allow_html=True)
    with c6:
        st.markdown(metric_card("Risk:Reward", m['risk_reward'], AMBER), unsafe_allow_html=True)

    # ── Charts Row ──
    ch1, ch2 = st.columns([1, 2])

    with ch1:
        # AI Performance Radar (use normalized metrics)
        categories = ['Win %', 'Consistency', 'Profit Factor', 'Recovery', 'Max DD', 'Avg Win/Loss']
        consistency = min(100, max(0, m['win_rate'] * 100))
        pf_norm = min(100, m['profit_factor'] * 30) if m['profit_factor'] < 100 else 100
        max_dd = state.get('risk_metrics', {}).get('max_drawdown', 0)
        dd_score = max(0, 100 - abs(max_dd) * 10)
        # Recovery: based on how much of max drawdown has been recovered
        cur_dd = state.get('risk_metrics', {}).get('current_drawdown', 0)
        recovery = max(0, min(100, 100 - abs(cur_dd) * 10)) if max_dd != 0 else 50
        avg_wl = 0
        if trade_stats.get('avg_loss', 0) > 0:
            avg_wl = min(100, (trade_stats.get('avg_win', 0) / trade_stats['avg_loss']) * 30)

        vals = [m['win_rate'] * 100, consistency, pf_norm, recovery, dd_score, avg_wl]
        fig = radar_chart(categories, vals)
        st.plotly_chart(fig, width="stretch", key="radar")

        ai_score = sum(vals) / len(vals)
        sc = GREEN if ai_score > 50 else (AMBER if ai_score > 30 else RED)
        _html(f"""
        <div style="text-align:center">
            <div style="font-size:0.7rem; color:#64748b">AI Score</div>
            <div style="font-size:1.8rem; font-weight:700; color:{sc}">{ai_score:.1f}</div>
        </div>
        """)

    with ch2:
        # Daily & Cumulative P&L Chart (dates = days with closed trades)
        if daily_pnl_live:
            daily_pnl = daily_pnl_live
            sorted_days = sorted(daily_pnl.keys())
            pnl_vals = [daily_pnl[d] for d in sorted_days]
            cum_pnl = []
            running = 0
            for v in pnl_vals:
                running += v
                cum_pnl.append(running)

            colors = [GREEN if v >= 0 else RED for v in pnl_vals]

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=sorted_days, y=pnl_vals, name='Daily P&L',
                marker_color=colors, opacity=0.7,
            ))
            fig2.add_trace(go.Scatter(
                x=sorted_days, y=cum_pnl, name='Cumulative P&L',
                line=dict(color=BLUE, width=2), yaxis='y2',
            ))
            fig2.update_layout(**plotly_layout(
                title=dict(text='Daily & Cumulative P&L (by close date)', font=dict(size=13)),
                yaxis=dict(title='Daily ($)', gridcolor='rgba(255,255,255,0.04)'),
                yaxis2=dict(title='Cumulative ($)', overlaying='y', side='right',
                            gridcolor='rgba(255,255,255,0.04)'),
                barmode='relative', height=350,
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
            ))
            st.plotly_chart(fig2, width="stretch", key="pnl_chart")
            st.caption("Bars show realized P&L on the day each trade closed. Days with no closed trades have no bar.")
        else:
            st.info("No trade data yet for P&L chart")

    # ── Calendar Heatmap ──
    st.markdown('<div class="section-title">P&L Calendar</div>', unsafe_allow_html=True)
    cal_month = st.selectbox("Month", list(range(1, 13)),
                             index=now.month - 1, format_func=lambda m: f"{now.year}-{m:02d}",
                             key="cal_month", label_visibility="collapsed")
    st.markdown(calendar_heatmap_html(daily_pnl_live, now.year, cal_month), unsafe_allow_html=True)

    # ── Open Positions ──
    st.markdown('<div class="section-title">Open Positions</div>', unsafe_allow_html=True)
    if open_positions:
        pos_cols = st.columns(min(len(open_positions), 4))
        for idx, (pos_asset, pos) in enumerate(open_positions.items()):
            col = pos_cols[idx % len(pos_cols)]
            with col:
                upnl = pos.get('unrealized_pnl', 0)
                upnl_color = GREEN if upnl >= 0 else RED
                dir_color = GREEN if pos.get('direction') == 'LONG' else RED
                _html(f"""
                <div class="pj-card" style="padding:12px; border-left:3px solid {dir_color}">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px">
                        <span style="font-weight:700; color:#e2e8f0; font-size:0.9rem">{h(pos_asset)}</span>
                        <span style="font-size:0.75rem; font-weight:600; color:{dir_color}; background:{dir_color}22; padding:2px 7px; border-radius:4px">{h(pos.get('direction','?'))}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#94a3b8">
                        <span>Entry</span><span style="color:#e2e8f0">${pos.get('entry_price', 0):,.4f}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#94a3b8">
                        <span>Current</span><span style="color:#e2e8f0">${pos.get('current_price', 0):,.4f}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#94a3b8">
                        <span>Size</span><span style="color:#e2e8f0">{pos.get('size', 0):.6f}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-top:6px; padding-top:6px; border-top:1px solid #1e2330">
                        <span style="color:#64748b">Unrealized P&L</span>
                        <span style="font-weight:700; color:{upnl_color}">${upnl:+,.2f}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.68rem; color:#475569; margin-top:4px">
                        <span>SL: ${pos.get('stop_loss',0):,.4f}</span>
                        <span>TP: ${pos.get('take_profit',0):,.4f}</span>
                    </div>
                </div>
                """)
    else:
        st.caption("No open positions")

    # ── Recent Trades + Polymarket ──
    st.markdown('<div class="section-title">Recent Activity</div>', unsafe_allow_html=True)
    bot_left, bot_right = st.columns([3, 2])

    with bot_left:
        st.markdown("**Recent Trades**")
        # Show closed trades first (most informative), then recent open entries
        _closed = [t for t in trades if isinstance(t, dict) and t.get('status') == 'CLOSED']
        _open_entries = [t for t in trades if isinstance(t, dict) and t.get('status') != 'CLOSED']

        def _sort_key(t):
            ts = t.get('timestamp') or t.get('exit_time') or ''
            if ts is None:
                return '0'
            if isinstance(ts, (int, float)):
                try:
                    return datetime.fromtimestamp(float(ts)).isoformat()
                except (ValueError, OSError):
                    return str(ts)
            return str(ts) if ts else '0'

        try:
            _closed_sorted = sorted(_closed, key=_sort_key, reverse=True)
            _open_sorted = sorted(_open_entries, key=_sort_key, reverse=True)
        except Exception:
            _closed_sorted = _closed
            _open_sorted = _open_entries
        recent = (_closed_sorted + _open_sorted)[:12]
        if recent:
            for t in recent:
                try:
                    pnl_v = float(t.get('pnl') or 0)
                except (TypeError, ValueError):
                    pnl_v = 0.0
                status = t.get('status', 'OPEN')
                cls = 'trade-win' if (status == 'CLOSED' and pnl_v > 0) else ('trade-loss' if (status == 'CLOSED' and pnl_v < 0) else '')
                pnl_c = GREEN if pnl_v > 0 else (RED if pnl_v < 0 else MUTED)
                asset = h(str(t.get('asset', '?')))
                side = h(str(t.get('side', '?')).upper())
                ts = _parse_ts(t.get('timestamp') or t.get('exit_time') or '')[:16]
                try:
                    conf = float(t.get('confidence', 0) or 0)
                except (TypeError, ValueError):
                    conf = 0
                status_badge = f'<span style="font-size:0.65rem; color:{"#22c55e" if status=="CLOSED" else "#64748b"}; margin-left:6px">{status}</span>'
                pnl_display = f'${pnl_v:+,.2f}' if status == 'CLOSED' else '(open)'
                _html(f"""
                <div class="trade-row {cls}">
                    <div>
                        <span style="font-weight:600; color:#e2e8f0">{asset}</span>
                        <span style="font-size:0.7rem; color:#64748b; margin-left:8px">{side} | {ts}</span>
                        {status_badge}
                    </div>
                    <div style="text-align:right">
                        <span style="font-weight:600; color:{pnl_c}">{pnl_display}</span>
                        <span style="font-size:0.7rem; color:#64748b; margin-left:8px">{conf:.0%}</span>
                    </div>
                </div>
                """)
        else:
            st.caption("No trades yet")

    with bot_right:
        st.markdown("**Polymarket Signals**")
        pm = state.get('polymarket', {})
        pm_markets = pm.get('top_markets', [])
        if pm_markets:
            _html(f"""
            <div style="font-size:0.7rem; color:#64748b; margin-bottom:8px">
                {pm.get('active_markets', 0)} active | {pm.get('liquid_markets', 0)} liquid markets
            </div>
            """)
            for m in pm_markets[:6]:
                prob = m.get('probability', 0)
                vol = m.get('volume', 0)
                q = h(m.get('question', '')[:80])
                bar_w = max(2, prob * 100)
                bar_color = GREEN if prob > 0.5 else RED
                _html(f"""
                <div class="pj-card" style="padding:10px; margin-bottom:6px">
                    <div style="font-size:0.72rem; color:#e2e8f0; margin-bottom:6px">{q}</div>
                    <div style="display:flex; align-items:center; gap:8px">
                        <div style="flex:1; background:#1e2330; border-radius:4px; height:6px; overflow:hidden">
                            <div style="width:{bar_w}%; height:100%; background:{bar_color}; border-radius:4px"></div>
                        </div>
                        <span style="font-size:0.7rem; font-weight:600; color:{bar_color}">{prob:.0%}</span>
                        <span style="font-size:0.65rem; color:#475569">${vol:,.0f}</span>
                    </div>
                </div>
                """)
        else:
            st.caption("No Polymarket data — waiting for first fetch")

# ══════════════════════════════════════════════
# TAB: AGENT OVERLAY
# ══════════════════════════════════════════════
with tab_agents:
    ao = state.get('agent_overlay', {})
    if not ao.get('enabled'):
        st.info("Agent overlay is not active. Start the trading system to see agent data.")
    else:
        dec = ao.get('last_decision', {})
        d = dec.get('direction', 0)
        dir_text = 'LONG' if d > 0 else ('SHORT' if d < 0 else 'FLAT')
        dir_color = GREEN if d > 0 else (RED if d < 0 else MUTED)

        # Status row
        s1, s2, s3, s4, s5 = st.columns(5)
        with s1:
            st.markdown(metric_card("Direction", dir_text, dir_color), unsafe_allow_html=True)
        with s2:
            st.markdown(metric_card("Confidence", f"{dec.get('confidence', 0):.1%}", BLUE), unsafe_allow_html=True)
        with s3:
            st.markdown(metric_card("Consensus", ao.get('consensus_level', 'N/A'), CYAN), unsafe_allow_html=True)
        with s4:
            st.markdown(metric_card("Data Quality", f"{ao.get('data_quality', 0):.0%}", GREEN), unsafe_allow_html=True)
        with s5:
            mode = ao.get('daily_pnl_mode', 'NORMAL')
            mode_colors = {'NORMAL': GREEN, 'CAUTION': AMBER, 'DEFENSIVE': RED,
                           'HALT': RED, 'PRESERVATION': CYAN, 'APPROACHING': AMBER}
            st.markdown(metric_card("PnL Mode", mode, mode_colors.get(mode, MUTED)), unsafe_allow_html=True)

        # Agent votes grid
        st.markdown('<div class="section-title">Agent Votes</div>', unsafe_allow_html=True)
        votes = ao.get('agent_votes', {})
        weights = ao.get('agent_weights', {})
        if votes:
            agent_list = list(votes.items())
            cols_per_row = 4
            for i in range(0, len(agent_list), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j >= len(agent_list):
                        break
                    name, v = agent_list[i + j]
                    vd = v.get('direction', 0) if isinstance(v, dict) else 0
                    vc = v.get('confidence', 0) if isinstance(v, dict) else 0
                    vr = h(v.get('reasoning', '') if isinstance(v, dict) else '')
                    vw = weights.get(name, 1.0)
                    dir_t = 'LONG' if vd > 0 else ('SHORT' if vd < 0 else 'FLAT')
                    dir_c = 'dir-long' if vd > 0 else ('dir-short' if vd < 0 else 'dir-flat')
                    with col:
                        _html(f"""
                        <div class="agent-card">
                            <div class="agent-name">{h(name.replace('_', ' '))}</div>
                            <div class="agent-dir {dir_c}">{dir_t}</div>
                            <div style="font-size:0.72rem; color:#64748b">
                                Conf: {vc:.0%} | W: {vw:.2f}x
                            </div>
                            <div style="font-size:0.65rem; color:#475569; margin-top:4px">{vr[:80]}</div>
                        </div>
                        """)
        else:
            st.caption("No agent votes yet — waiting for first cycle")

# ══════════════════════════════════════════════
# TAB: OVERVIEW (TradingView + 9-Layer)
# ══════════════════════════════════════════════
with tab_overview:
    # TradingView chart
    st.markdown('<div class="section-title">Market Chart</div>', unsafe_allow_html=True)
    tv_html = """
    <div class="pj-card" style="padding:0; overflow:hidden; border-radius:10px">
    <div style="height:500px">
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
    new TradingView.widget({
        "container_id": "tv_chart_overview",
        "autosize": true, "symbol": "BINANCE:BTCUSDT", "interval": "5",
        "timezone": "Etc/UTC", "theme": "dark", "style": "1",
        "locale": "en", "toolbar_bg": "#0f1117",
        "enable_publishing": false, "hide_top_toolbar": false,
        "hide_legend": false, "save_image": false,
        "backgroundColor": "#0f1117", "gridColor": "rgba(255,255,255,0.04)"
    });
    </script>
    <div id="tv_chart_overview" style="height:100%"></div>
    </div></div>
    """
    st.components.v1.html(tv_html, height=520)

    # 9-Layer Status
    st.markdown('<div class="section-title">9-Layer System Status</div>', unsafe_allow_html=True)
    layers = state.get('layers', {})
    layer_names = {
        'L1': 'Quant Engine', 'L2': 'Sentiment', 'L3': 'Risk',
        'L4': 'Signal Fusion', 'L5': 'Execution', 'L6': 'Strategist',
        'L7': 'Memory', 'L8': 'Meta-Learning', 'L9': 'Evolution Portal',
    }
    for lid in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9']:
        ldata = layers.get(lid, {})
        status = h(str(ldata.get('status', 'INIT')))
        progress = ldata.get('progress', 0)
        metric = h(str(ldata.get('metric', '')))
        sc = GREEN if status == 'OK' else (AMBER if status == 'WARN' else RED)
        bar_w = max(2, min(100, progress * 100))
        _html(f"""
        <div style="display:flex; align-items:center; gap:12px; padding:8px 0; border-bottom:1px solid #1e2330">
            <div style="width:30px; font-size:0.7rem; color:#64748b; font-weight:600">{lid}</div>
            <div style="width:130px; font-size:0.75rem; color:#e2e8f0">{layer_names.get(lid, lid)}</div>
            <div style="flex:1">
                <div class="layer-bar-bg">
                    <div class="layer-bar-fill" style="width:{bar_w}%; background:{sc}"></div>
                </div>
            </div>
            <div style="width:60px; text-align:right">
                <span class="badge {'badge-green' if status == 'OK' else ('badge-amber' if status == 'WARN' else 'badge-red')}">{status}</span>
            </div>
            <div style="width:120px; font-size:0.7rem; color:#64748b; text-align:right">{str(metric)[:20]}</div>
        </div>
        """)

