"""
Paper Trading Dashboard — Robinhood Real Prices
=================================================
Shows live paper trading: entries, exits, open positions, P&L.
"""
import os
import json
import streamlit as st
from datetime import datetime, timedelta, timezone

from src.dashboard.theme import (
    MARKETEDGE_CSS, GREEN, RED, BLUE, AMBER, CYAN, PURPLE, WHITE, MUTED,
    plotly_layout,
)

st.markdown(MARKETEDGE_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <div>
        <div class="title">PAPER TRADING</div>
        <div class="subtext">Robinhood Real Prices | Read-Only | No Real Orders</div>
    </div>
</div>
""", unsafe_allow_html=True)

def _to_local(utc_str: str) -> str:
    """Convert UTC ISO timestamp to local time display string."""
    if not utc_str:
        return '---'
    try:
        # Parse UTC timestamp (handles both +00:00 and Z suffix)
        s = utc_str.replace('Z', '+00:00')
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local_dt = dt.astimezone()  # Converts to system local timezone
        return local_dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return utc_str[:19]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')


# ── Data Loaders ──

@st.cache_data(ttl=5)
def load_paper_state():
    path = os.path.join(LOG_DIR, 'robinhood_paper_state.json')
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


@st.cache_data(ttl=5)
def load_paper_trades():
    path = os.path.join(LOG_DIR, 'robinhood_paper.jsonl')
    events = []
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except Exception:
            pass
    return events


@st.cache_data(ttl=10)
def get_robinhood_live():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from src.integrations.robinhood_crypto import RobinhoodCryptoClient
        client = RobinhoodCryptoClient()
        if not client.authenticated:
            return {}
        btc = client.get_best_price("BTC-USD")
        eth = client.get_best_price("ETH-USD")
        acct = client.get_account()
        holdings = client.get_holdings()
        return {
            'btc': btc.get('results', [{}])[0] if btc and 'results' in btc else {},
            'eth': eth.get('results', [{}])[0] if eth and 'results' in eth else {},
            'account': acct or {},
            'holdings': holdings.get('results', []) if holdings else [],
            'connected': True,
        }
    except Exception as e:
        return {'connected': False, 'error': str(e)}


# ── Load Data ──
state = load_paper_state()
events = load_paper_trades()
live = get_robinhood_live()

entries = [e for e in events if e.get('event') == 'ENTRY']
exits = [e for e in events if e.get('event') == 'EXIT']

# ── Connection Status ──
connected = live.get('connected', False)
status_color = GREEN if connected else RED
status_text = "CONNECTED" if connected else "OFFLINE"

st.markdown(f"""
<div style="display:flex; align-items:center; gap:8px; margin-bottom:16px">
    <div style="width:10px; height:10px; border-radius:50%; background:{status_color}"></div>
    <span style="color:{status_color}; font-weight:600; font-size:0.85rem">Robinhood {status_text}</span>
    <span style="color:{MUTED}; font-size:0.75rem; margin-left:auto">Auto-refresh every 10s</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# ROW 1: Account + Paper Stats
# ══════════════════════════════════════════════════════════════════
acct = live.get('account', {})
buying_power = acct.get('buying_power', '0')
equity = state.get('equity', 0)
initial = state.get('initial_capital', 0)
stats = state.get('stats', {})
total_pnl = stats.get('total_pnl_usd', 0)
pnl_color = GREEN if total_pnl >= 0 else RED
wins = stats.get('wins', 0)
losses = stats.get('losses', 0)
total_trades = wins + losses
win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""<div class="pj-card" style="padding:14px; text-align:center">
        <div style="font-size:0.6rem; color:{MUTED}; text-transform:uppercase">Real Buying Power</div>
        <div style="font-size:1.2rem; font-weight:700; color:{WHITE}">${float(buying_power):,.2f}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""<div class="pj-card" style="padding:14px; text-align:center">
        <div style="font-size:0.6rem; color:{MUTED}; text-transform:uppercase">Paper Equity</div>
        <div style="font-size:1.2rem; font-weight:700; color:{WHITE}">${equity:,.2f}</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""<div class="pj-card" style="padding:14px; text-align:center">
        <div style="font-size:0.6rem; color:{MUTED}; text-transform:uppercase">Total P&L</div>
        <div style="font-size:1.2rem; font-weight:700; color:{pnl_color}">${total_pnl:+,.2f}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""<div class="pj-card" style="padding:14px; text-align:center">
        <div style="font-size:0.6rem; color:{MUTED}; text-transform:uppercase">Trades (W/L)</div>
        <div style="font-size:1.2rem; font-weight:700; color:{WHITE}">{wins}<span style="color:{GREEN}">W</span> / {losses}<span style="color:{RED}">L</span></div>
    </div>""", unsafe_allow_html=True)

with c5:
    wr_color = GREEN if win_rate > 50 else (AMBER if win_rate > 40 else RED) if total_trades > 0 else MUTED
    st.markdown(f"""<div class="pj-card" style="padding:14px; text-align:center">
        <div style="font-size:0.6rem; color:{MUTED}; text-transform:uppercase">Win Rate</div>
        <div style="font-size:1.2rem; font-weight:700; color:{wr_color}">{win_rate:.0f}%</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# ROW 2: Live Prices
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title" style="margin-top:20px">LIVE ROBINHOOD PRICES</div>', unsafe_allow_html=True)

pc1, pc2 = st.columns(2)

for col, asset, data_key in [(pc1, "BTC", "btc"), (pc2, "ETH", "eth")]:
    d = live.get(data_key, {})
    bid = float(d.get('bid_inclusive_of_sell_spread', 0))
    ask = float(d.get('ask_inclusive_of_buy_spread', 0))
    mid = float(d.get('price', 0))
    spread_pct = ((ask - bid) / mid * 100) if mid > 0 else 0
    ts = d.get('timestamp', '')[:19] if d.get('timestamp') else '---'
    spread_color = GREEN if spread_pct < 1.0 else (AMBER if spread_pct < 2.0 else RED)

    with col:
        st.markdown(f"""<div class="pj-card" style="padding:16px">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px">
                <span style="font-size:1rem; font-weight:700; color:{WHITE}">{asset}-USD</span>
                <span style="font-size:0.7rem; color:{MUTED}">{ts}</span>
            </div>
            <div style="font-size:1.5rem; font-weight:700; color:{CYAN}; margin-bottom:8px">${mid:,.2f}</div>
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px">
                <div>
                    <div style="font-size:0.55rem; color:{MUTED}; text-transform:uppercase">Bid</div>
                    <div style="font-size:0.85rem; color:{GREEN}">${bid:,.2f}</div>
                </div>
                <div>
                    <div style="font-size:0.55rem; color:{MUTED}; text-transform:uppercase">Ask</div>
                    <div style="font-size:0.85rem; color:{RED}">${ask:,.2f}</div>
                </div>
                <div>
                    <div style="font-size:0.55rem; color:{MUTED}; text-transform:uppercase">Spread</div>
                    <div style="font-size:0.85rem; color:{spread_color}">{spread_pct:.2f}%</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# ROW 3: Open Paper Positions
# ══════════════════════════════════════════════════════════════════
# Cross-check: remove positions whose last JSONL event is EXIT (handles stale state file)
_last_event = {}
for ev in events:
    _last_event[ev.get('asset', '')] = ev.get('event', '')
open_positions = {a: p for a, p in state.get('positions', {}).items() if _last_event.get(a) != 'EXIT'}
st.markdown(f'<div class="section-title" style="margin-top:20px">OPEN POSITIONS ({len(open_positions)})</div>', unsafe_allow_html=True)

if open_positions:
    for asset, pos in open_positions.items():
        direction = pos.get('direction', '?')
        entry_p = pos.get('entry_price', 0)
        current_p = pos.get('current_price', 0)
        pnl_pct = pos.get('current_pnl_pct', 0)
        pnl_usd = pos.get('current_pnl_usd', 0)
        bars = pos.get('bars_held', 0)
        sl = pos.get('sl_price', 0)
        entry_spread = pos.get('entry_spread_pct', 0)
        qty = pos.get('quantity', 0)
        score = pos.get('score', 0)
        conf = pos.get('llm_confidence', 0)
        entry_time = _to_local(pos.get('entry_time', ''))
        p_color = GREEN if pnl_pct >= 0 else RED
        dir_color = GREEN if direction == "LONG" else RED

        st.markdown(f"""<div class="pj-card" style="padding:14px; margin-bottom:8px; border-left: 3px solid {dir_color}">
            <div style="display:flex; justify-content:space-between; align-items:center">
                <div>
                    <span style="font-size:1rem; font-weight:700; color:{dir_color}">{direction}</span>
                    <span style="font-size:1rem; font-weight:600; color:{WHITE}; margin-left:8px">{asset}/USD</span>
                    <span style="font-size:0.75rem; color:{MUTED}; margin-left:12px">{bars} bars | Score: {score} | Conf: {conf:.0%}</span>
                </div>
                <div style="text-align:right">
                    <div style="font-size:1.1rem; font-weight:700; color:{p_color}">{pnl_pct:+.2f}%</div>
                    <div style="font-size:0.8rem; color:{p_color}">${pnl_usd:+,.2f}</div>
                </div>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr 1fr; gap:8px; margin-top:10px">
                <div>
                    <div style="font-size:0.55rem; color:{MUTED}">Entry Price</div>
                    <div style="font-size:0.8rem; color:{WHITE}">${entry_p:,.2f}</div>
                </div>
                <div>
                    <div style="font-size:0.55rem; color:{MUTED}">Current Price</div>
                    <div style="font-size:0.8rem; color:{WHITE}">${current_p:,.2f}</div>
                </div>
                <div>
                    <div style="font-size:0.55rem; color:{MUTED}">Quantity</div>
                    <div style="font-size:0.8rem; color:{WHITE}">{qty:.6f}</div>
                </div>
                <div>
                    <div style="font-size:0.55rem; color:{MUTED}">Stop Loss</div>
                    <div style="font-size:0.8rem; color:{RED}">${sl:,.2f}</div>
                </div>
                <div>
                    <div style="font-size:0.55rem; color:{MUTED}">Entry Time</div>
                    <div style="font-size:0.8rem; color:{MUTED}">{entry_time}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""<div class="pj-card" style="padding:20px; text-align:center; color:{MUTED}">
        No open positions — waiting for signal
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# ROW 4: Trade Log (entries + exits combined, like real exchange)
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title" style="margin-top:20px">TRADE LOG</div>', unsafe_allow_html=True)

if events:
    # Build table — show ALL events (entries and exits) like a real exchange log
    rows_html = ""
    running_pnl = 0.0

    for ev in reversed(events[-50:]):
        event_type = ev.get('event', '?')
        asset = ev.get('asset', '?')
        direction = ev.get('direction', '?')
        ts = _to_local(ev.get('timestamp', ''))

        if event_type == 'ENTRY':
            fill_price = ev.get('fill_price', 0)
            qty = ev.get('quantity', 0)
            spread = ev.get('spread_pct', 0)
            score = ev.get('score', 0)
            conf = ev.get('llm_confidence', 0)
            sl = ev.get('sl', 0)
            dir_color = GREEN if direction == "LONG" else RED

            rows_html += f"""
            <tr style="border-bottom:1px solid #1e2330">
                <td style="padding:8px; font-size:0.75rem; color:{MUTED}">{ts}</td>
                <td style="padding:8px; font-size:0.75rem; color:{WHITE}; font-weight:600">{asset}/USD</td>
                <td style="padding:8px; font-size:0.75rem; color:{CYAN}; font-weight:600">OPEN</td>
                <td style="padding:8px; font-size:0.75rem; color:{dir_color}; font-weight:600">{direction}</td>
                <td style="padding:8px; font-size:0.75rem; color:{WHITE}">${fill_price:,.2f}</td>
                <td style="padding:8px; font-size:0.75rem; color:{WHITE}">—</td>
                <td style="padding:8px; font-size:0.75rem; color:{WHITE}">{qty:.6f}</td>
                <td style="padding:8px; font-size:0.75rem; color:{MUTED}">—</td>
                <td style="padding:8px; font-size:0.75rem; color:{MUTED}">—</td>
                <td style="padding:8px; font-size:0.75rem; color:{AMBER}">{spread:.2f}%</td>
                <td style="padding:8px; font-size:0.7rem; color:{MUTED}">S:{score} C:{conf:.0%} SL:${sl:,.0f}</td>
            </tr>"""

        elif event_type == 'EXIT':
            entry_p = ev.get('entry_price', 0)
            exit_p = ev.get('exit_price', 0)
            pnl_pct = ev.get('pnl_pct', 0)
            pnl_usd = ev.get('pnl_usd', 0)
            reason = ev.get('reason', '')
            bars = ev.get('bars_held', 0)
            spread = ev.get('entry_spread_pct', 0)
            running_pnl += pnl_usd
            p_color = GREEN if pnl_usd >= 0 else RED
            dir_color = GREEN if direction == "LONG" else RED

            rows_html += f"""
            <tr style="border-bottom:1px solid #1e2330; background:rgba({'76,175,80' if pnl_usd >= 0 else '244,67,54'},0.05)">
                <td style="padding:8px; font-size:0.75rem; color:{MUTED}">{ts}</td>
                <td style="padding:8px; font-size:0.75rem; color:{WHITE}; font-weight:600">{asset}/USD</td>
                <td style="padding:8px; font-size:0.75rem; color:{AMBER}; font-weight:600">CLOSE</td>
                <td style="padding:8px; font-size:0.75rem; color:{dir_color}; font-weight:600">{direction}</td>
                <td style="padding:8px; font-size:0.75rem; color:{WHITE}">${entry_p:,.2f}</td>
                <td style="padding:8px; font-size:0.75rem; color:{WHITE}">${exit_p:,.2f}</td>
                <td style="padding:8px; font-size:0.75rem; color:{WHITE}">{bars}bars</td>
                <td style="padding:8px; font-size:0.75rem; color:{p_color}; font-weight:700">{pnl_pct:+.2f}%</td>
                <td style="padding:8px; font-size:0.75rem; color:{p_color}; font-weight:700">${pnl_usd:+,.2f}</td>
                <td style="padding:8px; font-size:0.75rem; color:{AMBER}">{spread:.2f}%</td>
                <td style="padding:8px; font-size:0.7rem; color:{MUTED}">{reason[:30]}</td>
            </tr>"""

    st.markdown(f"""
    <div style="overflow-x:auto">
    <table style="width:100%; border-collapse:collapse; background:#1a1e2e; border-radius:8px; overflow:hidden">
        <thead>
            <tr style="border-bottom:2px solid #2a2e3e">
                <th style="padding:8px; font-size:0.6rem; color:{MUTED}; text-transform:uppercase; text-align:left">Time</th>
                <th style="padding:8px; font-size:0.6rem; color:{MUTED}; text-transform:uppercase; text-align:left">Market</th>
                <th style="padding:8px; font-size:0.6rem; color:{MUTED}; text-transform:uppercase; text-align:left">Action</th>
                <th style="padding:8px; font-size:0.6rem; color:{MUTED}; text-transform:uppercase; text-align:left">Side</th>
                <th style="padding:8px; font-size:0.6rem; color:{MUTED}; text-transform:uppercase; text-align:left">Entry Price</th>
                <th style="padding:8px; font-size:0.6rem; color:{MUTED}; text-transform:uppercase; text-align:left">Exit Price</th>
                <th style="padding:8px; font-size:0.6rem; color:{MUTED}; text-transform:uppercase; text-align:left">Qty/Bars</th>
                <th style="padding:8px; font-size:0.6rem; color:{MUTED}; text-transform:uppercase; text-align:left">P&L %</th>
                <th style="padding:8px; font-size:0.6rem; color:{MUTED}; text-transform:uppercase; text-align:left">Realized P&L</th>
                <th style="padding:8px; font-size:0.6rem; color:{MUTED}; text-transform:uppercase; text-align:left">Spread</th>
                <th style="padding:8px; font-size:0.6rem; color:{MUTED}; text-transform:uppercase; text-align:left">Details</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    # Running P&L summary under table
    if exits:
        rpnl_color = GREEN if running_pnl >= 0 else RED
        st.markdown(f"""<div style="margin-top:8px; padding:10px; text-align:right; font-size:0.85rem">
            <span style="color:{MUTED}">Realized P&L:</span>
            <span style="color:{rpnl_color}; font-weight:700; margin-left:8px">${running_pnl:+,.2f}</span>
        </div>""", unsafe_allow_html=True)
else:
    st.info("No paper trades yet. Start the system with `python -m src.main`")

# ══════════════════════════════════════════════════════════════════
# ROW 5: Equity Curve
# ══════════════════════════════════════════════════════════════════
if exits:
    st.markdown('<div class="section-title" style="margin-top:20px">EQUITY CURVE</div>', unsafe_allow_html=True)

    import plotly.graph_objects as go

    eq_values = [initial if initial > 0 else 16445.79]
    running_eq = eq_values[0]

    for ex in exits:
        running_eq += ex.get('pnl_usd', 0)
        eq_values.append(running_eq)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(eq_values))),
        y=eq_values,
        mode='lines+markers',
        line=dict(color=CYAN, width=2),
        marker=dict(size=4),
        name='Equity',
        hovertext=[f"${v:,.2f}" for v in eq_values],
    ))

    for i, ex in enumerate(exits):
        pnl = ex.get('pnl_usd', 0)
        color = GREEN if pnl >= 0 else RED
        fig.add_trace(go.Scatter(
            x=[i + 1],
            y=[eq_values[i + 1]],
            mode='markers',
            marker=dict(color=color, size=8, symbol='circle'),
            showlegend=False,
            hovertext=f"{ex.get('asset', '?')} {ex.get('direction', '?')}: ${pnl:+,.2f} ({ex.get('reason', '')})",
        ))

    fig.update_layout(
        **plotly_layout(),
        height=300,
        xaxis_title="Trade #",
        yaxis_title="Equity ($)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Footer ──
total_signals = stats.get('total_signals', 0)
price_snapshots = stats.get('price_snapshots', 0)
st.markdown(f"""
<div style="margin-top:24px; padding:12px; border-top:1px solid #1e2330; display:flex; justify-content:space-between; font-size:0.7rem; color:{MUTED}">
    <span>Entries: {len(entries)} | Exits: {len(exits)} | Signals: {total_signals}</span>
    <span>Robinhood Crypto API v1 | Read-Only</span>
    <span>Last update: {_to_local(state.get('timestamp', ''))}</span>
</div>
""", unsafe_allow_html=True)
