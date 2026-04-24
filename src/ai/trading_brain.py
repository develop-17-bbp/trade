"""
Crypto Trading Brain v2.1 — Full-Potential Pattern-Finding Brain
================================================================

Major upgrade over v2.0:
- SPECIALIZED MODEL ROLES: Mistral = Pattern Scanner, Llama = Risk Analyst
- MULTI-PASS ANALYSIS: Pass 1 = scan patterns, Pass 2 = decide based on findings
- EXPLICIT PATTERN RECOGNITION: Candlestick + chart patterns identified by name
- WINNER DNA EXTRACTION: Analyzes L4+ winners to find common traits
- CROSS-ASSET INTELLIGENCE: BTC context fed when analyzing ETH and vice versa
- CONFIDENCE CALIBRATION: Tracks LLM accuracy, adjusts future confidence
- ORDER FLOW INTELLIGENCE: OB imbalance/walls fed to brain

Combines 10 capabilities:
1. Chain-of-Thought (CoT) Multi-Pass Reasoning
2. Trade Memory & Pattern Matching
3. Winner DNA Extractor (what do L4+ trades have in common?)
4. Regime-Specific Strategy Selector
5. Funding Rate Signal
6. Specialized Multi-Model Consensus (different roles per model)
7. Kelly Criterion Position Sizing
8. Session-Aware Trading Filter
9. Confidence Calibrator (track LLM accuracy, adjust scores)
10. Cross-Asset Intelligence (BTC<->ETH correlation)

Imported and used by executor.py — same interface as v2.0.
"""

import json
import math
import os
import re
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# ---------------------------------------------------------------------------
# 1. Pattern Scanner Prompt (Mistral's specialty)
# ---------------------------------------------------------------------------

def build_pattern_scanner_prompt(
    asset: str,
    signal: str,
    price: float,
    entry_score: int,
    slope_pct: float,
    min_trend_bars: int,
    ema_separation: float,
    ema_direction: str,
    htf_alignment: str,
    current_atr: float,
    current_ema: float,
    support: float,
    resistance: float,
    candle_lines: str,
    orch_result: dict,
    math_filter_warnings: list,
    score_reasons: list,
    htf_1h_direction: str,
    htf_4h_direction: str,
    is_reversal_signal: bool,
    memory_summary: str,
    regime_info: dict,
    funding_info: dict,
    session_info: dict,
    winner_dna: str,
    cross_asset_context: str,
    edge_stats: Optional[dict] = None,
    spread_round_trip: float = 0.0,
) -> str:
    """Build PATTERN SCANNER prompt — Mistral's job is to FIND patterns, not decide."""

    warnings_str = "; ".join(math_filter_warnings) if math_filter_warnings else "None"
    reasons_str = "; ".join(str(r) for r in score_reasons) if score_reasons else "None"
    regime_str = regime_info.get("regime", "UNKNOWN")
    funding_str = funding_info.get("signal", "NEUTRAL")
    funding_rate = funding_info.get("funding_rate", 0.0)

    direction_word = "CALL/LONG" if signal in ("BUY", "LONG") else "PUT/SHORT"
    signal_word = "CALL" if signal in ("BUY", "LONG") else "PUT"

    agent_lines = orch_result.get('agent_summary', []) if isinstance(orch_result, dict) else []
    agent_data = chr(10).join(agent_lines[:8]) if agent_lines else "No agent data"

    edge_str = ""
    if edge_stats:
        w = edge_stats.get('wins', 0)
        l = edge_stats.get('losses', 0)
        wr = edge_stats.get('win_rate', 0.5)
        edge_str = f"Track Record: {w}W/{l}L ({wr:.0%} win rate)"

    prompt = f"""You are a PATTERN RECOGNITION SPECIALIST for {asset}/USDT 5-minute perpetual futures.

YOUR ONLY JOB: Scan the candle data below and identify ALL patterns you see. Do NOT make a trade decision.
You are the EYES of the system — find every pattern, formation, and anomaly.

STRATEGY CONTEXT (so you know what matters most):
Our system trades EMA(8) NEW LINES (72% WR, PF 1.19 in 6-month backtest).
The 28% losers share patterns: late entries, declining volume, reversal wicks, fighting macro trend.
PAY SPECIAL ATTENTION to these danger signals — they predict the hard stop deaths.
{"" if spread_round_trip < 1.0 else f"SPREAD COST: This exchange has {spread_round_trip:.1f}% round-trip spread. ONLY flag patterns suggesting {spread_round_trip*2:.0f}%+ moves. Small patterns (<{spread_round_trip:.0f}% expected move) are IRRELEVANT — they lose money after spread."}

═══════════════════════════════════════════════════════════
MARKET DATA
═══════════════════════════════════════════════════════════
Asset: {asset}/USDT | Price: ${price:,.2f} | EMA(8): ${current_ema:.2f} ({ema_direction})
Signal: {signal_word} ({direction_word}) | Entry Score: {entry_score}/10
EMA Slope: {slope_pct:+.3f}%/bar | Trend Bars: {min_trend_bars} | EMA Sep: {ema_separation:.2f}%
ATR(14): ${current_atr:.2f} | Support: ${support:.2f} | Resistance: ${resistance:.2f}
Timeframes: 5m={ema_direction}, 1h={htf_1h_direction}, 4h={htf_4h_direction} | Alignment: {htf_alignment}/3
Regime: {regime_str} | Funding: {funding_str} ({funding_rate:.4f}%)
Is Reversal: {is_reversal_signal}
{edge_str}

CANDLE DATA — ANALYZE ALL TIMEFRAMES:
(5-minute candles are your PRIMARY signal. 1m shows micro-structure. 15m/1h show bigger picture.)
{candle_lines}

QUANT AGENT DATA:
{agent_data}

{f'CROSS-ASSET CONTEXT: {cross_asset_context}' if cross_asset_context else ''}

═══════════════════════════════════════════════════════════
SCAN FOR THESE PATTERNS (report ALL you find):
═══════════════════════════════════════════════════════════

1. CANDLESTICK PATTERNS (last 5 candles):
   - Engulfing (bullish/bearish)? Hammer/Shooting star? Doji? Three soldiers/crows?
   - Pin bar (long wick rejection)? Inside bar (compression)? Marubozu (strong conviction)?
   - Morning/Evening star? Harami? Tweezer top/bottom?

2. PRICE ACTION PATTERNS (last 10-20 candles):
   - Double top/bottom forming? Head & shoulders? Inverse H&S?
   - Flag/pennant (continuation)? Wedge (rising/falling)?
   - Channel (ascending/descending/horizontal)?
   - Triangle (symmetrical/ascending/descending)?
   - Breakout or breakdown from consolidation?

3. VOLUME PATTERNS:
   - Volume climax (spike + reversal)? Volume dry-up (squeeze building)?
   - Volume confirming trend? Volume divergence (price up, volume down = weak)?
   - Accumulation (high volume at lows)? Distribution (high volume at highs)?

4. MOMENTUM PATTERNS:
   - Exhaustion (price pushing but candles getting smaller)?
   - Acceleration (candles getting bigger = momentum building)?
   - Divergence vs EMA (price making new high but EMA flattening)?
   - Squeeze (ATR contracting = big move coming)?

5. STRUCTURAL PATTERNS:
   - Clean impulse (5-wave move)? Corrective ABC?
   - Higher highs + higher lows (uptrend intact)?
   - Lower highs + lower lows (downtrend intact)?
   - Range-bound (equal highs and lows)?

6. MULTI-TIMEFRAME ALIGNMENT:
   - Do 1m candles show micro-momentum in trade direction? Or micro-reversal starting?
   - Does 15m structure confirm the 5m signal? (same trend, or diverging?)
   - Does 1h show the bigger trend? Is 5m signal WITH or AGAINST the 1h trend?
   - Are multiple timeframes forming the SAME pattern? (e.g., flag on 5m AND 15m = stronger)

7. KEY LEVEL INTERACTION:
   - Testing support from above? Testing resistance from below?
   - Breakout above resistance with volume? Breakdown below support?
   - False breakout (went above resistance then came back)?
   - Price at round number ($60K, $70K, $3000, etc.)?

8. INSTITUTIONAL LIQUIDITY ANALYSIS (if data provided below):
   - Fair Value Gaps (FVG): Is price approaching/inside an unfilled gap? (gaps act as magnets)
   - Order Blocks (OB): Is price near a bullish/bearish OB? (institutional buy/sell zones)
   - For LONG: bullish OB below = support, bearish OB above = resistance target
   - For SHORT: bearish OB above = resistance, bullish OB below = support target
   - FVG near price = high probability zone for reaction

9. MARKET STRUCTURE (if BOS/CHoCH data provided):
   - Break of Structure (BOS): Has price broken a key HH or LL? (trend continuation signal)
   - Change of Character (CHoCH): Has trend character shifted? (early reversal warning)
   - HH+HL sequence = bullish structure. LH+LL sequence = bearish structure.
   - Signal ALIGNED with structure = higher probability. AGAINST structure = risky.

10. PROFIT PROTECTION CONTEXT (if provided):
   - If system is in profit: higher bar needed to risk gains
   - Trade quality score: low quality = likely L1 exit
   - Win probability estimate: P(loss) > 35% = dangerous

SYSTEM CONTEXT (for reference — focus on PATTERNS above):
{warnings_str}

ENTRY QUALITY: {reasons_str}

{f'WINNER DNA (what L4+ trades looked like): {winner_dna}' if winner_dna else ''}

RESPOND WITH ONLY JSON — list every pattern you found:
{{"candlestick_patterns": ["<pattern1>", "<pattern2>"], "price_action_patterns": ["<pattern>"], "volume_pattern": "<confirming/diverging/climax/dry-up/neutral>", "momentum": "<accelerating/decelerating/exhausting/squeezing>", "structure": "<impulse/corrective/ranging/breakout>", "key_level_action": "<testing_support/testing_resistance/breakout/false_breakout/clear>", "institutional_zones": "<at_ob/near_fvg/clear/in_liquidity_void>", "market_structure_trend": "<bullish_HH_HL/bearish_LH_LL/transitioning/unclear>", "bos_choch": "<BOS_bullish/BOS_bearish/CHoCH_detected/none>", "pattern_bias": "<BULLISH/BEARISH/NEUTRAL>", "pattern_strength": <1-10>, "strongest_signal": "<the single most important pattern you found>", "danger_pattern": "<the most dangerous pattern that could kill this trade>", "resembles_winner": <true/false based on winner DNA similarity>}}"""

    return prompt


# ---------------------------------------------------------------------------
# 2. Risk Analyst Prompt (Llama's specialty)
# ---------------------------------------------------------------------------

def build_risk_analyst_prompt(
    asset: str,
    signal: str,
    price: float,
    entry_score: int,
    slope_pct: float,
    min_trend_bars: int,
    ema_separation: float,
    ema_direction: str,
    htf_alignment: str,
    current_atr: float,
    current_ema: float,
    support: float,
    resistance: float,
    candle_lines: str,
    htf_1h_direction: str,
    htf_4h_direction: str,
    is_reversal_signal: bool,
    memory_summary: str,
    regime_info: dict,
    funding_info: dict,
    session_info: dict,
    pattern_scan_result: dict,
    winner_dna: str,
    cross_asset_context: str,
    math_filter_warnings: list = None,
    score_reasons: list = None,
    edge_stats: Optional[dict] = None,
    spread_round_trip: float = 0.0,
) -> str:
    """Build RISK ANALYST + DECISION prompt — Llama's job is to decide using pattern data."""

    warnings_str = "; ".join(math_filter_warnings) if math_filter_warnings else "None"
    reasons_str = "; ".join(str(r) for r in score_reasons) if score_reasons else "None"
    regime_str = regime_info.get("regime", "UNKNOWN")
    funding_str = funding_info.get("signal", "NEUTRAL")
    funding_rate = funding_info.get("funding_rate", 0.0)
    session_ok = "YES" if session_info.get("allowed", True) else "REDUCED"
    session_mult = session_info.get("multiplier", 1.0)

    direction_word = "CALL/LONG" if signal in ("BUY", "LONG") else "PUT/SHORT"
    signal_word = "CALL" if signal in ("BUY", "LONG") else "PUT"

    edge_str = ""
    if edge_stats:
        w = edge_stats.get('wins', 0)
        l = edge_stats.get('losses', 0)
        wr = edge_stats.get('win_rate', 0.5)
        edge_str = f"Track Record: {w}W/{l}L ({wr:.0%} win rate)"

    # Format pattern scan results
    patterns = pattern_scan_result or {}
    candle_pats = ", ".join(patterns.get("candlestick_patterns", [])) or "None found"
    pa_pats = ", ".join(patterns.get("price_action_patterns", [])) or "None found"
    vol_pat = patterns.get("volume_pattern", "unknown")
    momentum = patterns.get("momentum", "unknown")
    structure = patterns.get("structure", "unknown")
    key_level = patterns.get("key_level_action", "unknown")
    pat_bias = patterns.get("pattern_bias", "NEUTRAL")
    pat_strength = patterns.get("pattern_strength", 5)
    strongest = patterns.get("strongest_signal", "None identified")
    danger = patterns.get("danger_pattern", "None identified")
    resembles_winner = patterns.get("resembles_winner", False)
    inst_zones = patterns.get("institutional_zones", "unknown")
    ms_trend = patterns.get("market_structure_trend", "unknown")
    bos_choch = patterns.get("bos_choch", "none")

    prompt = f"""You are the DECISION-MAKING RISK ANALYST for an automated {asset}/USDT trading system.

A Pattern Scanner (separate AI) has already analyzed the charts. Your job:
1. Use the pattern findings to assess if this trade has a CLEAR DIRECTIONAL EDGE
2. Cross-reference with our historical win/loss DNA
3. Make the FINAL go/no-go decision — but LEAN TOWARD ENTERING when patterns confirm direction

═══════════════════════════════════════════════════════════
PROVEN STRATEGY — EMA(8) TREND LINE (6-month backtest: 72% WR, PF 1.19)
═══════════════════════════════════════════════════════════
ENTRY: EMA(8) forms NEW LINE (direction changes after 3+ bars opposite). Price on correct side.
EXIT (priority order):
  1. Hard stop -2% (emergency)
  2. EMA new line exit — ONLY when in profit (100% WR in backtest)
  3. EMA line-following SL — tracks EMA with buffer after 5 min
  4. Ratchet: breakeven at 1.0%, lock profits from 1.5%+

KEY FACTS FROM 6-MONTH BACKTEST:
- 72% of trades are winners when entry score >= 7
- EMA exit on profitable trades = 100% WR. On losing trades = 18% WR (system handles this)
- SL exits (delayed activation) = 68-78% WR
- Hard stop losers (-2%) are the ONLY consistent losers = 28% of trades
- YOUR JOB: Confirm direction is correct and this isn't a hard-stop setup
- Hard-stop setups look like: late entry, declining volume, reversal wicks, fighting macro trend
- ENTER when patterns support direction. Missing a winner costs more than a small loss.

═══════════════════════════════════════════════════════════
PATTERN SCAN RESULTS (from Pattern Scanner AI)
═══════════════════════════════════════════════════════════
Candlestick Patterns: {candle_pats}
Price Action Patterns: {pa_pats}
Volume Pattern: {vol_pat}
Momentum: {momentum}
Structure: {structure}
Key Level Action: {key_level}
Overall Pattern Bias: {pat_bias} (strength {pat_strength}/10)
Strongest Signal: {strongest}
Danger Pattern: {danger}
Institutional Zones: {inst_zones}
Market Structure Trend: {ms_trend}
BOS/CHoCH Status: {bos_choch}
Resembles Past Winners: {"YES" if resembles_winner else "NO"}

═══════════════════════════════════════════════════════════
TRADE SETUP
═══════════════════════════════════════════════════════════
Signal: {signal_word} ({direction_word}) | Price: ${price:,.2f} | EMA(8): ${current_ema:.2f} ({ema_direction})
Entry Score: {entry_score}/10 | EMA Slope: {slope_pct:+.3f}%/bar | Trend Bars: {min_trend_bars}
EMA Separation: {ema_separation:.2f}% | Is Reversal: {is_reversal_signal}
ATR: ${current_atr:.2f} | Support: ${support:.2f} | Resistance: ${resistance:.2f}
Room to Target: {"~" + f"${resistance - price:.2f} ({((resistance - price) / price * 100):.1f}%)" if signal in ("BUY", "LONG") else "~" + f"${price - support:.2f} ({((price - support) / price * 100):.1f}%)"}

HIGHER TIMEFRAMES: 5m={ema_direction}, 1h={htf_1h_direction}, 4h={htf_4h_direction} | Alignment: {htf_alignment}/3
Regime: {regime_str} | Session: {session_ok} ({session_mult:.1f}x) | Funding: {funding_str} ({funding_rate:.4f}%)
{edge_str}

TRADE MEMORY: {memory_summary}
{f'CROSS-ASSET: {cross_asset_context}' if cross_asset_context else ''}
{f'WINNER DNA: {winner_dna}' if winner_dna else ''}

SYSTEM NOTES (context, not blockers — YOU make the final call):
{warnings_str}
Entry Quality: {reasons_str}
NOTE: These warnings are INFORMATIONAL. A warning does NOT mean "don't trade". Many profitable trades had warnings.

═══════════════════════════════════════════════════════════
YOUR ANALYSIS — Think step by step:
═══════════════════════════════════════════════════════════

STEP 1 — PATTERN ALIGNMENT:
Does the pattern scan SUPPORT this {signal_word} entry?
- Pattern bias ({pat_bias}) aligns with {signal_word}?
- Strongest signal ({strongest}) — does it confirm or contradict?
- Danger pattern ({danger}) — how serious is it?
- Does this setup resemble our historical winners?

STEP 2 — TREND CONTINUATION POTENTIAL:
- Is there room to run? Check distance to next support/resistance.
- Momentum is {momentum} — enough fuel for continuation?
- Volume is {vol_pat} — does it support the move?
{"" if spread_round_trip < 1.0 else f"""
STEP 2.5 — SPREAD ECONOMICS (THIS EXCHANGE HAS WIDE SPREADS):
- Round-trip spread cost: {spread_round_trip:.1f}% (paid on entry + exit)
- Breakeven requires {spread_round_trip:.1f}%+ move
- Good trade = {spread_round_trip * 1.5:.1f}%+ expected move (covers spread + profit)
- Scalps under {spread_round_trip * 0.5:.1f}% are not viable on this exchange
- Swing trades (multi-hour to multi-day) with trend alignment CAN overcome spread cost
"""}
STEP 3 — INSTITUTIONAL & STRUCTURAL ANALYSIS:
- Check FVG/Order Block data: is price in a liquidity void (easy movement) or at a reaction zone (reversal risk)?
- Check Market Structure: is BOS confirmed in our direction? Or is CHoCH warning of reversal?
- HH+HL sequence for longs = GOOD. LH+LL sequence for longs = FIGHTING STRUCTURE.
- Profit Protector says {warnings_str.split('PROFIT PROTECTOR')[1][:60] if 'PROFIT PROTECTOR' in warnings_str else 'no comment'} — weigh this.

STEP 4 — RISK ASSESSMENT:
- If patterns show exhaustion/divergence, this trade dies at L1
- If near key level with no breakout confirmation, this is a trap
- Higher TF misalignment = fighting the macro trend = L1 death
- Entry score {entry_score}/10: below 5 = likely L1-L2 exit
- If opposing Order Block is nearby, price may reverse before L4+

STEP 5 — FINAL DECISION:
{"" if spread_round_trip < 1.0 else f"- HIGH-SPREAD EXCHANGE: proceed=true if expected move > {spread_round_trip * 1.5:.0f}% with trend alignment. REJECT only clear counter-trend traps."}
- Patterns STRONG + structure aligned + multi-day momentum? proceed=true, confidence 0.80-0.95
- Patterns confirm direction but some risk? proceed=true, confidence 0.65-0.80
- Patterns CONTRADICT signal or exhaustion clear? proceed=false, high risk score
- No clear edge either way but trend present? proceed=true with lower confidence (0.55-0.65)
{"- IMPORTANT: On this exchange, prefer swing trades with trend alignment. Missing a trending move costs more than a small spread loss." if spread_round_trip > 1.0 else "- IMPORTANT: When in doubt and patterns lean toward the signal direction, ENTER. Missing a winner costs more than a small L1 loss."}

RESPOND WITH ONLY JSON:
{{"proceed": <true/false>, "confidence": <0.0-1.0>, "position_size_pct": <1-20>, "risk_score": <0-10>, "trade_quality": <0-10>, "predicted_l_level": "<L1/L2/L3/L4/L5+>", "pattern_alignment": "<CONFIRMS/CONTRADICTS/NEUTRAL>", "bull_case": "<specific pattern-backed reason for L4+>", "bear_case": "<specific pattern/risk that kills this trade>", "facilitator_verdict": "<one sentence: enter or reject and WHY>"}}"""

    return prompt


# ---------------------------------------------------------------------------
# 3. Trade Memory & Pattern Matching (enhanced)
# ---------------------------------------------------------------------------

class TradeMemory:
    """Stores and queries historical trade outcomes for pattern matching."""

    def __init__(self, journal_path: str = "logs/trading_journal.jsonl", exchange: str = ""):
        self.journal_path = journal_path
        self.exchange = exchange  # Filter trades by exchange (empty = all)
        self.trades: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        try:
            path = Path(self.journal_path)
            if not path.exists():
                print(f"[BRAIN:MEMORY] Journal not found at {self.journal_path}")
                return
            all_count = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("action", "").startswith("close_") and entry.get("pnl_usd") is not None:
                            all_count += 1
                            # Filter by exchange if set (old entries without tag are included for backward compat)
                            if self.exchange:
                                entry_ex = entry.get("exchange", "")
                                if entry_ex and entry_ex != self.exchange:
                                    continue
                            self.trades.append(entry)
                    except json.JSONDecodeError:
                        continue
            ex_tag = f" ({self.exchange})" if self.exchange else ""
            print(f"[BRAIN:MEMORY] Loaded {len(self.trades)}/{all_count} closed trades{ex_tag} from journal")
        except Exception as e:
            print(f"[BRAIN:MEMORY] Error loading journal: {e}")

    def reload(self):
        self.trades = []
        self._load()

    def query_similar(
        self, asset: str, direction: str, entry_score: float = 0,
        slope: float = 0, ema_separation: float = 0,
    ) -> str:
        try:
            similar = []
            for t in self.trades:
                t_asset = t.get("asset", "")
                t_action = t.get("action", "")
                t_dir = "long" if "long" in t_action else "short" if "short" in t_action else ""
                if t_asset != asset or t_dir != direction:
                    continue
                t_conf = t.get("confidence", 0.5)
                t_score_approx = t_conf * 10
                if entry_score > 0 and abs(t_score_approx - entry_score) > 2:
                    continue
                similar.append(t)
            similar = similar[-10:]
            if not similar:
                return "No similar past setups found."
            total = len(similar)
            pnls = [t.get("pnl_usd", 0) for t in similar]
            avg_pnl = sum(pnls) / total if total > 0 else 0
            l4_plus = sum(1 for t in similar if "L4" in t.get("sl_progression", "") or "L5" in t.get("sl_progression", ""))
            hard_stop = sum(1 for t in similar if t.get("pnl_usd", 0) <= -2)
            early_stop = total - l4_plus - hard_stop
            return (
                f"Last {total} similar setups: "
                f"{l4_plus} reached L4+, {hard_stop} hit hard stop, "
                f"{early_stop} stopped at L1-L2. Avg PnL: ${avg_pnl:.2f}"
            )
        except Exception as e:
            return f"Memory query error: {e}"

    def get_other_asset_context(self, current_asset: str) -> str:
        """Get recent performance of OTHER assets for cross-asset intelligence."""
        try:
            other_recent = []
            for t in self.trades[-30:]:
                if t.get("asset", "") != current_asset:
                    other_recent.append(t)
            if not other_recent:
                return ""
            # Summarize other asset performance
            by_asset = defaultdict(list)
            for t in other_recent[-10:]:
                by_asset[t.get("asset", "?")].append(t.get("pnl_usd", 0))
            parts = []
            for a, pnls in by_asset.items():
                avg = sum(pnls) / len(pnls) if pnls else 0
                wins = sum(1 for p in pnls if p > 0)
                parts.append(f"{a}: {wins}/{len(pnls)} wins, avg ${avg:.2f}")
            return "; ".join(parts)
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# 4. Winner DNA Extractor
# ---------------------------------------------------------------------------

class WinnerDNAExtractor:
    """Analyzes L4+ winning trades to find what they have in common.
    This DNA is fed to the LLM so it knows what a good setup looks like."""

    def __init__(self, trades: List[Dict[str, Any]]):
        self.trades = trades
        self._dna_cache: str = ""
        self._last_compute: float = 0

    def get_dna(self) -> str:
        """Get winner DNA string. Cached for 10 minutes."""
        now = time.time()
        if self._dna_cache and (now - self._last_compute) < 600:
            return self._dna_cache
        self._dna_cache = self._compute_dna()
        self._last_compute = now
        return self._dna_cache

    def _compute_dna(self) -> str:
        try:
            # Find L4+ winners
            winners = []
            losers = []
            for t in self.trades:
                sl = t.get("sl_progression", "")
                pnl = t.get("pnl_pct", t.get("pnl_usd", 0))
                if isinstance(pnl, str):
                    try:
                        pnl = float(pnl)
                    except:
                        pnl = 0

                # L4+ = good trade (SL progression reached L4 or higher, or pnl > 2%)
                is_winner = ("L4" in sl or "L5" in sl or "L6" in sl or
                             "L7" in sl or "L8" in sl or float(pnl) > 2.0)
                if is_winner:
                    winners.append(t)
                elif float(pnl) < -0.5:
                    losers.append(t)

            if len(winners) < 3:
                return "Not enough L4+ winners yet to extract DNA (need 3+)."

            # Analyze winners
            w_confs = [t.get("confidence", 0.5) for t in winners]
            w_durations = [t.get("duration_minutes", 0) for t in winners]
            w_pnls = [t.get("pnl_pct", 0) for t in winners]

            # Analyze losers for contrast
            l_confs = [t.get("confidence", 0.5) for t in losers] if losers else [0.5]
            l_durations = [t.get("duration_minutes", 0) for t in losers] if losers else [0]

            avg_w_conf = sum(w_confs) / len(w_confs) if w_confs else 0.5
            avg_w_dur = sum(w_durations) / len(w_durations) if w_durations else 0
            avg_w_pnl = sum(w_pnls) / len(w_pnls) if w_pnls else 0
            avg_l_conf = sum(l_confs) / len(l_confs) if l_confs else 0.5
            avg_l_dur = sum(l_durations) / len(l_durations) if l_durations else 0

            # Count direction bias
            long_wins = sum(1 for t in winners if "long" in t.get("action", "").lower())
            short_wins = len(winners) - long_wins

            # Count assets
            asset_wins = defaultdict(int)
            for t in winners:
                asset_wins[t.get("asset", "?")] += 1

            # SL progression analysis — how many levels did winners hit?
            sl_depths = []
            for t in winners:
                sl = t.get("sl_progression", "L1")
                levels = sl.split("->")
                sl_depths.append(len(levels))
            avg_sl_depth = sum(sl_depths) / len(sl_depths) if sl_depths else 1

            dna_parts = [
                f"L4+ WINNER DNA ({len(winners)} trades analyzed):",
                f"- Avg confidence at entry: {avg_w_conf:.2f} (losers: {avg_l_conf:.2f})",
                f"- Avg duration: {avg_w_dur:.0f}min (losers: {avg_l_dur:.0f}min)",
                f"- Avg profit: {avg_w_pnl:.2f}%",
                f"- Direction: {long_wins} LONG wins, {short_wins} SHORT wins",
                f"- Avg SL levels traversed: {avg_sl_depth:.1f}",
                f"- By asset: {', '.join(f'{a}={c}' for a, c in asset_wins.items())}",
            ]

            # Key insight: what separates winners from losers
            if avg_w_conf > avg_l_conf + 0.1:
                dna_parts.append(f"- KEY: Winners had HIGHER entry confidence ({avg_w_conf:.2f} vs {avg_l_conf:.2f})")
            if avg_w_dur > avg_l_dur * 1.5:
                dna_parts.append(f"- KEY: Winners held LONGER ({avg_w_dur:.0f}min vs {avg_l_dur:.0f}min)")

            return chr(10).join(dna_parts)
        except Exception as e:
            return f"Winner DNA computation error: {e}"


# ---------------------------------------------------------------------------
# 5. Confidence Calibrator
# ---------------------------------------------------------------------------

class ConfidenceCalibrator:
    """Tracks LLM prediction accuracy and adjusts future confidence scores.

    If the LLM says conf=0.80 but those trades win 40%, the calibrated
    confidence should be 0.40 — not the inflated LLM number.
    """

    def __init__(self, journal_path: str = "logs/trading_journal.jsonl", exchange: str = ""):
        self.exchange = exchange
        self.calibration_data: Dict[str, List] = {}  # bucket -> list of (predicted_conf, actual_win)
        self._bucket_size = 0.1  # Group confidences into 0.1 buckets
        self._load_calibration(journal_path)

    def _load_calibration(self, journal_path: str):
        """Build calibration from historical trades."""
        try:
            path = Path(journal_path)
            if not path.exists():
                return
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line)
                        if not t.get("action", "").startswith("close_"):
                            continue
                        # Filter by exchange (old entries without tag included for backward compat)
                        if self.exchange:
                            t_ex = t.get("exchange", "")
                            if t_ex and t_ex != self.exchange:
                                continue
                        conf = t.get("confidence", 0)
                        pnl = t.get("pnl_usd", 0)
                        if conf > 0:
                            bucket = f"{int(conf * 10) / 10:.1f}"
                            if bucket not in self.calibration_data:
                                self.calibration_data[bucket] = []
                            self.calibration_data[bucket].append({
                                "conf": conf,
                                "win": 1 if pnl > 0 else 0,
                                "pnl": pnl,
                            })
                    except (json.JSONDecodeError, ValueError):
                        continue

            total = sum(len(v) for v in self.calibration_data.values())
            if total > 0:
                print(f"[BRAIN:CALIBRATE] Loaded {total} trades for confidence calibration")
        except Exception as e:
            print(f"[BRAIN:CALIBRATE] Error loading: {e}")

    def calibrate(self, raw_confidence: float) -> float:
        """Adjust LLM confidence based on historical accuracy.

        If LLM says 0.8 confidence but 0.8-bucket trades only win 40%,
        return 0.40 instead.
        """
        try:
            bucket = f"{int(raw_confidence * 10) / 10:.1f}"
            data = self.calibration_data.get(bucket, [])

            if len(data) < 5:
                # Not enough data to calibrate — trust the LLM's judgment
                return raw_confidence  # No haircut — let LLM confidence through

            actual_win_rate = sum(d["win"] for d in data) / len(data)

            # Blend: 60% calibrated + 40% raw (don't fully override LLM)
            calibrated = actual_win_rate * 0.6 + raw_confidence * 0.4

            return round(max(0.0, min(1.0, calibrated)), 3)
        except Exception:
            return raw_confidence  # Trust LLM on calibration error

    def get_calibration_report(self) -> str:
        """Return a brief calibration summary."""
        try:
            parts = []
            for bucket in sorted(self.calibration_data.keys()):
                data = self.calibration_data[bucket]
                if len(data) >= 3:
                    wr = sum(d["win"] for d in data) / len(data)
                    parts.append(f"conf={bucket}: actual_win={wr:.0%} (n={len(data)})")
            return "; ".join(parts) if parts else "No calibration data yet"
        except Exception:
            return "Calibration error"


# ---------------------------------------------------------------------------
# 6. Regime-Specific Strategy Selector
# ---------------------------------------------------------------------------

class RegimeSelector:
    """Determines market regime and filters allowed trade types."""

    def evaluate(self, closes=None, volumes=None, orch_result=None):
        regime = "UNKNOWN"
        if orch_result and isinstance(orch_result, dict):
            regime_raw = str(orch_result.get("regime", "")).upper()
            if regime_raw in ("TRENDING", "RANGING", "VOLATILE", "CHOPPY"):
                regime = regime_raw
        if regime == "UNKNOWN" and closes and len(closes) >= 10:
            regime = self._compute_regime(closes, volumes)
        return self._build_result(regime)

    def _compute_regime(self, closes, volumes=None):
        try:
            returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
            if not returns:
                return "UNKNOWN"
            avg_ret = sum(returns) / len(returns)
            volatility = (sum((r - avg_ret)**2 for r in returns) / len(returns)) ** 0.5
            pos_count = sum(1 for r in returns if r > 0)
            consistency = pos_count / len(returns)

            # Enhanced: also check ATR expansion/contraction
            if volatility > 0.005:
                if consistency > 0.65 or consistency < 0.35:
                    return "VOLATILE"
                return "CHOPPY"
            elif consistency > 0.6 or consistency < 0.4:
                return "TRENDING"
            else:
                return "RANGING"
        except Exception:
            return "UNKNOWN"

    def _build_result(self, regime):
        configs = {
            "TRENDING": {"regime": "TRENDING", "allowed_types": ["trend_following"], "size_multiplier": 1.0, "min_score": 5},
            "RANGING": {"regime": "RANGING", "allowed_types": ["mean_reversion", "reversal"], "size_multiplier": 1.0, "min_score": 5},
            "VOLATILE": {"regime": "VOLATILE", "allowed_types": ["trend_following", "mean_reversion", "reversal"], "size_multiplier": 0.5, "min_score": 7},
            "CHOPPY": {"regime": "CHOPPY", "allowed_types": [], "size_multiplier": 0.0, "min_score": 10},
        }
        return configs.get(regime, {"regime": "UNKNOWN", "allowed_types": ["trend_following", "mean_reversion", "reversal"], "size_multiplier": 0.8, "min_score": 6})


# ---------------------------------------------------------------------------
# 7. Funding Rate Signal
# ---------------------------------------------------------------------------

class FundingRateSignal:
    def get_signal(self, asset, exchange_client=None):
        default = {"funding_rate": 0.0, "signal": "NEUTRAL", "strength": 0.0}
        try:
            funding_rate = 0.0
            if exchange_client is not None:
                funding_rate = self._fetch(asset, exchange_client)
            if funding_rate == 0.0:
                return default
            signal = "NEUTRAL"
            strength = 0.0
            if funding_rate > 0.0001:
                signal = "CONTRARIAN_SHORT"
                strength = min(1.0, abs(funding_rate) / 0.001)
            elif funding_rate < -0.0001:
                signal = "CONTRARIAN_LONG"
                strength = min(1.0, abs(funding_rate) / 0.001)
            return {"funding_rate": funding_rate * 100, "signal": signal, "strength": round(strength, 3)}
        except Exception as e:
            return default

    def _fetch(self, asset, client):
        try:
            if hasattr(client, "get_funding_rate"):
                result = client.get_funding_rate(asset)
                if isinstance(result, (int, float)):
                    return float(result)
                if isinstance(result, dict):
                    return float(result.get("funding_rate", result.get("rate", 0)))
            if hasattr(client, "funding_rate"):
                fr = client.funding_rate
                if callable(fr):
                    return float(fr(asset))
                if isinstance(fr, dict):
                    return float(fr.get(asset, 0))
        except Exception:
            pass
        return 0.0


# ---------------------------------------------------------------------------
# 8. Specialized Multi-Model Consensus (Different Roles!)
# ---------------------------------------------------------------------------

class MultiModelConsensus:
    """Queries models with SPECIALIZED ROLES:
    - Mistral = Pattern Scanner (finds patterns, doesn't decide)
    - Llama = Risk Analyst (uses pattern data to make final decision)

    This is a TWO-PASS system:
    Pass 1: Scanner (right brain) scans for patterns (what do you SEE?)
    Pass 2: Analyst (left brain) decides using patterns + risk analysis (should we TRADE?)
    """

    # Fine-tuned models with trading-specific system prompts, temperature, and output format
    # Deployed via Ollama custom Modelfiles on GPU server.
    # Fallback chains ordered most-specialized → most-generic; each tier
    # is only used if Ollama doesn't have the previous one pulled.
    MODEL_SCANNER = "act-scanner"        # QLoRA fine-tuned scanner (falls back to nexus-scanner → devstral → deepseek-r1:7b)
    MODEL_ANALYST = "act-analyst"        # QLoRA fine-tuned analyst (falls back to nexus-analyst → qwen3:32b → deepseek-r1:32b)
    MODEL_SCANNER_FALLBACKS = ["nexus-scanner", "devstral:24b", "deepseek-r1:7b"]
    MODEL_ANALYST_FALLBACKS = ["nexus-analyst", "qwen3:32b", "deepseek-r1:32b"]

    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_base_url = ollama_base_url.rstrip("/")
        # Cache of installed Ollama models — refreshed via /api/tags on demand.
        # Using a 5-minute TTL so a mid-cycle `ollama pull` is picked up quickly
        # without hammering the tags endpoint every call.
        self._available_models: set = set()
        self._available_models_ts: float = 0.0
        self._AVAIL_TTL = 300

    def _get_available_models(self) -> set:
        """Return the set of model names Ollama currently has pulled.

        Cached for 5 min. On failure (Ollama down), returns an empty set AND
        sets the timestamp so callers fall through to try all models rather
        than skipping everything. Result: cold-start still attempts every
        model in the fallback chain instead of going silent.
        """
        import time as _t
        now = _t.time()
        if now - self._available_models_ts < self._AVAIL_TTL and self._available_models:
            return self._available_models
        try:
            resp = requests.get(f"{self.ollama_base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                tags = resp.json().get("models", []) or []
                self._available_models = {m.get("name", "") for m in tags if m.get("name")}
                self._available_models_ts = now
        except Exception:
            # Leave cache untouched; next call retries
            pass
        return self._available_models

    def _filter_available(self, chain: List[str]) -> List[str]:
        """Drop model names Ollama doesn't have. If the tags endpoint is
        unreachable we get an empty set back and fall through unchanged —
        caller sees every model try once (and fail loudly) rather than
        silently skipping all models."""
        avail = self._get_available_models()
        if not avail:
            return chain                                  # unknown — try all
        return [m for m in chain if m in avail] or chain  # if nothing matches, still try

    def query_two_pass(self, scanner_prompt: str, analyst_prompt_builder, **analyst_kwargs) -> Dict[str, Any]:
        """
        Two-pass evaluation:
        1. Mistral scans patterns (Pass 1)
        2. Llama analyzes risk + decides using Mistral's findings (Pass 2)
        """
        # PASS 1: Pattern scan with Mistral (try fine-tuned, fall back through chain)
        print(f"  [BRAIN:PASS1] Mistral scanning patterns...")
        pattern_result = None
        scanner_chain = self._filter_available([self.MODEL_SCANNER] + self.MODEL_SCANNER_FALLBACKS)
        for _model in scanner_chain:
            try:
                pattern_result = self._call_ollama(_model, scanner_prompt)
                break
            except Exception as _e:
                print(f"  [BRAIN:PASS1] {_model} failed: {str(_e)[:60]}")
                continue
        if pattern_result is None:
            print(f"  [BRAIN:PASS1] All scanner models failed — using empty patterns")
            pattern_result = self._empty_pattern_result()

        # Validate pattern result
        pattern_result = self._validate_patterns(pattern_result)
        pat_bias = pattern_result.get("pattern_bias", "NEUTRAL")
        pat_strength = pattern_result.get("pattern_strength", 5)
        strongest = pattern_result.get("strongest_signal", "None")
        print(f"  [BRAIN:PASS1] Patterns: bias={pat_bias} strength={pat_strength}/10 key={strongest}")

        # PASS 2: Risk analysis + decision with Llama (try fine-tuned, fall back through chain)
        print(f"  [BRAIN:PASS2] Llama analyzing risk + deciding...")
        analyst_prompt = analyst_prompt_builder(pattern_scan_result=pattern_result, **analyst_kwargs)
        decision_result = None
        analyst_chain = self._filter_available([self.MODEL_ANALYST] + self.MODEL_ANALYST_FALLBACKS)
        for _model in analyst_chain:
            try:
                decision_result = self._call_ollama(_model, analyst_prompt)
                break
            except Exception as _e:
                print(f"  [BRAIN:PASS2] {_model} failed: {str(_e)[:60]}")
                continue
        if decision_result is None:
            print(f"  [BRAIN:PASS2] All analyst models failed — rejecting trade (safe default)")
            decision_result = {
                "proceed": False, "confidence": 0.2, "risk_score": 8,
                "trade_quality": 2, "predicted_l_level": "L1",
                "bull_case": "Analyst unavailable", "bear_case": "all models failed",
                "facilitator_verdict": "REJECTED — analyst error",
            }

        # PASS 3 (optional): If Llama says YES, quick Mistral verification
        # "Does the pattern data actually support this decision?"
        if decision_result.get("proceed", False):
            verification = self._quick_verify(pattern_result, decision_result)
            if not verification["confirmed"]:
                print(f"  [BRAIN:VERIFY] Pattern contradiction detected: {verification['reason']}")
                decision_result["confidence"] = decision_result.get("confidence", 0.5) * 0.7
                if decision_result["confidence"] < 0.45:
                    decision_result["proceed"] = False
                    decision_result["facilitator_verdict"] = f"REJECTED — pattern contradiction: {verification['reason']}"

        # ── Capture for fine-tuning data collection ──
        self._last_scanner_prompt = scanner_prompt
        self._last_scanner_output = pattern_result
        self._last_analyst_prompt = analyst_prompt
        self._last_analyst_output = decision_result

        # Merge pattern + decision results
        return self._merge_results(pattern_result, decision_result)

    def _quick_verify(self, patterns: dict, decision: dict) -> dict:
        """Quick sanity check: do patterns actually support the decision?"""
        try:
            pat_bias = patterns.get("pattern_bias", "NEUTRAL").upper()
            proceed = decision.get("proceed", False)
            conf = decision.get("confidence", 0.5)

            # If patterns say BEARISH but decision says proceed with HIGH confidence on a LONG → contradiction
            # We can't check direction directly here, but we can check for mismatches
            danger = patterns.get("danger_pattern", "").lower()

            # If danger pattern mentions reversal/exhaustion and confidence is high
            if danger and any(w in danger for w in ["reversal", "exhaustion", "divergence", "trap"]):
                if conf > 0.7:
                    return {"confirmed": False, "reason": f"High confidence ({conf:.2f}) despite danger: {danger}"}

            # Pattern strength < 3 but confidence > 0.8 = suspicious
            pat_strength = patterns.get("pattern_strength", 5)
            if pat_strength < 3 and conf > 0.75:
                return {"confirmed": False, "reason": f"Weak patterns (str={pat_strength}) but overconfident ({conf:.2f})"}

            return {"confirmed": True, "reason": ""}
        except Exception:
            return {"confirmed": True, "reason": ""}

    def _call_ollama(self, model: str, prompt: str) -> Dict[str, Any]:
        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 1500},
        }
        resp = requests.post(url, json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data.get("response", "")
        return self._parse_llm_json(raw_text)

    def _parse_llm_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        # Remove markdown fences
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()

        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Find JSON in text — try largest block first
        depth = 0
        start = -1
        best_json = None
        for i, c in enumerate(text):
            if c == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = text[start:i+1]
                    try:
                        parsed = json.loads(candidate)
                        if best_json is None or len(candidate) > len(json.dumps(best_json)):
                            best_json = parsed
                    except json.JSONDecodeError:
                        # Try fixing trailing commas
                        fixed = re.sub(r',\s*}', '}', candidate)
                        fixed = re.sub(r',\s*]', ']', fixed)
                        try:
                            best_json = json.loads(fixed)
                        except json.JSONDecodeError:
                            pass

        if best_json:
            return best_json

        return {"proceed": False, "confidence": 0.2, "risk_score": 8,
                "parse_error": True, "facilitator_verdict": "REJECTED — unparseable LLM response"}

    def _empty_pattern_result(self):
        return {
            "candlestick_patterns": [], "price_action_patterns": [],
            "volume_pattern": "unknown", "momentum": "unknown",
            "structure": "unknown", "key_level_action": "unknown",
            "pattern_bias": "NEUTRAL", "pattern_strength": 3,
            "strongest_signal": "None (scanner failed)",
            "danger_pattern": "Unknown (scanner failed)",
            "resembles_winner": False,
        }

    def _validate_patterns(self, result: dict) -> dict:
        """Ensure pattern result has all expected fields."""
        defaults = self._empty_pattern_result()
        for key, default in defaults.items():
            if key not in result:
                result[key] = default
        # Clamp pattern_strength
        try:
            result["pattern_strength"] = max(1, min(10, int(result["pattern_strength"])))
        except (ValueError, TypeError):
            result["pattern_strength"] = 5
        return result

    def _merge_results(self, patterns: dict, decision: dict) -> Dict[str, Any]:
        """Merge pattern scan + analyst decision into final result."""
        return {
            "proceed": bool(decision.get("proceed", False)),
            "confidence": round(float(decision.get("confidence", 0.0)), 3),
            "risk_score": int(decision.get("risk_score", 7)),
            "trade_quality": int(decision.get("trade_quality", 5)),
            "predicted_l_level": str(decision.get("predicted_l_level", "L1")),
            "position_size_pct": float(decision.get("position_size_pct", 50)),
            "bull_case": str(decision.get("bull_case", "")),
            "bear_case": str(decision.get("bear_case", "")),
            "facilitator_verdict": str(decision.get("facilitator_verdict", "")),
            "pattern_alignment": str(decision.get("pattern_alignment", "NEUTRAL")),
            # Pattern details for logging
            "patterns": {
                "candlestick": patterns.get("candlestick_patterns", []),
                "price_action": patterns.get("price_action_patterns", []),
                "volume": patterns.get("volume_pattern", "unknown"),
                "momentum": patterns.get("momentum", "unknown"),
                "structure": patterns.get("structure", "unknown"),
                "key_level": patterns.get("key_level_action", "unknown"),
                "bias": patterns.get("pattern_bias", "NEUTRAL"),
                "strength": patterns.get("pattern_strength", 5),
                "strongest": patterns.get("strongest_signal", "None"),
                "danger": patterns.get("danger_pattern", "None"),
                "resembles_winner": patterns.get("resembles_winner", False),
            },
            "model_votes": {
                "mistral_scanner": {
                    "role": "pattern_scanner",
                    "bias": patterns.get("pattern_bias", "NEUTRAL"),
                    "strength": patterns.get("pattern_strength", 5),
                },
                "llama_analyst": {
                    "role": "risk_analyst",
                    "proceed": decision.get("proceed", False),
                    "confidence": decision.get("confidence", 0.0),
                    "risk_score": decision.get("risk_score", 7),
                },
            },
        }


# ---------------------------------------------------------------------------
# 9. Kelly Criterion Position Sizing
# ---------------------------------------------------------------------------

class KellySizer:
    MIN_TRADES_FOR_KELLY = 20

    def compute_from_history(self, pnl_history, base_size_pct, equity, max_size_pct=100.0):
        try:
            if not pnl_history or len(pnl_history) < self.MIN_TRADES_FOR_KELLY:
                return base_size_pct
            wins = [p for p in pnl_history if p > 0]
            losses = [abs(p) for p in pnl_history if p < 0]
            if not wins or not losses:
                return base_size_pct
            win_rate = len(wins) / len(pnl_history)
            avg_win = sum(wins) / len(wins)
            avg_loss = sum(losses) / len(losses)
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
            kelly = win_rate - (1 - win_rate) / win_loss_ratio
            half_kelly = kelly / 2.0
            size_pct = half_kelly * 100.0
            return round(max(1.0, min(max_size_pct, size_pct)), 2)
        except Exception:
            return base_size_pct


# ---------------------------------------------------------------------------
# 10. Session-Aware Trading Filter
# ---------------------------------------------------------------------------

class SessionFilter:
    def is_good_session(self):
        try:
            now = datetime.now(timezone.utc)
            hour = now.hour
            weekday = now.weekday()
            # Weekends: crypto volume drops 40-60% — skip new entries
            if weekday >= 5:
                return {"allowed": False, "multiplier": 0.0, "reason": f"Weekend — no new entries (day={weekday})"}
            # Dead hours UTC 00:00-07:00: low volume, wide spreads, stop hunts
            if 0 <= hour < 7:
                return {"allowed": False, "multiplier": 0.0, "reason": f"Dead hours — no new entries (UTC {hour}:00)"}
            # Late US / Asia overlap UTC 20:00-24:00: reduced quality
            if 20 <= hour < 24:
                return {"allowed": True, "multiplier": 0.6, "reason": f"Reduced session (UTC {hour}:00)"}
            # Prime US session UTC 13:00-20:00: highest volume and trend clarity
            if 13 <= hour < 20:
                return {"allowed": True, "multiplier": 1.3, "reason": f"Prime US session (UTC {hour}:00)"}
            # EU session UTC 07:00-13:00: good volume
            return {"allowed": True, "multiplier": 1.0, "reason": f"EU session (UTC {hour}:00)"}
        except Exception:
            return {"allowed": True, "multiplier": 1.0, "reason": "Error"}


# ---------------------------------------------------------------------------
# Main Orchestrator: TradingBrainV2
# ---------------------------------------------------------------------------

class TradingBrainV2:
    """
    v2.1 — Full-potential pattern-finding brain.

    TWO-PASS LLM evaluation:
      Pass 1 (Mistral): "What patterns do you SEE?" → structured pattern scan
      Pass 2 (Llama):   "Given these patterns, should we TRADE?" → risk-aware decision
      Verify:           If Llama says YES, check patterns don't contradict

    Plus: Winner DNA extraction, confidence calibration, cross-asset intel.
    Same interface as v2.0 — drop-in replacement.
    """

    def __init__(self, ollama_base_url="http://localhost:11434", journal_path="logs/trading_journal.jsonl", exchange: str = "", config: dict = None,
                 economic_intelligence=None, llm_memory=None, finetune_enricher=None):
        self.config = config or {}
        self.exchange = exchange  # Exchange tag for filtering (e.g., 'bybit', 'delta')
        self.memory = TradeMemory(journal_path, exchange=exchange)
        self.regime = RegimeSelector()
        self.funding = FundingRateSignal()
        self.consensus = MultiModelConsensus(ollama_base_url)
        self.kelly = KellySizer()
        self.session = SessionFilter()
        self.calibrator = ConfidenceCalibrator(journal_path, exchange=exchange)
        self.winner_dna = WinnerDNAExtractor(self.memory.trades)

        # v8.0: Dynamic intelligence layers
        self._economic_intelligence = economic_intelligence
        self._llm_memory = llm_memory
        self._finetune_enricher = finetune_enricher

        cal_report = self.calibrator.get_calibration_report()
        print(f"[BRAIN] TradingBrainV2.1 ACTIVE — 2-pass (Mistral=scanner + Llama=analyst)")
        print(f"[BRAIN] Confidence calibration: {cal_report}")
        print(f"[BRAIN] Winner DNA: {self.winner_dna.get_dna()[:120]}")
        if self._economic_intelligence:
            print(f"[BRAIN] v8.0: EconomicIntelligence connected (12 macro layers)")
        if self._llm_memory:
            print(f"[BRAIN] v8.0: LLM Memory connected (dynamic few-shot prompts)")
        if self._finetune_enricher:
            print(f"[BRAIN] v8.0: FinetuneEnricher connected (enriched training data)")

    def evaluate_trade(
        self, asset, signal, price, entry_score, slope_pct, min_trend_bars,
        ema_separation, ema_direction, htf_alignment, closes, volumes,
        current_atr, current_ema, support, resistance, candle_lines,
        orch_result, pnl_history, math_filter_warnings, score_reasons,
        htf_1h_direction, htf_4h_direction, is_reversal_signal, equity,
        exchange_client=None, edge_stats=None,
    ) -> Dict[str, Any]:
        try:
            return self._evaluate_internal(
                asset=asset, signal=signal, price=price, entry_score=entry_score,
                slope_pct=slope_pct, min_trend_bars=min_trend_bars,
                ema_separation=ema_separation, ema_direction=ema_direction,
                htf_alignment=htf_alignment, closes=closes, volumes=volumes,
                current_atr=current_atr, current_ema=current_ema,
                support=support, resistance=resistance, candle_lines=candle_lines,
                orch_result=orch_result, pnl_history=pnl_history,
                math_filter_warnings=math_filter_warnings, score_reasons=score_reasons,
                htf_1h_direction=htf_1h_direction, htf_4h_direction=htf_4h_direction,
                is_reversal_signal=is_reversal_signal, equity=equity,
                exchange_client=exchange_client, edge_stats=edge_stats,
            )
        except Exception as e:
            print(f"[BRAIN:{asset}] CRITICAL ERROR: {e}")
            return self._safe_default(asset, str(e))

    def _evaluate_internal(
        self, asset, signal, price, entry_score, slope_pct, min_trend_bars,
        ema_separation, ema_direction, htf_alignment, closes, volumes,
        current_atr, current_ema, support, resistance, candle_lines,
        orch_result, pnl_history, math_filter_warnings, score_reasons,
        htf_1h_direction, htf_4h_direction, is_reversal_signal, equity,
        exchange_client, edge_stats,
    ) -> Dict[str, Any]:

        direction = "long" if "long" in signal.lower() or "buy" in signal.lower() else "short"

        # --- Step 1: Pre-LLM filters (fast, no API calls) ---
        session_info = self.session.is_good_session()
        session_mult = session_info.get("multiplier", 1.0)
        print(f"[BRAIN:{asset}] Session: {session_info['reason']} (mult={session_mult})")
        # Hard block during dead/weekend hours — no new entries
        if not session_info.get("allowed", True):
            print(f"[BRAIN:{asset}] SESSION BLOCK: {session_info['reason']} — skipping entry")
            return {
                "proceed": False, "confidence": 0.0, "risk_score": 10,
                "trade_quality": 0, "position_size_pct": 0,
                "facilitator_verdict": f"SESSION BLOCK: {session_info['reason']}",
                "session_blocked": True,
            }

        regime_info = self.regime.evaluate(closes=closes, volumes=volumes, orch_result=orch_result)
        regime_name = regime_info.get("regime", "UNKNOWN")
        print(f"[BRAIN:{asset}] Regime: {regime_name}")

        # ── Regime context for LLM (no hard blocks — LLM sees everything and decides) ──
        regime_allowed = regime_info.get("allowed_types", [])
        trade_type = "reversal" if is_reversal_signal else "trend_following"
        if slope_pct is not None and abs(slope_pct) < 0.1 and not is_reversal_signal:
            trade_type = "mean_reversion"

        # Add regime context (informational, not fear-based)
        if regime_name == "CHOPPY":
            math_filter_warnings.append(f"REGIME: CHOPPY — tighter SL may be needed")
        if regime_allowed and trade_type not in regime_allowed:
            math_filter_warnings.append(f"REGIME NOTE: {trade_type} in {regime_name} market")
        print(f"[BRAIN:{asset}] Regime context: {regime_name} | trade_type={trade_type} | warnings fed to LLM")

        # --- Step 2: Gather intelligence (parallel where possible) ---
        funding_info = self.funding.get_signal(asset, exchange_client)
        memory_summary = self.memory.query_similar(asset, direction, entry_score, slope_pct, ema_separation)
        cross_asset_ctx = self.memory.get_other_asset_context(asset)
        winner_dna_str = self.winner_dna.get_dna()

        print(f"[BRAIN:{asset}] Funding: {funding_info.get('signal')} | Memory: {memory_summary[:60]}")
        if cross_asset_ctx:
            print(f"[BRAIN:{asset}] Cross-asset: {cross_asset_ctx[:80]}")

        # --- Step 3: TWO-PASS LLM evaluation ---
        # v8.0: Inject memory + macro intelligence into prompts
        _v8_context_block = ""
        if self._finetune_enricher:
            try:
                _v8_ctx = {'asset': asset, 'regime': regime_name, 'direction': signal,
                           'volatility': float(current_atr / price * 100) if price > 0 else 0.15}
                _v8_context_block = self._finetune_enricher.build_enriched_prompt_block(_v8_ctx)
            except Exception as _e:
                logger.debug(f"[BRAIN] v8.0 enricher failed: {_e}")
        elif self._economic_intelligence:
            try:
                _v8_context_block = self._economic_intelligence.get_llm_context_block()
            except Exception:
                pass
        if self._llm_memory and not _v8_context_block:
            try:
                _sig = {'market_regime': regime_name, 'action_taken': signal}
                _v8_context_block = self._llm_memory.build_dynamic_prompt_context(_sig)
            except Exception:
                pass

        # Append v8.0 context to memory_summary so it flows into both prompts
        if _v8_context_block:
            memory_summary = (memory_summary or "") + "\n\n" + _v8_context_block

        # Pass 1: Mistral scans patterns
        scanner_prompt = build_pattern_scanner_prompt(
            asset=asset, signal=signal, price=price, entry_score=entry_score,
            slope_pct=slope_pct, min_trend_bars=min_trend_bars,
            ema_separation=ema_separation, ema_direction=ema_direction,
            htf_alignment=htf_alignment, current_atr=current_atr,
            current_ema=current_ema, support=support, resistance=resistance,
            candle_lines=candle_lines, orch_result=orch_result if isinstance(orch_result, dict) else {},
            math_filter_warnings=math_filter_warnings, score_reasons=score_reasons,
            htf_1h_direction=htf_1h_direction, htf_4h_direction=htf_4h_direction,
            is_reversal_signal=is_reversal_signal, memory_summary=memory_summary,
            regime_info=regime_info, funding_info=funding_info,
            session_info=session_info, winner_dna=winner_dna_str,
            cross_asset_context=cross_asset_ctx, edge_stats=edge_stats,
            spread_round_trip=self.config.get('exchanges', [{}])[0].get('round_trip_spread_pct', 0.0) if self.config.get('exchanges') else 0.0,
        )

        # Pass 2: Llama decides (builder function receives pattern_scan_result from Pass 1)
        analyst_kwargs = dict(
            asset=asset, signal=signal, price=price, entry_score=entry_score,
            slope_pct=slope_pct, min_trend_bars=min_trend_bars,
            ema_separation=ema_separation, ema_direction=ema_direction,
            htf_alignment=htf_alignment, current_atr=current_atr,
            current_ema=current_ema, support=support, resistance=resistance,
            candle_lines=candle_lines, htf_1h_direction=htf_1h_direction,
            htf_4h_direction=htf_4h_direction, is_reversal_signal=is_reversal_signal,
            memory_summary=memory_summary, regime_info=regime_info,
            funding_info=funding_info, session_info=session_info,
            winner_dna=winner_dna_str, cross_asset_context=cross_asset_ctx,
            math_filter_warnings=math_filter_warnings, score_reasons=score_reasons,
            edge_stats=edge_stats,
            spread_round_trip=self.config.get('exchanges', [{}])[0].get('round_trip_spread_pct', 0.0) if self.config.get('exchanges') else 0.0,
        )

        consensus_result = self.consensus.query_two_pass(
            scanner_prompt=scanner_prompt,
            analyst_prompt_builder=build_risk_analyst_prompt,
            **analyst_kwargs,
        )

        # --- Step 4: Extract and calibrate ---
        proceed = consensus_result.get("proceed", False)
        raw_confidence = consensus_result.get("confidence", 0.0)
        risk_score = consensus_result.get("risk_score", 7)
        trade_quality = consensus_result.get("trade_quality", 5)
        position_size_pct = float(consensus_result.get("position_size_pct", 50))
        patterns = consensus_result.get("patterns", {})
        model_votes = consensus_result.get("model_votes", {})

        # Calibrate confidence using historical accuracy
        calibrated_conf = self.calibrator.calibrate(raw_confidence)
        if abs(calibrated_conf - raw_confidence) > 0.1:
            print(f"[BRAIN:{asset}] CALIBRATED confidence: {raw_confidence:.2f} -> {calibrated_conf:.2f}")
        confidence = calibrated_conf

        # ── Confidence stuck detection ──
        # If LLM returns identical confidence 5+ times in a row, it's not discriminating
        if not hasattr(self, '_recent_confidences'):
            self._recent_confidences = []
        self._recent_confidences.append(round(raw_confidence, 2))
        if len(self._recent_confidences) > 10:
            self._recent_confidences = self._recent_confidences[-10:]
        if len(self._recent_confidences) >= 5:
            last_5 = self._recent_confidences[-5:]
            if len(set(last_5)) == 1:  # All identical
                # Force variation based on actual trade quality metrics
                score_factor = entry_score / 10.0  # 0.0-1.0
                regime_name_local = regime_info.get("regime", "UNKNOWN")
                regime_factor = 0.8 if regime_name_local in ['CHOPPY', 'CRISIS', 'VOLATILE'] else 1.0
                old_conf = confidence
                confidence = confidence * score_factor * regime_factor
                print(f"[BRAIN:{asset}] CONFIDENCE STUCK at {raw_confidence:.2f} x5 — forced variation: {old_conf:.2f} -> {confidence:.2f} (score={score_factor:.1f} regime={regime_factor:.1f})")

        # Pattern-based confidence adjustment
        pat_strength = patterns.get("strength", 5)
        pat_bias = patterns.get("bias", "NEUTRAL")
        resembles_winner = patterns.get("resembles_winner", False)

        # If patterns are strong and align with trade direction, boost confidence
        signal_is_long = signal in ("BUY", "LONG")
        pattern_aligns = (pat_bias == "BULLISH" and signal_is_long) or (pat_bias == "BEARISH" and not signal_is_long)

        if pattern_aligns and pat_strength >= 7:
            confidence = min(1.0, confidence * 1.15)
            print(f"[BRAIN:{asset}] Pattern boost: {pat_bias} str={pat_strength} -> conf {confidence:.2f}")
        elif not pattern_aligns and pat_bias != "NEUTRAL" and pat_strength >= 6:
            confidence = confidence * 0.75
            print(f"[BRAIN:{asset}] Pattern penalty: {pat_bias} contradicts {signal} -> conf {confidence:.2f}")

        if resembles_winner and proceed:
            confidence = min(1.0, confidence * 1.1)
            print(f"[BRAIN:{asset}] Winner DNA match -> conf {confidence:.2f}")

        # Re-check proceed after calibration — only reject at very low confidence
        if proceed and confidence < 0.30:
            proceed = False
            print(f"[BRAIN:{asset}] REJECTED after calibration: conf {confidence:.2f} < 0.30")

        consensus_type = "TWO_PASS_YES" if proceed else "TWO_PASS_NO"

        # Log patterns
        pats_found = patterns.get("candlestick", []) + patterns.get("price_action", [])
        if pats_found:
            print(f"[BRAIN:{asset}] PATTERNS: {', '.join(str(p) for p in pats_found[:5])}")
        print(f"[BRAIN:{asset}] Volume={patterns.get('volume','?')} Momentum={patterns.get('momentum','?')} Structure={patterns.get('structure','?')}")
        print(f"[BRAIN:{asset}] DECISION: proceed={proceed} conf={confidence:.2f} risk={risk_score} quality={trade_quality}")

        # --- Step 5: Kelly sizing ---
        pnl_values = []
        if pnl_history:
            for p in pnl_history:
                if isinstance(p, (int, float)):
                    pnl_values.append(float(p))
                elif isinstance(p, dict):
                    val = p.get("pnl_usd", p.get("pnl", 0))
                    if val:
                        pnl_values.append(float(val))

        kelly_size = self.kelly.compute_from_history(pnl_values, position_size_pct, equity)
        kelly_size = kelly_size * session_mult * regime_info.get("size_multiplier", 1.0)
        kelly_size = round(max(1.0, min(100.0, kelly_size)), 2)

        # ── Record decision for fine-tuning data collection ──
        try:
            if hasattr(self, '_training_collector') and self._training_collector:
                self._training_collector.record_decision(
                    asset=asset, price=price,
                    scanner_prompt=getattr(self.consensus, '_last_scanner_prompt', ''),
                    scanner_output=getattr(self.consensus, '_last_scanner_output', {}),
                    analyst_prompt=getattr(self.consensus, '_last_analyst_prompt', ''),
                    analyst_output=getattr(self.consensus, '_last_analyst_output', {}),
                    market_context={
                        'regime': regime_name,
                        'hurst': regime_info.get('hurst', 0),
                        'entry_score': entry_score,
                        'htf_alignment': htf_alignment,
                        'ema_direction': ema_direction,
                        'signal': signal,
                    },
                )
        except Exception:
            pass  # Never block trading for data collection

        return self._build_result(
            proceed=proceed, confidence=confidence, position_size_pct=kelly_size,
            risk_score=risk_score, trade_quality=trade_quality,
            predicted_l_level=consensus_result.get("predicted_l_level", "L1"),
            bull_case=consensus_result.get("bull_case", ""),
            bear_case=consensus_result.get("bear_case", ""),
            facilitator_verdict=consensus_result.get("facilitator_verdict", ""),
            regime=regime_name, session_mult=session_mult, kelly_size=kelly_size,
            memory_summary=memory_summary, funding_signal=funding_info.get("signal", "NEUTRAL"),
            model_votes=model_votes, consensus=consensus_type, patterns=patterns,
        )

    def _build_result(self, proceed, confidence, position_size_pct, risk_score,
                      trade_quality, predicted_l_level, bull_case, bear_case,
                      facilitator_verdict, regime, session_mult, kelly_size,
                      memory_summary, funding_signal, model_votes, consensus,
                      patterns=None):
        return {
            "proceed": proceed,
            "confidence": round(float(confidence), 3),
            "position_size_pct": round(float(position_size_pct), 2),
            "risk_score": int(risk_score),
            "trade_quality": int(trade_quality),
            "predicted_l_level": str(predicted_l_level),
            "bull_case": str(bull_case)[:200],
            "bear_case": str(bear_case)[:200],
            "facilitator_verdict": str(facilitator_verdict)[:200],
            "brain_details": {
                "regime": regime,
                "session_multiplier": round(float(session_mult), 2),
                "kelly_size": round(float(kelly_size), 2),
                "memory_summary": str(memory_summary)[:300],
                "funding_signal": str(funding_signal),
                "model_votes": model_votes,
                "consensus": consensus,
                "patterns": patterns or {},
            },
        }

    def _safe_default(self, asset, error_msg=""):
        return self._build_result(
            proceed=False, confidence=0.0, position_size_pct=0,
            risk_score=10, trade_quality=0, predicted_l_level="L0",
            bull_case="Error", bear_case=error_msg[:200],
            facilitator_verdict="REJECTED — brain error",
            regime="ERROR", session_mult=1.0, kelly_size=0,
            memory_summary="N/A", funding_signal="N/A",
            model_votes={}, consensus="ERROR", patterns={},
        )
