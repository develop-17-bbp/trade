"""
Loss Prevention Guardian Agent
================================
Special agent with absolute VETO power over all trading decisions.
Tracks daily PnL and enforces progressive risk modes from NORMAL to HALT.
"""

from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent, AgentVote, DataIntegrityReport, AuditResult


# PnL mode thresholds and parameters
_MODES = {
    'PRESERVATION': {'min_pnl': 1.0, 'min_conf': 0.85, 'pos_scale': 0.5, 'min_agents': 0},
    'APPROACHING':  {'min_pnl': 0.8, 'min_conf': 0.75, 'pos_scale': 0.7, 'min_agents': 0},
    'NORMAL':       {'min_pnl': 0.0, 'min_conf': 0.0,  'pos_scale': 1.0, 'min_agents': 0},
    'CAUTION':      {'min_pnl': -0.5, 'min_conf': 0.0, 'pos_scale': 0.7, 'min_agents': 6},
    'DEFENSIVE':    {'min_pnl': -1.0, 'min_conf': 0.0, 'pos_scale': 0.5, 'min_agents': 7},
    'HALT':         {'min_pnl': float('-inf'), 'min_conf': 1.0, 'pos_scale': 0.0, 'min_agents': 999},
}


class LossPreventionGuardian(BaseAgent):
    """Enforces daily PnL-based risk modes with absolute veto power."""

    def __init__(self, name: str = 'loss_prevention', config: Dict = None):
        super().__init__(name=name, config=config)
        self._intraday_peak_pnl: float = 0.0
        self._max_drawdown_pct: float = 2.0

    def analyze(self, quant_state: Dict, context: Dict) -> AgentVote:
        daily_pnl = context.get('daily_pnl', 0.0)
        if daily_pnl is None:
            daily_pnl = 0.0

        # Track intraday peak for drawdown detection
        if daily_pnl > self._intraday_peak_pnl:
            self._intraday_peak_pnl = daily_pnl

        # Determine mode
        mode = self._determine_mode(daily_pnl)
        mode_params = _MODES[mode]

        # Build reasoning and flags
        flags: List[str] = []
        veto = False
        position_scale = mode_params['pos_scale']
        min_agents = mode_params['min_agents']

        # --- HALT mode: absolute veto ---
        if mode == 'HALT':
            veto = True
            flags.append(f"HALT: daily PnL {daily_pnl:.2f}% below -1.0% threshold")

        # --- Intraday drawdown check ---
        intraday_dd = self._intraday_peak_pnl - daily_pnl
        if intraday_dd > self._max_drawdown_pct:
            flags.append(f"DRAWDOWN_PAUSE: intraday drawdown {intraday_dd:.2f}% exceeds {self._max_drawdown_pct}% limit")
            veto = True
            position_scale = 0.0

        # --- Preservation / Approaching confidence gate ---
        raw_conf = context.get('raw_confidence', 0.5)
        if mode == 'PRESERVATION' and raw_conf < 0.85:
            flags.append(f"PRESERVATION: confidence {raw_conf:.2f} below 0.85 threshold")
            position_scale = 0.0
        elif mode == 'APPROACHING' and raw_conf < 0.75:
            flags.append(f"APPROACHING: confidence {raw_conf:.2f} below 0.75 threshold")
            position_scale = 0.0

        # --- High correlation with open positions check ---
        corr_veto = self._check_position_correlation(quant_state, context)
        if corr_veto:
            flags.append("CORRELATION_VETO: new trade highly correlated with open position")
            veto = True
            position_scale = 0.0

        # Build direction: 0 (flat) if veto or scale=0, else pass through
        direction = 0 if (veto or position_scale == 0) else context.get('raw_signal', 0)

        reasoning_parts = [
            f"[MODE={mode}]",
            f"[DAILY_PNL={daily_pnl:.2f}%]",
            f"[POS_SCALE={position_scale:.2f}]",
        ]
        if veto:
            reasoning_parts.append("[VETO=TRUE]")
        if min_agents > 0:
            reasoning_parts.append(f"[MIN_AGENTS={min_agents}]")
        if flags:
            reasoning_parts.append(f"[FLAGS={len(flags)}]")

        return AgentVote(
            direction=direction,
            confidence=1.0 if veto else 0.8,
            position_scale=position_scale,
            reasoning=" ".join(reasoning_parts),
            veto=veto,
            metadata={
                'mode': mode,
                'daily_pnl': daily_pnl,
                'min_agents_required': min_agents,
                'intraday_drawdown': round(intraday_dd, 4),
                'intraday_peak': round(self._intraday_peak_pnl, 4),
                'flags': flags,
                'position_scale': position_scale,
            },
        )

    def _determine_mode(self, daily_pnl: float) -> str:
        """Determine PnL protection mode based on current daily PnL percentage."""
        if daily_pnl >= 1.0:
            return 'PRESERVATION'
        elif daily_pnl >= 0.8:
            return 'APPROACHING'
        elif daily_pnl >= 0.0:
            return 'NORMAL'
        elif daily_pnl >= -0.5:
            return 'CAUTION'
        elif daily_pnl >= -1.0:
            return 'DEFENSIVE'
        else:
            return 'HALT'

    def _check_position_correlation(self, quant_state: Dict, context: Dict) -> bool:
        """
        Check if a new trade would be highly correlated with existing open positions.
        Uses simple heuristic: same asset + same direction = correlated.
        """
        open_positions = context.get('open_positions', [])
        if not open_positions:
            return False

        asset = context.get('asset', '')
        raw_signal = context.get('raw_signal', 0)
        if raw_signal == 0:
            return False

        for pos in open_positions:
            if not isinstance(pos, dict):
                continue
            pos_asset = pos.get('asset', '') or pos.get('symbol', '')
            pos_side = pos.get('side', '') or pos.get('direction', '')

            if pos_asset == asset:
                # Same asset, same direction → highly correlated
                if (raw_signal > 0 and str(pos_side).lower() in ('long', 'buy', '1')) or \
                   (raw_signal < 0 and str(pos_side).lower() in ('short', 'sell', '-1')):
                    return True
        return False

    def reset_intraday(self):
        """Called at start of new trading day to reset peak tracking."""
        self._intraday_peak_pnl = 0.0
