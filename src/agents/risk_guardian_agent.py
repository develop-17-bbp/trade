"""
Risk Guardian Agent
=====================
Computes composite risk score from Monte Carlo VaR, EVT tail risk,
Hawkes process, VPIN, and drawdown. Enforces position sizing limits
based on risk level.
"""

from typing import Dict, Any
from src.agents.base_agent import BaseAgent, AgentVote, DataIntegrityReport, AuditResult


# Risk level thresholds and position scales
_RISK_LEVELS = {
    'LOW':     {'max_risk': 0.3, 'pos_scale': 1.0},
    'MEDIUM':  {'max_risk': 0.5, 'pos_scale': 0.7},
    'HIGH':    {'max_risk': 0.7, 'pos_scale': 0.4},
    'EXTREME': {'max_risk': 1.0, 'pos_scale': 0.0},
}

# Composite risk weights
_WEIGHTS = {
    'mc_risk': 0.30,
    'evt_risk': 0.25,
    'hawkes_pctl': 0.20,
    'vpin': 0.15,
    'drawdown_ratio': 0.10,
}


class RiskGuardianAgent(BaseAgent):
    """Computes composite risk and enforces position sizing limits."""

    def __init__(self, name: str = 'risk_guardian', config: Dict = None):
        super().__init__(name=name, config=config)

    def analyze(self, quant_state: Dict, context: Dict) -> AgentVote:
        # --- Extract risk components ---
        mc_risk = self._safe_get(quant_state, 'monte_carlo_risk', 'risk_score', default=0.5)
        mc_prob_profit = self._safe_get(quant_state, 'monte_carlo_risk', 'prob_profit', default=0.5)
        mc_pos_scale = self._safe_get(quant_state, 'monte_carlo_risk', 'position_scale', default=1.0)

        evt_risk = self._safe_get(quant_state, 'evt_tail_risk', 'risk_score', default=0.5)
        evt_xi = self._safe_get(quant_state, 'evt_tail_risk', 'xi', default=0.0)
        evt_pos_scale = self._safe_get(quant_state, 'evt_tail_risk', 'position_scale', default=1.0)

        hawkes_intensity = self._safe_get(quant_state, 'hawkes', 'intensity', default=0.0)
        hawkes_trade_allowed = self._safe_get(quant_state, 'hawkes', 'trade_allowed', default=True)
        hawkes_pctl = min(1.0, max(0.0, hawkes_intensity))  # Normalize to 0-1

        vpin = self._safe_get(quant_state, 'vpin', 'vpin', default=0.0)
        vpin = min(1.0, max(0.0, vpin))

        # Drawdown ratio from context or quant_state
        drawdown_ratio = self._safe_get(context, 'drawdown_ratio', default=0.0)
        if drawdown_ratio == 0.0:
            drawdown_ratio = self._safe_get(quant_state, 'drawdown', 'ratio', default=0.0)
        drawdown_ratio = min(1.0, max(0.0, drawdown_ratio))

        # --- Compute composite risk ---
        composite_risk = (
            _WEIGHTS['mc_risk'] * mc_risk +
            _WEIGHTS['evt_risk'] * evt_risk +
            _WEIGHTS['hawkes_pctl'] * hawkes_pctl +
            _WEIGHTS['vpin'] * vpin +
            _WEIGHTS['drawdown_ratio'] * drawdown_ratio
        )
        composite_risk = min(1.0, max(0.0, composite_risk))

        # --- Determine risk level and base position scale ---
        if composite_risk < 0.3:
            risk_level = 'LOW'
            position_scale = 1.0
        elif composite_risk < 0.5:
            risk_level = 'MEDIUM'
            position_scale = 0.7
        elif composite_risk < 0.7:
            risk_level = 'HIGH'
            position_scale = 0.4
        else:
            risk_level = 'EXTREME'
            position_scale = 0.0

        # --- Override checks ---
        flags = []

        # Hawkes trade_allowed=False → position_scale=0
        if not hawkes_trade_allowed:
            position_scale = 0.0
            flags.append("HAWKES_BLOCKED: trade_allowed=False")

        # EVT xi > 0.3 → halve position
        if evt_xi > 0.3:
            position_scale *= 0.5
            flags.append(f"EVT_HEAVY_TAIL: xi={evt_xi:.3f}>0.3, position halved")

        # MC prob_profit < 0.4 → FLAT
        if mc_prob_profit < 0.4:
            position_scale = 0.0
            flags.append(f"MC_LOW_PROFIT: prob_profit={mc_prob_profit:.3f}<0.4, going FLAT")

        position_scale = min(1.0, max(0.0, position_scale))

        # Direction: 0 if extreme risk or position_scale is 0
        direction = 0 if position_scale == 0 else context.get('raw_signal', 0)
        confidence = max(0.0, 1.0 - composite_risk)

        reasoning = (
            f"[COMPOSITE_RISK={composite_risk:.3f}] [RISK_LEVEL={risk_level}] "
            f"[POS_SCALE={position_scale:.2f}] "
            f"[MC={mc_risk:.2f}] [EVT={evt_risk:.2f}] [HAWKES={hawkes_pctl:.2f}] "
            f"[VPIN={vpin:.2f}] [DD={drawdown_ratio:.2f}]"
        )
        if flags:
            reasoning += f" [FLAGS={'; '.join(flags)}]"

        return AgentVote(
            direction=direction,
            confidence=round(confidence, 4),
            position_scale=round(position_scale, 4),
            reasoning=reasoning,
            veto=(risk_level == 'EXTREME'),
            metadata={
                'composite_risk': round(composite_risk, 4),
                'risk_level': risk_level,
                'components': {
                    'mc_risk': round(mc_risk, 4),
                    'evt_risk': round(evt_risk, 4),
                    'hawkes_pctl': round(hawkes_pctl, 4),
                    'vpin': round(vpin, 4),
                    'drawdown_ratio': round(drawdown_ratio, 4),
                },
                'overrides': flags,
                'mc_prob_profit': round(mc_prob_profit, 4),
                'evt_xi': round(evt_xi, 4),
                'hawkes_trade_allowed': hawkes_trade_allowed,
            },
        )
