"""
Authority Compliance Guardian Agent
=====================================
Enforces the non-negotiable authority rules from the official trading PDF.
Has absolute VETO power — any authority violation blocks the trade entirely.

This agent is the deterministic counterpart to LoRA-learned behavior:
the LLM might *prefer* good trades after fine-tuning, but only this guardian
*guarantees* hard rules are honored (wick entries, small-body candles,
higher-TF disagreement, stop widening, adding to losers, news blackouts).
"""

from typing import Dict, Any, List

from src.agents.base_agent import BaseAgent, AgentVote
from src.ai.authority_rules import (
    validate_authority_entry,
    get_asset_permitted_types,
    AUTHORITY_STRATEGIES,
)


class AuthorityComplianceGuardian(BaseAgent):
    """Validates every proposed trade against the authority PDF directives.

    VETO triggers:
      - Asset traded outside its permitted types (e.g., swinging ETH)
      - Higher-TF trend disagrees with entry direction
      - Entry on a wick (no candle close)
      - Entry on a small-body candle (body below 10-50 bar avg)
      - Stop-loss widened after entry
      - Adding to a losing position
      - News blackout window active
      - Mean-reversion used outside CHOP/LOW_VOL regime
      - Fakeout filters not all cleared on 5m/15m

    Missing context is treated as unknown — rules that cannot be verified
    emit a reasoning flag but do not veto. Only deterministically failed
    rules veto.
    """

    def __init__(self, name: str = 'authority_compliance', config: Dict = None):
        super().__init__(name=name, config=config)
        self._veto_count: int = 0
        self._check_count: int = 0

    def analyze(self, quant_state: Dict, context: Dict) -> AgentVote:
        self._check_count += 1

        raw_signal = context.get('raw_signal', 0)

        # No trade proposed — nothing to enforce, stand down
        if raw_signal == 0:
            return AgentVote(
                direction=0,
                confidence=1.0,
                position_scale=1.0,
                reasoning="[AUTHORITY=STANDBY] no trade proposed",
                veto=False,
                metadata={'checks_run': self._check_count, 'vetoes_total': self._veto_count},
            )

        ok, violations = validate_authority_entry(quant_state, context)

        # Auxiliary informational check: report permitted types for this asset
        asset = str(context.get('asset', '')).upper().strip()
        permitted = get_asset_permitted_types(asset) if asset else []

        if not ok:
            self._veto_count += 1
            reasoning = (
                f"[AUTHORITY_VETO] {len(violations)} violation(s): "
                + " | ".join(violations[:3])  # cap reasoning length
            )
            if len(violations) > 3:
                reasoning += f" | (+{len(violations) - 3} more)"

            return AgentVote(
                direction=0,
                confidence=1.0,
                position_scale=0.0,
                reasoning=reasoning,
                veto=True,
                metadata={
                    'violations': violations,
                    'violation_count': len(violations),
                    'asset': asset,
                    'permitted_trade_types': permitted,
                    'checks_run': self._check_count,
                    'vetoes_total': self._veto_count,
                },
            )

        # All rules satisfied — pass through the signal unchanged
        return AgentVote(
            direction=raw_signal,
            confidence=0.9,  # high confidence in authority compliance itself
            position_scale=1.0,
            reasoning=(
                f"[AUTHORITY=COMPLIANT] asset={asset} "
                f"permitted={permitted} signal={raw_signal}"
            ),
            veto=False,
            metadata={
                'asset': asset,
                'permitted_trade_types': permitted,
                'known_strategies': [s['name'] for s in AUTHORITY_STRATEGIES],
                'checks_run': self._check_count,
                'vetoes_total': self._veto_count,
            },
        )

    def get_compliance_stats(self) -> Dict[str, Any]:
        """Return agent lifetime compliance statistics."""
        veto_rate = self._veto_count / self._check_count if self._check_count > 0 else 0.0
        return {
            'total_checks': self._check_count,
            'total_vetoes': self._veto_count,
            'veto_rate': round(veto_rate, 4),
        }
