"""
Data Integrity Validator Agent (PRE-ANALYSIS GATE)
====================================================
Validates ALL quant model outputs before any agent sees them.
Checks ranges, cross-model consistency, NaN/Inf, staleness, anomalies.
"""

import math
from typing import Dict, Any, List, Tuple
from src.agents.base_agent import BaseAgent, AgentVote, DataIntegrityReport, AuditResult


# Safe defaults for NaN/Inf replacement
_SAFE_DEFAULTS = {
    'rsi_14': 50.0, 'adx': 20.0, 'macd_hist': 0.0,
    'hurst': 0.5, 'confidence': 0.5, 'crisis_prob': 0.0,
    'kappa': 0.5, 'z_score': 0.0, 'snr': 1.0, 'slope': 0.0,
    'intensity': 0.0, 'risk_score': 0.5, 'var_95': 0.0,
    'xi': 0.0, 'ewma_vol': 0.01, 'garch_vol': 0.01,
    'score': 0.0, 'freshness': 1.0, 'last_price': 0.0,
    'volume_change_pct': 0.0, 'position_scale': 1.0,
}


class DataIntegrityValidator(BaseAgent):
    """Pre-analysis gate that validates and sanitizes quant model outputs."""

    def __init__(self, name: str = 'data_integrity', config: Dict = None):
        super().__init__(name=name, config=config)
        self._last_state_hash: str = ""

    def analyze(self, quant_state: Dict, context: Dict) -> AgentVote:
        report = DataIntegrityReport()
        flags: List[str] = []
        inconsistencies: List[Dict] = []
        confidence_adj: Dict[str, float] = {}
        sanitized = self._deep_copy(quant_state)

        # --- Step 1: NaN/Inf sanitization ---
        nan_count = self._sanitize_nans(sanitized, flags)

        # --- Step 2: Range validation ---
        range_violations = self._validate_ranges(sanitized, flags)

        # --- Step 3: Cross-model consistency ---
        self._check_cross_model(sanitized, flags, inconsistencies, confidence_adj)

        # --- Step 4: Staleness detection ---
        state_hash = str(sorted(str(v) for v in self._flatten(sanitized)))
        if state_hash == self._last_state_hash and self._last_state_hash:
            flags.append("STALE_DATA: quant state unchanged from last cycle")
        self._last_state_hash = state_hash

        # --- Step 5: Anomaly detection (>3 models at extremes) ---
        extreme_count = self._count_extremes(sanitized)
        if extreme_count > 3:
            flags.append(f"ANOMALY: {extreme_count} models at extreme values simultaneously")

        # --- Compute quality score ---
        penalty = (nan_count * 0.05) + (range_violations * 0.08) + (len(inconsistencies) * 0.06)
        if "STALE_DATA" in " ".join(flags):
            penalty += 0.15
        if extreme_count > 3:
            penalty += 0.1
        quality_score = max(0.0, min(1.0, 1.0 - penalty))

        # --- Determine recommendation ---
        if quality_score >= 0.7:
            recommendation = "PROCEED"
        elif quality_score >= 0.4:
            recommendation = "PROCEED_WITH_CAUTION"
        else:
            recommendation = "HALT_BAD_DATA"

        report.is_valid = quality_score >= 0.4
        report.quality_score = round(quality_score, 4)
        report.sanitized_state = sanitized
        report.flags = flags
        report.inconsistencies = inconsistencies
        report.confidence_adjustments = confidence_adj
        report.recommendation = recommendation

        reasoning_parts = [f"[QUALITY={quality_score:.2f}]", f"[RECOMMENDATION={recommendation}]"]
        if flags:
            reasoning_parts.append(f"[FLAGS={len(flags)}]")
        if inconsistencies:
            reasoning_parts.append(f"[INCONSISTENCIES={len(inconsistencies)}]")

        return AgentVote(
            direction=0,
            confidence=quality_score,
            position_scale=1.0 if recommendation == "PROCEED" else (0.7 if recommendation == "PROCEED_WITH_CAUTION" else 0.0),
            reasoning=" ".join(reasoning_parts),
            veto=(recommendation == "HALT_BAD_DATA"),
            metadata={"data_integrity_report": report.__dict__},
        )

    # ------------------------------------------------------------------
    def _sanitize_nans(self, state: Dict, flags: List[str]) -> int:
        """Replace NaN/Inf values with safe defaults. Returns count replaced."""
        count = 0
        for section_key, section in state.items():
            if not isinstance(section, dict):
                continue
            for k, v in section.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    safe = _SAFE_DEFAULTS.get(k, 0.0)
                    section[k] = safe
                    flags.append(f"NaN_SANITIZED: {section_key}.{k} replaced with {safe}")
                    count += 1
        return count

    def _validate_ranges(self, state: Dict, flags: List[str]) -> int:
        """Validate known value ranges. Returns violation count."""
        violations = 0
        checks: List[Tuple[str, str, float, float]] = [
            ('trend', 'rsi_14', 0.0, 100.0),
            ('hurst', 'hurst', 0.0, 1.0),
            ('hurst', 'confidence', 0.0, 1.0),
            ('ou_process', 'kappa', 0.0, float('inf')),
            ('kalman', 'snr', 0.0, float('inf')),
        ]
        for section, key, lo, hi in checks:
            val = self._safe_get(state, section, key, default=None)
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                continue
            if val < lo or val > hi:
                flags.append(f"RANGE_VIOLATION: {section}.{key}={val} outside [{lo},{hi}]")
                violations += 1
                # Clamp
                state[section][key] = max(lo, min(hi if hi != float('inf') else val, val))

        # HMM probs sum check
        hmm_probs = self._safe_get(state, 'hmm_regime', 'probs', default=None)
        if isinstance(hmm_probs, (list, tuple)) and hmm_probs:
            prob_sum = sum(hmm_probs)
            if abs(prob_sum - 1.0) > 0.05:
                flags.append(f"RANGE_VIOLATION: hmm_regime.probs sum={prob_sum:.4f} != 1.0")
                violations += 1

        # Prices > 0, volumes >= 0, volatility >= 0
        price = self._safe_get(state, 'price_stats', 'last_price', default=None)
        if isinstance(price, (int, float)) and price <= 0:
            flags.append(f"RANGE_VIOLATION: price_stats.last_price={price} must be >0")
            violations += 1

        for vk in ('ewma_vol', 'garch_vol'):
            vol = self._safe_get(state, 'volatility', vk, default=None)
            if isinstance(vol, (int, float)) and vol < 0:
                flags.append(f"RANGE_VIOLATION: volatility.{vk}={vol} must be >=0")
                violations += 1
                state.setdefault('volatility', {})[vk] = 0.0

        return violations

    def _check_cross_model(self, state: Dict, flags: List[str],
                           inconsistencies: List[Dict], conf_adj: Dict[str, float]):
        """Check cross-model consistency."""
        hurst_regime = self._safe_get(state, 'hurst', 'regime', default='')
        adx = self._safe_get(state, 'trend', 'adx', default=25.0)
        hmm_regime = self._safe_get(state, 'hmm_regime', 'regime', default='')
        garch_vol = self._safe_get(state, 'volatility', 'garch_vol', default=0.01)
        ou_stationary = self._safe_get(state, 'ou_process', 'is_stationary', default=False)
        kalman_slope = self._safe_get(state, 'kalman', 'slope', default=0.0)
        macd_hist = self._safe_get(state, 'trend', 'macd_hist', default=0.0)
        mc_var = self._safe_get(state, 'monte_carlo_risk', 'var_95', default=0.0)
        evt_xi = self._safe_get(state, 'evt_tail_risk', 'xi', default=0.0)

        # Hurst trending but ADX < 15
        if str(hurst_regime).lower() == 'trending' and adx < 15:
            entry = {"models": ["hurst", "trend"], "issue": "Hurst=trending but ADX<15"}
            inconsistencies.append(entry)
            flags.append("INCONSISTENCY: Hurst trending but ADX<15")
            conf_adj['hurst'] = conf_adj.get('hurst', 1.0) * 0.7

        # HMM crisis but GARCH vol low
        if str(hmm_regime).lower() == 'crisis' and garch_vol < 0.02:
            entry = {"models": ["hmm_regime", "volatility"], "issue": "HMM=crisis but GARCH vol low"}
            inconsistencies.append(entry)
            flags.append("INCONSISTENCY: HMM crisis but low GARCH vol")
            conf_adj['hmm_regime'] = conf_adj.get('hmm_regime', 1.0) * 0.7

        # OU stationary but Hurst trending
        if ou_stationary and str(hurst_regime).lower() == 'trending':
            entry = {"models": ["ou_process", "hurst"], "issue": "OU=stationary but Hurst=trending"}
            inconsistencies.append(entry)
            flags.append("INCONSISTENCY: OU stationary but Hurst trending")
            conf_adj['ou_process'] = conf_adj.get('ou_process', 1.0) * 0.7
            conf_adj['hurst'] = conf_adj.get('hurst', 1.0) * 0.8

        # Kalman slope>0 but MACD hist<0
        if kalman_slope > 0 and macd_hist < 0:
            entry = {"models": ["kalman", "trend"], "issue": "Kalman slope>0 but MACD hist<0"}
            inconsistencies.append(entry)
            flags.append("INCONSISTENCY: Kalman slope>0 but MACD hist<0")
            conf_adj['kalman'] = conf_adj.get('kalman', 1.0) * 0.8

        # MC VaR safe but EVT heavy tails
        if abs(mc_var) < 0.02 and evt_xi > 0.3:
            entry = {"models": ["monte_carlo_risk", "evt_tail_risk"],
                     "issue": "MC VaR safe but EVT shows heavy tails"}
            inconsistencies.append(entry)
            flags.append("INCONSISTENCY: MC VaR safe but EVT heavy tails → conservative")
            conf_adj['monte_carlo_risk'] = conf_adj.get('monte_carlo_risk', 1.0) * 0.6

    def _count_extremes(self, state: Dict) -> int:
        """Count how many models are at extreme values."""
        count = 0
        rsi = self._safe_get(state, 'trend', 'rsi_14', default=50.0)
        if rsi > 85 or rsi < 15:
            count += 1
        hurst = self._safe_get(state, 'hurst', 'hurst', default=0.5)
        if hurst > 0.9 or hurst < 0.1:
            count += 1
        crisis = self._safe_get(state, 'hmm_regime', 'crisis_prob', default=0.0)
        if crisis > 0.85:
            count += 1
        z = self._safe_get(state, 'ou_process', 'z_score', default=0.0)
        if abs(z) > 3.0:
            count += 1
        risk = self._safe_get(state, 'monte_carlo_risk', 'risk_score', default=0.5)
        if risk > 0.9:
            count += 1
        evt_risk = self._safe_get(state, 'evt_tail_risk', 'risk_score', default=0.5)
        if evt_risk > 0.9:
            count += 1
        return count

    def _deep_copy(self, d: Dict) -> Dict:
        """Simple deep copy for nested dicts of primitives."""
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out[k] = self._deep_copy(v)
            elif isinstance(v, list):
                out[k] = list(v)
            else:
                out[k] = v
        return out

    def _flatten(self, d: Dict) -> List:
        """Flatten nested dict values for hashing."""
        vals = []
        for v in d.values():
            if isinstance(v, dict):
                vals.extend(self._flatten(v))
            else:
                vals.append(v)
        return vals
