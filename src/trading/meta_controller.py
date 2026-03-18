"""
L4 Meta-Controller Orchestration Layer
========================================
Fuses v5.5 (LightGBM Core) and v6.0 (RL Agent) actions via an XGBoost
Arbitrator (mocked conceptually here) trained on historical alignments.

Quant Finance Integration:
  - HMM regime → dynamic model weight allocation (crisis→RL, bull→LGB)
  - Kalman SNR → confidence multiplier (high SNR = clear signal)
  - MC risk score → position scale reduction under elevated risk
"""
from typing import Dict, Tuple, Optional

class MetaController:
    """
    Arbitrates and outputs a Unified Signal with Confidence and Size Rec.
    Veto Logic (Enhanced): LightGBM Long + FinBERT Bearish -> Halve size (v5.5);
    RL simulates flip risk -> Full veto if >10% drawdown prob.
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        # optional directional bias: positive to favor longs, negative for shorts
        # small values (0.0-0.1) gently tilt confidence
        self.bias = self.config.get('bias', 0.0)

        # Fix A: MetaSizer loaded once at init (not per-call) to avoid overhead
        try:
            from src.models.meta_sizer import MetaSizer
            self._meta_sizer = MetaSizer()
        except Exception:
            self._meta_sizer = None

    def arbitrate(self,
                  lgb_class: int, lgb_conf: float,
                  rl_action: int, rl_prob: float,
                  features: Dict[str, float],
                  finbert_score: float,
                  patch_result: Optional[Dict] = None,
                  asset_name: str = 'BTC',
                  agentic_bias: float = 0.0,
                  hmm_regime: str = 'sideways',
                  hmm_crisis_prob: float = 0.0,
                  kalman_snr: float = 1.0,
                  mc_risk_score: float = 0.5,
                  mc_position_scale: float = 1.0,
                  hawkes_intensity: float = 0.0,
                  hawkes_trade_allowed: bool = True,
                  alpha_freshness: float = 1.0,
                  evt_risk_score: float = 0.3,
                  evt_position_scale: float = 1.0,
                  agentic_enhanced: Optional[Dict] = None) -> Tuple[int, float, float]:
        """
        Returns:
           final_direction (-1, 0, 1)
           final_confidence (0.0 - 1.0)
           position_scale (0.0 - 1.0) -> Multiplier for sizing (1.0 = full size)
        """
        if not features:
            return 0, 1.0, 0.0

        # ── Hawkes Event Clustering Veto ──
        # When event intensity is spiking (clustering), defer new entries
        if not hawkes_trade_allowed:
            return 0, 1.0, 0.0

        vol = features.get('ewma_vol', 0.0)

        # ── HMM Regime-Based Dynamic Weighting ──
        # Replaces simple vol threshold with 4-state regime model
        if hmm_regime == 'crisis' or hmm_crisis_prob > 0.5:
            # Crisis: RL dominates (designed for drawdown management)
            lgb_weight = 0.10
            rl_weight = 0.75
            patch_weight = 0.15
        elif hmm_regime == 'bull':
            # Bull: LGB + RL balanced (Fix E: RL weight raised 0.25→0.40 for trend capture)
            lgb_weight = 0.40
            rl_weight = 0.40
            patch_weight = 0.20
        elif hmm_regime == 'bear':
            # Bear: balanced with RL edge (needs drawdown protection)
            lgb_weight = 0.30
            rl_weight = 0.50
            patch_weight = 0.20
        else:
            # Sideways or unknown: use vol-based fallback
            if vol > 0.04:
                lgb_weight = 0.15
                rl_weight = 0.70
                patch_weight = 0.15
            else:
                lgb_weight = 0.50
                rl_weight = 0.30
                patch_weight = 0.20

        position_scale = 1.0

        # ── HMM Crisis Veto ──
        if hmm_crisis_prob > 0.70:
            position_scale *= 0.3  # Severe reduction in crisis regime

        # --- Veto Rules ---
        # LightGBM Long + FinBERT Bearish -> Halve size (v5.5)
        if lgb_class == 1 and finbert_score < -0.1:
            position_scale *= 0.5
        elif lgb_class == -1 and finbert_score > 0.1:
            position_scale *= 0.5

        # Meta-Controller Veto (Drawdown Risk from RL)
        if lgb_class != 0 and rl_prob < 0.30:
            if lgb_conf < 0.55:
                return 0, 1.0, 0.0

        # Fix C: Veto on directional disagreement with weak signal
        lgb_direction = 1 if lgb_class > 0 else (-1 if lgb_class < 0 else 0)
        rl_direction = 1 if rl_action > 0 else (-1 if rl_action < 0 else 0)
        if lgb_direction != 0 and rl_direction != 0 and lgb_direction != rl_direction:
            if min(lgb_conf, abs(rl_prob - 0.5) * 2) < 0.50:
                return 0, 1.0, 0.0  # VETO: directional disagreement with weak signal

        # PatchTST Liquidity Shock Veto (Loss Prevention)
        if patch_result and patch_result.get('liquidity_shock_prob', 0) > 0.70:
            # High probability of flash crash - full veto
            return 0, 1.0, 0.0

        # --- Agreement / Disagreement Logic ---
        if lgb_class == rl_action:
            final_class = lgb_class
            # Aggregate confidence
            final_conf = min(1.0, (lgb_conf * lgb_weight) + (rl_prob * rl_weight))

            # Boost scale if alignment is strong
            if lgb_class != 0 and (vol * 2 > 0.02) and (features.get('vol_adj_momentum', 0) > 0.85):
                position_scale = min(1.0, position_scale * 1.5)  # 1% stretch target unlocked!
        else:
            # Weighted vote score
            lgb_score = lgb_class * lgb_conf
            rl_score = rl_action * rl_prob
            patch_score = 0.0
            if patch_result:
                patch_score = patch_result.get('prediction', 0) * patch_result.get('confidence', 0)

            combined_score = (lgb_score * lgb_weight) + (rl_score * rl_weight) + (patch_score * patch_weight)

            if combined_score > 0.35:
                final_class = 1
                final_conf = abs(combined_score)
            elif combined_score < -0.35:
                final_class = -1
                final_conf = abs(combined_score)
            else:
                final_class = 0
                final_conf = 1.0 - abs(combined_score)
                position_scale = 0.0

        # ── Kalman SNR Confidence Multiplier ──
        # High SNR = clear signal → boost confidence; Low SNR = noise → reduce
        if kalman_snr > 2.0:
            final_conf = min(1.0, final_conf * 1.15)  # +15% confidence boost
        elif kalman_snr < 0.3:
            final_conf *= 0.7  # -30% confidence penalty (noisy regime)

        # apply bias toward a direction by adjusting confidence
        if self.bias and final_class != 0:
            adj = self.bias if final_class > 0 else -self.bias
            final_conf = float(min(1.0, max(0.0, float(final_conf) + float(adj))))

        # --- Agentic Macro Bias ---
        if agentic_bias and final_class != 0:
            agent_adj = agentic_bias if final_class > 0 else -agentic_bias
            final_conf = float(min(1.0, max(0.0, float(final_conf) + float(agent_adj))))

        # --- KELLY SIZING ---
        # Fix A: use pre-loaded MetaSizer; win_loss_ratio=3.0 matches new atr_tp(4.5)/atr_stop(1.5)
        if self._meta_sizer is not None:
            try:
                kelly_factor = self._meta_sizer.size(features, win_prob=final_conf, win_loss_ratio=3.0)
                kelly_factor = max(0.0, min(1.0, kelly_factor))
            except Exception:
                kelly_factor = 0.5
        else:
            kelly_factor = 0.5
        position_scale = float(position_scale) * kelly_factor

        # --- CORRELATION VETO ---
        # Fix B: more gradual correlation-based scaling (was 0.90 binary)
        from src.trading.correlation_monitor import CorrelationMonitor
        corr_mon = CorrelationMonitor()
        if asset_name != 'BTC':
            corr = corr_mon.get_correlation('BTC', asset_name)
            if corr > 0.85:
                position_scale *= 0.4
            elif corr > 0.70:
                position_scale *= 0.6
            elif corr > 0.60:
                position_scale *= 0.8

        # ── Monte Carlo Risk Scaling ──
        # MC-derived position scale caps sizing based on forward-looking VaR
        if mc_risk_score > 0.7:
            position_scale *= 0.5  # Cut size in half under elevated risk
        position_scale = min(position_scale, mc_position_scale)

        # ── Alpha Decay Confidence Adjustment ──
        # Stale signals get reduced confidence; fresh signals preserved
        if alpha_freshness < 1.0:
            final_conf *= alpha_freshness

        # ── EVT Tail Risk Scaling ──
        # Extreme Value Theory: heavy tails → reduce position
        if evt_risk_score > 0.7:
            position_scale *= 0.4  # Heavy tail regime
        elif evt_risk_score > 0.5:
            position_scale *= 0.7
        position_scale = min(position_scale, evt_position_scale)

        # ── Agent Intelligence Overlay ──
        # Blend existing pipeline (40%) with agent overlay (60%)
        if agentic_enhanced:
            ae = agentic_enhanced
            blend_w = ae.get('blend_weight', 0.60)
            existing_score = final_class * final_conf
            agent_score = ae.get('direction', 0) * ae.get('confidence', 0.0)
            blended = (1 - blend_w) * existing_score + blend_w * agent_score

            # Fix D: dynamic entry threshold scales with blend_weight
            _entry_thresh = 0.15 + 0.20 * blend_w  # 0.30 agents → thresh=0.21; 0.60 agents → thresh=0.27
            if blended > _entry_thresh:
                final_class = 1
            elif blended < -_entry_thresh:
                final_class = -1
            else:
                final_class = 0

            final_conf = float(min(1.0, abs(blended) * 1.5))
            position_scale *= ae.get('position_scale', 1.0)

            # Loss Prevention Guardian VETO overrides everything
            if ae.get('veto'):
                final_class = 0
                position_scale = 0.0

            # Data quality scaling
            dq = ae.get('data_quality', 1.0)
            if dq < 0.7:
                final_conf *= dq

        return final_class, float(final_conf), float(position_scale)
