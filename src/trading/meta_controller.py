"""
L4 Meta-Controller Orchestration Layer
========================================
Fuses v5.5 (LightGBM Core) and v6.0 (RL Agent) actions via an XGBoost
Arbitrator (mocked conceptually here) trained on historical alignments.
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

    def arbitrate(self, 
                  lgb_class: int, lgb_conf: float, 
                  rl_action: int, rl_prob: float, 
                  features: Dict[str, float],
                  finbert_score: float,
                  asset_name: str = 'BTC',
                  agentic_bias: float = 0.0) -> Tuple[int, float, float]:
        """
        Returns: 
           final_direction (-1, 0, 1)
           final_confidence (0.0 - 1.0)
           position_scale (0.0 - 1.0) -> Multiplier for sizing (1.0 = full size)
        """
        if not features:
            return 0, 1.0, 0.0

        vol = features.get('ewma_vol', 0.0)
        
        # Dynamic Weighting Based on Volatility Regime
        # Ramps up RL weight to 80% and LightGBM to 20% in high volatility
        if vol > 0.04:  # Simulated threshold
            lgb_weight = 0.20
            rl_weight = 0.80
        else:
            lgb_weight = 0.60
            rl_weight = 0.40

        position_scale = 1.0
        
        # --- Veto Rules ---
        # LightGBM Long + FinBERT Bearish -> Halve size (v5.5)
        if lgb_class == 1 and finbert_score < -0.1:
            position_scale *= 0.5
        elif lgb_class == -1 and finbert_score > 0.1:
            position_scale *= 0.5

        # Meta-Controller Veto (Drawdown Risk from RL)
        # RELAXED for testnet: Allow trades if LightGBM confident, even if RL is uncertain
        # Only veto if BOTH disagreetrongly (RL prob < 0.30)
        if lgb_class != 0 and rl_prob < 0.30:
            # Both models disagree strongly - be cautious
            if lgb_conf < 0.55:  # Also LightGBM not confident
                return 0, 1.0, 0.0  # Full veto only in extreme disagreement

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

            combined_score = (lgb_score * lgb_weight) + (rl_score * rl_weight)

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

        # apply bias toward a direction by adjusting confidence
        if self.bias and final_class != 0:
            adj = self.bias if final_class > 0 else -self.bias
            final_conf = float(min(1.0, max(0.0, float(final_conf) + float(adj))))
            
        # --- NEW: Agentic Macro Bias ---
        if agentic_bias and final_class != 0:
            # Shift confidence based on LLM Macro Analysis
            agent_adj = agentic_bias if final_class > 0 else -agentic_bias
            final_conf = float(min(1.0, max(0.0, float(final_conf) + float(agent_adj))))

        # --- NEW ACCURACY STEP: KELLY SIZING ---
        from src.models.meta_sizer import MetaSizer
        ms = MetaSizer()
        kelly_factor = ms.size(features, win_prob=final_conf, win_loss_ratio=2.0)
        position_scale = float(position_scale) * kelly_factor

        # --- NEW ACCURACY STEP: CORRELATION VETO ---
        from src.trading.correlation_monitor import CorrelationMonitor
        corr_mon = CorrelationMonitor()
        # If this is the second trade in a highly correlated market, reduce size
        if asset_name != 'BTC':
            corr = corr_mon.get_correlation('BTC', asset_name)
            if corr > 0.90:
                position_scale *= 0.5 # Reduce exposure to correlated "herds"
        
        return final_class, float(final_conf), float(position_scale)
