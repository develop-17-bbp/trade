"""
Sentiment Decoder Agent
========================
Fuses FinBERT NLP sentiment, Fear/Greed index, on-chain whale data,
funding rate, and open interest into a single directional vote.
Includes contrarian logic for extreme sentiment regimes.
"""

from typing import Dict, Any
from src.agents.base_agent import BaseAgent, AgentVote


class SentimentDecoderAgent(BaseAgent):
    """Decodes multi-source sentiment into a trading signal."""

    def __init__(self, name: str = 'sentiment_decoder', config: Dict = None):
        super().__init__(name=name, config=config)

    def analyze(self, quant_state: Dict, context: Dict) -> AgentVote:
        # --- Extract inputs ---
        sentiment = quant_state.get('sentiment', {})
        finbert_score = sentiment.get('score', 0.0)
        sentiment_data = context.get('sentiment_data', {})

        ext_feats = context.get('ext_feats', {})
        funding_rate = ext_feats.get('funding_rate', 0.0)
        open_interest = ext_feats.get('open_interest', 0.0)
        fear_greed = ext_feats.get('fear_greed_index', 50.0)

        on_chain = context.get('on_chain', {})
        whale_sentiment = on_chain.get('whale_sentiment', 'neutral')
        liquidation_risk = on_chain.get('liquidation_risk', 0.0)

        raw_signal = context.get('raw_signal', 0)

        # --- Determine direction and flags ---
        direction = 0
        position_scale = 1.0
        reasons = []
        contrarian_flag = False

        # Primary bullish: strong FinBERT + whale bullish + low funding
        if finbert_score > 0.5 and whale_sentiment == 'bullish' and funding_rate < 0.0001:
            direction = 1
            position_scale = 0.9
            reasons.append(f"[FINBERT={finbert_score:.2f}] bullish + whale bullish + low funding")

        # Primary bearish: strong FinBERT negative + whale bearish + high liq risk
        elif finbert_score < -0.5 and whale_sentiment == 'bearish' and liquidation_risk > 0.6:
            direction = -1
            position_scale = 0.9
            reasons.append(f"[FINBERT={finbert_score:.2f}] bearish + whale bearish + [LIQ_RISK={liquidation_risk:.2f}]")

        # Contrarian buy: extreme fear + FinBERT bearish (crowd is wrong)
        elif fear_greed < 20 and finbert_score < -0.3:
            direction = 1
            position_scale = 0.6
            contrarian_flag = True
            reasons.append(f"CONTRARIAN BUY: [FEAR_GREED={fear_greed:.0f}] extreme fear + [FINBERT={finbert_score:.2f}]")

        # Contrarian sell: extreme greed + FinBERT bullish
        elif fear_greed > 80 and finbert_score > 0.3:
            direction = -1
            position_scale = 0.6
            contrarian_flag = True
            reasons.append(f"CONTRARIAN SELL: [FEAR_GREED={fear_greed:.0f}] extreme greed + [FINBERT={finbert_score:.2f}]")

        # Mild directional lean from FinBERT alone
        elif abs(finbert_score) > 0.25:
            direction = 1 if finbert_score > 0 else -1
            position_scale = 0.5
            reasons.append(f"Mild sentiment lean [FINBERT={finbert_score:.2f}]")

        # --- Funding rate overcrowding penalty ---
        if funding_rate > 0.0005 and direction >= 0:
            position_scale *= 0.6
            reasons.append(f"Overcrowded longs [FUNDING={funding_rate:.4f}] reducing long bias")
        elif funding_rate < -0.0005 and direction <= 0:
            position_scale *= 0.6
            reasons.append(f"Overcrowded shorts [FUNDING={funding_rate:.4f}] reducing short bias")

        # --- Open interest divergence ---
        if open_interest > 0:
            reasons.append(f"[OI={open_interest:.0f}]")

        # --- Confidence calculation ---
        sent_conf = sentiment_data.get('confidence', 0.5)
        confidence = abs(finbert_score) * 0.7 + sent_conf * 0.3
        confidence = max(0.05, min(1.0, confidence))

        # Contrarian trades get slightly lower confidence
        if contrarian_flag:
            confidence *= 0.85

        # Clamp position scale
        position_scale = max(0.0, min(1.0, position_scale))

        reasoning = "; ".join(reasons) if reasons else "No clear sentiment signal"

        return AgentVote(
            direction=direction,
            confidence=round(confidence, 4),
            position_scale=round(position_scale, 4),
            reasoning=reasoning,
            metadata={
                'contrarian_flag': contrarian_flag,
                'finbert_score': finbert_score,
                'fear_greed': fear_greed,
                'whale_sentiment': whale_sentiment,
                'funding_rate': funding_rate,
            },
        )
