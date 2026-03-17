"""
Polymarket Arbitrage Agent
===========================
Analyzes prediction market probabilities from Polymarket to identify
crypto events where the market price significantly differs from
AI-estimated probability. Signals when mispricing detected.

Inspired by 0xdE17: AI evaluates "true probability" using trend,
sentiment, and on-chain data, then signals when market is wrong.
"""

import logging
from typing import Dict, Any

from src.agents.base_agent import BaseAgent, AgentVote

logger = logging.getLogger(__name__)


class PolymarketArbitrageAgent(BaseAgent):
    """
    13th agent: uses Polymarket prediction market data to generate
    probability-arbitrage trading signals for crypto assets.
    """

    def __init__(self, name: str = 'polymarket_arb', config: Dict = None):
        super().__init__(name=name, config=config)
        cfg = config or {}
        self.divergence_threshold = cfg.get('divergence_threshold', 0.15)
        self.volume_threshold = cfg.get('volume_threshold', 10000)
        self.max_signal_strength = cfg.get('max_signal_strength', 0.8)
        self._fetcher = None

    def _get_fetcher(self):
        """Lazy-init the fetcher to avoid import at module load."""
        if self._fetcher is None:
            from src.data.polymarket_fetcher import PolymarketFetcher
            self._fetcher = PolymarketFetcher(config={
                'divergence_threshold': self.divergence_threshold,
                'volume_threshold': self.volume_threshold,
                'max_signal_strength': self.max_signal_strength,
            })
        return self._fetcher

    def analyze(self, quant_state: Dict, context: Dict) -> AgentVote:
        """
        Analyze Polymarket probabilities vs AI-estimated probabilities.
        """
        asset = context.get('asset', 'BTC')
        reasons = []

        try:
            fetcher = self._get_fetcher()
            markets = fetcher.fetch_all_crypto_probabilities()

            if not markets:
                return AgentVote(
                    direction=0, confidence=0.0, position_scale=1.0,
                    reasoning=f"[POLYMARKET] No crypto markets available"
                )

            # Filter to markets relevant to this asset
            asset_lower = asset.lower()
            asset_names = {
                'BTC': ['bitcoin', 'btc'],
                'ETH': ['ethereum', 'eth'],
                'SOL': ['solana', 'sol'],
                'AAVE': ['aave'],
            }
            keywords = asset_names.get(asset, [asset_lower])
            relevant = [
                m for m in markets
                if any(kw in m['question'].lower() for kw in keywords)
                and m['volume'] >= self.volume_threshold
            ]

            if not relevant:
                # No relevant markets — use general crypto sentiment from Polymarket
                liquid = [m for m in markets if m['volume'] >= self.volume_threshold]
                if liquid:
                    avg_prob = sum(m['market_probability'] for m in liquid) / len(liquid)
                    reasons.append(f"[PM_GENERAL] {len(liquid)} liquid crypto markets, avg_prob={avg_prob:.2f}")
                    # General crypto market probability as a weak signal
                    return AgentVote(
                        direction=0, confidence=0.1, position_scale=1.0,
                        reasoning=' | '.join(reasons),
                        metadata={'polymarket_markets': len(liquid), 'avg_prob': avg_prob}
                    )
                return AgentVote(
                    direction=0, confidence=0.0, position_scale=1.0,
                    reasoning=f"[POLYMARKET] No relevant markets for {asset}"
                )

            # Build AI probability estimates using available signals
            ai_estimates = self._estimate_probabilities(relevant, quant_state, context)

            # Find divergences
            divergences = fetcher.find_probability_divergences(ai_estimates)

            if not divergences:
                reasons.append(f"[PM_ALIGNED] {len(relevant)} markets for {asset}, no significant divergence")
                return AgentVote(
                    direction=0, confidence=0.15, position_scale=1.0,
                    reasoning=' | '.join(reasons),
                    metadata={'polymarket_markets': len(relevant)}
                )

            # Aggregate divergence signals
            bullish_score = 0.0
            bearish_score = 0.0
            total_weight = 0.0

            for d in divergences:
                # Weight by divergence magnitude and volume
                vol_weight = min(1.0, d['volume'] / 100_000)
                w = d['abs_divergence'] * (0.5 + 0.5 * vol_weight)
                total_weight += w

                if d['direction_signal'] == 'BULLISH':
                    bullish_score += w
                else:
                    bearish_score += w

                reasons.append(
                    f"[PM_DIV={d['divergence']:+.2f}] "
                    f"mkt={d['market_prob']:.2f} ai={d['ai_prob']:.2f} "
                    f"vol=${d['volume']:,.0f} | {d['question'][:60]}"
                )

            # Determine direction
            if total_weight == 0:
                return AgentVote(direction=0, confidence=0.0, reasoning='No weight')

            net_score = (bullish_score - bearish_score) / total_weight

            if net_score > 0.1:
                direction = 1  # LONG
            elif net_score < -0.1:
                direction = -1  # SHORT
            else:
                direction = 0  # FLAT

            confidence = min(self.max_signal_strength, abs(net_score))
            position_scale = min(1.0, len(divergences) / 3.0)

            return AgentVote(
                direction=direction,
                confidence=round(confidence, 4),
                position_scale=round(position_scale, 4),
                reasoning=' | '.join(reasons[:3]),  # Top 3 divergences
                metadata={
                    'polymarket_divergences': len(divergences),
                    'net_score': round(net_score, 4),
                    'bullish_score': round(bullish_score, 4),
                    'bearish_score': round(bearish_score, 4),
                }
            )

        except Exception as e:
            logger.warning(f"[PolymarketAgent] Error: {e}")
            return AgentVote(
                direction=0, confidence=0.0, position_scale=1.0,
                reasoning=f"[POLYMARKET_ERROR] {str(e)[:80]}"
            )

    def _estimate_probabilities(
        self, markets: list, quant_state: Dict, context: Dict
    ) -> Dict[str, float]:
        """
        Estimate "true probability" for each market using available signals.
        This is the core of the probability-arbitrage strategy.
        """
        estimates = {}

        # Extract available signals
        raw_signal = context.get('raw_signal', 0)
        raw_conf = context.get('raw_confidence', 0.5)
        sentiment = context.get('sentiment_data', {})
        sent_score = sentiment.get('composite_score', 0.0) if isinstance(sentiment, dict) else 0.0
        on_chain = context.get('on_chain', {})
        whale = on_chain.get('whale_sentiment', 'neutral')

        # Current trend direction (from quant models)
        trend_bias = 0.0
        if raw_signal > 0:
            trend_bias = raw_conf * 0.3
        elif raw_signal < 0:
            trend_bias = -raw_conf * 0.3

        # Sentiment bias
        sent_bias = sent_score * 0.2  # [-0.2, +0.2]

        # Whale bias
        whale_bias = 0.0
        if whale == 'bullish':
            whale_bias = 0.1
        elif whale == 'bearish':
            whale_bias = -0.1

        combined_bias = trend_bias + sent_bias + whale_bias  # [-0.6, +0.6]

        for m in markets:
            cid = m['condition_id']
            mkt_prob = m['market_probability']

            # Classify event as bullish or bearish for crypto
            q = m['question'].lower()
            is_bullish_event = any(kw in q for kw in [
                'above', 'reach', 'rise', 'break', 'new high', 'approval',
                'etf', 'adopt', 'bullish', 'hit', 'surpass', 'exceed',
            ])
            is_bearish_event = any(kw in q for kw in [
                'below', 'crash', 'fall', 'drop', 'ban', 'bearish',
                'reject', 'decline', 'regulation', 'restrict',
            ])

            # AI estimate: start from market price and adjust by our signal bias
            if is_bullish_event:
                ai_prob = mkt_prob + combined_bias
            elif is_bearish_event:
                ai_prob = mkt_prob - combined_bias
            else:
                # Neutral event — harder to estimate, keep close to market
                ai_prob = mkt_prob + combined_bias * 0.3

            # Clamp to valid probability range
            ai_prob = max(0.01, min(0.99, ai_prob))
            estimates[cid] = ai_prob

        return estimates
