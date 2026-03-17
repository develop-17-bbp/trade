"""
Polymarket Prediction Market Data Fetcher
==========================================
Fetches event probabilities from Polymarket's Gamma API (read-only, no auth).
Identifies crypto-related prediction markets and computes probability divergences
between market prices and AI-estimated probabilities.

Inspired by 0xdE17 strategy: find events where market price significantly
differs from actual probability, and use that as a trading signal.
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.data.base_fetcher import CachedFetcher, CACHE_TTL_MEDIUM

logger = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"

# Keywords to identify crypto-related events
CRYPTO_KEYWORDS = re.compile(
    r'\b(bitcoin|btc|ethereum|eth|solana|sol|crypto|blockchain|'
    r'defi|nft|binance|coinbase|stablecoin|usdt|usdc|'
    r'altcoin|token|mining|halving|etf|sec.*crypto|'
    r'ripple|xrp|cardano|ada|polygon|matic|avalanche|avax|'
    r'dogecoin|doge|shiba|pepe|meme.*coin)\b',
    re.IGNORECASE
)


class PolymarketFetcher(CachedFetcher):
    """
    Fetches crypto-related prediction market data from Polymarket.
    Uses the Gamma API (public, no authentication required).

    The Yes token price in a binary market IS the market-implied probability.
    """

    def __init__(self, config: Optional[Dict] = None, timeout: int = 10):
        super().__init__(timeout=timeout)
        cfg = config or {}
        self.divergence_threshold = cfg.get('divergence_threshold', 0.15)
        self.volume_threshold = cfg.get('volume_threshold', 10000)
        self.max_signal_strength = cfg.get('max_signal_strength', 0.8)

    def fetch_crypto_markets(self) -> List[Dict]:
        """
        Fetch active events from Polymarket and filter to crypto-related ones.
        Returns list of dicts with market data.
        """
        cached = self._get_cached('crypto_markets')
        if cached is not None:
            return cached

        try:
            data = self._safe_get(
                f"{GAMMA_API_BASE}/events",
                params={'active': 'true', 'closed': 'false', 'limit': 100}
            )
            if not data or not isinstance(data, list):
                logger.warning("[Polymarket] No events returned or unexpected format")
                return []

            crypto_events = []
            for event in data:
                title = event.get('title', '') or ''
                description = event.get('description', '') or ''
                text = f"{title} {description}"

                if not CRYPTO_KEYWORDS.search(text):
                    continue

                markets = event.get('markets', [])
                for market in markets:
                    outcome_prices = market.get('outcomePrices', '[]')
                    outcomes = market.get('outcomes', '[]')

                    # Parse prices (may be JSON string or list)
                    if isinstance(outcome_prices, str):
                        try:
                            import json
                            outcome_prices = json.loads(outcome_prices)
                        except Exception:
                            outcome_prices = []
                    if isinstance(outcomes, str):
                        try:
                            import json
                            outcomes = json.loads(outcomes)
                        except Exception:
                            outcomes = []

                    yes_price = 0.0
                    if outcome_prices and len(outcome_prices) > 0:
                        try:
                            yes_price = float(outcome_prices[0])
                        except (ValueError, TypeError):
                            pass

                    volume = 0.0
                    try:
                        volume = float(market.get('volume', 0) or 0)
                    except (ValueError, TypeError):
                        pass

                    crypto_events.append({
                        'event_title': title,
                        'question': market.get('question', title),
                        'condition_id': market.get('conditionId', ''),
                        'market_probability': yes_price,
                        'volume': volume,
                        'end_date': market.get('endDate', ''),
                        'outcomes': outcomes,
                        'liquidity': float(market.get('liquidityNum', 0) or 0),
                        'active': market.get('active', True),
                    })

            self._set_cached('crypto_markets', crypto_events, CACHE_TTL_MEDIUM)
            logger.info(f"[Polymarket] Found {len(crypto_events)} crypto markets from {len(data)} total events")
            return crypto_events

        except Exception as e:
            logger.warning(f"[Polymarket] Failed to fetch markets: {e}")
            return []

    def fetch_all_crypto_probabilities(self) -> List[Dict]:
        """
        Get all crypto market probabilities in a flat list.
        Each item has: question, market_probability, volume, condition_id.
        """
        return self.fetch_crypto_markets()

    def find_probability_divergences(
        self, ai_estimates: Dict[str, float]
    ) -> List[Dict]:
        """
        Compare market probabilities against AI-estimated probabilities.

        Args:
            ai_estimates: Dict mapping condition_id to AI-estimated probability (0-1)

        Returns:
            List of divergence dicts for events exceeding threshold.
        """
        markets = self.fetch_all_crypto_probabilities()
        divergences = []

        for m in markets:
            cid = m['condition_id']
            if cid not in ai_estimates:
                continue

            ai_prob = ai_estimates[cid]
            mkt_prob = m['market_probability']
            div = ai_prob - mkt_prob  # Positive = AI thinks more likely than market

            if abs(div) < self.divergence_threshold:
                continue
            if m['volume'] < self.volume_threshold:
                continue

            # Direction signal: if AI thinks event is MORE likely than market prices,
            # and the event is bullish for crypto, then bullish signal
            direction_signal = 'BULLISH' if div > 0 else 'BEARISH'

            divergences.append({
                'question': m['question'],
                'condition_id': cid,
                'market_prob': round(mkt_prob, 4),
                'ai_prob': round(ai_prob, 4),
                'divergence': round(div, 4),
                'abs_divergence': round(abs(div), 4),
                'direction_signal': direction_signal,
                'volume': m['volume'],
                'liquidity': m.get('liquidity', 0),
            })

        # Sort by absolute divergence descending
        divergences.sort(key=lambda x: x['abs_divergence'], reverse=True)
        return divergences

    def get_summary_for_dashboard(self) -> Dict[str, Any]:
        """
        Compact summary for dashboard display.
        Returns active market count, top divergences, and timing.
        """
        markets = self.fetch_all_crypto_probabilities()

        # Filter to liquid markets only
        liquid = [m for m in markets if m['volume'] >= self.volume_threshold]

        # Compute naive divergences (no AI estimates — just report market data)
        top_markets = sorted(liquid, key=lambda x: x['volume'], reverse=True)[:10]

        return {
            'active_markets': len(markets),
            'liquid_markets': len(liquid),
            'top_markets': [
                {
                    'question': m['question'][:100],
                    'probability': round(m['market_probability'], 3),
                    'volume': round(m['volume'], 0),
                }
                for m in top_markets
            ],
            'top_divergences': [],  # Populated when AI estimates are available
            'avg_divergence': 0.0,
            'last_fetch': datetime.now().isoformat(),
        }
