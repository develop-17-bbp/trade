"""
L2 Sentiment Layer — Advanced Sentiment Analysis
==================================================
Two-tier sentiment with time-decayed weighted aggregation.

Tier 1: Fast rule-based keyword scoring (< 1ms)
Tier 2: Transformer-based (FinBERT / CryptoBERT / RoBERTa) — queued (< 10s)

Aggregation equation (per asset):
  S_t = Σ_i c_i · w(t_i) · s_i  /  Σ_i c_i · w(t_i)
  where:
    s_i = sentiment score ∈ [-1, +1]
    c_i = confidence ∈ [0, 1]
    w(t_i) = e^{-γ(t - t_i)}  (exponential time-decay)

Event impact multipliers apply to specific event types
(regulatory, hack, ETF, macro, etc.).
"""

import math
import time
from typing import List, Dict, Optional, Tuple
from enum import Enum
from typing import Any

# try import FinBERT service for optional hooking
type_FinBERT = None
try:
    from src.ai.finbert_service import FinBERTService
    type_FinBERT = FinBERTService
except Exception:
    FinBERTService = None
    type_FinBERT = None


class SentimentLabel(Enum):
    STRONG_POSITIVE = "STRONG_POSITIVE"
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    STRONG_NEGATIVE = "STRONG_NEGATIVE"


class SentimentResult:
    """Individual sentiment analysis result."""
    __slots__ = ('text', 'label', 'score', 'confidence', 'timestamp', 'source',
                 'event_type', 'tickers')

    def __init__(self, text: str, label: str, score: float, confidence: float,
                 timestamp: float = 0.0, source: str = '', event_type: str = 'general',
                 tickers: Optional[List[str]] = None):
        self.text = text
        self.label = label
        self.score = score          # -1 to +1
        self.confidence = confidence  # 0 to 1
        self.timestamp = timestamp or time.time()
        self.source = source
        self.event_type = event_type
        self.tickers = tickers or []

    def to_dict(self) -> Dict:
        return {
            'label': self.label,
            'score': self.score,
            'confidence': self.confidence,
            'event_type': self.event_type,
            'text': self.text[:100],
        }


# Lazy import for transformers
_pipeline = None
_SentenceTransformer = None


def _load_transformers():
    global _pipeline, _SentenceTransformer
    if _pipeline is not None:
        return
    try:
        from transformers import pipeline as _p
        _pipeline = _p
        print("[Sentiment] Transformers library detected")
    except ImportError:
        _pipeline = False
        print("[Sentiment] Transformers library NOT installed; using rule-based only")
    try:
        from sentence_transformers import SentenceTransformer as _ST
        _SentenceTransformer = _ST
        print("[Sentiment] Sentence-Transformers library detected")
    except ImportError:
        _SentenceTransformer = False
        print("[Sentiment] Sentence-Transformers library NOT installed")


class SentimentPipeline:
    """
    Two-tier sentiment analysis with time-decayed aggregation.

    Tier 1: Rule-based keyword + bigram scoring (always available, < 1ms)
    Tier 2: HuggingFace transformer (optional, queued)

    Event impact multipliers boost/dampen sentiment for specific news types.
    """

    # ---- Expanded keyword lexicons ----
    POSITIVE_WORDS = {
        'good', 'bull', 'bullish', 'positive', 'gain', 'gains', 'up', 'surge',
        'surging', 'rally', 'rallying', 'moon', 'green', 'buy', 'buying',
        'crush', 'pump', 'soar', 'skyrocket', 'hodl', 'breakout', 'ath',
        'all-time high', 'approval', 'approved', 'adopt', 'adoption',
        'partnership', 'upgrade', 'record', 'profit', 'profitable',
        'accumulate', 'accumulation', 'institutional', 'inflow', 'inflows',
        'recovery', 'recover', 'growth', 'growing', 'optimistic', 'strong',
        'strength', 'milestone', 'launch', 'launched', 'innovation',
    }

    NEGATIVE_WORDS = {
        'bad', 'bear', 'bearish', 'negative', 'loss', 'losses', 'down',
        'drop', 'dropping', 'red', 'sell', 'selling', 'selloff', 'sell-off',
        'dump', 'dump', 'crash', 'crashing', 'plunge', 'plunging',
        'capitulation', 'rug', 'rugpull', 'scam', 'fraud', 'hack', 'hacked',
        'exploit', 'vulnerability', 'ban', 'banned', 'fine', 'fined',
        'investigation', 'lawsuit', 'sued', 'outflow', 'outflows',
        'fear', 'panic', 'collapse', 'collapsed', 'bankrupt', 'bankruptcy',
        'liquidation', 'liquidated', 'warning', 'caution', 'risk', 'risky',
        'decline', 'declining', 'recession', 'inflation', 'bubble',
    }

    # Bigrams for more nuanced scoring
    POSITIVE_BIGRAMS = [
        'all time high', 'spot etf', 'etf approved', 'etf approval',
        'price target', 'bull market', 'bull run', 'going up',
        'institutional adoption', 'mass adoption', 'buy signal',
    ]

    NEGATIVE_BIGRAMS = [
        'all time low', 'bear market', 'death cross', 'going down',
        'sell signal', 'market crash', 'flash crash', 'circuit breaker',
        'bank run', 'credit crunch', 'rate hike', 'rate increase',
    ]

    # Event impact multipliers (boost sentiment magnitude for high-impact events)
    EVENT_MULTIPLIERS = {
        'regulatory': 1.8,    # regulatory news has outsized impact
        'hack': 2.0,          # security breaches are very negative
        'etf': 1.5,           # ETF news moves markets
        'macro': 1.3,         # macro events matter
        'exchange': 1.4,      # exchange issues affect confidence
        'adoption': 1.2,      # adoption is positive catalyst
        'general': 1.0,
    }

    # Source credibility weights
    SOURCE_WEIGHTS = {
        'newsapi': 0.9,
        'cryptopanic': 0.8,
        'coingecko': 0.7,
        'reddit': 0.5,          # Reddit is noisy
        'default': 0.6,
    }

    def __init__(self,
                 sentiment_model: str = 'cardiffnlp/twitter-roberta-base-sentiment',
                 embed_model: str = 'all-MiniLM-L6-v2',
                 device: str = 'cpu',
                 use_transformer: bool = False,
                 decay_gamma: float = 0.001):
        """If `sentiment_model` is "finbert" or contains "finbert" we will
        delegate Tier-2 scoring to the specialized :class:`FinBERTService`.
        """
        """
        Args:
            decay_gamma: time-decay rate (per second). 0.001 ≈ 50% decay in ~11 min.
        """
        self.device = device
        self.sentiment_model_name = sentiment_model
        self.embed_model_name = embed_model
        self.use_transformer = use_transformer
        self.decay_gamma = decay_gamma
        self.sentiment = None
        self.embedder = None
        self._sentiment_loaded = False
        self._embedder_loaded = False

    # -------------------------------------------------------------------
    # Tier 1: Rule-based sentiment
    # -------------------------------------------------------------------
    def _rule_based_score(self, text: str) -> Tuple[float, float]:
        """
        Fast keyword + bigram scoring.
        Returns (score ∈ [-1,1], confidence ∈ [0,1]).
        """
        t_low = text.lower()
        pos_count = 0
        neg_count = 0

        # Unigram scoring
        for w in self.POSITIVE_WORDS:
            if w in t_low:
                pos_count += 1
        for w in self.NEGATIVE_WORDS:
            if w in t_low:
                neg_count += 1

        # Bigram scoring (higher weight)
        for bg in self.POSITIVE_BIGRAMS:
            if bg in t_low:
                pos_count += 2
        for bg in self.NEGATIVE_BIGRAMS:
            if bg in t_low:
                neg_count += 2

        total = pos_count + neg_count
        if total == 0:
            return 0.0, 0.3  # neutral, low confidence

        raw_score = (pos_count - neg_count) / max(total, 1)
        confidence = min(0.95, 0.3 + total * 0.1)

        return max(-1.0, min(1.0, raw_score)), confidence

    # -------------------------------------------------------------------
    # Tier 2: Transformer-based sentiment
    # -------------------------------------------------------------------
    def _transformer_score(self, texts: List[str]) -> List[Tuple[float, float]]:
        """
        Use HuggingFace transformer or FinBERT service for sentiment scoring.
        Returns list of (score, confidence) tuples.
        """
        # if FinBERT is available and chosen, delegate entirely
        if type_FinBERT and ('finbert' in self.sentiment_model_name.lower()):
            svc = FinBERTService(model_name=self.sentiment_model_name,
                                 device=self.device)
            scored = svc.score(texts)
            return [(r['score'], r['confidence']) for r in scored]

        # otherwise fall back to generic HF pipeline
        _load_transformers()
        if not self._sentiment_loaded and _pipeline and _pipeline is not False:
            try:
                self.sentiment = _pipeline(
                    'sentiment-analysis',
                    model=self.sentiment_model_name,
                    device=self.device,
                    truncation=True,
                    max_length=512,
                )
                self._sentiment_loaded = True
            except Exception as e:
                print(f"Warning: Failed to load transformer: {e}")
                self.sentiment = None
                self._sentiment_loaded = True

        if self.sentiment is None:
            return [self._rule_based_score(t) for t in texts]

        results: List[Tuple[float, float]] = []
        try:
            preds = self.sentiment(texts)
            for p in preds:
                label = p.get('label', 'NEUTRAL').upper()
                conf = p.get('score', 0.5)
                # Map: POSITIVE → +1, NEGATIVE → -1, NEUTRAL → 0
                if 'POS' in label or 'LABEL_2' in label:
                    score = conf
                elif 'NEG' in label or 'LABEL_0' in label:
                    score = -conf
                else:
                    score = 0.0
                results.append((score, conf))
        except Exception:
            results = [self._rule_based_score(t) for t in texts]

        return results

    # -------------------------------------------------------------------
    # Main analysis method
    # -------------------------------------------------------------------
    def analyze(self, texts: List[str],
                timestamps: Optional[List[float]] = None,
                sources: Optional[List[str]] = None,
                event_types: Optional[List[str]] = None
                ) -> List[Dict]:
        """
        Analyze sentiment of texts.
        Returns list of dicts with label, score, confidence.
        Backward-compatible with existing code.
        """
        if not texts:
            return []

        n = len(texts)
        if timestamps is None:
            timestamps = [time.time()] * n
        if sources is None:
            sources = ['default'] * n
        if event_types is None:
            event_types = ['general'] * n

        results: List[Dict] = []

        # Score each text
        if self.use_transformer:
            scores = self._transformer_score(texts)
        else:
            scores = [self._rule_based_score(t) for t in texts]

        for i, (score, conf) in enumerate(scores):
            # Apply event multiplier
            event_mult = self.EVENT_MULTIPLIERS.get(event_types[i], 1.0)
            adjusted_score = max(-1.0, min(1.0, score * event_mult))

            # Determine label
            if adjusted_score > 0.5:
                label = SentimentLabel.STRONG_POSITIVE.value
            elif adjusted_score > 0.1:
                label = SentimentLabel.POSITIVE.value
            elif adjusted_score < -0.5:
                label = SentimentLabel.STRONG_NEGATIVE.value
            elif adjusted_score < -0.1:
                label = SentimentLabel.NEGATIVE.value
            else:
                label = SentimentLabel.NEUTRAL.value

            # Source credibility
            src = sources[i] if i < len(sources) else 'default'
            src_weight = self.SOURCE_WEIGHTS.get('default', 0.6)
            for src_key, w in self.SOURCE_WEIGHTS.items():
                if src_key in src:
                    src_weight = w
                    break

            results.append({
                'label': label,
                'score': adjusted_score,
                'confidence': conf * src_weight,
                'event_type': event_types[i],
                'source': src,
            })

        return results

    # -------------------------------------------------------------------
    # Time-Decayed Weighted Aggregation
    # -------------------------------------------------------------------
    def aggregate_sentiment(self, sentiments: List[Dict],
                             timestamps: Optional[List[float]] = None,
                             current_time: Optional[float] = None
                             ) -> Dict:
        """
        Compute time-decayed weighted aggregate sentiment.

        S_t = Σ_i c_i · w(t_i) · s_i / Σ_i c_i · w(t_i)
        where w(t_i) = e^{-γ(t - t_i)}

        Returns: {
          'aggregate_score': float [-1, +1],
          'aggregate_label': str,
          'confidence': float [0, 1],
          'num_sources': int,
          'freshness': float [0, 1],
        }
        """
        if not sentiments:
            return {
                'aggregate_score': 0.0,
                'aggregate_label': 'NEUTRAL',
                'confidence': 0.0,
                'num_sources': 0,
                'freshness': 0.0,
            }

        now = current_time or time.time()
        n = len(sentiments)
        if timestamps is None:
            timestamps = [now] * n

        weighted_sum = 0.0
        weight_total = 0.0

        for i, sent in enumerate(sentiments):
            s_i = sent.get('score', 0.0)
            c_i = sent.get('confidence', 0.5)
            t_i = timestamps[i] if i < len(timestamps) else now

            # Time-decay kernel: w(t_i) = e^{-γ(t - t_i)}
            age = max(0.0, now - t_i)
            w_i = math.exp(-self.decay_gamma * age)

            weighted_sum += c_i * w_i * s_i
            weight_total += c_i * w_i

        if weight_total == 0:
            agg_score = 0.0
        else:
            agg_score = weighted_sum / weight_total

        # Overall confidence from weight density
        agg_confidence = min(1.0, weight_total / max(n, 1))

        # Freshness: average recency of sources
        ages = [max(0.0, now - t) for t in timestamps]
        avg_age = sum(ages) / n if n else 0
        freshness = math.exp(-self.decay_gamma * avg_age)

        # Label
        if agg_score > 0.5:
            label = SentimentLabel.STRONG_POSITIVE.value
        elif agg_score > 0.1:
            label = SentimentLabel.POSITIVE.value
        elif agg_score < -0.5:
            label = SentimentLabel.STRONG_NEGATIVE.value
        elif agg_score < -0.1:
            label = SentimentLabel.NEGATIVE.value
        else:
            label = SentimentLabel.NEUTRAL.value

        return {
            'aggregate_score': max(-1.0, min(1.0, agg_score)),
            'aggregate_label': label,
            'confidence': agg_confidence,
            'num_sources': n,
            'freshness': freshness,
        }

    # -------------------------------------------------------------------
    # Embeddings (optional)
    # -------------------------------------------------------------------
    def embed(self, texts: List[str]):
        """Generate sentence embeddings using SentenceTransformer."""
        if not texts:
            return []
        _load_transformers()
        if not self._embedder_loaded and _SentenceTransformer and _SentenceTransformer is not False:
            try:
                self.embedder = _SentenceTransformer(self.embed_model_name)
                self._embedder_loaded = True
            except Exception as e:
                print(f"Warning: Failed to load embedder: {e}")
                self.embedder = None
                self._embedder_loaded = True

        if self.embedder is not None:
            try:
                return self.embedder.encode(texts)
            except Exception:
                pass
        return [[0.0]] * len(texts)
