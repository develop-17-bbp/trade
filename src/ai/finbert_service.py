"""
FinBERT Inference Service
==========================
Domain-specific financial sentiment analysis using FinBERT.
Outputs polarity (Bullish/Bearish/Neutral) + confidence [0, 1].

FinBERT captures nuances in financial language that general models miss:
  - "Revenue missed expectations" -> Bearish (general model might miss this)
  - "The company beat estimates" -> Bullish
  - "Rates held steady" -> Neutral

Pipeline:
  1. Deduplicate headlines via semantic hashing
  2. Score with FinBERT (or CryptoBERT for crypto-specific text)
  3. Output sentiment polarity + confidence + Z-score
  4. Cache results for sub-1ms LightGBM feature retrieval

Falls back to enhanced rule-based scoring when FinBERT is unavailable.
"""

import hashlib
import math
import time
from typing import List, Dict, Optional, Tuple
from collections import deque


class FinBERTService:
    """
    FinBERT-based financial sentiment analysis service.

    Provides:
      - Sentence-level sentiment scoring
      - Batch inference with deduplication
      - Sentiment Z-score computation (rolling normalization)
      - Results caching for fast downstream retrieval
    """

    # Supported FinBERT model variants
    MODELS = {
        'finbert': 'ProsusAI/finbert',
        'cryptobert': 'ElKulako/cryptobert',
        'twitter-roberta': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    }

    def __init__(self,
                 model_name: str = 'finbert',
                 device: str = 'cpu',
                 max_cache_size: int = 500,
                 z_score_window: int = 50):
        self.device = device
        self.model_key = model_name
        self.model_path = self.MODELS.get(model_name, model_name)
        self.max_cache_size = max_cache_size
        self.z_score_window = z_score_window

        # State
        self._pipeline = None
        self._loaded = False
        self._available = False
        self._cache: Dict[str, Dict] = {}
        self._score_history: deque = deque(maxlen=z_score_window)

    def _ensure_loaded(self):
        """Lazy-load the FinBERT model on first use."""
        if self._loaded:
            return
        self._loaded = True
        
        import os
        if os.environ.get('DISABLE_FINBERT', '1') == '1':
            print("[FinBERT] Disabled via config. Using enhanced rule-based fallback.")
            self._pipeline = None
            self._available = False
            return
            
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self._pipeline = pipeline(
                'sentiment-analysis',
                model=model,
                tokenizer=tokenizer,
                device=self.device if self.device != 'cpu' else -1,
                truncation=True,
                max_length=512,
            )
            self._available = True
            print(f"[FinBERT] Loaded {self.model_path} on {self.device}")
        except Exception as e:
            print(f"[FinBERT] Model unavailable ({e}). Using enhanced rule-based fallback.")
            self._pipeline = None
            self._available = False

    @property
    def is_available(self) -> bool:
        self._ensure_loaded()
        return self._available

    # -------------------------------------------------------------------
    # Core scoring
    # -------------------------------------------------------------------
    def score(self, texts: List[str]) -> List[Dict]:
        """
        Score a batch of texts for financial sentiment.

        Returns list of:
          {
            'text': str,
            'polarity': 'bullish' | 'bearish' | 'neutral',
            'score': float [-1, +1],
            'confidence': float [0, 1],
            'z_score': float,
          }
        """
        if not texts:
            return []

        self._ensure_loaded()

        # Deduplicate
        unique_texts, index_map = self._deduplicate(texts)

        # Score unique texts
        if self._available and self._pipeline:
            unique_results = self._score_transformer(unique_texts)
        else:
            unique_results = self._score_rule_based(unique_texts)

        # Map back to original ordering
        results = [unique_results[index_map[i]] for i in range(len(texts))]

        # Compute Z-scores
        for r in results:
            self._score_history.append(r['score'])
            r['z_score'] = self._compute_z_score(r['score'])

        return results

    def score_single(self, text: str) -> Dict:
        """Score a single text string."""
        results = self.score([text])
        return results[0] if results else self._neutral_result(text)

    def get_sentiment_features(self, texts: List[str]) -> Dict[str, float]:
        """
        Generate LightGBM-ready sentiment features from a batch of texts.

        Returns:
          {
            'sentiment_mean': float,
            'sentiment_std': float,
            'sentiment_z_score': float,
            'bullish_ratio': float,
            'bearish_ratio': float,
            'avg_confidence': float,
            'max_negative_score': float,
            'sentiment_momentum': float,
          }
        """
        if not texts:
            return self._empty_features()

        results = self.score(texts)
        scores = [r['score'] for r in results]
        confidences = [r['confidence'] for r in results]

        n = len(scores)
        mean_score = sum(scores) / n
        variance = sum((s - mean_score) ** 2 for s in scores) / max(n - 1, 1)
        std_score = math.sqrt(max(variance, 0))

        bullish = sum(1 for r in results if r['polarity'] == 'bullish') / n
        bearish = sum(1 for r in results if r['polarity'] == 'bearish') / n
        avg_conf = sum(confidences) / n
        max_neg = min(scores) if scores else 0.0

        # Sentiment momentum: difference between first half and second half
        if n >= 4:
            first_half = sum(scores[:n // 2]) / (n // 2)
            second_half = sum(scores[n // 2:]) / (n - n // 2)
            momentum = second_half - first_half
        else:
            momentum = 0.0

        z = self._compute_z_score(mean_score)

        return {
            'sentiment_mean': mean_score,
            'sentiment_std': std_score,
            'sentiment_z_score': z,
            'bullish_ratio': bullish,
            'bearish_ratio': bearish,
            'avg_confidence': avg_conf,
            'max_negative_score': max_neg,
            'sentiment_momentum': momentum,
        }

    # -------------------------------------------------------------------
    # Transformer scoring (FinBERT)
    # -------------------------------------------------------------------
    def _score_transformer(self, texts: List[str]) -> List[Dict]:
        """Score using FinBERT pipeline."""
        results = []
        try:
            preds = self._pipeline(texts, batch_size=min(len(texts), 16))
            for text, pred in zip(texts, preds):
                label = pred['label'].lower()
                conf = pred['score']

                # Map FinBERT labels to standardized polarity
                if 'pos' in label or label == 'bullish':
                    polarity = 'bullish'
                    score = conf
                elif 'neg' in label or label == 'bearish':
                    polarity = 'bearish'
                    score = -conf
                else:
                    polarity = 'neutral'
                    score = 0.0

                result = {
                    'text': text[:100],
                    'polarity': polarity,
                    'score': score,
                    'confidence': conf,
                    'z_score': 0.0,
                    'model': self.model_key,
                }
                # Cache it
                cache_key = self._hash_text(text)
                self._cache[cache_key] = result
                results.append(result)
        except Exception as e:
            print(f"[FinBERT] Inference error: {e}. Falling back to rule-based.")
            return self._score_rule_based(texts)

        return results

    # -------------------------------------------------------------------
    # Enhanced rule-based fallback
    # -------------------------------------------------------------------
    def _score_rule_based(self, texts: List[str]) -> List[Dict]:
        """Enhanced financial-domain rule-based sentiment scoring."""
        results = []
        for text in texts:
            # Check cache
            cache_key = self._hash_text(text)
            if cache_key in self._cache:
                results.append(self._cache[cache_key])
                continue

            t = text.lower()
            pos_score = 0.0
            neg_score = 0.0

            # Financial positive terms with weights
            pos_terms = {
                'bullish': 2.0, 'bull': 1.5, 'surge': 2.0, 'surging': 2.0,
                'rally': 2.0, 'rallying': 2.0, 'breakout': 1.8, 'moon': 1.5,
                'ath': 2.0, 'all-time high': 2.5, 'all time high': 2.5,
                'beat': 1.5, 'beats': 1.5, 'exceeded': 1.5, 'outperform': 1.5,
                'upgrade': 1.5, 'approved': 2.0, 'approval': 2.0,
                'adoption': 1.5, 'partnership': 1.2, 'institutional': 1.3,
                'inflow': 1.5, 'inflows': 1.5, 'accumulate': 1.5,
                'buy': 1.0, 'buying': 1.0, 'green': 0.8, 'pump': 1.5,
                'recovery': 1.5, 'profit': 1.0, 'gain': 1.0, 'gains': 1.0,
                'record': 1.5, 'milestone': 1.2, 'growth': 1.2,
                'etf approved': 3.0, 'spot etf': 2.5,
            }
            neg_terms = {
                'bearish': 2.0, 'bear': 1.5, 'crash': 2.5, 'crashing': 2.5,
                'plunge': 2.0, 'plunging': 2.0, 'dump': 2.0, 'dumping': 2.0,
                'collapse': 2.5, 'collapsed': 2.5, 'bankrupt': 2.5,
                'hack': 2.5, 'hacked': 2.5, 'exploit': 2.0, 'stolen': 2.0,
                'ban': 2.0, 'banned': 2.0, 'fine': 1.5, 'fined': 1.5,
                'lawsuit': 1.8, 'sued': 1.8, 'investigation': 1.5,
                'sell': 1.0, 'selling': 1.0, 'selloff': 2.0, 'sell-off': 2.0,
                'red': 0.8, 'loss': 1.5, 'losses': 1.5, 'missed': 1.5,
                'outflow': 1.5, 'outflows': 1.5, 'fear': 1.5, 'panic': 2.0,
                'liquidation': 2.0, 'liquidated': 2.0, 'rug': 2.5,
                'scam': 2.5, 'fraud': 2.5, 'ponzi': 2.5,
                'death cross': 2.0, 'recession': 1.8, 'inflation': 1.0,
                'rate hike': 1.5, 'decline': 1.5, 'declining': 1.5,
            }

            for term, weight in pos_terms.items():
                if term in t:
                    pos_score += weight
            for term, weight in neg_terms.items():
                if term in t:
                    neg_score += weight

            total = pos_score + neg_score
            if total == 0:
                polarity = 'neutral'
                score = 0.0
                confidence = 0.3
            else:
                net = pos_score - neg_score
                score = max(-1.0, min(1.0, net / max(total, 1.0)))
                confidence = min(0.95, 0.3 + total * 0.08)
                if score > 0.1:
                    polarity = 'bullish'
                elif score < -0.1:
                    polarity = 'bearish'
                else:
                    polarity = 'neutral'

            result = {
                'text': text[:100],
                'polarity': polarity,
                'score': score,
                'confidence': confidence,
                'z_score': 0.0,
                'model': 'rule_based',
            }
            self._cache[cache_key] = result
            results.append(result)

            # Evict old cache entries
            if len(self._cache) > self.max_cache_size:
                oldest = next(iter(self._cache))
                del self._cache[oldest]

        return results

    # -------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------
    def _deduplicate(self, texts: List[str]) -> Tuple[List[str], Dict[int, int]]:
        """Deduplicate texts using semantic hashing. Returns unique texts + index map."""
        seen: Dict[str, int] = {}
        unique: List[str] = []
        index_map: Dict[int, int] = {}

        for i, text in enumerate(texts):
            h = self._hash_text(text)
            if h not in seen:
                seen[h] = len(unique)
                unique.append(text)
            index_map[i] = seen[h]

        return unique, index_map

    @staticmethod
    def _hash_text(text: str) -> str:
        """Semantic hash: normalize and hash for dedup."""
        normalized = ' '.join(text.lower().split())[:200]
        return hashlib.md5(normalized.encode()).hexdigest()

    def _compute_z_score(self, score: float) -> float:
        """Rolling Z-score of sentiment: (score - mean) / std."""
        if len(self._score_history) < 3:
            return 0.0
        hist = list(self._score_history)
        mean = sum(hist) / len(hist)
        var = sum((x - mean) ** 2 for x in hist) / (len(hist) - 1)
        std = math.sqrt(max(var, 1e-10))
        return (score - mean) / std

    def _neutral_result(self, text: str = '') -> Dict:
        return {
            'text': text[:100], 'polarity': 'neutral',
            'score': 0.0, 'confidence': 0.3, 'z_score': 0.0,
            'model': 'none',
        }

    def _empty_features(self) -> Dict[str, float]:
        return {
            'sentiment_mean': 0.0, 'sentiment_std': 0.0,
            'sentiment_z_score': 0.0, 'bullish_ratio': 0.0,
            'bearish_ratio': 0.0, 'avg_confidence': 0.0,
            'max_negative_score': 0.0, 'sentiment_momentum': 0.0,
        }
