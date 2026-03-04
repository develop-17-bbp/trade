from typing import List, Dict
import time
from src.ai.sentiment import SentimentPipeline

class NewsAggregator:
    def __init__(self, sentiment: SentimentPipeline, decay_gamma: float = 0.5):
        self.sentiment = sentiment
        self.gamma = decay_gamma

    def aggregate(self, headlines: List[str], now: float = None) -> Dict:
        """Return aggregated sentiment score and details.

        headlines: list of strings (most recent first).
        Uses rule-based sentiment by default (fast). Each headline treated with unit confidence.
        Applies exponential time decay: weight_i = exp(-gamma * dt)
        dt uses headline index as proxy for age if timestamp not provided.
        """
        if now is None:
            now = time.time()
        if not headlines:
            return {'S_t': 0.0, 'count': 0, 'details': []}
        scores = self.sentiment.analyze(headlines)
        weighted_sum = 0.0
        weight_total = 0.0
        details = []
        for i, (h, s) in enumerate(zip(headlines, scores)):
            # approximate dt by index
            dt = i
            w = math_exp(-self.gamma * dt)
            val = 0.0
            if isinstance(s, dict):
                # if label provided map POSITIVE/NEGATIVE to 1/-1
                label = s.get('label', '').upper()
                score = s.get('score', 0.5)
                if label.startswith('POS'):
                    val = score
                elif label.startswith('NEG'):
                    val = -score
                else:
                    val = 0.0
            else:
                # fallback numeric
                try:
                    val = float(s)
                except Exception:
                    val = 0.0
            weighted_sum += val * w
            weight_total += w
            details.append({'headline': h, 'val': val, 'weight': w})
        S_t = weighted_sum / weight_total if weight_total > 0 else 0.0
        return {'S_t': S_t, 'count': len(headlines), 'details': details}

def math_exp(x: float) -> float:
    try:
        import math
        return math.exp(x)
    except Exception:
        return 0.0
