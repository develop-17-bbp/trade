
import pytest
from src.ai.sentiment import SentimentPipeline


def test_transformer_sentiment_produces_nonzero_scores():
    # this test will import transformers; if not installed, it will skip gracefully
    try:
        sp = SentimentPipeline(use_transformer=True)
    except Exception as e:
        pytest.skip(f"Transformer environment unavailable: {e}")

    # simple positive/negative examples
    texts = ["Bitcoin surges after ETF approval", "Major hack causes crash"]
    results = sp.analyze(texts)
    assert len(results) == 2
    scores = [r['score'] for r in results]
    # one score should be positive, the other negative (or at least non-neutral)
    assert any(s > 0 for s in scores)
    assert any(s < 0 for s in scores)


def test_pipeline_uses_finbert(monkeypatch):
    """When sentiment_model contains 'finbert', the pipeline should delegate
    scoring to FinBERTService.score()."""
    try:
        from src.ai.sentiment import SentimentPipeline, FinBERTService
    except ImportError:
        pytest.skip("SentimentPipeline/FinBERTService import failed")

    called = {'count': 0}

    def fake_score(self, texts):
        called['count'] += 1
        # return neutral scores for simplicity
        return [{'text': t, 'polarity': 'neutral', 'score': 0.0, 'confidence': 0.5} for t in texts]

    monkeypatch.setattr(FinBERTService, 'score', fake_score)

    sp = SentimentPipeline(use_transformer=True, sentiment_model='finbert')
    texts = ["test"]
    res = sp.analyze(texts)
    assert called['count'] == 1
    assert res[0]['score'] == 0.0
    # confidence is multiplied by default source weight (≈0.6 -> 0.3)
    assert abs(res[0]['confidence'] - 0.3) < 1e-6
