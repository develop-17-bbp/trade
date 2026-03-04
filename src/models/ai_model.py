"""
AI Model — Sentence Transformer Embeddings (Lazy-loaded)
=========================================================
Provides text embedding capabilities using SentenceTransformers.
Lazy-loads the model on first use to avoid slow imports.
"""

_SentenceTransformer = None


def _load():
    global _SentenceTransformer
    if _SentenceTransformer is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer
    except ImportError:
        _SentenceTransformer = False


class AIModel:
    """Sentence-transformer embedding model (lazy-loaded)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        _load()
        if _SentenceTransformer and _SentenceTransformer is not False:
            try:
                self.model = _SentenceTransformer(self.model_name, device=self.device)
            except Exception as e:
                print(f"Warning: Failed to load embedding model: {e}")
                self.model = None
        self._loaded = True

    def embed_text(self, texts):
        """Generate embeddings for a list of texts."""
        self._ensure_loaded()
        if self.model is not None:
            try:
                return self.model.encode(texts)
            except Exception:
                pass
        return [[0.0]] * len(texts)
