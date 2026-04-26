import math
import os
import sqlite3
import json
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from src.models.trade_trace import TradeTrace


# C19 (WebCryptoAgent-inspired): weight similarity by exp(-age / λ) so
# retrieval prefers RECENT experience over stale-but-similar. Lambda
# in hours — 168 (1 week) balances "recent regime" vs "enough samples
# to matter." Overridable per-call.
DEFAULT_TIME_DECAY_LAMBDA_H = 168.0

class MemoryVault:
    """
    Long-term tactical memory using SQLite + Semantic Embeddings.
    Custom implementation to avoid Pydantic v1/v2 issues on Python 3.14.
    """
    
    def __init__(self, db_path: str = "memory/experience_vault"):
        # Ensure the directory exists
        os.makedirs(db_path, exist_ok=True)
        self.db_file = os.path.join(db_path, "vault.db")
        
        # Initialize embedding model lazily
        self._model = None
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_experiences (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                asset TEXT,
                regime TEXT,
                document TEXT,
                metadata TEXT,
                embedding BLOB
            )
        ''')
        conn.commit()
        conn.close()

    @property
    def model(self):
        """Lazy-load the embedding model -- on CPU by default to keep
        GPU memory exclusive for Ollama. MiniLM is small (~80 MB) and
        runs fine on CPU; embedding latency is irrelevant compared to
        the 30B analyst's inference time. Operators on big GPUs can
        opt back in via ACT_EMBEDDER_DEVICE=cuda:0."""
        if self._model is None:
            import os as _os
            from sentence_transformers import SentenceTransformer
            _device = _os.environ.get("ACT_EMBEDDER_DEVICE", "cpu").strip() or "cpu"
            self._model = SentenceTransformer(
                'all-MiniLM-L6-v2', device=_device,
            )
        return self._model

    def store_trade(self, trace: TradeTrace) -> None:
        """Store completed trade in memory."""
        try:
            doc = trace.to_embedding_text()
            meta = trace.to_metadata()
            embedding = self.model.encode(doc).astype(np.float32).tobytes()
            
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            trade_id = f"{trace.asset}_{trace.timestamp.timestamp()}"
            
            cursor.execute('''
                INSERT OR REPLACE INTO trade_experiences 
                (id, timestamp, asset, regime, document, metadata, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_id,
                trace.timestamp.isoformat(),
                trace.asset,
                trace.market_regime,
                doc,
                json.dumps(meta),
                embedding
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"  [MEMORY] Error storing trade experience: {e}")

    def find_similar_trades(self,
                           asset: str,
                           current_regime: str,
                           current_funding: float,
                           current_sentiment: Dict,
                           proposed_signal: int,
                           top_k: int = 3,
                           time_decay_lambda_h: Optional[float] = None,
                           regime_bonus: float = 0.10) -> List[Dict]:
        """Query memory for similar past situations.

        C19 scoring combines three terms:
          * cosine embedding similarity (semantic match)
          * exp(-age_hours / λ) age decay (recent regimes preferred)
          * +regime_bonus if row's regime matches current_regime exactly

        Set time_decay_lambda_h=0 to disable decay (legacy behaviour).
        """
        query_text = f"""
        Asset: {asset}
        Market regime: {current_regime}
        Funding rate: {current_funding:.4f}
        Sentiment: bullish={current_sentiment.get('bullish', 0)*100:.0f}% bearish={current_sentiment.get('bearish', 0)*100:.0f}%
        Signal: {'LONG' if proposed_signal == 1 else 'SHORT' if proposed_signal == -1 else 'FLAT'}
        """
        lam = DEFAULT_TIME_DECAY_LAMBDA_H if time_decay_lambda_h is None else float(time_decay_lambda_h)
        now_ts = time.time()

        try:
            query_embedding = self.model.encode(query_text).astype(np.float32)

            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()

            cursor.execute(
                'SELECT id, timestamp, regime, metadata, embedding, document '
                'FROM trade_experiences WHERE asset = ?', (asset,),
            )
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return []

            results = []
            for row_id, ts_str, regime, meta_json, emb_blob, doc in rows:
                row_embedding = np.frombuffer(emb_blob, dtype=np.float32)

                dot_product = np.dot(query_embedding, row_embedding)
                norm_q = np.linalg.norm(query_embedding)
                norm_r = np.linalg.norm(row_embedding)
                cosine = dot_product / (norm_q * norm_r) if norm_q > 0 and norm_r > 0 else 0.0

                # Age decay
                age_factor = 1.0
                if lam > 0 and ts_str:
                    try:
                        row_ts = datetime.fromisoformat(ts_str).timestamp()
                        age_h = max(0.0, (now_ts - row_ts) / 3600.0)
                        age_factor = math.exp(-age_h / lam)
                    except Exception:
                        age_factor = 1.0

                # Regime bonus
                regime_match = 1.0 + regime_bonus if (regime or "").lower() == (current_regime or "").lower() else 1.0

                weighted = float(cosine) * age_factor * regime_match

                results.append({
                    'id': row_id,
                    'metadata': json.loads(meta_json),
                    'document': doc,
                    'similarity': float(cosine),
                    'age_factor': round(age_factor, 4),
                    'regime_match': regime_match > 1.0,
                    'weighted_score': float(weighted),
                })

            results.sort(key=lambda x: x['weighted_score'], reverse=True)
            return results[:top_k]

        except Exception as e:
            print(f"  [MEMORY] Search error: {e}")
            return []

    def get_regime_stats(self, asset: str, regime: str) -> Dict:
        """Historical performance stats for specific conditions."""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute('SELECT metadata FROM trade_experiences WHERE asset = ? AND regime = ?', (asset, regime))
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return {'count': 0, 'avg_pnl': 0.0}
            
            pnls = [json.loads(r[0])['pnl_pct'] for r in rows]
            return {
                'count': len(pnls),
                'avg_pnl': sum(pnls) / len(pnls),
                'max_pnl': max(pnls),
                'min_pnl': min(pnls)
            }
        except Exception:
            return {'count': 0, 'avg_pnl': 0.0}
