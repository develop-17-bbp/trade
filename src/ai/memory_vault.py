import os
import sqlite3
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from src.models.trade_trace import TradeTrace

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
        """Lazy-load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            # Using a lightweight model
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
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
                           top_k: int = 3) -> List[Dict]:
        """Query memory for similar past situations using cosine similarity."""
        
        query_text = f"""
        Asset: {asset}
        Market regime: {current_regime}
        Funding rate: {current_funding:.4f}
        Sentiment: bullish={current_sentiment.get('bullish', 0)*100:.0f}% bearish={current_sentiment.get('bearish', 0)*100:.0f}%
        Signal: {'LONG' if proposed_signal == 1 else 'SHORT' if proposed_signal == -1 else 'FLAT'}
        """
        
        try:
            query_embedding = self.model.encode(query_text).astype(np.float32)
            
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Filter by asset to keep it efficient
            cursor.execute('SELECT id, metadata, embedding, document FROM trade_experiences WHERE asset = ?', (asset,))
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return []
            
            results = []
            for row_id, meta_json, emb_blob, doc in rows:
                row_embedding = np.frombuffer(emb_blob, dtype=np.float32)
                
                # Manual cosine similarity Calculation
                # cosine_similarity = (A . B) / (||A|| * ||B||)
                dot_product = np.dot(query_embedding, row_embedding)
                norm_q = np.linalg.norm(query_embedding)
                norm_r = np.linalg.norm(row_embedding)
                similarity = dot_product / (norm_q * norm_r) if norm_q > 0 and norm_r > 0 else 0
                
                results.append({
                    'id': row_id,
                    'metadata': json.loads(meta_json),
                    'document': doc,
                    'similarity': float(similarity)
                })
                
            # Sort by similarity descending
            results.sort(key=lambda x: x['similarity'], reverse=True)
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
