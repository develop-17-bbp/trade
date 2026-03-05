from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class TradeTrace(BaseModel):
    """Encoded trade + context for ChromaDB storage."""
    timestamp: datetime
    asset: str
    market_regime: str  # TRENDING, RANGING, VOLATILE, CHOPPY
    funding_rate: float
    
    # Market sentiment snapshot
    sentiment: Dict[str, float]  # {bullish, bearish, neutral}
    
    # Agent decision context
    agent_bias: float
    proposed_signal: int  # -1, 0, 1
    signal_confidence: float
    
    # Price action
    price: Dict[str, float]  # {open, high, low, close}
    volume: float
    
    # Trade outcome
    entry_price: float
    exit_price: float
    holding_bars: int
    pnl: float
    pnl_pct: float
    exit_reason: str  # "tp", "sl", "signal_reversal", "manual"
    
    # Agent's reasoning (for audit trail)
    reasoning_trace: str
    
    def to_metadata(self) -> Dict:
        """Convert to ChromaDB metadata."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'asset': self.asset,
            'regime': self.market_regime,
            'pnl_pct': float(self.pnl_pct),
            'holding_bars': int(self.holding_bars),
            'exit_reason': self.exit_reason,
            'proposed_signal': int(self.proposed_signal)
        }
    
    def to_embedding_text(self) -> str:
        """Generate text for vector embedding."""
        return f"""
        Asset: {self.asset}
        Market regime: {self.market_regime}
        Funding rate: {self.funding_rate:.4f}
        Sentiment: bullish={self.sentiment.get('bullish', 0)*100:.0f}% bearish={self.sentiment.get('bearish', 0)*100:.0f}%
        Signal: {'LONG' if self.proposed_signal == 1 else 'SHORT' if self.proposed_signal == -1 else 'FLAT'}
        Confidence: {self.signal_confidence:.0f}%
        Price: {self.price.get('close', 0):.2f}
        Outcome: {self.pnl_pct:+.2f}% over {self.holding_bars} bars
        Reasoning: {self.reasoning_trace}
        """
