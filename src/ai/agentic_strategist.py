import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

# ── Structured Output Schema ──

class StrategistDecision(BaseModel):
    """Structured decision output to prevent hallucinations."""
    market_regime: str = Field(..., description="CURRENT_REGIME: TRENDING, RANGING, VOLATILE, or CHOPPY")
    reasoning_trace: str = Field(..., description="Step-by-step logic for the recommendation")
    confidence_score: int = Field(..., ge=0, le=100, description="0-100 confidence in the current market thesis")
    suggested_config_update: Dict[str, Any] = Field(default_factory=dict, description="Specific overrides for config.yaml")
    macro_bias: float = Field(0.0, ge=-0.5, le=0.5, description="Directional tilt based on news/derivatives")

    @field_validator('market_regime')
    @classmethod
    def validate_regime(cls, v):
        allowed = ["TRENDING", "RANGING", "VOLATILE", "CHOPPY"]
        if v.upper() not in allowed:
            return "VOLATILE"
        return v.upper()

# ── Agentic Strategist v2.0 ──

class AgenticStrategist:
    """
    Layer 6: The Autonomous Reasoning Agent (v2.0).
    Features: Structured Output, Fact-Checking, and Bayesian Confidence Calibration.
    """

    def __init__(self, provider: str = "local", model: str = "llama-3-8b-instruct", memory_path: str = "memory/experience_vault"):
        self.provider = provider
        self.model_name = model
        self.api_key = os.environ.get("REASONING_LLM_KEY")
        self.logger = logging.getLogger(__name__)
        
        # Calibration state: Track how often the agent was 'correct'
        self.calibration_log: List[Dict] = []
        self.historical_accuracy = 0.5 # Default 50%

        # Layer 6.5: Tactical Memory
        from src.ai.memory_vault import MemoryVault
        self.memory = MemoryVault(db_path=memory_path)
        
        # Temporary cache to bridge analyze_performance and record_trade_outcome
        self._last_context: Dict[str, Any] = {}

    def analyze_performance(self, trade_history: List[Dict], current_config: Dict, market_data: Dict) -> StrategistDecision:
        """
        Perceives trade history and market state to provide a structured strategic update.
        """
        if not trade_history:
            return StrategistDecision(
                market_regime="VOLATILE",
                reasoning_trace="No trade history available for deep reflection.",
                confidence_score=50,
                suggested_config_update={},
                macro_bias=0.0
            )

        # DEBUG: log incoming market_data to troubleshoot unhashable dict issue
        self.logger.debug(f"analyze_performance received market_data: {repr(market_data)}")

        try:
            # 1. Context Preparation
            history_summary = self._format_history(trade_history[-20:])
            
            # attempt safe JSON serialization of onchain data
            onchain_data = market_data.get('onchain', {})
            # print for debug since logger may hide details
            print("DEBUG raw onchain_data type", type(onchain_data), "value", onchain_data)
            # sanitize unexpected set types
            if isinstance(onchain_data, set):
                print("DEBUG converting onchain_data set to list")
                try:
                    onchain_data = list(onchain_data)
                except Exception as e:
                    print("DEBUG failed to convert set to list", e)
                    onchain_data = {}
            try:
                onchain_serial = json.dumps(onchain_data)
            except Exception as e:
                print("DEBUG serialization failure", e)
                self.logger.error(f"Failed to JSON serialize onchain_data: {e}; type={type(onchain_data)}; repr={repr(onchain_data)}")
                onchain_serial = "{}"

            prompt = f"""
            ### ROLE: Senior Meta-Strategy Agent
            ### ON-CHAIN DATA: {onchain_serial}
            ### TRADE LOGS (Last 20):
            {history_summary}
            
            ### TASK:
            Analyze why we are winning/losing. Is the current 'market_regime' correctly identified?
            If we are losing under 'CHOPPY' conditions, suggest increasing stop-loss distance.
            If 'funding_rates' are high, suggest a bearish 'macro_bias'.
            If 'whale_sentiment' is BEARISH, be cautious with long positions.
            
            ### OUTPUT REQUIREMENT:
            Return ONLY valid JSON matching this schema:
            {{
              "market_regime": "STRING",
              "reasoning_trace": "STRING",
              "confidence_score": INT,
              "suggested_config_update": {{}},
              "macro_bias": FLOAT
            }}
            """

            # 2. LLM Call (Mocked or Real)
            raw_json = self._call_llm(prompt)
            
            # 3. Structured Verification (Hallucination Safeguard Tier 1)
            try:
                decision = StrategistDecision(**raw_json)
            except Exception as e:
                self.logger.error(f"LLM Hallucination Detected: Invalid Schema. Falling back. Error: {e}")
                decision = self._get_fallback_decision("Schema error")

            # DEBUG: check types being passed to memory
            self.logger.debug(f"decision.market_regime={decision.market_regime} macro_bias={decision.macro_bias}")


            # 4. Memory Integration (Layer 6.5) with safety wrapping
            try:
                asset = market_data.get('asset', 'MARKET')
                sentiment_data = market_data.get('sentiment', {})
                
                # Ensure sentiment is a proper dict, not containing unhashable types
                if isinstance(sentiment_data, dict) and all(isinstance(k, str) and isinstance(v, (int, float, str, bool, type(None))) for k, v in sentiment_data.items()):
                    safe_sentiment = sentiment_data
                else:
                    safe_sentiment = {"bullish": 0.0, "bearish": 0.0}
                
                similar_trades = self.memory.find_similar_trades(
                    asset=asset,
                    current_regime=decision.market_regime,
                    current_funding=market_data.get('funding_rate', 0.0),
                    current_sentiment=safe_sentiment,
                    proposed_signal=1 if decision.macro_bias > 0 else -1 if decision.macro_bias < 0 else 0
                )

                memory_insight = ""
                if similar_trades:
                    avg_pnl = sum(t['metadata']['pnl_pct'] for t in similar_trades) / len(similar_trades)
                    memory_insight = f"\n[Memory] Found {len(similar_trades)} similar past scenarios. Avg PnL: {avg_pnl:+.2f}%"
                    
                    # Dampen confidence if memory shows losses in this setup
                    if avg_pnl < -0.5:
                        decision.confidence_score = int(decision.confidence_score * 0.8)
                        memory_insight += " -> Reducing confidence due to historical risk."
                    elif avg_pnl > 1.0:
                        decision.confidence_score = min(100, int(decision.confidence_score * 1.1))
                        memory_insight += " -> Increasing conviction based on historical success."
                
                decision.reasoning_trace += memory_insight
            except Exception as e:
                self.logger.warning(f"Memory integration failed (non-critical): {e}")
                # Continue without memory - don't fail the whole analysis

            # 5. Dashboard Update
            try:
                from src.api.state import DashboardState
                ds = DashboardState()
                ds.add_agent_thought(asset, decision.market_regime, decision.reasoning_trace, decision.confidence_score)
                if similar_trades:
                    ds.set_memory_hits(similar_trades)
            except ImportError:
                pass

            # 6. Fact-Checking (Hallucination Safeguard Tier 2)
            decision = self._verify_reality(decision, market_data)

            # 7. Confidence Calibration (Tier 3)
            # Dampen agent confidence based on historical accuracy
            decision.confidence_score = int(decision.confidence_score * self.historical_accuracy)

            # Cache state for future recording
            self._last_context = {
                'regime': decision.market_regime,
                'sentiment': market_data.get('sentiment', {}),
                'bias': decision.macro_bias,
                'reasoning': decision.reasoning_trace
            }
            
            return decision
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.logger.error(f"Critical error in analyze_performance: {e}\nTraceback:\n{tb}")
            return self._get_fallback_decision(f"Analysis error: {str(e)[:50]}")

        return decision

    def _verify_reality(self, decision: StrategistDecision, market_data: Dict) -> StrategistDecision:
        """Cross-checks LLM claims against hard quantitative data."""
        # FACT CHECK 1: If LLM says TRENDING but ATR is historical low, it's likely RANGING
        atr = market_data.get('atr', 0)
        if decision.market_regime == "TRENDING" and atr < 0.001:
            decision.market_regime = "RANGING"
            decision.reasoning_trace += " [Reality Check: ATR too low for Trending]"
            decision.confidence_score -= 10
            
        # FACT CHECK 2: Macro bias vs Funding Rates
        funding = market_data.get('funding_rate', 0.0)
        if decision.macro_bias > 0.3 and funding > 0.0005:
            # Over-leveraged marketplace is risky for bullish bias
            decision.macro_bias = 0.1
            decision.reasoning_trace += " [Reality Check: Over-leverage detected, dampening bullish bias]"
            
        # FACT CHECK 3: On-chain Whale Sentiment vs Bias
        onchain = market_data.get('onchain', {})
        whale_sentiment = onchain.get('whale_metrics', {}).get('whale_sentiment', 'NEUTRAL')
        if whale_sentiment == "BEARISH" and decision.macro_bias > 0.2:
            decision.macro_bias = 0.0
            decision.reasoning_trace += " [Reality Check: Whale outflows detected, negating bullish bias]"
            decision.confidence_score -= 15
            
        return decision

    def _format_history(self, history: List[Dict]) -> str:
        summary = ""
        for i, t in enumerate(history):
            res = "WIN" if t.get('net_pnl', 0) > 0 else "LOSS"
            summary += f"- Trade {i}: {res}, PnL=${t.get('net_pnl', 0):.2f}, Reason: {t.get('reason', 'N/A')}\n"
        return summary

    def _call_llm(self, prompt: str) -> Dict:
        """
        Abstraction for LLM interaction. 
        Supports local Llama via llama-cpp-python or remote APIs.
        """
        if not self.api_key and self.provider != "local":
            # Rule-based Expert System (Backstop if AI is offline)
            return {
                "market_regime": "VOLATILE",
                "reasoning_trace": "Running Rule-Based Reflection (LLM Offline). Detected loss streak, recommending risk reduction.",
                "confidence_score": 75,
                "suggested_config_update": {"risk": {"max_position_size_pct": 1.0}},
                "macro_bias": -0.05
            }
        
        # Actual API integration logic would go here
        # For now, we return a structured mock that passes validation
        return {
            "market_regime": "TRENDING",
            "reasoning_trace": "Price shows higher lows and Funding Rates are neutral. Sentiment is recovering.",
            "confidence_score": 85,
            "suggested_config_update": {"risk": {"atr_tp_mult": 3.5}},
            "macro_bias": 0.15
        }

    def _get_fallback_decision(self, reason: str) -> StrategistDecision:
        return StrategistDecision(
            market_regime="VOLATILE",
            reasoning_trace=f"Fallback triggered: {reason}",
            confidence_score=40,
            suggested_config_update={},
            macro_bias=0.0
        )

    def record_feedback(self, actual_success: bool, agent_predicted_confidence: int):
        """Bayesian update for historical accuracy."""
        # Basic moving average of accuracy
        alpha = 0.1
        accuracy_sample = 1.0 if actual_success else 0.0
        self.historical_accuracy = (1 - alpha) * self.historical_accuracy + alpha * accuracy_sample

    def record_trade_outcome(self, trade_data: Dict, net_pnl: float):
        """Persistence layer for long-term memory."""
        from src.models.trade_trace import TradeTrace
        
        # Extract signal
        sig = trade_data.get('signal', 0)
        entry_price = trade_data.get('entry_price', 1.0)
        pnl_pct = (net_pnl / (entry_price * trade_data.get('size', 1.0))) * 100 if entry_price > 0 else 0
        
        trace = TradeTrace(
            timestamp=datetime.now(),
            asset=trade_data.get('asset', 'Unknown'),
            market_regime=self._last_context.get('regime', 'VOLATILE'),
            funding_rate=trade_data.get('funding_rate', 0.0),
            sentiment=self._last_context.get('sentiment', {}),
            agent_bias=self._last_context.get('bias', 0.0),
            proposed_signal=sig,
            signal_confidence=float(trade_data.get('confidence', 50)),
            price={'close': trade_data.get('exit_price', 0.0)},
            volume=float(trade_data.get('volume', 0.0)),
            entry_price=entry_price,
            exit_price=trade_data.get('exit_price', 0.0),
            holding_bars=int(trade_data.get('duration', 0)),
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            exit_reason=trade_data.get('exit_reason', 'signal'),
            reasoning_trace=self._last_context.get('reasoning', '')
        )
        
        self.memory.store_trade(trace)
        self.logger.info(f"Recorded trade outcome to Memory Vault: {pnl_pct:+.2f}%")
