#!/usr/bin/env python3
"""
Verification Script: Per-Trade LLM Reasoning (Layer 6)
Test the new analyze_trade() method to show per-trade explanations.
"""

import sys
from datetime import datetime


def test_per_trade_reasoning():
    """Test per-trade LLM reasoning with sample trade data."""
    
    print("\n" + "="*80)
    print("LAYER 6: PER-TRADE LLM REASONING - VERIFICATION TEST")
    print("="*80)
    
    try:
        from src.ai.agentic_strategist import AgenticStrategist
        print("\n[OK] Imported AgenticStrategist")
    except ImportError as e:
        print(f"\n[ERROR] Failed to import: {e}")
        return False
    
    try:
        from src.monitoring.journal import TradingJournal
        print("[OK] Imported TradingJournal")
    except ImportError as e:
        print(f"[ERROR] Failed to import TradingJournal: {e}")
        return False
    
    # Initialize components
    print("\n[INITIALIZATION]")
    strategist = AgenticStrategist(provider="ollama", model="neural-chat")
    print("[OK] Strategist initialized (Ollama mode)")
    
    journal = TradingJournal()
    recent_count = len(journal.trades)
    print(f"[OK] Journal loaded ({recent_count} total trades)")
    
    # ──────────────────────────────────────────────────────────────────────────────
    # TEST 1: BULLISH BTC ENTRY
    # ──────────────────────────────────────────────────────────────────────────────
    print("\n" + "-"*80)
    print("TEST 1: BULLISH BTC ENTRY")
    print("-"*80)
    
    test_case_1 = {
        'asset': 'BTC_USDT',
        'entry_price': 69700.0,
        'entry_side': 'BUY',
        'l1_signal': {
            'confidence': 75.0,
            'prediction': 'BUY',
            'top_features': ['rsi_14', 'macd_signal', 'sma_50']
        },
        'l2_sentiment': {
            'sentiment_score': 0.45,
            'confidence': 80.0,
            'news_count': 5,
            'source_breakdown': {'NewsAPI': 2, 'CryptoPanic': 2, 'Reddit': 1},
            'label': 'BULLISH'
        },
        'l3_risk': {
            'vpin': 0.60,
            'funding_rate': -0.02,
            'liquidation_levels': "$66,000 | $73,400",
            'position_concentration': 'MEDIUM'
        },
        'market_data': {
            'regime': 'TRENDING',
            'atr': 500.0,
            'trend_direction': 'UP',
            'volatility': 0.03
        },
        'recent_trades': [
            {'asset': 'BTC_USDT', 'side': 'BUY', 'price': 69200, 'pnl': 450},
            {'asset': 'BTC_USDT', 'side': 'BUY', 'price': 68900, 'pnl': 800}
        ]
    }
    
    print("\nInput Signals:")
    print(f"  - Asset: {test_case_1['asset']}")
    print(f"  - Price: ${test_case_1['entry_price']:,.2f}")
    print(f"  - L1 Confidence: {test_case_1['l1_signal']['confidence']:.0f}%")
    print(f"  - L2 Sentiment: {test_case_1['l2_sentiment']['label']} ({test_case_1['l2_sentiment']['sentiment_score']:+.2f})")
    print(f"  - L3 VPIN: {test_case_1['l3_risk']['vpin']:.2f} (non-toxic)")
    print(f"  - Market Regime: {test_case_1['market_data']['regime']}")
    
    print("\nCalling strategist.analyze_trade()...")
    try:
        reasoning_1 = strategist.analyze_trade(
            asset=test_case_1['asset'],
            entry_price=test_case_1['entry_price'],
            entry_side=test_case_1['entry_side'],
            l1_signal=test_case_1['l1_signal'],
            l2_sentiment=test_case_1['l2_sentiment'],
            l3_risk=test_case_1['l3_risk'],
            market_data=test_case_1['market_data'],
            recent_trades=test_case_1['recent_trades']
        )
        print(f"\n[L6-REASONING (Test 1)]:\n{reasoning_1}\n")
        print("[OK] Test 1 PASSED - Generated bullish entry reasoning")
    except Exception as e:
        print(f"\n[ERROR] Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ──────────────────────────────────────────────────────────────────────────────
    # TEST 2: BEARISH ETH EXIT (Loss Aversion)
    # ──────────────────────────────────────────────────────────────────────────────
    print("\n" + "-"*80)
    print("TEST 2: BEARISH ETH EXIT (Loss Aversion Scenario)")
    print("-"*80)
    
    test_case_2 = {
        'asset': 'ETH_USDT',
        'entry_price': 2300.0,
        'entry_side': 'SELL',
        'l1_signal': {
            'confidence': 55.0,  # Lower confidence
            'prediction': 'SELL',
            'top_features': ['atr_14', 'bb_width', 'volume_sma']
        },
        'l2_sentiment': {
            'sentiment_score': -0.25,
            'confidence': 65.0,
            'news_count': 2,
            'source_breakdown': {'NewsAPI': 1, 'CryptoPanic': 1},
            'label': 'BEARISH'
        },
        'l3_risk': {
            'vpin': 0.45,  # Low toxicity
            'funding_rate': 0.05,  # Bearish
            'liquidation_levels': "$2,200 | $2,400",
            'position_concentration': 'LOW'
        },
        'market_data': {
            'regime': 'RANGING',
            'atr': 80.0,
            'trend_direction': 'DOWN',
            'volatility': 0.025
        },
        'recent_trades': []
    }
    
    print("\nInput Signals:")
    print(f"  - Asset: {test_case_2['asset']}")
    print(f"  - Price: ${test_case_2['entry_price']:,.2f}")
    print(f"  - L1 Confidence: {test_case_2['l1_signal']['confidence']:.0f}% (MEDIUM)")
    print(f"  - L2 Sentiment: {test_case_2['l2_sentiment']['label']} ({test_case_2['l2_sentiment']['sentiment_score']:+.2f})")
    print(f"  - L3 VPIN: {test_case_2['l3_risk']['vpin']:.2f} (low toxicity)")
    print(f"  - Market Regime: {test_case_2['market_data']['regime']}")
    
    print("\nCalling strategist.analyze_trade()...")
    try:
        reasoning_2 = strategist.analyze_trade(
            asset=test_case_2['asset'],
            entry_price=test_case_2['entry_price'],
            entry_side=test_case_2['entry_side'],
            l1_signal=test_case_2['l1_signal'],
            l2_sentiment=test_case_2['l2_sentiment'],
            l3_risk=test_case_2['l3_risk'],
            market_data=test_case_2['market_data'],
            recent_trades=test_case_2['recent_trades']
        )
        print(f"\n[L6-REASONING (Test 2)]:\n{reasoning_2}\n")
        print("[OK] Test 2 PASSED - Generated bearish exit reasoning")
    except Exception as e:
        print(f"\n[ERROR] Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ──────────────────────────────────────────────────────────────────────────────
    # TEST 3: VERIFY JOURNAL METHOD
    # ──────────────────────────────────────────────────────────────────────────────
    print("\n" + "-"*80)
    print("TEST 3: VERIFY JOURNAL.GET_RECENT_TRADES() METHOD")
    print("-"*80)
    
    try:
        recent = journal.get_recent_trades(limit=5)
        print(f"\nRetrieved {len(recent)} recent trades:")
        for i, trade in enumerate(recent, 1):
            print(f"  {i}. {trade.get('asset')} {trade.get('side')} @ ${trade.get('price', 'N/A')}")
        print("\n[OK] Test 3 PASSED - Journal method working")
    except Exception as e:
        print(f"\n[ERROR] Test 3 FAILED: {e}")
        return False
    
    return True


def show_summary():
    """Show implementation summary."""
    
    print("\n" + "="*80)
    print("IMPLEMENTATION SUMMARY")
    print("="*80)
    print("\n[CHANGES MADE]:")
    print("\n1. New Method: AgenticStrategist.analyze_trade()")
    print("   Location: src/ai/agentic_strategist.py")
    print("   Purpose: Generate per-trade LLM reasoning during execution")
    
    print("\n2. Integration: executor.py _execute_autonomous_trade()")
    print("   Location: src/trading/executor.py (lines ~1260-1310)")
    print("   Purpose: Call analyze_trade() before logging each trade")
    
    print("\n3. Journal Enhancement: TradingJournal.get_recent_trades()")
    print("   Location: src/monitoring/journal.py")
    print("   Purpose: Provide trade history context for LLM analysis")
    
    print("\n[KEY FEATURES]:")
    print("  - Per-trade reasoning during execution (not end of session)")
    print("  - Considers L1/L2/L3 signals and market regime")
    print("  - Falls back to rule-based if Ollama unavailable")
    print("  - Rate-limited if using Gemini (15 calls/min)")
    print("  - Reasoning stored in trade journal")
    
    print("\n[CONFIGURATION]:")
    print("  - LLM Provider: Ollama (local, no rate limits)")
    print("  - Model: neural-chat")
    print("  - Endpoint: http://localhost:11434")
    print("  - Fallback: Rule-based reasoning")
    
    print("\n[NEXT STEPS]:")
    print("  1. Run live trading to see per-trade reasoning in action")
    print("  2. Monitor logs for [L6-REASONING] entries")
    print("  3. Check trading journal for enhanced reasoning")
    print("  4. Generate trade reports with per-trade explanations")


if __name__ == "__main__":
    try:
        print("\n" + "="*80)
        print("TESTING PER-TRADE LLM REASONING IMPLEMENTATION")
        print("="*80)
        
        success = test_per_trade_reasoning()
        
        if success:
            show_summary()
            print("\n[SUCCESS] All tests passed!")
            print("\nDocumentation: See L6_PER_TRADE_REASONING_GUIDE.md")
            sys.exit(0)
        else:
            print("\n[FAILED] Verification did not complete")
            sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
