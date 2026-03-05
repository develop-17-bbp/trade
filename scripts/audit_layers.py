import sys
import os
import traceback
from pprint import pprint

print("Starting Full Top-to-Bottom Architecture Audit...\n")
results = {}

def check(layer_name, func):
    try:
        func()
        results[layer_name] = "PASSED"
        print(f"[OK] {layer_name}")
    except Exception as e:
        results[layer_name] = f"FAILED: {type(e).__name__} - {str(e)}"
        print(f"[FAIL] {layer_name}")
        traceback.print_exc(file=sys.stdout)

# Layer 1: Quantitative Core
def check_layer_1():
    from src.indicators.indicators import add_all_indicators
    from src.models.lightgbm_classifier import LightGBMClassifier
    import pandas as pd
    df = pd.DataFrame({'close': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], 'high': [1]*20, 'low': [1]*20, 'volume': [1]*20})
    df = add_all_indicators(df)
    assert 'rsi_14' in df.columns
    model = LightGBMClassifier()

check("Layer 1: Quantitative Core (Indicators & LightGBM)", check_layer_1)

# Layer 1.5: Auto-Retrain
def check_layer_1_5():
    from src.models.auto_retrain import ModelOptimizer
    opt = ModelOptimizer(n_trials=1)

check("Layer 1.5: Auto-Retrain Engine (Optuna)", check_layer_1_5)

# Layer 2: Alternative Alpha (Sentiment)
def check_layer_2():
    from src.ai.sentiment import SentimentPipeline
    from src.data.news_fetcher import NewsFetcher
    nf = NewsFetcher(newsapi_key="mock", cryptopanic_token="mock")
    sp = SentimentPipeline(use_transformer=False)

check("Layer 2: Alternative Alpha (Sentiment & News)", check_layer_2)

# Layer 3: Risk Management
def check_layer_3():
    from src.risk.dynamic_manager import DynamicRiskManager
    from src.risk.manager import RiskManager
    rm = RiskManager()
    drm = DynamicRiskManager(initial_capital=100000)

check("Layer 3: Risk Management (Circuit Breakers)", check_layer_3)

# Layer 4: On-Chain Intelligence
def check_layer_4():
    from src.data.on_chain_fetcher import OnChainFetcher
    from src.integrations.on_chain_portfolio import OnChainPortfolioManager
    ocf = OnChainFetcher()
    ocpm = OnChainPortfolioManager()

check("Layer 4: On-Chain Intelligence", check_layer_4)

# Layer 5: Arbitration Engine
def check_layer_5():
    from src.trading.meta_controller import MetaController
    mc = MetaController()

check("Layer 5: Arbitration Engine (MetaController)", check_layer_5)

# Layer 6: Agentic Strategist
def check_layer_6():
    from src.ai.agentic_strategist import AgenticStrategist
    # We mock API keys
    os.environ['OPENAI_API_KEY'] = 'mock'
    ag = AgenticStrategist(provider='mock')

check("Layer 6: Agentic Strategist (LLM Reasoning)", check_layer_6)

# Layer 6.5: Memory Vault
def check_layer_6_5():
    from src.ai.memory_vault import MemoryVault
    mv = MemoryVault(persist_directory='./test_chroma')

check("Layer 6.5: Memory Vault (ChromaDB)", check_layer_6_5)

# Phase 6: Meta-Learning
def check_phase_6():
    from src.ai.advanced_learning import AdvancedLearningEngine
    from src.ai.reinforcement_learning import ReinforcementLearningAgent, AdaptiveAlgorithmLayer
    ale = AdvancedLearningEngine()
    rla = ReinforcementLearningAgent()
    aal = AdaptiveAlgorithmLayer()

check("Phase 6: Meta-Learning Engine", check_phase_6)

# Phase 5: Autonomous Trading Desk
def check_phase_5():
    from src.portfolio.allocator import PortfolioAllocator
    from src.portfolio.hedger import PortfolioHedger
    from src.monitoring.health_checker import SystemHealthChecker
    from src.execution.failover import ExecutionFailoverController
    from src.execution.router import ExecutionRouter, ExecutionMode
    pa = PortfolioAllocator()
    ph = PortfolioHedger()
    shc = SystemHealthChecker()
    # Mock price source
    efc = ExecutionFailoverController(primary_exchange="mock")
    er = ExecutionRouter(mode=ExecutionMode.TESTNET)

check("Phase 5: Autonomous Trading Desk (Allocator, Hedger, Failover)", check_phase_5)

# Phase 3: Dashboard State Sync
def check_phase_3():
    from src.api.state import DashboardState
    ds1 = DashboardState()
    ds2 = DashboardState()
    assert ds1 is ds2

check("Phase 3: Dashboard State Config", check_phase_3)

print("\n--- Summary ---")
for k, v in results.items():
    print(f"{k}: {v}")

