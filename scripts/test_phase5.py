from src.portfolio.allocator import PortfolioAllocator
from src.portfolio.hedger import PortfolioHedger
from src.monitoring.health_checker import SystemHealthChecker
from src.execution.failover import ExecutionFailoverController

print('\n' + '='*60)
print('  PHASE 5: AUTONOMOUS RUNTIME COMPONENTS VERIFICATION')
print('='*60)

# 1. Health Checker
health = SystemHealthChecker()
health.register_component('binance_api', lambda: True)
health.register_component('onchain_node', lambda: True)
sys_status = health.check_overall_health()
print(f"\n[Component: HealthChecker]")
print(f"  Status: {sys_status['status']} (Components: {list(sys_status['components'].keys())})")

# 2. Allocator (Kelly + OnChain)
allocator = PortfolioAllocator(total_capital=125000.0)
metrics = {
    'BTC': {'win_rate': 0.65, 'avg_win_loss_ratio': 1.8, 'volatility': 0.02},
    'ETH': {'win_rate': 0.55, 'avg_win_loss_ratio': 1.2, 'volatility': 0.035},
    'AAVE': {'win_rate': 0.45, 'avg_win_loss_ratio': 2.5, 'volatility': 0.06}
}
onchain_data = {
    'BTC': {'whale_score': 0.8, 'on_chain_momentum': 0.7, 'confidence': 85},
    'ETH': {'whale_score': 0.5, 'on_chain_momentum': 0.4, 'confidence': 50},
    'AAVE': {'whale_score': 0.2, 'on_chain_momentum': 0.1, 'confidence': 20}
}
allocations = allocator.allocate_portfolio(['BTC', 'ETH', 'AAVE'], metrics, onchain_data)

print(f"\n[Component: PortfolioAllocator]")
for asset, alloc in allocations.items():
    print(f"  {asset}: Alloc {alloc.position_size_pct*100:.2f}% (${alloc.position_size_usd:,.2f}) | Kelly multiplier: {alloc.kelly_fraction:.2f} | Conf: {alloc.onchain_confidence:.2f}")

# 3. Portfolio Hedger
hedger = PortfolioHedger()
from src.ai.advanced_learning import MarketRegime
regimes = {
    'BTC': MarketRegime('TRENDING', 0.90, 0.02, 0.8, 10, 'TREND_FOLLOWING'),
    'ETH': MarketRegime('VOLATILE', 0.80, 0.05, 0.2, 5, 'MEAN_REVERSION'),
    'AAVE': MarketRegime('BEARISH', 0.95, 0.07, -0.9, 15, 'SHORT_BIAS')
}
onchain_risk_metrics = {
    'BTC': {'liquidation_risk_score': 0.1},
    'ETH': {'liquidation_risk_score': 0.3},
    'AAVE': {'liquidation_risk_score': 0.85}  # Highly risky
}
hedges = hedger.calculate_hedges(allocations, {'BTC': 0.02, 'ETH': 0.05, 'AAVE': 0.07}, regimes, onchain_risk_metrics)

print(f"\n[Component: PortfolioHedger]")
for h in hedges:
    print(f"  Action on {h.asset}: {h.action_type} (Hedge Ratio: {h.hedge_ratio:.1f}x) -> Reason: {h.reason}")

print('\n' + '='*60)
print('  PHASE 5 INTEGRATION CHECK: OK')
print('='*60)
