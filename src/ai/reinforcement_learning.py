"""
PHASE 6: Reinforcement Learning for Adaptive Strategy Discovery
==================================================================
Self-modifying algorithmic trading system using policy gradient methods
and reward optimization for optimal strategy generation.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradeAction:
    """Represents a trade action in RL environment."""
    action_type: str  # BUY, SELL, HOLD
    position_size: float  # 0-100% of portfolio
    stop_loss_pct: float
    take_profit_pct: float
    timestamp: str


@dataclass
class MarketState:
    """State representation for RL agent."""
    price: float
    returns_5m: float
    returns_15m: float
    returns_1h: float
    volatility: float
    rsi: float
    macd_signal: float
    momentum: float
    volume_ratio: float
    trend_strength: float
    zscore: float
    time_of_day: int  # 0-23
    day_of_week: int  # 0-6


class ReinforcementLearningAgent:
    """
    RL Agent that learns optimal trading policies through interaction with markets.
    Implements policy gradient optimization.
    """
    
    def __init__(self, learning_rate: float = 0.001, gamma: float = 0.99):
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        
        # Policy network weights (simplified linear policy)
        self.policy_weights = {
            "action_preferences": {
                "BUY": np.random.randn(10),
                "SELL": np.random.randn(10),
                "HOLD": np.random.randn(10)
            },
            "position_size_weights": np.random.randn(10)
        }
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=10000)
        self.episode_rewards = []
        self.action_history = []
        
    def construct_state_vector(self, market_state: MarketState) -> np.ndarray:
        """Convert market state to feature vector for policy network."""
        features = [
            np.log(market_state.price),
            market_state.returns_5m,
            market_state.returns_15m,
            market_state.returns_1h,
            market_state.volatility,
            market_state.rsi / 100,  # Normalize
            market_state.macd_signal,
            market_state.momentum,
            market_state.volume_ratio,
            market_state.trend_strength,
            market_state.zscore,
            np.sin(2 * np.pi * market_state.time_of_day / 24),  # Cyclic encoding for time
            np.cos(2 * np.pi * market_state.time_of_day / 24),
            np.sin(2 * np.pi * market_state.day_of_week / 7),   # Cyclic encoding for day
            np.cos(2 * np.pi * market_state.day_of_week / 7),
        ]
        return np.array(features)
    
    def select_action(self, market_state: MarketState, epsilon: float = 0.1) -> TradeAction:
        """
        Select action using epsilon-greedy policy:
        - With probability (1-epsilon): use learned policy
        - With probability epsilon: explore random action
        """
        state_vector = self.construct_state_vector(market_state)
        
        if np.random.rand() < epsilon:
            # Exploration: random action
            action_type = np.random.choice(["BUY", "SELL", "HOLD"])
            position_size = np.random.uniform(0, 2.5)
        else:
            # Exploitation: use learned policy
            action_scores = {}
            for action, weights in self.policy_weights["action_preferences"].items():
                action_scores[action] = np.dot(weights, state_vector[:len(weights)])
            
            action_type = max(action_scores, key=action_scores.get)
            
            # Calculate position size
            pos_score = np.dot(self.policy_weights["position_size_weights"], state_vector[:10])
            # Sigmoid to bound between 0-2.5
            position_size = 2.5 / (1 + np.exp(-pos_score))
        
        # Dynamic stop loss and take profit
        stop_loss = 2.0 + (market_state.volatility * 100)  # Wider stops in volatile markets
        take_profit = 4.0 + (market_state.volatility * 100)
        
        action = TradeAction(
            action_type=action_type,
            position_size=position_size,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            timestamp=datetime.now().isoformat()
        )
        
        return action
    
    def compute_reward(self, previous_state: MarketState, action: TradeAction, 
                      current_state: MarketState, trade_result: Dict[str, float]) -> float:
        """
        Compute reward signal for reinforcement learning.
        Combines P&L, risk, and consistency metrics.
        """
        pnl_return = trade_result.get("pnl_pct", 0)
        sharpe_component = trade_result.get("sharpe_ratio", 0) * 0.1
        drawdown_penalty = -trade_result.get("max_drawdown", 0) * 0.15
        
        # Action appropriateness
        action_bonus = 0
        if action.action_type == "BUY" and current_state.returns_5m > 0:
            action_bonus = 0.2
        elif action.action_type == "SELL" and current_state.returns_5m < 0:
            action_bonus = 0.2
        elif action.action_type == "HOLD" and abs(current_state.returns_5m) < 0.01:
            action_bonus = 0.1
        
        # Risk-adjusted reward
        risk_adjustment = 1 / (1 + abs(current_state.volatility) * 10)
        
        reward = (pnl_return * 0.5) + sharpe_component + drawdown_penalty + action_bonus
        reward *= risk_adjustment
        
        return reward
    
    def learn_from_trajectory(self, trajectory: List[Tuple[MarketState, TradeAction, float]]):
        """
        Learn from a complete trading trajectory using policy gradient.
        """
        if not trajectory:
            return
        
        # Compute returns for each step (discounted sum of future rewards)
        returns = []
        cumulative_return = 0
        for _, _, reward in reversed(trajectory):
            cumulative_return = reward + self.gamma * cumulative_return
            returns.insert(0, cumulative_return)
        
        # Normalize returns for stability
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        
        # Policy gradient update
        for i, (state, action, _) in enumerate(trajectory):
            state_vector = self.construct_state_vector(state)
            advantage = returns[i]
            
            # Update action preference weights
            action_weights = self.policy_weights["action_preferences"][action.action_type]
            action_weights += self.learning_rate * advantage * state_vector[:len(action_weights)]
            
            # Update position size weights
            self.policy_weights["position_size_weights"] += (
                self.learning_rate * advantage * (action.position_size - 1.0) * state_vector[:10]
            )
        
        logger.info(f"[RL-LEARNING] Updated policy from trajectory of length {len(trajectory)}")
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """Get summary of learned policy."""
        return {
            "learning_rate": self.learning_rate,
            "discount_factor": self.gamma,
            "action_preferences": {
                k: float(np.mean(v)) for k, v in self.policy_weights["action_preferences"].items()
            },
            "position_size_weight_magnitude": float(np.linalg.norm(self.policy_weights["position_size_weights"])),
            "experience_buffer_size": len(self.experience_buffer),
            "timestamp": datetime.now().isoformat()
        }


class AdaptiveAlgorithmLayer:
    """
    Self-modifying algorithm layer that adapts to market conditions.
    Learns and adjusts trading logic in real-time.
    """
    
    def __init__(self):
        self.rl_agent = ReinforcementLearningAgent()
        self.algorithm_variants = {
            "aggressive": {},
            "conservative": {},
            "neutral": {}
        }
        self.current_variant = "neutral"
        self.variant_performance = {"aggressive": 0, "conservative": 0, "neutral": 0}
        self.algorithm_history = []
        
    def adapt_to_market_conditions(self, market_metrics: Dict[str, float], 
                                  onchain_signals: Optional[Dict[str, float]] = None):
        """
        Dynamically select algorithm variant based on market conditions.
        """
        volatility = market_metrics.get("volatility", 0.03)
        trend_strength = abs(market_metrics.get("trend_strength", 0))
        momentum = market_metrics.get("momentum", 0)
        
        # Incorporate onchain signals
        onchain_influence = 0.0
        if onchain_signals:
            whale_score = onchain_signals.get("whale_score", 0.0)
            onchain_momentum = onchain_signals.get("on_chain_momentum", 0.0)
            liquidation_risk = onchain_signals.get("liquidation_risk", 50.0)
            
            # Whale activity and onchain momentum influence decision
            onchain_influence = (abs(whale_score) + abs(onchain_momentum)) * 0.5
            
            # High liquidation risk suggests more conservative approach
            if liquidation_risk > 70:
                volatility += 0.02  # Treat as higher volatility
        
        # Decision logic for variant selection (enhanced with onchain)
        if volatility > 0.08 or (onchain_signals and onchain_signals.get("liquidation_risk", 50) > 70):
            # High volatility or liquidation risk: use conservative approach
            new_variant = "conservative"
        elif volatility < 0.02 and trend_strength > 0.05 and onchain_influence > 0.3:
            # Low volatility with strong trend and onchain support: aggressive
            new_variant = "aggressive"
        else:
            # Neutral conditions
            new_variant = "neutral"
        
        if new_variant != self.current_variant:
            logger.info(f"[ADAPTIVE-ALGORITHM] Switching from {self.current_variant} to {new_variant}")
            self.algorithm_history.append({
                "from": self.current_variant,
                "to": new_variant,
                "reason": f"volatility={volatility:.4f}, trend={trend_strength:.4f}",
                "timestamp": datetime.now().isoformat()
            })
            self.current_variant = new_variant
        
        return self.get_algorithm_config()
    
    def get_algorithm_config(self) -> Dict[str, Any]:
        """Get current algorithm configuration."""
        configs = {
            "aggressive": {
                "position_size_mult": 2.0,
                "profit_target_mult": 3.0,
                "stop_loss_mult": 1.5,
                "entry_threshold": 0.6,
                "risk_limit_daily": 4.0,
                "max_concurrent_positions": 5
            },
            "conservative": {
                "position_size_mult": 0.5,
                "profit_target_mult": 1.5,
                "stop_loss_mult": 2.0,
                "entry_threshold": 0.8,
                "risk_limit_daily": 1.5,
                "max_concurrent_positions": 2
            },
            "neutral": {
                "position_size_mult": 1.0,
                "profit_target_mult": 2.0,
                "stop_loss_mult": 1.8,
                "entry_threshold": 0.7,
                "risk_limit_daily": 2.5,
                "max_concurrent_positions": 3
            }
        }
        return configs[self.current_variant]
    
    def record_performance(self, variant: str, trade_result: Dict[str, float]):
        """Track variant performance for adaptive selection."""
        current_perf = self.variant_performance[variant]
        new_perf = trade_result.get("pnl_pct", 0) + trade_result.get("sharpe_ratio", 0) * 0.1
        self.variant_performance[variant] = current_perf * 0.8 + new_perf * 0.2  # EMA
    
    def suggest_algorithm_improvement(self, recent_trades: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Analyze recent trades and suggest algorithm improvements.
        """
        if not recent_trades:
            return {"suggestion": "No trades to analyze", "change": None}
        
        win_rate = sum(1 for t in recent_trades if t.get("pnl_pct", 0) > 0) / len(recent_trades)
        avg_win = np.mean([t.get("pnl_pct", 0) for t in recent_trades if t.get("pnl_pct", 0) > 0]) if win_rate > 0 else 0
        avg_loss = abs(np.mean([t.get("pnl_pct", 0) for t in recent_trades if t.get("pnl_pct", 0) < 0])) if (1 - win_rate) > 0 else 0
        profit_factor = (avg_win * win_rate) / (avg_loss * (1 - win_rate)) if avg_loss > 0 else 1
        
        improvement = None
        if win_rate < 0.45:
            improvement = "Reduce position sizes and tighten entry criteria"
        elif profit_factor < 1.5:
            improvement = "Increase take profit targets or reduce position size to improve risk-reward"
        elif len(recent_trades) < 10:
            improvement = "Gather more samples before suggesting changes"
        else:
            improvement = "Algorithm performing well - no changes suggested"
        
        return {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "suggestion": improvement,
            "timestamp": datetime.now().isoformat()
        }


class SelfModifyingStrategyEngine:
    """
    Creates and evolves trading strategies using genetic algorithms and
    reinforcement learning. Strategies modify themselves based on performance.
    """
    
    def __init__(self, population_size: int = 10):
        self.population_size = population_size
        self.population = self._initialize_population()
        self.fitness_history = []
        self.generation = 0
        
    def _initialize_population(self) -> List[Dict[str, Any]]:
        """Initialize random population of strategies."""
        population = []
        for _ in range(self.population_size):
            strategy = {
                "id": np.random.randint(0, 1000000),
                "params": {
                    "rsi_period": int(np.random.uniform(7, 28)),
                    "rsi_buy": int(np.random.uniform(20, 40)),
                    "rsi_sell": int(np.random.uniform(60, 80)),
                    "ma_fast": int(np.random.uniform(5, 20)),
                    "ma_slow": int(np.random.uniform(30, 100)),
                    "atr_mult": np.random.uniform(1.5, 3.0),
                },
                "fitness": 0.0,
                "age": 0
            }
            population.append(strategy)
        return population
    
    def evaluate_population(self, backtest_results: Dict[int, Dict[str, float]]):
        """Evaluate fitness of all strategies in population."""
        for strategy in self.population:
            strategy_id = strategy["id"]
            if strategy_id in backtest_results:
                result = backtest_results[strategy_id]
                # Fitness = Sharpe ratio + return bonus - drawdown penalty
                fitness = (
                    result.get("sharpe_ratio", 0) * 0.4 +
                    min(result.get("total_return", 0), 0.5) * 0.3 +
                    (1 - min(result.get("max_drawdown", 0.5), 0.5)) * 0.3
                )
                strategy["fitness"] = fitness
            
            strategy["age"] += 1
    
    def selection(self) -> List[Dict[str, Any]]:
        """Tournament selection - select best strategies."""
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda x: x["fitness"], reverse=True)
        # Keep top 40%
        elite = sorted_pop[:max(1, int(0.4 * len(sorted_pop)))]
        return elite
    
    def crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover to create offspring."""
        child = {
            "id": np.random.randint(0, 1000000),
            "params": {},
            "fitness": 0.0,
            "age": 0
        }
        
        # Inherit parameters from both parents
        for param in parent1["params"]:
            if np.random.rand() < 0.5:
                child["params"][param] = parent1["params"][param]
            else:
                child["params"][param] = parent2["params"][param]
        
        return child
    
    def mutate(self, strategy: Dict, mutation_rate: float = 0.1):
        """Randomly mutate strategy parameters."""
        for param in strategy["params"]:
            if np.random.rand() < mutation_rate:
                value = strategy["params"][param]
                if isinstance(value, int):
                    strategy["params"][param] = int(value + np.random.randn() * 5)
                else:
                    strategy["params"][param] = value + np.random.randn() * 0.2
    
    def evolve_population(self, backtest_results: Dict[int, Dict[str, float]]):
        """Evolve population using genetic algorithm."""
        # Evaluate current population
        self.evaluate_population(backtest_results)
        self.fitness_history.append(max([s["fitness"] for s in self.population]))
        
        # Selection
        elite = self.selection()
        if not elite:
            elite = self.population[:1]
        
        # Create new population
        new_population = elite.copy()
        
        while len(new_population) < self.population_size:
            # Crossover
            parent1 = np.random.choice(elite)
            parent2 = np.random.choice(elite)
            child = self.crossover(parent1, parent2)
            
            # Mutate
            self.mutate(child)
            
            new_population.append(child)
        
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        logger.info(f"[GENETIC-ALGORITHM] Generation {self.generation}: Best fitness = {self.fitness_history[-1]:.4f}")
    
    def get_best_strategy(self) -> Dict[str, Any]:
        """Return best strategy from current population."""
        return max(self.population, key=lambda x: x["fitness"])
    
    def get_diversity_metrics(self) -> Dict[str, float]:
        """Compute population diversity metrics."""
        if not self.population:
            return {}
        
        # Parameter variance across population
        param_variance = {}
        all_params = self.population[0]["params"].keys()
        
        for param in all_params:
            values = [s["params"][param] for s in self.population]
            param_variance[param] = float(np.std(values))
        
        return {
            "mean_variance": float(np.mean(list(param_variance.values()))),
            "param_variances": param_variance,
            "generation": self.generation
        }

