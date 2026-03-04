"""
Hyperparameter Optimization Service (Fine-Tuning)
================================================
Uses Optuna to find the best LightGBM parameters for current market conditions.
Implements Walk-Forward Validation to prevent overfitting.
"""
import optuna
import numpy as np
import lightgbm as lgb
from typing import List, Dict, Tuple, Any
from sklearn.model_selection import TimeSeriesSplit

class ModelTuner:
    def __init__(self, n_trials: int = 50):
        self.n_trials = n_trials

    def tune_lightgbm(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Runs Bayesian optimization to find optimal LightGBM hyperparameters.
        """
        def objective(trial):
            param = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            }

            # TimeSeriesSplit for Walk-Forward Validation
            tscv = TimeSeriesSplit(n_splits=3)
            losses = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                dtrain = lgb.Dataset(X_train, label=y_train)
                dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

                gbm = lgb.train(param, dtrain, num_boost_round=100, 
                                valid_sets=[dval], 
                                callbacks=[lgb.early_stopping(stopping_rounds=10)])
                
                losses.append(gbm.best_score['valid_0']['multi_logloss'])

            return np.mean(losses)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        print(f"[OPTUNA] Best value: {study.best_value}")
        print(f"[OPTUNA] Best params: {study.best_params}")
        
        return study.best_params

    def optimize_meta_weights(self, 
                              component_returns: Dict[str, List[float]], 
                              target_metric: str = 'sharpe') -> Dict[str, float]:
        """
        Finds the best ensemble weights for sub-strategies.
        """
        def objective(trial):
            weights = {name: trial.suggest_float(name, 0.0, 1.0) for name in component_returns.keys()}
            total = sum(weights.values())
            if total == 0: return -1e6
            
            # Normalize
            weights = {k: v/total for k, v in weights.items()}
            
            # Weighted returns
            portfolio_ret = np.zeros_like(next(iter(component_returns.values())))
            for name, rets in component_returns.items():
                portfolio_ret += np.array(rets) * weights[name]
            
            if target_metric == 'sharpe':
                vol = np.std(portfolio_ret)
                return np.mean(portfolio_ret) / vol if vol > 0 else 0
            return np.sum(portfolio_ret)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30)
        
        total = sum(study.best_params.values())
        return {k: v/total for k, v in study.best_params.items()}
