"""
Hyperparameter search harness
==============================
Provides a simple wrapper around Optuna for sweeping strategy parameters.  The
functionology is illustrative; users can extend with real evaluation code.
"""

import optuna


def objective(trial):
    # example parameters
    bias = trial.suggest_float('bias', -0.1, 0.1)
    l1_weight = trial.suggest_float('l1_weight', 0.0, 1.0)
    # placeholder evaluation using a random score
    import random
    return random.random()


def run_study(n_trials: int = 50):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
