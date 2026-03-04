from src.models.numerical_models import L1SignalEngine

cfg = {'forecast': {'use_lgbm': False, 'use_fingpt': False}}
engine = L1SignalEngine(cfg)
series = [float(i) for i in range(100, 150)]
result = engine.generate_signals(series)
print('Keys in result:')
for key in sorted(result.keys()):
    print(f'  {key}')
print(f'\nTotal keys: {len(result)}')
print(f'Has forecast_signal: {"forecast_signal" in result}')
