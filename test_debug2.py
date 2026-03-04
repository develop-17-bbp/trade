import sys
sys.path.insert(0, '/Users/convo/trade')

# Force reload
if 'src' in sys.modules:
    import importlib
    del sys.modules['src.models.numerical_models']
    del sys.modules['src.models']
    del sys.modules['src']

from src.models.numerical_models import L1SignalEngine

# Create a custom version that prints what it's returning
original_init = L1SignalEngine.__init__

def new_init(self, cfg=None):
    original_init(self, cfg)
    
# Monkey patch to see what's happening
original_generate = L1SignalEngine.generate_signals

def debug_generate(self, closes, highs=None, lows=None, volumes=None):
    result = original_generate(self, closes, highs, lows, volumes)
    print(f"DEBUG: Result keys = {list(result.keys())}")
    return result

L1SignalEngine.generate_signals = debug_generate

cfg = {'forecast': {'use_lgbm': False, 'use_fingpt': False}}
engine = L1SignalEngine(cfg)
series = [float(i) for i in range(100, 150)]
result = engine.generate_signals(series)
print(f"Has forecast_signal: {'forecast_signal' in result}")
if 'forecast_signal' in result:
    print(f"forecast_signal length: {len(result['forecast_signal'])}")
