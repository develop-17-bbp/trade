from src.models.numerical_models import L1SignalEngine
import inspect

# Get source code
source = inspect.getsource(L1SignalEngine.generate_signals)

# Find the return statement
lines = source.split('\n')
for i, line in enumerate(lines):
    if "'forecast_signal'" in line:
        print(f"Line {i}: {repr(line)}")
        
# Also print the actual bytecode to see what it does
import dis
print("\n\nBytecode for generate_signals:")
dis.dis(L1SignalEngine.generate_signals)
