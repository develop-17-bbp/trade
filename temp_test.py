from src.ai.agentic_strategist import AgenticStrategist
import traceback

s = AgenticStrategist()
print('Instance created')

data = {'atr':0.0,'funding_rate':0.0,'onchain':{},'asset':'BTC','sentiment':{'bullish':0.0,'bearish':0.0}}

try:
    dec = s.analyze_performance([],{},data)
    print('Decision',dec)
except Exception as e:
    traceback.print_exc()
