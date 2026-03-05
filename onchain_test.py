from src.data.on_chain_fetcher import OnChainFetcher
import json

f = OnChainFetcher()
print('onchain output repr:', repr(f.get_market_context('BTC')))

try:
    print('json dumps success', json.dumps(f.get_market_context('BTC')))
except Exception as e:
    print('json dumps error', e)
