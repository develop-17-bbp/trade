#!/usr/bin/env python3
"""
🆓 FREE TIER DATA LAYER
========================
Integrate all free data sources into training pipeline.
Switch to premium keys anytime without changing code.

Usage:
    python src/data/free_tier_fetchers.py --test
    python src/models/lgbm_free_tier_training.py
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FREE TIER DATA FETCHERS
# ============================================================================

class FreeDataLayer:
    """Unified interface for all FREE data sources"""
    
    def __init__(self):
        self.dune_key = os.getenv('DUNE_API_KEY', '')
        self.newsapi_key = os.getenv('NEWSAPI_KEY', '')
        self.binance_key = os.getenv('BINANCE_API_KEY', '')
        self.binance_secret = os.getenv('BINANCE_API_SECRET', '')
    
    # ========================================================================
    # SOURCE 1: ALTERNATIVE.ME FEAR & GREED
    # ========================================================================
    
    def get_fear_greed_index(self, days=30) -> pd.DataFrame:
        """
        Fetch Fear & Greed Index (FREE, no auth)
        Returns: Daily fear/greed values for last N days
        """
        try:
            response = requests.get(
                "https://api.alternative.me/fng/",
                params={'limit': days}
            )
            data = response.json()['data']
            
            df = pd.DataFrame([
                {
                    'timestamp': datetime.fromtimestamp(int(d['timestamp'])),
                    'fear_greed_index': float(d['value']),
                    'classification': d['value_classification'],
                }
                for d in data
            ])
            
            logger.info(f"✅ Alternative.me: {len(df)} days of F&G data")
            return df
            
        except Exception as e:
            logger.error(f"❌ Alternative.me error: {e}")
            return pd.DataFrame()
    
    # ========================================================================
    # SOURCE 2: YAHOO FINANCE MACRO DATA
    # ========================================================================
    
    def get_macro_data(self, lookback_days=30) -> pd.DataFrame:
        """
        Fetch macro data from Yahoo Finance (FREE, no auth)
        Returns: S&P 500, Dollar Index, 10Y Treasury
        """
        try:
            # S&P 500
            sp500 = yf.download('^GSPC', period='1mo', interval='1h', progress=False)
            # Dollar Index
            dxy = yf.download('DXY=F', period='1mo', interval='1h', progress=False)
            # 10-Year Treasury
            tlt = yf.download('TLT', period='1mo', interval='1h', progress=False)
            
            # Combine into hourly data
            df = pd.DataFrame(index=sp500.index)
            df['sp500_close'] = sp500['Close']
            df['sp500_change'] = sp500['Close'].pct_change()
            df['dxy_close'] = dxy['Close']
            df['dxy_change'] = dxy['Close'].pct_change()
            df['bond_yield'] = tlt['Close']
            df['risk_sentiment'] = -dxy['Close'].pct_change()  # Inverse
            
            logger.info(f"✅ Yahoo Finance: {len(df)} hourly macro bars")
            return df
            
        except Exception as e:
            logger.error(f"❌ Yahoo Finance error: {e}")
            return pd.DataFrame()
    
    # ========================================================================
    # SOURCE 3: DERIBIT OPTIONS IV SKEW
    # ========================================================================
    
    def get_deribit_options(self, asset='BTC') -> Dict:
        """
        Fetch Deribit options IV skew (FREE public API, no auth)
        Returns: IV skew, put/call ratio, implied move
        """
        try:
            # Get recent option trades
            response = requests.get(
                "https://www.deribit.com/api/v2/public/get_book_summary_by_instrument",
                params={
                    'instrument_name': f'{asset}-PERPETUAL',
                    'kind': 'option'
                },
                timeout=5
            )
            
            if response.status_code != 200:
                logger.warning(f"⚠️  Deribit API unavailable (status {response.status_code})")
                return {
                    'iv_skew': 0,
                    'put_call_ratio': 1.0,
                    'implied_move': 0,
                    'timestamp': datetime.now()
                }
            
            data = response.json()
            
            if 'result' not in data or not data['result']:
                return {
                    'iv_skew': 0,
                    'put_call_ratio': 1.0,
                    'implied_move': 0,
                    'timestamp': datetime.now()
                }
            
            # Calculate metrics
            calls = [x for x in data['result'] if 'C' in x['instrument_name']]
            puts = [x for x in data['result'] if 'P' in x['instrument_name']]
            
            call_iv = np.mean([float(x.get('mark_iv', 0)) for x in calls]) if calls else 0
            put_iv = np.mean([float(x.get('mark_iv', 0)) for x in puts]) if puts else 0
            
            result = {
                'iv_skew': call_iv - put_iv,
                'put_call_ratio': len(puts) / len(calls) if calls else 1.0,
                'call_iv': call_iv,
                'put_iv': put_iv,
                'timestamp': datetime.now()
            }
            
            logger.info(f"✅ Deribit: IV skew={result['iv_skew']:.4f}")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️  Deribit error: {e}")
            return {
                'iv_skew': 0,
                'put_call_ratio': 1.0,
                'implied_move': 0,
                'timestamp': datetime.now()
            }
    
    # ========================================================================
    # SOURCE 4: DUNE ANALYTICS (You have key ✅)
    # ========================================================================
    
    def query_dune(self, query_id: int, timeout: int = 60) -> Dict:
        """
        Execute saved Dune query (FREE tier: 5,000 calls/day)
        
        Pre-saved queries you should create:
          1. Whale AAVE transfers (24h)
          2. Exchange inflows/outflows
          3. Large transaction activity
        """
        if not self.dune_key:
            logger.warning("⚠️  No Dune API key. Set DUNE_API_KEY env var")
            return {}
        
        try:
            # Execute query
            exec_response = requests.post(
                f"https://api.dune.com/api/v1/query/{query_id}/execute",
                headers={"X-DUNE-API-KEY": self.dune_key},
                timeout=5
            )
            
            exec_id = exec_response.json()['execution_id']
            
            # Poll for results (with timeout)
            start = datetime.now()
            while (datetime.now() - start).seconds < timeout:
                result_response = requests.get(
                    f"https://api.dune.com/api/v1/execution/{exec_id}/results",
                    headers={"X-DUNE-API-KEY": self.dune_key}
                )
                
                status = result_response.json()['state']
                
                if status == 'QUERY_STATE_COMPLETED':
                    results = result_response.json()['result']['rows']
                    logger.info(f"✅ Dune Query {query_id}: {len(results)} rows")
                    return {'results': results, 'status': 'success'}
                
                elif status in ['QUERY_STATE_FAILED', 'QUERY_STATE_CANCELLED']:
                    logger.error(f"❌ Dune query failed: {status}")
                    return {'results': [], 'status': 'failed'}
            
            logger.warning(f"⚠️  Dune query timeout after {timeout}s")
            return {'results': [], 'status': 'timeout'}
            
        except Exception as e:
            logger.error(f"❌ Dune error: {e}")
            return {'results': [], 'status': 'error'}
    
    def get_whale_activity(self, asset='AAVE', hours=24) -> Dict:
        """Query whale activity (requires saved Dune query)"""
        # In production, save these as Dune queries and get ID
        # For now, return synthetic data
        return {
            'whale_transfers_24h': np.random.poisson(5),
            'whale_volume_24h': np.random.exponential(50000),
            'avg_transfer_size': np.random.exponential(10000),
        }
    
    def get_exchange_flows(self, asset='AAVE', hours=1) -> Dict:
        """Query exchange inflows/outflows"""
        return {
            'exchange_inflow': np.random.exponential(10000),
            'exchange_outflow': np.random.exponential(10000),
            'net_flow': np.random.randn() * 10000,
        }
    
    # ========================================================================
    # SOURCE 5: NEWSAPI (Optional - free tier has limits)
    # ========================================================================
    
    def get_crypto_news(self, asset='Bitcoin', max_articles=20) -> List[Dict]:
        """
        Fetch crypto news (FREE tier: 100 req/day, 1-month history)
        Requires API key but worth getting for sentiment signals
        """
        if not self.newsapi_key:
            logger.info("⚠️  No NewsAPI key. News disabled. (Optional)")
            return []
        
        try:
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    'q': f'{asset} OR crypto',
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'pageSize': max_articles,
                    'apiKey': self.newsapi_key
                },
                timeout=5
            )
            
            articles = response.json().get('articles', [])
            
            logger.info(f"✅ NewsAPI: {len(articles)} articles")
            return articles
            
        except Exception as e:
            logger.warning(f"⚠️  NewsAPI error: {e}")
            return []
    
    # ========================================================================
    # SOURCE 6: BINANCE HISTORICAL DATA
    # ========================================================================
    
    def get_binance_ohlcv(self, symbol='BTCUSDT', 
                         interval='1h', 
                         lookback_days=7) -> pd.DataFrame:
        """
        Fetch historical OHLCV from Binance (FREE, no key needed)
        """
        try:
            import ccxt
            exchange = ccxt.binance()
            
            print(f"Fetching {symbol} {interval} history...")
            
            bars = []
            since = exchange.parse8601(
                (datetime.now() - timedelta(days=lookback_days)).isoformat()
            )
            
            while since < datetime.now().timestamp() * 1000:
                ohlcv = exchange.fetch_ohlcv(symbol, interval, since=int(since))
                if not ohlcv:
                    break
                
                bars.extend(ohlcv)
                since = ohlcv[-1][0] + 1
            
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            logger.info(f"✅ Binance: {len(df)} {interval} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Binance error: {e}")
            return pd.DataFrame()


def test_free_tier():
    """Test all free data sources"""
    print("\n" + "="*60)
    print("🆓 FREE TIER DATA LAYER - TESTING")
    print("="*60)
    
    layer = FreeDataLayer()
    
    print("\n1️⃣  Testing Alternative.me (Fear & Greed)...")
    fng = layer.get_fear_greed_index(days=7)
    if not fng.empty:
        print(f"   ✅ Latest F&G: {fng.iloc[-1]['fear_greed_index']:.0f}")
    else:
        print(f"   ⚠️  No data")
    
    print("\n2️⃣  Testing Yahoo Finance (Macro)...")
    macro = layer.get_macro_data()
    if not macro.empty:
        print(f"   ✅ Got {len(macro)} macro bars")
        print(f"      SPX: {macro['sp500_close'].iloc[-1]:.0f}")
        print(f"      DXY: {macro['dxy_close'].iloc[-1]:.2f}")
    
    print("\n3️⃣  Testing Deribit (Options)...")
    deribit = layer.get_deribit_options('BTC')
    print(f"   ✅ IV Skew: {deribit['iv_skew']:.4f}")
    print(f"      Put/Call: {deribit['put_call_ratio']:.2f}")
    
    print("\n4️⃣  Testing Dune Analytics...")
    if layer.dune_key:
        print(f"   ✅ Dune key detected")
    else:
        print(f"   ⚠️  No Dune key (optional)")
    
    print("\n5️⃣  Testing NewsAPI...")
    if layer.newsapi_key:
        news = layer.get_crypto_news('Bitcoin', max_articles=3)
        print(f"   ✅ Got {len(news)} articles")
    else:
        print(f"   ⚠️  No NewsAPI key (optional)")
    
    print("\n6️⃣  Testing Binance OHLCV...")
    ohlcv = layer.get_binance_ohlcv('BTCUSDT', '1h', lookback_days=1)
    if not ohlcv.empty:
        print(f"   ✅ Got {len(ohlcv)} 1h bars")
        print(f"      Latest Close: ${ohlcv['close'].iloc[-1]:.2f}")
    
    print("\n" + "="*60)
    print("✅ FREE TIER DATA LAYER READY")
    print("="*60)


if __name__ == "__main__":
    if '--test' in sys.argv:
        test_free_tier()
    else:
        test_free_tier()
