"""
Robinhood Crypto API — Read-Only Client
========================================
Official Robinhood Crypto Trading API (https://docs.robinhood.com/crypto/trading/)
ED25519 signature authentication, read-only endpoints only.

Endpoints:
  - Account info & buying power
  - Holdings (BTC, ETH, etc.)
  - Best bid/ask quotes (real-time)
  - Estimated execution price
  - Trading pairs metadata
  - Order history (read-only)

Usage:
    client = RobinhoodCryptoClient()
    if client.authenticated:
        print(client.get_account())
        print(client.get_best_price("BTC-USD"))
        print(client.get_holdings())
"""

import os
import time
import json
import base64
import logging
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

BASE_URL = "https://trading.robinhood.com"

# Try to import nacl for ED25519 signing
try:
    from nacl.signing import SigningKey
    NACL_AVAILABLE = True
except ImportError:
    NACL_AVAILABLE = False
    logger.warning("pynacl not installed — run: pip install pynacl")


class RobinhoodCryptoClient:
    """
    Read-only client for Robinhood Crypto Trading API.
    Uses ED25519 signature authentication.
    """

    def __init__(self, api_key: str = None, private_key_b64: str = None):
        """
        Initialize with API credentials.

        Args:
            api_key: Robinhood API key (or from ROBINHOOD_API_KEY env)
            private_key_b64: Base64-encoded ED25519 private key (or from ROBINHOOD_PRIVATE_KEY env)
        """
        self.api_key = api_key or os.environ.get("ROBINHOOD_API_KEY", "")
        self._private_key_b64 = private_key_b64 or os.environ.get("ROBINHOOD_PRIVATE_KEY", "")
        self._signing_key = None
        self.authenticated = False
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        self._last_request = 0
        self._min_interval = 3.0  # rate limit: ~1 req/3s (Robinhood undocumented, 403 if too fast)

        # Init signing key
        if self.api_key and self._private_key_b64 and NACL_AVAILABLE:
            try:
                seed = base64.b64decode(self._private_key_b64)
                self._signing_key = SigningKey(seed)
                self.authenticated = True
                logger.info("[Robinhood] ED25519 signing key loaded")
            except Exception as e:
                logger.error(f"[Robinhood] Failed to load signing key: {e}")
                self.authenticated = False
        elif not NACL_AVAILABLE:
            logger.warning("[Robinhood] pynacl not installed")
        elif not self.api_key:
            logger.info("[Robinhood] No API key configured (set ROBINHOOD_API_KEY)")

    def _sign_request(self, path: str, method: str, body: str = "") -> Dict[str, str]:
        """Build signed headers for a request."""
        timestamp = int(datetime.now(tz=timezone.utc).timestamp())
        message = f"{self.api_key}{timestamp}{path}{method}{body}"
        signed = self._signing_key.sign(message.encode("utf-8"))
        return {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
            "x-timestamp": str(timestamp),
        }

    def _throttle(self):
        """Simple rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    def _get(self, path: str, params: Dict = None) -> Optional[Dict]:
        """Signed GET request."""
        if not self.authenticated:
            logger.warning("[Robinhood] Not authenticated")
            return None

        self._throttle()

        # Build full path with query string for signature
        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            sign_path = f"{path}?{qs}"
        else:
            sign_path = path

        headers = self._sign_request(sign_path, "GET")
        url = f"{BASE_URL}{path}"

        try:
            resp = self._session.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.warning(f"[Robinhood] GET {path} → {resp.status_code}: {resp.text[:200]}")
                return {"error": resp.status_code, "detail": resp.text[:500]}
        except requests.exceptions.Timeout:
            logger.warning(f"[Robinhood] Timeout: GET {path}")
            return None
        except Exception as e:
            logger.error(f"[Robinhood] Request failed: {e}")
            return None

    # ── Account ──

    def get_account(self) -> Optional[Dict]:
        """Get account info (account_number, status, buying_power)."""
        return self._get("/api/v1/crypto/trading/accounts/")

    # ── Holdings ──

    def get_holdings(self, assets: List[str] = None) -> Optional[Dict]:
        """
        Get crypto holdings.

        Args:
            assets: Optional filter e.g. ["BTC", "ETH"]
        """
        params = {}
        if assets:
            # Robinhood expects repeated asset_code params
            params = {"asset_code": ",".join(assets)}
        return self._get("/api/v1/crypto/trading/holdings/", params=params if params else None)

    # ── Market Data ──

    def get_best_price(self, symbol: str = None) -> Optional[Dict]:
        """
        Get best bid/ask prices.

        Args:
            symbol: e.g. "BTC-USD" or None for all
        """
        params = {"symbol": symbol} if symbol else None
        return self._get("/api/v1/crypto/marketdata/best_bid_ask/", params=params)

    def get_estimated_price(self, symbol: str, side: str = "both",
                            quantity: float = 1.0) -> Optional[Dict]:
        """
        Get estimated execution price for a quantity.

        Args:
            symbol: e.g. "BTC-USD"
            side: "bid", "ask", or "both"
            quantity: amount to estimate
        """
        params = {"symbol": symbol, "side": side, "quantity": str(quantity)}
        return self._get("/api/v1/crypto/marketdata/estimated_price/", params=params)

    # ── Trading Pairs ──

    def get_trading_pairs(self, symbols: List[str] = None) -> Optional[Dict]:
        """
        Get available trading pairs.

        Args:
            symbols: Optional filter e.g. ["BTC-USD", "ETH-USD"]
        """
        params = {}
        if symbols:
            params = {"symbol": ",".join(symbols)}
        return self._get("/api/v1/crypto/trading/trading_pairs/",
                         params=params if params else None)

    # ── Orders (read-only) ──

    def get_orders(self, symbol: str = None, state: str = None,
                   limit: int = 20) -> Optional[Dict]:
        """
        Get order history (read-only).

        Args:
            symbol: e.g. "BTC-USD"
            state: "open", "filled", "canceled", "failed"
            limit: max results
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        if state:
            params["state"] = state
        if limit:
            params["limit"] = str(limit)
        return self._get("/api/v1/crypto/trading/orders/",
                         params=params if params else None)

    def get_order(self, order_id: str) -> Optional[Dict]:
        """Get a specific order by ID."""
        return self._get(f"/api/v1/crypto/trading/orders/{order_id}/")

    # ── Convenience ──

    def get_btc_price(self) -> Optional[float]:
        """Quick helper: get BTC mid price."""
        data = self.get_best_price("BTC-USD")
        if data and "results" in data and data["results"]:
            r = data["results"][0]
            bid = float(r.get("bid_inclusive_of_sell_spread", 0))
            ask = float(r.get("ask_inclusive_of_buy_spread", 0))
            if bid and ask:
                return (bid + ask) / 2
        return None

    def get_eth_price(self) -> Optional[float]:
        """Quick helper: get ETH mid price."""
        data = self.get_best_price("ETH-USD")
        if data and "results" in data and data["results"]:
            r = data["results"][0]
            bid = float(r.get("bid_inclusive_of_sell_spread", 0))
            ask = float(r.get("ask_inclusive_of_buy_spread", 0))
            if bid and ask:
                return (bid + ask) / 2
        return None

    def status(self) -> str:
        """Return a quick status string."""
        if not NACL_AVAILABLE:
            return "UNAVAILABLE (pip install pynacl)"
        if not self.authenticated:
            return "NOT CONFIGURED (set ROBINHOOD_API_KEY + ROBINHOOD_PRIVATE_KEY)"
        # Test with account endpoint
        acct = self.get_account()
        if acct and "error" not in acct:
            return f"CONNECTED | account={acct.get('account_number', '?')} | status={acct.get('status', '?')} | buying_power=${acct.get('buying_power', '?')}"
        return f"AUTH ERROR: {acct}"


class RobinhoodPaperTracker:
    """
    Paper trading layer on top of read-only Robinhood API.

    Logs our system's signals alongside real Robinhood prices to measure
    how the strategy would perform on real market data (bid/ask spread included).

    No orders are placed — purely observational.
    """

    def __init__(self, client: RobinhoodCryptoClient, log_path: str = None):
        self.client = client
        self.log_path = log_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "logs", "robinhood_paper.jsonl"
        )
        self.signals: List[Dict] = []

    def record_signal(self, asset: str, direction: str, score: int,
                      ml_confidence: float = None, llm_confidence: float = None):
        """
        Record a trading signal and snapshot the real Robinhood price.

        Args:
            asset: "BTC" or "ETH"
            direction: "LONG" or "SHORT"
            score: entry score from strategy
            ml_confidence: ML model confidence
            llm_confidence: LLM confidence
        """
        symbol = f"{asset}-USD"
        price_data = self.client.get_best_price(symbol)

        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "asset": asset,
            "direction": direction,
            "score": score,
            "ml_confidence": ml_confidence,
            "llm_confidence": llm_confidence,
        }

        if price_data and "results" in price_data and price_data["results"]:
            r = price_data["results"][0]
            entry["bid"] = r.get("bid_inclusive_of_sell_spread")
            entry["ask"] = r.get("ask_inclusive_of_buy_spread")
            entry["spread_buy"] = r.get("buy_spread")
            entry["spread_sell"] = r.get("sell_spread")
            entry["rh_timestamp"] = r.get("timestamp")
        else:
            entry["bid"] = None
            entry["ask"] = None
            entry["error"] = "price_fetch_failed"

        self.signals.append(entry)

        # Append to JSONL log
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"[RobinhoodPaper] Log write failed: {e}")

        return entry

    def summary(self) -> Dict:
        """Quick summary of recorded signals."""
        if not self.signals:
            return {"total_signals": 0}
        return {
            "total_signals": len(self.signals),
            "long_signals": sum(1 for s in self.signals if s["direction"] == "LONG"),
            "short_signals": sum(1 for s in self.signals if s["direction"] == "SHORT"),
            "with_price": sum(1 for s in self.signals if s.get("bid") is not None),
            "price_errors": sum(1 for s in self.signals if s.get("error")),
        }
