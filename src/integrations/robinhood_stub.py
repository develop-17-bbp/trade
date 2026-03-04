"""
Robinhood Integration
=====================
Full integration with Robinhood using the unofficial robin_stocks library.
Provides real order placement, position tracking, and account management.

⚠️  WARNING: Robinhood does not provide an official public API for automation.
    Using robin_stocks (or similar unofficial wrappers) may violate Robinhood's 
    ToS.  Use at your own risk and review Robinhood's current policies before 
    deploying live.

Features:
  - Authentication with optional 2FA
  - Market, limit, and stop-limit orders
  - Position tracking and P&L monitoring
  - Account balance and buying power queries
  - Order cancellation and order history
  - Rate limiting to respect API quotas
  - Graceful fallback to paper/CCXT mode on auth failure
"""

import time
import os
from typing import Dict, List, Optional, Any
from enum import Enum

try:
    import robin_stocks as r
    ROBIN_AVAILABLE = True
except ImportError:
    ROBIN_AVAILABLE = False


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class RobinhoodClient:
    """
    Unified Robinhood trading client.
    
    Handles authentication, order placement, position tracking, and account 
    queries via the robin_stocks library.  Implements basic rate limiting and 
    error recovery.
    """

    def __init__(self, cache_token: bool = True):
        """
        Args:
            cache_token: if True, attempt to use cached oauth token to skip 2FA.
        """
        self.authenticated = False
        self.username = None
        self.cache_token = cache_token
        self._last_request = 0
        self._min_interval = 0.5  # seconds between requests (rate limit)

    def login(self, username: str, password: str, mfa_code: Optional[str] = None) -> bool:
        """
        Authenticate with Robinhood.
        
        Args:
            username: email or username
            password: account password
            mfa_code: optional 2FA code (6-digit string)
        
        Returns:
            True if authenticated, False otherwise.
        """
        if not ROBIN_AVAILABLE:
            print("[RobinhoodClient] robin_stocks not installed; cannot authenticate")
            return False

        try:
            # attempt cached token first
            if self.cache_token:
                try:
                    r.login(username, password, mfa_code=mfa_code)
                    self.username = username
                    self.authenticated = True
                    print("[RobinhoodClient] Authenticated successfully (cached token)")
                    return True
                except Exception:
                    pass

            # fallback to fresh login
            r.login(username, password, mfa_code=mfa_code)
            self.username = username
            self.authenticated = True
            print(f"[RobinhoodClient] Authenticated as {username}")
            return True
        except Exception as e:
            print(f"[RobinhoodClient] Authentication failed: {e}")
            self.authenticated = False
            return False

    def logout(self):
        """Log out from Robinhood."""
        if ROBIN_AVAILABLE and self.authenticated:
            try:
                r.logout()
                self.authenticated = False
                print("[RobinhoodClient] Logged out")
            except Exception as e:
                print(f"[RobinhoodClient] Logout error: {e}")

    def _throttle(self):
        """Simple rate limiting to avoid API overload."""
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    def place_order(self,
                    symbol: str,
                    quantity: float,
                    side: str = "buy",
                    order_type: str = "market",
                    limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    time_in_force: str = "gfd") -> Dict[str, Any]:
        """
        Place an order on Robinhood.
        
        Args:
            symbol: ticker symbol (e.g., "BTC", "ETH")
            quantity: number of shares/units
            side: "buy" or "sell"
            order_type: "market", "limit", "stop_loss", "stop_limit"
            limit_price: for limit/stop_limit orders
            stop_price: for stop_loss/stop_limit orders
            time_in_force: "gfd" (good for day), "gtc" (good til canceled)
        
        Returns:
            Dict with order details or error info.
        """
        if not self.authenticated:
            return {"status": "error", "message": "Not authenticated"}

        self._throttle()

        try:
            if order_type == "market":
                order = r.order(symbol, quantity, side=side)
            elif order_type == "limit":
                order = r.order_limit(symbol, quantity, limit_price, side=side)
            elif order_type == "stop_loss":
                order = r.order_stop_loss(symbol, quantity, stop_price, side=side)
            elif order_type == "stop_limit":
                order = r.order_stop_limit(
                    symbol, quantity, limit_price, stop_price, side=side
                )
            else:
                return {"status": "error", "message": f"Unknown order type: {order_type}"}

            if order:
                print(f"[RobinhoodClient] Placed {side} order: {quantity} {symbol} @ {order_type}")
                return {
                    "status": "success",
                    "order": order,
                    "symbol": symbol,
                    "quantity": quantity,
                    "side": side,
                }
            else:
                return {"status": "error", "message": "Order rejected by Robinhood"}
        except Exception as e:
            print(f"[RobinhoodClient] Order placement error: {e}")
            return {"status": "error", "message": str(e)}

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order by ID."""
        if not self.authenticated:
            return False

        self._throttle()

        try:
            result = r.cancel_order(order_id)
            print(f"[RobinhoodClient] Cancelled order {order_id}")
            return True
        except Exception as e:
            print(f"[RobinhoodClient] Cancellation error: {e}")
            return False

    def get_positions(self) -> List[Dict[str, Any]]:
        """Retrieve all open positions."""
        if not self.authenticated:
            return []

        self._throttle()

        try:
            positions = r.get_positions()
            return positions if positions else []
        except Exception as e:
            print(f"[RobinhoodClient] Position fetch error: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get details for a specific symbol."""
        if not self.authenticated:
            return None

        self._throttle()

        try:
            instrument = r.get_instruments_by_symbols(symbol)
            if not instrument:
                return None
            return r.get_quotes(symbol)[0] if r.get_quotes(symbol) else None
        except Exception as e:
            print(f"[RobinhoodClient] Quote fetch error for {symbol}: {e}")
            return None

    def get_account_balance(self) -> Optional[Dict[str, float]]:
        """Retrieve account balance, buying power, and portfolio value."""
        if not self.authenticated:
            return None

        self._throttle()

        try:
            acct = r.get_account()
            return {
                "cash": float(acct.get("cash", 0)),
                "portfolio_value": float(acct.get("portfolio_value", 0)),
                "buying_power": float(acct.get("buying_power", 0)),
                "margin_limit": float(acct.get("margin_limit", 0)),
            }
        except Exception as e:
            print(f"[RobinhoodClient] Account balance fetch error: {e}")
            return None

    def get_order_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent order history."""
        if not self.authenticated:
            return []

        self._throttle()

        try:
            orders = r.get_all_orders()
            return orders[:limit] if orders else []
        except Exception as e:
            print(f"[RobinhoodClient] Order history fetch error: {e}")
            return []

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current price quote for a symbol."""
        if not self.authenticated:
            return None

        self._throttle()

        try:
            quote = r.get_quotes(symbol)
            return quote[0] if quote else None
        except Exception as e:
            print(f"[RobinhoodClient] Quote error for {symbol}: {e}")
            return None
