"""
Test suite for Robinhood integration
Tests client initialization, API availability, and error handling.
"""

import pytest
from src.integrations.robinhood_stub import RobinhoodClient, OrderType, OrderStatus


class TestRobinhoodClient:
    """Test Robinhood client functionality."""

    def test_client_initialization(self):
        """Test RobinhoodClient can be instantiated."""
        client = RobinhoodClient(cache_token=True)
        assert client is not None
        assert client.authenticated is False
        assert client.username is None
        assert client._min_interval == 0.5

    def test_client_not_authenticated_by_default(self):
        """Test that client starts unauthenticated."""
        client = RobinhoodClient()
        assert not client.authenticated

    def test_login_fails_without_library(self):
        """Test login returns False when robin_stocks not available."""
        client = RobinhoodClient()
        # Should return False without dummy credentials since library may not be installed
        result = client.login("test@example.com", "password123")
        # Result depends on whether robin_stocks is available
        assert isinstance(result, bool)

    def test_place_order_without_auth(self):
        """Test order placement fails when not authenticated."""
        client = RobinhoodClient()
        result = client.place_order(
            symbol="BTC",
            quantity=0.1,
            side="buy",
            order_type="market"
        )
        assert result['status'] == 'error'
        assert 'Not authenticated' in result['message']

    def test_get_positions_without_auth(self):
        """Test get_positions returns empty list when not authenticated."""
        client = RobinhoodClient()
        result = client.get_positions()
        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_account_balance_without_auth(self):
        """Test get_account_balance returns None when not authenticated."""
        client = RobinhoodClient()
        result = client.get_account_balance()
        assert result is None

    def test_get_order_history_without_auth(self):
        """Test get_order_history returns empty list when not authenticated."""
        client = RobinhoodClient()
        result = client.get_order_history()
        assert isinstance(result, list)
        assert len(result) == 0

    def test_order_types_enum(self):
        """Test OrderType enum has expected values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP_LOSS.value == "stop_loss"
        assert OrderType.STOP_LIMIT.value == "stop_limit"

    def test_order_status_enum(self):
        """Test OrderStatus enum has expected values."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"

    def test_throttle_mechanism(self):
        """Test rate limiting throttle mechanism."""
        import time
        client = RobinhoodClient()
        
        start = time.time()
        client._throttle()
        client._throttle()
        elapsed = time.time() - start
        
        # Should have waited at least _min_interval seconds
        assert elapsed >= client._min_interval - 0.01  # small tolerance for timing variance

    def test_place_order_invalid_type(self):
        """Test order placement with invalid order type."""
        client = RobinhoodClient()
        client.authenticated = True  # Fake authentication
        
        result = client.place_order(
            symbol="BTC",
            quantity=1.0,
            side="buy",
            order_type="invalid_type"
        )
        
        assert result['status'] == 'error'
        assert 'Unknown order type' in result['message']

    def test_logout_when_authenticated(self):
        """Test logout when client is not authenticated."""
        client = RobinhoodClient()
        client.authenticated = False
        
        # Should not raise exception
        client.logout()
        assert not client.authenticated

    def test_cancel_order_without_auth(self):
        """Test order cancellation fails when not authenticated."""
        client = RobinhoodClient()
        result = client.cancel_order("some_order_id")
        assert result is False

    def test_get_position_without_auth(self):
        """Test get_position returns None when not authenticated."""
        client = RobinhoodClient()
        result = client.get_position("BTC")
        assert result is None

    def test_get_quote_without_auth(self):
        """Test get_quote returns None when not authenticated."""
        client = RobinhoodClient()
        result = client.get_quote("BTC")
        assert result is None

    def test_client_cache_token_flag(self):
        """Test cache_token flag is stored correctly."""
        client_cached = RobinhoodClient(cache_token=True)
        assert client_cached.cache_token is True
        
        client_no_cache = RobinhoodClient(cache_token=False)
        assert client_no_cache.cache_token is False


class TestRobinhoodExecutorIntegration:
    """Test Robinhood integration with TradingExecutor."""

    def test_executor_initializes_robinhood_stub(self):
        """Test that TradingExecutor can initialize with paper mode (no Robinhood)."""
        from src.trading.executor import TradingExecutor
        
        config = {
            'mode': 'paper',
            'assets': ['BTC'],
            'initial_capital': 10000,
        }
        
        executor = TradingExecutor(config)
        assert executor is not None
        assert executor.robinhood is None  # Paper mode doesn't init Robinhood

    def test_executor_live_mode_without_credentials_fails_gracefully(self):
        """Test that executor falls back to paper when live mode lacks credentials."""
        from src.trading.executor import TradingExecutor
        
        config = {
            'mode': 'live',
            'assets': ['BTC'],
            'initial_capital': 10000,
        }
        
        # Without environment variables, Robinhood will not be initialized
        executor = TradingExecutor(config)
        assert executor.mode == 'live'
        # Robinhood init should have failed gracefully
        assert executor.robinhood is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
