"""
Smart order router
===================
Routes orders across multiple venues based on latency and rate-limit models.
This stub simply prints the order and returns a mock fill.
"""

from typing import Dict, Any


def route_order(symbol: str, quantity: float, side: str = "buy") -> Dict[str, Any]:
    print(f"[SmartRouter] Routing {side} {quantity} of {symbol}")
    return {"status": "filled", "price": None, "quantity": quantity}
