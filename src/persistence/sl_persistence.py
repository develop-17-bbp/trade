"""
SL Crash Persistence — Survive Bot Crashes with Open Positions
===============================================================
Robinhood has NO exchange-native stop-loss orders for crypto.
If the bot crashes with an open position, nothing protects you.

This module persists SL state to disk on EVERY L-level update,
enabling instant recovery on restart.

Architecture:
  Every SL update → atomic write to sl_state.json
  Bot startup → check for orphaned positions → resume SL monitoring
  Position close → clear state file

The atomic write (tmp file + os.replace) prevents half-written files
even if the crash happens during the save itself.
"""

import json
import os
import time
import threading
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SL_STATE_DIR = os.path.join(PROJECT_ROOT, 'data')
SL_STATE_FILE = os.path.join(SL_STATE_DIR, 'sl_state.json')


@dataclass
class PositionState:
    """Complete snapshot of an open position's SL state."""
    asset: str
    direction: str                  # LONG or SHORT
    entry_price: float
    quantity: float
    current_sl: float
    sl_levels: List[str] = field(default_factory=lambda: ['L1'])
    peak_price: float = 0.0
    entry_time: float = 0.0
    confidence: float = 0.0
    trade_timeframe: str = '4h'
    hurst: float = 0.5
    breakeven_moved: bool = False
    order_id: str = ''


class SLPersistenceManager:
    """
    Persists open position SL state to disk for crash recovery.

    Usage:
        manager = SLPersistenceManager()

        # On every SL update:
        manager.save_position(asset, position_dict)

        # On position close:
        manager.clear_position(asset)

        # On startup:
        orphans = manager.recover_all()
        for asset, state in orphans.items():
            executor.positions[asset] = state
    """

    def __init__(self, state_file: str = SL_STATE_FILE):
        self.state_file = state_file
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

    def save_position(self, asset: str, position: Dict):
        """
        Save position state to disk. Call on EVERY SL update.
        Uses atomic write (tmp + replace) to prevent corruption.
        """
        with self._lock:
            try:
                # Load existing state (may have multiple positions)
                all_positions = self._load_raw()

                # Update this asset's state
                state = {
                    'asset': asset,
                    'direction': position.get('direction', 'LONG'),
                    'entry_price': position.get('entry_price', 0),
                    'quantity': position.get('qty', 0),
                    'current_sl': position.get('sl', 0),
                    'sl_levels': position.get('sl_levels', ['L1']),
                    'peak_price': position.get('peak_price', 0),
                    'entry_time': position.get('entry_time', 0),
                    'confidence': position.get('confidence', 0),
                    'trade_timeframe': position.get('trade_timeframe', '4h'),
                    'hurst': position.get('hurst', 0.5),
                    'breakeven_moved': position.get('breakeven_moved', False),
                    'order_id': position.get('order_id', ''),
                    'saved_at': time.time(),
                    'saved_at_iso': time.strftime('%Y-%m-%dT%H:%M:%S'),
                }
                all_positions[asset] = state

                # Atomic write
                tmp = self.state_file + '.tmp'
                with open(tmp, 'w') as f:
                    json.dump(all_positions, f, indent=2)
                os.replace(tmp, self.state_file)

                logger.debug(f"[SL-PERSIST] Saved {asset} SL=${state['current_sl']:,.2f} L{len(state['sl_levels'])}")

            except Exception as e:
                logger.error(f"[SL-PERSIST] Save failed for {asset}: {e}")

    def clear_position(self, asset: str):
        """Remove a position from state after it's closed."""
        with self._lock:
            try:
                all_positions = self._load_raw()
                if asset in all_positions:
                    del all_positions[asset]
                    tmp = self.state_file + '.tmp'
                    with open(tmp, 'w') as f:
                        json.dump(all_positions, f, indent=2)
                    os.replace(tmp, self.state_file)
                    logger.info(f"[SL-PERSIST] Cleared {asset} state")

                # If no positions left, remove file entirely
                if not all_positions and os.path.exists(self.state_file):
                    os.remove(self.state_file)

            except Exception as e:
                logger.error(f"[SL-PERSIST] Clear failed for {asset}: {e}")

    def recover_all(self) -> Dict[str, Dict]:
        """
        On startup, load all orphaned positions.
        Returns dict of {asset: position_dict} ready to inject into executor.positions.
        """
        all_positions = self._load_raw()
        if not all_positions:
            return {}

        recovered = {}
        for asset, state in all_positions.items():
            age_min = (time.time() - state.get('entry_time', time.time())) / 60
            sl_level = len(state.get('sl_levels', ['L1']))

            print(f"  [SL-RECOVER] Found orphaned {asset} position:")
            print(f"    Direction: {state.get('direction')} | Entry: ${state.get('entry_price', 0):,.2f}")
            print(f"    Current SL: ${state.get('current_sl', 0):,.2f} | Level: L{sl_level}")
            print(f"    Age: {age_min:.0f} min | Saved: {state.get('saved_at_iso', '?')}")

            # Reconstruct executor-compatible position dict
            recovered[asset] = {
                'direction': state.get('direction', 'LONG'),
                'side': 'buy' if state.get('direction') == 'LONG' else 'sell',
                'entry_price': state.get('entry_price', 0),
                'qty': state.get('quantity', 0),
                'sl': state.get('current_sl', 0),
                'sl_levels': state.get('sl_levels', ['L1']),
                'peak_price': state.get('peak_price', 0),
                'entry_time': state.get('entry_time', time.time()),
                'confidence': state.get('confidence', 0),
                'trade_timeframe': state.get('trade_timeframe', '4h'),
                'hurst': state.get('hurst', 0.5),
                'hurst_regime': 'unknown',
                'breakeven_moved': state.get('breakeven_moved', False),
                'order_id': state.get('order_id', ''),
                'reasoning': 'recovered from crash',
                'predicted_l_level': '?',
                'risk_score': 5,
                'bear_risk': 5,
                'is_reversal': False,
                'entry_tag': 'crash_recovery',
                'dca_count': 0,
                'rl_state': None,
                'rl_action_idx': 0,
                'agent_votes': {},
                '_recovered': True,
            }

        if recovered:
            print(f"  [SL-RECOVER] Recovered {len(recovered)} position(s) from crash state")
        return recovered

    def _load_raw(self) -> Dict:
        """Load raw state dict from disk."""
        if not os.path.exists(self.state_file):
            return {}
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def has_orphans(self) -> bool:
        """Quick check if there are any saved positions."""
        return os.path.exists(self.state_file) and os.path.getsize(self.state_file) > 2
