"""
PHASE 5: Trading Journal
========================
Maintains a rich, persistent log of all trades for post-trade analysis.
Tracks entry/exit reasons, market regime at time of trade, and alpha decay.
Supports optional AES encryption for at-rest security.
"""

import json
import os
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use centralized path management; fall back to legacy in-repo paths for backward compatibility
try:
    from src.core.paths import TRADING_JOURNAL_FILE as _JOURNAL_ENC_PATH, get_log_path as _get_log_path
    ENCRYPTED_JOURNAL_FILE = str(_JOURNAL_ENC_PATH)
    JOURNAL_FILE = str(_get_log_path("trading_journal.json"))
    # During transition: if old path exists but new path doesn't, use old path
    _legacy_enc = "logs/trading_journal.enc"
    _legacy_plain = "logs/trading_journal.json"
    if not _JOURNAL_ENC_PATH.exists() and os.path.exists(_legacy_enc):
        ENCRYPTED_JOURNAL_FILE = _legacy_enc
    if not _get_log_path("trading_journal.json").exists() and os.path.exists(_legacy_plain):
        JOURNAL_FILE = _legacy_plain
except Exception:
    JOURNAL_FILE = "logs/trading_journal.json"
    ENCRYPTED_JOURNAL_FILE = "logs/trading_journal.enc"


def _get_encryption_key() -> Optional[bytes]:
    """Derive AES key from JOURNAL_ENCRYPTION_KEY env var."""
    key_str = os.environ.get('JOURNAL_ENCRYPTION_KEY')
    if not key_str:
        return None
    return hashlib.sha256(key_str.encode()).digest()


def _encrypt_data(data: bytes, key: bytes) -> bytes:
    """AES-256-CBC encryption with PKCS7 padding."""
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives import padding
        iv = os.urandom(16)
        padder = padding.PKCS7(128).padder()
        padded = padder.update(data) + padder.finalize()
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded) + encryptor.finalize()
        return base64.b64encode(iv + ciphertext)
    except ImportError:
        logger.warning("cryptography package not installed. Journal stored unencrypted.")
        return data


def _decrypt_data(data: bytes, key: bytes) -> bytes:
    """AES-256-CBC decryption."""
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives import padding
        raw = base64.b64decode(data)
        iv = raw[:16]
        ciphertext = raw[16:]
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        padded = decryptor.update(ciphertext) + decryptor.finalize()
        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(padded) + unpadder.finalize()
    except ImportError:
        return data


class TradingJournal:
    """
    Persistent repository for trade outcomes and metadata.
    Enables 'Alpha Review' and strategy refinement.
    Set JOURNAL_ENCRYPTION_KEY env var to enable AES-256 encryption at rest.
    """
    def __init__(self):
        self._ensure_log_dir()
        self._encryption_key = _get_encryption_key()
        self.trades = self._load_journal()

    def _ensure_log_dir(self):
        # Ensure both the new runtime dir and legacy dir exist
        os.makedirs(os.path.dirname(os.path.abspath(JOURNAL_FILE)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(ENCRYPTED_JOURNAL_FILE)), exist_ok=True)

    def _load_journal(self) -> List[Dict]:
        # Try encrypted file first
        if self._encryption_key and os.path.exists(ENCRYPTED_JOURNAL_FILE):
            try:
                with open(ENCRYPTED_JOURNAL_FILE, 'rb') as f:
                    decrypted = _decrypt_data(f.read(), self._encryption_key)
                    return json.loads(decrypted.decode('utf-8'))
            except Exception as e:
                logger.error(f"Failed to load encrypted journal: {e}")
                return []

        # Fall back to plaintext
        if os.path.exists(JOURNAL_FILE):
            try:
                with open(JOURNAL_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load journal: {e}")
                return []
        return []

    def log_trade(self, 
                  asset: str, 
                  side: str, 
                  quantity: float, 
                  price: float, 
                  regime: str, 
                  strategy_name: str,
                  confidence: float,
                  reasoning: str,
                  order_id: str,
                  feature_vector: Optional[Dict] = None,
                  model_signal: int = 0):
        """Record trade with professional Audit/Compliance logging."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "asset": asset,
            "side": side,
            "quantity": quantity,
            "price": price,
            "regime": regime,
            "strategy": strategy_name,
            "confidence": confidence,
            "reasoning": reasoning,
            "order_id": order_id,
            "model_decision": model_signal,
            "audit_trail": {
                "feature_snapshot": feature_vector or {},
                "execution_mode": "LIVE" if "live" in reasoning.lower() else "SHADOW"
            },
            "status": "OPEN",
            "pnl": 0.0,
            "exit_price": None,
            "exit_time": None,
        }
        # Tamper-proof hash chain: each entry includes hash of previous
        entry["prev_hash"] = self._get_chain_hash()
        entry["entry_hash"] = self._compute_entry_hash(entry)
        self.trades.append(entry)
        self._save_journal()
        logger.info(f"Journaled {side} trade for {asset} (Order: {order_id}) - Audit Trail Saved.")

    def close_trade(self, order_id: str, exit_price: float, pnl: float, reason: str = ""):
        """Update existing trade with exit data."""
        for entry in self.trades:
            if entry["order_id"] == order_id:
                entry["status"] = "CLOSED"
                entry["exit_price"] = exit_price
                entry["exit_time"] = datetime.now().isoformat()
                entry["pnl"] = pnl
                if reason:
                    entry["reason"] = reason
                # Calculate R-Multiple
                try:
                    entry["return_pct"] = (exit_price - entry["price"]) / entry["price"] if entry["side"] == "buy" else (entry["price"] - exit_price) / entry["price"]
                except ZeroDivisionError:
                    entry["return_pct"] = 0.0
                break
        self._save_journal()

    def _save_journal(self):
        try:
            data = json.dumps(self.trades, indent=2)
            if self._encryption_key:
                encrypted = _encrypt_data(data.encode('utf-8'), self._encryption_key)
                with open(ENCRYPTED_JOURNAL_FILE, 'wb') as f:
                    f.write(encrypted)
            else:
                with open(JOURNAL_FILE, 'w') as f:
                    f.write(data)
        except Exception as e:
            logger.error(f"Failed to save journal: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Generate high-level stats for the dashboard."""
        closed_trades = [t for t in self.trades if t["status"] == "CLOSED"]
        if not closed_trades:
            return {"total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0}
        
        wins = [t for t in closed_trades if t["pnl"] > 0]
        return {
            "total_trades": len(self.trades),
            "closed_trades": len(closed_trades),
            "win_rate": len(wins) / len(closed_trades),
            "total_pnl": sum(t["pnl"] for t in closed_trades),
            "avg_return": sum(t.get("return_pct", 0) for t in closed_trades) / len(closed_trades)
        }

    def get_recent_trades(self, limit: int = 5) -> List[Dict]:
        """Get the most recent trades for context in LLM analysis."""
        return self.trades[-limit:] if self.trades else []

    # ── Tamper-Proof Hash Chain ──

    def _compute_entry_hash(self, entry: Dict) -> str:
        """SHA-256 hash of an entry (excluding its own hash field)."""
        hashable = {k: v for k, v in entry.items() if k != "entry_hash"}
        content = json.dumps(hashable, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_chain_hash(self) -> str:
        """Get hash of the last entry in the chain (or genesis hash)."""
        if not self.trades:
            return hashlib.sha256(b"GENESIS").hexdigest()
        last = self.trades[-1]
        return last.get("entry_hash", hashlib.sha256(b"LEGACY").hexdigest())

    def verify_chain_integrity(self) -> Dict[str, Any]:
        """
        Verify the hash chain integrity of the entire journal.
        Returns dict with is_valid, tampered_indices, and total_entries.
        """
        if not self.trades:
            return {"is_valid": True, "tampered_indices": [], "total_entries": 0}

        tampered = []
        expected_prev = hashlib.sha256(b"GENESIS").hexdigest()

        for i, entry in enumerate(self.trades):
            # Skip legacy entries without hash chain
            if "entry_hash" not in entry:
                continue

            # Verify prev_hash linkage
            if entry.get("prev_hash") != expected_prev:
                tampered.append(i)

            # Verify self-hash integrity
            computed = self._compute_entry_hash(entry)
            if computed != entry["entry_hash"]:
                tampered.append(i)

            expected_prev = entry["entry_hash"]

        return {
            "is_valid": len(tampered) == 0,
            "tampered_indices": tampered,
            "total_entries": len(self.trades),
            "chain_entries": sum(1 for t in self.trades if "entry_hash" in t),
        }
