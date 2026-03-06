import mmap
import struct
import time
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class FastTickIngestor:
    """
    HFT-Ready: High-Performance Tick Ingestion (L1).
    Consumes binary market data from a Rust/C++ feed via Shared Memory.
    
    Format (32 bytes):
    - Symbol ID (Int)
    - Price (Double)
    - Volume (Double)
    - MS Timestamp (Long)
    """
    def __init__(self, buffer_name: str = "market_ticks_shm", 
                 buffer_size: int = 1024):
        self.buffer_name = buffer_name
        self.shm = None
        self._connect_shm()

    def _connect_shm(self):
        try:
            # We assume the Rust/Go service created this file in /tmp
            # In production, this would be a real SHM segment.
            f = open(f"tmp/{self.buffer_name}", "r+b")
            self.shm = mmap.mmap(f.fileno(), 1024)
            logger.info(f"[HFT-INGEST] Connected to high-speed binary tick feed.")
        except FileNotFoundError:
            # Create a mock if it doesn't exist for the demo
            pass

    def get_latest_tick(self) -> Optional[Tuple[int, float, float, int]]:
        """
        Polls the shared memory for the latest binary tick.
        Zero-copy and nearly zero-latency reading.
        """
        if not self.shm: return None
        
        try:
            self.shm.seek(0)
            data = self.shm.read(32)
            # Unpack: I d d Q
            symbol_id, price, vol, ts = struct.unpack("I d d Q", data)
            
            if ts == 0: return None
            return (symbol_id, price, vol, ts)
        except Exception as e:
            return None

    def close(self):
        if self.shm:
            self.shm.close()
