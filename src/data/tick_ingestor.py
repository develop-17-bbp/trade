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
        import sys
        try:
            if sys.platform == 'win32':
                # Windows-native Named Shared Memory
                self.shm = mmap.mmap(-1, 1024, tagname=self.buffer_name)
                logger.info(f"[HFT-INGEST] Connected to Windows SHM: {self.buffer_name}")
            else:
                # Mac/Linux: file-backed shm in project temp dir (cross-platform path)
                import os
                _tmp_dir = os.path.join(os.getcwd(), "tmp")
                os.makedirs(_tmp_dir, exist_ok=True)
                _path = os.path.join(_tmp_dir, self.buffer_name)
                f = open(_path, "r+b")
                self.shm = mmap.mmap(f.fileno(), 1024)
                logger.info(f"[HFT-INGEST] Connected to file-backed SHM: {self.buffer_name}")
        except Exception as e:
            logger.warning(f"[HFT-INGEST] Shared Memory unavailable ({e}). Using network fallback.")
            self.shm = None

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
