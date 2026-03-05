"""
PHASE 5: System Health Monitoring & 24/7 Operations
===================================================
Continuous market monitoring, automated maintenance routines, and alerting system.
"""

import time
import threading
import logging
from typing import Dict, Any, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

class SystemHealthChecker:
    """
    Monitors system health, data staleness, and API connectivity.
    Triggers recovery actions if services degrade.
    """
    def __init__(self, check_interval_sec: int = 60):
        self.check_interval_sec = check_interval_sec
        self.running = False
        self.monitor_thread = None
        self.components = {}
        self.last_alerts = {}
        
    def register_component(self, name: str, check_fn: Callable[[], bool]):
        """Register a component to be monitored via a boolean-returning function."""
        self.components[name] = check_fn
        logger.info(f"[HEALTH] Registered component: {name}")
        
    def start(self):
        """Start the background health checking thread."""
        if self.running:
            return
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("[HEALTH] 24/7 Health Monitoring Started.")
        
    def stop(self):
        """Stop the background monitoring thread."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
    def _monitor_loop(self):
        while self.running:
            for name, check_fn in self.components.items():
                try:
                    is_healthy = check_fn()
                    if not is_healthy:
                        self._trigger_alert(name, "Health check returned False.")
                except Exception as e:
                    self._trigger_alert(name, f"Exception during health check: {e}")
                    
            time.sleep(self.check_interval_sec)
            
    def _trigger_alert(self, component: str, reason: str):
        """Alert and attempt self-healing triggers."""
        now = time.time()
        # Rate limit alerts to once every 5 minutes per component
        if component in self.last_alerts and (now - self.last_alerts[component]) < 300:
            return
            
        self.last_alerts[component] = now
        logger.error(f"[🚨 HEALTH ALERT] Component {component} is failing! Reason: {reason}")
        
        # Self-healing logic hooks go here
        # e.g., if API fails, trigger fallback to backup API
        
    def check_overall_health(self) -> Dict[str, Any]:
        """Provides an immediate assessment of all registered components."""
        status = {}
        all_healthy = True
        for name, check_fn in self.components.items():
            try:
                healthy = check_fn()
                status[name] = "OK" if healthy else "FAILED"
                if not healthy:
                    all_healthy = False
            except Exception as e:
                status[name] = f"ERROR: {e}"
                all_healthy = False
                
        return {
            "status": "HEALTHY" if all_healthy else "DEGRADED",
            "timestamp": datetime.now().isoformat(),
            "components": status
        }
