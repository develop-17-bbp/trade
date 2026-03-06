import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class KafkaProducerWrapper:
    """
    Enterprise Data Infrastructure: Kafka Streaming Layer.
    Publishes market ticks, features, model scores, and execution events.
    
    If no broker is connected, it falls back to a fail-safe local audit file 
    to ensure NO SILENT FAILURES.
    """
    def __init__(self, bootstrap_servers: str = "localhost:9092", 
                 enabled: bool = False):
        self.enabled = enabled
        self.bootstrap = bootstrap_servers
        self.producer = None
        
        if self.enabled:
            # Only try importing if enabled to avoid missing dependency errors
            try:
                from kafka import KafkaProducer
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    acks='all', # Guarantees delivery to all replicas
                    retries=5
                )
                logger.info(f"[KAFKA] Connected to {self.bootstrap}")
            except Exception as e:
                logger.error(f"[KAFKA] Connection failed: {e}. Falling back to local audit logs.")
                self.enabled = False

    def publish(self, topic: str, message: Dict[str, Any], key: Optional[str] = None):
        """
        Broadcasting trading events to Kafka for ClickHouse ingestion.
        """
        # Add metadata for audit
        payload = {
            "timestamp_ms": int(datetime.now().timestamp() * 1000),
            "payload": message,
            "topic": topic
        }

        if self.producer and self.enabled:
            try:
                self.producer.send(
                    topic, 
                    value=payload, 
                    key=key.encode('utf-8') if key else None
                )
            except Exception as e:
                logger.error(f"[KAFKA-SEND-ERROR] Topic {topic}: {e}")
                self._fallback_log(topic, payload)
        else:
            self._fallback_log(topic, payload)

    def _fallback_log(self, topic: str, payload: Dict[str, Any]):
        """
        Fail-safe persistence to ensure every trading decision is recorded.
        """
        try:
            with open("logs/audit_failover.jsonl", "a") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            logger.critical(f"[AUDIT-FATAL] Final failover logging failed: {e}")

    def close(self):
        if self.producer:
            self.producer.flush()
            self.producer.close()
