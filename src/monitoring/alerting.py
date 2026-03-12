"""
Alerting System for Critical Trading Events
=============================================
Sends alerts via webhook (Slack/Discord), Telegram, and local JSONL log
for kill switch triggers, circuit breaker activations, API failures,
exchange connectivity loss, and daily loss limit hits.
"""

import json
import logging
import os
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

VALID_LEVELS = ("INFO", "WARNING", "CRITICAL")

LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


class AlertManager:
    """
    Centralized alerting with rate-limiting and multi-channel delivery.

    Config keys:
        webhook_url            - Slack/Discord-compatible webhook endpoint
        telegram_bot_token     - Telegram Bot API token
        telegram_chat_id       - Telegram chat/channel ID
        alert_cooldown_minutes - minimum minutes between duplicate alerts (default 5)
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.webhook_url = config.get("webhook_url") or os.getenv("ALERT_WEBHOOK_URL")
        self.telegram_bot_token = config.get("telegram_bot_token") or os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = config.get("telegram_chat_id") or os.getenv("TELEGRAM_CHAT_ID")
        self.cooldown_seconds = int(config.get("alert_cooldown_minutes", 5)) * 60

        # title -> last-sent unix timestamp
        self._cooldowns: Dict[str, float] = {}
        self._lock = threading.Lock()

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._log_path = LOGS_DIR / "alerts.jsonl"

        logger.info("[ALERTING] AlertManager initialised (cooldown=%ds)", self.cooldown_seconds)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_alert(
        self,
        level: str,
        title: str,
        message: str,
        data: Optional[dict] = None,
    ) -> bool:
        """
        Dispatch an alert on all configured channels.

        Returns True if the alert was actually sent (not rate-limited).
        """
        level = level.upper()
        if level not in VALID_LEVELS:
            logger.warning("[ALERTING] Invalid level '%s', defaulting to WARNING", level)
            level = "WARNING"

        if not self._check_cooldown(title):
            logger.debug("[ALERTING] Suppressed (cooldown): %s", title)
            return False

        timestamp = datetime.now(timezone.utc).isoformat()

        # Always log locally
        self._send_log(level, title, message, data, timestamp)

        # Build payload for external channels
        payload = {
            "level": level,
            "title": title,
            "message": message,
            "timestamp": timestamp,
        }
        if data:
            payload["data"] = data

        # Webhook (Slack / Discord)
        if self.webhook_url:
            self._send_webhook(payload)

        # Telegram
        if self.telegram_bot_token and self.telegram_chat_id:
            icon = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(level, "")
            text = f"{icon} *{level}: {title}*\n{message}"
            if data:
                text += f"\n```{json.dumps(data, indent=2)}```"
            self._send_telegram(text)

        logger.info("[ALERTING] Sent %s alert: %s", level, title)
        return True

    # ------------------------------------------------------------------
    # Cooldown
    # ------------------------------------------------------------------

    def _check_cooldown(self, title: str) -> bool:
        """Return True if the alert should be sent (cooldown expired or first time)."""
        now = time.monotonic()
        with self._lock:
            last = self._cooldowns.get(title)
            if last is not None and (now - last) < self.cooldown_seconds:
                return False
            self._cooldowns[title] = now
            return True

    # ------------------------------------------------------------------
    # Delivery channels
    # ------------------------------------------------------------------

    def _send_webhook(self, payload: dict) -> None:
        """POST JSON to a Slack/Discord-compatible webhook URL."""
        try:
            import requests

            body = {
                "text": f"[{payload['level']}] {payload['title']}: {payload['message']}",
                "attachments": [
                    {
                        "color": {"INFO": "#36a64f", "WARNING": "#daa038", "CRITICAL": "#d00000"}.get(
                            payload["level"], "#cccccc"
                        ),
                        "fields": [
                            {"title": "Level", "value": payload["level"], "short": True},
                            {"title": "Time", "value": payload["timestamp"], "short": True},
                        ],
                    }
                ],
            }
            if payload.get("data"):
                body["attachments"][0]["fields"].append(
                    {"title": "Details", "value": json.dumps(payload["data"], indent=2), "short": False}
                )

            resp = requests.post(self.webhook_url, json=body, timeout=10)
            resp.raise_for_status()
        except Exception as exc:
            logger.error("[ALERTING] Webhook delivery failed: %s", exc)

    def _send_telegram(self, text: str) -> None:
        """Send a message via the Telegram Bot API."""
        try:
            import requests

            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            resp = requests.post(
                url,
                json={
                    "chat_id": self.telegram_chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.error("[ALERTING] Telegram delivery failed: %s", exc)

    def _send_log(
        self,
        level: str,
        title: str,
        message: str,
        data: Optional[dict] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Append a JSON line to the local alerts log file."""
        entry = {
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "level": level,
            "title": title,
            "message": message,
        }
        if data:
            entry["data"] = data
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.error("[ALERTING] Failed to write alert log: %s", exc)


# ----------------------------------------------------------------------
# Module-level singleton & convenience helper
# ----------------------------------------------------------------------

_singleton: Optional[AlertManager] = None
_singleton_lock = threading.Lock()


def _get_singleton() -> AlertManager:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = AlertManager()
    return _singleton


def alert_critical(title: str, message: str, data: Optional[dict] = None) -> bool:
    """Fire a CRITICAL alert using the module-level singleton AlertManager."""
    return _get_singleton().send_alert("CRITICAL", title, message, data)
