"""LLM output scrubber — strip secrets + PII before storage / logging.

Reference: arXiv:2602.21496 — "Beyond Refusal: Probing the Limits of
Agentic Self-Correction for Semantic Sensitive Information" (Feb 2026).
They demonstrate that LLM refusal training + 'do not output secrets'
system prompts is insufficient: adversarial inputs can still cause the
model to echo sensitive data (API keys, credentials, PII) verbatim in
outputs.

ACT's defense: a deterministic regex pass over every LLM output BEFORE
it reaches `warm_store`, `brain_memory`, or operator-facing logs. The
model might still generate a secret; the scrubber ensures it never
persists.

Patterns intentionally OVER-scrub (err on false-positive side) rather
than under-scrub — a false positive replaces a random 20-char string
with `[REDACTED:SECRET]` which is recoverable; a false negative leaks
a real secret to disk, which is not.

Kept pure-stdlib (regex) — no extra dep, no crash on import.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Pattern = (label, compiled regex, replacement). Ordered by specificity
# — more-specific patterns run first so 'API key' isn't mistaken for
# 'long hex string'.
#
# All patterns use word-boundary or delimiter anchors where possible.

_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    # Common API-key shapes (prefix + base64/hex)
    # OpenAI keys come in variants: sk-XXX... (classic) and sk-proj-XXX...
    # (project-scoped, 2026). Both end with alnum; middle may contain -/_.
    ("openai_key",       re.compile(r"\bsk-(?:proj-|svcacct-|[A-Za-z0-9])[A-Za-z0-9_\-]{18,}\b"),
     "[REDACTED:OPENAI_KEY]"),
    ("anthropic_key",    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{18,}\b"),
     "[REDACTED:ANTHROPIC_KEY]"),
    ("google_ai_key",    re.compile(r"\bAIza[0-9A-Za-z_\-]{35,}\b"),
     "[REDACTED:GOOGLE_AI_KEY]"),
    ("aws_access_key",   re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
     "[REDACTED:AWS_KEY]"),
    ("aws_secret",       re.compile(r"\b(?:aws_secret_access_key|AWS_SECRET)[\"\s:=]+[A-Za-z0-9/+=]{40}\b", re.IGNORECASE),
     "[REDACTED:AWS_SECRET]"),
    ("github_token",     re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
     "[REDACTED:GITHUB_TOKEN]"),
    ("slack_token",      re.compile(r"\bxox[abpors]-[A-Za-z0-9\-]{10,}\b"),
     "[REDACTED:SLACK_TOKEN]"),
    ("private_key_pem",  re.compile(r"-----BEGIN (?:RSA |EC |)PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |)PRIVATE KEY-----"),
     "[REDACTED:PRIVATE_KEY]"),
    ("jwt",              re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"),
     "[REDACTED:JWT]"),

    # PII
    ("email",            re.compile(r"\b[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,253}\.[A-Za-z]{2,}\b"),
     "[REDACTED:EMAIL]"),
    ("credit_card",      re.compile(r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6011)[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),
     "[REDACTED:CARD]"),
    ("us_ssn",           re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
     "[REDACTED:SSN]"),
    ("phone_us",         re.compile(r"\b(?:\+?1[\s\-])?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b"),
     "[REDACTED:PHONE]"),

    # Robinhood / crypto-specific
    ("rh_account_num",   re.compile(r"\b(?:Robinhood|RH)[^\d]{0,10}(\d{9,12})\b", re.IGNORECASE),
     "[REDACTED:RH_ACCOUNT]"),
    ("ed25519_priv",     re.compile(r"(?:private\s*key|secret)[\"\s:=]+[A-Fa-f0-9]{64,}", re.IGNORECASE),
     "[REDACTED:PRIV_KEY]"),
    ("bitcoin_priv_wif", re.compile(r"\b[KL5][1-9A-HJ-NP-Za-km-z]{50,51}\b"),
     "[REDACTED:BTC_WIF]"),
    ("ethereum_priv",    re.compile(r"\b0x[a-fA-F0-9]{64}\b"),
     "[REDACTED:ETH_PRIVKEY]"),
]


class ScrubResult:
    """Plain class (Py3.14 dataclass anonymous-load quirk)."""

    def __init__(self, text: str, redactions: Optional[List[Dict[str, Any]]] = None) -> None:
        self.text = text
        self.redactions: List[Dict[str, Any]] = list(redactions or [])

    @property
    def any_redacted(self) -> bool:
        return len(self.redactions) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "redaction_count": len(self.redactions),
            "redacted_labels": [r["label"] for r in self.redactions],
        }


def scrub(text: Any) -> ScrubResult:
    """Redact known secrets + PII from `text`. Non-string input coerced
    to str; None / empty returns empty ScrubResult."""
    if text is None:
        return ScrubResult(text="")
    s = str(text)
    redactions: List[Dict[str, Any]] = []
    out = s
    for label, pat, repl in _PATTERNS:
        def _record(match, _label=label, _repl=repl):
            redactions.append({
                "label": _label,
                "position": match.start(),
                "length": match.end() - match.start(),
            })
            return _repl
        out = pat.sub(_record, out)
    if redactions:
        logger.debug("output_scrubber: %d redactions (%s)",
                     len(redactions),
                     ", ".join(sorted({r["label"] for r in redactions})))
    return ScrubResult(text=out, redactions=redactions)


def scrub_dict(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively scrub string values in a dict. Lists are walked;
    numbers/bools/None pass through."""
    if not d:
        return {}
    out: Dict[str, Any] = {}
    for k, v in d.items():
        out[k] = _scrub_any(v)
    return out


def _scrub_any(v: Any) -> Any:
    if v is None or isinstance(v, (int, float, bool)):
        return v
    if isinstance(v, str):
        return scrub(v).text
    if isinstance(v, list):
        return [_scrub_any(x) for x in v]
    if isinstance(v, dict):
        return scrub_dict(v)
    return v


def is_enabled() -> bool:
    """Gated by `ACT_SCRUB_LLM_OUTPUT`. On by default — turn off with
    `ACT_SCRUB_LLM_OUTPUT=0` only if you have downstream consumers that
    need raw text (very unusual)."""
    env = (os.environ.get("ACT_SCRUB_LLM_OUTPUT") or "1").strip().lower()
    return env in ("1", "true", "yes", "on")
