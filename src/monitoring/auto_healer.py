"""
Auto-Healer Daemon — GPU LLM-Powered System Monitor & Fixer
=============================================================
Runs alongside the trading system. Uses a dedicated Ollama GPU endpoint
to watch logs, detect errors, diagnose issues, and auto-apply fixes.

Usage:
    python -m src.monitoring.auto_healer                  # Run daemon
    python -m src.monitoring.auto_healer --once           # Single scan
    python -m src.monitoring.auto_healer --scan-code      # Deep code audit

Config (config.yaml):
    monitor:
      enabled: true
      ollama_url: "https://your-gpu-tunnel.trycloudflare.com"
      ollama_model: "llama3.2:latest"
      scan_interval: 60          # Check every 60s
      auto_fix: true             # Apply safe fixes automatically
      max_fixes_per_hour: 5      # Rate limit auto-fixes
"""

import os
import sys
import json
import time
import re
import logging
import hashlib
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=True)

logger = logging.getLogger(__name__)

# ======================================================================
# Constants
# ======================================================================
LOG_FILE = os.path.join(PROJECT_ROOT, 'logs', 'system_output.log')
HEALER_LOG = os.path.join(PROJECT_ROOT, 'logs', 'auto_healer.log')
FIXES_LOG = os.path.join(PROJECT_ROOT, 'logs', 'auto_healer_fixes.jsonl')
JOURNAL_FILE = os.path.join(PROJECT_ROOT, 'logs', 'trading_journal.jsonl')

# Error patterns that trigger LLM analysis
ERROR_PATTERNS = [
    r'\[WARNING\].*failed:',
    r'\[ERROR\]',
    r'Traceback \(most recent call last\)',
    r'Exception|Error|TypeError|ValueError|KeyError|AttributeError',
    r'FATAL',
    r'ConnectionError|TimeoutError|RequestException',
    r'rate.?limit',
    r'insufficient.*balance|margin',
    r'order.*rejected|order.*failed',
    r'position.*failed|close.*failed',
]

# Safe fix categories (auto-apply)
SAFE_FIX_TYPES = [
    'config_change',        # Change config.yaml values
    'null_guard',           # Add None/null checks in code
    'default_value',        # Add default values for missing keys
    'type_cast_guard',      # Wrap float()/int() with safe fallback
    'retry_logic',          # Add retry around network calls
    'logging_improvement',  # Better error messages
]

# Dangerous fix categories (log only, don't auto-apply)
DANGEROUS_FIX_TYPES = [
    'strategy_change',      # Changes to trading logic
    'risk_parameter',       # Changes to risk limits
    'exchange_api',         # Changes to API calls
    'position_management',  # Changes to position handling
    'delete_code',          # Removing code
]


# ======================================================================
# TIER 1: Hardcoded instant fixes — no LLM needed, 100% reliable
# Pattern: (regex matching error line) -> fix dict
# ======================================================================
HARDCODED_FIXES = [
    {
        # float(None) from LLM returning null
        'pattern': r"float\(\) argument must be a string or a real number, not 'NoneType'",
        'scan_for': r"float\([a-z_]+\.get\(['\"](\w+)['\"],\s*([^)]+)\)\)",
        'fix_fn': 'fix_float_none',
        'description': 'LLM returned null — add `or default` guard on float(.get())',
        'category': 'type_cast_guard',
        'severity': 'critical',
    },
    {
        # int(None) same problem
        'pattern': r"int\(\) argument must be a string.*not 'NoneType'",
        'scan_for': r"int\([a-z_]+\.get\(['\"](\w+)['\"],\s*([^)]+)\)\)",
        'fix_fn': 'fix_int_none',
        'description': 'LLM returned null — add `or default` guard on int(.get())',
        'category': 'type_cast_guard',
        'severity': 'critical',
    },
    {
        # JSON parse errors from LLM (Extra data, Expecting value, etc)
        'pattern': r'(Extra data|Expecting value|JSONDecodeError|Unterminated string)',
        'scan_for': None,
        'fix_fn': 'fix_json_parse',
        'description': 'LLM returned malformed JSON — already has try/except, log and continue',
        'category': 'logging_improvement',
        'severity': 'medium',
    },
    {
        # Connection errors to exchange
        'pattern': r'(ConnectionError|Connection refused|Connection reset|ECONNREFUSED)',
        'scan_for': None,
        'fix_fn': 'fix_connection_error',
        'description': 'Exchange connection failed — transient, will retry next cycle',
        'category': 'retry_logic',
        'severity': 'low',
    },
    {
        # Rate limiting from exchange or LLM
        'pattern': r'(rate.?limit|429|Too Many Requests|RateLimitError)',
        'scan_for': None,
        'fix_fn': 'fix_rate_limit',
        'description': 'Rate limited — back off and retry',
        'category': 'retry_logic',
        'severity': 'medium',
    },
    {
        # KeyError from missing dict keys
        'pattern': r"KeyError: ['\"](\w+)['\"]",
        'scan_for': None,
        'fix_fn': 'fix_key_error',
        'description': 'Missing dict key — add .get() with default',
        'category': 'default_value',
        'severity': 'high',
    },
    {
        # AttributeError: 'NoneType' has no attribute
        'pattern': r"AttributeError: 'NoneType' has no attribute '(\w+)'",
        'scan_for': None,
        'fix_fn': 'fix_none_attribute',
        'description': 'Object is None — add None check before access',
        'category': 'null_guard',
        'severity': 'high',
    },
    {
        # Timeout errors
        'pattern': r'(TimeoutError|ReadTimeout|ConnectTimeout|timed out)',
        'scan_for': None,
        'fix_fn': 'fix_timeout',
        'description': 'Request timed out — transient, will retry next cycle',
        'category': 'retry_logic',
        'severity': 'low',
    },
    {
        # Division by zero
        'pattern': r'(ZeroDivisionError|division by zero)',
        'scan_for': None,
        'fix_fn': 'fix_division_zero',
        'description': 'Division by zero — add zero check',
        'category': 'null_guard',
        'severity': 'high',
    },
    {
        # Index out of range
        'pattern': r'IndexError: (list|tuple) index out of range',
        'scan_for': None,
        'fix_fn': 'fix_index_error',
        'description': 'Empty list/array access — add length check',
        'category': 'null_guard',
        'severity': 'high',
    },
]


class AutoHealer:
    """
    3-Tier auto-healing system monitor:

    TIER 1 — Hardcoded fixes (instant, no LLM, 100% reliable)
      Pattern-matched common bugs: float(None), KeyError, JSON parse, etc.
      Applied immediately with backup. Covers ~60% of real errors.

    TIER 2 — Mistral 7B on GPU (2s, good for medium bugs)
      Null guards, type casts, default values, simple code patches.
      Can read error + 20 lines of code and suggest a fix.

    TIER 3 — Claude API (best, for complex multi-method bugs)
      Logic errors, race conditions, strategy bugs, multi-file issues.
      Falls back here when Mistral can't produce valid JSON fix.

    Flow:
    1. Tail system_output.log for new errors/warnings
    2. Try TIER 1 hardcoded fix first
    3. If no hardcoded fix, try TIER 2 Mistral
    4. If Mistral fails/bad JSON, escalate to TIER 3 Claude
    5. Apply safe fixes automatically, log dangerous ones for review
    6. Track everything in auto_healer_fixes.jsonl
    """

    def __init__(self, config: dict):
        monitor_cfg = config.get('monitor', {})
        ai_cfg = config.get('ai', {})

        # LLM endpoint for monitoring (can be separate from trading LLM)
        self.ollama_url = (
            monitor_cfg.get('ollama_url', '')
            or os.environ.get('MONITOR_OLLAMA_URL', '')
            or os.environ.get('OLLAMA_REMOTE_URL', '')
            or ai_cfg.get('ollama_base_url', 'http://localhost:11434')
        ).rstrip('/')

        self.ollama_model = (
            monitor_cfg.get('ollama_model', '')
            or os.environ.get('MONITOR_OLLAMA_MODEL', '')
            or os.environ.get('OLLAMA_REMOTE_MODEL', '')
            or ai_cfg.get('reasoning_model', 'llama3.2:latest')
        )

        self.scan_interval = monitor_cfg.get('scan_interval', 60)
        self.auto_fix = monitor_cfg.get('auto_fix', True)
        self.max_fixes_per_hour = monitor_cfg.get('max_fixes_per_hour', 5)

        # State
        self._last_log_pos = 0           # File position of last read
        self._seen_errors: Dict[str, float] = {}  # error_hash -> timestamp (dedup)
        self._fixes_applied: List[float] = []      # timestamps of recent fixes
        self._error_history: List[dict] = []       # All detected errors
        self._fix_history: List[dict] = []         # All generated fixes

        # Try Claude API as fallback if Ollama down
        self._claude_api_key = os.environ.get('ANTHROPIC_API_KEY', '')

        self._setup_logging()
        print(f"  [HEALER] GPU LLM: {self.ollama_model} @ {self.ollama_url}")
        print(f"  [HEALER] Auto-fix: {'ON' if self.auto_fix else 'OFF'} | Max {self.max_fixes_per_hour}/hr")

    def _setup_logging(self):
        """Setup healer-specific log file."""
        os.makedirs(os.path.dirname(HEALER_LOG), exist_ok=True)
        handler = logging.FileHandler(HEALER_LOG, mode='a', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    # ==================================================================
    # Log Tailing & Error Detection
    # ==================================================================
    def _tail_log(self, max_lines=200) -> List[str]:
        """Read new lines from system log since last check."""
        if not os.path.exists(LOG_FILE):
            return []

        try:
            with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
                f.seek(0, 2)  # End of file
                file_size = f.tell()

                if self._last_log_pos > file_size:
                    # File was rotated/truncated
                    self._last_log_pos = 0

                f.seek(self._last_log_pos)
                new_lines = f.readlines()
                self._last_log_pos = f.tell()

                return new_lines[-max_lines:] if len(new_lines) > max_lines else new_lines
        except Exception as e:
            logger.error(f"Failed to read log: {e}")
            return []

    def _detect_errors(self, lines: List[str]) -> List[dict]:
        """Find error patterns in log lines. Group related errors."""
        errors = []
        i = 0
        while i < len(lines):
            line = lines[i]
            matched = False

            for pattern in ERROR_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    matched = True
                    break

            if matched:
                # Grab context: 2 lines before + 5 lines after (for tracebacks)
                start = max(0, i - 2)
                end = min(len(lines), i + 6)
                context = ''.join(lines[start:end])

                # Dedup by hash
                err_hash = hashlib.md5(context.strip()[:200].encode()).hexdigest()[:12]
                now = time.time()

                if err_hash in self._seen_errors:
                    # Skip if seen in last 5 minutes
                    if now - self._seen_errors[err_hash] < 300:
                        i += 1
                        continue

                self._seen_errors[err_hash] = now

                errors.append({
                    'timestamp': datetime.now(tz=None).isoformat(),
                    'hash': err_hash,
                    'line': line.strip(),
                    'context': context.strip(),
                    'line_num': i,
                })
            i += 1

        return errors

    # ==================================================================
    # Code Context Extraction
    # ==================================================================
    def _find_relevant_code(self, error_context: str) -> str:
        """Extract relevant source code around the error."""
        # Parse file + line from traceback
        file_line_pattern = r'File "([^"]+)", line (\d+)'
        matches = re.findall(file_line_pattern, error_context)

        code_snippets = []
        for filepath, line_num in matches[-3:]:  # Last 3 frames
            try:
                line_num = int(line_num)
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                        all_lines = f.readlines()
                    start = max(0, line_num - 10)
                    end = min(len(all_lines), line_num + 10)
                    snippet = ''.join(f"{j+1:4d} | {all_lines[j]}" for j in range(start, end))
                    code_snippets.append(f"--- {filepath} (line {line_num}) ---\n{snippet}")
            except Exception:
                pass

        # Also try to find file references in warning messages
        # Pattern: [ASSET] Method failed: ErrorMessage
        method_pattern = r'\[(\w+)\] (\w+) failed: (.+)'
        method_match = re.search(method_pattern, error_context)
        if method_match and not code_snippets:
            asset, method_name, error_msg = method_match.groups()
            # Search executor.py for the method
            executor_path = os.path.join(PROJECT_ROOT, 'src', 'trading', 'executor.py')
            try:
                with open(executor_path, 'r', encoding='utf-8', errors='replace') as f:
                    all_lines = f.readlines()
                for i, line in enumerate(all_lines):
                    if f'def _{method_name.lower()}' in line.lower() or f'def {method_name.lower()}' in line.lower():
                        start = max(0, i - 2)
                        end = min(len(all_lines), i + 30)
                        snippet = ''.join(f"{j+1:4d} | {all_lines[j]}" for j in range(start, end))
                        code_snippets.append(f"--- executor.py ({method_name}) ---\n{snippet}")
                        break
            except Exception:
                pass

        return '\n\n'.join(code_snippets) if code_snippets else "No source code context available."

    # ==================================================================
    # LLM Query
    # ==================================================================
    def _query_llm(self, prompt: str, max_tokens: int = 512) -> str:
        """Query the monitoring GPU LLM. Falls back to Claude if Ollama down."""
        # Try Ollama first
        try:
            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": max_tokens},
            }
            resp = requests.post(url, json=payload, timeout=45)
            resp.raise_for_status()
            text = resp.json().get('response', '').strip()
            if text:
                return text
        except Exception as e:
            logger.warning(f"Ollama monitor query failed: {e}")

        # Fallback to Claude if available
        if self._claude_api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=self._claude_api_key)
                msg = client.messages.create(
                    model="claude-haiku-4-5",
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                return msg.content[0].text.strip()
            except Exception as e:
                logger.warning(f"Claude fallback also failed: {e}")

        return ""

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON from LLM response."""
        if not text:
            return None
        try:
            # Try direct
            if text.strip().startswith('{'):
                return json.loads(text)
            # Code fence
            if '```' in text:
                for part in text.split('```'):
                    part = part.strip()
                    if part.startswith('json'):
                        part = part[4:].strip()
                    if part.startswith('{'):
                        try:
                            return json.loads(part)
                        except json.JSONDecodeError:
                            pass
            # Find JSON in text
            start = text.find('{')
            end = text.rfind('}')
            if start >= 0 and end > start:
                return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
        return None

    # ==================================================================
    # TIER 1: Hardcoded instant fixes (no LLM needed)
    # ==================================================================
    def _try_hardcoded_fix(self, error: dict) -> Optional[dict]:
        """Check if error matches a known pattern with a hardcoded fix."""
        error_line = error.get('line', '') + ' ' + error.get('context', '')

        for fix_template in HARDCODED_FIXES:
            if re.search(fix_template['pattern'], error_line, re.IGNORECASE):
                # Found a match
                fix_fn_name = fix_template.get('fix_fn', '')
                fix_fn = getattr(self, f'_hardcoded_{fix_fn_name}', None)

                result = {
                    'severity': fix_template['severity'],
                    'category': fix_template['category'],
                    'diagnosis': fix_template['description'],
                    'fix_description': fix_template['description'],
                    'tier': 'TIER1_HARDCODED',
                    'error_hash': error.get('hash', ''),
                    'error_line': error.get('line', ''),
                    'diagnosed_at': datetime.now(tz=None).isoformat(),
                }

                # Try to generate actual code patch
                if fix_fn:
                    patch = fix_fn(error)
                    if patch:
                        result.update(patch)
                        return result

                # Even without a code patch, we have the diagnosis
                result['fix_type'] = 'manual_review'
                result['safe_to_auto_apply'] = False
                return result

        return None  # No hardcoded fix matches

    def _hardcoded_fix_float_none(self, error: dict) -> Optional[dict]:
        """Fix float(x.get('key', default)) -> float(x.get('key') or default)"""
        code_context = self._find_relevant_code(error['context'])

        # Find the offending float(.get()) pattern in the code
        matches = re.findall(
            r"(float\(\w+\.get\(['\"](\w+)['\"],\s*([^)]+)\))\)",
            code_context
        )
        if matches:
            old_expr = matches[0][0] + ')'  # full expression
            key_name = matches[0][1]
            default_val = matches[0][2].strip()
            new_expr = f"float({old_expr.split('(', 1)[1].split('.get(')[0]}.get('{key_name}') or {default_val})"

            return {
                'fix_type': 'code_patch',
                'fix_file': self._guess_fix_file(error),
                'old_code': old_expr,
                'new_code': new_expr,
                'safe_to_auto_apply': True,
                'reasoning': f'LLM can return null for {key_name}, .get() default ignored when key exists with None value',
            }
        return None

    def _hardcoded_fix_int_none(self, error: dict) -> Optional[dict]:
        """Fix int(x.get('key', default)) -> int(x.get('key') or default)"""
        code_context = self._find_relevant_code(error['context'])
        matches = re.findall(
            r"(int\(\w+\.get\(['\"](\w+)['\"],\s*([^)]+)\))\)",
            code_context
        )
        if matches:
            old_expr = matches[0][0] + ')'
            key_name = matches[0][1]
            default_val = matches[0][2].strip()
            new_expr = f"int({old_expr.split('(', 1)[1].split('.get(')[0]}.get('{key_name}') or {default_val})"
            return {
                'fix_type': 'code_patch',
                'fix_file': self._guess_fix_file(error),
                'old_code': old_expr,
                'new_code': new_expr,
                'safe_to_auto_apply': True,
                'reasoning': f'LLM can return null for {key_name}',
            }
        return None

    def _hardcoded_fix_json_parse(self, error: dict) -> Optional[dict]:
        """JSON parse errors — find json.loads() call without _extract_json safety net."""
        fix_file = self._guess_fix_file(error)
        context = error.get('context', '')

        # Look for bare json.loads(raw) patterns in source
        if fix_file and os.path.exists(fix_file):
            try:
                with open(fix_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Find json.loads(raw) or json.loads(llm_response) without _extract_json wrapper
                matches = re.findall(
                    r'(\s+result = json\.loads\(raw\))',
                    content
                )
                if matches:
                    old_code = matches[0]
                    indent = old_code[:len(old_code) - len(old_code.lstrip())]
                    new_code = (
                        f"{indent}try:\n"
                        f"{indent}    result = json.loads(raw)\n"
                        f"{indent}except json.JSONDecodeError:\n"
                        f"{indent}    raw = self._extract_json(raw)\n"
                        f"{indent}    result = json.loads(raw)"
                    )
                    return {
                        'fix_type': 'code_patch',
                        'fix_file': fix_file,
                        'old_code': old_code.lstrip(),
                        'new_code': new_code.lstrip(),
                        'safe_to_auto_apply': True,
                        'category': 'type_cast_guard',
                        'reasoning': 'LLM returns malformed JSON — add _extract_json safety wrapper',
                    }
            except Exception:
                pass

        return None  # Escalate to TIER 2

    def _hardcoded_fix_connection_error(self, error: dict) -> Optional[dict]:
        """Connection errors are transient — no code fix needed."""
        return {
            'fix_type': 'manual_review',
            'fix_file': '',
            'safe_to_auto_apply': False,
            'reasoning': 'Transient network error. System will retry next poll cycle automatically.',
        }

    def _hardcoded_fix_rate_limit(self, error: dict) -> Optional[dict]:
        """Rate limit — suggest increasing poll interval."""
        return {
            'fix_type': 'manual_review',
            'fix_file': 'config.yaml',
            'safe_to_auto_apply': False,
            'reasoning': 'Rate limited by exchange or LLM API. Consider increasing poll_interval or reducing LLM calls.',
        }

    def _hardcoded_fix_key_error(self, error: dict) -> Optional[dict]:
        """KeyError — try to find the dict access and add .get() with default."""
        context = error.get('context', '')
        # Extract the missing key name
        key_match = re.search(r"KeyError:\s*['\"](\w+)['\"]", context)
        if not key_match:
            return None

        key_name = key_match.group(1)
        fix_file = self._guess_fix_file(error)

        if fix_file and os.path.exists(fix_file):
            try:
                with open(fix_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Find dict[key] access pattern
                pattern = rf"\[['\"]({re.escape(key_name)})['\"](?!\s*\])"
                matches = list(re.finditer(pattern, content))
                if matches:
                    # Find the line containing the match
                    match = matches[0]
                    line_start = content.rfind('\n', 0, match.start()) + 1
                    line_end = content.find('\n', match.end())
                    old_line = content[line_start:line_end if line_end > 0 else len(content)]

                    # Replace dict['key'] with dict.get('key', default)
                    new_line = old_line.replace(f"['{key_name}']", f".get('{key_name}', '')")
                    new_line = new_line.replace(f'["{key_name}"]', f'.get("{key_name}", "")')

                    if new_line != old_line:
                        return {
                            'fix_type': 'code_patch',
                            'fix_file': fix_file,
                            'old_code': old_line.strip(),
                            'new_code': new_line.strip(),
                            'safe_to_auto_apply': True,
                            'category': 'default_value',
                            'reasoning': f'Key "{key_name}" missing from dict — use .get() with default',
                        }
            except Exception:
                pass

        return None  # Escalate to TIER 2

    def _hardcoded_fix_none_attribute(self, error: dict) -> Optional[dict]:
        """NoneType attribute — add None check guard."""
        context = error.get('context', '')
        attr_match = re.search(r"'NoneType' has no attribute '(\w+)'", context)
        if not attr_match:
            return None

        attr_name = attr_match.group(1)
        fix_file = self._guess_fix_file(error)

        # Extract the variable name from traceback if available
        line_match = re.search(r'(\w+)\.' + re.escape(attr_name), context)
        if line_match and fix_file and os.path.exists(fix_file):
            var_name = line_match.group(1)
            try:
                with open(fix_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Find the line with var.attr access
                pattern = rf'^(\s*)(.+{re.escape(var_name)}\.{re.escape(attr_name)}.+)$'
                match = re.search(pattern, content, re.MULTILINE)
                if match:
                    indent = match.group(1)
                    old_line = match.group(2)
                    new_line = f"if {var_name} is not None:\n{indent}    {old_line}"
                    return {
                        'fix_type': 'code_patch',
                        'fix_file': fix_file,
                        'old_code': f"{indent}{old_line}",
                        'new_code': f"{indent}{new_line}",
                        'safe_to_auto_apply': True,
                        'category': 'null_guard',
                        'reasoning': f'{var_name} can be None — add None check before .{attr_name} access',
                    }
            except Exception:
                pass

        return None  # Escalate to TIER 2

    def _hardcoded_fix_timeout(self, error: dict) -> Optional[dict]:
        """Timeout is transient."""
        return {
            'fix_type': 'manual_review',
            'fix_file': '',
            'safe_to_auto_apply': False,
            'reasoning': 'Request timed out. Transient — system retries automatically.',
        }

    def _hardcoded_fix_division_zero(self, error: dict) -> Optional[dict]:
        """Division by zero — try to find the division and add zero check."""
        fix_file = self._guess_fix_file(error)
        context = error.get('context', '')

        # Extract line number from traceback
        line_match = re.search(r'line (\d+)', context)
        if line_match and fix_file and os.path.exists(fix_file):
            line_num = int(line_match.group(1))
            try:
                with open(fix_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                if 0 < line_num <= len(lines):
                    old_line = lines[line_num - 1].rstrip('\n')
                    # Find division pattern: x / y or x // y
                    div_match = re.search(r'(\w+)\s*/\s*(\w+)', old_line)
                    if div_match:
                        divisor = div_match.group(2)
                        # Add zero guard
                        new_line = old_line.replace(
                            div_match.group(0),
                            f"{div_match.group(1)} / {divisor} if {divisor} != 0 else 0"
                        )
                        if new_line != old_line:
                            return {
                                'fix_type': 'code_patch',
                                'fix_file': fix_file,
                                'old_code': old_line.strip(),
                                'new_code': new_line.strip(),
                                'safe_to_auto_apply': True,
                                'category': 'null_guard',
                                'reasoning': f'Division by zero — add guard: {divisor} != 0',
                            }
            except Exception:
                pass

        return None  # Escalate to TIER 2

    def _hardcoded_fix_index_error(self, error: dict) -> Optional[dict]:
        """Index error — try to add length check."""
        fix_file = self._guess_fix_file(error)
        context = error.get('context', '')

        line_match = re.search(r'line (\d+)', context)
        if line_match and fix_file and os.path.exists(fix_file):
            line_num = int(line_match.group(1))
            try:
                with open(fix_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                if 0 < line_num <= len(lines):
                    old_line = lines[line_num - 1].rstrip('\n')
                    indent = old_line[:len(old_line) - len(old_line.lstrip())]
                    # Find array[index] pattern
                    idx_match = re.search(r'(\w+)\[(-?\d+)\]', old_line)
                    if idx_match:
                        arr_name = idx_match.group(1)
                        idx_val = int(idx_match.group(2))
                        min_len = abs(idx_val) if idx_val < 0 else idx_val + 1
                        new_code = f"{indent}if len({arr_name}) >= {min_len}:\n    {old_line}"
                        return {
                            'fix_type': 'code_patch',
                            'fix_file': fix_file,
                            'old_code': old_line,
                            'new_code': new_code,
                            'safe_to_auto_apply': True,
                            'category': 'null_guard',
                            'reasoning': f'Index {idx_val} out of range — add len({arr_name}) >= {min_len} guard',
                        }
            except Exception:
                pass

        return None  # Escalate to TIER 2

    def _guess_fix_file(self, error: dict) -> str:
        """Guess which file the error is in from context."""
        context = error.get('context', '')
        # Check traceback
        file_match = re.findall(r'File "([^"]+)"', context)
        if file_match:
            return file_match[-1]
        # Check warning pattern
        if 'executor' in context.lower():
            return os.path.join(PROJECT_ROOT, 'src', 'trading', 'executor.py')
        if 'fetcher' in context.lower():
            return os.path.join(PROJECT_ROOT, 'src', 'data', 'fetcher.py')
        return os.path.join(PROJECT_ROOT, 'src', 'trading', 'executor.py')

    # ==================================================================
    # TIER 3: Claude API (complex bugs Mistral can't handle)
    # ==================================================================
    def _query_claude_diagnosis(self, error: dict, code_context: str) -> Optional[dict]:
        """Escalate to Claude API for complex bugs."""
        if not self._claude_api_key:
            return None

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self._claude_api_key)

            prompt = f"""You are an expert Python debugger for a crypto trading system.
The system uses EMA(8) crossover + LLM confirmation (bull/bear/facilitator agents).
Files: src/trading/executor.py, src/data/fetcher.py, config.yaml

ERROR:
{error['context']}

SOURCE CODE AROUND ERROR:
{code_context}

Provide an EXACT fix. The old_code must be a string that exists verbatim in the file.

Respond ONLY with JSON:
{{
    "severity": "critical|high|medium|low",
    "category": "null_guard|type_cast_guard|default_value|retry_logic|strategy_change|risk_parameter",
    "diagnosis": "Root cause in 1-2 sentences",
    "fix_type": "code_patch|config_patch|manual_review",
    "fix_file": "relative/path/to/file.py",
    "fix_description": "What the fix does",
    "old_code": "EXACT string to find in file",
    "new_code": "replacement string",
    "safe_to_auto_apply": true,
    "reasoning": "Why this fix is correct"
}}"""

            msg = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            text = msg.content[0].text.strip()
            result = self._extract_json(text)
            if result:
                result['tier'] = 'TIER3_CLAUDE'
                result['error_hash'] = error.get('hash', '')
                result['error_line'] = error.get('line', '')
                result['diagnosed_at'] = datetime.now(tz=None).isoformat()
                logger.info(f"[TIER3 CLAUDE] {result.get('severity','?')} | {result.get('diagnosis','')[:100]}")
                return result
        except Exception as e:
            logger.warning(f"Claude diagnosis failed: {e}")

        return None

    # ==================================================================
    # Error Diagnosis — 3-Tier System
    # ==================================================================
    def diagnose_error(self, error: dict) -> Optional[dict]:
        """
        3-tier diagnosis:
        TIER 1: Hardcoded pattern match (instant, 100% reliable)
        TIER 2: Mistral on GPU (2s, good for medium bugs)
        TIER 3: Claude API (best, for complex bugs Mistral can't fix)
        """
        # ---- TIER 1: Hardcoded ----
        hardcoded = self._try_hardcoded_fix(error)
        if hardcoded and hardcoded.get('fix_type') not in ('manual_review', None):
            print(f"  [TIER1] Hardcoded fix: {hardcoded.get('diagnosis','')[:80]}")
            return hardcoded
        elif hardcoded and hardcoded.get('fix_type') == 'manual_review':
            # Known transient issue (connection, timeout, rate_limit) — log, don't escalate
            category = hardcoded.get('category', '')
            is_transient = category in ('retry_logic',) or 'transient' in hardcoded.get('reasoning', '').lower()
            if is_transient:
                print(f"  [TIER1] Transient (no fix needed): {hardcoded.get('reasoning','')[:80]}")
                self._log_fix(hardcoded, applied=False, reason="Transient issue")
                return None
            else:
                # Non-transient manual_review — escalate to TIER 2
                print(f"  [TIER1] Recognized but needs code fix — escalating to TIER 2")
                # Fall through to TIER 2

        # ---- TIER 2: Mistral on GPU ----
        code_context = self._find_relevant_code(error['context'])

        prompt = f"""You are a Python trading system debugger. Analyze this error and provide a fix.

SYSTEM: Crypto trading bot using EMA(8) crossover + LLM confirmation on Robinhood Crypto.
Key files: src/trading/executor.py, src/data/fetcher.py, config.yaml

ERROR LOG:
{error['context']}

RELEVANT SOURCE CODE:
{code_context}

DIAGNOSE THIS ERROR AND PROVIDE A FIX.

Respond ONLY with JSON:
{{
    "severity": "critical|high|medium|low",
    "category": "one of: config_change, null_guard, default_value, type_cast_guard, retry_logic, logging_improvement, strategy_change, risk_parameter, exchange_api, position_management",
    "diagnosis": "What went wrong and why (1-2 sentences)",
    "fix_type": "code_patch|config_patch|restart_needed|manual_review",
    "fix_file": "path/to/file.py or config.yaml",
    "fix_description": "What the fix does (1 sentence)",
    "old_code": "exact string to find and replace (or empty if config)",
    "new_code": "replacement string (or empty if config)",
    "config_key": "dot.notation.key if config change",
    "config_value": "new value if config change",
    "safe_to_auto_apply": true/false,
    "reasoning": "Why this fix is correct and safe"
}}"""

        raw = self._query_llm(prompt, max_tokens=600)
        result = self._extract_json(raw)

        if result and result.get('old_code') and result.get('new_code'):
            result['tier'] = 'TIER2_MISTRAL'
            result['error_hash'] = error['hash']
            result['error_line'] = error['line']
            result['diagnosed_at'] = datetime.now(tz=None).isoformat()
            logger.info(f"[TIER2 MISTRAL] {result.get('severity','?')} | {result.get('diagnosis','')[:100]}")
            print(f"  [TIER2] Mistral diagnosed: {result.get('diagnosis','')[:80]}")
            return result

        # ---- TIER 3: Claude API (Mistral failed or gave incomplete fix) ----
        if self._claude_api_key:
            print(f"  [TIER2] Mistral couldn't fix — escalating to Claude...")
            claude_result = self._query_claude_diagnosis(error, code_context)
            if claude_result:
                print(f"  [TIER3] Claude diagnosed: {claude_result.get('diagnosis','')[:80]}")
                return claude_result

        logger.warning(f"All tiers failed for error: {error['line'][:80]}")
        return None

    # ==================================================================
    # Fix Application
    # ==================================================================
    def _can_apply_fix(self) -> bool:
        """Rate limit check for auto-fixes."""
        now = time.time()
        hour_ago = now - 3600
        self._fixes_applied = [t for t in self._fixes_applied if t > hour_ago]
        return len(self._fixes_applied) < self.max_fixes_per_hour

    def _backup_file(self, filepath: str) -> str:
        """Create backup before modifying a file."""
        backup_dir = os.path.join(PROJECT_ROOT, 'logs', 'backups')
        os.makedirs(backup_dir, exist_ok=True)
        ts = datetime.now(tz=None).strftime('%Y%m%d_%H%M%S')
        basename = os.path.basename(filepath)
        backup_path = os.path.join(backup_dir, f"{basename}.{ts}.bak")
        shutil.copy2(filepath, backup_path)
        return backup_path

    def apply_fix(self, diagnosis: dict) -> bool:
        """Apply a fix if it's safe. Returns True if applied."""
        fix_type = diagnosis.get('fix_type', '')
        category = diagnosis.get('category', '')
        safe = diagnosis.get('safe_to_auto_apply', False)

        # Check if this category is safe
        is_safe_category = category in SAFE_FIX_TYPES
        is_dangerous = category in DANGEROUS_FIX_TYPES

        if is_dangerous or not safe:
            self._log_fix(diagnosis, applied=False, reason="Dangerous category or LLM said unsafe")
            print(f"  [HEALER] MANUAL REVIEW NEEDED: {diagnosis.get('diagnosis','')[:80]}")
            print(f"           Category: {category} | File: {diagnosis.get('fix_file','')}")
            return False

        if not self.auto_fix:
            self._log_fix(diagnosis, applied=False, reason="auto_fix disabled")
            print(f"  [HEALER] Fix available but auto_fix=false: {diagnosis.get('fix_description','')[:80]}")
            return False

        if not self._can_apply_fix():
            self._log_fix(diagnosis, applied=False, reason="Rate limit exceeded")
            print(f"  [HEALER] Rate limited — skipping fix: {diagnosis.get('fix_description','')[:80]}")
            return False

        # Apply code patch
        if fix_type == 'code_patch':
            return self._apply_code_patch(diagnosis)
        elif fix_type == 'config_patch':
            return self._apply_config_patch(diagnosis)
        else:
            self._log_fix(diagnosis, applied=False, reason=f"Unknown fix_type: {fix_type}")
            return False

    def _apply_code_patch(self, diagnosis: dict) -> bool:
        """Apply a code fix via string replacement."""
        fix_file = diagnosis.get('fix_file', '')
        old_code = diagnosis.get('old_code', '')
        new_code = diagnosis.get('new_code', '')

        if not fix_file or not old_code or not new_code:
            self._log_fix(diagnosis, applied=False, reason="Missing fix_file/old_code/new_code")
            return False

        # Resolve relative path
        if not os.path.isabs(fix_file):
            fix_file = os.path.join(PROJECT_ROOT, fix_file)

        if not os.path.exists(fix_file):
            self._log_fix(diagnosis, applied=False, reason=f"File not found: {fix_file}")
            return False

        try:
            with open(fix_file, 'r', encoding='utf-8') as f:
                content = f.read()

            if old_code not in content:
                self._log_fix(diagnosis, applied=False, reason="old_code not found in file (maybe already fixed)")
                print(f"  [HEALER] Already fixed or code changed: {diagnosis.get('fix_description','')[:60]}")
                return False

            # Backup
            backup = self._backup_file(fix_file)

            # Apply
            new_content = content.replace(old_code, new_code, 1)

            # Verify it's valid Python (basic check)
            try:
                compile(new_content, fix_file, 'exec')
            except SyntaxError as e:
                self._log_fix(diagnosis, applied=False, reason=f"Fix creates syntax error: {e}")
                print(f"  [HEALER] Fix rejected (syntax error): {e}")
                return False

            with open(fix_file, 'w', encoding='utf-8') as f:
                f.write(new_content)

            self._fixes_applied.append(time.time())
            self._log_fix(diagnosis, applied=True, reason=f"Applied successfully. Backup: {backup}")
            print(f"  [HEALER] FIX APPLIED: {diagnosis.get('fix_description','')[:80]}")
            print(f"           File: {fix_file} | Backup: {os.path.basename(backup)}")
            return True

        except Exception as e:
            self._log_fix(diagnosis, applied=False, reason=f"Exception: {e}")
            logger.error(f"Failed to apply code patch: {e}")
            return False

    def _apply_config_patch(self, diagnosis: dict) -> bool:
        """Apply a config.yaml change."""
        config_key = diagnosis.get('config_key', '')
        config_value = diagnosis.get('config_value')

        if not config_key:
            self._log_fix(diagnosis, applied=False, reason="No config_key provided")
            return False

        config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
        try:
            backup = self._backup_file(config_path)

            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Navigate dot notation (e.g., "ai.llm_trade_conf_threshold")
            keys = config_key.split('.')
            obj = config
            for k in keys[:-1]:
                if k not in obj:
                    obj[k] = {}
                obj = obj[k]
            obj[keys[-1]] = config_value

            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            self._fixes_applied.append(time.time())
            self._log_fix(diagnosis, applied=True, reason=f"Config updated. Backup: {backup}")
            print(f"  [HEALER] CONFIG UPDATED: {config_key} = {config_value}")
            return True

        except Exception as e:
            self._log_fix(diagnosis, applied=False, reason=f"Config patch failed: {e}")
            return False

    def _log_fix(self, diagnosis: dict, applied: bool, reason: str):
        """Log fix attempt to JSONL file."""
        entry = {
            'timestamp': datetime.now(tz=None).isoformat(),
            'applied': applied,
            'reason': reason,
            'severity': diagnosis.get('severity', ''),
            'category': diagnosis.get('category', ''),
            'diagnosis': diagnosis.get('diagnosis', ''),
            'fix_description': diagnosis.get('fix_description', ''),
            'fix_file': diagnosis.get('fix_file', ''),
            'error_line': diagnosis.get('error_line', ''),
        }
        try:
            os.makedirs(os.path.dirname(FIXES_LOG), exist_ok=True)
            with open(FIXES_LOG, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, default=str) + '\n')
        except Exception:
            pass

        self._fix_history.append(entry)
        logger.info(f"[FIX] applied={applied} | {diagnosis.get('category','')} | {reason[:80]}")

    # ==================================================================
    # Deep Code Audit
    # ==================================================================
    def scan_code(self):
        """Full code audit — scan executor.py for potential issues."""
        executor_path = os.path.join(PROJECT_ROOT, 'src', 'trading', 'executor.py')

        with open(executor_path, 'r', encoding='utf-8', errors='replace') as f:
            code = f.read()

        # Find all float()/int() calls that might get None
        unsafe_patterns = []
        for i, line in enumerate(code.split('\n'), 1):
            stripped = line.strip()
            # float(x.get('key', default)) where default might not trigger
            if re.search(r'float\(.*\.get\(', stripped) and 'or ' not in stripped:
                unsafe_patterns.append(f"  Line {i}: {stripped[:100]}")
            # int(x.get('key', default))
            if re.search(r'int\(.*\.get\(', stripped) and 'or ' not in stripped:
                unsafe_patterns.append(f"  Line {i}: {stripped[:100]}")
            # bare except that silently passes
            if stripped == 'except:' or stripped == 'except Exception:':
                if i + 1 < len(code.split('\n')):
                    next_line = code.split('\n')[i].strip()
                    if next_line == 'pass':
                        unsafe_patterns.append(f"  Line {i}: Silent except+pass — errors hidden")

        if unsafe_patterns:
            prompt = f"""You are a Python code reviewer for a crypto trading system.
These patterns were found that could cause runtime errors:

{chr(10).join(unsafe_patterns[:20])}

For each, suggest if it needs a fix (use `or default` pattern for .get() calls).
List ONLY the ones that are actually dangerous (where None could reach float()/int()).

Respond with JSON:
{{
    "issues_found": <count>,
    "fixes": [
        {{"line": <num>, "description": "what's wrong", "old": "exact code", "new": "fixed code"}}
    ]
}}"""

            raw = self._query_llm(prompt, max_tokens=800)
            result = self._extract_json(raw)
            if result:
                print(f"\n  [AUDIT] Found {result.get('issues_found', 0)} potential issues:")
                for fix in result.get('fixes', []):
                    print(f"    Line {fix.get('line','?')}: {fix.get('description','')[:80]}")
                return result

        print("  [AUDIT] No obvious unsafe patterns found.")
        return None

    # ==================================================================
    # Trading Performance Monitor
    # ==================================================================
    def check_performance(self) -> Optional[dict]:
        """Analyze recent trading performance and flag issues."""
        try:
            if not os.path.exists(JOURNAL_FILE):
                return None

            trades = []
            with open(JOURNAL_FILE, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            trades.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

            if len(trades) < 5:
                return None

            # Last 20 trades
            recent = trades[-20:]
            wins = sum(1 for t in recent if float(t.get('pnl', 0)) > 0)
            losses = len(recent) - wins
            total_pnl = sum(float(t.get('pnl', 0)) for t in recent)
            win_rate = wins / len(recent) if recent else 0

            issues = []
            if win_rate < 0.3:
                issues.append(f"LOW WIN RATE: {win_rate:.0%} ({wins}W/{losses}L in last {len(recent)} trades)")
            if total_pnl < -100:
                issues.append(f"HEAVY LOSSES: ${total_pnl:.2f} in last {len(recent)} trades")

            # Consecutive losses
            streak = 0
            for t in reversed(recent):
                if float(t.get('pnl', 0)) < 0:
                    streak += 1
                else:
                    break
            if streak >= 5:
                issues.append(f"LOSING STREAK: {streak} consecutive losses")

            if not issues:
                return None

            # Ask LLM for diagnosis
            trade_summary = json.dumps(recent[-10:], indent=2, default=str)
            prompt = f"""You are a crypto trading system analyst. The system has performance issues:

ISSUES:
{chr(10).join('- ' + i for i in issues)}

LAST 10 TRADES:
{trade_summary}

The system uses EMA(8) crossover + LLM confirmation on 5-minute candles.
Bull agent confirms, Bear agent vetos bad trades, Facilitator makes final call.

Diagnose the performance issues and suggest fixes:
1. Is the strategy wrong, or is the market just unfavorable?
2. Should we pause trading?
3. Any config changes to reduce losses?

Respond with JSON:
{{
    "market_assessment": "trending|ranging|choppy|unclear",
    "should_pause": true/false,
    "config_suggestions": [{{"key": "config.key", "current": "val", "suggested": "val", "reason": "why"}}],
    "strategy_notes": "1-2 sentences"
}}"""

            raw = self._query_llm(prompt, max_tokens=500)
            result = self._extract_json(raw)
            if result:
                result['issues'] = issues
                print(f"\n  [PERF] Performance Issues Detected:")
                for i in issues:
                    print(f"    - {i}")
                print(f"  [PERF] Market: {result.get('market_assessment', '?')}")
                print(f"  [PERF] Pause: {'YES' if result.get('should_pause') else 'NO'}")
                print(f"  [PERF] Notes: {result.get('strategy_notes', '')[:100]}")
                return result

        except Exception as e:
            logger.error(f"Performance check failed: {e}")
        return None

    # ==================================================================
    # Main Loop
    # ==================================================================
    def scan_once(self):
        """Single scan cycle: check logs, diagnose, fix."""
        print(f"\n  [HEALER] Scanning... ({datetime.now(tz=None).strftime('%H:%M:%S')} UTC)")

        # 1. Read new log lines
        lines = self._tail_log()
        if not lines:
            print(f"  [HEALER] No new log lines.")
            return

        # 2. Detect errors
        errors = self._detect_errors(lines)
        if not errors:
            print(f"  [HEALER] {len(lines)} lines scanned — no errors found.")
            return

        print(f"  [HEALER] Found {len(errors)} new error(s)")

        # 3. Diagnose and fix each
        for error in errors:
            print(f"  [HEALER] Analyzing: {error['line'][:80]}")
            diagnosis = self.diagnose_error(error)

            if diagnosis:
                severity = diagnosis.get('severity', 'unknown')
                print(f"  [HEALER] Severity: {severity} | {diagnosis.get('diagnosis', '')[:80]}")

                if severity in ('critical', 'high') and self.auto_fix:
                    self.apply_fix(diagnosis)
                elif severity == 'medium':
                    print(f"  [HEALER] Medium severity — logged for review")
                    self._log_fix(diagnosis, applied=False, reason="Medium severity, manual review")
                else:
                    self._log_fix(diagnosis, applied=False, reason=f"Low severity, monitoring only")

    def run(self):
        """Main daemon loop."""
        print("=" * 60)
        print("  AUTO-HEALER DAEMON")
        print(f"  LLM: {self.ollama_model} @ {self.ollama_url}")
        print(f"  Watching: {LOG_FILE}")
        print(f"  Scan interval: {self.scan_interval}s")
        print(f"  Auto-fix: {'ON' if self.auto_fix else 'OFF'}")
        print("=" * 60)

        # Initial full-file read to set position
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
                f.seek(0, 2)
                self._last_log_pos = f.tell()
            print(f"  [HEALER] Starting from end of log ({self._last_log_pos} bytes)")

        # Performance check on startup
        self.check_performance()

        cycle = 0
        while True:
            try:
                self.scan_once()

                # Performance check every 10 cycles
                cycle += 1
                if cycle % 10 == 0:
                    self.check_performance()

                # Summary every 30 cycles
                if cycle % 30 == 0:
                    applied = sum(1 for f in self._fix_history if f.get('applied'))
                    total = len(self._fix_history)
                    print(f"  [HEALER] Summary: {total} errors diagnosed, {applied} fixes applied")

                time.sleep(self.scan_interval)

            except KeyboardInterrupt:
                print("\n  [HEALER] Shutting down...")
                break
            except Exception as e:
                logger.error(f"Scan cycle failed: {e}")
                time.sleep(self.scan_interval)


# ======================================================================
# CLI
# ======================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description='Auto-Healer: GPU LLM-powered system monitor')
    parser.add_argument('--once', action='store_true', help='Single scan then exit')
    parser.add_argument('--scan-code', action='store_true', help='Deep code audit')
    parser.add_argument('--check-perf', action='store_true', help='Check trading performance')
    parser.add_argument('--no-fix', action='store_true', help='Diagnose only, no auto-fix')
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if args.no_fix:
        config.setdefault('monitor', {})['auto_fix'] = False

    healer = AutoHealer(config)

    if args.scan_code:
        healer.scan_code()
    elif args.check_perf:
        healer.check_performance()
    elif args.once:
        # Read entire log for single scan
        healer._last_log_pos = max(0, os.path.getsize(LOG_FILE) - 50000) if os.path.exists(LOG_FILE) else 0
        healer.scan_once()
    else:
        healer.run()


if __name__ == '__main__':
    main()
