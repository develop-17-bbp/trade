"""
Universal Pluggable LLM Provider Interface
=============================================
Drop-in any LLM (Gemini, OpenAI, Claude, Ollama, vLLM, Together, Groq, DeepSeek,
Mistral, Cohere, local HuggingFace, or any OpenAI-compatible endpoint).

Every provider returns structured JSON. Math injection + prompt constraints are
applied BEFORE the prompt reaches the LLM — the LLM never gets to hallucinate
numbers, only interpret pre-computed quant data.

Usage:
    from src.ai.llm_provider import LLMRouter, LLMConfig
    router = LLMRouter()
    router.add_provider('gemini', LLMConfig(provider='google', api_key='...', model='gemini-2.5-flash'))
    router.add_provider('local', LLMConfig(provider='ollama', model='mistral'))
    result = router.query(prompt, fallback_chain=['gemini', 'local'])
"""

import os
import json
import re
import time
import logging
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

# Pre-compiled regexes for LLM JSON cleanup
_RE_JS_COMMENT = re.compile(r'//[^\n"]*(?=\n|$)')  # JS comments (not inside strings)
_RE_TRAILING_COMMA_OBJ = re.compile(r',\s*}')
_RE_TRAILING_COMMA_ARR = re.compile(r',\s*]')

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """Configuration for a single LLM provider."""
    provider: str                # google, openai, anthropic, ollama, vllm, together, groq, deepseek, mistral, cohere, custom
    api_key: str = ''
    model: str = ''
    base_url: str = ''           # For custom OpenAI-compatible endpoints
    temperature: float = 0.1     # Low temp for deterministic trading decisions
    max_tokens: int = 2000
    timeout: int = 120
    rate_limit_rpm: int = 15     # Requests per minute
    json_mode: bool = True       # Force JSON output when supported
    system_prompt: str = ''      # Override system prompt (else uses default)
    extra_params: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Base Provider Interface
# ─────────────────────────────────────────────────────────────

class BaseLLMProvider(ABC):
    """Abstract base for all LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._call_times: List[float] = []

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = '') -> Dict:
        """Send prompt, return parsed JSON dict."""
        pass

    def _throttle(self):
        """Rate limiting."""
        now = time.time()
        window = 60.0
        self._call_times = [t for t in self._call_times if now - t < window]
        if len(self._call_times) >= self.config.rate_limit_rpm:
            sleep_time = window - (now - self._call_times[0])
            if sleep_time > 0:
                logger.info(f"Rate limit: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        self._call_times.append(time.time())

    def _parse_json(self, text: str) -> Dict:
        """Extract JSON from LLM response text, handling truncation and comments."""
        text = text.strip()
        # Strip markdown code blocks
        if text.startswith('```'):
            lines = text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            text = '\n'.join(lines)
        # Strip JS-style comments and trailing commas from LLM output
        text = _RE_JS_COMMENT.sub('', text)
        text = _RE_TRAILING_COMMA_OBJ.sub('}', text)
        text = _RE_TRAILING_COMMA_ARR.sub(']', text)
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try finding JSON object in text
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        # Handle truncated JSON: try to repair by closing open strings/braces
        if start >= 0:
            fragment = text[start:]
            # Close any open string
            if fragment.count('"') % 2 == 1:
                fragment += '"'
            # Balance braces
            open_braces = fragment.count('{') - fragment.count('}')
            if open_braces > 0:
                fragment += '}' * open_braces
            try:
                return json.loads(fragment)
            except json.JSONDecodeError:
                pass
        logger.warning(f"Failed to parse JSON from LLM response: {text[:200]}")
        return {'error': 'json_parse_failed', 'raw_text': text[:500]}


# ─────────────────────────────────────────────────────────────
# Provider Implementations
# ─────────────────────────────────────────────────────────────

class GoogleGeminiProvider(BaseLLMProvider):
    """Google Gemini (new unified google-genai SDK)."""
    def generate(self, prompt: str, system_prompt: str = '') -> Dict:
        self._throttle()
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.config.api_key)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
        )
        if self.config.json_mode:
            config.response_mime_type = 'application/json'
        response = client.models.generate_content(
            model=self.config.model or 'gemini-2.5-flash',
            contents=prompt,
            config=config,
        )
        return self._parse_json(response.text)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT models (also works for Azure OpenAI)."""
    def generate(self, prompt: str, system_prompt: str = '') -> Dict:
        self._throttle()
        from openai import OpenAI
        client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url or None,
            timeout=self.config.timeout,
        )
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        kwargs = {
            'model': self.config.model or 'gpt-4-turbo',
            'messages': messages,
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
        }
        if self.config.json_mode:
            kwargs['response_format'] = {'type': 'json_object'}

        response = client.chat.completions.create(**kwargs)
        return self._parse_json(response.choices[0].message.content)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude models."""
    def generate(self, prompt: str, system_prompt: str = '') -> Dict:
        self._throttle()
        from anthropic import Anthropic
        client = Anthropic(api_key=self.config.api_key)
        response = client.messages.create(
            model=self.config.model or 'claude-sonnet-4-20250514',
            max_tokens=self.config.max_tokens,
            system=system_prompt or 'You are a quantitative trading analyst. Return only valid JSON.',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=self.config.temperature,
        )
        return self._parse_json(response.content[0].text)


class OllamaProvider(BaseLLMProvider):
    """Ollama local models (also LM Studio, vLLM, any OpenAI-compatible local)."""
    def generate(self, prompt: str, system_prompt: str = '') -> Dict:
        self._throttle()
        import requests
        base = self.config.base_url or 'http://127.0.0.1:11434'
        model_id = self.config.model or 'mistral:latest'

        # Use short connect timeout (3s) — if Ollama isn't running, fail fast
        # Read timeout stays longer (60s) for actual inference
        local_timeout = (3, 60)

        # Try OpenAI-compatible endpoint first
        endpoints = [
            f'{base}/v1/chat/completions',
            f'{base}/api/generate',
        ]

        for url in endpoints:
            try:
                if '/api/generate' in url:
                    payload = {
                        'model': model_id,
                        'prompt': f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
                        'stream': False,
                        'options': {'temperature': self.config.temperature},
                    }
                    resp = requests.post(url, json=payload, timeout=local_timeout)
                    if resp.ok:
                        data = resp.json()
                        return self._parse_json(data.get('response', '{}'))
                else:
                    payload = {
                        'model': model_id,
                        'messages': [],
                        'temperature': self.config.temperature,
                        'max_tokens': self.config.max_tokens,
                    }
                    if system_prompt:
                        payload['messages'].append({'role': 'system', 'content': system_prompt})
                    payload['messages'].append({'role': 'user', 'content': prompt})

                    resp = requests.post(url, json=payload, timeout=local_timeout)
                    if resp.ok:
                        data = resp.json()
                        text = data['choices'][0]['message']['content']
                        return self._parse_json(text)
            except Exception as e:
                logger.debug(f"Endpoint {url} failed: {e}")
                continue

        logger.error(f"All Ollama endpoints failed for model {model_id}")
        return {'error': 'all_endpoints_failed'}


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    Generic OpenAI-compatible provider.
    Works with: Together, Groq, DeepSeek, Mistral, Fireworks, Anyscale, vLLM, etc.
    Just set base_url to the provider's endpoint.
    """
    def generate(self, prompt: str, system_prompt: str = '') -> Dict:
        self._throttle()
        from openai import OpenAI
        client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        kwargs = {
            'model': self.config.model,
            'messages': messages,
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
        }
        if self.config.json_mode:
            try:
                kwargs['response_format'] = {'type': 'json_object'}
            except Exception:
                pass  # Not all compatible APIs support this

        response = client.chat.completions.create(**kwargs)
        return self._parse_json(response.choices[0].message.content)


class CohereProvider(BaseLLMProvider):
    """Cohere Command models."""
    def generate(self, prompt: str, system_prompt: str = '') -> Dict:
        self._throttle()
        import cohere
        client = cohere.Client(self.config.api_key)
        response = client.chat(
            model=self.config.model or 'command-r-plus',
            preamble=system_prompt or 'You are a quantitative trading analyst. Return only valid JSON.',
            message=prompt,
            temperature=self.config.temperature,
        )
        return self._parse_json(response.text)


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace Inference API or local transformers."""
    def generate(self, prompt: str, system_prompt: str = '') -> Dict:
        self._throttle()
        if self.config.api_key:
            # Use HF Inference API
            import requests
            headers = {'Authorization': f'Bearer {self.config.api_key}'}
            url = f'https://api-inference.huggingface.co/models/{self.config.model}'
            payload = {
                'inputs': f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
                'parameters': {
                    'temperature': self.config.temperature,
                    'max_new_tokens': self.config.max_tokens,
                    'return_full_text': False,
                }
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=self.config.timeout)
            if resp.ok:
                data = resp.json()
                text = data[0]['generated_text'] if isinstance(data, list) else str(data)
                return self._parse_json(text)
            return {'error': f'HF API error: {resp.status_code}'}
        else:
            # Local transformers
            from transformers import pipeline
            pipe = pipeline('text-generation', model=self.config.model,
                            device_map='auto', torch_dtype='auto')
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            result = pipe(full_prompt, max_new_tokens=self.config.max_tokens,
                          temperature=self.config.temperature, return_full_text=False)
            return self._parse_json(result[0]['generated_text'])


# ─────────────────────────────────────────────────────────────
# Provider Registry & Router
# ─────────────────────────────────────────────────────────────

PROVIDER_MAP = {
    'google': GoogleGeminiProvider,
    'gemini': GoogleGeminiProvider,
    'openai': OpenAIProvider,
    'gpt': OpenAIProvider,
    'anthropic': AnthropicProvider,
    'claude': AnthropicProvider,
    'ollama': OllamaProvider,
    'local': OllamaProvider,
    'lmstudio': OllamaProvider,
    'vllm': OllamaProvider,
    'together': OpenAICompatibleProvider,
    'groq': OpenAICompatibleProvider,
    'deepseek': OpenAICompatibleProvider,
    'mistral': OpenAICompatibleProvider,
    'fireworks': OpenAICompatibleProvider,
    'anyscale': OpenAICompatibleProvider,
    'cohere': CohereProvider,
    'huggingface': HuggingFaceProvider,
    'hf': HuggingFaceProvider,
    'custom': OpenAICompatibleProvider,
}

# Known base URLs for convenience
KNOWN_BASE_URLS = {
    'together': 'https://api.together.xyz/v1',
    'groq': 'https://api.groq.com/openai/v1',
    'deepseek': 'https://api.deepseek.com/v1',
    'mistral': 'https://api.mistral.ai/v1',
    'fireworks': 'https://api.fireworks.ai/inference/v1',
    'anyscale': 'https://api.endpoints.anyscale.com/v1',
}


class LLMRouter:
    """
    Routes queries to LLM providers with fallback chains, caching, and monitoring.
    """

    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl = 300  # 5 minute cache
        self._stats: Dict[str, Dict] = {}

    def add_provider(self, name: str, config: LLMConfig) -> 'LLMRouter':
        """Register an LLM provider."""
        provider_type = config.provider.lower()
        cls = PROVIDER_MAP.get(provider_type)
        if cls is None:
            raise ValueError(f"Unknown provider: {provider_type}. Available: {list(PROVIDER_MAP.keys())}")

        # Auto-fill base_url for known providers
        if not config.base_url and provider_type in KNOWN_BASE_URLS:
            config.base_url = KNOWN_BASE_URLS[provider_type]

        # Auto-detect API key from env if not provided
        if not config.api_key:
            env_keys = {
                'google': 'REASONING_LLM_KEY',
                'gemini': 'REASONING_LLM_KEY',
                'openai': 'OPENAI_API_KEY',
                'anthropic': 'ANTHROPIC_API_KEY',
                'together': 'TOGETHER_API_KEY',
                'groq': 'GROQ_API_KEY',
                'deepseek': 'DEEPSEEK_API_KEY',
                'mistral': 'MISTRAL_API_KEY',
                'cohere': 'COHERE_API_KEY',
                'huggingface': 'HUGGINGFACE_TOKEN',
            }
            env_key = env_keys.get(provider_type, '')
            if env_key:
                config.api_key = os.environ.get(env_key, '')

        self.providers[name] = cls(config)
        self._stats[name] = {'calls': 0, 'errors': 0, 'avg_latency': 0.0}
        logger.info(f"Registered LLM provider: {name} ({provider_type}/{config.model})")
        return self

    def add_from_env(self) -> 'LLMRouter':
        """
        Auto-detect and register providers from environment variables.
        Scans for known API keys and creates providers automatically.
        """
        env_map = [
            # gemini-2.5-flash: 20 req/day free tier (primary — best quality)
            ('REASONING_LLM_KEY', 'google', 'gemini-2.5-flash', 'gemini'),
            # gemini-2.0-flash-lite: disabled — this API key only has quota for gemini-2.5-flash
            # ('REASONING_LLM_KEY', 'google', 'gemini-2.0-flash-lite', 'gemini_fast'),
            ('OPENAI_API_KEY', 'openai', 'gpt-4-turbo', 'openai'),
            ('ANTHROPIC_API_KEY', 'anthropic', 'claude-sonnet-4-20250514', 'claude'),
            ('TOGETHER_API_KEY', 'together', 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', 'together'),
            ('GROQ_API_KEY', 'groq', 'llama-3.1-70b-versatile', 'groq'),
            ('DEEPSEEK_API_KEY', 'deepseek', 'deepseek-chat', 'deepseek'),
            ('COHERE_API_KEY', 'cohere', 'command-r-plus', 'cohere'),
        ]

        # ── Remote Ollama GPU is the PRIMARY and ONLY LLM ──
        remote_ollama_url = os.environ.get('OLLAMA_REMOTE_URL', '').strip()
        if remote_ollama_url:
            self.add_provider('remote_gpu', LLMConfig(
                provider='ollama',
                model=os.environ.get('OLLAMA_REMOTE_MODEL', 'mistral'),
                base_url=remote_ollama_url,
                timeout=120,
            ))
            logger.info(f"Registered remote Ollama GPU: {remote_ollama_url}")
        else:
            # Fallback: try local Ollama if no remote URL
            local_base_url = (
                os.environ.get('LLM_BASE_URL', '').strip()
                or os.environ.get('OLLAMA_BASE_URL', '').strip()
                or os.environ.get('OLLAMA_HOST', '').strip()
            )
            _local_model = os.environ.get('OLLAMA_MODEL', '').strip() or 'mistral:latest'
            self.add_provider('local', LLMConfig(
                provider='ollama',
                model=_local_model,
                base_url=local_base_url,
            ))

        return self

    def query(self, prompt: str, system_prompt: str = '',
              fallback_chain: Optional[List[str]] = None,
              cache: bool = True) -> Dict:
        """
        Query LLM with automatic fallback chain.

        Args:
            prompt: The user prompt (after math injection + constraints)
            system_prompt: System-level instructions
            fallback_chain: List of provider names to try in order
            cache: Whether to cache results

        Returns:
            Parsed JSON dict from LLM
        """
        chain = fallback_chain or list(self.providers.keys())

        # Check cache
        if cache:
            cache_key = hashlib.md5(f"{prompt}{system_prompt}".encode()).hexdigest()
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                if time.time() - entry['time'] < self._cache_ttl:
                    return entry['data']

        for provider_name in chain:
            if provider_name not in self.providers:
                continue

            provider = self.providers[provider_name]
            start = time.time()

            try:
                result = provider.generate(prompt, system_prompt)
                latency = time.time() - start

                # Update stats
                stats = self._stats[provider_name]
                stats['calls'] += 1
                stats['avg_latency'] = (
                    (stats['avg_latency'] * (stats['calls'] - 1) + latency) / stats['calls']
                )

                # json_parse_failed with raw_text means the LLM DID respond — just not valid JSON.
                # Treat as a usable partial result (caller can use raw_text for free-form responses).
                _is_real_error = (
                    'error' in result and
                    not (result.get('error') == 'json_parse_failed' and result.get('raw_text'))
                )
                if not _is_real_error:
                    # Cache successful result
                    if cache:
                        self._cache[cache_key] = {'data': result, 'time': time.time()}

                    logger.info(f"LLM [{provider_name}] responded in {latency:.1f}s")
                    return result
                else:
                    logger.warning(f"LLM [{provider_name}] returned error: {result.get('error')}")
                    self._stats[provider_name]['errors'] += 1

            except Exception as e:
                latency = time.time() - start
                self._stats[provider_name]['errors'] += 1
                logger.warning(f"LLM [{provider_name}] failed ({latency:.1f}s): {e}")
                continue

        # All providers failed — return safe default (rule-based fallback used)
        logger.warning("All LLM providers failed — using rule-based fallback")
        return {
            'error': 'all_providers_failed',
            'market_regime': 'UNKNOWN',
            'reasoning_trace': 'All LLM providers unavailable. Using rule-based fallback.',
            'confidence_score': 0,
            'suggested_config_update': {},
            'macro_bias': 0.0,
        }

    def get_stats(self) -> Dict:
        """Return provider performance stats."""
        return dict(self._stats)

    def list_providers(self) -> List[str]:
        """List registered provider names."""
        return list(self.providers.keys())
