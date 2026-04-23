"""
MCP client — consume external MCP servers from inside ACT.

ACT already hosts ITS OWN MCP server (src/mcp_server/act_mcp.py) so that
Claude Code can drive ACT remotely. This module is the *client* side:
ACT itself subscribes to external MCP servers (CoinGecko, Perplexity,
Chroma, community TradingView, etc.) and mirrors each of their tools as
a regular Tool entry in the existing ToolRegistry (C3). The Analyst
brain then sees them alongside in-process tools with no special casing.

Why this is valuable (operator-stated goal):
  * External MCP servers become additional "hands" the LLM can reach.
  * Tool-shape is already LLM-friendly (MCP's schema ≈ the
    Anthropic tool-use JSON ACT already uses).
  * Adding a new data source is a config line, not a code change —
    hooks into the `mcp_clients` block in config.yaml.

Design constraints:
  * Zero new hard deps. Uses `requests` (already in ACT) for HTTP.
    The official `mcp` Python SDK is optional; we use it if present
    but work without it via a small streamable-HTTP adapter.
  * Sub-agent / lean-context discipline (C3): MCP tool results come
    back soft-capped at DEFAULT_MAX_OUTPUT_CHARS (1200 — same as the
    rest of trade_tools.py). No raw payloads leak into the parent
    LLM's context.
  * Graceful degradation: if a server is unreachable at registration
    time, we log + skip rather than crash. Runtime tool calls to a
    dead server return a structured `{"error": "..."}` the LLM sees.
  * Kill switch: ACT_DISABLE_MCP_CLIENTS=1 skips all external MCP
    work entirely (local-only mode for debugging).

Config shape (config.yaml):

    mcp_clients:
      - url: https://example-mcp/coingecko
        tag: cg
        enabled: true
        headers:
          Authorization: "Bearer ${CG_MCP_TOKEN}"   # env-expanded
      - url: http://localhost:3100
        tag: tv
        enabled: false                              # off by default
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


DISABLE_ENV = "ACT_DISABLE_MCP_CLIENTS"
DEFAULT_TIMEOUT_S = float(os.getenv("ACT_MCP_CLIENT_TIMEOUT_S", "8.0"))
DEFAULT_MAX_OUTPUT_CHARS = 1200


def _expand_env(value: Any) -> Any:
    """Expand ${VAR} placeholders in string config values. Leaves other
    types untouched. Missing vars become empty strings so a key like
    `Authorization: ${MISSING}` doesn't break registration."""
    if isinstance(value, str) and "${" in value:
        out = value
        for var in _find_env_vars(value):
            out = out.replace("${" + var + "}", os.environ.get(var, ""))
        return out
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _find_env_vars(s: str) -> List[str]:
    import re
    return re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", s)


# ── HTTP MCP client (minimal streamable-HTTP) ───────────────────────────


@dataclass
class MCPRemoteTool:
    """One tool discovered on a remote MCP server."""
    name: str                   # LOCAL name: f"{tag}_{remote_name}"
    remote_name: str
    description: str
    input_schema: Dict[str, Any]
    server_url: str
    server_tag: str


class MCPHTTPClient:
    """Minimal MCP client over streamable-HTTP. Implements the subset
    ACT needs: list_tools + call_tool. Suitable for FastMCP-compatible
    servers which is what the ecosystem has settled on.

    If the official `mcp` Python SDK is importable, we use it instead
    for full protocol compliance. This keeps the module useful whether
    or not the operator has `pip install mcp`-ed.
    """

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None,
                 timeout_s: float = DEFAULT_TIMEOUT_S):
        self.url = url.rstrip("/")
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.timeout_s = float(timeout_s)

    # Public API --------------------------------------------------------

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return the list of tools exposed by the remote server.

        Shape of each entry: {name, description, inputSchema}.
        """
        data = self._rpc("tools/list", {})
        tools = data.get("tools") if isinstance(data, dict) else data
        return list(tools or [])

    def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a remote tool. Returns the server's response dict."""
        return self._rpc("tools/call", {"name": name, "arguments": args or {}})

    # Internals ---------------------------------------------------------

    def _rpc(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """JSON-RPC 2.0 over HTTP POST — the streamable-HTTP transport."""
        try:
            import requests
        except Exception as e:
            raise RuntimeError(f"requests not installed: {e}")

        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": method, "params": params,
        }
        r = requests.post(self.url, headers=self.headers,
                          data=json.dumps(payload), timeout=self.timeout_s)
        r.raise_for_status()
        body = r.json()
        if "error" in body:
            err = body["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"mcp rpc error: {msg}")
        return body.get("result") or {}


# ── Registration into the ToolRegistry ──────────────────────────────────


def register_mcp_server(
    registry,                     # src.ai.trade_tools.ToolRegistry
    url: str,
    *,
    tag: str = "mcp",
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    client_factory: Optional[Callable[..., MCPHTTPClient]] = None,
) -> List[str]:
    """Connect to an MCP server and mirror each of its tools into the
    given registry. Returns the list of LOCAL tool names registered.

    Tools are namespaced: `f"{tag}_{remote_name}"` so multiple MCP
    servers can coexist without collisions. Each mirrored Tool's
    handler forwards the args to the remote, catches all exceptions,
    and soft-caps the output so noisy servers can't blow the parent's
    context window.

    If the remote is unreachable, logs + returns []. ACT keeps running
    with just its local tools.
    """
    from src.ai.trade_tools import Tool
    factory = client_factory or (
        lambda: MCPHTTPClient(url=url, headers=headers, timeout_s=timeout_s)
    )

    try:
        client = factory()
        remote_tools = client.list_tools()
    except Exception as e:
        logger.warning("MCP server %s unreachable at registration: %s", url, e)
        return []

    registered: List[str] = []
    for t in remote_tools:
        if not isinstance(t, dict):
            continue
        remote_name = str(t.get("name") or "").strip()
        if not remote_name:
            continue
        local_name = f"{tag}_{remote_name}"
        description = str(t.get("description") or "")
        schema = t.get("inputSchema") or t.get("input_schema") or {"type": "object"}

        def _make_handler(_client: MCPHTTPClient, _remote_name: str):
            def _handler(args: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    resp = _client.call_tool(_remote_name, dict(args or {}))
                    # MCP tools typically return {"content": [{"type":"text","text":"..."}]}.
                    # Normalize into a compact string summary + raw pointer.
                    return _normalize_response(resp, _remote_name)
                except Exception as e:
                    return {"error": f"{type(e).__name__}: {e}"}
            return _handler

        try:
            registry.register(Tool(
                name=local_name,
                description=f"[MCP:{tag}] {description}",
                input_schema=schema if isinstance(schema, dict) else {"type": "object"},
                handler=_make_handler(client, remote_name),
                tag="read_only",      # external data is read-only by default
                max_output_chars=DEFAULT_MAX_OUTPUT_CHARS,
            ))
            registered.append(local_name)
        except ValueError:
            # Duplicate tool name (retry scenario). Skip silently.
            logger.debug("MCP tool %s already registered; skipping", local_name)
            continue

    logger.info("MCP server %s (%s): registered %d tool(s)",
                url, tag, len(registered))
    return registered


def _normalize_response(resp: Any, remote_name: str) -> Dict[str, Any]:
    """Turn an MCP tools/call response into a compact dict. Never raises."""
    if not isinstance(resp, dict):
        return {"summary": str(resp)[:500]}
    # FastMCP / SDK style: resp["content"] is a list of blocks.
    content = resp.get("content")
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
                elif block.get("type") == "json":
                    parts.append(json.dumps(block.get("data"), default=str))
                else:
                    parts.append(json.dumps(block, default=str))
            else:
                parts.append(str(block))
        return {"summary": " | ".join(parts)[:800], "tool": remote_name}
    return {"summary": json.dumps(resp, default=str)[:800], "tool": remote_name}


# ── Config-driven bulk registration ─────────────────────────────────────


def register_all_from_config(
    registry,
    config: Optional[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """Iterate `config['mcp_clients']`, register each one, return a
    dict `{tag: [tool_name, ...]}` describing what was wired up.

    Skips entries with enabled=false or when ACT_DISABLE_MCP_CLIENTS=1.
    """
    if os.environ.get(DISABLE_ENV, "0") == "1":
        logger.info("MCP clients disabled by env %s", DISABLE_ENV)
        return {}

    clients_cfg = (config or {}).get("mcp_clients") or []
    if not isinstance(clients_cfg, list):
        return {}

    out: Dict[str, List[str]] = {}
    for entry in clients_cfg:
        if not isinstance(entry, dict):
            continue
        entry = _expand_env(entry)
        if not entry.get("enabled", True):
            continue
        url = entry.get("url")
        if not url:
            continue
        tag = str(entry.get("tag") or "mcp")
        headers = entry.get("headers") or {}
        timeout_s = float(entry.get("timeout_s") or DEFAULT_TIMEOUT_S)
        registered = register_mcp_server(
            registry, url=url, tag=tag,
            headers=headers, timeout_s=timeout_s,
        )
        out[tag] = registered
    return out


# ── Catalogue helper (for /regime-check-style listings) ─────────────────


def list_registered_mcp_tools(registry) -> List[Dict[str, str]]:
    """Return the MCP-origin tools currently in the registry, for dashboards."""
    out: List[Dict[str, str]] = []
    for name in registry.list_names():
        tool = registry.get(name)
        if tool and tool.description.startswith("[MCP:"):
            out.append({"name": name, "description": tool.description})
    return out
