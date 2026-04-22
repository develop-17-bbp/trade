"""MCP (Model Context Protocol) server exposing ACT runtime artifacts to
Claude Code sessions over HTTP.

Run on the GPU box with:
    python -m src.mcp_server.act_mcp

Routed via Cloudflare tunnel (infra/cloudflared/config.yml) to `mcp.<domain>`,
gated by Cloudflare Access. Secondary token check via X-MCP-Token header.
"""
