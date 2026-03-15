from __future__ import annotations

import base64
import os

from pydantic_ai.mcp import MCPServerStreamableHTTP


def build_langfuse_mcp_server() -> MCPServerStreamableHTTP | None:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")
    if not public_key or not secret_key or not host:
        return None
    endpoint = _build_mcp_endpoint(host)
    headers = {"Authorization": f"Basic {_build_basic_auth_token(public_key, secret_key)}"}
    return MCPServerStreamableHTTP(
        endpoint,
        headers=headers,
        tool_prefix="langfuse",
    )


def _build_mcp_endpoint(host: str) -> str:
    normalized = host.rstrip("/")
    if normalized.endswith("/api/public/mcp"):
        return normalized
    return f"{normalized}/api/public/mcp"


def _build_basic_auth_token(public_key: str, secret_key: str) -> str:
    raw = f"{public_key}:{secret_key}".encode("utf-8")
    return base64.b64encode(raw).decode("ascii")
