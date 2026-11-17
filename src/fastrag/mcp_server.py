"""MCP server exposing fastRag's retrieval tools to MCP-capable agent
platforms (Claude Desktop, IDE agents, etc.). It calls the HTTP API with a
tenant API key, so the key's group ceiling applies as usual.

    FASTRAG_MCP_API_URL=http://fastrag:8000 FASTRAG_MCP_TOKEN=fr_... fastrag mcp

Needs the `mcp` extra.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


def build_server() -> Any:
    try:  # mcp >= 2
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server  # type: ignore[attr-defined,no-redef]

    base = os.environ.get("FASTRAG_MCP_API_URL", "http://localhost:8000").rstrip("/")
    token = os.environ.get("FASTRAG_MCP_TOKEN")
    if not token:
        raise SystemExit("set FASTRAG_MCP_TOKEN to a tenant API key")
    profile = os.environ.get("FASTRAG_MCP_PROFILE", "customer")
    client = httpx.AsyncClient(base_url=base, headers={"Authorization": f"Bearer {token}"}, timeout=15)
    mcp = Server("fastrag")

    async def post(path: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = await client.post(path, json=body)
        if resp.status_code == 422:
            return {"error": resp.json()}
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    @mcp.tool()
    async def search_documents(query: str, top_k: int = 5, collections: list[str] | None = None) -> dict[str, Any]:
        """Search company documents (policies, project info, FAQs) with a standalone question,
        written in the documents' language (translate other languages to Vietnamese first).
        Returns passages with citations and evidence signals. If signals.nothing_relevant is
        true, do not answer from these passages."""
        return await post(
            "/v1/search", {"query": query, "top_k": top_k, "collections": collections, "profile": profile}
        )

    @mcp.tool()
    async def query_listings(filters: dict[str, Any], sort: str | None = None, limit: int = 10) -> dict[str, Any]:
        """Find property listings. Filters: price_min/price_max ("3,5 tỷ"), area_min/area_max,
        bedrooms_min, district ("Q7"), project, property_type, status, direction, keywords.
        Sort: price_asc, price_desc, area_asc, area_desc, newest. Quote as_of when stating
        prices or availability."""
        return await post("/v1/listings/query", {"filters": filters, "sort": sort, "limit": limit, "profile": profile})

    @mcp.tool()
    async def get_listings(ids: list[str]) -> dict[str, Any]:
        """Refresh listings by the ids returned earlier (current price, status, as_of)."""
        return await post("/v1/listings/lookup", {"ids": ids, "profile": profile})

    return mcp


def main() -> None:
    build_server().run()
