"""MCPBridge — MCP (Model Context Protocol) tool discovery and calling.

Extracted from agent-purple's mcp_bridge.py and shared so all agents
can use the same JSON-RPC 2.0 MCP interface without reimplementing it.

Protocol: POST {endpoint}/mcp?session_id=...
Methods: tools/list, tools/call
"""
from __future__ import annotations

import httpx

TOOL_TIMEOUT = 30.0


class MCPBridge:
    """Discover and call MCP tools from a green agent endpoint.

    Usage:
        bridge = MCPBridge(endpoint="http://green-agent:9009")
        tools = await bridge.discover(session_id="abc")
        result = await bridge.call("query_soql", {"query": "SELECT..."}, session_id="abc")
    """

    def __init__(self, endpoint: str, timeout: float = TOOL_TIMEOUT):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    async def discover(self, session_id: str = "") -> list[dict]:
        """List available tools via MCP tools/list.

        Returns Anthropic-format tool dicts (name, description, input_schema).
        Returns [] if endpoint has no MCP or is unreachable.
        """
        url = f"{self.endpoint}/mcp"
        if session_id:
            url = f"{url}?session_id={session_id}"

        payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
                data = resp.json()
        except Exception:
            return []

        tools = data.get("result", {}).get("tools", [])
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": (
                    t.get("inputSchema") or t.get("input_schema") or
                    {"type": "object", "properties": {}}
                ),
            }
            for t in tools
        ]

    async def call(
        self,
        tool_name: str,
        arguments: dict,
        session_id: str = "",
    ) -> dict:
        """Call a tool via MCP tools/call.

        Returns the parsed tool result dict.
        Raises on network/parse errors.
        """
        url = f"{self.endpoint}/mcp?session_id={session_id}"
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            data = resp.json()

        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")

        result = data.get("result", {})
        content = result.get("content", [])
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and first.get("type") == "text":
                import json
                try:
                    return json.loads(first["text"])
                except (json.JSONDecodeError, KeyError):
                    return {"text": first.get("text", "")}
        return result
