"""MCP scientific cartography integration tests."""

from __future__ import annotations

import json
import subprocess
import sys


def _request(req_id: int, method: str, params: dict | None = None) -> str:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {},
        },
        sort_keys=True,
    )


def _run_mcp_requests(requests: list[str]) -> list[dict]:
    init = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0"},
            },
        },
        sort_keys=True,
    )
    initialized = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}, sort_keys=True)
    result = subprocess.run(
        [sys.executable, "-m", "mcp_server.server"],
        input="\n".join([init, initialized, *requests]) + "\n",
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def test_mcp_server_exposes_scientific_cartography_tool():
    """Active package MCP server should expose get_atlas_data."""
    responses = _run_mcp_requests([_request(2, "tools/list")])
    tools = responses[-1]["result"]["tools"]
    names = {tool["name"] for tool in tools}
    assert "get_atlas_data" in names


def test_get_atlas_data_schemas_returns_package_schemas():
    """get_atlas_data(category='schemas') should return real cartography schema modules."""
    responses = _run_mcp_requests(
        [
            _request(
                2,
                "tools/call",
                {"name": "get_atlas_data", "arguments": {"category": "schemas"}},
            )
        ]
    )
    content = responses[-1]["result"]["content"][0]["text"]
    payload = json.loads(content)
    assert payload["module"] == "scientific_cartography"
    assert "program_schema" in payload["schemas"]
    assert "disease_schema" in payload["schemas"]
    assert payload["governance"]["read_only_diagnostic"] is True
