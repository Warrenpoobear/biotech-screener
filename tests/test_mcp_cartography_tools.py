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
    """Send JSON-RPC requests and read replies before closing stdin.

    Closing stdin as soon as the writes finish makes mcp 1.22 drop the
    in-flight response with ClosedResourceError. The pipe stays open until
    every request id has a reply.
    """
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
    wanted = {1}
    for raw in requests:
        wanted.add(json.loads(raw)["id"])
    proc = subprocess.Popen(
        [sys.executable, "-m", "mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(init + "\n" + initialized + "\n" + "\n".join(requests) + "\n")
    proc.stdin.flush()
    messages = []
    seen: set[int] = set()
    try:
        while seen != wanted:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            message = json.loads(line)
            messages.append(message)
            if "id" in message:
                seen.add(message["id"])
    finally:
        proc.kill()
        proc.wait(timeout=5)
    missing = wanted - seen
    assert not missing, f"MCP server did not answer ids {sorted(missing)}"
    return messages


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
