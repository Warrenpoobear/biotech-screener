import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _request(request_id: int, method: str, params: dict | None = None) -> str:
    payload = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        payload["params"] = params
    return json.dumps(payload, sort_keys=True)


def _content_length_frame(payload: str) -> str:
    return f"Content-Length: {len(payload.encode('utf-8'))}\r\n\r\n{payload}"


def _decode_content_length_response(output: str) -> dict:
    delimiter = "\r\n\r\n" if "\r\n\r\n" in output else "\n\n"
    header, body = output.split(delimiter, 1)
    assert header.startswith("Content-Length: ")
    expected_length = int(header.split(": ", 1)[1])
    assert len(body.encode("utf-8")) == expected_length
    return json.loads(body)


def test_hermes_mcp_stdio_exposes_read_only_tools():
    requests = [
        _request(
            1,
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0"},
            },
        ),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}, sort_keys=True),
        _request(2, "tools/list", {}),
        _request(3, "tools/call", {"name": "fleet_context_snapshot", "arguments": {}}),
    ]

    result = subprocess.run(
        [sys.executable, "-m", "mcp_server.hermes_server"],
        input="\n".join(requests) + "\n",
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    by_id = {response["id"]: response for response in responses if "id" in response}

    assert by_id[1]["result"]["serverInfo"]["name"] == "Hermes"
    tools = {tool["name"] for tool in by_id[2]["result"]["tools"]}
    assert {
        "agents_get",
        "agents_list",
        "fleet_context_snapshot",
        "knowledge_read",
        "skills_read",
    }.issubset(tools)

    snapshot_content = by_id[3]["result"]["content"][0]["text"]
    snapshot = json.loads(snapshot_content)
    assert snapshot["server"] == "Hermes"
    assert snapshot["mode"] == "read-only"
    assert snapshot["agent_registry"]["exists"] is True


def test_knowledge_read_contradiction_ledger_prefers_latest_md(tmp_path, monkeypatch):
    """MCP knowledge_read must find contradiction_ledger/latest.md (not only .json)."""
    ledger_dir = PROJECT_ROOT / "artifacts" / "ops" / "contradiction_ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    md_path = ledger_dir / "latest.md"
    md_path.write_text("# Contradiction Ledger\n\n## 1. Hard Contradictions (0)\n\nNone.\n")

    request = _request(4, "tools/call", {"name": "knowledge_read", "arguments": {"artifact": "contradiction_ledger"}})
    result = subprocess.run(
        [sys.executable, "-m", "mcp_server.hermes_server"],
        input=request + "\n",
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout.strip().splitlines()[-1])
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["exists"] is True
    assert payload["selected_path"].endswith("contradiction_ledger/latest.md")
    assert "Hard Contradictions" in payload["content"]


def test_hermes_mcp_stdio_accepts_content_length_framing():
    request = _request(
        1,
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "0"},
        },
    )

    result = subprocess.run(
        [sys.executable, "-m", "mcp_server.hermes_server"],
        input=_content_length_frame(request),
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    response = _decode_content_length_response(result.stdout)
    assert response["id"] == 1
    assert response["result"]["serverInfo"]["name"] == "Hermes"
