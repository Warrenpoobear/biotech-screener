---
name: cursor-hermes-mcp-debug
description: "Diagnose Cursor Hermes MCP discovery failures for the biotech screener repo. Covers repo-native server startup, cached discovery errors, direct JSON-RPC smoke tests, and local reload requirements."
when_to_use: "Hermes MCP shows red/error in Cursor, fleet_context_snapshot is unavailable, GetMcpTools reports a discovery failure, or .cursor/mcp.json was changed and Cursor still shows the old server state."
---

# Cursor Hermes MCP Debugger

For the biotech screener repo. Diagnose-only unless the operator asks for a config
change. This is IDE/tooling infrastructure only; it must not change scoring,
rulesets, production weights, or enforcement behavior.

---

## Current architecture

Hermes MCP is repo-native. Cursor should launch the server from the checkout:

```json
{
  "mcpServers": {
    "hermes": {
      "command": "python3",
      "args": ["-m", "mcp_server.hermes_server"],
      "env": {
        "HERMES_REPO": "${workspaceFolder}",
        "HERMES_AGENTS_DIR": "${workspaceFolder}/agents",
        "PYTHONPATH": "${workspaceFolder}"
      }
    }
  }
}
```

The server implementation lives at:

```text
mcp_server/hermes_server.py
```

It is stdlib-only and read-only. It does not require the external
`../hermes-agent` checkout, the Python MCP SDK, or a running Hermes desktop/local
process.

---

## Tools exposed

Expected `tools/list` result:

- `fleet_context_snapshot`
- `agents_list`
- `agents_get`
- `skills_read`
- `knowledge_read`

All tools are read-only. They read repo files such as `agents/AGENT_REGISTRY.json`,
agent identity documents, `.cursor/rules/hermes-context.mdc`, and generated
`artifacts/ops/*` knowledge-layer outputs if present.

---

## Failure taxonomy

### Class A - Stale Cursor MCP discovery cache

**Signature:** Direct server smoke tests pass, but Cursor still reports:

```text
This MCP server failed during live tool discovery. Its tools are unavailable until the connection is fixed.
```

**Cause:** Cursor's MCP host cached a previous failed discovery attempt. This can
happen after `.cursor/mcp.json` changes from the old external checkout command to
the repo-native server.

**Resolution:** Reload the Cursor window or restart Cursor in the repo.

```text
Cmd/Ctrl+Shift+P -> Developer: Reload Window
```

After reload, MCP settings should rediscover `hermes` and show the five tools.

---

### Class B - Cursor cannot resolve `python3`

**Signature:** Direct shell smoke works in an interactive terminal, but Cursor MCP
still fails after reload.

**Cause:** Cursor's environment cannot resolve `python3`, especially on local
Windows/WSL2 or non-standard Python installs.

**Diagnostic:**

```bash
command -v python3
python3 --version
```

**Resolution:** Set `.cursor/mcp.json` `command` to the absolute interpreter path
visible to Cursor, for example:

```json
"command": "/usr/bin/python3"
```

Keep the args and env unchanged.

---

### Class C - Module import fails outside repo cwd

**Signature:**

```text
No module named mcp_server.hermes_server
```

**Cause:** Cursor launched the process from a working directory outside the repo
and `PYTHONPATH` is missing.

**Resolution:** Keep this env entry in `.cursor/mcp.json`:

```json
"PYTHONPATH": "${workspaceFolder}"
```

---

### Class D - Old external Hermes checkout path

**Signature:** `.cursor/mcp.json` points at:

```text
${workspaceFolder}/../hermes-agent/hermes-mcp-serve
```

and discovery fails in cloud or fresh checkouts.

**Cause:** The sibling `../hermes-agent` checkout is not portable and may not exist.

**Resolution:** Replace it with the repo-native command shown in "Current
architecture".

---

## Direct smoke test

Run from any directory:

```bash
HERMES_REPO=/workspace \
HERMES_AGENTS_DIR=/workspace/agents \
PYTHONPATH=/workspace \
python3 -m mcp_server.hermes_server <<'EOF'
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"manual-test","version":"0"}}}
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"fleet_context_snapshot","arguments":{}}}
EOF
```

Expected:

- `initialize` returns `serverInfo.name == "Hermes"`
- `tools/list` returns the five read-only tools
- `fleet_context_snapshot` returns agent registry metadata

For a local checkout, replace `/workspace` with the repo path.

---

## Test files

Focused regression coverage:

```text
tests/test_hermes_mcp_server.py
```

It verifies:

- newline-delimited stdio JSON-RPC
- `Content-Length` stdio framing
- tool discovery
- `fleet_context_snapshot`

---

## Decision rule

If direct smoke tests pass but Cursor MCP remains red, do not keep changing the
server. Reload Cursor first. Treat a still-red MCP entry after reload as an
environment/config discovery issue, not a Hermes server implementation issue.
