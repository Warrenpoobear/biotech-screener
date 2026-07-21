"""Read-only biotech-screener MCP server (`biotech-mcp`).

A stdlib-only, JSON-RPC 2.0 / NDJSON MCP server exposing a small, typed,
**read-only** view of the biotech model's existing diagnostic artifacts so
Hermes (and other MCP clients) can answer questions like "what changed in
today's snapshot?", "are the governance gates passing?", "is Event-EV
feature coverage starved?", "which disease maps have unknown mappings?",
"did Semgrep find anything governance-relevant?" — without touching
production code, config, jobs, git, or the network.

Hard constraints (enforced by construction — there is no code path that does
any of these): no shell escape, no arbitrary file read (every tool is pinned
to a known artifact under a fixed subtree, with date inputs validated against
a strict YYYY-MM-DD pattern and path-escape guards), no git write, no network
I/O, no mutation, no trading, no "repair" tool. `run_readonly_diagnostics`
aggregates already-generated artifacts in-process; it does NOT execute the
diagnostics scripts.

Style mirrors mcp_server/hermes_server.py (the repo's existing read-only MCP).
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

SERVER_NAME = "biotech-mcp"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2025-06-18"

REPO_ROOT = Path(os.environ.get("BIOTECH_MCP_REPO", Path(__file__).resolve().parent.parent)).resolve()

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_OUTPUT_FRAMING = "ndjson"

# Bounded fan-out / output caps (read-only safety + context hygiene).
MAX_LEDGER_RECORDS = 50
MAX_CSV_SAMPLE_ROWS = 5
MAX_DIR_ENTRIES = 500


# --------------------------------------------------------------------------- #
# JSON-RPC / MCP transport (mirrors hermes_server.py)
# --------------------------------------------------------------------------- #
def main() -> int:
    for line in _read_message_payloads():
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            _write(_error_response(None, -32700, f"Parse error: {exc.msg}"))
            continue

        messages = message if isinstance(message, list) else [message]
        responses = [_handle_message(item) for item in messages]
        responses = [response for response in responses if response is not None]
        if not responses:
            continue
        _write(responses if isinstance(message, list) else responses[0])
    return 0


def _read_message_payloads():
    global _OUTPUT_FRAMING
    stdin = sys.stdin.buffer
    while True:
        raw_line = stdin.readline()
        if raw_line == b"":
            return
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith(b"content-length:"):
            _OUTPUT_FRAMING = "content-length"
            length_text = stripped.split(b":", 1)[1].strip()
            try:
                content_length = int(length_text)
            except ValueError:
                yield "{"
                continue
            while True:
                header_line = stdin.readline()
                if header_line in (b"", b"\n", b"\r\n"):
                    break
            body = stdin.read(content_length)
            if body == b"":
                return
            yield body.decode("utf-8")
            continue
        yield raw_line.decode("utf-8").strip()


def _handle_message(message: Any) -> dict[str, Any] | None:
    if not isinstance(message, dict):
        return _error_response(None, -32600, "Invalid Request")
    request_id = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}
    if "id" not in message:
        return None
    if not isinstance(method, str):
        return _error_response(request_id, -32600, "Invalid Request")
    try:
        if method == "initialize":
            return _success_response(request_id, _initialize_result(params))
        if method == "ping":
            return _success_response(request_id, {})
        if method == "tools/list":
            return _success_response(request_id, {"tools": _tool_definitions()})
        if method == "tools/call":
            return _success_response(request_id, _call_tool(params))
        if method == "resources/list":
            return _success_response(request_id, {"resources": []})
        if method == "prompts/list":
            return _success_response(request_id, {"prompts": []})
    except ValueError as exc:
        return _error_response(request_id, -32602, str(exc))
    except Exception as exc:  # pragma: no cover - defensive MCP boundary
        print(f"{SERVER_NAME} MCP internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return _error_response(request_id, -32603, "Internal error")
    return _error_response(request_id, -32601, f"Method not found: {method}")


def _initialize_result(params: dict[str, Any]) -> dict[str, Any]:
    requested_version = params.get("protocolVersion") if isinstance(params, dict) else None
    return {
        "protocolVersion": requested_version or PROTOCOL_VERSION,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "instructions": (
            "Read-only diagnostic view of the biotech screener: snapshots, "
            "governance gate verdicts, phase-2 health, rankings schema, "
            "Event-EV feature coverage, forward-eval IC ledger, Scientific "
            "Cartography status and disease-map artifacts, and Semgrep rules. "
            "This server never mutates data, config, jobs, git, or artifacts, "
            "and never executes scripts or network calls."
        ),
    }


# --------------------------------------------------------------------------- #
# Tool registry
# --------------------------------------------------------------------------- #
def _tool_definitions() -> list[dict[str, Any]]:
    date_prop = {
        "type": "string",
        "description": "Snapshot date YYYY-MM-DD. Omit for the latest available.",
        "pattern": r"^\d{4}-\d{2}-\d{2}$",
    }
    return [
        _tool(
            "list_snapshots",
            "List dated snapshot directories under data/snapshots/ with key-file presence.",
            {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "default": 30,
                    "description": "Max snapshots to return (most recent first).",
                }
            },
        ),
        _tool(
            "read_latest_snapshot_manifest",
            "Read snapshot_manifest.json for a snapshot (latest if date omitted).",
            {"date": date_prop},
        ),
        _tool(
            "read_gate_verdicts",
            "Read recent governance gate verdicts from artifacts/gate_verdict_ledger.jsonl "
            "(most recent first), plus optional per-snapshot EES gate diagnostics.",
            {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 5,
                    "description": "Number of recent ledger records to return.",
                },
                "date": date_prop,
            },
        ),
        _tool(
            "read_phase2_health",
            "Read phase2_health.json for a snapshot (latest if date omitted).",
            {"date": date_prop},
        ),
        _tool(
            "read_rankings_schema",
            "Read the rankings.csv schema (column headers, row count, small sample) "
            "for a snapshot. Schema-focused: does not return the full table.",
            {
                "date": date_prop,
                "sample_rows": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5,
                    "default": 0,
                    "description": "Optional number of sample rows (<=5).",
                },
            },
        ),
        _tool(
            "read_event_ev_feature_coverage",
            "Read the Scientific Cartography landscape_feature_coverage_report.json "
            "(Event-EV / expectation-model feature coverage) for the latest or given date.",
            {"date": date_prop},
        ),
        _tool(
            "read_forward_eval_ic_ledger",
            "Read artifacts/readiness/forward_eval_ic_baseline.json, returning the "
            "most recent IC observations and window/status fields.",
            {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                    "description": "Number of most-recent observations to return.",
                }
            },
        ),
        _tool(
            "read_scientific_cartography_status",
            "Read scientific_cartography_status.json (governance flags, warnings, errors) "
            "for the latest or given cartography run date.",
            {"date": date_prop},
        ),
        _tool(
            "list_disease_map_artifacts",
            "List the disease-map / asset-indication artifacts in a Scientific Cartography "
            "run directory, with the disease_map_summary if present.",
            {"date": date_prop},
        ),
        _tool(
            "read_semgrep_findings",
            "Read the Semgrep governance rule inventory from .semgrep/. Note: scan findings "
            "are not persisted in-repo (CI-generated); this reports the ruleset, not results.",
            {},
        ),
        _tool(
            "run_readonly_diagnostics",
            "Aggregate a read-only health rollup across the latest snapshot, gate verdicts, "
            "phase-2 health, cartography status, and IC ledger. Reads existing artifacts only "
            "— does NOT execute any script, mutate, or call the network.",
            {"date": date_prop},
        ),
    ]


def _tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        },
    }


def _call_tool(params: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise ValueError("tools/call params must be an object")
    name = params.get("name")
    arguments = params.get("arguments") or {}
    if not isinstance(name, str):
        raise ValueError("tools/call requires string name")
    if not isinstance(arguments, dict):
        raise ValueError("tools/call arguments must be an object")

    tools = {
        "list_snapshots": _list_snapshots,
        "read_latest_snapshot_manifest": _read_latest_snapshot_manifest,
        "read_gate_verdicts": _read_gate_verdicts,
        "read_phase2_health": _read_phase2_health,
        "read_rankings_schema": _read_rankings_schema,
        "read_event_ev_feature_coverage": _read_event_ev_feature_coverage,
        "read_forward_eval_ic_ledger": _read_forward_eval_ic_ledger,
        "read_scientific_cartography_status": _read_scientific_cartography_status,
        "list_disease_map_artifacts": _list_disease_map_artifacts,
        "read_semgrep_findings": _read_semgrep_findings,
        "run_readonly_diagnostics": _run_readonly_diagnostics,
    }
    if name not in tools:
        raise ValueError(f"Unknown tool: {name}")
    result = tools[name](arguments)
    return {"content": [{"type": "text", "text": _json_text(result)}], "isError": False}


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #
SNAPSHOT_KEY_FILES = (
    "snapshot_manifest.json",
    "phase2_health.json",
    "rankings.csv",
    "ees_gate_diagnostics.json",
)


def _snapshots_dir() -> Path:
    return REPO_ROOT / "data" / "snapshots"


def _cartography_dir() -> Path:
    return REPO_ROOT / "artifacts" / "scientific_cartography"


def _list_snapshot_dates() -> list[str]:
    base = _snapshots_dir()
    if not base.is_dir():
        return []
    dates = [p.name for p in base.iterdir() if p.is_dir() and DATE_RE.fullmatch(p.name)]
    return sorted(dates, reverse=True)


def _list_cartography_dates() -> list[str]:
    base = _cartography_dir()
    if not base.is_dir():
        return []
    dates = [p.name for p in base.iterdir() if p.is_dir() and DATE_RE.fullmatch(p.name)]
    return sorted(dates, reverse=True)


def _resolve_date(value: Any, available: list[str], *, label: str) -> str:
    if value is None:
        if not available:
            raise ValueError(f"No {label} available")
        return available[0]
    if not isinstance(value, str) or not DATE_RE.fullmatch(value):
        raise ValueError("date must be YYYY-MM-DD")
    return value


def _safe_subdir(base: Path, date: str) -> Path:
    """Resolve base/date with a path-escape guard (date already regex-validated)."""
    path = (base / date).resolve()
    if base.resolve() not in path.parents:
        raise ValueError("path escapes the permitted directory")
    return path


def _list_snapshots(arguments: dict[str, Any]) -> dict[str, Any]:
    limit = _int_arg(arguments.get("limit"), default=30, lo=1, hi=500)
    dates = _list_snapshot_dates()[:limit]
    rows = []
    for date in dates:
        snap = _safe_subdir(_snapshots_dir(), date)
        rows.append(
            {
                "date": date,
                "path": _repo_relative(snap),
                "key_files": {f: (snap / f).is_file() for f in SNAPSHOT_KEY_FILES},
            }
        )
    return {
        "snapshots_dir": _repo_relative(_snapshots_dir()),
        "count": len(rows),
        "latest": dates[0] if dates else None,
        "snapshots": rows,
    }


def _read_latest_snapshot_manifest(arguments: dict[str, Any]) -> dict[str, Any]:
    date = _resolve_date(arguments.get("date"), _list_snapshot_dates(), label="snapshot")
    snap = _safe_subdir(_snapshots_dir(), date)
    return {"date": date, **_read_json_artifact(snap / "snapshot_manifest.json")}


def _read_gate_verdicts(arguments: dict[str, Any]) -> dict[str, Any]:
    limit = _int_arg(arguments.get("limit"), default=5, lo=1, hi=MAX_LEDGER_RECORDS)
    ledger = REPO_ROOT / "artifacts" / "gate_verdict_ledger.jsonl"
    result: dict[str, Any] = {"ledger": _read_jsonl_tail(ledger, limit)}
    if arguments.get("date") is not None:
        date = _resolve_date(arguments.get("date"), _list_snapshot_dates(), label="snapshot")
        snap = _safe_subdir(_snapshots_dir(), date)
        result["ees_gate_diagnostics"] = {"date": date, **_read_json_artifact(snap / "ees_gate_diagnostics.json")}
    return result


def _read_phase2_health(arguments: dict[str, Any]) -> dict[str, Any]:
    date = _resolve_date(arguments.get("date"), _list_snapshot_dates(), label="snapshot")
    snap = _safe_subdir(_snapshots_dir(), date)
    return {"date": date, **_read_json_artifact(snap / "phase2_health.json")}


def _read_rankings_schema(arguments: dict[str, Any]) -> dict[str, Any]:
    date = _resolve_date(arguments.get("date"), _list_snapshot_dates(), label="snapshot")
    sample_rows = _int_arg(arguments.get("sample_rows"), default=0, lo=0, hi=MAX_CSV_SAMPLE_ROWS)
    snap = _safe_subdir(_snapshots_dir(), date)
    path = snap / "rankings.csv"
    summary = _file_summary(path)
    if not path.is_file():
        return {"date": date, **summary, "exists": False}
    headers: list[str] = []
    row_count = 0
    sample: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            headers = next(reader)
        except StopIteration:
            headers = []
        for row in reader:
            row_count += 1
            if len(sample) < sample_rows:
                sample.append(dict(zip(headers, row)))
    out = {
        "date": date,
        **summary,
        "exists": True,
        "format": "csv",
        "column_count": len(headers),
        "columns": headers,
        "row_count": row_count,
    }
    if sample_rows:
        out["sample"] = sample
    return out


def _read_event_ev_feature_coverage(arguments: dict[str, Any]) -> dict[str, Any]:
    date = _resolve_date(arguments.get("date"), _list_cartography_dates(), label="cartography run")
    run = _safe_subdir(_cartography_dir(), date)
    return {"date": date, **_read_json_artifact(run / "landscape_feature_coverage_report.json")}


def _read_forward_eval_ic_ledger(arguments: dict[str, Any]) -> dict[str, Any]:
    limit = _int_arg(arguments.get("limit"), default=10, lo=1, hi=MAX_LEDGER_RECORDS)
    path = REPO_ROOT / "artifacts" / "readiness" / "forward_eval_ic_baseline.json"
    artifact = _read_json_artifact(path)
    content = artifact.get("content")
    if isinstance(content, dict):
        observations = content.get("observations")
        if isinstance(observations, list):
            trimmed = dict(content)
            trimmed["observations"] = observations[-limit:]
            trimmed["observations_total"] = len(observations)
            trimmed["observations_returned"] = min(limit, len(observations))
            artifact = {**artifact, "content": trimmed}
    return artifact


def _read_scientific_cartography_status(arguments: dict[str, Any]) -> dict[str, Any]:
    date = _resolve_date(arguments.get("date"), _list_cartography_dates(), label="cartography run")
    run = _safe_subdir(_cartography_dir(), date)
    return {"date": date, **_read_json_artifact(run / "scientific_cartography_status.json")}


def _list_disease_map_artifacts(arguments: dict[str, Any]) -> dict[str, Any]:
    date = _resolve_date(arguments.get("date"), _list_cartography_dates(), label="cartography run")
    run = _safe_subdir(_cartography_dir(), date)
    summary = _file_summary(run)
    if not run.is_dir():
        return {"date": date, **summary, "exists": False}
    files = []
    for path in sorted(run.iterdir())[:MAX_DIR_ENTRIES]:
        if path.is_file():
            files.append({"name": path.name, "size_bytes": path.stat().st_size})
    out = {
        "date": date,
        "path": _repo_relative(run),
        "exists": True,
        "file_count": len(files),
        "files": files,
    }
    disease_summary = run / "disease_map_summary.json"
    if disease_summary.is_file():
        out["disease_map_summary"] = _read_json(disease_summary)
    return out


def _read_semgrep_findings(_arguments: dict[str, Any]) -> dict[str, Any]:
    semgrep_dir = REPO_ROOT / ".semgrep"
    out: dict[str, Any] = {
        "semgrep_dir": _repo_relative(semgrep_dir),
        "exists": semgrep_dir.is_dir(),
        "findings_persisted": False,
        "note": (
            "Semgrep findings are not stored in-repo; they are produced by the "
            "semgrep-governance-audit CI workflow (currently inactive — Actions "
            "budget exhausted) and by the local pre-commit hook. This tool reports "
            "the rule inventory, not scan results."
        ),
        "rule_files": [],
        "rules": [],
    }
    if not semgrep_dir.is_dir():
        return out
    rule_ids: list[str] = []
    for rule_file in sorted(semgrep_dir.glob("*.y*ml")):
        out["rule_files"].append(_repo_relative(rule_file))
        for match in re.finditer(
            r"^\s*-?\s*id:\s*(.+?)\s*$", rule_file.read_text(encoding="utf-8", errors="replace"), flags=re.MULTILINE
        ):
            rule_ids.append(match.group(1).strip().strip("'\""))
    out["rules"] = rule_ids
    out["rule_count"] = len(rule_ids)
    return out


def _run_readonly_diagnostics(arguments: dict[str, Any]) -> dict[str, Any]:
    """In-process read-only rollup. Executes nothing; mutates nothing."""
    snap_dates = _list_snapshot_dates()
    cart_dates = _list_cartography_dates()
    requested = arguments.get("date")
    snap_date = _resolve_date(requested, snap_dates, label="snapshot") if snap_dates else None

    checks: list[dict[str, Any]] = []

    def check(name: str, exists: bool, status: Any = None, detail: Any = None) -> None:
        checks.append({"check": name, "exists": exists, "status": status, "detail": detail})

    if snap_date:
        snap = _safe_subdir(_snapshots_dir(), snap_date)
        manifest = snap / "snapshot_manifest.json"
        check("snapshot_manifest", manifest.is_file(), detail=snap_date)
        ph = snap / "phase2_health.json"
        ph_status = _read_json(ph).get("status") if ph.is_file() else None
        check("phase2_health", ph.is_file(), status=ph_status)
        check("rankings_csv", (snap / "rankings.csv").is_file())
    else:
        check("snapshot_manifest", False, detail="no snapshots present")

    ledger = REPO_ROOT / "artifacts" / "gate_verdict_ledger.jsonl"
    latest_gate = _read_jsonl_tail(ledger, 1)
    gate_records = latest_gate.get("records", [])
    gate_status = gate_records[-1].get("overall_status") if gate_records else None
    check("gate_verdicts", ledger.is_file(), status=gate_status)

    ic_path = REPO_ROOT / "artifacts" / "readiness" / "forward_eval_ic_baseline.json"
    ic = _read_json(ic_path) if ic_path.is_file() else {}
    check("forward_eval_ic", ic_path.is_file(), status=ic.get("path_c_status"))

    if cart_dates:
        cart = _safe_subdir(_cartography_dir(), cart_dates[0])
        cs = cart / "scientific_cartography_status.json"
        cs_status = _read_json(cs).get("status") if cs.is_file() else None
        check("scientific_cartography", cs.is_file(), status=cs_status, detail=cart_dates[0])
    else:
        check("scientific_cartography", False, detail="no cartography runs present")

    return {
        "mode": "read-only",
        "executes_scripts": False,
        "mutates": False,
        "network": False,
        "latest_snapshot": snap_date,
        "latest_cartography_run": cart_dates[0] if cart_dates else None,
        "checks": checks,
        "summary": {
            "checks_total": len(checks),
            "artifacts_present": sum(1 for c in checks if c["exists"]),
            "artifacts_missing": sum(1 for c in checks if not c["exists"]),
        },
    }


# --------------------------------------------------------------------------- #
# Read helpers (bounded, read-only)
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {"value": data}


def _read_json_artifact(path: Path) -> dict[str, Any]:
    summary = _file_summary(path)
    if not path.is_file():
        return {**summary, "exists": False}
    try:
        with path.open("r", encoding="utf-8") as handle:
            content = json.load(handle)
    except json.JSONDecodeError as exc:
        return {**summary, "exists": True, "format": "json", "parse_error": exc.msg}
    return {**summary, "exists": True, "format": "json", "content": content}


def _read_jsonl_tail(path: Path, limit: int) -> dict[str, Any]:
    summary = _file_summary(path)
    if not path.is_file():
        return {**summary, "exists": False, "records": []}
    records: list[Any] = []
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"_unparseable_line": line[:500]})
    return {
        **summary,
        "exists": True,
        "format": "jsonl",
        "records_total": total,
        "records_returned": min(limit, len(records)),
        "records": records[-limit:],
    }


def _file_summary(path: Path) -> dict[str, Any]:
    return {"path": _repo_relative(path), "exists": path.exists(), "is_file": path.is_file()}


def _int_arg(value: Any, *, default: int, lo: int, hi: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("expected an integer argument")
    if value < lo or value > hi:
        raise ValueError(f"value must be between {lo} and {hi}")
    return value


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _success_response(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _write(message: Any) -> None:
    text = _json_text(message)
    if _OUTPUT_FRAMING == "content-length":
        encoded = text.encode("utf-8")
        sys.stdout.buffer.write(f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii") + encoded)
        sys.stdout.buffer.flush()
        return
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    if "--health" in sys.argv:
        print(json.dumps({"ok": True, "server": SERVER_NAME, "mode": "stdio", "version": SERVER_VERSION}))
        raise SystemExit(0)
    if "--help" in sys.argv:
        print("Usage: python tools/biotech_mcp_server.py [--health]  (read-only MCP over stdio)")
        raise SystemExit(0)
    raise SystemExit(main())
