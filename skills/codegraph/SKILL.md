# Codegraph Repo Intelligence Skill

## Purpose

Use Codegraph as the first-pass structural map for biotech-screener code navigation, dependency tracing, and blast-radius review. Codegraph is a repo map, not a proof engine: use it to find symbols and call edges quickly, then confirm literals, cron boundaries, dynamic-dispatch gaps, and production-path claims with targeted file reads or text search.

## Activation

Use this skill when a task asks to:

- find where a function, class, constant, schema, or signal is defined
- trace callers/callees or production paths
- assess blast radius before edits
- compare duplicate definitions or schema drift risk
- validate codegraph setup in Cursor Cloud

Do not use this skill as sole evidence for file-path literals, shell/cron/subprocess boundaries, or final production-path proof.

---

## Startup Check

From the repo root:

```bash
codegraph status
```

Healthy state:

- project path is the current workspace
- index statistics are present
- status says index is up to date

If missing or stale:

```bash
codegraph sync
# or, after branch switches / large changes:
codegraph index
```

Cursor Cloud agents install and maintain codegraph through `.cursor/environment.json`. Cursor MCP launches through `.cursor/mcp.json`:

```bash
codegraph serve --mcp --path ${workspaceFolder}
```

Hermes registration remains deferred; do not add codegraph to Hermes until the ambiguity/dynamic-dispatch fallback policy is implemented.

---

## Standard Workflow

> **Tool names differ by context.**
> From an IDE/agent session use the **MCP tools** (`codegraph_search`, `codegraph_callers`, `codegraph_callees`, `codegraph_impact`, `codegraph_context`).
> From a bash shell use the **CLI equivalents** (`codegraph query`, `codegraph callers`, `codegraph callees`, `codegraph impact`, `codegraph context`).
> The table below uses MCP names. CLI equivalents are shown inline.

1. Search broadly for the target symbol or concept:
   - MCP: `codegraph_search("save_validation_snapshot")`
   - CLI: `codegraph query "save_validation_snapshot"`
2. Disambiguate common names by file/path context before drawing conclusions.
3. Inspect dependency direction:
   - MCP: `codegraph_callers("save_validation_snapshot")` / `codegraph_callees("save_validation_snapshot")`
   - CLI: `codegraph callers "save_validation_snapshot"` / `codegraph callees "save_validation_snapshot"`
4. Before edits, check blast radius:
   - MCP: `codegraph_impact("SNAPSHOT_COLUMNS")`
   - CLI: `codegraph impact "SNAPSHOT_COLUMNS"`
5. For unknown subsystems, generate focused context:
   - MCP: `codegraph_context("trace snapshot columns drift risk")`
   - CLI: `codegraph context "trace snapshot columns drift risk"`
6. Confirm anything outside static Python/JS call edges with targeted read/search.

---

## Biotech-Specific Patterns

### Snapshot / rankings path

Known split:

- Live: `run_screen.py -> save_validation_snapshot() -> _write_snapshot()`
- Bundle/backfill: `scripts/run_screen_from_bundle.py -> main() -> run_batch() -> run_screen_for_date() -> _write_snapshot()`

Always clarify which path is in scope and confirm `"rankings.csv"` or other output path literals outside codegraph.

### Signal input tracing

Use callers/callees to map composer functions and input-surface functions for fields such as `clinical_score`, `final_score`, `financial_score`, and expectation-layer fields. Separate production callers from tests before proposing changes.

### Drift-risk checks

For duplicate constants or schema lists, use `codegraph query` and `codegraph impact`, then inspect the authoritative definition. Current known example:

- `SNAPSHOT_COLUMNS` in `run_screen_columns.py`
- duplicate list in `scripts/run_screen_from_bundle.py`

---

## Limits and Required Fallbacks

| Area | Codegraph use | Required fallback |
| --- | --- | --- |
| Symbol definitions | `query`, MCP `codegraph_search` | file read when editing |
| Call direction | `callers`, `callees` | inspect dynamic dispatch manually |
| Flow paths | MCP `codegraph_trace` | inspect break point; do not infer missing edge |
| File names / strings | candidate symbol location only | targeted text search/read |
| Cron / shell / subprocess | orienting only | inspect cron/script boundary |
| Common names | search candidates | file-qualified disambiguation |

## Output Checklist

When reporting codegraph findings, include:

- production symbols
- test-only symbols
- ambiguous symbols
- dynamic-dispatch gaps
- file-path literals needing confirmation
- recommended next action
