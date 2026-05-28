# Codegraph Operator Runbook

**Status:** Claude/Cursor approved · Cursor Cloud repo config active · Hermes registration deferred

**Last updated:** 2026-05-28

**Index location:** `.codegraph/` (repo-contained; `.codegraph/.gitignore` excludes `*.db`, `*.db-wal`, `*.db-shm` — ships with codegraph, no manual gitignore entry needed)

**Tool version:** codegraph v0.9.6 · Node 22 · npm global install

**Skill:** `skills/codegraph/SKILL.md`

---

## Operating Rule

```
Codegraph first, grep/read second, edit third.
```

1. `codegraph_search` / `query` — find candidate symbols
2. `codegraph_node` — inspect exact symbol signature
3. `codegraph_callers` / `codegraph_callees` — map dependency direction
4. `codegraph_trace` — only between Python symbols (not file-path literals)
5. `grep` / `read` — for file paths, string literals, cron/shell boundaries, dynamic-dispatch gaps
6. Propose patch only after this map is complete

---

## Standard Preflight (use at start of any code-review or implementation task)

```text
Before reading files manually or editing anything, run a codegraph preflight.

Required:
1. codegraph_search for the target symbol/concept
2. codegraph_node for the exact candidate symbol
3. codegraph_callers and codegraph_callees
4. codegraph_impact if any edit is being considered
5. grep/read only to confirm dynamic hops, file paths, or string literals

Return:
- production symbols
- test-only symbols
- ambiguous symbols
- dynamic-dispatch gaps
- file-path literals requiring grep/read
- recommended next action

Do not edit files until this map is complete.
```

---

## Tool-Selection Table

| Goal | Start with | Then use | Avoid |
|---|---|---|---|
| Find symbol / function | `codegraph_search` | `codegraph_node` | Broad `context` first |
| Find upstream usage | `codegraph_callers` | targeted read | Guessing from names |
| Find downstream calls | `codegraph_callees` | `codegraph_node` | Treating params as callees |
| Find path between functions | `codegraph_trace` | `codegraph_node` at break point | Tracing to file-path literals |
| Find file output path | `codegraph_search` + grep/read | targeted read | `trace` to string literal |
| Explore unknown subsystem | `codegraph_context` | `search` / `node` | Treating context output as proof |
| Before any edit | `codegraph_impact` | tests + grep/read | Editing from search results alone |

---

## Biotech-Specific Workflow Templates

### A. Production path tracing

*Use for: `rankings.csv`, snapshots, phase files, 13F outputs, KG outputs.*

```text
Use codegraph to trace `<artifact>` production.

Rules:
- codegraph for symbols and function paths
- grep/read for file-path literals
- do not infer across dynamic-dispatch gaps
- separate live production from backfill/bundle paths

Return:
1. entry point
2. writer function
3. output path (grep/read to confirm string literal)
4. upstream scoring/decision functions
5. ambiguity / gaps
6. exact final confirmation needed
```

**Known split in this repo:** `rankings.csv` has two production paths:
- Live: `run_screen.py → save_validation_snapshot() → _write_snapshot()`
- Bundle/backfill: `run_screen_from_bundle.py → main() → run_batch() → run_screen_for_date() → _write_snapshot()`

Always clarify which path is in scope.

---

### B. Signal input tracing

*Use for: `clinical_score`, `final_score`, `financial_score`, `catalyst_score`, expectation-layer fields.*

```text
Use codegraph to trace the input surface for `<score_or_signal>`.

Return:
1. production composer function
2. upstream callers
3. direct input parameters
4. functions constructing each input
5. output field written
6. tests touching this path
7. dynamic / ambiguous areas

Do not edit.
```

**Benchmark result:** Module 4 `clinical_score` input map — 2 tool calls (`callers` + `callees`), zero file reads, complete z_* input surface from `save_validation_snapshot` → 6 component functions in `common/clinical_calendar_alpha.py`.

---

### C. Governance boundary check

*Use before any change that might touch ranker / selector / sizing / final_score / KG.*

```text
Use codegraph to classify whether this proposed change touches:
- selector
- ranker
- sizing
- final_score
- decision_engine enforcement
- production KG
- snapshot writer / schema

If yes: mark it gated and stop.
If no: explain why it is plumbing / test / docs only.
```

---

### D. Drift-risk detection

*Use for: duplicate constants, schema lists, copied pipelines, bundle-vs-live outputs.*

```text
Use codegraph to find duplicate definitions and drift risks for `<contract_or_column_list>`.

Return:
1. authoritative definition
2. duplicate definitions
3. production consumers
4. test consumers
5. mismatch risk
6. minimal guardrail test proposal
```

**Known risk in this repo:** `SNAPSHOT_COLUMNS` has an authoritative definition in `run_screen_columns.py` and a separate copy in `run_screen_from_bundle.py`. Schema divergence between these two could make live and bundle `rankings.csv` outputs differ silently.

**Status: covered.** Contract 6 in `tests/test_contract_output_schemas.py` enforces three invariants (added 2026-05-24, verified passing):
- 6a: Bundle is a strict subset of live (naming drift guard)
- 6b: Both paths share the required scoring/decision parity surface
- 6c: Live path retains required scoring and expectation-layer columns

---

### E. Blast-radius review

*Use before accepting any patch.*

```text
Use codegraph_impact, callers, and callees on each changed function.

Return:
1. direct callers
2. direct callees
3. production consumers
4. test consumers
5. affected artifacts
6. whether this is: safe / gated / blocked
```

---

## Known Limitations Policy

Codegraph proof is partial unless all of the following hold:

1. **Symbol is file-disambiguated** — common names (`main`, `run`, `load`, `test`, `_write_snapshot`) match 2–10+ symbols across the repo. Always include filename context in the query (e.g. `codegraph_explore "main run_batch run_screen_from_bundle.py"`).
2. **Dynamic-dispatch breaks are manually inspected** — `codegraph_trace` returns a diagnostic when the call path is mediated by a variable, conditional, or callback. That break point must be inspected with `codegraph_node` + grep/read before the path is considered confirmed.
3. **File-path literals are confirmed with grep/read** — `codegraph_trace` cannot target string literals like `"rankings.csv"`. Use grep after the symbol path is known.
4. **Cron/shell/subprocess boundaries are separately verified** — static graph has no edges across `subprocess.run()`, crontab entries, or shell invocations. These boundaries require manual inspection of `crontab -l` and the script's subprocess calls.
5. **Production and test symbols are separated** — `codegraph_callers` returns all callers including test helpers. Always filter for production callers (typically in `scripts/`, `tools/`, `run_*.py`, `common/`, `decision_engine.py`).

---

## Index Maintenance (WSL / Manual Sync)

Git hooks were **intentionally declined** (WSL `/mnt/` path; `offerWatchFallback` prompt answered "I'll run sync myself"). Do not enable hooks.

Run `codegraph index` (full reindex) or `codegraph sync` (incremental) after:

- Branch switches
- Pulls / merges
- Generated code or file moves
- Large refactors (10+ files changed)
- Before serious dependency tracing

```bash
codegraph status          # confirm index state
codegraph sync            # incremental (fast, use after small changes)
codegraph index           # full reindex (use after merges / branch switches)
codegraph status          # verify node/edge counts unchanged or updated
```

Current index: **1,677 files · 50,419 nodes · 113,867 edges · 108.5 MB**

### Cursor Cloud Persistence

Repo-level Cursor Cloud setup lives in:

- `.cursor/environment.json` — installs `@colbymchenry/codegraph@0.9.6`, installs Python deps from `requirements.txt`, then syncs or initializes the local index from the project root.
- `.cursor/mcp.json` — registers the Cursor MCP server as `codegraph serve --mcp --path ${workspaceFolder}`.

The install command must remain idempotent. Keep `.codegraph/` database files gitignored.

---

## Hermes Registration

Hermes agents run autonomously on cron. All five acceptance gates are now implemented in `common/codegraph_guard.py`. Hermes agents must import and use `CodegraphGuard` rather than calling the codegraph CLI directly.

### Acceptance gates (all implemented)

| Gate | Implementation |
|---|---|
| Dynamic-dispatch break | `_gate_dynamic_dispatch()` — warns + instructs fallback; never halts silently |
| Ambiguous symbol | `_gate_ambiguous()` — warns + requires `file_hint` before proceeding |
| File-path literal | `_gate_file_path()` — returns UNVERIFIED + rg fallback instruction |
| Cron/shell boundary | `_gate_cron_shell()` — warns to inspect subprocess/crontab manually |
| Partial graph proof | Emits `[PARTIAL PROOF]` on empty results or dynamic-dispatch gaps |

### Hermes agent usage

```python
from common.codegraph_guard import CodegraphGuard

cg = CodegraphGuard()

# Before any write/mutation:
result = cg.callers("my_function")
if not result.is_trustworthy:
    # heed result.warnings before proceeding
    logger.warning(result.format())

# Tier 3 boundary check:
hit, surfaces = cg.tier3_gate("my_function")
if hit:
    raise RuntimeError(f"Tier 3 surface reached: {surfaces}. Operator approval required.")
```

### Registration state

| Target | Registered | Notes |
|---|---|---|
| Cursor IDE | ✅ `.cursor/mcp.json` | MCP tools available |
| Cloud Agent env | ✅ `.cursor/environment.json` | v0.9.6 pinned |
| Hermes agents | ✅ `common/codegraph_guard.py` | Use guard, not raw CLI |

---

## Registration State

| Target | Registered | Notes |
|---|---|---|
| Claude Code (global) | ✅ | `codegraph install --target claude --location global` |
| Cursor IDE | ✅ repo config | `.cursor/mcp.json` + `.cursor/environment.json` |
| Hermes agents | ✅ | `common/codegraph_guard.py` — use guard, not raw CLI |

---

## Rollback

```bash
# Remove index from repo
codegraph uninit .

# Remove Claude Code MCP registration
codegraph uninstall --target claude --location global

# Remove binary
rm -rf ~/.codegraph/ && npm uninstall -g @colbymchenry/codegraph
```
