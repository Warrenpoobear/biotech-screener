# HOT Memory (≤100 lines)

<!-- Critical patterns — always loaded. Promote here after 3x recurrence. -->

## Code Style
- Use plain strings for static markdown table headers, not f-strings (flake8 F541). Recurrence: 5x.
- Remove unused imports before committing — black reformats but flake8 F401 catches unused. Recurrence: 4x.

## Research Signals
- Raw count-based features (event counts, failure counts, trial counts) always correlate with company size. Must residualize against pipeline breadth or market cap before testing. Recurrence: 3x (PI trial count, graveyard burden, catalyst density).

## Portfolio Construction
- Shadow portfolio drag is from construction policy (flat 3% C-tier weights), not from ranking model defects. Tier-weighted policy (A=4/B=2.5/C=1/D=0) improved +1.60pp. CRITICAL finding.
- Headwind + deep_drawdown names bleed at 2.3x the rate of non-headwind names. Exit overlay adds +0.22pp.

## API Patterns
- Open Targets GraphQL search returns generic SearchResult — inline fragments (... on Drug, ... on Disease) are silently ignored. Always use two-step: search → get ID → fetch by ID.

## Ops
- run_screen.py --snapshot-dir appends date as subdirectory. Pass parent dir to avoid double nesting.
- Weekend/non-trading day: run_daily_production.py correctly blocks. Use run_screen.py directly for manual weekend runs.
- Cursor Cloud agents need Python deps from `pip install -r requirements.txt` before running run_screen.py/pytest. `pytest-xdist` is NOT required — `pyproject.toml` uses `-q -m 'not network'`. Missing dotenv indicates environment.json drift. (LRN-20260528-002)
- codegraph installed: v0.9.6, pinned in environment.json. Use MCP tools (codegraph_search, etc.) from IDE; CLI equivalents (codegraph query, etc.) from bash. Preflight mandatory before any Tier 2+ edit. (LRN-20260528-001/003)
- Hermes agents use `common/codegraph_guard.py` (CodegraphGuard) — all 5 acceptance gates enforced. Do NOT call codegraph CLI directly from agent code. (LRN-20260528-004)
- GitHub Actions "job was not started because an Actions budget is preventing further use" is provider budget/quota, not a code failure. Do not patch PR code for that signal.
- Track B fail-closed governance contracts live in draft PR #304 as expected-red spec tests only. Do not make them pass or touch ranker/final_score, snapshot writer, promotion, selector, sizing, or KG behavior without explicit governance clearance.
- Repo-native Hermes MCP can work in Cursor Cloud while production Hermes/Hermes Link runtime is absent. Treat cloud knowledge-layer warnings as stale until refreshed on the local/production runtime.
