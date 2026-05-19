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
- "Latest dated subdir" lookups MUST filter by `^\d{4}-\d{2}-\d{2}$` before sorting. Non-date siblings (e.g., `data/snapshots/resolutions/`) lex-sort AFTER `YYYY-MM-DD` and poison `sorted()[-1]`. Bit telegram `/status` and `/snap`.
- Deprecating a hook script via `.sh.deprecated` rename: ALWAYS grep `.claude/settings.json`, `.github/workflows/`, and crontabs for the original name. Silent broken hooks survive indefinitely.

## Producer/Schema Design
- When a runtime function evaluates a seed value, overwrite the SAME field name documented in the spec. Don't bifurcate (`status` + `eval`) — internal readers degrade to `.get("new", item["old"])` everywhere and the on-disk artifact shows contradictory values. Spec 089 first_fire_ledger violated this; fixed in 37d9997.
