# CLAUDE.md — Wake Robin Biotech Screener

## Project Identity
Institutional-grade biotech investment screening system.
Outputs must be reproducible, auditable, and deterministic.
Every decision must be traceable to a data source with a timestamp.

## North Star Rule
Backtest systems NEVER directly modify production screening behavior.
They produce evidence and proposals only. Governance review required before
any backtest finding changes a production signal weight.

## CCFT Principles (Non-Negotiable)
All data fixtures must be:
- **Canonical**: single authoritative source per data type
- **Complete**: no silent nulls or missing fields without explicit flags
- **Frozen**: historical snapshots are immutable once written
- **Timestamped**: `data_available_timestamp <= as_of_date` always enforced

## Active Ruleset
- **ID**: `8887576e` (v1.14.0)
- **File**: `production_data/decision_rulesets/v1.14.0_coinvest_only_selector.json`
- **Key**: coinvest-only selector (coinvest_score_z 100%), pairwise_minimal ranker (ordinal-only), EW Top-30
- **Pinned in**: `run_screen.py` AND `run_phase2_snapshot_delta.py` (must stay in sync)

## Production Mental Model
> Coinvest selects (sole institutional signal as of v1.14.0), financial penalizes
> "safe but less catalytic" names, clinical is weak/conditional and under review.
> inst_delta_z zeroed in selector 2026-05-04 (ALERT: mean_ic=-0.097). Active in ranker (NW-t=+3.32).

## Architecture Freeze
v1.14.0 freeze in effect until post-h20d checkpoint (~2026-05-26).
No new enforcement logic or scoring changes. CI fixes and monitoring allowed.

## Universal Coding Rules

### Decimal Arithmetic Mandate
- All **scoring** arithmetic MUST use `Decimal` (never `float`). Init from strings: `Decimal("500000000")`.
- Statistical analysis (IC, Spearman, bootstrap) may use float/numpy/scipy.
- `exp()` in sigmoid: compute in float, convert to Decimal before re-entering scoring paths.
- Rounding: `ROUND_HALF_UP`. Scores to 2 dp, rates to 4 dp.

### Point-in-Time (PIT) Enforcement
- All dates ISO 8601 (`YYYY-MM-DD`). Never call `datetime.now()` — derive from `as_of_date`.
- Standard PIT: `source_date <= as_of_date - 1 day`.
- Strict PIT: `source_date < as_of_date - 2 days` (intraday data).
- Lookahead (`age_days < 0`): **reject unconditionally**.

### Deterministic Output
- Same inputs MUST produce byte-identical outputs.
- JSON serialization: sorted keys. List operations: deterministic sort keys.
- SHA256 content hash in every output. No external API calls during scoring.
- Random seed: 42. No overwriting existing run directories.

### Governance Metadata
Every pipeline output MUST include `_governance` block with run_id, score_version,
schema_version, parameters_hash, pit_cutoff, as_of_date.

## Before Writing Any Code
1. State which module this change belongs to
2. Identify: new signal, validation change, or infrastructure change
3. Write the failing test FIRST — show the red test before implementation
4. Confirm no look-ahead bias: what is the data_available_timestamp?
5. Classify by governance tier (Tier 0-4 per `governance/AGENT_ROUTING_POLICY.md`)

## Canonical Commands
```bash
pytest -p no:warnings                          # tests
ruff check src tests scripts tools             # lint
ruff format --check src tests scripts tools    # format
python tools/run_daily_production.py           # daily pipeline (13 steps, ~100 min)
```
Always warm 8-K cache BEFORE running screen.

## Anti-Patterns (Do Not Do)
1. `float` in scoring paths (use `Decimal`)
2. `datetime.now()` anywhere (use explicit `as_of_date`)
3. Refactor and add features in the same commit
4. Change production weights without ablation test (Sharpe delta >= 0.1)
5. Raw EDGAR XML as source of truth (use canonical summary)
6. Overwrite existing run directories
7. Push red main (WIP commits stay local)
8. Use PubMed h-index API, options flow, or CapIQ (not approved sources)
9. Introduce survivorship bias (graveyard list at `data/graveyard/`)

## Test Requirements
Every new signal must include:
1. Unit test with known fixture input -> expected output
2. Leakage test confirming data_available_timestamp compliance
3. Ablation test stub showing Sharpe contribution >= 0.1

## Coding Standards
- All outputs: `encoding='utf-8'`, `lineterminator='\n'`, `quoting=csv.QUOTE_MINIMAL`
- SHA256 hash every scored output for audit trail
- Use PIT fixtures in tests — never fetch live data

## Data Provenance
- Holdings truth source: `production_data/institutional_summary.json` (canonical)
- CUSIP-first reasoning, never issuer name strings
- Raw EDGAR XML is debug-only — never build narratives from raw parses
- If raw count != summary count: investigate summary pipeline first

## Detailed Context (loaded automatically by path)
- Production pipeline & decision engine: `.claude/rules/production-pipeline.md`
- Research, backtests, dead lanes, trust buckets: `.claude/rules/research-backtest.md`
- Governance tiers & promotion rules: `.claude/rules/governance.md`
- Signal status & spec tracker: `.claude/rules/signals.md`
- Volatile operational state (13F, IC, freeze): `.claude/rules/operational-state.md`
- Testing & CI: `.claude/rules/testing.md`
- External AI landscape & competitive intel: `.claude/rules/external-intel.md`
- Options expression & long-call contracts: `.claude/rules/options-expression.md`

## Key File Locations
| Area | File |
|------|------|
| Main orchestrator | `run_screen.py` |
| Decision Engine | `decision_engine.py` |
| Selector Engine | `selector_engine.py` |
| Ranker Engine | `ranker_engine.py` |
| Daily Production | `tools/run_daily_production.py` |
| Shadow Portfolio | `tools/live_shadow_portfolio.py` |
| Promotion Battery | `scripts/research/run_promotion_battery.py` |
| Ruleset Manifest | `production_data/decision_rulesets/manifest.json` |
| Cache Warmer | `warm_caches.py` |
| Event Ledger | `event_ledger.py` |
| Governance Policy | `governance/AGENT_ROUTING_POLICY.md` |
| Agent Registry | `agents/AGENT_REGISTRY.json` |
| Expression Layer | `event_ev/expression_layer.py` |
| Data Explorer | `tools/data_explorer/agent.py` |

## What to Update After Every Session
- [ ] Active ruleset version and key settings
- [ ] Trust bucket changes (provisional -> safe, or new invalid entries)
- [ ] Dead-lane additions (newly killed signals/lanes)
- [ ] Architecture freeze status (lift date, post-freeze priorities)
- [ ] Active spec status (resolved, blocked)
- [ ] Forward shadow & IC checkpoint
