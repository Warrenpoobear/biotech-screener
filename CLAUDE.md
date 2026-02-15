# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**biotech-screener** is a deterministic, point-in-time (PIT) safe biotech investment screening system developed by Wake Robin Capital Management. It implements a multi-module pipeline that combines financial health analysis, clinical trial catalysts, and clinical development metrics to produce ranked investment opportunities, processed through a Decision Engine that assigns actionable tiers and position sizes.

**Key Principles:**
- **Determinism**: Same inputs always produce byte-identical outputs (no `random`, no `datetime.now()`)
- **Point-in-Time Safety**: Prevents lookahead bias by enforcing strict PIT cutoffs (`source_date <= as_of_date - 1`)
- **Fail-Closed**: Validates data and stops on errors rather than gracefully degrading
- **Audit Trail**: Complete machine-readable governance metadata for reproducibility
- **Decimal-Only Arithmetic**: All financial calculations use `Decimal` (never floats)
- **Stdlib-Only Core**: Zero external dependencies in scoring modules

## Quick Start

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests (~6870 tests)
pytest tests/ -v

# Run the full screening pipeline with Decision Engine + Phase-2 health gate
python run_screen.py --as-of-date 2026-02-14 --data-dir production_data \
  --output results.json --phase2

# Run without Phase-2 health gate
python run_screen.py --as-of-date 2026-02-14 --data-dir production_data \
  --output results.json

# Run with strict health gate (FAIL=exit 1, WARN=exit 2)
python run_screen.py --as-of-date 2026-02-14 --data-dir production_data \
  --output results.json --phase2 --strict
```

## Architecture Overview

```
Universe (Module 1)
  → Financial Health (Module 2)
  → Catalyst Events (Module 3) ← CT.gov + SEC 8-K + SEC Multi-Form + FDA Calendar
  → Clinical Development (Module 4)
  → Composite Scoring (Module 5) [Legacy — being superseded by Decision Engine]
  → Decision Engine (post-processing)
      L0: Eligibility gate (archetype + score floor)
      L2: Catalyst + drawdown overlays
      L4: Dev tier assignment (A/B/C/D)
      L3: Position sizing
  → Phase-2 Health Gate (snapshot delta + guardrails)
  → Output: rankings.csv + catalyst_source_mix.json + decision_portfolio.csv
```

## Decision Engine (v1.3.0+)

The Decision Engine (`decision_engine.py`, ~620 lines) is the primary post-processing layer that converts raw Module 5 composite scores into actionable investment tiers and position sizes. It supersedes Module 5's built-in position sizing.

### Key Concepts

- **Tiers**: A (highest conviction) → B → C → D (lowest). A-tier requires `score_rank_pct >= a_floor` AND catalyst strength NEAR or MID.
- **Catalyst strength bands**: NEAR (< catalyst_near days), MID (< catalyst_mid days), FAR, MISSING
- **Catalyst mode**: `specific_days` (dated event), `blended_window` (days_to_catalyst=0 + in_optimal_window), `no_upcoming`, `missing`
- **Catalyst priority**: FDA=1 (highest), CTGOV/SEC=2, FEDERAL_REGISTER=1, corporate=3, none=9, unknown=99
- **Layers**: L0 (eligibility) → L2 (overlays) → L4 (dev tier) → L3 (sizing)

### DecisionRuleset

Externalized, frozen dataclass with all tunable parameters. Stored as JSON in `production_data/decision_rulesets/`.

```python
from decision_engine import DecisionRuleset

# Load from JSON (file-content hash becomes ruleset_id)
ruleset = DecisionRuleset.from_json("production_data/decision_rulesets/v1.3.2_candidate.json")
print(ruleset.ruleset_id)  # e.g. "96f655ee"

# Key operational parameters
ruleset.a_floor          # 0.60 — minimum score_rank_pct for A-tier
ruleset.catalyst_near    # 120 days
ruleset.catalyst_mid     # 180 days
ruleset.tier_filter      # ["A", "B"]
ruleset.top_k            # 20 — max portfolio names
ruleset.catalyst_priority_mode  # "off"|"tiebreaker"|"blended"
```

**Pinned IDs:**
- `PHASE2_PINNED_RULESET_ID` in `run_screen.py` = `"96f655ee"` (must match delta module)
- `PHASE2_PINNED_RULESET_ID` in `run_phase2_snapshot_delta.py` = `"96f655ee"` (must match run_screen)
- Both pins MUST be updated together — `run_screen.py` imports the delta module's pin

### Ruleset Promotion Pipeline

```bash
# Bump version and create candidate
python scripts/bump_ruleset.py --from-json production_data/decision_rulesets/v1.json

# Promote candidate to active
python scripts/promote_ruleset.py production_data/decision_rulesets/v1.3.2_candidate.json
```

### Snapshot Outputs

Each screen run produces in `data/snapshots/{date}/`:
- `rankings.csv` — full universe with 20+ decision columns
- `catalyst_source_mix.json` — event source/confidence/precision distributions
- `decision_portfolio.csv` — actionable portfolio (tier_filter + top_k applied)

Key CSV columns: `eligible`, `tier_dev`, `actionable_rank`, `target_weight_pct`, `catalyst_mode`, `catalyst_days`, `catalyst_strength`, `catalyst_source`, `catalyst_event_type`, `catalyst_priority`, `de_drawdown_missing_reason`

## Phase-2 Health Gate

The Phase-2 pipeline (`run_phase2_snapshot_delta.py`, ~1220 lines) compares consecutive snapshots and enforces guardrails.

### Health Gate Cascade
- **FAIL** (exit 1): ruleset mismatch, zero eligible, optionality broken (coverage < 80%), coverage < 40%
- **WARN** (exit 2): A-count low, weight L1 > 55%, catalyst drop > 5pp, no A-tier regime, coverage < 60%
- **OK** (exit 0): all checks pass

### Pinned Thresholds
- `production_data/phase2_health_thresholds/v1.json` (ID: `c0e01f42`)
- `Phase2HealthThresholds` frozen dataclass with `thresholds_id` (sha256[:8])

### Delta Report
Shows turnover, +/- names, tier distribution, catalyst mode transitions, weight changes.

## Module 3: Catalyst Events (Multi-Source)

Module 3 now integrates four data sources:

| Source | Key | Description |
|--------|-----|-------------|
| ClinicalTrials.gov | `CTGOV_CALENDAR` | Trial milestones, completion dates |
| FDA Calendar | `FDA_CALENDAR` | PDUFA dates, AdCom meetings |
| SEC 8-K Filings | `SEC_8K_FILING` | Material event disclosures |
| SEC Multi-Form | `SEC_10Q/10K/6K_FILING` | Quarterly/annual/foreign filings |
| Federal Register | `FEDERAL_REGISTER` | FDA regulatory notices |

### Quality Gating (3-layer filter)
1. **Source triage**: `prefer_exhibits_only=True` for 10-Q/10-K (skip main body, fetch exhibit 99.x only)
2. **Relevance filter**: `require_biopharma_context=True` + `block_boilerplate=True` for multi-form
3. **Hard gate at merge**: `_MF_ALLOWED_CONF = {MED, HIGH}`, `_MF_ALLOWED_PREC = {"DAY", "WEEK", "MONTH", "QUARTER"}`

### Configuration
In `Module3Config`: `enable_sec_multi_form` and `enable_fda_regulatory` accept `"off"`, `"cache_only"`, or `"live"`. Production default is `"cache_only"`.

### Source Mix Sidecar
Written to `catalyst_source_mix.json` alongside `rankings.csv`. Contains `total_events`, `unique_tickers_with_events`, `by_source`, `by_confidence`, `by_date_precision`.

## Drift Monitoring

`scripts/run_drift_report.py` — daily guardrails and rollback trigger detection.

Tracks catalyst source mix, event type distribution, tier distribution, catalyst mode transitions, and flags anomalies against historical baselines.

```bash
python scripts/run_drift_report.py \
  --snapshot data/snapshots/2026-02-14 \
  --output /tmp/drift_report/
```

## Backtest Harness

### Rank IC Backtest (`run_rank_ic_backtest.py`)

```bash
# Basic rank IC
python run_rank_ic_backtest.py --signal score_rank_pct --subset dev

# Group-by analysis
python run_rank_ic_backtest.py --signal score_rank_pct --group-by tier_dev --subset dev

# Flip signal direction
python run_rank_ic_backtest.py --signal clinical_score --flip-signal
```

Available `--group-by`: `tier_dev`, `size_band`, `mom_state`, `eligible`, `tier_reason`, `catalyst_mode`, `severity`, `archetype`

Available `--subset`: `dev` (drug_developer), `commercial` (commercial_*)

### Walk-Forward Panel
```bash
python run_rank_ic_backtest.py --emit-panel output/panel.csv
```
Emits 18 PANEL_COLUMNS for ALL dev tickers across all archive dates.

### Ruleset Calibration
```bash
python scripts/calibrate_ruleset_from_panel.py --panel output/panel.csv
```
2D sweep over (a_floor x catalyst_near). Best: a_floor=0.60, catalyst_near=120 (separation=+0.97pp).

### Archives
- `data/archives/` — 33 `.tar.gz` files (2024-01-31 through 2026-02-07)
- Three data regimes: 2024 (catalyst_broken), 2025 (well-formed), 2026-01 (no_portfolio+optionality_broken)

## Core Modules (Pipeline)

| Module | File | Purpose |
|--------|------|---------|
| **Module 1** | `module_1_universe.py` | Universe filtering, status gates, shell company detection |
| **Module 2** | `module_2_financial.py` | Financial health scoring (burn rate, dilution, liquidity) |
| **Module 3** | `module_3_catalyst.py` | Multi-source catalyst event detection and scoring |
| **Module 4** | `module_4_clinical_dev_v2.py` | Clinical development scoring with PoS integration |
| **Module 5** | `module_5_composite_with_defensive.py` | Legacy composite ranking (being superseded by Decision Engine) |
| **Decision Engine** | `decision_engine.py` | Tier assignment, position sizing, catalyst overlays |
| **Phase-2 Delta** | `run_phase2_snapshot_delta.py` | Health gate + snapshot comparison |

## Data Pipeline (`wake_robin_data_pipeline/`)

### Collectors
| Collector | File | Purpose |
|-----------|------|---------|
| SEC 8-K | `collectors/sec_8k_catalyst_collector.py` | 8-K catalyst events + multi-form (10-Q/10-K/6-K) |
| FDA AdCom | `collectors/fda_adcom_collector.py` | PDUFA dates, AdCom meetings, Federal Register notices |
| Market Data | `market_data_provider.py` | Price history via yfinance |
| Morningstar | `morningstar_data_provider.py` | Fundamentals (requires SDK) |

### Key Data Files
| File | Size | Content |
|------|------|---------|
| `production_data/universe.json` | ~1MB | 353 tickers |
| `production_data/trial_records.json` | ~5.8MB | Clinical trials (17,420 interventions) |
| `production_data/financial_records.json` | — | Financial metrics |
| `production_data/market_data.json` | ~307KB | Market metrics |
| `data/price_history.csv` | — | Daily OHLCV prices |

### Cache Structure
```
wake_robin_data_pipeline/cache/sec/8k_catalysts/
  8k_catalysts_{date}_{PATTERN_VERSION}.json    # SEC 8-K events
  sec_filings_{date}_{PATTERN_VERSION}.json     # SEC multi-form events
wake_robin_data_pipeline/cache/fda/
  fda_regulatory_{date}.json                    # Federal Register events
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_drift_report.py` | Daily drift monitoring + rollback triggers |
| `scripts/calibrate_ruleset_from_panel.py` | 2D sweep for optimal ruleset params |
| `scripts/bump_ruleset.py` | Create new ruleset candidate |
| `scripts/promote_ruleset.py` | Promote candidate to active |
| `scripts/audit_catalyst_coverage.py` | Comprehensive offline catalyst coverage audit |
| `scripts/audit_drawdown_coverage.py` | Drawdown data coverage diagnostic |
| `scripts/backfill_missing_prices.py` | Manual yfinance price backfill (NOT for CI) |
| `scripts/build_multi_form_caches.py` | Build SEC multi-form caches for archive dates |
| `scripts/compare_ablation_snapshots.py` | Compare two snapshot folders (ablation analysis) |
| `scripts/run_phase2_health_calibration.py` | Replay archives to calibrate health thresholds |

## MCP Server

`mcp_server/` package with 12 registered tools via FastMCP.

- `mcp_server/app.py` — FastMCP singleton (all tools import from here)
- `mcp_server/server.py` — entry point, imports tool modules to register decorators
- `mcp_server/config.py` — path constants
- `mcp_server/tools/` — 4 modules: universe, price, screening, fundamentals
- **Critical**: no `print()` in server code, logging to stderr only
- `.mcp.json` at project root for both official Morningstar + custom server

## Coding Conventions

### Use Decimal for All Financial Calculations

```python
from decimal import Decimal

# CORRECT
cash = Decimal("500000000")
runway = (cash / abs(burn)).quantize(Decimal("0.01"))

# WRONG - Never use floats for money
cash = 500000000.0
```

### PIT Safety Pattern

```python
from common.pit_enforcement import compute_pit_cutoff, is_pit_admissible

def process_data(records, as_of_date: str):
    pit_cutoff = compute_pit_cutoff(as_of_date)  # as_of_date - 1
    for record in records:
        if not is_pit_admissible(record.get("source_date"), pit_cutoff):
            continue  # Skip future data
```

### Deterministic Hashing

```python
import hashlib, json
from datetime import date
from decimal import Decimal

def stable_json_dumps(obj):
    def default_serializer(o):
        if isinstance(o, date): return o.isoformat()
        if isinstance(o, Decimal): return str(o)
        raise TypeError(f"Cannot serialize {type(o)}")
    return json.dumps(obj, sort_keys=True, default=default_serializer)
```

### Fail-Loud Validation

```python
# CORRECT - Explicit tracking of failures
def validate_tickers(tickers: list[str]) -> ValidationResult:
    valid, invalid = [], {}
    for ticker in tickers:
        is_valid, reason = is_valid_ticker(ticker)
        if is_valid: valid.append(ticker)
        else: invalid[ticker] = reason
    return ValidationResult(valid=valid, invalid=invalid)
```

## Testing

**~6870 tests across 105+ test files.**

### Key Test Files

| File | Tests | Purpose |
|------|-------|---------|
| `tests/test_decision_engine.py` | ~100+ | Decision engine tiers, sizing, rulesets |
| `tests/test_phase2_health_gate.py` | 18 | Health gate FAIL/WARN/OK paths |
| `tests/test_phase2_delta_hook.py` | 9 | Snapshot delta comparisons |
| `tests/test_sec_multi_form.py` | 29+ | SEC multi-form collection, cache, quality gate |
| `tests/test_fda_regulatory_notices.py` | 25+ | Federal Register, product map, dedup |
| `tests/test_audit_catalyst_coverage.py` | 8 | Catalyst coverage audit |
| `tests/test_hydrate_drawdown.py` | 22 | Drawdown hydration, alias resolution |
| `tests/test_decision_engine_qa_report.py` | 41 | QA gate cascade |
| `tests/integration/test_run_screen.py` | — | End-to-end pipeline |

### Running Tests

```bash
# Full suite
pytest tests/ -v

# Specific module
pytest tests/test_decision_engine.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html

# Fast subset (decision engine only)
pytest tests/test_decision_engine.py tests/test_phase2_health_gate.py -x
```

## Common Gotchas

1. **Never use `datetime.now()`** — Pass explicit `as_of_date` parameter
2. **Never use `float` for money** — Use `Decimal` with string initialization
3. **Always check PIT admissibility** before using data
4. **component_scores is a LIST of dicts** — iterate to find `name=="clinical"`, not dict key access
5. **Pinned IDs must stay in sync** — `run_screen.py` and `run_phase2_snapshot_delta.py` both have `PHASE2_PINNED_RULESET_ID`; `run_screen.py` imports the delta module's pin as `DELTA_PINNED_ID`
6. **SEC collector needs universe as list of DICTS** — `collect_sec_filing_events(universe, ...)` expects `[{"ticker": "ACME", ...}]`, not `["ACME"]`
7. **`days_to_catalyst=0` + `in_optimal_window=True`** = blended proximity mode, NOT "no catalyst"
8. **Archive provenance trap** — full re-enrichment overwrites ALL fields; use `--catalyst-only` to preserve non-catalyst columns
9. **Catalyst data source mapping** — Sponsorship=`rec["smart_money_signal"]`+`rec["coinvest"]`, Catalyst=`rec["catalyst_decay"]` (TOP-LEVEL), Momentum=`score_breakdown.enhancements.momentum.alpha_60d`
10. **`from_json()` migration** — `enable_catalyst_priority=true` + no mode field → auto-migrates to `catalyst_priority_mode="tiebreaker"`
11. **Hash outputs for reproducibility** — Use `stable_json_dumps()` for deterministic serialization
12. **Don't silently drop invalid data** — Track and report validation failures

## Important Files Reference

| File | Purpose |
|------|---------|
| `run_screen.py` | Main pipeline orchestrator (~5000+ lines) |
| `decision_engine.py` | Tier assignment + position sizing (~620 lines) |
| `run_phase2_snapshot_delta.py` | Health gate + delta report (~1220 lines) |
| `module_3_catalyst.py` | Multi-source catalyst detection |
| `wake_robin_data_pipeline/collectors/sec_8k_catalyst_collector.py` | SEC 8-K + multi-form collection |
| `wake_robin_data_pipeline/collectors/fda_adcom_collector.py` | FDA + Federal Register collection |
| `common/score_to_er.py` | `attach_rank_and_z()` for score_rank_pct |
| `common/integration_contracts.py` | Module boundary types and schema validation |
| `enrich_archive_inputs.py` | Archive re-enrichment (catalyst-only mode) |
| `run_rank_ic_backtest.py` | Backtest harness with signals + group-by |
| `scripts/run_drift_report.py` | Drift monitoring + rollback triggers |
| `tests/conftest.py` | Shared test fixtures |

## Recent Changes

### v2.1.0 (February 2026 - Current)

- **Decision Engine v1.3.0+**: Tier assignment (A/B/C/D), position sizing, catalyst overlays, frozen DecisionRuleset dataclass
- **catalyst_priority_mode**: "off"|"tiebreaker"|"blended" — supersedes legacy `enable_catalyst_priority` bool
- **Phase-2 Health Gate**: FAIL/WARN/OK cascade, snapshot delta comparison, pinned thresholds
- **Catalyst Coverage Expansion**: SEC multi-form (10-Q/10-K/6-K) + Federal Register FDA notices
- **Quality Gating**: 3-layer filter (source triage → relevance filter → hard gate); reduced multi-form events from 3665 to 518 (86% reduction)
- **Source Mix Sidecar**: `catalyst_source_mix.json` per snapshot for ablation analysis
- **Drawdown Coverage**: 99.5% dev coverage, MIN_BARS_FOR_ESTIMATE=126, alias resolution
- **Walk-Forward Calibration**: 2D sweep (a_floor x catalyst_near), best separation +0.97pp
- **Drift Monitoring**: Daily guardrail checks with new source/event type keys
- **Ablation Tooling**: `compare_ablation_snapshots.py` with `--json-out`
- **MCP Server**: 12 tools via FastMCP for interactive queries
- **6870+ tests passing**

### v1.5.1 (January 2026)

- CUSIP resolution fixes (102/102 resolved)
- Smart money coverage 48% → 55.6%
- Composite scoring enhancements E1-E6
- Observability: confidence_factors in score_breakdown

### v1.5.0 (January 2026)

- Enhancements enabled by default
- Valuation coverage 85% → 100%
- Clinical trials pagination fix (100 → 1000 limit)
