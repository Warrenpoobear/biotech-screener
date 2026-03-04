# Biotech Screener

Point-in-time safe biotech investment screening system with clinical trials data, institutional holdings analysis, and multi-factor composite ranking.

## Architecture

```
                        ┌──────────────────────────────┐
                        │     run_screen.py             │
                        │  (Deterministic Orchestrator) │
                        └──────────┬───────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
     ┌────────▼──────┐  ┌─────────▼────────┐  ┌────────▼──────────┐
     │  Modules 1-5  │  │ decision_engine  │  │  Hydration Layer  │
     │  (Screening)  │  │  (Post-process)  │  │  (Price-derived)  │
     └───────┬───────┘  └────────┬─────────┘  └────────┬──────────┘
             │                   │                      │
     ┌───────▼───────────────────▼──────────────────────▼──────┐
     │                    Data Layer                            │
     │  PIT caches │ CTgov trials │ SEC 13F │ Price history    │
     └─────────────────────────────────────────────────────────┘
```

**Modules:**
- **Module 1** — Universe filtering (353 tickers from `universe.json`)
- **Module 2** — Financial health & survivability
- **Module 3** — Catalyst detection (CTgov, SEC 8-K, FDA, EU/EEA registries)
- **Module 4** — Clinical development scoring (calendar alpha v2)
- **Module 5** — Composite ranking with defensive overlay

**Decision Engine** — Pure post-processing layer (eligibility gates, tier labels, sizing bands, overlay signals). Driven by externalized rulesets with promotion governance.

## Directory Structure

```
├── run_screen.py              # Main orchestrator
├── decision_engine.py         # Decision engine v1 (~620 lines)
├── common/                    # Shared utilities
│   ├── clinical_calendar_alpha.py   # Calendar alpha v2 scoring
│   ├── data_quality.py              # Circuit breakers & validation
│   ├── input_validation.py          # Type/range/enum checks
│   ├── logging_config.py            # Rotating handlers, sanitization
│   ├── pit_enforcement.py           # Point-in-time discipline
│   └── staleness_gates.py           # Data freshness enforcement
├── tools/                     # Operational tooling
│   ├── run_daily_production.py      # Daily automated runner
│   ├── maintain_universe.py         # Universe audit/add/retire
│   ├── warm_13f_cache.py            # PIT-safe 13F cache builder
│   ├── warm_price_cache.py          # PIT price cache + forward eval
│   ├── ruleset_health_monitor.py    # Post-promotion drift detection
│   ├── weekly_health_packet.py      # Multi-signal health rollup
│   ├── live_performance_tracker.py  # Portfolio PnL attribution
│   ├── send_alert.py               # Slack/email notifications
│   └── data_integrity_audit.py      # Invariant checks & price recompute
├── scripts/                   # Research, evaluation & backfill
│   ├── build_*_pit.py               # PIT feature builders (3)
│   ├── eval_forward_returns.py      # Forward return IC evaluation
│   ├── compare_rulesets_replay.py   # A/B replay harness
│   └── research/                    # Signal research scripts
├── wake_robin_data_pipeline/  # Data collection & providers
├── collectors/                # Trial registry collectors (EU/EEA)
├── mcp_server/                # MCP server (12 tools)
├── production_data/           # Universe, rulesets, alpha tables
├── data/                      # PIT caches, snapshots, archives
├── tests/                     # 291 test files
└── .github/workflows/         # 8 CI/CD pipelines
```

## Quickstart

```bash
# Install
pip install -r requirements.txt
pip install -e .

# Run tests
python -m pytest tests/ -q \
  --ignore=tests/integration \
  --ignore=tests/test_minimum_suite.py \
  --ignore=tests/test_golden_baseline.py \
  --ignore=tests/test_run_screen.py \
  --ignore=tests/test_module_5_defensive.py

# Lint
flake8 common/ decision_engine.py tools/ scripts/
black --check common/ decision_engine.py tools/ scripts/

# Run a screen (requires data caches)
python run_screen.py --as-of-date 2026-03-04 --data-dir ./data --output results.json
```

## Daily Production Workflow

The production pipeline runs via GitHub Actions (`phase2-daily-production.yml`) on weekdays at 14:00 UTC:

1. **Warm caches** — CTgov trials, SEC 13F holdings, PIT prices
2. **Run screen** — Modules 1-5 + decision engine + hydration
3. **Health gates** — 24 gates (HARD FAIL / SOFT WARN / PASS)
4. **Snapshot** — Atomic write of rankings.csv + metadata if gates pass
5. **Drift report** — Compare against baseline, trigger alerts if degraded
6. **Alerts** — Slack/email on WARN or FAIL

See `RUNBOOK.md` for full operational reference.

## Decision Engine & Rulesets

The decision engine is parameterized by frozen `DecisionRuleset` JSON files in `production_data/decision_rulesets/`. Rulesets control:

- Sort anchor & signal weights (clinical, calendar alpha, institutional delta)
- Tier thresholds (A/B/C/D for dev and commercial)
- Eligibility gates (financial health, coverage)
- Sizing bands and rebalance buffer

**Promotion governance:** Candidate rulesets go through automated evaluation (`ruleset-promotion-gate.yml`), comparing IS/OOS IC, turnover, and overlap against the active baseline. See `promotion_governance.md`.

## CI/CD Pipelines

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `tests.yml` | push/PR | Lint + 291 unit tests with coverage |
| `phase2-daily-production.yml` | cron (weekdays) | Full production run |
| `ruleset-promotion-gate.yml` | dispatch | Automated A/B evaluation for ruleset candidates |
| `ruleset-release.yml` | dispatch | Versioned ruleset promotion |
| `replay-regression.yml` | dispatch | Historical regression testing |
| `rebuild-alpha-cohort-table.yml` | dispatch | Alpha cohort table rebuild |
| `publish-inputs-bundle.yml` | dispatch | Pre-built input bundle packaging |

## Key Design Principles

- **PIT-safe**: All features use strict `< as_of_date` discipline. No future data leakage.
- **Deterministic**: Fixed seeds, decimal arithmetic, stable sort order. Same inputs = same outputs.
- **Gate-protected**: 24 health gates enforce data quality before any snapshot is promoted.
- **Externalized config**: All tuning parameters live in ruleset JSON files, not in code.
- **Auditable**: Every production run writes provenance metadata (git SHA, gate verdicts, data sources).

## License

MIT
