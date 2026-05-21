---
paths:
  - tools/run_daily_production.py
  - tools/cron_daily_production.sh
  - tools/build_ops_digest.py
  - tools/weekly_readiness_scorecard.py
  - tools/build_data_collection_health.py
  - src/wake_robin_screener/decision_engine.py
  - src/wake_robin_screener/selector_engine.py
  - src/wake_robin_screener/ranker_engine.py
  - src/wake_robin_screener/ranker_v2_pairwise.py
  - warm_caches.py
  - event_ledger.py
  - run_screen.py
---

# Production Pipeline & Decision Engine

## Daily Pipeline (13 Steps)
**Runner**: `tools/run_daily_production.py`
**Cron**: 5:30 PM ET weekdays + `@reboot` catch-up
**Timeout**: 6000s (100 min) — covers worst-case AACT + tail steps. Previous 4500s killed mid-AACT on Mondays.

### Steps (in order)
1. Price refresh
2. Cache warm (including FDA) — **always warm 8-K cache BEFORE screen**
3. Screen (with `--inputs-manifest write`)
4. Audit
5. Gates
6. Manifest + promotion
7. Drift report
8. Action packet
9. Shadow portfolio
10. Trade plan
11. Portfolio report
12. Readiness scorecard
13. Ops digest + PIT backfill (optional)

### Pipeline Artifacts
- Ops digest: `artifacts/ops_digest/YYYY-MM-DD_digest.md`
- Readiness: `tools/weekly_readiness_scorecard.py` -> READY / REVIEW / HOLD
- Health checks: collection health (INFO/WARN/FAIL with weekend-safe price fallback)

## Decision Engine Architecture (v1.14.0)

### Pipeline Flow
```
Modules 1-5 -> Decision Engine (gates, tiers, sizing)
           -> Selector Engine (B6: coinvest_score_z 100%, inst_delta_z zeroed)
           -> Ranker Engine (pairwise_minimal: 6 features, top-60 cohort, ordinal-only)
           -> Sort by final_score -> EW Top-30 -> rankings.csv
```

### Core Files
| Component | File |
|-----------|------|
| Decision Engine | `decision_engine.py` — L0 gates -> L2 overlays -> L4 tiers -> L3 sizing -> sort key |
| Selector Engine | `selector_engine.py` — B6 selector (5 blocks, coinvest dominant) |
| Pairwise Ranker | `ranker_v2_pairwise.py` — pairwise_minimal (6 features, ordinal-only) |
| Legacy Ranker | `ranker_engine.py` — clinical_50 (fallback, bounded +/-15%) |

### Sort Anchor
`selector_score` (uses `final_score` = ranker_v2_score for cohort members).
All downstream consumers use `actionable_rank` (driven by selector/ranker, not composite_rank).

## Event Ledger & Cache Warming
- Event ledger: `build_event_ledger()` — 7+ sources (CTGov, merged trials, SEC 8-K, SEC multi-form, FDA ADCOM, FDA regulatory, PDUFA manual, EMA)
- Cache warmer: `warm_caches.py --sources sec_8k,ctgov,sec_13f,fda_adcom,fda_regulatory,euctr,ctis,isrctn,merged_trials`
- EU/EEA registries: `euctr_collector.py`, `ctis_collector.py`, `isrctn_collector.py`
- Trial merger: `trial_registry_merger.py` — cross-registry dedup by NCT/EudraCT IDs

## Shadow Portfolio
- **File**: `tools/live_shadow_portfolio.py` (902 lines)
- **Policy**: `production_data/portfolio_policy.json` (v3), $500k, 55/25/10/10 bucket split
- Family sleeves: REGULATORY/CLINICAL split per bucket with time-ladder sub-buckets

## Adding a 13F Manager
Use `tools/onboard_manager.py` — never edit `production_data/manager_registry.json` directly.
```bash
python tools/onboard_manager.py --cik 1802528 --name "Fairmount Funds Management" \
  --aum-b 1.3 --style concentrated_clinical_stage --tier elite_core --notes "..."
```
Partial reruns: `--skip-registry`, `--skip-backfill`, `--skip-current`, `--skip-test`.
