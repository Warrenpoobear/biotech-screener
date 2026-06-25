---
name: production-pipeline
description: Pipeline architecture, runtime behavior, cron orchestration
metadata:
  type: workflow
  status: active
  paths:
    - src/wake_robin_screener/decision_engine.py
    - src/wake_robin_screener/selector_engine.py
    - tools/run_daily_production.py
    - tools/warm_caches.py
---

# Production Pipeline Rules

---

## Decision Engine Architecture (v1.14.0)

**Core files:**
- `decision_engine.py` — L0 gates -> L2 overlays -> L4 tiers -> L3 sizing -> sort key
- `selector_engine.py` — B6 selector (5 blocks, coinvest+inst dominant)
- `ranker_v2_pairwise.py` — pairwise_minimal ranker (6 features, ordinal-only)
- `ranker_engine.py` — clinical_50 ranker (legacy/fallback, bounded +/-15%)

**Pipeline flow:**
```
Modules 1-5 -> Decision Engine (gates, tiers, sizing)
           -> Selector Engine (B6: coinvest_score_z 100%, inst_delta_z zeroed 2026-05-04)
           -> Ranker Engine (pairwise_minimal: 6 features, top-60 cohort, ordinal-only)
           -> Sort by final_score -> EW Top-30 -> rankings.csv
```

**Sort anchor:** `selector_score` (uses `final_score` = ranker_v2_score for cohort members)
All downstream consumers use `actionable_rank` (now driven by selector/ranker, not composite_rank).

**Statistical QA:** `common/stats/` (6 modules), `scripts/research/checklist_v2_rerun.py`

---

## Event Ledger & Cache Warming

- **Event ledger**: `build_event_ledger()` in `event_ledger.py` — 7+ sources (CTGov, merged trials, SEC 8-K, SEC multi-form, FDA ADCOM, FDA regulatory, PDUFA manual, EMA)
- **Cache warmer**: `warm_caches.py --sources sec_8k,ctgov,sec_13f,fda_adcom,fda_regulatory,euctr,ctis,isrctn,merged_trials`
- **EU/EEA registries**: `euctr_collector.py`, `ctis_collector.py`, `isrctn_collector.py` in `wake_robin_data_pipeline/collectors/`
- **Trial merger**: `trial_registry_merger.py` — cross-registry dedup by NCT/EudraCT IDs
- **Always warm 8-K cache BEFORE running screen**

---

## Daily Production Pipeline

- **Runner**: `tools/run_daily_production.py` — 13-step orchestrator
- **Cron**: 5:30 PM ET weekdays + `@reboot` catch-up for missed runs
- **Steps**: price refresh -> cache warm (incl. FDA) -> screen (with `--inputs-manifest write`) -> audit -> gates -> manifest + promotion -> drift report -> action packet -> shadow portfolio -> trade plan -> portfolio report -> readiness scorecard -> ops digest -> PIT backfill (optional)
- **Ops digest**: `tools/build_ops_digest.py` -> `artifacts/ops_digest/YYYY-MM-DD_digest.md` — single-screen actionable summary
- **Readiness**: `tools/weekly_readiness_scorecard.py` -> READY / REVIEW / HOLD verdict
- **Health checks**: collection health (INFO/WARN/FAIL with weekend-safe price fallback), phase-2 health, exposure metrics

---

## OpenClaw Ops Agent

- **Workspace:** `agents/ops/` — SOUL.md, TOOLS.md, HEARTBEAT.md
- **Role:** read-mostly operator — runs pipeline, reads digest, refuses ruleset mutation
- **Model:** DeepSeek v4 flash via OpenRouter on operator WSL gateway (2026-05-20+)
- **Fleet:** 29 active agents + Hermes governance jobs; registry `agents/AGENT_REGISTRY.json`
- **Evening catchup (22:00 ET):** `fleet_completion_audit` → `fleet_ops_status` → `fleet_crontab_verify`
- **Operator host:** `bash tools/run_operator_host_setup.sh` after `git pull`
- **Architecture freeze:** scoped production model freeze (2026-06-20) — no selector/ranker/sizing changes without lift

---

## Shadow Portfolio

- **File**: `tools/live_shadow_portfolio.py` (902 lines)
- **Policy**: `production_data/portfolio_policy.json` (v3), $500k, 55/25/10/10 bucket split
- **Family sleeves**: REGULATORY/CLINICAL split per bucket with time-ladder sub-buckets
- **Regulatory sleeve A/B**: +1.85pp 63d, +1.59pp 84d (positive but coverage-limited)

---

## Data Provenance Rules

- **Holdings truth source:** `production_data/institutional_summary.json` is canonical. It has CUSIP->ticker resolution, issuer normalization, and corporate action handling.
- **Raw EDGAR XML is debug-only.** Never build a narrative (e.g., "8 new entrants") from raw filing parses unless it matches the canonical summary. Raw issuer strings are unreliable — different filings use different names for the same entity.
- **CUSIP-first, not issuer-first.** Always reason from CUSIP -> canonical ticker, never from issuer name strings.
- **If raw count != summary count:** investigate the summary pipeline first. The summary is more likely correct.
