---
name: institutional-signal
triggers:
  - 13F signal
  - coinvest score
  - institutional holdings
  - manager registry
  - 13F refresh
  - cohort quarantine
  - inst_delta_z
  - SEC EDGAR 13F
  - manager onboarding
description: >
  Reference for the 13F institutional signal pipeline — from SEC EDGAR filing
  ingestion through coinvest_score_z (100% selector weight in v1.14.0) and
  cohort quarantine governance. Covers manager registry, refresh cycle, and
  insider diagnostic isolation guard.
---

# Institutional Signal Skill

## Purpose

Reference for the 13F institutional signal pipeline - from SEC EDGAR filing ingestion through coinvest_score_z production signal and cohort quarantine governance. This is the dominant selector signal (100% weight in v1.14.0).

This skill is organized into two sections:

1. **Framework Reference** - Stable architecture, rules, and processes (changes only with code updates)
2. **Operational State** - Volatile status snapshots that require periodic refresh

---

# SECTION 1: FRAMEWORK REFERENCE

---

## Architecture Overview

```
SEC EDGAR 13F-HR filings
  -> warm_13f_cache.py (per-CIK PIT cache dirs)
  -> build_institutional_summary.py (canonical summary)
  -> coinvest_score_z (selector signal, 100% weight v1.14.0)
  -> inst_delta_z (diagnostic/export only in v1.14.0)
```

## Manager Registry

**File**: `production_data/manager_registry.json`
**Never edit directly** - use `tools/onboard_manager.py`

### Tiers

| Tier | Description | Signal Weight |
|------|-------------|--------------|
| elite_core | Highest-conviction biotech-focused managers | Full weight |
| conditional | Broader institutional managers with biotech exposure | Reduced weight |

### Onboarding Flow

```bash
python tools/onboard_manager.py \
  --cik 1802528 \
  --name "Fairmount Funds Management" \
  --aum-b 1.3 \
  --style concentrated_clinical_stage \
  --tier elite_core \
  --notes "..."
```

One-shot flow: registry append -> backfill across every existing PIT dir (lookback=40, approx 10y) -> warm current as-of date -> run `tools/test_manager_integration.py` (6/6 gate).

Partial reruns: `--skip-registry`, `--skip-backfill`, `--skip-current`, `--skip-test`.

---

## coinvest_score_z

The production selector signal. Measures institutional co-investment conviction across elite biotech managers.

### Key Properties

- Drives 100% of selector weight (v1.14.0, coinvest-only after inst_delta_z demotion)
- Correlation with final_score: rho = +0.882 (double-count concern, documented in T1 ranker anatomy)
- Checklist v2: 3/5 standalone; prior bundle with inst_delta is historical evidence only
- Collapse guard: SD floor = 0.10 (below this, snapshot integrity check FAILs)

### Data Flow

1. PIT cache: `data/caches/sec_13f/PIT/{YYYY-MM-DD}/` per manager
2. Canonical summary: `production_data/institutional_summary.json`
3. Delta computation: `institutional_summary_delta.json` (pre vs post refresh)
4. Score: z-scored across eligible universe per snapshot

---

## inst_delta_z

Quarter-over-quarter change in institutional holdings. Measures whether smart money is accumulating or distributing.

### Governance Rules

- Reinstatement requires IC recovery evidence documented in governance log
- Current production status: selector weight 0%, absent from the production 2-feature ranker
- Historical audit showed positive within-cohort discrimination (NW-t = +3.32), but this is not active production behavior
- When zeroed, selector runs on coinvest_score_z alone and ranker uses only `coinvest_score_z` + `financial_score`

---

## insider_net_buy_value_90d (Spec 104, Diagnostic Only)

Form 4-derived insider buying signal. Tracks net insider purchase value over a trailing 90-day window.

### Status: DIAGNOSTIC ONLY

- Listed in `DIAGNOSTIC_FIELDS`, NOT in `ALPHA_FEATURE_REGISTRY`
- Tracked and exported for observability
- Does NOT enter the scoring model, ranker, or selector
- Does NOT affect ranks, actions, or position sizing

### Blank vs. Zero Semantics (CRITICAL)

| Value | Meaning |
|-------|---------|
| NaN / None / blank | Not fetched, no Form 4 coverage for this ticker |
| 0.0 | Fetched successfully, no insider buy activity in 90-day window |

Never collapse blank and zero. Never impute zero for missing or blank for zero.

### Expectation Model Isolation Guard (Spec 104, R4a)

The expectation model has an `insider_net_buy_z` weight that activates silently if `insider_net_buy_value_90d` flows into `market_features`. Spec 104 requires an explicit guard: either runtime assertion that the field is NOT in `market_features`, or weight zeroing, or a pre-inference drop guard.

### Promotion Criteria (future, not current build)

Requires ALL of: 20+ stable snapshots with >= 60% non-null coverage, blank/zero integrity verified, IC > 0 at p < 0.05, Checklist v2 battery pass, explicit written approval.

---

## 13F Refresh Cycle

SEC 13F filings have a 45-day lag from quarter-end. Filings typically cluster in the final 3 business days before the deadline.

### Pre-Refresh Readiness (`tools/prep_13f_refresh.py`)

5 guards, all must PASS:

| Guard | Check |
|-------|-------|
| 1 | Most recent snapshot has valid institutional_summary_delta.json |
| 2 | coinvest_score_z has healthy variance (SD > 0.10) |
| 3 | PIT cache has entries within 3 days of today |
| 4 | SEC EDGAR endpoint is reachable |
| 5 | Dry-run: build_institutional_summary() produces valid output (>=80% coverage) |

Writes baseline artifact: `artifacts/13f_pre_refresh_baseline_{date}.json`

### Cohort Quarantine (`tools/check_13f_cohort_quarantine.py`)

Run after new filings land. Compares pre-refresh vs post-refresh snapshots.

**Sections:**

- A: Manager-level diff (filing counts, coverage)
- B: Coverage diff (tickers_with_signal, signal_coverage_pct)
- C: Per-ticker score diff (coinvest_score_z, inst_delta_z distributions)
- D: Top-30 churn (Jaccard similarity, entries/exits, rank movement)

**Verdicts:**

| Verdict | Meaning | Action |
|---------|---------|--------|
| CLEAN | Normal refresh, minimal churn | Proceed |
| QUARANTINE | Significant score/rank disruption | Hold for review |
| PRODUCER_AUDIT_REQUIRED | Anomalous coverage or manager changes | Deep investigation |

Telegram alerting on QUARANTINE/PRODUCER_AUDIT_REQUIRED (suppressible with `--no-alert`).

### Contamination Window

After adding new managers, a contamination window opens (typically 20 trading days). IC measurements during this window are flagged as contaminated and excluded from clean IC calculations.

---

## Data Provenance Rules

- **Holdings truth source**: `production_data/institutional_summary.json` is canonical
- **CUSIP-first, not issuer-first**: Always reason from CUSIP -> canonical ticker
- **Raw EDGAR XML is debug-only**: Never build narratives from raw filing parses
- **If raw count != summary count**: investigate the summary pipeline first

---

## Source Files

| Component | File |
|----------|------|
| Manager Onboarding | `tools/onboard_manager.py` |
| 13F Cache Warmer | `tools/warm_13f_cache.py` |
| Institutional Summary Builder | `build_institutional_summary.py` |
| 13F Refresh Readiness | `tools/prep_13f_refresh.py` |
| Cohort Quarantine | `tools/check_13f_cohort_quarantine.py` |
| Snapshot Collapse Guards | `tools/verify_snapshot_integrity.py` |
| Manager Registry | `production_data/manager_registry.json` |
| Institutional Summary | `production_data/institutional_summary.json` |

---

# SECTION 2: OPERATIONAL STATE

> **SNAPSHOT DATA** - The values below are point-in-time and go stale. Verify against current pipeline output before citing.

---

## inst_delta_z Current Status

*Last reviewed: 2026-05-04*

- **Zeroed in selector** (2026-05-04): ALERT confirmed, mean IC = -0.097 over 36 dates
- **Absent from production ranker**: the live pairwise artifact has only `coinvest_score_z` and `financial_score`
- **Historical audit note**: prior within-top-30 analysis found positive discrimination (NW-t = +3.32), retained as context only
- **Reinstatement conditions**: documented in governance log, requires IC recovery evidence

## 13F Filing Cycle Status

*Last reviewed: 2026-05-24*

- **Completed cycle**: Q1 2026 (period ending March 31, 2026) -- ALL THREE FILED May 15, 2026
- **Accession numbers**: Fairmount 0001104659-26-062419, Deep Track 0001856083-26-000003, Logos Global 0001172661-26-002196
- **Filing pattern**: All three filed on deadline day (May 15), consistent with Q1 2025 pattern
- **CIKs**: Fairmount 0001802528, Deep Track 0001856083, Logos Global 0001792126
- **Post-filing action sequence**: (1) Warm 13F cache, (2) Run cohort quarantine, (3) Check collapse guards (coinvest_score_z SD), (4) Refresh IC decomposition, (5) 5-day observation window before treating as production-grade
- **Next cycle**: Q2 2026 (period ending June 30, 2026). Filing deadline ~August 14, 2026. Monitor EDGAR starting ~August 11.

## Q1 2026 13F Cohort Quarantine Status

*Last reviewed: 2026-05-24*

- **Verdict**: QUARANTINE ACTIVE — Jaccard 0.364 (gate requires ≥ 0.70)
- **49/55 managers filed** (84.9%); structural cohort shift, not filing lag
- **Top-30 entering**: ALMS, APGE, ARWR, CMPS, DRUG, MLTX, MLYS, NRIX, RYTM, SNDX, SYRE, TRVI, TYRA, URGN
- **Top-30 leaving**: ANNX, ARGX, AXSM, BCRX, BLTE, CMPX, ERAS, INSM, KYMR, ORIC, SLDB, SLN, SRRK, TSHA
- **Re-decision gate**: condition-based (Jaccard ≥ 0.70 + inst_delta_z mean abs delta < 0.50 + ≥10 post-refresh snapshots + no active incidents)
- **Earliest plausible re-decision**: 2026-06-15; more likely 2026-07-01+
- **Gate results documented**: `artifacts/audit/13f_q1_2026_refresh_gates_2026_05_24.md`
