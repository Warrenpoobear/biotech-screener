# Catalyst & Event Resolution Skill

## Purpose

Reference for the catalyst and event resolution pipeline - from multi-source event ingestion through catalyst timing/quality signals used in the screener's production scoring.

This skill is organized into two sections:

1. **Framework Reference** - Stable architecture, sources, and signal definitions (changes only with code updates)
2. **Operational State** - Volatile snapshots that require periodic refresh

---

# SECTION 1: FRAMEWORK REFERENCE

---

## Architecture Overview

```
7+ Event Sources (CTGov, SEC 8-K, FDA ADCOM, FDA Regulatory, PDUFA, EMA, merged trials)
  -> event_ledger.py (unified event ledger)
  -> catalyst_resolution_tracker.py (per-ticker resolution files)
  -> catalyst_decay_w (timing signal, production)
  -> catalyst_quality / binary_quality_score (quality signal, Spec 078)
  -> event_ev_p_hit (EV binder, Spec 077, prospective accumulation)
```

## Event Ledger

**Builder**: `event_ledger.py` - `build_event_ledger()`

### Sources (7+)

| Source | Data |
| --- | --- |
| ClinicalTrials.gov (AACT) | Trial status changes, phase transitions |
| SEC 8-K | Material events (earnings, FDA actions) |
| FDA ADCOM | Advisory committee meetings |
| FDA Regulatory | Approval/CRL/priority review decisions |
| PDUFA Manual | Target action dates |
| EMA | European regulatory decisions |
| Merged Trials | Cross-registry deduplicated (NCT/EudraCT) |

### EU/EEA Registry Collectors

- `euctr_collector.py` - EU Clinical Trials Register
- `ctis_collector.py` - Clinical Trials Information System
- `isrctn_collector.py` - ISRCTN Registry
- `trial_registry_merger.py` - Cross-registry dedup by NCT/EudraCT IDs

---

## catalyst_decay_w (Timing Signal)

Measures proximity to the next known catalyst event. Near-term catalyst = higher weight.

### Key Properties

- Production signal in rankings.csv
- Signal primarily discriminates in lower quartile (~15-18 tickers); top-60 tends toward ceiling effect (median = 1.000)
- IC tests blocked on Spec 071 Lane 2 + Gate 4
- Requires >= 30 post-PIT HIT/MISS outcomes for formal evaluation

### Monitoring (updated 2026-05-13)

Shadow-track catalyst_decay_w + binary_quality_score distributions in top-60 monthly (Spec 097). No formal IC claims until gates clear.

**Spec 097 monitoring framework** (canonicalized 2026-05-13):

- Event-EV prospective monitoring with Brier score gate (Brier <= 0.08)
- Minimum n >= 30 calibration threshold required
- Tier-wise validation

**Spec 098 monitoring framework** (canonicalized 2026-05-13):

- Catalyst timing prospective monitor
- Correlation > 0.15 gate required
- Tier-wise validation

---

## catalyst_quality / binary_quality_score (Quality Signal)

Classification of catalyst event quality (Spec 078).

### Key Properties

- binary_quality_score has meaningful variability (IQR ~0.2)
- Joint opportunity (timing + quality): typically ~38% of top-60 tickers

### CTGOV_CALENDAR Dependency

A material share of top-60 catalysts are sourced from ClinicalTrials.gov calendar. Lane 2 dependency confirmed material - some false catalysts expected in any given top-60.

---

## event_ev_p_hit (EV Binder, Spec 077)

Bayesian expected value estimate for catalyst events, binding EV artifacts to resolution outcomes.

### Design

- Forward-only (no backfill)
- Writes null where no EV artifact match exists (correct behavior)
- Prospective sample accumulation required before evaluation

### Gate Requirements

| Gate | Requirement |
| --- | --- |
| Gate 3 | >= 15 non-null event_ev_p_hit records |
| Gate 4 | >= 30 post-PIT HIT/MISS with non-null |
| Spec 079 | Calibration review at n >= 30 |

### Calibration Bias Risk

EV Bayesian priors derived from FDA historical precedent and endpoint type, fit before post-PIT-fix period. If priors are miscalibrated (e.g., FDA accelerated-approval scrutiny shift), values may be systematically biased. Risk level: MEDIUM-HIGH.

---

## Catalyst Resolution Tracker (CRT)

Per-ticker resolution files tracking catalyst event outcomes.

**Location**: `data/snapshots/resolutions/{YYYY-MM}/`

### Resolution States

| State | Meaning |
| --- | --- |
| HIT | Catalyst event occurred and was positive |
| MISS | Catalyst event occurred and was negative |
| PENDING | Event not yet resolved |
| EXPIRED | Event window passed without resolution |

### watchlist_current.json

- Today-only aggregator regenerated on every cron run
- NOT tracked in git (gitignored after contaminating commits)
- History captured by per-ticker resolution files
- Freshness check: as_of_date within 3 days (WARN if stale)

---

## AACT Pipeline

ClinicalTrials.gov data ingestion via AACT database.

### Timing

- Pipeline timeout: 6000s (100 min) to cover worst-case AACT + tail steps
- Monday runs are typically longest (weekend AACT batch)
- Previous 4500s (75 min) timeout was killing the pipeline mid-AACT

### Cache Warming

```bash
warm_caches.py --sources sec_8k,ctgov,sec_13f,fda_adcom,fda_regulatory,euctr,ctis,isrctn,merged_trials
```

Always warm 8-K cache BEFORE running screen.

---

## Composite Integration

Catalyst signals enter Module 5 composite via Module 3:

| Weight Set | Catalyst Weight |
| --- | --- |
| V3 Enhanced | Part of remaining allocation |
| V3 Default | 25% (legacy) |

---

## Source Files

| Component | File |
| --- | --- |
| Event Ledger Builder | `event_ledger.py` |
| Catalyst Resolution Tracker | `catalyst_resolution_tracker.py` |
| Module 3 Scoring | `module_3_scoring_v2.py` |
| Snapshot Column Spec | `run_screen_columns.py` |
| Cache Warmer | `warm_caches.py` |
| AACT Collector | `wake_robin_data_pipeline/collectors/` |
| Trial Registry Merger | `trial_registry_merger.py` |

---

# SECTION 2: OPERATIONAL STATE

> **SNAPSHOT DATA** - The values below are point-in-time and go stale. Verify against current pipeline output before citing.

---

## event_ev_p_hit Gate Progress

*Last reviewed: 2026-05-28*

- Binder shipped and operational (forward-only, confirmed present 2026-05-08)
- Spec 087 B1b: PASS (first-fire validation complete 2026-05-13)
- Non-null records accumulated: 0 (as of 2026-05-08); accumulating with each pipeline run
- Gate 3 (n >= 15): **0/15** - accumulating
- Gate 4 (n >= 30): **0/30** - blocked on Gate 3
- Next monthly check: 2026-06-08

## Catalyst Resolution Ledger (CRL) — Initialized 2026-05-25

*Last reviewed: 2026-06-08*

The CRL tracks manual resolution of high-profile PDUFAs and catalyst events from the operator's watchlist. This is separate from the pipeline CRT (which is auto-populated). The CRL is stored at: `content://collections/biotech-screener/Biotech-Earnings-Post-Mortem-Ledger`.

### Resolved Entries (summary)

The CRL contained 10 resolved entries as of 2026-06-06, including GILD Hepcludex (APPROVED), MNKD Afrezza pediatric (APPROVED), LPCN 1154 (Ph3 FAILED), GALT/MDGL (NEUTRAL conference data), SMMT HARMONi-6 (Ph3 OS MET, sell-the-news), COGT PEAK (data confirmation NEUTRAL), RVMD RASolute 302 (landmark Ph3 OS), CING (CRL inferred), and ARVN vepdegestrant (APPROVED one day early).

### Cumulative Stats (n=10 resolved + CING pending official 8-K, as of 2026-06-06)
- PDUFAs resolved: 4 (GILD, MNKD, CING, ARVN) | Approvals: 3 | CRLs: 1 (CING inferred) | Approval rate: 75% (of confirmed PDUFAs)
- Ph3 binary readouts: 3 (LPCN failed, SMMT met, RVMD met) | Hit rate: 67%
- Conference data (non-binary): 3 (GALT, MDGL — NEUTRAL; COGT — data confirmation NEUTRAL)
- Sell-the-news pattern confirmed: MNKD (-6.4% D+1), SMMT (-10.4% D+1)
- Coinvest names positively resolved: COGT (Fairmount) + RVMD (Logos/Baker Bros) — both data confirmations

## High-Priority Upcoming Catalysts (as of 2026-06-25)

*Last reviewed: 2026-06-25. Verify against current BPIQ calendar before acting.*

*Data-quality note (2026-06-26): An "IMCR tebentafusp PDUFA Jun 18" row was removed as a verified false catalyst. KIMMTRAK (tebentafusp) was FDA-approved Jan 2022; the only active tebentafusp program (TEBE-AM) is still Phase 3. There is no 2026 IMCR PDUFA. Do not re-add.*

| Date | Ticker | Event | Notes |
| --- | --- | --- | --- |
| Tue Jun 30 | **VRDN** | **veligrotug PDUFA** (thyroid eye disease, Priority Review) | **HIGHEST PRIORITY.** FM 14.04% (3.9M sh) + DT 5.4M sh; $350M raise complete; analyst avg PT $36.14 (12 Buy/1 Hold/1 Sell). |
| Wed Jul 1 | KURA | tipifarnib PDUFA | -- |
| Thu Jul 17 | CELC | gedatolisib PDUFA | Post-ASCO data |
| Sun Aug 24 | BIIB | LEQEMBI SC PDUFA | Extended from May 24 |

## BioShort / Hedge Report Forward Analysis (Spec 092)

*Added: 2026-05-13. Spec 092 Phases A-D all complete.*

Historical backfill of hedge report features across 146 snapshots with forward return analysis (pseudo-PIT):

| Metric | Value |
| --- | --- |
| DEFER verdict accuracy (T+5) | 60.5% (129 samples) |
| Median T+5 return | +0.63% |
| Median T+20 return | +2.49% |
| Median 20d max drawdown | -2.86% |

Research-mode isolation verified: 100% success rate, 0 writes to live output/hedge_report/ path.

**Caveat**: Pseudo-PIT (features computed with current logic on historical snapshots). No promotion claims supported per Spec 092 section A6 - descriptive analysis only.

## binary_quality_score Coverage

*Last reviewed: 2026-05-08*

- 261/261 (100%) catalyst rows classified in current snapshot
- Rising trend in May: n(>0.7) grew from 24 to 34 tickers in top-60
- Joint opportunity (timing + quality): median 23/60 tickers (38%)

## CTGOV_CALENDAR Dependency

*Last reviewed: 2026-05-08*

- ~48% of top-60 catalysts sourced from ClinicalTrials.gov calendar
- ~6 estimated false catalysts in current top-60 (Lane 2 dependency)
- BCRX excluded from monitoring

## catalyst_decay_w Coverage

*Last reviewed: 2026-05-08*

- 299/299 coverage in recent snapshots
- Median = 1.000 in top-60 (ceiling effect confirmed)

## External Catalyst Platforms and FDA Initiative (May 2026)

### Catalyst Calendar Platforms

- BiotechSigns: 970 companies, 74,988 active signals (PDUFA, Phase readouts, insider filings)
- CatalystAlert: 1,624 companies, 14,310 drug pipelines, 3,815 upcoming catalysts
- BioCatalysts.AI: Bio-Score algorithm predicting volatility magnitude per catalyst event
- PDUFA.BIO: 200+ PDUFA events scored by ODIN (96.2% verified accuracy)

### ODIN Confidence Tiers

ODIN assigns binary confidence tiers that correlate with post-approval stock performance:

- TIER_1: >85% approval probability
- TIER_2: 70-85%
- TIER_3: 40-70%
- TIER_4: <40%

Potential external benchmark for CRT catalyst_quality classification.

### FDA Real-Time Clinical Trial Initiative

If the FDA's RTCT initiative succeeds (20-40% trial duration reduction projected), the binary catalyst model evolves. Monitor as Tier 4 governance question for catalyst_decay_w and catalyst_quality calibration.
