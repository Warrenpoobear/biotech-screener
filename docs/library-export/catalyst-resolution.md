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

### Resolved Entries

**Resolution #1 — GILD Hepcludex (bulevirtide-gmod) 8.5 mg**
- Indication: Chronic Hepatitis Delta Virus (HDV)
- Event type: PDUFA/NDA (Accelerated Approval)
- Decision date: 2026-05-22
- Outcome: APPROVED
- D0 move: +3.04% (close $134.36); D+1 ~$134.34 (held, pre-Memorial Day)
- Coinvest: NO (large-cap pharma, not in Tier 1 registry)
- Surprise factor: LOW (first and only FDA-approved HDV treatment in US; prior EEA approval)
- Note: BPIQ calendar listed event for May 27-30 window; resolved 5 days early on May 22

**TNGX — OPEN ID RESOLVED (2026-05-27)**
- Original flag: "Friday reporter" from Week 4 post-mortem; D+1 = Tue May 27
- Resolved: Actual earnings date was May 13, not May 23. D0 reaction: -10.8% to -17.3% (zero Q1 revenue, Gilead partnership ended)
- May 26 post-holiday close: $20.04 / -0.99% — flat/uninformative
- Classification: NEUTRAL / ID CLOSED. Excluded from PDUFA ledger (earnings catalyst, not PDUFA/AdCom/Ph3)

**Resolution #2 — LPCN 1154 (oral brexanolone)**
- Indication: Postpartum Depression
- Event type: Ph3 (ASCP Annual Meeting oral presentation, 2026-05-26)
- Outcome: FAILED primary endpoint (overall population). Subgroup excluding outlier site was statistically significant (treatment effect -7.1 at 12h), but not a clean top-line win.
- Score: NEGATIVE
- D0 move: -2.16% (2026-05-27); D+1: +0.89% ($2.28, 2026-05-28)
- Coinvest: NO
- Surprise factor: LOW (primary miss previously disclosed; ASCP was subgroup data)
- Path forward: FDA guidance meeting requested; Breakthrough Therapy + Fast Track applications filed; new validation trial planned (interim Q1 2027). Debt-free.

**Resolution #3 — GALT belapectin (GR-MD-02) — NEUTRAL conference data**
- Indication: MASH Cirrhosis / Portal Hypertension
- Event type: EASL Congress 2026 secondary analyses (2026-05-27)
- Outcome: NEUTRAL (secondary/poster analyses from previously published NAVIGATE trial, not new top-line)
- Score: NEUTRAL
- D0: -3.23% ($2.70); D+1: Flat ($2.70)
- Volume: 2.81x average on presentation day
- Coinvest: NO
- Notes: NAVIGATE results published Hepatology May 11. Belapectin 2mg/kg: stat sig liver stiffness worsening reduction (11.7% vs 23.9%, p=0.03). FDA Type-C meeting Q2 2026.

**Resolution #4 — MDGL resmetirom (Rezdiffra) — NEUTRAL post-approval data**
- Indication: MASH (FDA-approved March 2024)
- Event type: EASL Congress 2026 secondary analyses (2026-05-27)
- Outcome: NEUTRAL (8 posters, secondary analyses; drug already approved)
- Score: NEUTRAL
- D0: +0.65% ($527.67)
- Coinvest: NO
- Notes: 44.4% of statin-treated patients achieved LDL-C <70 mg/dL at 52 wks; 91% improved/stabilized liver stiffness at 1yr. Not a binary catalyst.

**Resolution #5 — MNKD Afrezza (inhaled insulin) Pediatric sBLA**
- Indication: Type 1 and Type 2 Diabetes (children and adolescents aged 6+)
- Event type: PDUFA/sBLA
- Decision date: 2026-05-29
- Outcome: APPROVED (first needle-free insulin option for pediatric patients; based on Phase 3 INHALE-1 trial)
- Score: POSITIVE
- D0 move: +4.81% intraday May 29 (to ~$3.815; up ~6% at midday peak per reports). First full trading day was May 29 itself (approval announced during market hours).
- D+1 move: -6.37% ($3.53, Jun 1) — sell-the-news
- Coinvest: NO (small-cap, not in Tier 1 registry)
- Surprise factor: LOW (FDA had released MNKD from postmarketing pulmonary malignancy requirement May 27 — constructive pre-signal)
- Note: Prior close $3.64 (+2.54% day before). Afrezza available at $35/month or less.

### Cumulative Stats (n=11 resolved, as of 2026-06-28)
- PDUFAs resolved: 5 (GILD, MNKD, CING, ARVN, VRDN) | Approvals: 4 | CRLs: 1 (CING inferred) | Approval rate: 80% (of confirmed PDUFAs)
- Ph3 binary readouts: 3 (LPCN failed, SMMT met, RVMD met) | Hit rate: 67%
- Conference data (non-binary): 3 (GALT, MDGL — NEUTRAL; COGT — data confirmation NEUTRAL)
- Sell-the-news pattern confirmed: MNKD (-6.4% D+1), SMMT (-10.4% D+1) — monitor VRDN D+1 (Jun 27/30) for pattern repeat
- Coinvest names positively resolved: COGT (Fairmount) + RVMD (Logos/Baker Bros) + **VRDN (FM 14.04% + DT 5.4M sh)** — VRDN is first Tier 1 binary PDUFA resolution with full dual-manager coinvest
- Coinvest names resolved: 3 of 11 (COGT, RVMD, VRDN)
- Outcome distribution: 6 POSITIVE/HIT, 2 NEGATIVE/MISS, 3 NEUTRAL (data confirmation)

**Resolution #11 — VRDN veligrotug (Lumvoa) PDUFA — APPROVED (4 days early)**
- Indication: Thyroid Eye Disease (TED) — active and chronic forms
- Event type: PDUFA/BLA (Priority Review + Breakthrough Therapy Designation; PDUFA target Jun 30, 2026)
- Decision date: 2026-06-26 (FDA approved 4 days ahead of PDUFA target)
- Outcome: APPROVED — Lumvoa™ (veligrotug-vvze) approved for TED; supported by Phase 3 THRIVE (active TED) + THRIVE-2 (chronic TED). Both met all primary and secondary endpoints. Proptosis and diplopia improvements began as early as 3 weeks. 5 IV infusions over 12 weeks. Immediate commercial launch announced.
- Score: POSITIVE
- D0 move: +6% after-hours Jun 26 (from ~$17.39 close)
- Coinvest: YES — FM 14.04% (3.9M sh + $20M add May 11) + DT 5.4M sh post-Q1 — largest active Tier 1 coinvest pair at time of resolution
- Surprise factor: LOW-MODERATE (approval broadly expected; 4-day early action was mild positive; stock had declined ~44% YTD from $31.12 to ~$17 by pre-PDUFA)
- Note: First commercial product for Viridian. Priced at parity with existing approved IGF-1R therapy on course-of-therapy basis. SQ formulation BLA filing expected early 2027.

**New resolution since last run (Resolution #10, added 2026-06-05):**

**Resolution #10 — ARVN vepdegestrant (VEPPANU) PDUFA — APPROVED (one day early)**
- Indication: ER-positive, HER2-negative, ESR1-mutated advanced/metastatic breast cancer
- Event type: PDUFA/NDA (PDUFA target Jun 5, 2026)
- Decision date: 2026-06-04 (FDA approved one day ahead of PDUFA target)
- Outcome: APPROVED — Veppanu (vepdegestrant/ARV-471) approved; Phase 3 VERITAC-2: 43% reduction in risk of disease progression or death vs. fulvestrant
- Score: POSITIVE
- D0 move: Pending resolution confirmed Jun 5; approval announced Jun 4
- Coinvest: NO (not in Tier 1 registry per Q1 2026 13F)
- Surprise factor: LOW (Rigel commercialization deal pre-signed; muted reaction expected)
- Note: Oral tablet once daily with food. Key safety: QTc interval prolongation and embryo-fetal toxicity.

**Previously added resolutions (Resolutions #6–#9, added 2026-06-02):**

**Resolution #6 — SMMT ivonescimab + chemo (HARMONi-6) — Ph3 OS MET PRIMARY (POSITIVE)**
- Indication: 1L advanced squamous NSCLC
- Event type: Ph3 OS (ASCO Plenary LBA4, 2026-05-31)
- Outcome: OS met primary endpoint — HR=0.66, median OS 27.9 vs 23.7 mo (+4.2 mo), p=0.0017; Lancet publication; first OS superiority over active PD-1 control
- Score: POSITIVE (data) but SELL-THE-NEWS on market reaction
- D0 (Jun 1 open): +10% overnight; D+1 close: -10.43% ($15.71) — China-only population (532 pts) weighed on market
- Coinvest: NOT confirmed — Fairmount, Deep Track, Logos do NOT hold SMMT per Q1 2026 13F verification
- Surprise factor: MODERATE-HIGH on magnitude; NDA not yet filed for sq-NSCLC in US

**Resolution #7 — COGT bezuclastinib + sunitinib (PEAK Ph3) — DATA CONFIRMATION (NEUTRAL)**
- Indication: Post-imatinib GIST
- Event type: Ph3 full data (ASCO Oral, 2026-05-30)
- Outcome: Confirmed — HR=0.50, mPFS 16.5 vs 9.2 mo, ORR 46% vs 26%; PDUFA Nov 30, 2026
- Score: NEUTRAL (buy-the-rumor fully priced; not a binary surprise)
- D0 (May 30): -1.19%; D+1 (Jun 1): -0.43% — flat
- Coinvest: Fairmount Funds $212.9M / 15.3% float (most concentrated Tier 1 coinvest in registry)
- Next binary: PDUFA Nov 30, 2026

**Resolution #8 — RVMD daraxonrasib (RASolute 302) — Ph3 OS LANDMARK (POSITIVE)**
- Indication: 2L+ metastatic PDAC
- Event type: Ph3 OS (ASCO Plenary LBA5, 2026-05-31)
- Outcome: Median OS 13.2 vs 6.7 mo (~2x improvement); mPFS 7.3 vs 3.5 mo; NEJM simultaneous publication
- Score: POSITIVE (landmark data)
- D0 (May 31 PM): +5.5% intraday; D+1 close (Jun 1): $157.48, +1.9% — muted; ~40% pre-ASCO gain already priced
- Coinvest: Logos Capital $194.5M + Baker Bros $929M — largest Tier 1 coinvest pair in registry

**Resolution #9 — CING CTx-1301 (dexmethylphenidate PTR) — PDUFA NEGATIVE (CRL inferred)**
- Indication: ADHD; PDUFA target date May 31 (FDA missed weekend, effective Jun 1)
- Outcome: CRL inferred — no approval 8-K found through Jun 1 EOD; stock -15.38% ($4.16 → $3.52)
- Prior CRL on CMC grounds; CMC information requests outstanding
- Status: PENDING official 8-K CRL confirmation

## High-Priority Upcoming Catalysts (as of 2026-07-12)

*Last reviewed: 2026-07-12. Verify against current BPIQ calendar before acting.*

*Data-quality note (2026-06-26): An "IMCR tebentafusp PDUFA Jun 18" row was removed as a verified false catalyst. KIMMTRAK (tebentafusp) was FDA-approved Jan 2022 (PDUFA Feb 23, 2022); the only active tebentafusp program (TEBE-AM, cutaneous melanoma) is still Phase 3. There is no 2026 IMCR PDUFA, so the Jun 18 entry was bad data (likely a Lane 2 / CTGOV false catalyst). Do not re-add.*

| Date | Ticker | Event | Notes |
| --- | --- | --- | --- |
| ~~Fri May 29~~ | ~~MNKD~~ | ~~Afrezza Pediatric PDUFA~~ | APPROVED. D0 +4.81%; D+1 -6.37% sell-the-news. |
| ~~Sun May 30~~ | ~~COGT~~ | ~~bezuclastinib PEAK Ph3 ASCO oral~~ | DATA CONFIRMATION (NEUTRAL). FM $212.9M / 15.3% float. PDUFA Nov 30 is next binary. |
| ~~Sun May 31~~ | ~~SMMT~~ | ~~ivonescimab HARMONi-6 OS (ASCO LBA4)~~ | POSITIVE OS (HR=0.66). D+1 -10.43% sell-the-news. China-only population. No Tier 1 coinvest. |
| ~~Sun May 31~~ | ~~RVMD~~ | ~~daraxonrasib RASolute 302 (ASCO LBA5)~~ | LANDMARK OS (2x, NEJM). D+1 +1.9% muted — pre-ASCO gain fully priced. Logos $194.5M + Baker Bros $929M. |
| ~~Sun May 31~~ | ~~CING~~ | ~~CTx-1301 ADHD PDUFA~~ | CRL INFERRED. -15.38% price action Jun 1. Awaiting official 8-K. |
| ~~Tue Jun 2~~ | ~~CELC~~ | ~~gedatolisib VIKTORIA-1 ASCO LBA1008~~ | ASCO data presented Jun 2; conference call Jun 2 9:45 AM CT. PDUFA Jul 17 is next binary. Pre-data -7.47% Jun 1. |
| ~~Fri Jun 5~~ | ~~ARVN~~ | ~~vepdegestrant (VEPPANU) PDUFA~~ | APPROVED Jun 4 (one day early). VERITAC-2: 43% reduction in PFS risk vs. fulvestrant. D0 = Jun 5 (yesterday). |
| ~~Tue Jun 30~~ | ~~VRDN~~ | ~~veligrotug PDUFA~~ (thyroid eye disease, Priority Review) | APPROVED Jun 26 (4 days early). Lumvoa™ (veligrotug-vvze). D0 +6% AH. See Resolution #11. FM 14.04% + DT 5.4M sh. |
| Wed Jul 1 | KURA | tipifarnib PDUFA | Outcome unconfirmed as of Jul 12 — no verified 8-K/press release found; leave PENDING. Do not flip status without primary-source confirmation. |
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

Research-mode isolation verified: 100% success rate, 0 writes to live output/hedge_report/ path. All Phase D outputs in `artifacts/research/bioshort_backfill/forward_analysis/`.

**Caveat**: Pseudo-PIT (features computed with current logic on historical snapshots). No promotion claims supported per Spec 092 section A6 - descriptive analysis only. Candidate for independent overlay signal or pre-trade timing filter, but requires true forward evidence before any production use.

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

If the FDA's RTCT initiative succeeds (20-40% trial duration reduction projected), the binary catalyst model evolves:

- Faster time-to-market increases NPV of pipeline assets
- Reduced trial costs improve capital efficiency for small-cap biotechs
- Real-time safety data reduces binary event risk for investors
- Adaptive designs enable mid-trial pivots preserving option value
Monitor as Tier 4 governance question for catalyst_decay_w and catalyst_quality calibration.
