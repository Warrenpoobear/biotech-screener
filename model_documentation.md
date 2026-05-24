# Wake Robin DEM — Model Documentation

**Version:** 1.7.2 (ruleset `8887576e`, v1.14.0 — 2026-05-04 demotion of `inst_delta_z`; see `RULESET_CHANGELOG.md`)
**Last updated:** 2026-05-24 (documentation sync to active v1.14.0: coinvest-only selector, capped Family C 2-feature ranker, freeze/quarantine active)
**Status:** Production — coinvest-only selector (`coinvest_score_z` 100%, `inst_delta_z` 0%) + pairwise_minimal ranker (2-feature, ordinal-only) + EW Top-30 | **FROZEN** until governance gates clear

---

## Recent Corrections & Governance Status (2026-05-17)

**Spec 100 IC Tooling Correction (Commit 2faa88e6)**
- **Change:** `run_rank_ic_backtest.py` now measures `final_score` (production ranker) instead of `composite_score`
- **Reason:** Prior IC claims using `composite_score` were measuring the wrong signal; ranker IC was unmeasurable
- **Status:** ✓ Corrected. Prior composite_score IC claims invalidated; final_score baseline established
- **Interpretation:** Deferred until post-freeze Checklist v2 battery

**Module 4 Clinical Score Denominator Fix (Commit 3ad7b904)**
- **Change:** Clinical normalization denominator corrected from 120 → 117
- **Reason:** Execution_score max is 22 (not 25); total raw max = 117, not 120
- **Effect:** clinical_score increases by factor 120/117 (~+2.56%); hard ceiling now 100.0 (was 97.5)
- **Tests:** 27/27 clinical tests pass; synthetic max case validates 100.0
- **Status:** ✓ Verified and merged (PR #288, awaiting review)

**13F Q1 2026 Cohort Quarantine (Active)**
- **Status:** Quarantine ACTIVE — 6/48 managers filed (as of 2026-05-15)
- **Trigger for clearance:** ≥34 managers filed + all 6 validation gates pass
- **Expected:** ~2026-05-23 (fuller filing coverage) → validation rerun → ~2026-05-26 clearance decision or extension
- **Enforcement:** No selector/ranker/sizing changes authorized until cohort Jaccard ≥0.70 and distortion clears
- **Monitoring:** Cron active weekdays 6:22 PM ET through 2026-06-20

**Architecture Freeze (Active)**
- **Period:** ~2026-04-25 through h20d checkpoint ~2026-05-26
- **Scope:** No model logic changes, no feature promotions, no ranking modifications
- **Exceptions:** Deterministic tooling (preflight, monitoring, verification) permitted; no alpha changes
- **Unblock:** Post-h20d checkpoint, pending 13F cohort clearance + Checklist v2 validation

---

## 0. Governing Principles

These principles are non-negotiable. Every feature, model change, data source,
and production decision must satisfy all of them. They override convenience,
speed, and aesthetics.

### Point-in-Time Safety (PIT)

No information from the future may influence a historical or live decision.

- Every data field has a `disclosed_at`, `collected_at`, or `filing_date` anchor
- Backtests use only data available at each snapshot date
- Trial records are filtered by `collected_at <= as_of_date`
- CRT resolutions are filtered by `resolution_date <= as_of_date`
- Price lookups for entry use backward-only tolerance (never forward)
- Price lookups for realized returns use forward tolerance (measuring outcomes, not decisions)
- Any field with unknown provenance is tagged `unknown` and excluded from live inference
- Pre-PIT-correction performance claims (+93.7pp) are permanently deprecated
- The only credible evidence is PIT-corrected backtest (+2.34pp/mo, t=2.60) and forward monitoring

PIT violations are treated as bugs, not tradeoffs. There is no "small" leakage.

### Determinism

The same inputs must always produce the same outputs.

- All scoring uses `Decimal(str(x))` arithmetic, never raw floats, in production paths
- JSON outputs use sorted keys and stable serialization
- No `datetime.now()` in scoring logic — all timestamps derive from `as_of_date`
- No randomness in any production path (no `random`, no sampling, no stochastic initialization)
- Snapshot promotion is atomic: staging → validation gates → promotion or rejection

### Fail-Closed Design

When in doubt, the system rejects rather than guesses.

- Validation gates default to FAIL unless all checks pass explicitly
- Missing data → ineligible or penalized, never imputed as favorable
- Unknown phase → conservative prior, not optimistic
- Absent options data → no options overlay, not synthetic pricing
- Gate failures block snapshot promotion (exit 1); warnings allow but flag (exit 2)

### Bounded Influence

No single feature, layer, or data source may dominate the model.

- Every new signal enters via bounded composition (CalendarAlpha `max_adjustment`, logit caps)
- Clinical transmission layer capped per-phase (Ph1 ±0.12, Ph2 ±0.25, Ph3 ±0.20 log-odds)
- Biomarker context score capped at [-0.05, 0.30]
- Protocol quality soft floors prevent early-phase collapse
- No feature weight exceeds the base prior's influence without Checklist v2 validation

### Evidence-Based Promotion

No signal enters production without measured justification.

- Checklist v2 required for any new signal promotion: Fama-MacBeth, bootstrap, FDR, LOSO, year stability
- All new features start as diagnostic overlays, not ranking inputs
- Shadow validation precedes production promotion (minimum 2-week forward window)
- Ablation required: old vs new, top-10/20 overlap, actionable count, by-phase effects
- Decision rule: promote only if changed decisions are directionally better, calibration improves,
  and no systematic failure mode is detected
- If results are mixed, prefer partial adoption or extended shadow over premature promotion

### Separation of Concerns

Different causal processes get different model paths.

- Clinical catalyst success (trial/FDA outcome) uses the outcome model + clinical transmission
- Market expectation (crowd belief) uses the expectation model with its own feature set
- Scenario payoff (price reaction) uses the payoff engine with analog-based distributions
- M&A event probability (deal close) will use a separate event sleeve, not the clinical p_hit path
- Research labels (HINT benchmark) are tagged `offline_eval_only` and never enter live scoring
- Read-only agents observe and report; they do not modify production state

### Data Hygiene

Trust the source hierarchy. Verify before acting on derived data.

- Structured ClinicalTrials.gov fields (allocation, masking, intervention_model) take precedence over text parsing
- FDA regulatory state comes from Drugs@FDA, not openFDA (which is enrichment, not source-of-truth)
- 13F institutional ownership is the source for coinvest/inst_delta, refreshed quarterly
- PubMed enrichment is optional (`--enrich-pubmed` flag) and cached (24h TTL); API failure is non-blocking
- Drug name map (`drug_name_map.json`) is curated from PDUFA + trial_records, not free-text extraction
- Herald press release classification is Herald-biased for Phase 1/2 (positive selection); CRT calibration
  excludes these phases (`HERALD_BIASED_PHASES`)
- Any benchmark dataset (HINT, BioTradingArena) stays in `research/` and is never imported by production code

### Reversibility

Every production change must be reversible within one commit.

- Old prior values preserved as named constants (e.g., `_PHASE_2_PRIOR_OLD = 0.310`)
- Feature flags control new behavior (e.g., `--enrich-pubmed`, clinical transmission behind shadow)
- CalendarAlpha weights are config values, not hardcoded logic
- Kill switches exist for options expression overlay, clinical transmission, and PubMed enrichment
- Snapshot promotion can be rolled back by reverting to the previous day's snapshot

### Intellectual Honesty

The model documents what it knows and what it doesn't.

- Historical backtest numbers from prior PIT v2 snapshots (ruleset `69a0c7f8`) were invalidated
  (2026-04-17 audit). Regenerated with then-current ruleset `2a3e79eb` (now retired 2026-05-04;
  current canonical is `8887576e` v1.14.0): +669% DEM vs +54% XBI
  over 75 monthly periods (Jan 2020 – Apr 2026), +2.39pp/mo excess (t=3.14). However, these
  remain **pseudo-PIT** (current code applied retroactively) and are **not credible as forward
  return claims**. Live period (Oct 2024+, 18 months): +112% DEM vs +41% XBI, t=1.24 —
  directionally encouraging but statistically underpowered. Forward shadow/live monitoring
  remains the only credible basis for promotion decisions.
- Biomarker selection is NOT globally positive (HINT Δ=-2.7%); old 1.20x boost was wrong and neutralized
- Clinical transmission is a validated filter (Brier improved), not yet proven alpha (returns identical in PIT-honest window)
- The 1-year paper return (+74.6%) carries survivorship bias, no execution costs, and no rebalancing friction
- Agent governance: agents observe and recommend, they do not execute trades or modify production state unilaterally

---

## 1. System Overview

Wake Robin is a systematic biotech screening and portfolio construction system.
It ranks ~297 biotech names by asymmetric event-driven upside potential, with
the goal of identifying names where clinical/regulatory catalysts create
favorable risk/reward ahead of binary outcomes.

The winning model is not "best science wins." It is **science + survivability +
catalyst timing + strategic relevance + price-vs-probability discipline**. A name
enters the portfolio not because it has the most promising drug, but because the
intersection of sponsorship quality, financial resilience, catalyst proximity, and
market mispricing creates an asymmetric setup. Names with great science but poor
timing, thin sponsorship, or already-priced expectations are filtered out — the
system explicitly penalizes "safe but less catalytic" names and traps where the
market has correctly priced above base rates.

### Architecture

```
Universe (M1) → Financial Health (M2) → Catalyst Events (M3) → Clinical Dev (M4)
→ Composite Scoring (M5) → Decision Engine (L0→L2→L4→L4b→L3)
→ Selector Engine (v1.14.0: coinvest_score_z 100%; inst_delta_z 0% diagnostic)
→ Ranker Engine (pairwise_minimal: 2 features, ordinal-only, top-60 cohort)
→ Sort by final_score → EW Top-30 → Portfolio Construction
→ Shadow Portfolio → Performance Attribution → Governance Gates
```

### Two-Stage Scoring (Spec 050, adopted 2026-04-03; QA revalidated 2026-04-04)

The model uses a **selector/ranker split**: one score to choose the shortlist, a different
score to rank within it. This was validated on true PIT data (67 monthly periods, Jun 2020 —
Apr 2026) at +2.34pp/mo net-of-cost, t=2.57.

> **Production mental model (2026-05-24):**
> coinvest selects the book; the capped 2-feature ranker orders within the
> top-60 by bounded coinvest support and the intentional financial_score
> stress-upside penalty. `inst_delta_z` is computed and exported as a
> diagnostic but has zero selector/ranker weight in v1.14.0.

**Stage 1 — Selector (v1.14.0 coinvest-only):** Institutional co-investment quality
determines which names enter the actionable book. Active ruleset `8887576e`
sets `coinvest_score_z` to 100% and `inst_delta_z` to 0% after the 2026-05-04
two-frame ALERT (`inst_delta_z` mean IC = -0.097 over 36 dates). The prior
65/35 B6 blend remains historical evidence, not the live selector. Clinical
quality was destructive as a selector (-0.53pp).

**Stage 2 — Ranker (pairwise_minimal, ordinal-only):** A 2-feature Bradley-Terry pairwise
model ranks within the selected top-60 cohort. Promoted 2026-04-05 after feature audit
confirmed the prior 5-feature model added noise. The ranker is **ordinal-only** — raw scores
are not calibrated (ECE = 0.129, verdict: POOR). Do not rank-weight or confidence-size.

Production model — live deployed weights (`production_data/ranker_v2_model.json`):
- `coinvest_score_z` (deployed weight **+0.02**, capped Family C live pilot; trained basis
  was +0.0613): selects high-sponsorship names within cohort. Internal code signal name is
  `sponsorship_score_z`; the artifact and live deployment use `coinvest_score_z`.
- `financial_score` (deployed weight **−0.0533**, unchanged from trained): penalizes
  financially safe names — those with less catalytic upside. Negative weight is correct and
  informative. Persists across all cohort widths, both bull and bear regimes. Note:
  `financial_score` in CSV is Module 5 rank-normalized (stage×size cohort), not raw Module 2
  output.

**Deployed artifact ≠ trained vector — Family C live pilot.** The live artifact applies a
cap to the positive feature: the deployed weight on `coinvest_score_z` is **+0.02**, down from
the trained `minimal_v2` weight of **+0.0613**. The `financial_score` weight is unchanged at
−0.0533. See the artifact's `provenance` block: `model_variant = deployed_live_pilot`,
`trained_basis = minimal_v2`, `deployment_delta = coinvest weight capped`. Operators should
read the live weights from the artifact, not from any external doc — this section is a guide,
the artifact is authoritative.

Dead features (confirmed noise, removed 2026-04-05):
- `momentum_delta_z`, `catalyst_decay_w`, `binary_quality_score`, `clinical_score_v2_z`
  all added noise to the pairwise model despite individual FM significance.
  Walk-forward: 2-feature spread +2.95%, IC +0.143 (t=2.98) — beats 5-feature on all metrics.

**Construction:** Equal-weight Top-30. Rank-weighting is not justified (RW-EW = -0.09pp,
t=-0.95). Pairwise calibration confirms: ordinal ranking only, no sizing from scores.

### Core Invariants

1. **Deterministic**: Same inputs → byte-identical outputs. No randomness. Full hash verification.
2. **Point-in-Time (PIT) Safe**: All data access satisfies `source_date <= as_of_date - 1`. Enforced via `pit_enforcement.py`.
3. **Fail-Closed**: Gates default FAIL unless proven otherwise. Validate and stop on errors.
4. **Decimal Arithmetic**: All financials use `Decimal`, never floats.
5. **Stdlib-Only Core**: Modules 1–5 have zero external dependencies.

---

## 2. Decision Engine

The DEM (Decision Engine Model) takes composite-scored names and applies a
layered decision framework to produce a final ranked list with tier assignments
and size bands.

### Layer 0 — Eligibility (Hard Gates)

Determines whether a name is eligible for ranking. All gates must pass.

| Gate | Threshold | Mode |
|------|-----------|------|
| Drawdown | ≤ -0.40 | Hard (v1.12.0) |
| Drawdown hard floor | ≤ -0.75 | Always fail |
| Drawdown relative to XBI | ≥ -0.25 | AND with drawdown gate |
| Financials missing | cash_total ≤ 0 | Bypass for mega-cap |
| Survivability | Red flag severity | Configurable threshold |
| Liquidity | Dollar volume | Configurable threshold |

**Key pitfall:** `financials_missing` gate requires `cash_total <= 0`. Fields
`missing_cash` / `missing_burn_data` are misleading for profitable companies
with positive cash flow.

### Layer 2 — Risk Flags & Momentum

Overlay risk signals on eligible names. These inform sizing and reporting but
do not gate eligibility.

| Signal | Threshold | Effect |
|--------|-----------|--------|
| Volatility (60d) | > 1.20 | Risk flag |
| Beta (XBI, 60d) | > 1.80 | Risk flag |
| Drawdown | < -0.35 | Risk flag |
| RSI (14d) | > 70.0 | Overbought flag |
| Confidence | < 0.30 | Low confidence flag |

**Momentum classification:**
- `alpha_60d_z > +0.05` → tailwind
- `alpha_60d_z < -0.05` → headwind
- else → neutral

### Layer 4 — Development Tier (Drug Developers)

Assigns A/B/C/D tier based on clinical optionality and catalyst presence.

| Tier | Criteria |
|------|----------|
| **A** | optionality ≥ 0.60 AND actionable catalyst (within 120d) |
| **B** | optionality ≥ 0.60 (no actionable catalyst) OR optionality ≥ 0.30 + actionable catalyst |
| **C** | optionality ≥ 0.30 (no catalyst) OR optionality < 0.30 |
| **D** | Ineligible or no optionality data |

### Layer 4b — Commercial Tier (Non-Drug Developers)

Same structure as Layer 4, but uses `commercial_quality_pct` with floors 0.85 / 0.60.

### Layer 3 — Position Sizing

Size bands: XS (0.15), S (0.30), M (0.60), L (1.00).

**Active modifiers (v1.12.0):**
- Cost haircut: enabled (bid-ask spread penalty)
- Catalyst tilt: disabled
- Catalyst type tilt: disabled
- Momentum tilt: disabled

### Sort Key (v1.13.0)

12-element tuple determining final rank order. The sort anchor is now `selector_score`
which uses `final_score` (selector + ranker adjustment) when the ranker is active.

```
(eligible, is_dev, tier_ord, catalyst_priority, catalyst_mode,
 catalyst_days, missing_count, -final_score,
 -sponsor_count, momentum_ord, anchor, ticker)
```

**Sort anchor:** `selector_score` → reads `final_score` (selector + ranker bounded adjustment)

**Selector:** A4 config in `run_screen.py` → `selector_engine.py` (`compute_selector_scores()`)

**Ranker:** clinical_50 config in `ranker_engine.py` → `compute_ranker_adjustments()`
- Activates for names with catalyst ≤ 120d in selector top-60
- Bounded at ±15% of selector_score
- Does NOT require options data (analyst rank model)

**Legacy sort signals:** Still computed but superseded by selector/ranker. The tier system
(A/B/C/D) is still emitted for backward compatibility but no longer drives ordering.

---

## 3. Signal Inventory

### Production Signals (Spec 050 + v1.14.0 governance update 2026-05-04)

**Selector (v1.14.0 coinvest-only) — active production configuration:**

| Signal | Role | Active Weight | Governance Evidence |
|--------|------|---------------|---------------------|
| **coinvest_score_z** | Selector | 100% | Surviving institutional selector signal in ruleset `8887576e`; prior 65% weight redistributed here. |
| **inst_delta_z** | Diagnostic/export only | 0% | Demoted 2026-05-04 after two-frame ALERT: mean IC = -0.097 over 36 dates, hit rate = 0.111. Reinstatement requires governance review. |
| **Prior B6 bundle** | Historical selector evidence | retired 65/35 blend | Checklist v2 bootstrap +2.42pp/mo, CI [1.25%, 3.70%], LOSO ROBUST. Historical evidence only; not the live v1.14.0 selector. |

**Ranker (pairwise_minimal) — 2 features, ordinal-only (ECE=0.129):**

| Signal | Role | Trained Weight | Deployed Weight | Walk-Forward | Interpretation |
|--------|------|----------------|-----------------|-------------|----------------|
| **sponsorship_score_z** (artifact: `coinvest_score_z`) | Ranker (positive) | +0.061 | **+0.02 (capped, Family C live pilot)** | Spread +2.95%, IC +0.143 (t=2.98) | Selects high-sponsorship names within top-60 |
| **financial_score** | Ranker (negative) | −0.053 | −0.053 (unchanged) | Same walk-forward | TRUE PENALTY — safe names have less catalytic upside |

Deployed weights are read live from `production_data/ranker_v2_model.json`; the `provenance`
block is authoritative. Trained weights are retained here for audit-trail comparison only.

**Overlay signals (not in selector/ranker weights):**

| Signal | Role | Checklist v2 | Status |
|--------|------|-------------|--------|
| **event_type_score** | Diagnostic/filter/sizer | **5/5 PASS** (FM incr NW-t=+2.34, FDR q=0.096) | Overlay only — does NOT improve B6 bundle |

### Shadow / Under Review

| Signal | Checklist v2 | Status |
|--------|-------------|--------|
| insider_exec_buy_value_90d | 1/5 (FRAGILE robustness, bootstrap CI includes 0) | Shadow only — downgraded |
| aact_execution_score | 1/5 (bear-unstable −1.86pp, bootstrap CI includes 0) | Shadow only — downgraded |
| clinical_score_v2_z (as ranker) | Negative within top-30, collider-amplified | Quarterly review — drop if drifts to zero |

### Rejected / Disabled Signals

| Signal | Reason | Status |
|--------|--------|--------|
| **clinical_score_v2_z as selector** | Δ=-0.68pp, negative IC, universally destructive (Spec 055) | REJECTED |
| **DEFAULT selector weights** | -0.53pp as selector | REJECTED (clinical 35%/catalyst 25% mix) |
| **clinical composites as ranker** | Negative across ALL robustness slices (Spec 055) | CLOSED |
| cal_alpha | Confirmed noise at all horizons | REMOVED in v1.12.0 |
| optionality as sort anchor | Underwater on PIT data | SUPERSEDED by B6 selector |
| sponsorship_binary | Δ=+0.25pp, t=1.25 | WORTHLESS — count granularity matters |
| total_volume_z | IC=-0.10 on PIT-native data | DEAD |
| quality_tiebreaks (Specs 030/031) | Economically immaterial | Lane EXHAUSTED |
| rank-weighting (any signal) | RW-EW = -0.09pp, t=-0.95; pairwise ECE=0.129 | NOT JUSTIFIED |
| options-as-alpha (Spec 053) | 37 signals tested, ALL fail as selector/ranker | CLOSED |
| static execution features (Spec 054) | PCD overdue, update recency, pipeline velocity all noise | CLOSED |

---

## 4. Data Sources & Architecture

### Data Architecture Overview

```
                              EXTERNAL APIS
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │  SEC EDGAR    │  │ClinicalTrials│  │  Tastytrade   │  │  Yahoo/MS    │
   │  (XBRL,13F,  │  │    .gov      │  │ Options API   │  │  Prices &    │
   │   8-K)        │  │  + AACT DB   │  │  (OAuth2)     │  │  Fundamentals│
   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
          │                 │                  │                  │
          ▼                 ▼                  ▼                  ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │                    DATA COLLECTION LAYER                             │
   │  sec_collector    trials_collector   options_diagnostics  yahoo_coll │
   │  sec_8k_catalyst  poll_ctgov_daily   options_history      warm_price │
   │  warm_13f_cache   fetch_aact_snap    massive_api          refresh_ms │
   │  build_pit_fin    fda_adcom_coll     event_quality        macro_coll │
   └──────────────────────────┬──────────────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │                    PERSISTENCE LAYER                                 │
   │                                                                      │
   │  production_data/          cache/                data/                │
   │  ├── universe.json         ├── ctgov/            ├── snapshots/       │
   │  ├── market_data.json      ├── sec/              ├── snapshots_pit_v2/│
   │  ├── financial_records.json├── fda/              ├── pit_archives/    │
   │  ├── price_history.csv     ├── morningstar_data/ ├── aact/snapshots/  │
   │  ├── pit_financials/       ├── press/            ├── press_releases/  │
   │  ├── ipo_dates.json        ├── market_data/      ├── sponsorship/     │
   │  ├── sponsorship_*.json    └── clinical/         ├── short_interest/  │
   │  ├── fda_designations.json                       └── condition_aliases│
   │  ├── regulatory_calendar*.json                                       │
   │  ├── purple_book.json                                                │
   │  ├── manager_registry.json                                           │
   │  └── adcom_outcomes.json                                             │
   └──────────────────────────┬──────────────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │                    SCREENING PIPELINE (run_screen.py)                │
   │                                                                      │
   │  M1 Universe ─► M2 Financial ─► M3 Catalyst ─► M4 Clinical          │
   │       │              │               │              │                │
   │       ▼              ▼               ▼              ▼                │
   │                M5 Composite Scoring                                  │
   │                       │                                              │
   │                       ▼                                              │
   │          Decision Engine (L0→L2→L4→L4b→L3)                          │
   │                       │                                              │
   │                       ▼                                              │
   │              rankings.csv + metadata.json                            │
   └──────────────────────────┬──────────────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │                    CONSUMPTION LAYER                                  │
   │                                                                      │
   │  Portfolio Construction    Benchmarking        Dashboard (React)     │
   │  Shadow Portfolio          CRT Calibration     Agent Fleet (22)      │
   │  Bioshort Hedge Report     Signal Research     Email Alerts          │
   │  Expression Overlay (062)  Data Explorer       Ops Digest            │
   └──────────────────────────────────────────────────────────────────────┘
```

### External API Sources

| Source | Data Provided | Auth | Refresh | Pipeline Entry |
|--------|--------------|------|---------|----------------|
| **SEC EDGAR** (data.sec.gov) | XBRL company facts, 13F-HR filings, 8-K filings | Public (rate-limited 10 req/s) | Daily (8-K, XBRL) / Quarterly (13F) | M2 financials, sponsorship signal |
| **ClinicalTrials.gov** | Trial registry (status, phases, dates, endpoints) | Public | Daily | M4 clinical development |
| **AACT** (ctti-clinicaltrials.org) | Bulk clinical trial mirror (580K trials) | Public (pipe files) | Weekly (Mon) | M4 enrichment, AACT delta features |
| **Tastytrade** | Options IV, Greeks, skew, term structure | OAuth2 (`TT_SECRET`, `TT_REFRESH`) | Daily (intraday capable) | Options diagnostics, EPD |
| **Massive** | Historical options chains, quotes | API key (`MASSIVE_API_KEY`) + S3 | Daily | Options history backfill |
| **Yahoo Finance** | Stock prices, balance sheets, income statements | Public (rate-limited) | Daily | Price history, market data |
| **Morningstar Direct** | Fundamental data, volatility, star ratings | JWT (`MD_AUTH_TOKEN`) | Daily | M2 enhancements, vol enrichment |
| **FRED** (St. Louis Fed) | VIX, TNX, IRX, fed rate, HYG, SPY | API key (`FRED_API_KEY`) | Daily | Regime classifier (7 feeds) |
| **BioTradingArena** (biotradingarena.com) | 655 validated catalyst cases with press releases, trial data, price action, ground truth | Bearer auth (API key) | Ad-hoc | CRT calibration benchmark |
| **OpenFDA** | Drug approvals, recalls, labels | Public | Weekly | Regulatory enrichment |
| **EMA** | European drug approvals (CHMP decisions) | Public | Monthly | Regulatory tracking |
| **xAI (Grok)** | LLM biotech news analysis, X Search | API key (`XAI_API_KEY`) | Ad-hoc | Event alerts, news triage |

### Production Data Files

Core inputs loaded by `run_screen.py` every run:

| File | Contents | Records | Refresh | Module |
|------|----------|---------|---------|--------|
| `production_data/universe.json` | Tracked biotech universe | 341 tickers | Manual (universe changes) | M1 |
| `production_data/price_history.csv` | Daily OHLCV prices | ~1,571 dates × 355 tickers | Daily | M2 (drawdown, beta, RSI), benchmarks |
| `production_data/market_data.json` | Market caps, volume, sector | 340 tickers | Daily | M1 filters, M2 sizing |
| `production_data/financial_records.json` | Balance sheet, income, cash flow | 340 tickers | Every 2-3 days | M2 (current-state fallback) |
| `production_data/pit_financials/{TICKER}.json` | EDGAR XBRL facts with filing dates | 339 tickers, all historical filings | Daily rebuild | M2 (PIT mode) |
| `production_data/ipo_dates.json` | First/last price dates per ticker | 355 tickers | From price_history.csv | PIT survivorship filter |
| `production_data/institutional_summary.json` | Sponsorship holdings, delta signals | 29 managers, 58.2% coverage | Quarterly (~May 15 next) | momentum_delta_z sort signal |
| `production_data/manager_registry.json` | Sponsorship manager metadata | ~100 managers | Quarterly | Sponsorship signal processing |
| `production_data/fda_designations.json` | Fast Track, Breakthrough, Orphan, Priority | 207 entries, 84 tickers | Manual | M4 regulatory scoring |
| `production_data/regulatory_calendar_manual.json` | Hand-curated PDUFA/ADCOM dates | 20-50 events | Manual | M3 catalyst detection |
| `production_data/adcom_outcomes.json` | FDA advisory committee voting history | 100+ decisions | Ad-hoc | M4 adcom vote scoring |
| `production_data/purple_book.json` | Biologics competition/exclusivity | 2,013 products, 49 tickers | Manual | Commercial-stage context |
| `production_data/portfolio_policy.json` | Construction rules (v3) | — | Manual | Portfolio construction |
| `production_data/decision_rulesets/v1.12.0_*.json` | Active decision engine config | — | Governed promotion | Decision engine |

### Cache Layer

Date-stamped caches for PIT-safe historical reruns:

| Cache | Contents | Path Pattern | Refresh | Size |
|-------|----------|-------------|---------|------|
| CTgov trial records | PIT-filtered clinical trials | `cache/ctgov/trial_records_{date}.json` | Daily | ~15 MB each |
| SEC 8-K catalysts | Corporate event filings | `cache/sec/8k_catalysts_{date}*.json` | Daily | ~2 MB each |
| FDA ADCOM calendar | Advisory committee schedule | `cache/fda/adcom_calendar_{date}.json` | Monthly | ~1 MB |
| Morningstar data | Fundamentals, vol, ratings | `cache/morningstar_data/` | Daily | ~40 MB total |
| Clinical features | Pre-computed M4 features | `cache/clinical/clinical_features_{date}.json` | Daily | ~5 MB each |
| Press releases | Company PR text + classification | `cache/press/` | Daily | ~50 MB total |
| Market data | Price/volume warm cache | `cache/market_data/` | Daily | ~30 MB total |

### Supplementary Data Sources

| Source | Records | Tickers Linked | Path | Status |
|--------|---------|---------------|------|--------|
| AACT clinical trials | 579,828 trials | 22,082 linked (303 universe tickers via 549 sponsor aliases) | `data/aact/snapshots/` | Live — weekly ingest (Mondays) |
| Purple Book biologics | 2,013 products | 530 products → 49 unique tickers (41 in universe) | `production_data/purple_book.json` | Live |
| Herald press releases | 4,380+ classified | 338 tickers | `data/press_releases/` | Live — daily collection |
| Short interest (FINRA) | 300+ tickers | 300+ | `data/short_interest.json` | Weekly |
| BioTradingArena benchmark | 655 validated catalyst cases | 212 tickers (130 overlap with universe) | `production_data/biotradingarena_benchmark.json` | Live — API fetch (2026-04-15) |
| DealForma deal comps | — | — | Spec 046 ready | Awaiting CSV export |
| Conference programs | 8 conferences (ASCO, AACR, ESMO, ASH, AAN, SABCS, SITC, ACR) | 5 AACR 2026 abstracts (first run) | `cache/conferences/` | Daily via Grok web search (6 AM ET) |
| EU trial registries | EUCTR, CTIS, ISRCTN | — | `cache/ema/` | Monthly |
| PubMed (NCBI E-utilities) | ~19K trial records searchable | Per-ticker drug/NCT search | `data/cache/pubmed/` | Optional — `--enrich-pubmed` flag |

### Event Evidence Snapshot (2026-04-15)

PIT-anchored evidence artifact per `(node_id, as_of_date)` materializing trial-design,
regulatory-designation, CRT-history, and PubMed literature data into a single frozen record.

**Schema** (`EventEvidenceSnapshot`, frozen dataclass):

| Field | Type | Source | Notes |
|-------|------|--------|-------|
| `phase` | str | CatalystNode / trial_records | Normalized ("1", "2", "3", etc.) |
| `randomized_flag` | bool? | trial_records.allocation | RANDOMIZED → True |
| `blinded_flag` | bool? | trial_records.masking | DOUBLE/SINGLE → True |
| `control_arm_flag` | bool? | trial_records.intervention_model | Not SINGLE_GROUP → True |
| `enrollment_n` | int? | trial_records.enrollment | Raw enrollment count |
| `primary_endpoint_text` | str? | trial_records.primary_endpoints[0] | Capped 500 chars |
| `endpoint_type` | str? | Inferred from endpoint text | EFFICACY/SAFETY/SURVIVAL/BIOMARKER/COMPOSITE |
| `prior_positive_readouts_n` | int? | CRT resolutions (PIT-filtered) | HIT count for ticker |
| `prior_negative_readouts_n` | int? | CRT resolutions (PIT-filtered) | MISS count for ticker |
| `orphan_flag` | bool? | CatalystNode.designations | "ODD" in list |
| `fast_track_flag` | bool? | CatalystNode.designations | "FT" in list |
| `breakthrough_flag` | bool? | CatalystNode.designations | "BTD" in list |
| `adcom_flag` | bool? | CatalystNode.adcom_outcome | Non-null → True |
| `safety_signal_flag` | bool? | Not yet wired | Always null in v1 |
| `literature_support_score` | float? | PubMed via NCBI E-utilities | [0, 1], null when --enrich-pubmed off |
| `evidence_confidence` | float? | Weighted source coverage | [0, 1]: trial 0.35 + nct 0.10 + desig 0.20 + CRT 0.20 + lit 0.15 |
| `ctgov_study_id` | str? | CatalystNode.nct_id / trial match | NCT ID |

**PubMed integration** (`data_sources/pubmed_client.py`):
- API: NCBI E-utilities (esearch + efetch), XML responses parsed to `PubMedArticle` objects
- Search strategy: (1) NCT ID exact match, (2) drug name + indication from `drug_name_map.json` (300 tickers)
- `literature_support_score` = 0.30×count + 0.25×recency + 0.25×journal_quality + 0.20×has_results
- Journal tiers: high-impact (NEJM, Lancet, JAMA, Nature, Cell, etc.), mid-impact (Clin Cancer Res, BMJ, etc.)
- Cache: `data/cache/pubmed/{sha1}.json`, 24h TTL
- Rate limit: 10 req/s with `NCBI_API_KEY`, 3 req/s anonymous
- Flags: `--enrich-pubmed` on `run_daily_production.py`, `--pubmed` on `build_event_ev_scores.py`
- Fallback: API failure → score stays null, evidence snapshot still built, no production blocking

**Model consumption**: `literature_support_score` feeds into the outcome model (`outcome_model.py`)
as a Bayesian likelihood update: `log_odds += (score - 0.3) * 0.3` (±0.15 max, centered at 0.3).
Higher publication volume in high-impact journals modestly increases P(HIT). Fires only when
score > 0; null/zero = no effect. Appears in `features_used.log_odds_updates.literature_support`.

**Validation**: `build_ev_validation.py` carries `literature_support_score` and `evidence_confidence`
into the validation ledger and computes a `by_literature` calibration split (Brier/hit-rate for
events with vs without literature). Promotion to binding requires forward evidence from this split.

### Phase 2 Prior Recalibration (2026-04-16)

HINT benchmark revealed our Phase 2 readout prior was materially miscalibrated:

| Metric | Old (0.310) | New (0.420) | HINT empirical |
|--------|-------------|-------------|----------------|
| Phase 2 prior | 0.310 | **0.420** | 0.492 |
| HINT Brier | — | — | 0.250 |
| Our Brier (matched) | 0.336 | ~0.26 (est.) | — |

**Ablation results** (418 events, 66 Phase 2):
- Actionable count: 30 → 30 (no change)
- Top-10 overlap: 7/10 (stable)
- Top-20 overlap: 16/20 (stable)
- Phase 2 mean p_hit: +0.098 (correct direction)
- Phase 2 mean EV: +9.4pp
- Phase 1, Phase 3: completely unaffected

**Decision:** Adopted 0.420 (conservative halfway between old 0.310 and HINT 0.492).
Aggressive 0.492 tested but deferred — same top-10/20 overlap but larger rank swings
in mid-table Phase 2 names. Old value preserved as `_PHASE_2_PRIOR_OLD = 0.310`.

Implementation: `event_ev/outcome_model.py:LITERATURE_PHASE_READOUT_PRIORS["2"]`.
Ablation: `research/phase2_recalibration_ablation.py`.

**Leaderboard surfacing**: `literature_support_score`, `evidence_confidence`, `randomized_flag`,
`blinded_flag`, `enrollment_n`, `endpoint_type`, `orphan_flag`, `breakthrough_flag`, `ctgov_study_id`
appear in EV leaderboard JSON. Full evidence snapshot serialized in `{date}_event_ev_full.json`.

Implementation: `event_ev/evidence_snapshot.py` (builder + PubMed enricher),
`data_sources/pubmed_client.py` (client), `event_ev/loaders.py:load_evidence_snapshots()`,
`production_data/drug_name_map.json` (300-ticker curated drug name lookup).

### Clinical Stack v2 (2026-04-16)

Four clinical layers integrated into the EV ranking path via a bounded transmission mechanism:

**1. Protocol quality** (`common/protocol_quality.py`):
Phase-conditional trial design rigor from structured ClinicalTrials.gov fields.
Features: comparator, randomization, blinding, endpoint specificity, multi-arm, complexity penalty.
Phase 1 weights lower than Phase 3. Soft floor prevents early-phase collapse.
Weight in CalendarAlpha: `w_protocol=0.08`.

**2. Conditional biomarker** (`common/biomarker_context.py`):
Context-dependent biomarker score replacing the old flat 1.20x boost (neutralized to 1.00 in PoSPriorEngine).
Conditional on phase × indication × protocol quality × endpoint specificity.
Score range [-0.05, 0.30]. Oncology + strong design = max; weak design capped at 0.18.
86/297 tickers have biomarker-selected trials.

**3. Endpoint quality v2** (`common/endpoint_quality.py`):
7-bucket endpoint classification (hard_clinical → validated_surrogate → objective_response →
symptom_functional → safety_tolerability → pk_pd_exploratory → vague_other).
Phase-aware multipliers: Ph3 OS = max, Ph3 safety-only = red flag, Ph1 safety = neutral.
Weight in CalendarAlpha: `w_endpoint_v2=0.08`.

**4. Clinical-to-p_hit transmission** (`event_ev/outcome_model.py`):
Bridges all three scores into the outcome model's logit-space update path.
Conservative weights: endpoint 0.08, protocol 0.06, biomarker 0.04.
Phase-aware caps: Ph1 ±0.12, Ph2 ±0.25, Ph3 ±0.20 log-odds.
Neutral anchors at population means (protocol 0.50, endpoint 0.55).
Sign-symmetric: strong setups boost p_hit, weak setups penalize.

**Status: behind flag (shadow validation through 2026-04-30).**
PIT-honest backtest (Apr 5-15): Brier 0.041→0.039, returns +5.37% (identical),
6 dropped names (safety-only endpoints), 0 gained. Validated filter, not yet proven alpha.

Forward validation: `tools/clinical_transmission_shadow.py` (daily, Step 5k.21b).
Outcome evaluation: `research/clinical_tx_outcome_eval.py`.
HINT benchmark: `research/hint_benchmark.py` (17,614 trials, 1,610 matched).

### HINT Research Integration (2026-04-16)

External benchmark from github.com/futianfan/clinical-trial-outcome-prediction.
Non-commercial research use only. Located in `research/` module (not production code).

Key findings:
- Phase 1: PoS v3 wins (Brier 0.218 vs 0.244)
- Phase 2: HINT wins (Brier 0.250 vs 0.336) → recalibrated prior
- Phase 3: Comparable (0.217 vs 0.219)
- Biomarker selection NOT globally positive (Δ=-2.7%)

Protocol feature extraction: 10 PIT-safe features from eligibility text.
Schema mapper: `research/hint_adapter.py`. Benchmark: `research/hint_benchmark.py`.

### PIT (Point-in-Time) Data Architecture

Historical backtests require data as-known-on each snapshot date. The PIT stack:

```
Current-state files          PIT-corrected path
─────────────────           ──────────────────
financial_records.json  ──►  pit_financials/{TICKER}.json (filed <= as_of_date)
universe.json           ──►  ipo_dates.json filter (first_price_date <= as_of_date)
trial_records.json      ──►  cache/ctgov/trial_records_{date}.json + posting filter
catalyst_events.json    ──►  CTgov fallback PIT safety net (posting_date <= as_of_date)
```

| PIT Component | Status | Notes |
|---------------|--------|-------|
| Survivorship filter (ipo_dates.json) | **Shipped** | 8,556 violations fixed |
| EDGAR PIT financials (filing-date gated) | **Shipped** | 339 tickers, all historical filings |
| CTgov PIT safety net | **Shipped** | Runtime filter on posting dates |
| Production data archiver | **Shipped** | SHA-256 manifests in `data/pit_archives/` |
| PIT v2 snapshot regeneration | **In progress** | 76 monthly dates via `regenerate_pit_v2_snapshots.py` |
| Catalyst look-ahead audit | Inconclusive | Retroactive generation makes this hard to clean |

### Data Refresh Pipeline

Daily production run (`tools/run_daily_production.py`, cron 5:30 PM ET):

```
Step 1: Archive production inputs (SHA-256 manifest)
Step 2: Refresh prices (Yahoo/Morningstar → price_history.csv)
Step 3: Refresh market data (→ market_data.json)
Step 4: Poll CTgov (→ cache/ctgov/trial_records_{date}.json)
Step 5: Run full screen (run_screen.py → data/snapshots/{date}/)
        ├── 5a-5d: Modules 1-4
        ├── 5e: Module 5 composite
        ├── 5f: Decision engine
        ├── 5g: Sponsorship momentum
        ├── 5h: Options diagnostics (Tastytrade)
        ├── 5i: Event premium decomposition
        ├── 5j: AACT delta pipeline (weekly — Mondays only)
        ├── 5k: Construction overlays
        │   └── 5k.21: Event EV scoring (evidence snapshots + optional PubMed via --enrich-pubmed)
        ├── 5l: Shadow portfolio
        ├── 5m: CRT pipeline
        └── 5o: Construction v2 shadow
Step 6: Gate validation (29 production checks)
Step 7: Agent fleet dispatch (ops → sentinel → qa → calibration)
```

### Data Quality Summary

| Dimension | Coverage | Notes |
|-----------|----------|-------|
| Tier/momentum/archetype | 100% | Core fields always populated |
| Catalyst fields | ~85% | Some names lack dated catalysts |
| Options (ATM IV) | 96% eligible | Liquidity (~42% liquid chains) is the real gate |
| RR 25d / implied move | ~39% | Gated by options chain liquidity |
| Sponsorship data | 58.2% ticker coverage | Next refresh ~May 15 (Q1 2026 filings) |
| FDA designations | 58.3% top-60 | 207 entries, 84 tickers |
| PIT financials | 99.4% universe | 339/341 tickers with EDGAR facts |
| AACT trial linkage | 303 tickers (549 sponsor aliases) | 22,082 linked trials |
| Drug name map | 300 tickers | From trial_records + PDUFA; `production_data/drug_name_map.json` |
| PubMed literature | Optional (--enrich-pubmed) | Cached per-ticker; 24h TTL in `data/cache/pubmed/` |

### Environment Variables

| Variable | Service | Purpose |
|----------|---------|---------|
| `TT_SECRET`, `TT_REFRESH` | Tastytrade | Options surface data (OAuth2) |
| `MASSIVE_API_KEY`, `MASSIVE_S3_*` | Massive | Historical options chains |
| `MD_AUTH_TOKEN` | Morningstar Direct | Fundamentals, vol, ratings |
| `FRED_API_KEY` | FRED (St. Louis Fed) | Macro regime feeds (VIX, rates) |
| `XAI_API_KEY` | xAI (Grok) | News analysis, X Search |
| `NCBI_API_KEY` | NCBI (PubMed) | Literature enrichment (10 req/s vs 3 anonymous) |
| `SMTP_USER`, `SMTP_PASSWORD` | Email | Alert delivery |

---

## 5. Portfolio Construction

### Standing Allocation Policy (2026-04-17)

Risk management lives at the **allocation layer above the model**, not inside DEM construction.
The DEM signal stays pure inside the portfolio; deployment risk is controlled by sleeve sizing.

| Role | Allocation | Core | Purpose |
|------|-----------|------|---------|
| Research / shadow | 100% DEM | — | Pure signal benchmark |
| **Initial production** | **30% DEM / 70% XBI** | XBI | Conservative start (live t=1.13) |
| Scaled production | 60% DEM / 40% XBI | XBI | Live Sharpe maximum (1.37) |

**Default core:** XBI (better drawdown than EW-All at every allocation level).
**Promotion from 30/70 to 60/40:** requires live net excess positive, Sharpe acceptable,
ex-tail robustness, no recurring failure mode. Do NOT promote on pseudo-PIT alone.

Always report three series: 100% DEM, 30/70 DEM/XBI, 60/40 DEM/XBI.
Default deployable headline = 30/70 DEM/XBI.

Evidence: capital allocation sweep (2026-04-17) showed alpha scales linearly with DEM weight,
no knee point, position caps are no-ops in Top-30 EW, internal overlay blending dilutes alpha
without meaningful drawdown improvement. See `research/capital_allocation_sweep.py`.

### Internal Construction (Spec 050, adopted 2026-04-03)

**Model:** B6 selector + pairwise_minimal ranker (ordinal-only)
**Construction:** Equal-weight Top-30
**Account:** $50,000 notional
**Rebalance:** Weekly (Friday)
**Cost budget:** 25 bps round-trip per turnover event

| Parameter | Value | Evidence |
|-----------|-------|----------|
| K (portfolio size) | **30** | K-sweep peak: +2.34pp net, t=2.60, stable K=25-35 |
| Weighting | **Equal-weight** | RW-EW = -0.09pp, t=-0.95 — do not rank-weight |
| Turnover | **~22%** monthly | Lower than old baseline (29%) |
| Rebalance buffer | 30 ranks | Existing, reduces churn |

**Sleeve budgets are RETIRED.** The fixed 55/25/10/10 allocation was the primary
construction damage mechanism (+153.6pp drag). Bucket labels survive as metadata only.

### Construction Diagnosis (2026-04-01, updated 2026-04-03)

**Original finding (2026-04-01):** The selection layer generates alpha but fixed sleeve
budgets destroy it. This remains true for the old optionality selector.

**Updated finding (2026-04-03):** The A4 sponsorship selector generates statistically
significant alpha on true PIT data, and EW Top-30 construction preserves it.

#### Selection-Only Benchmark (EW Top-20, PIT, 2020-2026)

| Metric | Value |
|--------|-------|
| Cumulative return | +151.1% |
| Cumulative excess vs XBI | **+95.2%** |
| Win rate (daily excess > 0) | 53.1% |
| Information ratio | 1.41 |
| Positive excess years | 6 of 7 |

#### Drag Decomposition

| Construction Layer | Excess vs XBI | Drag from Prior |
|-------------------|---------------|-----------------|
| EW Top-20 (pure selection) | +95.2% | — |
| EW Bucketed (sleeves, EW within) | +19.3% | -75.95% (50% of total drag) |
| Policy-Weighted (55/25/10/10) | -28.4% | -47.62% (31%) |
| Full Shadow (all rules) | -58.4% | -30.01% (19%) |
| **Total construction drag** | | **+153.6%** |

**Root cause:** Fixed sleeve budget allocation, not per-name caps or other rules.
Sleeve labels as metadata are harmless (loose sleeves = EW). The damage comes
from forcing capital into the 91-180d bucket at 55%.

#### Regime Asymmetry

| Regime | EW Top-20 Excess | IR |
|--------|-----------------|-----|
| Bear XBI (daily return ≤ 0) | **+102.8%** | **3.35** |
| Bull XBI (daily return > 0) | -7.6% | -0.21 |

The selector's edge is a **bear-market phenomenon**. Optionality-anchored names
hold value during selloffs because their catalyst-driven upside is less correlated
with sector beta. The model is a **downside-protection engine**, not an all-weather
momentum strategy.

#### Construction v2 Candidates

| Candidate | Full-History IR | 2024-2026 IR | 2026 YTD IR | Mean Turnover |
|-----------|----------------|-------------|-------------|---------------|
| **EW Top-30** | **1.51** | **2.70** | **2.64** | **12.8%** |
| EW Top-20 | 1.41 | 1.99 | 1.37 | 20.6% |
| Rank-Weighted Top-20 | 0.86 | 1.06 | 1.08 | 20.6% |

**EW Top-30 is the leading candidate** for construction v2: higher IR, lower
turnover, strong across all recent windows.

#### Transaction Cost Analysis

Estimated cost drag by candidate across full history (390 periods), assuming
conservative round-trip costs for small/mid-cap biotech:

| Candidate | Gross Excess | Drag (30 bps) | Drag (50 bps) | Drag (80 bps) | Net Excess (50 bps) |
|-----------|-------------|--------------|--------------|--------------|-------------------|
| **EW Top-30** | +95.8% | -6.0% | **-10.0%** | -16.0% | **+85.8%** |
| EW Top-20 | +95.2% | -9.6% | -16.0% | -25.7% | +79.2% |
| Rank-Weighted Top-20 | +59.8% | -9.6% | -16.0% | -25.7% | +43.8% |

**EW Top-30 wins net of costs.** Its lower turnover (12.8% vs 20.6%) saves ~6%
over full history at 50 bps. The advantage *widens* after transaction costs.

Cost drag is real (~10-16% over 6 years) but small relative to the 153.6%
construction drag from sleeve budgets. Transaction costs are not the primary
problem — fixed budget allocation is.

Rebalance cost model: `common/rebalance_cost_model.py` (17 tests).
Components: spread estimation by market-cap bucket, impact by ADV, portfolio
cost aggregation, rebalance threshold gate (only trade if expected alpha >
2× estimated cost).

#### Standing Benchmarks (established 2026-04-01)

| Benchmark | Role | Status |
|-----------|------|--------|
| **EW Top-30** | New default control | Active |
| Rank-Weighted Top-30 | Shadow overlay (regime-dependent, not always-on) | Shadow |
| Current shadow (sleeve-budget) | Legacy comparator to beat | Legacy |

#### Operating Conclusion (updated 2026-05-24)

> The DEM uses a two-stage scoring architecture: **coinvest selects** (v1.14.0
> coinvest-only selector), the capped **2-feature pairwise_minimal ranker** orders
> within the top-60, **financial_score penalizes** safe-but-uncatalytic names, and
> the portfolio is held equal-weight. The prior B6 selector remains historical
> evidence; `inst_delta_z` is now diagnostic-only. The pairwise ranker is
> ordinal-only (ECE = 0.129) — no rank-weighting or sizing.

> The model is a **bear/neutral alpha engine**: strong in distress and consolidation
> (+3.37pp bear, +6.23pp neutral), with bounded underperformance in sharp biotech
> rallies (-0.37pp bull). This is structural — sponsorship is a quality
> signal, and quality lags beta in risk-on environments.

> Fixed sleeve budgeting has been retired. Rank-weighting is not justified.
> The correct construction is EW Top-30.

#### Construction v2 Shadow (live since 2026-04-01)

Construction v2 runs daily as Step 5o in the production pipeline, tracking two
variants alongside the legacy shadow:

| Variant | Rule | Status |
|---------|------|--------|
| **EW Top-30** | Equal-weight top-30 by DEM rank | **New default control** |
| **Regime-Conditioned** | Bear: top-20, Bull: top-30 (XBI 20d + hysteresis) | Shadow overlay |
| Legacy shadow | Full sleeve-budget construction | Legacy comparator |

**Regime classifier:** XBI 20-day return, bear < -2%, bull > +2%, min 5-day duration.

**Backfill (March 1 - April 1, 17 periods):**
- EW Top-30: -6.97%, +2.15% excess vs XBI
- Legacy shadow: -5.54%, -4.86% excess vs XBI

**Do not carry forward:** fixed sleeve budgets, dynamic caps, always-on rank-weighting.

#### Options Signal Work (built 2026-04-01)

**Event premium decomposition** (`common/event_premium_decomp.py`, 26 tests):
Decomposes the options surface into 8 within-top-30 ranking features:
`epd_event_premium_ratio`, `epd_term_slope_z`, `epd_skew_richness_z`,
`epd_iv_momentum`, `epd_implied_vs_realized_ratio`, `epd_iv_per_catalyst_day`,
`epd_surface_regime`, `epd_quality`. Runs daily as Step 5l.4b. 28/30 top names
at full quality using 254K rows of historical IV features.

**Options cohort diagnostics** (`scripts/research/options_cohort_diagnostics.py`):
Cuts by hard/soft, regulatory/clinical, near/mid/far, liquid/thin, surface type.
Key finding: eligible-name options coverage is 96% (not 65.6%), liquidity (42%)
is the real gate. Highest-dispersion cohort: hard catalyst + regulatory + liquid.

**CRT × options join** (`scripts/research/build_crt_options_join.py`):
27 resolutions joined with prediction-time options state. Foundation for the
catalyst EV model. Realized-return backfill needed.

**Options EV pilot** (`scripts/research/options_ev_pilot.py`):
First directional results on 905 observations (5 dates, h5 horizon):
- EPR signal is inverted across full universe (high EPR = overpriced) but
  **positive inside top-30** (event-loaded +4.4% vs flat +2.9%)
- IV regime: NORMAL +1.3%, ELEVATED +0.7%, EXTREME -1.1%
- Hard catalysts outperform soft by +59bps
- Caveat: 5 dates only, directional not conclusive

**Next:** h20/h63 returns mature mid-April; rerun EV pilot and ranker with
options-populated window.

#### Next Construction Experiments

1. **Correlation/concentration penalty** — lightweight risk overlay on EW Top-30
2. **Regime classifier stability testing** — monitor flip frequency, turnover on transitions
3. All candidates must survive transaction-cost gate before promotion

---

## 6. Expectation Error Model (EES v2 / Trap Gate + v3 Conditional)

A pre-trade risk control layer that identifies structural expectation errors in biotech event pricing. The core insight: names that appear "cheap" relative to base-rate event outcomes are behavioral traps — the market is right to price them above historical norms.

**Status:** FROZEN (2026-04-12). Checklist v2: **5/5 PASS**. Production gates deployed.

### Architecture

Two independent binary gates applied before portfolio construction:

| Gate | Formula | Threshold | IC (63d) | Role |
|------|---------|-----------|----------|------|
| **Trap T20** | `-(0.50 × base_rate_gap + 0.50 × conditional_misprice)` | Exclude bottom 20% | +0.084 (t=9.9) | Remove structural traps |
| **Quality Q15** | `-1.0 × timing_decay_risk` | Exclude bottom 15% | +0.078 (t=18.1) | Remove timing-uncertain structures |

Final eligibility: `ees_eligible = quality_gate AND trap_gate`. Observed: ~80% of universe passes (60 names excluded by trap, 0 by quality alone).

### Sub-Scores

Six normalized inputs (all ≈[-1, +1]):

| Score | Input | What It Detects |
|-------|-------|-----------------|
| **base_rate_gap** | `(implied_move - historical_p50) / historical_iqr` | Market price deviates from category base rate |
| **conditional_misprice** | `(expected_move - implied_move) / (\|implied_move\| + ε)` | Implied move ≠ scenario-weighted expected move |
| **timing_decay_risk** | Event date precision (DAY=0, QUARTER=1) | Expensive priced move with uncertain timing |
| **divergence** | `(implied_event_move - realized_vol) / max(both)` | Options mispricing vs historical vol |
| **crowding_bias** | Normalized short interest percentile | Bearish consensus |
| **slippage_penalty** | **Always 0** — removed (PIT look-ahead bias found 2026-04-12) | — |

Base-rate table (absolute % moves by category):

| Category | p50 | IQR |
|----------|-----|-----|
| CLINICAL \| phase3 | 35.0% | 37.0% |
| CLINICAL \| phase2 | 35.0% | 40.0% |
| CLINICAL \| early | 25.0% | 38.0% |
| REGULATORY \| any | 19.0–23.5% | 27–30% |
| SAFETY \| any | 20.0% | 35.0% |

### Conviction Sizing (α=1.5)

After gate filtering, survivors are sized by conviction:

```
weight_i ∝ (B6_percentile ^ 1.5) × trap_strength × liquidity_cap
```

- **B6 conviction**: Power-law concentration from selector rank
- **Trap scaling**: Reduce weight for names barely passing trap gate
- **Liquidity cap**: `≤ 2% of 20d avg dollar volume / NAV` (sizing only, not a filter)
- **Single-name cap**: 10% max
- **Dust filter**: Drop positions < 0.5%

No liquidity filters — illiquidity is where the edge lives in biotech.

### Execution Guardrails

Applied post-sizing, protective only:

| Threshold | Value | Action |
|-----------|-------|--------|
| Participation > 5% ADV | Scale down to 5% | 1 name affected at $5M NAV |
| Participation > 20% ADV | Skip entirely | 0 names affected at $5M NAV |

Capacity: $50M+ viable. At $5M NAV, only 1 trade exceeds 5% ADV.

### Checklist v2 Results

| Test | Result |
|------|--------|
| Bootstrap CI | [+0.24%, +1.17%], excludes zero |
| LOSO | 6/9 periods pass (67%) |
| FDR | p < 0.000001 (BH q=0.10) |
| FM incremental vs B6 | β=+0.036 (t=6.46) at 20d, +0.064 (t=13.59) at 63d |
| Dependence-adjusted t | t=6.69 (ρ=0.24, n_eff=159) |

### Backtested Performance (PIT-safe, 2020–2026)

| Metric | EW Top-30 (baseline) | + Trap T20 | + Trap + Timing |
|--------|----------------------|------------|-----------------|
| Sharpe | 0.211 | 0.221 | **0.414** (2.4×) |
| Mean return | +1.98%/mo | +2.34%/mo | +4.04%/mo |
| Hit rate | 59% | 68% | — |

Trap-removed names return -0.12% vs kept +2.21% (drag = -2.33% at 20d). Rolling IC: 100% positive across 152 windows.

### Daily Monitoring

`tools/trapops_monitor.py` runs at 6:17 AM ET weekdays:
- **Module A**: Selection diff (top-30 overlap, trap-removed names)
- **Module B**: Execution stress (participation rates, scaled/skipped)
- **Module C**: Trap attribution (realized returns by gate bucket)
- **Module D**: Health alerts (GREEN/YELLOW/RED)

Regime detector auto-switches to conservative mode (Q20/T30) when: correlation > 0.40, eligible% < 50%, trap rate > 35%, or output-side underperformance.

### EES v3 — Conditional Mispricing Model (diagnostic overlay)

**Status:** Production-emitting since 2026-04-14. Checklist v2: **4/5 PASS** (WS4 pending forward data). Does NOT affect ranking or selection yet.

A two-factor model replacing the v2 trap/quality overlay with PIT-validated signals:

```
ees_v3_score = 0.70 × z(conditional_misprice_score)
             + 0.30 × z(conditional_expected_move)
```

| Factor | IC | NW-t | Decile Spread | Role |
|--------|-----|------|--------------|------|
| conditional_misprice_score | +0.089 | 2.07 | +6.9pp | Primary alpha — scenario-EV vs market price |
| conditional_expected_move | +0.026 | 1.83 | +4.2pp | Stability overlay — orthogonal (r < 0.15) |

**Key corrections from v2:**
- `base_rate_gap_score` is **anti-predictive** (IC -0.090). Market is right to price above base rates. Removed from alpha.
- `trap_overlay_score` is dead (IC ~0) — base_rate and misprice cancel each other.
- Unit fix: `priced_move_pct` converted ×100 in Phase 2z (was decimal, now percentage points) with sanity check warning if values outside [1, 500].
- Soft scaling: `tanh(raw / 2)` replaces hard [-1, +1] clamp — preserves tail ordering.
- Distribution diagnostics: every run logs unique count and ceiling %, warns if >20% saturated.

**Conditional model** (`event_ev/conditional_model.py`): detects mispriced conditionals where market prices a generic event outcome but subgroup odds differ (biomarker-selected trials, enriched designs, validated mechanisms, platform track record). Outputs `conditional_expected_move` and `conditional_gap_score`.

**Execution capacity** (`event_ev/execution_capacity.py`): post-sizing guardrail. Checklist: 6/6 PASS at $3M NAV. Participation limits: scale down >5% ADV, skip >20% ADV.

### Forward Monitor

`tools/ees_v3_forward_monitor.py` tracks rolling evidence toward WS4 clearance (dependence-adjusted t ≥ 1.65). Re-scores historical snapshots on-the-fly for pre-v3 dates, reads native v3 columns from 2026-04-14+.

Current forward state (re-scored, 433 snapshots):

| Signal | Mean IC | rho1 | n_eff | t_adj | Status |
|--------|---------|------|-------|-------|--------|
| conditional_expected_move | +0.025 | 0.75 | 54 | +0.99 | Leading candidate — only positive signal |
| ees_v3_score | -0.023 | 0.62 | 93 | -1.23 | Negative (misprice saturation in pre-v3 data) |
| conditional_misprice_score | -0.077 | 0.34 | 32 | -4.12 | Contaminated by pre-fix unit mismatch |

Native v3 snapshots begin 2026-04-14. Clean forward evidence requires ~21 trading days (h20 returns). WS4 clearance expected after accumulation in production archives.

### Runway-to-Catalyst Severity (risk-control overlay)

**Status:** Production-emitting since 2026-04-15. **Risk-control overlay** — not alpha.
Does not affect ranking or selection. Its job is to stop the portfolio from drifting
into fragile, dilution-prone micro-cap lottery exposure.

**Not alpha. Variance control.** Backfill validation (75,567 observations, 377 dates,
228 tickers) shows severity IC is +0.017 — the *wrong sign* for a return predictor.
Higher severity names have higher raw returns (micro-cap lottery premium) but 4x the
volatility (111% vs 28%) and +32 skewness. Risk-adjusted return is identical between
gate pass and fail (Sharpe 0.068 vs 0.067). The feature does not predict returns; it
controls the *kind* of returns you hold.

One feature computed once, consumed across four layers via **dual-severity paths** (v1.1):

- **Truth severity**: "can they survive to the catalyst?" Uses T1/T2 decisive catalysts only.
- **EV severity**: "what financing damage even if they do?" Uses actual catalyst timing for any tier.

**Formula:**
```
runway_buffer = months_to_cash_out - months_to_decisive_catalyst
severity = sigmoid(-(buffer - 3) / 2) + financing_adjustment + market_adjustment
```

Catalyst decisiveness tiers: T1 (regulatory/PDUFA) = 1.0, T2 (pivotal Phase 3) = 0.85, T3 (conference) = 0.50, T4 (routine) = 0.20, T5 (unknown) = 0.10. T1/T2 decisive for truth gate; all tiers used by EV/sizing.

**Catalyst priority fix (2026-04-15):** `_find_nearest_catalyst_event` now has a Tier P priority override: T1/T2 events within 180 days always win over nearer T3 CT.gov milestones. PDUFA manual entries are checked alongside M3 events. T1 beats T2 at equal distance. Before fix: 3/14 PDUFAs correctly T1. After: 10/10 future PDUFAs correctly T1.

**Four consumption layers:**

| Layer | Severity Path | Effect | Threshold |
|-------|--------------|--------|-----------|
| Truth gate | Truth severity | Hard fail — name ineligible | truth_severity > 0.92 |
| EV layer | EV severity | dilution_haircut = 0.35 × ev_severity | Continuous |
| Portfolio sizing | EV severity | size_multiplier = 1 - 0.60 × ev_severity | Continuous |
| Crowd-belief | EV severity | Distortion input for expectation model | Continuous |

**Production (2026-04-15):** safe=149, moderate=94, elevated=18, critical=16, extreme=21. 113 unique severity values. 12 truth-gate failures (all micro-cap, all defensible).

**Backfill validation verdict:**

| Check | Obs | Finding |
|-------|-----|---------|
| Severity quintile monotonicity | 75,567 | NO — Q5 (highest severity) has best mean return (+4.20%) |
| Gate fail vs pass (risk-adjusted) | 75,567 | NEUTRAL — Sharpe 0.067 vs 0.068. Fails have 4x vol, +32 skew |
| Bucket monotonicity | 75,567 | NO — moderate Sharpe 0.117 (best), extreme 0.066 (lottery) |
| Severity-return IC | 377 dates | +0.017 (positive = wrong sign for alpha, 63% of dates) |
| Gate-fail audit | 12 fails | PASS — all micro-cap, all defensible |

**Policy:** Never promote into ranking. Keep as truth gate + EV haircut + sizing overlay.

Implementation: `event_ev/runway_severity.py` (v1.1). Backfill: `scripts/research/backfill_runway_severity.py`.

### Dead Lanes

| Feature | Why Dead |
|---------|----------|
| Slippage penalty (market_cap) | PIT look-ahead bias (IC +0.107 → -0.088 when corrected) |
| Original alpha direction (unflipped) | IC -0.006, t=-0.9 |
| Crowding bias (short interest) | IC ≈ 0 at all horizons |
| Trap as ranker (continuous) | Gates strictly dominate; trap is binary veto, not ranking signal |
| Liquidity filters | Would destroy alpha |
| base_rate_gap_score as alpha | Anti-predictive (IC -0.090); market is right |

---

## 7. Catalyst Resolution Tracker (CRT)

Prediction → resolution → calibration loop for hard catalysts.

**Status:** All 4 phases shipped (Spec 042)
**Resolutions:** 14 seeded (6 HIT, 8 MISS)
**Calibration:** Monotonic hit rate by DEM rank (100% → 67% → 33% → 0%)

### CRT Architecture

```
Watchlist Builder → Source Adapters (8-K, CTgov, PDUFA, Manual)
→ Outcome Classification → Price Direction → Resolution Record
→ Calibration Rollup → Governance Triggers
```

### Resolution Record Schema

- `ticker`, `catalyst_date`, `catalyst_type`, `prediction_dem_rank`, `prediction_tier`
- `outcome`: HIT / MISS / EXOGENOUS / INFORMATIONAL
- `price_direction`: up / down / flat
- `event_outcome`: scored against event result, not price

### RR Adjudication Policy

Score `event_outcome`, not `price_direction`. BIIB excluded as EXOGENOUS (M&A).
Current scorable: 1/3 (PVLA correct, CELC+TBPH wrong). Gate: BIIB PDUFA May 24.

### BioTradingArena External Benchmark (2026-04-15)

**Source:** [biotradingarena.com](https://www.biotradingarena.com) — open benchmark of 655 validated
biotech catalyst cases with de-identified press releases, CT.gov trial records, company
fundamentals, PubMed articles, price action (day before/of/after), and ground-truth
impact labels (7 categories: very_negative → very_positive).

**Dataset:** `production_data/biotradingarena_benchmark.json` (11.3 MB)
- 393 oncology + 262 non-oncology cases, 2015-01-09 to 2025-12-23
- 212 unique tickers, 130 overlap with our 342-ticker universe
- Event types: FDA approval (250), Phase 2/3 readouts (282), topline (123)
- API: REST with bearer auth, endpoints at `/api/benchmark/cases`

**Calibration script:** `scripts/research/crt_bta_calibration.py`

**Calibration results (2026-04-15):**

| Metric | Value |
|--------|-------|
| Overall predicted hit rate | 56.8% |
| BTA realized hit rate | 54.4% |
| Overall gap | -2.4pp (decent) |

Calibration by predicted quintile (non-neutral events only, n=226):

| Quintile | N | Predicted | Realized | Gap |
|----------|---:|---:|---:|---:|
| Q1 (lowest) | 45 | 34.3% | 55.6% | +21.2% |
| Q2 | 45 | 47.8% | 53.3% | +5.6% |
| Q3 | 45 | 58.0% | 48.9% | -9.1% |
| Q4 | 45 | 68.6% | 57.8% | -10.9% |
| Q5 (highest) | 46 | 75.1% | 56.5% | -18.6% |

**Critical findings:**
- **Quintile separation is flat** — model does not discriminate high vs low probability events well
- **FDA rejection blind spot**: predicted 72.1% vs realized 23.1% (-49pp). Model applies unconditional REGULATORY base rate to rejections. **Fix: add event direction as input.**
- **Biomarker selection uplift NOT confirmed**: selected 57.4% vs unselected 59.7%. Treat skeptically.
- **Mechanism class gradient confirmed**: semi_validated (65%) > validated (59%) > novel (54%) > unknown (47%). Keep.

**Implications for model:**
1. Event direction (approval vs rejection) is mandatory for regulatory PoS
2. Recalibrate confidence — model overconfident at top, underconfident at bottom
3. Downweight biomarker uplift assumption until reconfirmed
4. Mechanism class logic is validated externally — preserve

---

## 8. Shadow Portfolio Performance

### What is the Shadow Portfolio?

The shadow portfolio is a paper-traded simulation of the DEM's output. It holds real
positions at real prices but executes no trades in a live brokerage account. Its purpose
is to accumulate forward-only evidence of the model's selection and construction quality
before committing real capital. The shadow runs daily as part of the production pipeline
and is the **only credible evidence** of model performance — all historical backtests
are informative but secondary to forward shadow results.

Two shadow tracks run in parallel:

| Track | Construction | Start | Positions | Purpose |
|-------|-------------|-------|-----------|---------|
| **Legacy shadow** | Sleeve-budget (55/25/10/10 split) | 2026-01-02 | ~50 names | Historical comparator (being retired) |
| **EW Top-30 shadow** | Equal-weight top 30 by B6 rank | 2026-03-03 | 30 names | Production-track candidate |

### Legacy Shadow (sleeve-budget, 2026-01-02 to present)

$50,000 notional. 71 position snapshots across 73 periods.

| Metric | Value |
|--------|-------|
| Cumulative return | +24.72% |
| Cumulative excess vs XBI | **+12.63%** |
| Max drawdown | 11.05% |
| Sharpe | 0.92 |
| Win rate | 45% |
| Total P&L | +$12,360 |

Sleeve attribution (cumulative P&L):
- Binary 0-30d: +$7,904 (64% of total)
- Binary 31-90d: +$3,280
- Binary 91-180d: +$625
- Less Binary: +$551

### EW Top-30 Shadow (production-track, 2026-03-03 to 2026-04-01)

$50,000 notional. 17 trading days. Equal-weight, no sleeves.

| Metric | EW Top-30 | XBI | Excess |
|--------|-----------|-----|--------|
| Cumulative return | -7.3% | -9.1% | **+1.8%** |
| Mean daily excess | — | — | +0.13%/day |
| Win rate (daily excess > 0) | — | — | 41% |

The EW Top-30 shadow outperformed both XBI (+1.8% excess) and the legacy shadow
(legacy was -2.5% cumulative, -2.6% excess over the same period) despite holding
fewer names (30 vs 51) on a $50,000 account. This validates the construction
diagnosis: sleeve budgets were destroying alpha.

### Shadow Governance

- **Readiness verdict**: Generated weekly by `tools/weekly_readiness_scorecard.py`
- **Go/no-go gate**: Requires positive excess at 63d+ before live capital deployment
- **Monitoring**: `shadow_watch` agent (daily), `shadow_monitor` agent (via heartbeat checks)
- **Artifacts**: `artifacts/live_shadow/` (positions, trades, attribution, alerts, diagnostics)

---

## 9. Governance Stack

### Ruleset Governance

- Active ruleset pinned in `run_screen.py` with hash verification
- Promotion requires: evidence packet, replay comparison, canary regression
- Promotion bars: +0.20pp at longest horizon, guardrail -0.05pp
- Rollback: any candidate can be disabled via ruleset toggle

### Production Gates (29 checks)

| Category | Gates | Current Status |
|----------|-------|---------------|
| Core | ruleset, trading_day, inputs | PASS |
| Data freshness | XBI staleness, market data, CTgov, sponsorship | PASS |
| Schema | market_data, DE schema, sort contrib | PASS |
| Drift | top-20/60 overlap, Spearman, migrations | WARN (66.7% top-20) |
| Risk | concentration, exposure, portfolio weights | PASS |
| Forward eval | rolling IC | WARN (IC -0.021 < floor 0.02) |
| Canary | 3 historical dates | PASS (INFO) |

### Signal Governance

All new signals follow: research → evidence packet → shadow → governed promotion.
Minimum bars: IC > 0.03 at 60d (sort signals), +0.20pp at longest horizon (ranking).

### Backtest Protocol

All backtests must be PIT-safe: use only data that was available on each snapshot date.

#### Data Regimes

| Period | Regime | Catalyst % | priced_move % | Sponsorship % | Snapshots | Notes |
|--------|--------|-----------|--------------|--------------|-----------|-------|
| 2020 | `sparse` | 12% | 0% | 100% | 52 | Selector signals fully populated; catalyst/options data thin |
| 2021–2022 | `sparse` | 14–16% | 16% | 100% | ~104 | Options coverage begins; catalyst still limited to CT_PRIMARY_COMPLETION |
| 2023–2024 | `maturing` | 18–23% | 22% | 100% | ~104 | FDA_DECISION appears; priced_move growing but still minority |
| 2025 | `well-formed` | 42–71% | 28% | 100% | 58 | DATA_READOUT diversifies; full pipeline running |
| 2026-01+ | `production` | 87% | 83% | 100% | 98+ | is_hard_catalyst, full options surface, live snapshots |

**Backtest applicability by signal type:**
- **Selector (B6 bundle)**: Valid across full 2020–2026 range. Sponsorship and momentum_delta are 100% populated in all 464 snapshots.
- **Trap gate (EES v2)**: Backtested 2020–2026, IC positive in 100% of 152 rolling windows. Sub-scores degrade gracefully when priced_move is missing (base_rate_gap and conditional_misprice both zero out → trap score = 0 → gate passes).
- **Options/catalyst signals**: Only reliable from late 2025+. Pre-2025 coverage too sparse for standalone evaluation.
- **Conditional model**: Requires priced_move (83% in 2026, <25% before 2025). Forward-validation only until accumulation in production archives.

The prior label `catalyst_broken` for 2020–2024 was overly broad — selector and trap signals are valid for that period. The limitation is specific to catalyst-dependent and options-dependent features.

#### Options Feature Coverage (2026-04-14 production snapshot)

Three coverage tiers, gated by different data dependencies:

**Tier 1 — High coverage (94–96%):** Features derived from ATM IV surface (Tastytrade daily).

| Feature | Fill | Gate |
|---------|------|------|
| opt_atm_iv | 96% | Options chain exists |
| opt_iv_regime | 96% | Same |
| atm_iv_change_5d | 94% | 5d price history + chain |
| iv_ramp_flag | 94% | Same |

11 tickers lack any options chain (micro-caps: CNTX, IKT, ARTV, DRUG, BLTE, etc.).

**Tier 2 — Medium coverage (42–83%):** Features requiring catalyst date + options chain.

| Feature | Fill | Gate |
|---------|------|------|
| priced_move_pct | 83% | Needs catalyst_date + chain (39 tickers have chain but no catalyst) |
| straddle_price | 83% | Same as priced_move |
| implied_event_move | 61% | Needs priced_move + event model derivation (65 tickers fail derivation) |
| actual_implied_move_pctile | 57% | Needs implied_event_move + historical distribution |
| opt_put_call_skew | 50% | Needs liquid 25d strikes |
| opt_rr_25d | 46% | Same — only liquid chains |
| rr_25d_trend_7d | 42% | Needs 7d RR history + liquidity |

The 83% → 61% drop from priced_move to implied_event_move is a derivation gap: 65 tickers have straddle prices but the event move decomposition fails (typically missing event type context or insufficient term structure).

**Tier 3 — Low coverage (0–31%):** Features requiring deep chain liquidity or multiple surface inputs.

| Feature | Fill | Gate |
|---------|------|------|
| options_quality_composite | 31% | Needs RR + skew + IV regime + catalyst proximity |
| iv_crush_breakeven_pct | 28% | Needs term structure around event date |
| crush_adjusted_implied_move | 28% | Same |
| ovf_composite (expression overlay) | 28% | Needs ≥3 of 6 expression flags |
| ovf_has_iv_ramp | 14% | Needs 5d+ IV history near catalyst |
| ovf_has_event_premium | 8% | Needs term structure + event isolation |
| pre_event_put_call_ratio | 0% | Not yet populated (feed pending) |
| ranker_options_block | 0% | Not yet populated (ranker paused) |

**Impact on production signals:**
- **Trap gate**: Unaffected. When priced_move is missing, both base_rate_gap and conditional_misprice zero out. The gate passes these names through (safe default — no information = no trap signal = no exclusion). The 50 names without priced_move are scored by the selector only.
- **Expression overlay (Spec 062)**: Shadow-only, 28% coverage. Low coverage is expected — the overlay fires only when multiple surface signals converge. Not gated for production use.
- **Quality gate**: Unaffected. Depends on timing_decay_risk (catalyst precision), not options data.
- **Conviction sizing**: Uses trap_overlay_score (100% populated). Liquidity cap uses dollar volume, not options data.

#### Safety Requirements

1. **PIT enforcement**: All features must use `pit_financials/`, PIT-corrected catalyst dates, and snapshot-frozen inputs. No current-state files in historical evaluation.
2. **Look-ahead prohibition**: No future prices, future resolutions, or post-publication data in feature construction. The PIT infrastructure (`scripts/archive_production_inputs.py`) freezes inputs daily.
3. **Survivorship bias**: Universe is as-of each snapshot date. Tickers that delist or merge are included up to their last valid date.
4. **Transaction costs**: Backtests must account for turnover. Historical PIT-corrected excess return is +2.34pp/mo net (t=2.60, 67 periods). Prior claims of +93.7pp are DEPRECATED (contained look-ahead bias).

#### Checklist v2 (Promotion Gate)

Any new signal or model change must pass all 6 modules before promotion:

| Module | Test | Threshold |
|--------|------|-----------|
| Feature Monotonicity (FM) | Decile spread, monotonic rank-return | Spread > 0, no inversions in top 3 deciles |
| Bootstrap | Resampled IC distribution | 95% CI excludes zero |
| FDR | Benjamini-Hochberg multiple testing correction | q < 0.10 across signal family |
| LOSO | Leave-one-season-out cross-validation | No single season drives >50% of total IC |
| Year Stability | Per-year IC sign consistency | Positive IC in ≥60% of calendar years |
| Ablation | Add/remove signal from production stack | Marginal contribution > 0 after controlling for existing signals |

Implementation: `common/stats/` (6 modules, 36 tests). Current stack: v1.14.0
coinvest-only selector + 2-feature pairwise ranker (deployed weights:
`coinvest_score_z` +0.02, `financial_score` -0.0533). Active ruleset `8887576e` v1.14.0 (was
`2a3e79eb` v1.13.0; v1.14.0 zeroed `inst_delta_z` selector weight as a
demotion-class hygiene patch — see `RULESET_CHANGELOG.md`).

#### Event Feedback Loop

Resolved events feed calibration through the event feedback pipeline:

```
Herald detects → CRT resolves (HIT/MISS) → join to T-1 snapshot
  → store in artifacts/event_feedback/ → weekly calibration metrics
```

- Daily: `build_event_feedback.py` materializes resolved events (35 resolved, 71% Herald match rate)
- Weekly: `build_event_feedback_metrics.py` computes source precision, confidence ECE, outcome confusion, regulatory calibration
- Read-only: metrics inform governance decisions but never auto-update model priors

---

## 10. OpenClaw Agent Fleet

26 agents on gateway ws://127.0.0.1:18789 (6 Sonnet 4.6, 15 Haiku 4.5, 1 main).

### Production Monitors (cron-scheduled)

| Agent | Name | Schedule | Role |
|-------|------|----------|------|
| ops | Packet | 5:00 PM ET weekdays | Duty officer, reads ops digest |
| sentinel | Vigil | 5:15 PM ET | Drift monitor, rollback advisor |
| qa | Litmus | 5:30 PM ET (via heartbeat checks) | Artifact validation, contract audit |
| calibration | Tuner | Fri (via heartbeat checks) | Evidence weighing, candidate review |
| ic_health_monitor | Canary | 5:45 PM ET (via heartbeat checks) | Signal decay watchdog |
| production_qa | Inspector | 5:45 PM ET (via heartbeat checks) | Post-production codebase review, lint, schema, distribution health |
| fleet_steward | Conductor | 6:15 PM ET (via heartbeat checks) | Fleet orchestration |

### Data & Collection

| Agent | Name | Role | Status |
|-------|------|------|--------|
| aact_trial_ingest | Archivist | Bulk AACT clinical trial warehouse | Live, 580K trials, weekly (Mon) |
| company_news_ingest | Herald | Deterministic PR collection + classification | RETIRED 2026-05-06 (P1 #5) — consolidated into `herald` (canonical news-ingest agent); historical artifacts preserved at `data/press_releases/` |
| herald | Herald | Press release collector and news summarizer | Live — 3 daily digests |
| ctgov_poller | Registry | ClinicalTrials.gov delta polling | Live, daily |
| earnings_calendar_sync | Bellringer | Earnings calendar maintenance | Live, 2x daily |
| grok_biotech_watch | Scout | Web sentinel for biotech signals | 1x daily weekdays (16:00 ET; reduced from 4×/day on 2026-05-06 — ROI audit; see RULESET_CHANGELOG / agent fleet audit P1 #3) |
| universe_maintenance | Gardener | Universe steward | Weekly (Mon) |
| data_auditor | Auditor | Pipeline integrity checks | Daily + weekly deep |
| biotech_news_digest | Herald Digest | News digest builder/formatter | 3x daily |

### Resolution & Analysis

| Agent | Name | Role | Status |
|-------|------|------|--------|
| crt_resolution_watcher | Verdict | Catalyst outcome tracker | Live, 98 resolutions |
| catalyst_delta | Pulse | Event-change detection | Live |
| postmortem | Record | Event resolution evidence archivist | Live |
| event_analyst | Analyst | Lesson aggregation from events | Weekly Friday (LLM 18:55 ET, builder 19:10 ET; reduced from daily on 2026-05-06 — P1 #4) |
| review_queue_steward | Triage | Review queue dispatcher | Live |

### Portfolio & Market

| Agent | Name | Role | Status |
|-------|------|------|--------|
| options_watch | Surface | Options volume/surface flags | Live |
| price_action_watch | Tape | Price/volume scanner | Live |
| shadow_monitor | Mirror | Shadow portfolio observer | Live — deterministic build daily via run_daily_production.py + Tier 2 heartbeat check; LLM cron retired 2026-05-06 (P1 #6) |
| shadow_watch | Mirror | Portfolio pattern monitor | Live |
| policy_shadow_watch | Shadow | Policy change comparator | Live |
| bioshort_watch | Hedge | Hedge fund governance monitor | SUPPRESSED 2026-05-06 (bioshort P2) — upstream `output/hedge_report/` unscheduled and 41-day stale; producer `tools/biotech_hedge_report.py` preserved; reactivation requires separate spec |
| calibration_evidence | Evidence | Calibration evidence builder | Weekly (Fri)

---

## 11. Dashboard

React + Vite 6 + Tailwind v3 + Recharts frontend with FastAPI backend.

### Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/api/rankings/{date}` | Full rankings table |
| `/api/ticker/{ticker}` | Merged ticker detail |
| `/api/aact/{ticker}` | AACT trial records for ticker |
| `/api/deal_comps/{ticker}` | DealForma deal comp context |
| `/api/purple_book/{ticker}` | Biologics competition context |
| `/api/crt/resolutions` | CRT resolution records |
| `/api/crt/calibration` | CRT calibration summary |
| `/api/shadow_performance` | Shadow portfolio timeseries |
| `/api/bioshort/verdict` | Hedge report verdict |
| `/api/herald/health` | Press release collection health |
| `/api/aact/health` | AACT ingest health |

### TickerDetail Tabs

Overview | Options | Portfolio | Trials | Deals | Bio | CRT

---

## 12. Roadmap (April-May 2026)

### Phase 1: This Week (April 1-7)

- [x] Selection-only benchmark → **DONE**: +95.2% excess, IR 1.41
- [x] Construction drag decomposition → **DONE**: 50/31/19 split identified
- [x] Construction v2 candidate pack → **DONE**: EW Top-30 leads (IR 2.64-2.81)
- [x] Transaction-cost / rebalance threshold model → **DONE**: EW Top-30 wins net of costs (+85.8% at 50 bps)
- [x] Monthly IC decomposition → **DONE**: see below
- [ ] total_volume_z validation (April 7)

#### Monthly IC Decomposition Results

| Scope | Mean IC (h20) | % Positive | Interpretation |
|-------|--------------|-----------|----------------|
| Full universe | -0.044 | 34.4% | Ranking is poor at ordering middle/bottom |
| **Top-30 only** | **-0.003** | **52.2%** | Within-top-30 ordering does not predict (t=-0.10) |

**Interpretation:** The DEM is a **filter/selector**, not a **ranker**. It identifies
a good bucket of ~30 names (EW Top-30 generates +95% excess) but does not
meaningfully distinguish rank #1 from rank #30 within that bucket. This explains
why EW outperforms rank-weighting and why top-20 and top-30 produce similar excess.

**Monthly pattern:** IC is episodic. Strong positive months (2023-04: +0.224,
2024-01: +0.171) interspersed with negative months. Last 6 months (2024 H2)
show deeply negative IC (-0.137 mean) driven by overall biotech selloff.

**Implication for construction:** EW is the correct weighting scheme. Rank-based
concentration would only help if within-top-30 IC were reliably positive, which
it is not. This further validates the EW Top-30 construction choice.

### Architectural Decision: Two-Stage Model

The monthly IC decomposition revealed that DEM is a **filter/selector**, not a
**ranker**. Full-universe IC is negative (-0.044), top-30-only IC is zero. The
model says "these 30 are interesting" but not "this one is better than that one."

**Architecture:**
1. **Stage 1 — Selector (DEM, proven):** Identifies the top-30 candidate set.
   Keep fixed. EW Top-30 is the correct construction because within-bucket
   ordering doesn't predict.
2. **Stage 2 — Ranker (new, shadow-first):** A dedicated within-top-30 model
   that answers "among names DEM already likes, which should get more capital?"

**Ranker features** (vary meaningfully inside the top bucket):
- Options mispricing: `actual_implied_move_pctile`, event premium, skew/RR
- AACT timeline deltas: PCD shifts, enrollment changes, results posted
- `momentum_delta_z` (only confirmed sort contributor)
- `total_volume_z` (if validated)
- Catalyst type differentiation (regulatory vs pivotal vs mid-stage)
- DealForma dealability priors (slow-moving)

**Promotion bar for ranker:**
- Top-30-only IC meaningfully positive and stable
- Rank-weighted-by-ranker Top-30 beats EW Top-30 net of costs
- Survives by regime
- Until then: selector = active, ranker = shadow

### Phase 2: April

- [ ] Lightweight risk layer (correlation penalty, vol targeting)
- [ ] Options coverage push to 80% + event-premium decomposition
- [ ] Herald precision audit
- [ ] Dashboard integration pass (AACT index optimization)

### Phase 3: May — Two-Stage Ranker

- [x] Build top-30 rank dataset → **DONE**: 7,157 feature rows, 86,759 pairwise rows
- [x] First ranker training → **DONE**: null result — correct (features empty in test window)
- [x] Ranker readiness gate → **DONE**: 8/30 eligible dates, blocked on options accumulation
- [ ] Accumulate ranker-ready snapshots (target: late April, ~30 eligible dates)
- [ ] AACT delta features for within-top-30 ranking
- [ ] Options event-premium decomposition for within-top-30 ranking
- [ ] Retrain ranker on ranker-ready window
- [ ] Validate: top-30-only IC, rank-weighted vs EW, pairwise accuracy
- [ ] DealForma dealability prior as slow-moving ranker feature
- [ ] Governed shadow review / promote-reject gates

**Ranker status:** Paused. First training produced a correct null result — all
models at coin-flip accuracy because test window (2023-2024) had zero options
coverage and sparse momentum_delta_z. The ranker concept is not falsified; it is
currently untestable on the available historical data. Readiness gate at
`output/ranker/ranker_data_readiness.json` tracks when training becomes viable.

### Operating Thesis (updated 2026-05-24)

> **coinvest selects, capped coinvest + financial_score ranks, financial_score
> penalizes "safe but less catalytic" names, and clinical is a weak/conditional
> feature under review.**
>
> v1.14.0 selector is coinvest-only: `coinvest_score_z` 100%, `inst_delta_z` 0%.
> The prior B6 selector (65/35) is historical validation evidence; `inst_delta_z`
> remains computed/exported but does not affect selector_score, actionable_rank,
> final_score, or top-30 membership.
>
> Pairwise_minimal ranker is 2-feature and ordinal-only (ECE = 0.129). Within the
> top-60 cohort, deployed `coinvest_score_z` is capped at +0.02 and
> `financial_score` is a true negative penalty (safe names underperform).
>
> EW Top-30 is the correct construction. Rank-weighting and confidence sizing are
> not justified — pairwise scores are not calibrated.
>
> The selector's edge is regime-dependent, strongest in bear biotech.
> Fixed sleeve budgets are retired. Bucket labels survive as metadata only.

---

## 13. Key Files

| File | Purpose |
|------|---------|
| `decision_engine.py` | DEM core — L0→L2→L4→L4b→L3 |
| `selector_engine.py` | v1.14.0 coinvest-only selector (`coinvest_score_z` 100%, `inst_delta_z` 0%) |
| `ranker_engine.py` | clinical_50 ranker (legacy bounded ±15%) |
| `ranker_v2_pairwise.py` | pairwise_minimal ranker (Bradley-Terry, 2 deployed features) |
| `run_screen.py` | Production pipeline orchestrator |
| `tools/run_daily_production.py` | Daily cron pipeline (Steps 1-6) |
| `tools/live_shadow_portfolio.py` | Shadow portfolio construction + PnL |
| `tools/catalyst_resolution_tracker.py` | CRT core |
| `tools/crt_calibration.py` | CRT calibration rollup |
| `tools/fetch_aact_snapshot.py` | AACT trial warehouse ingest |
| `event_ev/expectation_error_model.py` | EES v2 — Trap gate + quality overlay |
| `event_ev/ees_v3.py` | EES v3 — conditional misprice + expected move (diagnostic) |
| `event_ev/conditional_model.py` | Biomarker/subgroup conditional mispricing detection |
| `event_ev/execution_capacity.py` | Post-sizing participation guardrails |
| `event_ev/portfolio_sizing.py` | Conviction sizing (α=1.5) |
| `event_ev/runway_severity.py` | Runway-to-catalyst severity (4-layer diagnostic) |
| `event_ev/evidence_snapshot.py` | PIT-anchored evidence snapshot builder (trial design + designations + CRT + PubMed) |
| `data_sources/pubmed_client.py` | NCBI E-utilities PubMed client (search, fetch, cache, literature scoring) |
| `common/protocol_quality.py` | Phase-conditional protocol quality score (HINT-derived) |
| `common/biomarker_context.py` | Conditional biomarker context score |
| `common/endpoint_quality.py` | Endpoint quality v2 (7-bucket, phase-aware) |
| `tools/clinical_transmission_shadow.py` | Daily shadow: default vs TX-enabled EV rankings |
| `research/clinical_tx_outcome_eval.py` | Outcome evaluation + alpha attribution framework |
| `research/hint_adapter.py` | HINT schema mapper + NCT matcher |
| `research/hint_benchmark.py` | Offline benchmark: PoS v3 vs HINT baselines |
| `research/hint_feature_extract.py` | Protocol feature extraction (10 PIT-safe features) |
| `research/full_current_model_backtest.py` | Full historical backtest (4-variant) |
| `scripts/research/crt_bta_calibration.py` | BioTradingArena calibration benchmark |
| `tools/ees_v3_forward_monitor.py` | Forward evidence tracker toward WS4 clearance |
| `tools/build_event_feedback.py` | Resolved event materializer (CRT→Herald→postmortem join) |
| `tools/build_event_feedback_metrics.py` | Weekly calibration metrics (source precision, ECE) |
| `common/stats/` | Statistical QA package (FM, bootstrap, FDR, LOSO, calibration) |
| `common/options_diagnostics.py` | Options surface data (Tastytrade) |
| `dashboard/app.py` | FastAPI backend |
| `frontend/dashboard/` | React frontend |
| `specs/SYSTEM_SPEC.md` | System invariants |
| `production_data/portfolio_policy.json` | Portfolio construction policy (v3) |
| `production_data/ranker_v2_model.json` | Pairwise minimal model weights |
| `production_data/decision_rulesets/v1.14.0_coinvest_only_selector.json` | Active ruleset |
| `scripts/research/checklist_v2_rerun.py` | Promotion Checklist v2 battery runner |
| `scripts/research/pairwise_feature_audit.py` | Within-cohort feature diagnostic |

---

## 14. Statistical QA Layer (Spec 055, 2026-04-04)

### Promotion Checklist v2

Any signal promotion now requires passing all 5 gates:

1. **Signal card**: Coverage ≥40%, selector Δ > 0, ranker IC > 0
2. **Fama-MacBeth incremental**: NW-t ≥ 1.96 with controls (sponsorship, momentum_delta, financial)
3. **Block bootstrap**: 95% CI on portfolio delta excludes zero (6-month blocks, n=10,000)
4. **BH FDR**: q-value < 0.10 within testing family
5. **LOSO robustness**: Worst-slice delta positive across year/regime/cap/catalyst/stage

### Checklist v2 Rerun Results (2026-04-04)

| Signal | G1 Card | G2 FM | G3 Boot | G4 FDR | G5 LOSO | Total | Verdict |
|--------|---------|-------|---------|--------|---------|-------|---------|
| sponsorship_score_z | PASS | FAIL | PASS | FAIL | PASS | 3/5 | SHADOW |
| momentum_delta_z | PASS | FAIL | PASS | FAIL | FAIL | 2/5 | NO_GO standalone |
| event_type_score | PASS | PASS | PASS | PASS | PASS | 5/5 | **PROMOTE (overlay)** |
| insider_exec_buy_value_90d | FAIL | PASS | FAIL | FAIL | FAIL | 1/5 | NO_GO |
| aact_execution_score | PASS | FAIL | FAIL | FAIL | FAIL | 1/5 | NO_GO |
| **B6 bundle** | — | — | **PASS** | — | **PASS** | — | **VALIDATED** |

### Pairwise Calibration Assessment

- Pairs evaluated: 33,093 (67 snapshots)
- Brier score: 0.2755
- ECE: 0.129 → **POOR — ordinal ranking only**
- Pairwise accuracy: 53.0%
- Platt-calibrated ECE: 0.013 (but raw scores are uncalibrated)

**Policy**: No rank-weighting, no confidence sizing. Pairwise scores determine ordering
only. Equal-weight construction is the correct response to ordinal-only ranking.

### Within-Cohort Feature Audit (2026-05-24 status)

| Feature | Evidence / Weight | Mechanism | Action |
|---------|-------------------|-----------|--------|
| financial_score | Deployed −0.0533; NW-t −3.41 in audit | TRUE PENALTY — persists all cohorts, all regimes | Keep negative weight; frozen |
| coinvest_score_z | Deployed +0.02; trained +0.0613 | Bounded institutional support within top-60 | Keep capped Family C weight; frozen |
| inst_delta_z | 0% selector weight; absent from production ranker | Diagnostic/export only after 2026-05-04 ALERT | Do not reinstate without governance review |
| clinical_score_v2_z | Historical audit NW-t −2.38 as ranker | Collider + weak penalty — vanishes in high-sponsorship stratum | Closed/diagnostic unless new evidence |

**Key insight**: The selector and ranker learn different structure. Coinvest gets
names into the room; within the room, the capped two-feature ranker uses bounded
coinvest support plus the `financial_score` stress-upside penalty. `inst_delta_z`
is no longer a production discriminator in v1.14.0.

### Infrastructure

| Script | Purpose |
|--------|---------|
| `common/stats/` | 6 modules: cross_sectional, bootstrap, multiple_testing, calibration, robustness, survival |
| `scripts/research/checklist_v2_rerun.py` | Targeted battery: Queue A (signals), B (calibration), C (B6 bundle) |
| `scripts/research/pairwise_feature_audit.py` | 6 diagnostic tests for within-cohort feature behavior |
| `scripts/research/statistical_methods_upgrade.py` | Full Spec 055 battery (broad, all signals) |
| `scripts/research/herald_precision_study.py` | Spec 056 — first Checklist v2 pass (event_type_score) |

---

## 15. Test Coverage

**14,519 tests** across **516 test files**. Pre-commit hooks enforce black, isort, flake8, and detect-secrets on every commit.

### Key Test Suites

| Suite | Tests | Focus |
|-------|-------|-------|
| test_decision_engine (5 files) | 267 | DEM layers, sort keys, eligibility, determinism, golden records |
| test_composite_v3 | 91 | Module 5 composite scoring |
| test_drift_report | 219 | Portfolio drift monitoring |
| test_ic_enhancements | 139 | IC measurement, evaluation |
| test_eval_forward_returns | 130 | Forward return evaluation harness |
| test_null_safety | 130 | Missing-data resilience |
| test_ema_committee_collector | 120 | EU regulatory data |
| test_defensive_overlay_adapter | 120 | Defensive overlay |
| test_production_hardening | 96 | Production robustness |
| test_accuracy_improvements | 93 | Scoring accuracy |
| test_catalyst_event_graph | 86 | Catalyst event modeling |
| test_decision_actionable_ordering | 84 | Actionable rank determinism |
| test_decision_ruleset | 84 | Ruleset governance |
| test_regime_engine | 83 | Market regime classification |
| test_expression_layer | 83 | Spec 062 options expression |
| test_options_diagnostics | 81 | Options surface diagnostics |
| test_clinical_v2_robustness | 75 | Clinical score robustness |
| test_institutional_delta | 75 | Sponsorship momentum signal |
| test_input_validation | 80 | Input schema enforcement |
| test_score_utils | 82 | Scoring utilities |
| test_expectation_error_model | 63 | EES/Trap gate |
| test_event_ev_engine | 68 | Event EV 6-layer Bayesian |
| test_evidence_snapshot | 33 | Evidence snapshot PIT safety, field tolerance, designation extraction |
| test_pubmed_client | 29 | PubMed XML parsing, literature scoring, cache, API key, drug map |
| test_protocol_quality | 19 | Protocol quality score, phase-conditional weights, CalendarAlpha integration |
| test_biomarker_context | 13 | Conditional biomarker detection, phase/indication/design conditioning |
| test_endpoint_quality | 17 | Endpoint bucket classification, phase scoring, multi-EP handling |
| test_clinical_transmission | 6 | Clinical-to-p_hit transmission, phase caps, sign symmetry |
| test_phase2_recalibration | 8 | Phase 2 prior value, old values preserved, CRT exclusion |
| test_hint_integration | 22 | HINT adapter, features, benchmark, PIT safety, no-production-import |
| test_phase2_daily | 143 | Daily production pipeline |
| test_classify_press_releases | 56 | Herald news classification |
| test_alpha_cohort | 56 | Alpha cohort analysis |
| test_financial_v2_golden | 42 | Module 2 financials |
| test_pipeline_robustness | 54 | Integration robustness |
| test_pit_enforcement | 39 | PIT safety checks |
| test_catalyst_resolution_tracker | 28 | CRT watchlist, classification |
| test_crt_real_record_fixtures | 19 | Real resolution records |
| test_aact_ingest | 28 | AACT normalization, linkage, deltas |
| test_expression_attribution | 40 | Spec 062 attribution |
| test_purple_book_features | 16 | Spec 047 biologics competition |
| test_news_feed | 24 | Spec 044 news schema |
| test_options_quality | 17 | Spec 045 quality layer |
| test_dealforma_features | 24 | Spec 046 deal comps |

### Coverage by Domain

| Domain | Files | Tests | Key Areas |
|--------|-------|-------|-----------|
| Decision engine & scoring | ~80 | ~3,200 | DEM, composite, sort, eligibility, determinism |
| Data collection & ingestion | ~60 | ~1,800 | SEC, CTgov, AACT, options, news, 13F |
| Portfolio construction | ~40 | ~1,100 | Sizing, rebalance, risk, cost model, drift |
| Governance & promotion | ~30 | ~900 | Rulesets, gates, canary, rollback, IC health |
| Clinical & catalyst | ~50 | ~1,500 | Module 3/4, catalyst events, CRT, POS priors |
| Options & expression | ~30 | ~800 | Surface, diagnostics, quality, expression layer |
| Backtesting & evaluation | ~25 | ~700 | Walk-forward, IC, forward eval, signal backtest |
| PIT & data integrity | ~20 | ~600 | PIT enforcement, audit, schema, validation |
| Integration & smoke | ~15 | ~300 | Pipeline smoke, determinism, robustness |
| Infrastructure | ~30 | ~500 | Logging, hashing, dates, types, manifests |

---

*Document updated 2026-04-14 (ruleset reference refreshed 2026-05-06). Active ruleset: 8887576e (v1.14.0; was 2a3e79eb v1.13.0 until 2026-05-04 demotion of `inst_delta_z` — demotion path, not Checklist v2 — see `RULESET_CHANGELOG.md` and `policy_demotion_path_2026_05_06.md`). QA baseline: Checklist v2 rerun (for the prior B6 65/35 bundle; v1.14.0 is a demotion-class change and does not require Checklist v2 retrospectively).*
