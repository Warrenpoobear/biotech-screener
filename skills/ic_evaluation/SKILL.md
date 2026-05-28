# IC & Signal Evaluation Skill

## Purpose

Reference for the statistical framework used to evaluate, promote, and demote signals in the biotech screener. Covers IC measurement, the Checklist v2 promotion battery, forward shadow monitoring, and the evidence hierarchy.

This skill is organized into two sections:

1. **Framework Reference** \- Stable rules, constants, and methodology \(changes only with code updates\)
2. **Operational State** \- Volatile snapshots that go stale and require periodic refresh

---

## Codegraph Preflight (mandatory before any code edit)

IC evaluation produces forward shadow evidence — Tier 2/3 boundary (artifacts feed promotion decisions). Before editing any symbol, run the standard preflight per `skills/codegraph/SKILL.md`:

1. `codegraph_search("<symbol>")` — locate the target
2. `codegraph_node("<symbol>", source=True)` — inspect signature and body
3. `codegraph_callers("<symbol>")` — identify all production callers
4. `codegraph_callees("<symbol>")` — map downstream dependencies
5. `codegraph_impact("<symbol>", depth=2)` — confirm blast radius

**Gate:** If impact reaches walk-forward harness, shadow-monitoring artifacts, or any governance evidence surface — Claude Code review is required before merge (Tier 2 trigger per `governance/AGENT_ROUTING_POLICY.md`).

---

# SECTION 1: FRAMEWORK REFERENCE

---

## IC Decomposition

**Tool**: `tools/ic_decomposition.py`

Measures Spearman IC of a signal vs forward returns, segmented by cohort and stage.

### Key Features

- Forward returns panel: PIT-safe, h5d \(5-day horizon\)
- Cohort-contamination tagging: dates after manager additions flagged as contaminated
- Pre/post IC reported separately \(clean vs contaminated windows\)
- Stage\_bucket segmentation \(early/mid/late\)
- catalyst\_quality segmentation \(auto-detects column presence\)
- Top-30 walk-forward: mean/median excess 5d return per snap date + cumulative

### IC Tooling Scope Gap — RESOLVED via Spec 100 \(2026-05-17\)

**Prior finding (Spec 095, 2026-05-13):** The IC backtest tool measured composite\_score IC, NOT production ranker final\_score IC. This created a confirmed conflation:

- composite\_rank correlates only 0.25 with actionable\_rank \(production\)
- Top-30 overlap: 7/30 \(23%\) between composite vs production rankings
- composite\_score weakly correlates \(0.13\) with final\_score
- IC backtest selected a completely different portfolio than production

**Spec 100 Implementation (2026-05-17):** Tool corrected to measure `final_score` (production ranker) by default.
- Default signal changed from `score_rank_pct` → `final_score`
- Metadata includes explicit `spec_100_status: "CORRECTED"`
- `composite_score` IC marked as ⚠️ INVALIDATED for promotion purposes
- `final_score` IC marked as ✓ Spec 100 corrected (production ranker IC)
- All future IC output states score field, tool version, and measurement status
- Prior composite_score IC claims remain invalidated; new final_score IC is baseline evidence

**Tooling specification:** `run_rank_ic_backtest.py --signal final_score` (now default)

### Interpretation Rules

- Serial correlation is heavy \(5-day windows overlap across daily snapshots\)
- 14 snap dates yields \~37 effective observations
- IC t-stats are indicative only, not promotion-grade
- Promotion requires Checklist v2 \(full battery below\)
- **Ranker IC is now measurable (Spec 100, 2026-05-17)** — tool corrected to measure `final_score` instead of prior composite\_score conflation. Historical composite\_score IC claims are invalidated; future ranker IC claims require final\_score evidence.

---

## Checklist v2 Promotion Battery

**File**: `scripts/research/checklist_v2_rerun.py`

5-gate statistical bar. A signal must pass ALL gates for promotion.

### Gate 1: Fama-MacBeth \(FM\) Regression

Cross-sectional regression with Newey-West corrected standard errors.

- Positive t-stat required
- Controls for size, momentum, and sector

### Gate 2: Bootstrap

- 1000 iterations of resampled IC
- 95% CI must exclude zero
- P\(>0\) threshold for confidence

### Gate 3: FDR \(False Discovery Rate\)

- Benjamini-Hochberg correction across all tested signals
- Controls for multiple comparison bias

### Gate 4: LOSO \(Leave-One-Slice-Out\)

- Robustness across all dimensions: time, sector, market cap, regime
- Must be ROBUST \(not fragile to any single slice removal\)

### Gate 5: Year Stability

- Signal must maintain positive IC across rolling annual windows
- 1-year stability gate for production promotion

### Scoring

| Score | Meaning |
| --- | --- |
| 5/5 | Full promotion eligible |
| 3-4/5 | Shadow research, monitor |
| 1-2/5 | Shadow only, do not promote |
| 0/5 | Dead lane |

---

## Forward Shadow Monitoring

**Tracker**: coinvest\_shadow\_tracker v2 \(7 arms, wired into `run_daily.py`\)

The ONLY true out-of-sample evidence. Accumulates daily from production.

### Arms

7 shadow arms tracking different signal combinations and construction variants.

### Evaluation Rules

- Evaluate after 30+ trading days of true-PIT daily production
- If forward evidence is positive: re-establish selector thesis from clean data
- If forward evidence is negative: selector needs structural re-examination
- Do NOT backfill from historical

---

## Evidence Hierarchy

| Rank | Source | Strength |
| --- | --- | --- |
| 1 | Checklist v2 rerun \(2026-04-04\) | STRONGEST \(signals\) |
| 2 | True PIT backtest \(Spec 050\) | STRONGEST \(portfolio\) |
| 3 | Pairwise feature audit \(2026-04-04\) | SUPPORTING |
| 4 | Forward shadow | MONITORING |
| 5 | Old PIT benchmark \(Spec 048\) | SUPERSEDED |

---

## IC Measurement Constants

| Constant | Value | Purpose |
| --- | --- | --- |
| MIN\_OBS\_IC | 10 | Minimum observations for IC |
| MIN\_OBS\_TSTAT | 20 | Minimum for t-statistic |
| MIN\_OBS\_BOOTSTRAP | 30 | Minimum for bootstrap CI |
| MIN\_ROLLING\_WINDOW | 12 weeks | Minimum rolling window |
| BOOTSTRAP\_ITERATIONS | 1000 | Bootstrap resampling count |
| TSTAT\_THRESHOLD\_95 | 2.0 | 95% confidence |
| TSTAT\_THRESHOLD\_99 | 2.58 | 99% confidence |

### Forward Return Horizons

| Horizon | Trading Days |
| --- | --- |
| 1w | 5 |
| 2w | 10 |
| 1m | 20 |
| 1.5m | 30 |
| 3m | 60 |
| 4.5m | 90 |

---

## Deprecated Evidence \(Do Not Cite\)

- All survivorship-only benchmark numbers \(+93.7pp, +110.5pp, etc.\)
- Pre-Checklist-v2 signal card t-stats
- "Bear IR 3.35" regime story from contaminated data
- Any promotion memo citing pre-Spec-050 selector performance
- **Any ranker IC claim based on composite\_score** \(Spec 095 finding, 2026-05-13; Spec 100 fix, 2026-05-17\) - these measured the wrong score field and are misattributed. Corrected ranker IC uses `final_score` measurement.

---

## Source Files

| Component | File |
| --- | --- |
| IC Decomposition | `tools/ic_decomposition.py` |
| Checklist v2 Rerun | `scripts/research/checklist_v2_rerun.py` |
| Statistical QA Package | `common/stats/` \(6 modules\) |
| Forward Shadow Tracker | Part of `run_daily.py` |
| Signal Evidence Runner | `scripts/run_signal_evidence.py` |

---

# SECTION 2: OPERATIONAL STATE

> **SNAPSHOT DATA** \- The values below are point-in-time and go stale. Verify against current pipeline output before citing.

---

## Current Signal Scores \(Checklist v2\)

*Last reviewed: 2026-05-08*

| Signal | Score | Status |
| --- | --- | --- |
| B6 bundle \(coinvest + inst\_delta\) | 5/5 | Production \(as bundle\) |
| event\_type\_score | 5/5 | Overlay only \(doesn't improve B6\) |
| coinvest\_score\_z standalone | 3/5 | Part of bundle |
| insider\_exec\_buy\_value\_90d | 1/5 | Shadow only, FRAGILE |
| aact\_execution\_score | 1/5 | Shadow only, bear-unstable |

### Insider Signal Status \(Spec 104, 2026-05-14\)

`insider_net_buy_value_90d` is DIAGNOSTIC ONLY. It is tracked in `DIAGNOSTIC_FIELDS` and explicitly excluded from `ALPHA_FEATURE_REGISTRY`. It does not enter the scoring model, ranker, or selector. The expectation model has an `insider_net_buy_z` weight that would activate silently if the field flowed into `market_features` -- an explicit isolation guard \(Spec 104 R4a\) prevents this.

Promotion to alpha requires ALL of: 20+ stable snapshots, >= 60% non-null coverage, IC > 0 at p < 0.05, Checklist v2 battery pass, and explicit written approval. Until all five are met, insider stays diagnostic. Do NOT evaluate insider IC for promotion purposes.

### Expectation Feature Coverage Prerequisite \(Spec 105, 2026-05-14\)

IC measurement on expectation-model signals \(`short_interest_pct`, `close_price`, `market_cap_mm`, `priced_move_pct`\) is only valid if those fields are actually flowing into the model at inference time. Spec 105 adds a production gate verifying:

1. All four fields present in `rankings.csv`
2. Per-field coverage above `FEATURE_COVERAGE_REQUIREMENTS` thresholds
3. Expectation model consumes these columns \(not just exports them\)

Any IC research on expectation-gap features against historical snapshots must verify the snapshot was post-wiring \(or backfilled per Spec 102 with `_backfill_version` set\). Pre-wiring snapshots that lack these fields are NOT valid for expectation IC measurement.

## coinvest\_score\_z IC Snapshot

*Last reviewed: 2026-05-13. Refresh after each 13F cycle and at scheduled checkpoints.*

| Window | n\_dates | Mean IC | Hit Rate |
| --- | --- | --- | --- |
| Pooled \(all\) | 14 | -0.031 | 28.6% |
| Pre-cohort \(clean\) | 9 | -0.051 | 11.1% |
| Post-cohort \(contaminated\) | 5 | -0.008 | 60.0% |

**Verdict**: OBSERVE. April selloff drove pre-cohort negativity. Post-cohort recovering.

**IMPORTANT**: These IC figures measure coinvest\_score\_z across the full eligible universe. They do NOT measure ranker IC within the top-60 cohort \(Spec 095 confirmed this gap on 2026-05-13\). Ranker-specific IC is UNMEASURED until Spec 100 tooling is implemented.

### Upcoming Checkpoints

- h20d horizon: 2026-05-26
- Post-Q1 2026 13F refresh: ALL THREE FILED May 15, 2026. Cache warm + cohort quarantine + IC decomposition refresh needed. 5-day observation window runs through \~May 22.
- Forward shadow 30+ trading day evaluation \(accumulating since 2026-04-03 -- should be at or past 30 trading days as of mid-May\)
- Spec 094 selector-only comparator rerun: target 2026-05-27 \(when post-PIT outcomes resolve\)
- Spec 100 ranker IC tooling fix: blocked on architecture freeze \(lifts \~2026-05-26\)

## Ranker IC Tooling Status \(Spec 095 / Spec 100\)

*Spec 095 finding: 2026-05-13 | Spec 100 fix implemented: 2026-05-17*

**Status: CORRECTED** — IC backtest tool now measures `final_score` (production ranker) by default.

- ✓ Spec 100 default signal changed to `final_score` (was `score_rank_pct`)
- ✓ Metadata explicitly labels final\_score IC as Spec 100 corrected
- ✓ composite\_score IC marked INVALIDATED for promotion purposes
- Prior composite\_score IC claims remain invalidated
- Ranker IC measurement now available for forward validation

## Forward Shadow Status

*Accumulating since: 2026-04-03. As of 2026-05-17, approximately 30+ trading days accumulated.*

Should be at or past the 30-day evaluation threshold. Architecture freeze in effect until post-h20d checkpoint (2026-05-26). Evaluate per the rules in Section 1 once confirmed >= 30 trading days of true-PIT daily production data.

---

## Governance Freeze Status (2026-05-17)

**Architecture Freeze** — Active through ~2026-05-26 (h20d checkpoint)
- No model logic changes, feature promotions, or ranking modifications authorized
- Deterministic tooling (preflight, monitoring, verification) permitted
- Spec 100 ranker IC evaluation deferred post-freeze

**13F Q1 2026 Cohort Quarantine** — Active
- 42/48 managers filed (as of 2026-05-19; up from 6/48 on 2026-05-15)
- Validation trigger: ~2026-05-23 (≥34 managers filed — threshold MET as of 2026-05-19)
- Clearance decision: ~2026-05-26 (requires Jaccard ≥0.70 + all 6 gates pass)
- No selector/ranker/sizing changes authorized until cohort clears
- IC health monitor ALERT as of 2026-05-19: lagging historical IC baseline, NOT a system failure — expected post-13F refresh transient

**Decision Gates Ahead**
- **May 19**: Phase 2 Step 3 verification ✓ (IC health monitor ALERT = lagging IC, not failure)
- **~May 23**: 13F refresh validation rerun — trigger MET (42 managers filed)
- **~May 26**: Architecture freeze lift + cohort clearance decision; h20d Decision Memo Draft ready (2026-05-21)
- **Post-May 26**: Spec 100 corrected final_score IC evaluation + Checklist v2 battery

**Interpretation**: All IC evaluation and ranker promotion decisions are deferred until post-freeze when full validation battery can be applied. Current Spec 100 baseline is ready but explicitly labeled for deferred interpretation.
