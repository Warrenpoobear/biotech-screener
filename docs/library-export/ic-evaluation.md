# IC & Signal Evaluation Skill

## Purpose

Reference for the statistical framework used to evaluate, promote, and demote signals in the biotech screener. Covers IC measurement, the Checklist v2 promotion battery, forward shadow monitoring, and the evidence hierarchy.

This skill is organized into two sections:

1. **Framework Reference** - Stable rules, constants, and methodology (changes only with code updates)
2. **Operational State** - Volatile snapshots that go stale and require periodic refresh

---

# SECTION 1: FRAMEWORK REFERENCE

---

## Operator Statistical Background

- **Operator**: Darren Schulz, CFA, CAIA — Director of Investments, Wake Robin (Holland, MI)
- **Quantitative background**: CFA credential covers quantitative methods, performance measurement, and attribution. Engages with academic finance research (Journal of Finance, SSRN quant finance papers). Follows systematic/quant strategies (AQR Capital).
- **IC interpretation authority**: The operator's institutional experience ($14B+ AUM, multi-asset-class portfolio oversight) provides the context for evaluating whether IC signals are economically meaningful vs. statistically spurious. All Checklist v2 promotion/demotion decisions require operator judgment on economic plausibility.
- **Performance measurement experience**: Career-long institutional benchmarking (Barclays Agg, custom composites), return attribution, and risk-adjusted performance evaluation. This background informs the evidence hierarchy and the distinction between backtest artifacts and true forward evidence.

## IC Decomposition

**Tool**: `tools/ic_decomposition.py`

Measures Spearman IC of a signal vs forward returns, segmented by cohort and stage.

### Key Features

- Forward returns panel: PIT-safe, h5d (5-day horizon)
- Cohort-contamination tagging: dates after manager additions flagged as contaminated
- Pre/post IC reported separately (clean vs contaminated windows)
- Stage_bucket segmentation (early/mid/late)
- catalyst_quality segmentation (auto-detects column presence)
- Top-30 walk-forward: mean/median excess 5d return per snap date + cumulative

### CRITICAL: IC Tooling Scope Gap (Spec 095, 2026-05-13)

**The IC backtest tool (`run_rank_ic_backtest.py`) measures composite_score IC, NOT production ranker final_score IC.** This is a confirmed conflation:

- composite_rank correlates only 0.25 with actionable_rank (production)
- Top-30 overlap: 7/30 (23%) between composite vs production rankings
- composite_score weakly correlates (0.13) with final_score
- IC backtest selects a completely different portfolio than production

**Consequence**: Ranker IC is UNMEASURED. Any IC claims based on composite_score are misattributed. Do NOT use prior IC evidence for ranker promotion until Spec 100 (tooling correction) is complete or outputs are explicitly relabeled.

**Spec 100 fix** (commit 2faa88e6, 2026-05-17): Code committed. Prior composite_score IC claims invalidated; final_score baseline established. Full operationalization is now unblocked: architecture freeze LIFTED 2026-05-26 (h20d checkpoint passed). The Checklist v2 battery rerun against final_score has NOT yet been executed — this is the highest-priority code change post-freeze. Until that rerun completes, ranker IC remains effectively unmeasured for promotion purposes.

### Interpretation Rules

- Serial correlation is heavy (5-day windows overlap across daily snapshots)
- 14 snap dates yields ~37 effective observations
- IC t-stats are indicative only, not promotion-grade
- Promotion requires Checklist v2 (full battery below)
- **Ranker IC is currently unmeasurable** — existing tools conflate composite_score with final_score (Spec 095). Blocks all ranker IC claims until Spec 100 is implemented.

### IC Tool Field-Declaration Rule (standing prevention rule)

Origin: failure F-2026-002 (MA, CRITICAL) — `run_rank_ic_backtest.py` silently measured composite_score IC while it was assumed to measure production final_score IC, invalidating a pervasive body of IC claims. To prevent recurrence:

1. Every IC measurement tool MUST declare, in its output header/metadata, exactly which score field it measures (e.g. `composite_score` vs `final_score`) and over which universe (full eligible vs. top-N cohort).
2. No IC claim is valid unless the declared field matches the production sort key it is being used to justify.
3. When citing any IC figure, state the field and universe alongside it. A bare "IC = x" with no field + universe is non-promotion-grade by default.

---

## Checklist v2 Promotion Battery

**File**: `scripts/research/checklist_v2_rerun.py`

5-gate statistical bar. A signal must pass ALL gates for promotion.

### Gate 1: Fama-MacBeth (FM) Regression

Cross-sectional regression with Newey-West corrected standard errors. Positive t-stat required. Controls for size, momentum, and sector.

### Gate 2: Bootstrap

- 1000 iterations of resampled IC
- 95% CI must exclude zero
- P(>0) threshold for confidence

### Gate 3: FDR (False Discovery Rate)

- Benjamini-Hochberg correction across all tested signals
- Controls for multiple comparison bias

### Gate 4: LOSO (Leave-One-Slice-Out)

- Robustness across all dimensions: time, sector, market cap, regime
- Must be ROBUST (not fragile to any single slice removal)

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

**Tracker**: coinvest_shadow_tracker v2 (7 arms, wired into `run_daily.py`)

The ONLY true out-of-sample evidence. Accumulates daily from production.

### Evaluation Rules

- Evaluate after 30+ trading days of true-PIT daily production
- If forward evidence is positive: re-establish selector thesis from clean data
- If forward evidence is negative: selector needs structural re-examination
- Do NOT backfill from historical

---

## Evidence Hierarchy

| Rank | Source | Strength |
| --- | --- | --- |
| 1 | Checklist v2 rerun (2026-04-04) | STRONGEST (signals) |
| 2 | True PIT backtest (Spec 050) | STRONGEST (portfolio) |
| 3 | Pairwise feature audit (2026-04-04) | SUPPORTING |
| 4 | Forward shadow | MONITORING |
| 5 | Old PIT benchmark (Spec 048) | SUPERSEDED |

---

## IC Measurement Constants

| Constant | Value | Purpose |
| --- | --- | --- |
| MIN_OBS_IC | 10 | Minimum observations for IC |
| MIN_OBS_TSTAT | 20 | Minimum for t-statistic |
| MIN_OBS_BOOTSTRAP | 30 | Minimum for bootstrap CI |
| MIN_ROLLING_WINDOW | 12 weeks | Minimum rolling window |
| BOOTSTRAP_ITERATIONS | 1000 | Bootstrap resampling count |
| TSTAT_THRESHOLD_95 | 2.0 | 95% confidence |
| TSTAT_THRESHOLD_99 | 2.58 | 99% confidence |

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

## Deprecated Evidence (Do Not Cite)

- All survivorship-only benchmark numbers (+93.7pp, +110.5pp, etc.)
- Pre-Checklist-v2 signal card t-stats
- "Bear IR 3.35" regime story from contaminated data
- Any promotion memo citing pre-Spec-050 selector performance
- **Any ranker IC claim based on composite_score** (Spec 095, 2026-05-13) - these measured the wrong score field and are misattributed

---

## Source Files

| Component | File |
| --- | --- |
| IC Decomposition | `tools/ic_decomposition.py` |
| Checklist v2 Rerun | `scripts/research/checklist_v2_rerun.py` |
| Statistical QA Package | `common/stats/` (6 modules) |
| Forward Shadow Tracker | Part of `run_daily.py` |
| Signal Evidence Runner | `scripts/run_signal_evidence.py` |

---

# SECTION 2: OPERATIONAL STATE

> **SNAPSHOT DATA** - The values below are point-in-time and go stale. Verify against current pipeline output before citing.

---

## Current Signal Scores (Checklist v2)

*Last reviewed: 2026-05-18*

| Signal | Score | Status |
| --- | --- | --- |
| B6 bundle (coinvest + inst_delta) | 5/5 | Production (as bundle) |
| event_type_score | 5/5 | Overlay only (doesn't improve B6) |
| coinvest_score_z standalone | 3/5 | Part of bundle |
| insider_exec_buy_value_90d | 1/5 | Shadow only, FRAGILE |
| aact_execution_score | 1/5 | Shadow only, bear-unstable |

## coinvest_score_z IC Snapshot

*Last reviewed: 2026-06-25. Refresh after each 13F cycle and at scheduled checkpoints.*

| Window | n_dates | Mean IC | Hit Rate |
| --- | --- | --- | --- |
| Pooled (all) | 14 | -0.031 | 28.6% |
| Pre-cohort (clean) | 9 | -0.051 | 11.1% |
| Post-cohort (contaminated) | 5 | -0.008 | 60.0% |

**Verdict**: OBSERVE. April selloff drove pre-cohort negativity. Post-cohort recovering.

**IMPORTANT**: These IC figures measure coinvest_score_z across the full eligible universe. They do NOT measure ranker IC within the top-60 cohort (Spec 095 confirmed this gap on 2026-05-13). Ranker-specific IC is UNMEASURED until Spec 100 tooling is implemented.

## Ranker IC Tooling Status (Spec 095 / Spec 100)

*Added: 2026-05-13. Updated: 2026-05-17.*

The existing IC backtest tool previously measured the WRONG score. Spec 100 commit (2faa88e6, 2026-05-17) corrected the tooling — `run_rank_ic_backtest.py` now supports score-field and universe parameters with explicit metadata output. Prior composite_score IC claims are invalidated; a final_score baseline has been established.

Architecture freeze LIFTED 2026-05-26 (h20d checkpoint passed). Checklist v2 battery rerun against final_score has NOT been executed yet. Until that rerun completes, ranker IC remains effectively unmeasured for promotion purposes. This is the highest-priority action item.

## Forward Shadow Status

*Accumulating since: 2026-04-03. As of 2026-06-25, approximately 62+ trading days accumulated.*

At or past the 30-day evaluation threshold. Architecture freeze LIFTED 2026-05-26. Evaluate per the rules in Section 1 — confirmed >= 30 trading days of true-PIT daily production data.
