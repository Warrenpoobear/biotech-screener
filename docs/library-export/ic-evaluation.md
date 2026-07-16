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
- **Quantitative background**: CFA credential covers quantitative methods, performance measurement, and attribution. Engages with academic finance research (Journal of Finance, SSRN quant finance papers). Follows systematic/quant strategies (AQR Capital). Published commentary on credit-equity market integration and CDX vs. SPX options pricing.
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

**Consequence**: Ranker IC is measured — by the forward_eval gate, not the composite_score IC backtest (different measurements; see the RECONCILED 2026-07-12 block below). Any IC claims based on composite_score are misattributed. Do NOT use prior IC evidence for ranker promotion until Spec 100 (tooling correction) is complete or outputs are explicitly relabeled.

**Spec 100 fix** (commit 2faa88e6, 2026-05-17): Code committed. Prior composite_score IC claims invalidated; final_score baseline established. Full operationalization is now unblocked: architecture freeze LIFTED 2026-05-26 (h20d checkpoint passed). The Checklist v2 battery rerun against final_score has NOT yet been executed — this is the highest-priority code change post-freeze. Until that rerun completes, ranker IC remains effectively unmeasured for promotion purposes.

> **Numbering note (2026-05-14):** "Spec 100" remains the ranker IC tooling correction. The expectation layer coverage verification spec was renumbered to Spec 105 (commit cb242311) to resolve the collision. No ambiguity remains.



**RECONCILED 2026-07-12 (repo commit `fbe9bf83`, PR #492).** This block supersedes the "Ranker IC is UNMEASURED / blocked until Spec 100" framing wherever it still appears in this skill — the Consequence line above, the Interpretation Rules below, the coinvest_score_z IC Snapshot note, and the Section 2 "Ranker IC Tooling Status" block. Spec 100 tooling IS implemented (commit `2faa88e6`); the "unmeasured" framing is retired. There are four distinct measurements — do not collapse them into one "ranker IC":

| Measurement | Score field | Universe / windows | Role |
| --- | --- | --- | --- |
| forward_eval gate (#487) | negated actionable_rank | eligible/tradeable set, de-overlapped | production forward rank-IC gate; −0.0597 WARN → +0.0332 PASS after the eligible-scope + de-overlap fix |
| IC dashboard | score_rank_pct | full, overlapping | diagnostic only, not promotion-grade |
| Spec 100 backtest | final_score | top-N cohort | corrected historical baseline; composite_score claims stay invalid |
| forward-shadow mandate SM-20260629-001 | 5d excess vs XBI | ISO-week, non-overlapping | operative out-of-sample gate: 0/20 eligible LIVE windows (code-enforced) |

Operative out-of-sample evidence = the DEM candidate forward-validation mandate (0/20 eligible LIVE windows; ~mid-Nov 2026 gate; 2026-09-30 checkpoint), detailed in Section 2. The final_score Checklist v2 battery-rerun status is NOT asserted here — verify against the repo before citing it as done or pending. Authority: artifacts/forward_validation/CANDIDATE.json, docs/model_documentation.md.


### Interpretation Rules

- Serial correlation is heavy (5-day windows overlap across daily snapshots)
- 14 snap dates yields ~37 effective observations
- IC t-stats are indicative only, not promotion-grade
- Promotion requires Checklist v2 (full battery below)
- **Ranker IC is measured, not unmeasurable** — existing tools conflate composite_score with final_score (Spec 095). Spec 100 is implemented (commit 2faa88e6); the composite_score backtest still measures the wrong field, but ranker IC is measured by the forward_eval gate and the operative out-of-sample evidence is the forward-validation mandate — see the RECONCILED 2026-07-12 block above.



### IC Tool Field-Declaration Rule (standing prevention rule)

Origin: failure F-2026-002 (MA, CRITICAL) — `run_rank_ic_backtest.py` silently measured composite_score IC while it was assumed to measure production final_score IC, invalidating a pervasive body of IC claims. To prevent recurrence:

1. Every IC measurement tool MUST declare, in its output header/metadata, exactly which score field it measures (e.g. `composite_score` vs `final_score`) and over which universe (full eligible vs. top-N cohort).
2. No IC claim is valid unless the declared field matches the production sort key it is being used to justify. An IC number measured on a different field than the one in production is misattributed, not evidence.
3. When citing any IC figure, state the field and universe alongside it. A bare "IC = x" with no field + universe is non-promotion-grade by default.

---

## Checklist v2 Promotion Battery

**File**: `scripts/research/checklist_v2_rerun.py`

5-gate statistical bar. A signal must pass ALL gates for promotion.

### Gate 1: Fama-MacBeth (FM) Regression

Cross-sectional regression with Newey-West corrected standard errors.

- Positive t-stat required
- Controls for size, momentum, and sector

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
- **Any backtest that frames "DEM" as a separate baseline algorithm competing with the A4 selector** (F-2026-010, 2026-06-27) - DEM = the stored production top-30 of the current pipeline (A4 selector + clinical_50 ranker -> actionable_rank). Prior "DEM vs A4" comparisons contrasted stored-production-output against fresh-recompute variants, not two independent algorithms. See D-2026-008 in decision-audit-trail for the full architecture verdict.

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

### Insider Signal Status (Spec 104, 2026-05-14)

`insider_net_buy_value_90d` is DIAGNOSTIC ONLY. It is tracked in `DIAGNOSTIC_FIELDS` and explicitly excluded from `ALPHA_FEATURE_REGISTRY`. It does not enter the scoring model, ranker, or selector. The expectation model has an `insider_net_buy_z` weight that would activate silently if the field flowed into `market_features` -- an explicit isolation guard (Spec 104 R4a) prevents this.

Promotion to alpha requires ALL of: 20+ stable snapshots, >= 60% non-null coverage, IC > 0 at p < 0.05, Checklist v2 battery pass, and explicit written approval. Until all five are met, insider stays diagnostic. Do NOT evaluate insider IC for promotion purposes.

### Expectation Feature Coverage Prerequisite (Spec 105, 2026-05-14)

IC measurement on expectation-model signals (`short_interest_pct`, `close_price`, `market_cap_mm`, `priced_move_pct`) is only valid if those fields are actually flowing into the model at inference time. Spec 105 adds a production gate verifying:

1. All four fields present in `rankings.csv`
2. Per-field coverage above `FEATURE_COVERAGE_REQUIREMENTS` thresholds
3. Expectation model consumes these columns (not just exports them)

Any IC research on expectation-gap features against historical snapshots must verify the snapshot was post-wiring (or backfilled per Spec 102 with `_backfill_version` set). Pre-wiring snapshots that lack these fields are NOT valid for expectation IC measurement.

## coinvest_score_z IC Snapshot

*Last reviewed: 2026-06-25. Refresh after each 13F cycle and at scheduled checkpoints.*

| Window | n_dates | Mean IC | Hit Rate |
| --- | --- | --- | --- |
| Pooled (all) | 14 | -0.031 | 28.6% |
| Pre-cohort (clean) | 9 | -0.051 | 11.1% |
| Post-cohort (contaminated) | 5 | -0.008 | 60.0% |

**Verdict**: OBSERVE. April selloff drove pre-cohort negativity. Post-cohort recovering.

**IMPORTANT**: These IC figures measure coinvest_score_z across the full eligible universe. They do NOT measure ranker IC within the top-60 cohort (Spec 095 confirmed this gap on 2026-05-13). Ranker-specific IC is UNMEASURED until Spec 100 tooling is implemented.

### Completed Milestones (as of 2026-06-25)

- h20d horizon: PASSED (2026-05-26 — architecture freeze LIFTED)
- Post-Q1 2026 13F refresh: COMPLETE — all three Tier 1 files May 15, 2026. Cache warm + cohort quarantine done. 5-day observation window through ~May 22: COMPLETE.
- Forward shadow 30+ trading day evaluation: THRESHOLD REACHED (37+ trading days as of 2026-05-26)
- Spec 094 selector-only comparator rerun: target was 2026-05-27 — status PENDING operator confirmation

### Active Action Items (post-freeze)

- Spec 100 ranker IC tooling fix: UNBLOCKED. Checklist v2 battery rerun against final_score is highest-priority post-freeze action. NOT YET EXECUTED (~47 days post-freeze-lift as of Jul 12).
- SM-20260629-001 forward-validation mandate: UNBLOCKED Jul 10 (cron guard fix PR #489). Live captures now accumulating. 0/20 mandate-eligible windows as of Jul 10 — monitor liveness via `tools/cron_forward_validation_liveness.sh` (crontab install pending).
- DEM architecture framing: RESOLVED 2026-06-27 (D-2026-008). DEM = stored production top-30, not a separate algorithm. Any IC or backtest work comparing "DEM vs A4" must be reframed as stored-output vs fresh-recompute before being cited as evidence. See selector-ranker skill Dead Lanes table for closed research lanes (A4 overlay, inst_delta_z relaxation, cat_opt).

## Ranker IC Tooling Status (Spec 095 / Spec 100)

*Added: 2026-05-13. Updated: 2026-05-17.*



> SUPERSEDED 2026-07-12 (PR #492): the "battery rerun NOT executed / ranker IC remains effectively unmeasured / highest-priority action item" framing in this block is retired. Spec 100 tooling is implemented — see the RECONCILED 2026-07-12 block in Section 1. The final_score battery-rerun status is verify-against-repo, not assumed here.


The existing IC backtest tool previously measured the WRONG score. Spec 100 commit (2faa88e6, 2026-05-17) corrected the tooling — `run_rank_ic_backtest.py` now supports score-field and universe parameters with explicit metadata output. Prior composite_score IC claims are invalidated; a final_score baseline has been established.

Architecture freeze LIFTED 2026-05-26 (h20d checkpoint passed). Operationalization is now unblocked:

- Checklist v2 battery rerun against final_score has NOT been executed yet
- Until that rerun completes, ranker IC remains effectively unmeasured for promotion purposes
- No new ranker promotion decisions can be made until the battery rerun is done
- This is the highest-priority action item — execute now that freeze is lifted

## Forward Shadow Status

*Accumulating since: 2026-04-03. As of 2026-07-12, approximately 70+ trading days accumulated.*

At or past the 30-day evaluation threshold. Architecture freeze LIFTED 2026-05-26. Evaluate per the rules in Section 1.

**SM-20260629-001 mandate UNBLOCKED (2026-07-10):** A `set -e` bug in `cron_daily_production.sh` was silently killing the forward-validation tail (including `run_forward_validation.py`) whenever `run_daily_production.py` exited with code 2 (WARN-status runs, which began 2026-04-17). Fixed in PR #489 (merged Jul 10). Live captures now running. The 10 backfilled legacy rows correctly count as 0 mandate-eligible windows; SM-20260629-001 requires 20/40/52 genuine live captures. Liveness monitor (`tools/cron_forward_validation_liveness.sh`) added Jul 10 — not yet installed in crontab (operator deploy step pending).

**Model hash (AST-v1, 2026-07-10):** Raw-bytes model hash replaced with AST-normalized fingerprint (HASH_SCHEME=ast-v1, hash=827c35a9ed3ee6e1) so behavior-neutral edits (type annotations, docstrings) no longer flip the frozen DEM candidate hash. Operator must re-register candidate under 827c35a9ed3ee6e1.

## External IC and Alpha Benchmarks (May 2026)

### Earnings Call NLP Alpha

- FinBERT on 6.5M S&P 500 earnings call sentences (2015-2025): 2.03% monthly long-short alpha (t-stat = 6.49)
- Speaker-weighted and section-specific sentiment substantially outperforms aggregate sentiment
- Outperforms traditional Loughran-McDonald dictionary approach
- Remains significant after controlling for SUE (standardized unexpected earnings)
- Price assimilation of soft information is sluggish - consistent with ~1 week exploitation window
- RavenPack Q&A sentiment: 4.1% annual excess from analyst questions, 3.7% from management answers (IR 0.78/0.75, 8-day holding)
- Key limitation: General-purpose LLMs (GPT-4, Claude) struggle with nuanced financial language; domain-specific FinBERT outperforms on this task

### Regime-Dependent Signal Performance

- Multiple 2025-2026 studies confirm LLM-based signals are strongly regime-dependent
- Strong in stable markets, degraded in high-volatility regimes
- Regime-aware agentic framework (Springer 2026): +0.373 Sharpe improvement net of transaction costs
- Wasserstein HMM: Sharpe 2.18 vs 1.59 baseline, max DD -5.43% vs -14.62% SPX
- Implication: DEM signals likely exhibit similar regime sensitivity; regime detection should be evaluated as an overlay
- The T5 Ablation Protocol Gate 3 (13F quarantine) is a primitive form of regime conditioning

### Agentic Factor Investing Alpha

- Autonomous factor investing framework (arXiv 2603.14288): 3.11 Sharpe ratio, 59.53% annualized returns
- Agentic AI nowcasting (arXiv 2601.11958): 2.43 Sharpe ratio, 18.4 bps daily alpha on Russell 1000
- MarketSenseAI multi-agent S&P 500 strong-buy portfolio: +2.18%/month vs +1.15% passive (+25.2pp compound excess, p=0.003)
- Key finding: governance-constrained AI (strict out-of-sample validation, economic rationale requirements) outperforms unconstrained optimization - validates DEM's CCFT approach

### Construction Drag (Cross-Literature Finding)

Multiple research papers confirm AI has concentrated on stock selection while portfolio construction remains underserved. The DEM's own analysis identifies construction drag as the binding constraint. The walk-forward harness in Q3 2026 priority lane is correctly positioned as the gate for addressing this.
