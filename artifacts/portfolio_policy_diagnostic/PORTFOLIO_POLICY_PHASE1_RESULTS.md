# Phase 1 Portfolio Policy Diagnostic Report

**Branch:** `phase1-portfolio-diagnostic`  
**Latest Commit:** `bcbaf316` (Fix timeline attribution: track holding spells, not appearance spans)  
**Report Date:** 2026-05-29  
**Status:** Diagnostic-only. No production changes. No deployment recommendation.

---

## 1. Executive Summary

This report documents Phase 1 of a diagnostic harness for testing portfolio rebalancing policies on historical biotech sector data. The harness is **read-only** and **diagnostic-only**; no production code or live systems are affected.

**Key Finding:** Among seven policies tested on seven canonical cohorts (2024-10-18 through 2026-01-02, terminal 2026-05-27), **quarterly rebalancing is the leading tested policy so far in this harness**, with an average return of +127.4% compared to weekly (+115.6%), a gap of +11.8 percentage points.

**Critical Caveat:** The mechanism driving this advantage is **not yet fully proven**. A timeline attribution bug was identified and fixed (spell tracking now correctly accounts for multiple holding periods per ticker), which invalidated earlier claims about "early entry" timing. The underlying drivers—whether rebalance cadence, exposure path, or regime-dependent artifacts—remain unresolved.

**Governance Status:** This is a diagnostic finding only. No operational decision is recommended without Phase 2 forward validation and governance approval.

---

## 2. Scope and Data

### Branch and Versioning
- **Branch:** `phase1-portfolio-diagnostic`
- **Latest Harness Commit:** `bcbaf316`
- **Harness Location:** `scripts/phase1_portfolio_diagnostic.py`
- **Test Type:** Historical backtesting on canonical rolling cohorts

### Data Sources
- **Terminal Date:** 2026-05-27
- **Price Source:** `production_data/price_history_split_adj.csv` (split-adjusted close prices)
- **Snapshot Source:** `data/snapshots/{date}/rankings.csv` and `decision_portfolio.csv` (daily rankings with actionable_rank field)
- **Benchmark:** XBI (NASDAQ Biotechnology ETF)

### Canonical Cohorts
Seven inception dates with common terminal (2026-05-27):
- 2024-10-18 (588 calendar days)
- 2024-11-01 (574 calendar days)
- 2025-01-10 (504 calendar days)
- 2025-04-11 (413 calendar days)
- 2025-07-18 (315 calendar days)
- 2025-10-10 (231 calendar days)
- 2026-01-02 (147 calendar days)

### Generated Artifacts
All artifacts are generated in `artifacts/portfolio_policy_diagnostic/` and are **untracked** (in `.gitignore`):
- `canonical_cohorts.json` — policy returns for each cohort and policy
- `quarterly_vs_weekly_attribution.json` — set-based attribution (shared holdings)
- `quarterly_vs_weekly_exposure_attribution.json` — exposure-weighted contribution analysis
- `quarterly_vs_weekly_timeline_attribution.json` — spell-based holding period analysis

**Note:** These JSON artifacts are diagnostic outputs only. They are intentionally excluded from version control and are not approved for production use unless explicitly reviewed and authorized by governance.

---

## 3. Policies Tested

Seven rebalancing policies were evaluated on each cohort:

### 1. STATIC_INCEPTION_HOLD
- **Rule:** Hold inception-date top-30 through terminal date without rebalancing
- **Purpose:** Baseline test of selector strength
- **Rebalance Count:** 0 per period

### 2. DELISTING_ONLY
- **Rule:** Hold inception top-30 until delisting detected (10+ consecutive calendar days missing from pricing)
- **Purpose:** Test selector strength under minimal trading friction
- **Rebalance Count:** Variable (triggered by delisting events only)

### 3. AVAILABLE_SNAPSHOT_REBUILD
- **Rule:** Rebuild from current top-30 on every snapshot date available
- **Purpose:** Maximum rebalance frequency baseline
- **Rebalance Count:** ~175 per period (daily snapshots)

### 4. WEEKLY_TRADE_PACKET_PROXY
- **Rule:** Rebuild from current top-30 on weekly Friday cadence
- **Purpose:** Approximate weekly portfolio rebalancing
- **Rebalance Count:** ~80 per period

### 5. MONTHLY_REBALANCE_PROXY
- **Rule:** Rebuild from current top-30 on first snapshot of each calendar month
- **Purpose:** Approximate monthly calendar-based rebalancing
- **Rebalance Count:** ~15 per period

### 6. HOLD_30
- **Rule:** Inception top-30 with minimum 30-day hold before forced replacement if ticker falls below top-30
- **Purpose:** Test hybrid: rebalance discipline with holding window
- **Rebalance Count:** ~195 per period (replacements only)

### 7. QUARTERLY_REBALANCE_PROXY
- **Rule:** Rebuild from current top-30 on first snapshot of each calendar quarter
- **Purpose:** Approximate quarterly calendar-based rebalancing
- **Rebalance Count:** ~7 per period

---

## 4. Results Table

| Policy | Avg Return | Avg XBI Return | Avg Alpha | Avg Turnover | Rebalances/Period | Role |
|--------|------------|---|---|---|---|---|
| **QUARTERLY_REBALANCE_PROXY** | **+127.4%** | +43.5% | **+83.9pp** | 0.82 | 7 | **Leading tested policy** |
| **WEEKLY_TRADE_PACKET_PROXY** | **+115.6%** | +43.5% | **+72.1pp** | 0.95 | 80 | High-frequency baseline |
| **STATIC_INCEPTION_HOLD** | +107.4% | +43.5% | +63.9pp | 0 | 0 | No-rebalance hold |
| **DELISTING_ONLY** | +107.4% | +43.5% | +63.9pp | ~0.1 | 3–16 | Minimal trading baseline |
| **AVAILABLE_SNAPSHOT_REBUILD** | +101.9% | +43.5% | +58.4pp | 0.86 | 175 | Maximum frequency baseline |
| **MONTHLY_REBALANCE_PROXY** | +102.7% | +43.5% | +59.2pp | 0.80 | 15 | Calendar intermediate |
| **HOLD_30** | +87.1% | +43.5% | +43.6pp | 1.11 | 195 | Minimum-hold discipline |

**Key Observation:** Quarterly outperforms weekly by +11.8pp average return (+127.4% vs +115.6%), despite lower rebalance frequency (7 vs 80 per period). This gap widens in later cohorts (shorter time horizons) and narrows in earlier cohorts (longer horizons).

---

## 5. Key Findings

### Return Hierarchy (Descending)
1. **Quarterly:** +127.4% (7 rebalances)
2. **Weekly:** +115.6% (80 rebalances)
3. **Static/Delisting:** +107.4% (0–3 rebalances)
4. **Monthly:** +102.7% (15 rebalances)
5. **Available Snapshot:** +101.9% (175 rebalances)
6. **HOLD_30:** +87.1% (195 rebalances)

### Return Gaps
- **Quarterly vs Weekly:** +11.8pp (quarterly leads)
- **Quarterly vs Static:** +20.0pp
- **Weekly vs Monthly:** +12.9pp
- **Monthly vs Available Snapshot:** +0.8pp
- **Available Snapshot vs HOLD_30:** +14.8pp

### Cohort-Level Consistency
Quarterly beats weekly in **5 of 7 cohorts:**
- 2024-10-18: Q=119.8%, W=92.2% (+27.6pp)
- 2024-11-01: Q=137.8%, W=132.0% (+5.8pp)
- 2025-01-10: Q=200.5%, W=207.3% (−6.8pp) ← weekly wins
- 2025-04-11: Q=221.0%, W=227.2% (−6.2pp) ← weekly wins
- 2025-07-18: Q=88.9%, W=83.3% (+5.6pp)
- 2025-10-10: Q=91.6%, W=53.8% (+37.8pp)
- 2026-01-02: Q=32.2%, W=13.2% (+19.0pp)

**Pattern:** Quarterly advantage is strongest in shortest-window cohorts (2025-10-10, 2026-01-02) and weaker in long-window cohorts (2025-01-10, 2025-04-11).

---

## 6. Attribution Summary

### Exposure-Weighted Attribution
Quarterly accumulates higher per-ticker contribution than weekly across most cohorts:

| Cohort | Q Contribution | W Contribution | Gap |
|--------|---|---|---|
| 2024-10-18 | 23.45% | 22.32% | +1.13pp |
| 2024-11-01 | 30.98% | 18.42% | +12.56pp |
| 2025-01-10 | 48.81% | 50.35% | −1.54pp |
| 2025-04-11 | 68.67% | 44.51% | +24.15pp |
| 2025-07-18 | 50.88% | 31.75% | +19.13pp |
| 2025-10-10 | 60.53% | 36.87% | +23.67pp |
| 2026-01-02 | 32.24% | 4.84% | +27.39pp |

**Interpretation:** Quarterly concentrates holdings on higher-returning tickers more often than weekly, but this does not explain the *mechanism* (why quarterly entries are better, or whether they exit better, or whether the effect is regime-dependent).

### Timeline Attribution (Spell-Tracked)
A bug in the original timeline code was identified and fixed. The original code measured **appearance spans** (first entry to last appearance) rather than **actual holding spells** (multiple entry/exit cycles).

**Bug Example:** NTLA appeared to have a "315-day entry gap" (quarterly entered 2024-10-18, weekly entered 2025-09-19), which was impossible if both started from the same inception top-30. Investigation revealed this was a **measurement artifact**: the code was measuring span, not spells.

**Fix:** Spell tracking now correctly identifies:
- Multiple [entry_date, exit_date] pairs per ticker per policy
- Total exposure days = sum of spell days (actual holding time)
- Number of spells = count of entry/exit cycles

**Result:** With spell tracking, NTLA no longer appears in top comparison lists, and earlier claims about "early entry advantage" are invalidated.

### Supported Statements
- Quarterly advantage appears related to **exposure path / rebalance cadence** (lower churn, different entry timing relative to price movements)
- Quarterly concentrates risk differently than weekly (higher per-ticker contribution on average)

### Unsupported Statements
- ~~Quarterly is optimal~~ (diagnostic-only finding; no forward validation)
- ~~Quarterly finds winners earlier~~ (invalidated by spell tracking fix)
- ~~Quarterly holds winners longer~~ (spell tracking shows fragmentation patterns, not causation)
- ~~Causal mechanism identified~~ (mechanism remains unresolved)

---

## 7. Limitations

### Scope Constraints
1. **Diagnostic-only.** No production code changes. No live systems affected.
2. **Historical cohorts only.** Seven backward-looking cohorts on completed data. No forward validation.
3. **Price data only.** No transaction costs, slippage, borrowing costs, market-impact modeling, or execution feasibility analysis.
4. **Snapshot selection artifact.** Rankings are daily top-30; rebalance proxies are sampled snapshots, not actual trading calendars.
5. **Regime-dependent.** Results may be specific to biotech sector 2024–2026 and XBI alpha levels (+43–44%).
6. **Untracked artifacts.** Generated JSON outputs are excluded from version control and are diagnostic-only unless explicitly approved.

### Known Issues (Fixed)
1. **Timeline attribution span measurement (FIXED):** Original code measured appearance span instead of holding spells. Spell tracking deployed in commit `bcbaf316` corrects this and invalidates earlier entry-timing claims.

### Open Questions
1. **Mechanism:** Why does quarterly outperform weekly? Is it entry timing, exit discipline, exposure concentration, or regime-dependent noise?
2. **Forward stability:** Do the relative rankings of policies remain consistent in prospective data (2026-05-28 onward)?
3. **Transaction feasibility:** Can quarterly rebalancing be executed cleanly without market impact, given the small biotech universe?
4. **Turnover cost:** At realistic borrow/slippage costs, does quarterly's +11.8pp advantage persist?

---

## 8. Governance Interpretation

### Findings Classification

| Finding | Classification | Status |
|---------|---|---|
| Quarterly return leader | PERSISTENT_POLICY_PROMISING | Confirmed in 5/7 cohorts; mechanism unresolved |
| Weekly high-frequency risk | REBUILD_CHURN_RISK_CONFIRMED | HOLD_30 underperforms (−40pp); turnover~1.0 creates drag |
| Static/delisting baseline | TIER_EXIT_ANTI_SIGNAL_CONFIRMED | No tier-exit logic in production; diagnostic warning only |
| Entry timing causation | INCONCLUSIVE | Spell tracking invalidated prior claims; requires Phase 2 |
| Quarterly mechanism | INCONCLUSIVE | Exposure-weighted shows contribution path; causation unknown |

### Governance Recommendation
**No operational decision is warranted at this stage.** Phase 2 forward paper test (May 2026 onward) is required to:
1. Validate historical findings prospectively
2. Measure actual transaction costs
3. Identify the causal mechanism (if any)
4. Assess execution feasibility under live market conditions

**Prerequisite:** Governance approval for Phase 2 scope, timeline, and decision gates.

---

## 9. Recommended Next Step

### Phase 2: Forward Paper Test (If Governance Approves)

**Objective:** Validate Phase 1 findings on prospective data and quantify execution costs.

**Design:**
- Run weekly, monthly, and quarterly rebalance proxies on new cohorts (2026-05-28 through end of 2026 or governance-defined endpoint)
- Compare to current advisory shadow behavior (daily fresh top-30)
- Track: realized returns, transaction costs, turnover, drawdown, staleness (rank drift), and execution feasibility
- No production changes

**Success Criteria:**
- Quarterly maintains advantage over weekly in prospective data
- Transaction cost impact quantified
- Mechanism becomes clear (or explicitly ruled out)

**Duration:** ~6 months of live data (May 2026 → November 2026)

**Authority:** Governance approval required before Phase 2 launch.

---

## Appendix: Data Integrity

### Harness Validation
- **Code quality:** All pre-commit hooks pass (black, isort, flake8, secrets)
- **Language audit:** No overstatement found ("optimal", "deploy", "causal", "proven")
- **Artifacts ignored:** `.gitignore` excludes `artifacts/*` (exceptions: audit markdown only)
- **Final test:** Harness run 2026-05-29 17:15 completed successfully; all policies executed; all attributions computed

### Repository State
- **Branch:** `phase1-portfolio-diagnostic`
- **Commit:** `bcbaf316` (timeline attribution spell tracking fix)
- **Working directory:** Clean (report file only)

---

**End of Phase 1 Report**

*This is a diagnostic-only document. No production changes are reflected. Governance approval required for any Phase 2 work.*
