# Decision Audit Trail

**Status:** ACTIVE (activated 2026-07-15)
**Created:** 2026-05-18
**Priority:** 4 of 7

## Purpose

Annotate key design decisions, parameter choices, and closed research lanes with their rationale, the evidence base at the time, and the conditions under which the decision should be revisited.

## Decision Entry Schema

```
decision_id: D-YYYY-NNN
date: YYYY-MM-DD
component: [skill or module name]
decision: One-line summary
rationale: Why this choice was made
evidence: What data/analysis supported the decision
alternatives_considered: What else was evaluated and why rejected
revisit_conditions: When this decision should be re-examined
related_specs: [spec numbers]
status: ACTIVE | SUPERSEDED | UNDER_REVIEW
```

---

## Catalog

### D-2026-001 | Gate 3 Penny Stock Threshold ($5.00)

- **Component:** biotech-validation (Gate 3)
- **Decision:** Exclude all tickers below $5.00 from the rankable universe
- **Rationale:** Not documented in any current skill. The W7 note explains the relationship between the $5.00 gate and the $2.00 financial-health penalty, but not why $5.00 was chosen.
- **Evidence:** MISSING. Presumably based on institutional liquidity constraints and SEC penny stock definitions.
- **Alternatives considered:** UNKNOWN.
- **Revisit conditions:** If the biotech universe shifts significantly in price distribution (e.g., many clinical-stage companies trading $3-5 after market correction), this gate could exclude otherwise legitimate candidates.
- **Status:** ACTIVE (needs rationale backfill)

### D-2026-002 | Construction Size K=30

- **Component:** selector-ranker
- **Decision:** Equal-weight top 30 names by final_score
- **Rationale:** Validated by PIT sweep (stable K=25-35 plateau, net-of-cost peak)
- **Evidence:** PIT sweep results exist but are not linked from the skill.
- **Alternatives considered:** K=20 (too concentrated, higher turnover), K=40 (dilution of conviction signal). RW-EW delta = -0.09pp showed rank-weighting does NOT help.
- **Revisit conditions:** If the rankable universe shrinks significantly (e.g., from coinvest-only selector reducing eligible names), K=30 may be too large. If universe expands, K=30 may be too small.
- **Status:** ACTIVE

### D-2026-003 | inst_delta_z Zeroing Threshold

- **Component:** institutional-signal, selector-ranker
- **Decision:** Zero inst_delta_z weight in selector (v1.14.0, 2026-05-04)
- **Rationale:** Mean IC = -0.097 over 36 dates, confirmed across two independent measurement frames
- **Evidence:** Two-frame IC confirmation. Negative IC means the signal was pointing the wrong direction.
- **Alternatives considered:** Reduce weight (e.g., 35% to 15%) instead of zeroing. Rejected because negative IC means any positive weight is actively harmful.
- **Revisit conditions:** Reinstatement requires IC recovery evidence documented in governance log. Signal remains active in ranker (NW-t = +3.32) where it operates within a different scope.
- **Status:** ACTIVE

### D-2026-004 | Contamination Window Duration (20 Trading Days)

- **Component:** institutional-signal
- **Decision:** After adding new managers, IC measurements during a 20-trading-day window are flagged as contaminated
- **Rationale:** Not explicitly documented. Presumably reflects time needed for score distribution to stabilize after manager addition.
- **Evidence:** MISSING. Was this calibrated empirically?
- **Alternatives considered:** UNKNOWN.
- **Revisit conditions:** If manager additions cause longer-duration score instability, or if clearance trigger (>=34/48 managers filed) interacts with contamination windows unexpectedly.
- **Status:** ACTIVE (needs rationale backfill)

### D-2026-005 | financial_score Negative Ranker Weight (Stress-Upside Thesis)

- **Component:** selector-ranker
- **Decision:** financial_score has weight -0.0533 in the production ranker, penalizing financially safe names
- **Rationale:** Intentional stress-upside thesis (Spec 074, reconfirmed Spec 093). Within the coinvest-selected universe, financially safe names are less catalytic.
- **Evidence:** Six-diagnostic audit confirmed TRUE PENALTY in both bull (NW-t = -3.42) and bear (-3.38) regimes. Persists across cohorts and regimes, ruling out artifact.
- **Alternatives considered:** Zeroing financial_score in ranker (would remove tilt). Using absolute value (would lose directional signal).
- **Revisit conditions:** If coinvest universe shifts toward earlier-stage cash-burning companies where financial health IS the binding constraint. Also revisit if bear-market drawdowns exceed historical bounds.
- **Status:** ACTIVE

### D-2026-006 | Dead Lane Closures (11 Research Lanes)

- **Component:** selector-ranker
- **Decision:** 11 research lanes closed/dead
- **Rationale and key learnings:**
  - *Options surface-shape (DEAD):* 50-month IC negative all horizons. Vol dominated by binary catalyst outcomes, not information flow.
  - *Options-as-alpha (Spec 053, CLOSED):* 37 signals tested, ALL fail. Options market for small-cap biotech too thin and binary-event-driven for continuous alpha.
  - *Static execution features (Spec 054, CLOSED):* All noise/destructive. Trade execution characteristics don't predict forward returns.
  - *Clinical composites as ranker (Spec 055, CLOSED):* Negative across ALL slices. Clinical stage doesn't predict relative performance within coinvest-selected universe (coinvest already incorporates clinical conviction).
  - *Fixed sleeve budgets (RETIRED):* Primary construction damage (+153.6pp drag). Forcing sector/stage allocation constraints destroyed the most value of any single design choice.
- **Revisit conditions:** Only with fundamentally new data (new options data source, or structural change to biotech market microstructure). Do not reopen on hope.
- **Status:** ACTIVE (all lanes remain closed)

### D-2026-007 | DEM Tier 6 Decision: Local LLMs Deferred

- **Component:** openclaw-agent-optimize
- **Decision:** Do not deploy local LLMs for agent inference at this time
- **Rationale:** Cost-performance analysis: local LLM breakeven vs frontier APIs is 3-6 months on DGX Spark; against cheap open-weight cloud APIs ($0.30/1M tokens), local never breaks even within hardware life.
- **Evidence:** Benchmark data in `local-llm-cost-vs-performance-agents-2026` (ai-projects). Qwen 2.5 Coder 32B scored 9.3/10 on tool calling but requires 24GB+ VRAM (current hardware: 16GB).
- **Alternatives considered:** DGX Spark purchase ($3K), cloud GPU rental, hybrid local+API routing
- **Revisit conditions:** If API costs increase significantly, if hardware costs drop, or if a breakthrough model fits in 16GB VRAM with comparable quality
- **Status:** ACTIVE



### D-2026-008 | DEM Is the Current Ranker; A4 / Inst-Relax / Cat-Opt Closed as DEM Proxies

- **Component:** selector-ranker
- **Date:** 2026-06-27
- **Decision:** Treat the stored production top-30 ("DEM") as the current ranker output, freeze the A4 selector overlay as a failed (anti-alpha) lane, and close A4-repair, inst_delta_z relaxation, and catalyst-optionality (cat_opt, incl. solvency variant) as DEM-proxy research lanes.
- **Rationale:** Architecture confirmed across snapshots: `actionable_rank` = `final_score` rank = A4 selector (`sel_score`) + clinical_50 ranker output. "DEM baseline" is NOT a separate algorithm — it is the stored top-30 of the full production pipeline (universe -> A4 selector -> clinical_50 ranker -> actionable_rank -> DEM top-30). The prior "DEM vs A4" framing compared stored production output against fresh recomputation variants (forward-filled inst_delta_z), not two independent algorithms. See F-2026-010.
- **Evidence:** 2025+ (n=14): DEM +7.898 pp/mo, t=2.167, hit 64.3% vs A4 +3.656 pp/mo, t=1.683 (DEM +4.2 pp/mo better). Full history (n=69): DEM +3.048, t=3.309 vs A4 +0.242. Downside worst-month: DEM -8.87 vs A4 -14.32 vs cat_opt -21.24. cat_opt did not recover DEM and worsened the left tail; cat_opt∩DEM overlap collapses post-2024. inst_delta relaxation failed to recover DEM. Concentration caveat: top-5 months (May/Jun/Jul 2025, Sep 2025, Feb 2026) = 100% of cumulative 2025+ alpha; DEM ex-best-5-months ≈ 0.
- **Alternatives considered:** A4 selector overlay (anti-alpha vs DEM — frozen); inst_delta_z relaxation (failed); catalyst-optionality selector + solvency variant (failed, worse left tail). All rejected as DEM proxies.
- **Revisit conditions:** DEM is a forward-shadow candidate, NOT investable production. Promotion gated on the DEM_REGIME_CONDITIONAL_ALPHA diagnostic — does the 2025+ edge survive outside the May–Jul 2025 biotech rally cluster, and is it alpha vs high-beta rally participation. Do not promote to capital on backtest strength alone. Do not reopen A4 / inst-relax / cat_opt without a new spec and fundamentally new evidence.
- **Related specs:** diagnostics DEM_CURRENT_RANKER_YTD_BACKTEST (2026-06-27); pending DEM_REGIME_CONDITIONAL_ALPHA
- **Status:** UNDER_REVIEW



### D-2026-009 | Verified Test-Hygiene Fixes Run In-Repo (Cursor Agent), Not Hand-Written in Sandbox

- **Component:** test-trust-audit
- **Date:** 2026-07-14
- **Decision:** For any authorized remediation of test-trust-audit findings, add/repair assertions via a Cursor background agent operating in the real repo, and require every modified test to be run to green. Do not commit assertions hand-written in the sandbox.
- **Rationale:** The sandbox can run the auditor (pure AST, no deps) but cannot import the biotech-screener package or load data fixtures, so any assertion authored there is unverified. Committing unverified assertions into a CI-red repo manufactures exactly the fake-green the audit exists to catch.
- **Evidence:** 2026-07-14 run — the sandbox tree held only `tools/` + `tests/` + `governance/` (no source package dirs), so return contracts could not be read and no test could execute. The Cursor agent `bc-6cfd427b` read each function's contract and ran each of 42 modified tests to green (PR #497); the 12 it could not verify were reverted and listed.
- **Alternatives considered:** Hand-write assertions in the sandbox and commit (rejected — unverified; violates honest-assertion discipline). Trust the detector and skip runtime verification (rejected — that is the failure mode under audit).
- **Revisit conditions:** If the sandbox gains the full package + fixtures and can run the suite, in-sandbox verified fixes become acceptable. Also revisit if Cursor repo access changes.
- **Related specs:** PR #496, PR #497; F-2026-011
- **Status:** ACTIVE

### D-2026-010 | Frozen-Path Exclusion for Test Edits by Code Inspection, Not Filename

- **Component:** test-trust-audit
- **Date:** 2026-07-14
- **Decision:** When excluding tests that touch frozen scoring / ranker / selector / sizing / final_score / PIT paths from an edit pass, decide by reading what each test actually exercises — not by matching filenames.
- **Rationale:** A filename filter silently under-excludes: innocuously named tests exercise frozen modules. Relying on names alone would let edits land on frozen-path tests and breach the freeze.
- **Evidence:** 2026-07-14 run — the filename filter flagged only 4 tests to exclude; code inspection (by the Cursor agent) caught 7 additional frozen-exposed tests it missed: `test_defensive_integration` (×3), `test_golden_baseline` (×2), `test_minimum_suite`, and `test_run_screen_units::test_live_source_mode` — exercising `module_5_composite` / `module_2_financial` / position-sizing despite benign names.
- **Alternatives considered:** Filename/path-prefix filter only (rejected — proven to under-exclude). Conservatively exclude whole directories (rejected — over-excludes and leaves real hollow tests unfixed).
- **Revisit conditions:** If the repo adds reliable per-test frozen-path markers/metadata a mechanical filter can trust, revisit the manual-inspection requirement.
- **Related specs:** PR #497; F-2026-011
- **Status:** ACTIVE

---

## Usage Rules

1. When a parameter or threshold is questioned, check this catalog first.
2. If no entry exists, flag the decision as needing rationale backfill (mark `evidence: MISSING`).
3. When conditions change (market regime shift, universe expansion, new data source), scan revisit_conditions for affected decisions.
4. SUPERSEDED decisions should retain their full history -- do not delete, only change status.
