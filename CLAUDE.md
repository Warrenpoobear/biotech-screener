# CLAUDE.md — Wake Robin Capital Management Biotech Screener

## Project Identity
This is an institutional-grade biotech investment screening system.
Outputs must be reproducible, auditable, and deterministic.
Every decision must be traceable to a data source with a timestamp.

## North Star Rule
Backtest systems NEVER directly modify production screening behavior.
They produce evidence and proposals only. Governance review required before
any backtest finding changes a production signal weight.

## CCFT Principles (Non-Negotiable)
All data fixtures must be:
- Canonical: single authoritative source per data type
- Complete: no silent nulls or missing fields without explicit flags
- Frozen: historical snapshots are immutable once written
- Timestamped: data_available_timestamp <= as_of_date always enforced

## Active Ruleset
- **ID**: `2a3e79eb` (v1.13.0)
- **File**: `production_data/decision_rulesets/v1.13.0_a4_selector_ranker.json`
- **Key settings**: sort_anchor=selector_score, B6 selector (coinvest 65% + inst_delta 35%), pairwise_minimal ranker (ordinal-only), EW Top-30
- **Prior ruleset**: `69a0c7f8` (v1.12.0) — RETIRED 2026-04-03
- **Pinned in**: `run_screen.py` AND `run_phase2_snapshot_delta.py` (must stay in sync)
- **Manifest**: 36+ entries, no dup IDs

---

## Current Operating Truths

Spec 050 (2026-04-03) replaced the old optionality-anchored selector with a two-stage
selector/ranker architecture. Checklist v2 rerun (2026-04-04) revalidated the live stack
under the Spec 055 statistical bar (FM, bootstrap, FDR, LOSO).

> **Production mental model: coinvest selects, inst_delta ranks, financial penalizes
> "safe but less catalytic" names, and clinical is a weak/conditional feature under review.**

1. **B6 selector + pairwise_minimal ranker (ordinal-only) + EW Top-30 is production.** True PIT backtest: +2.34pp/mo net-of-cost, t=2.57, 69% hit rate, 67 monthly periods (Jun 2020 — Apr 2026).
2. **B6 selector validated under Checklist v2.** Bootstrap: +2.42pp/mo, 95% CI [1.25%, 3.70%], P(>0)=99.99%. LOSO: ROBUST across all dimensions. Neither component survives standalone, but the bundle is real.
3. **Selector and ranker learn different structure.** B6 (coinvest 65% + inst_delta 35%) selects which 30 names. Within top-30: inst_delta is the dominant positive discriminator (NW-t=+3.32), financial_score is a true negative penalty (NW-t=−3.41), coinvest washes out (+0.49).
4. **Pairwise ranker is ordinal-only.** ECE=0.129 (POOR calibration). No rank-weighting, no confidence sizing. Equal-weight is the correct construction.
5. **EW Top-30 is the correct construction.** RW-EW = -0.09pp, t=-0.95. Pairwise calibration confirms.
6. **K=30 validated by sweep.** Net-of-cost peak at +2.34pp, stable K=25-35 plateau.
7. **Bear/neutral alpha engine.** Bear: +3.37pp (75% hit), neutral: +6.23pp (93% hit), bull: -0.37pp (50% hit). Worst months are all bull regime.
8. **event_type_score is the only 5/5 Checklist v2 pass.** Use as overlay/diagnostic/sizer only — does NOT improve B6 bundle.
9. **insider_exec and aact_execution downgraded.** Both 1/5 under Checklist v2. Shadow only.
10. **Forward shadow accumulating daily** (7 arms in coinvest_shadow_tracker v2, wired into run_daily.py).

---

## Trust Buckets

### Safe to use now (production-grade evidence)
- **B6 selector + pairwise_minimal ranker (ordinal-only) + EW Top-30**: true PIT validated, t=2.57, 67 periods
- **B6 bundle revalidated under Checklist v2** (2026-04-04): bootstrap CI [1.25%, 3.70%], LOSO ROBUST
- **Pairwise ordinal-only policy**: ECE=0.129, no rank-weighting or confidence sizing
- Selector engine (`selector_engine.py`), ranker engines (`ranker_engine.py`, `ranker_v2_pairwise.py`): 48+ tests
- Statistical QA package (`common/stats/`): FM, bootstrap, FDR, LOSO, calibration — 36 tests
- PIT validation audit framework, PIT financial regeneration infrastructure
- K=30 validated by sweep (stable K=25-35 plateau)
- Forward shadow tracker (7 arms, wired into daily cron)
- event_type_score as overlay/diagnostic (5/5 Checklist v2 pass, but not selector weight)
- Event EV promotion ladder infrastructure (Spec 061): 4-stage default-OFF ladder wired into decision engine + run_screen.py injection

### Deprecated (do not cite)
- **All survivorship-only benchmark numbers** (+93.7pp, +110.5pp, etc.)
- **Old optionality-anchored selector** — underwater on PIT data (-25pp cumulative)
- **DEFAULT selector weights** (clinical 35%, catalyst 25%) — destructive as selector (-0.53pp)
- **clinical_score_v2_z as selector anchor** — negative delta (-0.68pp), universally destructive
- **Pre-Checklist-v2 signal card t-stats** — superseded by FM/bootstrap/FDR/LOSO findings
- **insider_exec_buy_value_90d optimistic reads** — 1/5 under Checklist v2, FRAGILE
- **aact_execution_score optimistic reads** — 1/5 under Checklist v2, bear-unstable
- Any promotion memo citing pre-Spec-050 selector performance
- "Bear IR 3.35" regime story from contaminated data

### Current evidence hierarchy
1. **Checklist v2 rerun (2026-04-04)**: B6 bundle bootstrap+LOSO validated — STRONGEST (for signals)
2. **True PIT backtest (Spec 050)**: A4+ranker +2.34pp net, t=2.57 — STRONGEST (for portfolio)
3. **Pairwise feature audit (2026-04-04)**: within-top-30 FM on ranker features — SUPPORTING
4. **Forward shadow**: accumulating daily since 2026-04-03 — MONITORING
5. **Old PIT benchmark (Spec 048)**: optionality selector underwater — SUPERSEDED by new selector

---

## Do Not Reopen Without New Evidence

These lanes have been tested and either died or were superseded. Do not spend research
hours here unless genuinely new data or a structural model change creates a reason to revisit.

| Lane | Status | Why closed |
|------|--------|-----------|
| Options surface-shape as systematic ranker | DEAD | 50-month backtest IC negative at all horizons |
| Options-as-alpha (Spec 053) | CLOSED | 37 signals tested, ALL fail as selector/ranker |
| Static execution features (Spec 054) | CLOSED | PCD overdue, update recency, pipeline velocity all noise/destructive |
| Clinical composites as ranker (Spec 055) | CLOSED | Negative across ALL robustness slices, universally destructive |
| `total_volume_z` | DEAD | IC=-0.10 on PIT-native data (109 obs) |
| Always-on rank-weighting (Top-20 or Top-30) | NOT PROMOTED | RW-EW = -0.09pp; pairwise ECE=0.129 confirms ordinal-only |
| Confidence/rank-weighted sizing | NOT JUSTIFIED | Pairwise scores not calibrated (ECE=0.129) |
| `insider_exec_buy_value_90d` | SHADOW ONLY | 1/5 Checklist v2, FRAGILE robustness |
| `aact_execution_score` | SHADOW ONLY | 1/5 Checklist v2, bear-unstable (−1.86pp) |
| Top-20 / pruner promotion story | DEPRECATED | PIT-financial correction shows both underwater vs XBI |
| Historical alpha narrative (+93pp / +110pp) | DEPRECATED | Inflated by financial look-ahead contamination |
| `cal_alpha` | REMOVED in v1.12.0 | Confirmed no-op, zero deltas at all horizons |
| Clinical sort signal | OFF | Insufficient IC, destructive as selector |
| Coinvest as standalone sort signal | SUPERSEDED | Now used as B6 selector anchor; standalone only 3/5 Checklist v2 |
| Quality tiebreaks (Specs 030/031) | EXHAUSTED | All economically immaterial |
| 91-180d drawdown gate | DEAD | Counterproductive at all thresholds |
| Dynamic caps | DEAD | Identical to plain EW |
| Fixed sleeve budgets | RETIRED | Primary construction damage mechanism (+153.6pp drag) |

---

## Current Promotion Story

1. **B6 selector + pairwise_minimal ranker (ordinal-only) + EW Top-30 is ADOPTED** (2026-04-03, revalidated 2026-04-04).
2. True PIT evidence: +2.34pp/mo net, t=2.57, 69% hit, beats XBI on return and risk.
3. **B6 bundle passes Checklist v2**: bootstrap +2.42pp/mo, CI [1.25%, 3.70%], LOSO ROBUST. Bundle > parts.
4. **Pairwise ordinal-only confirmed**: ECE=0.129. Do not rank-weight or confidence-size.
5. **Within-cohort roles clear**: coinvest selects, inst_delta ranks, financial penalizes safe names.
6. **event_type_score**: 5/5 Checklist v2 but overlay only — does not improve B6 bundle.
7. **Forward shadow is the validation layer.** 7 arms accumulating daily. Evaluate after 30 trading days.
8. **K=30 is validated** by PIT sweep (stable K=25-35 plateau, net-of-cost peak).
9. **Regime caveat**: this is a bear/neutral alpha engine. Expect bounded underperformance in strong bull.
10. The governance hold (Spec 048) **succeeded**: it prevented the old optionality selector from being institutionalized on contaminated data, which led to finding the better B6 selector.
11. **Event EV promotion ladder (Spec 061)**: infrastructure implemented (2026-04-08). 4 stages (tiebreaker → rank overlay → sizing overlay → composite), all default OFF. Wired into decision engine + run_screen.py. Requires forward evidence + governance review before activation.

---

## PIT Rules

1. **Never call the historical set "true PIT"** unless archived raw inputs, archived code, AND archived derived artifacts all exist as-of each date.
2. Historical benchmark outputs must carry `pseudo_pit_version` (1=contaminated, 2=cleaned).
3. Benchmark reruns must use the PIT-aware paths: `--pit-mode survivorship` or `--pit-mode full`.
4. Long-history conclusions are **provisional** until PIT-v2 financial rerun lands.
5. The forward monitor is the only true out-of-sample evidence. Accumulate it.

---

## Canonical Benchmark Commands

```bash
# Survivorship-cleaned selection benchmark (current baseline)
python3 scripts/research/build_selection_benchmark.py --pit-mode survivorship --top-n 20 --also-top30

# Monthly IC / selection benchmark
python3 scripts/research/selection_benchmark.py --pit-mode survivorship

# Ranker evaluation (inst_delta_z within top-30)
python3 scripts/research/ranker_evaluation_harness.py --signal inst_delta_z --pit-mode survivorship

# Construction v2 benchmark (all variants)
python3 scripts/research/construction_v2_benchmark.py --pit-mode survivorship

# PIT-financials snapshot regeneration (heavy lift, ~2h)
python3 scripts/research/regenerate_pit_v2_snapshots.py

# Run benchmarks on PIT-financial-corrected snapshots
python3 scripts/research/build_selection_benchmark.py --pit-mode survivorship --top-n 20 --also-top30 --snapshot-dir data/snapshots_pit_v2
```

---

## Heavy-Lift Jobs

- **PIT financial regeneration is COMPLETE.** 76 monthly dates in `data/snapshots_pit_v2/`, 72/72 OK, 0 errors.
- **Result: historical alpha collapsed.** All pre-correction claims are deprecated.
- **Next heavy lift: forward monitor accumulation.** No compute needed — just time. Evaluate after 30+ trading days of true-PIT daily production.
- **Event EV forward accumulation**: daily artifacts in `artifacts/event_ev/` via Step 5k.21. EV fields now injected into rankings.csv (Spec 061). Evaluate readiness via `event_ev.promotion_ladder.evaluate_ev_readiness()`.
- **If forward evidence is positive:** re-establish selector thesis from clean data. Do not backfill from historical.
- **If forward evidence is negative:** the selector needs structural re-examination.

---

## What to Update After Every Session

- [ ] Current benchmark winner (Top-20 vs Top-30, any new candidate)
- [ ] Trust bucket changes (provisional → safe, or new invalid entries)
- [ ] Dead-lane list (add any newly killed signals/lanes)
- [ ] PIT version / contamination status
- [ ] Active heavy-lift job status

## Decision Engine Architecture (v1.5.0)

**Core files:**
- `decision_engine.py` — L0 gates → L2 overlays → L4 tiers → L3 sizing → sort key + Event EV ladder (Spec 061)
- `selector_engine.py` — B6 selector (5 blocks, coinvest+inst dominant)
- `ranker_v2_pairwise.py` — pairwise_minimal ranker (6 features, ordinal-only)
- `ranker_engine.py` — clinical_50 ranker (legacy/fallback, bounded ±15%)
- `event_ev/promotion_ladder.py` — EV stage enum, bucket classifier, cohort loader, z-scores, readiness evaluator

**Pipeline flow:**
```
Modules 1-5 → Decision Engine (gates, tiers, sizing)
           → Selector Engine (B6: coinvest 65% + inst_delta 35%)
           → Ranker Engine (pairwise_minimal: 6 features, top-60 cohort, ordinal-only)
           → Event EV injection (Spec 061: load artifacts → z-score → bucket → inject)
           → Sort by final_score → EW Top-30 → rankings.csv
```

**Sort anchor:** `selector_score` (uses `final_score` = ranker_v2_score for cohort members)
All downstream consumers use `actionable_rank` (now driven by selector/ranker, not composite_rank).

**Statistical QA:** `common/stats/` (6 modules), `scripts/research/checklist_v2_rerun.py`

## Promotion Governance
- **Manifest**: `production_data/decision_rulesets/manifest.json` — all rulesets tracked with status (active/candidate/retired)
- **Promotion battery**: `scripts/research/run_promotion_battery.py` → bucketed verdicts + weekly live-sim → PASS/FAIL
- **Promote script**: `scripts/promote_ruleset.py` — blocks promotion unless battery PASS
- **Health monitor**: `tools/ruleset_health_monitor.py` — post-promotion drift detection
- **Rollback**: `scripts/promote_ruleset.py --rollback --reason "..."` — first-class with auto-LKG discovery

## Event Ledger & Cache Warming
- **Event ledger**: `build_event_ledger()` in `event_ledger.py` — 7+ sources (CTGov, merged trials, SEC 8-K, SEC multi-form, FDA ADCOM, FDA regulatory, PDUFA manual, EMA)
- **Cache warmer**: `warm_caches.py --sources sec_8k,ctgov,sec_13f,fda_adcom,fda_regulatory,euctr,ctis,isrctn,merged_trials`
- **EU/EEA registries**: `euctr_collector.py`, `ctis_collector.py`, `isrctn_collector.py` in `wake_robin_data_pipeline/collectors/`
- **Trial merger**: `trial_registry_merger.py` — cross-registry dedup by NCT/EudraCT IDs
- Always warm 8-K cache BEFORE running screen

## Daily Production Pipeline
- **Runner**: `tools/run_daily_production.py` — 13-step orchestrator
- **Cron**: 5:30 PM ET weekdays + `@reboot` catch-up for missed runs
- **Steps**: price refresh → cache warm (incl. FDA) → screen (with `--inputs-manifest write`) → audit → gates → manifest + promotion → drift report → action packet → shadow portfolio → trade plan → portfolio report → readiness scorecard → ops digest → PIT backfill (optional)
- **Ops digest**: `tools/build_ops_digest.py` → `artifacts/ops_digest/YYYY-MM-DD_digest.md` — single-screen actionable summary
- **Readiness**: `tools/weekly_readiness_scorecard.py` → READY / REVIEW / HOLD verdict
- **Health checks**: collection health (INFO/WARN/FAIL with weekend-safe price fallback), phase-2 health, exposure metrics

## OpenClaw Ops Agent
- **Workspace**: `agents/ops/` — SOUL.md (boundaries), TOOLS.md (daily working set), HEARTBEAT.md (3-check)
- **Role**: read-mostly operator — runs pipeline, reads digest, surfaces action items, refuses to modify rulesets
- **Gateway**: 127.0.0.1:18789, loopback only, auth via setup token

## Shadow Portfolio
- **File**: `tools/live_shadow_portfolio.py` (902 lines)
- **Policy**: `production_data/portfolio_policy.json` (v3), $500k, 55/25/10/10 bucket split
- **Family sleeves**: REGULATORY/CLINICAL split per bucket with time-ladder sub-buckets
- **Regulatory sleeve A/B**: +1.85pp 63d, +1.59pp 84d (positive but coverage-limited)

## Before Writing Any Code
1. State which module this change belongs to
2. Identify whether this is a new signal, validation change, or infrastructure change
3. Write the failing test FIRST — show me the red test before any implementation
4. Confirm no look-ahead bias: what is the data_available_timestamp?

## Coding Standards
- All outputs: encoding='utf-8', lineterminator='\n', quoting=csv.QUOTE_MINIMAL
- SHA256 hash every scored output for audit trail
- Identical inputs must produce byte-identical outputs — no random seeds, no datetime.now()
- Use Point-in-Time fixtures — never fetch live data in tests

## What NOT To Do
- Do not refactor and add features in the same commit
- Do not change production agent weights without an ablation test showing Sharpe delta
- Do not use PubMed h-index API, options flow, or CapIQ — see approved data sources
- Do not introduce survivorship bias — graveyard list is at data/graveyard/

## Test Requirements
Every new signal must include:
1. Unit test with known fixture input → expected output
2. Leakage test confirming data_available_timestamp compliance
3. Ablation test stub showing Sharpe contribution ≥ 0.1

## Long-Call Contract Recommendations (Post-Screen)

When producing long-call candidates from the screen output, also recommend the best executable long-call contract for each surviving candidate.

**Goal:** For every name that passes the long-call filter, recommend:
1. One primary contract
2. One backup contract
3. Or explicitly mark `NO_TRADE` if no contract is liquid / priced well enough

Do NOT just say "buy calls." Pick an actual strike + expiry from the chain data available in the repo/output.

### Step 1 — Expiry selection
- Base case: choose the first liquid expiry that is AFTER the catalyst date and still leaves 14–35 calendar days of cushion after the event
- If catalyst_days is 21–45: allow tighter post-event cushion of 7–21 days
- Avoid expiries that occur BEFORE the catalyst
- Avoid very long expiries unless all nearer expiries are illiquid or the event date is uncertain
- Prefer standard monthly expiries over odd weeklies when liquidity is similar

### Step 2 — Strike selection
- Target call delta between 0.30 and 0.50
- Higher-conviction names: prefer 0.40–0.50 delta
- Lower-conviction / higher-IV names: prefer 0.30–0.40 delta
- Avoid ultra-OTM lottery strikes unless premium is tiny and liquidity is still acceptable
- Avoid deep ITM unless spread/liquidity is clearly superior and thesis is very high conviction

### Step 3 — Liquidity filter
Reject contracts if any of these are true:
- open_interest is too low
- volume is too low
- bid/ask spread is too wide
- pricing looks stale

If the repo does not have exact spread fields, use the best liquidity proxies available and state the limitation.

### Step 4 — Entry economics
For each candidate contract, compute or estimate:
- mid premium
- breakeven move to expiry
- event-date implied move
- crush-adjusted move if available
- delta
- DTE

Prefer contracts where:
- directional thesis is confirmed by RR / skew
- implied move is not already extreme
- the contract still has room to profit after likely post-event IV compression
- premium at risk is reasonable relative to conviction

### Step 5 — Rank contracts
Choose the primary contract by this priority:
1. Expiry appropriately covering the catalyst
2. Strongest liquidity
3. Delta in target band
4. Best breakeven vs thesis
5. Cleaner spread / execution quality

Choose one backup contract that is either:
- one strike lower/higher with similar expiry, or
- next best expiry with similar delta profile

### Output format for each candidate
```
ticker:
  catalyst: <event_type> in <N> days
  thesis: <1-2 lines>
  primary_contract:
    expiry:
    DTE:
    strike:
    option_type: CALL
    delta:
    premium_or_mid:
    open_interest:
    volume:
    spread_or_liquidity_proxy:
    breakeven_move_pct:
    why_this_contract:
  backup_contract:
    <same fields>
  no_trade_reason: <if applicable>
```

### Important constraints
- If exact contract-chain data is unavailable from the snapshot alone, look for the nearest chain artifact/cache already produced by the repo for that date
- If the contract recommendation depends on missing chain fields, say so explicitly and give the best constrained recommendation possible
- Do not change DEM scoring or ranking logic
- This is a post-screen execution recommendation only

## Key File Locations
| Area | File |
|------|------|
| Main orchestrator | `run_screen.py` |
| Decision Engine | `decision_engine.py` |
| Selector Engine | `selector_engine.py` |
| Ranker Engine | `ranker_engine.py` |
| Calendar Alpha | `common/clinical_calendar_alpha.py` |
| Options Provider | `common/options_history_massive.py` |
| Daily Production | `tools/run_daily_production.py` |
| Shadow Portfolio | `tools/live_shadow_portfolio.py` |
| Trade Plan | `tools/build_trade_plan.py` |
| Portfolio Report | `tools/build_portfolio_report.py` |
| Readiness Scorecard | `tools/weekly_readiness_scorecard.py` |
| Ops Digest | `tools/build_ops_digest.py` |
| Collection Health | `tools/build_data_collection_health.py` |
| Hedge Report | `tools/biotech_hedge_report.py` |
| Promotion Battery | `scripts/research/run_promotion_battery.py` |
| Signal Evidence | `scripts/run_signal_evidence.py` |
| Ruleset Manifest | `production_data/decision_rulesets/manifest.json` |
| Cache Warmer | `warm_caches.py` |
| Event Ledger | `event_ledger.py` |
| Event EV Ladder | `event_ev/promotion_ladder.py` |
| Cron Wrapper | `tools/cron_daily_production.sh` |
| Ops Agent Workspace | `agents/ops/` |
