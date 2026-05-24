---
name: operational-state
description: Live operational status; volatile, updated weekly or upon governance change
metadata:
  type: operational
  status: active
---

# Operational State

**Last updated:** 2026-05-22  
**Update cadence:** weekly or upon governance-state change  
**Authority:** operational status snapshot only; not architectural specification  
**Staleness risk:** MEDIUM (reference `governance.md` for timeless process)

---

## Active Ruleset

- **ID**: `8887576e` (v1.14.0)
- **File**: `production_data/decision_rulesets/v1.14.0_coinvest_only_selector.json`
- **Key settings**: sort_anchor=selector_score, coinvest-only selector (coinvest_score_z 100%), pairwise_minimal ranker (ordinal-only), EW Top-30. inst_delta_z zeroed in selector 2026-05-04 (ALERT: mean_ic=-0.097, two-frame confirmed).
- **Prior ruleset**: `2a3e79eb` (v1.13.0) â€” RETIRED 2026-05-04
- **Pinned in**: `run_screen.py` AND `run_phase2_snapshot_delta.py` (must stay in sync)
- **Manifest**: 36+ entries, no dup IDs

---

## Architecture Freeze Status

- **v1.14.0 freeze in effect** until post-h20d checkpoint (~2026-05-26)
- No new enforcement logic or scoring changes until then
- `ranker_active_contract.py` exists on unmerged branch (`hygiene/ranker-active-contract-2026-04-30`), deferred to post-freeze
- Manual spot-checks via snapshot_integrity verifier in the interim
- Spec 100 (ranker IC tooling correction) is highest-priority code change post-freeze

---

## Heavy-Lift Jobs

- **PIT financial regeneration is COMPLETE.** 76 monthly dates in `data/snapshots_pit_v2/`, 72/72 OK, 0 errors.
- **Result: historical alpha collapsed.** All pre-correction claims are deprecated.
- **Next heavy lift: forward monitor accumulation.** No compute needed â€” just time. Evaluate after 30+ trading days of true-PIT daily production.
- **If forward evidence is positive:** re-establish selector thesis from clean data. Do not backfill from historical.
- **If forward evidence is negative:** the selector needs structural re-examination.

---

## 13F Cycle Status (Q1 2026 â€” COMPLETE)

*Updated: 2026-05-16*

- **All three tracked managers filed Q1 2026 13F-HR on deadline day (May 15, 2026)**
- Accession numbers: Fairmount 0001104659-26-062419, Deep Track 0001856083-26-000003, Logos Global 0001172661-26-002196
- CIKs: Fairmount 0001802528, Deep Track 0001856083, Logos Global 0001792126
- Filing pattern: all three filed on deadline day, consistent with Q1 2025 pattern (all May 15, 2025)

**Key changes:**
- **Fairmount**: Added DAMORA THERAPEUTICS ($225.7M, 16.3% of portfolio â€” largest new, NOT signaled by 13D/13G). Massive APGE trim (-85.4%), COGT trim (-38.9%). Exits: KINIKSA, NUVALENT. VRDN held (3.9M shares at 3/31). Post-Q1: VRDN stake raised to 14.04% via $20M purchase May 11.
- **Deep Track**: AUM $6,124M (+9.2%). 63 positions (was 55). 16 new positions including ALMS ($149M), NUVL ($141M), GMAB ($98M), DFNT ($57M). Exits: DVAX ($242M largest). VRDN: 1.4M shares at 3/31, accumulated to 5.4M post-Q1 per 13G.
- **Logos Global**: AUM $2,003M (+21.0%). 66 positions. Massive CNTA add (+963%, now $84.4M). New: UTHR ($47M), MDGL ($44.5M), XENE ($26M). 15 exits including CDTX ($68.5M).
- **Top coinvest**: VRDN (FM 14.04% + DT 5.30%) entering Ph3 TED readout. ORKA coinvest (FM + DT). Triple overlap on CRESCENT BIOPHARMA only. DT+Logos 22 overlaps.

**13D/13G pre-signal validation**: ~60-70% of major moves captured pre-filing. Largest surprises (DAMORA for Fairmount, CNTA for Logos) were invisible until 13F-HR.

**Post-filing action sequence**: (1) Warm 13F cache, (2) Run cohort quarantine, (3) Check collapse guards (coinvest_score_z SD), (4) Refresh IC decomposition, (5) 5-day observation window.

**Next cycle**: Q2 2026 (period ending June 30, 2026). Filing deadline ~August 14, 2026. Monitor EDGAR starting ~August 11.

---

## Forward Shadow & IC Status

*Updated: 2026-05-16*

- **Forward shadow**: accumulating since 2026-04-03. ~30+ trading days as of mid-May. Approaching or at evaluation threshold.
- **coinvest_score_z IC** (last measured 2026-05-13): Pooled mean IC = -0.031 (14 dates, 28.6% hit rate). Pre-cohort (clean): -0.051 (11.1% hit). Post-cohort (contaminated): -0.008 (60.0% hit). Verdict: OBSERVE.
- **Ranker IC**: UNMEASURED. Existing tools conflate composite_score with final_score (Spec 095). All prior ranker IC claims are misattributed. Blocked until Spec 100.
- **inst_delta_z**: zeroed in selector since 2026-05-04. Active in ranker (NW-t = +3.32). Reinstatement requires IC recovery evidence.

**Sequential gate (CRITICAL):**
1. Wait for Q1 2026 13F filing (completed 2026-05-15)
2. Warm 13F cache via `tools/warm_13f_cache.py`
3. Run cohort quarantine check
4. Then refresh IC decomposition

---

## Active Spec Status

*Updated: 2026-05-16*

### Recently Resolved
| Spec | Title | Status | Commit |
|------|-------|--------|--------|
| 093 | financial_score sign direction | RESOLVED (INTENTIONAL_STRESS_UPSIDE) | 2026-05-13 audit |
| 101 | Runway Severity v1.1 Export Contract | CLOSED | eaa4ea87 + cba4ee0f |
| 087 B0 | Stale-Propagation Guard | CLOSED | 0f0c7952 |
| 087 B2 | Dashboard Freshness Envelope | CLOSED | 400a6cd9 |
| 088 B | Catalyst Delta v2 Filter Companion | SHIPPED | 5ca4b033 |

### Active / Blocked
| Spec | Title | Status | Blocker |
|------|-------|--------|---------|
| 094 | Selector-only comparator | RANKER_UNPROVEN | Rerun target 2026-05-27 |
| 095 | Evaluation scope (IC tooling gap) | CURRENT_TOOLS_CONFLATED | Blocks ranker IC claims |
| 100 | Ranker IC tooling correction | Spec written, no impl | Architecture freeze (~May 26) |
| 104 | Insider diagnostic stabilization | MEASURED | Isolation guard (R4a) |
| 105 | Expectation layer coverage verification | CODE-CLOSED | Pending live QA |
| 102 | Historical backfill for expectation research | DRAFT | â€” |

### Monitoring
| Spec | Purpose | Gate | Next Review |
|------|---------|------|-------------|
| 096 | Gate/ranker separation doctrine | Defines promotion paths | Ongoing |
| 097 | Event-EV prospective monitoring | Brier <= 0.08, n >= 30 | Monthly |
| 098 | Catalyst timing prospective monitor | Correlation > 0.15 | Monthly |
| 099 | Clinical orthogonality audit | Pre-promotion gate | Before clinical promotion |

---

## Hermes Skills Hub — Sync State

*Last sync: 2026-05-24*

- **biotech-screener `f3ab726b`** — 15 reference docs in `docs/hermes_skills/`
- **hermes-agent `654172c06`** — 7 new live skills installed:
  - `biotech-screener/`: dossier-generation, validation
  - `software-development/`: browser-automation
  - `autonomous-ai-agents/`: self-improving
  - `investment-frameworks/` (new category): pe-pacing, sfo-liquidity-architecture, spending-liquidity
- **Canonical 7 biotech hub skills preserved** (sizes unchanged): catalyst-resolution, clinical-scoring, financial-health, ic-evaluation, institutional-signal, screener-ops, selector-ranker
- Gateway reload and smoke checks passed (dossier-generation + validation)
- `validation` skill includes Spec 104 insider isolation guard
- Governance label: Tooling/knowledge-layer expansion. No model, ranker, selector, sizing, or alpha-promotion change.
## What to Update After Every Session

- [ ] Current benchmark winner (Top-20 vs Top-30, any new candidate)
- [ ] Active heavy-lift job status
- [ ] 13F cycle status (next filing deadline, post-filing action items)
- [ ] Active spec status (newly resolved, newly blocked)
- [ ] Forward shadow & IC status (trading days accumulated, next checkpoint)
- [ ] Governance artifact status (PRs merged, enforcement layers pending)
