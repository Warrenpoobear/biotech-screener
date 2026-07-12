---
name: operational-state
description: Live operational status; volatile, updated weekly or upon governance change
metadata:
  type: operational
  status: active
---

# Operational State

**Last updated:** 2026-07-12  
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

**MODEL BEHAVIOR REMAINS FROZEN. The 2026-06-24 "lift" was tooling-scoped only — do not read it as authorization to change the model.**

- **INC-2026-06-20 scoped freeze — narrowly lifted 2026-06-24:** operator authorized the lift **solely to unblock Spec 100** (ranker IC tooling correction). Spec 100 is implemented (commit `2faa88e6`); that lift is spent.
- **Currently binding (2026-07-12):**
  - **DEM candidate freeze + NO_MODEL_CHANGE forward-validation window** — `docs/FORWARD_VALIDATION_PROTOCOL.md` §1 (RATIFIED 2026-06-28; mandate SM-20260629-001, 0/20 eligible LIVE windows). No ranker/selector/feature/eligibility/sizing/rebalance-policy/price-source change; any authorized bug fix resets the out-of-sample clock.
  - **FROZEN (BLOCKED_LEVEL_0)** per `docs/model_documentation.md` (canonical; synced 2026-07-10) — model doc governs where this snapshot disagrees.
- **Candidate identity:** `model_hash=827c35a9ed3ee6e1` / `hash_scheme=ast-v1` (legacy `a9983a67c6954813`; `registered=2026-06-26` unchanged) — `artifacts/forward_validation/CANDIDATE.json`.
- `ranker_active_contract.py` (branch `hygiene/ranker-active-contract-2026-04-30`) remains deferred
- All prior ranker IC claims based on `composite_score` remain non-authoritative; ranker IC = `final_score` per Spec 100, interpretation deferred pending governance battery

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

**Next cycle (Q2 2026):** Period ending June 30, 2026. Filing deadline ~August 14, 2026. Monitor EDGAR starting ~August 11.

---

## Forward Shadow & IC Status

*Updated: 2026-06-24*

- **Forward shadow**: accumulating since 2026-04-03. Enough calendar/trading time has elapsed for IC refresh; gated on explicit operator freeze lift.
- **score_rank_pct IC** (weekly sweep 2026-06-24): mean_ic = **+0.0432**, hit_rate = 54.3%, N = 35 dates. Verdict: **HEALTHY**. This cleared the Path C IC gate and the Day-5 IC checkpoint.
- **Path C**: FORMALLY CLOSED 2026-06-24. IC gate met (mean_ic +0.0432 > +0.03). Catalyst-timing override policy remains in production weights; no reversion required. Transition to Path A durable gates.
- **IC checkpoint (Phase 2)**: Extended to 2026-07-01 (coincides with h20d re-eval gate). Next formal IC review: 2026-07-01. Early-review trigger: score_rank_pct mean_ic drops below 0.00 in any intervening daily snapshot.
- **coinvest_score_z IC** (last measured 2026-05-13): Pooled mean IC = -0.031 (14 dates). Refresh pending freeze lift.
- **Ranker IC**: UNMEASURED. Existing tools conflate composite_score with final_score (Spec 095). All prior ranker IC claims remain non-authoritative until Spec 100 reviewed.
- **inst_delta_z**: zeroed in selector since 2026-05-04. Active in ranker (NW-t = +3.32). Reinstatement requires IC recovery evidence.
- **PIT/backtest research (2026-06-22)**: Autonomous PIT/backtest research quarantined (PR #379). Not accepted evidence. Do not cite for model decisions without explicit operator acceptance.

## h20d Re-Evaluation Gate

*Due: 2026-07-01*

- Override granted 2026-05-26 (OPTION_B_OVERRIDE_2026_05_26) despite failed 13F Jaccard (0.463).
- Current metrics (2026-06-23): Filing coverage 100% ✓, signal coverage 85.22% ✓, Top-30 Jaccard 0.875 ✓. All observable metrics favorable.
- Q1 2026 13F data not yet promoted to production (quarantine script deferred). Run after first post-promotion snapshot:
  ```bash
  python3 tools/check_13f_cohort_quarantine.py \
      --pre-date <last-pre-promo> --post-date <first-post-promo>
  ```
- Override posture: **MAINTAINED** pending script confirmation.
- Hard gate: 2026-07-01. Failure triggers: Jaccard < 0.40 or coverage drop ≥ 10pp.

**Sequential gate (post-freeze):**
1. Explicit operator freeze lift
2. Run h20d quarantine script (2026-07-01 gate)
3. Warm Q2 13F cache via `tools/warm_13f_cache.py` (period ends Jun 30; filing deadline ~Aug 14)
4. Run cohort quarantine check
5. Refresh IC decomposition

---

## Active Spec Status

*Updated: 2026-06-24*

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
| 094 | Selector-only comparator | RANKER_UNPROVEN | Blocked by scoped freeze |
| 095 | Evaluation scope (IC tooling gap) | CURRENT_TOOLS_CONFLATED | Blocks ranker IC claims |
| 100 | Ranker IC tooling correction | Spec written, no impl | Blocked by scoped freeze |
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

*Last sync: 2026-05-30 · `python3 tools/sync_hermes_skills.py --register-meta`*

- **Agent registry:** 34 agents in `agents/AGENT_REGISTRY.json` (31 active + 2 deprecated + 1 shadow); 4 Hermes governance jobs including `hermes-contradiction-detector`
- **Registry:** 31 skills in `docs/hermes_skills/_meta.json` (16 Cursor `SKILL.md` mirrors + 3 `REFERENCE.md` + 12 Hermes-native)
- **Sync map:** `tools/sync_hermes_skills.py` · audit: `tools/audit_hermes_skills.py`
- **Authoritative (no overwrite from `skills/`):** `memory-steward` — canonical copy at `~/.hermes/skills/devops/memory-steward/`; repo backup `.hermes/skills/devops/memory-steward.SKILL.md`
- **Recent additions (since 2026-05-30):** `biotech-mcp` registered as read-only diagnostic MCP server (PR #27233ec6); daily state brief operational (PR #374)
- **weekly-skill-harvester:** MANUALIZED (Option C, PR #373) — no auto-push to main; operator reviews candidates before promotion. See `docs/governance/HARVESTER_MANUALIZATION_2026_06_22.md`.
- **Hermes-only sections preserved on sync:** Path C block in `screener-ops.md`
- **Town-Hermes bridge:** Phase B event wiring complete (2026-05-30, all event types); live email delivery awaits `OPERATOR_DELIVERY_DRY_RUN=0`
- Governance label: Tooling/knowledge-layer expansion only. No model, ranker, selector, sizing, or alpha-promotion change.

---

## Governance Events (2026-06-20 to 2026-06-24)

- **INC-2026-06-20-AUTOPUSH:** `weekly-skill-harvester` auto-committed to main in an unattended run. Harvester manualized (Option C, PR #373); pre-push hook added; Semgrep supply-chain rules deployed (PR #370); scoped production model freeze in effect. Closeout: `docs/incident/INC-2026-06-20-AUTOPUSH/`.
- **OpenClaw fenced (PR #372):** `**` wildcards and shell write permissions removed from exec-approvals. Exec allowlist now read-only. Hermes is primary orchestrator. Hard-retire deferred (natural on reboot). See `docs/governance/OPENCLAW_FENCE_DECISION_2026_06_22.md`.
- **Semgrep Phase 1 ERROR gate active (PR #376):** CI now blocks on Semgrep governance rule violations. Rules cover ranker/selector/final_score mutation guard, PIT annotation enforcement, model-output-as-signal guard, supply-chain checks.
- **Autonomous research quarantined (PR #379):** An autonomous PIT/backtest research run was quarantined. Outputs are not accepted evidence and must not be used for model decisions. See `docs/governance/AUTONOMOUS_RESEARCH_QUARANTINE_2026_06_22.md`.
- **Platform roadmap 9/9 COMPLETE (2026-06-23):** All operator-approved workstream items done. EES chain closed (validation→attribution→shadow monitor→guardrail design-only). Active path: daily shadow monitor run only. `OPENCLAW_STATUS: RETIRED`.
- **EES shadow monitor chain complete (2026-06-23):** Daily run operational. Gates unmet (need 20 completed 5d + 20 completed 20d exits). Observation-only, non-blocking. Script: `python3 scripts/research/ees_v2_phase3_shadow_monitor.py --as-of-date YYYY-MM-DD`.
- **Path C formally closed (2026-06-24):** IC gate met (score_rank_pct mean_ic +0.0432, hit_rate 54.3%, N=35). Catalyst-timing policy remains in weights; no reversion. Transition to Path A durable gates effective immediately.
- **Phase 2 agentic portfolio rebalanced (2026-06-24):** Robinhood account 802349084 rebalanced to 30 equal-weight positions (~$10.82 each, $324.50 total). 13 pending buys blocked by T+1 settlement; execute at open 2026-06-25. Operational rules at `production_data/AGENTIC_ACCOUNT_RULES.md`.

---

## What to Update After Every Session

- [ ] Current benchmark winner (Top-20 vs Top-30, any new candidate)
- [ ] Active heavy-lift job status
- [ ] 13F cycle status (next filing deadline, post-filing action items)
- [ ] Active spec status (newly resolved, newly blocked)
- [ ] Forward shadow & IC status (trading days accumulated, next checkpoint)
- [ ] Governance artifact status (PRs merged, enforcement layers pending)
