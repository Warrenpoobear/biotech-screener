---
name: operational-state
description: Live operational status; volatile, updated weekly or upon governance change
metadata:
  type: operational
  status: active
---

# Operational State

**Last updated:** 2026-06-24  
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

**SCOPED PRODUCTION MODEL FREEZE — effective 2026-06-20 (INC-2026-06-20-AUTOPUSH)**

- **Frozen:** ranker, selector, sizing logic, `final_score`, portfolio/snapshot files, gate configs
- **Safe lanes:** expectation field verification, Event EV shadow, Sci-Cart artifacts, observability/diagnostics, Hermes read-only queries
- h20d checkpoint has passed; original v1.14.0 structural freeze superseded by the scoped freeze above
- `ranker_active_contract.py` (branch `hygiene/ranker-active-contract-2026-04-30`) remains deferred
- Spec 100 (ranker IC tooling correction) remains highest-priority post-freeze code change
- **Lift condition:** explicit operator clearance

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

*Updated: 2026-06-22*

- **Forward shadow**: accumulating since 2026-04-03. As of 2026-06-22, enough calendar/trading time has elapsed for a future IC refresh, but refresh is gated on explicit operator freeze lift.
- **coinvest_score_z IC** (last measured 2026-05-13): Pooled mean IC = -0.031 (14 dates, 28.6% hit rate). Pre-cohort (clean): -0.051 (11.1% hit). Post-cohort (contaminated): -0.008 (60.0% hit). Verdict: OBSERVE. Refresh pending freeze lift.
- **Ranker IC**: UNMEASURED. Existing tools conflate composite_score with final_score (Spec 095). All prior ranker IC claims remain non-authoritative until corrected tooling is reviewed (Spec 100).
- **inst_delta_z**: zeroed in selector since 2026-05-04. Active in ranker (NW-t = +3.32). Reinstatement requires IC recovery evidence.
- **PIT/backtest research (2026-06-22)**: Autonomous PIT/backtest research was quarantined on 2026-06-22 (PR #379) and is **not accepted evidence**. Do not cite performance numbers or use outputs for model decisions until manually reviewed and explicitly accepted by operator.

**Sequential gate (post-freeze):**
1. Explicit operator freeze lift
2. Warm Q2 13F cache via `tools/warm_13f_cache.py` (period ends Jun 30; filing deadline ~Aug 14)
3. Run cohort quarantine check
4. Refresh IC decomposition

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

## Governance Events (2026-06-20 to 2026-06-22)

- **INC-2026-06-20-AUTOPUSH:** `weekly-skill-harvester` auto-committed to main in an unattended run. Harvester manualized (Option C, PR #373); pre-push hook added; Semgrep supply-chain rules deployed (PR #370); scoped production model freeze in effect. Closeout: `docs/incident/INC-2026-06-20-AUTOPUSH/`.
- **OpenClaw fenced (PR #372):** `**` wildcards and shell write permissions removed from exec-approvals. Exec allowlist now read-only. Hermes is primary orchestrator. Hard-retire deferred (natural on reboot). See `docs/governance/OPENCLAW_FENCE_DECISION_2026_06_22.md`.
- **Semgrep Phase 1 ERROR gate active (PR #376):** CI now blocks on Semgrep governance rule violations. Rules cover ranker/selector/final_score mutation guard, PIT annotation enforcement, model-output-as-signal guard, supply-chain checks.
- **Autonomous research quarantined (PR #379):** An autonomous PIT/backtest research run was quarantined. Outputs are not accepted evidence and must not be used for model decisions. See `docs/governance/AUTONOMOUS_RESEARCH_QUARANTINE_2026_06_22.md`.

---

## Agent Fleet Self-Learning (2026-06-24)

*Updated post-PR #397/#398, phase-2 branch*

- **Heartbeat:** 27 specialized checks in `tools/agent_heartbeat_checks.py`; Hermes on-demand jobs SKIP by design.
- **Telemetry:** `log_agent_run` on daily builders + `data_auditor` + `ops_supervisor` + Hermes `run_job.py` exits.
- **Outcome feedback:** all daily builders + qa/supervisor/herald/auditor/postmortem/CRT/event_binder/Hermes exits (policy_shadow overlap ≥80%, universe zero stale-alert, grok zero high alerts, etc.).
- **Herald recovery:** `tools/herald_recovery.py` / `herald_recovery.sh`; `herald_health_check.py --recover` for F-2026-005.
- **Cron integration:** evening catchup fully deterministic; watchdog phase-2 recovery uses builders (not LLM); `install_agent_fleet_crontab.sh` for WSL install.
- **Rule 12:** `docs/governance/RULE_12_PROMOTION_CHECKLIST.md` — weekly workflow + stalled-loop gates.
- **Stalled loops:** F-2026-005 (Herald host recovery), F-2026-006 (Actions budget) — `SELFIMPROVE_GATES_MET=1` blocked until closed.

---

## What to Update After Every Session

- [ ] Current benchmark winner (Top-20 vs Top-30, any new candidate)
- [ ] Active heavy-lift job status
- [ ] 13F cycle status (next filing deadline, post-filing action items)
- [ ] Active spec status (newly resolved, newly blocked)
- [ ] Forward shadow & IC status (trading days accumulated, next checkpoint)
- [ ] Governance artifact status (PRs merged, enforcement layers pending)
