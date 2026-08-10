# Held-Spec Ledger — 2026-08-10

Generated: 2026-08-10T12:50:02Z (cron, hermes-held-spec-ledger)
Prior ledger: artifacts/audit/held_spec_ledger_2026_08_03.md
Evidence window checked: 2026-08-03 → 2026-08-10 (gate_verdict_ledger, phase2_health,
scorecard_2026-08-10, EES v3 shadow status 08-04/08-05/08-06/08-07, WEEKLY_SIGNAL_REGIME_SWEEP_2026_08_09,
forward_eval_ic_baseline.json, git log/status, AGENT_REGISTRY.json, CHANGELOG.md tail,
specs/changes/spec_114/spec_115/spec_087_b2)

## Overall readiness snapshot
- Gate ledger (2026-08-10, ruleset 8887576e): n_fail=0, n_pass=39, n_warn=3, overall_status=WARN.
  Chronic WARNs unchanged: `regulatory_calendar`, `ruleset_health`; new/recurring `pnl_attribution` WARN
  (first seen 08-07, still WARN 08-10). `working_tree_clean` gate=PASS at 2026-08-10T00:00:00Z generation
  time — **contradicted by live `git status -s`** run this cycle (6 uncommitted paths, see §4). Not
  treated as a gate defect; flagged as a timing gap (gate snapshot predates later same-day file touches).
- Weekly scorecard (2026-08-10): **HOLD**. Blocking check `bucket_drift_vs_policy` FAIL at **31.67pp**
  (91-180d bucket 23.3% vs 55% policy, Δ31.7pp) — improved from 08-03 ledger's 35.0pp print but still an
  active blocking FAIL, three consecutive HOLD verdicts now. `pre_trade_gate` WARN (`alpha_health`) persists.
- phase2_health (2026-08-10): WARN, reason `headwind_exposure` (headwind_weight_pct=36.63%, portfolio_size=30).
  Distinct driver from 08-03 ledger's `catalyst_7d_count_high` — that reason has cleared
  (catalyst_le_7d_weight_pct now 0.0%); new driver is fresh, not a carryover.
- forward_eval_ic_baseline.json: **unchanged**, still stale Path C window (2026-05-27→06-03), single
  observation, IC_UNOBSERVABLE_EXPECTED. Now >2 months stale on this artifact specifically; gate ledger's
  `forward_eval: PASS` remains the more current signal (unchanged convention from prior ledger).
- score_rank_pct signal health: **fresher evidence found** — `WEEKLY_SIGNAL_REGIME_SWEEP_2026_08_09.md`
  reports HEALTHY, mean IC +0.0401, hit rate 82.05% (N=39), no WARN/ALERT signals. This resolves the
  08-03 ledger's staleness concern for this branch.
- EES v3 raw_veto_core shadow: 20d gate remains MET (66 obs as of 08-07 memo, up from 62 on 08-01). Alpha+
  rate and mean 20d alpha continued to **decline**: 83.1%→81.0%, +7.0pp→+6.4pp (08-01→08-07). Trend now
  three consecutive prints down from the 07-28 peak (88.5%/+7.9pp). Still FREEZE_ACTIVE/DIAGNOSTIC_ONLY;
  no operator promote/hold/reject ruling found in commits or memos 08-03→08-10.
- **New AGENT_REGISTRY evidence**: `hermes-first-fire-validator` (Spec 089 Phase 2), `hermes-held-spec-ledger`
  (Spec 089/090 Phase 2), and `hermes-contradiction-detector` (Spec 090 Phase B) all now present with
  `status: active`, cron-enabled, observe_only authority. This is further-phase implementation beyond the
  08-03 ledger's "Spec 089 Phase 1" note — governance/ops tooling lane, no scoring/ranking/selector surface.
- Universe-hygiene / `production_data/universe.json` modification flagged NEEDS_OPERATOR_DECISION in the
  08-03 ledger is **still uncommitted, unresolved, now spanning the full window** — see §4.
- Q1/Q2 2026 13F like-for-like retest: due ~2026-08-14, now **4 days out** (was 11 days out in 08-03
  ledger). No new artifact found confirming a scheduled/completed run.

---

## 1. ACTIONABLE BLOCKERS

| Item | Trigger | Status | Action required |
|---|---|---|---|
| EES v3 raw_veto_core production promotion | Gate MET since 2026-06-25 (46 days); no operator ruling found through 08-07 memo | NEEDS_OPERATOR_DECISION — longer overdue, and underlying alpha metrics now declining 3 prints running (88.5%→83.1%→81.0% alpha+ rate) | Operator must issue explicit promote/hold/reject ruling; review the declining trend as part of the decision, not as noise |
| bucket_drift_vs_policy FAIL | 35.0pp (07-31) → 31.67pp (08-10); 91-180d bucket actual 23.3% vs 55% target | ACTIVE FAIL, blocking trade (scorecard=HOLD, 3rd consecutive) | Portfolio/policy review of 91-180d bucket allocation; note partial improvement but still failing |
| Universe.json + universe_hygiene_audit (06-28-dated files) uncommitted modification | First flagged 08-03 ledger; still modified in `git status -s` this cycle, no commit/memo found in 6+ weeks | NEEDS_OPERATOR_DECISION — **escalated**, unresolved for a full ledger cycle with no explanation | Operator to confirm this is a benign in-place regeneration vs. an uncommunicated universe change before it ages further |
| Spec 114 freeze-lift decision | Shadow/non-routing halves merged 07-29/07-30; routing half still DRAFT | NEEDS_OPERATOR_DECISION (unchanged) | Operator to rule explicitly on freeze lift for the routing-affecting half (resets OOS clock per spec) |
| Spec 115 Phase 2 full remediation (2a vs 2b) + Phase 3 | Phase 1 + partial Phase 2 (visibility-only) landed 07-29; no further commits found 08-03→08-10 | NEEDS_OPERATOR_DECISION (unchanged) | Operator to choose 2a (relocate evidence ledger) vs 2b (commit-after-capture); do not start Phase 3 early |
| Q2 2026 13F like-for-like re-test | Due ~2026-08-14 (4 days out) | Scheduled, imminent | Confirm run occurs on schedule; check next ledger cycle |
| Spec 087 B2 dashboard freshness envelope disposition | B1b closed 05-14 / re-confirmed CLOSED 07-18; B2 unresolved since ≥07-13 ledger | STILL UNRESOLVED — no new evidence 08-03→08-10 | Operator to confirm formally whether B2 is ACTIVE; transitively blocks Spec 088 |

---

## 2. HELD BRANCHES

### Spec 087 B1b — first-fire validation
- Status: **CLOSED** (unchanged)
- Last evidence: `artifacts/audit/spec_087_b1b_formal_closure_2026_05_14.md`; commit `f9b5ad76` (07-18) confirms stale ledger entries cleared with real evidence
- Blocker: none
- Next allowed action: none
- Explicitly not allowed: reopening without new operator directive
- Runtime risk: NONE
- Alert condition: n/a

### Spec 087 B2 — dashboard freshness envelope
- Status: **NEEDS_OPERATOR_DECISION** (unchanged, now stale ≥4 weeks)
- Last evidence: `specs/changes/spec_087_b2_dashboard_freshness_envelope.md` (Status: PHASE B2 DESIGN, ready for implementation planning); no commit found 08-03→08-10 referencing activation
- Blocker: no explicit operator ruling that B1b closure unblocks B2 implementation
- Next allowed action: operator review + written authorization to transition to ACTIVE
- Explicitly not allowed: silent promotion to ACTIVE
- Runtime risk: LOW
- Alert condition: any commit touching dashboard freshness logic without a spec-status update

### Spec 087C — bioshort alpha research
- Status: HELD (needs ≥4 fresh weekly hedge reports) — **count not independently re-verified this cycle**, marked `[?]`
- Last evidence: `spec_087c_phase_b_disposition_2026_05_14.md` — no newer evidence found
- Blocker: weekly hedge-report accumulation
- Next allowed action: continue accumulation; verify count next cycle
- Explicitly not allowed: production activation
- Runtime risk: NONE
- Alert condition: 4th fresh weekly report landing

### bioshort_watch LLM reactivation
- Status: **CLOSED / SUPERSEDED** (unchanged) — AGENT_REGISTRY.json confirms `status: active`, reactivated 2026-07-18, cron ID `9b7546acf514`, sunset_review_date `2026-09-30` (~51 days out)
- Last evidence: AGENT_REGISTRY.json (re-verified 2026-08-10)
- Blocker: none
- Next allowed action: monitor cadence
- Explicitly not allowed: n/a
- Runtime risk: LOW
- Alert condition: sunset review 2026-09-30 approaching, or cron gap >~24 days

### Spec 088 Phase B — catalyst_delta filtered artifacts
- Status: HELD — blocked on Spec 087 B2 close (unchanged)
- Last evidence: `specs/changes/spec_088_phase_a_catalyst_delta_filter_design_2026_05_07.md`; no new commits touching `tools/build_catalyst_delta.py` filtering logic found 08-03→08-10
- Blocker: Spec 087 B2 still NEEDS_OPERATOR_DECISION
- Next allowed action: none until 087 chain resolved
- Explicitly not allowed: any artifact-level filtering change to production catalyst_delta tool
- Runtime risk: LOW
- Alert condition: any commit touching `tools/build_catalyst_delta.py` filtering logic

### watchlist_current.json disposition
- Status: **NEEDS_OPERATOR_DECISION** (carried forward, **not independently re-searched this cycle** — marked `[?]`)
- Last evidence: negative search result, 2026-08-03 (prior ledger)
- Blocker: unclear whether removed/renamed artifact or stale doc reference
- Next allowed action: operator to confirm scrub-vs-regenerate
- Explicitly not allowed: fabricating or regenerating without operator direction
- Runtime risk: LOW
- Alert condition: any code path reading this path at runtime and failing silently

### score_rank_pct SPEC_REQUIRED
- Status: HELD (gate does not expire); IC health **re-verified this cycle — HEALTHY**
- Last evidence: `WEEKLY_SIGNAL_REGIME_SWEEP_2026_08_09.md` — mean IC +0.0401, hit rate 82.05% (N=39), no WARN/ALERT
- Blocker: CRT + IC + PIT + Checklist v2 all still required before any weight change; none evidenced
- Next allowed action: continue monitoring; IC print is current (08-09, 1 day old)
- Explicitly not allowed: any scoring/ranking/weight change without full evidence chain
- Runtime risk: LOW (improved — fresher print than prior cycle)
- Alert condition: any commit touching `score_rank_pct` logic without full evidence chain; OR next sweep print showing degraded IC

### EES v3 veto_monitor (raw_veto_core) — production promotion decision
- Status: HELD (shadow) — gate MET 2026-06-25, still FREEZE_ACTIVE/DIAGNOSTIC_ONLY as of 08-07 memo
- Last evidence: `artifacts/readiness/EES_V3_RAW_VETO_SHADOW_STATUS_2026_08_07.md` — 20d gate MET (66/20), cumulative alpha+ rate 81.0%, mean 20d veto alpha +6.4%
- Blocker: explicit operator approval for production promotion; none found through 08-10
- Next allowed action: operator reviews freeze-lift memo + current shadow status; issue explicit ruling
- Explicitly not allowed: any change to `mutation_authority` or enabling production decisioning without sign-off
- Runtime risk: LOW (diagnostic-only) — **watch: alpha+ rate/mean-alpha declining 3 consecutive prints**
- Alert condition: any commit flipping `production_decisioning` to true or removing FREEZE_ACTIVE without an operator-attributed commit

### Spec 113 — construction signal PIT snapshots
- Status: HELD — **DRAFT**, unchanged
- Last evidence: `specs/changes/spec_113_construction_signal_pit_snapshots_2026_07_05.md`; no new commits found 08-03→08-10
- Blocker: operator approval to move to IN PROGRESS
- Next allowed action: none until approval
- Explicitly not allowed: any implementation commits against this spec
- Runtime risk: NONE
- Alert condition: any commit referencing spec_113 implementation without a preceding approval record

### Spec 114 — CT.gov date-precision provenance
- Status: HELD — freeze-lift-requiring (routing) half is NEEDS_OPERATOR_DECISION; shadow/non-routing half CLOSED (merged) — unchanged
- Last evidence: commits `5d475997` (07-29), `577d1549` (07-30); spec file still DRAFT for the routing portion; no new commits 08-03→08-10
- Blocker: operator freeze lift for the routing half (resets OOS clock, n=4 mandate-eligible windows / 52-window gate)
- Next allowed action: operator decision on whether cost (OOS reset) justifies fixing routing defect now vs. deferring
- Explicitly not allowed: any change to `catalyst_decay_w`/`catalyst_tilt_mult`/`target_weight_pct` routing logic without freeze lift
- Runtime risk: LOW (non-routing capture inert to production scoring)
- Alert condition: any commit modifying routing/precision-to-tilt logic without an operator-attributed freeze-lift commit

### Spec 115 — protect mandate evidence from git-state interference
- Status: HELD — Phase 1 + Phase 2(visibility) CLOSED (merged); Phase 2(full) + Phase 3 NEEDS_OPERATOR_DECISION — unchanged
- Last evidence: commit `dd13e097` (07-29, PR #540); no new commits 08-03→08-10
- Blocker: operator choice between 2a (relocate evidence ledger outside git) and 2b (commit-after-capture); Phase 3 sequenced after 1+2 observed working
- Next allowed action: operator decision on 2a vs 2b; do not start Phase 3 early
- Explicitly not allowed: any cron job push capability (hard invariant); any evidence-ledger relocation without sign-off
- Runtime risk: MEDIUM (unchanged — Mode B exposure window remains open until 2a/2b lands)
- Alert condition: any further mandate-eligible window loss to git mechanics; any uncommitted-capture WARN naming new dates

### Q1 2026 13F cohort — quarantine
- Status: **CLOSED — decision-grade** (unchanged); Q2 2026 retest due ~2026-08-14, now **4 days out**
- Last evidence: `artifacts/readiness/H20D_JACCARD_FINAL_2026_07_17.md`
- Blocker: none currently — retest imminent
- Next allowed action: Q1 2026 13F consumable as decision-grade until Q2 retest supersedes
- Explicitly not allowed: n/a
- Runtime risk: NONE
- Alert condition: Q2 2026 retest date (~2026-08-14) arriving without a scheduled run

### Spec 089 — Hermes Knowledge Layer (Phase 2 now evidenced) / Spec 090 Phase B
- Status: **ACTIVE** — Phase 1 (ledger generator) previously confirmed; **Phase 2 now evidenced** via AGENT_REGISTRY entries `hermes-first-fire-validator` and `hermes-held-spec-ledger` (both `status: active`, cron-enabled); Spec 090 Phase B evidenced via `hermes-contradiction-detector` (`status: active`)
- Last evidence: `agents/AGENT_REGISTRY.json` (read 2026-08-10) — no corresponding commit SHA independently confirmed this cycle, flagged `[?]`
- Blocker: none identified this window
- Next allowed action: continue phased implementation per spec; confirm scope of any further phases with operator
- Explicitly not allowed: scope expansion beyond documented phases without spec update
- Runtime risk: LOW (ops/governance tooling, observe_only authority, not scoring/ranking/selector surface)
- Alert condition: any commit under these specs touching ranking/scoring/selector logic (out of stated scope)

---

## 3. RECENTLY CLOSED (since 2026-08-03 ledger)

| Item | Closure evidence | Date |
|---|---|---|
| Spec 087 B1b re-confirmation (stale ledger entries cleared with real evidence) | commit `f9b5ad76` | 2026-07-18 (pre-existing, newly reviewed this cycle) |
| score_rank_pct IC-health staleness concern | `WEEKLY_SIGNAL_REGIME_SWEEP_2026_08_09.md` — fresh HEALTHY print | 2026-08-09 |
| phase2_health `catalyst_7d_count_high` driver | cleared — `catalyst_le_7d_weight_pct` now 0.0% (new driver `headwind_exposure` open) | 2026-08-10 |

No closures this window touch scoring/ranking/selector logic per evidence reviewed.

---

## 4. UNCOMMITTED WORKING TREE

| Path | Type | Disposition |
|---|---|---|
| `artifacts/universe_hygiene/xbi_ibb_universe_audit_2026_06_28/*` (6 files) | Modified | **Unresolved, escalated** — flagged in 08-03 ledger, still modified 7 days later with no commit/memo explaining the change |
| `production_data/universe.json` | Modified | **Unresolved, escalated** — same window as above; universe composition is scoring-adjacent; still no explanation found |
| `production_data/market_data.json` | Modified | Routine daily-refresh pattern — low risk |
| `production_data/options_snapshot_latest.json` | Modified | Routine daily-refresh pattern — low risk |
| `production_data/trial_records.json` | Modified | Routine daily-refresh pattern — low risk |
| `production_data/options_snapshot_2026-08-10.json` | Untracked (new) | Routine dated snapshot — low risk, pending commit |
| `"data/snapshots - Shortcut.lnk"` | Untracked (new) | **Anomalous** (unchanged from 08-03 ledger) — Windows shortcut file inside a tracked data path; recommend operator delete, do not commit |

**Escalation note:** the `universe.json` + universe-hygiene-audit modification cluster is now unresolved across a full ledger cycle (≥7 days) with no commit or memo — escalating from "flag" to "actionable blocker" this cycle (see §1).

---

## 5. RECOMMENDED NEXT OPERATOR DECISIONS (rank-ordered, max 5)

1. **Explain and resolve the `production_data/universe.json` + universe-hygiene-audit (06-28-dated files) uncommitted modification.** Now unresolved across two full ledger cycles (≥7 days) with no commit/memo; universe composition is scoring-adjacent.
2. **Rule on EES v3 raw_veto_core production promotion.** Gate MET for 46 days; alpha+ rate/mean-alpha have declined for 3 consecutive prints (88.5%→83.1%→81.0%; +7.9pp→+7.0pp→+6.4pp) — review trend as part of the decision.
3. **Resolve `bucket_drift_vs_policy` FAIL.** Improved 35.0pp→31.67pp but still an active blocking FAIL (3 consecutive HOLD verdicts); portfolio/policy review of 91-180d bucket allocation needed.
4. **Rule on Spec 114 freeze lift and Spec 115 Phase 2 (2a vs 2b).** No movement since 07-29/07-30; both decisions gate further remediation and carry OOS-clock/exposure-window costs the longer they sit open.
5. **Confirm Spec 087 B2 dashboard freshness envelope disposition in writing.** Unresolved since ≥07-13 ledger (now ≥4 weeks), transitively blocking Spec 088.

---

*Ledger produced by read-only tooling (run_readonly_diagnostics, gate_verdicts, phase2_health, forward_eval_ic_ledger) plus terminal git/file inspection. No production files were modified. No scoring, ranking, or selector logic was inspected for change — only status/read. Ledger and its published copies are the only files written by this run.*
