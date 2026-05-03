# Governance Log Entry — inst_delta_z disposition

**Status:** FILED. Decision recorded; no production change applied.

---

## Decision Record

**Date:** 2026-05-03
**Decision-maker:** Darren Schulz / Wake Robin Capital Management
**Trigger:** ic_health_monitor SIGNAL_ALERT on inst_delta_z, fleet receipt 2026-05-01

**Active ruleset before decision:** `2a3e79eb` (v1.13.0)
**Active ruleset after decision:** `2a3e79eb` (v1.13.0) — unchanged

## Evidence Reviewed

- `INST_DELTA_Z_GOVERNANCE_REVIEW_2026_05_03.md` — facts, options, comparator behavior
- `INST_DELTA_Z_SHARED_REGIME_CHECK_2026_05_03.md` — coinvest+inst_delta_z anti-correlated (ρ=-0.57, p=0.0003), institutional lane not broadly broken
- `artifacts/ic_dashboard/2026-05-01_dashboard.json` — inst_delta_z mean IC -0.1006, hit rate 8.6%, n=35
- `artifacts/ic_dashboard/2026-05-03_shared_regime_check.json` — coinvest mean IC +0.1033, hit rate 91.4%

## Disposition

[ ] Option A — Drop inst_delta_z entirely (RankerSignalSpec weight 0.45 → 0.0)
[ ] Option B — Reduce ranker weight (0.45 → ____ )
[X] **Option C — Watch only, no production change** (re-evaluate 2026-06-15 post-Q1 13F refresh)
[ ] Option D (other)

### Rationale

35-date sample with a -0.10 mean IC and structural anti-correlation against coinvest_score_z is sufficient evidence to flag, but operator chose to defer remediation pending the Q1 2026 13F refresh (~2026-05-15) and one full post-refresh IC dashboard cycle (~2026-06-15). No production change is being made on stale-input evidence; review will reopen automatically when fresh data arrives. The parent memo and shared-regime check stand as the audit trail for this deferral.

### Side-effects covered

- [X] Selector inst_delta contribution (0.35 in B6 selector) — left as-is
- [X] 91-180d institutional bucket weight (0.30 in decision_engine.py:279) — left as-is
- [ ] SOUL.md weight-string drift (says 0.30, ranker says 0.45) — separate ticket TBD (not blocking; documentation issue)
- [X] Golden snapshots — not required (Option C, no production change)
- [X] ic_health_monitor SIGNAL_ALERT acknowledgement — left active (signal is doing its job; do not silence)

### Re-evaluation trigger

- [X] Calendar: re-evaluate after first IC dashboard with majority post-Q1-2026-13F-refresh data (~2026-06-15)
- [ ] Threshold: not set
- [X] Manual: may reopen earlier by request

## Related artifacts

- Parent memo: `INST_DELTA_Z_GOVERNANCE_REVIEW_2026_05_03.md`
- Shared-regime check: `INST_DELTA_Z_SHARED_REGIME_CHECK_2026_05_03.md`
- Backtest sweep template (not filed under Option C): `INST_DELTA_Z_SPEC_TICKET_TEMPLATE_2026_05_03.md` — retained as reference if disposition changes
- Ruleset changelog entry: N/A (no ruleset bump)

## Sign-off

- [X] Decision recorded
- [X] Production change applied — N/A (Option C)
- [X] Verification run — N/A (Option C)
- [X] Memo cross-linked from parent

---

_Filed 2026-05-03. Next review trigger: ~2026-06-15 (first post-Q1-13F-refresh IC dashboard cycle)._
