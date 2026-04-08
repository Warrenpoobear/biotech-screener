# Spec 062 — Stage 1 Activation, EV Feed Verification, and Readiness Monitoring

**Status**: IMPLEMENTING
**Author**: Claude / arrenchulz
**Date**: 2026-04-08
**Ruleset impact**: YES (Stage 1 tiebreaker activation in candidate ruleset)
**Depends on**: Spec 061 (Event EV Promotion Ladder), Spec 060 (Daily EV Scoring)

---

## Objective

Activate Stage 1 (tiebreaker) of the Event EV promotion ladder as a
candidate ruleset, verify the EV data feed is healthy, and surface daily
readiness monitoring so activation decisions are evidence-based.

This is NOT a full promotion. Stage 1 is the lowest-risk integration:
it only affects names with identical sort keys, and it has zero effect
when keys are distinct. Stages 2-4 remain explicitly deferred.

## Scope

### 1. Stage 1 Candidate Ruleset

Create `v1.14.0_a4_ev_tiebreaker.json` with `event_ev_stage: "tiebreaker"`.
All other fields match v1.13.0. Register in manifest as `candidate` (not `active`).

### 2. EV Feed Health Verification

New tool: `tools/verify_event_ev_feed.py`
- Reads the latest rankings.csv and checks that event_ev_* columns are
  populated for names that have daily EV artifacts
- Reports coverage: N names with EV data / N eligible names
- Reports staleness: artifact date vs snapshot date
- Reports expectation model input coverage: which market features are
  present (short_interest_pct, close_price, market_cap_mm, priced_move_pct)
- Exits 0 (healthy) / 1 (degraded) / 2 (missing)

### 3. Daily Readiness Artifact in Ops Digest

Wire `evaluate_ev_readiness()` into the ops digest:
- New section "Event EV Readiness" in digest JSON and markdown
- Shows: n_daily_artifacts, avg coverage, max gap, per-stage readiness
- Attention level: INFO when accumulating, REVIEW when Stage 1 ready

### 4. Stage 1 Shadow Impact Measurement

New tool: `tools/measure_ev_tiebreak_impact.py`
- Reads a rankings.csv snapshot
- Computes sort keys with and without Stage 1 tiebreaker
- Reports: tie frequency, names reordered, top-30 membership changes
- Produces a structured JSON artifact for trend tracking

### 5. Explicit Deferral

Stages 2, 3, 4 are NOT activated. The timing hazard model is research-only
and non-stationary. Event EV depends partly on timing quality, so anything
beyond a tiebreaker requires more forward evidence.

## Acceptance Criteria

- [ ] Candidate ruleset v1.14.0 registered in manifest
- [ ] `verify_event_ev_feed.py` runs and reports coverage
- [ ] Ops digest includes "Event EV Readiness" section
- [ ] `measure_ev_tiebreak_impact.py` quantifies tie frequency and reordering
- [ ] Zero behavioral change to production (candidate only, not active)

## Deferral Policy

Do not promote Stage 1 to active until:
1. 30+ trading days of EV artifacts accumulated
2. EV feed coverage >= 30% of top-30
3. Shadow impact measurement shows ties occur and EV changes ordering
4. Governance review completed
