# Change Spec: Preserve CT.gov date precision through the catalyst pipeline

**Status**: DRAFT
**Author**: Claude Code (root-cause trace from the 2026-07-27 catch-up backfill)
**Date**: 2026-07-28
**Ruleset impact**: **YES** — reclassifying precision moves `catalyst_decay_w` → `catalyst_tilt_mult` → `target_weight_pct`. Sizing is frozen under the DEM NO_MODEL_CHANGE window, so this spec **requires an explicit operator freeze lift** and will reset the out-of-sample clock (`docs/FORWARD_VALIDATION_PROTOCOL.md` §1).

Tracking issue: #535. Related monitor-only fixes (already landed separately, no freeze implication): #533, #534 via PR #536.

---

## Objective

CT.gov reports many completion dates as month-only and explicitly `ESTIMATED`. The pipeline snaps those to the 1st of the month, discards the precision, and then asserts `date_precision=DAY` — so the model treats a placeholder as an exact-day catalyst. Restore the source's own precision so imprecise dates route through the existing `far_window` path instead of the `specific_day` path.

## Root cause — verified chain

Three separate defects compose. Each is individually small; together they manufacture exact dates that no source ever asserted.

### D1 (root) — `data_sources/ctgov_client.py:252-271` destroys precision at parse time

```python
def _parse_date(self, date_struct: Dict[str, Any]) -> Optional[str]:
    date_str = date_struct.get("date")
    ...
    if len(date_str) == 7:      # "2024-06"
        return f"{date_str}-01"        # <-- month -> 1st, precision discarded
    elif len(date_str) == 4:    # "2024"
        return f"{date_str}-01-01"     # <-- year  -> Jan 1, precision discarded
    else:
        return date_str[:10]
```

The function returns a bare ISO string. Both the **month-only granularity** and the struct's **`type` field (`ACTUAL` vs `ESTIMATED`)** are dropped, and neither is persisted anywhere — the raw payload is not cached, so downstream code cannot recover them even in principle.

Live verification against the CT.gov v2 API for the four trials behind the 2026-08-01 cluster in the 2026-07-27 snapshot:

| NCT | ticker | raw `primaryCompletionDateStruct` | stored `primary_completion_date` |
|---|---|---|---|
| NCT06669754 | PHVS | `{"date": "2026-08", "type": "ESTIMATED"}` | `2026-08-01` |
| NCT06727565 | RCUS | `{"date": "2026-08", "type": "ESTIMATED"}` | `2026-08-01` |
| NCT06775379 | XENE | `{"date": "2026-08", "type": "ESTIMATED"}` | `2026-08-01` |
| NCT07298330 | MIRM | `{"date": "2026-08", "type": "ESTIMATED"}` | `2026-08-01` |

CT.gov said "sometime in August 2026, estimated." The pipeline recorded "Saturday 1 August 2026, day-precision."

### D2 — `module_3_catalyst.py:319` defaults the stamp to `DAY`

```python
event_date_end = None
date_precision = "DAY"
...
if calendar_catalyst.event_type == "READOUT_WINDOW":
    ...
    date_precision = "RANGE"
```

Every non-`READOUT_WINDOW` `CTGOV_CALENDAR` event is stamped `DAY` unconditionally. The stamp is not derived from the source or from the granularity of the underlying date.

### D3 — two contradictory precision authorities

`common/event_quality_features.py:151-163` already declares the correct answer:

```python
_SOURCE_PRECISION: Dict[str, str] = {
    "SEC_8K_FILING": "DAY",
    "PDUFA_MANUAL": "DAY",
    ...
    "CTGOV_CALENDAR": "MONTH",      # <-- correct
    "CTGOV_PCD_FAR": "QUARTER",
}
_PRECISION_CONFIDENCE = {"DAY": 0.95, "MONTH": 0.60, "QUARTER": 0.40, ...}
```

So the repo holds both `CTGOV_CALENDAR == MONTH` (quality layer) and `CTGOV_CALENDAR == DAY` (the stamped field that actually drives routing). This is why the observed `clinical_date_confidence` for the affected names was 0.48–0.57 (MONTH-ish) while `catalyst_date_precision` claimed `DAY` (0.95) — the internal inconsistency is already visible in the output.

## Observed effect in production

`data/snapshots/2026-07-27/`, 302-ticker universe:

| Metric | Value |
|---|---|
| DAY-precision catalyst dates | 230 |
| …on the 1st of a month | 143 (62%) |
| …on end-of-month (28–31) | 62 |
| …on a **weekend** | 46 (20%) |
| DAY rows sourced `CTGOV_CALENDAR` | **189** |
| DAY rows carrying a non-1.0 `catalyst_tilt_mult` | **178** |

Largest single-date clusters, all `precision=DAY`: 2026-09-01 (34), 2026-12-01 (24), 2026-08-01 (23), 2026-10-01 (13). Distribution is identical on 2026-07-24, so this is chronic, not a regression.

Routing consequence — the two paths differ materially:

| | `catalyst_mode` | `catalyst_strength` | `catalyst_decay_w` | `catalyst_tilt_mult` |
|---|---|---|---|---|
| snapped, stamped DAY (NRIX/RCUS/PHVS/XENE/MIRM) | `specific_days` | near | 1.0 | 1.2 |
| honest MONTH (TRVI, `lower=2026-07-15 upper=2026-08-14`) | `far_window` | far | 0.15 | 0.9 |

Secondary consequence: month-boundary clustering makes `catalyst_le_7d_count` cross `warn_catalyst_le_7d_count: 4` whenever a run date falls within 7 days of the 1st, producing a recurring monthly `--strict` exit 2. That is what failed the 2026-07-27 run (count 1 → 6, `catalyst_le_7d_weight_pct` 3.33% → 19.98%).

## PIT / Data Constraints

- [x] No lookahead — this changes only how an already-known date's *precision* is recorded; no new data is read and no future information enters
- [x] Data source: CT.gov API v2 `protocolSection.statusModule.*DateStruct` via `data_sources/ctgov_client.py`
- [ ] Historical availability: **the raw `date`/`type` fields were never persisted.** `cache/ctgov/trial_records_*.json` holds only post-snap values, so historical precision cannot be reconstructed without re-fetching. Re-fetching current CT.gov data for historical dates would be a PIT violation.
- [x] Known gaps: precision for all pre-change snapshots is unrecoverable — see Non-Goals

## Inputs

| Input | Source | Schema |
|-------|--------|--------|
| `primaryCompletionDateStruct.date` | CT.gov v2 API | `str`, `"YYYY"` \| `"YYYY-MM"` \| `"YYYY-MM-DD"` |
| `primaryCompletionDateStruct.type` | CT.gov v2 API | `str`, `ACTUAL` \| `ESTIMATED` |
| `_SOURCE_PRECISION` | `common/event_quality_features.py:151` | `Dict[str, str]`, existing, unchanged |

## Outputs

| Output | Destination | Schema |
|--------|-------------|--------|
| `primary_completion_date_precision` | `cache/ctgov/trial_records_*.json` (new field) | `DAY` \| `MONTH` \| `YEAR` |
| `primary_completion_date_type` | `cache/ctgov/trial_records_*.json` (new field) | `ACTUAL` \| `ESTIMATED` |
| `catalyst_date_precision` | `rankings.csv` (existing) | derived, not defaulted to `DAY` |
| `catalyst_date_lower` / `_upper` | `rankings.csv` (existing) | true month/year bounds, not `lower == upper` |

## Invariants

1. **Never assert precision the source did not provide.** A month-only source date must not emit `date_precision=DAY`, and must not emit `catalyst_date_lower == catalyst_date_upper`.
2. **Single source of truth.** `_SOURCE_PRECISION` and the emitted `date_precision` must agree for every source, enforced by test.
3. **Deterministic** — identical inputs produce byte-identical outputs (CLAUDE.md coding standards).
4. **No silent nulls** (CCFT: Complete). A trial whose precision cannot be determined is flagged, not defaulted.
5. **Conservative on ambiguity.** Where precision is unknown, prefer the *weaker* claim, which routes to `far_window` and a ≤1.0 tilt. Never the stronger.
6. **`ACTUAL` vs `ESTIMATED` is preserved** and available downstream, whether or not this spec consumes it.

## Failure Modes

| Scenario | Expected behavior |
|----------|-------------------|
| `date_struct` missing `date` | return `None` as today; precision `UNKNOWN`; no event emitted |
| `date` is `"YYYY"` | date `YYYY-01-01`, precision `YEAR`, bounds `YYYY-01-01`..`YYYY-12-31` |
| `date` is `"YYYY-MM"` | date unchanged from today's `YYYY-MM-01`, precision `MONTH`, bounds first..last of month |
| `date` is `"YYYY-MM-DD"` | precision `DAY`, bounds equal — the only case where `lower == upper` is legitimate |
| `type` absent | `type` recorded as `UNKNOWN`, not assumed `ACTUAL` |
| cached record lacks the new fields (pre-change cache) | precision `UNKNOWN` → `far_window`; WARN once per run, do not fail |
| source not in `_SOURCE_PRECISION` | precision `UNKNOWN`, WARN; do not default to `DAY` |

## Validation Plan

### Tests (write BEFORE implementation)

- [ ] `test_parse_date_month_only_returns_month_precision` — `{"date": "2026-08"}` → `("2026-08-01", "MONTH")`, proves D1 fixed
- [ ] `test_parse_date_year_only_returns_year_precision` — `{"date": "2026"}` → `("2026-01-01", "YEAR")`
- [ ] `test_parse_date_full_returns_day_precision` — `{"date": "2026-08-14"}` → `("2026-08-14", "DAY")`
- [ ] `test_parse_date_preserves_estimated_type` — `ESTIMATED` survives to the stored record
- [ ] `test_ctgov_calendar_event_not_stamped_day_for_month_source` — proves D2 fixed
- [ ] `test_source_precision_table_matches_emitted_precision` — proves D3; parametrised over every key in `_SOURCE_PRECISION`
- [ ] `test_month_precision_routes_to_far_window` — asserts `catalyst_mode == "far_window"`, `decay_w == 0.15`
- [ ] `test_no_day_precision_catalyst_on_weekend` — regression guard on the 46-name symptom
- [ ] `test_month_precision_bounds_are_not_collapsed` — `lower != upper` for MONTH
- [ ] `test_legacy_cache_without_precision_fields_degrades_to_unknown` — back-compat, WARN not FAIL
- [ ] `test_parse_date_deterministic` — same inputs → same outputs
- [ ] Use PIT fixtures only; never fetch live CT.gov in tests (CLAUDE.md)

### Evaluation (ranking/sizing change)

Because `catalyst_tilt_mult` feeds `target_weight_pct` and **not** `final_score` (`decision_engine.py:2581` "affects weight only"; sole consumer `decision_engine.py:2280`), the expected ranking delta is **zero**. This must be verified, not assumed:

- [ ] Re-run 2026-07-27 with the fix; assert `final_score` unchanged for all 302 rows (bitwise)
- [ ] Assert Top-30 membership and `actionable_rank` unchanged
- [ ] Report `target_weight_pct` delta distribution across the ~178 affected rows
- [ ] Confirm `catalyst_le_7d_count` falls from 6 to ≈1 on 2026-07-27
- [ ] Confirm `phase2_health` returns OK on 2026-07-27, removing the false `--strict` exit 2
- [ ] Hash rotation entry in `governance/HASH_ROTATIONS.md` if any production hash moves

### Integration

- [ ] Full suite passes
- [ ] No pre-commit hook failures (black 120, isort, flake8, semgrep governance gate)
- [ ] `catalyst_source_mix.json` `by_date_precision` shows a `MONTH` bucket, which today is absent entirely

## Expected Effect Size

**Structural correctness fix. No direct IC or ranking impact expected.**

- `final_score`, selector, ranker, Top-30 membership: **no change expected** (tilt is sizing-only)
- `target_weight_pct`: changes for up to **178 of 302** rows. Affected names move from `decay_w=1.0`/`tilt=1.2` to `far_window` `decay_w=0.15`/`tilt=0.9`, i.e. **lower** weights. Inert for the live agentic account, which is equal-weight per `production_data/AGENTIC_ACCOUNT_RULES.md`; live only under model-weight sizing.
- Monitoring: removes 3 spurious health flags per month-boundary window and the recurring false `--strict` exit 2.
- Honest caveat: I have **not** established whether catalyst proximity reaches `final_score` through a separate channel (`module_3_catalyst.py:2498` `time_decay_score` → `module_5_scoring_v3` catalyst contribution). Empirically `final_score` moved <1e-4 across a 3-day proximity change on 2026-07-24 → 07-27, which suggests the channel is insensitive, but the bitwise check in the Evaluation section is the actual gate. **If `final_score` moves at all, this becomes a Tier 3/4 alpha change and needs the full promotion battery, not just a freeze lift.**

## Non-Goals

- Does **not** backfill or correct historical snapshots. The raw `date`/`type` fields were never persisted, so historical precision is unrecoverable; re-fetching current CT.gov data for past as-of dates would be a PIT violation. Pre-change snapshots keep their `DAY` stamps and must be read with this defect in mind.
- Does **not** change `_SOURCE_PRECISION` values, thresholds, `warn_catalyst_le_7d_count`, or any tilt multiplier. Only which precision each event is *assigned*.
- Does **not** consume the newly preserved `ACTUAL`/`ESTIMATED` type in scoring. It is recorded for future use; using it would be a separate spec.
- Does **not** touch `READOUT_WINDOW` events, which already correctly emit `RANGE`.
- Does **not** address the `Portfolio weights` WARN (`weight sum=16.46%`, 184 rows missing `target_weight_pct`) — chronic and unrelated, identical on 2026-07-24.

---

## Governance

- **Tier**: 3 (catalyst pipeline + sizing surface). Highest affected tier governs.
- **Freeze**: blocked by the DEM candidate freeze / NO_MODEL_CHANGE window. Requires explicit operator lift; the lift resets the out-of-sample clock, currently at n=4 mandate-eligible windows against a 52-window gate. **The cost of fixing this is a restart of forward-validation evidence — that trade-off is the operator's call, and it is the main reason this spec is DRAFT rather than implemented.**
- **Precedent**: same defect class as the KYMR catalyst misdate (sev3, 2026-07-02).

## Implementation Log

*(none — DRAFT, awaiting operator decision on the freeze lift)*

---

*Template version: 1.0.0 — see specs/SYSTEM_SPEC.md for system invariants*
