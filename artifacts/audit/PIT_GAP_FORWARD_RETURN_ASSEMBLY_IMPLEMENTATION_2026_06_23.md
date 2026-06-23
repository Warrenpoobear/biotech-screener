# PIT Gap Forward Return Assembly — Implementation Audit Memo

**Date:** 2026-06-23  
**Spec reference:** `artifacts/audit/PIT_GAP_FORWARD_RETURN_ASSEMBLY_SPEC_2026_06_22.md`  
**Method decision:** `artifacts/audit/PIT_GAP_FORWARD_RETURN_METHOD_DECISION_2026_06_22.md`  
**Script:** `scripts/research/pit_gap_forward_returns.py`  
**Status:** IMPLEMENTATION_COMPLETE_PENDING_OPERATOR_REVIEW

---

## Implementation Approach

Fresh implementation from spec pseudocode. No code copied from PR #382.

**Method A (primary, same-archive basis):**
- Horizons: 1d, 3d, 5d, 20d only. 60d horizon is absent from the implementation entirely — no `60` appears in `HORIZONS_A`, no 60d column is computed or emitted.
- Archive resolution: per-snapshot; if `data/pit_archives/{snap_date}/` exists, use it; otherwise fall back to nearest prior date. Fallback is flagged via `archive_fallback=True`.
- Prices are loaded once per unique archive date and cached.
- XBI anchor resolved the same way as individual tickers; excess return = actual − XBI.
- Output: `artifacts/audit/gap_panel_method_a_{run_date}.csv` (gitignored).

**Method B (sensitivity only, single May 7 archive):**
- Horizons: 1d, 3d, 5d, 20d, 60d.
- All prices sourced from `data/pit_archives/2026-05-07/price_history.csv` only.
- Every row carries `sensitivity_label="SENSITIVITY_ONLY_NOT_PRIMARY_EVIDENCE"`.
- CSV header includes two comment lines before column names (per spec §4).
- Output: `artifacts/audit/gap_panel_method_b_sensitivity_{run_date}.csv` (gitignored).

---

## Pre-PR Validation Checklist

### Generated CSVs not in PR diff
- [x] Both output files land in `artifacts/audit/` which is gitignored (`artifacts/audit/*`). Only `.md` files in that directory are tracked (`!artifacts/audit/*.md`). CSV outputs will not appear in the PR diff.

### No production files in PR diff
- [x] The script imports only: `csv`, `hashlib`, `json`, `logging`, `os`, `sys`, `datetime`, `pathlib`. No production modules imported.
- [x] No `ranker_engine`, `selector_engine`, `decision_engine`, `run_screen`, `final_score`, `sizing`, `run_phase2_snapshot_delta` imports anywhere in the script.
- [x] No reads of `data/universe_prices.csv`, `data/indices_prices.csv`, or `data/snapshots/_forward_returns_panel.csv`.
- [x] No writes to any `data/snapshots/` path.
- [x] PR diff contains only: `scripts/research/pit_gap_forward_returns.py` and this memo.

### Method A produces no 60d conclusion
- [x] `HORIZONS_A = [1, 3, 5, 20]` — no `60` in this list.
- [x] No `60d` column in the Method A output schema (`write_method_a_csv` columns list).
- [x] `summary["no_60d_conclusion"] = True` is set in the Method A summary.
- [x] Log line at runtime: `"NO 60d horizon — no 60d conclusion possible from Method A"`.
- [x] `compute_return` is never called with `horizon=60` in Method A path.

### Method B rows labeled SENSITIVITY_ONLY_NOT_PRIMARY_EVIDENCE
- [x] `SENSITIVITY_LABEL = "SENSITIVITY_ONLY_NOT_PRIMARY_EVIDENCE"` constant defined.
- [x] Every row emitted in `run_method_b()` carries `"sensitivity_label": SENSITIVITY_LABEL`.
- [x] ATXS excluded rows also carry the sensitivity label.
- [x] CSV file begins with two comment header lines before column names (per spec §4).
- [x] Validation check: rows missing label logged as `FAIL` (must be 0).
- [x] Output filename includes `_sensitivity_` to make method obvious.

### ATXS excluded after 2026-01-23
- [x] `ATXS_EXCLUSION_AFTER = "2026-01-23"` constant.
- [x] Both Method A and Method B: `if ticker == "ATXS" and snap_date > ATXS_EXCLUSION_AFTER`.
- [x] Excluded rows: `anchor_close=None`, all returns `None`, `atxs_excluded=True`.
- [x] Pre-exclusion rows (snap_date ≤ 2026-01-23): ATXS processed normally (anchor resolved from prices).
- [x] Validation check §7.4 verifies both post-exclusion (null required) and pre-exclusion (non-null expected).

### No PR #382 code copied
- [x] Implementation written from scratch from the spec pseudocode (§5, §6, §7).
- [x] `assemble_gap_forward_returns.py` (the PR #382 script name) was NOT opened or read during this session.
- [x] The new script is named `pit_gap_forward_returns.py` as required.
- [x] No structural resemblance to any quarantined code; all logic derived from spec pseudocode directly.

---

## All 9 Validation Checks Implemented

| Check | Spec § | Implementation |
|-------|--------|----------------|
| Archive date resolution | 7.1 | `resolve_archive()` + summary counts |
| Same-archive basis (Method A) | 7.2 | Enforced: one `prices` dict per row, loaded from single `arch_date` |
| May 7 archive basis (Method B) | 7.3 | Precondition check + `last_date` log |
| ATXS exclusion | 7.4 | `v4_atxs_exclusion` check in `run_validation_checks()` |
| Anchor coverage threshold ≥28 | 7.5 | `v5_anchor_coverage` flags LOW_COVERAGE snapshots |
| XBI coverage | 7.6 | `v6_xbi_coverage` flags null XBI snapshots |
| Continuity flagging | 7.7 | `flag_continuity()` + binary-event review note |
| Manifest SHA256 handling | 7.8 | `check_manifest()` returns PASS/STALE_MANIFEST/MISSING_MANIFEST |
| No production file modification | 7.9 | Import list inspection + design enforcement |

---

## Acceptance Thresholds

Logged at runtime, not asserted as conclusions:

| Threshold | Method | Condition |
|-----------|--------|-----------|
| 5d coverage ≥40 snapshots | A | Logged; `acceptance_5d_meets_threshold` field |
| 20d coverage ≥25 snapshots | A | Logged; `acceptance_20d_meets_threshold` field |
| 60d NEVER | A | Hard invariant — not computed |
| 60d coverage ≥20 snapshots | B | Logged; sensitivity only |

---

## Governance Boundaries

- **Production model freeze ACTIVE:** No ranker/selector/sizing/final_score/gate/snapshot changes.
- **PR #382 quarantined:** Not merged, not copied, not referenced.
- **No live data fetch:** Script has zero network calls. All reads from local `data/pit_archives/` and `data/snapshots/` only.
- **Outputs are research artifacts:** CSV files in gitignored `artifacts/audit/`. Only this markdown memo is committed.
- **No trading/investment conclusion:** Script emits numerical returns only; no buy/sell/hold recommendation.

---

*Prepared 2026-06-23 for operator review.*
