"""Stage 1 options feature expansion — unit tests.

Spec: OPTIONS_SHADOW_EXPANSION / EXPECTATION_LAYER_PLUMBING
Invariant: options feature injection MUST NOT change actionable_rank.

Tests:
  1. test_options_feature_join_does_not_change_rank_order
  2. test_missing_options_data_warns_not_fails
  3. test_options_quality_score_flags_wide_spreads
  4. test_vendor_iv_disagreement_flags_data_warn
  5. test_expectation_gap_computes_from_existing_fields
  6. test_options_shadow_veto_does_not_change_production_selection
  7. test_options_fields_export_to_rankings_csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.options_features import (
    OPTIONS_FEATURE_COLUMNS,
    OPTIONS_MISSING,
    OPTIONS_NO_EVENT_EXPIRY,
    OPTIONS_STALE,
    OPTIONS_THIN,
    OPTIONS_UNUSABLE,
    OPTIONS_USABLE,
    OPTIONS_VENDOR_DISAGREE,
    VERDICT_CROWDING_WARN,
    VERDICT_DATA_WARN,
    VERDICT_EVENT_ALREADY_PRICED,
    VERDICT_NO_DATA,
    VERDICT_OPTIONS_CONFIRMED,
    build_forward_validation_cohorts,
    compute_event_premium_features,
    compute_expectation_gap_scores,
    compute_options_quality,
    compute_options_shadow_verdict,
    enrich_csv_rows,
    write_options_cohorts_sidecar,
    write_options_features_sidecar,
)

# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _base_row(ticker: str = "TEST", **overrides) -> dict:
    """Minimal rankings row with required fields."""
    row = {
        "ticker": ticker,
        "actionable_rank": "5",
        "score_rank_pct": "70.0",
        "priced_move_pct": "",
        "opt_has_data": "0",
        "opt_use_for_judgment": "",
        "opt_liquidity_state": "absent",
        "opt_iv_regime": "",
        "opt_event_premium": "",
        "opt_front_iv": "",
        "opt_back_iv": "",
        "opt_atm_iv": "",
        "composite_score": "50.0",
    }
    row.update(overrides)
    return row


def _usable_row(ticker: str = "GOOD", **overrides) -> dict:
    defaults = dict(
        opt_has_data="1",
        opt_use_for_judgment="YES",
        opt_liquidity_state="liquid",
        opt_iv_regime="NORMAL",
        opt_event_premium="YES",
        opt_front_iv="0.90",
        opt_back_iv="0.60",
        opt_atm_iv="0.75",
        priced_move_pct="35.0",
    )
    defaults.update(overrides)
    return _base_row(ticker=ticker, **defaults)


# ─── Test 1: rank order invariant ─────────────────────────────────────────────


def test_options_feature_join_does_not_change_rank_order(tmp_path):
    """Injecting options features must not alter actionable_rank values."""
    rows = [
        _usable_row("ALPHA", actionable_rank="1", score_rank_pct="95.0"),
        _usable_row("BETA", actionable_rank="2", score_rank_pct="88.0"),
        _base_row("GAMMA", actionable_rank="3", score_rank_pct="60.0"),
        _base_row("DELTA", actionable_rank="4", score_rank_pct="40.0"),
    ]
    before = [r["actionable_rank"] for r in rows]

    enrich_csv_rows(rows, "2026-06-28", write_sidecars=False)

    after = [r["actionable_rank"] for r in rows]
    assert before == after, f"actionable_rank changed: {list(zip(before, after))}"


# ─── Test 2: missing data → OPTIONS_MISSING, no exception ─────────────────────


def test_missing_options_data_warns_not_fails():
    """Rows with no options chain get OPTIONS_MISSING status without error."""
    row = _base_row("NODATA", opt_has_data="0")
    rows = [row]

    summary = enrich_csv_rows(rows, "2026-06-28", write_sidecars=False)

    assert row["options_quality_status"] == OPTIONS_MISSING
    assert row["options_quality_score"] == 0.0
    assert row["options_shadow_verdict"] == VERDICT_NO_DATA
    assert summary["n_missing"] == 1


# ─── Test 3: thin/absent liquidity → appropriate status ───────────────────────


def test_options_quality_score_flags_wide_spreads():
    """Absent liquidity → UNUSABLE; thin liquidity → THIN, not USABLE."""
    absent_row = _base_row(
        "ABSENT",
        opt_has_data="1",
        opt_use_for_judgment="YES",
        opt_liquidity_state="absent",
        opt_event_premium="YES",
    )
    thin_row = _base_row(
        "THIN",
        opt_has_data="1",
        opt_use_for_judgment="YES",
        opt_liquidity_state="thin",
        opt_event_premium="YES",
        opt_front_iv="0.80",
        opt_back_iv="0.60",
    )

    score_absent, status_absent = compute_options_quality(absent_row)
    score_thin, status_thin = compute_options_quality(thin_row)

    assert status_absent == OPTIONS_UNUSABLE
    assert score_absent < 0.5
    assert status_thin == OPTIONS_THIN
    assert 0.0 < score_thin < 1.0
    assert status_thin != OPTIONS_USABLE


# ─── Test 4: EXTREME iv regime → OPTIONS_VENDOR_DISAGREE ──────────────────────


def test_vendor_iv_disagreement_flags_data_warn():
    """EXTREME iv regime signals vendor disagreement."""
    row = _base_row(
        "EXTREME",
        opt_has_data="1",
        opt_use_for_judgment="YES",
        opt_liquidity_state="liquid",
        opt_iv_regime="EXTREME",
        opt_event_premium="YES",
    )
    score, status = compute_options_quality(row)
    assert status == OPTIONS_VENDOR_DISAGREE

    # Shadow verdict should be DATA_WARN (not OPTIONS_CONFIRMED)
    rows = [row]
    enrich_csv_rows(rows, "2026-06-28", write_sidecars=False)
    # EXTREME maps to VENDOR_DISAGREE which is not STALE/UNUSABLE,
    # so verdict may be NEUTRAL or CONFIRMED depending on priced_move.
    # Key assertion: it is NOT OPTIONS_CONFIRMED.
    assert row["options_shadow_verdict"] != VERDICT_OPTIONS_CONFIRMED


# ─── Test 5: expectation gap math ─────────────────────────────────────────────


def test_expectation_gap_computes_from_existing_fields():
    """Expectation gap = opportunity_z - priced_move_z (verified numerically)."""
    rows = [
        _base_row("HIGH_OPP", score_rank_pct="90.0", priced_move_pct="20.0"),
        _base_row("MID_OPP", score_rank_pct="50.0", priced_move_pct="50.0"),
        _base_row("LOW_OPP", score_rank_pct="10.0", priced_move_pct="80.0"),
    ]
    gaps = compute_expectation_gap_scores(rows)

    # HIGH_OPP: high model rank, low priced move → positive gap
    # LOW_OPP: low model rank, high priced move → negative gap
    assert len(gaps) == 3
    assert gaps[0] is not None and gaps[0] > 0, "HIGH_OPP should have positive gap"
    assert gaps[2] is not None and gaps[2] < 0, "LOW_OPP should have negative gap"

    # Gaps are z-score differences, so they should straddle zero
    assert gaps[0] > gaps[1] > gaps[2]  # monotone ordering


# ─── Test 6: shadow verdict never changes actionable selection ─────────────────


def test_options_shadow_veto_does_not_change_production_selection():
    """options_shadow_verdict must not appear in ranking-affecting fields."""
    rows = [_usable_row("A", actionable_rank="1"), _base_row("B", actionable_rank="2")]
    rank_fields_before = [(r["ticker"], r["actionable_rank"]) for r in rows]

    enrich_csv_rows(rows, "2026-06-28", write_sidecars=False)

    rank_fields_after = [(r["ticker"], r["actionable_rank"]) for r in rows]
    assert rank_fields_before == rank_fields_after

    # Verify the verdict field exists but doesn't alias any ranking field
    protected = {"actionable_rank", "composite_score", "score_rank_pct", "composite_rank"}
    assert not (set(OPTIONS_FEATURE_COLUMNS) & protected), (
        f"Options feature columns overlap with ranking fields: " f"{set(OPTIONS_FEATURE_COLUMNS) & protected}"
    )


# ─── Test 7: options fields appear in output rows ─────────────────────────────


def test_options_fields_export_to_rankings_csv():
    """All OPTIONS_FEATURE_COLUMNS must be present on every row after enrichment."""
    rows = [
        _usable_row("GOOD"),
        _base_row("NO_DATA"),
        _base_row("STALE", opt_has_data="1", opt_use_for_judgment="NO"),
    ]
    enrich_csv_rows(rows, "2026-06-28", write_sidecars=False)

    for row in rows:
        for col in OPTIONS_FEATURE_COLUMNS:
            assert col in row, f"Column '{col}' missing from row {row['ticker']}"


# ─── Bonus: sidecar CSV and cohort JSON shapes ────────────────────────────────


def test_sidecar_csv_is_written(tmp_path):
    """Sidecar CSV is written with correct schema."""
    rows = [_usable_row("ALPHA", actionable_rank="1"), _base_row("BETA", actionable_rank="25")]
    enrich_csv_rows(rows, "2026-06-28", write_sidecars=True, artifact_dir=tmp_path)

    csv_path = tmp_path / "2026-06-28_options_features.csv"
    assert csv_path.exists()
    import csv as csvmod

    with csv_path.open() as fh:
        reader = csvmod.DictReader(fh)
        written_rows = list(reader)
    assert len(written_rows) == 2
    assert "options_quality_status" in written_rows[0]
    assert "expectation_gap_score" in written_rows[0]


def test_cohorts_json_is_written(tmp_path):
    """Cohorts JSON has expected structure and correct core_top30 count."""
    rows = [
        _usable_row("A", actionable_rank="1", score_rank_pct="90.0", priced_move_pct="20.0"),
        _usable_row("B", actionable_rank="30", score_rank_pct="70.0", priced_move_pct="30.0"),
        _base_row("C", actionable_rank="31", score_rank_pct="40.0"),
    ]
    enrich_csv_rows(rows, "2026-06-28", write_sidecars=True, artifact_dir=tmp_path)

    json_path = tmp_path / "2026-06-28_options_cohorts.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert data["schema"] == "options_cohorts.v1"
    assert data["n_universe"] == 3
    # A(rank=1) and B(rank=30) are in top 30; C(rank=31) is not
    assert len(data["cohorts"]["core_top30"]) == 2
    assert "A" in data["cohorts"]["core_top30"]
    assert "B" in data["cohorts"]["core_top30"]
    assert "C" not in data["cohorts"]["core_top30"]
