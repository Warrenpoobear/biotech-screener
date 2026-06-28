"""Tests for the rank-depth shadow validation layer.

VALIDATION_INFRASTRUCTURE / RANK_DEPTH_SHADOW_TRACKING / NO_MODEL_CHANGE.

Covers:
  1. Cohort labeling (add_rank_depth_cohorts)
  2. Top-60 price-refresh scope selection (_load_top_n_tickers_from_prior_snapshot)
  3. Default daily-production behavior unchanged (MODE_DEFAULTS + scope resolver)
  4. Forward ledger emits top30 / rank31_60 / top60 cohorts
  5. Top-60 sidecar export (annotation only)
"""

from __future__ import annotations

import csv

import pandas as pd

from run_screen import add_rank_depth_cohorts, export_rank_depth_top60_sidecar
from tools.fill_forward_returns import COHORT_RANGES, compute_cohort_returns
from tools.run_daily_production import (
    MODE_DEFAULTS,
    _load_top_n_tickers_from_prior_snapshot,
    resolve_validation_rank_depth_scope,
)
from tools.run_forward_validation import build_cohort_baskets, get_rank_band

# ---------------------------------------------------------------------------
# 1. Cohort labeling
# ---------------------------------------------------------------------------


def test_rank_depth_cohorts_labeling():
    df = pd.DataFrame(
        {
            "ticker": [f"T{i}" for i in range(1, 71)],
            "actionable_rank": list(range(1, 71)),
        }
    )
    out = add_rank_depth_cohorts(df)

    assert out["is_top30"].sum() == 30
    assert out["is_rank31_60"].sum() == 30
    assert out["is_top60"].sum() == 60
    assert set(out.loc[out["actionable_rank"].between(31, 60), "rank_depth_cohort"]) == {"rank31_60"}
    assert set(out.loc[out["actionable_rank"].between(1, 30), "rank_depth_cohort"]) == {"top30"}
    assert set(out.loc[out["actionable_rank"] > 60, "rank_depth_cohort"]) == {"outside_top60"}


def test_rank_depth_cohorts_handles_blank_ranks():
    # Ineligible rows carry blank actionable_rank -> outside_top60, no crash.
    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "actionable_rank": [1, "", 45],
        }
    )
    out = add_rank_depth_cohorts(df)
    assert list(out["rank_depth_cohort"]) == ["top30", "outside_top60", "rank31_60"]
    assert list(out["is_top60"]) == [True, False, True]


# ---------------------------------------------------------------------------
# 2. Top-60 price-refresh scope selection
# ---------------------------------------------------------------------------


def _write_prior_snapshot(snap_root, date, n_rows):
    d = snap_root / date
    d.mkdir(parents=True)
    with open(d / "rankings.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "actionable_rank"])
        for i in range(1, n_rows + 1):
            w.writerow([f"TCK{i}", i])
    return d


def test_top60_price_scope_loads_60_and_excludes_xbi(tmp_path):
    _write_prior_snapshot(tmp_path, "2026-06-26", n_rows=70)

    top60 = _load_top_n_tickers_from_prior_snapshot(tmp_path, "2026-06-28", top_n=60)
    assert top60 is not None
    assert len(top60) == 60
    assert "XBI" not in top60  # XBI is appended separately by refresh_prices()

    top30 = _load_top_n_tickers_from_prior_snapshot(tmp_path, "2026-06-28", top_n=30)
    assert len(top30) == 30
    # top30 is a strict prefix of top60 (same rank order)
    assert top60[:30] == top30


# ---------------------------------------------------------------------------
# 3. Default daily-production behavior unchanged
# ---------------------------------------------------------------------------


def test_daily_production_default_scope_unchanged():
    assert MODE_DEFAULTS["daily-production"]["price_refresh_scope"] == "full"
    assert MODE_DEFAULTS["daily-validation"]["price_refresh_scope"] == "top30"
    # No --validation-rank-depth -> scope left at the mode default.
    assert resolve_validation_rank_depth_scope(None, scope_explicitly_set=False, current_scope="full") == "full"


def test_validation_rank_depth_widens_scope():
    assert resolve_validation_rank_depth_scope(60, scope_explicitly_set=False, current_scope="full") == "top60"
    assert resolve_validation_rank_depth_scope(30, scope_explicitly_set=False, current_scope="full") == "top30"
    # Explicit --price-refresh-scope always wins over --validation-rank-depth.
    assert resolve_validation_rank_depth_scope(60, scope_explicitly_set=True, current_scope="full") == "full"


# ---------------------------------------------------------------------------
# 4. Forward ledger emits rank-depth cohorts
# ---------------------------------------------------------------------------


def _ranked_rows(n):
    return [{"ticker": f"T{i}", "actionable_rank": str(i)} for i in range(1, n + 1)]


def test_build_cohort_baskets_emits_all_cohorts():
    baskets = build_cohort_baskets(_ranked_rows(70))
    assert set(baskets) == {"top30", "rank31_60", "top60"}
    assert len(baskets["top30"]) == 30
    assert len(baskets["rank31_60"]) == 30
    assert len(baskets["top60"]) == 60
    assert baskets["rank31_60"][0] == "T31"
    assert baskets["rank31_60"][-1] == "T60"


def test_get_rank_band_31_60():
    band = get_rank_band(_ranked_rows(70), 31, 60)
    assert [e["rank"] for e in band] == list(range(31, 61))


def test_compute_cohort_returns_structure_and_ranges():
    cohorts = build_cohort_baskets(_ranked_rows(70))
    # Empty universe_dates -> no forward endpoint -> all returns pending, but the
    # cohort structure + rank ranges must still be emitted.
    out = compute_cohort_returns(cohorts, effective_start="2026-06-26", universe_dates=[], xbi_start=100.0)
    assert set(out) == {"top30", "rank31_60", "top60"}
    for cohort in ("top30", "rank31_60", "top60"):
        rec = out[cohort]["5d"]
        assert (rec["rank_min"], rec["rank_max"]) == COHORT_RANGES[cohort]
        assert rec["ew_return"] is None  # pending (no end date)
        assert rec["n_names"] == len(cohorts[cohort])


# ---------------------------------------------------------------------------
# 5. Top-60 sidecar export
# ---------------------------------------------------------------------------


def test_sidecar_export_is_top60_only(tmp_path):
    csv_rows = []
    for i in range(1, 71):
        csv_rows.append(
            {
                "ticker": f"T{i}",
                "company_name": f"Co {i}",
                "actionable_rank": str(i),
                "final_score": f"{1.0 / i:.4f}",
                "market_cap_mm": "100",
                "close_price": "10",
                "catalyst_bucket": "",
                "next_catalyst_date": "",
                "catalyst_event_type": "",
                "inst_delta_z": "",
                "coinvest_score_z": "",
                "expectation_error_score": "",
                "ees_v3_gate": "",
                "opt_use_for_judgment": "",
                "risk_flags": "",
            }
        )
    # add an ineligible row (blank rank) — must be excluded
    csv_rows.append({"ticker": "INELIG", "actionable_rank": ""})

    out_path = export_rank_depth_top60_sidecar(csv_rows, tmp_path, "2026-06-28")
    assert out_path.exists()

    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 60  # only ranks 1-60
    assert all(r["as_of_date"] == "2026-06-28" for r in rows)
    assert {r["rank_depth_cohort"] for r in rows} == {"top30", "rank31_60"}
    assert sum(1 for r in rows if r["is_top30"] == "True") == 30
    assert sum(1 for r in rows if r["is_rank31_60"] == "True") == 30
    assert "INELIG" not in {r["ticker"] for r in rows}
    # rank-ordered
    assert [int(r["actionable_rank"]) for r in rows] == list(range(1, 61))
