"""Tests for scripts/pnl_attribution.py."""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from scripts.pnl_attribution import (
    SCHEMA_VERSION,
    AttributionResult,
    PositionPnL,
    _bucket_catalyst,
    _bucket_inst_delta,
    _find_prior_date,
    build_portfolio,
    check_pnl_attribution_file,
    compute_attribution,
    compute_portfolio_turnover,
    load_rankings_map,
    write_attribution_json,
    write_attribution_md,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _write_price_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = ["date", "ticker", "close"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_rankings_csv(snap_dir: Path, tickers_data: List[Dict[str, str]]) -> None:
    """Write rankings.csv with full field set."""
    snap_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ticker", "actionable_rank", "eligible", "tier_dev", "tier_any",
        "catalyst_mode", "coinvest_tag", "inst_delta_z",
        "composite_rank", "composite_score", "archetype",
        "target_weight_pct",
    ]
    with open(snap_dir / "rankings.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in tickers_data:
            writer.writerow(row)


def _write_metadata(snap_dir: Path, date_str: str) -> None:
    snap_dir.mkdir(parents=True, exist_ok=True)
    with open(snap_dir / "metadata.json", "w") as f:
        json.dump({"as_of_date": date_str}, f)


def _make_snapshot(
    root: Path,
    date_str: str,
    tickers: List[str],
    prices: Dict[str, float] | None = None,
) -> Path:
    """Create a snapshot dir with rankings + metadata."""
    snap_dir = root / date_str
    data = []
    for i, t in enumerate(tickers):
        data.append({
            "ticker": t, "actionable_rank": str(i + 1),
            "eligible": "1", "tier_dev": "A", "tier_any": "A",
            "catalyst_mode": "specific_days", "coinvest_tag": "elite_3",
            "inst_delta_z": "0.8",
            "composite_rank": str(i + 1), "composite_score": "50.0",
            "archetype": "drug_developer", "target_weight_pct": "5.0",
        })
    _write_rankings_csv(snap_dir, data)
    _write_metadata(snap_dir, date_str)
    return snap_dir


# ---------------------------------------------------------------------------
# Unit tests: portfolio turnover
# ---------------------------------------------------------------------------

class TestPortfolioTurnover:
    def test_identical_weights(self):
        w = {"A": 0.5, "B": 0.5}
        assert compute_portfolio_turnover(w, w) == 0.0

    def test_complete_rotation(self):
        w0 = {"A": 0.5, "B": 0.5}
        w1 = {"C": 0.5, "D": 0.5}
        turn = compute_portfolio_turnover(w0, w1)
        # |0-0.5|*2 + |0.5-0|*2 = 2.0 → 0.5*2.0 = 1.0
        assert abs(turn - 1.0) < 1e-6

    def test_partial_change(self):
        w0 = {"A": 0.5, "B": 0.5}
        w1 = {"A": 0.5, "C": 0.5}
        turn = compute_portfolio_turnover(w0, w1)
        # |0.5-0.5| + |0-0.5| + |0.5-0| = 1.0 → 0.5
        assert abs(turn - 0.5) < 1e-6

    def test_empty(self):
        assert compute_portfolio_turnover({}, {}) == 0.0


# ---------------------------------------------------------------------------
# Unit tests: net cost
# ---------------------------------------------------------------------------

class TestNetCost:
    def test_cost_deduction(self, tmp_dir):
        """Verify net = gross - turnover * cost_bps/10000."""
        snap_root = tmp_dir / "snapshots"
        d0 = _make_snapshot(snap_root, "2025-01-02", ["A", "B"])
        d1 = _make_snapshot(snap_root, "2025-01-03", ["A", "C"])  # B exits, C enters

        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, [
            {"date": "2025-01-02", "ticker": "A", "close": "100"},
            {"date": "2025-01-02", "ticker": "B", "close": "50"},
            {"date": "2025-01-03", "ticker": "A", "close": "102"},
            {"date": "2025-01-03", "ticker": "B", "close": "52"},
            {"date": "2025-01-03", "ticker": "C", "close": "200"},
        ])

        result = compute_attribution(
            d0_dir=d0, d1_dir=d1,
            price_csv=price_csv, cost_bps=30, top_k=2,
        )

        assert result.gross_return is not None
        assert result.turnover > 0
        expected_net = result.gross_return - result.turnover * 30 / 10_000
        assert abs(result.net_return - expected_net) < 1e-8


# ---------------------------------------------------------------------------
# Unit tests: schema version
# ---------------------------------------------------------------------------

class TestSchema:
    def test_schema_present(self, tmp_dir):
        snap_root = tmp_dir / "snapshots"
        d0 = _make_snapshot(snap_root, "2025-01-02", ["A"])
        d1 = _make_snapshot(snap_root, "2025-01-03", ["A"])

        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, [
            {"date": "2025-01-02", "ticker": "A", "close": "100"},
            {"date": "2025-01-03", "ticker": "A", "close": "105"},
        ])

        result = compute_attribution(d0, d1, price_csv)
        assert result.schema == SCHEMA_VERSION

        # Verify JSON output has schema
        out_path = tmp_dir / "pnl_attribution.json"
        write_attribution_json(result, out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert data["schema"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Unit tests: breakdown sums
# ---------------------------------------------------------------------------

class TestBreakdownSums:
    def test_action_breakdown_sums_to_gross(self, tmp_dir):
        """Sum of all action breakdowns should equal gross return."""
        snap_root = tmp_dir / "snapshots"
        d0 = _make_snapshot(snap_root, "2025-01-02", ["A", "B", "C"])
        d1 = _make_snapshot(snap_root, "2025-01-03", ["A", "B", "D"])

        price_csv = tmp_dir / "prices.csv"
        _write_price_csv(price_csv, [
            {"date": "2025-01-02", "ticker": "A", "close": "100"},
            {"date": "2025-01-02", "ticker": "B", "close": "50"},
            {"date": "2025-01-02", "ticker": "C", "close": "200"},
            {"date": "2025-01-03", "ticker": "A", "close": "110"},
            {"date": "2025-01-03", "ticker": "B", "close": "48"},
            {"date": "2025-01-03", "ticker": "C", "close": "190"},
            {"date": "2025-01-03", "ticker": "D", "close": "75"},
        ])

        result = compute_attribution(d0, d1, price_csv, top_k=3)

        # Sum of action contributions
        action_total = sum(v["total_contribution"] for v in result.by_action.values())
        assert abs(action_total - result.gross_return) < 1e-6


# ---------------------------------------------------------------------------
# Unit tests: gate check
# ---------------------------------------------------------------------------

class TestGateCheck:
    def test_missing_file_cold_start(self, tmp_dir):
        status, detail, _, _ = check_pnl_attribution_file(tmp_dir)
        assert status == "PASS"
        assert "cold-start" in detail

    def test_valid_file(self, tmp_dir):
        data = {
            "schema": SCHEMA_VERSION,
            "as_of_date": "2025-01-03",
            "prior_date": "2025-01-02",
            "coverage_pct": 0.95,
            "gross_return": 0.0123,
            "turnover": 0.15,
        }
        with open(tmp_dir / "pnl_attribution.json", "w") as f:
            json.dump(data, f)

        status, detail, _, _ = check_pnl_attribution_file(tmp_dir)
        assert status == "PASS"

    def test_low_coverage_warns(self, tmp_dir):
        data = {
            "schema": SCHEMA_VERSION,
            "coverage_pct": 0.50,
        }
        with open(tmp_dir / "pnl_attribution.json", "w") as f:
            json.dump(data, f)

        status, detail, value, threshold = check_pnl_attribution_file(
            tmp_dir, min_coverage_pct=80.0,
        )
        assert status == "WARN"
        assert value == 50.0

    def test_wrong_schema_warns(self, tmp_dir):
        data = {"schema": "wrong_schema.v99"}
        with open(tmp_dir / "pnl_attribution.json", "w") as f:
            json.dump(data, f)

        status, detail, _, _ = check_pnl_attribution_file(tmp_dir)
        assert status == "WARN"

    def test_corrupt_json_warns(self, tmp_dir):
        with open(tmp_dir / "pnl_attribution.json", "w") as f:
            f.write("{broken json")

        status, detail, _, _ = check_pnl_attribution_file(tmp_dir)
        assert status == "WARN"


# ---------------------------------------------------------------------------
# Unit tests: bucket functions
# ---------------------------------------------------------------------------

class TestBuckets:
    def test_catalyst_buckets(self):
        assert _bucket_catalyst("specific_days") == "specific_days"
        assert _bucket_catalyst("missing") == "none"
        assert _bucket_catalyst("") == "none"

    def test_inst_delta_buckets(self):
        assert _bucket_inst_delta("1.5") == "positive"
        assert _bucket_inst_delta("-0.8") == "negative"
        assert _bucket_inst_delta("0.2") == "neutral"
        assert _bucket_inst_delta("") == "none"
        assert _bucket_inst_delta("nan") == "none"


# ---------------------------------------------------------------------------
# Unit tests: prior date finder
# ---------------------------------------------------------------------------

class TestFindPriorDate:
    def test_finds_most_recent(self, tmp_dir):
        snap_root = tmp_dir / "snapshots"
        _make_snapshot(snap_root, "2025-01-01", ["A"])
        _make_snapshot(snap_root, "2025-01-02", ["A"])
        _make_snapshot(snap_root, "2025-01-03", ["A"])

        prior = _find_prior_date(snap_root, "2025-01-03")
        assert prior == "2025-01-02"

    def test_no_prior(self, tmp_dir):
        snap_root = tmp_dir / "snapshots"
        _make_snapshot(snap_root, "2025-01-01", ["A"])

        prior = _find_prior_date(snap_root, "2025-01-01")
        assert prior is None


# ---------------------------------------------------------------------------
# Integration: output writers
# ---------------------------------------------------------------------------

class TestOutputWriters:
    def test_json_md_roundtrip(self, tmp_dir):
        result = AttributionResult(
            as_of_date="2025-01-03", prior_date="2025-01-02",
            n_positions_d0=20, n_positions_d1=20,
            n_priced=18, coverage_pct=0.90,
            gross_return=0.0123, turnover=0.15,
            cost_bps=30, net_return=0.012,
        )

        write_attribution_json(result, tmp_dir / "pnl_attribution.json")
        write_attribution_md(result, tmp_dir / "pnl_attribution.md")

        assert (tmp_dir / "pnl_attribution.json").exists()
        assert (tmp_dir / "pnl_attribution.md").exists()

        with open(tmp_dir / "pnl_attribution.json") as f:
            data = json.load(f)
        assert data["schema"] == SCHEMA_VERSION
        assert data["gross_return"] == 0.0123
