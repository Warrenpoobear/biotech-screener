"""Tests for scripts/make_ic_onepager.py — IC one-pager generator."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from make_ic_onepager import generate_ic_onepager


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def _minimal_health() -> dict:
    return {
        "status": "OK",
        "reasons": [],
        "metrics": {
            "a_count": 5,
            "catalyst_coverage_pct": 80.0,
            "name_turnover_pct": 10.0,
            "weight_l1_delta": 3.5,
        },
    }


def _minimal_delta() -> dict:
    return {
        "prior": {"date": "2026-02-15"},
        "portfolio_turnover": {
            "entrants": ["NEW1"],
            "exits": ["OLD1"],
            "name_turnover_pct": 5.0,
            "weight_l1_delta": 2.0,
        },
        "catalyst_coverage": {
            "current": {
                "n_dev": 100,
                "n_specific": 75,
                "pct": 75.0,
            },
        },
        "top_catalysts": {
            "current": [
                {"ticker": "AAA", "catalyst_days": 3, "tier_dev": "A"},
                {"ticker": "BBB", "catalyst_days": 10, "tier_dev": "B"},
                {"ticker": "CCC", "catalyst_days": 15, "tier_dev": "A"},
                {"ticker": "DDD", "catalyst_days": 20, "tier_dev": "C"},
                {"ticker": "EEE", "catalyst_days": 30, "tier_dev": "D"},
                {"ticker": "FFF", "catalyst_days": 50, "tier_dev": "B"},
            ],
        },
    }


def _minimal_source_mix() -> dict:
    return {
        "total_events": 500,
        "unique_tickers_with_events": 200,
        "by_source": {
            "CTGOV_CALENDAR": 400,
            "SEC_8K_FILING": 80,
            "FDA_ADCOM_CALENDAR": 20,
        },
    }


def _minimal_portfolio() -> list[dict]:
    return [
        {
            "ticker": "AAA",
            "tier_dev": "A",
            "actionable_rank": "1",
            "target_weight_pct": "5.50",
            "catalyst_days": "30",
            "catalyst_mode": "specific_days",
            "mom_state": "tailwind",
            "risk_flags": "",
            "size_band": "L",
        },
        {
            "ticker": "BBB",
            "tier_dev": "A",
            "actionable_rank": "2",
            "target_weight_pct": "4.00",
            "catalyst_days": "10",
            "catalyst_mode": "specific_days",
            "mom_state": "headwind",
            "risk_flags": "deep_drawdown",
            "size_band": "M",
        },
    ]


def _minimal_rankings() -> list[dict]:
    return [
        {
            "ticker": "AAA",
            "tier_dev": "A",
            "eligible": "1",
            "ineligible_reasons": "",
            "risk_flags": "",
            "top_3_drivers": "clinical_score:+22.0;financial_score:+10.0",
            "catalyst_reason_detail": "tier=A;reason=high_opt",
        },
        {
            "ticker": "BBB",
            "tier_dev": "A",
            "eligible": "1",
            "ineligible_reasons": "",
            "risk_flags": "deep_drawdown",
            "top_3_drivers": "momentum_score:+15.0;catalyst_score:+12.0",
            "catalyst_reason_detail": "tier=A;reason=high_opt",
        },
        {
            "ticker": "CCC",
            "tier_dev": "B",
            "eligible": "1",
            "ineligible_reasons": "",
            "risk_flags": "",
            "top_3_drivers": "smart_money_score:+20.0",
            "catalyst_reason_detail": "",
        },
        {
            "ticker": "DDD",
            "tier_dev": "C",
            "eligible": "0",
            "ineligible_reasons": "low_confidence",
            "risk_flags": "",
            "top_3_drivers": "",
            "catalyst_reason_detail": "",
        },
    ]


def _build_full_snapshot(tmp_path: Path) -> Path:
    """Build a complete minimal snapshot directory."""
    snap = tmp_path / "2026-02-16"
    snap.mkdir()
    _write_json(snap / "phase2_health.json", _minimal_health())
    _write_json(snap / "phase2_run_delta_details.json", _minimal_delta())
    _write_json(snap / "catalyst_source_mix.json", _minimal_source_mix())
    _write_csv(snap / "decision_portfolio.csv", _minimal_portfolio())
    _write_csv(snap / "rankings.csv", _minimal_rankings())
    return snap


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFullOnepager:
    def test_all_sections_present(self, tmp_path: Path) -> None:
        snap = _build_full_snapshot(tmp_path)
        md = generate_ic_onepager(snap)

        assert "# IC Run Summary" in md
        assert "## Health Gate" in md
        assert "## Portfolio" in md
        assert "## Delta vs Prior" in md
        assert "## Catalyst Coverage" in md
        assert "## Tier Distribution" in md
        assert "## Exceptions" in md

    def test_health_values(self, tmp_path: Path) -> None:
        snap = _build_full_snapshot(tmp_path)
        md = generate_ic_onepager(snap)

        assert "Health Gate: OK" in md
        assert "A-tier: 5" in md
        assert "80.0%" in md

    def test_date_in_title(self, tmp_path: Path) -> None:
        snap = _build_full_snapshot(tmp_path)
        md = generate_ic_onepager(snap)
        assert "2026-02-16" in md


class TestMissingFiles:
    def test_missing_delta_graceful(self, tmp_path: Path) -> None:
        snap = _build_full_snapshot(tmp_path)
        (snap / "phase2_run_delta_details.json").unlink()
        md = generate_ic_onepager(snap)

        assert "## Delta vs Prior" in md
        assert "Not available" in md
        # Other sections still present
        assert "## Health Gate: OK" in md
        assert "## Portfolio" in md

    def test_missing_health_graceful(self, tmp_path: Path) -> None:
        snap = _build_full_snapshot(tmp_path)
        (snap / "phase2_health.json").unlink()
        md = generate_ic_onepager(snap)

        assert "## Health Gate" in md
        assert "Not available" in md
        # Portfolio still works
        assert "## Portfolio (2 positions)" in md

    def test_missing_source_mix_graceful(self, tmp_path: Path) -> None:
        snap = _build_full_snapshot(tmp_path)
        (snap / "catalyst_source_mix.json").unlink()
        md = generate_ic_onepager(snap)

        # Catalyst section still has coverage from delta
        assert "## Catalyst Coverage" in md
        assert "specific_days: 75/100" in md

    def test_missing_all_jsons(self, tmp_path: Path) -> None:
        snap = _build_full_snapshot(tmp_path)
        for f in ["phase2_health.json", "phase2_run_delta_details.json",
                   "catalyst_source_mix.json"]:
            (snap / f).unlink()
        md = generate_ic_onepager(snap)
        # Should still produce a document with portfolio and tiers
        assert "## Portfolio (2 positions)" in md
        assert "## Tier Distribution" in md


class TestPortfolioExplainability:
    def test_top_3_drivers_in_table(self, tmp_path: Path) -> None:
        snap = _build_full_snapshot(tmp_path)
        md = generate_ic_onepager(snap)

        assert "clinical_score:+22.0" in md
        assert "momentum_score:+15.0" in md

    def test_missing_rankings_no_crash(self, tmp_path: Path) -> None:
        snap = _build_full_snapshot(tmp_path)
        (snap / "rankings.csv").unlink()
        md = generate_ic_onepager(snap)

        # Portfolio section still works, just no driver column
        assert "## Portfolio (2 positions)" in md
        # Tier section shows not available
        assert "## Tier Distribution" in md


class TestTierCounts:
    def test_tier_counts_match_rankings(self, tmp_path: Path) -> None:
        snap = _build_full_snapshot(tmp_path)
        md = generate_ic_onepager(snap)

        # Rankings: A=2, B=1, C=1, D=0
        assert "| A | 2 | 2 |" in md
        assert "| B | 1 | 0 |" in md
        assert "| C | 1 | 0 |" in md
        assert "| D | 0 | 0 |" in md


class TestExceptions:
    def test_ineligible_counted(self, tmp_path: Path) -> None:
        snap = _build_full_snapshot(tmp_path)
        md = generate_ic_onepager(snap)

        assert "Ineligible: 1" in md
        assert "low_confidence" in md

    def test_risk_flags_reported(self, tmp_path: Path) -> None:
        snap = _build_full_snapshot(tmp_path)
        md = generate_ic_onepager(snap)

        assert "Risk flags in portfolio: 1" in md
        assert "BBB: deep_drawdown" in md

    def test_health_warnings_when_warn(self, tmp_path: Path) -> None:
        snap = _build_full_snapshot(tmp_path)
        health = _minimal_health()
        health["status"] = "WARN"
        health["reasons"] = ["high_turnover", "catalyst_drop"]
        _write_json(snap / "phase2_health.json", health)

        md = generate_ic_onepager(snap)
        assert "Health Gate: WARN" in md
        assert "high_turnover" in md
        assert "catalyst_drop" in md


class TestDelta:
    def test_entrants_exits(self, tmp_path: Path) -> None:
        snap = _build_full_snapshot(tmp_path)
        md = generate_ic_onepager(snap)

        assert "Entrants: NEW1" in md
        assert "Exits: OLD1" in md
        assert "2026-02-15" in md

    def test_null_prior_graceful(self, tmp_path: Path) -> None:
        snap = _build_full_snapshot(tmp_path)
        delta = _minimal_delta()
        delta["prior"] = None
        _write_json(snap / "phase2_run_delta_details.json", delta)
        md = generate_ic_onepager(snap)

        assert "## Delta vs Prior (?)" in md

    def test_nearest_catalysts(self, tmp_path: Path) -> None:
        snap = _build_full_snapshot(tmp_path)
        md = generate_ic_onepager(snap)

        assert "AAA (3d, A)" in md
        assert "BBB (10d, B)" in md
        # Only first 5
        assert "FFF" not in md


class TestCLI:
    def test_cli_writes_output(self, tmp_path: Path) -> None:
        snap = _build_full_snapshot(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "make_ic_onepager.py"),
                "--snapshot-dir",
                str(snap),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        out_path = snap / "ic_onepager.md"
        assert out_path.exists()
        content = out_path.read_text()
        assert "# IC Run Summary" in content

    def test_cli_bad_dir(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "make_ic_onepager.py"),
                "--snapshot-dir",
                str(tmp_path / "nonexistent"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
