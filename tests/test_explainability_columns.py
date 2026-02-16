"""Tests for explainability columns (top_3_drivers, catalyst_reason_detail)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decision_engine import compute_decision_fields, DecisionRuleset


# =============================================================================
# Helpers
# =============================================================================

def _base_rec(**overrides):
    rec = {
        "ticker": "TEST",
        "composite_score": 50.0,
        "severity": "NONE",
        "fundamental_red_flag": False,
        "confidence_overall": 0.65,
        "defensive_features": {
            "vol_60d": 0.80,
            "beta_xbi_60d": 1.20,
            "drawdown": -0.15,
            "rsi_14d": 50.0,
        },
        "smart_money_signal": {
            "tier1_holders": ["FundA", "FundB"],
            "holders_increasing": ["FundA"],
            "holders_decreasing": [],
            "overlap_count": 2,
        },
        "coinvest": {"tier1_count": 3},
        "catalyst_decay": {
            "days_to_catalyst": 45,
            "in_optimal_window": True,
        },
        "score_breakdown": {
            "enhancements": {
                "momentum": {"alpha_60d": 0.08},
            }
        },
    }
    rec.update(overrides)
    return rec


def _compute_top_3_drivers(row):
    """Replicate the top_3_drivers logic from run_screen.py for unit tests."""
    score_cols = [
        "clinical_score", "catalyst_score", "momentum_score",
        "financial_score", "smart_money_score",
    ]
    vals = {}
    for sc in score_cols:
        v = row.get(sc)
        if v is not None and v != "":
            try:
                vals[sc] = float(v)
            except (ValueError, TypeError):
                pass
    if not vals:
        return ""
    mean = sum(vals.values()) / len(vals)
    devs = [(k, v - mean) for k, v in vals.items()]
    devs.sort(key=lambda x: (-abs(x[1]), x[0]))
    return ";".join(f"{k}:{d:+.1f}" for k, d in devs[:3])


def _compute_catalyst_reason_detail(row):
    """Replicate the catalyst_reason_detail logic from run_screen.py."""
    cat_days = row.get("catalyst_days", "")
    cat_days_str = str(cat_days) if cat_days != "" else "NA"
    decay_w = row.get("catalyst_decay_w", "")
    decay_w_str = str(decay_w) if decay_w != "" else "NA"
    return (
        f"tier={row.get('tier_dev', '')}"
        f";reason={row.get('tier_reason', '')}"
        f";cat_days={cat_days_str}"
        f";strength={row.get('catalyst_strength', '')}"
        f";decay_w={decay_w_str}"
    )


# =============================================================================
# top_3_drivers tests
# =============================================================================

class TestTop3Drivers:
    def test_format(self):
        """Verify semicolon-separated, signed decimals."""
        row = {
            "clinical_score": 80.0,
            "catalyst_score": 60.0,
            "momentum_score": 40.0,
            "financial_score": 50.0,
            "smart_money_score": 70.0,
        }
        result = _compute_top_3_drivers(row)
        parts = result.split(";")
        assert len(parts) == 3
        for part in parts:
            name, val = part.split(":")
            assert name in row
            # Should have sign prefix
            assert val[0] in ("+", "-")
            float(val)  # Should parse as float

    def test_deterministic(self):
        """Same input should always give the same output."""
        row = {
            "clinical_score": 80.0,
            "catalyst_score": 60.0,
            "momentum_score": 40.0,
            "financial_score": 50.0,
            "smart_money_score": 70.0,
        }
        r1 = _compute_top_3_drivers(row)
        r2 = _compute_top_3_drivers(row)
        assert r1 == r2

    def test_missing_score_handled(self):
        """Missing score should be excluded from ranking."""
        row = {
            "clinical_score": 80.0,
            "catalyst_score": None,
            "momentum_score": 40.0,
            "financial_score": "",
            "smart_money_score": 70.0,
        }
        result = _compute_top_3_drivers(row)
        parts = result.split(";")
        assert len(parts) == 3
        # None and "" scores excluded
        for part in parts:
            name, _ = part.split(":")
            assert name in ("clinical_score", "momentum_score", "smart_money_score")

    def test_all_missing_returns_empty(self):
        """If all scores are missing, return empty string."""
        row = {
            "clinical_score": None,
            "catalyst_score": "",
            "momentum_score": None,
        }
        assert _compute_top_3_drivers(row) == ""

    def test_fewer_than_3_scores(self):
        """If only 2 scores exist, return 2."""
        row = {
            "clinical_score": 80.0,
            "catalyst_score": 40.0,
        }
        result = _compute_top_3_drivers(row)
        parts = result.split(";")
        assert len(parts) == 2

    def test_highest_deviation_first(self):
        """Score with largest deviation from mean should be first."""
        row = {
            "clinical_score": 100.0,  # +40 from mean of 60
            "catalyst_score": 60.0,   # 0
            "momentum_score": 60.0,   # 0
            "financial_score": 60.0,  # 0
            "smart_money_score": 20.0,  # -40
        }
        result = _compute_top_3_drivers(row)
        parts = result.split(";")
        # clinical_score and smart_money_score have |dev|=40, both first/second
        # Ties broken by col name asc
        first_name = parts[0].split(":")[0]
        assert first_name in ("clinical_score", "smart_money_score")

    def test_tie_broken_by_name(self):
        """When |dev| ties, sort by column name ascending."""
        row = {
            "clinical_score": 70.0,
            "financial_score": 30.0,
        }
        # mean = 50, both |dev| = 20
        result = _compute_top_3_drivers(row)
        parts = result.split(";")
        assert parts[0].startswith("clinical_score:")
        assert parts[1].startswith("financial_score:")


# =============================================================================
# catalyst_reason_detail tests
# =============================================================================

class TestCatalystReasonDetail:
    def test_with_days(self):
        """Should include cat_days=N when days present."""
        rec = _base_rec()
        decision = compute_decision_fields(rec, "drug_developer", 0.75)
        detail = _compute_catalyst_reason_detail(decision)
        assert "cat_days=45" in detail
        assert "tier=A" in detail
        assert "strength=near" in detail

    def test_missing_days(self):
        """Should show cat_days=NA when no catalyst data."""
        rec = _base_rec(catalyst_decay={})
        decision = compute_decision_fields(rec, "drug_developer", 0.75)
        detail = _compute_catalyst_reason_detail(decision)
        assert "cat_days=NA" in detail
        assert "strength=missing" in detail

    def test_decay_w_present(self):
        """Should include decay_w value."""
        rec = _base_rec()
        decision = compute_decision_fields(rec, "drug_developer", 0.75)
        detail = _compute_catalyst_reason_detail(decision)
        assert "decay_w=" in detail
        assert "decay_w=NA" not in detail

    def test_full_format(self):
        """Detail should have all 5 fields."""
        rec = _base_rec()
        decision = compute_decision_fields(rec, "drug_developer", 0.75)
        detail = _compute_catalyst_reason_detail(decision)
        fields = detail.split(";")
        assert len(fields) == 5
        field_names = [f.split("=")[0] for f in fields]
        assert field_names == ["tier", "reason", "cat_days", "strength", "decay_w"]

    def test_non_dev_empty_tier(self):
        """Non-dev archetype should have empty tier."""
        rec = _base_rec()
        decision = compute_decision_fields(rec, "commercial_biotech", 0.75)
        detail = _compute_catalyst_reason_detail(decision)
        assert detail.startswith("tier=;reason=;")


# =============================================================================
# SNAPSHOT_COLUMNS contract test
# =============================================================================

class TestSnapshotColumns:
    def test_snapshot_columns_include_new(self):
        from run_screen import SNAPSHOT_COLUMNS
        assert "top_3_drivers" in SNAPSHOT_COLUMNS
        assert "catalyst_reason_detail" in SNAPSHOT_COLUMNS
        assert "catalyst_decay_w" in SNAPSHOT_COLUMNS
