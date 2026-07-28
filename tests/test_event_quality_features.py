"""Tests for event quality features (regulatory_quality, clinical_quality, etc.)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.event_quality_features import (
    _SOURCE_PRECISION,
    EVENT_QUALITY_COLUMNS,
    OPTIONS_QUALITY_COLUMNS,
    PRECISION_WEAKNESS,
    compute_event_quality_features,
    compute_options_quality_composite,
    source_precision_floor,
    weakest_precision,
)


class TestPrecisionHelpers:
    """Shadow-only precision helpers — see specs/changes/spec_114_*.md.

    These exist so the CT.gov client can record what precision a corrected
    parser *would* assign, without any routing or scoring change.
    """

    def test_every_source_precision_value_is_orderable(self):
        """D3 guard: _SOURCE_PRECISION is the single authority, so every value
        it declares must be rankable by PRECISION_WEAKNESS."""
        for source, precision in _SOURCE_PRECISION.items():
            assert precision in PRECISION_WEAKNESS, f"{source} -> {precision} not orderable"

    def test_source_precision_floor_reads_the_table(self):
        assert source_precision_floor("CTGOV_CALENDAR") == _SOURCE_PRECISION["CTGOV_CALENDAR"]
        assert source_precision_floor("SEC_8K_FILING") == "DAY"
        assert source_precision_floor("CTGOV_PCD_FAR") == "QUARTER"

    def test_source_precision_floor_unknown_source(self):
        """Never default an unrecognised source to DAY."""
        assert source_precision_floor("NOT_A_REAL_SOURCE") == "UNKNOWN"

    def test_weakest_precision_prefers_the_weaker_claim(self):
        assert weakest_precision("DAY", "MONTH") == "MONTH"
        assert weakest_precision("MONTH", "DAY") == "MONTH"
        assert weakest_precision("QUARTER", "MONTH") == "QUARTER"
        assert weakest_precision("DAY", "DAY") == "DAY"

    def test_weakest_precision_ignores_unrankable(self):
        assert weakest_precision("MONTH", "NONSENSE") == "MONTH"

    def test_weakest_precision_no_known_input(self):
        assert weakest_precision("NONSENSE") == "UNKNOWN"
        assert weakest_precision() == "UNKNOWN"


class TestRegulatoryQuality:
    def test_pdufa_regulatory_high(self):
        row = {
            "catalyst_family": "REGULATORY",
            "catalyst_event_type": "PDUFA",
            "catalyst_source": "SEC_8K_FILING",
        }
        f = compute_event_quality_features(row)
        assert f["regulatory_quality"] > 0.8

    def test_adcom_regulatory(self):
        row = {
            "catalyst_family": "REGULATORY",
            "catalyst_event_type": "FDA_ADCOM",
            "catalyst_source": "FDA_FEDREG",
        }
        f = compute_event_quality_features(row)
        assert f["regulatory_quality"] > 0.7
        assert f["has_adcom"] == 1

    def test_non_regulatory_gets_zero(self):
        row = {
            "catalyst_family": "CLINICAL",
            "catalyst_event_type": "DATA_READOUT",
            "catalyst_source": "CTGOV_CALENDAR",
        }
        f = compute_event_quality_features(row)
        assert f["regulatory_quality"] == 0.0

    def test_submission_lower_than_pdufa(self):
        pdufa = compute_event_quality_features(
            {"catalyst_family": "REGULATORY", "catalyst_event_type": "PDUFA", "catalyst_source": "PDUFA_MANUAL"}
        )
        sub = compute_event_quality_features(
            {
                "catalyst_family": "REGULATORY",
                "catalyst_event_type": "FDA_SUBMISSION",
                "catalyst_source": "CTGOV_CALENDAR",
            }
        )
        assert pdufa["regulatory_quality"] > sub["regulatory_quality"]


class TestClinicalQuality:
    def test_data_readout_phase3(self):
        row = {
            "catalyst_family": "CLINICAL",
            "catalyst_event_type": "DATA_READOUT",
            "lead_program_phase": "3.0",
            "design_quality_score": "0.8",
            "program_count": "3",
        }
        f = compute_event_quality_features(row)
        assert f["clinical_quality"] > 0.7

    def test_pcd_phase1_lower(self):
        row = {
            "catalyst_family": "CLINICAL",
            "catalyst_event_type": "CT_PRIMARY_COMPLETION",
            "lead_program_phase": "1.0",
            "design_quality_score": "0.3",
            "program_count": "1",
        }
        f = compute_event_quality_features(row)
        assert f["clinical_quality"] < 0.5

    def test_non_clinical_gets_zero(self):
        row = {
            "catalyst_family": "REGULATORY",
            "catalyst_event_type": "PDUFA",
        }
        f = compute_event_quality_features(row)
        assert f["clinical_quality"] == 0.0

    def test_more_programs_higher_quality(self):
        base = {
            "catalyst_family": "CLINICAL",
            "catalyst_event_type": "DATA_READOUT",
            "lead_program_phase": "2.0",
            "design_quality_score": "0.5",
        }
        one = compute_event_quality_features({**base, "program_count": "1"})
        five = compute_event_quality_features({**base, "program_count": "5"})
        assert five["clinical_quality"] > one["clinical_quality"]


class TestHasAdcom:
    def test_adcom_flagged(self):
        f = compute_event_quality_features({"catalyst_family": "REGULATORY", "catalyst_event_type": "FDA_ADCOM"})
        assert f["has_adcom"] == 1

    def test_pdufa_not_adcom(self):
        f = compute_event_quality_features({"catalyst_family": "REGULATORY", "catalyst_event_type": "PDUFA"})
        assert f["has_adcom"] == 0

    def test_empty_not_adcom(self):
        f = compute_event_quality_features({})
        assert f["has_adcom"] == 0


class TestSingleAssetRisk:
    def test_single_program(self):
        f = compute_event_quality_features(
            {"catalyst_family": "CLINICAL", "catalyst_event_type": "DATA_READOUT", "program_count": "1"}
        )
        assert f["single_asset_risk"] == 1

    def test_multi_program(self):
        f = compute_event_quality_features(
            {"catalyst_family": "CLINICAL", "catalyst_event_type": "DATA_READOUT", "program_count": "3"}
        )
        assert f["single_asset_risk"] == 0

    def test_missing_program_count(self):
        f = compute_event_quality_features({"catalyst_family": "CLINICAL", "catalyst_event_type": "DATA_READOUT"})
        # missing → 0 (not single asset)
        assert f["single_asset_risk"] == 0


class TestEventQualityColumns:
    def test_all_columns_present(self):
        f = compute_event_quality_features({})
        for col in EVENT_QUALITY_COLUMNS:
            assert col in f

    def test_columns_list(self):
        assert "regulatory_quality" in EVENT_QUALITY_COLUMNS
        assert "clinical_quality" in EVENT_QUALITY_COLUMNS
        assert "has_adcom" in EVENT_QUALITY_COLUMNS
        assert "single_asset_risk" in EVENT_QUALITY_COLUMNS


# ---------------------------------------------------------------------------
# Options quality composite
# ---------------------------------------------------------------------------


class TestOptionsQualityComposite:
    def _base_row(self, **overrides):
        row = {
            "opt_use_for_judgment": "YES",
            "opt_event_premium": "YES",
            "opt_liquidity_ok": "1",
            "opt_iv_regime": "NORMAL",
            "opt_term_slope": "-0.20",
            "opt_put_call_skew": "0.05",
        }
        row.update(overrides)
        return row

    def test_options_quality_no_data(self):
        row = {"opt_use_for_judgment": ""}
        f = compute_options_quality_composite(row)
        assert f["options_quality_composite"] == ""

    def test_options_quality_not_usable(self):
        row = {"opt_use_for_judgment": "NO"}
        f = compute_options_quality_composite(row)
        assert f["options_quality_composite"] == ""

    def test_options_quality_full_signal(self):
        # YES + event premium(+0.40) + liquid(+0.20) + normal IV(0)
        # + slope -0.20 → +0.20 * (0.20/0.30) ≈ +0.1333
        # + skew 0.05 → +0.20 * (0.05/0.10) = +0.10
        # total ≈ 0.40 + 0.20 + 0.1333 + 0.10 = 0.8333
        row = self._base_row()
        f = compute_options_quality_composite(row)
        val = f["options_quality_composite"]
        assert isinstance(val, float)
        assert 0.55 < val < 0.90

    def test_options_quality_extreme_iv(self):
        row = self._base_row(opt_iv_regime="EXTREME")
        f = compute_options_quality_composite(row)
        base = compute_options_quality_composite(self._base_row())
        assert f["options_quality_composite"] < base["options_quality_composite"]

    def test_options_quality_no_event_premium(self):
        row = self._base_row(opt_event_premium="NO")
        f = compute_options_quality_composite(row)
        base = compute_options_quality_composite(self._base_row())
        # Missing event premium removes 0.40
        assert f["options_quality_composite"] < base["options_quality_composite"] - 0.3

    def test_options_quality_skew_bonus(self):
        no_skew = self._base_row(opt_put_call_skew="0")
        with_skew = self._base_row(opt_put_call_skew="0.08")
        f_no = compute_options_quality_composite(no_skew)
        f_yes = compute_options_quality_composite(with_skew)
        assert f_yes["options_quality_composite"] > f_no["options_quality_composite"]

    def test_options_quality_capped_at_1(self):
        # Max everything: event(0.40) + liq(0.20) + slope(0.20) + skew(0.20) = 1.0
        row = self._base_row(opt_term_slope="-0.50", opt_put_call_skew="0.20")
        f = compute_options_quality_composite(row)
        assert f["options_quality_composite"] <= 1.0

    def test_options_quality_columns_list(self):
        assert "options_quality_composite" in OPTIONS_QUALITY_COLUMNS
