#!/usr/bin/env python3
"""
Tests for common/types.py

Tests shared type definitions including enums and dataclasses.
Covers:
- Severity enum values
- StatusGate enum values
- SecurityRecord dataclass
- CatalystRecord dataclass
- ClinicalScore dataclass
- CompositeRecord dataclass
"""

import pytest
from decimal import Decimal
from common.types import (
    Severity,
    StatusGate,
    SecurityRecord,
    CatalystRecord,
    ClinicalScore,
    CompositeRecord,
)


class TestSeverityEnum:
    """Tests for Severity enum."""

    def test_severity_none_value(self):
        """NONE should have value 'none'."""
        assert Severity.NONE.value == "none"

    def test_severity_sev1_value(self):
        """SEV1 should have value 'sev1'."""
        assert Severity.SEV1.value == "sev1"

    def test_severity_sev2_value(self):
        """SEV2 should have value 'sev2'."""
        assert Severity.SEV2.value == "sev2"

    def test_severity_sev3_value(self):
        """SEV3 should have value 'sev3'."""
        assert Severity.SEV3.value == "sev3"

    def test_severity_all_values(self):
        """All severity levels should be present."""
        values = [s.value for s in Severity]
        assert set(values) == {"none", "sev1", "sev2", "sev3"}

    def test_severity_from_value(self):
        """Should be able to create from string value."""
        assert Severity("none") == Severity.NONE
        assert Severity("sev1") == Severity.SEV1
        assert Severity("sev2") == Severity.SEV2
        assert Severity("sev3") == Severity.SEV3

    def test_severity_invalid_value_raises(self):
        """Invalid value should raise ValueError."""
        with pytest.raises(ValueError):
            Severity("invalid")

    def test_severity_comparison(self):
        """Severity enum members should be comparable."""
        assert Severity.NONE == Severity.NONE
        assert Severity.SEV1 != Severity.SEV2


class TestStatusGateEnum:
    """Tests for StatusGate enum."""

    def test_status_gate_active_value(self):
        """ACTIVE should have value 'active'."""
        assert StatusGate.ACTIVE.value == "active"

    def test_status_gate_excluded_shell_value(self):
        """EXCLUDED_SHELL should have value 'excluded_shell'."""
        assert StatusGate.EXCLUDED_SHELL.value == "excluded_shell"

    def test_status_gate_excluded_delisted_value(self):
        """EXCLUDED_DELISTED should have value 'excluded_delisted'."""
        assert StatusGate.EXCLUDED_DELISTED.value == "excluded_delisted"

    def test_status_gate_excluded_acquired_value(self):
        """EXCLUDED_ACQUIRED should have value 'excluded_acquired'."""
        assert StatusGate.EXCLUDED_ACQUIRED.value == "excluded_acquired"

    def test_status_gate_excluded_missing_data_value(self):
        """EXCLUDED_MISSING_DATA should have value 'excluded_missing_data'."""
        assert StatusGate.EXCLUDED_MISSING_DATA.value == "excluded_missing_data"

    def test_status_gate_excluded_small_cap_value(self):
        """EXCLUDED_SMALL_CAP should have value 'excluded_small_cap'."""
        assert StatusGate.EXCLUDED_SMALL_CAP.value == "excluded_small_cap"

    def test_status_gate_not_found_value(self):
        """NOT_FOUND should have value 'not_found'."""
        assert StatusGate.NOT_FOUND.value == "not_found"

    def test_status_gate_all_values(self):
        """All status gates should be present."""
        expected = {
            "active", "excluded_shell", "excluded_delisted",
            "excluded_acquired", "excluded_missing_data",
            "excluded_small_cap", "not_found"
        }
        values = {s.value for s in StatusGate}
        assert values == expected

    def test_status_gate_from_value(self):
        """Should be able to create from string value."""
        assert StatusGate("active") == StatusGate.ACTIVE
        assert StatusGate("excluded_shell") == StatusGate.EXCLUDED_SHELL


class TestSecurityRecord:
    """Tests for SecurityRecord dataclass."""

    def test_security_record_minimal(self):
        """Minimal record with only ticker."""
        record = SecurityRecord(ticker="ACME")
        assert record.ticker == "ACME"
        assert record.status == StatusGate.ACTIVE  # Default
        assert record.severity == Severity.NONE  # Default
        assert record.flags == []  # Default empty list

    def test_security_record_full(self):
        """Full record with all fields."""
        record = SecurityRecord(
            ticker="ACME",
            status=StatusGate.ACTIVE,
            market_cap_mm=Decimal("5000"),
            cash_mm=Decimal("500"),
            debt_mm=Decimal("100"),
            runway_months=Decimal("24"),
            severity=Severity.SEV1,
            flags=["low_cash"],
        )

        assert record.ticker == "ACME"
        assert record.status == StatusGate.ACTIVE
        assert record.market_cap_mm == Decimal("5000")
        assert record.cash_mm == Decimal("500")
        assert record.debt_mm == Decimal("100")
        assert record.runway_months == Decimal("24")
        assert record.severity == Severity.SEV1
        assert record.flags == ["low_cash"]

    def test_security_record_flags_default_empty(self):
        """Flags should default to empty list, not None."""
        record = SecurityRecord(ticker="TEST")
        assert record.flags == []
        assert record.flags is not None

    def test_security_record_flags_mutable(self):
        """Flags list should be mutable."""
        record = SecurityRecord(ticker="TEST")
        record.flags.append("new_flag")
        assert "new_flag" in record.flags

    def test_security_record_with_none_values(self):
        """Optional fields can be None."""
        record = SecurityRecord(
            ticker="TEST",
            market_cap_mm=None,
            cash_mm=None,
        )
        assert record.market_cap_mm is None
        assert record.cash_mm is None


class TestCatalystRecord:
    """Tests for CatalystRecord dataclass."""

    def test_catalyst_record_minimal(self):
        """Minimal record with required fields."""
        record = CatalystRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            phase="Phase 3",
        )

        assert record.ticker == "ACME"
        assert record.nct_id == "NCT12345678"
        assert record.phase == "Phase 3"

    def test_catalyst_record_full(self):
        """Full record with all fields."""
        record = CatalystRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            phase="Phase 3",
            primary_completion_date="2026-06-15",
            study_status="Active, not recruiting",
            indication="Solid Tumors",
        )

        assert record.primary_completion_date == "2026-06-15"
        assert record.study_status == "Active, not recruiting"
        assert record.indication == "Solid Tumors"

    def test_catalyst_record_defaults(self):
        """Optional fields default to None."""
        record = CatalystRecord(ticker="TEST", nct_id="NCT000", phase="Phase 1")
        assert record.primary_completion_date is None
        assert record.study_status is None
        assert record.indication is None


class TestClinicalScore:
    """Tests for ClinicalScore dataclass."""

    def test_clinical_score_creation(self):
        """Create clinical score with all fields."""
        score = ClinicalScore(
            ticker="ACME",
            phase_score=Decimal("25"),
            design_score=Decimal("20"),
            execution_score=Decimal("15"),
            endpoint_score=Decimal("10"),
            total_score=Decimal("70"),
            lead_phase="Phase 3",
        )

        assert score.ticker == "ACME"
        assert score.phase_score == Decimal("25")
        assert score.total_score == Decimal("70")
        assert score.lead_phase == "Phase 3"
        assert score.flags == []

    def test_clinical_score_flags_default(self):
        """Flags should default to empty list."""
        score = ClinicalScore(
            ticker="TEST",
            phase_score=Decimal("0"),
            design_score=Decimal("0"),
            execution_score=Decimal("0"),
            endpoint_score=Decimal("0"),
            total_score=Decimal("0"),
            lead_phase="Preclinical",
        )
        assert score.flags == []

    def test_clinical_score_with_flags(self):
        """Create score with flags."""
        score = ClinicalScore(
            ticker="TEST",
            phase_score=Decimal("25"),
            design_score=Decimal("20"),
            execution_score=Decimal("15"),
            endpoint_score=Decimal("10"),
            total_score=Decimal("70"),
            lead_phase="Phase 3",
            flags=["strong_data", "accelerated_pathway"],
        )
        assert score.flags == ["strong_data", "accelerated_pathway"]


class TestCompositeRecord:
    """Tests for CompositeRecord dataclass."""

    def test_composite_record_creation(self):
        """Create composite record with all required fields."""
        record = CompositeRecord(
            ticker="ACME",
            composite_score=Decimal("75.5"),
            composite_rank=5,
            clinical_dev_raw=Decimal("80"),
            financial_raw=Decimal("70"),
            catalyst_raw=Decimal("65"),
            clinical_dev_normalized=Decimal("85"),
            financial_normalized=Decimal("72"),
            catalyst_normalized=Decimal("68"),
            uncertainty_penalty=Decimal("5"),
            missing_subfactor_pct=Decimal("10"),
            market_cap_bucket="mid",
            stage_bucket="phase_3",
            severity=Severity.NONE,
            flags=[],
        )

        assert record.ticker == "ACME"
        assert record.composite_score == Decimal("75.5")
        assert record.composite_rank == 5
        assert record.rankable == True  # Default

    def test_composite_record_with_missing_scores(self):
        """Create record with missing raw scores."""
        record = CompositeRecord(
            ticker="INCOMPLETE",
            composite_score=Decimal("50"),
            composite_rank=10,
            clinical_dev_raw=None,
            financial_raw=Decimal("60"),
            catalyst_raw=None,
            clinical_dev_normalized=None,
            financial_normalized=Decimal("65"),
            catalyst_normalized=None,
            uncertainty_penalty=Decimal("15"),
            missing_subfactor_pct=Decimal("40"),
            market_cap_bucket="small",
            stage_bucket="phase_2",
            severity=Severity.SEV1,
            flags=["missing_clinical", "missing_catalyst"],
        )

        assert record.clinical_dev_raw is None
        assert record.catalyst_raw is None
        assert record.missing_subfactor_pct == Decimal("40")

    def test_composite_record_not_rankable(self):
        """Create non-rankable record."""
        record = CompositeRecord(
            ticker="EXCLUDED",
            composite_score=Decimal("0"),
            composite_rank=0,
            clinical_dev_raw=None,
            financial_raw=None,
            catalyst_raw=None,
            clinical_dev_normalized=None,
            financial_normalized=None,
            catalyst_normalized=None,
            uncertainty_penalty=Decimal("100"),
            missing_subfactor_pct=Decimal("100"),
            market_cap_bucket="unknown",
            stage_bucket="unknown",
            severity=Severity.SEV3,
            flags=["excluded"],
            rankable=False,
        )

        assert record.rankable == False


class TestEnumInteroperability:
    """Tests for enum interoperability with dataclasses."""

    def test_security_record_with_enum_values(self):
        """Dataclass should work with enum values."""
        record = SecurityRecord(
            ticker="TEST",
            status=StatusGate.EXCLUDED_ACQUIRED,
            severity=Severity.SEV2,
        )

        # Can access enum value
        assert record.status.value == "excluded_acquired"
        assert record.severity.value == "sev2"

        # Can compare with enum
        assert record.status == StatusGate.EXCLUDED_ACQUIRED
        assert record.severity == Severity.SEV2

    def test_severity_ordering_for_filtering(self):
        """Severity levels can be used for filtering."""
        records = [
            SecurityRecord(ticker="A", severity=Severity.NONE),
            SecurityRecord(ticker="B", severity=Severity.SEV1),
            SecurityRecord(ticker="C", severity=Severity.SEV2),
            SecurityRecord(ticker="D", severity=Severity.SEV3),
        ]

        # Filter out hard-gated (SEV3)
        active = [r for r in records if r.severity != Severity.SEV3]
        assert len(active) == 3
        assert all(r.severity != Severity.SEV3 for r in active)

    def test_status_gate_for_exclusion_check(self):
        """StatusGate can be used to check exclusion."""
        record = SecurityRecord(ticker="TEST", status=StatusGate.EXCLUDED_SHELL)

        excluded_statuses = {
            StatusGate.EXCLUDED_SHELL,
            StatusGate.EXCLUDED_DELISTED,
            StatusGate.EXCLUDED_ACQUIRED,
            StatusGate.EXCLUDED_MISSING_DATA,
            StatusGate.EXCLUDED_SMALL_CAP,
        }

        is_excluded = record.status in excluded_statuses
        assert is_excluded == True
