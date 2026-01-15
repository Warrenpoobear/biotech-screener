#!/usr/bin/env python3
"""
test_catalyst_enhanced.py - Tests for Enhanced Catalyst Engines

Tests:
- Probability Engine (FDA base rates, TDQS, competitive, sponsor)
- Timing Engine (enrollment, sponsor delays, clustering, PDUFA)
- Governance Engine (fail-closed, black swan, validation)
- Enhanced Scoring Integration

All tests are deterministic and PIT-safe.
"""

import pytest
from datetime import date
from decimal import Decimal

# Import modules under test
from catalyst_probability_engine import (
    ProbabilityEngine,
    Phase,
    TherapeuticArea,
    TrialDesignProfile,
    TrialDesign,
    EndpointType,
    CompetitiveLandscape,
    SponsorTrackRecord,
    parse_phase,
    parse_therapeutic_area,
    FDA_BASE_RATES,
)
from catalyst_timing_engine import (
    TimingEngine,
    SponsorDelayProfile,
    SponsorReliability,
    PDUFADate,
    PDUFAType,
    estimate_enrollment_rate,
)
from catalyst_governance_engine import (
    GovernanceEngine,
    ValidationRule,
    ValidationSeverity,
    BlackSwanType,
    BlackSwanEvent,
    fail_closed_validate,
    FailClosedError,
)
from catalyst_enhanced_scoring import (
    EnhancedCatalystScoringEngine,
    EnhancedCatalystScore,
)


# =============================================================================
# PROBABILITY ENGINE TESTS
# =============================================================================

class TestProbabilityEngine:
    """Tests for ProbabilityEngine."""

    def test_fda_base_rates_exist(self):
        """FDA base rates should be defined for all phases."""
        assert Phase.P1 in FDA_BASE_RATES
        assert Phase.P2 in FDA_BASE_RATES
        assert Phase.P3 in FDA_BASE_RATES
        assert Phase.NDA_BLA in FDA_BASE_RATES

    def test_fda_base_rates_range(self):
        """FDA base rates should be between 0 and 1."""
        for phase, rates in FDA_BASE_RATES.items():
            for ta, rate in rates.items():
                assert Decimal("0") <= rate <= Decimal("1"), f"{phase} {ta}: {rate}"

    def test_probability_engine_basic(self):
        """Basic probability estimation should work."""
        engine = ProbabilityEngine()
        estimate = engine.estimate_probability(
            ticker="TEST",
            nct_id="NCT00000001",
            phase=Phase.P2,
            therapeutic_area=TherapeuticArea.ONCOLOGY,
            as_of_date=date(2026, 1, 12),
        )

        assert estimate.ticker == "TEST"
        assert estimate.nct_id == "NCT00000001"
        assert Decimal("0") < estimate.adjusted_probability < Decimal("1")
        assert estimate.confidence_interval_low < estimate.confidence_interval_high

    def test_probability_determinism(self):
        """Same inputs should produce same outputs."""
        engine = ProbabilityEngine()

        estimate1 = engine.estimate_probability(
            ticker="TEST",
            nct_id="NCT00000001",
            phase=Phase.P3,
            therapeutic_area=TherapeuticArea.CNS,
            as_of_date=date(2026, 1, 12),
        )

        estimate2 = engine.estimate_probability(
            ticker="TEST",
            nct_id="NCT00000001",
            phase=Phase.P3,
            therapeutic_area=TherapeuticArea.CNS,
            as_of_date=date(2026, 1, 12),
        )

        assert estimate1.adjusted_probability == estimate2.adjusted_probability
        assert estimate1.estimate_id == estimate2.estimate_id

    def test_tdqs_computation(self):
        """TDQS should compute correctly."""
        profile = TrialDesignProfile(
            design_type=TrialDesign.RANDOMIZED_CONTROLLED,
            primary_endpoint=EndpointType.OVERALL_SURVIVAL,
            target_enrollment=500,
            actual_enrollment=500,
            has_control_arm=True,
            has_biomarker_selection=True,
            has_breakthrough_designation=True,
        )

        tdqs = profile.compute_tdqs()
        assert tdqs > Decimal("80")  # High quality trial

    def test_tdqs_low_quality(self):
        """Low quality trial should have low TDQS."""
        profile = TrialDesignProfile(
            design_type=TrialDesign.SINGLE_ARM,
            primary_endpoint=EndpointType.BIOMARKER,
            target_enrollment=100,
            actual_enrollment=50,
            has_control_arm=False,
            has_biomarker_selection=False,
        )

        tdqs = profile.compute_tdqs()
        assert tdqs < Decimal("60")  # Low quality trial

    def test_competitive_adjustment(self):
        """Competitive landscape should affect probability."""
        # Crowded market
        crowded = CompetitiveLandscape(
            indication="breast cancer",
            n_approved_therapies=10,
            n_phase3_competitors=5,
            is_first_in_class=False,
            unmet_need_level="LOW",
        )
        assert crowded.compute_competitive_adjustment() < Decimal("1.0")

        # First-in-class with high unmet need
        novel = CompetitiveLandscape(
            indication="rare disease X",
            n_approved_therapies=0,
            n_phase3_competitors=0,
            is_first_in_class=True,
            unmet_need_level="HIGH",
        )
        assert novel.compute_competitive_adjustment() > Decimal("1.0")

    def test_sponsor_adjustment(self):
        """Sponsor track record should affect probability."""
        # Good track record
        good_sponsor = SponsorTrackRecord(
            sponsor_name="PFIZER",
            n_approvals_5yr=10,
            n_failures_5yr=2,
            n_crl_5yr=0,
        )
        assert good_sponsor.compute_sponsor_adjustment() >= Decimal("1.0")

        # Poor track record
        poor_sponsor = SponsorTrackRecord(
            sponsor_name="SMALLBIO",
            n_approvals_5yr=0,
            n_failures_5yr=5,
            n_crl_5yr=3,
            has_manufacturing_issues=True,
        )
        assert poor_sponsor.compute_sponsor_adjustment() < Decimal("1.0")

    def test_parse_phase(self):
        """Phase parsing should work."""
        assert parse_phase("Phase 1") == Phase.P1
        assert parse_phase("Phase 2") == Phase.P2
        assert parse_phase("Phase 3") == Phase.P3
        assert parse_phase("II") == Phase.P2
        assert parse_phase("III") == Phase.P3
        assert parse_phase("unknown") == Phase.UNKNOWN

    def test_parse_therapeutic_area(self):
        """Therapeutic area parsing should work."""
        assert parse_therapeutic_area("breast cancer") == TherapeuticArea.ONCOLOGY
        assert parse_therapeutic_area("Alzheimer's disease") == TherapeuticArea.CNS
        assert parse_therapeutic_area("diabetes") == TherapeuticArea.METABOLIC
        assert parse_therapeutic_area("rheumatoid arthritis") == TherapeuticArea.IMMUNOLOGY


# =============================================================================
# TIMING ENGINE TESTS
# =============================================================================

class TestTimingEngine:
    """Tests for TimingEngine."""

    def test_sponsor_delay_database(self):
        """Sponsor profiles should load from database."""
        profile = SponsorDelayProfile.from_database("PFIZER")
        assert profile.tier == SponsorReliability.TIER_1
        assert profile.delay_factor < Decimal("1.10")

    def test_sponsor_delay_unknown(self):
        """Unknown sponsors should get default profile."""
        profile = SponsorDelayProfile.from_database("UNKNOWN_SPONSOR_XYZ")
        assert profile.tier == SponsorReliability.UNKNOWN

    def test_timing_estimate_basic(self):
        """Basic timing estimation should work."""
        engine = TimingEngine()
        estimate = engine.estimate_readout_date(
            ticker="TEST",
            nct_id="NCT00000001",
            phase="P3",
            as_of_date=date(2026, 1, 12),
            target_enrollment=500,
            current_enrollment=400,
            enrollment_rate_per_month=Decimal("20"),
            sponsor_name="PFIZER",
        )

        assert estimate.ticker == "TEST"
        assert estimate.estimated_readout_days is not None or estimate.expected_primary_completion is None

    def test_timing_with_pcd(self):
        """Timing with PCD should use PCD as fallback."""
        engine = TimingEngine()
        estimate = engine.estimate_readout_date(
            ticker="TEST",
            nct_id="NCT00000001",
            phase="P3",
            as_of_date=date(2026, 1, 12),
            expected_primary_completion="2026-06-15",
            sponsor_name="MERCK",
        )

        assert estimate.estimated_readout_date is not None

    def test_cluster_detection(self):
        """Catalyst clustering should detect clusters."""
        engine = TimingEngine()

        # Multiple catalysts within 90 days
        catalyst_dates = [
            ("2026-02-01", "DATA_READOUT", Decimal("10")),
            ("2026-02-15", "PDUFA", Decimal("15")),
            ("2026-03-01", "CONFERENCE", Decimal("5")),
        ]

        clusters = engine.detect_clusters(catalyst_dates, "TEST")
        assert len(clusters) == 1
        assert clusters[0].is_convex
        assert clusters[0].n_catalysts == 3

    def test_no_cluster_far_apart(self):
        """Catalysts far apart should not cluster."""
        engine = TimingEngine()

        catalyst_dates = [
            ("2026-02-01", "DATA_READOUT", Decimal("10")),
            ("2026-06-01", "PDUFA", Decimal("15")),
        ]

        clusters = engine.detect_clusters(catalyst_dates, "TEST")
        assert len(clusters) == 0

    def test_pdufa_tracking(self):
        """PDUFA dates should be tracked correctly."""
        engine = TimingEngine()

        pdufa = PDUFADate(
            ticker="TEST",
            drug_name="TestDrug",
            indication="cancer",
            pdufa_type=PDUFAType.PRIORITY,
            action_date="2026-03-15",
            is_confirmed=True,
        )
        engine.add_pdufa_date(pdufa)

        upcoming = engine.get_upcoming_pdufas("TEST", date(2026, 1, 12))
        assert len(upcoming) == 1
        assert upcoming[0].days_until(date(2026, 1, 12)) == 62

    def test_enrollment_rate_estimation(self):
        """Enrollment rate should be estimated from history."""
        history = [
            ("2025-10-01", 100),
            ("2025-11-01", 120),
            ("2025-12-01", 145),
            ("2026-01-01", 170),
        ]

        rate = estimate_enrollment_rate(history)
        assert rate is not None
        assert rate > Decimal("0")


# =============================================================================
# GOVERNANCE ENGINE TESTS
# =============================================================================

class TestGovernanceEngine:
    """Tests for GovernanceEngine."""

    def test_pit_violation_detected(self):
        """PIT violations should be detected."""
        engine = GovernanceEngine(date(2026, 1, 12))

        events = [
            {
                "ticker": "TEST",
                "nct_id": "NCT00000001",
                "event_type": "DATA_READOUT",
                "source_date": "2026-01-15",  # Future date!
            }
        ]

        result = engine.validate_pit_compliance(events)
        assert not result.is_valid
        assert result.n_fatal > 0

    def test_pit_valid(self):
        """Valid PIT data should pass."""
        engine = GovernanceEngine(date(2026, 1, 12))

        events = [
            {
                "ticker": "TEST",
                "nct_id": "NCT00000001",
                "event_type": "DATA_READOUT",
                "source_date": "2026-01-10",  # Past date
            }
        ]

        result = engine.validate_pit_compliance(events)
        assert result.is_valid

    def test_critical_fields_missing(self):
        """Missing critical fields should be detected."""
        engine = GovernanceEngine(date(2026, 1, 12))

        events = [
            {
                "nct_id": "NCT00000001",
                # Missing ticker and event_type
            }
        ]

        result = engine.validate_critical_fields(events)
        assert not result.is_valid
        assert result.n_fatal >= 2

    def test_staleness_warning(self):
        """Stale data should trigger warnings/errors."""
        engine = GovernanceEngine(date(2026, 1, 12))

        # 100 days old = warning
        result_warning = engine.validate_data_staleness(["2025-10-04"])
        assert result_warning.n_warning > 0

        # 200 days old = error
        result_error = engine.validate_data_staleness(["2025-06-25"])
        assert result_error.n_error > 0

    def test_duplicate_detection(self):
        """Duplicate event IDs should be detected."""
        engine = GovernanceEngine(date(2026, 1, 12))

        event_ids = ["abc123", "def456", "abc123", "ghi789", "abc123"]
        result = engine.validate_duplicates(event_ids)

        assert result.n_error > 0

    def test_black_swan_detection(self):
        """Black swan events should be detected."""
        engine = GovernanceEngine(date(2026, 1, 12))

        events = [
            {
                "ticker": "TEST",
                "nct_id": "NCT00000001",
                "event_type": "STATUS_CHANGE",
                "new_value": "Clinical Hold - Full",
                "event_date": "2026-01-10",
            }
        ]

        black_swans = engine.detect_black_swans(events)
        assert len(black_swans) > 0
        assert black_swans[0].event_type == BlackSwanType.CLINICAL_HOLD

    def test_fail_closed_raises(self):
        """Fail-closed should raise on fatal violations."""
        events = [
            {
                "ticker": "TEST",
                "nct_id": "NCT00000001",
                "event_type": "DATA_READOUT",
                "source_date": "2027-01-01",  # Future date = fatal
            }
        ]

        with pytest.raises(FailClosedError):
            fail_closed_validate(events, date(2026, 1, 12), raise_on_fatal=True)

    def test_governance_score_adjustment(self):
        """Governance should adjust scores appropriately."""
        engine = GovernanceEngine(date(2026, 1, 12))

        # Clean validation
        clean_result = engine.run_all_validations([])
        clean_score, clean_adj = engine.compute_governance_score(clean_result, Decimal("70"))
        assert clean_score >= Decimal("70")

        # With errors
        engine2 = GovernanceEngine(date(2026, 1, 12))
        engine2.black_swans.append(BlackSwanEvent(
            ticker="TEST",
            event_type=BlackSwanType.CLINICAL_HOLD,
            event_date="2026-01-10",
            description="Test hold",
            severity_score=Decimal("90"),
        ))

        error_result = engine2.run_all_validations([])
        error_score, error_adj = engine2.compute_governance_score(error_result, Decimal("70"))
        assert error_score < Decimal("70")


# =============================================================================
# ENHANCED SCORING TESTS
# =============================================================================

class TestEnhancedScoring:
    """Tests for EnhancedCatalystScoringEngine."""

    def test_enhanced_scoring_basic(self):
        """Basic enhanced scoring should work."""
        engine = EnhancedCatalystScoringEngine(date(2026, 1, 12))

        score = engine.compute_enhanced_score(
            ticker="TEST",
            base_catalyst_score=Decimal("60"),
            base_proximity_score=Decimal("10"),
            base_delta_score=Decimal("5"),
        )

        assert score.ticker == "TEST"
        assert Decimal("0") <= score.enhanced_score <= Decimal("100")
        assert len(score.score_components) == 4

    def test_enhanced_scoring_with_trial_data(self):
        """Enhanced scoring with trial data should adjust score."""
        engine = EnhancedCatalystScoringEngine(date(2026, 1, 12))

        trial_data = {
            "nct_id": "NCT00000001",
            "phase": "Phase 3",
            "indication": "breast cancer",
            "target_enrollment": 500,
            "current_enrollment": 450,
            "enrollment_rate": 20,
            "primary_completion_date": "2026-06-15",
            "sponsor": "PFIZER",
        }

        score = engine.compute_enhanced_score(
            ticker="TEST",
            base_catalyst_score=Decimal("60"),
            base_proximity_score=Decimal("10"),
            base_delta_score=Decimal("5"),
            trial_data=trial_data,
        )

        assert score.probability_of_success > Decimal("0")
        assert score.estimated_readout_days is not None

    def test_enhanced_scoring_determinism(self):
        """Same inputs should produce same enhanced scores."""
        engine1 = EnhancedCatalystScoringEngine(date(2026, 1, 12))
        engine2 = EnhancedCatalystScoringEngine(date(2026, 1, 12))

        score1 = engine1.compute_enhanced_score(
            ticker="TEST",
            base_catalyst_score=Decimal("55"),
            base_proximity_score=Decimal("8"),
            base_delta_score=Decimal("3"),
        )

        score2 = engine2.compute_enhanced_score(
            ticker="TEST",
            base_catalyst_score=Decimal("55"),
            base_proximity_score=Decimal("8"),
            base_delta_score=Decimal("3"),
        )

        assert score1.enhanced_score == score2.enhanced_score
        assert score1.calculation_hash == score2.calculation_hash

    def test_batch_scoring(self):
        """Batch scoring should work for multiple tickers."""
        engine = EnhancedCatalystScoringEngine(date(2026, 1, 12))

        tickers_data = {
            "AAAA": {
                "base_catalyst_score": 60,
                "base_proximity_score": 10,
                "base_delta_score": 5,
            },
            "BBBB": {
                "base_catalyst_score": 45,
                "base_proximity_score": 0,
                "base_delta_score": -5,
            },
            "CCCC": {
                "base_catalyst_score": 75,
                "base_proximity_score": 20,
                "base_delta_score": 10,
            },
        }

        results = engine.batch_compute_enhanced_scores(tickers_data)

        assert len(results) == 3
        assert "AAAA" in results
        assert "BBBB" in results
        assert "CCCC" in results

    def test_score_serialization(self):
        """Enhanced scores should serialize to dict correctly."""
        engine = EnhancedCatalystScoringEngine(date(2026, 1, 12))

        score = engine.compute_enhanced_score(
            ticker="TEST",
            base_catalyst_score=Decimal("60"),
            base_proximity_score=Decimal("10"),
            base_delta_score=Decimal("5"),
        )

        score_dict = score.to_dict()

        assert "score_id" in score_dict
        assert "ticker" in score_dict
        assert "enhanced_score" in score_dict
        assert "probability" in score_dict
        assert "timing" in score_dict
        assert "governance" in score_dict
        assert "calculation_hash" in score_dict


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests across all engines."""

    def test_full_pipeline(self):
        """Full pipeline from events to enhanced score should work."""
        as_of_date = date(2026, 1, 12)

        events = [
            {
                "ticker": "TEST",
                "nct_id": "NCT00000001",
                "event_type": "CT_STATUS_UPGRADE",
                "event_date": "2026-01-10",
                "source_date": "2026-01-10",
                "disclosed_at": "2026-01-10",
            }
        ]

        trial_data = {
            "nct_id": "NCT00000001",
            "phase": "Phase 3",
            "indication": "lung cancer",
            "sponsor": "ROCHE",
        }

        engine = EnhancedCatalystScoringEngine(as_of_date)

        score = engine.compute_enhanced_score(
            ticker="TEST",
            base_catalyst_score=Decimal("65"),
            base_proximity_score=Decimal("15"),
            base_delta_score=Decimal("8"),
            trial_data=trial_data,
            events=events,
        )

        assert score.governance_passed
        assert score.n_black_swans == 0
        assert score.enhanced_score > Decimal("0")

    def test_black_swan_impact(self):
        """Black swan events should significantly impact score."""
        as_of_date = date(2026, 1, 12)

        # Events with black swan
        events = [
            {
                "ticker": "TEST",
                "nct_id": "NCT00000001",
                "event_type": "STATUS_CHANGE",
                "new_value": "Terminated",
                "event_date": "2026-01-10",
                "source_date": "2026-01-10",
                "disclosed_at": "2026-01-10",
            }
        ]

        engine = EnhancedCatalystScoringEngine(as_of_date)

        score_with_bs = engine.compute_enhanced_score(
            ticker="TEST",
            base_catalyst_score=Decimal("70"),
            base_proximity_score=Decimal("15"),
            base_delta_score=Decimal("10"),
            events=events,
        )

        # Score without black swan
        score_without_bs = engine.compute_enhanced_score(
            ticker="TEST2",
            base_catalyst_score=Decimal("70"),
            base_proximity_score=Decimal("15"),
            base_delta_score=Decimal("10"),
            events=[],
        )

        assert score_with_bs.n_black_swans > 0
        assert score_with_bs.enhanced_score < score_without_bs.enhanced_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
