#!/usr/bin/env python3
"""
test_pos_coverage_regression.py

Regression tests for PoS indication coverage to prevent the
"coverage improved but ranks got weird" failure mode.

Tests:
1. Coverage test - Assert PoS indication coverage >= X% for standard universe
2. No-overlap test - Verify patterns don't conflict with higher-priority rules
3. PIT sanity test - Assert v3 overrides have required temporal fields

Run: pytest tests/test_pos_coverage_regression.py -v

Author: Wake Robin Capital Management
"""

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Set

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from indication_mapper import IndicationMapper, MappingValidationError
from pos_engine import ProbabilityOfSuccessEngine


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mapper():
    """Create indication mapper instance."""
    return IndicationMapper()


@pytest.fixture
def pos_engine():
    """Create PoS engine instance."""
    return ProbabilityOfSuccessEngine()


@pytest.fixture
def as_of_date():
    """Standard as_of_date for tests."""
    return date(2026, 1, 15)


@pytest.fixture
def sample_ticker_universe() -> List[str]:
    """Representative sample of biotech tickers for coverage testing."""
    return [
        # Large cap
        "MRNA", "BNTX", "VRTX", "REGN", "BIIB", "GILD", "AMGN",
        # Oncology focus
        "SGEN", "INCY", "EXEL", "JAZZ", "ARVN", "NUVL", "RVMD",
        # Rare disease
        "ALNY", "BMRN", "SRPT", "CRSP", "BEAM", "EDIT", "NTLA",
        # CNS
        "NBIX", "AXSM", "ACAD", "BHVN", "PRTA",
        # Autoimmune
        "ARGX", "AUPH", "ARQT",
        # Other
        "CYTK", "URGN", "MDGL", "INSM",
    ]


# ============================================================================
# 1. COVERAGE REGRESSION TESTS
# ============================================================================

class TestCoverageRegression:
    """Tests to ensure PoS indication coverage doesn't regress."""

    MINIMUM_COVERAGE_PCT = 70.0  # Minimum acceptable coverage percentage

    def test_ticker_override_coverage_minimum(self, mapper, sample_ticker_universe, as_of_date):
        """Assert coverage is at or above minimum threshold for sample universe."""
        covered = 0
        uncovered = []

        for ticker in sample_ticker_universe:
            result = mapper.map_ticker(ticker, conditions=[], as_of_date=as_of_date)
            if result["indication"] is not None:
                covered += 1
            else:
                uncovered.append(ticker)

        coverage_pct = (covered / len(sample_ticker_universe)) * 100

        assert coverage_pct >= self.MINIMUM_COVERAGE_PCT, (
            f"Coverage {coverage_pct:.1f}% below minimum {self.MINIMUM_COVERAGE_PCT}%. "
            f"Uncovered tickers: {uncovered}"
        )

    def test_v3_overrides_count_minimum(self, mapper):
        """Assert minimum number of PIT-safe v3 overrides exist."""
        MIN_V3_OVERRIDES = 5  # Minimum v3 overrides expected

        v3_count = len(mapper.ticker_overrides_v3)
        assert v3_count >= MIN_V3_OVERRIDES, (
            f"Only {v3_count} v3 overrides found, expected at least {MIN_V3_OVERRIDES}"
        )

    def test_legacy_overrides_count(self, mapper):
        """Assert legacy overrides are still available for backwards compatibility."""
        MIN_LEGACY_OVERRIDES = 100  # Minimum legacy overrides expected

        legacy_count = len(mapper.ticker_overrides)
        assert legacy_count >= MIN_LEGACY_OVERRIDES, (
            f"Only {legacy_count} legacy overrides found, expected at least {MIN_LEGACY_OVERRIDES}"
        )

    def test_all_categories_represented_in_patterns(self, mapper):
        """Ensure all expected indication categories have patterns."""
        expected_categories = {
            "oncology", "rare_disease", "infectious_disease",
            "cardiovascular", "cns", "autoimmune", "metabolic",
            "respiratory", "ophthalmology", "gi_hepatology", "urology"
        }

        actual_categories = set(mapper.condition_patterns.keys())
        missing = expected_categories - actual_categories

        assert not missing, f"Missing categories in condition_patterns: {missing}"

    def test_pos_engine_has_all_indication_benchmarks(self, pos_engine):
        """Ensure PoS engine has benchmarks for all key categories."""
        # Get phase 3 benchmarks as reference
        phase_3_benchmarks = pos_engine.benchmarks.get("phase_3", {})

        expected_indications = {
            "oncology", "rare_disease", "infectious_disease", "neurology",
            "cardiovascular", "immunology", "metabolic", "respiratory",
            "ophthalmology", "urology"  # urology was added
        }

        actual_indications = set(phase_3_benchmarks.keys()) - {"_description", "all_indications"}
        missing = expected_indications - actual_indications

        assert not missing, f"Missing indications in PoS benchmarks: {missing}"


# ============================================================================
# 2. NO-OVERLAP / NO-CONTRADICTION TESTS
# ============================================================================

class TestNoOverlapContradiction:
    """Tests to ensure mapping rules don't conflict."""

    def test_mapper_validation_passes(self, mapper):
        """Mapper should pass internal validation without errors."""
        assert mapper.is_valid(), (
            f"Mapper validation failed with errors: {mapper.get_validation_errors()}"
        )

    def test_no_duplicate_patterns_across_categories(self, mapper):
        """No pattern should appear in multiple categories."""
        pattern_to_categories: Dict[str, List[str]] = {}

        for category, patterns in mapper.condition_patterns.items():
            for pattern in patterns:
                if pattern not in pattern_to_categories:
                    pattern_to_categories[pattern] = []
                pattern_to_categories[pattern].append(category)

        duplicates = {p: cats for p, cats in pattern_to_categories.items() if len(cats) > 1}
        assert not duplicates, f"Duplicate patterns across categories: {duplicates}"

    def test_v3_override_does_not_contradict_legacy(self, mapper, as_of_date):
        """V3 override indications should match legacy overrides when both exist."""
        contradictions = []

        for ticker in mapper.ticker_overrides_v3:
            if ticker in mapper.ticker_overrides:
                v3_indication = mapper.ticker_overrides_v3[ticker].get("indication")
                legacy_indication = mapper.ticker_overrides[ticker]

                # They should match (or v3 should be an alias of legacy)
                if v3_indication != legacy_indication:
                    # Check if they're equivalent via alias
                    alias_resolved = mapper.category_aliases.get(legacy_indication, legacy_indication)
                    v3_alias_resolved = mapper.category_aliases.get(v3_indication, v3_indication)

                    if v3_indication != legacy_indication and alias_resolved != v3_alias_resolved:
                        contradictions.append({
                            "ticker": ticker,
                            "v3": v3_indication,
                            "legacy": legacy_indication
                        })

        assert not contradictions, f"V3/legacy contradictions found: {contradictions}"

    def test_category_aliases_are_consistent(self, mapper):
        """Category aliases should point to valid PoS benchmark categories."""
        valid_pos_categories = {
            "oncology", "rare_disease", "infectious_disease", "neurology",
            "cardiovascular", "immunology", "metabolic", "respiratory",
            "dermatology", "ophthalmology", "gastroenterology", "hematology",
            "urology", "all_indications"
        }

        invalid_aliases = {}
        for alias_from, alias_to in mapper.category_aliases.items():
            if alias_to not in valid_pos_categories:
                invalid_aliases[alias_from] = alias_to

        assert not invalid_aliases, f"Aliases point to invalid categories: {invalid_aliases}"

    def test_mapper_and_pos_engine_category_alignment(self, mapper, pos_engine, as_of_date):
        """Categories from mapper should be recognized by PoS engine."""
        # Test each category from mapper
        test_conditions = {
            "oncology": ["breast cancer"],
            "rare_disease": ["orphan disease"],
            "infectious_disease": ["viral infection"],
            "cns": ["alzheimer disease"],  # Should alias to neurology
            "autoimmune": ["rheumatoid arthritis"],  # Should alias to immunology
            "metabolic": ["type 2 diabetes"],
            "respiratory": ["asthma treatment"],
            "ophthalmology": ["macular degeneration"],
            "gi_hepatology": ["liver disease"],  # Should alias to gastroenterology
            "urology": ["bladder dysfunction"],
            "cardiovascular": ["heart failure"],
        }

        alignment_issues = []

        for category, conditions in test_conditions.items():
            # Get indication from mapper
            mapper_result = mapper.map_ticker(
                ticker="TEST",
                conditions=conditions,
                as_of_date=as_of_date
            )
            mapper_indication = mapper_result.get("indication")

            if mapper_indication:
                # Check if PoS engine recognizes it
                pos_normalized = pos_engine._normalize_indication(mapper_indication)
                if pos_normalized == "all_indications" and mapper_indication != "other":
                    alignment_issues.append({
                        "mapper_category": category,
                        "mapper_indication": mapper_indication,
                        "pos_normalized": pos_normalized
                    })

        assert not alignment_issues, f"Mapper/PoS alignment issues: {alignment_issues}"


# ============================================================================
# 3. PIT SANITY TESTS
# ============================================================================

class TestPITSanity:
    """Tests for Point-in-Time safety of ticker overrides."""

    def test_all_v3_overrides_have_required_fields(self, mapper):
        """Every v3 override must have effective_from and evidence."""
        missing_fields = []

        required_fields = {"indication", "effective_from", "evidence"}

        for ticker, override in mapper.ticker_overrides_v3.items():
            missing = required_fields - set(override.keys())
            if missing:
                missing_fields.append({
                    "ticker": ticker,
                    "missing": list(missing)
                })

        assert not missing_fields, f"V3 overrides missing required fields: {missing_fields}"

    def test_v3_effective_from_dates_are_valid(self, mapper):
        """All effective_from dates should be valid ISO format."""
        invalid_dates = []

        for ticker, override in mapper.ticker_overrides_v3.items():
            effective_from = override.get("effective_from", "")
            try:
                date.fromisoformat(effective_from)
            except (ValueError, TypeError):
                invalid_dates.append({
                    "ticker": ticker,
                    "effective_from": effective_from
                })

        assert not invalid_dates, f"Invalid effective_from dates: {invalid_dates}"

    def test_v3_overrides_not_from_future(self, mapper):
        """V3 override effective_from should not be in the future."""
        today = date.today()
        future_overrides = []

        for ticker, override in mapper.ticker_overrides_v3.items():
            effective_from = override.get("effective_from", "")
            try:
                eff_date = date.fromisoformat(effective_from)
                if eff_date > today:
                    future_overrides.append({
                        "ticker": ticker,
                        "effective_from": effective_from
                    })
            except (ValueError, TypeError):
                pass  # Already caught in other test

        assert not future_overrides, f"V3 overrides with future effective_from: {future_overrides}"

    def test_v3_override_pit_admissibility(self, mapper):
        """V3 overrides should be PIT-admissible for dates after effective_from."""
        test_cases = []

        for ticker, override in mapper.ticker_overrides_v3.items():
            effective_from = override.get("effective_from", "")
            try:
                eff_date = date.fromisoformat(effective_from)

                # Test date before effective_from - should NOT be admissible
                before_date = date(eff_date.year - 1, eff_date.month, eff_date.day)
                is_admissible_before = mapper._is_v3_override_pit_admissible(override, before_date)

                # Test date after effective_from - should be admissible
                after_date = date(eff_date.year + 1, eff_date.month, eff_date.day)
                is_admissible_after = mapper._is_v3_override_pit_admissible(override, after_date)

                if is_admissible_before or not is_admissible_after:
                    test_cases.append({
                        "ticker": ticker,
                        "effective_from": effective_from,
                        "admissible_before": is_admissible_before,
                        "admissible_after": is_admissible_after
                    })

            except (ValueError, TypeError):
                pass  # Skip invalid dates

        assert not test_cases, f"PIT admissibility failures: {test_cases}"

    def test_v3_evidence_is_nct_or_documented(self, mapper):
        """V3 override evidence should be NCT ID or documented source."""
        # NCT IDs start with "NCT" followed by 8 digits
        import re
        nct_pattern = re.compile(r"^NCT\d{8}$")

        suspicious_evidence = []

        for ticker, override in mapper.ticker_overrides_v3.items():
            evidence = override.get("evidence", "")

            # Evidence should be NCT ID or have source_type specified
            has_nct = nct_pattern.match(evidence) is not None
            has_source_type = override.get("source_type") is not None

            if not has_nct and not has_source_type:
                suspicious_evidence.append({
                    "ticker": ticker,
                    "evidence": evidence
                })

        assert not suspicious_evidence, (
            f"V3 overrides with suspicious evidence (no NCT ID or source_type): "
            f"{suspicious_evidence}"
        )


# ============================================================================
# 4. DETERMINISM AND REPRODUCIBILITY TESTS
# ============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_mapper_produces_deterministic_output(self, mapper, as_of_date):
        """Same inputs should always produce same outputs."""
        tickers = ["MRNA", "VRTX", "UNKNOWN"]
        conditions = {"MRNA": ["covid vaccine"], "VRTX": ["cystic fibrosis"], "UNKNOWN": ["cancer"]}

        results1 = {
            t: mapper.map_ticker(t, conditions.get(t, []), as_of_date)
            for t in tickers
        }

        # Clear audit trail and run again
        mapper.clear_audit_trail()

        results2 = {
            t: mapper.map_ticker(t, conditions.get(t, []), as_of_date)
            for t in tickers
        }

        for ticker in tickers:
            assert results1[ticker]["indication"] == results2[ticker]["indication"], (
                f"Non-deterministic indication for {ticker}"
            )
            assert results1[ticker]["source"] == results2[ticker]["source"], (
                f"Non-deterministic source for {ticker}"
            )

    def test_precedence_is_deterministic(self, mapper, as_of_date):
        """Precedence rules should produce consistent results."""
        # MRNA is in both v3 and legacy - v3 should take precedence
        result = mapper.map_ticker("MRNA", conditions=["vaccine"], as_of_date=as_of_date)

        # Should use v3 since as_of_date is after effective_from
        assert result["source"] == "ticker_override_v3", (
            f"Expected ticker_override_v3 precedence, got {result['source']}"
        )


# ============================================================================
# 5. CATEGORY ALIAS RESOLUTION TESTS
# ============================================================================

class TestCategoryAliasResolution:
    """Tests for category alias resolution between mapper and PoS engine."""

    def test_cns_maps_to_neurology(self, mapper, pos_engine, as_of_date):
        """CNS category should map to neurology for PoS engine."""
        # Get a ticker with CNS indication from mapper
        result = mapper.map_ticker(
            "BIIB",
            conditions=["alzheimer disease"],
            as_of_date=as_of_date
        )

        indication = result.get("indication")

        # PoS engine should recognize it
        pos_normalized = pos_engine._normalize_indication(indication)
        assert pos_normalized == "neurology", (
            f"Expected 'neurology', got '{pos_normalized}' for CNS indication"
        )

    def test_autoimmune_maps_to_immunology(self, mapper, pos_engine, as_of_date):
        """Autoimmune category should map to immunology for PoS engine."""
        result = mapper.map_ticker(
            "ARGX",
            conditions=["autoimmune disease"],
            as_of_date=as_of_date
        )

        indication = result.get("indication")
        pos_normalized = pos_engine._normalize_indication(indication)

        assert pos_normalized == "immunology", (
            f"Expected 'immunology', got '{pos_normalized}' for autoimmune indication"
        )

    def test_gi_hepatology_maps_to_gastroenterology(self, mapper, pos_engine, as_of_date):
        """gi_hepatology category should map to gastroenterology for PoS engine."""
        result = mapper.map_ticker(
            "TEST",
            conditions=["liver disease", "hepatic disorder"],
            as_of_date=as_of_date
        )

        indication = result.get("indication")
        pos_normalized = pos_engine._normalize_indication(indication)

        assert pos_normalized == "gastroenterology", (
            f"Expected 'gastroenterology', got '{pos_normalized}' for gi_hepatology indication"
        )

    def test_urology_passes_through(self, mapper, pos_engine, as_of_date):
        """Urology category should be recognized by both mapper and PoS engine."""
        result = mapper.map_ticker(
            "URGN",
            conditions=["bladder dysfunction"],
            as_of_date=as_of_date
        )

        indication = result.get("indication")
        assert indication == "urology", f"Expected 'urology', got '{indication}'"

        pos_normalized = pos_engine._normalize_indication(indication)
        assert pos_normalized == "urology", (
            f"Expected 'urology', got '{pos_normalized}' from PoS engine"
        )


# ============================================================================
# 6. INTEGRATION TESTS
# ============================================================================

class TestMapperPosEngineIntegration:
    """Integration tests for mapper and PoS engine working together."""

    def test_full_pipeline_for_sample_tickers(
        self, mapper, pos_engine, sample_ticker_universe, as_of_date
    ):
        """Test full pipeline from mapper to PoS engine for sample universe."""
        results = []
        failures = []

        for ticker in sample_ticker_universe:
            # Step 1: Map indication
            mapper_result = mapper.map_ticker(
                ticker=ticker,
                conditions=[],
                as_of_date=as_of_date
            )
            indication = mapper_result.get("indication")

            # Step 2: Calculate PoS score
            pos_result = pos_engine.calculate_pos_score(
                base_stage="phase_2",
                indication=indication,
                as_of_date=as_of_date
            )

            # Step 3: Verify PoS engine recognized the indication
            if indication and pos_result["indication_normalized"] == "all_indications":
                failures.append({
                    "ticker": ticker,
                    "mapper_indication": indication,
                    "pos_normalized": pos_result["indication_normalized"]
                })

            results.append({
                "ticker": ticker,
                "indication": indication,
                "pos_score": pos_result["pos_score"],
                "loa_provenance": pos_result["loa_provenance"]
            })

        # Allow some failures but not too many
        MAX_ALLOWED_FAILURES = 3
        assert len(failures) <= MAX_ALLOWED_FAILURES, (
            f"Too many mapper->PoS integration failures ({len(failures)}): {failures}"
        )
