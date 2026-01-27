#!/usr/bin/env python3
"""
Tests for Partnership Validation Engine
"""

import pytest
from decimal import Decimal
from datetime import date

from partnership_engine import (
    PartnershipEngine,
    PartnerTier,
    DealType,
    PartnershipStrength,
    Partnership,
)


class TestPartnershipEngine:
    """Tests for PartnershipEngine."""

    @pytest.fixture
    def engine(self):
        return PartnershipEngine()

    @pytest.fixture
    def sample_partnerships(self):
        return [
            {
                "ticker": "ACME",
                "partner_name": "Pfizer",
                "deal_type": "exclusive_license",
                "announcement_date": "2025-06-15",
                "upfront_payment": 150,
                "total_potential_value": 1200,
                "indication": "oncology",
                "asset_name": "ACM-123",
                "is_active": True,
            },
            {
                "ticker": "ACME",
                "partner_name": "Roche",
                "deal_type": "co_development",
                "announcement_date": "2024-03-20",
                "upfront_payment": 75,
                "total_potential_value": 600,
                "indication": "immunology",
                "asset_name": "ACM-456",
                "is_active": True,
            },
            {
                "ticker": "BIOTECH",
                "partner_name": "Small Therapeutics",
                "deal_type": "collaboration",
                "announcement_date": "2025-01-10",
                "upfront_payment": 10,
                "total_potential_value": 50,
                "indication": "rare disease",
                "is_active": True,
            },
            {
                "ticker": "NOPART",
                "partner_name": "",
                "deal_type": "",
                "announcement_date": "2020-01-01",
            },
        ]

    def test_initialization(self, engine):
        """Engine initializes with empty state."""
        assert engine.partnerships_by_ticker == {}
        assert engine.profiles == {}
        assert engine.audit_trail == []
        assert engine._data_loaded is False

    def test_classify_partner_top_10(self, engine):
        """Top 10 pharma companies are classified correctly."""
        assert engine.classify_partner("Pfizer") == PartnerTier.TOP_10_PHARMA
        assert engine.classify_partner("Johnson & Johnson") == PartnerTier.TOP_10_PHARMA
        assert engine.classify_partner("Roche") == PartnerTier.TOP_10_PHARMA
        assert engine.classify_partner("Merck & Co") == PartnerTier.TOP_10_PHARMA
        assert engine.classify_partner("AbbVie Inc") == PartnerTier.TOP_10_PHARMA
        assert engine.classify_partner("Eli Lilly") == PartnerTier.TOP_10_PHARMA
        assert engine.classify_partner("AstraZeneca") == PartnerTier.TOP_10_PHARMA
        assert engine.classify_partner("Bristol-Myers Squibb") == PartnerTier.TOP_10_PHARMA
        assert engine.classify_partner("Sanofi") == PartnerTier.TOP_10_PHARMA
        assert engine.classify_partner("Novartis") == PartnerTier.TOP_10_PHARMA

    def test_classify_partner_major(self, engine):
        """Major pharma companies are classified correctly."""
        assert engine.classify_partner("Biogen") == PartnerTier.MAJOR_PHARMA
        assert engine.classify_partner("Moderna") == PartnerTier.MAJOR_PHARMA
        assert engine.classify_partner("Takeda") == PartnerTier.MAJOR_PHARMA
        assert engine.classify_partner("BioNTech") == PartnerTier.MAJOR_PHARMA

    def test_classify_partner_academic(self, engine):
        """Academic institutions are classified correctly."""
        assert engine.classify_partner("Harvard University") == PartnerTier.ACADEMIC
        assert engine.classify_partner("NIH") == PartnerTier.ACADEMIC
        assert engine.classify_partner("Memorial Sloan Kettering Cancer Center") == PartnerTier.ACADEMIC
        assert engine.classify_partner("MIT Research Institute") == PartnerTier.ACADEMIC

    def test_classify_partner_mid_tier(self, engine):
        """Mid-tier companies are classified correctly."""
        assert engine.classify_partner("Acme Pharmaceuticals") == PartnerTier.MID_TIER
        assert engine.classify_partner("XYZ Therapeutics") == PartnerTier.MID_TIER

    def test_classify_partner_small(self, engine):
        """Small biotechs are classified correctly."""
        assert engine.classify_partner("Tiny Biotech") == PartnerTier.SMALL_BIOTECH
        assert engine.classify_partner("Unknown Company") == PartnerTier.SMALL_BIOTECH

    def test_classify_deal_type(self, engine):
        """Deal types are classified correctly."""
        assert engine.classify_deal_type("exclusive license agreement") == DealType.EXCLUSIVE_LICENSE
        assert engine.classify_deal_type("non-exclusive license") == DealType.NON_EXCLUSIVE_LICENSE
        assert engine.classify_deal_type("co-development agreement") == DealType.CO_DEVELOPMENT
        assert engine.classify_deal_type("acquisition option agreement") == DealType.ACQUISITION_OPTION
        assert engine.classify_deal_type("option to license") == DealType.OPTION_AGREEMENT
        assert engine.classify_deal_type("research collaboration") == DealType.RESEARCH_COLLABORATION
        assert engine.classify_deal_type("supply and manufacturing") == DealType.SUPPLY_AGREEMENT
        assert engine.classify_deal_type("strategic collaboration") == DealType.COLLABORATION

    def test_load_partnerships(self, engine, sample_partnerships):
        """Partnerships are loaded correctly."""
        loaded = engine.load_partnerships(sample_partnerships)

        assert loaded >= 3  # At least 3 valid partnerships
        assert "ACME" in engine.partnerships_by_ticker
        assert "BIOTECH" in engine.partnerships_by_ticker
        assert len(engine.partnerships_by_ticker["ACME"]) == 2

    def test_build_profile(self, engine, sample_partnerships):
        """Profile is built correctly."""
        engine.load_partnerships(sample_partnerships)
        as_of = date(2026, 1, 26)

        profile = engine.build_profile("ACME", as_of)

        assert profile.ticker == "ACME"
        assert len(profile.partnerships) == 2
        assert profile.total_upfront == Decimal("225")  # 150 + 75
        assert profile.total_potential_value == Decimal("1800")  # 1200 + 600
        assert profile.top_tier_count == 2  # Pfizer + Roche
        assert profile.active_partnerships == 2
        assert profile.strength == PartnershipStrength.EXCEPTIONAL

    def test_build_profile_no_partnerships(self, engine):
        """Profile for ticker with no partnerships."""
        as_of = date(2026, 1, 26)

        profile = engine.build_profile("UNKNOWN", as_of)

        assert profile.ticker == "UNKNOWN"
        assert len(profile.partnerships) == 0
        assert profile.strength == PartnershipStrength.WEAK

    def test_score_ticker_with_partnerships(self, engine, sample_partnerships):
        """Score ticker with partnership data."""
        engine.load_partnerships(sample_partnerships)
        as_of = date(2026, 1, 26)

        result = engine.score_ticker("ACME", as_of)

        assert result["ticker"] == "ACME"
        assert result["partnership_score"] >= Decimal("0")
        assert result["partnership_score"] <= Decimal("100")
        assert result["partnership_strength"] == "exceptional"
        assert result["partnership_count"] == 2
        assert result["top_tier_partners"] == 2
        assert "Pfizer" in result["top_partners"]
        assert "Roche" in result["top_partners"]

    def test_score_ticker_without_partnerships(self, engine):
        """Score ticker without partnerships returns neutral score."""
        as_of = date(2026, 1, 26)

        result = engine.score_ticker("UNKNOWN", as_of)

        assert result["ticker"] == "UNKNOWN"
        assert result["partnership_score"] == Decimal("50")  # Neutral
        assert result["partnership_strength"] == "unknown"
        assert result["partnership_count"] == 0

    def test_score_components(self, engine, sample_partnerships):
        """Score components are calculated correctly."""
        engine.load_partnerships(sample_partnerships)
        as_of = date(2026, 1, 26)

        result = engine.score_ticker("ACME", as_of)

        assert "score_components" in result
        components = result["score_components"]
        assert "tier_score" in components
        assert "value_score" in components
        assert "recency_score" in components
        assert "diversity_score" in components

    def test_recency_scoring(self, engine):
        """Recent deals score higher than old deals."""
        old_partnerships = [
            {
                "ticker": "OLD",
                "partner_name": "Pfizer",
                "deal_type": "exclusive_license",
                "announcement_date": "2020-01-01",
                "upfront_payment": 100,
                "total_potential_value": 500,
                "is_active": True,
            },
        ]
        recent_partnerships = [
            {
                "ticker": "RECENT",
                "partner_name": "Pfizer",
                "deal_type": "exclusive_license",
                "announcement_date": "2025-10-01",
                "upfront_payment": 100,
                "total_potential_value": 500,
                "is_active": True,
            },
        ]

        engine.load_partnerships(old_partnerships + recent_partnerships)
        as_of = date(2026, 1, 26)

        old_score = engine.score_ticker("OLD", as_of)
        recent_score = engine.score_ticker("RECENT", as_of)

        # Recent deal should score higher due to recency component
        assert recent_score["score_components"]["recency_score"] > old_score["score_components"]["recency_score"]

    def test_value_scoring_tiers(self, engine):
        """Deal value tiers are scored correctly."""
        partnerships = [
            {
                "ticker": "MEGA",
                "partner_name": "Partner",
                "announcement_date": "2025-01-01",
                "total_potential_value": 1500,  # $1.5B
                "is_active": True,
            },
            {
                "ticker": "LARGE",
                "partner_name": "Partner",
                "announcement_date": "2025-01-01",
                "total_potential_value": 600,  # $600M
                "is_active": True,
            },
            {
                "ticker": "MODEST",
                "partner_name": "Partner",
                "announcement_date": "2025-01-01",
                "total_potential_value": 30,  # $30M
                "is_active": True,
            },
        ]

        engine.load_partnerships(partnerships)
        as_of = date(2026, 1, 26)

        mega_score = engine.score_ticker("MEGA", as_of)["score_components"]["value_score"]
        large_score = engine.score_ticker("LARGE", as_of)["score_components"]["value_score"]
        modest_score = engine.score_ticker("MODEST", as_of)["score_components"]["value_score"]

        assert mega_score == Decimal("100")
        assert large_score == Decimal("80")
        assert modest_score == Decimal("40")

    def test_diversity_scoring(self, engine):
        """Multiple partnerships score higher on diversity."""
        # Single partnership
        single = [
            {
                "ticker": "SINGLE",
                "partner_name": "Partner1",
                "announcement_date": "2025-01-01",
                "is_active": True,
            },
        ]
        # Multiple partnerships
        multiple = [
            {
                "ticker": "MULTI",
                "partner_name": f"Partner{i}",
                "announcement_date": "2025-01-01",
                "is_active": True,
            }
            for i in range(4)
        ]

        engine.load_partnerships(single + multiple)
        as_of = date(2026, 1, 26)

        single_diversity = engine.score_ticker("SINGLE", as_of)["score_components"]["diversity_score"]
        multi_diversity = engine.score_ticker("MULTI", as_of)["score_components"]["diversity_score"]

        assert multi_diversity > single_diversity
        assert multi_diversity == Decimal("100")  # 4+ partnerships
        assert single_diversity == Decimal("40")  # 1 partnership

    def test_score_universe(self, engine, sample_partnerships):
        """Score entire universe."""
        universe = [
            {"ticker": "ACME"},
            {"ticker": "BIOTECH"},
            {"ticker": "UNKNOWN"},
        ]
        as_of = date(2026, 1, 26)

        result = engine.score_universe(universe, sample_partnerships, as_of)

        assert result["diagnostic_counts"]["total_scored"] == 3
        assert result["diagnostic_counts"]["with_partnerships"] >= 2
        assert "strength_distribution" in result["diagnostic_counts"]
        assert "provenance" in result

    def test_partnership_strength_exceptional(self, engine):
        """Exceptional strength with multiple top-tier deals."""
        partnerships = [
            {
                "ticker": "EXCEPT",
                "partner_name": "Pfizer",
                "announcement_date": "2025-01-01",
                "total_potential_value": 500,
                "is_active": True,
            },
            {
                "ticker": "EXCEPT",
                "partner_name": "Roche",
                "announcement_date": "2025-01-01",
                "total_potential_value": 500,
                "is_active": True,
            },
        ]
        engine.load_partnerships(partnerships)
        as_of = date(2026, 1, 26)

        profile = engine.build_profile("EXCEPT", as_of)

        assert profile.strength == PartnershipStrength.EXCEPTIONAL
        assert profile.top_tier_count == 2

    def test_partnership_strength_from_mega_deal(self, engine):
        """Exceptional strength from mega deal value alone."""
        partnerships = [
            {
                "ticker": "MEGA",
                "partner_name": "Small Therapeutics",  # Not top tier
                "announcement_date": "2025-01-01",
                "total_potential_value": 1500,  # But mega deal
                "is_active": True,
            },
        ]
        engine.load_partnerships(partnerships)
        as_of = date(2026, 1, 26)

        profile = engine.build_profile("MEGA", as_of)

        assert profile.strength == PartnershipStrength.EXCEPTIONAL

    def test_get_partnership_details(self, engine, sample_partnerships):
        """Get detailed partnership info."""
        engine.load_partnerships(sample_partnerships)

        details = engine.get_partnership_details("ACME")

        assert len(details) == 2
        assert details[0]["partner_name"] == "Pfizer"
        assert details[0]["partner_tier"] == "top_10_pharma"

    def test_audit_trail(self, engine, sample_partnerships):
        """Audit trail is maintained."""
        engine.load_partnerships(sample_partnerships)
        as_of = date(2026, 1, 26)
        engine.score_ticker("ACME", as_of)

        trail = engine.get_audit_trail()
        assert len(trail) > 0
        assert trail[0]["ticker"] == "ACME"

        engine.clear_audit_trail()
        assert len(engine.get_audit_trail()) == 0

    def test_strength_rating(self, engine):
        """Strength rating categories are correct."""
        assert engine.get_strength_rating(Decimal("85")) == "exceptional"
        assert engine.get_strength_rating(Decimal("70")) == "strong"
        assert engine.get_strength_rating(Decimal("50")) == "moderate"
        assert engine.get_strength_rating(Decimal("20")) == "weak"

    def test_point_in_time_filtering(self, engine):
        """Only partnerships announced before as_of_date are included."""
        partnerships = [
            {
                "ticker": "PIT",
                "partner_name": "Pfizer",
                "announcement_date": "2024-01-01",  # Before as_of
                "is_active": True,
            },
            {
                "ticker": "PIT",
                "partner_name": "Roche",
                "announcement_date": "2027-01-01",  # After as_of
                "is_active": True,
            },
        ]
        engine.load_partnerships(partnerships)
        as_of = date(2026, 1, 26)

        profile = engine.build_profile("PIT", as_of)

        # Only the 2024 partnership should be included
        assert len(profile.partnerships) == 1
        assert profile.partnerships[0].partner_name == "Pfizer"


class TestPartnerTierScoring:
    """Tests for partner tier scoring adjustments."""

    @pytest.fixture
    def engine(self):
        return PartnershipEngine()

    def test_all_tiers_have_scores(self):
        """All partner tiers have score values."""
        for tier in PartnerTier:
            assert tier in PartnershipEngine.TIER_SCORES

    def test_all_deal_types_have_multipliers(self):
        """All deal types have multipliers."""
        for deal_type in DealType:
            assert deal_type in PartnershipEngine.DEAL_TYPE_MULTIPLIERS

    def test_top_tier_scores_highest(self, engine):
        """Top 10 pharma scores highest."""
        top_10_score = engine.TIER_SCORES[PartnerTier.TOP_10_PHARMA]
        major_score = engine.TIER_SCORES[PartnerTier.MAJOR_PHARMA]
        mid_score = engine.TIER_SCORES[PartnerTier.MID_TIER]
        small_score = engine.TIER_SCORES[PartnerTier.SMALL_BIOTECH]

        assert top_10_score > major_score
        assert major_score > mid_score
        assert mid_score > small_score

    def test_exclusive_license_multiplier_highest(self, engine):
        """Exclusive license has highest multiplier."""
        exclusive = engine.DEAL_TYPE_MULTIPLIERS[DealType.EXCLUSIVE_LICENSE]
        non_exclusive = engine.DEAL_TYPE_MULTIPLIERS[DealType.NON_EXCLUSIVE_LICENSE]
        supply = engine.DEAL_TYPE_MULTIPLIERS[DealType.SUPPLY_AGREEMENT]

        assert exclusive > non_exclusive
        assert exclusive > supply
