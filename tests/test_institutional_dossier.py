#!/usr/bin/env python3
"""
Tests for institutional_dossier_generator.py
"""

import pytest
from decimal import Decimal
from datetime import date, datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from institutional_dossier_generator import (
    InstitutionalDossierGenerator,
    DossierContent,
    PriceTarget,
    Catalyst,
    RiskFactor,
    TechnicalLevel,
    InstitutionalHolder,
)


class TestPriceTarget:
    """Tests for PriceTarget dataclass."""

    def test_price_target_creation(self):
        pt = PriceTarget(
            scenario="bull",
            target_price=25.00,
            probability=0.35,
            upside_pct=150.0,
            rationale="Successful Phase 3 data",
        )
        assert pt.scenario == "bull"
        assert pt.target_price == 25.00
        assert pt.probability == 0.35
        assert pt.upside_pct == 150.0

    def test_price_target_weighted_return(self):
        pt = PriceTarget(
            scenario="bull",
            target_price=30.00,
            probability=0.40,
            upside_pct=100.0,
            rationale="Test",
        )
        weighted = pt.upside_pct * pt.probability
        assert weighted == 40.0


class TestCatalyst:
    """Tests for Catalyst dataclass."""

    def test_catalyst_creation(self):
        catalyst = Catalyst(
            event="Phase 3 Data Readout",
            expected_date="Q2 2026",
            timeframe="near",
            probability_of_success=0.65,
            impact_on_success="+150%",
            impact_on_failure="-60%",
            key_metrics=["Primary endpoint", "Safety"],
        )
        assert catalyst.event == "Phase 3 Data Readout"
        assert catalyst.timeframe == "near"
        assert catalyst.probability_of_success == 0.65


class TestRiskFactor:
    """Tests for RiskFactor dataclass."""

    def test_risk_factor_creation(self):
        risk = RiskFactor(
            category="clinical",
            description="Phase 3 trial failure risk",
            probability=0.35,
            impact_severity="high",
            mitigation="Diversified pipeline",
        )
        assert risk.category == "clinical"
        assert risk.impact_severity == "high"


class TestDossierContent:
    """Tests for DossierContent dataclass."""

    def test_dossier_content_defaults(self):
        content = DossierContent(
            ticker="TEST",
            company_name="Test Company",
            generation_date="2026-01-12",
        )
        assert content.ticker == "TEST"
        assert content.recommendation == "HOLD"  # default
        assert content.price_targets == []
        assert content.risk_factors == []

    def test_dossier_content_full(self):
        content = DossierContent(
            ticker="GOSS",
            company_name="Gossamer Bio",
            generation_date="2026-01-12",
            recommendation="BUY",
            current_price=2.50,
            market_cap=500_000_000,
            composite_score=72.5,
            catalyst_score=65.0,
            probability_score=55.0,
            timing_score=60.0,
            governance_score=48.0,
        )
        assert content.ticker == "GOSS"
        assert content.recommendation == "BUY"
        assert content.composite_score == 72.5


class TestInstitutionalDossierGenerator:
    """Tests for InstitutionalDossierGenerator."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create a generator with temp output dir."""
        return InstitutionalDossierGenerator(
            snapshot_path="nonexistent.json",  # Will use empty companies
            output_dir=str(tmp_path / "dossiers"),
        )

    def test_generator_init(self, generator):
        assert generator.companies == {}
        assert generator.output_dir.exists()

    def test_get_recommendation_strong_buy(self, generator):
        assert generator._get_recommendation(85) == "STRONG BUY"

    def test_get_recommendation_buy(self, generator):
        assert generator._get_recommendation(70) == "BUY"

    def test_get_recommendation_hold(self, generator):
        assert generator._get_recommendation(55) == "HOLD"

    def test_get_recommendation_sell(self, generator):
        assert generator._get_recommendation(40) == "SELL"

    def test_get_recommendation_strong_sell(self, generator):
        assert generator._get_recommendation(20) == "STRONG SELL"

    def test_calculate_probability_weighted_return(self, generator):
        targets = [
            PriceTarget("bull", 20.0, 0.30, 100.0, ""),
            PriceTarget("base", 12.0, 0.50, 20.0, ""),
            PriceTarget("bear", 5.0, 0.20, -50.0, ""),
        ]
        # 100*0.30 + 20*0.50 + (-50)*0.20 = 30 + 10 - 10 = 30
        result = generator._calculate_probability_weighted_return(targets)
        assert result == 30.0

    def test_calculate_probability_weighted_return_empty(self, generator):
        result = generator._calculate_probability_weighted_return([])
        assert result == 0.0

    def test_determine_catalyst_timeframe_near(self, generator):
        assert generator._determine_catalyst_timeframe(30) == "near"
        assert generator._determine_catalyst_timeframe(90) == "near"

    def test_determine_catalyst_timeframe_mid(self, generator):
        assert generator._determine_catalyst_timeframe(180) == "mid"
        assert generator._determine_catalyst_timeframe(365) == "mid"

    def test_determine_catalyst_timeframe_long(self, generator):
        assert generator._determine_catalyst_timeframe(400) == "long"
        assert generator._determine_catalyst_timeframe(None) == "long"

    def test_format_currency_billions(self, generator):
        assert generator._format_currency(5_000_000_000) == "$5.0B"

    def test_format_currency_millions(self, generator):
        assert generator._format_currency(150_000_000) == "$150.0M"

    def test_format_currency_thousands(self, generator):
        assert generator._format_currency(75_000) == "$75.0K"

    def test_estimate_runway(self, generator):
        # $120M cash, $10M/month burn = 12 months
        runway = generator._estimate_runway(120_000_000, 10_000_000)
        assert runway == 12.0

    def test_estimate_runway_zero_burn(self, generator):
        runway = generator._estimate_runway(100_000_000, 0)
        assert runway == 0.0

    def test_build_price_targets_algorithmic(self, generator):
        targets = generator._build_price_targets(
            current_price=10.0,
            composite_score=70.0,
            probability_score=60.0,
            analyst_targets={},
        )
        assert len(targets) == 3
        assert targets[0].scenario == "bull"
        assert targets[1].scenario == "base"
        assert targets[2].scenario == "bear"
        # Bull should have positive upside
        assert targets[0].upside_pct > 0
        # Bear should have negative upside
        assert targets[2].upside_pct < 0

    def test_build_price_targets_from_analyst(self, generator):
        analyst_targets = {
            "bull": {"price": 20.0, "probability": 0.30, "rationale": "Best case"},
            "base": {"price": 12.0, "probability": 0.50, "rationale": "Expected"},
            "bear": {"price": 5.0, "probability": 0.20, "rationale": "Worst case"},
        }
        targets = generator._build_price_targets(
            current_price=10.0,
            composite_score=70.0,
            probability_score=60.0,
            analyst_targets=analyst_targets,
        )
        assert len(targets) == 3
        assert targets[0].target_price == 20.0
        assert targets[0].probability == 0.30

    def test_build_dossier_content_minimal(self, generator):
        content = generator.build_dossier_content(ticker="TEST")
        assert content.ticker == "TEST"
        assert content.company_name == "TEST"  # Falls back to ticker
        assert content.recommendation in [
            "STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"
        ]

    def test_build_dossier_content_with_enhanced_score(self, generator):
        enhanced_score = {
            "enhanced_score": "75.50",
            "base_scores": {"catalyst_score": "65.0"},
            "probability": {"pos": "0.60"},
            "timing": {"cluster_convexity_bonus": "5.0"},
            "governance": {"penalty": "2.0"},
        }
        content = generator.build_dossier_content(
            ticker="TEST",
            enhanced_score=enhanced_score,
        )
        assert content.composite_score == 75.50
        assert content.catalyst_score == 65.0
        assert content.recommendation == "BUY"

    def test_generate_markdown_basic(self, generator):
        content = DossierContent(
            ticker="TEST",
            company_name="Test Bio Inc",
            generation_date="2026-01-12 10:00:00",
            recommendation="BUY",
            current_price=15.00,
            market_cap=1_000_000_000,
            composite_score=72.0,
        )
        markdown = generator.generate_markdown(content)

        assert "# INSTITUTIONAL INVESTMENT DOSSIER: Test Bio Inc (TEST)" in markdown
        assert "Recommendation: **BUY**" in markdown
        assert "Current Price | $15.00" in markdown
        assert "Composite Score | 72.0/100" in markdown

    def test_generate_markdown_with_catalysts(self, generator):
        content = DossierContent(
            ticker="TEST",
            company_name="Test Bio",
            generation_date="2026-01-12",
            catalysts=[
                Catalyst(
                    event="Phase 3 Data",
                    expected_date="Q2 2026",
                    timeframe="near",
                    probability_of_success=0.65,
                    impact_on_success="+100%",
                    impact_on_failure="-50%",
                    key_metrics=["Primary endpoint"],
                ),
            ],
        )
        markdown = generator.generate_markdown(content)
        assert "Near-Term Catalysts" in markdown
        assert "Phase 3 Data" in markdown
        assert "65%" in markdown

    def test_generate_markdown_with_risks(self, generator):
        content = DossierContent(
            ticker="TEST",
            company_name="Test Bio",
            generation_date="2026-01-12",
            risk_factors=[
                RiskFactor(
                    category="clinical",
                    description="Trial failure risk",
                    probability=0.35,
                    impact_severity="high",
                    mitigation="Pipeline diversification",
                ),
            ],
        )
        markdown = generator.generate_markdown(content)
        assert "RISK FRAMEWORK" in markdown
        assert "Trial failure risk" in markdown
        assert "35%" in markdown
        assert "HIGH" in markdown

    def test_generate_dossier_full(self, generator, tmp_path):
        enhanced_score = {
            "enhanced_score": "68.75",
            "base_scores": {"catalyst_score": "60.0"},
            "probability": {"pos": "0.55"},
            "timing": {"cluster_convexity_bonus": "3.0"},
            "governance": {"penalty": "0", "n_black_swans": "0"},
        }

        additional_context = {
            "thesis_points": [
                "Strong Phase 3 data expected",
                "Underappreciated pipeline value",
            ],
            "variant_perception": "Market underestimates near-term catalysts",
            "lead_program": "Lead Asset Phase 3",
            "moa": "Novel mechanism of action",
            "tam": "$5B addressable market",
        }

        markdown = generator.generate_dossier(
            ticker="TEST",
            enhanced_score=enhanced_score,
            additional_context=additional_context,
            save=True,
        )

        # Check file was saved
        output_file = tmp_path / "dossiers" / "TEST_institutional_dossier.md"
        assert output_file.exists()

        # Check content
        assert "TEST" in markdown
        assert "68.8" in markdown  # Rounded to 1 decimal place
        assert "Strong Phase 3 data expected" in markdown
        assert "Novel mechanism of action" in markdown

    def test_generate_dossier_no_save(self, generator):
        markdown = generator.generate_dossier(
            ticker="NOSAVE",
            save=False,
        )
        assert "NOSAVE" in markdown
        # File should not exist
        output_file = generator.output_dir / "NOSAVE_institutional_dossier.md"
        assert not output_file.exists()


class TestDossierIntegration:
    """Integration tests for dossier generation."""

    @pytest.fixture
    def generator_with_data(self, tmp_path):
        """Create generator with mock company data."""
        import json

        snapshot_path = tmp_path / "snapshot.json"
        companies = [
            {
                "ticker": "INTG",
                "market_data": {
                    "company_name": "Integration Test Bio",
                    "price": 12.50,
                    "market_cap": 800_000_000,
                    "volume_avg_30d": 500_000,
                    "52_week_high": 18.00,
                    "52_week_low": 8.00,
                },
                "financials": {
                    "cash": 200_000_000,
                    "debt": 50_000_000,
                    "revenue_ttm": 25_000_000,
                },
                "clinical": {
                    "lead_stage": "phase_3",
                    "active_trials": 8,
                    "total_trials": 15,
                },
                "data_quality": {
                    "overall_coverage": 85,
                    "financial_coverage": 90,
                    "sec_stale": False,
                },
            },
        ]
        with open(snapshot_path, "w") as f:
            json.dump(companies, f)

        return InstitutionalDossierGenerator(
            snapshot_path=str(snapshot_path),
            output_dir=str(tmp_path / "output"),
        )

    def test_full_integration_with_snapshot_data(self, generator_with_data):
        """Test dossier generation with actual company data."""
        content = generator_with_data.build_dossier_content(
            ticker="INTG",
            enhanced_score={
                "enhanced_score": "74.25",
                "base_scores": {"catalyst_score": "70.0"},
                "probability": {"pos": "0.62"},
                "timing": {"cluster_convexity_bonus": "4.0"},
                "governance": {"penalty": "0"},
            },
        )

        assert content.ticker == "INTG"
        assert content.company_name == "Integration Test Bio"
        assert content.current_price == 12.50
        assert content.market_cap == 800_000_000
        assert content.cash_position == 200_000_000
        assert content.clinical_stage == "phase_3"
        assert content.active_trials == 8
        assert content.data_coverage == 85
        assert content.recommendation == "BUY"

    def test_markdown_generation_completeness(self, generator_with_data):
        """Verify all sections are present in generated markdown."""
        markdown = generator_with_data.generate_dossier(
            ticker="INTG",
            save=False,
        )

        required_sections = [
            "EXECUTIVE SUMMARY",
            "INVESTMENT THESIS",
            "CATALYST ANALYSIS",
            "SCIENTIFIC & CLINICAL REVIEW",
            "COMMERCIAL OPPORTUNITY",
            "RISK FRAMEWORK",
            "VALUATION & PRICE TARGET",
            "INSTITUTIONAL POSITIONING",
            "TECHNICAL SETUP",
            "FINAL RECOMMENDATION",
        ]

        for section in required_sections:
            assert section in markdown, f"Missing section: {section}"

    def test_price_targets_sum_to_100(self, generator_with_data):
        """Verify price target probabilities sum to approximately 100%."""
        content = generator_with_data.build_dossier_content(ticker="INTG")

        total_prob = sum(pt.probability for pt in content.price_targets)
        assert 0.99 <= total_prob <= 1.01, f"Probabilities sum to {total_prob}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
