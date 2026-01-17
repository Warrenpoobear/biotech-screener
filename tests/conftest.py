#!/usr/bin/env python3
"""
Shared test fixtures for biotech-screener test suite.

Provides reusable fixtures for:
- Sample data (universe, financial, trials, market)
- Temporary data directories
- Common test utilities
- Standard as_of_date for deterministic tests
"""

import json
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add parent to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# STANDARD DATES
# ============================================================================

@pytest.fixture
def as_of_date() -> date:
    """Standard as_of_date for deterministic tests."""
    return date(2026, 1, 15)


@pytest.fixture
def as_of_date_str() -> str:
    """Standard as_of_date as string for CLI tests."""
    return "2026-01-15"


@pytest.fixture
def historical_date() -> date:
    """Historical date for PIT testing."""
    return date(2025, 6, 15)


# ============================================================================
# SAMPLE UNIVERSE DATA
# ============================================================================

@pytest.fixture
def sample_universe() -> List[Dict[str, Any]]:
    """Minimal valid universe for testing."""
    return [
        {
            "ticker": "ACME",
            "company_name": "Acme Therapeutics",
            "sector": "Biotechnology",
            "market_cap": 5_000_000_000,
            "is_active": True,
        },
        {
            "ticker": "BETA",
            "company_name": "Beta Pharma",
            "sector": "Biotechnology",
            "market_cap": 2_000_000_000,
            "is_active": True,
        },
        {
            "ticker": "GAMMA",
            "company_name": "Gamma BioSciences",
            "sector": "Biotechnology",
            "market_cap": 800_000_000,
            "is_active": True,
        },
    ]


@pytest.fixture
def universe_with_invalid_tickers() -> List[Dict[str, Any]]:
    """Universe containing invalid ticker formats."""
    return [
        {"ticker": "VALID", "company_name": "Valid Corp", "is_active": True},
        {"ticker": "123INVALID", "company_name": "Invalid Corp", "is_active": True},
        {"ticker": "", "company_name": "Empty Ticker", "is_active": True},
        {"ticker": "TOOLONGTICKERXYZ", "company_name": "Long Ticker", "is_active": True},
    ]


# ============================================================================
# SAMPLE FINANCIAL DATA
# ============================================================================

@pytest.fixture
def sample_financial_records() -> List[Dict[str, Any]]:
    """Financial records with cash, burn, runway data."""
    return [
        {
            "ticker": "ACME",
            "Cash": "500000000",
            "MarketableSecurities": "100000000",
            "CFO_quarterly": "-50000000",
            "CFO": "-200000000",
            "FCF": "-220000000",
            "NetIncome": "-180000000",
            "R&D": "150000000",
            "shares_outstanding": "100000000",
            "shares_outstanding_prior": "95000000",
            "as_of_date": "2025-12-31",
            "filing_date": "2026-01-10",
        },
        {
            "ticker": "BETA",
            "Cash": "200000000",
            "CFO_quarterly": "-30000000",
            "CFO": "-120000000",
            "shares_outstanding": "50000000",
            "as_of_date": "2025-12-31",
            "filing_date": "2026-01-08",
        },
        {
            "ticker": "GAMMA",
            "Cash": "50000000",
            "CFO_quarterly": "-20000000",
            "CFO": "-80000000",
            "shares_outstanding": "30000000",
            "as_of_date": "2025-12-31",
            "filing_date": "2026-01-12",
        },
    ]


@pytest.fixture
def financial_record_minimal() -> Dict[str, Any]:
    """Minimal financial record with only required fields."""
    return {
        "ticker": "MINIMAL",
        "Cash": "100000000",
        "as_of_date": "2025-12-31",
    }


@pytest.fixture
def financial_record_full() -> Dict[str, Any]:
    """Complete financial record with all fields."""
    return {
        "ticker": "COMPLETE",
        "Cash": "500000000",
        "MarketableSecurities": "100000000",
        "CFO_quarterly": "-50000000",
        "CFO_YTD": "-150000000",
        "CFO": "-200000000",
        "FCF_quarterly": "-55000000",
        "FCF": "-220000000",
        "NetIncome": "-180000000",
        "R&D": "150000000",
        "shares_outstanding": "100000000",
        "shares_outstanding_prior": "95000000",
        "burn_history": [-48000000, -52000000, -49000000, -51000000],
        "as_of_date": "2025-12-31",
        "filing_date": "2026-01-10",
    }


# ============================================================================
# SAMPLE CLINICAL TRIAL DATA
# ============================================================================

@pytest.fixture
def sample_trial_records() -> List[Dict[str, Any]]:
    """Clinical trial records for testing."""
    return [
        {
            "nct_id": "NCT12345678",
            "sponsor_ticker": "ACME",
            "brief_title": "Phase 3 Study of ACM-101 in Solid Tumors",
            "phase": "Phase 3",
            "overall_status": "Active, not recruiting",
            "primary_completion_date": "2026-06-15",
            "enrollment": 450,
            "conditions": ["Solid Tumors", "Cancer"],
            "interventions": [{"name": "ACM-101", "type": "Drug"}],
            "start_date": "2024-01-15",
        },
        {
            "nct_id": "NCT23456789",
            "sponsor_ticker": "ACME",
            "brief_title": "Phase 2 Study of ACM-102 in Rare Disease",
            "phase": "Phase 2",
            "overall_status": "Recruiting",
            "primary_completion_date": "2027-03-01",
            "enrollment": 80,
            "conditions": ["Rare Genetic Disorder"],
            "interventions": [{"name": "ACM-102", "type": "Biological"}],
            "start_date": "2025-06-01",
        },
        {
            "nct_id": "NCT34567890",
            "sponsor_ticker": "BETA",
            "brief_title": "Phase 1/2 Study of BET-201",
            "phase": "Phase 1/Phase 2",
            "overall_status": "Recruiting",
            "primary_completion_date": "2026-12-01",
            "enrollment": 120,
            "conditions": ["Autoimmune Disease"],
            "interventions": [{"name": "BET-201", "type": "Drug"}],
            "start_date": "2025-03-15",
        },
    ]


@pytest.fixture
def trial_with_pcd_imminent(as_of_date) -> Dict[str, Any]:
    """Trial with primary completion date within 90 days."""
    return {
        "nct_id": "NCT99999999",
        "sponsor_ticker": "URGENT",
        "phase": "Phase 3",
        "overall_status": "Active, not recruiting",
        "primary_completion_date": "2026-03-01",  # Within 90 days of as_of_date
        "enrollment": 300,
    }


# ============================================================================
# SAMPLE MARKET DATA
# ============================================================================

@pytest.fixture
def sample_market_data() -> List[Dict[str, Any]]:
    """Market data with prices and volumes."""
    return [
        {
            "ticker": "ACME",
            "price": "45.50",
            "volume": 1_500_000,
            "market_cap": 4_550_000_000,
            "avg_volume_20d": 1_200_000,
            "vol_60d": "0.35",
            "corr_xbi": "0.65",
            "drawdown_60d": "-0.12",
            "as_of_date": "2026-01-15",
        },
        {
            "ticker": "BETA",
            "price": "22.30",
            "volume": 800_000,
            "market_cap": 1_115_000_000,
            "avg_volume_20d": 600_000,
            "vol_60d": "0.45",
            "corr_xbi": "0.72",
            "drawdown_60d": "-0.18",
            "as_of_date": "2026-01-15",
        },
        {
            "ticker": "GAMMA",
            "price": "8.75",
            "volume": 2_000_000,
            "market_cap": 262_500_000,
            "avg_volume_20d": 1_800_000,
            "vol_60d": "0.65",
            "corr_xbi": "0.80",
            "drawdown_60d": "-0.25",
            "as_of_date": "2026-01-15",
        },
    ]


@pytest.fixture
def market_snapshot() -> Dict[str, Any]:
    """Market regime snapshot for enhancement modules."""
    return {
        "vix": "18.5",
        "xbi_vs_spy_30d": "0.02",
        "fed_rate_change_3m": "-0.25",
        "as_of_date": "2026-01-15",
    }


# ============================================================================
# SHORT INTEREST DATA
# ============================================================================

@pytest.fixture
def sample_short_interest() -> List[Dict[str, Any]]:
    """Short interest data for testing."""
    return [
        {
            "ticker": "ACME",
            "short_interest_pct": "15.5",
            "days_to_cover": "6.2",
            "short_interest_change_pct": "-5.0",
            "as_of_date": "2026-01-10",
        },
        {
            "ticker": "SQUEEZE",
            "short_interest_pct": "42.0",
            "days_to_cover": "12.5",
            "short_interest_change_pct": "8.0",
            "as_of_date": "2026-01-10",
        },
    ]


# ============================================================================
# CO-INVEST SIGNALS
# ============================================================================

@pytest.fixture
def sample_coinvest_signals() -> List[Dict[str, Any]]:
    """Co-invest signals from elite managers."""
    return [
        {
            "ticker": "ACME",
            "overlap_count": 3,
            "tier_1_count": 2,
            "conviction_score": 72.5,
            "holders": ["Baker Bros", "Perceptive", "RA Capital"],
            "as_of_quarter": "2025-Q3",
        },
        {
            "ticker": "BETA",
            "overlap_count": 2,
            "tier_1_count": 1,
            "conviction_score": 45.0,
            "holders": ["OrbiMed", "Deerfield"],
            "as_of_quarter": "2025-Q3",
        },
    ]


# ============================================================================
# TEMPORARY DATA DIRECTORIES
# ============================================================================

@pytest.fixture
def sample_data_dir(
    tmp_path,
    sample_universe,
    sample_financial_records,
    sample_trial_records,
    sample_market_data,
) -> Path:
    """Create temporary data directory with sample files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Write all required files
    (data_dir / "universe.json").write_text(
        json.dumps(sample_universe, indent=2)
    )
    (data_dir / "financial_records.json").write_text(
        json.dumps(sample_financial_records, indent=2)
    )
    (data_dir / "trial_records.json").write_text(
        json.dumps(sample_trial_records, indent=2)
    )
    (data_dir / "market_data.json").write_text(
        json.dumps(sample_market_data, indent=2)
    )

    # Create ctgov_state directory for Module 3
    ctgov_state = data_dir / "ctgov_state"
    ctgov_state.mkdir()

    return data_dir


@pytest.fixture
def full_sample_data_dir(
    sample_data_dir,
    sample_coinvest_signals,
    sample_short_interest,
    market_snapshot,
) -> Path:
    """Data directory with all optional files included."""
    # Add optional files
    (sample_data_dir / "coinvest_signals.json").write_text(
        json.dumps(sample_coinvest_signals, indent=2)
    )
    (sample_data_dir / "short_interest.json").write_text(
        json.dumps(sample_short_interest, indent=2)
    )
    (sample_data_dir / "market_snapshot.json").write_text(
        json.dumps(market_snapshot, indent=2)
    )

    return sample_data_dir


@pytest.fixture
def empty_data_dir(tmp_path) -> Path:
    """Empty data directory for missing file tests."""
    data_dir = tmp_path / "empty_data"
    data_dir.mkdir()
    return data_dir


# ============================================================================
# UTILITY FIXTURES
# ============================================================================

@pytest.fixture
def active_tickers(sample_universe) -> List[str]:
    """List of active ticker symbols."""
    return [r["ticker"] for r in sample_universe if r.get("is_active", True)]


@pytest.fixture
def decimal_precision() -> int:
    """Standard decimal precision for score comparisons."""
    return 2


def assert_decimal_equal(actual: Decimal, expected: Decimal, precision: int = 2):
    """Assert two Decimals are equal to given precision."""
    quantize_str = "0." + "0" * precision
    assert actual.quantize(Decimal(quantize_str)) == expected.quantize(Decimal(quantize_str)), \
        f"Expected {expected}, got {actual}"


def assert_score_bounded(score: Decimal, min_val: Decimal = Decimal("0"), max_val: Decimal = Decimal("100")):
    """Assert score is within valid bounds."""
    assert min_val <= score <= max_val, \
        f"Score {score} out of bounds [{min_val}, {max_val}]"


# ============================================================================
# MOCKING HELPERS
# ============================================================================

@pytest.fixture
def mock_date_today(monkeypatch, as_of_date):
    """Mock date.today() to return as_of_date (use sparingly - prefer explicit dates)."""
    import datetime

    class MockDate(datetime.date):
        @classmethod
        def today(cls):
            return as_of_date

    monkeypatch.setattr(datetime, "date", MockDate)


# ============================================================================
# POS ENGINE FIXTURES
# ============================================================================

@pytest.fixture
def pos_universe() -> List[Dict[str, Any]]:
    """Universe data for PoS engine testing."""
    return [
        {"ticker": "ONCO", "base_stage": "phase_3", "indication": "oncology"},
        {"ticker": "RARE", "base_stage": "phase_3", "indication": "rare disease"},
        {"ticker": "NEURO", "base_stage": "phase_2", "indication": "neurology"},
        {"ticker": "CARDIO", "base_stage": "phase_1", "indication": "cardiovascular"},
        {"ticker": "UNKNOWN", "base_stage": "phase_2", "indication": None},
    ]


# ============================================================================
# DEFENSIVE OVERLAY FIXTURES
# ============================================================================

@pytest.fixture
def defensive_features_elite() -> Dict[str, str]:
    """Defensive features for an elite diversifier."""
    return {
        "corr_xbi": "0.25",
        "vol_60d": "0.30",
        "drawdown_60d": "-0.08",
    }


@pytest.fixture
def defensive_features_high_corr() -> Dict[str, str]:
    """Defensive features for high-correlation stock."""
    return {
        "corr_xbi": "0.88",
        "vol_60d": "0.55",
        "drawdown_60d": "-0.22",
    }


@pytest.fixture
def defensive_features_placeholder() -> Dict[str, str]:
    """Defensive features with placeholder correlation."""
    return {
        "corr_xbi": "0.50",  # Placeholder value
        "vol_60d": "0.40",
    }
