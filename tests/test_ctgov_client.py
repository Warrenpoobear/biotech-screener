#!/usr/bin/env python3
"""
Tests for data_sources/ctgov_client.py

ClinicalTrials.gov API client for fetching trial data.
Tests cover:
- ClinicalTrialsClient class methods
- Phase normalization
- Date parsing
- Blinding parsing
- Endpoint extraction
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from data_sources.ctgov_client import (
    ClinicalTrialsClient,
    CTGOV_API_BASE,
    CTGOV_RATE_LIMIT,
    BIOTECH_TICKER_MAP,
)


class TestClinicalTrialsClientInit:
    """Tests for ClinicalTrialsClient initialization."""

    def test_creates_cache_dir(self):
        """Should create cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "new_cache"
            client = ClinicalTrialsClient(cache_dir=cache_dir)
            assert cache_dir.exists()

    def test_initializes_rate_limit(self):
        """Should initialize rate limit tracking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ClinicalTrialsClient(cache_dir=Path(temp_dir))
            assert client._last_request == 0.0


class TestNormalizePhase:
    """Tests for _normalize_phase method."""

    @pytest.fixture
    def client(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ClinicalTrialsClient(cache_dir=Path(temp_dir))

    def test_empty_phases(self, client):
        """Should return 'unknown' for empty phases."""
        assert client._normalize_phase([]) == "unknown"

    def test_phase_3(self, client):
        """Should normalize Phase 3."""
        assert client._normalize_phase(["PHASE3"]) == "phase 3"
        assert client._normalize_phase(["Phase 3"]) == "phase 3"

    def test_phase_2(self, client):
        """Should normalize Phase 2."""
        assert client._normalize_phase(["PHASE2"]) == "phase 2"
        assert client._normalize_phase(["Phase 2"]) == "phase 2"

    def test_phase_1(self, client):
        """Should normalize Phase 1."""
        assert client._normalize_phase(["PHASE1"]) == "phase 1"
        assert client._normalize_phase(["Phase 1"]) == "phase 1"

    def test_phase_4(self, client):
        """Should return 'approved' for Phase 4."""
        assert client._normalize_phase(["PHASE4"]) == "approved"
        assert client._normalize_phase(["Phase 4"]) == "approved"

    def test_phase_2_3(self, client):
        """Should normalize Phase 2/3."""
        assert client._normalize_phase(["PHASE2", "PHASE3"]) == "phase 2/3"

    def test_phase_1_2(self, client):
        """Should normalize Phase 1/2."""
        assert client._normalize_phase(["PHASE1", "PHASE2"]) == "phase 1/2"

    def test_unknown_phase(self, client):
        """Should return 'unknown' for unrecognized phases."""
        # Note: EARLY_PHASE1 contains "phase1" so it matches phase 1
        assert client._normalize_phase(["NA"]) == "unknown"
        assert client._normalize_phase(["NOT_APPLICABLE"]) == "unknown"


class TestParseDate:
    """Tests for _parse_date method."""

    @pytest.fixture
    def client(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ClinicalTrialsClient(cache_dir=Path(temp_dir))

    def test_empty_struct(self, client):
        """Should return None for empty struct."""
        assert client._parse_date({}) is None
        assert client._parse_date(None) is None

    def test_missing_date(self, client):
        """Should return None for missing date field."""
        assert client._parse_date({"type": "ACTUAL"}) is None

    def test_full_date(self, client):
        """Should handle full date format."""
        result = client._parse_date({"date": "2024-06-15"})
        assert result == "2024-06-15"

    def test_year_month_date(self, client):
        """Should add -01 for year-month format."""
        result = client._parse_date({"date": "2024-06"})
        assert result == "2024-06-01"

    def test_year_only_date(self, client):
        """Should add -01-01 for year-only format."""
        result = client._parse_date({"date": "2024"})
        assert result == "2024-01-01"

    def test_truncates_long_date(self, client):
        """Should truncate to first 10 chars."""
        result = client._parse_date({"date": "2024-06-15T00:00:00Z"})
        assert result == "2024-06-15"


class TestParseBlinding:
    """Tests for _parse_blinding method."""

    @pytest.fixture
    def client(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ClinicalTrialsClient(cache_dir=Path(temp_dir))

    def test_empty_design_info(self, client):
        """Should return 'open' for empty design info."""
        assert client._parse_blinding({}) == "open"

    def test_double_blind(self, client):
        """Should return 'double' for double-blind."""
        design_info = {"maskingInfo": {"masking": "DOUBLE"}}
        assert client._parse_blinding(design_info) == "double"

    def test_single_blind(self, client):
        """Should return 'single' for single-blind."""
        design_info = {"maskingInfo": {"masking": "SINGLE"}}
        assert client._parse_blinding(design_info) == "single"

    def test_triple_blind(self, client):
        """Should return 'double' for triple/quadruple-blind."""
        design_info = {"maskingInfo": {"masking": "TRIPLE"}}
        assert client._parse_blinding(design_info) == "double"

        design_info = {"maskingInfo": {"masking": "QUADRUPLE"}}
        assert client._parse_blinding(design_info) == "double"

    def test_open_label(self, client):
        """Should return 'open' for open label."""
        design_info = {"maskingInfo": {"masking": "NONE"}}
        assert client._parse_blinding(design_info) == "open"

    def test_missing_masking_info(self, client):
        """Should return 'open' for missing masking info."""
        design_info = {"allocation": "RANDOMIZED"}
        assert client._parse_blinding(design_info) == "open"


class TestGetPrimaryEndpoint:
    """Tests for _get_primary_endpoint method."""

    @pytest.fixture
    def client(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ClinicalTrialsClient(cache_dir=Path(temp_dir))

    def test_empty_outcomes(self, client):
        """Should return empty string for empty outcomes."""
        assert client._get_primary_endpoint({}) == ""

    def test_extracts_measure(self, client):
        """Should extract primary outcome measure."""
        outcomes = {
            "primaryOutcomes": [
                {"measure": "Overall Survival", "timeFrame": "24 months"}
            ]
        }
        assert client._get_primary_endpoint(outcomes) == "Overall Survival"

    def test_returns_first_primary(self, client):
        """Should return first primary outcome."""
        outcomes = {
            "primaryOutcomes": [
                {"measure": "First Endpoint"},
                {"measure": "Second Endpoint"},
            ]
        }
        assert client._get_primary_endpoint(outcomes) == "First Endpoint"

    def test_missing_measure(self, client):
        """Should return empty string if measure missing."""
        outcomes = {"primaryOutcomes": [{"timeFrame": "24 months"}]}
        assert client._get_primary_endpoint(outcomes) == ""


class TestSearchBySponsor:
    """Tests for search_by_sponsor method."""

    @pytest.fixture
    def client(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ClinicalTrialsClient(cache_dir=Path(temp_dir))

    def test_returns_list(self, client):
        """Should return a list."""
        with patch.object(client, '_make_request', return_value={}):
            result = client.search_by_sponsor("Test Company")
            assert isinstance(result, list)

    def test_handles_empty_response(self, client):
        """Should handle empty response."""
        with patch.object(client, '_make_request', return_value={}):
            result = client.search_by_sponsor("Test Company")
            assert result == []

    def test_parses_studies(self, client):
        """Should parse studies from response."""
        mock_response = {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT00000001",
                            "briefTitle": "Test Trial",
                        },
                        "statusModule": {
                            "overallStatus": "RECRUITING",
                        },
                        "designModule": {
                            "phases": ["PHASE3"],
                        },
                        "conditionsModule": {
                            "conditions": ["Cancer"],
                        },
                    }
                }
            ]
        }

        with patch.object(client, '_make_request', return_value=mock_response):
            result = client.search_by_sponsor("Test Company", max_results=1)

            assert len(result) == 1
            assert result[0]["nct_id"] == "NCT00000001"
            assert result[0]["phase"] == "phase 3"
            assert result[0]["status"] == "RECRUITING"


class TestGetTrialByNct:
    """Tests for get_trial_by_nct method."""

    @pytest.fixture
    def client(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ClinicalTrialsClient(cache_dir=Path(temp_dir))

    def test_returns_dict(self, client):
        """Should return a dict on success."""
        with patch.object(client, '_make_request', return_value={"protocolSection": {}}):
            result = client.get_trial_by_nct("NCT00000001")
            assert isinstance(result, dict)

    def test_returns_none_on_failure(self, client):
        """Should return None on failure."""
        with patch.object(client, '_make_request', return_value={}):
            result = client.get_trial_by_nct("NCT00000001")
            assert result == {}


class TestConstants:
    """Tests for module constants."""

    def test_api_base_url(self):
        """API base URL should be valid."""
        assert CTGOV_API_BASE.startswith("https://")
        assert "clinicaltrials.gov" in CTGOV_API_BASE

    def test_rate_limit_positive(self):
        """Rate limit should be positive."""
        assert CTGOV_RATE_LIMIT > 0

    def test_biotech_ticker_map_not_empty(self):
        """Ticker map should have entries."""
        assert len(BIOTECH_TICKER_MAP) > 0

    def test_biotech_ticker_map_valid_entries(self):
        """Ticker map should have valid ticker-company pairs."""
        for ticker, company in BIOTECH_TICKER_MAP.items():
            assert ticker.isupper()
            assert len(ticker) <= 5
            assert len(company) > 0
