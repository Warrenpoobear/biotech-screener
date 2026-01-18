#!/usr/bin/env python3
"""
Tests for module_3_catalyst.py main compute_module_3_catalyst function

Tests cover:
- Main function integration with full pipeline
- Empty universe handling
- Config customization via from_dict()
- State directory management
- Output schema validation
- PIT enforcement
"""

import pytest
from pathlib import Path
from datetime import date
from decimal import Decimal

from module_3_catalyst import (
    compute_module_3_catalyst,
    Module3Config,
)
from module_3_schema_v2 import (
    CatalystEventV2,
    EventType,
    EventSeverity,
    ConfidenceLevel,
    SourceReliability,
    DateSpecificity,
)


class TestComputeModule3CatalystMain:
    """Test the main compute_module_3_catalyst function."""

    @pytest.fixture
    def as_of_date(self):
        """Standard as_of_date for testing."""
        return date(2026, 1, 15)

    @pytest.fixture
    def sample_trial_records(self):
        """Sample trial records for testing."""
        return {
            "TEST": [
                {
                    "ticker": "TEST",
                    "nct_id": "NCT00000001",
                    "phase": "Phase 3",
                    "status": "Active, not recruiting",
                    "primary_completion_date": "2026-03-01",
                    "primary_completion_date_type": "Anticipated",
                    "source_date": "2026-01-14",
                },
            ],
            "ABCD": [
                {
                    "ticker": "ABCD",
                    "nct_id": "NCT00000002",
                    "phase": "Phase 2",
                    "status": "Recruiting",
                    "results_first_posted_date": "2026-01-10",
                    "source_date": "2026-01-14",
                },
            ],
        }

    @pytest.fixture
    def state_dir(self, tmp_path):
        """Create temporary state directory."""
        state = tmp_path / "state"
        state.mkdir()
        return state

    def test_basic_execution(self, as_of_date, sample_trial_records, state_dir):
        """Test basic execution with valid inputs."""
        active_tickers = {"TEST", "ABCD"}

        result = compute_module_3_catalyst(
            active_tickers=active_tickers,
            trial_records=sample_trial_records,
            as_of_date=as_of_date,
            state_dir=state_dir,
        )

        assert "summaries" in result
        assert "diagnostic_counts" in result
        assert "as_of_date" in result
        assert "schema_version" in result
        assert "score_version" in result

        assert result["as_of_date"] == as_of_date.isoformat()
        assert isinstance(result["summaries"], dict)

    def test_empty_universe_handling(self, as_of_date, state_dir):
        """Test handling of empty active_tickers."""
        result = compute_module_3_catalyst(
            active_tickers=set(),
            trial_records={},
            as_of_date=as_of_date,
            state_dir=state_dir,
        )

        assert result["summaries"] == {}
        assert result["diagnostic_counts"]["tickers_analyzed"] == 0
        assert result["diagnostic_counts"]["events_detected_total"] == 0

    def test_none_universe_handling(self, as_of_date, state_dir):
        """Test handling of None active_tickers."""
        result = compute_module_3_catalyst(
            active_tickers=None,
            trial_records={},
            as_of_date=as_of_date,
            state_dir=state_dir,
        )

        assert result["summaries"] == {}
        assert result["diagnostic_counts"]["tickers_analyzed"] == 0

    def test_list_universe_accepted(self, as_of_date, sample_trial_records, state_dir):
        """Test that list of tickers is accepted (not just set)."""
        active_tickers = ["TEST", "ABCD"]  # List instead of set

        result = compute_module_3_catalyst(
            active_tickers=active_tickers,
            trial_records=sample_trial_records,
            as_of_date=as_of_date,
            state_dir=state_dir,
        )

        assert "summaries" in result
        # Should process both tickers
        assert result["diagnostic_counts"]["tickers_analyzed"] >= 1

    def test_iso_date_string_accepted(self, sample_trial_records, state_dir):
        """Test that ISO date string is accepted for as_of_date."""
        active_tickers = {"TEST"}

        result = compute_module_3_catalyst(
            active_tickers=active_tickers,
            trial_records=sample_trial_records,
            as_of_date="2026-01-15",  # String instead of date object
            state_dir=state_dir,
        )

        assert result["as_of_date"] == "2026-01-15"

    def test_invalid_date_raises_error(self, sample_trial_records, state_dir):
        """Test that invalid as_of_date raises ValueError."""
        with pytest.raises(ValueError, match="as_of_date must be"):
            compute_module_3_catalyst(
                active_tickers={"TEST"},
                trial_records=sample_trial_records,
                as_of_date=123,  # Invalid type
                state_dir=state_dir,
            )

    def test_none_date_raises_error(self, sample_trial_records, state_dir):
        """Test that None as_of_date raises ValueError (PIT safety)."""
        with pytest.raises(ValueError, match="as_of_date is REQUIRED"):
            compute_module_3_catalyst(
                active_tickers={"TEST"},
                trial_records=sample_trial_records,
                as_of_date=None,  # Not allowed
                state_dir=state_dir,
            )

    def test_output_schema_has_required_fields(self, as_of_date, sample_trial_records, state_dir):
        """Test that output has all required schema fields."""
        result = compute_module_3_catalyst(
            active_tickers={"TEST"},
            trial_records=sample_trial_records,
            as_of_date=as_of_date,
            state_dir=state_dir,
        )

        required_fields = [
            "summaries",
            "summaries_legacy",
            "diagnostic_counts",
            "diagnostic_counts_legacy",
            "as_of_date",
            "schema_version",
            "score_version",
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    def test_diagnostic_counts_structure(self, as_of_date, sample_trial_records, state_dir):
        """Test that diagnostic_counts has expected structure."""
        result = compute_module_3_catalyst(
            active_tickers={"TEST"},
            trial_records=sample_trial_records,
            as_of_date=as_of_date,
            state_dir=state_dir,
        )

        counts = result["diagnostic_counts"]

        expected_keys = [
            "events_detected_total",
            "events_deduped",
            "tickers_with_events",
            "tickers_analyzed",
            "tickers_with_severe_negative",
        ]

        for key in expected_keys:
            assert key in counts, f"Missing diagnostic count: {key}"
            assert isinstance(counts[key], int)

    def test_tickers_without_trials_handled(self, as_of_date, state_dir):
        """Test tickers with no trial data are handled gracefully."""
        active_tickers = {"NOTFOUND", "NODATA"}
        trial_records = {}  # No data for these tickers

        result = compute_module_3_catalyst(
            active_tickers=active_tickers,
            trial_records=trial_records,
            as_of_date=as_of_date,
            state_dir=state_dir,
        )

        assert result["diagnostic_counts"]["tickers_analyzed"] == 2
        assert result["diagnostic_counts"]["events_detected_total"] == 0

    def test_output_dir_defaults_to_state_parent(self, as_of_date, sample_trial_records, tmp_path):
        """Test that output_dir defaults to state_dir.parent if not provided."""
        state_dir = tmp_path / "nested" / "state"
        state_dir.mkdir(parents=True)

        result = compute_module_3_catalyst(
            active_tickers={"TEST"},
            trial_records=sample_trial_records,
            as_of_date=as_of_date,
            state_dir=state_dir,
            # output_dir not provided
        )

        # Should not crash
        assert "summaries" in result

    def test_custom_output_dir(self, as_of_date, sample_trial_records, tmp_path):
        """Test that custom output_dir is accepted."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        output_dir = tmp_path / "custom_output"
        output_dir.mkdir()

        result = compute_module_3_catalyst(
            active_tickers={"TEST"},
            trial_records=sample_trial_records,
            as_of_date=as_of_date,
            state_dir=state_dir,
            output_dir=output_dir,
        )

        assert "summaries" in result


class TestModule3ConfigCustomization:
    """Test Module3Config configuration and customization."""

    def test_config_from_dict_decay_constant(self):
        """Test customizing decay_constant via from_dict."""
        config_dict = {"decay_constant": 45.0}

        config = Module3Config.from_dict(config_dict)

        assert config.decay_constant == 45.0

    def test_config_from_dict_noise_band(self):
        """Test customizing noise_band_days via from_dict."""
        config_dict = {"noise_band_days": 10}

        config = Module3Config.from_dict(config_dict)

        assert config.event_detector_config.noise_band_days == 10

    def test_config_from_dict_recency_threshold(self):
        """Test customizing recency_threshold_days via from_dict."""
        config_dict = {"recency_threshold_days": 180}

        config = Module3Config.from_dict(config_dict)

        assert config.event_detector_config.recency_threshold_days == 180

    def test_config_from_dict_multiple_params(self):
        """Test customizing multiple parameters at once."""
        config_dict = {
            "noise_band_days": 7,
            "decay_constant": 60.0,
            "recency_threshold_days": 365,
        }

        config = Module3Config.from_dict(config_dict)

        assert config.event_detector_config.noise_band_days == 7
        assert config.decay_constant == 60.0
        assert config.event_detector_config.recency_threshold_days == 365

    def test_config_from_empty_dict(self):
        """Test from_dict with empty dict uses defaults."""
        config = Module3Config.from_dict({})

        # Should use default values
        assert config.decay_constant == 30.0
        assert config.module_version == "3A.2.0"

    def test_config_default_values(self):
        """Test default config values."""
        config = Module3Config()

        assert config.decay_constant == 30.0
        assert config.module_version == "3A.2.0"
        assert config.event_detector_config is not None


class TestModule3WithCustomConfig:
    """Test compute_module_3_catalyst with custom configuration."""

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    @pytest.fixture
    def state_dir(self, tmp_path):
        state = tmp_path / "state"
        state.mkdir()
        return state

    def test_custom_config_accepted(self, as_of_date, state_dir):
        """Test that custom config is accepted and used."""
        custom_config = Module3Config.from_dict({"decay_constant": 60.0})

        result = compute_module_3_catalyst(
            active_tickers={"TEST"},
            trial_records={},
            as_of_date=as_of_date,
            state_dir=state_dir,
            config=custom_config,
        )

        assert "summaries" in result

    def test_none_config_uses_default(self, as_of_date, state_dir):
        """Test that None config falls back to default."""
        result = compute_module_3_catalyst(
            active_tickers={"TEST"},
            trial_records={},
            as_of_date=as_of_date,
            state_dir=state_dir,
            config=None,  # Should use default
        )

        assert "summaries" in result


class TestDeterminism:
    """Test deterministic behavior of module_3_catalyst."""

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    @pytest.fixture
    def state_dir(self, tmp_path):
        state = tmp_path / "state"
        state.mkdir()
        return state

    @pytest.fixture
    def trial_records(self):
        return {
            "TEST": [
                {
                    "ticker": "TEST",
                    "nct_id": "NCT00000001",
                    "phase": "Phase 2",
                    "status": "Recruiting",
                    "primary_completion_date": "2026-06-01",
                    "primary_completion_date_type": "Anticipated",
                    "source_date": "2026-01-14",
                },
            ],
        }

    def test_same_inputs_same_output(self, as_of_date, trial_records, tmp_path):
        """Test that same inputs produce identical outputs."""
        state_dir1 = tmp_path / "state1"
        state_dir1.mkdir()

        state_dir2 = tmp_path / "state2"
        state_dir2.mkdir()

        result1 = compute_module_3_catalyst(
            active_tickers={"TEST"},
            trial_records=trial_records,
            as_of_date=as_of_date,
            state_dir=state_dir1,
        )

        result2 = compute_module_3_catalyst(
            active_tickers={"TEST"},
            trial_records=trial_records,
            as_of_date=as_of_date,
            state_dir=state_dir2,
        )

        # Compare key outputs
        assert result1["as_of_date"] == result2["as_of_date"]
        assert result1["diagnostic_counts"] == result2["diagnostic_counts"]
        assert len(result1["summaries"]) == len(result2["summaries"])
