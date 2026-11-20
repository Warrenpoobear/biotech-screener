#!/usr/bin/env python3
"""
Tests for trials_collector.py

Covers:
- _clean_company_name suffix stripping
- SPONSOR_ALIASES structure
- get_cache_path sanitized filename generation
- is_cache_valid freshness check
- _search_single_sponsor CT.gov API parsing
- search_company_trials multi-step search + dedup
- aggregate_trial_stats phase/status counting
- fetch_trials_data full fetch + lead_stage
- collect_trials_data caching entry point
- collect_batch rate-limited batch collection
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from wake_robin_data_pipeline.collectors.trials_collector import (
    SPONSOR_ALIASES,
    _clean_company_name,
    _search_single_sponsor,
    aggregate_trial_stats,
    collect_batch,
    collect_trials_data,
    fetch_trials_data,
    get_cache_path,
    is_cache_valid,
    search_company_trials,
)

# ============================================================================
# HELPERS
# ============================================================================


def _make_trial(
    nct_id: str,
    phase: str = "PHASE2",
    status: str = "RECRUITING",
    condition: str = "Cancer",
    sponsor: str = "Acme Inc.",
) -> dict:
    """Build a minimal trial dict matching _search_single_sponsor output."""
    return {
        "nct_id": nct_id,
        "title": f"Study {nct_id}",
        "status": status,
        "phase": phase,
        "condition": condition,
        "start_date": "2024-01",
        "completion_date": "2026-06",
        "enrollment": 100,
        "sponsor": sponsor,
    }


def _make_ctgov_response(trials: list[dict]) -> dict:
    """Build a CT.gov API v2 JSON response from simplified trial dicts."""
    studies = []
    for t in trials:
        studies.append(
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": t["nct_id"],
                        "briefTitle": t.get("title", ""),
                    },
                    "statusModule": {
                        "overallStatus": t.get("status", "RECRUITING"),
                        "startDateStruct": {"date": t.get("start_date", "")},
                        "completionDateStruct": {"date": t.get("completion_date", "")},
                        "enrollmentInfo": {"count": t.get("enrollment", 0)},
                    },
                    "designModule": {
                        "phases": t.get("phase", "").split(", ") if t.get("phase") else [],
                    },
                    "conditionsModule": {
                        "conditions": t.get("condition", "").split(", ") if t.get("condition") else [],
                    },
                    "sponsorCollaboratorsModule": {
                        "leadSponsor": {"name": t.get("sponsor", "")},
                    },
                }
            }
        )
    return {"studies": studies}


# ============================================================================
# _clean_company_name
# ============================================================================


class TestCleanCompanyName:
    def test_strips_inc(self):
        assert _clean_company_name("Vertex Pharmaceuticals Inc.") == "Vertex"

    def test_strips_inc_no_dot(self):
        assert _clean_company_name("Vertex Inc") == "Vertex"

    def test_strips_corporation(self):
        assert _clean_company_name("Acme Corporation") == "Acme"

    def test_strips_plc(self):
        assert _clean_company_name("Acme plc") == "Acme"

    def test_strips_plc_upper(self):
        assert _clean_company_name("Acme PLC") == "Acme"

    def test_strips_ltd(self):
        assert _clean_company_name("Immunocore Ltd") == "Immunocore"

    def test_strips_limited(self):
        assert _clean_company_name("Autolus Limited") == "Autolus"

    def test_strips_sa(self):
        assert _clean_company_name("Acme S.A.") == "Acme"

    def test_strips_nv(self):
        assert _clean_company_name("CureVac N.V.") == "CureVac"

    def test_strips_holdings(self):
        assert _clean_company_name("Scholar Rock Holdings") == "Scholar Rock"

    def test_strips_group(self):
        assert _clean_company_name("Replimune Group") == "Replimune"

    def test_strips_american_suffix(self):
        assert _clean_company_name("SomeCo - American") == "SomeCo"

    def test_strips_se(self):
        assert _clean_company_name("BioNTech SE") == "BioNTech"

    def test_strips_bv(self):
        assert _clean_company_name("Pharvaris BV") == "Pharvaris"

    def test_no_suffix_unchanged(self):
        assert _clean_company_name("Moderna") == "Moderna"

    def test_multiple_suffixes_stripped(self):
        # The function iterates ALL suffixes, so multiple can match in sequence.
        # "Acme Therapeutics Inc." -> strip " Inc." -> "Acme Therapeutics"
        # -> strip " Therapeutics" -> "Acme"
        result = _clean_company_name("Acme Therapeutics Inc.")
        assert result == "Acme"

    def test_single_pass_suffix_order_matters(self):
        # Suffix list is iterated once: " Pharmaceuticals" is checked before " plc".
        # "Jazz Pharmaceuticals plc" does NOT end with " Pharmaceuticals" (ends with " plc"),
        # so only " plc" is stripped. The earlier " Pharmaceuticals" suffix was already checked.
        assert _clean_company_name("Jazz Pharmaceuticals plc") == "Jazz Pharmaceuticals"

    def test_empty_string(self):
        assert _clean_company_name("") == ""


# ============================================================================
# SPONSOR_ALIASES
# ============================================================================


class TestSponsorAliases:
    def test_is_dict(self):
        assert isinstance(SPONSOR_ALIASES, dict)

    def test_values_are_lists(self):
        for ticker, aliases in SPONSOR_ALIASES.items():
            assert isinstance(aliases, list), f"{ticker} aliases is not a list"
            assert len(aliases) > 0, f"{ticker} has empty alias list"

    def test_mrna_aliases(self):
        assert "MRNA" in SPONSOR_ALIASES
        aliases = SPONSOR_ALIASES["MRNA"]
        assert "ModernaTX, Inc." in aliases
        assert "Moderna" in aliases

    def test_roiv_has_subsidiaries(self):
        aliases = SPONSOR_ALIASES["ROIV"]
        assert any("Immunovant" in a for a in aliases)
        assert any("Roivant" in a for a in aliases)


# ============================================================================
# get_cache_path
# ============================================================================


class TestGetCachePath:
    def test_returns_path(self):
        result = get_cache_path("Vertex Pharmaceuticals")
        assert isinstance(result, Path)
        assert result.name == "Vertex Pharmaceuticals.json"

    def test_sanitizes_special_chars(self):
        result = get_cache_path("Zai Lab (Shanghai) Co., Ltd.")
        # Parentheses, commas, dots stripped
        assert "(" not in result.name
        assert "," not in result.name
        assert result.suffix == ".json"

    def test_preserves_hyphens_and_underscores(self):
        result = get_cache_path("My-Company_Name")
        assert "My-Company_Name" in result.stem


# ============================================================================
# is_cache_valid
# ============================================================================


class TestIsCacheValid:
    def test_nonexistent_file(self, tmp_path):
        assert is_cache_valid(tmp_path / "no_such_file.json") is False

    def test_fresh_cache(self, tmp_path):
        cache_file = tmp_path / "fresh.json"
        cache_file.write_text("{}")
        assert is_cache_valid(cache_file, max_age_hours=1) is True

    def test_stale_cache(self, tmp_path):
        cache_file = tmp_path / "stale.json"
        cache_file.write_text("{}")
        # Set mtime to 25 hours ago
        old_time = time.time() - 25 * 3600
        os.utime(cache_file, (old_time, old_time))
        assert is_cache_valid(cache_file, max_age_hours=24) is False

    def test_custom_max_age(self, tmp_path):
        cache_file = tmp_path / "custom.json"
        cache_file.write_text("{}")
        # 2 hours old
        old_time = time.time() - 2 * 3600
        os.utime(cache_file, (old_time, old_time))
        assert is_cache_valid(cache_file, max_age_hours=1) is False
        assert is_cache_valid(cache_file, max_age_hours=3) is True


# ============================================================================
# _search_single_sponsor
# ============================================================================


class TestSearchSingleSponsor:
    @patch("wake_robin_data_pipeline.collectors.trials_collector._get_session")
    def test_parses_response(self, mock_get_session):
        trials = [_make_trial("NCT001"), _make_trial("NCT002", phase="PHASE3")]
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_ctgov_response(trials)
        mock_resp.raise_for_status = MagicMock()
        mock_get_session.return_value.get.return_value = mock_resp

        result = _search_single_sponsor("Acme", max_results=50)
        assert len(result) == 2
        assert result[0]["nct_id"] == "NCT001"
        assert result[1]["nct_id"] == "NCT002"
        assert result[1]["phase"] == "PHASE3"

    @patch("wake_robin_data_pipeline.collectors.trials_collector._get_session")
    def test_empty_response(self, mock_get_session):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"studies": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get_session.return_value.get.return_value = mock_resp

        result = _search_single_sponsor("NothingCorp")
        assert result == []

    @patch("wake_robin_data_pipeline.collectors.trials_collector._get_session")
    def test_max_results_capped_at_100(self, mock_get_session):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"studies": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get_session.return_value.get.return_value = mock_resp

        _search_single_sponsor("Test", max_results=200)
        call_args = mock_get_session.return_value.get.call_args
        assert call_args[1]["params"]["pageSize"] == 100

    @patch("wake_robin_data_pipeline.collectors.trials_collector._get_session")
    def test_missing_fields_default_gracefully(self, mock_get_session):
        # Minimal study with missing modules
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"studies": [{"protocolSection": {}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_get_session.return_value.get.return_value = mock_resp

        result = _search_single_sponsor("Sparse")
        assert len(result) == 1
        assert result[0]["nct_id"] == ""
        assert result[0]["phase"] == ""
        assert result[0]["enrollment"] == 0


# ============================================================================
# search_company_trials
# ============================================================================


class TestSearchCompanyTrials:
    @patch("wake_robin_data_pipeline.collectors.trials_collector._search_single_sponsor")
    def test_alias_search_for_known_ticker(self, mock_search):
        """MRNA should search each alias, then cleaned name."""
        mock_search.side_effect = [
            [_make_trial("NCT001")],  # ModernaTX, Inc.
            [_make_trial("NCT002")],  # Moderna TX
            [_make_trial("NCT001"), _make_trial("NCT003")],  # Moderna (NCT001 is dup)
            [_make_trial("NCT004")],  # cleaned company name
        ]

        result = search_company_trials("Moderna Inc.", ticker="MRNA", max_results=50)
        nct_ids = {t["nct_id"] for t in result}
        # NCT001 should appear only once (deduplication)
        assert nct_ids == {"NCT001", "NCT002", "NCT003", "NCT004"}
        assert len(result) == 4

    @patch("wake_robin_data_pipeline.collectors.trials_collector._search_single_sponsor")
    def test_no_ticker_uses_cleaned_name(self, mock_search):
        mock_search.return_value = [_make_trial("NCT010")]

        result = search_company_trials("Vertex Pharmaceuticals", ticker=None)
        assert len(result) == 1
        # Should have called with cleaned name "Vertex"
        mock_search.assert_called_once_with("Vertex", 50)

    @patch("wake_robin_data_pipeline.collectors.trials_collector._search_single_sponsor")
    def test_fallback_to_original_name_when_cleaned_returns_nothing(self, mock_search):
        """When aliases don't exist and cleaned name returns nothing, try original."""
        mock_search.side_effect = [
            [],  # cleaned name returns nothing
            [_make_trial("NCT099")],  # original name succeeds
        ]
        result = search_company_trials("SomeBiotech Holdings Inc.", ticker="FAKE")
        assert len(result) == 1
        assert result[0]["nct_id"] == "NCT099"

    @patch("wake_robin_data_pipeline.collectors.trials_collector._search_single_sponsor")
    def test_dedup_across_aliases(self, mock_search):
        """Same NCT ID from multiple alias searches should be kept once."""
        shared_trial = _make_trial("NCT_SHARED")
        mock_search.return_value = [shared_trial]

        result = search_company_trials("Genmab A/S", ticker="GMAB", max_results=50)
        nct_ids = [t["nct_id"] for t in result]
        assert nct_ids.count("NCT_SHARED") == 1

    @patch("wake_robin_data_pipeline.collectors.trials_collector._search_single_sponsor")
    def test_exception_returns_empty(self, mock_search):
        """Top-level exception should return empty list."""
        mock_search.side_effect = Exception("network error")
        result = search_company_trials("Crash Corp", ticker=None)
        assert result == []

    @patch("wake_robin_data_pipeline.collectors.trials_collector._search_single_sponsor")
    def test_max_results_limit(self, mock_search):
        """Result list should be truncated to max_results."""
        trials = [_make_trial(f"NCT{i:03d}") for i in range(10)]
        mock_search.return_value = trials

        result = search_company_trials("Big Pharma", ticker=None, max_results=3)
        assert len(result) == 3


# ============================================================================
# aggregate_trial_stats
# ============================================================================


class TestAggregateTrialStats:
    def test_empty_trials(self):
        stats = aggregate_trial_stats([])
        assert stats["total_trials"] == 0
        assert stats["by_phase"] == {}
        assert stats["by_status"] == {}
        assert stats["active_trials"] == 0
        assert stats["completed_trials"] == 0

    def test_counts_by_phase(self):
        trials = [
            _make_trial("NCT1", phase="PHASE1"),
            _make_trial("NCT2", phase="PHASE2"),
            _make_trial("NCT3", phase="PHASE2"),
        ]
        stats = aggregate_trial_stats(trials)
        assert stats["by_phase"]["PHASE1"] == 1
        assert stats["by_phase"]["PHASE2"] == 2

    def test_counts_by_status(self):
        trials = [
            _make_trial("NCT1", status="RECRUITING"),
            _make_trial("NCT2", status="RECRUITING"),
            _make_trial("NCT3", status="COMPLETED"),
            _make_trial("NCT4", status="TERMINATED"),
        ]
        stats = aggregate_trial_stats(trials)
        assert stats["by_status"]["RECRUITING"] == 2
        assert stats["by_status"]["COMPLETED"] == 1
        assert stats["by_status"]["TERMINATED"] == 1

    def test_active_trials_count(self):
        trials = [
            _make_trial("NCT1", status="RECRUITING"),
            _make_trial("NCT2", status="ACTIVE_NOT_RECRUITING"),
            _make_trial("NCT3", status="ENROLLING_BY_INVITATION"),
            _make_trial("NCT4", status="COMPLETED"),
            _make_trial("NCT5", status="TERMINATED"),
        ]
        stats = aggregate_trial_stats(trials)
        assert stats["active_trials"] == 3
        assert stats["completed_trials"] == 1

    def test_conditions_collected(self):
        trials = [
            _make_trial("NCT1", condition="Cancer"),
            _make_trial("NCT2", condition="Diabetes"),
            _make_trial("NCT3", condition="Cancer"),  # duplicate
        ]
        stats = aggregate_trial_stats(trials)
        assert stats["conditions"] == ["Cancer", "Diabetes"]

    def test_missing_phase_defaults_to_unknown(self):
        trial = {"nct_id": "NCT1", "status": "RECRUITING"}
        # No 'phase' key at all
        stats = aggregate_trial_stats([trial])
        assert stats["by_phase"].get("UNKNOWN", 0) == 1


# ============================================================================
# fetch_trials_data
# ============================================================================


class TestFetchTrialsData:
    @patch("wake_robin_data_pipeline.collectors.trials_collector.search_company_trials")
    def test_success(self, mock_search):
        mock_search.return_value = [
            _make_trial("NCT1", phase="PHASE3", status="RECRUITING"),
            _make_trial("NCT2", phase="PHASE2", status="COMPLETED"),
        ]
        data = fetch_trials_data("VRTX", "Vertex Pharmaceuticals")
        assert data["success"] is True
        assert data["ticker"] == "VRTX"
        assert data["summary"]["total_trials"] == 2
        assert data["summary"]["lead_stage"] == "phase_3"
        assert data["provenance"]["source"] == "ClinicalTrials.gov API v2"

    @patch("wake_robin_data_pipeline.collectors.trials_collector.search_company_trials")
    def test_lead_stage_phase2(self, mock_search):
        mock_search.return_value = [_make_trial("NCT1", phase="PHASE2")]
        data = fetch_trials_data("TEST", "TestCo")
        assert data["summary"]["lead_stage"] == "phase_2"

    @patch("wake_robin_data_pipeline.collectors.trials_collector.search_company_trials")
    def test_lead_stage_commercial(self, mock_search):
        mock_search.return_value = [_make_trial("NCT1", phase="PHASE4")]
        data = fetch_trials_data("TEST", "TestCo")
        assert data["summary"]["lead_stage"] == "commercial"

    @patch("wake_robin_data_pipeline.collectors.trials_collector.search_company_trials")
    def test_lead_stage_preclinical(self, mock_search):
        mock_search.return_value = [_make_trial("NCT1", phase="EARLY_PHASE1")]
        data = fetch_trials_data("TEST", "TestCo")
        assert data["summary"]["lead_stage"] == "preclinical"

    @patch("wake_robin_data_pipeline.collectors.trials_collector.search_company_trials")
    def test_lead_stage_unknown_no_trials(self, mock_search):
        mock_search.return_value = []
        data = fetch_trials_data("TEST", "TestCo")
        assert data["summary"]["lead_stage"] == "unknown"

    @patch("wake_robin_data_pipeline.collectors.trials_collector.search_company_trials")
    def test_exception_returns_failure(self, mock_search):
        mock_search.side_effect = RuntimeError("boom")
        data = fetch_trials_data("TEST", "TestCo")
        assert data["success"] is False
        assert "boom" in data["error"]

    @patch("wake_robin_data_pipeline.collectors.trials_collector.search_company_trials")
    def test_trials_truncated_to_10(self, mock_search):
        mock_search.return_value = [_make_trial(f"NCT{i:03d}") for i in range(20)]
        data = fetch_trials_data("TEST", "TestCo")
        assert len(data["trials"]) == 10


# ============================================================================
# collect_trials_data (caching)
# ============================================================================


class TestCollectTrialsData:
    @patch("wake_robin_data_pipeline.collectors.trials_collector.fetch_trials_data")
    @patch("wake_robin_data_pipeline.collectors.trials_collector.get_cache_path")
    def test_returns_cached_when_valid(self, mock_cache_path, mock_fetch, tmp_path):
        cache_file = tmp_path / "Cached.json"
        cached_data = {"success": True, "ticker": "TST", "cached_field": True}
        cache_file.write_text(json.dumps(cached_data))
        mock_cache_path.return_value = cache_file

        result = collect_trials_data("TST", "Cached Co")
        assert result["from_cache"] is True
        assert result["cached_field"] is True
        mock_fetch.assert_not_called()

    @patch("wake_robin_data_pipeline.collectors.trials_collector.fetch_trials_data")
    @patch("wake_robin_data_pipeline.collectors.trials_collector.get_cache_path")
    def test_fetches_when_no_cache(self, mock_cache_path, mock_fetch, tmp_path):
        cache_file = tmp_path / "NoCacheYet.json"
        mock_cache_path.return_value = cache_file
        mock_fetch.return_value = {"success": True, "ticker": "NEW"}

        result = collect_trials_data("NEW", "New Co")
        assert result["from_cache"] is False
        mock_fetch.assert_called_once()
        # Successful result should be written to cache
        assert cache_file.exists()

    @patch("wake_robin_data_pipeline.collectors.trials_collector.fetch_trials_data")
    @patch("wake_robin_data_pipeline.collectors.trials_collector.get_cache_path")
    def test_force_refresh_bypasses_cache(self, mock_cache_path, mock_fetch, tmp_path):
        cache_file = tmp_path / "ForceRefresh.json"
        cache_file.write_text(json.dumps({"success": True, "old": True}))
        mock_cache_path.return_value = cache_file
        mock_fetch.return_value = {"success": True, "fresh": True}

        result = collect_trials_data("TST", "Test Co", force_refresh=True)
        assert result["from_cache"] is False
        assert result["fresh"] is True
        mock_fetch.assert_called_once()

    @patch("wake_robin_data_pipeline.collectors.trials_collector.fetch_trials_data")
    @patch("wake_robin_data_pipeline.collectors.trials_collector.get_cache_path")
    def test_failed_fetch_not_cached(self, mock_cache_path, mock_fetch, tmp_path):
        cache_file = tmp_path / "FailedFetch.json"
        mock_cache_path.return_value = cache_file
        mock_fetch.return_value = {"success": False, "error": "timeout"}

        result = collect_trials_data("FAIL", "Fail Co")
        assert result["success"] is False
        assert not cache_file.exists()


# ============================================================================
# collect_batch
# ============================================================================


class TestCollectBatch:
    @patch("wake_robin_data_pipeline.collectors.trials_collector.collect_trials_data")
    @patch("wake_robin_data_pipeline.collectors.trials_collector.time.sleep")
    def test_collects_all(self, mock_sleep, mock_collect):
        mock_collect.return_value = {
            "success": True,
            "summary": {"total_trials": 5, "active_trials": 2, "lead_stage": "phase_2"},
            "from_cache": False,
        }
        ticker_map = {"AAA": "Alpha Inc.", "BBB": "Beta Corp."}
        results = collect_batch(ticker_map, delay_seconds=0.5)

        assert len(results) == 2
        assert "AAA" in results
        assert "BBB" in results
        assert mock_collect.call_count == 2

    @patch("wake_robin_data_pipeline.collectors.trials_collector.collect_trials_data")
    @patch("wake_robin_data_pipeline.collectors.trials_collector.time.sleep")
    def test_rate_limiting_called(self, mock_sleep, mock_collect):
        mock_collect.return_value = {
            "success": True,
            "summary": {"total_trials": 1, "active_trials": 0, "lead_stage": "unknown"},
            "from_cache": False,
        }
        ticker_map = {"A": "Co A", "B": "Co B", "C": "Co C"}
        collect_batch(ticker_map, delay_seconds=1.0)
        # Sleep called between non-cached requests (not after the last one)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(1.0)

    @patch("wake_robin_data_pipeline.collectors.trials_collector.collect_trials_data")
    @patch("wake_robin_data_pipeline.collectors.trials_collector.time.sleep")
    def test_no_sleep_for_cached(self, mock_sleep, mock_collect):
        mock_collect.return_value = {
            "success": True,
            "summary": {"total_trials": 1, "active_trials": 0, "lead_stage": "unknown"},
            "from_cache": True,
        }
        ticker_map = {"A": "Co A", "B": "Co B"}
        collect_batch(ticker_map, delay_seconds=1.0)
        mock_sleep.assert_not_called()

    @patch("wake_robin_data_pipeline.collectors.trials_collector.collect_trials_data")
    @patch("wake_robin_data_pipeline.collectors.trials_collector.time.sleep")
    def test_empty_batch(self, mock_sleep, mock_collect):
        results = collect_batch({})
        assert results == {}
        mock_collect.assert_not_called()


# ---------------------------------------------------------------------------
# SUPN after the INDV -> SUPN ticker change (2026-11-03)
# ---------------------------------------------------------------------------


def test_supn_aliases_cover_both_legacy_sponsors():
    """Indivior acquired Supernus and renamed itself Supernus, Inc. (INDV -> SUPN).

    The EU registry collectors search SPONSOR_ALIASES plus the universe company
    name, so SUPN must carry both sponsor families whatever the company name says.
    """
    aliases = SPONSOR_ALIASES["SUPN"]
    assert any("Indivior" in a for a in aliases)
    assert any("Supernus" in a for a in aliases)
    assert "INDV" not in SPONSOR_ALIASES  # retired ticker; universe has no INDV row
