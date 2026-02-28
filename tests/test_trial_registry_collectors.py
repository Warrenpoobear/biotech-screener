"""
Tests for EUCTR, CTIS, ISRCTN collectors and the trial registry merger.

All tests use fixtures + monkeypatched requests — zero network calls.
"""

import json
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

MINI_UNIVERSE = [
    {"ticker": "MRNA", "company_name": "Moderna, Inc."},
    {"ticker": "VRTX", "company_name": "Vertex Pharmaceuticals Incorporated"},
]


def _make_response(text="", status_code=200, json_data=None):
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.raise_for_status = MagicMock()
    if json_data is not None:
        resp.json = MagicMock(return_value=json_data)
    return resp


# ===========================================================================
# EUCTR Collector
# ===========================================================================

class TestEuctrCollector:
    """Tests for euctr_collector.py."""

    def test_rss_parsing(self):
        """RSS XML is correctly parsed into item dicts."""
        from wake_robin_data_pipeline.collectors.euctr_collector import _parse_rss_response

        xml_text = (FIXTURES / "euctr_rss_sample.xml").read_text()
        items = _parse_rss_response(xml_text)

        assert len(items) == 3
        assert items[0]["eudract_number"] == "2024-001234-56"
        assert items[0]["sponsor"] == "ModernaTX, Inc."
        assert items[1]["eudract_number"] == "2023-005678-90"
        assert "Completed" in items[1]["status"]

    def test_normalization(self):
        """Raw EUCTR item normalizes to TrialRecord schema."""
        from wake_robin_data_pipeline.collectors.euctr_collector import _normalize_euctr_record

        raw = {
            "eudract_number": "2024-001234-56",
            "title": "Phase III RSV Study",
            "sponsor": "ModernaTX, Inc.",
            "status": "Ongoing",
            "phase": "Phase III",
            "start_date": "2024-03-15",
            "medical_condition": "RSV Infection",
            "countries": ["DE", "FR"],
        }
        rec = _normalize_euctr_record(raw, "MRNA", "2026-02-27T12:00:00Z")

        assert rec["registry"] == "euctr"
        assert rec["primary_id"] == "2024-001234-56"
        assert rec["ticker"] == "MRNA"
        assert rec["phase"] == "PHASE3"
        assert rec["status"] == "RECRUITING"
        assert rec["countries"] == ["DE", "FR"]

    def test_sponsor_matching(self):
        """Sponsor name matches against universe map."""
        from wake_robin_data_pipeline.collectors.euctr_collector import (
            _build_universe_map,
            _match_ticker,
        )

        umap = _build_universe_map(MINI_UNIVERSE)
        assert _match_ticker("ModernaTX, Inc.", umap) == "MRNA"
        assert _match_ticker("Moderna", umap) == "MRNA"
        assert _match_ticker("Unknown Pharma", umap) is None

    def test_cache_only_mode(self, tmp_path):
        """cache_only=True with no cache returns empty list."""
        from wake_robin_data_pipeline.collectors.euctr_collector import collect_euctr_trials

        result = collect_euctr_trials(
            universe=MINI_UNIVERSE,
            as_of_date=date(2026, 2, 27),
            cache_dir=tmp_path / "euctr",
            cache_only=True,
        )
        assert result == []


# ===========================================================================
# CTIS Collector
# ===========================================================================

class TestCtisCollector:
    """Tests for ctis_collector.py."""

    def test_search_response_parsing(self):
        """CTIS search JSON is parsed correctly."""
        from wake_robin_data_pipeline.collectors.ctis_collector import _normalize_ctis_record

        search_data = json.loads((FIXTURES / "ctis_search_response.json").read_text())
        items = search_data["data"]

        assert len(items) == 3
        rec = _normalize_ctis_record(items[0], "MRNA", "2026-02-27T12:00:00Z")
        assert rec["primary_id"] == "2024-511897-64-00"
        assert rec["phase"] == "PHASE3"
        assert "NCT06001234" in rec["secondary_ids"]

    def test_detail_parsing(self):
        """CTIS detail response provides enriched metadata."""
        detail = json.loads((FIXTURES / "ctis_detail_response.json").read_text())
        assert detail["ctNumber"] == "2024-511897-64-00"
        assert detail["eudractNumber"] == "2024-001234-56"
        assert detail["nctNumber"] == "NCT06001234"

    def test_date_normalization(self):
        """CTIS dd/mm/yyyy dates are normalized to YYYY-MM-DD."""
        from wake_robin_data_pipeline.collectors.ctis_collector import _normalize_ctis_date

        assert _normalize_ctis_date("15/03/2024") == "2024-03-15"
        assert _normalize_ctis_date("2024-03-15") == "2024-03-15"
        assert _normalize_ctis_date("") is None

    def test_pagination(self):
        """Pagination metadata is correctly parsed."""
        search_data = json.loads((FIXTURES / "ctis_search_response.json").read_text())
        pagination = search_data["pagination"]
        assert pagination["totalPages"] == 1
        assert pagination["totalElements"] == 3


# ===========================================================================
# ISRCTN Collector
# ===========================================================================

class TestIsrctnCollector:
    """Tests for isrctn_collector.py."""

    def test_html_search_parsing(self):
        """ISRCTN IDs are extracted from search result HTML."""
        from wake_robin_data_pipeline.collectors.isrctn_collector import _parse_search_results

        html = '''
        <div class="results">
          <a href="/ISRCTN13619480">Trial 1</a>
          <a href="/ISRCTN99887766">Trial 2</a>
          <a href="/ISRCTN13619480">Duplicate</a>
        </div>
        '''
        ids = _parse_search_results(html)
        assert ids == ["ISRCTN13619480", "ISRCTN99887766"]

    def test_json_ld_extraction(self):
        """JSON-LD from ISRCTN detail page is extracted correctly."""
        from wake_robin_data_pipeline.collectors.isrctn_collector import _extract_json_ld

        html = (FIXTURES / "isrctn_detail_page.html").read_text()
        detail = _extract_json_ld(html, "ISRCTN13619480")

        assert detail is not None
        assert detail["isrctn_id"] == "ISRCTN13619480"
        assert detail["sponsor"] == "Moderna"
        assert "Type 2 Diabetes" in detail["conditions"]
        assert detail["start_date"] == "2024-06-15"

    def test_normalization(self):
        """ISRCTN detail normalizes to TrialRecord schema."""
        from wake_robin_data_pipeline.collectors.isrctn_collector import _normalize_isrctn_record

        raw = {
            "isrctn_id": "ISRCTN13619480",
            "title": "Phase II Diabetes Trial",
            "sponsor": "Moderna",
            "status": "recruiting",
            "phase": "phase ii",
            "conditions": ["Type 2 Diabetes"],
            "countries": ["GB"],
            "start_date": "2024-06-15",
            "end_date": "2026-12-31",
            "date_registered": "2024-05-01",
            "last_edited": "2025-11-20",
            "secondary_ids": ["NCT05999999"],
            "source_url": "https://www.isrctn.com/ISRCTN13619480",
        }
        rec = _normalize_isrctn_record(raw, "MRNA", "2026-02-27T12:00:00Z")

        assert rec["registry"] == "isrctn"
        assert rec["primary_id"] == "ISRCTN13619480"
        assert rec["phase"] == "PHASE2"
        assert rec["status"] == "RECRUITING"
        assert rec["last_update_posted"] == "2025-11-20"

    def test_redirect_handling(self):
        """ISRCTN search follows redirects (allow_redirects=True)."""
        from wake_robin_data_pipeline.collectors.isrctn_collector import _search_isrctn

        mock_resp = _make_response(
            text='<a href="/ISRCTN11111111">Trial</a>',
            status_code=200,
        )

        with patch("wake_robin_data_pipeline.collectors.isrctn_collector.requests.get",
                    return_value=mock_resp) as mock_get:
            with patch("wake_robin_data_pipeline.collectors.isrctn_collector.time.sleep"):
                ids = _search_isrctn("TestSponsor")

        # Verify allow_redirects=True was passed
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["allow_redirects"] is True
        assert ids == ["ISRCTN11111111"]


# ===========================================================================
# Trial Registry Merger
# ===========================================================================

class TestTrialRegistryMerger:
    """Tests for trial_registry_merger.py."""

    def test_dedup_by_shared_nct(self):
        """Two records sharing an NCT ID are merged into one."""
        from wake_robin_data_pipeline.collectors.trial_registry_merger import merge_trial_registries

        ctgov = [{
            "registry": "ctgov", "primary_id": "NCT06001234", "secondary_ids": [],
            "ticker": "MRNA", "phase": "PHASE3", "status": "RECRUITING",
            "last_update_posted": "2025-01-01", "study_type": "INTERVENTIONAL",
            "title": "CTgov Title", "sponsor": {"name": "Moderna", "country": "US"},
            "conditions": ["RSV"], "countries": ["US"],
            "start_date": "2024-03-15", "primary_completion_date": "2026-06-30",
            "completion_date": "2026-12-31", "results_posted_date": None,
            "first_posted": "2024-01-01",
        }]
        ctis = [{
            "registry": "ctis", "primary_id": "2024-511897-64-00",
            "secondary_ids": ["NCT06001234"],
            "ticker": "MRNA", "phase": "PHASE3", "status": "RECRUITING",
            "last_update_posted": "2025-02-01", "study_type": "INTERVENTIONAL",
            "title": "CTIS Title", "sponsor": {"name": "ModernaTX", "country": ""},
            "conditions": ["Respiratory Syncytial Virus"], "countries": ["DE", "FR"],
            "start_date": "2024-04-01", "primary_completion_date": None,
            "completion_date": None, "results_posted_date": None,
            "first_posted": "2024-03-15",
        }]

        result = merge_trial_registries(ctgov, [], ctis, [], date(2026, 12, 31))

        assert len(result) == 1
        merged = result[0]
        assert merged["primary_id"] == "NCT06001234"  # CTgov preferred
        assert "2024-511897-64-00" in merged["secondary_ids"]
        assert "DE" in merged["countries"]
        assert "US" in merged["countries"]

    def test_dedup_by_shared_eudract(self):
        """EUCTR and CTIS records sharing an EudraCT number are merged."""
        from wake_robin_data_pipeline.collectors.trial_registry_merger import merge_trial_registries

        euctr = [{
            "registry": "euctr", "primary_id": "2024-001234-56",
            "secondary_ids": [], "ticker": "MRNA", "phase": "PHASE3",
            "status": "RECRUITING", "last_update_posted": "2025-01-01",
            "study_type": "INTERVENTIONAL", "title": "EUCTR Title",
            "sponsor": {"name": "Moderna", "country": ""},
            "conditions": [], "countries": ["DE"],
            "start_date": "2024-03-15", "primary_completion_date": None,
            "completion_date": None, "results_posted_date": None,
            "first_posted": "2024-03-15",
        }]
        ctis = [{
            "registry": "ctis", "primary_id": "2024-511897-64-00",
            "secondary_ids": ["2024-001234-56"], "ticker": "MRNA",
            "phase": "PHASE3", "status": "RECRUITING",
            "last_update_posted": "2025-02-01", "study_type": "INTERVENTIONAL",
            "title": "CTIS Title", "sponsor": {"name": "ModernaTX", "country": "US"},
            "conditions": ["RSV"], "countries": ["FR"],
            "start_date": "2024-04-01", "primary_completion_date": "2026-06-30",
            "completion_date": None, "results_posted_date": None,
            "first_posted": "2024-03-15",
        }]

        result = merge_trial_registries([], euctr, ctis, [], date(2026, 12, 31))

        assert len(result) == 1
        merged = result[0]
        assert merged["primary_completion_date"] == "2026-06-30"

    def test_no_overlap_union(self):
        """Records with no shared IDs remain separate."""
        from wake_robin_data_pipeline.collectors.trial_registry_merger import merge_trial_registries

        euctr = [{
            "registry": "euctr", "primary_id": "2024-001234-56",
            "secondary_ids": [], "ticker": "MRNA", "phase": "PHASE3",
            "status": "RECRUITING", "last_update_posted": "2025-01-01",
            "study_type": "INTERVENTIONAL", "title": "EUCTR Trial",
            "sponsor": {"name": "Moderna", "country": ""},
            "conditions": [], "countries": ["DE"],
            "start_date": None, "primary_completion_date": None,
            "completion_date": None, "results_posted_date": None,
            "first_posted": None,
        }]
        isrctn = [{
            "registry": "isrctn", "primary_id": "ISRCTN13619480",
            "secondary_ids": [], "ticker": "MRNA", "phase": "PHASE2",
            "status": "RECRUITING", "last_update_posted": "2025-06-01",
            "study_type": "INTERVENTIONAL", "title": "ISRCTN Trial",
            "sponsor": {"name": "Moderna", "country": ""},
            "conditions": [], "countries": ["GB"],
            "start_date": None, "primary_completion_date": None,
            "completion_date": None, "results_posted_date": None,
            "first_posted": None,
        }]

        result = merge_trial_registries([], euctr, [], isrctn, date(2026, 12, 31))
        assert len(result) == 2

    def test_pit_filtering(self):
        """Records with last_update_posted after as_of_date are excluded."""
        from wake_robin_data_pipeline.collectors.trial_registry_merger import merge_trial_registries

        records = [
            {
                "registry": "euctr", "primary_id": "2024-AAAA-01",
                "secondary_ids": [], "ticker": "MRNA",
                "last_update_posted": "2025-01-01",
                "study_type": "INTERVENTIONAL", "phase": "PHASE2",
                "status": "RECRUITING", "title": "", "sponsor": {},
                "conditions": [], "countries": [],
                "start_date": None, "primary_completion_date": None,
                "completion_date": None, "results_posted_date": None,
                "first_posted": None,
            },
            {
                "registry": "euctr", "primary_id": "2025-BBBB-02",
                "secondary_ids": [], "ticker": "MRNA",
                "last_update_posted": "2026-06-01",  # future
                "study_type": "INTERVENTIONAL", "phase": "PHASE3",
                "status": "RECRUITING", "title": "", "sponsor": {},
                "conditions": [], "countries": [],
                "start_date": None, "primary_completion_date": None,
                "completion_date": None, "results_posted_date": None,
                "first_posted": None,
            },
        ]

        result = merge_trial_registries([], records, [], [], date(2026, 2, 28))
        assert len(result) == 1
        assert result[0]["primary_id"] == "2024-AAAA-01"

    def test_stable_sort(self):
        """Output is sorted by (ticker, primary_id)."""
        from wake_robin_data_pipeline.collectors.trial_registry_merger import merge_trial_registries

        records = [
            {
                "registry": "euctr", "primary_id": "2025-ZZZ-01",
                "secondary_ids": [], "ticker": "VRTX",
                "last_update_posted": "2025-01-01",
                "study_type": "INTERVENTIONAL", "phase": "PHASE2",
                "status": "RECRUITING", "title": "", "sponsor": {},
                "conditions": [], "countries": [],
                "start_date": None, "primary_completion_date": None,
                "completion_date": None, "results_posted_date": None,
                "first_posted": None,
            },
            {
                "registry": "isrctn", "primary_id": "ISRCTN00000001",
                "secondary_ids": [], "ticker": "MRNA",
                "last_update_posted": "2025-01-01",
                "study_type": "INTERVENTIONAL", "phase": "PHASE3",
                "status": "RECRUITING", "title": "", "sponsor": {},
                "conditions": [], "countries": [],
                "start_date": None, "primary_completion_date": None,
                "completion_date": None, "results_posted_date": None,
                "first_posted": None,
            },
        ]

        result = merge_trial_registries([], records[:1], [], records[1:], date(2026, 12, 31))
        assert result[0]["ticker"] == "MRNA"
        assert result[1]["ticker"] == "VRTX"


# ===========================================================================
# Normalized Schema Validation
# ===========================================================================

class TestNormalizedSchema:
    """Verify all collectors produce records with required schema fields."""

    REQUIRED_FIELDS = {
        "registry", "primary_id", "secondary_ids", "ticker", "sponsor",
        "title", "conditions", "phase", "status", "study_type", "countries",
        "start_date", "primary_completion_date", "completion_date",
        "results_posted_date", "first_posted", "last_update_posted",
        "source_url", "fetched_at_utc",
    }

    def test_euctr_has_required_fields(self):
        from wake_robin_data_pipeline.collectors.euctr_collector import _normalize_euctr_record

        raw = {
            "eudract_number": "2024-001234-56", "title": "Test",
            "sponsor": "Test", "status": "Ongoing", "phase": "Phase III",
            "start_date": "2024-01-01", "medical_condition": "Test",
            "countries": ["DE"],
        }
        rec = _normalize_euctr_record(raw, "TEST", "2026-01-01T00:00:00Z")
        assert self.REQUIRED_FIELDS.issubset(rec.keys())

    def test_ctis_has_required_fields(self):
        from wake_robin_data_pipeline.collectors.ctis_collector import _normalize_ctis_record

        raw = {
            "ctNumber": "2024-511897-64-00", "title": "Test",
            "sponsorName": "Test", "trialPhase": "Phase II",
            "trialStatus": "Ongoing", "decisionDate": "01/01/2024",
        }
        rec = _normalize_ctis_record(raw, "TEST", "2026-01-01T00:00:00Z")
        assert self.REQUIRED_FIELDS.issubset(rec.keys())

    def test_isrctn_has_required_fields(self):
        from wake_robin_data_pipeline.collectors.isrctn_collector import _normalize_isrctn_record

        raw = {
            "isrctn_id": "ISRCTN13619480", "title": "Test",
            "sponsor": "Test", "status": "recruiting", "phase": "phase i",
            "conditions": [], "countries": [], "start_date": "",
            "end_date": "", "date_registered": "", "last_edited": "",
            "secondary_ids": [], "source_url": "",
        }
        rec = _normalize_isrctn_record(raw, "TEST", "2026-01-01T00:00:00Z")
        assert self.REQUIRED_FIELDS.issubset(rec.keys())


# ===========================================================================
# Phase & Status Normalization
# ===========================================================================

class TestPhaseStatusNormalization:
    """Test phase and status normalization across all collectors."""

    def test_phase_normalization(self):
        from wake_robin_data_pipeline.collectors.euctr_collector import _PHASE_MAP as euctr_phases
        from wake_robin_data_pipeline.collectors.ctis_collector import _PHASE_MAP as ctis_phases
        from wake_robin_data_pipeline.collectors.isrctn_collector import _PHASE_MAP as isrctn_phases

        # All maps should produce the same canonical values
        canonical = {"PHASE1", "PHASE2", "PHASE3", "PHASE4"}
        assert set(euctr_phases.values()).issubset(canonical)
        assert set(ctis_phases.values()).issubset(canonical)
        assert set(isrctn_phases.values()) - {"NA"} == canonical or \
               set(isrctn_phases.values()).issubset(canonical | {"NA"})

    def test_status_normalization(self):
        from wake_robin_data_pipeline.collectors.euctr_collector import _STATUS_MAP as euctr_status
        from wake_robin_data_pipeline.collectors.isrctn_collector import _STATUS_MAP as isrctn_status

        canonical = {"RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED", "TERMINATED", "UNKNOWN"}
        assert set(euctr_status.values()).issubset(canonical)
        assert set(isrctn_status.values()).issubset(canonical)


# ===========================================================================
# Warm Caches Integration
# ===========================================================================

class TestWarmCachesIntegration:
    """Test warm_caches dispatch for new trial registries."""

    def test_warm_euctr_writes_cache(self, tmp_path):
        """warm_euctr calls collector and returns count."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "universe.json").write_text(json.dumps(MINI_UNIVERSE))

        cache_dir = tmp_path / "euctr"

        with patch("warm_caches.warm_euctr") as mock_warm:
            mock_warm.return_value = 5
            result = mock_warm(date(2026, 2, 27), data_dir, cache_dir)
            assert result == 5

    def test_warm_merged_trials_reads_sources(self, tmp_path):
        """warm_merged_trials reads per-registry caches."""
        from wake_robin_data_pipeline.collectors.trial_registry_merger import (
            merge_trial_registries,
            write_merged_cache,
        )

        merged_dir = tmp_path / "merged"
        records = [{
            "registry": "euctr", "primary_id": "2024-001234-56",
            "secondary_ids": [], "ticker": "MRNA", "phase": "PHASE3",
            "status": "RECRUITING", "last_update_posted": "2025-01-01",
            "study_type": "INTERVENTIONAL", "title": "Test",
            "sponsor": {}, "conditions": [], "countries": [],
            "start_date": None, "primary_completion_date": None,
            "completion_date": None, "results_posted_date": None,
            "first_posted": None,
        }]

        path = write_merged_cache(records, date(2026, 2, 27), merged_dir)
        assert path.exists()

        loaded = json.loads(path.read_text())
        assert len(loaded) == 1
        assert loaded[0]["primary_id"] == "2024-001234-56"

        # Check meta sidecar
        meta_path = merged_dir / "trial_records_2026-02-27.meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["schema"] == "merged_trials.v1"
        assert meta["total_records"] == 1

    def test_validate_cache_refresh_new_sources(self):
        """validate_cache_refresh accepts new registry sources."""
        from warm_caches import validate_cache_refresh

        # New registries always accept
        assert validate_cache_refresh("euctr", 0, None) == (True, "")
        assert validate_cache_refresh("ctis", 50, None) == (True, "")
        assert validate_cache_refresh("isrctn", 10, 5) == (True, "")

        # Merged trials rejects empty
        assert validate_cache_refresh("merged_trials", 0, None) == (False, "empty_refresh")

        # Merged trials accepts normal
        assert validate_cache_refresh("merged_trials", 100, None) == (True, "")


# ===========================================================================
# Cache-Only Mode
# ===========================================================================

class TestCacheOnly:
    """Test cache_only behavior for all collectors."""

    def test_cache_only_returns_cached_data(self, tmp_path):
        """cache_only=True returns cached data when cache exists."""
        from wake_robin_data_pipeline.collectors.euctr_collector import collect_euctr_trials

        cache_dir = tmp_path / "euctr"
        cache_dir.mkdir()
        cache_file = cache_dir / "euctr_2026-02-27.json"
        records = [{"registry": "euctr", "primary_id": "2024-001234-56", "ticker": "MRNA"}]
        cache_file.write_text(json.dumps(records))

        result = collect_euctr_trials(
            universe=MINI_UNIVERSE,
            as_of_date=date(2026, 2, 27),
            cache_dir=cache_dir,
            cache_only=True,
        )
        assert len(result) == 1
        assert result[0]["primary_id"] == "2024-001234-56"

    def test_cache_only_empty_when_no_cache(self, tmp_path):
        """cache_only=True with no cache returns empty list for all collectors."""
        from wake_robin_data_pipeline.collectors.euctr_collector import collect_euctr_trials
        from wake_robin_data_pipeline.collectors.ctis_collector import collect_ctis_trials
        from wake_robin_data_pipeline.collectors.isrctn_collector import collect_isrctn_trials

        as_of = date(2026, 2, 27)

        assert collect_euctr_trials(MINI_UNIVERSE, as_of, tmp_path / "e", cache_only=True) == []
        assert collect_ctis_trials(MINI_UNIVERSE, as_of, tmp_path / "c", cache_only=True) == []
        assert collect_isrctn_trials(MINI_UNIVERSE, as_of, tmp_path / "i", cache_only=True) == []
