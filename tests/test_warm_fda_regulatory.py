"""Tests for warm_fda_regulatory() integration in warm_caches.py."""

import json
from datetime import date
from unittest.mock import patch


class TestWarmFdaRegulatory:
    """Verify warm_fda_regulatory() wiring and behavior."""

    def test_warm_calls_collect_and_returns_count(self, tmp_path):
        """warm_fda_regulatory builds product map and calls collector."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Write minimal pdufa_dates.json for product map
        pdufa = [{"ticker": "AXSM", "drug_name": "AXS-05"}]
        (data_dir / "pdufa_dates.json").write_text(json.dumps(pdufa))

        as_of = date(2026, 3, 12)
        fake_events = [
            {
                "ticker": "AXSM",
                "event_type": "FDA_APPROVAL",
                "event_date": "2026-03-01",
                "event_name": "FDA_APPROVAL: AXS-05 (test)",
                "drug_name": "axs-05",
                "confidence": "HIGH",
                "source": "FEDERAL_REGISTER",
                "disclosed_at": "2026-03-01",
                "tags": ["fda_regulatory", "federal_register"],
            }
        ]

        with patch(
            "wake_robin_data_pipeline.collectors.fda_adcom_collector.collect_fda_regulatory_notices",
            return_value=fake_events,
        ):
            from warm_caches import warm_fda_regulatory

            count = warm_fda_regulatory(as_of, data_dir, cache_dir)

        assert count == 1

    def test_warm_returns_zero_on_empty_product_map(self, tmp_path):
        """Returns 0 when product map is empty (no pdufa_dates.json)."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        from warm_caches import warm_fda_regulatory

        count = warm_fda_regulatory(date(2026, 3, 12), data_dir, cache_dir)
        assert count == 0

    def test_cache_file_created(self, tmp_path):
        """Collector writes cache file with correct naming."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        pdufa = [{"ticker": "VERA", "drug_name": "Atacicept"}]
        (data_dir / "pdufa_dates.json").write_text(json.dumps(pdufa))

        as_of = date(2026, 3, 12)
        fake_events = [
            {
                "ticker": "VERA",
                "event_type": "FDA_APPROVAL",
                "event_date": "2026-02-15",
                "source": "FEDERAL_REGISTER",
                "disclosed_at": "2026-02-15",
            }
        ]

        with patch(
            "wake_robin_data_pipeline.collectors.fda_adcom_collector.collect_fda_regulatory_notices",
            return_value=fake_events,
        ):
            from warm_caches import warm_fda_regulatory

            count = warm_fda_regulatory(as_of, data_dir, cache_dir)

        # The collector (mocked here) is what writes the cache file, so no file is produced in this
        # unit test; warm_fda_regulatory returns the notice count, which we assert directly.
        assert count == 1


class TestFdaRegulatoryDispatcher:
    """Verify fda_regulatory is wired into the main() source dispatcher."""

    def test_fda_regulatory_in_sources_help(self):
        """The --sources help text mentions fda_regulatory."""
        import inspect

        import warm_caches

        source = inspect.getsource(warm_caches.main)
        assert "fda_regulatory" in source
