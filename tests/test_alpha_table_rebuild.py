"""Tests for adaptive alpha cohort table building and rebuild policy."""
import hashlib
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_alpha_cohort_table_oos import (
    build_oos_table,
    parse_train_mode,
    write_table_with_marker,
    _build_equal_weight_table,
    _build_weighted_table,
    ALL_KEYS,
)


# =============================================================================
# parse_train_mode
# =============================================================================

class TestParseTrainMode:

    def test_expanding(self):
        assert parse_train_mode("expanding") == ("expanding", None)

    def test_trailing(self):
        assert parse_train_mode("trailing-6") == ("trailing", 6)

    def test_trailing_large(self):
        assert parse_train_mode("trailing-100") == ("trailing", 100)

    def test_decay(self):
        assert parse_train_mode("decay-3") == ("decay", 3)

    def test_case_insensitive(self):
        assert parse_train_mode("EXPANDING") == ("expanding", None)
        assert parse_train_mode("Trailing-6") == ("trailing", 6)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown train mode"):
            parse_train_mode("bogus")

    def test_trailing_zero_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_train_mode("trailing-0")

    def test_decay_zero_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_train_mode("decay-0")

    def test_trailing_no_param_raises(self):
        with pytest.raises(ValueError, match="Invalid train mode"):
            parse_train_mode("trailing-")

    def test_trailing_nonnumeric_raises(self):
        with pytest.raises(ValueError, match="Invalid train mode"):
            parse_train_mode("trailing-abc")


# =============================================================================
# _build_equal_weight_table
# =============================================================================

class TestBuildEqualWeightTable:

    def _make_entry(self, date, rows, excess):
        return {"date": date, "rows": rows, "excess": excess}

    def test_single_date(self):
        rows = [
            {"ticker": "AAAA", "alpha_cohort_key": "early|near_0_30|pos"},
            {"ticker": "BBBB", "alpha_cohort_key": "mid|none|nonpos"},
        ]
        excess = {"AAAA": 0.05, "BBBB": -0.03}
        # Monkey-patch compute_alpha_cohort_key to return pre-set key
        for r in rows:
            r["_key"] = r["alpha_cohort_key"]
        entry = self._make_entry("2025-01-31", rows, excess)

        with patch(
            "scripts.build_alpha_cohort_table_oos.compute_alpha_cohort_key",
            side_effect=lambda row: row.get("_key", "early|none|pos"),
        ):
            table = _build_equal_weight_table([entry], shrink_k=50)

        assert "cells" in table
        assert table["cells"]["early|near_0_30|pos"]["mean_excess_ret_6m"] == 0.05
        assert table["cells"]["early|near_0_30|pos"]["n"] == 1
        assert table["cells"]["mid|none|nonpos"]["mean_excess_ret_6m"] == -0.03
        assert table["cells"]["mid|none|nonpos"]["n"] == 1
        # Unpopulated cells should be 0
        assert table["cells"]["late|far_271_540|pos"]["n"] == 0

    def test_all_36_keys_present(self):
        entry = self._make_entry("2025-01-31", [], {})
        table = _build_equal_weight_table([entry], shrink_k=50)
        assert len(table["cells"]) == 36
        for key in ALL_KEYS:
            assert key in table["cells"]


# =============================================================================
# _build_weighted_table
# =============================================================================

class TestBuildWeightedTable:

    def test_decay_weights_favor_recent(self):
        """With exponential decay, recent observations have more influence."""
        rows_old = [{"ticker": "A", "_key": "early|none|pos"}]
        rows_new = [{"ticker": "A", "_key": "early|none|pos"}]
        excess_old = {"A": -0.10}
        excess_new = {"A": 0.10}

        cache = [
            {"date": "2024-01-31", "rows": rows_old, "excess": excess_old},
            {"date": "2025-01-31", "rows": rows_new, "excess": excess_new},
        ]
        # Weights: older=0.5, newer=1.0 (half-life = 1)
        weights = [0.5, 1.0]

        with patch(
            "scripts.build_alpha_cohort_table_oos.compute_alpha_cohort_key",
            side_effect=lambda row: row.get("_key", "early|none|pos"),
        ):
            table = _build_weighted_table(cache, weights, shrink_k=50)

        cell = table["cells"]["early|none|pos"]
        # Weighted mean: (0.5 * -0.10 + 1.0 * 0.10) / (0.5 + 1.0) = 0.05/1.5 ≈ 0.0333
        expected = (0.5 * -0.10 + 1.0 * 0.10) / (0.5 + 1.0)
        assert abs(cell["mean_excess_ret_6m"] - expected) < 1e-4

    def test_zero_weight_excluded(self):
        """Entries with weight=0 are excluded."""
        rows = [{"ticker": "A", "_key": "mid|none|pos"}]
        cache = [
            {"date": "2024-01-31", "rows": rows, "excess": {"A": 0.50}},
            {"date": "2025-01-31", "rows": rows, "excess": {"A": 0.10}},
        ]
        weights = [0.0, 1.0]

        with patch(
            "scripts.build_alpha_cohort_table_oos.compute_alpha_cohort_key",
            side_effect=lambda row: row.get("_key", "mid|none|pos"),
        ):
            table = _build_weighted_table(cache, weights, shrink_k=50)

        # Only the second entry should contribute
        assert table["cells"]["mid|none|pos"]["mean_excess_ret_6m"] == 0.1


# =============================================================================
# _resolve_alpha_table_path (run_screen.py integration)
# =============================================================================

class TestCheckArtifactMarker:
    """Tests for _check_artifact_marker (provenance validation)."""

    def test_no_marker_file(self, tmp_path):
        from run_screen import _check_artifact_marker
        table = tmp_path / "v1_2026-02-25.json"
        table.write_text('{"cells": {}}')
        present, valid, data = _check_artifact_marker(table)
        assert present is False
        assert valid is False
        assert data is None

    def test_valid_marker_matches(self, tmp_path):
        from run_screen import _check_artifact_marker
        table = tmp_path / "v1_2026-02-25.json"
        content = b'{"cells": {}}'
        table.write_bytes(content)
        sha = hashlib.sha256(content).hexdigest()
        marker = {"sha256": sha, "artifact_id": 42, "as_of_date": "2026-02-25"}
        table.with_suffix(".artifact.json").write_text(json.dumps(marker))

        present, valid, data = _check_artifact_marker(table)
        assert present is True
        assert valid is True
        assert data["artifact_id"] == 42

    def test_marker_sha_mismatch(self, tmp_path):
        from run_screen import _check_artifact_marker
        table = tmp_path / "v1_2026-02-25.json"
        table.write_bytes(b'{"cells": {}}')
        marker = {"sha256": "0000bad0000", "artifact_id": 99}
        table.with_suffix(".artifact.json").write_text(json.dumps(marker))

        present, valid, data = _check_artifact_marker(table)
        assert present is True
        assert valid is False

    def test_corrupt_marker_json(self, tmp_path):
        from run_screen import _check_artifact_marker
        table = tmp_path / "v1_2026-02-25.json"
        table.write_bytes(b'{"cells": {}}')
        table.with_suffix(".artifact.json").write_text("NOT JSON {{{")

        present, valid, data = _check_artifact_marker(table)
        assert present is True
        assert valid is False
        assert data is None

    def test_marker_missing_sha_field(self, tmp_path):
        from run_screen import _check_artifact_marker
        table = tmp_path / "v1_2026-02-25.json"
        table.write_bytes(b'{"cells": {}}')
        marker = {"artifact_id": 1}  # no sha256 key
        table.with_suffix(".artifact.json").write_text(json.dumps(marker))

        present, valid, data = _check_artifact_marker(table)
        assert present is True
        assert valid is False


class TestResolveAlphaTablePath:

    def test_never_policy_returns_static(self):
        from run_screen import _resolve_alpha_table_path
        from decision_engine import DecisionRuleset
        import logging

        rs = DecisionRuleset(alpha_table_rebuild_policy="never")
        logger = logging.getLogger("test")
        path, source = _resolve_alpha_table_path(rs, "2026-02-24", logger)
        expected = Path(__file__).resolve().parent.parent / rs.alpha_cohort_table_path
        assert path == expected
        assert source == "static"

    def test_if_missing_artifact_source(self, tmp_path, monkeypatch):
        """Dated file + valid marker → source='artifact'."""
        from decision_engine import DecisionRuleset
        import logging, run_screen

        # Set up fake project root
        daily_dir = tmp_path / "production_data" / "alpha_cohort_tables" / "daily"
        daily_dir.mkdir(parents=True)
        dated_file = daily_dir / "v1_2026-02-24.json"
        content = b'{"cells": {}}'
        dated_file.write_bytes(content)
        sha = hashlib.sha256(content).hexdigest()
        dated_file.with_suffix(".artifact.json").write_text(
            json.dumps({"sha256": sha, "artifact_id": 1})
        )

        monkeypatch.setattr(run_screen, "__file__", str(tmp_path / "run_screen.py"))

        rs = DecisionRuleset(alpha_table_rebuild_policy="if_missing")
        logger = logging.getLogger("test")
        path, source = run_screen._resolve_alpha_table_path(rs, "2026-02-24", logger)
        assert source == "artifact"
        assert path == dated_file

    def test_if_missing_preexisting_local(self, tmp_path, monkeypatch):
        """Dated file exists but no marker → source='preexisting_local'."""
        from decision_engine import DecisionRuleset
        import logging, run_screen

        daily_dir = tmp_path / "production_data" / "alpha_cohort_tables" / "daily"
        daily_dir.mkdir(parents=True)
        dated_file = daily_dir / "v1_2026-02-24.json"
        dated_file.write_text('{"cells": {}}')
        # No .artifact.json marker

        monkeypatch.setattr(run_screen, "__file__", str(tmp_path / "run_screen.py"))

        rs = DecisionRuleset(alpha_table_rebuild_policy="if_missing")
        logger = logging.getLogger("test")
        path, source = run_screen._resolve_alpha_table_path(rs, "2026-02-24", logger)
        assert source == "preexisting_local"
        assert path == dated_file

    def test_if_missing_rebuilds_in_run(self, tmp_path, monkeypatch):
        """Dated file missing + build succeeds → source='rebuilt_in_run'."""
        from decision_engine import DecisionRuleset
        import logging, run_screen

        # daily dir exists but no dated file
        daily_dir = tmp_path / "production_data" / "alpha_cohort_tables" / "daily"
        daily_dir.mkdir(parents=True)

        monkeypatch.setattr(run_screen, "__file__", str(tmp_path / "run_screen.py"))

        fake_table = {"cells": {}, "_build_info": {"as_of_date": "2026-02-24"}}
        rs = DecisionRuleset(alpha_table_rebuild_policy="if_missing")
        logger = logging.getLogger("test")

        with patch("scripts.build_alpha_cohort_table_oos.build_oos_table", return_value=fake_table):
            path, source = run_screen._resolve_alpha_table_path(rs, "2026-02-24", logger)

        assert source == "rebuilt_in_run"
        assert path.name == "v1_2026-02-24.json"
        assert path.exists()  # file was written

    def test_if_missing_fallback_static(self, tmp_path, monkeypatch):
        """Dated file missing + build returns None → source='static_fallback'."""
        from decision_engine import DecisionRuleset
        import logging, run_screen

        monkeypatch.setattr(run_screen, "__file__", str(tmp_path / "run_screen.py"))
        # Create static fallback path
        static_path = tmp_path / "production_data" / "alpha_cohort_tables" / "v1_alpha_cohort_table.json"
        static_path.parent.mkdir(parents=True, exist_ok=True)
        static_path.write_text('{"cells": {}}')

        rs = DecisionRuleset(alpha_table_rebuild_policy="if_missing")
        logger = logging.getLogger("test")

        with patch("scripts.build_alpha_cohort_table_oos.build_oos_table", return_value=None):
            path, source = run_screen._resolve_alpha_table_path(rs, "2026-02-24", logger)

        assert source == "static_fallback"


# =============================================================================
# build_oos_table with mocked archives
# =============================================================================

class TestBuildOosTableMocked:
    """Test build_oos_table with mocked archive discovery and providers."""

    def test_returns_none_with_no_archives(self):
        with patch("scripts.build_alpha_cohort_table_oos.discover_archives", return_value=[]):
            result = build_oos_table(
                as_of_date="2026-01-01",
                train_mode="expanding",
                min_train_dates=2,
            )
        assert result is None

    def test_returns_none_below_min_dates(self):
        """If only 1 archive available but min_train_dates=2, returns None."""
        fake_path = Path("/fake/archive.tar.gz")
        with patch("scripts.build_alpha_cohort_table_oos.discover_archives",
                    return_value=[("2025-06-30", fake_path)]), \
             patch("scripts.build_alpha_cohort_table_oos.MorningstarReturnsProvider"), \
             patch("scripts.build_alpha_cohort_table_oos.CSVReturnsProvider"), \
             patch("scripts.build_alpha_cohort_table_oos.ChainedReturnsProvider") as mock_chain:
            mock_provider = MagicMock()
            mock_provider.get_last_date.return_value = None  # no last date check
            mock_chain.return_value = mock_provider
            result = build_oos_table(
                as_of_date="2026-01-01",
                train_mode="expanding",
                min_train_dates=2,
            )
        assert result is None

    def test_pit_filter_excludes_future_archives(self):
        """Archives on or after as_of_date are excluded."""
        archives = [
            ("2025-06-30", Path("/fake/a1.tar.gz")),
            ("2026-01-01", Path("/fake/a2.tar.gz")),  # same as as_of_date
            ("2026-02-01", Path("/fake/a3.tar.gz")),  # after as_of_date
        ]
        with patch("scripts.build_alpha_cohort_table_oos.discover_archives",
                    return_value=archives), \
             patch("scripts.build_alpha_cohort_table_oos.MorningstarReturnsProvider"), \
             patch("scripts.build_alpha_cohort_table_oos.CSVReturnsProvider"), \
             patch("scripts.build_alpha_cohort_table_oos.ChainedReturnsProvider") as mock_chain:
            mock_provider = MagicMock()
            mock_provider.get_last_date.return_value = None
            mock_chain.return_value = mock_provider
            # Only 1 archive before 2026-01-01, need 2 → None
            result = build_oos_table(
                as_of_date="2026-01-01",
                train_mode="expanding",
                min_train_dates=2,
            )
        assert result is None

    def test_trailing_mode_limits_window(self):
        """trailing-2 should only use last 2 dates even if more available."""
        fake_archives = [
            ("2025-01-31", Path("/fake/a1.tar.gz")),
            ("2025-02-28", Path("/fake/a2.tar.gz")),
            ("2025-03-31", Path("/fake/a3.tar.gz")),
            ("2025-04-30", Path("/fake/a4.tar.gz")),
        ]
        loaded_dates = []

        def fake_load_rankings(tar_path):
            return [
                {"ticker": f"T{i}", "archetype": "drug_developer",
                 "tier_dev": "A", "clinical_score": "0.5"}
                for i in range(6)
            ]

        def fake_fwd_returns(provider, tickers, date_str, horizon):
            loaded_dates.append(date_str)
            return {f"T{i}": 0.01 * i for i in range(6)}

        with patch("scripts.build_alpha_cohort_table_oos.discover_archives",
                    return_value=fake_archives), \
             patch("scripts.build_alpha_cohort_table_oos.MorningstarReturnsProvider"), \
             patch("scripts.build_alpha_cohort_table_oos.CSVReturnsProvider"), \
             patch("scripts.build_alpha_cohort_table_oos.ChainedReturnsProvider") as mock_chain, \
             patch("scripts.build_alpha_cohort_table_oos.add_trading_days",
                   return_value="2025-12-31"), \
             patch("scripts.build_alpha_cohort_table_oos.load_rankings_dicts",
                   side_effect=fake_load_rankings), \
             patch("scripts.build_alpha_cohort_table_oos.compute_forward_returns",
                   side_effect=fake_fwd_returns), \
             patch("scripts.build_alpha_cohort_table_oos.backfill_clinical_z_tier"), \
             patch("scripts.build_alpha_cohort_table_oos.compute_alpha_cohort_key",
                   return_value="early|none|pos"):
            mock_provider = MagicMock()
            mock_provider.get_last_date.return_value = None
            mock_chain.return_value = mock_provider

            result = build_oos_table(
                as_of_date="2026-01-01",
                train_mode="trailing-2",
                min_train_dates=2,
            )

        assert result is not None
        # Only last 2 dates should be loaded
        assert loaded_dates == ["2025-03-31", "2025-04-30"]

    def test_table_schema_valid(self):
        """Built table has all required schema fields."""
        fake_archives = [
            ("2025-01-31", Path("/fake/a1.tar.gz")),
            ("2025-02-28", Path("/fake/a2.tar.gz")),
        ]

        def fake_load_rankings(tar_path):
            return [
                {"ticker": f"T{i}", "archetype": "drug_developer",
                 "tier_dev": "A", "clinical_score": "0.5"}
                for i in range(6)
            ]

        with patch("scripts.build_alpha_cohort_table_oos.discover_archives",
                    return_value=fake_archives), \
             patch("scripts.build_alpha_cohort_table_oos.MorningstarReturnsProvider"), \
             patch("scripts.build_alpha_cohort_table_oos.CSVReturnsProvider"), \
             patch("scripts.build_alpha_cohort_table_oos.ChainedReturnsProvider") as mock_chain, \
             patch("scripts.build_alpha_cohort_table_oos.add_trading_days",
                   return_value="2025-12-31"), \
             patch("scripts.build_alpha_cohort_table_oos.load_rankings_dicts",
                   side_effect=fake_load_rankings), \
             patch("scripts.build_alpha_cohort_table_oos.compute_forward_returns",
                   return_value={f"T{i}": 0.01 * i for i in range(6)}), \
             patch("scripts.build_alpha_cohort_table_oos.backfill_clinical_z_tier"), \
             patch("scripts.build_alpha_cohort_table_oos.compute_alpha_cohort_key",
                   return_value="early|none|pos"):
            mock_provider = MagicMock()
            mock_provider.get_last_date.return_value = None
            mock_chain.return_value = mock_provider

            result = build_oos_table(
                as_of_date="2026-01-01",
                train_mode="expanding",
                min_train_dates=2,
            )

        assert result is not None
        assert result["_schema"]["name"] == "alpha_cohort_table"
        assert result["_schema"]["version"] == "1"
        assert "shrink_k_default" in result
        assert "alpha_clip" in result
        assert len(result["cells"]) == 36
        for key in ALL_KEYS:
            assert key in result["cells"]
        assert result["_build_info"]["as_of_date"] == "2026-01-01"
        assert result["_build_info"]["n_train_dates"] == 2


# =============================================================================
# write_table_with_marker
# =============================================================================

class TestWriteTableWithMarker:
    """Tests for artifact marker write alongside table JSON."""

    def _sample_table(self):
        return {
            "_schema": {"name": "alpha_cohort_table", "version": "1"},
            "shrink_k_default": 50,
            "alpha_clip": {"min": -0.10, "max": 0.10},
            "cells": {"early|none|pos": {"mean_excess_ret_6m": 0.05, "n": 10}},
            "_build_info": {
                "as_of_date": "2026-03-01",
                "train_mode": "trailing-6",
                "horizon": 84,
                "n_train_dates": 6,
                "train_dates": ["2025-04-30"],
            },
        }

    def test_writes_table_file(self, tmp_path):
        dest = tmp_path / "daily" / "v1_2026-03-01.json"
        write_table_with_marker(self._sample_table(), dest, as_of_date="2026-03-01")
        assert dest.exists()
        data = json.loads(dest.read_text())
        assert data["_schema"]["name"] == "alpha_cohort_table"

    def test_writes_marker_file(self, tmp_path):
        dest = tmp_path / "v1_2026-03-01.json"
        write_table_with_marker(self._sample_table(), dest, as_of_date="2026-03-01")
        marker_path = dest.with_suffix(".artifact.json")
        assert marker_path.exists()

    def test_marker_sha256_matches_table(self, tmp_path):
        dest = tmp_path / "v1_2026-03-01.json"
        write_table_with_marker(self._sample_table(), dest, as_of_date="2026-03-01")
        marker = json.loads(dest.with_suffix(".artifact.json").read_text())
        actual_sha = hashlib.sha256(dest.read_bytes()).hexdigest()
        assert marker["sha256"] == actual_sha

    def test_marker_fields_present(self, tmp_path):
        dest = tmp_path / "v1_2026-03-01.json"
        write_table_with_marker(self._sample_table(), dest, as_of_date="2026-03-01")
        marker = json.loads(dest.with_suffix(".artifact.json").read_text())
        assert marker["as_of_date"] == "2026-03-01"
        assert "built_at_utc" in marker
        assert "sha256" in marker
        assert marker["target_path"] == str(dest)
        assert marker["source"] == "oos_build"
        assert marker["train_mode"] == "trailing-6"
        assert marker["n_train_dates"] == 6

    def test_custom_source_tag(self, tmp_path):
        dest = tmp_path / "v1_2026-03-01.json"
        write_table_with_marker(
            self._sample_table(), dest,
            as_of_date="2026-03-01", source="rebuilt_in_run",
        )
        marker = json.loads(dest.with_suffix(".artifact.json").read_text())
        assert marker["source"] == "rebuilt_in_run"

    def test_creates_parent_dirs(self, tmp_path):
        dest = tmp_path / "a" / "b" / "c" / "v1.json"
        write_table_with_marker(self._sample_table(), dest, as_of_date="2026-03-01")
        assert dest.exists()

    def test_validated_by_check_artifact_marker(self, tmp_path):
        """Marker written by write_table_with_marker passes _check_artifact_marker."""
        from run_screen import _check_artifact_marker
        dest = tmp_path / "v1_2026-03-01.json"
        write_table_with_marker(self._sample_table(), dest, as_of_date="2026-03-01")
        present, valid, data = _check_artifact_marker(dest)
        assert present is True
        assert valid is True

    def test_rebuilt_in_run_gets_artifact_source(self, tmp_path, monkeypatch):
        """When _resolve_alpha_table_path rebuilds, the marker lets reload detect 'artifact'."""
        from run_screen import _check_artifact_marker
        dest = tmp_path / "v1_2026-03-01.json"
        write_table_with_marker(
            self._sample_table(), dest,
            as_of_date="2026-03-01", source="rebuilt_in_run",
        )
        # On next load, _check_artifact_marker will find valid marker
        present, valid, _ = _check_artifact_marker(dest)
        assert present and valid
