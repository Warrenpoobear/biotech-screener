"""Tests for scripts/fetch_alpha_table_artifact.py (mocked GitHub API)."""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fetch_alpha_table_artifact import (
    EXIT_API_ERROR,
    EXIT_NOT_FOUND,
    EXIT_OK,
    fetch_alpha_table,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_artifact_zip(table_dict: dict | None = None) -> bytes:
    """Build an in-memory zip containing a synthetic alpha table JSON."""
    if table_dict is None:
        table_dict = {
            "schema": "alpha_cohort_table.v1",
            "cells": {"drug_developer|near|positive": {"mean_excess_ret_6m": 0.03, "n": 12}},
            "_build_info": {"as_of_date": "2026-02-25"},
        }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("v1_2026-02-25.json", json.dumps(table_dict))
    return buf.getvalue()


def _mock_list_response(artifacts: list, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"total_count": len(artifacts), "artifacts": artifacts}
    resp.text = json.dumps({"total_count": len(artifacts), "artifacts": artifacts})
    return resp


def _mock_download_response(content: bytes, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    return resp


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

class TestSuccessPath:

    def test_fetches_and_writes_table(self, tmp_path):
        dest = tmp_path / "daily" / "v1_2026-02-25.json"
        artifact = {
            "id": 42,
            "name": "alpha-cohort-table-2026-02-25",
            "archive_download_url": "https://api.github.com/repos/test/test/actions/artifacts/42/zip",
            "expired": False,
            "size_in_bytes": 1234,
        }
        zip_bytes = _make_artifact_zip()

        with patch("fetch_alpha_table_artifact.requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_list_response([artifact]),
                _mock_download_response(zip_bytes),
            ]
            rc = fetch_alpha_table(
                "2026-02-25", dest, token="fake-token",
            )

        assert rc == EXIT_OK
        assert dest.exists()
        data = json.loads(dest.read_text())
        assert data["schema"] == "alpha_cohort_table.v1"

    def test_creates_parent_dirs(self, tmp_path):
        dest = tmp_path / "deep" / "nested" / "v1_2026-02-25.json"
        artifact = {
            "id": 1,
            "name": "alpha-cohort-table-2026-02-25",
            "archive_download_url": "https://example.com/zip",
            "expired": False,
            "size_in_bytes": 100,
        }
        with patch("fetch_alpha_table_artifact.requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_list_response([artifact]),
                _mock_download_response(_make_artifact_zip()),
            ]
            rc = fetch_alpha_table("2026-02-25", dest, token="fake-token")

        assert rc == EXIT_OK
        assert dest.exists()


# ---------------------------------------------------------------------------
# Not found (exit 3)
# ---------------------------------------------------------------------------

class TestNotFound:

    def test_no_artifacts_returns_3(self, tmp_path):
        dest = tmp_path / "v1_2026-02-25.json"
        with patch("fetch_alpha_table_artifact.requests.get") as mock_get:
            mock_get.return_value = _mock_list_response([])
            rc = fetch_alpha_table("2026-02-25", dest, token="fake-token")
        assert rc == EXIT_NOT_FOUND
        assert not dest.exists()

    def test_expired_artifact_returns_3(self, tmp_path):
        dest = tmp_path / "v1_2026-02-25.json"
        artifact = {
            "id": 1,
            "name": "alpha-cohort-table-2026-02-25",
            "archive_download_url": "https://example.com/zip",
            "expired": True,
            "size_in_bytes": 100,
        }
        with patch("fetch_alpha_table_artifact.requests.get") as mock_get:
            mock_get.return_value = _mock_list_response([artifact])
            rc = fetch_alpha_table("2026-02-25", dest, token="fake-token")
        assert rc == EXIT_NOT_FOUND


# ---------------------------------------------------------------------------
# Auth / API errors (exit 2)
# ---------------------------------------------------------------------------

class TestApiError:

    def test_no_token_returns_2(self, tmp_path):
        dest = tmp_path / "v1_2026-02-25.json"
        rc = fetch_alpha_table("2026-02-25", dest, token="")
        assert rc == EXIT_API_ERROR

    def test_none_token_returns_2(self, tmp_path):
        dest = tmp_path / "v1_2026-02-25.json"
        rc = fetch_alpha_table("2026-02-25", dest, token=None)
        assert rc == EXIT_API_ERROR

    def test_401_returns_2(self, tmp_path):
        dest = tmp_path / "v1_2026-02-25.json"
        resp = MagicMock()
        resp.status_code = 401
        resp.text = "Unauthorized"
        with patch("fetch_alpha_table_artifact.requests.get") as mock_get:
            mock_get.return_value = resp
            rc = fetch_alpha_table("2026-02-25", dest, token="bad-token")
        assert rc == EXIT_API_ERROR

    def test_403_returns_2(self, tmp_path):
        dest = tmp_path / "v1_2026-02-25.json"
        resp = MagicMock()
        resp.status_code = 403
        resp.text = "Forbidden"
        with patch("fetch_alpha_table_artifact.requests.get") as mock_get:
            mock_get.return_value = resp
            rc = fetch_alpha_table("2026-02-25", dest, token="bad-token")
        assert rc == EXIT_API_ERROR

    def test_network_error_returns_2(self, tmp_path):
        import requests as req
        dest = tmp_path / "v1_2026-02-25.json"
        with patch("fetch_alpha_table_artifact.requests.get") as mock_get:
            mock_get.side_effect = req.ConnectionError("network down")
            rc = fetch_alpha_table("2026-02-25", dest, token="fake-token")
        assert rc == EXIT_API_ERROR

    def test_download_failure_returns_2(self, tmp_path):
        dest = tmp_path / "v1_2026-02-25.json"
        artifact = {
            "id": 1,
            "name": "alpha-cohort-table-2026-02-25",
            "archive_download_url": "https://example.com/zip",
            "expired": False,
            "size_in_bytes": 100,
        }
        with patch("fetch_alpha_table_artifact.requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_list_response([artifact]),
                _mock_download_response(b"not a zip", status_code=200),
            ]
            # Bad zip content → API_ERROR
            rc = fetch_alpha_table("2026-02-25", dest, token="fake-token")
        assert rc == EXIT_API_ERROR

    def test_download_http_error_returns_2(self, tmp_path):
        dest = tmp_path / "v1_2026-02-25.json"
        artifact = {
            "id": 1,
            "name": "alpha-cohort-table-2026-02-25",
            "archive_download_url": "https://example.com/zip",
            "expired": False,
            "size_in_bytes": 100,
        }
        with patch("fetch_alpha_table_artifact.requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_list_response([artifact]),
                _mock_download_response(b"", status_code=500),
            ]
            rc = fetch_alpha_table("2026-02-25", dest, token="fake-token")
        assert rc == EXIT_API_ERROR


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_zip_with_no_json_returns_2(self, tmp_path):
        """Zip exists but contains no .json file."""
        dest = tmp_path / "v1_2026-02-25.json"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("readme.txt", "not a table")
        artifact = {
            "id": 1,
            "name": "alpha-cohort-table-2026-02-25",
            "archive_download_url": "https://example.com/zip",
            "expired": False,
            "size_in_bytes": 100,
        }
        with patch("fetch_alpha_table_artifact.requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_list_response([artifact]),
                _mock_download_response(buf.getvalue()),
            ]
            rc = fetch_alpha_table("2026-02-25", dest, token="fake-token")
        assert rc == EXIT_API_ERROR

    def test_idempotent_overwrite(self, tmp_path):
        """Running twice overwrites the destination cleanly."""
        dest = tmp_path / "v1_2026-02-25.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text('{"old": true}')

        table = {"schema": "alpha_cohort_table.v1", "cells": {}, "new": True}
        artifact = {
            "id": 1,
            "name": "alpha-cohort-table-2026-02-25",
            "archive_download_url": "https://example.com/zip",
            "expired": False,
            "size_in_bytes": 100,
        }
        with patch("fetch_alpha_table_artifact.requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_list_response([artifact]),
                _mock_download_response(_make_artifact_zip(table)),
            ]
            rc = fetch_alpha_table("2026-02-25", dest, token="fake-token")
        assert rc == EXIT_OK
        data = json.loads(dest.read_text())
        assert data.get("new") is True
