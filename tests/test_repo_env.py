"""Tests for common.repo_env."""

from __future__ import annotations

from pathlib import Path

from common.repo_env import load_repo_dotenv


def test_load_repo_dotenv_reads_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('FIRECRAWL_API_KEY="fc-from-dotenv"\n', encoding="utf-8")  # pragma: allowlist secret
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)

    assert load_repo_dotenv(tmp_path) is True
    import os

    assert os.environ.get("FIRECRAWL_API_KEY") == "fc-from-dotenv"


def test_load_repo_dotenv_missing_file(tmp_path, monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    assert load_repo_dotenv(tmp_path) is False
