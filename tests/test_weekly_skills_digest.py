"""Tests for tools/weekly_skills_digest.py."""

from __future__ import annotations

from datetime import date

from tools.weekly_skills_digest import generate_digest


def test_generate_digest_dry_run():
    path = generate_digest(date(2026, 6, 24), dry_run=True)
    assert path.name == "weekly_digest_2026-06-24.md"


def test_generate_digest_writes_file(tmp_path, monkeypatch):
    import tools.weekly_skills_digest as mod

    monkeypatch.setattr(mod, "OUT_DIR", tmp_path)
    out = generate_digest(date(2026, 6, 24), dry_run=False)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Weekly Skills Learning Digest" in text
    assert "Promotion checklist" not in text  # digest uses operator checklist
    assert "Operator checklist" in text
