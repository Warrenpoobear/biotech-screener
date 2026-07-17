"""Tests for archive_production_inputs — forward-shadow evidence archiving (B2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.archive_production_inputs as arch


def _mk(p: Path, content: str = "{}\n") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def test_evidence_ledgers_archived(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    monkeypatch.setattr(arch, "SCRIPT_DIR", repo)
    for _, rel in arch._EVIDENCE_FILES:
        _mk(repo / rel, '{"x": 1}\n')
    data_dir = tmp_path / "production_data"
    data_dir.mkdir()
    archive_root = tmp_path / "pit_archives"

    rc = arch.archive("2026-07-17", data_dir, archive_root)

    assert rc == 0
    dest = archive_root / "2026-07-17"
    manifest = json.loads((dest / "manifest.json").read_text())
    for arch_name, _ in arch._EVIDENCE_FILES:
        assert (dest / arch_name).exists(), f"{arch_name} not copied"
        assert arch_name in manifest["files"], f"{arch_name} missing from manifest"
        assert manifest["files"][arch_name]["size_bytes"] > 0


def test_missing_evidence_is_nonfatal(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    monkeypatch.setattr(arch, "SCRIPT_DIR", repo)
    # create only the first evidence file; the rest are absent
    name0, rel0 = arch._EVIDENCE_FILES[0]
    _mk(repo / rel0, '{"x": 1}\n')
    data_dir = tmp_path / "production_data"
    data_dir.mkdir()
    archive_root = tmp_path / "pit_archives"

    rc = arch.archive("2026-07-17", data_dir, archive_root)

    assert rc == 0  # present ones archived; missing ones WARN but are non-fatal
    dest = archive_root / "2026-07-17"
    assert (dest / name0).exists()
    manifest = json.loads((dest / "manifest.json").read_text())
    assert name0 in manifest["files"]
    # a missing evidence file is simply not archived
    assert not (dest / arch._EVIDENCE_FILES[1][0]).exists()
    assert arch._EVIDENCE_FILES[1][0] not in manifest["files"]


def test_evidence_archive_names_flat_and_unique():
    names = [n for n, _ in arch._EVIDENCE_FILES]
    for n in names:
        assert "/" not in n and "\\" not in n, f"{n} is not a flat filename"
    assert len(names) == len(set(names)), "duplicate evidence archive names"
    # every source is repo-relative under artifacts/
    for _, rel in arch._EVIDENCE_FILES:
        assert rel.startswith("artifacts/"), rel
