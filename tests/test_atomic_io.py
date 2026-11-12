"""common/atomic_io: production_data writes must not write through hard links.

production_data files are hard-linked into staged PIT input dirs. An in-place
write (open "w", write_text, append) changes every linked copy; these helpers
must replace the inode instead so staged copies keep their captured bytes.
"""

from __future__ import annotations

import json
import os

import pytest

from common.atomic_io import atomic_append_line, atomic_write_json, atomic_write_text


@pytest.fixture()
def linked(tmp_path):
    live = tmp_path / "production_data" / "x.json"
    live.parent.mkdir()
    live.write_text("OLD", encoding="utf-8")
    os.chmod(live, 0o640)
    staged = tmp_path / "staging" / "x.json"
    staged.parent.mkdir()
    os.link(live, staged)
    assert os.stat(live).st_nlink == 2
    return live, staged


def test_write_text_does_not_touch_hard_linked_copy(linked):
    live, staged = linked
    atomic_write_text(live, "NEW")
    assert live.read_text() == "NEW"
    assert staged.read_text() == "OLD"
    assert os.stat(live).st_nlink == 1


def test_write_preserves_mode(linked):
    live, _ = linked
    atomic_write_text(live, "NEW")
    assert os.stat(live).st_mode & 0o777 == 0o640


def test_write_json_roundtrip_and_new_file(tmp_path):
    p = tmp_path / "new" / "y.json"
    atomic_write_json(p, {"a": 1}, trailing_newline=True)
    assert json.loads(p.read_text()) == {"a": 1}
    assert p.read_text().endswith("\n")


def test_append_line_does_not_touch_hard_linked_copy(linked):
    live, staged = linked
    atomic_append_line(live, "line2")
    assert live.read_text() == "OLD\nline2\n"
    assert staged.read_text() == "OLD"


def test_failed_write_leaves_original_and_no_temp(linked, monkeypatch):
    live, staged = linked

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_text(live, "NEW")
    assert live.read_text() == "OLD"
    assert [p.name for p in live.parent.iterdir()] == ["x.json"]
