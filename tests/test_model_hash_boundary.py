"""Tests for the behavioral model-hash boundary (HASH_SCHEME = ast-v1, widened to ast-v2 on 2026-11-15).

Regression guard for the 2026-07-10 finding: the raw-bytes model hash flipped
the frozen forward-validation candidate (a9983a67 -> a7a80e85) on a one-line
type-annotation edit despite byte-for-byte identical runtime behavior. The
behavioral fingerprint must be insensitive to annotations/docstrings/comments/
whitespace and sensitive to real logic changes.

See docs/governance/2026-07-10-dem-candidate-hash-equivalence.md.
"""

from __future__ import annotations

import tools.run_forward_validation as m


def _hash_with_files(monkeypatch, tmp_path, sources: dict[str, str]) -> str:
    paths = []
    for name, src in sources.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
        paths.append(p)
    monkeypatch.setattr(m, "MODEL_FILES", paths)
    return m.compute_model_hash()


BASE = '''\
"""Original docstring."""
import math

THRESHOLD = 0.5


def score(alpha: float, beta: float = 1.0) -> float:
    """Compute a score."""
    # weighting comment
    total: float = alpha * 2 + beta
    return total
'''

# Same runtime behavior — only annotations, docstrings, comments, whitespace differ.
COSMETIC = """\
import math

THRESHOLD = 0.5

def score(alpha, beta=1.0):
    total = alpha * 2 + beta
    return total
"""

# Real logic change: the multiplier 2 -> 3.
LOGIC = '''\
"""Original docstring."""
import math

THRESHOLD = 0.5


def score(alpha: float, beta: float = 1.0) -> float:
    """Compute a score."""
    total: float = alpha * 3 + beta
    return total
'''


def test_cosmetic_edits_do_not_change_hash(monkeypatch, tmp_path):
    h_base = _hash_with_files(monkeypatch, tmp_path / "a", {"m.py": BASE})
    h_cos = _hash_with_files(monkeypatch, tmp_path / "b", {"m.py": COSMETIC})
    assert h_base == h_cos, "annotation/docstring/comment/whitespace edits must not move the hash"


def test_logic_edit_changes_hash(monkeypatch, tmp_path):
    h_base = _hash_with_files(monkeypatch, tmp_path / "a", {"m.py": BASE})
    h_logic = _hash_with_files(monkeypatch, tmp_path / "b", {"m.py": LOGIC})
    assert h_base != h_logic, "a real logic change must move the hash"


def test_legacy_scheme_is_fooled_by_annotations(monkeypatch, tmp_path):
    """Documents the original defect the behavioral scheme fixes."""
    for name, src in {"a": BASE, "b": COSMETIC}.items():
        p = tmp_path / name
        p.mkdir()
        (p / "m.py").write_text(src, encoding="utf-8")
    monkeypatch.setattr(m, "MODEL_FILES", [tmp_path / "a" / "m.py"])
    legacy_base = m.compute_model_hash_legacy()
    monkeypatch.setattr(m, "MODEL_FILES", [tmp_path / "b" / "m.py"])
    legacy_cos = m.compute_model_hash_legacy()
    assert legacy_base != legacy_cos  # raw bytes differ -> false drift


def test_scheme_tag_is_mixed_in(monkeypatch, tmp_path):
    """A scheme bump must change the hash so schemes never collide."""
    p = tmp_path / "m.py"
    p.write_text(BASE, encoding="utf-8")
    monkeypatch.setattr(m, "MODEL_FILES", [p])
    baseline = m.compute_model_hash()
    monkeypatch.setattr(m, "HASH_SCHEME", "ast-v-other")
    assert m.compute_model_hash() != baseline


def test_bare_annotation_dropped_valued_annotation_preserved(monkeypatch, tmp_path):
    # `x: int` (no value) has no runtime effect and must be ignored; `x: int = 5`
    # must be preserved as a real assignment (so changing its value moves the hash).
    with_bare = "def f():\n    x: int\n    return 1\n"
    without = "def f():\n    return 1\n"
    h1 = _hash_with_files(monkeypatch, tmp_path / "a", {"m.py": with_bare})
    h2 = _hash_with_files(monkeypatch, tmp_path / "b", {"m.py": without})
    assert h1 == h2

    valued_5 = "def f():\n    x: int = 5\n    return x\n"
    valued_6 = "def f():\n    x: int = 6\n    return x\n"
    h5 = _hash_with_files(monkeypatch, tmp_path / "c", {"m.py": valued_5})
    h6 = _hash_with_files(monkeypatch, tmp_path / "d", {"m.py": valued_6})
    assert h5 != h6


# --- ast-v2: selector config and order mode are inside the fingerprint (2026-11-15) ---

import json  # noqa: E402

RUN_SCREEN_STUB = """\
import os

A4_SELECTOR_CONFIG = SelectorConfig(
    block_weights=(BlockWeight("institutional", {weight}), BlockWeight("catalyst", 0.15)),
)

OTHER_SETTING = {other}


def unrelated():
    return {other}
"""


def _hash_with_config(monkeypatch, tmp_path, weight, other=1, names=("A4_SELECTOR_CONFIG",)):
    tmp_path.mkdir(parents=True, exist_ok=True)
    engine = tmp_path / "engine.py"
    engine.write_text(BASE, encoding="utf-8")
    run_screen = tmp_path / "run_screen.py"
    run_screen.write_text(RUN_SCREEN_STUB.format(weight=weight, other=other), encoding="utf-8")
    monkeypatch.setattr(m, "MODEL_FILES", [engine])
    monkeypatch.setattr(m, "MODEL_CONFIG_ASSIGNMENTS", {run_screen: names})
    return m.compute_model_hash()


def test_selector_weight_change_moves_hash(monkeypatch, tmp_path):
    assert _hash_with_config(monkeypatch, tmp_path / "a", 0.65) != _hash_with_config(monkeypatch, tmp_path / "b", 0.50)


def test_unrelated_run_screen_edit_does_not_move_hash(monkeypatch, tmp_path):
    h1 = _hash_with_config(monkeypatch, tmp_path / "a", 0.65, other=1)
    h2 = _hash_with_config(monkeypatch, tmp_path / "b", 0.65, other=2)
    assert h1 == h2


def test_missing_covered_assignment_moves_hash(monkeypatch, tmp_path):
    present = _hash_with_config(monkeypatch, tmp_path / "a", 0.65)
    renamed = _hash_with_config(monkeypatch, tmp_path / "b", 0.65, names=("A5_SELECTOR_CONFIG",))
    assert present != renamed
    assert "MISSING" in m._assignment_fingerprint(tmp_path / "b" / "run_screen.py", ("A5_SELECTOR_CONFIG",))


def test_real_repo_covers_selector_config_and_ranker_mode():
    assert m.HASH_SCHEME == "ast-v2"
    assert any(p.name == "ranker_mode.py" for p in m.MODEL_FILES)
    (path,) = [p for p in m.MODEL_CONFIG_ASSIGNMENTS if p.name == "run_screen.py"]
    fingerprint = m._assignment_fingerprint(path, m.MODEL_CONFIG_ASSIGNMENTS[path])
    assert "MISSING" not in fingerprint and "institutional" in fingerprint


def test_registered_candidate_matches_the_code():
    """CANDIDATE.json must be re-registered in the same change that moves the hash."""
    candidate = json.loads(m.CANDIDATE_FILE.read_text())
    assert candidate["hash_scheme"] == m.HASH_SCHEME
    assert candidate["model_hash"] == m.compute_model_hash()


def test_ast_v1_value_is_still_reproducible():
    candidate = json.loads(m.CANDIDATE_FILE.read_text())
    prior = [h for h in candidate.get("prior_model_hashes", []) if h["hash_scheme"] == "ast-v1"]
    assert prior and prior[0]["model_hash"] == m.compute_model_hash_ast_v1()
