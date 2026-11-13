"""ranker_mode="selector_only": production order is the selector score, nothing else.

Classification: OPERATOR_AUTHORIZED_MODEL_CHANGE_2026_11_13 (Tier 3)
"""

from __future__ import annotations

import re
from pathlib import Path

from common.ranker_mode import DEFAULT_RANKER_MODE, RANKER_MODES, apply_selector_only_ranking

REPO_ROOT = Path(__file__).resolve().parents[1]


def _rows():
    return [
        {"ticker": "AAA", "selector_score": 0.90, "financial_score": 10, "coinvest_score_z": 2.0},
        {"ticker": "BBB", "selector_score": "0.95", "financial_score": 90, "coinvest_score_z": 2.5},
        {"ticker": "CCC", "selector_score": 0.10, "financial_score": 5, "coinvest_score_z": -1.0},
    ]


def test_default_is_selector_only_and_rollback_modes_exist():
    assert DEFAULT_RANKER_MODE == "selector_only"
    assert RANKER_MODES == ("selector_only", "pairwise_minimal", "clinical_50")


def test_final_score_is_the_selector_score_for_every_eligible_row():
    rows = _rows()
    apply_selector_only_ranking(rows)
    assert [r["final_score"] for r in rows] == [0.90, 0.95, 0.10]
    assert {r["ranker_active"] for r in rows} == {"0"}
    assert sorted(rows, key=lambda r: -r["final_score"])[0]["ticker"] == "BBB"


def test_ranker_scores_are_kept_as_shadow_and_do_not_change_the_order():
    # The pairwise ranker would put AAA first (low financial_score); selector order has BBB first.
    shadow = {
        "AAA": {"ranker_v2_score": 0.71, "ranker_v2_rank": 1},
        "BBB": {"ranker_v2_score": 0.40, "ranker_v2_rank": 2},
        "CCC": {"ranker_v2_score": None, "ranker_v2_rank": None},
    }
    rows = _rows()
    n = apply_selector_only_ranking(rows, shadow)
    by = {r["ticker"]: r for r in rows}
    assert n == 2
    assert by["AAA"]["ranker_v2_rank"] == 1 and by["AAA"]["ranker_v2_score"] == 0.71
    assert by["CCC"]["ranker_v2_score"] == "" and by["CCC"]["ranker_v2_rank"] == ""
    assert [r["ticker"] for r in sorted(rows, key=lambda r: -r["final_score"])] == ["BBB", "AAA", "CCC"]


def test_final_score_ignores_financial_score():
    a, b = _rows(), _rows()
    for r in b:
        r["financial_score"] = 100 - r["financial_score"]
    apply_selector_only_ranking(a)
    apply_selector_only_ranking(b)
    assert [r["final_score"] for r in a] == [r["final_score"] for r in b]


def test_missing_or_bad_selector_score_becomes_zero_not_an_error():
    rows = [{"ticker": "AAA", "selector_score": ""}, {"ticker": "BBB"}, {"ticker": "CCC", "selector_score": "nan"}]
    apply_selector_only_ranking(rows)
    assert [r["final_score"] for r in rows] == [0.0, 0.0, 0.0]


def test_options_and_ranker_blocks_are_blank():
    rows = _rows()
    apply_selector_only_ranking(rows)
    for r in rows:
        assert r["ranker_options_block"] == r["ranker_inst_block"] == r["ranker_aact_block"] == ""
        assert r["ranker_adjustment"] == 0.0


def test_run_screen_wires_the_default_and_the_choices():
    src = (REPO_ROOT / "run_screen.py").read_text(encoding="utf-8")
    assert "from common.ranker_mode import DEFAULT_RANKER_MODE, RANKER_MODES, apply_selector_only_ranking" in src
    assert "choices=list(RANKER_MODES)" in src
    assert len(re.findall(r"DEFAULT_RANKER_MODE", src)) >= 5  # import, CLI default, two signatures, getattr
    assert 'if ranker_mode == "selector_only":' in src
