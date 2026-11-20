"""Sign-corrected copies of IC dashboard artifacts written before the 2026-11-12 fix.

Until PR #652 the dashboard negated ``score_rank_pct`` (which is higher-is-better), so
every stored ``score_rank_pct`` IC has the wrong sign. The originals are left as
written (docs/governance/2026-11-12-ic-dashboard-sign-correction.md §4); the tool
writes corrected copies beside them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.correct_ic_dashboard_sign import (  # noqa: E402
    correct_dashboard,
    correct_history_row,
    is_pre_fix_dashboard,
    write_corrected_copies,
)


def _pre_fix_dashboard():
    return {
        "as_of_date": "2026-11-11",
        "attention": "LOW",
        "signals": {
            "score_rank_pct": {
                "mean_ic": 0.1764,
                "hit_rate": 1.0,
                "n_dates": 40,
                "health": "HEALTHY",
                "latest_ic": 0.0068,
                "per_date": [{"date": "2026-08-18", "ic": 0.1399, "n_obs": 295}],
            },
            "inst_delta_z": {"mean_ic": 0.05, "hit_rate": 0.6, "n_dates": 40, "health": "HEALTHY", "per_date": []},
        },
    }


def _post_fix_dashboard():
    d = _pre_fix_dashboard()
    d["as_of_date"] = "2026-11-12"
    d["signals"]["actionable_rank"] = {"mean_ic": 0.02, "health": "WEAK", "per_date": []}
    return d


def test_detects_pre_fix_by_missing_actionable_rank():
    assert is_pre_fix_dashboard(_pre_fix_dashboard())
    assert not is_pre_fix_dashboard(_post_fix_dashboard())


def test_flips_score_rank_pct_and_rederives_health():
    out = correct_dashboard(_pre_fix_dashboard())
    s = out["signals"]["score_rank_pct"]
    assert s["mean_ic"] == -0.1764
    assert s["latest_ic"] == -0.0068
    assert s["hit_rate"] == 0.0
    assert s["per_date"][0]["ic"] == -0.1399
    assert s["health"] == "ALERT"
    assert out["attention"] == "HIGH"
    # other signals were not negated by the old code and stay as written
    assert out["signals"]["inst_delta_z"]["mean_ic"] == 0.05
    assert out["sign_correction"]["fields"] == ["score_rank_pct"]


def test_insufficient_and_no_data_labels_are_kept():
    d = _pre_fix_dashboard()
    d["signals"]["score_rank_pct"]["health"] = "INSUFFICIENT_DATES"
    assert correct_dashboard(d)["signals"]["score_rank_pct"]["health"] == "INSUFFICIENT_DATES"


def test_history_row_flip():
    row = {"date": "2026-11-11", "attention": "LOW", "score_rank_pct_ic": 0.1764, "score_rank_pct_health": "HEALTHY"}
    out = correct_history_row(row)
    assert out["score_rank_pct_ic"] == -0.1764
    assert out["score_rank_pct_health"] == "ALERT"
    assert out["attention"] == "HIGH"
    assert out["sign_corrected"] is True
    null_row = {"date": "2026-03-27", "score_rank_pct_ic": None, "score_rank_pct_health": "NO_DATA"}
    assert correct_history_row(null_row)["score_rank_pct_health"] == "NO_DATA"


def test_post_fix_history_row_unchanged():
    row = {"date": "2026-11-12", "score_rank_pct_ic": -0.0253, "actionable_rank_ic": 0.02}
    assert correct_history_row(row) == {**row, "sign_corrected": False}


def test_write_leaves_originals_untouched(tmp_path):
    src = tmp_path / "ic_dashboard"
    src.mkdir()
    pre = src / "2026-11-11_dashboard.json"
    post = src / "2026-11-12_dashboard.json"
    pre.write_text(json.dumps(_pre_fix_dashboard()))
    post.write_text(json.dumps(_post_fix_dashboard()))
    hist = src / "history.jsonl"
    hist.write_text(
        json.dumps({"date": "2026-11-11", "score_rank_pct_ic": 0.1764, "score_rank_pct_health": "HEALTHY"})
        + "\n"
        + json.dumps({"date": "2026-11-12", "score_rank_pct_ic": -0.0253, "actionable_rank_ic": 0.02})
        + "\n"
    )
    before = {p.name: p.read_bytes() for p in src.iterdir()}
    summary = write_corrected_copies(src)
    assert {p.name: p.read_bytes() for p in src.iterdir() if p.is_file()} == before
    out = src / "sign_corrected"
    assert (
        json.loads((out / "2026-11-11_dashboard.json").read_text())["signals"]["score_rank_pct"]["mean_ic"] == -0.1764
    )
    assert not (out / "2026-11-12_dashboard.json").exists()
    rows = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    assert [r["sign_corrected"] for r in rows] == [True, False]
    assert summary["dashboards_corrected"] == 1
    # deterministic: a second run produces identical bytes
    first = (out / "history.jsonl").read_bytes()
    write_corrected_copies(src)
    assert (out / "history.jsonl").read_bytes() == first
