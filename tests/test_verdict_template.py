#!/usr/bin/env python3
"""Tests for the research verdict template in run_audited_backtest.py.

Covers _compute_verdict (pure function) and _write_verdict (file output).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "research"))

from run_audited_backtest import _compute_verdict, _write_verdict, _load_eval_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summary(
    horizons=(84, 126),
    net_84: float = 0.040,
    net_126: float = 0.047,
    turnover: float = 0.021,
) -> Dict[str, Any]:
    """Minimal eval summary.json structure."""
    by_horizon: Dict[str, Any] = {}
    for h in horizons:
        net = net_84 if h == 84 else net_126
        by_horizon[str(h)] = {
            "mean_net_return": net,
            "mean_excess_return": net + 0.001,
            "mean_hedged_return": net - 0.003,
            "mean_turnover": turnover,
            "mean_ic": -0.05,
            "ic_t_stat": -10.0,
        }
    return {
        "horizons": list(horizons),
        "n_evaluated": 282,
        "n_skipped": 0,
        "by_horizon": by_horizon,
    }


def _verdict(cand: Dict, baseline: Optional[Dict] = None, **kw) -> Dict:
    return _compute_verdict(cand, baseline, **kw)


# ---------------------------------------------------------------------------
# _compute_verdict — schema
# ---------------------------------------------------------------------------

class TestComputeVerdictSchema:
    def test_schema_field(self):
        v = _verdict(_summary())
        assert v["schema"] == "verdict.v1"

    def test_required_keys(self):
        v = _verdict(_summary())
        for key in ["verdict", "verdict_reasons", "results", "thresholds",
                    "n_evaluated", "primary_horizon", "guardrail_horizon"]:
            assert key in v

    def test_primary_is_max_horizon(self):
        v = _verdict(_summary(horizons=(63, 84, 126)))
        assert v["primary_horizon"] == 126

    def test_guardrail_is_second_longest(self):
        v = _verdict(_summary(horizons=(84, 126)))
        assert v["guardrail_horizon"] == 84

    def test_single_horizon_no_guardrail(self):
        v = _verdict(_summary(horizons=(126,)))
        assert v["guardrail_horizon"] is None


# ---------------------------------------------------------------------------
# _compute_verdict — NEEDS_MORE cases
# ---------------------------------------------------------------------------

class TestNeedsMore:
    def test_no_baseline_is_needs_more(self):
        v = _verdict(_summary(), baseline=None)
        assert v["verdict"] == "NEEDS_MORE"
        assert any("no baseline" in r for r in v["verdict_reasons"])

    def test_too_few_dates_is_needs_more(self):
        s = _summary()
        s["n_evaluated"] = 10
        v = _verdict(s, _summary())
        assert v["verdict"] == "NEEDS_MORE"
        assert any("min" in r for r in v["verdict_reasons"])

    def test_custom_min_dates(self):
        s = _summary()
        s["n_evaluated"] = 30
        # passes with min_dates=20
        v = _verdict(s, _summary(), min_dates=20)
        assert v["verdict"] != "NEEDS_MORE"


# ---------------------------------------------------------------------------
# _compute_verdict — PROMOTE
# ---------------------------------------------------------------------------

class TestPromote:
    def test_promote_when_both_pass(self):
        # cand net_126 = 0.050, base net_126 = 0.047 → Δ = +0.30pp ≥ +0.20pp
        # cand net_84  = 0.042, base net_84  = 0.040 → Δ = +0.20pp ≥ −0.05pp
        v = _verdict(
            _summary(net_84=0.042, net_126=0.050),
            _summary(net_84=0.040, net_126=0.047),
        )
        assert v["verdict"] == "PROMOTE"

    def test_promote_reasons_mention_both_horizons(self):
        v = _verdict(
            _summary(net_84=0.042, net_126=0.050),
            _summary(net_84=0.040, net_126=0.047),
        )
        combined = " ".join(v["verdict_reasons"])
        assert "126d" in combined
        assert "84d" in combined

    def test_delta_values_correct(self):
        v = _verdict(
            _summary(net_126=0.050),
            _summary(net_126=0.047),
        )
        delta = v["results"]["126"]["delta_net_pp"]
        assert delta == pytest.approx(0.30, abs=0.001)

    def test_single_horizon_promote_no_guardrail(self):
        s = _summary(horizons=(126,), net_126=0.050)
        b = _summary(horizons=(126,), net_126=0.047)
        v = _verdict(s, b)
        assert v["verdict"] == "PROMOTE"
        assert v["guardrail_horizon"] is None


# ---------------------------------------------------------------------------
# _compute_verdict — ARCHIVE
# ---------------------------------------------------------------------------

class TestArchive:
    def test_archive_when_primary_fails(self):
        # Δ126d = −0.054pp, Δ84d = −0.020pp (guardrail passes, primary fails)
        v = _verdict(
            _summary(net_84=0.040, net_126=0.0469),
            _summary(net_84=0.040, net_126=0.047),  # Δ = −0.01pp < +0.20pp
        )
        assert v["verdict"] == "ARCHIVE"

    def test_archive_when_guardrail_fails(self):
        # big primary gain but guardrail fails
        v = _verdict(
            _summary(net_84=0.035, net_126=0.055),
            _summary(net_84=0.040, net_126=0.047),
            # 84d Δ = −0.5pp < −0.05pp → fail
        )
        assert v["verdict"] == "ARCHIVE"

    def test_archive_reasons_mention_failing_horizon(self):
        v = _verdict(
            _summary(net_126=0.0469),
            _summary(net_126=0.047),
        )
        assert any("126d" in r for r in v["verdict_reasons"])

    def test_no_delta_when_no_baseline(self):
        v = _verdict(_summary())
        for row in v["results"].values():
            assert row["delta_net_pp"] is None


# ---------------------------------------------------------------------------
# _compute_verdict — threshold boundary
# ---------------------------------------------------------------------------

class TestBoundary:
    def test_exactly_at_primary_threshold_promotes(self):
        # Δ126d = exactly +0.20pp
        v = _verdict(
            _summary(net_126=0.04900),
            _summary(net_126=0.04700),
        )
        assert v["verdict"] == "PROMOTE"

    def test_just_below_primary_threshold_archives(self):
        # Δ126d = +0.1999pp
        v = _verdict(
            _summary(net_126=0.048999),
            _summary(net_126=0.047000),
        )
        assert v["verdict"] == "ARCHIVE"

    def test_exactly_at_guardrail_threshold_promotes(self):
        # primary passes (+0.30pp), guardrail exactly at −0.05pp
        v = _verdict(
            _summary(net_84=0.03950, net_126=0.050),
            _summary(net_84=0.04000, net_126=0.047),
        )
        assert v["verdict"] == "PROMOTE"

    def test_just_below_guardrail_archives(self):
        v = _verdict(
            _summary(net_84=0.03949, net_126=0.050),
            _summary(net_84=0.04000, net_126=0.047),
        )
        assert v["verdict"] == "ARCHIVE"


# ---------------------------------------------------------------------------
# _load_eval_summary
# ---------------------------------------------------------------------------

class TestLoadEvalSummary:
    def test_loads_valid_summary(self, tmp_path):
        s = _summary()
        (tmp_path / "summary.json").write_text(json.dumps(s))
        loaded = _load_eval_summary(tmp_path)
        assert loaded is not None
        assert loaded["n_evaluated"] == 282

    def test_returns_none_if_missing(self, tmp_path):
        assert _load_eval_summary(tmp_path) is None

    def test_returns_none_if_corrupt(self, tmp_path):
        (tmp_path / "summary.json").write_text("{bad json")
        assert _load_eval_summary(tmp_path) is None


# ---------------------------------------------------------------------------
# _write_verdict (file output)
# ---------------------------------------------------------------------------

class TestWriteVerdict:
    def _run(self, tmp_path: Path, verdict_override: Optional[str] = None):
        cand = _summary()
        baseline = _summary(net_84=0.040, net_126=0.047)
        v = _compute_verdict(cand, baseline)
        if verdict_override:
            v["verdict"] = verdict_override
            v["verdict_reasons"] = [f"forced {verdict_override}"]
        _write_verdict(
            tmp_path, v,
            name="test_run",
            git_sha="abc12345",
            run_id="20260305T120000Z",
            candidate_ruleset="production_data/decision_rulesets/candidate.json",
            baseline_ruleset="production_data/decision_rulesets/baseline.json",
            date_from="2020-03-31",
            date_to="2024-12-31",
            relaxed=False,
        )
        return tmp_path

    def test_files_created(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / "VERDICT.md").is_file()
        assert (tmp_path / "VERDICT.json").is_file()

    def test_json_schema(self, tmp_path):
        self._run(tmp_path)
        d = json.loads((tmp_path / "VERDICT.json").read_text())
        assert d["schema"] == "verdict.v1"
        assert d["name"] == "test_run"
        assert d["git_sha"] == "abc12345"

    def test_md_contains_verdict(self, tmp_path):
        self._run(tmp_path, verdict_override="ARCHIVE")
        md = (tmp_path / "VERDICT.md").read_text()
        assert "ARCHIVE" in md

    def test_promote_md_includes_promote_command(self, tmp_path):
        self._run(tmp_path, verdict_override="PROMOTE")
        md = (tmp_path / "VERDICT.md").read_text()
        assert "promote_ruleset.py" in md

    def test_archive_md_includes_archive_note(self, tmp_path):
        self._run(tmp_path, verdict_override="ARCHIVE")
        md = (tmp_path / "VERDICT.md").read_text()
        assert "research_archive" in md

    def test_md_contains_git_sha(self, tmp_path):
        self._run(tmp_path)
        md = (tmp_path / "VERDICT.md").read_text()
        assert "abc12345" in md

    def test_md_contains_horizons(self, tmp_path):
        self._run(tmp_path)
        md = (tmp_path / "VERDICT.md").read_text()
        assert "126d" in md
        assert "84d" in md
