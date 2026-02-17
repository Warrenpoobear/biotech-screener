"""Tests for _hydrate_far_horizon_catalysts() — far-horizon catalyst hydration."""
from __future__ import annotations

import pytest

from decision_engine import DecisionRuleset
from run_screen import _hydrate_far_horizon_catalysts


def _make_row(ticker, catalyst_mode="no_upcoming", **extra):
    """Build a minimal csv_row dict."""
    row = {
        "ticker": ticker,
        "catalyst_mode": catalyst_mode,
        "catalyst_days": "",
        "catalyst_source": "",
        "catalyst_event_type": "",
        "catalyst_strength": "",
        "catalyst_decay_w": "",
    }
    row.update(extra)
    return row


def _make_trial(ticker, study_type="INTERVENTIONAL", status="RECRUITING",
                pcd=None):
    """Build a minimal trial_record dict."""
    t = {"ticker": ticker, "study_type": study_type, "status": status}
    if pcd:
        t["primary_completion_date"] = pcd
    return t


DEFAULT_RULESET = DecisionRuleset(far_window_days=540, far_window_decay_mult=0.15)


class TestHydrateFarHorizonCatalysts:
    """Tests for _hydrate_far_horizon_catalysts()."""

    def test_active_trial_within_far_window_overridden(self):
        """Ticker with active interventional trial, PCD at ~350d → overridden."""
        rows = [_make_row("ACAD")]
        # PCD 350 days from 2026-02-14 → 2027-01-30
        trials = [_make_trial("ACAD", pcd="2027-01-30")]
        n = _hydrate_far_horizon_catalysts(rows, trials, "2026-02-14", DEFAULT_RULESET)
        assert n == 1
        assert rows[0]["catalyst_mode"] == "far_window"
        assert rows[0]["catalyst_source"] == "CTGOV_PCD_FAR"
        assert rows[0]["catalyst_event_type"] == "CT_PRIMARY_COMPLETION"
        assert rows[0]["catalyst_strength"] == "far"
        assert rows[0]["catalyst_decay_w"] == 0.15
        # catalyst_days should be roughly 350
        assert 340 <= rows[0]["catalyst_days"] <= 360

    def test_pcd_beyond_far_window_not_overridden(self):
        """Ticker with PCD at 600d (> far_window_days=540) → NOT overridden."""
        rows = [_make_row("TOOFAR")]
        # PCD 600 days from 2026-02-14 → 2027-10-07
        trials = [_make_trial("TOOFAR", pcd="2027-10-07")]
        n = _hydrate_far_horizon_catalysts(rows, trials, "2026-02-14", DEFAULT_RULESET)
        assert n == 0
        assert rows[0]["catalyst_mode"] == "no_upcoming"

    def test_terminal_trials_not_counted(self):
        """Ticker with only terminated/withdrawn trials → NOT overridden."""
        rows = [_make_row("TERM")]
        trials = [
            _make_trial("TERM", status="TERMINATED", pcd="2027-01-30"),
            _make_trial("TERM", status="WITHDRAWN", pcd="2026-12-01"),
        ]
        n = _hydrate_far_horizon_catalysts(rows, trials, "2026-02-14", DEFAULT_RULESET)
        assert n == 0
        assert rows[0]["catalyst_mode"] == "no_upcoming"

    def test_specific_days_not_touched(self):
        """Ticker with specific_days catalyst_mode → NOT touched."""
        rows = [_make_row("GOOD", catalyst_mode="specific_days", catalyst_days="60")]
        trials = [_make_trial("GOOD", pcd="2027-01-30")]
        n = _hydrate_far_horizon_catalysts(rows, trials, "2026-02-14", DEFAULT_RULESET)
        assert n == 0
        assert rows[0]["catalyst_mode"] == "specific_days"
        assert rows[0]["catalyst_days"] == "60"

    def test_blended_window_not_touched(self):
        """Ticker with blended_window catalyst_mode → NOT touched."""
        rows = [_make_row("BLD", catalyst_mode="blended_window")]
        trials = [_make_trial("BLD", pcd="2027-01-30")]
        n = _hydrate_far_horizon_catalysts(rows, trials, "2026-02-14", DEFAULT_RULESET)
        assert n == 0
        assert rows[0]["catalyst_mode"] == "blended_window"

    def test_no_trial_records_no_override(self):
        """Ticker with no matching trial_records → NOT overridden."""
        rows = [_make_row("LONELY")]
        trials = []
        n = _hydrate_far_horizon_catalysts(rows, trials, "2026-02-14", DEFAULT_RULESET)
        assert n == 0
        assert rows[0]["catalyst_mode"] == "no_upcoming"

    def test_observational_trial_not_counted(self):
        """Ticker with only OBSERVATIONAL trials → NOT overridden."""
        rows = [_make_row("OBS")]
        trials = [_make_trial("OBS", study_type="OBSERVATIONAL", pcd="2027-01-30")]
        n = _hydrate_far_horizon_catalysts(rows, trials, "2026-02-14", DEFAULT_RULESET)
        assert n == 0

    def test_past_pcd_not_counted(self):
        """Ticker with only past PCD → NOT overridden."""
        rows = [_make_row("PAST")]
        trials = [_make_trial("PAST", pcd="2025-01-01")]
        n = _hydrate_far_horizon_catalysts(rows, trials, "2026-02-14", DEFAULT_RULESET)
        assert n == 0

    def test_missing_mode_also_overridden(self):
        """Ticker with 'missing' catalyst_mode is also eligible for override."""
        rows = [_make_row("MSS", catalyst_mode="missing")]
        trials = [_make_trial("MSS", pcd="2027-01-30")]
        n = _hydrate_far_horizon_catalysts(rows, trials, "2026-02-14", DEFAULT_RULESET)
        assert n == 1
        assert rows[0]["catalyst_mode"] == "far_window"

    def test_picks_min_future_pcd(self):
        """When multiple future PCDs, picks the nearest one."""
        rows = [_make_row("MULTI")]
        trials = [
            _make_trial("MULTI", pcd="2027-06-01"),  # ~472d
            _make_trial("MULTI", pcd="2027-01-01"),  # ~321d (closer)
            _make_trial("MULTI", pcd="2026-12-01"),  # ~290d (closest)
        ]
        n = _hydrate_far_horizon_catalysts(rows, trials, "2026-02-14", DEFAULT_RULESET)
        assert n == 1
        assert rows[0]["catalyst_days"] < 300  # closest PCD wins

    def test_custom_decay_mult(self):
        """Custom far_window_decay_mult is used in override."""
        rs = DecisionRuleset(far_window_days=540, far_window_decay_mult=0.25)
        rows = [_make_row("CUST")]
        trials = [_make_trial("CUST", pcd="2027-01-30")]
        _hydrate_far_horizon_catalysts(rows, trials, "2026-02-14", rs)
        assert rows[0]["catalyst_decay_w"] == 0.25

    def test_far_window_days_zero_no_overrides(self):
        """When far_window_days=0, no overrides occur (all PCDs are > 0)."""
        rs = DecisionRuleset(far_window_days=0, far_window_decay_mult=0.15)
        rows = [_make_row("ZERO")]
        trials = [_make_trial("ZERO", pcd="2027-01-30")]
        n = _hydrate_far_horizon_catalysts(rows, trials, "2026-02-14", rs)
        assert n == 0

    def test_return_count_matches_overrides(self):
        """Return count equals number of overridden tickers."""
        rows = [
            _make_row("A", catalyst_mode="no_upcoming"),
            _make_row("B", catalyst_mode="specific_days"),  # not touched
            _make_row("C", catalyst_mode="missing"),
            _make_row("D", catalyst_mode="no_upcoming"),    # no trial
        ]
        trials = [
            _make_trial("A", pcd="2027-01-30"),
            _make_trial("B", pcd="2027-01-30"),  # won't be touched (specific_days)
            _make_trial("C", pcd="2027-01-30"),
            # no trial for D
        ]
        n = _hydrate_far_horizon_catalysts(rows, trials, "2026-02-14", DEFAULT_RULESET)
        assert n == 2  # A and C overridden
        assert rows[0]["catalyst_mode"] == "far_window"
        assert rows[1]["catalyst_mode"] == "specific_days"
        assert rows[2]["catalyst_mode"] == "far_window"
        assert rows[3]["catalyst_mode"] == "no_upcoming"


class TestRulesetValidation:
    """Tests for far_window DecisionRuleset field validation."""

    def test_negative_far_window_days_raises(self):
        with pytest.raises(ValueError, match="far_window_days"):
            DecisionRuleset(far_window_days=-1)

    def test_zero_far_window_days_ok(self):
        rs = DecisionRuleset(far_window_days=0)
        assert rs.far_window_days == 0

    def test_decay_mult_zero_raises(self):
        with pytest.raises(ValueError, match="far_window_decay_mult"):
            DecisionRuleset(far_window_decay_mult=0.0)

    def test_decay_mult_above_one_raises(self):
        with pytest.raises(ValueError, match="far_window_decay_mult"):
            DecisionRuleset(far_window_decay_mult=1.5)

    def test_decay_mult_one_ok(self):
        rs = DecisionRuleset(far_window_decay_mult=1.0)
        assert rs.far_window_decay_mult == 1.0
