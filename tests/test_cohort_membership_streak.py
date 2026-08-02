#!/usr/bin/env python3
"""Tests for cohort_membership_streak + cohort_churn_alert (audit follow-ups).

Display/monitoring only — none of these helpers should mutate scoring,
selector, ranker, or decision-engine fields.
"""

import csv
import json
from pathlib import Path

from run_screen import _annotate_cohort_membership_streaks, _classify_cohort_churn_severity, _write_cohort_churn_alert
from run_screen_columns import SNAPSHOT_COLUMNS


def _write_csv(path: Path, rows):
    fields = sorted({k for r in rows for k in r.keys()}) or ["ticker"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


class TestCohortStreakAnnotation:
    def test_in_cohort_today_with_no_prior_history(self, tmp_path):
        snap = tmp_path / "2026-04-25"
        snap.mkdir()
        rows = [{"ticker": "ABCD", "ranker_v2_score": "0.5"}]
        _annotate_cohort_membership_streaks(rows, snap)
        assert rows[0]["cohort_membership"] == "in"
        assert rows[0]["cohort_membership_streak"] == 1

    def test_out_cohort_today_with_no_prior_history(self, tmp_path):
        snap = tmp_path / "2026-04-25"
        snap.mkdir()
        rows = [{"ticker": "EFGH", "ranker_v2_score": ""}]
        _annotate_cohort_membership_streaks(rows, snap)
        assert rows[0]["cohort_membership"] == "out"
        assert rows[0]["cohort_membership_streak"] == 1

    def test_streak_grows_when_state_persists(self, tmp_path):
        for d in ("2026-04-23", "2026-04-24"):
            _write_csv(tmp_path / d / "rankings.csv", [{"ticker": "ARVN", "ranker_v2_score": "0.5"}])
        snap = tmp_path / "2026-04-25"
        snap.mkdir()
        rows = [{"ticker": "ARVN", "ranker_v2_score": "0.5"}]
        _annotate_cohort_membership_streaks(rows, snap)
        assert rows[0]["cohort_membership"] == "in"
        # 1 (today) + 2 prior days "in" = streak of 3
        assert rows[0]["cohort_membership_streak"] == 3

    def test_streak_resets_on_state_change(self, tmp_path):
        # Two days "in" then today "out" → streak=1
        for d in ("2026-04-23", "2026-04-24"):
            _write_csv(tmp_path / d / "rankings.csv", [{"ticker": "ERAS", "ranker_v2_score": "0.5"}])
        snap = tmp_path / "2026-04-25"
        snap.mkdir()
        rows = [{"ticker": "ERAS", "ranker_v2_score": ""}]
        _annotate_cohort_membership_streaks(rows, snap)
        assert rows[0]["cohort_membership"] == "out"
        assert rows[0]["cohort_membership_streak"] == 1

    def test_streak_breaks_on_intermediate_flip(self, tmp_path):
        # 04-23 in, 04-24 out, today in → streak=1 (04-24 out broke the chain)
        _write_csv(tmp_path / "2026-04-23" / "rankings.csv", [{"ticker": "X", "ranker_v2_score": "0.5"}])
        _write_csv(tmp_path / "2026-04-24" / "rankings.csv", [{"ticker": "X", "ranker_v2_score": ""}])
        snap = tmp_path / "2026-04-25"
        snap.mkdir()
        rows = [{"ticker": "X", "ranker_v2_score": "0.6"}]
        _annotate_cohort_membership_streaks(rows, snap)
        assert rows[0]["cohort_membership_streak"] == 1

    def test_walkback_cap_respected(self, tmp_path):
        # 50 days of "in" history; walkback cap=10 → max streak = 11.
        for i in range(50):
            d = f"2026-{(i // 31) + 1:02d}-{(i % 31) + 1:02d}"
            _write_csv(tmp_path / d / "rankings.csv", [{"ticker": "Z", "ranker_v2_score": "0.5"}])
        snap = tmp_path / "2026-04-25"
        snap.mkdir()
        rows = [{"ticker": "Z", "ranker_v2_score": "0.5"}]
        _annotate_cohort_membership_streaks(rows, snap, max_walkback_days=10)
        # 1 (today) + at most 10 prior = streak <= 11
        assert rows[0]["cohort_membership_streak"] <= 11

    def test_skips_suffixed_snapshot_dirs(self, tmp_path):
        # __pre_… and __stale_… variants must be ignored.
        _write_csv(tmp_path / "2026-04-24" / "rankings.csv", [{"ticker": "Y", "ranker_v2_score": "0.5"}])
        _write_csv(tmp_path / "2026-04-24__pre_xyz" / "rankings.csv", [{"ticker": "Y", "ranker_v2_score": ""}])
        snap = tmp_path / "2026-04-25"
        snap.mkdir()
        rows = [{"ticker": "Y", "ranker_v2_score": "0.5"}]
        _annotate_cohort_membership_streaks(rows, snap)
        # Suffixed dir should be ignored; only plain 2026-04-24 counts.
        assert rows[0]["cohort_membership_streak"] == 2

    def test_handles_none_snap_path_gracefully(self):
        rows = [
            {"ticker": "A", "ranker_v2_score": "0.5"},
            {"ticker": "B", "ranker_v2_score": ""},
        ]
        _annotate_cohort_membership_streaks(rows, None)
        assert rows[0]["cohort_membership"] == "in"
        assert rows[0]["cohort_membership_streak"] == 1
        assert rows[1]["cohort_membership"] == "out"
        assert rows[1]["cohort_membership_streak"] == 1

    def test_does_not_mutate_other_fields(self, tmp_path):
        snap = tmp_path / "2026-04-25"
        snap.mkdir()
        rows = [
            {
                "ticker": "ABCD",
                "ranker_v2_score": "0.5",
                "actionable_rank": 7,
                "composite_score": 0.099,
                "tier_any": "A",
            }
        ]
        before = dict(rows[0])
        _annotate_cohort_membership_streaks(rows, snap)
        for k, v in before.items():
            assert rows[0][k] == v, f"existing field {k} was mutated"


class TestCohortChurnAlertSeverity:
    def test_below_threshold_is_info(self):
        assert _classify_cohort_churn_severity(0.0) == "info"
        assert _classify_cohort_churn_severity(5.0) == "info"
        assert _classify_cohort_churn_severity(9.99) == "info"

    def test_at_or_above_threshold_is_warn(self):
        assert _classify_cohort_churn_severity(10.0) == "warn"
        assert _classify_cohort_churn_severity(15.0) == "warn"

    def test_custom_threshold(self):
        assert _classify_cohort_churn_severity(7.5, threshold_pct=5.0) == "warn"
        assert _classify_cohort_churn_severity(4.9, threshold_pct=5.0) == "info"


class TestCohortChurnAlertWriter:
    def test_writes_alert_with_no_prior_snapshot(self, tmp_path):
        snap = tmp_path / "2026-04-25"
        snap.mkdir()
        rows = [
            {"ticker": "A", "ranker_v2_score": "0.5"},
            {"ticker": "B", "ranker_v2_score": ""},
        ]
        alert = _write_cohort_churn_alert(rows, snap)
        assert alert is not None
        assert alert["prior_snapshot"] is None
        assert alert["today_cohort_n"] == 1
        assert alert["churn_n"] is None
        assert (snap / "cohort_churn_alert.json").exists()

    def test_zero_churn_when_cohort_unchanged(self, tmp_path):
        _write_csv(
            tmp_path / "2026-04-24" / "rankings.csv",
            [
                {"ticker": "A", "ranker_v2_score": "0.5"},
                {"ticker": "B", "ranker_v2_score": ""},
            ],
        )
        snap = tmp_path / "2026-04-25"
        snap.mkdir()
        rows = [
            {"ticker": "A", "ranker_v2_score": "0.6"},
            {"ticker": "B", "ranker_v2_score": ""},
        ]
        alert = _write_cohort_churn_alert(rows, snap)
        assert alert["churn_n"] == 0
        assert alert["churn_pct"] == 0.0
        assert alert["severity"] == "info"

    def test_warn_severity_at_10_percent(self, tmp_path):
        # Yesterday: 10 names in cohort. Today: 9 stay, 1 left, 1 new joined.
        # max(left, joined)/max_cohort = 1/10 = 10% → warn.
        prior = [{"ticker": f"T{i}", "ranker_v2_score": "0.5"} for i in range(10)]
        _write_csv(tmp_path / "2026-04-24" / "rankings.csv", prior)
        snap = tmp_path / "2026-04-25"
        snap.mkdir()
        # Drop T9, add NEW
        today = [{"ticker": f"T{i}", "ranker_v2_score": "0.5"} for i in range(9)]
        today.append({"ticker": "NEW", "ranker_v2_score": "0.5"})
        alert = _write_cohort_churn_alert(today, snap)
        assert alert["churn_pct"] >= 10.0
        assert alert["severity"] == "warn"
        assert "T9" in alert["names_left"]
        assert "NEW" in alert["names_joined"]

    def test_info_severity_below_threshold(self, tmp_path):
        # 50 names; 2 swap → 4% churn → info.
        prior = [{"ticker": f"T{i}", "ranker_v2_score": "0.5"} for i in range(50)]
        _write_csv(tmp_path / "2026-04-24" / "rankings.csv", prior)
        snap = tmp_path / "2026-04-25"
        snap.mkdir()
        today = [{"ticker": f"T{i}", "ranker_v2_score": "0.5"} for i in range(48)]
        today.extend(
            [
                {"ticker": "NEW1", "ranker_v2_score": "0.5"},
                {"ticker": "NEW2", "ranker_v2_score": "0.5"},
            ]
        )
        alert = _write_cohort_churn_alert(today, snap)
        assert alert["severity"] == "info"

    def test_artifact_is_valid_json(self, tmp_path):
        _write_csv(tmp_path / "2026-04-24" / "rankings.csv", [{"ticker": "A", "ranker_v2_score": "0.5"}])
        snap = tmp_path / "2026-04-25"
        snap.mkdir()
        rows = [{"ticker": "A", "ranker_v2_score": "0.5"}]
        _write_cohort_churn_alert(rows, snap)
        with open(snap / "cohort_churn_alert.json") as f:
            data = json.load(f)
        for k in ("as_of", "today_cohort_n", "names_left", "names_joined", "severity", "threshold_pct"):
            assert k in data


class TestSchemaIntegrity:
    def test_cohort_membership_in_snapshot_columns(self):
        assert "cohort_membership" in SNAPSHOT_COLUMNS
        assert "cohort_membership_streak" in SNAPSHOT_COLUMNS

    def test_no_duplicates(self):
        assert len(SNAPSHOT_COLUMNS) == len(set(SNAPSHOT_COLUMNS))


class TestStagingDirPriorSnapshotResolution:
    """Production runs write into a throwaway staging dir, not data/snapshots.

    `tools/run_daily_production.py` creates the staging root via
    `tempfile.mkdtemp(prefix=f"phase2_staging_{as_of_date}_")` and passes
    `--prior-snapshot-dir <real snapshots root>`. Deriving the search root from
    `snap_path.parent` therefore finds no history: the staging parent holds
    exactly one dated dir (today's). These tests pin the fix — both helpers must
    honour an explicit prior-snapshot root the way the rest of
    `save_validation_snapshot` already does via `_prior_dir`.
    """

    @staticmethod
    def _staging_and_history(tmp_path, history):
        """Build (staging_snap_path, real_snapshots_root) mirroring production."""
        real_root = tmp_path / "data" / "snapshots"
        for date, rows in history.items():
            _write_csv(real_root / date / "rankings.csv", rows)
        staging_root = tmp_path / "phase2_staging_2026-04-25_xyz"
        snap = staging_root / "2026-04-25"
        snap.mkdir(parents=True)
        return snap, real_root

    def test_churn_alert_uses_prior_snapshot_dir(self, tmp_path):
        snap, real_root = self._staging_and_history(
            tmp_path,
            {
                "2026-04-24": [
                    {"ticker": "A", "ranker_v2_score": "0.5"},
                    {"ticker": "B", "ranker_v2_score": "0.5"},
                ]
            },
        )
        rows = [
            {"ticker": "A", "ranker_v2_score": "0.6"},
            {"ticker": "B", "ranker_v2_score": ""},
        ]
        alert = _write_cohort_churn_alert(rows, snap, prior_snapshot_dir=real_root)
        assert alert["prior_snapshot"] == "2026-04-24"
        assert alert["churn_n"] == 1
        assert alert["churn_pct"] == 50.0
        assert alert["severity"] == "warn"
        assert alert["names_left"] == ["B"]

    def test_streak_walks_back_via_prior_snapshot_dir(self, tmp_path):
        snap, real_root = self._staging_and_history(
            tmp_path,
            {
                "2026-04-22": [{"ticker": "A", "ranker_v2_score": "0.5"}],
                "2026-04-23": [{"ticker": "A", "ranker_v2_score": "0.5"}],
                "2026-04-24": [{"ticker": "A", "ranker_v2_score": "0.5"}],
            },
        )
        rows = [{"ticker": "A", "ranker_v2_score": "0.6"}]
        _annotate_cohort_membership_streaks(rows, snap, prior_snapshot_dir=real_root)
        assert rows[0]["cohort_membership"] == "in"
        assert rows[0]["cohort_membership_streak"] == 4

    def test_todays_dir_in_history_is_not_self_compared(self, tmp_path):
        """A same-day rerun leaves 2026-04-25 in the real root; it must be skipped."""
        snap, real_root = self._staging_and_history(
            tmp_path,
            {
                "2026-04-24": [{"ticker": "A", "ranker_v2_score": "0.5"}],
                "2026-04-25": [{"ticker": "A", "ranker_v2_score": ""}],
            },
        )
        rows = [{"ticker": "A", "ranker_v2_score": "0.6"}]
        alert = _write_cohort_churn_alert(rows, snap, prior_snapshot_dir=real_root)
        assert alert["prior_snapshot"] == "2026-04-24"

    def test_omitting_prior_snapshot_dir_preserves_sibling_behaviour(self, tmp_path):
        """Back-compat: with no explicit root, both helpers still use snap_path.parent."""
        _write_csv(tmp_path / "2026-04-24" / "rankings.csv", [{"ticker": "A", "ranker_v2_score": "0.5"}])
        snap = tmp_path / "2026-04-25"
        snap.mkdir()
        rows = [{"ticker": "A", "ranker_v2_score": "0.6"}]
        alert = _write_cohort_churn_alert(rows, snap)
        _annotate_cohort_membership_streaks(rows, snap)
        assert alert["prior_snapshot"] == "2026-04-24"
        assert rows[0]["cohort_membership_streak"] == 2
