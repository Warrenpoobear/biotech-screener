"""Tests for common/pi_features.py (Spec 032)."""

from datetime import date

from common.pi_features import compute_pi_features_universe, is_pit_admitted, normalize_pi_name, z_score_pi_features

# ---------------------------------------------------------------------------
# normalize_pi_name
# ---------------------------------------------------------------------------


class TestNormalizePiName:
    def test_strip_md(self):
        assert normalize_pi_name("John Smith, M.D.") == "john smith"

    def test_strip_phd(self):
        assert normalize_pi_name("Jane Doe, PhD") == "jane doe"

    def test_strip_md_phd(self):
        assert normalize_pi_name("Alice Brown, M.D., Ph.D.") == "alice brown"

    def test_strip_do(self):
        assert normalize_pi_name("Bob Jones, D.O.") == "bob jones"

    def test_strip_dr_prefix(self):
        assert normalize_pi_name("Dr. John Smith") == "john smith"

    def test_strip_professor_prefix(self):
        assert normalize_pi_name("Professor Jane Doe") == "jane doe"

    def test_strip_prof_prefix(self):
        assert normalize_pi_name("Prof. Alice Brown") == "alice brown"

    def test_lowercase(self):
        assert normalize_pi_name("JOHN SMITH") == "john smith"

    def test_whitespace_collapse(self):
        assert normalize_pi_name("  John   Smith  ") == "john smith"

    def test_trailing_comma(self):
        assert normalize_pi_name("John Smith,") == "john smith"

    def test_trailing_period(self):
        assert normalize_pi_name("John Smith.") == "john smith"

    def test_empty(self):
        assert normalize_pi_name("") == ""

    def test_none_like(self):
        assert normalize_pi_name("   ") == ""

    def test_deterministic(self):
        name = "Dr. Ashley L Lynch, M.D."
        assert normalize_pi_name(name) == normalize_pi_name(name)

    def test_complex_credentials(self):
        assert normalize_pi_name("Jane Doe, MD, FACP, MPH") == "jane doe"

    def test_jr_suffix(self):
        assert normalize_pi_name("John Smith Jr.") == "john smith"

    def test_preserves_middle_name(self):
        assert normalize_pi_name("John Michael Smith, MD") == "john michael smith"


# ---------------------------------------------------------------------------
# PIT gate
# ---------------------------------------------------------------------------


class TestPitAdmitted:
    def test_admitted_by_first_posted(self):
        trial = {"first_posted": "2025-01-01", "last_update_posted": "2025-06-01"}
        assert is_pit_admitted(trial, date(2025, 3, 1)) is True

    def test_rejected_future_first_posted(self):
        trial = {"first_posted": "2025-06-01", "last_update_posted": "2025-07-01"}
        assert is_pit_admitted(trial, date(2025, 3, 1)) is False

    def test_admitted_same_day(self):
        trial = {"first_posted": "2025-03-01"}
        assert is_pit_admitted(trial, date(2025, 3, 1)) is True

    def test_no_pit_dates(self):
        trial = {"title": "some trial"}
        assert is_pit_admitted(trial, date(2025, 3, 1)) is False

    def test_partial_date_month_only(self):
        trial = {"first_posted": "2025-01"}
        assert is_pit_admitted(trial, date(2025, 1, 15)) is True


# ---------------------------------------------------------------------------
# compute_pi_features_universe
# ---------------------------------------------------------------------------


def _make_trial(ticker, nct_id, phase="PHASE2", status="RECRUITING", first_posted="2024-01-01"):
    return {
        "ticker": ticker,
        "nct_id": nct_id,
        "phase": phase,
        "status": status,
        "first_posted": first_posted,
        "last_update_posted": first_posted,
        "study_type": "INTERVENTIONAL",
    }


def _make_pi_index(entries):
    """entries: list of (nct_id, name)"""
    idx = {}
    for nct_id, name in entries:
        norm = normalize_pi_name(name)
        idx.setdefault(nct_id, []).append((norm, "PRINCIPAL_INVESTIGATOR"))
    return idx


class TestComputePiFeaturesUniverse:
    def test_basic_single_ticker(self):
        trials = [_make_trial("ACME", "NCT001")]
        pi_idx = _make_pi_index([("NCT001", "Dr. John Smith, MD")])
        result = compute_pi_features_universe(trials, pi_idx, {"ACME"}, date(2025, 1, 1))

        assert "ACME" in result
        f = result["ACME"]
        assert f["pi_count"] == 1
        assert f["pi_max_trial_count"] == 1
        assert f["n_trials_admitted"] == 1
        assert f["n_trials_with_pi"] == 1

    def test_pi_cross_company_experience(self):
        """PI appears in trials at two different companies — experience counted globally."""
        trials = [
            _make_trial("ACME", "NCT001"),
            _make_trial("ACME", "NCT002"),
            _make_trial("BETA", "NCT003"),
        ]
        pi_idx = _make_pi_index(
            [
                ("NCT001", "John Smith, MD"),
                ("NCT002", "John Smith, MD"),
                ("NCT003", "John Smith, MD"),  # same PI at different company
            ]
        )
        result = compute_pi_features_universe(trials, pi_idx, {"ACME", "BETA"}, date(2025, 1, 1))

        # ACME's best PI (john smith) has 3 trials globally
        assert result["ACME"]["pi_max_trial_count"] == 3
        # BETA's best PI also has 3
        assert result["BETA"]["pi_max_trial_count"] == 3

    def test_pit_gate_excludes_future(self):
        trials = [
            _make_trial("ACME", "NCT001", first_posted="2024-01-01"),
            _make_trial("ACME", "NCT002", first_posted="2026-01-01"),  # future
        ]
        pi_idx = _make_pi_index(
            [
                ("NCT001", "John Smith"),
                ("NCT002", "Jane Doe"),
            ]
        )
        result = compute_pi_features_universe(trials, pi_idx, {"ACME"}, date(2025, 6, 1))

        assert result["ACME"]["pi_count"] == 1  # only John Smith admitted
        assert result["ACME"]["n_trials_admitted"] == 1

    def test_missing_aact_data_graceful(self):
        trials = [_make_trial("ACME", "NCT001")]
        pi_idx = {}  # no AACT data at all
        result = compute_pi_features_universe(trials, pi_idx, {"ACME"}, date(2025, 1, 1))

        assert result["ACME"]["pi_count"] == 0
        assert result["ACME"]["pi_max_trial_count"] == 0
        assert result["ACME"]["pi_experience_z"] == 0.0

    def test_ticker_not_in_trials(self):
        """Universe ticker with no trials at all."""
        result = compute_pi_features_universe([], {}, {"ACME"}, date(2025, 1, 1))

        assert result["ACME"]["pi_count"] == 0
        assert result["ACME"]["n_trials_admitted"] == 0

    def test_late_stage_counting(self):
        trials = [
            _make_trial("ACME", "NCT001", phase="PHASE1"),
            _make_trial("ACME", "NCT002", phase="PHASE3"),
            _make_trial("ACME", "NCT003", phase="PHASE2"),
        ]
        pi_idx = _make_pi_index(
            [
                ("NCT001", "John Smith"),
                ("NCT002", "John Smith"),
                ("NCT003", "John Smith"),
            ]
        )
        result = compute_pi_features_universe(trials, pi_idx, {"ACME"}, date(2025, 1, 1))

        assert result["ACME"]["pi_max_trial_count"] == 3
        assert result["ACME"]["pi_max_late_stage_count"] == 2  # Phase 2 + Phase 3

    def test_completed_counting(self):
        trials = [
            _make_trial("ACME", "NCT001", status="COMPLETED"),
            _make_trial("ACME", "NCT002", status="RECRUITING"),
            _make_trial("ACME", "NCT003", status="COMPLETED"),
        ]
        pi_idx = _make_pi_index(
            [
                ("NCT001", "John Smith"),
                ("NCT002", "John Smith"),
                ("NCT003", "John Smith"),
            ]
        )
        result = compute_pi_features_universe(trials, pi_idx, {"ACME"}, date(2025, 1, 1))

        assert result["ACME"]["pi_max_completed_count"] == 2

    def test_multiple_pis_max_selected(self):
        """Ticker has two PIs with different experience — max wins."""
        trials = [
            _make_trial("ACME", "NCT001"),
            _make_trial("BETA", "NCT002"),
            _make_trial("BETA", "NCT003"),
        ]
        pi_idx = _make_pi_index(
            [
                ("NCT001", "Junior PI"),  # 1 trial
                ("NCT001", "Senior PI"),  # also on NCT001
                ("NCT002", "Senior PI"),  # 2 trials total for Senior
                ("NCT003", "Senior PI"),  # 3 trials total for Senior
            ]
        )
        result = compute_pi_features_universe(trials, pi_idx, {"ACME"}, date(2025, 1, 1))

        # ACME has Junior PI (1 trial) and Senior PI (3 trials) → max = 3
        assert result["ACME"]["pi_max_trial_count"] == 3
        assert result["ACME"]["pi_count"] == 2


# ---------------------------------------------------------------------------
# z_score
# ---------------------------------------------------------------------------


class TestZScore:
    def test_z_score_basic(self):
        features = {
            "A": {"pi_count": 1, "pi_max_trial_count": 10, "pi_experience_z": 0.0},
            "B": {"pi_count": 1, "pi_max_trial_count": 20, "pi_experience_z": 0.0},
            "C": {"pi_count": 1, "pi_max_trial_count": 30, "pi_experience_z": 0.0},
        }
        z_score_pi_features(features)

        # Mean = 20, std = sqrt(200/3) ≈ 8.165
        assert features["A"]["pi_experience_z"] < 0
        assert abs(features["B"]["pi_experience_z"]) < 0.01  # near zero (mean)
        assert features["C"]["pi_experience_z"] > 0

    def test_z_score_zero_pi_excluded(self):
        features = {
            "A": {"pi_count": 1, "pi_max_trial_count": 10, "pi_experience_z": 0.0},
            "B": {"pi_count": 0, "pi_max_trial_count": 0, "pi_experience_z": 0.0},
            "C": {"pi_count": 1, "pi_max_trial_count": 20, "pi_experience_z": 0.0},
        }
        z_score_pi_features(features)

        # B should stay at 0.0 (excluded from z-score)
        assert features["B"]["pi_experience_z"] == 0.0
        # A and C should be symmetric around 0
        assert features["A"]["pi_experience_z"] < 0
        assert features["C"]["pi_experience_z"] > 0

    def test_z_score_deterministic(self):
        features1 = {
            "A": {"pi_count": 1, "pi_max_trial_count": 5, "pi_experience_z": 0.0},
            "B": {"pi_count": 1, "pi_max_trial_count": 15, "pi_experience_z": 0.0},
        }
        features2 = {
            "A": {"pi_count": 1, "pi_max_trial_count": 5, "pi_experience_z": 0.0},
            "B": {"pi_count": 1, "pi_max_trial_count": 15, "pi_experience_z": 0.0},
        }
        z_score_pi_features(features1)
        z_score_pi_features(features2)

        assert features1["A"]["pi_experience_z"] == features2["A"]["pi_experience_z"]
        assert features1["B"]["pi_experience_z"] == features2["B"]["pi_experience_z"]

    def test_single_ticker_no_crash(self):
        features = {
            "A": {"pi_count": 1, "pi_max_trial_count": 10, "pi_experience_z": 0.0},
        }
        z_score_pi_features(features)
        # With only one value, z-score can't be computed meaningfully; value stays neutral (no crash)
        assert features["A"]["pi_experience_z"] == 0.0
