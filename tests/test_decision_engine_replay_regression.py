"""
Decision Engine Replay Regression Tests
========================================

Loads compact fixtures extracted from 3 real archives (2025-04-30, 2025-07-31,
2025-10-31), runs the decision engine with the Phase-2 production ruleset
(181346fe, a_floor=0.60, catalyst_near=120, catalyst_mid=180, drawdown_rel_xbi_gate=-0.15), and asserts on 7 categories of metrics:

1. Real data ingest — engine doesn't crash, universe size is sane
2. Schema on real data — all expected columns present, valid enums
3. Score dispersion — tiers not flat, optionality not degenerate
4. Tier distribution — counts within historical bounds
5. Coverage metrics — eligibility, catalyst, sponsor, momentum feed health
6. Golden tickers — specific ticker×date → exact tier+band
7. Rank stability — cross-date overlap and swing within bounds

Fixtures: tests/fixtures/phase2_replays/{2025-04-30,2025-07-31,2025-10-31}.json
Runtime: <5 seconds (JSON load + engine calls for ~960 tickers total)
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# ---------------------------------------------------------------------------
# Imports from project
# ---------------------------------------------------------------------------
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import (
    DECISION_COLUMNS,
    DecisionRuleset,
    compute_decision_fields,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "phase2_replays"
RULESET_PATH = (
    PROJECT_ROOT / "production_data" / "decision_rulesets" / "v1.2.1_candidate.json"
)
DATES = ["2025-04-30", "2025-07-31", "2025-10-31"]
EXPECTED_RULESET_ID = "5a9faad9"

# Valid enum values for categorical fields
VALID_TIERS = {"A", "B", "C", "D", ""}
VALID_BANDS = {"L", "M", "S", "XS"}
VALID_ELIGIBLE = {"0", "1"}
VALID_MOM_STATES = {"tailwind", "neutral", "headwind", ""}
VALID_CATALYST_MODES = {"specific_days", "blended_window", "no_upcoming", "missing", ""}
VALID_RUNWAY_BUCKETS = {"critical", "short", "adequate", ""}


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------
def _load_fixture(date_str: str) -> Dict[str, Any]:
    path = FIXTURE_DIR / f"{date_str}.json"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def _run_engine(
    fixture: Dict[str, Any], ruleset: DecisionRuleset
) -> List[Dict[str, Any]]:
    """Run compute_decision_fields on every ticker in fixture, return results."""
    results = []
    for ticker in fixture["tickers"]:
        rec = fixture["recs"].get(ticker)
        if rec is None:
            continue
        fields = compute_decision_fields(
            rec=rec,
            archetype=fixture["archetypes"].get(ticker, ""),
            optionality_pct_dev=fixture["optionalities"].get(ticker),
            ruleset=ruleset,
        )
        fields["ticker"] = ticker
        fields["archetype"] = fixture["archetypes"].get(ticker, "")
        fields["optionality"] = fixture["optionalities"].get(ticker)
        results.append(fields)
    return results


# ---------------------------------------------------------------------------
# Module-scoped fixtures (cached — engine runs once per date)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ruleset() -> DecisionRuleset:
    if not RULESET_PATH.exists():
        pytest.skip(f"Ruleset not found: {RULESET_PATH}")
    return DecisionRuleset.from_json(str(RULESET_PATH))


@pytest.fixture(scope="module")
def all_snapshots(
    ruleset: DecisionRuleset,
) -> Dict[str, Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Returns {date_str: (fixture_dict, engine_results_list)}."""
    snapshots = {}
    for date_str in DATES:
        fixture = _load_fixture(date_str)
        results = _run_engine(fixture, ruleset)
        snapshots[date_str] = (fixture, results)
    return snapshots


def _dev_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter to drug_developer archetype (has tier_dev != "")."""
    return [r for r in results if r.get("tier_dev") != ""]


# ============================================================================
# 1. TestRealDataIngest — engine processes real data without crashing
# ============================================================================
class TestRealDataIngest:
    """Engine doesn't crash, output sizes match input, universe size is sane."""

    @pytest.mark.parametrize("date_str", DATES)
    def test_no_crash(self, all_snapshots, date_str):
        """Engine returns results for every ticker without exceptions."""
        fixture, results = all_snapshots[date_str]
        assert len(results) == len(fixture["tickers"])

    @pytest.mark.parametrize("date_str", DATES)
    def test_universe_size_in_range(self, all_snapshots, date_str):
        """Universe has 280-400 tickers (historical range)."""
        _, results = all_snapshots[date_str]
        assert 280 <= len(results) <= 400, (
            f"{date_str}: {len(results)} tickers outside [280, 400]"
        )

    @pytest.mark.parametrize("date_str", DATES)
    def test_all_tickers_produce_output(self, all_snapshots, date_str):
        """Every ticker in fixture gets a result dict with fields."""
        fixture, results = all_snapshots[date_str]
        result_tickers = {r["ticker"] for r in results}
        fixture_tickers = set(fixture["tickers"])
        assert result_tickers == fixture_tickers


# ============================================================================
# 2. TestSchemaOnRealData — columns present, valid enums
# ============================================================================
class TestSchemaOnRealData:
    """All decision columns present; categorical values are valid enums."""

    @pytest.mark.parametrize("date_str", DATES)
    def test_all_decision_columns_present(self, all_snapshots, date_str):
        _, results = all_snapshots[date_str]
        for r in results:
            missing = [c for c in DECISION_COLUMNS if c not in r]
            assert not missing, f"{r['ticker']}: missing columns {missing}"

    @pytest.mark.parametrize("date_str", DATES)
    def test_tier_dev_valid(self, all_snapshots, date_str):
        _, results = all_snapshots[date_str]
        for r in results:
            assert r["tier_dev"] in VALID_TIERS, (
                f"{r['ticker']}: invalid tier_dev={r['tier_dev']!r}"
            )

    @pytest.mark.parametrize("date_str", DATES)
    def test_size_band_valid(self, all_snapshots, date_str):
        _, results = all_snapshots[date_str]
        for r in results:
            assert r["size_band"] in VALID_BANDS, (
                f"{r['ticker']}: invalid size_band={r['size_band']!r}"
            )

    @pytest.mark.parametrize("date_str", DATES)
    def test_eligible_valid(self, all_snapshots, date_str):
        _, results = all_snapshots[date_str]
        for r in results:
            assert r["eligible"] in VALID_ELIGIBLE, (
                f"{r['ticker']}: invalid eligible={r['eligible']!r}"
            )

    @pytest.mark.parametrize("date_str", DATES)
    def test_mom_state_valid(self, all_snapshots, date_str):
        _, results = all_snapshots[date_str]
        for r in results:
            assert r["mom_state"] in VALID_MOM_STATES, (
                f"{r['ticker']}: invalid mom_state={r['mom_state']!r}"
            )

    @pytest.mark.parametrize("date_str", DATES)
    def test_catalyst_mode_valid(self, all_snapshots, date_str):
        _, results = all_snapshots[date_str]
        for r in results:
            assert r["catalyst_mode"] in VALID_CATALYST_MODES, (
                f"{r['ticker']}: invalid catalyst_mode={r['catalyst_mode']!r}"
            )

    @pytest.mark.parametrize("date_str", DATES)
    def test_runway_bucket_valid(self, all_snapshots, date_str):
        _, results = all_snapshots[date_str]
        for r in results:
            assert r["runway_bucket"] in VALID_RUNWAY_BUCKETS, (
                f"{r['ticker']}: invalid runway_bucket={r['runway_bucket']!r}"
            )

    @pytest.mark.parametrize("date_str", DATES)
    def test_ruleset_id_matches(self, all_snapshots, date_str):
        _, results = all_snapshots[date_str]
        for r in results:
            assert r["decision_engine_ruleset_id"] == EXPECTED_RULESET_ID, (
                f"{r['ticker']}: ruleset_id={r['decision_engine_ruleset_id']}"
            )


# ============================================================================
# 3. TestScoreDispersion — tiers not flat, optionality not degenerate
# ============================================================================
class TestScoreDispersion:
    """Ensures non-trivial distributional properties — not all one tier,
    not all one band, optionality has real variance."""

    @pytest.mark.parametrize("date_str", DATES)
    def test_tier_dispersion_at_least_3(self, all_snapshots, date_str):
        """Dev tickers produce at least 3 distinct tiers."""
        _, results = all_snapshots[date_str]
        dev = _dev_results(results)
        tiers = {r["tier_dev"] for r in dev}
        assert len(tiers) >= 3, f"{date_str}: only {tiers}"

    @pytest.mark.parametrize("date_str", DATES)
    def test_band_dispersion_at_least_3(self, all_snapshots, date_str):
        """At least 3 distinct size bands across all tickers."""
        _, results = all_snapshots[date_str]
        bands = {r["size_band"] for r in results}
        assert len(bands) >= 3, f"{date_str}: only {bands}"

    @pytest.mark.parametrize("date_str", DATES)
    def test_optionality_stddev(self, all_snapshots, date_str):
        """Dev-stage optionality has stddev > 0.15 (not degenerate)."""
        fixture, results = all_snapshots[date_str]
        dev = _dev_results(results)
        opt_vals = [
            r["optionality"] for r in dev if r["optionality"] is not None
        ]
        assert len(opt_vals) >= 10, f"{date_str}: only {len(opt_vals)} opt values"
        std = statistics.stdev(opt_vals)
        assert std > 0.15, f"{date_str}: optionality std={std:.3f} < 0.15"

    @pytest.mark.parametrize("date_str", DATES)
    def test_not_all_same_eligible(self, all_snapshots, date_str):
        """Not 100% eligible or 100% ineligible."""
        _, results = all_snapshots[date_str]
        eligible_vals = {r["eligible"] for r in results}
        assert len(eligible_vals) == 2, (
            f"{date_str}: all tickers have eligible={eligible_vals}"
        )


# ============================================================================
# 4. TestTierDistribution — counts within historical bounds
# ============================================================================
class TestTierDistribution:
    """Tier percentages fall within historical norms.

    Calibrated from 3 snapshots (2025-04-30, 2025-07-31, 2025-10-31):
    - A: 1-3% → bounds [0%, 10%]
    - B: 7-10% → bounds [3%, 25%]
    - C: 12-14% → bounds [5%, 30%]
    - D: 76-77% → bounds [50%, 90%]
    """

    @pytest.mark.parametrize("date_str", DATES)
    def test_tier_pcts_in_bounds(self, all_snapshots, date_str):
        _, results = all_snapshots[date_str]
        dev = _dev_results(results)
        n_dev = len(dev)
        assert n_dev > 0, f"{date_str}: no dev tickers"

        tier_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        for r in dev:
            t = r.get("tier_dev", "")
            if t in tier_counts:
                tier_counts[t] += 1

        bounds = {
            "A": (0.0, 10.0),
            "B": (3.0, 25.0),
            "C": (5.0, 30.0),
            "D": (50.0, 90.0),
        }
        for tier, (lo, hi) in bounds.items():
            pct = tier_counts[tier] / n_dev * 100
            assert lo <= pct <= hi, (
                f"{date_str}: tier {tier} = {pct:.1f}% "
                f"(count={tier_counts[tier]}/{n_dev}) outside [{lo}, {hi}]"
            )

    @pytest.mark.parametrize("date_str", DATES)
    def test_ab_count_positive(self, all_snapshots, date_str):
        """At least 1 A-tier and at least 3 B-tier tickers."""
        _, results = all_snapshots[date_str]
        dev = _dev_results(results)
        a_count = sum(1 for r in dev if r["tier_dev"] == "A")
        b_count = sum(1 for r in dev if r["tier_dev"] == "B")
        assert a_count >= 1, f"{date_str}: A-count={a_count}"
        assert b_count >= 3, f"{date_str}: B-count={b_count}"


# ============================================================================
# 5. TestCoverageMetrics — feed health checks
# ============================================================================
class TestCoverageMetrics:
    """Ensures data feeds haven't collapsed.

    Calibrated from 3 snapshots:
    - Eligibility: 23-24% of dev → bounds [15%, 40%]
    - Catalyst coverage: 24-26% → bounds [15%, 50%]
    - Sponsor coverage: 88-89% → bounds [60%, 100%]
    - Momentum coverage: 100% → bounds [80%, 100%]
    """

    @pytest.mark.parametrize("date_str", DATES)
    def test_eligibility_rate(self, all_snapshots, date_str):
        """Dev eligibility between 15-40%."""
        _, results = all_snapshots[date_str]
        dev = _dev_results(results)
        n_dev = len(dev)
        n_eligible = sum(1 for r in dev if r["eligible"] == "1")
        pct = n_eligible / n_dev * 100 if n_dev else 0
        assert 15 <= pct <= 40, (
            f"{date_str}: dev eligibility = {pct:.1f}% ({n_eligible}/{n_dev})"
        )

    @pytest.mark.parametrize("date_str", DATES)
    def test_catalyst_coverage(self, all_snapshots, date_str):
        """Catalyst coverage (non-missing mode) between 15-50%."""
        _, results = all_snapshots[date_str]
        dev = _dev_results(results)
        n_dev = len(dev)
        n_with = sum(
            1 for r in dev if r.get("catalyst_mode") not in ("missing", "")
        )
        pct = n_with / n_dev * 100 if n_dev else 0
        assert 15 <= pct <= 50, (
            f"{date_str}: catalyst coverage = {pct:.1f}% ({n_with}/{n_dev})"
        )

    @pytest.mark.parametrize("date_str", DATES)
    def test_sponsor_coverage(self, all_snapshots, date_str):
        """Sponsor coverage (tier1_count not blank) between 60-100%."""
        _, results = all_snapshots[date_str]
        dev = _dev_results(results)
        n_dev = len(dev)
        n_with = sum(
            1 for r in dev if r.get("sponsor_tier1_count", "") != ""
        )
        pct = n_with / n_dev * 100 if n_dev else 0
        assert 60 <= pct <= 100, (
            f"{date_str}: sponsor coverage = {pct:.1f}% ({n_with}/{n_dev})"
        )

    @pytest.mark.parametrize("date_str", DATES)
    def test_momentum_coverage(self, all_snapshots, date_str):
        """Momentum coverage (mom_state not blank) between 80-100%."""
        _, results = all_snapshots[date_str]
        dev = _dev_results(results)
        n_dev = len(dev)
        n_with = sum(1 for r in dev if r.get("mom_state", "") != "")
        pct = n_with / n_dev * 100 if n_dev else 0
        assert 80 <= pct <= 100, (
            f"{date_str}: momentum coverage = {pct:.1f}% ({n_with}/{n_dev})"
        )


# ============================================================================
# 6. TestGoldenTickers — exact tier+band pins
# ============================================================================
class TestGoldenTickers:
    """Pinned regressions: specific ticker×date → exact tier and band.

    If any of these change, it means the engine or the fixture data has
    shifted in a way that requires investigation.
    """

    # (date, ticker, expected_tier, expected_band, expected_reason_substring)
    GOLDEN_PINS = [
        # 2025-04-30: A-tier names
        ("2025-04-30", "MRUS", "A", "L", "high_opt+catalyst_near"),
        ("2025-04-30", "LQDA", "A", "L", "high_opt+catalyst_near"),
        # 2025-04-30: B-tier names
        ("2025-04-30", "PVLA", "B", "L", "high_opt+no_catalyst_data"),
        ("2025-04-30", "AKRO", "B", "L", "high_opt+no_catalyst_data"),
        # 2025-07-31: A-tier names
        ("2025-07-31", "ABEO", "A", "L", "high_opt+catalyst_near"),
        ("2025-07-31", "ELDN", "A", "L", "high_opt+catalyst_near"),
        # 2025-07-31: B-tier names
        ("2025-07-31", "BLTE", "B", "M", "mod_opt+catalyst_near"),
        ("2025-07-31", "PVLA", "A", "L", "high_opt+catalyst_mid"),
        # 2025-10-31: A-tier (pinned by test_phase2_portfolio_regression)
        ("2025-10-31", "CNTX", "A", "L", "high_opt+catalyst_near"),
        ("2025-10-31", "KALV", "A", "L", "high_opt+catalyst_near"),
        # 2025-10-31: B-tier
        ("2025-10-31", "ARTV", "B", "L", "high_opt+no_catalyst_data"),
        ("2025-10-31", "PVLA", "B", "L", "mod_opt+catalyst_near"),
    ]

    @pytest.mark.parametrize(
        "date_str,ticker,exp_tier,exp_band,exp_reason",
        GOLDEN_PINS,
        ids=[f"{d}_{t}" for d, t, *_ in GOLDEN_PINS],
    )
    def test_golden_pin(
        self,
        all_snapshots,
        date_str: str,
        ticker: str,
        exp_tier: str,
        exp_band: str,
        exp_reason: str,
    ):
        _, results = all_snapshots[date_str]
        match = [r for r in results if r["ticker"] == ticker]
        assert match, f"{date_str}: ticker {ticker} not found in results"
        r = match[0]
        assert r["tier_dev"] == exp_tier, (
            f"{date_str}/{ticker}: tier={r['tier_dev']!r}, expected={exp_tier!r}"
        )
        assert r["size_band"] == exp_band, (
            f"{date_str}/{ticker}: band={r['size_band']!r}, expected={exp_band!r}"
        )
        assert r["tier_reason"] == exp_reason, (
            f"{date_str}/{ticker}: reason={r['tier_reason']!r}, "
            f"expected={exp_reason!r}"
        )


# ============================================================================
# 7. TestRankStability — cross-date overlap and swing
# ============================================================================
class TestRankStability:
    """Cross-date consistency checks.

    The top tier-A+B names shouldn't completely churn between adjacent 3-month
    snapshots. Eligibility counts shouldn't swing wildly.
    """

    def _ab_set(self, results: List[Dict[str, Any]]) -> set:
        """Return set of A+B tickers."""
        return {
            r["ticker"]
            for r in _dev_results(results)
            if r["tier_dev"] in ("A", "B")
        }

    def _top_n(self, results: List[Dict[str, Any]], n: int) -> set:
        """Return top-N tickers by tier priority (A first, then B)."""
        tier_order = {"A": 0, "B": 1, "C": 2, "D": 3}
        band_order = {"L": 0, "M": 1, "S": 2, "XS": 3}
        dev = _dev_results(results)
        dev.sort(
            key=lambda r: (
                tier_order.get(r["tier_dev"], 9),
                band_order.get(r["size_band"], 9),
            )
        )
        return {r["ticker"] for r in dev[:n]}

    def test_top25_overlap_apr_jul(self, all_snapshots):
        """Top-25 overlap between Apr and Jul >= 30%."""
        _, results_apr = all_snapshots["2025-04-30"]
        _, results_jul = all_snapshots["2025-07-31"]
        top_apr = self._top_n(results_apr, 25)
        top_jul = self._top_n(results_jul, 25)
        overlap = len(top_apr & top_jul)
        pct = overlap / 25 * 100
        assert pct >= 30, f"Apr→Jul top-25 overlap={pct:.0f}% < 30%"

    def test_top25_overlap_jul_oct(self, all_snapshots):
        """Top-25 overlap between Jul and Oct >= 30%."""
        _, results_jul = all_snapshots["2025-07-31"]
        _, results_oct = all_snapshots["2025-10-31"]
        top_jul = self._top_n(results_jul, 25)
        top_oct = self._top_n(results_oct, 25)
        overlap = len(top_jul & top_oct)
        pct = overlap / 25 * 100
        assert pct >= 30, f"Jul→Oct top-25 overlap={pct:.0f}% < 30%"

    def test_top50_overlap_apr_oct(self, all_snapshots):
        """Top-50 overlap between Apr and Oct >= 40%."""
        _, results_apr = all_snapshots["2025-04-30"]
        _, results_oct = all_snapshots["2025-10-31"]
        top_apr = self._top_n(results_apr, 50)
        top_oct = self._top_n(results_oct, 50)
        overlap = len(top_apr & top_oct)
        pct = overlap / 50 * 100
        assert pct >= 40, f"Apr→Oct top-50 overlap={pct:.0f}% < 40%"

    def test_ab_swing_bounded(self, all_snapshots):
        """A+B count doesn't swing more than 20pp between adjacent dates."""
        counts = []
        for date_str in DATES:
            _, results = all_snapshots[date_str]
            dev = _dev_results(results)
            n_dev = len(dev)
            n_ab = sum(1 for r in dev if r["tier_dev"] in ("A", "B"))
            counts.append(n_ab / n_dev * 100 if n_dev else 0)

        for i in range(len(counts) - 1):
            swing = abs(counts[i + 1] - counts[i])
            assert swing <= 20, (
                f"{DATES[i]}→{DATES[i+1]}: A+B swing = {swing:.1f}pp > 20pp"
            )

    def test_eligible_count_ratio_bounded(self, all_snapshots):
        """Eligible counts across dates shouldn't diverge more than 1.5x."""
        eligible_counts = []
        for date_str in DATES:
            _, results = all_snapshots[date_str]
            n = sum(1 for r in results if r["eligible"] == "1")
            eligible_counts.append(n)

        min_c = min(eligible_counts)
        max_c = max(eligible_counts)
        assert min_c > 0, "At least one date has 0 eligible tickers"
        ratio = max_c / min_c
        assert ratio <= 1.5, (
            f"Eligible count ratio = {ratio:.2f} > 1.5 "
            f"(min={min_c}, max={max_c})"
        )
