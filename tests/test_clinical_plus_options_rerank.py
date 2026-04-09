"""Synthetic rerank regression: clinical_plus_options mode.

Proves that candidate ruleset 6bc25923 (clinical_plus_options) changes
ONLY the intended REGULATORY names via options_quality_composite while
leaving CLINICAL names on the clinical_quality path — identical to active
ruleset 7177a4ea for that family.

This is a pure-unit proof that the live ranking path is wired correctly,
independent of real data accumulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import DecisionRuleset, _build_sort_contributions, compute_actionable_sort_key

# ---------------------------------------------------------------------------
# Active and candidate rulesets (mirroring production JSON)
# ---------------------------------------------------------------------------

ACTIVE_RS = DecisionRuleset(
    binary_91_180_sort_mode="clinical_quality",
    binary_91_180_clinical_quality_weight=0.5,
    binary_91_180_options_quality_weight=0.0,
    binary_91_180_flatten_tier_sort=True,
    sort_anchor="optionality_pct",
    enable_calendar_alpha_sort=True,
    calendar_alpha_sort_weight=0.3,
    enable_institutional_sort_signal=True,
    institutional_sort_weight=0.3,
    enable_clinical_sort_signal=False,
    enable_coinvest_sort_signal=False,
    catalyst_priority_mode="tiebreaker",
    rebalance_buffer_ranks=30,
)

CANDIDATE_RS = DecisionRuleset(
    binary_91_180_sort_mode="clinical_plus_options",
    binary_91_180_clinical_quality_weight=0.5,
    binary_91_180_options_quality_weight=0.5,
    binary_91_180_flatten_tier_sort=True,
    sort_anchor="optionality_pct",
    enable_calendar_alpha_sort=True,
    calendar_alpha_sort_weight=0.3,
    enable_institutional_sort_signal=True,
    institutional_sort_weight=0.3,
    enable_clinical_sort_signal=False,
    enable_coinvest_sort_signal=False,
    catalyst_priority_mode="tiebreaker",
    rebalance_buffer_ranks=30,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fields(
    ticker="TEST",
    family="CLINICAL",
    bucket="less_binary",
    clinical_composite="0.70",
    options_composite="0.60",
    optionality=50.0,
    regulatory_days="",
    has_regulatory_180d="0",
    regulatory_event_type="",
    **extra,
):
    """Build a minimal decision_fields dict for sort key computation."""
    d = {
        "ticker": ticker,
        "eligible": "1",
        "tier_dev": "B",
        "catalyst_mode": "specific_days",
        "catalyst_days": "120",
        "catalyst_bucket": bucket,
        "catalyst_family": family,
        "stage_bucket": "mid",
        "mom_state": "neutral",
        "sponsor_tier1_count": "0",
        "clinical_quality_composite": clinical_composite,
        "options_quality_composite": options_composite,
        "binary_quality_score": "0.5",
        "clinical_score_z_tier": "0",
        "inst_delta_z": "0",
        "clinical_score_v2_z": "0",
        "alpha_cohort_pct": "0",
        "clinical_optionality_pct_dev": str(optionality),
        "regulatory_days": regulatory_days,
        "has_regulatory_upcoming_180d": has_regulatory_180d,
        "regulatory_event_type": regulatory_event_type,
    }
    d.update(extra)
    return d


def _sort_key(fields, ruleset, optionality=50.0):
    return compute_actionable_sort_key(
        decision_fields=fields,
        archetype="drug_developer",
        optionality=optionality,
        composite_rank=100,
        ticker=fields["ticker"],
        ruleset=ruleset,
        tiebreaker_pct=optionality,
        alpha_raw=0.0,
    )


def _contribs(fields, ruleset):
    return _build_sort_contributions(fields, ruleset, alpha_raw=0.0, catalyst_bonus=0.0)


def _contrib_names(contribs):
    return [c.name for c in contribs]


def _contrib_delta(contribs, name):
    matches = [c for c in contribs if c.name == name]
    return float(matches[0].delta) if matches else None


# ---------------------------------------------------------------------------
# 1. CLINICAL family: identical under both rulesets
# ---------------------------------------------------------------------------


class TestClinicalFamilyIdentical:
    """CLINICAL names use clinical_quality_91_180 under both rulesets.
    Active (clinical_quality mode) and candidate (clinical_plus_options)
    must produce the SAME contribution for CLINICAL family."""

    def test_clinical_quality_fires_under_active(self):
        fields = _make_fields(family="CLINICAL", clinical_composite="0.80")
        contribs = _contribs(fields, ACTIVE_RS)
        assert "clinical_quality_91_180" in _contrib_names(contribs)
        assert _contrib_delta(contribs, "clinical_quality_91_180") == pytest.approx(0.40)

    def test_clinical_quality_fires_under_candidate(self):
        fields = _make_fields(family="CLINICAL", clinical_composite="0.80")
        contribs = _contribs(fields, CANDIDATE_RS)
        assert "clinical_quality_91_180" in _contrib_names(contribs)
        assert _contrib_delta(contribs, "clinical_quality_91_180") == pytest.approx(0.40)

    def test_options_quality_does_not_fire_for_clinical(self):
        """Neither ruleset should add options_quality for CLINICAL family."""
        fields = _make_fields(family="CLINICAL", options_composite="0.90")
        for rs in [ACTIVE_RS, CANDIDATE_RS]:
            contribs = _contribs(fields, rs)
            assert "options_quality_91_180" not in _contrib_names(contribs)

    def test_clinical_sort_key_identical(self):
        """Full sort key for CLINICAL family is the same under both rulesets."""
        fields = _make_fields(
            ticker="CLIN1",
            family="CLINICAL",
            clinical_composite="0.75",
            options_composite="0.60",
        )
        key_active = _sort_key(fields, ACTIVE_RS)
        key_candidate = _sort_key(fields, CANDIDATE_RS)
        assert key_active == key_candidate


# ---------------------------------------------------------------------------
# 2. REGULATORY family: candidate adds options_quality, active does not
# ---------------------------------------------------------------------------


class TestRegulatoryFamilyDivergence:
    """REGULATORY names should get options_quality_91_180 under candidate
    but NOT under active. This is the intended divergence."""

    def test_active_no_options_contribution(self):
        fields = _make_fields(
            family="REGULATORY",
            options_composite="0.80",
            regulatory_days="120",
            has_regulatory_180d="1",
            regulatory_event_type="PDUFA",
        )
        contribs = _contribs(fields, ACTIVE_RS)
        assert "options_quality_91_180" not in _contrib_names(contribs)

    def test_candidate_adds_options_contribution(self):
        fields = _make_fields(
            family="REGULATORY",
            options_composite="0.80",
            regulatory_days="120",
            has_regulatory_180d="1",
            regulatory_event_type="PDUFA",
        )
        contribs = _contribs(fields, CANDIDATE_RS)
        assert "options_quality_91_180" in _contrib_names(contribs)
        assert _contrib_delta(contribs, "options_quality_91_180") == pytest.approx(0.40)

    def test_clinical_quality_does_not_fire_for_regulatory(self):
        """Neither ruleset should add clinical_quality for REGULATORY family."""
        fields = _make_fields(
            family="REGULATORY",
            clinical_composite="0.90",
            regulatory_days="120",
            has_regulatory_180d="1",
            regulatory_event_type="PDUFA",
        )
        for rs in [ACTIVE_RS, CANDIDATE_RS]:
            contribs = _contribs(fields, rs)
            assert "clinical_quality_91_180" not in _contrib_names(contribs)

    def test_sort_key_diverges_with_options_data(self):
        """Candidate sorts REGULATORY names differently when OQC > 0."""
        fields = _make_fields(
            ticker="REG1",
            family="REGULATORY",
            options_composite="0.70",
            regulatory_days="120",
            has_regulatory_180d="1",
            regulatory_event_type="PDUFA",
        )
        key_active = _sort_key(fields, ACTIVE_RS)
        key_candidate = _sort_key(fields, CANDIDATE_RS)
        assert key_active != key_candidate
        # Candidate should rank this name HIGHER (lower sort key anchor)
        assert key_candidate < key_active

    def test_sort_key_identical_when_oqc_zero(self):
        """If options_quality_composite is 0, candidate = active."""
        fields = _make_fields(
            ticker="REG2",
            family="REGULATORY",
            options_composite="0.0",
            regulatory_days="120",
            has_regulatory_180d="1",
            regulatory_event_type="PDUFA",
        )
        key_active = _sort_key(fields, ACTIVE_RS)
        key_candidate = _sort_key(fields, CANDIDATE_RS)
        assert key_active == key_candidate

    def test_sort_key_identical_when_oqc_empty(self):
        """If options_quality_composite is empty string, candidate = active."""
        fields = _make_fields(
            ticker="REG3",
            family="REGULATORY",
            options_composite="",
            regulatory_days="120",
            has_regulatory_180d="1",
            regulatory_event_type="PDUFA",
        )
        key_active = _sort_key(fields, ACTIVE_RS)
        key_candidate = _sort_key(fields, CANDIDATE_RS)
        assert key_active == key_candidate


# ---------------------------------------------------------------------------
# 3. Non-less_binary buckets: no change under either ruleset
# ---------------------------------------------------------------------------


class TestNonLessBinaryUnchanged:
    """Names outside less_binary should never get clinical_quality or
    options_quality contributions, regardless of ruleset."""

    @pytest.mark.parametrize("bucket", ["binary_now", "build_window", "core"])
    def test_no_contributions_outside_less_binary(self, bucket):
        for family in ["CLINICAL", "REGULATORY"]:
            fields = _make_fields(
                family=family,
                bucket=bucket,
                clinical_composite="0.80",
                options_composite="0.80",
            )
            for rs in [ACTIVE_RS, CANDIDATE_RS]:
                contribs = _contribs(fields, rs)
                names = _contrib_names(contribs)
                assert "clinical_quality_91_180" not in names
                assert "options_quality_91_180" not in names


# ---------------------------------------------------------------------------
# 4. Multi-name ranking: candidate reorders REGULATORY, preserves CLINICAL
# ---------------------------------------------------------------------------


class TestMultiNameReranking:
    """Simulate a mixed CLINICAL+REGULATORY cohort in less_binary and verify
    the candidate changes REGULATORY ordering without touching CLINICAL."""

    def _build_cohort(self):
        return [
            _make_fields(
                ticker="CLIN_HIGH",
                family="CLINICAL",
                clinical_composite="0.90",
                options_composite="0.10",
                optionality=60.0,
            ),
            _make_fields(
                ticker="CLIN_LOW",
                family="CLINICAL",
                clinical_composite="0.30",
                options_composite="0.90",
                optionality=55.0,
            ),
            _make_fields(
                ticker="REG_HIGH_OQC",
                family="REGULATORY",
                clinical_composite="0.10",
                options_composite="0.85",
                optionality=50.0,
                regulatory_days="120",
                has_regulatory_180d="1",
                regulatory_event_type="PDUFA",
            ),
            _make_fields(
                ticker="REG_LOW_OQC",
                family="REGULATORY",
                clinical_composite="0.10",
                options_composite="0.10",
                optionality=50.2,
                regulatory_days="120",
                has_regulatory_180d="1",
                regulatory_event_type="PDUFA",
            ),
        ]

    def _rank(self, cohort, ruleset):
        keyed = [(f, _sort_key(f, ruleset, float(f["clinical_optionality_pct_dev"]))) for f in cohort]
        keyed.sort(key=lambda x: x[1])
        return [f["ticker"] for f, _ in keyed]

    def test_clinical_order_preserved(self):
        cohort = self._build_cohort()
        active_order = self._rank(cohort, ACTIVE_RS)
        cand_order = self._rank(cohort, CANDIDATE_RS)

        # Extract CLINICAL-only ordering from each
        clin_active = [t for t in active_order if t.startswith("CLIN_")]
        clin_cand = [t for t in cand_order if t.startswith("CLIN_")]
        assert clin_active == clin_cand

    def test_regulatory_order_changes(self):
        cohort = self._build_cohort()
        active_order = self._rank(cohort, ACTIVE_RS)
        cand_order = self._rank(cohort, CANDIDATE_RS)

        # Under active: REGULATORY order is pure optionality (REG_LOW_OQC slightly ahead)
        reg_active = [t for t in active_order if t.startswith("REG_")]
        # Under candidate: REG_HIGH_OQC gets +0.5*0.85 = 0.425 bonus, overcoming the gap
        reg_cand = [t for t in cand_order if t.startswith("REG_")]
        assert reg_active != reg_cand
        # REG_HIGH_OQC should rank higher under candidate
        assert reg_cand.index("REG_HIGH_OQC") < reg_cand.index("REG_LOW_OQC")


# ---------------------------------------------------------------------------
# 5. Weight sensitivity
# ---------------------------------------------------------------------------


class TestWeightSensitivity:
    """Verify that the options_quality_91_180 contribution scales linearly
    with the weight parameter."""

    @pytest.mark.parametrize(
        "weight,expected_delta",
        [
            (0.0, 0.0),
            (0.25, 0.175),  # 0.25 * 0.70
            (0.5, 0.35),  # 0.5 * 0.70
            (1.0, 0.70),  # 1.0 * 0.70
        ],
    )
    def test_weight_scales_linearly(self, weight, expected_delta):
        rs = DecisionRuleset(
            binary_91_180_sort_mode="clinical_plus_options",
            binary_91_180_options_quality_weight=weight,
        )
        fields = _make_fields(
            family="REGULATORY",
            options_composite="0.70",
            regulatory_days="120",
            has_regulatory_180d="1",
            regulatory_event_type="PDUFA",
        )
        contribs = _contribs(fields, rs)
        if weight == 0.0:
            oq = [c for c in contribs if c.name == "options_quality_91_180"]
            assert len(oq) == 1
            assert oq[0].delta == 0.0
        else:
            delta = _contrib_delta(contribs, "options_quality_91_180")
            assert delta == pytest.approx(expected_delta)


# ---------------------------------------------------------------------------
# 6. Secondary regulatory path: PDUFA behind a closer clinical primary
# ---------------------------------------------------------------------------


class TestSecondaryRegulatoryPath:
    """Step 10 should fire based on the secondary regulatory fields
    (regulatory_days, has_regulatory_upcoming_180d), NOT the primary
    catalyst_family/catalyst_bucket. This covers the VERA-like scenario
    where a PDUFA at 116d is masked by a CTGov event at 2d."""

    def test_clinical_primary_with_pdufa_secondary(self):
        """Primary is CLINICAL/binary_now, but PDUFA at 116d via secondary path."""
        fields = _make_fields(
            ticker="VERA",
            family="CLINICAL",
            bucket="binary_now",
            options_composite="0.70",
            clinical_composite="0.50",
            regulatory_days="116",
            has_regulatory_180d="1",
            regulatory_event_type="PDUFA",
        )
        contribs = _contribs(fields, CANDIDATE_RS)
        assert "options_quality_91_180" in _contrib_names(contribs)
        assert _contrib_delta(contribs, "options_quality_91_180") == pytest.approx(0.35)

    def test_pdufa_outside_less_binary_window_no_fire(self):
        """regulatory_days=30 is inside build_window, not less_binary."""
        fields = _make_fields(
            family="REGULATORY",
            bucket="binary_now",
            regulatory_days="30",
            has_regulatory_180d="1",
            regulatory_event_type="PDUFA",
        )
        contribs = _contribs(fields, CANDIDATE_RS)
        assert "options_quality_91_180" not in _contrib_names(contribs)

    def test_pdufa_at_boundary_91_fires(self):
        """regulatory_days=91 is > 90, should fire."""
        fields = _make_fields(
            regulatory_days="91",
            has_regulatory_180d="1",
            regulatory_event_type="PDUFA",
        )
        contribs = _contribs(fields, CANDIDATE_RS)
        assert "options_quality_91_180" in _contrib_names(contribs)

    def test_pdufa_at_boundary_180_fires(self):
        """regulatory_days=180 is <= 180, should fire."""
        fields = _make_fields(
            regulatory_days="180",
            has_regulatory_180d="1",
            regulatory_event_type="PDUFA",
        )
        contribs = _contribs(fields, CANDIDATE_RS)
        assert "options_quality_91_180" in _contrib_names(contribs)

    def test_pdufa_at_181_no_fire(self):
        """regulatory_days=181 is > 180, should NOT fire."""
        fields = _make_fields(
            regulatory_days="181",
            has_regulatory_180d="1",
            regulatory_event_type="PDUFA",
        )
        contribs = _contribs(fields, CANDIDATE_RS)
        assert "options_quality_91_180" not in _contrib_names(contribs)

    def test_no_regulatory_flag_no_fire(self):
        """has_regulatory_upcoming_180d=0 blocks Step 10 even with valid days."""
        fields = _make_fields(
            regulatory_days="120",
            has_regulatory_180d="0",
            regulatory_event_type="PDUFA",
        )
        contribs = _contribs(fields, CANDIDATE_RS)
        assert "options_quality_91_180" not in _contrib_names(contribs)

    def test_active_ruleset_never_fires(self):
        """Active ruleset (clinical_quality mode) never produces
        options_quality_91_180 regardless of secondary fields."""
        fields = _make_fields(
            family="CLINICAL",
            bucket="binary_now",
            regulatory_days="116",
            has_regulatory_180d="1",
            regulatory_event_type="PDUFA",
            options_composite="0.80",
        )
        contribs = _contribs(fields, ACTIVE_RS)
        assert "options_quality_91_180" not in _contrib_names(contribs)
