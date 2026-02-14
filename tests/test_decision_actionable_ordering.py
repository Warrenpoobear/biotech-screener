"""Tests for Decision Engine Phase 2 — Actionable ordering and target weights."""

import pytest
from decision_engine import (
    compute_actionable_sort_key,
    compute_target_weights,
    resolve_catalyst_priority,
    DecisionRuleset,
    SIZING_WEIGHTS,
    ACTIONABLE_COLUMNS,
)


def _make_fields(
    eligible="1",
    tier_dev="B",
    catalyst_mode="specific_days",
    catalyst_days=60,
    mom_state="neutral",
    sponsor_tier1_count=0,
    size_band="M",
    risk_flags="",
    ineligible_reasons="",
):
    """Build a minimal decision_fields dict for testing."""
    return {
        "eligible": eligible,
        "tier_dev": tier_dev,
        "catalyst_mode": catalyst_mode,
        "catalyst_days": catalyst_days,
        "mom_state": mom_state,
        "sponsor_tier1_count": sponsor_tier1_count,
        "size_band": size_band,
        "risk_flags": risk_flags,
        "ineligible_reasons": ineligible_reasons,
    }


def _sort_key(fields, archetype="drug_developer", optionality=0.50,
              composite_rank=100, ticker="TEST", **kwargs):
    return compute_actionable_sort_key(
        decision_fields=fields,
        archetype=archetype,
        optionality=optionality,
        composite_rank=composite_rank,
        ticker=ticker,
        **kwargs,
    )


# ── Test 1: Ordering stability with 10 synthetic tickers ──

def test_ordering_stability_10_tickers():
    """10 synthetic tickers with known fields produce exact expected order."""
    tickers = [
        # (ticker, archetype, fields_kwargs, optionality, composite_rank)
        ("ALPHA", "drug_developer", dict(eligible="1", tier_dev="A", catalyst_mode="specific_days", catalyst_days=30), 0.80, 5),
        ("BRAVO", "drug_developer", dict(eligible="1", tier_dev="A", catalyst_mode="specific_days", catalyst_days=90), 0.70, 10),
        ("CHARLIE", "drug_developer", dict(eligible="1", tier_dev="B", catalyst_mode="blended_window", catalyst_days=0), 0.50, 15),
        ("DELTA", "drug_developer", dict(eligible="1", tier_dev="B", catalyst_mode="no_upcoming", catalyst_days=0), 0.40, 20),
        ("ECHO", "drug_developer", dict(eligible="1", tier_dev="C", catalyst_mode="missing", catalyst_days=""), 0.20, 25),
        ("FOXTROT", "commercial_biotech", dict(eligible="1", tier_dev="", catalyst_mode="missing", catalyst_days=""), None, 2),
        ("GOLF", "commercial_biotech", dict(eligible="1", tier_dev="", catalyst_mode="missing", catalyst_days=""), None, 8),
        ("HOTEL", "drug_developer", dict(eligible="0", tier_dev="D", catalyst_mode="missing", catalyst_days=""), 0.10, 50),
        ("INDIA", "drug_developer", dict(eligible="0", tier_dev="D", catalyst_mode="missing", catalyst_days=""), 0.05, 60),
        ("JULIET", "commercial_biotech", dict(eligible="0", tier_dev="", catalyst_mode="missing", catalyst_days=""), None, 3),
    ]

    rows = []
    for ticker, arch, kwargs, opt, cr in tickers:
        fields = _make_fields(**kwargs)
        key = _sort_key(fields, archetype=arch, optionality=opt,
                        composite_rank=cr, ticker=ticker)
        rows.append((ticker, key))

    sorted_rows = sorted(rows, key=lambda x: x[1])
    result_order = [t for t, _ in sorted_rows]

    expected = [
        "ALPHA", "BRAVO",      # A-tier dev, specific_days, days 30 < 90
        "CHARLIE", "DELTA",    # B-tier dev, blended < no_upcoming
        "ECHO",                # C-tier dev
        "FOXTROT", "GOLF",     # eligible non-dev, composite_rank 2 < 8
        "HOTEL", "INDIA",      # ineligible dev
        "JULIET",              # ineligible non-dev
    ]
    assert result_order == expected


# ── Test 2: Tier priority ──

def test_tier_priority_a_above_b():
    """A-tier ticker always ranks above B-tier regardless of other fields."""
    a_fields = _make_fields(tier_dev="A", catalyst_mode="no_upcoming", catalyst_days=0,
                            sponsor_tier1_count=0)
    b_fields = _make_fields(tier_dev="B", catalyst_mode="specific_days", catalyst_days=10,
                            sponsor_tier1_count=5, mom_state="tailwind")

    key_a = _sort_key(a_fields, optionality=0.30, composite_rank=200, ticker="AAA")
    key_b = _sort_key(b_fields, optionality=0.90, composite_rank=1, ticker="BBB")

    assert key_a < key_b


# ── Test 3: Catalyst tiebreak within same tier ──

def test_catalyst_tiebreak_within_tier():
    """specific_days < blended_window < no_upcoming < missing within same tier."""
    modes = ["specific_days", "blended_window", "no_upcoming", "missing"]
    keys = []
    for mode in modes:
        f = _make_fields(tier_dev="B", catalyst_mode=mode, catalyst_days=60 if mode == "specific_days" else "")
        keys.append(_sort_key(f, ticker=mode.upper()))

    for i in range(len(keys) - 1):
        assert keys[i] < keys[i + 1], f"{modes[i]} should sort before {modes[i+1]}"


# ── Test 4: Days tiebreak within catalyst_mode ──

def test_days_tiebreak_within_mode():
    """days=30 sorts before days=90 within specific_days mode."""
    f30 = _make_fields(tier_dev="B", catalyst_mode="specific_days", catalyst_days=30)
    f90 = _make_fields(tier_dev="B", catalyst_mode="specific_days", catalyst_days=90)

    key_30 = _sort_key(f30, ticker="AAA")
    key_90 = _sort_key(f90, ticker="BBB")

    assert key_30 < key_90


# ── Test 5: Optionality tiebreak ──

def test_optionality_tiebreak():
    """Higher optionality wins within same tier+catalyst."""
    f_high = _make_fields(tier_dev="B", catalyst_mode="specific_days", catalyst_days=60)
    f_low = _make_fields(tier_dev="B", catalyst_mode="specific_days", catalyst_days=60)

    key_high = _sort_key(f_high, optionality=0.80, ticker="AAA")
    key_low = _sort_key(f_low, optionality=0.20, ticker="BBB")

    assert key_high < key_low


# ── Test 6: Non-dev after dev ──

def test_nondev_after_dev():
    """commercial_biotech ticker sorts after all drug_developer tickers."""
    dev_fields = _make_fields(tier_dev="D", catalyst_mode="missing")
    comm_fields = _make_fields(tier_dev="", catalyst_mode="missing")

    key_dev = _sort_key(dev_fields, archetype="drug_developer",
                        optionality=0.01, composite_rank=999, ticker="ZZZ")
    key_comm = _sort_key(comm_fields, archetype="commercial_biotech",
                         optionality=0.99, composite_rank=1, ticker="AAA")

    assert key_dev < key_comm


# ── Test 7: Ineligible at bottom ──

def test_ineligible_at_bottom():
    """Ineligible tickers sort last regardless of tier."""
    elig = _make_fields(eligible="1", tier_dev="D", catalyst_mode="missing")
    inelig = _make_fields(eligible="0", tier_dev="A", catalyst_mode="specific_days",
                          catalyst_days=10)

    key_elig = _sort_key(elig, optionality=0.01, composite_rank=999, ticker="ZZZ")
    key_inelig = _sort_key(inelig, optionality=0.99, composite_rank=1, ticker="AAA")

    assert key_elig < key_inelig


# ── Test 8: Target weights sum to 100% ──

def test_target_weights_sum_to_100():
    """Eligible rows' target_weight_pct sums to 100.0."""
    rows = [
        {"size_band": "L", "ticker": "A"},
        {"size_band": "M", "ticker": "B"},
        {"size_band": "S", "ticker": "C"},
        {"size_band": "XS", "ticker": "D"},
        {"size_band": "M", "ticker": "E"},
    ]
    compute_target_weights(rows)

    total = sum(r["target_weight_pct"] for r in rows)
    assert abs(total - 100.0) < 0.05, f"Weights sum to {total}, expected ~100.0"


# ── Test 9: Weight proportionality ──

def test_weight_proportionality():
    """L weight > M weight > S weight > XS weight."""
    rows = [
        {"size_band": "L", "ticker": "A"},
        {"size_band": "M", "ticker": "B"},
        {"size_band": "S", "ticker": "C"},
        {"size_band": "XS", "ticker": "D"},
    ]
    compute_target_weights(rows)

    w_l = rows[0]["target_weight_pct"]
    w_m = rows[1]["target_weight_pct"]
    w_s = rows[2]["target_weight_pct"]
    w_xs = rows[3]["target_weight_pct"]

    assert w_l > w_m > w_s > w_xs


# ── Test 10: Determinism ──

def test_determinism():
    """Sorting same input twice gives identical result."""
    tickers_data = [
        ("C", "drug_developer", dict(tier_dev="B", catalyst_mode="specific_days", catalyst_days=60), 0.50, 10),
        ("A", "drug_developer", dict(tier_dev="A", catalyst_mode="specific_days", catalyst_days=30), 0.80, 5),
        ("B", "drug_developer", dict(tier_dev="B", catalyst_mode="blended_window", catalyst_days=0), 0.40, 15),
    ]

    def do_sort():
        items = []
        for ticker, arch, kwargs, opt, cr in tickers_data:
            fields = _make_fields(**kwargs)
            key = _sort_key(fields, archetype=arch, optionality=opt,
                            composite_rank=cr, ticker=ticker)
            items.append((ticker, key))
        return [t for t, _ in sorted(items, key=lambda x: x[1])]

    result1 = do_sort()
    result2 = do_sort()
    assert result1 == result2


# ── Test 11: Observe mode keeps composite_rank order ──

def test_observe_mode_composite_rank_order():
    """In observe mode the CSV would be sorted by composite_rank.

    This test verifies that actionable_rank is still computed correctly
    even when the final sort is by composite_rank (the re-sort is done
    in run_screen.py, not in the decision_engine; here we verify that
    sort keys are independent of final display order).
    """
    # Two tickers: ticker B has better tier but worse composite_rank
    b_fields = _make_fields(tier_dev="A", catalyst_mode="specific_days", catalyst_days=30)
    a_fields = _make_fields(tier_dev="C", catalyst_mode="missing")

    key_b = _sort_key(b_fields, composite_rank=50, ticker="B")
    key_a = _sort_key(a_fields, composite_rank=10, ticker="A")

    # Actionable: B sorts first (better tier)
    assert key_b < key_a

    # Composite rank: A sorts first (rank 10 < 50)
    composite_order = sorted(
        [("A", 10), ("B", 50)],
        key=lambda x: x[1],
    )
    assert composite_order[0][0] == "A"


# ── Additional: ACTIONABLE_COLUMNS export ──

def test_actionable_columns_export():
    """ACTIONABLE_COLUMNS contains expected column names."""
    assert "actionable_rank" in ACTIONABLE_COLUMNS
    assert "target_weight_pct" in ACTIONABLE_COLUMNS
    assert len(ACTIONABLE_COLUMNS) == 2


# ── Additional: SIZING_WEIGHTS values ──

def test_sizing_weights_values():
    """SIZING_WEIGHTS has expected keys and values."""
    assert SIZING_WEIGHTS == {"L": 1.00, "M": 0.60, "S": 0.30, "XS": 0.15}


# ── Edge case: all same size_band ──

def test_target_weights_uniform_band():
    """When all rows have the same band, weights are equal."""
    rows = [
        {"size_band": "M", "ticker": "A"},
        {"size_band": "M", "ticker": "B"},
        {"size_band": "M", "ticker": "C"},
    ]
    compute_target_weights(rows)

    for r in rows:
        assert abs(r["target_weight_pct"] - 100.0 / 3) < 0.1


# ── Edge case: empty eligible list ──

def test_target_weights_empty_list():
    """Empty list returns empty list."""
    result = compute_target_weights([])
    assert result == []


# ── Edge case: missing optionality ──

def test_sort_key_missing_optionality():
    """None optionality uses 0.0 fallback (sorts after real values)."""
    f = _make_fields(tier_dev="B", catalyst_mode="specific_days", catalyst_days=60)

    key_with = _sort_key(f, optionality=0.50, ticker="AAA")
    key_without = _sort_key(f, optionality=None, ticker="AAA")

    # With real optionality (0.50 → negated = -0.50) sorts before None (→ 0.0)
    assert key_with < key_without


# ── Catalyst source/type priority tests ──

def _priority_ruleset():
    """Ruleset with catalyst priority enabled (tiebreaker mode)."""
    return DecisionRuleset(catalyst_priority_mode="tiebreaker")


def _disabled_ruleset():
    """Ruleset with catalyst priority disabled (off mode, default)."""
    return DecisionRuleset(catalyst_priority_mode="off")


# -- resolve_catalyst_priority unit tests --

class TestResolveCatalystPriority:
    """Unit tests for resolve_catalyst_priority()."""

    def test_fda_decision_fda_calendar(self):
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("FDA_DECISION", "FDA_CALENDAR", rs) == 1

    def test_fda_adcom_fda_calendar(self):
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("FDA_ADCOM", "FDA_CALENDAR", rs) == 1

    def test_data_readout_ctgov(self):
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("DATA_READOUT", "CTGOV_CALENDAR", rs) == 2

    def test_data_readout_wildcard_source(self):
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("DATA_READOUT", "SEC_8K_FILING", rs) == 2

    def test_trial_ongoing_any_source(self):
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("TRIAL_ONGOING", "SOME_SOURCE", rs) == 3

    def test_any_type_corporate_calendar(self):
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("SOME_TYPE", "CORPORATE_CALENDAR", rs) == 3

    def test_any_type_sec_8k(self):
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("SOME_TYPE", "SEC_8K_FILING", rs) == 2

    def test_no_catalyst_returns_default(self):
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("", "", rs) == 9
        assert resolve_catalyst_priority("none", "none", rs) == 9
        assert resolve_catalyst_priority(None, None, rs) == 9

    def test_unknown_returns_unknown_priority(self):
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("unknown", "FDA_CALENDAR", rs) == 99
        assert resolve_catalyst_priority("DATA_READOUT", "unknown", rs) == 99

    def test_unmatched_rule_returns_default(self):
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("NOVEL_TYPE", "NOVEL_SOURCE", rs) == 9

    def test_case_insensitive(self):
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("fda_decision", "fda_calendar", rs) == 1
        assert resolve_catalyst_priority("data_readout", "ctgov_calendar", rs) == 2

    # -- Expanded FDA type coverage (v1.3.1 priority map) --

    def test_fda_pdufa_date_sec_8k(self):
        """FDA_PDUFA_DATE from SEC_8K_FILING → priority 1 (not wildcard 2)."""
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("FDA_PDUFA_DATE", "SEC_8K_FILING", rs) == 1

    def test_fda_adcom_sec_8k(self):
        """FDA_ADCOM from SEC_8K_FILING → priority 1."""
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("FDA_ADCOM", "SEC_8K_FILING", rs) == 1

    def test_fda_crl_any_source(self):
        """FDA_CRL from any source → priority 1."""
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("FDA_CRL", "SEC_8K_FILING", rs) == 1
        assert resolve_catalyst_priority("FDA_CRL", "FDA_CALENDAR", rs) == 1

    def test_fda_rtf_any_source(self):
        """FDA_RTF from any source → priority 1."""
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("FDA_RTF", "SEC_8K_FILING", rs) == 1

    def test_fda_warning_letter_any_source(self):
        """FDA_WARNING_LETTER from any source → priority 1."""
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("FDA_WARNING_LETTER", "SEC_8K_FILING", rs) == 1

    def test_fda_approval_any_source(self):
        """FDA_APPROVAL from any source → priority 1."""
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("FDA_APPROVAL", "CORPORATE_CALENDAR", rs) == 1

    def test_data_readout_sec_8k_still_priority_2(self):
        """DATA_READOUT from SEC_8K_FILING → priority 2 (unchanged)."""
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("DATA_READOUT", "SEC_8K_FILING", rs) == 2

    # -- New multi-form SEC sources (v1.3.1 priority map) --

    def test_sec_10q_wildcard_type_priority_2(self):
        """Any event_type from SEC_10Q_FILING → priority 2."""
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("SOME_TYPE", "SEC_10Q_FILING", rs) == 2

    def test_sec_10k_wildcard_type_priority_2(self):
        """Any event_type from SEC_10K_FILING → priority 2."""
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("SOME_TYPE", "SEC_10K_FILING", rs) == 2

    def test_sec_6k_wildcard_type_priority_2(self):
        """Any event_type from SEC_6K_FILING → priority 2."""
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("SOME_TYPE", "SEC_6K_FILING", rs) == 2

    def test_federal_register_wildcard_type_priority_3(self):
        """Any event_type from FEDERAL_REGISTER → priority 3 (demoted: procedural notices)."""
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("SOME_TYPE", "FEDERAL_REGISTER", rs) == 3

    def test_fda_approval_from_federal_register_priority_1(self):
        """FDA_APPROVAL from FEDERAL_REGISTER → matches FDA_APPROVAL event rule first → 1."""
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("FDA_APPROVAL", "FEDERAL_REGISTER", rs) == 1

    def test_data_readout_from_10q_priority_2(self):
        """DATA_READOUT from SEC_10Q_FILING → matches DATA_READOUT wildcard → 2."""
        rs = _priority_ruleset()
        assert resolve_catalyst_priority("DATA_READOUT", "SEC_10Q_FILING", rs) == 2


# -- Sort key integration tests with catalyst priority --

class TestCatalystPriorityOrdering:
    """Integration tests: catalyst priority in sort key."""

    def test_fda_beats_ctgov_same_tier(self):
        """FDA event sorts before CTGOV readout within same tier."""
        rs = _priority_ruleset()
        f_fda = _make_fields(tier_dev="A", catalyst_mode="specific_days", catalyst_days=60)
        f_ctgov = _make_fields(tier_dev="A", catalyst_mode="specific_days", catalyst_days=60)

        key_fda = _sort_key(f_fda, ticker="FDA_TICK",
                            catalyst_event_type="FDA_DECISION",
                            catalyst_source="FDA_CALENDAR", ruleset=rs)
        key_ctgov = _sort_key(f_ctgov, ticker="CTG_TICK",
                              catalyst_event_type="DATA_READOUT",
                              catalyst_source="CTGOV_CALENDAR", ruleset=rs)
        assert key_fda < key_ctgov

    def test_unknown_sinks_below_dated(self):
        """Unknown source/type sorts after dated catalyst at same tier."""
        rs = _priority_ruleset()
        f_dated = _make_fields(tier_dev="B", catalyst_mode="specific_days", catalyst_days=60)
        f_unknown = _make_fields(tier_dev="B", catalyst_mode="specific_days", catalyst_days=60)

        key_dated = _sort_key(f_dated, ticker="DATED",
                              catalyst_event_type="DATA_READOUT",
                              catalyst_source="CTGOV_CALENDAR", ruleset=rs)
        key_unknown = _sort_key(f_unknown, ticker="UNK",
                                catalyst_event_type="unknown",
                                catalyst_source="unknown", ruleset=rs)
        assert key_dated < key_unknown

    def test_no_catalyst_below_dated(self):
        """No catalyst (empty type/source) sorts after dated catalyst."""
        rs = _priority_ruleset()
        f_dated = _make_fields(tier_dev="B", catalyst_mode="specific_days", catalyst_days=60)
        f_none = _make_fields(tier_dev="B", catalyst_mode="specific_days", catalyst_days=60)

        key_dated = _sort_key(f_dated, ticker="DATED",
                              catalyst_event_type="DATA_READOUT",
                              catalyst_source="CTGOV_CALENDAR", ruleset=rs)
        key_none = _sort_key(f_none, ticker="NONE_TICK",
                             catalyst_event_type="", catalyst_source="",
                             ruleset=rs)
        assert key_dated < key_none

    def test_disabled_no_effect_on_ordering(self):
        """When disabled, catalyst priority does not change relative order."""
        rs = _disabled_ruleset()
        f1 = _make_fields(tier_dev="B", catalyst_mode="specific_days", catalyst_days=60)
        f2 = _make_fields(tier_dev="B", catalyst_mode="specific_days", catalyst_days=60)

        key_fda = _sort_key(f1, ticker="FDA_TICK",
                            catalyst_event_type="FDA_DECISION",
                            catalyst_source="FDA_CALENDAR", ruleset=rs)
        key_ctgov = _sort_key(f2, ticker="CTG_TICK",
                              catalyst_event_type="DATA_READOUT",
                              catalyst_source="CTGOV_CALENDAR", ruleset=rs)
        # cat_priority is 0 for both when disabled — order falls through
        # to cat_mode_ord (same) → cat_days (same) → ticker alpha
        assert key_ctgov < key_fda  # CTG_TICK < FDA_TICK alphabetically

    def test_tier_still_dominates_over_priority(self):
        """Tier ordering still takes precedence over catalyst priority."""
        rs = _priority_ruleset()
        f_a = _make_fields(tier_dev="A", catalyst_mode="specific_days", catalyst_days=60)
        f_b = _make_fields(tier_dev="B", catalyst_mode="specific_days", catalyst_days=60)

        # B-tier has FDA (priority=1), A-tier has generic (priority=9)
        key_a = _sort_key(f_a, ticker="A_TICK",
                          catalyst_event_type="", catalyst_source="",
                          ruleset=rs)
        key_b = _sort_key(f_b, ticker="B_TICK",
                          catalyst_event_type="FDA_DECISION",
                          catalyst_source="FDA_CALENDAR", ruleset=rs)
        assert key_a < key_b  # A-tier wins despite worse catalyst priority


# ── Catalyst Priority Mode tests ──

def _mode_ruleset(mode, **kwargs):
    """Build a ruleset with a specific catalyst_priority_mode."""
    return DecisionRuleset(catalyst_priority_mode=mode, **kwargs)


class TestCatalystPriorityModes:
    """Tests for off / tiebreaker / blended catalyst priority modes."""

    def test_off_mode_identical_to_disabled(self):
        """mode='off' produces same ordering as enable_catalyst_priority=False."""
        rs_off = _mode_ruleset("off")
        rs_disabled = DecisionRuleset(enable_catalyst_priority=False)

        f = _make_fields(tier_dev="A", catalyst_mode="specific_days", catalyst_days=60)

        key_off = _sort_key(f, ticker="X", composite_rank=10,
                            catalyst_event_type="FDA_DECISION",
                            catalyst_source="FDA_CALENDAR", ruleset=rs_off)
        key_dis = _sort_key(f, ticker="X", composite_rank=10,
                            catalyst_event_type="FDA_DECISION",
                            catalyst_source="FDA_CALENDAR", ruleset=rs_disabled)
        assert key_off == key_dis

    def test_tiebreaker_comp_rank_dominates(self):
        """In tiebreaker mode, FDA at rank 20 sorts AFTER generic at rank 10."""
        rs = _mode_ruleset("tiebreaker")
        f_fda = _make_fields(tier_dev="A", catalyst_mode="specific_days", catalyst_days=60)
        f_gen = _make_fields(tier_dev="A", catalyst_mode="specific_days", catalyst_days=60)

        key_fda = _sort_key(f_fda, ticker="FDA_T", composite_rank=20,
                            catalyst_event_type="FDA_DECISION",
                            catalyst_source="FDA_CALENDAR", ruleset=rs)
        key_gen = _sort_key(f_gen, ticker="GEN_T", composite_rank=10,
                            catalyst_event_type="DATA_READOUT",
                            catalyst_source="CTGOV_CALENDAR", ruleset=rs)
        assert key_gen < key_fda  # rank 10 beats rank 20 despite FDA priority

    def test_tiebreaker_same_rank_fda_wins(self):
        """In tiebreaker mode, FDA at rank 10 sorts before CTGOV at rank 10."""
        rs = _mode_ruleset("tiebreaker")
        f_fda = _make_fields(tier_dev="A", catalyst_mode="specific_days", catalyst_days=60)
        f_ctgov = _make_fields(tier_dev="A", catalyst_mode="specific_days", catalyst_days=60)

        key_fda = _sort_key(f_fda, ticker="FDA_T", composite_rank=10,
                            catalyst_event_type="FDA_DECISION",
                            catalyst_source="FDA_CALENDAR", ruleset=rs)
        key_ctgov = _sort_key(f_ctgov, ticker="CTG_T", composite_rank=10,
                              catalyst_event_type="DATA_READOUT",
                              catalyst_source="CTGOV_CALENDAR", ruleset=rs)
        assert key_fda < key_ctgov  # same rank, FDA priority 1 < CTGOV priority 2

    def test_blended_small_rank_jump(self):
        """In blended mode, FDA at rank 12 (bonus=5 → eff 7) beats generic at rank 10."""
        rs = _mode_ruleset("blended")
        f_fda = _make_fields(tier_dev="A", catalyst_mode="specific_days", catalyst_days=60)
        f_gen = _make_fields(tier_dev="A", catalyst_mode="specific_days", catalyst_days=60)

        key_fda = _sort_key(f_fda, ticker="FDA_T", composite_rank=12,
                            catalyst_event_type="FDA_DECISION",
                            catalyst_source="FDA_CALENDAR", ruleset=rs)
        key_gen = _sort_key(f_gen, ticker="GEN_T", composite_rank=10,
                            catalyst_event_type="", catalyst_source="",
                            ruleset=rs)
        # FDA: effective_rank = 12 - 5 = 7.0; generic: 10 - 0 = 10.0
        assert key_fda < key_gen

    def test_blended_large_gap_no_swap(self):
        """In blended mode, FDA at rank 20 (bonus=5 → eff 15) still after generic rank 10."""
        rs = _mode_ruleset("blended")
        f_fda = _make_fields(tier_dev="A", catalyst_mode="specific_days", catalyst_days=60)
        f_gen = _make_fields(tier_dev="A", catalyst_mode="specific_days", catalyst_days=60)

        key_fda = _sort_key(f_fda, ticker="FDA_T", composite_rank=20,
                            catalyst_event_type="FDA_DECISION",
                            catalyst_source="FDA_CALENDAR", ruleset=rs)
        key_gen = _sort_key(f_gen, ticker="GEN_T", composite_rank=10,
                            catalyst_event_type="", catalyst_source="",
                            ruleset=rs)
        # FDA: effective_rank = 20 - 5 = 15.0; generic: 10 - 0 = 10.0
        assert key_gen < key_fda

    def test_tier_still_dominates_all_modes(self):
        """A-tier always beats B-tier regardless of mode."""
        for mode in ("off", "tiebreaker", "blended"):
            rs = _mode_ruleset(mode)
            f_a = _make_fields(tier_dev="A", catalyst_mode="specific_days", catalyst_days=60)
            f_b = _make_fields(tier_dev="B", catalyst_mode="specific_days", catalyst_days=60)

            key_a = _sort_key(f_a, ticker="A_T", composite_rank=50,
                              catalyst_event_type="", catalyst_source="",
                              ruleset=rs)
            key_b = _sort_key(f_b, ticker="B_T", composite_rank=1,
                              catalyst_event_type="FDA_DECISION",
                              catalyst_source="FDA_CALENDAR", ruleset=rs)
            assert key_a < key_b, f"A-tier should beat B-tier in mode={mode}"

    def test_no_tier_changes_tiebreaker(self):
        """Tiebreaker mode does not change tier assignments — same tier_ord."""
        rs_off = _mode_ruleset("off")
        rs_tb = _mode_ruleset("tiebreaker")
        f = _make_fields(tier_dev="A", catalyst_mode="specific_days", catalyst_days=60)

        key_off = _sort_key(f, ticker="X", composite_rank=10, ruleset=rs_off)
        key_tb = _sort_key(f, ticker="X", composite_rank=10, ruleset=rs_tb)
        # tier_ord is at index 2 in both tuples
        assert key_off[2] == key_tb[2]

    def test_mode_validation_rejects_invalid(self):
        """Invalid catalyst_priority_mode raises ValueError."""
        import pytest as _pytest
        with _pytest.raises(ValueError, match="catalyst_priority_mode"):
            DecisionRuleset(catalyst_priority_mode="invalid")

    def test_from_json_migration_enable_to_tiebreaker(self):
        """Old JSON with enable_catalyst_priority=true migrates to mode='tiebreaker'."""
        import json
        import tempfile
        import os

        # Build a minimal valid ruleset JSON with enable_catalyst_priority=true
        # but NO catalyst_priority_mode key
        base = DecisionRuleset(enable_catalyst_priority=True)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        )
        try:
            d = {}
            from dataclasses import fields as dc_fields
            for f in dc_fields(base):
                val = getattr(base, f.name)
                if f.name == "sizing_weights":
                    d[f.name] = dict(val)
                else:
                    d[f.name] = val
            # Remove the new fields to simulate old-format JSON
            d.pop("catalyst_priority_mode", None)
            d.pop("catalyst_priority_rank_bonuses", None)
            json.dump(d, tmp, indent=2, sort_keys=True)
            tmp.write("\n")
            tmp.close()

            loaded = DecisionRuleset.from_json(tmp.name)
            assert loaded.catalyst_priority_mode == "tiebreaker"
        finally:
            os.unlink(tmp.name)
