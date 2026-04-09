"""Contract test: decision engine determinism.

Proves that compute_decision_fields produces byte-identical output when
called twice with the same inputs.  Catches floating-point ordering bugs,
dict iteration non-determinism, and accidental use of random().
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import DecisionRuleset, compute_actionable_sort_key, compute_decision_fields, compute_sort_contribs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rec(
    ticker: str = "ACME",
    *,
    composite_rank: int = 50,
    optionality_pct: float = 0.65,
    drawdown: float = -0.15,
    vol_60d: float = 0.55,
    alpha_60d: float = 0.03,
    catalyst_days: int = 45,
    tier1_count: int = 3,
    severity: str = "NONE",
    confidence: float = 0.72,
    red_flag: bool = False,
) -> Dict[str, Any]:
    """Build a realistic rec for determinism testing."""
    return {
        "ticker": ticker,
        "severity": severity,
        "confidence_overall": confidence,
        "fundamental_red_flag": red_flag,
        "fundamental_red_flag_reasons": [],
        "flags": [],
        "attn_flags": [],
        "catalyst_decay": {
            "days_to_catalyst": catalyst_days,
            "in_optimal_window": True,
        },
        "smart_money_signal": {
            "tier1_holders": tier1_count,
            "holders_increasing": 2,
            "holders_decreasing": 0,
            "overlap_count": 1,
            "tier_breakdown": {},
        },
        "coinvest": {"tier1_count": tier1_count},
        "defensive_features": {
            "drawdown": drawdown,
            "vol_60d": vol_60d,
            "beta_xbi_60d": 1.1,
            "rsi_14d": 55.0,
        },
        "score_breakdown": {
            "enhancements": {
                "momentum": {"alpha_60d": alpha_60d},
            },
        },
        "momentum_signal": {"alpha_60d": alpha_60d},
        "survivability_signal": {
            "coverage": [],
            "metrics": {"cash_total": 500_000_000, "burn_ttm": 50_000_000},
        },
        "composite_rank": composite_rank,
        "composite_score": 7.5,
        "score_rank_pct": optionality_pct,
    }


def _canonical(d: Dict[str, Any]) -> str:
    """Deterministic JSON string for comparison."""
    return json.dumps(d, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Determinism contracts
# ---------------------------------------------------------------------------


class TestDecisionFieldsDeterminism:
    """Contract: identical inputs → identical outputs, every time."""

    RULESETS = [
        DecisionRuleset(),  # default
        DecisionRuleset(sort_anchor="optionality_pct"),
        DecisionRuleset(
            enable_clinical_sort_signal=True,
            clinical_sort_weight=1.0,
        ),
        DecisionRuleset(
            enable_missingness_sort_penalty=True,
            enable_missingness_size_penalty=True,
        ),
        DecisionRuleset(
            catalyst_priority_mode="blended",
        ),
    ]

    ARCHETYPES = ["drug_developer", "commercial_biotech", "commercial_pharma", "platform_biotech"]

    def test_single_rec_deterministic(self):
        """Same rec → same fields, 10 consecutive calls."""
        rec = _make_rec()
        rs = DecisionRuleset()
        results = [_canonical(compute_decision_fields(rec, archetype="drug_developer", ruleset=rs)) for _ in range(10)]
        assert len(set(results)) == 1, "Decision fields not deterministic across 10 calls"

    def test_all_rulesets_deterministic(self):
        """Every ruleset variant produces identical output on repeat."""
        rec = _make_rec()
        for rs in self.RULESETS:
            a = _canonical(compute_decision_fields(rec, archetype="drug_developer", ruleset=rs))
            b = _canonical(compute_decision_fields(rec, archetype="drug_developer", ruleset=rs))
            assert a == b, f"Non-deterministic with ruleset {rs.ruleset_id}"

    def test_all_archetypes_deterministic(self):
        """Every archetype produces identical output on repeat."""
        rec = _make_rec()
        rs = DecisionRuleset()
        for arch in self.ARCHETYPES:
            a = _canonical(compute_decision_fields(rec, archetype=arch, ruleset=rs))
            b = _canonical(compute_decision_fields(rec, archetype=arch, ruleset=rs))
            assert a == b, f"Non-deterministic with archetype {arch}"

    def test_multi_rec_batch_deterministic(self):
        """Batch of 20 recs → same fields in same order."""
        recs = [
            _make_rec(
                ticker=f"T{i:03d}",
                composite_rank=i * 5,
                optionality_pct=0.30 + i * 0.03,
                drawdown=-0.05 - i * 0.02,
                catalyst_days=30 + i * 10,
            )
            for i in range(20)
        ]
        rs = DecisionRuleset()

        def _run_batch():
            return [_canonical(compute_decision_fields(r, archetype="drug_developer", ruleset=rs)) for r in recs]

        a = _run_batch()
        b = _run_batch()
        assert a == b, "Batch determinism failed"


class TestSortKeyDeterminism:
    """Contract: sort keys are deterministic and produce stable ordering."""

    def test_sort_key_deterministic(self):
        """Same inputs → same sort key tuple."""
        fields = compute_decision_fields(_make_rec(), archetype="drug_developer")
        rs = DecisionRuleset()
        keys = [
            compute_actionable_sort_key(
                fields,
                archetype="drug_developer",
                optionality=0.65,
                composite_rank=50,
                ticker="ACME",
                tiebreaker_pct=0.65,
                alpha_raw=0.03,
                ruleset=rs,
            )
            for _ in range(10)
        ]
        assert len(set(keys)) == 1, "Sort key not deterministic"

    def test_sort_contribs_deterministic(self):
        """Sort contributions are deterministic."""
        fields = compute_decision_fields(_make_rec(), archetype="drug_developer")
        rs = DecisionRuleset(
            enable_clinical_sort_signal=True,
            enable_calendar_alpha_sort=True,
        )
        results = [
            compute_sort_contribs(
                fields,
                archetype="drug_developer",
                ruleset=rs,
                tiebreaker_pct=0.65,
                alpha_raw=0.03,
            )
            for _ in range(10)
        ]
        totals = [r[0] for r in results]
        maps = [json.dumps(r[1], sort_keys=True, default=str) for r in results]
        assert len(set(totals)) == 1, "Sort contrib totals not deterministic"
        assert len(set(maps)) == 1, "Sort contrib maps not deterministic"

    def test_ordering_stable_across_runs(self):
        """20 tickers sorted twice → identical rank order."""
        recs = [
            _make_rec(
                ticker=f"T{i:03d}",
                composite_rank=i * 5,
                optionality_pct=0.30 + i * 0.03,
            )
            for i in range(20)
        ]
        rs = DecisionRuleset(sort_anchor="optionality_pct")

        def _rank_order():
            items = []
            for r in recs:
                fields = compute_decision_fields(r, archetype="drug_developer", ruleset=rs)
                key = compute_actionable_sort_key(
                    fields,
                    archetype="drug_developer",
                    optionality=r["score_rank_pct"],
                    composite_rank=r["composite_rank"],
                    ticker=r["ticker"],
                    tiebreaker_pct=r["score_rank_pct"],
                    alpha_raw=0.03,
                    ruleset=rs,
                )
                items.append((r["ticker"], key))
            items.sort(key=lambda x: x[1])
            return [t for t, _ in items]

        a = _rank_order()
        b = _rank_order()
        assert a == b, "Sort ordering not stable"


class TestEdgeCaseDeterminism:
    """Contract: edge-case inputs don't introduce non-determinism."""

    def test_none_fields_deterministic(self):
        """Rec with many None/missing fields → deterministic."""
        rec = {
            "ticker": "EMPTY",
            "severity": "NONE",
            "confidence_overall": None,
            "fundamental_red_flag": False,
            "fundamental_red_flag_reasons": [],
            "flags": [],
            "attn_flags": [],
            "catalyst_decay": None,
            "smart_money_signal": {},
            "coinvest": {},
            "defensive_features": {},
            "score_breakdown": {},
            "momentum_signal": {},
        }
        a = _canonical(compute_decision_fields(rec, archetype="drug_developer"))
        b = _canonical(compute_decision_fields(rec, archetype="drug_developer"))
        assert a == b

    def test_ineligible_rec_deterministic(self):
        """Ineligible rec (deep drawdown) → deterministic."""
        rec = _make_rec(drawdown=-0.80)
        a = _canonical(compute_decision_fields(rec, archetype="drug_developer"))
        b = _canonical(compute_decision_fields(rec, archetype="drug_developer"))
        assert a == b

    def test_red_flag_rec_deterministic(self):
        """Red-flagged rec → deterministic."""
        rec = _make_rec(red_flag=True)
        a = _canonical(compute_decision_fields(rec, archetype="drug_developer"))
        b = _canonical(compute_decision_fields(rec, archetype="drug_developer"))
        assert a == b
