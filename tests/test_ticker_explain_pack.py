"""Tests for scripts/generate_ticker_explain_pack.py — ticker explain pack."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_ticker_explain_pack import (
    DEFAULT_A_FLOOR,
    MAX_AUTO_TICKERS,
    _build_eligibility_map,
    auto_select_tickers,
    build_explain_pack,
    compute_ticker_delta,
    extract_dossier,
    main,
    render_dossier_md,
    render_explain_pack_md,
)
from run_phase2_snapshot_delta import SnapshotData


# ---------------------------------------------------------------------------
# Synthetic data builders (self-contained, mirroring test_drift_report.py)
# ---------------------------------------------------------------------------
def _make_rankings(
    n_dev: int = 50,
    include_attribution_cols: bool = True,
    ruleset_id: str = "bf6815e2",
) -> pd.DataFrame:
    """Build a synthetic rankings DataFrame with attribution columns."""
    import random

    rng = random.Random(42)

    n_a = max(1, int(n_dev * 0.05))
    n_b = max(1, int(n_dev * 0.30))
    n_c = max(1, int(n_dev * 0.40))
    n_d = n_dev - n_a - n_b - n_c

    tiers = ["A"] * n_a + ["B"] * n_b + ["C"] * n_c + ["D"] * max(0, n_d)
    actual_n = len(tiers)

    n_eligible = int(actual_n * 0.50)
    eligible = ["1"] * n_eligible + ["0"] * (actual_n - n_eligible)

    opt_vals = [round(0.5 + rng.gauss(0, 0.30), 4) for _ in range(actual_n)]

    data = {
        "ticker": [f"T{i:03d}" for i in range(actual_n)],
        "archetype": ["drug_developer"] * actual_n,
        "tier_dev": tiers,
        "catalyst_mode": ["specific_days"] * n_eligible + ["no_upcoming"] * (actual_n - n_eligible),
        "catalyst_days": list(range(10, 10 + actual_n)),
        "actionable_rank": list(range(1, actual_n + 1)),
        "risk_flags": [""] * actual_n,
        "target_weight_pct": [round(100 / actual_n, 2)] * actual_n,
        "decision_engine_ruleset_id": [ruleset_id] * actual_n,
        "composite_rank": list(range(1, actual_n + 1)),
        "composite_score": [50.0 - i * 0.3 for i in range(actual_n)],
        "clinical_optionality_pct_dev": opt_vals,
        "size_band": ["L"] * (actual_n // 2) + ["M"] * (actual_n - actual_n // 2),
        "size_reasons": [""] * actual_n,
        "decision_engine_version": ["v1.3.0"] * actual_n,
        "eligible": eligible,
        "mom_state": ["tailwind"] * (actual_n // 2) + ["headwind"] * (actual_n - actual_n // 2),
        "de_drawdown": [-0.10 + i * 0.005 for i in range(actual_n)],
        "de_drawdown_rel_xbi": [-0.05 + i * 0.003 for i in range(actual_n)],
        "de_drawdown_missing_reason": [""] * actual_n,
        "severity": [""] * actual_n,
        "catalyst_in_window": ["1"] * n_eligible + ["0"] * (actual_n - n_eligible),
        "sponsor_tier1_count": [2] * actual_n,
        "sponsor_net_buying": [3.0] * actual_n,
    }

    if include_attribution_cols:
        inelig_reasons = []
        for i in range(actual_n):
            if eligible[i] == "0":
                if i % 3 == 0:
                    inelig_reasons.append("fundamental_red_flag")
                elif i % 3 == 1:
                    inelig_reasons.append("deep_drawdown")
                else:
                    inelig_reasons.append("fundamental_red_flag|sev3")
            else:
                inelig_reasons.append("")
        data["ineligible_reasons"] = inelig_reasons

        bands_cycle = ["near", "mid", "far", "missing"]
        strength_vals = []
        for i in range(actual_n):
            if eligible[i] == "1":
                strength_vals.append(bands_cycle[i % len(bands_cycle)])
            else:
                strength_vals.append("missing")
        data["catalyst_strength"] = strength_vals

        tier_reasons = []
        for t in tiers:
            if t == "A":
                tier_reasons.append("high_opt+catalyst_near")
            elif t == "B":
                tier_reasons.append("mod_opt")
            elif t == "C":
                tier_reasons.append("low_opt")
            else:
                tier_reasons.append("ineligible")
        data["tier_reason"] = tier_reasons

    return pd.DataFrame(data)


def _make_snapshot(
    date: str,
    rankings: pd.DataFrame | None = None,
    ruleset_id: str = "bf6815e2",
) -> SnapshotData:
    if rankings is None:
        rankings = _make_rankings(ruleset_id=ruleset_id)
    portfolio = rankings.head(20).copy()
    return SnapshotData(
        date=date,
        path=Path(f"/fake/{date}"),
        rankings=rankings,
        portfolio=portfolio,
        metadata={"as_of_date": date},
        ruleset_id=ruleset_id,
        has_native_portfolio=True,
    )


# ============================================================================
# TestAutoSelection
# ============================================================================
class TestAutoSelection:
    def test_auto_selects_churn_tickers(self):
        """Swapping ranks forces churn — dropped + added appear."""
        r1 = _make_rankings()
        r2 = _make_rankings()
        # Swap T000 and T024 actionable ranks so one drops out of top-25
        r2.loc[r2["ticker"] == "T000", "actionable_rank"] = 30
        r2.loc[r2["ticker"] == "T024", "actionable_rank"] = 30
        # Push a new ticker into top-25
        r2.loc[r2["ticker"] == "T049", "actionable_rank"] = 1

        prior = _make_snapshot("2026-02-07", r1)
        current = _make_snapshot("2026-02-09", r2)

        tickers = auto_select_tickers(current, prior)
        # The swapped tickers should appear (either dropped or added)
        assert len(tickers) > 0
        assert isinstance(tickers, list)
        # Verify sorted
        assert tickers == sorted(tickers)

    def test_auto_selects_strength_movers(self):
        """Changing catalyst_strength triggers selection."""
        r1 = _make_rankings()
        r2 = _make_rankings()
        # Change T001's strength from mid to far (degraded)
        r2.loc[r2["ticker"] == "T001", "catalyst_strength"] = "far"

        prior = _make_snapshot("2026-02-07", r1)
        current = _make_snapshot("2026-02-09", r2)

        tickers = auto_select_tickers(current, prior)
        assert "T001" in tickers

    def test_auto_selects_eligibility_flippers(self):
        """Flip eligible status → ticker appears."""
        r1 = _make_rankings()
        r2 = _make_rankings()
        # T000 is eligible in r1 (idx 0, eligible="1"); flip to ineligible
        r2.loc[r2["ticker"] == "T000", "eligible"] = "0"

        prior = _make_snapshot("2026-02-07", r1)
        current = _make_snapshot("2026-02-09", r2)

        tickers = auto_select_tickers(current, prior)
        assert "T000" in tickers

    def test_auto_selection_deduplication(self):
        """Ticker appearing in both churn + strength counted once."""
        r1 = _make_rankings()
        r2 = _make_rankings()
        # T001: change strength AND push out of top-25
        r2.loc[r2["ticker"] == "T001", "catalyst_strength"] = "far"
        r2.loc[r2["ticker"] == "T001", "actionable_rank"] = 30

        prior = _make_snapshot("2026-02-07", r1)
        current = _make_snapshot("2026-02-09", r2)

        tickers = auto_select_tickers(current, prior)
        assert tickers.count("T001") <= 1

    def test_auto_selection_cap(self):
        """More than MAX_AUTO_TICKERS movers → capped."""
        r1 = _make_rankings(n_dev=100)
        r2 = _make_rankings(n_dev=100)
        # Flip eligibility on 60 tickers to exceed cap
        for i in range(60):
            ticker = f"T{i:03d}"
            old_val = r1.loc[r1["ticker"] == ticker, "eligible"].values[0]
            new_val = "0" if old_val == "1" else "1"
            r2.loc[r2["ticker"] == ticker, "eligible"] = new_val

        prior = _make_snapshot("2026-02-07", r1)
        current = _make_snapshot("2026-02-09", r2)

        tickers = auto_select_tickers(current, prior)
        assert len(tickers) <= MAX_AUTO_TICKERS


# ============================================================================
# TestExtractDossier
# ============================================================================
class TestExtractDossier:
    def test_all_fields_populated(self):
        """Known row returns all expected sections."""
        rankings = _make_rankings()
        dossier = extract_dossier("T000", rankings)
        assert dossier["found"] is True
        assert dossier["ticker"] == "T000"
        assert dossier["archetype"] == "drug_developer"
        assert isinstance(dossier["gates"], dict)
        assert len(dossier["gates"]) == 4
        assert dossier["composite_rank"] is not None
        assert dossier["composite_score"] is not None
        assert dossier["tier_dev"] in ("A", "B", "C", "D")

    def test_missing_ticker(self):
        """Unknown ticker returns found=False."""
        rankings = _make_rankings()
        dossier = extract_dossier("ZZZZZ", rankings)
        assert dossier["found"] is False
        assert dossier["ticker"] == "ZZZZZ"

    def test_missing_columns_graceful(self):
        """Missing catalyst_strength column → empty default."""
        rankings = _make_rankings(include_attribution_cols=False)
        # catalyst_strength is only added by include_attribution_cols
        dossier = extract_dossier("T000", rankings)
        assert dossier["found"] is True
        assert dossier["catalyst_strength"] == ""

    def test_gates_breakdown(self):
        """Ineligible ticker with pipe-separated gates → correct fail/pass."""
        rankings = _make_rankings()
        # Find an ineligible ticker with "fundamental_red_flag|sev3"
        inelig = rankings[rankings["ineligible_reasons"] == "fundamental_red_flag|sev3"]
        if inelig.empty:
            pytest.skip("No pipe-separated gate row in synthetic data")
        ticker = inelig.iloc[0]["ticker"]

        dossier = extract_dossier(ticker, rankings)
        assert dossier["gates"]["fundamental_red_flag"] == "fail"
        assert dossier["gates"]["sev3"] == "fail"
        assert dossier["gates"]["deep_drawdown"] == "pass"
        assert dossier["gates"]["adv_fail"] == "pass"

    def test_optionality_vs_a_floor(self):
        """Optionality above/below a_floor correctly annotated."""
        rankings = _make_rankings()
        # Set a known optionality value
        rankings.loc[rankings["ticker"] == "T000", "clinical_optionality_pct_dev"] = 0.70

        dossier_above = extract_dossier("T000", rankings, a_floor=0.60)
        assert dossier_above["clears_a_floor"] is True

        dossier_below = extract_dossier("T000", rankings, a_floor=0.80)
        assert dossier_below["clears_a_floor"] is False


# ============================================================================
# TestTickerDelta
# ============================================================================
class TestTickerDelta:
    def test_string_delta_tier_change(self):
        """Tier change B→A appears in delta."""
        r1 = _make_rankings()
        r2 = _make_rankings()
        # Change T005's tier_dev
        r1.loc[r1["ticker"] == "T005", "tier_dev"] = "B"
        r2.loc[r2["ticker"] == "T005", "tier_dev"] = "A"

        delta = compute_ticker_delta("T005", r2, r1)
        assert delta is not None
        assert delta["in_current"] is True
        assert delta["in_prior"] is True
        assert "tier_dev" in delta
        assert delta["tier_dev"]["prior"] == "B"
        assert delta["tier_dev"]["current"] == "A"

    def test_numeric_delta_rank_change(self):
        """Rank change 15→8 has change=-7."""
        r1 = _make_rankings()
        r2 = _make_rankings()
        r1.loc[r1["ticker"] == "T005", "actionable_rank"] = 15
        r2.loc[r2["ticker"] == "T005", "actionable_rank"] = 8

        delta = compute_ticker_delta("T005", r2, r1)
        assert delta is not None
        assert "actionable_rank" in delta
        assert delta["actionable_rank"]["prior"] == 15.0
        assert delta["actionable_rank"]["current"] == 8.0
        assert delta["actionable_rank"]["change"] == -7.0

    def test_absent_in_prior(self):
        """Ticker in current but not prior → in_prior=False."""
        r1 = _make_rankings()
        r2 = _make_rankings()
        # Remove T000 from prior
        r1 = r1[r1["ticker"] != "T000"]

        delta = compute_ticker_delta("T000", r2, r1)
        assert delta is not None
        assert delta["in_current"] is True
        assert delta["in_prior"] is False
        # No field-level deltas when one side missing
        assert "tier_dev" not in delta

    def test_no_changes_sparse_delta(self):
        """Identical rows → only in_current/in_prior keys."""
        r1 = _make_rankings()
        r2 = _make_rankings()

        delta = compute_ticker_delta("T005", r2, r1)
        assert delta is not None
        assert delta["in_current"] is True
        assert delta["in_prior"] is True
        # No field-level keys beyond in_current/in_prior
        field_keys = set(delta.keys()) - {"in_current", "in_prior"}
        assert len(field_keys) == 0


# ============================================================================
# TestMarkdownOutput
# ============================================================================
class TestMarkdownOutput:
    def test_header_format(self):
        """Header follows '### TICKER (X-tier, eligible)' pattern."""
        rankings = _make_rankings()
        dossier = extract_dossier("T000", rankings)
        md = render_dossier_md(dossier)
        tier = dossier["tier_dev"]
        elig = "eligible" if dossier["eligible"] else "ineligible"
        assert f"### T000 ({tier}-tier, {elig})" in md

    def test_gates_all_pass_text(self):
        """Eligible ticker with no gate failures → 'all pass'."""
        rankings = _make_rankings()
        # T000 is eligible — should have all gates pass
        dossier = extract_dossier("T000", rankings)
        md = render_dossier_md(dossier)
        assert "all pass" in md

    def test_delta_absent_without_prior(self):
        """Without delta data, no 'Delta:' line appears."""
        rankings = _make_rankings()
        dossier = extract_dossier("T000", rankings)
        # No delta key
        assert "delta" not in dossier
        md = render_dossier_md(dossier)
        assert "**Delta:**" not in md


# ============================================================================
# TestJsonOutput
# ============================================================================
class TestJsonOutput:
    def test_json_serializable(self):
        """Pack can be serialized to JSON without errors."""
        current = _make_snapshot("2026-02-09")
        prior = _make_snapshot("2026-02-07")
        tickers = ["T000", "T001", "T005"]
        pack = build_explain_pack(current, prior, tickers)
        # Must not raise
        serialized = json.dumps(pack)
        assert isinstance(serialized, str)

    def test_json_top_level_keys(self):
        """Pack has all expected top-level keys."""
        current = _make_snapshot("2026-02-09")
        pack = build_explain_pack(current, None, ["T000"])
        expected_keys = {
            "generated_at",
            "snapshot_date",
            "prev_snapshot_date",
            "a_floor",
            "tickers_selected_by",
            "ticker_count",
            "explains",
            "gate_pressure",
        }
        assert expected_keys == set(pack.keys())


# ============================================================================
# TestCli
# ============================================================================
class TestCli:
    def test_end_to_end(self, tmp_path):
        """Write snapshots to temp dir, run main(), verify outputs exist."""
        # Write current snapshot
        current_dir = tmp_path / "2026-02-09"
        current_dir.mkdir()
        rankings = _make_rankings()
        rankings.to_csv(current_dir / "rankings.csv", index=False)
        with open(current_dir / "metadata.json", "w") as f:
            json.dump({"as_of_date": "2026-02-09"}, f)

        # Write prior snapshot
        prior_dir = tmp_path / "2026-02-07"
        prior_dir.mkdir()
        r_prior = _make_rankings()
        # Introduce some changes
        r_prior.loc[r_prior["ticker"] == "T001", "catalyst_strength"] = "far"
        r_prior.to_csv(prior_dir / "rankings.csv", index=False)
        with open(prior_dir / "metadata.json", "w") as f:
            json.dump({"as_of_date": "2026-02-07"}, f)

        output_dir = tmp_path / "output"

        main([
            "--snapshot-dir", str(current_dir),
            "--prev-snapshot-dir", str(prior_dir),
            "--output-dir", str(output_dir),
        ])

        assert (output_dir / "ticker_explain_pack.json").exists()
        assert (output_dir / "ticker_explain_pack.md").exists()

        # Verify JSON is valid
        with open(output_dir / "ticker_explain_pack.json") as f:
            pack = json.load(f)
        assert pack["snapshot_date"] == "2026-02-09"
        assert pack["prev_snapshot_date"] == "2026-02-07"
        assert pack["ticker_count"] >= 1


# ============================================================================
# TestExplanation
# ============================================================================
class TestExplanation:
    def test_dossier_has_explanation_key(self):
        """extract_dossier() output has 'explanation' key with correct structure."""
        rankings = _make_rankings()
        dossier = extract_dossier("T000", rankings)
        assert dossier["found"] is True
        assert "explanation" in dossier

        explanation = dossier["explanation"]
        expected_domains = {"eligibility", "tier", "sizing", "risk", "overlays"}
        assert set(explanation.keys()) == expected_domains

        # Check sub-structure integrity
        assert isinstance(explanation["eligibility"]["passed"], bool)
        assert isinstance(explanation["eligibility"]["failed_gates"], list)
        assert isinstance(explanation["tier"]["tier"], str)
        assert isinstance(explanation["tier"]["primary_reasons"], list)
        assert isinstance(explanation["sizing"]["band"], str)
        assert isinstance(explanation["sizing"]["modifiers"], list)
        assert isinstance(explanation["risk"]["flags"], list)
        assert isinstance(explanation["overlays"]["catalyst_mode"], str)

    def test_explanation_codes_match_dossier_fields(self):
        """explanation.tier.tier == dossier.tier_dev, etc."""
        rankings = _make_rankings()
        for ticker in ["T000", "T005", "T010"]:
            row = rankings[rankings["ticker"] == ticker]
            if row.empty:
                continue
            dossier = extract_dossier(ticker, rankings)
            if not dossier["found"]:
                continue

            explanation = dossier["explanation"]
            assert explanation["tier"]["tier"] == dossier["tier_dev"], (
                f"[{ticker}] tier mismatch: explanation={explanation['tier']['tier']}, "
                f"dossier={dossier['tier_dev']}"
            )
            assert explanation["sizing"]["band"] == dossier["size_band"], (
                f"[{ticker}] band mismatch: explanation={explanation['sizing']['band']}, "
                f"dossier={dossier['size_band']}"
            )


# ============================================================================
# TestDossierMargins
# ============================================================================
class TestDossierMargins:
    """Tests for margin/counterfactual fields in extract_dossier()."""

    def test_dossier_has_margin_keys(self):
        """Dossier includes gate_margins and tier_margins."""
        rankings = _make_rankings()
        dossier = extract_dossier("T000", rankings)
        assert "gate_margins" in dossier
        assert "tier_margins" in dossier
        gm = dossier["gate_margins"]
        assert "dd_abs_margin" in gm
        assert "rescued_by_rel" in gm
        assert "first_failed_effective_gate" in gm
        assert "counterfactual" in gm

    def test_eligible_ticker_empty_counterfactual(self):
        """Eligible ticker has empty counterfactual and no effective first-fail."""
        rankings = _make_rankings()
        # T000 is eligible
        dossier = extract_dossier("T000", rankings)
        gm = dossier["gate_margins"]
        assert gm["counterfactual"] == {}
        assert gm["first_failed_effective_gate"] == ""

    def test_ineligible_ticker_has_counterfactual(self):
        """Ineligible ticker has non-empty counterfactual."""
        rankings = _make_rankings()
        # Find an ineligible ticker with fundamental_red_flag
        inelig = rankings[rankings["ineligible_reasons"] == "fundamental_red_flag"]
        if inelig.empty:
            pytest.skip("No red_flag ticker in synthetic data")
        ticker = inelig.iloc[0]["ticker"]
        dossier = extract_dossier(ticker, rankings)
        gm = dossier["gate_margins"]
        assert gm["first_failed_effective_gate"] == "fundamental_red_flag"
        assert "fundamental_red_flag" in gm["counterfactual"]

    def test_counterfactual_minimal_one_field(self):
        """require_both counterfactual flips only 1 drawdown field."""
        rankings = _make_rankings()
        # Find a deep_drawdown ineligible ticker
        inelig = rankings[rankings["ineligible_reasons"] == "deep_drawdown"]
        if inelig.empty:
            pytest.skip("No deep_drawdown ticker in synthetic data")
        ticker = inelig.iloc[0]["ticker"]
        # Set both drawdowns breaching
        rankings.loc[rankings["ticker"] == ticker, "de_drawdown"] = -0.50
        rankings.loc[rankings["ticker"] == ticker, "de_drawdown_rel_xbi"] = -0.30
        dossier = extract_dossier(ticker, rankings)
        gm = dossier["gate_margins"]
        cf = gm["counterfactual"]
        # Minimal: only 1 drawdown field (not both)
        dd_keys = {"drawdown", "drawdown_rel_xbi"} & set(cf.keys())
        assert len(dd_keys) == 1


# ============================================================================
# TestPressureHeader
# ============================================================================
class TestPressureHeader:
    """Tests for gate_pressure in explain pack."""

    def test_pack_has_pressure_key(self):
        """build_explain_pack includes gate_pressure dict."""
        current = _make_snapshot("2026-02-09")
        pack = build_explain_pack(current, None, ["T000"])
        assert "gate_pressure" in pack
        p = pack["gate_pressure"]
        assert "dd_abs_near_gate_pct" in p
        assert "rescued_share_pct" in p
        assert "eligible_count" in p

    def test_pressure_in_markdown(self):
        """Pressure Context header rendered in markdown."""
        current = _make_snapshot("2026-02-09")
        pack = build_explain_pack(current, None, ["T000"])
        md = render_explain_pack_md(pack)
        assert "Pressure Context" in md
        assert "DD abs near gate" in md

    def test_rescued_excludes_ineligible(self):
        """Ineligible ticker with rescued_by_rel=True must NOT count in rescued_share."""
        rankings = _make_rankings(n_dev=4)
        # Make ALL tickers ineligible via fundamental_red_flag, but give them
        # drawdowns that trigger rescued_by_rel (abs breaches, rel passes).
        rankings["eligible"] = "0"
        rankings["ineligible_reasons"] = "fundamental_red_flag"
        rankings["de_drawdown"] = -0.50       # breaches -0.40 gate
        rankings["de_drawdown_rel_xbi"] = 0.0  # passes -0.20 gate
        current = _make_snapshot("2026-02-09", rankings)
        pack = build_explain_pack(current, None, ["T000"])
        p = pack["gate_pressure"]
        assert p["rescued_count"] == 0
        assert p["rescued_share_pct"] == 0.0


# ============================================================================
# TestReversalMarkdown
# ============================================================================
class TestReversalMarkdown:
    """Tests for Decision Reversal markdown block."""

    def test_eligible_no_reversal_gate(self):
        """Eligible ticker does NOT show 'Decision Reversal: fundamental_red_flag'."""
        rankings = _make_rankings()
        dossier = extract_dossier("T000", rankings)
        md = render_dossier_md(dossier)
        # Should not show any failed gate reversal
        assert "Decision Reversal: fundamental_red_flag" not in md

    def test_ineligible_shows_reversal(self):
        """Ineligible ticker shows Decision Reversal block."""
        rankings = _make_rankings()
        inelig = rankings[rankings["ineligible_reasons"] == "fundamental_red_flag"]
        if inelig.empty:
            pytest.skip("No red_flag ticker")
        ticker = inelig.iloc[0]["ticker"]
        dossier = extract_dossier(ticker, rankings)
        md = render_dossier_md(dossier)
        assert "Decision Reversal" in md
        assert "fundamental_red_flag" in md
