"""Tests for two-baseline factor drift monitor."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_factor_drift import (
    _check_momentum_alert,
    _check_signal_drift_alerts,
    _is_eligible,
    _mean,
    _std,
    build_factor_drift,
    compute_factor_tilts,
    compute_hhi,
    compute_jaccard,
    compute_momentum_mix,
    compute_rank_correlation,
    compute_signal_moments,
    compute_trailing_comparison,
    generate_alerts,
    load_trailing_history,
)

# ---------------------------------------------------------------------------
# _is_eligible
# ---------------------------------------------------------------------------


def test_eligible_1():
    assert _is_eligible({"eligible": "1"}) is True


def test_eligible_1_0():
    assert _is_eligible({"eligible": "1.0"}) is True


def test_eligible_true():
    assert _is_eligible({"eligible": "true"}) is True


def test_eligible_0():
    assert _is_eligible({"eligible": "0"}) is False


def test_eligible_missing():
    assert _is_eligible({}) is False


# ---------------------------------------------------------------------------
# compute_factor_tilts
# ---------------------------------------------------------------------------


def test_factor_tilts_basic():
    port = [{"de_beta_xbi_60d": "1.2", "de_vol_60d": "0.5", "de_drawdown": "-0.3", "de_rsi_14d": "50"}]
    univ = [
        {"de_beta_xbi_60d": "1.0", "de_vol_60d": "0.5", "de_drawdown": "-0.3", "de_rsi_14d": "50"},
        {"de_beta_xbi_60d": "1.0", "de_vol_60d": "0.5", "de_drawdown": "-0.3", "de_rsi_14d": "50"},
    ]
    result = compute_factor_tilts(port, univ)
    assert result["de_beta_xbi_60d"]["port_mean"] == 1.2
    assert result["de_beta_xbi_60d"]["univ_mean"] == 1.0
    assert result["de_beta_xbi_60d"]["tilt"] == 0.2  # (1.2 - 1.0) / 1.0


def test_factor_tilts_zero_universe_mean():
    """If universe mean is zero, tilt should be None."""
    port = [{"de_beta_xbi_60d": "0.5"}]
    univ = [{"de_beta_xbi_60d": "0"}]
    result = compute_factor_tilts(port, univ)
    assert result["de_beta_xbi_60d"]["tilt"] is None


# ---------------------------------------------------------------------------
# compute_momentum_mix
# ---------------------------------------------------------------------------


def test_momentum_mix():
    rows = [
        {"mom_state": "tailwind"},
        {"mom_state": "tailwind"},
        {"mom_state": "neutral"},
        {"mom_state": "headwind"},
    ]
    result = compute_momentum_mix(rows)
    assert result["tailwind"] == 50.0
    assert result["neutral"] == 25.0
    assert result["headwind"] == 25.0


def test_momentum_mix_empty():
    result = compute_momentum_mix([])
    for s in ["tailwind", "neutral", "headwind"]:
        assert result[s] == 0.0


# ---------------------------------------------------------------------------
# compute_signal_moments
# ---------------------------------------------------------------------------


def test_signal_moments():
    rows = [{"financial_score": "10"}, {"financial_score": "20"}]
    result = compute_signal_moments(rows, rows)
    assert result["financial_score"]["port_mean"] == 15.0


# ---------------------------------------------------------------------------
# compute_jaccard
# ---------------------------------------------------------------------------


def test_jaccard_identical():
    assert compute_jaccard({"A", "B", "C"}, {"A", "B", "C"}) == 1.0


def test_jaccard_disjoint():
    assert compute_jaccard({"A"}, {"B"}) == 0.0


def test_jaccard_partial():
    # 2/4 = 0.5
    assert compute_jaccard({"A", "B", "C"}, {"B", "C", "D"}) == 0.5


def test_jaccard_empty():
    assert compute_jaccard(set(), {"A"}) is None


# ---------------------------------------------------------------------------
# compute_rank_correlation
# ---------------------------------------------------------------------------


def test_rank_corr_identical():
    ranks = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    assert compute_rank_correlation(ranks, ranks) == 1.0


def test_rank_corr_reversed():
    curr = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    prev = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
    rho = compute_rank_correlation(curr, prev)
    assert rho == -1.0


def test_rank_corr_too_few():
    curr = {"A": 1, "B": 2}
    prev = {"A": 1, "B": 2}
    assert compute_rank_correlation(curr, prev) is None


# ---------------------------------------------------------------------------
# compute_hhi
# ---------------------------------------------------------------------------


def test_hhi_equal_weight():
    weights = [100 / 30] * 30
    hhi = compute_hhi(weights)
    assert 330 < hhi < 340  # ~333.3


def test_hhi_concentrated():
    weights = [100.0] + [0.0] * 29
    hhi = compute_hhi(weights)
    assert hhi == 10000.0


def test_hhi_empty():
    assert compute_hhi([]) == 0.0


# ---------------------------------------------------------------------------
# Alert generation
# ---------------------------------------------------------------------------


def test_factor_tilt_yellow_alert():
    tilts = {
        "de_beta_xbi_60d": {"port_mean": 1.0, "univ_mean": 0.8, "tilt": 0.15},
        "de_vol_60d": {"port_mean": 1.0, "univ_mean": 1.0, "tilt": 0.0},
        "de_drawdown": {"port_mean": 1.0, "univ_mean": 1.0, "tilt": 0.0},
        "de_rsi_14d": {"port_mean": 1.0, "univ_mean": 1.0, "tilt": 0.0},
    }
    alerts = generate_alerts(tilts, 0.9, 333, {"tailwind": 33, "neutral": 33, "headwind": 33}, {}, 100, [])
    factor_alerts = [a for a in alerts if a["code"] == "FACTOR_TILT"]
    assert len(factor_alerts) == 1
    assert factor_alerts[0]["level"] == "YELLOW"


def test_factor_tilt_red_alert():
    tilts = {
        "de_beta_xbi_60d": {"port_mean": 1.0, "univ_mean": 0.8, "tilt": 0.25},
        "de_vol_60d": {"port_mean": 1.0, "univ_mean": 1.0, "tilt": 0.0},
        "de_drawdown": {"port_mean": 1.0, "univ_mean": 1.0, "tilt": 0.0},
        "de_rsi_14d": {"port_mean": 1.0, "univ_mean": 1.0, "tilt": 0.0},
    }
    alerts = generate_alerts(tilts, 0.9, 333, {"tailwind": 33, "neutral": 33, "headwind": 33}, {}, 100, [])
    factor_alerts = [a for a in alerts if a["code"] == "FACTOR_TILT"]
    assert len(factor_alerts) == 1
    assert factor_alerts[0]["level"] == "RED"


def test_jaccard_red_alert():
    tilts = {c: {"tilt": 0.0} for c in ["de_beta_xbi_60d", "de_vol_60d", "de_drawdown", "de_rsi_14d"]}
    alerts = generate_alerts(tilts, 0.65, 333, {"tailwind": 33, "neutral": 33, "headwind": 33}, {}, 100, [])
    jaccard_alerts = [a for a in alerts if a["code"] == "JACCARD_LOW"]
    assert len(jaccard_alerts) == 1
    assert jaccard_alerts[0]["level"] == "RED"


def test_jaccard_yellow_alert():
    tilts = {c: {"tilt": 0.0} for c in ["de_beta_xbi_60d", "de_vol_60d", "de_drawdown", "de_rsi_14d"]}
    alerts = generate_alerts(tilts, 0.75, 333, {"tailwind": 33, "neutral": 33, "headwind": 33}, {}, 100, [])
    jaccard_alerts = [a for a in alerts if a["code"] == "JACCARD_LOW"]
    assert len(jaccard_alerts) == 1
    assert jaccard_alerts[0]["level"] == "YELLOW"


def test_no_alerts_clean():
    tilts = {c: {"tilt": 0.05} for c in ["de_beta_xbi_60d", "de_vol_60d", "de_drawdown", "de_rsi_14d"]}
    alerts = generate_alerts(tilts, 0.90, 333, {"tailwind": 33, "neutral": 33, "headwind": 33}, {}, 100, [])
    assert len(alerts) == 0


def test_hhi_drift_yellow():
    tilts = {c: {"tilt": 0.0} for c in ["de_beta_xbi_60d", "de_vol_60d", "de_drawdown", "de_rsi_14d"]}
    trailing = [{"hhi": 333.0}] * 10
    alerts = generate_alerts(tilts, 0.90, 400, {"tailwind": 33, "neutral": 33, "headwind": 33}, {}, 100, trailing)
    hhi_alerts = [a for a in alerts if a["code"] == "HHI_DRIFT"]
    assert len(hhi_alerts) == 1
    assert hhi_alerts[0]["level"] == "YELLOW"


def test_signal_drift_red():
    tilts = {c: {"tilt": 0.0} for c in ["de_beta_xbi_60d", "de_vol_60d", "de_drawdown", "de_rsi_14d"]}
    moments = {"financial_score": {"port_mean": 20.0, "port_std": 5.0, "univ_mean": 10.0, "univ_std": 3.0}}
    # Trail with stable mean ~10 and some variance
    trailing = [{"signal_moments": {"financial_score": {"port_mean": 10.0 + i * 0.1}}} for i in range(10)]
    alerts = generate_alerts(tilts, 0.90, 333, {"tailwind": 33, "neutral": 33, "headwind": 33}, moments, 100, trailing)
    sig_alerts = [a for a in alerts if a["code"] == "SIGNAL_DRIFT"]
    assert len(sig_alerts) >= 1
    assert sig_alerts[0]["level"] == "RED"


def test_universe_size_alert():
    tilts = {c: {"tilt": 0.0} for c in ["de_beta_xbi_60d", "de_vol_60d", "de_drawdown", "de_rsi_14d"]}
    trailing = [{"universe_size": 100}] * 10
    alerts = generate_alerts(tilts, 0.90, 333, {"tailwind": 33, "neutral": 33, "headwind": 33}, {}, 75, trailing)
    size_alerts = [a for a in alerts if a["code"] == "UNIVERSE_SIZE"]
    assert len(size_alerts) == 1
    assert size_alerts[0]["level"] == "RED"


# ---------------------------------------------------------------------------
# compute_trailing_comparison
# ---------------------------------------------------------------------------


def test_trailing_comparison():
    tilts = {"de_beta_xbi_60d": {"port_mean": 1.5, "univ_mean": 1.0, "tilt": 0.5}}
    trailing = [
        {"portfolio_vs_universe": {"de_beta_xbi_60d": {"port_mean": 1.0}}},
        {"portfolio_vs_universe": {"de_beta_xbi_60d": {"port_mean": 1.1}}},
        {"portfolio_vs_universe": {"de_beta_xbi_60d": {"port_mean": 1.05}}},
    ]
    result = compute_trailing_comparison(tilts, trailing)
    assert "de_beta_xbi_60d" in result
    assert result["de_beta_xbi_60d"]["drift_z"] is not None
    assert result["de_beta_xbi_60d"]["drift_z"] > 0  # 1.5 is above trail avg ~1.05


def test_trailing_comparison_empty():
    tilts = {"de_beta_xbi_60d": {"port_mean": 1.0, "univ_mean": 1.0, "tilt": 0.0}}
    assert compute_trailing_comparison(tilts, []) == {}


# ---------------------------------------------------------------------------
# load_trailing_history
# ---------------------------------------------------------------------------


def test_load_trailing_empty(tmp_path):
    result = load_trailing_history("2026-04-05", tmp_path)
    assert result == []


def test_load_trailing_with_files(tmp_path):
    for d in ["2026-04-01", "2026-04-02", "2026-04-03"]:
        path = tmp_path / f"{d}_factor_drift.json"
        path.write_text(json.dumps({"as_of_date": d, "hhi": 333}))
    result = load_trailing_history("2026-04-05", tmp_path, n=20)
    assert len(result) == 3
    # Should be oldest-first
    assert result[0]["as_of_date"] == "2026-04-01"
    assert result[2]["as_of_date"] == "2026-04-03"


def test_load_trailing_respects_cutoff(tmp_path):
    for d in ["2026-04-01", "2026-04-02", "2026-04-06"]:
        path = tmp_path / f"{d}_factor_drift.json"
        path.write_text(json.dumps({"as_of_date": d}))
    result = load_trailing_history("2026-04-05", tmp_path, n=20)
    assert len(result) == 2  # 2026-04-06 excluded


# ---------------------------------------------------------------------------
# Integration: build_factor_drift
# ---------------------------------------------------------------------------


def test_build_factor_drift_integration(tmp_path):
    """Full integration test with synthetic snapshot."""
    snap_dir = tmp_path / "snapshots"
    art_dir = tmp_path / "artifacts"
    date = "2026-04-05"
    (snap_dir / date).mkdir(parents=True)

    # Write a minimal rankings.csv
    header = "ticker,actionable_rank,eligible,target_weight_pct,mom_state,de_beta_xbi_60d,de_vol_60d,de_drawdown,de_rsi_14d,financial_score,coinvest_score_z,inst_delta_z,selector_score,clinical_score_v2_z\n"
    rows = []
    for i in range(60):
        ticker = f"T{i:03d}"
        eligible = "1" if i < 50 else "0"
        rank = i + 1
        mom = ["tailwind", "neutral", "headwind"][i % 3]
        rows.append(f"{ticker},{rank},{eligible},3.33,{mom},1.0,0.5,-0.3,50,10,0.5,0.2,80,1.0")

    (snap_dir / date / "rankings.csv").write_text(header + "\n".join(rows))

    result = build_factor_drift(
        date,
        snapshots_dir=snap_dir,
        artifacts_dir=art_dir,
        top_k=30,
    )

    assert "error" not in result
    assert result["as_of_date"] == date
    assert result["universe_size"] == 50
    assert result["n_portfolio"] == 30
    assert result["attention"] in ("GREEN", "YELLOW", "RED")
    assert "de_beta_xbi_60d" in result["portfolio_vs_universe"]
    assert "financial_score" in result["signal_moments"]
    assert "portfolio" in result["momentum_mix"]
    assert isinstance(result["alerts"], list)

    # Check artifact was written
    out = art_dir / "factor_drift" / f"{date}_factor_drift.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["schema"] == "factor_drift.v1"


# ---------------------------------------------------------------------------
# _mean / _std edge cases
# ---------------------------------------------------------------------------


def test_mean_empty():
    assert _mean([]) is None


def test_mean_with_nan():
    assert _mean([1.0, float("nan"), 3.0]) == 2.0


def test_std_single():
    assert _std([5.0]) is None


def test_std_pair():
    result = _std([10.0, 20.0])
    assert abs(result - 7.0711) < 0.01


# ---------------------------------------------------------------------------
# _check_momentum_alert — trailing history shape (issue #533)
# ---------------------------------------------------------------------------

STEADY_MIX = {"tailwind": 60.0, "neutral": 20.0, "headwind": 20.0}
UNIVERSE_MIX = {"tailwind": 38.8, "neutral": 20.1, "headwind": 41.1}


def _nested_history(port_mix, n=5):
    """History entries as build_factor_drift writes them: nested under 'portfolio'."""
    return [{"momentum_mix": {"portfolio": dict(port_mix), "universe": dict(UNIVERSE_MIX)}} for _ in range(n)]


def test_momentum_alert_reads_nested_history_shape():
    """A steady portfolio mix must not alert.

    build_factor_drift writes momentum_mix as {"portfolio": ..., "universe": ...}.
    Reading it as a flat {state: pct} mapping silently yields a 0% baseline for
    every state, so the whole current mix is reported as a shift.
    """
    assert _check_momentum_alert(STEADY_MIX, _nested_history(STEADY_MIX)) == []


def test_momentum_alert_flat_legacy_history_still_supported():
    """Pre-nesting history entries stored the portfolio mix flat."""
    trailing = [{"momentum_mix": dict(STEADY_MIX)} for _ in range(5)]
    assert _check_momentum_alert(STEADY_MIX, trailing) == []


def test_momentum_alert_detects_real_shift_with_nested_history():
    """A genuine shift still alerts, and reports the true trailing baseline."""
    prior = {"tailwind": 40.0, "neutral": 30.0, "headwind": 30.0}
    current = {"tailwind": 80.0, "neutral": 10.0, "headwind": 10.0}
    alerts = _check_momentum_alert(current, _nested_history(prior))

    assert {a["code"] for a in alerts} == {"MOMENTUM_SHIFT"}
    tailwind = next(a for a in alerts if a["detail"].startswith("tailwind"))
    assert "vs trail avg 40%" in tailwind["detail"]
    assert "shift=40pp" in tailwind["detail"]


def test_momentum_alert_no_history():
    assert _check_momentum_alert(STEADY_MIX, []) == []


# ---------------------------------------------------------------------------
# _check_signal_drift_alerts — marginal-crossing legibility (issue #534)
# ---------------------------------------------------------------------------


def test_signal_drift_marginal_crossing_shows_two_decimals():
    """A z just over the 1.0 threshold must not render as a bare "1.0".

    Rounding to one decimal makes every marginal crossing print the threshold
    value itself, so unrelated signals with very different deltas both display
    "z=1.0" and look artificially saturated.
    """
    col = "inst_delta_z"
    # trail means 1,2,3 -> mean 2.0, sample std 1.0; current 3.02 -> z = 1.02
    trailing = [{"signal_moments": {col: {"port_mean": v}}} for v in (1.0, 2.0, 3.0)]
    moments = {col: {"port_mean": 3.02}}

    alerts = _check_signal_drift_alerts(moments, trailing)

    assert len(alerts) == 1
    assert alerts[0]["level"] == "YELLOW"
    assert "z=1.02" in alerts[0]["detail"]


def test_signal_drift_degenerate_trailing_std_suppressed():
    """A constant trailing window has no dispersion — no z is defined."""
    col = "selector_score"
    trailing = [{"signal_moments": {col: {"port_mean": 0.5}}} for _ in range(5)]
    moments = {col: {"port_mean": 0.9}}
    assert _check_signal_drift_alerts(moments, trailing) == []
