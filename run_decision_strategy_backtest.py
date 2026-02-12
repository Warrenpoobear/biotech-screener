#!/usr/bin/env python3
"""
Strategy-Level Decision Engine Backtest

Replays archive history to measure whether the full decision engine policy
(filter eligible A+B → order by actionable sort key → size by band) produces
better risk-adjusted portfolios than the legacy composite-rank workflow.

Outputs:
  - decision_portfolio.csv       Per-date per-ticker positions + forward returns
  - strategy_backtest_summary.csv  Per-date aggregates + POOLED row
  - strategy_backtest_details.json Full nested results
  - strategy_backtest_report.txt   Narrative report

Usage:
    python run_decision_strategy_backtest.py [options]

    --ruleset PATH       DecisionRuleset JSON (default: built-in default)
    --tier-filter A,B    Comma-separated tiers to include (default: A,B)
    --top-k 20           Max portfolio positions (default: 20)
    --output-dir PATH    Output directory (default: output/)
    --start DATE         Archive start date filter
    --end DATE           Archive end date filter
    --dry-run            Print config + archive count, exit
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.cost_model import (
    CostEstimate,
    CostSchedule,
    DEFAULT_SCHEDULE,
    compute_cost_telemetry,
    estimate_trade_cost,
)
from decision_engine import (
    DecisionRuleset,
    DEFAULT_RULESET,
    compute_actionable_sort_key,
    compute_decision_fields,
    compute_gate_margins,
    compute_target_weights,
    compute_tier_margins,
)
from run_decision_ruleset_sweep import (
    ArchiveData,
    ForwardReturnData,
    compute_snapshot_returns,
    init_providers,
    load_archive_data,
)
from run_rank_ic_backtest import (
    ARCHIVE_DIR,
    OUTPUT_DIR,
    _winsorize,
    compute_as_of_fence,
    compute_forward_returns,
    compute_residual_returns,
    discover_archives,
    estimate_xbi_betas,
    compute_xbi_regime,
)


# =============================================================================
# CONSTANTS
# =============================================================================

HORIZONS = [20, 60]
DEFAULT_TIER_FILTER = ["A", "B"]
DEFAULT_TOP_K = 20

PANEL_COLUMNS = [
    "as_of_date", "ticker", "tier", "band", "eligible", "weight",
    "ineligible_reasons", "first_failed_gate",
    "tier_reason", "risk_flags", "catalyst_mode", "catalyst_strength", "mom_state",
    "optionality", "catalyst_days_raw", "ruleset_id",
    "drawdown_abs", "drawdown_xbi", "drawdown_rel_xbi",
    "dd_abs_margin", "dd_rel_margin", "rescued_by_rel", "dd_rel_margin_rescued",
    "optionality_margin_a", "actionable_catalyst",
    "fwd_ret_20d", "fwd_ret_60d", "fwd_max_dd_20d", "fwd_max_dd_60d",
    "fwd_dd_missing_reason",
    "adv_dollars", "est_cost_bps", "participation_pct",
    "fwd_ret_20d_net", "fwd_ret_60d_net",
    "catalyst_tilt_mult", "catalyst_tilt_applied",
]


# =============================================================================
# MULTI-HORIZON FORWARD RETURNS
# =============================================================================

class MultiHorizonReturns(NamedTuple):
    """Forward return data across multiple horizons for one snapshot date."""
    raw: Dict[int, Dict[str, float]]        # {20: {ticker: ret}, 60: {...}}
    resid: Dict[int, Dict[str, float]]       # {20: {ticker: ret}, 60: {...}}
    xbi: Dict[int, Optional[float]]          # {20: xbi_ret, 60: xbi_ret}
    betas: Dict[str, float]
    regime: str


def compute_multi_horizon_returns(
    chained,
    csv_provider,
    tickers: List[str],
    as_of_date: str,
    horizons: List[int] = HORIZONS,
) -> MultiHorizonReturns:
    """Compute forward raw + residual returns at multiple horizons.

    Betas are estimated once and reused across horizons.
    """
    # Compute raw returns per horizon
    raw: Dict[int, Dict[str, float]] = {}
    for h in horizons:
        raw[h] = compute_forward_returns(chained, tickers, as_of_date, h)

    # Betas: use tickers that have returns at the longest horizon
    longest_h = max(horizons)
    beta_tickers = list(raw[longest_h].keys())
    betas = estimate_xbi_betas(csv_provider, beta_tickers, as_of_date)

    # XBI returns per horizon
    xbi: Dict[int, Optional[float]] = {}
    for h in horizons:
        xbi_fwd = compute_forward_returns(chained, ["XBI"], as_of_date, h)
        xbi[h] = xbi_fwd.get("XBI")

    # Residual returns per horizon
    resid: Dict[int, Dict[str, float]] = {}
    for h in horizons:
        resid[h] = compute_residual_returns(raw[h], xbi[h], betas)

    # Regime from trailing XBI return
    regimes = compute_xbi_regime(csv_provider, [as_of_date])
    regime = regimes.get(as_of_date, "UNKNOWN")

    return MultiHorizonReturns(
        raw=raw,
        resid=resid,
        xbi=xbi,
        betas=betas,
        regime=regime,
    )


# =============================================================================
# HELPERS
# =============================================================================

def _percentile(vals: List[float], pct: float) -> float:
    """Compute the given percentile from a sorted list. pct in [0, 100]."""
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + frac * (s[hi] - s[lo])


def _safe_div(num: float, denom: float) -> float:
    """Safe division returning 0.0 on zero denom."""
    return num / denom if denom else 0.0


# =============================================================================
# PORTFOLIO CONSTRUCTION
# =============================================================================

def build_strategy_portfolio(
    archive_data: ArchiveData,
    ruleset: DecisionRuleset,
    tier_filter: List[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Build the decision-engine strategy portfolio for one snapshot.

    1. Compute decision fields for each dev-stage ticker
    2. Filter to eligible + tier in tier_filter
    3. Sort by actionable sort key
    4. Take top-k
    5. Assign weights via compute_target_weights

    Returns list of position dicts.
    """
    # Build composite_rank map from rows
    rank_by_ticker: Dict[str, int] = {}
    for row in archive_data.rows:
        ticker = row.get("ticker", "").strip().upper()
        rank_str = row.get("composite_rank", "").strip()
        if ticker and rank_str:
            try:
                rank_by_ticker[ticker] = int(rank_str)
            except ValueError:
                pass

    # Cost schedule for est_cost_bps computation (when cost haircut is enabled)
    if ruleset.enable_cost_haircut:
        cost_schedule = CostSchedule(
            impact_cap_bps=ruleset.cost_impact_cap_bps,
        )
        # Representative weight for cost estimation (actual weights unknown yet)
        rep_weight = 100.0 / max(top_k, 1)
    else:
        cost_schedule = None
        rep_weight = 0.0

    # Compute decision fields for dev-stage tickers
    candidates: List[Tuple[Tuple, Dict[str, Any]]] = []
    for ticker in archive_data.tickers:
        archetype = archive_data.archetypes.get(ticker, "")
        if archetype != "drug_developer":
            continue

        rec = archive_data.recs.get(ticker)
        if rec is None:
            continue

        opt = archive_data.optionalities.get(ticker)

        # Compute est_cost_bps from archive market data when cost haircut is on
        est_cost_bps = None
        if cost_schedule is not None:
            md = archive_data.market_data.get(ticker, {})
            adv = (md.get("avg_volume") or 0) * (md.get("price") or 0)
            if adv > 0:
                cost_est = estimate_trade_cost(rep_weight, adv, cost_schedule)
                est_cost_bps = cost_est.round_trip_bps

        fields = compute_decision_fields(
            rec, archetype, opt, ruleset=ruleset, est_cost_bps=est_cost_bps,
        )

        # Filter: eligible + tier in filter
        if fields.get("eligible") != "1":
            continue
        tier = fields.get("tier_dev", "")
        if tier not in tier_filter:
            continue

        composite_rank = rank_by_ticker.get(ticker)
        sort_key = compute_actionable_sort_key(
            fields, archetype, opt, composite_rank, ticker
        )

        candidates.append((sort_key, {
            "ticker": ticker,
            "tier_dev": tier,
            "size_band": fields.get("size_band", ""),
            "catalyst_mode": fields.get("catalyst_mode", ""),
            "catalyst_days": fields.get("catalyst_days", ""),
            "mom_state": fields.get("mom_state", ""),
            "tier_reason": fields.get("tier_reason", ""),
            "size_reasons": fields.get("size_reasons", ""),
            "risk_flags": fields.get("risk_flags", ""),
            "composite_rank": composite_rank,
            "optionality": opt,
            "est_cost_bps": est_cost_bps,
            "cost_mult": fields.get("cost_mult", ""),
            "cost_bucket": fields.get("cost_bucket", ""),
            "cost_haircut_applied": fields.get("cost_haircut_applied", ""),
            "catalyst_tilt_mult": fields.get("catalyst_tilt_mult", 1.0),
            "catalyst_tilt_applied": fields.get("catalyst_tilt_applied", "0"),
        }))

    # Sort by actionable key and take top-k
    candidates.sort(key=lambda x: x[0])
    selected = [c[1] for c in candidates[:top_k]]

    # Assign actionable_rank (1-indexed)
    for i, pos in enumerate(selected, 1):
        pos["actionable_rank"] = i

    # Compute target weights
    selected = compute_target_weights(selected, ruleset=ruleset)

    # Rename field for consistency
    for pos in selected:
        pos["weight_pct"] = pos.pop("target_weight_pct", 0.0)

    return selected


def build_composite_baseline(
    archive_data: ArchiveData,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Build baseline portfolio: top-K dev-stage by composite_rank, equal weight.

    This represents the pre-decision-engine workflow: just take the best
    composite-ranked drug developers.
    """
    dev_rows: List[Tuple[int, str]] = []
    for row in archive_data.rows:
        ticker = row.get("ticker", "").strip().upper()
        archetype = archive_data.archetypes.get(ticker, "")
        if archetype != "drug_developer":
            continue
        rank_str = row.get("composite_rank", "").strip()
        if not rank_str:
            continue
        try:
            rank = int(rank_str)
        except ValueError:
            continue
        dev_rows.append((rank, ticker))

    dev_rows.sort(key=lambda x: x[0])
    selected = dev_rows[:top_k]

    if not selected:
        return []

    weight = round(100.0 / len(selected), 2)
    return [
        {"ticker": t, "composite_rank": r, "weight_pct": weight}
        for r, t in selected
    ]


def build_universe_baseline(
    archive_data: ArchiveData,
    ruleset: DecisionRuleset,
) -> List[Dict[str, Any]]:
    """Build universe baseline: all eligible dev-stage tickers, equal weight.

    This measures the broad eligible universe without tier/rank filtering.
    """
    eligible_tickers: List[str] = []
    for ticker in archive_data.tickers:
        archetype = archive_data.archetypes.get(ticker, "")
        if archetype != "drug_developer":
            continue
        rec = archive_data.recs.get(ticker)
        if rec is None:
            continue
        opt = archive_data.optionalities.get(ticker)
        fields = compute_decision_fields(rec, archetype, opt, ruleset=ruleset)
        if fields.get("eligible") == "1":
            eligible_tickers.append(ticker)

    if not eligible_tickers:
        return []

    weight = round(100.0 / len(eligible_tickers), 2)
    return [{"ticker": t, "weight_pct": weight} for t in eligible_tickers]


def build_all_dev_decisions(
    archive_data: ArchiveData,
    ruleset: DecisionRuleset,
) -> List[Dict[str, Any]]:
    """Compute decision fields for ALL dev-stage tickers (not just filtered portfolio).

    Returns one dict per dev-stage ticker with raw decision output.
    No filtering or sorting — includes ineligible tickers.
    """
    results: List[Dict[str, Any]] = []
    for ticker in archive_data.tickers:
        archetype = archive_data.archetypes.get(ticker, "")
        if archetype != "drug_developer":
            continue

        rec = archive_data.recs.get(ticker)
        if rec is None:
            continue

        opt = archive_data.optionalities.get(ticker)
        fields = compute_decision_fields(rec, archetype, opt, ruleset=ruleset)

        # Gate margin diagnostics (panel-only)
        margins = compute_gate_margins(rec, ruleset)
        tier_margins = compute_tier_margins(
            opt, fields.get("catalyst_strength", "missing"), ruleset
        )

        df = rec.get("defensive_features") or {}
        results.append({
            "ticker": ticker,
            "tier_dev": fields.get("tier_dev", ""),
            "band": fields.get("size_band", ""),
            "eligible": fields.get("eligible", "0"),
            "ineligible_reasons": fields.get("ineligible_reasons", ""),
            "first_failed_gate": margins.get("first_failed_gate", ""),
            "tier_reason": fields.get("tier_reason", ""),
            "risk_flags": fields.get("risk_flags", ""),
            "catalyst_mode": fields.get("catalyst_mode", ""),
            "catalyst_strength": fields.get("catalyst_strength", ""),
            "catalyst_days": fields.get("catalyst_days", ""),
            "mom_state": fields.get("mom_state", ""),
            "optionality": opt,
            "drawdown_abs": df.get("drawdown"),
            "drawdown_xbi": df.get("drawdown_xbi"),
            "drawdown_rel_xbi": df.get("drawdown_rel_xbi"),
            "dd_abs_margin": margins["dd_abs_margin"],
            "dd_rel_margin": margins["dd_rel_margin"],
            "rescued_by_rel": margins["rescued_by_rel"],
            "dd_rel_margin_rescued": fields.get("dd_rel_margin_rescued", "0"),
            "optionality_margin_a": tier_margins["optionality_margin_a"],
            "actionable_catalyst": tier_margins["actionable_catalyst"],
            "catalyst_tilt_mult": fields.get("catalyst_tilt_mult", 1.0),
            "catalyst_tilt_applied": fields.get("catalyst_tilt_applied", "0"),
        })

    return results


# Min trailing bars required for drawdown estimate (match run_screen.py)
_MIN_BARS_DRAWDOWN = 126
_DRAWDOWN_WINDOW = 252


def hydrate_archive_drawdown(
    archive_data: ArchiveData,
    csv_provider,
    as_of_date: str,
) -> None:
    """Compute trailing drawdown from price CSV and inject into archive recs.

    Populates defensive_features with: drawdown, drawdown_xbi, drawdown_rel_xbi.
    Modifies archive_data.recs in place.
    """
    from datetime import date as _date

    ref = _date.fromisoformat(as_of_date)

    def _trailing_dd(ticker: str) -> Optional[float]:
        prices = csv_provider._prices.get(ticker.upper())
        if not prices:
            return None
        # Sorted dates <= ref
        valid = sorted((d, float(p)) for d, p in prices.items() if d <= ref)
        if len(valid) < _MIN_BARS_DRAWDOWN:
            return None
        closes = [c for _, c in valid]
        look = closes[-_DRAWDOWN_WINDOW:] if len(closes) >= _DRAWDOWN_WINDOW else closes
        peak = max(look)
        if peak <= 0:
            return None
        return round((closes[-1] / peak) - 1.0, 6)

    xbi_dd = _trailing_dd("XBI")

    for ticker, rec in archive_data.recs.items():
        df = rec.get("defensive_features")
        if df is None:
            df = {}
            rec["defensive_features"] = df
        # Only hydrate if missing
        if df.get("drawdown") is None:
            dd = _trailing_dd(ticker)
            if dd is not None:
                df["drawdown"] = dd
                df["drawdown_missing_reason"] = ""
            else:
                df.setdefault("drawdown_missing_reason", "no_price_series")
        df["drawdown_xbi"] = xbi_dd
        dd = df.get("drawdown")
        if dd is not None and xbi_dd is not None:
            df["drawdown_rel_xbi"] = round(dd - xbi_dd, 6)
        else:
            df["drawdown_rel_xbi"] = None


def write_panel_csv(
    panel_rows: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Write the walk-forward panel CSV with stable column order."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PANEL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(panel_rows)


# =============================================================================
# PORTFOLIO EVALUATION
# =============================================================================

def evaluate_portfolio(
    positions: List[Dict[str, Any]],
    returns: MultiHorizonReturns,
    horizons: List[int],
) -> Dict[str, Any]:
    """Evaluate a portfolio against multi-horizon returns.

    Returns per-horizon metrics: weighted return, equal-weight return,
    hit rate, left tail (P5, P10, max loss), and per-position details.
    """
    result: Dict[str, Any] = {"n_positions": len(positions)}
    per_position: List[Dict[str, Any]] = []

    for h in horizons:
        raw_rets = returns.raw.get(h, {})
        resid_rets = returns.resid.get(h, {})

        # Match positions to returns
        matched_raw: List[Tuple[float, float]] = []  # (weight, return)
        matched_resid: List[Tuple[float, float]] = []
        position_returns: List[float] = []
        position_resid_returns: List[float] = []

        for pos in positions:
            ticker = pos["ticker"]
            weight = pos.get("weight_pct", 0.0)
            if isinstance(weight, str):
                try:
                    weight = float(weight)
                except (ValueError, TypeError):
                    weight = 0.0

            raw_ret = raw_rets.get(ticker)
            resid_ret = resid_rets.get(ticker)

            if raw_ret is not None:
                matched_raw.append((weight, raw_ret))
                position_returns.append(raw_ret)
            if resid_ret is not None:
                matched_resid.append((weight, resid_ret))
                position_resid_returns.append(resid_ret)

            # Store per-position data (first horizon pass only)
            if h == horizons[0]:
                pos_detail = {"ticker": ticker, "weight_pct": weight}
            else:
                # Find existing detail
                pos_detail = next(
                    (p for p in per_position if p["ticker"] == ticker), None
                )
                if pos_detail is None:
                    pos_detail = {"ticker": ticker, "weight_pct": weight}
                    per_position.append(pos_detail)

            pos_detail[f"fwd_{h}d"] = raw_ret
            pos_detail[f"resid_{h}d"] = resid_ret

            if h == horizons[0] and pos_detail not in per_position:
                per_position.append(pos_detail)

        # Coverage
        coverage = len(matched_raw)
        result[f"coverage_{h}d"] = coverage

        # Weighted return (renormalize weights for coverage)
        if matched_raw:
            total_weight = sum(w for w, _ in matched_raw)
            if total_weight > 0:
                weighted_ret = sum(
                    (w / total_weight) * r for w, r in matched_raw
                )
            else:
                weighted_ret = 0.0
            result[f"weighted_ret_{h}d_pct"] = round(weighted_ret * 100, 4)
        else:
            result[f"weighted_ret_{h}d_pct"] = None

        # Weighted residual return
        if matched_resid:
            total_weight_r = sum(w for w, _ in matched_resid)
            if total_weight_r > 0:
                weighted_resid = sum(
                    (w / total_weight_r) * r for w, r in matched_resid
                )
            else:
                weighted_resid = 0.0
            result[f"weighted_resid_{h}d_pct"] = round(weighted_resid * 100, 4)
        else:
            result[f"weighted_resid_{h}d_pct"] = None

        # Equal-weight return
        if position_returns:
            ew_ret = mean(position_returns)
            result[f"ew_ret_{h}d_pct"] = round(ew_ret * 100, 4)
        else:
            result[f"ew_ret_{h}d_pct"] = None

        # Equal-weight residual
        if position_resid_returns:
            ew_resid = mean(position_resid_returns)
            result[f"ew_resid_{h}d_pct"] = round(ew_resid * 100, 4)
        else:
            result[f"ew_resid_{h}d_pct"] = None

        # Hit rate (raw)
        if position_returns:
            hit_rate = sum(1 for r in position_returns if r > 0) / len(position_returns)
            result[f"hit_rate_{h}d_pct"] = round(hit_rate * 100, 2)
        else:
            result[f"hit_rate_{h}d_pct"] = None

        # Left tail (on raw returns as pct)
        if position_returns:
            ret_pcts = [r * 100 for r in position_returns]
            result[f"p5_{h}d_pct"] = round(_percentile(ret_pcts, 5), 2)
            result[f"p10_{h}d_pct"] = round(_percentile(ret_pcts, 10), 2)
            result[f"max_loss_{h}d_pct"] = round(min(ret_pcts), 2)
        else:
            result[f"p5_{h}d_pct"] = None
            result[f"p10_{h}d_pct"] = None
            result[f"max_loss_{h}d_pct"] = None

    result["per_position"] = per_position
    return result


# =============================================================================
# TURNOVER
# =============================================================================

def compute_portfolio_turnover(
    prev_positions: List[Dict[str, Any]],
    curr_positions: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute position and weight turnover between two snapshots.

    Position turnover: |symmetric_diff| / max(|prev|, |curr|)
    Weight turnover:   sum(|w_new - w_old|) / 2 for union of tickers
    """
    prev_tickers = {p["ticker"] for p in prev_positions}
    curr_tickers = {p["ticker"] for p in curr_positions}

    # Position turnover: fraction of names changed relative to union size
    union_size = len(prev_tickers | curr_tickers) or 1
    sym_diff = prev_tickers.symmetric_difference(curr_tickers)
    position_turnover = len(sym_diff) / union_size

    # Weight turnover
    prev_weights = {p["ticker"]: p.get("weight_pct", 0.0) for p in prev_positions}
    curr_weights = {p["ticker"]: p.get("weight_pct", 0.0) for p in curr_positions}

    all_tickers = prev_tickers | curr_tickers
    weight_turnover = sum(
        abs(
            (float(curr_weights.get(t, 0.0)) if curr_weights.get(t, 0.0) != "" else 0.0)
            - (float(prev_weights.get(t, 0.0)) if prev_weights.get(t, 0.0) != "" else 0.0)
        )
        for t in all_tickers
    ) / 2.0

    return {
        "position_turnover": round(position_turnover, 4),
        "weight_turnover_pct": round(weight_turnover, 2),
    }


# =============================================================================
# SNAPSHOT RESULT
# =============================================================================

class SnapshotResult(NamedTuple):
    """Result for one snapshot date."""
    date_str: str
    regime: str
    strategy_eval: Dict[str, Any]
    baseline_topk_eval: Dict[str, Any]
    baseline_universe_eval: Dict[str, Any]
    strategy_positions: List[Dict[str, Any]]
    baseline_topk_positions: List[Dict[str, Any]]
    baseline_universe_positions: List[Dict[str, Any]]
    turnover: Optional[Dict[str, float]]


# =============================================================================
# MAIN LOOP
# =============================================================================

def run_strategy_backtest(
    archives: List[Tuple[str, Path]],
    chained,
    csv_provider,
    ruleset: DecisionRuleset,
    tier_filter: List[str],
    top_k: int,
    horizons: List[int] = HORIZONS,
    emit_panel: bool = False,
) -> Tuple[List[SnapshotResult], List[Dict[str, Any]], Dict[str, Any]]:
    """Run the strategy backtest across all usable archives.

    Returns (results, panel_rows, cost_telemetry). panel_rows is populated only
    when emit_panel=True. cost_telemetry is populated only when emit_panel=True
    and aggregates coverage/cap-binding across all snapshots.
    """
    results: List[SnapshotResult] = []
    panel_rows: List[Dict[str, Any]] = []
    all_cost_estimates: List[CostEstimate] = []
    total_positions = 0
    prev_strategy: List[Dict[str, Any]] = []

    # Compute as-of fence
    snapshot_dates = [d for d, _ in archives]
    last_date = csv_provider.get_last_date()
    max_horizon = max(horizons)
    usable_dates, fence_skipped = compute_as_of_fence(
        snapshot_dates, last_date, max_horizon
    )
    usable_set = set(usable_dates)

    print(f"  Archives: {len(archives)} total, {len(usable_dates)} usable "
          f"(max horizon={max_horizon}d)")
    if fence_skipped:
        print(f"  Skipped {len(fence_skipped)} archives past return fence")

    for i, (date_str, tar_path) in enumerate(archives):
        if date_str not in usable_set:
            continue

        print(f"  [{i+1}/{len(archives)}] {date_str} ... ", end="", flush=True)

        # Load archive data
        archive_data = load_archive_data(tar_path, date_str)

        # Hydrate drawdown from price CSV (archives don't store it)
        hydrate_archive_drawdown(archive_data, csv_provider, date_str)

        # Compute multi-horizon returns
        mh_returns = compute_multi_horizon_returns(
            chained, csv_provider, archive_data.tickers, date_str, horizons
        )

        # Build portfolios
        strategy_positions = build_strategy_portfolio(
            archive_data, ruleset, tier_filter, top_k
        )
        baseline_topk_positions = build_composite_baseline(
            archive_data, top_k
        )
        baseline_universe_positions = build_universe_baseline(
            archive_data, ruleset
        )

        # Panel: all dev-stage decisions + forward returns + forward max-DD
        if emit_panel:
            all_devs = build_all_dev_decisions(archive_data, ruleset)

            # Build weight map from strategy positions
            weight_map: Dict[str, float] = {}
            for pos in strategy_positions:
                weight_map[pos["ticker"]] = pos.get("weight_pct", 0.0)

            # Cost schedule: use ruleset cap when cost haircut is enabled
            if ruleset.enable_cost_haircut:
                cost_schedule = CostSchedule(
                    impact_cap_bps=ruleset.cost_impact_cap_bps,
                )
            else:
                cost_schedule = DEFAULT_SCHEDULE
            cost_estimates: List[CostEstimate] = []

            for dev in all_devs:
                ticker = dev["ticker"]
                fwd_20 = mh_returns.raw.get(20, {}).get(ticker)
                fwd_60 = mh_returns.raw.get(60, {}).get(ticker)

                dd_20, dd_20_reason = csv_provider.get_forward_max_drawdown(
                    ticker, date_str, 20
                )
                dd_60, dd_60_reason = csv_provider.get_forward_max_drawdown(
                    ticker, date_str, 60
                )
                # Use the longer-horizon reason as representative
                dd_reason = dd_60_reason or dd_20_reason

                # Optionality + catalyst_days as raw values for re-tiering
                opt_val = dev.get("optionality")
                opt_str = round(opt_val, 6) if opt_val is not None else ""
                cat_days = dev.get("catalyst_days", "")

                # Cost model: ADV from market_data, cost only for portfolio positions
                md = archive_data.market_data.get(ticker, {})
                adv = (md.get("avg_volume") or 0) * (md.get("price") or 0)
                weight = weight_map.get(ticker, 0.0)
                if weight > 0 and adv > 0:
                    cost = estimate_trade_cost(weight, adv, cost_schedule)
                    est_cost_bps = cost.round_trip_bps
                    participation = cost.participation_pct
                    cost_estimates.append(cost)
                else:
                    est_cost_bps = 0.0
                    participation = 0.0

                # Net returns: gross minus round-trip cost (bps→%)
                cost_pct = est_cost_bps / 100.0  # bps to percentage points
                fwd_20_pct = round(fwd_20 * 100, 4) if fwd_20 is not None else ""
                fwd_60_pct = round(fwd_60 * 100, 4) if fwd_60 is not None else ""
                fwd_20_net = round(fwd_20_pct - cost_pct, 4) if fwd_20_pct != "" else ""
                fwd_60_net = round(fwd_60_pct - cost_pct, 4) if fwd_60_pct != "" else ""

                dd_abs = dev.get("drawdown_abs")
                dd_xbi = dev.get("drawdown_xbi")
                dd_rel = dev.get("drawdown_rel_xbi")
                panel_rows.append({
                    "as_of_date": date_str,
                    "ticker": ticker,
                    "tier": dev["tier_dev"],
                    "band": dev["band"],
                    "eligible": dev["eligible"],
                    "weight": weight_map.get(ticker, 0.0),
                    "ineligible_reasons": dev.get("ineligible_reasons", ""),
                    "first_failed_gate": dev.get("first_failed_gate", ""),
                    "tier_reason": dev["tier_reason"],
                    "risk_flags": dev["risk_flags"],
                    "catalyst_mode": dev["catalyst_mode"],
                    "catalyst_strength": dev.get("catalyst_strength", ""),
                    "mom_state": dev["mom_state"],
                    "optionality": opt_str,
                    "catalyst_days_raw": cat_days,
                    "ruleset_id": ruleset.ruleset_id,
                    "drawdown_abs": round(dd_abs, 4) if dd_abs is not None else "",
                    "drawdown_xbi": round(dd_xbi, 4) if dd_xbi is not None else "",
                    "drawdown_rel_xbi": round(dd_rel, 4) if dd_rel is not None else "",
                    "dd_abs_margin": round(dev["dd_abs_margin"], 4) if isinstance(dev.get("dd_abs_margin"), (int, float)) else "",
                    "dd_rel_margin": round(dev["dd_rel_margin"], 4) if isinstance(dev.get("dd_rel_margin"), (int, float)) else "",
                    "rescued_by_rel": "1" if dev.get("rescued_by_rel") else "0",
                    "dd_rel_margin_rescued": dev.get("dd_rel_margin_rescued", "0"),
                    "optionality_margin_a": round(dev["optionality_margin_a"], 4) if isinstance(dev.get("optionality_margin_a"), (int, float)) else "",
                    "actionable_catalyst": "1" if dev.get("actionable_catalyst") else "0",
                    "fwd_ret_20d": fwd_20_pct,
                    "fwd_ret_60d": fwd_60_pct,
                    "fwd_max_dd_20d": round(dd_20 * 100, 4) if dd_20 is not None else "",
                    "fwd_max_dd_60d": round(dd_60 * 100, 4) if dd_60 is not None else "",
                    "fwd_dd_missing_reason": dd_reason,
                    "adv_dollars": round(adv, 2),
                    "est_cost_bps": round(est_cost_bps, 2),
                    "participation_pct": round(participation, 4),
                    "fwd_ret_20d_net": fwd_20_net,
                    "fwd_ret_60d_net": fwd_60_net,
                })

            # Accumulate cost telemetry across snapshots
            all_cost_estimates.extend(cost_estimates)
            total_positions += len(strategy_positions)

        # Evaluate
        strategy_eval = evaluate_portfolio(
            strategy_positions, mh_returns, horizons
        )
        baseline_topk_eval = evaluate_portfolio(
            baseline_topk_positions, mh_returns, horizons
        )
        baseline_universe_eval = evaluate_portfolio(
            baseline_universe_positions, mh_returns, horizons
        )

        # Turnover
        turnover = None
        if prev_strategy:
            turnover = compute_portfolio_turnover(prev_strategy, strategy_positions)
        prev_strategy = strategy_positions

        n_strat = strategy_eval.get("n_positions", 0)
        w60 = strategy_eval.get("weighted_resid_60d_pct")
        w60_str = f"{w60:+.2f}%" if w60 is not None else "n/a"
        print(f"n={n_strat}, regime={mh_returns.regime}, "
              f"strat_resid_60d={w60_str}")

        results.append(SnapshotResult(
            date_str=date_str,
            regime=mh_returns.regime,
            strategy_eval=strategy_eval,
            baseline_topk_eval=baseline_topk_eval,
            baseline_universe_eval=baseline_universe_eval,
            strategy_positions=strategy_positions,
            baseline_topk_positions=baseline_topk_positions,
            baseline_universe_positions=baseline_universe_positions,
            turnover=turnover,
        ))

    # Compute aggregate cost telemetry
    if emit_panel and total_positions > 0:
        cost_sched = (
            CostSchedule(impact_cap_bps=ruleset.cost_impact_cap_bps)
            if ruleset.enable_cost_haircut
            else DEFAULT_SCHEDULE
        )
        cost_telem = compute_cost_telemetry(
            total_positions, all_cost_estimates, cost_sched
        )
    else:
        cost_telem = {}

    return results, panel_rows, cost_telem


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_results(
    results: List[SnapshotResult],
    horizons: List[int] = HORIZONS,
) -> Dict[str, Any]:
    """Aggregate snapshot results into pooled statistics."""
    if not results:
        return {}

    agg: Dict[str, Any] = {
        "n_snapshots": len(results),
        "date_range": f"{results[0].date_str} to {results[-1].date_str}",
    }

    # Collect per-portfolio per-horizon metrics
    for portfolio_key in ["strategy", "baseline_topk", "baseline_universe"]:
        for h in horizons:
            for metric in ["weighted_ret", "weighted_resid", "ew_ret", "ew_resid"]:
                field = f"{metric}_{h}d_pct"
                eval_key = f"{portfolio_key}_eval"

                vals = []
                for r in results:
                    ev = getattr(r, eval_key)
                    v = ev.get(field)
                    if v is not None:
                        vals.append(v)

                if vals:
                    agg[f"{portfolio_key}_{field}_mean"] = round(mean(vals), 4)
                    agg[f"{portfolio_key}_{field}_median"] = round(median(vals), 4)
                    winsorized = _winsorize(vals)
                    agg[f"{portfolio_key}_{field}_winsor"] = round(mean(winsorized), 4)
                else:
                    agg[f"{portfolio_key}_{field}_mean"] = None
                    agg[f"{portfolio_key}_{field}_median"] = None
                    agg[f"{portfolio_key}_{field}_winsor"] = None

            # Hit rate
            hr_field = f"hit_rate_{h}d_pct"
            hr_vals = []
            for r in results:
                ev = getattr(r, f"{portfolio_key}_eval")
                v = ev.get(hr_field)
                if v is not None:
                    hr_vals.append(v)
            if hr_vals:
                agg[f"{portfolio_key}_{hr_field}_mean"] = round(mean(hr_vals), 2)
            else:
                agg[f"{portfolio_key}_{hr_field}_mean"] = None

    # Strategy-vs-baseline spread (per date, then average)
    for h in horizons:
        spreads_topk: List[float] = []
        spreads_univ: List[float] = []
        for r in results:
            strat_v = r.strategy_eval.get(f"weighted_resid_{h}d_pct")
            topk_v = r.baseline_topk_eval.get(f"ew_resid_{h}d_pct")
            univ_v = r.baseline_universe_eval.get(f"ew_resid_{h}d_pct")
            if strat_v is not None and topk_v is not None:
                spreads_topk.append(strat_v - topk_v)
            if strat_v is not None and univ_v is not None:
                spreads_univ.append(strat_v - univ_v)

        if spreads_topk:
            agg[f"spread_vs_topk_{h}d_mean"] = round(mean(spreads_topk), 4)
            agg[f"spread_vs_topk_{h}d_median"] = round(median(spreads_topk), 4)
        else:
            agg[f"spread_vs_topk_{h}d_mean"] = None
            agg[f"spread_vs_topk_{h}d_median"] = None

        if spreads_univ:
            agg[f"spread_vs_universe_{h}d_mean"] = round(mean(spreads_univ), 4)
            agg[f"spread_vs_universe_{h}d_median"] = round(median(spreads_univ), 4)
        else:
            agg[f"spread_vs_universe_{h}d_mean"] = None
            agg[f"spread_vs_universe_{h}d_median"] = None

    # Per-regime split
    regime_results: Dict[str, List[SnapshotResult]] = {}
    for r in results:
        regime_results.setdefault(r.regime, []).append(r)

    regime_stats: Dict[str, Dict[str, Any]] = {}
    for regime, regime_snapshots in sorted(regime_results.items()):
        rs: Dict[str, Any] = {"n_snapshots": len(regime_snapshots)}
        for h in horizons:
            for metric in ["weighted_resid"]:
                field = f"{metric}_{h}d_pct"
                vals = [
                    snap.strategy_eval.get(field)
                    for snap in regime_snapshots
                    if snap.strategy_eval.get(field) is not None
                ]
                if vals:
                    rs[f"strategy_{field}_mean"] = round(mean(vals), 4)
                else:
                    rs[f"strategy_{field}_mean"] = None

                bvals = [
                    snap.baseline_topk_eval.get(f"ew_resid_{h}d_pct")
                    for snap in regime_snapshots
                    if snap.baseline_topk_eval.get(f"ew_resid_{h}d_pct") is not None
                ]
                if bvals:
                    rs[f"baseline_topk_ew_resid_{h}d_pct_mean"] = round(mean(bvals), 4)
                else:
                    rs[f"baseline_topk_ew_resid_{h}d_pct_mean"] = None

        regime_stats[regime] = rs
    agg["per_regime"] = regime_stats

    # Turnover
    turnover_pos = [r.turnover["position_turnover"]
                    for r in results if r.turnover is not None]
    turnover_wt = [r.turnover["weight_turnover_pct"]
                   for r in results if r.turnover is not None]
    agg["mean_position_turnover"] = round(mean(turnover_pos), 4) if turnover_pos else None
    agg["mean_weight_turnover_pct"] = round(mean(turnover_wt), 2) if turnover_wt else None

    # Top tickers by frequency across snapshots
    ticker_counter: Counter = Counter()
    for r in results:
        for pos in r.strategy_positions:
            ticker_counter[pos["ticker"]] += 1

    top_tickers = ticker_counter.most_common(20)
    agg["top_tickers"] = [
        {"ticker": t, "appearances": c, "pct_of_snapshots": round(c / len(results) * 100, 1)}
        for t, c in top_tickers
    ]
    agg["unique_ticker_count"] = len(ticker_counter)

    # Mean position count
    pos_counts = [r.strategy_eval.get("n_positions", 0) for r in results]
    agg["mean_n_positions"] = round(mean(pos_counts), 1) if pos_counts else 0

    return agg


# =============================================================================
# OUTPUT
# =============================================================================

def write_portfolio_csv(
    results: List[SnapshotResult],
    output_path: Path,
    horizons: List[int] = HORIZONS,
) -> None:
    """Write per-date per-ticker portfolio CSV."""
    fieldnames = [
        "date", "ticker", "tier_dev", "actionable_rank", "weight_pct",
        "size_band", "catalyst_mode", "mom_state",
    ]
    for h in horizons:
        fieldnames.extend([f"fwd_{h}d", f"resid_{h}d"])

    rows: List[Dict[str, Any]] = []
    for r in results:
        for pos in r.strategy_eval.get("per_position", []):
            row: Dict[str, Any] = {"date": r.date_str}
            # Find position metadata from strategy_positions
            meta = next(
                (p for p in r.strategy_positions if p["ticker"] == pos["ticker"]),
                {},
            )
            row["ticker"] = pos["ticker"]
            row["tier_dev"] = meta.get("tier_dev", "")
            row["actionable_rank"] = meta.get("actionable_rank", "")
            row["weight_pct"] = pos.get("weight_pct", "")
            row["size_band"] = meta.get("size_band", "")
            row["catalyst_mode"] = meta.get("catalyst_mode", "")
            row["mom_state"] = meta.get("mom_state", "")
            for h in horizons:
                val = pos.get(f"fwd_{h}d")
                row[f"fwd_{h}d"] = round(val * 100, 4) if val is not None else ""
                rval = pos.get(f"resid_{h}d")
                row[f"resid_{h}d"] = round(rval * 100, 4) if rval is not None else ""
            rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(
    results: List[SnapshotResult],
    agg: Dict[str, Any],
    output_path: Path,
    horizons: List[int] = HORIZONS,
) -> None:
    """Write per-date summary CSV with a POOLED row."""
    fieldnames = ["date", "regime", "n_positions"]
    for h in horizons:
        fieldnames.extend([
            f"strategy_weighted_resid_{h}d_pct",
            f"strategy_ew_resid_{h}d_pct",
            f"baseline_topk_ew_resid_{h}d_pct",
            f"baseline_universe_ew_resid_{h}d_pct",
            f"spread_vs_topk_{h}d",
            f"hit_rate_{h}d_pct",
        ])
    fieldnames.extend(["position_turnover", "weight_turnover_pct"])

    rows: List[Dict[str, Any]] = []
    for r in results:
        row: Dict[str, Any] = {
            "date": r.date_str,
            "regime": r.regime,
            "n_positions": r.strategy_eval.get("n_positions", 0),
        }
        for h in horizons:
            row[f"strategy_weighted_resid_{h}d_pct"] = r.strategy_eval.get(
                f"weighted_resid_{h}d_pct", ""
            )
            row[f"strategy_ew_resid_{h}d_pct"] = r.strategy_eval.get(
                f"ew_resid_{h}d_pct", ""
            )
            row[f"baseline_topk_ew_resid_{h}d_pct"] = r.baseline_topk_eval.get(
                f"ew_resid_{h}d_pct", ""
            )
            row[f"baseline_universe_ew_resid_{h}d_pct"] = r.baseline_universe_eval.get(
                f"ew_resid_{h}d_pct", ""
            )
            strat_v = r.strategy_eval.get(f"weighted_resid_{h}d_pct")
            topk_v = r.baseline_topk_eval.get(f"ew_resid_{h}d_pct")
            if strat_v is not None and topk_v is not None:
                row[f"spread_vs_topk_{h}d"] = round(strat_v - topk_v, 4)
            else:
                row[f"spread_vs_topk_{h}d"] = ""
            row[f"hit_rate_{h}d_pct"] = r.strategy_eval.get(
                f"hit_rate_{h}d_pct", ""
            )
        if r.turnover:
            row["position_turnover"] = r.turnover["position_turnover"]
            row["weight_turnover_pct"] = r.turnover["weight_turnover_pct"]
        else:
            row["position_turnover"] = ""
            row["weight_turnover_pct"] = ""
        rows.append(row)

    # POOLED row
    pooled: Dict[str, Any] = {
        "date": "POOLED",
        "regime": "",
        "n_positions": agg.get("mean_n_positions", ""),
    }
    for h in horizons:
        pooled[f"strategy_weighted_resid_{h}d_pct"] = agg.get(
            f"strategy_weighted_resid_{h}d_pct_mean", ""
        )
        pooled[f"strategy_ew_resid_{h}d_pct"] = agg.get(
            f"strategy_ew_resid_{h}d_pct_mean", ""
        )
        pooled[f"baseline_topk_ew_resid_{h}d_pct"] = agg.get(
            f"baseline_topk_ew_resid_{h}d_pct_mean", ""
        )
        pooled[f"baseline_universe_ew_resid_{h}d_pct"] = agg.get(
            f"baseline_universe_ew_resid_{h}d_pct_mean", ""
        )
        pooled[f"spread_vs_topk_{h}d"] = agg.get(
            f"spread_vs_topk_{h}d_mean", ""
        )
        pooled[f"hit_rate_{h}d_pct"] = agg.get(
            f"strategy_hit_rate_{h}d_pct_mean", ""
        )
    pooled["position_turnover"] = agg.get("mean_position_turnover", "")
    pooled["weight_turnover_pct"] = agg.get("mean_weight_turnover_pct", "")
    rows.append(pooled)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_details_json(
    results: List[SnapshotResult],
    agg: Dict[str, Any],
    config: Dict[str, Any],
    output_path: Path,
) -> None:
    """Write full nested JSON details."""
    per_date = []
    for r in results:
        per_date.append({
            "date": r.date_str,
            "regime": r.regime,
            "strategy": {
                "n_positions": r.strategy_eval.get("n_positions", 0),
                "metrics": {k: v for k, v in r.strategy_eval.items()
                            if k not in ("per_position", "n_positions")},
                "positions": [
                    {k: v for k, v in p.items()}
                    for p in r.strategy_positions
                ],
            },
            "baseline_topk": {
                "n_positions": r.baseline_topk_eval.get("n_positions", 0),
                "metrics": {k: v for k, v in r.baseline_topk_eval.items()
                            if k not in ("per_position", "n_positions")},
            },
            "baseline_universe": {
                "n_positions": r.baseline_universe_eval.get("n_positions", 0),
                "metrics": {k: v for k, v in r.baseline_universe_eval.items()
                            if k not in ("per_position", "n_positions")},
            },
            "turnover": r.turnover,
        })

    output = {
        "metadata": config,
        "per_date": per_date,
        "aggregate": agg,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)


def generate_report(
    agg: Dict[str, Any],
    config: Dict[str, Any],
    results: List[SnapshotResult],
    output_path: Path,
    horizons: List[int] = HORIZONS,
) -> None:
    """Generate narrative text report."""
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("STRATEGY-LEVEL DECISION ENGINE BACKTEST")
    lines.append("=" * 72)
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    # Section 1: Configuration
    lines.append("1. CONFIGURATION")
    lines.append("-" * 40)
    lines.append(f"  Ruleset ID:    {config.get('ruleset_id', 'default')}")
    lines.append(f"  Tier filter:   {config.get('tier_filter', [])}")
    lines.append(f"  Top-K:         {config.get('top_k', 20)}")
    lines.append(f"  Horizons:      {config.get('horizons', HORIZONS)}")
    lines.append(f"  Archives used: {agg.get('n_snapshots', 0)}")
    lines.append(f"  Date range:    {agg.get('date_range', 'n/a')}")
    lines.append(f"  Avg positions: {agg.get('mean_n_positions', 0)}")
    lines.append(f"  Unique tickers: {agg.get('unique_ticker_count', 0)}")
    lines.append("")

    # Section 2: Headline results table
    lines.append("2. HEADLINE RESULTS (mean across snapshots, %)")
    lines.append("-" * 40)

    hdr = (f"  {'Portfolio':<22} "
           + " ".join(f"{'resid_'+str(h)+'d':>12}" for h in horizons)
           + f" {'hit_60d':>8}")
    lines.append(hdr)
    lines.append(f"  {'-'*22} " + " ".join(f"{'-'*12}" for _ in horizons) + f" {'-'*8}")

    for label, key_prefix in [
        ("Strategy (weighted)", "strategy_weighted_resid"),
        ("Strategy (EW)", "strategy_ew_resid"),
        ("Baseline top-K (EW)", "baseline_topk_ew_resid"),
        ("Baseline universe(EW)", "baseline_universe_ew_resid"),
    ]:
        vals = []
        for h in horizons:
            v = agg.get(f"{key_prefix}_{h}d_pct_mean")
            vals.append(f"{v:>+12.2f}" if v is not None else f"{'n/a':>12}")
        hit_v = agg.get(f"{key_prefix.split('_resid')[0]}_hit_rate_60d_pct_mean")
        hit_str = f"{hit_v:>8.1f}" if hit_v is not None else f"{'n/a':>8}"
        lines.append(f"  {label:<22} " + " ".join(vals) + f" {hit_str}")

    lines.append("")

    # Strategy-vs-baseline spread
    lines.append("  Spread vs baseline (strategy weighted resid - baseline EW resid):")
    for h in horizons:
        sp_topk = agg.get(f"spread_vs_topk_{h}d_mean")
        sp_univ = agg.get(f"spread_vs_universe_{h}d_mean")
        sp_topk_str = f"{sp_topk:+.2f}%" if sp_topk is not None else "n/a"
        sp_univ_str = f"{sp_univ:+.2f}%" if sp_univ is not None else "n/a"
        lines.append(f"    {h}d: vs top-K={sp_topk_str}, vs universe={sp_univ_str}")
    lines.append("")

    # Winsorized results
    lines.append("  Winsorized means (5% each tail):")
    for label, key_prefix in [
        ("Strategy (weighted)", "strategy_weighted_resid"),
        ("Baseline top-K (EW)", "baseline_topk_ew_resid"),
    ]:
        vals = []
        for h in horizons:
            v = agg.get(f"{key_prefix}_{h}d_pct_winsor")
            vals.append(f"{v:>+12.2f}" if v is not None else f"{'n/a':>12}")
        lines.append(f"    {label:<22} " + " ".join(vals))
    lines.append("")

    # Section 3: Regime split
    lines.append("3. REGIME SPLIT")
    lines.append("-" * 40)
    per_regime = agg.get("per_regime", {})
    for regime, rs in sorted(per_regime.items()):
        lines.append(f"  {regime} (n={rs.get('n_snapshots', 0)}):")
        for h in horizons:
            s_v = rs.get(f"strategy_weighted_resid_{h}d_pct_mean")
            b_v = rs.get(f"baseline_topk_ew_resid_{h}d_pct_mean")
            s_str = f"{s_v:+.2f}%" if s_v is not None else "n/a"
            b_str = f"{b_v:+.2f}%" if b_v is not None else "n/a"
            lines.append(f"    {h}d: strategy={s_str}, baseline_topk={b_str}")
    lines.append("")

    # Section 4: Turnover
    lines.append("4. TURNOVER")
    lines.append("-" * 40)
    pos_to = agg.get("mean_position_turnover")
    wt_to = agg.get("mean_weight_turnover_pct")
    lines.append(f"  Mean position turnover: "
                 f"{pos_to:.1%}" if pos_to is not None else "  Mean position turnover: n/a")
    lines.append(f"  Mean weight turnover:   "
                 f"{wt_to:.1f}%" if wt_to is not None else "  Mean weight turnover:   n/a")
    lines.append("")

    # Section 5: Top positions
    lines.append("5. MOST FREQUENT POSITIONS")
    lines.append("-" * 40)
    top_tickers = agg.get("top_tickers", [])
    lines.append(f"  {'Ticker':<8} {'Appearances':>12} {'% of snapshots':>15}")
    lines.append(f"  {'-'*8} {'-'*12} {'-'*15}")
    for tt in top_tickers[:15]:
        lines.append(f"  {tt['ticker']:<8} {tt['appearances']:>12} "
                     f"{tt['pct_of_snapshots']:>14.1f}%")
    lines.append("")

    # Section 6: Per-date detail
    lines.append("6. PER-DATE SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  {'Date':<12} {'Regime':<9} {'N':>3} "
                 f"{'Strat_60d':>10} {'TopK_60d':>10} {'Spread':>8} {'PTO':>6}")
    lines.append(f"  {'-'*12} {'-'*9} {'-'*3} {'-'*10} {'-'*10} {'-'*8} {'-'*6}")
    for r in results:
        s60 = r.strategy_eval.get("weighted_resid_60d_pct")
        b60 = r.baseline_topk_eval.get("ew_resid_60d_pct")
        spread = (s60 - b60) if s60 is not None and b60 is not None else None
        pto = r.turnover["position_turnover"] if r.turnover else None

        parts = [f"  {r.date_str:<12} {r.regime:<9} "
                 f"{r.strategy_eval.get('n_positions', 0):>3}"]
        parts.append(f" {s60:>+10.2f}" if s60 is not None else f" {'n/a':>10}")
        parts.append(f" {b60:>+10.2f}" if b60 is not None else f" {'n/a':>10}")
        parts.append(f" {spread:>+8.2f}" if spread is not None else f" {'n/a':>8}")
        parts.append(f" {pto:>6.1%}" if pto is not None else f" {'n/a':>6}")
        lines.append("".join(parts))
    lines.append("")

    # Section 7: Caveats
    lines.append("7. CAVEATS")
    lines.append("-" * 40)
    lines.append("  - Forward returns use Morningstar+CSV fallback (price return, no dividends).")
    lines.append("  - Residual returns subtract beta*XBI to isolate alpha.")
    lines.append("  - Strategy uses size-band weighting; baselines use equal weight.")
    lines.append("  - Small sample sizes: interpret spreads directionally, not as precise estimates.")
    lines.append("  - Turnover assumes monthly rebalance frequency (archive cadence).")
    lines.append("  - No transaction costs, slippage, or capacity constraints modeled.")
    lines.append("")
    lines.append("=" * 72)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Strategy-Level Decision Engine Backtest"
    )
    parser.add_argument(
        "--ruleset", type=str, default=None,
        help="DecisionRuleset JSON path (default: built-in default)",
    )
    parser.add_argument(
        "--tier-filter", type=str, default="A,B",
        help="Comma-separated tiers to include (default: A,B)",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Max portfolio positions (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Archive start date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="Archive end date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--emit-panel", type=str, default=None,
        help="Write per-ticker walk-forward panel CSV to PATH",
    )
    parser.add_argument(
        "--archive-dir", type=str, default=None,
        help=f"Archive directory (default: {ARCHIVE_DIR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config + archive count, exit",
    )
    args = parser.parse_args()

    # Load ruleset
    if args.ruleset:
        ruleset = DecisionRuleset.from_json(args.ruleset)
        print(f"Loaded ruleset from {args.ruleset}")
    else:
        ruleset = DEFAULT_RULESET
        print(f"Using default ruleset")

    tier_filter = [t.strip().upper() for t in args.tier_filter.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "ruleset_id": ruleset.ruleset_id,
        "tier_filter": tier_filter,
        "top_k": args.top_k,
        "horizons": HORIZONS,
        "start": args.start,
        "end": args.end,
        "generated": datetime.now().isoformat(timespec="seconds"),
    }

    print(f"\nStrategy Backtest Configuration:")
    print(f"  Ruleset ID:  {ruleset.ruleset_id}")
    print(f"  Tier filter: {tier_filter}")
    print(f"  Top-K:       {args.top_k}")
    print(f"  Horizons:    {HORIZONS}")

    # Discover archives
    arch_dir = Path(args.archive_dir) if args.archive_dir else ARCHIVE_DIR
    archives = discover_archives(arch_dir, start=args.start, end=args.end)
    print(f"  Archives:    {len(archives)} found")

    if args.dry_run:
        print("\n[DRY RUN] Exiting before computation.")
        for d, p in archives:
            print(f"    {d}  {p.name}")
        return

    if not archives:
        print("ERROR: No archives found. Check --start/--end or archive directory.")
        sys.exit(1)

    # Initialize providers
    print("\nInitializing returns providers ...")
    chained, _ms_provider, csv_provider = init_providers()

    # Run backtest
    print("\nRunning strategy backtest ...")
    results, panel_rows, cost_telemetry = run_strategy_backtest(
        archives=archives,
        chained=chained,
        csv_provider=csv_provider,
        ruleset=ruleset,
        tier_filter=tier_filter,
        top_k=args.top_k,
        horizons=HORIZONS,
        emit_panel=bool(args.emit_panel),
    )

    if not results:
        print("ERROR: No usable snapshots produced results.")
        sys.exit(1)

    # Aggregate
    print("\nAggregating results ...")
    agg = aggregate_results(results, HORIZONS)

    # Write outputs
    portfolio_csv = output_dir / "decision_portfolio.csv"
    summary_csv = output_dir / "strategy_backtest_summary.csv"
    details_json = output_dir / "strategy_backtest_details.json"
    report_txt = output_dir / "strategy_backtest_report.txt"

    write_portfolio_csv(results, portfolio_csv, HORIZONS)
    print(f"  Wrote {portfolio_csv}")

    write_summary_csv(results, agg, summary_csv, HORIZONS)
    print(f"  Wrote {summary_csv}")

    if cost_telemetry:
        config["cost_telemetry"] = cost_telemetry
    write_details_json(results, agg, config, details_json)
    print(f"  Wrote {details_json}")

    generate_report(agg, config, results, report_txt, HORIZONS)
    print(f"  Wrote {report_txt}")

    if args.emit_panel and panel_rows:
        panel_path = Path(args.emit_panel)
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        write_panel_csv(panel_rows, panel_path)
        print(f"  Wrote panel CSV: {panel_path} ({len(panel_rows)} rows)")

    # Print headline
    print("\n" + "=" * 60)
    print("HEADLINE RESULTS")
    print("=" * 60)
    for h in HORIZONS:
        s = agg.get(f"strategy_weighted_resid_{h}d_pct_mean")
        b = agg.get(f"baseline_topk_ew_resid_{h}d_pct_mean")
        sp = agg.get(f"spread_vs_topk_{h}d_mean")
        s_str = f"{s:+.2f}%" if s is not None else "n/a"
        b_str = f"{b:+.2f}%" if b is not None else "n/a"
        sp_str = f"{sp:+.2f}%" if sp is not None else "n/a"
        print(f"  {h}d: strategy={s_str}, baseline={b_str}, spread={sp_str}")

    pos_to = agg.get("mean_position_turnover")
    wt_to = agg.get("mean_weight_turnover_pct")
    if pos_to is not None:
        print(f"  Turnover: position={pos_to:.1%}, weight={wt_to:.1f}%")
    print(f"  Snapshots: {agg.get('n_snapshots', 0)}, "
          f"Unique tickers: {agg.get('unique_ticker_count', 0)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
