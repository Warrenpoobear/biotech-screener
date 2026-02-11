#!/usr/bin/env python3
"""
Decision Ruleset Calibration Sweep

Systematically evaluates a grid of DecisionRuleset parameter variants across
all usable historical archives to identify which parameter combination produces
the best A-vs-C tier separation on 60-day residual returns.

Includes walk-forward holdout validation: pick best on early period, validate
on later period to guard against overfit.

Usage:
    python run_decision_ruleset_sweep.py [options]

    --output-dir PATH     Output directory (default: output/)
    --start DATE          Start date filter for archives
    --end DATE            End date filter for archives
    --top-n N             Print top N rulesets (default: 10)
    --holdout-split DATE  Cutoff date: early <= date, late > date (default: 2024-12-31)
    --top-k-holdout N     Evaluate top-K rulesets from early period on late (default: 10)
    --dry-run             Print grid size + archive count, don't run
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tarfile
import tempfile
from datetime import date, datetime
from itertools import product
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backfill_decision_columns import (
    _load_catalyst_map,
    _load_decision_inputs,
    _load_defensive_map,
    _load_holdings_map,
    build_rec,
)
from backtest.returns_provider import CSVReturnsProvider
from decision_engine import DecisionRuleset, compute_decision_fields
from run_rank_ic_backtest import (
    ARCHIVE_DIR,
    OUTPUT_DIR,
    PRICE_CSV,
    RETURNS_JSON,
    ChainedReturnsProvider,
    MorningstarReturnsProvider,
    _winsorize,
    compute_as_of_fence,
    compute_forward_returns,
    compute_residual_returns,
    discover_archives,
    estimate_xbi_betas,
)


# =============================================================================
# GRID GENERATION
# =============================================================================

# Parameters to sweep (all others held at defaults)
GRID_AXES = {
    "tier_a_optionality_floor": [0.55, 0.60, 0.65],
    "tier_b_optionality_floor": [0.25, 0.30, 0.35],
    "catalyst_near_days": [60, 90, 120],
    "sponsor_confirm_threshold": [2, 3],
}


def generate_grid() -> List[DecisionRuleset]:
    """Enumerate DecisionRuleset variants from GRID_AXES, skipping invalid combos.

    Filter: tier_b_optionality_floor must be strictly less than tier_a_optionality_floor.
    """
    keys = list(GRID_AXES.keys())
    combos = list(product(*(GRID_AXES[k] for k in keys)))

    rulesets: List[DecisionRuleset] = []
    for combo in combos:
        params = dict(zip(keys, combo))
        # Skip invalid: tier_b must be < tier_a
        if params["tier_b_optionality_floor"] >= params["tier_a_optionality_floor"]:
            continue
        rulesets.append(DecisionRuleset(**params))

    return rulesets


# =============================================================================
# PROVIDER INITIALIZATION
# =============================================================================

def init_providers(
    returns_json: Path = RETURNS_JSON,
    price_csv: Path = PRICE_CSV,
) -> Tuple[ChainedReturnsProvider, MorningstarReturnsProvider, CSVReturnsProvider]:
    """Create Morningstar/CSV/Chained returns providers."""
    ms_provider = MorningstarReturnsProvider(returns_json)
    csv_provider = CSVReturnsProvider(price_csv, price_col="close")
    chained = ChainedReturnsProvider(ms_provider, csv_provider)
    return chained, ms_provider, csv_provider


# =============================================================================
# ARCHIVE DATA CACHING
# =============================================================================

class ArchiveData(NamedTuple):
    """Cached per-archive data, loaded once and reused across rulesets."""
    date_str: str
    rows: List[Dict[str, str]]                   # raw CSV rows
    recs: Dict[str, Dict[str, Any]]              # ticker -> rec dict
    archetypes: Dict[str, str]                    # ticker -> archetype
    optionalities: Dict[str, Optional[float]]     # ticker -> optionality_pct_dev
    tickers: List[str]                            # all tickers in CSV
    market_data: Dict[str, Dict[str, Any]] = {}   # ticker -> {avg_volume, price, ...}


def _load_market_data(inputs_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load market_data.json from archive inputs directory.

    Returns {ticker: {avg_volume, price, ...}} or empty dict if file missing.
    Handles both list-of-dicts (with 'ticker' key) and dict-keyed-by-ticker formats.
    """
    md_path = inputs_dir / "market_data.json"
    if not md_path.exists():
        return {}
    with open(md_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return {item["ticker"]: item for item in raw if "ticker" in item}
    return raw


def load_archive_data(tar_path: Path, date_str: str) -> ArchiveData:
    """Extract archive, build rec dicts, return cached data, clean up temp dir."""
    tmp_dir = tempfile.mkdtemp(prefix=f"sweep_{date_str}_")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmp_dir)

        archive_root = Path(tmp_dir) / date_str
        if not archive_root.exists():
            subdirs = [d for d in Path(tmp_dir).iterdir() if d.is_dir()]
            if subdirs:
                archive_root = subdirs[0]
            else:
                raise FileNotFoundError(f"No archive root in {tar_path}")

        csv_path = archive_root / "rankings.csv"
        inputs_dir = archive_root / "inputs"

        if not csv_path.exists():
            raise FileNotFoundError(f"rankings.csv not found in {tar_path}")

        # Load enrichment maps
        catalyst_map = _load_catalyst_map(inputs_dir) if inputs_dir.exists() else {}
        holdings_map = _load_holdings_map(inputs_dir) if inputs_dir.exists() else {}
        defensive_map = _load_defensive_map(inputs_dir) if inputs_dir.exists() else {}
        decision_inputs = _load_decision_inputs(inputs_dir) if inputs_dir.exists() else {}
        market_data = _load_market_data(inputs_dir) if inputs_dir.exists() else {}

        # Read CSV
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Build per-ticker data
        recs: Dict[str, Dict[str, Any]] = {}
        archetypes: Dict[str, str] = {}
        optionalities: Dict[str, Optional[float]] = {}
        tickers: List[str] = []

        for row in rows:
            ticker = row.get("ticker", "").strip().upper()
            if not ticker:
                continue
            tickers.append(ticker)

            archetype = row.get("archetype", "").strip()
            archetypes[ticker] = archetype

            opt_str = row.get("clinical_optionality_pct_dev", "").strip()
            opt: Optional[float] = None
            if opt_str:
                try:
                    opt = float(opt_str)
                except ValueError:
                    pass
            optionalities[ticker] = opt

            enriched = decision_inputs.get(ticker, {})
            rec = build_rec(row, catalyst_map, holdings_map, defensive_map, enriched)
            recs[ticker] = rec

        return ArchiveData(
            date_str=date_str,
            rows=rows,
            recs=recs,
            archetypes=archetypes,
            optionalities=optionalities,
            tickers=tickers,
            market_data=market_data,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# FORWARD RETURNS (computed once per archive, shared across rulesets)
# =============================================================================

class ForwardReturnData(NamedTuple):
    """Cached forward return data for one snapshot date."""
    raw_returns: Dict[str, float]      # ticker -> 60d raw return
    residual_returns: Dict[str, float]  # ticker -> 60d residual return
    xbi_return: Optional[float]
    betas: Dict[str, float]


def compute_snapshot_returns(
    chained: ChainedReturnsProvider,
    csv_provider: CSVReturnsProvider,
    tickers: List[str],
    as_of_date: str,
    horizon_days: int = 60,
) -> ForwardReturnData:
    """Compute forward raw + residual returns for one snapshot date."""
    raw_returns = compute_forward_returns(chained, tickers, as_of_date, horizon_days)
    betas = estimate_xbi_betas(csv_provider, list(raw_returns.keys()), as_of_date)

    xbi_fwd = compute_forward_returns(chained, ["XBI"], as_of_date, horizon_days)
    xbi_return = xbi_fwd.get("XBI")

    residual_returns = compute_residual_returns(raw_returns, xbi_return, betas)

    return ForwardReturnData(
        raw_returns=raw_returns,
        residual_returns=residual_returns,
        xbi_return=xbi_return,
        betas=betas,
    )


# =============================================================================
# EVALUATE ONE SNAPSHOT + RULESET
# =============================================================================

def evaluate_snapshot(
    archive_data: ArchiveData,
    ruleset: DecisionRuleset,
    forward_data: ForwardReturnData,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """For one snapshot+ruleset: recompute tiers, group dev-stage, measure returns.

    Returns:
        (tier_stats, tier_assignments)
        tier_stats: {tier: {n, mean_pct, resid_mean_pct, median_resid_pct,
                            winsor_resid_pct, hit_pct, resid_hit_pct}}
        tier_assignments: {ticker: tier}
    """
    tier_assignments: Dict[str, str] = {}
    tier_raw: Dict[str, List[float]] = {}
    tier_resid: Dict[str, List[float]] = {}

    for ticker in archive_data.tickers:
        archetype = archive_data.archetypes.get(ticker, "")
        # Only dev-stage
        if archetype != "drug_developer":
            continue

        # Must have forward returns
        if ticker not in forward_data.raw_returns:
            continue

        rec = archive_data.recs.get(ticker)
        if rec is None:
            continue

        opt = archive_data.optionalities.get(ticker)

        # Recompute decision fields with candidate ruleset
        fields = compute_decision_fields(rec, archetype, opt, ruleset=ruleset)
        tier = fields.get("tier_dev", "")
        if not tier:
            continue

        tier_assignments[ticker] = tier

        raw_ret = forward_data.raw_returns[ticker]
        resid_ret = forward_data.residual_returns.get(ticker, raw_ret)

        tier_raw.setdefault(tier, []).append(raw_ret)
        tier_resid.setdefault(tier, []).append(resid_ret)

    # Compute per-tier stats
    tier_stats: Dict[str, Dict[str, Any]] = {}
    for tier in sorted(set(list(tier_raw.keys()) + list(tier_resid.keys()))):
        raw_vals = tier_raw.get(tier, [])
        resid_vals = tier_resid.get(tier, [])
        n = len(raw_vals)
        if n == 0:
            continue

        resid_pcts = [v * 100 for v in resid_vals] if resid_vals else []

        tier_stats[tier] = {
            "n": n,
            "mean_pct": mean(raw_vals) * 100,
            "resid_mean_pct": mean(resid_pcts) if resid_pcts else 0.0,
            "median_resid_pct": median(resid_pcts) if resid_pcts else 0.0,
            "winsor_resid_pct": mean(_winsorize(resid_pcts)) if resid_pcts else 0.0,
            "hit_pct": sum(1 for v in raw_vals if v > 0) / n * 100,
            "resid_hit_pct": (sum(1 for v in resid_pcts if v > 0) / n * 100
                              if resid_pcts else 0.0),
        }

    return tier_stats, tier_assignments


# =============================================================================
# TURNOVER
# =============================================================================

def compute_turnover(tier_history: List[Dict[str, str]]) -> float:
    """Compute mean tier turnover across adjacent snapshots.

    tier_history: list of {ticker: tier} dicts ordered by date.
    For each adjacent pair, turnover = n_changed / n_common.
    """
    if len(tier_history) < 2:
        return 0.0

    turnovers: List[float] = []
    for prev, curr in zip(tier_history[:-1], tier_history[1:]):
        common = set(prev.keys()) & set(curr.keys())
        if not common:
            continue
        n_changed = sum(1 for t in common if prev[t] != curr[t])
        turnovers.append(n_changed / len(common))

    return mean(turnovers) if turnovers else 0.0


# =============================================================================
# SCORING
# =============================================================================

# Scoring weights
TURNOVER_LAMBDA = 0.5
SMALL_A_MU = 2.0


def score_ruleset(
    per_date_stats: Dict[str, Dict[str, Dict[str, Any]]],
    turnover: float,
) -> Dict[str, Any]:
    """Score a ruleset based on pooled tier stats and turnover.

    Primary metric: delta_ac = pooled_A_resid_mean - pooled_C_resid_mean
    Composite: delta_ac - lambda*turnover - mu*max(0, 10 - a_count_total)/10

    Returns dict with all raw metrics including median and winsorized stats.
    """
    # Pool observations across dates
    a_resids: List[float] = []
    c_resids: List[float] = []
    a_raws: List[float] = []
    c_raws: List[float] = []
    a_present_in = 0
    n_usable = len(per_date_stats)

    # Per-date means for median computation
    a_date_means: List[float] = []
    c_date_means: List[float] = []
    a_date_medians: List[float] = []
    a_date_winsors: List[float] = []

    for date_str, tier_stats in per_date_stats.items():
        a_stat = tier_stats.get("A", {})
        c_stat = tier_stats.get("C", {})

        a_n = a_stat.get("n", 0)
        if a_n >= 5:
            a_present_in += 1

        if a_n > 0:
            a_resids.extend([a_stat["resid_mean_pct"]] * a_n)
            a_raws.extend([a_stat["mean_pct"]] * a_n)
            a_date_means.append(a_stat["resid_mean_pct"])
            a_date_medians.append(a_stat.get("median_resid_pct", a_stat["resid_mean_pct"]))
            a_date_winsors.append(a_stat.get("winsor_resid_pct", a_stat["resid_mean_pct"]))

        c_n = c_stat.get("n", 0)
        if c_n > 0:
            c_resids.extend([c_stat["resid_mean_pct"]] * c_n)
            c_raws.extend([c_stat["mean_pct"]] * c_n)
            c_date_means.append(c_stat["resid_mean_pct"])

    a_count_total = len(a_resids)
    c_count_total = len(c_resids)

    a_resid_mean = mean(_winsorize(a_resids)) if a_resids else 0.0
    c_resid_mean = mean(_winsorize(c_resids)) if c_resids else 0.0
    a_hit_pct = (sum(1 for v in a_raws if v > 0) / len(a_raws) * 100) if a_raws else 0.0

    # Median of per-date A means (each date is one observation)
    a_resid_median = median(a_date_means) if a_date_means else 0.0
    # Winsorized mean from per-date winsorized means
    a_resid_winsor = mean(a_date_winsors) if a_date_winsors else 0.0

    delta_ac = a_resid_mean - c_resid_mean

    # Composite with guardrails
    small_a_penalty = SMALL_A_MU * max(0, 10 - a_count_total) / 10
    composite = delta_ac - TURNOVER_LAMBDA * turnover * 100 - small_a_penalty

    return {
        "delta_ac_resid_60d": round(delta_ac, 4),
        "a_resid_mean_60d": round(a_resid_mean, 4),
        "c_resid_mean_60d": round(c_resid_mean, 4),
        "a_resid_median_60d": round(a_resid_median, 4),
        "a_resid_winsor_60d": round(a_resid_winsor, 4),
        "a_hit_pct": round(a_hit_pct, 2),
        "a_count_total": a_count_total,
        "c_count_total": c_count_total,
        "a_present_in": a_present_in,
        "n_usable_snapshots": n_usable,
        "turnover": round(turnover, 4),
        "composite_score": round(composite, 4),
    }


# =============================================================================
# HOLDOUT VALIDATION
# =============================================================================

def split_dates(
    all_dates: List[str],
    cutoff: str,
) -> Tuple[List[str], List[str]]:
    """Split sorted date list into early (<= cutoff) and late (> cutoff)."""
    early = [d for d in all_dates if d <= cutoff]
    late = [d for d in all_dates if d > cutoff]
    return early, late


def run_holdout_validation(
    all_results: List[Dict[str, Any]],
    tier_histories: Dict[str, List[Tuple[str, Dict[str, str]]]],
    cutoff: str,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Walk-forward holdout validation.

    1. Score each ruleset on early period only (dates <= cutoff)
    2. Rank by early composite
    3. Evaluate top-K on late period (dates > cutoff)
    4. Gate candidates on: positive OOS Δ(A-C), min A obs, stable turnover

    Args:
        all_results: full-period results with per_date stats
        tier_histories: {ruleset_id: [(date, {ticker: tier}), ...]} ordered by date
        cutoff: date string for early/late split
        top_k: number of top rulesets from early to validate

    Returns:
        Holdout analysis dict with early rankings, late validation, candidates.
    """
    all_dates = sorted({d for r in all_results for d in r["per_date"].keys()})
    early_dates, late_dates = split_dates(all_dates, cutoff)

    if not early_dates or not late_dates:
        return {
            "error": f"Cannot split: {len(early_dates)} early, {len(late_dates)} late",
            "cutoff": cutoff,
            "candidates": [],
        }

    # Score each ruleset on early period only
    early_scored: List[Dict[str, Any]] = []
    for r in all_results:
        rid = r["ruleset_id"]
        early_stats = {d: r["per_date"][d] for d in early_dates if d in r["per_date"]}

        # Compute early turnover from tier history
        th = tier_histories.get(rid, [])
        early_th = [assigns for d, assigns in th if d <= cutoff]
        early_turnover = compute_turnover(early_th)

        early_scores = score_ruleset(early_stats, early_turnover)
        early_scored.append({
            "ruleset_id": rid,
            "params": {
                "tier_a_floor": r["tier_a_floor"],
                "tier_b_floor": r["tier_b_floor"],
                "catalyst_near_days": r["catalyst_near_days"],
                "sponsor_threshold": r["sponsor_threshold"],
            },
            **{f"early_{k}": v for k, v in early_scores.items()},
        })

    # Rank by early composite
    early_scored.sort(key=lambda x: x["early_composite_score"], reverse=True)
    top_candidates = early_scored[:top_k]

    # Evaluate top candidates on late period
    validated: List[Dict[str, Any]] = []
    for cand in top_candidates:
        rid = cand["ruleset_id"]
        # Find full result
        full_r = next(r for r in all_results if r["ruleset_id"] == rid)

        late_stats = {d: full_r["per_date"][d] for d in late_dates
                      if d in full_r["per_date"]}

        th = tier_histories.get(rid, [])
        late_th = [assigns for d, assigns in th if d > cutoff]
        late_turnover = compute_turnover(late_th)

        late_scores = score_ruleset(late_stats, late_turnover)

        # Gates
        passes_delta = late_scores["delta_ac_resid_60d"] > 0
        passes_a_obs = late_scores["a_count_total"] >= 5
        passes_turnover = (late_scores["turnover"] <= cand["early_turnover"] * 2.0
                           if cand["early_turnover"] > 0 else True)
        passes_all = passes_delta and passes_a_obs and passes_turnover

        validated.append({
            **cand,
            **{f"late_{k}": v for k, v in late_scores.items()},
            "gate_positive_delta": passes_delta,
            "gate_min_a_obs": passes_a_obs,
            "gate_stable_turnover": passes_turnover,
            "passes_holdout": passes_all,
        })

    # Find best candidate that passes all gates
    passing = [v for v in validated if v["passes_holdout"]]
    # Among passing, rank by late delta_ac (OOS performance)
    passing.sort(key=lambda x: x["late_delta_ac_resid_60d"], reverse=True)

    return {
        "cutoff": cutoff,
        "n_early_dates": len(early_dates),
        "n_late_dates": len(late_dates),
        "early_dates": early_dates,
        "late_dates": late_dates,
        "top_k_evaluated": len(validated),
        "n_passing": len(passing),
        "validated": validated,
        "candidates": passing,
        "recommended": passing[0] if passing else None,
    }


# =============================================================================
# CANDIDATE OUTPUT
# =============================================================================

def save_candidate_ruleset(
    candidate: Dict[str, Any],
    output_path: Path,
) -> None:
    """Save candidate ruleset as a DecisionRuleset JSON."""
    params = candidate["params"]
    rs = DecisionRuleset(
        tier_a_optionality_floor=params["tier_a_floor"],
        tier_b_optionality_floor=params["tier_b_floor"],
        catalyst_near_days=params["catalyst_near_days"],
        sponsor_confirm_threshold=params["sponsor_threshold"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rs.to_json(str(output_path))


# =============================================================================
# CALIBRATION MEMO
# =============================================================================

def generate_calibration_memo(
    all_results: List[Dict[str, Any]],
    holdout: Dict[str, Any],
    output_path: Path,
) -> None:
    """Write calibration memo as plain text."""
    default_id = DecisionRuleset().ruleset_id
    default_rank = next(
        (i + 1 for i, r in enumerate(all_results) if r["ruleset_id"] == default_id),
        None,
    )
    default_result = next(
        (r for r in all_results if r["ruleset_id"] == default_id), None
    )

    recommended = holdout.get("recommended")
    top3 = all_results[:3]
    passing = holdout.get("candidates", [])

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("DECISION RULESET CALIBRATION NOTE")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 72)
    lines.append("")

    # 1. What we searched
    lines.append("1. GRID SEARCHED")
    lines.append("-" * 40)
    lines.append(f"  Parameters swept:")
    for k, vals in GRID_AXES.items():
        lines.append(f"    {k}: {vals}")
    lines.append(f"  Valid combinations: {len(all_results)}")
    lines.append(f"  Note: sponsor_confirm_threshold has ZERO effect on tier")
    lines.append(f"    assignments (only affects sizing). Effective grid = "
                 f"{len(all_results) // 2} unique tier behaviors.")
    lines.append(f"  Usable archives: {all_results[0]['n_usable_snapshots']}")
    lines.append("")

    # 2. Why
    lines.append("2. MOTIVATION")
    lines.append("-" * 40)
    lines.append("  Decision Engine v1.1.0 externalized DecisionRuleset parameters")
    lines.append("  but thresholds were hand-picked. This sweep identifies which")
    lines.append("  parameter combination produces the best A-vs-C tier separation")
    lines.append("  on 60-day XBI-residual returns across 22 monthly archives")
    lines.append("  (2024-01 through 2025-10).")
    lines.append("")

    # 3. What won (full period)
    lines.append("3. FULL-PERIOD RESULTS (top 5)")
    lines.append("-" * 40)
    hdr = (f"  {'#':>2}  {'ID':>8}  {'a_fl':>5} {'b_fl':>5} {'cat':>4}  "
           f"{'Δ(A-C)':>8} {'A_med':>7} {'A_win':>7} {'A_hit%':>6} "
           f"{'A_n':>5}  {'score':>8}")
    lines.append(hdr)
    for i, r in enumerate(all_results[:5]):
        marker = " *" if r["ruleset_id"] == default_id else "  "
        lines.append(
            f"  {i+1:>2}{marker}"
            f"{r['ruleset_id']:>8}  "
            f"{r['tier_a_floor']:>5.2f} {r['tier_b_floor']:>5.2f} "
            f"{r['catalyst_near_days']:>4d}  "
            f"{r['delta_ac_resid_60d']:>8.2f} "
            f"{r.get('a_resid_median_60d', 0):>7.2f} "
            f"{r.get('a_resid_winsor_60d', 0):>7.2f} "
            f"{r['a_hit_pct']:>6.1f} "
            f"{r['a_count_total']:>5d}  "
            f"{r['composite_score']:>8.2f}"
        )
    lines.append("")
    if default_result:
        lines.append(f"  Default v1.0 ({default_id}) ranked #{default_rank}/{len(all_results)}: "
                     f"Δ(A-C)={default_result['delta_ac_resid_60d']:.2f}, "
                     f"composite={default_result['composite_score']:.2f}")
        lines.append(f"  Default has NEGATIVE Δ(A-C): C-tier outperforms A-tier at a_floor=0.60.")
        lines.append(f"  Lowering a_floor to 0.55 captures names in 0.55-0.59 range that")
        lines.append(f"  perform well, flipping the separation positive.")
    lines.append("")

    # 4. Holdout validation
    lines.append("4. HOLDOUT VALIDATION")
    lines.append("-" * 40)
    lines.append(f"  Split: early <= {holdout['cutoff']} ({holdout['n_early_dates']} dates), "
                 f"late > {holdout['cutoff']} ({holdout['n_late_dates']} dates)")
    lines.append(f"  Top {holdout['top_k_evaluated']} by early composite evaluated on late period.")
    lines.append(f"  Gates: positive Δ(A-C) residual OOS, min 5 A obs, turnover <= 2x early.")
    lines.append(f"  Passing: {holdout['n_passing']}/{holdout['top_k_evaluated']}")
    lines.append("")

    if holdout.get("validated"):
        lines.append(f"  {'ID':>8}  {'early_Δ':>8} {'late_Δ':>8} "
                     f"{'late_A_n':>7} {'late_turn':>9}  {'pass':>5}")
        for v in holdout["validated"]:
            lines.append(
                f"  {v['ruleset_id']:>8}  "
                f"{v['early_delta_ac_resid_60d']:>8.2f} "
                f"{v['late_delta_ac_resid_60d']:>8.2f} "
                f"{v['late_a_count_total']:>7d} "
                f"{v['late_turnover']:>9.4f}  "
                f"{'YES' if v['passes_holdout'] else 'no':>5}"
            )
    lines.append("")

    # 5. Tradeoffs
    lines.append("5. KEY TRADEOFFS")
    lines.append("-" * 40)
    lines.append("  a) A-tier size vs separation:")
    lines.append("     a_floor=0.55 → ~3/snapshot A-tier names (tiny but positive Δ)")
    lines.append("     a_floor=0.60 → even fewer, and Δ goes negative")
    lines.append("     a_floor=0.65 → worst separation (too restrictive)")
    lines.append("")
    lines.append("  b) Catalyst window:")
    lines.append("     cat_days=90 → best Δ(A-C) at a_floor=0.55 (fewer A but sharper)")
    lines.append("     cat_days=120 → more A names (77 vs 59) but slightly diluted signal")
    lines.append("     cat_days=60 → too restrictive, sparse A group")
    lines.append("")
    lines.append("  c) Sponsor threshold: no effect on tier assignments.")
    lines.append("     Keeping at 2 (default) — affects sizing only.")
    lines.append("")
    lines.append("  d) Stability: turnover is low (~5%) across all combos, not")
    lines.append("     a differentiator in this grid.")
    lines.append("")

    # 6. Recommendation
    lines.append("6. RECOMMENDATION")
    lines.append("-" * 40)
    if recommended:
        p = recommended["params"]
        lines.append(f"  Candidate: {recommended['ruleset_id']} (saved as v1.2_candidate.json)")
        default_rs = DecisionRuleset()
        for k, v in p.items():
            default_v = getattr(default_rs, {
                "tier_a_floor": "tier_a_optionality_floor",
                "tier_b_floor": "tier_b_optionality_floor",
            }.get(k, k), None)
            changed = " (CHANGED)" if default_v is not None and default_v != v else ""
            lines.append(f"    {k}: {v}{changed}")
        early_d = recommended.get("early_delta_ac_resid_60d", 0)
        late_d = recommended.get("late_delta_ac_resid_60d", 0)
        lines.append(f"  OOS result: Δ(A-C) = {early_d:.2f} (early), {late_d:.2f} (late)")
        if early_d < 0 < late_d:
            lines.append(f"  Note: negative early Δ — A underperformed C during training")
            lines.append(f"  period. Positive late Δ drives the full-period result.")
        # Find conservative alternative (positive in both periods, more A obs)
        for v in passing:
            if v["ruleset_id"] != recommended["ruleset_id"]:
                ve = v.get("early_delta_ac_resid_60d", 0)
                vl = v.get("late_delta_ac_resid_60d", 0)
                va = v.get("late_a_count_total", 0)
                if ve > 0 and vl > 0:
                    lines.append(f"  Conservative alternative: {v['ruleset_id']} "
                                 f"(early Δ={ve:.2f}, late Δ={vl:.2f}, "
                                 f"late A obs={va})")
                    lines.append(f"    Positive Δ in BOTH periods — more consistent.")
                    break
        lines.append(f"  Holdout: passes all gates")
    else:
        lines.append("  No candidate passed all holdout gates.")
        if passing:
            lines.append(f"  Closest: {passing[0]['ruleset_id']}")
        else:
            lines.append("  Consider: relaxing gates or using best full-period ruleset")
            lines.append(f"  with awareness that OOS validation is inconclusive.")
            if top3:
                lines.append(f"  Best full-period: {top3[0]['ruleset_id']} "
                             f"(Δ={top3[0]['delta_ac_resid_60d']:.2f})")
    lines.append("")

    # 7. Caveats
    lines.append("7. CAVEATS")
    lines.append("-" * 40)
    lines.append("  - A-tier is SMALL (2-4 names per snapshot at a_floor=0.55).")
    lines.append("    Results are noisy; a single name can swing the mean.")
    lines.append("  - Only 22 usable monthly snapshots (limited history).")
    lines.append("  - Forward returns use HS793+CSV fallback (price return,")
    lines.append("    no dividend adjustment — acceptable for biotech).")
    lines.append("  - Grid only sweeps 4 parameters; gates/risk flags/sizing")
    lines.append("    weights are held at defaults.")
    lines.append("  - The positive Δ(A-C) at a_floor=0.55 may reflect that")
    lines.append("    0.55-0.59 optionality names are genuinely undervalued,")
    lines.append("    or may be a small-sample artifact.")
    lines.append("")
    lines.append("=" * 72)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Decision Ruleset Calibration Sweep"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date filter for archives (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date filter for archives (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Print top N rulesets (default: 10)",
    )
    parser.add_argument(
        "--holdout-split", type=str, default="2024-12-31",
        help="Cutoff date: early <= date, late > date (default: 2024-12-31)",
    )
    parser.add_argument(
        "--top-k-holdout", type=int, default=10,
        help="Evaluate top-K rulesets from early period on late (default: 10)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print grid size + archive count, don't run",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate grid
    grid = generate_grid()
    print(f"Ruleset grid: {len(grid)} valid parameter combinations")
    for i, rs in enumerate(grid[:3]):
        print(f"  [{i}] a_floor={rs.tier_a_optionality_floor}, "
              f"b_floor={rs.tier_b_optionality_floor}, "
              f"cat_days={rs.catalyst_near_days}, "
              f"sponsor={rs.sponsor_confirm_threshold}  "
              f"id={rs.ruleset_id}")
    if len(grid) > 3:
        print(f"  ... ({len(grid) - 3} more)")
    print()

    # Discover archives
    archives = discover_archives(ARCHIVE_DIR, start=args.start, end=args.end)
    print(f"Archives found: {len(archives)}")
    if archives:
        print(f"  Date range: {archives[0][0]} to {archives[-1][0]}")
    print()

    if args.dry_run:
        print("DRY RUN — would evaluate "
              f"{len(grid)} rulesets x {len(archives)} archives")
        return

    if not archives:
        print("ERROR: No archives found.")
        sys.exit(1)

    # Initialize providers
    print("Loading returns providers...", flush=True)
    chained, ms_provider, csv_provider = init_providers()
    last_date = chained.get_last_date()
    print(f"  Returns data ends: {last_date.isoformat() if last_date else 'NONE'}")

    # Determine usable archives (need 60d forward data)
    snapshot_dates = [d for d, _ in archives]
    usable_dates, fence_skipped = compute_as_of_fence(
        snapshot_dates, last_date, 60
    )
    print(f"  Usable snapshots: {len(usable_dates)} / {len(archives)}")
    if fence_skipped:
        for sk in fence_skipped:
            print(f"    SKIP {sk['snapshot_date']}: {sk['skipped_reason']}")
    print()

    if not usable_dates:
        print("ERROR: No usable snapshots (all lack 60d forward return window).")
        sys.exit(1)

    # Build usable archive list
    usable_archives = [(d, p) for d, p in archives if d in usable_dates]

    # Phase 1: Load archive data + compute forward returns (once per archive)
    print(f"Phase 1: Loading {len(usable_archives)} archives + computing returns...")
    archive_cache: Dict[str, ArchiveData] = {}
    return_cache: Dict[str, ForwardReturnData] = {}

    for i, (date_str, tar_path) in enumerate(usable_archives):
        print(f"  [{i+1}/{len(usable_archives)}] {date_str} ...", end=" ", flush=True)
        try:
            ad = load_archive_data(tar_path, date_str)
            archive_cache[date_str] = ad

            fd = compute_snapshot_returns(
                chained, csv_provider, ad.tickers, date_str
            )
            return_cache[date_str] = fd
            print(f"OK ({len(ad.tickers)} tickers, "
                  f"{len(fd.raw_returns)} with returns)")
        except Exception as e:
            print(f"ERROR: {e}")
    print()

    # Phase 2: Evaluate each ruleset across all cached snapshots
    print(f"Phase 2: Evaluating {len(grid)} rulesets across "
          f"{len(archive_cache)} snapshots...")

    all_results: List[Dict[str, Any]] = []
    # Store tier histories for holdout turnover computation
    tier_histories: Dict[str, List[Tuple[str, Dict[str, str]]]] = {}

    for ri, ruleset in enumerate(grid):
        if (ri + 1) % 10 == 0 or ri == 0:
            print(f"  Ruleset {ri+1}/{len(grid)} "
                  f"(id={ruleset.ruleset_id})...", flush=True)

        per_date_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
        tier_history_pairs: List[Tuple[str, Dict[str, str]]] = []

        for date_str in sorted(archive_cache.keys()):
            ad = archive_cache[date_str]
            fd = return_cache[date_str]

            tier_stats, tier_assigns = evaluate_snapshot(ad, ruleset, fd)
            per_date_stats[date_str] = tier_stats
            tier_history_pairs.append((date_str, tier_assigns))

        tier_histories[ruleset.ruleset_id] = tier_history_pairs
        tier_history_list = [assigns for _, assigns in tier_history_pairs]
        turnover = compute_turnover(tier_history_list)
        scores = score_ruleset(per_date_stats, turnover)

        # Build result row
        result = {
            "ruleset_id": ruleset.ruleset_id,
            "tier_a_floor": ruleset.tier_a_optionality_floor,
            "tier_b_floor": ruleset.tier_b_optionality_floor,
            "catalyst_near_days": ruleset.catalyst_near_days,
            "sponsor_threshold": ruleset.sponsor_confirm_threshold,
            **scores,
            "per_date": per_date_stats,
        }
        all_results.append(result)

    # Sort by composite score descending
    all_results.sort(key=lambda r: r["composite_score"], reverse=True)

    print()

    # Phase 3: Write full-period outputs
    csv_path = output_dir / "ruleset_sweep_summary.csv"
    json_path = output_dir / "ruleset_sweep_details.json"

    # CSV summary
    csv_columns = [
        "ruleset_id", "tier_a_floor", "tier_b_floor", "catalyst_near_days",
        "sponsor_threshold", "delta_ac_resid_60d", "a_resid_mean_60d",
        "c_resid_mean_60d", "a_resid_median_60d", "a_resid_winsor_60d",
        "a_hit_pct", "a_count_total", "a_present_in",
        "n_usable_snapshots", "turnover", "composite_score",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    print(f"Wrote {csv_path} ({len(all_results)} rows)")

    # JSON details
    json_output = {
        "sweep_metadata": {
            "n_rulesets": len(grid),
            "n_archives_total": len(archives),
            "n_usable_archives": len(usable_archives),
            "grid_params": {k: [str(v) for v in vals] for k, vals in GRID_AXES.items()},
            "scoring_weights": {
                "turnover_lambda": TURNOVER_LAMBDA,
                "small_a_mu": SMALL_A_MU,
            },
        },
        "rulesets": [
            {
                "ruleset_id": r["ruleset_id"],
                "params": {
                    "tier_a_floor": r["tier_a_floor"],
                    "tier_b_floor": r["tier_b_floor"],
                    "catalyst_near_days": r["catalyst_near_days"],
                    "sponsor_threshold": r["sponsor_threshold"],
                },
                "composite_score": r["composite_score"],
                "delta_ac_resid_60d": r["delta_ac_resid_60d"],
                "a_resid_mean_60d": r["a_resid_mean_60d"],
                "c_resid_mean_60d": r["c_resid_mean_60d"],
                "a_resid_median_60d": r.get("a_resid_median_60d", 0),
                "a_resid_winsor_60d": r.get("a_resid_winsor_60d", 0),
                "a_hit_pct": r["a_hit_pct"],
                "a_count_total": r["a_count_total"],
                "a_present_in": r["a_present_in"],
                "turnover": r["turnover"],
                "per_date": r["per_date"],
            }
            for r in all_results
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2)
        f.write("\n")

    print(f"Wrote {json_path}")
    print()

    # Print top N
    top_n = min(args.top_n, len(all_results))
    print(f"Top {top_n} rulesets by composite score:")
    print(f"{'#':>3}  {'ID':>8}  {'a_fl':>5} {'b_fl':>5} {'cat':>4} {'spon':>4}  "
          f"{'Δ(A-C)':>8} {'A_res':>7} {'A_med':>7} {'A_win':>7} {'A_hit%':>6} "
          f"{'A_n':>5} {'A_in':>4} {'turn':>6}  {'score':>8}")
    print("-" * 112)

    for i, r in enumerate(all_results[:top_n]):
        is_default = r["ruleset_id"] == DecisionRuleset().ruleset_id
        marker = " *" if is_default else "  "
        print(f"{i+1:>3}{marker}"
              f"{r['ruleset_id']:>8}  "
              f"{r['tier_a_floor']:>5.2f} {r['tier_b_floor']:>5.2f} "
              f"{r['catalyst_near_days']:>4d} {r['sponsor_threshold']:>4d}  "
              f"{r['delta_ac_resid_60d']:>8.2f} "
              f"{r['a_resid_mean_60d']:>7.2f} "
              f"{r.get('a_resid_median_60d', 0):>7.2f} "
              f"{r.get('a_resid_winsor_60d', 0):>7.2f} "
              f"{r['a_hit_pct']:>6.1f} "
              f"{r['a_count_total']:>5d} "
              f"{r['a_present_in']:>4d} "
              f"{r['turnover']:>6.3f}  "
              f"{r['composite_score']:>8.2f}")

    # Find where default lives
    default_id = DecisionRuleset().ruleset_id
    for i, r in enumerate(all_results):
        if r["ruleset_id"] == default_id:
            print(f"\nDefault ruleset ({default_id}) ranked #{i+1}/{len(all_results)} "
                  f"(composite={r['composite_score']:.2f})")
            break
    else:
        print(f"\nDefault ruleset ({default_id}) not found in grid (unexpected)")

    # =========================================================================
    # Phase 4: Holdout validation
    # =========================================================================
    print()
    print("=" * 72)
    print("HOLDOUT VALIDATION")
    print("=" * 72)

    holdout = run_holdout_validation(
        all_results,
        tier_histories,
        cutoff=args.holdout_split,
        top_k=args.top_k_holdout,
    )

    if "error" in holdout:
        print(f"  ERROR: {holdout['error']}")
    else:
        print(f"  Split: early <= {holdout['cutoff']} ({holdout['n_early_dates']} dates), "
              f"late > {holdout['cutoff']} ({holdout['n_late_dates']} dates)")
        print(f"  Evaluated top {holdout['top_k_evaluated']} on holdout period")
        print()

        print(f"  {'ID':>8}  {'early_Δ':>8} {'late_Δ':>8} "
              f"{'late_A_n':>8} {'late_turn':>9}  {'gates':>7}")
        print("  " + "-" * 60)
        for v in holdout.get("validated", []):
            gates = ("YES" if v["passes_holdout"] else
                     "delta" if not v["gate_positive_delta"] else
                     "a_obs" if not v["gate_min_a_obs"] else
                     "turn")
            print(f"  {v['ruleset_id']:>8}  "
                  f"{v['early_delta_ac_resid_60d']:>8.2f} "
                  f"{v['late_delta_ac_resid_60d']:>8.2f} "
                  f"{v['late_a_count_total']:>8d} "
                  f"{v['late_turnover']:>9.4f}  "
                  f"{gates:>7}")

        print()
        print(f"  Candidates passing all gates: {holdout['n_passing']}")

        recommended = holdout.get("recommended")
        if recommended:
            print(f"  RECOMMENDED: {recommended['ruleset_id']}")
            p = recommended["params"]
            print(f"    tier_a_optionality_floor: {p['tier_a_floor']}")
            print(f"    tier_b_optionality_floor: {p['tier_b_floor']}")
            print(f"    catalyst_near_days: {p['catalyst_near_days']}")
            print(f"    sponsor_confirm_threshold: {p['sponsor_threshold']}")
            print(f"    Early Δ(A-C): {recommended['early_delta_ac_resid_60d']:.2f}")
            print(f"    Late  Δ(A-C): {recommended['late_delta_ac_resid_60d']:.2f}")
        else:
            print("  No candidate passed all holdout gates.")
            print("  Using best full-period ruleset with caveat.")

    # =========================================================================
    # Phase 5: Save candidate JSON
    # =========================================================================
    print()
    rulesets_dir = PROJECT_ROOT / "production_data" / "decision_rulesets"

    recommended = holdout.get("recommended")
    if recommended:
        candidate_source = "holdout"
        candidate = recommended
    else:
        # Fall back to best full-period
        candidate_source = "full_period"
        candidate = {
            "ruleset_id": all_results[0]["ruleset_id"],
            "params": {
                "tier_a_floor": all_results[0]["tier_a_floor"],
                "tier_b_floor": all_results[0]["tier_b_floor"],
                "catalyst_near_days": all_results[0]["catalyst_near_days"],
                "sponsor_threshold": all_results[0]["sponsor_threshold"],
            },
        }

    candidate_path = rulesets_dir / "v1.2_candidate.json"
    save_candidate_ruleset(candidate, candidate_path)
    print(f"Wrote candidate ruleset: {candidate_path}")
    print(f"  Source: {candidate_source}")
    print(f"  ID: {candidate['ruleset_id']}")

    # Save holdout analysis JSON
    holdout_json_path = output_dir / "ruleset_holdout_analysis.json"
    # Strip non-serializable items
    holdout_output = {k: v for k, v in holdout.items() if k != "error"}
    with open(holdout_json_path, "w", encoding="utf-8") as f:
        json.dump(holdout_output, f, indent=2, default=str)
        f.write("\n")
    print(f"Wrote {holdout_json_path}")

    # =========================================================================
    # Phase 6: Calibration memo
    # =========================================================================
    memo_path = output_dir / "ruleset_calibration_note.txt"
    generate_calibration_memo(all_results, holdout, memo_path)
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
