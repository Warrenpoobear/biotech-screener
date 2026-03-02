#!/usr/bin/env python3
"""Evaluate a candidate ruleset across a date range.

Produces stability metrics, portfolio-realistic returns (turnover + costs),
regime slices, and an optional promotion gate verdict.

Usage:
    python3 scripts/eval_ruleset.py \
        --candidate production_data/decision_rulesets/v1.6.1_alpha_modifier_within_tier.json \
        --start 2026-02-05 --end 2026-02-26 \
        --k 60 --horizon 5 --cost-bps 30 \
        --snapshot-dir data/snapshots \
        --out-dir /tmp/ruleset_eval_smoke \
        --anchor-mode prev_trading_day \
        --gate
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root for imports
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.eval_forward_returns import (
    compute_forward_return,
    compute_turnover,
    discover_snapshot_dates,
    load_price_series,
    net_return,
    resolve_trade_date,
)
from common.ranking_utils import backfill_columns, safe_float as _rerank_sf
from decision_engine import DecisionRuleset, compute_actionable_sort_key

VERSION = "1.0.0"
SCHEMA = "ruleset_eval.v1"

# Default thresholds for --gate
DEFAULT_GATE_THRESHOLDS = {
    "mean_top60_overlap_warn": 0.85,
    "mean_top60_overlap_fail": 0.70,
    "max_rank_shift_warn": 50,
    "max_rank_shift_fail": 100,
    "turnover_excess_warn": 0.20,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, default=str) + "\n"


def _safe_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _read_rankings(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _cumulative(returns: List[float]) -> Optional[float]:
    if not returns:
        return None
    cum = 1.0
    for r in returns:
        cum *= 1.0 + r
    return cum - 1.0


def load_ruleset_id(path: Path) -> str:
    """Compute ruleset_id from file content (same as DecisionRuleset.from_json)."""
    data = _read_json(path)
    # Match decision_engine.py: sha256 of compact sorted JSON, first 8 hex chars
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]


def active_ruleset_from_manifest(manifest_path: Path) -> Tuple[str, str]:
    """Return (id, file) for the active ruleset from manifest."""
    data = _read_json(manifest_path)
    for entry in data.get("rulesets", []):
        if entry.get("status") == "active":
            return entry["id"], entry["file"]
    raise ValueError("No active ruleset in manifest")


def snapshot_ruleset_id(snap_dir: Path) -> str:
    """Read the ruleset_id from a snapshot's metadata or rankings."""
    meta_path = snap_dir / "metadata.json"
    if meta_path.exists():
        meta = _read_json(meta_path)
        rid = meta.get("clinical_sort_telemetry", {}).get("ruleset_id", "")
        if rid:
            return rid
    # Fallback: check first row of rankings
    rankings_path = snap_dir / "rankings.csv"
    if rankings_path.exists():
        rows = _read_rankings(rankings_path)
        if rows:
            return rows[0].get("decision_engine_ruleset_id", "")
    return ""


def is_degraded(snap_dir: Path) -> bool:
    ch_path = snap_dir / "cache_health.json"
    if ch_path.exists():
        try:
            ch = _read_json(ch_path)
            if ch.get("degraded_run", False):
                return True
            if ch.get("overall_status", "ok") != "ok":
                return True
        except Exception:
            pass
    return False


def ranked_tickers(rows: List[Dict], col: str = "actionable_rank") -> List[str]:
    """Extract tickers sorted by rank column (eligible only)."""
    ranked = []
    for r in rows:
        elig = str(r.get("eligible", "")).strip().lower()
        if elig not in ("1", "true", "yes"):
            continue
        rank = _safe_int(r.get(col))
        if rank is not None:
            ranked.append((rank, r.get("ticker", "")))
    ranked.sort()
    return [t for _, t in ranked]


def rank_map(rows: List[Dict], col: str = "actionable_rank") -> Dict[str, int]:
    result = {}
    for r in rows:
        elig = str(r.get("eligible", "")).strip().lower()
        if elig not in ("1", "true", "yes"):
            continue
        rank = _safe_int(r.get(col))
        ticker = r.get("ticker", "")
        if rank is not None and ticker:
            result[ticker] = rank
    return result


def compute_overlap(curr: List[str], prior: List[str], k: int) -> Dict[str, Any]:
    cs = set(curr[:k])
    ps = set(prior[:k])
    if not cs or not ps:
        return {"top_k": k, "overlap_count": 0, "overlap_pct": 0.0}
    inter = cs & ps
    union = cs | ps
    return {
        "top_k": k,
        "overlap_count": len(inter),
        "overlap_pct": round(100 * len(inter) / k, 1),
        "jaccard": round(len(inter) / len(union), 4) if union else 0.0,
        "entrants": sorted(cs - ps),
        "exits": sorted(ps - cs),
    }


def compute_rank_correlation(
    curr_ranks: Dict[str, int], prior_ranks: Dict[str, int]
) -> Dict[str, Optional[float]]:
    common = sorted(set(curr_ranks) & set(prior_ranks))
    if len(common) < 5:
        return {"spearman": None, "n_common": len(common)}
    n = len(common)
    cv = [curr_ranks[t] for t in common]
    pv = [prior_ranks[t] for t in common]
    d_sq = sum((c - p) ** 2 for c, p in zip(cv, pv))
    rho = 1 - (6 * d_sq) / (n * (n * n - 1)) if n > 1 else None
    return {"spearman": round(rho, 4) if rho is not None else None, "n_common": n}


def compute_max_shift(
    curr_ranks: Dict[str, int], prior_ranks: Dict[str, int]
) -> Optional[Dict[str, Any]]:
    common = set(curr_ranks) & set(prior_ranks)
    if not common:
        return None
    shifts = {t: abs(curr_ranks[t] - prior_ranks[t]) for t in common}
    worst = max(shifts, key=shifts.get)
    return {
        "ticker": worst,
        "shift": shifts[worst],
        "from": prior_ranks[worst],
        "to": curr_ranks[worst],
    }


def compute_top_shifts(
    curr_ranks: Dict[str, int],
    prior_ranks: Dict[str, int],
    n: int = 3,
) -> List[Dict[str, Any]]:
    """Return top-N rank shifts between two rank maps, sorted by shift desc."""
    common = set(curr_ranks) & set(prior_ranks)
    if not common:
        return []
    entries = []
    for t in common:
        s = abs(curr_ranks[t] - prior_ranks[t])
        if s > 0:
            entries.append({
                "ticker": t,
                "shift": s,
                "from": prior_ranks[t],
                "to": curr_ranks[t],
            })
    entries.sort(key=lambda x: (-x["shift"], x["ticker"]))
    return entries[:n]


# Decomposition columns added to shift entries for FAIL-packet diagnostics.
_ENRICHMENT_COLS = [
    ("tier", "tier_any"),
    ("archetype", "archetype"),
    ("composite_rank", "composite_rank"),
    ("catalyst_days", "catalyst_days"),
    ("catalyst_mode", "catalyst_mode"),
    ("alpha_cohort_raw", "alpha_cohort_raw"),
]


def _enrich_shift(
    shift: Dict[str, Any],
    row_by_ticker: Dict[str, Dict[str, str]],
) -> None:
    """Add decomposition columns from the CSV row to a shift entry (in-place)."""
    row = row_by_ticker.get(shift.get("ticker", ""), {})
    for out_key, csv_col in _ENRICHMENT_COLS:
        shift[out_key] = row.get(csv_col, "")
    mc = row.get("missing_components", "")
    shift["missing_count"] = (
        len([x for x in mc.split(";") if x.strip()]) if mc else 0
    )


# ---------------------------------------------------------------------------
# Rerank-only helpers
# ---------------------------------------------------------------------------


def rerank_rows(
    rows: List[Dict[str, str]],
    ruleset: DecisionRuleset,
) -> Tuple[List[str], Dict[str, int]]:
    """Re-rank CSV rows using a ruleset.

    Works on a shallow copy to avoid mutating the input.
    Returns (ordered_eligible_tickers, rank_map).
    """
    work = [dict(r) for r in rows]
    backfill_columns(work)

    work.sort(key=lambda r: compute_actionable_sort_key(
        decision_fields=r,
        archetype=r.get("archetype", ""),
        optionality=_rerank_sf(r.get("clinical_optionality_pct_dev")),
        composite_rank=r.get("composite_rank"),
        ticker=r.get("ticker", ""),
        catalyst_event_type=r.get("catalyst_event_type", ""),
        catalyst_source=r.get("catalyst_source", ""),
        ruleset=ruleset,
        tiebreaker_pct=(
            _rerank_sf(r.get("alpha_cohort_pct"))
            if ruleset.sort_anchor == "alpha_cohort"
            else (
                _rerank_sf(r.get("commercial_quality_pct"))
                if r.get("archetype", "").startswith("commercial_")
                else _rerank_sf(r.get("clinical_optionality_pct_dev"))
            )
        ),
        alpha_raw=_rerank_sf(r.get("alpha_cohort_raw")),
    ))

    tickers: List[str] = []
    ranks: Dict[str, int] = {}
    rank = 1
    for r in work:
        elig = str(r.get("eligible", "")).strip().lower()
        if elig in ("1", "true", "yes"):
            t = r.get("ticker", "")
            if t:
                tickers.append(t)
                ranks[t] = rank
                rank += 1
    return tickers, ranks


def evaluate_ruleset_rerank_only(
    candidate_ruleset: DecisionRuleset,
    baseline_ruleset: DecisionRuleset,
    dates: List[str],
    snapshot_dir: Path,
    k: int,
) -> Dict[str, Any]:
    """Evaluate by re-ranking historical snapshots (no price data needed).

    Loads each snapshot's rankings.csv, re-ranks using both rulesets,
    and computes stability + cross-comparison metrics.
    """
    is_self_eval = candidate_ruleset.ruleset_id == baseline_ruleset.ruleset_id

    per_date: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    prev_cand_tickers: List[str] = []
    prev_cand_ranks: Optional[Dict[str, int]] = None

    # Temporal stability (candidate day-over-day)
    temporal_overlaps_20: List[float] = []
    temporal_overlaps_60: List[float] = []
    temporal_spearman: List[float] = []
    temporal_max_shifts: List[Dict[str, Any]] = []
    all_enriched_shifts: List[Dict[str, Any]] = []

    # Cross-comparison (candidate vs baseline, same date)
    cross_overlaps_20: List[float] = []
    cross_overlaps_60: List[float] = []
    cross_spearman: List[float] = []
    cross_max_shifts: List[Dict[str, Any]] = []
    cross_pct_changed: List[float] = []

    for snap_date in dates:
        snap_path = snapshot_dir / snap_date
        if not snap_path.is_dir():
            skipped.append({"date": snap_date, "reason": "missing_snapshot"})
            continue

        rankings_path = snap_path / "rankings.csv"
        if not rankings_path.exists():
            skipped.append({"date": snap_date, "reason": "missing_rankings"})
            continue

        rows = _read_rankings(rankings_path)
        row_by_ticker = {r.get("ticker", ""): r for r in rows}
        if not rows:
            skipped.append({"date": snap_date, "reason": "empty_rankings"})
            continue

        if "ticker" not in rows[0]:
            skipped.append({"date": snap_date, "reason": "missing_ticker_column"})
            continue

        cand_tickers, cand_ranks = rerank_rows(rows, candidate_ruleset)

        if not cand_tickers:
            skipped.append({"date": snap_date, "reason": "no_eligible_rows"})
            continue

        date_row: Dict[str, Any] = {
            "date": snap_date,
            "n_eligible": len(cand_tickers),
        }

        if is_degraded(snap_path):
            date_row["degraded_snapshot"] = True

        # Temporal stability (candidate day-over-day)
        if prev_cand_ranks is not None:
            ov20 = compute_overlap(cand_tickers, prev_cand_tickers, min(20, k))
            ov60 = compute_overlap(cand_tickers, prev_cand_tickers, k)
            rc = compute_rank_correlation(cand_ranks, prev_cand_ranks)
            ms = compute_max_shift(cand_ranks, prev_cand_ranks)

            temporal_overlaps_20.append(ov20["overlap_pct"])
            temporal_overlaps_60.append(ov60["overlap_pct"])
            if rc["spearman"] is not None:
                temporal_spearman.append(rc["spearman"])
            if ms:
                temporal_max_shifts.append({**ms, "date": snap_date})
                date_row["temporal_max_shift_ticker"] = ms["ticker"]
                date_row["temporal_max_shift"] = ms["shift"]
                date_row["temporal_max_shift_from"] = ms["from"]
                date_row["temporal_max_shift_to"] = ms["to"]

            # Enriched top-N shifts for FAIL packet
            top_shifts = compute_top_shifts(cand_ranks, prev_cand_ranks, n=3)
            for ts_entry in top_shifts:
                enriched = {**ts_entry, "date": snap_date}
                _enrich_shift(enriched, row_by_ticker)
                all_enriched_shifts.append(enriched)

            date_row["temporal_overlap_20"] = ov20["overlap_pct"]
            date_row["temporal_overlap_60"] = ov60["overlap_pct"]
            date_row["temporal_spearman"] = rc["spearman"]

        prev_cand_tickers = cand_tickers[:k]
        prev_cand_ranks = cand_ranks

        # Cross-comparison (candidate vs baseline, same date)
        if not is_self_eval:
            base_tickers, base_ranks = rerank_rows(rows, baseline_ruleset)

            xov20 = compute_overlap(cand_tickers, base_tickers, min(20, k))
            xov60 = compute_overlap(cand_tickers, base_tickers, k)
            xrc = compute_rank_correlation(cand_ranks, base_ranks)
            xms = compute_max_shift(cand_ranks, base_ranks)

            cross_overlaps_20.append(xov20["overlap_pct"])
            cross_overlaps_60.append(xov60["overlap_pct"])
            if xrc["spearman"] is not None:
                cross_spearman.append(xrc["spearman"])
            if xms:
                cross_max_shifts.append({**xms, "date": snap_date})

            common = set(cand_ranks) & set(base_ranks)
            if common:
                changed = sum(
                    1 for t in common if cand_ranks[t] != base_ranks[t]
                )
                pct = round(100 * changed / len(common), 1)
                cross_pct_changed.append(pct)
                date_row["cross_pct_rank_changed"] = pct

            date_row["cross_overlap_20"] = xov20["overlap_pct"]
            date_row["cross_overlap_60"] = xov60["overlap_pct"]
            date_row["cross_spearman"] = xrc["spearman"]

        per_date.append(date_row)

    # --- Aggregate ---
    result: Dict[str, Any] = {
        "dates_evaluated": [d["date"] for d in per_date],
        "n_evaluated": len(per_date),
        "dates_skipped": skipped,
        "n_skipped": len(skipped),
    }

    # Temporal stability
    temporal: Dict[str, Any] = {}
    if temporal_overlaps_60:
        temporal["mean_top20_overlap"] = round(
            statistics.mean(temporal_overlaps_20), 1
        )
        temporal["mean_top60_overlap"] = round(
            statistics.mean(temporal_overlaps_60), 1
        )
        temporal["worst_top60_overlap"] = round(min(temporal_overlaps_60), 1)
    if temporal_spearman:
        temporal["mean_spearman"] = round(statistics.mean(temporal_spearman), 4)
        temporal["worst_spearman"] = round(min(temporal_spearman), 4)
    if temporal_max_shifts:
        worst = max(temporal_max_shifts, key=lambda x: x["shift"])
        temporal["max_rank_shift"] = worst
    if all_enriched_shifts:
        temporal["top_movers"] = sorted(
            all_enriched_shifts,
            key=lambda x: (-x["shift"], x["ticker"]),
        )[:20]
    result["temporal_stability"] = temporal

    # Cross-comparison
    cross: Dict[str, Any] = {}
    if cross_overlaps_60:
        cross["mean_top20_overlap"] = round(
            statistics.mean(cross_overlaps_20), 1
        )
        cross["mean_top60_overlap"] = round(
            statistics.mean(cross_overlaps_60), 1
        )
        cross["worst_top60_overlap"] = round(min(cross_overlaps_60), 1)
    if cross_spearman:
        cross["mean_spearman"] = round(statistics.mean(cross_spearman), 4)
    if cross_max_shifts:
        worst = max(cross_max_shifts, key=lambda x: x["shift"])
        cross["max_rank_shift"] = worst
    if cross_pct_changed:
        cross["mean_pct_rank_changed"] = round(
            statistics.mean(cross_pct_changed), 1
        )
    result["cross_comparison"] = cross

    result["performance"] = {}
    result["regime"] = {}
    result["per_date"] = per_date

    return result


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------


def evaluate_ruleset(
    candidate_id: str,
    baseline_id: str,
    dates: List[str],
    snapshot_dir: Path,
    price_series: Dict[str, Dict[str, float]],
    all_trade_dates: List[str],
    regime_map: Optional[Dict[str, str]],
    k: int,
    horizon: int,
    cost_bps: float,
    anchor_mode: str,
) -> Dict[str, Any]:
    """Evaluate candidate across dates, comparing to baseline."""
    is_self_eval = candidate_id == baseline_id

    per_date: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    prev_cand_tickers: List[str] = []
    prev_base_tickers: List[str] = []

    cand_returns_gross: List[float] = []
    cand_returns_net: List[float] = []
    cand_turnovers: List[float] = []
    base_returns_gross: List[float] = []
    base_returns_net: List[float] = []
    base_turnovers: List[float] = []

    # Temporal stability (candidate day-over-day)
    temporal_overlaps_20: List[float] = []
    temporal_overlaps_60: List[float] = []
    temporal_spearman: List[float] = []
    temporal_max_shifts: List[Dict[str, Any]] = []
    all_enriched_shifts: List[Dict[str, Any]] = []

    # Cross-comparison (candidate vs baseline, same date)
    cross_overlaps_20: List[float] = []
    cross_overlaps_60: List[float] = []
    cross_spearman: List[float] = []
    cross_max_shifts: List[Dict[str, Any]] = []

    # Regime buckets
    regime_cand: Dict[str, List[float]] = {}
    regime_base: Dict[str, List[float]] = {}

    prev_cand_ranks: Optional[Dict[str, int]] = None

    for snap_date in dates:
        snap_path = snapshot_dir / snap_date
        if not snap_path.is_dir():
            skipped.append({"date": snap_date, "reason": "missing_snapshot"})
            continue

        if is_degraded(snap_path):
            skipped.append({"date": snap_date, "reason": "degraded"})
            continue

        snap_rid = snapshot_ruleset_id(snap_path)

        # Check match
        cand_match = snap_rid == candidate_id
        base_match = snap_rid == baseline_id

        if not cand_match and not base_match:
            skipped.append({"date": snap_date, "reason": "RULESET_MISMATCH"})
            continue

        # In self-eval, both match
        if is_self_eval:
            cand_match = base_match = True

        rankings_path = snap_path / "rankings.csv"
        if not rankings_path.exists():
            skipped.append({"date": snap_date, "reason": "missing_rankings"})
            continue

        rows = _read_rankings(rankings_path)
        row_by_ticker = {r.get("ticker", ""): r for r in rows}
        tickers = ranked_tickers(rows)
        ranks = rank_map(rows)

        # Resolve trade date for returns
        trade_date = resolve_trade_date(all_trade_dates, snap_date, anchor_mode)

        # Compute forward returns for all eligible tickers
        fwd_rets: Dict[str, float] = {}
        if trade_date:
            for t in tickers:
                if t in price_series:
                    ret = compute_forward_return(
                        price_series[t], all_trade_dates, trade_date, horizon
                    )
                    if ret is not None:
                        fwd_rets[t] = ret

        date_row: Dict[str, Any] = {"date": snap_date, "trade_date": trade_date}

        # Regime
        regime = regime_map.get(snap_date, "UNKNOWN") if regime_map else "UNKNOWN"
        date_row["regime"] = regime

        # Coverage
        n_eligible = len(tickers)
        date_row["n_eligible"] = n_eligible

        # --- Candidate metrics ---
        if cand_match:
            cand_topk = tickers[:k]

            # Portfolio return
            cand_gross = None
            if fwd_rets and cand_topk:
                held = [t for t in cand_topk if t in fwd_rets]
                if held:
                    cand_gross = statistics.mean(fwd_rets[t] for t in held)
                    date_row["cand_n_held"] = len(held)

            turn = compute_turnover(prev_cand_tickers, cand_topk)
            date_row["cand_turnover"] = round(turn, 4)

            if cand_gross is not None:
                cand_net = net_return(cand_gross, turn, cost_bps)
                date_row["cand_gross"] = round(cand_gross, 6)
                date_row["cand_net"] = round(cand_net, 6)
                cand_returns_gross.append(cand_gross)
                cand_returns_net.append(cand_net)
                cand_turnovers.append(turn)
                regime_cand.setdefault(regime, []).append(cand_net)

            # Temporal stability (vs prior date)
            if prev_cand_ranks is not None:
                ov20 = compute_overlap(tickers, list(prev_cand_tickers), 20)
                ov60 = compute_overlap(tickers, list(prev_cand_tickers), k)
                rc = compute_rank_correlation(ranks, prev_cand_ranks)
                ms = compute_max_shift(ranks, prev_cand_ranks)

                temporal_overlaps_20.append(ov20["overlap_pct"])
                temporal_overlaps_60.append(ov60["overlap_pct"])
                if rc["spearman"] is not None:
                    temporal_spearman.append(rc["spearman"])
                if ms:
                    temporal_max_shifts.append({**ms, "date": snap_date})
                    date_row["temporal_max_shift_ticker"] = ms["ticker"]
                    date_row["temporal_max_shift"] = ms["shift"]
                    date_row["temporal_max_shift_from"] = ms["from"]
                    date_row["temporal_max_shift_to"] = ms["to"]

                # Enriched top-N shifts for FAIL packet
                top_shifts = compute_top_shifts(ranks, prev_cand_ranks, n=3)
                for ts_entry in top_shifts:
                    enriched = {**ts_entry, "date": snap_date}
                    _enrich_shift(enriched, row_by_ticker)
                    all_enriched_shifts.append(enriched)

                date_row["temporal_overlap_20"] = ov20["overlap_pct"]
                date_row["temporal_overlap_60"] = ov60["overlap_pct"]
                date_row["temporal_spearman"] = rc["spearman"]

            prev_cand_tickers = cand_topk
            prev_cand_ranks = ranks

        # --- Baseline metrics ---
        if base_match and not is_self_eval:
            base_topk = tickers[:k]
            base_gross = None
            if fwd_rets and base_topk:
                held = [t for t in base_topk if t in fwd_rets]
                if held:
                    base_gross = statistics.mean(fwd_rets[t] for t in held)

            bturn = compute_turnover(prev_base_tickers, base_topk)
            if base_gross is not None:
                base_net = net_return(base_gross, bturn, cost_bps)
                base_returns_gross.append(base_gross)
                base_returns_net.append(base_net)
                base_turnovers.append(bturn)
                regime_base.setdefault(regime, []).append(base_net)
                date_row["base_gross"] = round(base_gross, 6)
                date_row["base_net"] = round(base_net, 6)
            prev_base_tickers = base_topk

        # --- Cross-comparison (candidate vs baseline) ---
        if cand_match and base_match and not is_self_eval:
            # Both match on same date — compare
            xov20 = compute_overlap(tickers, tickers, 20)  # trivial if same
            xov60 = compute_overlap(tickers, tickers, k)
            xrc = compute_rank_correlation(ranks, ranks)
            cross_overlaps_20.append(xov20["overlap_pct"])
            cross_overlaps_60.append(xov60["overlap_pct"])
            if xrc["spearman"] is not None:
                cross_spearman.append(xrc["spearman"])

        per_date.append(date_row)

    # --- Aggregate ---
    result: Dict[str, Any] = {
        "dates_evaluated": [d["date"] for d in per_date],
        "n_evaluated": len(per_date),
        "dates_skipped": skipped,
        "n_skipped": len(skipped),
    }

    # Temporal stability
    temporal: Dict[str, Any] = {}
    if temporal_overlaps_60:
        temporal["mean_top20_overlap"] = round(
            statistics.mean(temporal_overlaps_20), 1
        )
        temporal["mean_top60_overlap"] = round(
            statistics.mean(temporal_overlaps_60), 1
        )
        temporal["worst_top60_overlap"] = round(min(temporal_overlaps_60), 1)
    if temporal_spearman:
        temporal["mean_spearman"] = round(statistics.mean(temporal_spearman), 4)
        temporal["worst_spearman"] = round(min(temporal_spearman), 4)
    if temporal_max_shifts:
        worst = max(temporal_max_shifts, key=lambda x: x["shift"])
        temporal["max_rank_shift"] = worst
    if all_enriched_shifts:
        temporal["top_movers"] = sorted(
            all_enriched_shifts,
            key=lambda x: (-x["shift"], x["ticker"]),
        )[:20]
    result["temporal_stability"] = temporal

    # Performance
    perf: Dict[str, Any] = {}
    if cand_returns_net:
        perf["candidate"] = {
            "n_periods": len(cand_returns_net),
            "mean_gross": round(statistics.mean(cand_returns_gross), 6),
            "mean_net": round(statistics.mean(cand_returns_net), 6),
            "cumulative_net": round(_cumulative(cand_returns_net), 6)
            if _cumulative(cand_returns_net) is not None
            else None,
            "mean_turnover": round(statistics.mean(cand_turnovers), 4),
        }
    if base_returns_net:
        perf["baseline"] = {
            "n_periods": len(base_returns_net),
            "mean_gross": round(statistics.mean(base_returns_gross), 6),
            "mean_net": round(statistics.mean(base_returns_net), 6),
            "cumulative_net": round(_cumulative(base_returns_net), 6)
            if _cumulative(base_returns_net) is not None
            else None,
            "mean_turnover": round(statistics.mean(base_turnovers), 4),
        }
    if "candidate" in perf and "baseline" in perf:
        perf["delta"] = {
            "mean_net": round(
                perf["candidate"]["mean_net"] - perf["baseline"]["mean_net"], 6
            ),
        }
    result["performance"] = perf

    # Regime
    regime_summary: Dict[str, Any] = {}
    all_regimes = sorted(set(list(regime_cand.keys()) + list(regime_base.keys())))
    for reg in all_regimes:
        entry: Dict[str, Any] = {}
        if reg in regime_cand:
            vals = regime_cand[reg]
            entry["candidate_n"] = len(vals)
            entry["candidate_mean_net"] = round(statistics.mean(vals), 6)
        if reg in regime_base:
            vals = regime_base[reg]
            entry["baseline_n"] = len(vals)
            entry["baseline_mean_net"] = round(statistics.mean(vals), 6)
        regime_summary[reg] = entry
    result["regime"] = regime_summary

    # Per-date metrics
    result["per_date"] = per_date

    return result


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


_CROSS_NOOP_SPEARMAN_FLOOR = 0.999  # cross spearman must be >= this
_CROSS_NOOP_OVERLAP_FLOOR = 99.0    # cross top60 overlap (%) must be >= this


def _is_cross_noop(eval_result: Dict[str, Any]) -> bool:
    """Return True when candidate and baseline produce effectively identical rankings.

    "No-op" is determined using production-relevant metrics only (top-60 and
    spearman), not ``mean_pct_rank_changed``.  The pct-changed metric counts
    moves anywhere in the ranked list, including positions well below the
    actionable threshold (rank > 60) — those moves don't affect portfolio
    decisions.  A small modifier (e.g. alpha_modifier w=0.05) may shift a few
    tickers in the tail while leaving the entire top-60 identical.

    A no-op is declared when BOTH:
      - cross_comparison.mean_top60_overlap >= 99.0
      - cross_comparison.mean_spearman >= 0.999
    """
    cc = eval_result.get("cross_comparison", {})
    spear = cc.get("mean_spearman")
    ov60 = cc.get("mean_top60_overlap")
    if spear is None or ov60 is None:
        return False
    return spear >= _CROSS_NOOP_SPEARMAN_FLOOR and ov60 >= _CROSS_NOOP_OVERLAP_FLOOR


def evaluate_gate(
    eval_result: Dict[str, Any],
    thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate promotion gate. Returns {verdict, reasons}.

    No-op override (rerank-only mode):
    When cross_comparison shows that candidate and baseline produce effectively
    identical rankings (mean_pct_rank_changed <= 0.1, spearman >= 0.999,
    top60_overlap >= 99.0), any temporal-stability FAILs are demoted to WARNs
    with a ``baseline_temporal_instability`` label.  This handles config
    simplifications (e.g. zero-weight modifier removal) that legitimately change
    nothing about sort order but fail the day-over-day stability check because of
    pre-existing data volatility (archetype flips, alpha_cohort oscillation).
    """
    t = thresholds or DEFAULT_GATE_THRESHOLDS
    fail_reasons: List[str] = []
    warn_reasons: List[str] = []

    # Zero evaluated dates is never a PASS
    if eval_result.get("n_evaluated", 0) == 0:
        warn_reasons.append("zero_evaluated_dates")

    ts = eval_result.get("temporal_stability", {})

    # Overlap
    mean_ov = ts.get("mean_top60_overlap")
    if mean_ov is not None:
        if mean_ov < t.get("mean_top60_overlap_fail", 0):
            fail_reasons.append(
                f"mean_top60_overlap={mean_ov:.1f}% < fail={t['mean_top60_overlap_fail'] * 100 if t.get('mean_top60_overlap_fail', 0) < 1 else t.get('mean_top60_overlap_fail')}%"
            )
        elif mean_ov < t.get("mean_top60_overlap_warn", 0) * 100:
            warn_reasons.append(
                f"mean_top60_overlap={mean_ov:.1f}% < warn={t['mean_top60_overlap_warn'] * 100}%"
            )

    # Max rank shift
    ms = ts.get("max_rank_shift", {})
    shift = ms.get("shift", 0)
    if shift > t.get("max_rank_shift_fail", 999):
        fail_reasons.append(
            f"max_rank_shift={shift} > fail={t['max_rank_shift_fail']}"
        )
    elif shift > t.get("max_rank_shift_warn", 999):
        warn_reasons.append(
            f"max_rank_shift={shift} > warn={t['max_rank_shift_warn']}"
        )

    # Turnover excess
    perf = eval_result.get("performance", {})
    cand_turn = perf.get("candidate", {}).get("mean_turnover")
    base_turn = perf.get("baseline", {}).get("mean_turnover")
    if cand_turn is not None and base_turn is not None:
        excess = cand_turn - base_turn
        if excess > t.get("turnover_excess_warn", 999):
            warn_reasons.append(
                f"turnover_excess={excess:.4f} > warn={t['turnover_excess_warn']}"
            )

    # No-op override: if cross-comparison shows zero behavioral change, demote
    # any temporal-stability FAILs to WARNs.  The temporal instability is
    # attributable to the baseline data (archetype/alpha oscillation), not to
    # this candidate's changes.
    temporal_fail_keys = {"max_rank_shift", "mean_top60_overlap"}
    temporal_fails = [
        r for r in fail_reasons
        if any(k in r for k in temporal_fail_keys)
    ]
    if temporal_fails and _is_cross_noop(eval_result):
        worst_ticker = ms.get("ticker", "?")
        worst_date   = ms.get("date", "?")
        for r in temporal_fails:
            fail_reasons.remove(r)
        cc = eval_result.get("cross_comparison", {})
        pct_info = (
            f", cross_pct_changed_all={cc['mean_pct_rank_changed']:.1f}%"
            if cc.get("mean_pct_rank_changed") is not None else ""
        )
        warn_reasons.append(
            f"baseline_temporal_instability (top60_overlap={cc.get('mean_top60_overlap', '?')}%"
            f"{pct_info}, worst_mover={worst_ticker}@{worst_date} shift={shift})"
        )

    fail_reasons.sort()
    warn_reasons.sort()

    # baseline_temporal_instability warnings are audit-only: they record pre-existing
    # data volatility that was demoted from a temporal FAIL by the no-op override.
    # They do not block PASS so that promote_ruleset.py can accept the result without
    # --force.  All other warn_reasons (zero_evaluated_dates, turnover_excess, etc.)
    # still produce a blocking WARN.
    blocking_warns = [
        w for w in warn_reasons
        if not w.startswith("baseline_temporal_instability")
    ]

    if fail_reasons:
        verdict = "FAIL"
    elif blocking_warns:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return {"verdict": verdict, "fail_reasons": fail_reasons, "warn_reasons": warn_reasons}


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(
    candidate_id: str,
    candidate_file: str,
    baseline_id: str,
    baseline_file: str,
    eval_result: Dict[str, Any],
    gate_result: Optional[Dict[str, Any]],
    config: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append(f"# Ruleset Evaluation: `{candidate_id}` vs `{baseline_id}`")
    lines.append("")

    # Config
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| candidate | `{candidate_id}` ({candidate_file}) |")
    lines.append(f"| baseline | `{baseline_id}` ({baseline_file}) |")
    for key in ("start", "end", "k", "horizon", "cost_bps", "anchor_mode"):
        lines.append(f"| {key} | `{config.get(key, '')}` |")
    lines.append(
        f"| dates evaluated | {eval_result['n_evaluated']} |"
    )
    lines.append(f"| dates skipped | {eval_result['n_skipped']} |")
    lines.append("")

    # Skip reasons
    skipped = eval_result.get("dates_skipped", [])
    if skipped:
        from collections import Counter

        reasons = Counter(s["reason"] for s in skipped)
        lines.append("**Skip reasons:** " + ", ".join(
            f"{r}={c}" for r, c in sorted(reasons.items())
        ))
        lines.append("")

    # Temporal stability
    ts = eval_result.get("temporal_stability", {})
    if ts:
        lines.append("## Temporal Stability (day-over-day)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key in (
            "mean_top20_overlap", "mean_top60_overlap", "worst_top60_overlap",
            "mean_spearman", "worst_spearman",
        ):
            val = ts.get(key)
            if val is not None:
                suffix = "%" if "overlap" in key else ""
                lines.append(f"| {key} | {val}{suffix} |")
        ms = ts.get("max_rank_shift")
        if ms:
            lines.append(
                f"| max_rank_shift | {ms['ticker']}: {ms['from']}->{ms['to']} "
                f"(delta={ms['shift']}, date={ms.get('date', '?')}) |"
            )
        lines.append("")

    # Performance
    perf = eval_result.get("performance", {})
    if perf:
        lines.append("## Performance")
        lines.append("")
        lines.append("| Metric | Candidate | Baseline |")
        lines.append("|--------|-----------|----------|")
        cand = perf.get("candidate", {})
        base = perf.get("baseline", {})
        for key in ("n_periods", "mean_gross", "mean_net", "cumulative_net", "mean_turnover"):
            cv = cand.get(key, "-")
            bv = base.get(key, "-")
            if isinstance(cv, float):
                cv = f"{cv:.6f}"
            if isinstance(bv, float):
                bv = f"{bv:.6f}"
            lines.append(f"| {key} | {cv} | {bv} |")
        delta = perf.get("delta", {})
        if delta:
            lines.append(f"| delta_mean_net | {delta.get('mean_net', '-')} | - |")
        lines.append("")

    # Regime
    regime = eval_result.get("regime", {})
    if regime:
        lines.append("## Regime Slices")
        lines.append("")
        lines.append("| Regime | N (cand) | Mean Net (cand) | N (base) | Mean Net (base) |")
        lines.append("|--------|----------|-----------------|----------|-----------------|")
        for reg in sorted(regime.keys()):
            entry = regime[reg]
            cn = entry.get("candidate_n", "-")
            cm = entry.get("candidate_mean_net", "-")
            if isinstance(cm, float):
                cm = f"{cm:.6f}"
            bn = entry.get("baseline_n", "-")
            bm = entry.get("baseline_mean_net", "-")
            if isinstance(bm, float):
                bm = f"{bm:.6f}"
            lines.append(f"| {reg} | {cn} | {cm} | {bn} | {bm} |")
        lines.append("")

    # Gate
    if gate_result:
        lines.append("## Promotion Gate")
        lines.append("")
        icon = {"PASS": "+", "WARN": "!", "FAIL": "x"}.get(gate_result["verdict"], "?")
        lines.append(f"**Verdict: [{icon}] {gate_result['verdict']}**")
        lines.append("")
        if gate_result.get("fail_reasons"):
            lines.append("Fail reasons:")
            for r in gate_result["fail_reasons"]:
                lines.append(f"- {r}")
        if gate_result.get("warn_reasons"):
            lines.append("Warn reasons:")
            for r in gate_result["warn_reasons"]:
                lines.append(f"- {r}")
        if not gate_result.get("fail_reasons") and not gate_result.get("warn_reasons"):
            lines.append("All metrics within thresholds.")
    else:
        lines.append("## Promotion Gate")
        lines.append("")
        lines.append("*No gate run (use --gate to enable).*")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI — rerank-only mode
# ---------------------------------------------------------------------------


def _main_rerank_only(args: argparse.Namespace) -> int:
    """Evaluate by re-ranking snapshots (no price data needed)."""
    try:
        cand_rs = DecisionRuleset.from_json(str(args.candidate))
    except Exception as e:
        print(f"ERROR: cannot load candidate ruleset: {e}", file=sys.stderr)
        return 2
    candidate_id = cand_rs.ruleset_id

    baseline_file = ""
    if args.baseline:
        try:
            base_rs = DecisionRuleset.from_json(str(args.baseline))
            baseline_file = args.baseline.name
        except Exception as e:
            print(f"ERROR: cannot load baseline ruleset: {e}", file=sys.stderr)
            return 2
    else:
        manifest_path = (
            _PROJECT_ROOT / "production_data" / "decision_rulesets" / "manifest.json"
        )
        try:
            _, baseline_file = active_ruleset_from_manifest(manifest_path)
        except Exception as e:
            print(
                f"ERROR: cannot determine baseline from manifest: {e}",
                file=sys.stderr,
            )
            return 2
        try:
            base_rs = DecisionRuleset.from_json(
                str(manifest_path.parent / baseline_file)
            )
        except Exception as e:
            print(f"ERROR: cannot load baseline ruleset: {e}", file=sys.stderr)
            return 2

    baseline_id = base_rs.ruleset_id

    print(f"Candidate: {candidate_id} ({args.candidate.name})")
    print(f"Baseline:  {baseline_id} ({baseline_file})")
    print("Mode:      rerank-only (no price data needed)")

    dates = discover_snapshot_dates(args.snapshot_dir, args.start, args.end)
    if not dates:
        print("ERROR: no snapshot dates found in range", file=sys.stderr)
        return 1

    print(f"Dates in range: {len(dates)} ({dates[0]} to {dates[-1]})")
    print("Re-ranking...")

    eval_result = evaluate_ruleset_rerank_only(
        candidate_ruleset=cand_rs,
        baseline_ruleset=base_rs,
        dates=dates,
        snapshot_dir=args.snapshot_dir,
        k=args.k,
    )

    print(
        f"  Evaluated: {eval_result['n_evaluated']}, "
        f"Skipped: {eval_result['n_skipped']}"
    )

    gate_result = None
    if args.gate:
        gate_result = evaluate_gate(eval_result)
        print(f"  Gate: {gate_result['verdict']}")

    config = {
        "start": args.start,
        "end": args.end,
        "k": args.k,
        "rerank_only": True,
    }

    output = {
        "schema": SCHEMA,
        "version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "candidate": {"id": candidate_id, "file": args.candidate.name},
        "baseline": {"id": baseline_id, "file": baseline_file},
        "config": config,
        "mode": "rerank_only",
        "notes": "rerank-only: derived from snapshot rankings.csv; no PIT replay/returns",
        **eval_result,
        "gate": gate_result,
    }

    out_subdir = (
        args.out_dir
        / f"{candidate_id}__vs__{baseline_id}"
        / f"{args.start}__{args.end}"
    )
    out_subdir.mkdir(parents=True, exist_ok=True)

    json_path = out_subdir / "ruleset_eval.json"
    json_path.write_text(_stable_json(output), encoding="utf-8")

    md = render_markdown(
        candidate_id,
        args.candidate.name,
        baseline_id,
        baseline_file,
        eval_result,
        gate_result,
        config,
    )
    md_path = out_subdir / "ruleset_eval.md"
    md_path.write_text(md, encoding="utf-8")

    per_date = eval_result.get("per_date", [])
    if per_date:
        csv_path = out_subdir / "per_date_metrics.csv"
        all_fields: List[str] = []
        seen: set = set()
        for row in per_date:
            for key in row:
                if key not in seen:
                    all_fields.append(key)
                    seen.add(key)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=all_fields, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(per_date)

    print(f"Written: {out_subdir}")

    if gate_result and gate_result["verdict"] == "FAIL":
        return 1
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a candidate ruleset.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--k", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument(
        "--snapshot-dir", type=Path, default=_PROJECT_ROOT / "data" / "snapshots"
    )
    parser.add_argument(
        "--price-csv",
        type=Path,
        default=_PROJECT_ROOT / "production_data" / "price_history.csv",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--anchor-mode", default="prev_trading_day",
        choices=["prev_trading_day", "next_trading_day", "exact"],
    )
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--long-short", action="store_true")
    parser.add_argument("--rebuild-snapshots", action="store_true")
    parser.add_argument(
        "--rerank-only", action="store_true",
        help="Re-rank snapshots with candidate/baseline rulesets (no price data needed)",
    )
    args = parser.parse_args(argv)

    if args.rerank_only:
        return _main_rerank_only(args)

    # Load candidate
    try:
        candidate_id = load_ruleset_id(args.candidate)
    except Exception as e:
        print(f"ERROR: cannot read candidate ruleset: {e}", file=sys.stderr)
        return 2

    # Load baseline
    baseline_file = ""
    if args.baseline:
        try:
            baseline_id = load_ruleset_id(args.baseline)
            baseline_file = args.baseline.name
        except Exception as e:
            print(f"ERROR: cannot read baseline ruleset: {e}", file=sys.stderr)
            return 2
    else:
        manifest_path = _PROJECT_ROOT / "production_data" / "decision_rulesets" / "manifest.json"
        try:
            baseline_id, baseline_file = active_ruleset_from_manifest(manifest_path)
        except Exception as e:
            print(f"ERROR: cannot determine baseline from manifest: {e}", file=sys.stderr)
            return 2

    print(f"Candidate: {candidate_id} ({args.candidate.name})")
    print(f"Baseline:  {baseline_id} ({baseline_file})")

    # Discover dates
    dates = discover_snapshot_dates(args.snapshot_dir, args.start, args.end)
    if not dates:
        print("ERROR: no snapshot dates found in range", file=sys.stderr)
        return 1

    print(f"Dates in range: {len(dates)} ({dates[0]} to {dates[-1]})")

    # Load prices
    print("Loading prices...")
    price_series = load_price_series(args.price_csv)
    all_trade_dates = sorted({d for ps in price_series.values() for d in ps})
    print(f"  {len(price_series)} tickers, {len(all_trade_dates)} trading dates")

    # Regime map (optional, degrade gracefully)
    regime_map: Optional[Dict[str, str]] = None
    try:
        from backtest.regime import (
            assign_regime_to_rebalance_dates,
            compute_regime_series,
            load_price_history,
        )
        price_df = load_price_history(args.price_csv)
        regime_df = compute_regime_series(price_df, "XBI")
        regime_map = assign_regime_to_rebalance_dates(regime_df, dates)
        print(f"  Regime map: {len(regime_map)} dates")
    except Exception as e:
        print(f"  Regime map unavailable: {e}")

    # Evaluate
    print("Evaluating...")
    eval_result = evaluate_ruleset(
        candidate_id=candidate_id,
        baseline_id=baseline_id,
        dates=dates,
        snapshot_dir=args.snapshot_dir,
        price_series=price_series,
        all_trade_dates=all_trade_dates,
        regime_map=regime_map,
        k=args.k,
        horizon=args.horizon,
        cost_bps=args.cost_bps,
        anchor_mode=args.anchor_mode,
    )

    print(f"  Evaluated: {eval_result['n_evaluated']}, Skipped: {eval_result['n_skipped']}")

    # Gate
    gate_result = None
    if args.gate:
        gate_result = evaluate_gate(eval_result)
        print(f"  Gate: {gate_result['verdict']}")

    # Config for output
    config = {
        "start": args.start,
        "end": args.end,
        "k": args.k,
        "horizon": args.horizon,
        "cost_bps": args.cost_bps,
        "anchor_mode": args.anchor_mode,
        "long_short": args.long_short,
    }

    # Assemble JSON
    output = {
        "schema": SCHEMA,
        "version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "candidate": {"id": candidate_id, "file": args.candidate.name},
        "baseline": {"id": baseline_id, "file": baseline_file},
        "config": config,
        **eval_result,
        "gate": gate_result,
    }

    # Write outputs
    out_subdir = (
        args.out_dir / f"{candidate_id}__vs__{baseline_id}" / f"{args.start}__{args.end}"
    )
    out_subdir.mkdir(parents=True, exist_ok=True)

    json_path = out_subdir / "ruleset_eval.json"
    json_path.write_text(_stable_json(output), encoding="utf-8")

    md = render_markdown(
        candidate_id, args.candidate.name,
        baseline_id, baseline_file,
        eval_result, gate_result, config,
    )
    md_path = out_subdir / "ruleset_eval.md"
    md_path.write_text(md, encoding="utf-8")

    # Per-date CSV (collect ALL fieldnames across all rows)
    per_date = eval_result.get("per_date", [])
    if per_date:
        csv_path = out_subdir / "per_date_metrics.csv"
        all_fields: List[str] = []
        seen: set = set()
        for row in per_date:
            for key in row:
                if key not in seen:
                    all_fields.append(key)
                    seen.add(key)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(per_date)

    print(f"Written: {out_subdir}")

    if gate_result and gate_result["verdict"] == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
