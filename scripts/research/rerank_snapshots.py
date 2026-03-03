#!/usr/bin/env python3
"""Re-rank historical snapshots through a target ruleset.

RESEARCH ONLY — not for production use.

Inputs:  snapshot-root with rankings.csv per date, ruleset JSON
Outputs: data/snapshots_reranked/{date}/rankings.csv + metadata.json

Reads each snapshot's rankings.csv, backfills missing columns with safe
defaults, re-computes the sort key using the target ruleset, assigns fresh
actionable_rank 1..N, and writes results to a staging directory.

This allows forward-return evaluation on a consistent ranking model
across dates that were originally produced with different rulesets.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.ranking_utils import backfill_columns, safe_float as _safe_float
from decision_engine import DecisionRuleset, compute_actionable_sort_key


# ---------------------------------------------------------------------------
# PIT coinvest helpers
# ---------------------------------------------------------------------------

# 13F filings become public ~45 days after quarter-end (SEC 45-day deadline).
_PIT_LAG_DAYS = 45


def _load_coinvest_pit_features(
    features_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """Load all pre-built coinvest feature files from *features_dir*.

    Returns a dict mapping ``quarter_end`` (ISO str) → full features dict.
    Only files whose name matches ``YYYY-MM-DD.json`` are loaded.
    """
    features: Dict[str, Dict[str, Any]] = {}
    if not features_dir.is_dir():
        return features
    for p in sorted(features_dir.glob("????-??-??.json")):
        try:
            with open(p) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            print(f"  WARNING: failed to load coinvest features from {p}")
            continue
        qe = p.stem  # filename without .json is the quarter_end date
        features[qe] = data
    return features


def _select_pit_quarter(
    snap_date_str: str,
    available_quarters: List[str],
    lag_days: int = _PIT_LAG_DAYS,
) -> Optional[str]:
    """Select the latest quarter_end where ``quarter_end + lag_days <= snap_date``.

    This replicates the 13F PIT rule: a filing becomes publicly available
    approximately 45 days after quarter-end (SEC filing deadline).

    Parameters
    ----------
    snap_date_str : str
        The snapshot date (ISO format, e.g. ``"2025-03-15"``).
    available_quarters : list[str]
        Sorted list of available quarter-end dates (ISO format).
    lag_days : int
        Number of days after quarter-end before the filing is public.

    Returns
    -------
    str | None
        The latest eligible quarter-end, or None if none qualify.
    """
    try:
        snap_date = date.fromisoformat(snap_date_str)
    except (ValueError, TypeError):
        return None

    best: Optional[str] = None
    for qe_str in sorted(available_quarters):
        try:
            qe = date.fromisoformat(qe_str)
        except (ValueError, TypeError):
            continue
        if qe + timedelta(days=lag_days) <= snap_date:
            best = qe_str
    return best


def _inject_coinvest_pit(
    rows: List[Dict[str, str]],
    tier1_counts: Dict[str, int],
) -> None:
    """Recompute ``coinvest_score_z`` cross-sectionally using PIT tier1_count.

    Matches the z-score formula used in run_screen.py (lines 3634-3647):
    population mean/std (ddof=0) across all rows in the snapshot.
    Writes results back into each row dict in-place.

    Parameters
    ----------
    rows : list[dict]
        Snapshot rows (mutated in-place).
    tier1_counts : dict[str, int]
        Ticker → tier1_count from the PIT coinvest features file.
    """
    vals = [float(tier1_counts.get(r.get("ticker", ""), 0)) for r in rows]
    n = len(vals)
    if n == 0:
        return
    mean_v = sum(vals) / n
    var_v = sum((v - mean_v) ** 2 for v in vals) / n
    std_v = var_v ** 0.5
    for r, v in zip(rows, vals):
        r["coinvest_score_z"] = str(
            round((v - mean_v) / std_v, 4) if std_v > 0 else 0.0
        )


def _select_prev_quarter(
    current_qe: str,
    available_quarters: List[str],
) -> Optional[str]:
    """Return the quarter immediately before *current_qe* in *available_quarters*.

    Returns None if *current_qe* is the first available quarter or not found.
    """
    sorted_q = sorted(available_quarters)
    try:
        idx = sorted_q.index(current_qe)
    except ValueError:
        return None
    return sorted_q[idx - 1] if idx > 0 else None


def _inject_coinvest_pit_delta(
    rows: List[Dict[str, str]],
    tier1_counts_curr: Dict[str, int],
    tier1_counts_prev: Optional[Dict[str, int]],
) -> None:
    """Compute and inject ``coinvest_delta_z`` = z-score of q/q tier1_count change.

    When *tier1_counts_prev* is None (no prior quarter available), all deltas
    are treated as 0 and ``coinvest_delta_z`` is set to "0.0" for every row.

    Writes results into each row dict in-place as ``coinvest_delta_z``.
    The caller should route this into the sort slot via ``--col-alias``
    (e.g. ``--col-alias coinvest_score_z=coinvest_delta_z``).

    Parameters
    ----------
    rows : list[dict]
        Snapshot rows (mutated in-place).
    tier1_counts_curr : dict[str, int]
        Ticker → tier1_count for the selected PIT quarter.
    tier1_counts_prev : dict[str, int] | None
        Ticker → tier1_count for the prior PIT quarter, or None if unavailable.
    """
    if tier1_counts_prev is None:
        for r in rows:
            r["coinvest_delta_z"] = "0.0"
        return

    vals = []
    for r in rows:
        tk = r.get("ticker", "")
        curr = float(tier1_counts_curr.get(tk, 0))
        prev = float(tier1_counts_prev.get(tk, 0))
        vals.append(curr - prev)

    n = len(vals)
    if n == 0:
        return
    mean_v = sum(vals) / n
    var_v = sum((v - mean_v) ** 2 for v in vals) / n
    std_v = var_v ** 0.5
    for r, v in zip(rows, vals):
        r["coinvest_delta_z"] = str(
            round((v - mean_v) / std_v, 4) if std_v > 0 else 0.0
        )


# ---------------------------------------------------------------------------
# Re-ranking
# ---------------------------------------------------------------------------

def rerank(
    rows: List[Dict[str, str]],
    ruleset: DecisionRuleset,
    col_aliases: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    """Re-sort rows using the target ruleset and assign fresh actionable_rank.

    col_aliases: maps destination column → source column.  Before sorting,
    each row's destination value is overwritten with the source value.
    E.g. ``{"clinical_score_z_tier": "clinical_alpha_z"}`` routes the
    clinical_alpha_z signal into the clinical-sort slot.
    """
    backfill_columns(rows)

    # Apply column aliases (pre-processing substitution for ablations)
    if col_aliases:
        for row in rows:
            for dest, src in col_aliases.items():
                row[dest] = row.get(src, "")

    # Sort using production-identical logic
    rows.sort(key=lambda r: compute_actionable_sort_key(
        decision_fields=r,
        archetype=r.get("archetype", ""),
        optionality=_safe_float(r.get("clinical_optionality_pct_dev")),
        composite_rank=r.get("composite_rank"),
        ticker=r.get("ticker", ""),
        catalyst_event_type=r.get("catalyst_event_type", ""),
        catalyst_source=r.get("catalyst_source", ""),
        ruleset=ruleset,
        tiebreaker_pct=(
            _safe_float(r.get("alpha_cohort_pct"))
            if ruleset.sort_anchor == "alpha_cohort"
            else (_safe_float(r.get("commercial_quality_pct"))
                  if r.get("archetype", "").startswith("commercial_")
                  else _safe_float(r.get("clinical_optionality_pct_dev")))
        ),
    ))

    # Assign actionable_rank
    rank = 1
    for r in rows:
        if r.get("eligible") == "1":
            r["actionable_rank"] = str(rank)
            rank += 1
        else:
            r["actionable_rank"] = ""

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-rank historical snapshots through a target ruleset",
    )
    parser.add_argument(
        "--snapshot-root", type=Path,
        default=PROJECT_ROOT / "data" / "snapshots",
    )
    parser.add_argument(
        "--out-root", type=Path,
        default=PROJECT_ROOT / "data" / "snapshots_reranked",
    )
    parser.add_argument(
        "--ruleset", type=Path, default=None,
        help="Path to ruleset JSON. Default: latest snapshot's decision_ruleset.json",
    )
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    parser.add_argument(
        "--min-cols", type=int, default=50,
        help="Skip snapshots with fewer than this many columns (default: 50)",
    )
    parser.add_argument(
        "--col-alias", action="append", default=[], metavar="DEST=SRC",
        help="Substitute column values before ranking. Format: DEST=SRC. "
             "E.g. --col-alias clinical_score_z_tier=clinical_alpha_z "
             "routes clinical_alpha_z into the clinical-sort slot. "
             "Can be repeated for multiple substitutions.",
    )
    parser.add_argument(
        "--coinvest-pit", action="store_true", default=False,
        help="Inject PIT-correct coinvest_score_z from pre-built quarterly "
             "features. Loads files from data/caches/sec_13f/coinvest_features/ "
             "and selects the latest quarter_end where "
             "quarter_end + 45 days <= snap_date.",
    )
    parser.add_argument(
        "--coinvest-pit-dir", type=Path,
        default=PROJECT_ROOT / "data" / "caches" / "sec_13f" / "coinvest_features",
        help="Directory containing pre-built coinvest feature JSON files. "
             "Defaults to data/caches/sec_13f/coinvest_features/.",
    )
    parser.add_argument(
        "--coinvest-delta", action="store_true", default=False,
        help="Inject coinvest_delta_z (q/q change in tier1_count, cross-sectionally "
             "z-scored) instead of tier1_count level. Requires --coinvest-pit. "
             "Route into sort via --col-alias coinvest_score_z=coinvest_delta_z.",
    )
    parser.add_argument(
        "--coinvest-negate", action="store_true", default=False,
        help="Negate the injected coinvest signal (coinvest_score_z or coinvest_delta_z) "
             "after PIT injection. Use with a ruleset that has coinvest_positive_only=false "
             "to implement a full-range CONTRA signal (penalise crowded names, reward "
             "uncrowded). Requires --coinvest-pit.",
    )
    parser.add_argument(
        "--require-pit-coinvest", action="store_true", default=False,
        help="Abort if --coinvest-pit is not also passed. Use this for formal "
             "evaluation runs to prevent accidental use of static holdings data.",
    )
    parser.add_argument(
        "--clinical-negate", action="store_true", default=False,
        help="Negate clinical_score_z_tier in every snapshot row before re-ranking. "
             "Used for CONTRA ablation: high-score names are penalised, low-score "
             "names are boosted. Pair with a ruleset that has clinical_positive_only=false "
             "for a full bidirectional CONTRA signal.",
    )
    parser.add_argument(
        "--preflight", action="store_true", default=False,
        help="Run snapshot preflight; skip dates that FAIL structural checks",
    )
    args = parser.parse_args()

    # Guardrail: validate coinvest flags
    if args.coinvest_negate and not args.coinvest_pit:
        print("ERROR: --coinvest-negate requires --coinvest-pit")
        sys.exit(1)
    if args.coinvest_delta and not args.coinvest_pit:
        print("ERROR: --coinvest-delta requires --coinvest-pit")
        sys.exit(1)
    if args.require_pit_coinvest and not args.coinvest_pit:
        print(
            "ERROR: --require-pit-coinvest is set but --coinvest-pit was not passed.\n"
            "  Re-ranked snapshots would use coinvest_score_z values already stored\n"
            "  in each snapshot, which may come from the static\n"
            "  production_data/holdings_snapshots.json (Q3 2025, no date field).\n"
            "  Add --coinvest-pit to inject PIT-correct quarterly values."
        )
        sys.exit(1)
    if not args.coinvest_pit:
        print(
            "WARNING: --coinvest-pit not set. Re-ranked snapshots will retain the\n"
            "  coinvest_score_z values already stored in each snapshot CSV. For\n"
            "  snapshots produced before the PIT pipeline existed, these values came\n"
            "  from production_data/holdings_snapshots.json (static Q3 2025 dump,\n"
            "  no date field — introduces look-ahead bias in historical backtests).\n"
            "  Metadata will be stamped with coinvest_source='static_holdings_snapshots'.\n"
            "  Pass --coinvest-pit to inject PIT-correct values.\n"
        )

    # Parse column aliases
    col_aliases: Optional[Dict[str, str]] = None
    if args.col_alias:
        col_aliases = {}
        for alias in args.col_alias:
            if "=" not in alias:
                print(f"ERROR: --col-alias must be DEST=SRC, got: {alias!r}")
                sys.exit(1)
            dest, src = alias.split("=", 1)
            col_aliases[dest.strip()] = src.strip()
        print(f"Column aliases: {col_aliases}")

    # Load ruleset
    if args.ruleset:
        ruleset = DecisionRuleset.from_json(str(args.ruleset))
    else:
        # Find latest snapshot with decision_ruleset.json
        dates = sorted([p.name for p in args.snapshot_root.iterdir()
                        if p.is_dir() and len(p.name) == 10 and p.name[4] == "-"
                        and (p / "decision_ruleset.json").exists()])
        if not dates:
            print("ERROR: No snapshot with decision_ruleset.json found")
            sys.exit(1)
        rs_path = args.snapshot_root / dates[-1] / "decision_ruleset.json"
        ruleset = DecisionRuleset.from_json(str(rs_path))
        print(f"Using ruleset from {rs_path} (ID: {ruleset.ruleset_id})")

    # Discover snapshots
    snap_dates = sorted([
        p.name for p in args.snapshot_root.iterdir()
        if p.is_dir() and len(p.name) == 10 and p.name[4] == "-"
        and (p / "rankings.csv").exists()
    ])

    if args.date_from:
        snap_dates = [d for d in snap_dates if d >= args.date_from]
    if args.date_to:
        snap_dates = [d for d in snap_dates if d <= args.date_to]

    print(f"Re-ranking {len(snap_dates)} snapshots → {args.out_root}")

    # Load PIT coinvest features once if requested
    coinvest_pit_features: Optional[Dict[str, Dict[str, Any]]] = None
    coinvest_quarters_sorted: List[str] = []
    if args.coinvest_pit:
        coinvest_pit_features = _load_coinvest_pit_features(args.coinvest_pit_dir)
        coinvest_quarters_sorted = sorted(coinvest_pit_features.keys())
        print(
            f"Loaded PIT coinvest features: {len(coinvest_pit_features)} quarters "
            f"({coinvest_quarters_sorted[0] if coinvest_quarters_sorted else 'none'} – "
            f"{coinvest_quarters_sorted[-1] if coinvest_quarters_sorted else 'none'})"
        )

    # Track sample PIT quarter mappings for provenance (first 5 printed)
    pit_sample_mappings: List[Tuple[str, Optional[str]]] = []

    n_ok = 0
    n_skip = 0
    for snap_date in snap_dates:
        src = args.snapshot_root / snap_date
        csv_path = src / "rankings.csv"

        # Load rankings
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
            rows = list(reader)

        if len(cols) < args.min_cols:
            print(f"  {snap_date}: SKIP ({len(cols)} cols < {args.min_cols})")
            n_skip += 1
            continue

        # Preflight check
        if args.preflight:
            from tools.snapshot_preflight import run_preflight as _run_preflight
            pf = _run_preflight(src, snap_date)
            if pf.status == "FAIL":
                details = "; ".join(
                    c.detail for c in pf.checks if c.status == "FAIL"
                )
                print(f"  {snap_date}: SKIP (preflight_FAIL: {details})")
                n_skip += 1
                continue

        # Inject PIT-correct coinvest signal if requested
        chosen_pit_quarter: Optional[str] = None
        if args.coinvest_pit and coinvest_pit_features is not None:
            chosen_pit_quarter = _select_pit_quarter(
                snap_date, coinvest_quarters_sorted
            )
            if len(pit_sample_mappings) < 5:
                pit_sample_mappings.append((snap_date, chosen_pit_quarter))
            if chosen_pit_quarter is not None:
                feat = coinvest_pit_features[chosen_pit_quarter]
                tier1_counts = {
                    tk: td.get("tier1_count", 0)
                    for tk, td in feat.get("tickers", {}).items()
                }
                if args.coinvest_delta:
                    # Q/Q delta mode: inject coinvest_delta_z
                    prev_qe = _select_prev_quarter(
                        chosen_pit_quarter, coinvest_quarters_sorted
                    )
                    feat_prev = coinvest_pit_features.get(prev_qe) if prev_qe else None
                    tier1_counts_prev: Optional[Dict[str, int]] = (
                        {
                            tk: td.get("tier1_count", 0)
                            for tk, td in feat_prev.get("tickers", {}).items()
                        }
                        if feat_prev else None
                    )
                    _inject_coinvest_pit_delta(rows, tier1_counts, tier1_counts_prev)
                else:
                    # Level mode: inject coinvest_score_z
                    _inject_coinvest_pit(rows, tier1_counts)
            else:
                # No eligible quarter — zero out coinvest signal
                if args.coinvest_delta:
                    for r in rows:
                        r["coinvest_delta_z"] = "0.0"
                else:
                    for r in rows:
                        r["coinvest_score_z"] = "0.0"

        # CONTRA: negate the injected coinvest column so crowded names are penalised
        if args.coinvest_pit and args.coinvest_negate:
            col_to_negate = "coinvest_delta_z" if args.coinvest_delta else "coinvest_score_z"
            for r in rows:
                try:
                    r[col_to_negate] = str(round(-float(r[col_to_negate]), 4))
                except (ValueError, TypeError):
                    pass

        # CONTRA: negate clinical_score_z_tier so low-scoring names are boosted
        if args.clinical_negate:
            for r in rows:
                try:
                    r["clinical_score_z_tier"] = str(
                        round(-float(r["clinical_score_z_tier"]), 4)
                    )
                except (ValueError, TypeError):
                    pass

        # Re-rank
        rows = rerank(rows, ruleset, col_aliases=col_aliases)

        # Write output
        out_dir = args.out_root / snap_date
        out_dir.mkdir(parents=True, exist_ok=True)

        # Use the full column set from the re-ranked rows
        out_cols = list(rows[0].keys()) if rows else cols
        with open(out_dir / "rankings.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        # Copy metadata.json (or create minimal one)
        meta_src = src / "metadata.json"
        meta = {}
        if meta_src.exists():
            with open(meta_src, encoding="utf-8") as f:
                meta = json.load(f)
        meta["as_of_date"] = snap_date
        meta["reranked_with_ruleset"] = ruleset.ruleset_id
        if args.clinical_negate:
            meta["clinical_negate"] = True
        if args.coinvest_pit:
            suffix = "_contra" if args.coinvest_negate else ""
            if args.coinvest_delta:
                meta["coinvest_source"] = f"PIT_13F_delta{suffix}"
            else:
                meta["coinvest_source"] = f"PIT_13F{suffix}"
            meta["coinvest_pit_quarter_end"] = chosen_pit_quarter or ""
        else:
            meta["coinvest_source"] = "static_holdings_snapshots"
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

        n_eligible = sum(1 for r in rows if r.get("eligible") == "1")
        print(f"  {snap_date}: OK ({len(rows)} rows, {n_eligible} eligible)")
        n_ok += 1

    if args.coinvest_pit and pit_sample_mappings:
        print("\nPIT quarter mapping (first 5 snapshot dates):")
        for sd, qe in pit_sample_mappings:
            print(f"  {sd} → used quarter {qe or 'NONE (no eligible quarter)'}")

    print(f"\nDone: {n_ok} re-ranked, {n_skip} skipped")


if __name__ == "__main__":
    main()
