#!/usr/bin/env python3
"""
Decision Engine Diagnostics.

Reads snapshots from data/snapshots/YYYY-MM-DD/rankings.csv and computes:
- Tier distribution per date (A/B/C/D counts)
- Tier churn between consecutive dates
- Top-20/60 Jaccard overlap + Spearman rank correlation
- Score contribution distribution by tier
- Missing-data sensitivity analysis

Usage:
    python3 scripts/decision_engine_diagnostics.py [--from-date 2026-01-15] [--to-date 2026-02-16]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_rankings(snapshot_dir: Path) -> list[dict]:
    csv_path = snapshot_dir / "rankings.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _discover_snapshots(
    base: Path, from_date: date | None, to_date: date | None,
) -> list[tuple[str, Path]]:
    """Return sorted (date_str, path) for snapshots in range."""
    results = []
    if not base.exists():
        return results
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        try:
            dt = date.fromisoformat(d.name)
        except ValueError:
            continue
        if from_date and dt < from_date:
            continue
        if to_date and dt > to_date:
            continue
        results.append((d.name, d))
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v: str) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _safe_int(v: str) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _spearman_corr(ranks_a: dict[str, int], ranks_b: dict[str, int]) -> float | None:
    """Spearman rank correlation between two {ticker: rank} dicts on common tickers."""
    common = set(ranks_a) & set(ranks_b)
    n = len(common)
    if n < 3:
        return None
    tickers = sorted(common)
    ra = [ranks_a[t] for t in tickers]
    rb = [ranks_b[t] for t in tickers]
    # Convert to ranks within common set
    def _rank(vals):
        indexed = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0] * len(vals)
        for rank, idx in enumerate(indexed, 1):
            ranks[idx] = rank
        return ranks
    ra_r = _rank(ra)
    rb_r = _rank(rb)
    d_sq = sum((a - b) ** 2 for a, b in zip(ra_r, rb_r))
    return 1 - (6 * d_sq) / (n * (n ** 2 - 1))


def _percentile(values: list[float], p: float) -> float:
    """Simple linear interpolation percentile."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "p25": None, "p75": None, "n": 0}
    return {
        "mean": round(sum(values) / len(values), 4),
        "median": round(_percentile(values, 0.5), 4),
        "p25": round(_percentile(values, 0.25), 4),
        "p75": round(_percentile(values, 0.75), 4),
        "n": len(values),
    }


# ---------------------------------------------------------------------------
# Diagnostic computations
# ---------------------------------------------------------------------------

def compute_tier_distribution(snapshots: list[tuple[str, list[dict]]]) -> list[dict]:
    """A/B/C/D/missing counts per date."""
    results = []
    for date_str, rows in snapshots:
        counts = Counter(r.get("tier_dev", "") for r in rows)
        results.append({
            "date": date_str,
            "total": len(rows),
            "A": counts.get("A", 0),
            "B": counts.get("B", 0),
            "C": counts.get("C", 0),
            "D": counts.get("D", 0),
            "no_tier": counts.get("", 0),
        })
    return results


def compute_tier_churn(snapshots: list[tuple[str, list[dict]]]) -> list[dict]:
    """Count names that changed tier between consecutive dates."""
    results = []
    for i in range(1, len(snapshots)):
        prev_date, prev_rows = snapshots[i - 1]
        curr_date, curr_rows = snapshots[i]
        prev_tiers = {r["ticker"]: r.get("tier_dev", "") for r in prev_rows}
        curr_tiers = {r["ticker"]: r.get("tier_dev", "") for r in curr_rows}
        common = set(prev_tiers) & set(curr_tiers)
        changed = sum(1 for t in common if prev_tiers[t] != curr_tiers[t])
        # Track direction of changes
        upgrades = sum(1 for t in common
                       if prev_tiers[t] in ("B", "C", "D") and curr_tiers[t] == "A")
        downgrades = sum(1 for t in common
                         if prev_tiers[t] == "A" and curr_tiers[t] in ("B", "C", "D"))
        results.append({
            "from_date": prev_date,
            "to_date": curr_date,
            "common_tickers": len(common),
            "tier_changes": changed,
            "churn_pct": round(100 * changed / len(common), 2) if common else 0,
            "A_upgrades": upgrades,
            "A_downgrades": downgrades,
        })
    return results


def compute_topk_stability(
    snapshots: list[tuple[str, list[dict]]], k: int = 20,
) -> list[dict]:
    """Jaccard overlap + Spearman of top-K actionable_rank across consecutive dates."""
    results = []
    for i in range(1, len(snapshots)):
        prev_date, prev_rows = snapshots[i - 1]
        curr_date, curr_rows = snapshots[i]

        def _topk_set(rows):
            ranked = [(r["ticker"], _safe_int(r.get("actionable_rank")))
                      for r in rows if r.get("actionable_rank")]
            ranked.sort(key=lambda x: x[1] or 9999)
            return {t for t, rank in ranked[:k]}

        def _rank_dict(rows):
            return {r["ticker"]: _safe_int(r.get("actionable_rank"))
                    for r in rows if r.get("actionable_rank") and _safe_int(r.get("actionable_rank"))}

        prev_top = _topk_set(prev_rows)
        curr_top = _topk_set(curr_rows)
        jaccard = _jaccard(prev_top, curr_top)

        prev_ranks = _rank_dict(prev_rows)
        curr_ranks = _rank_dict(curr_rows)
        spearman = _spearman_corr(prev_ranks, curr_ranks)

        results.append({
            "from_date": prev_date,
            "to_date": curr_date,
            "k": k,
            "jaccard": round(jaccard, 4),
            "spearman": round(spearman, 4) if spearman is not None else None,
        })
    return results


def compute_score_contributions(
    snapshots: list[tuple[str, list[dict]]],
) -> dict[str, dict]:
    """Mean/median of component scores, grouped by tier_dev."""
    SCORE_COLS = ["clinical_score", "catalyst_score", "momentum_score",
                  "financial_score", "smart_money_score"]
    by_tier: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for _, rows in snapshots:
        for r in rows:
            tier = r.get("tier_dev", "none")
            if tier == "":
                tier = "none"
            for col in SCORE_COLS:
                v = _safe_float(r.get(col))
                if v is not None:
                    by_tier[tier][col].append(v)

    result = {}
    for tier in sorted(by_tier):
        result[tier] = {col: _stats(by_tier[tier][col]) for col in SCORE_COLS}
    return result


def compute_tier_reason_distribution(
    snapshots: list[tuple[str, list[dict]]],
) -> dict[str, int]:
    """Aggregate tier_reason counts across all dates."""
    counts: Counter = Counter()
    for _, rows in snapshots:
        for r in rows:
            reason = r.get("tier_reason", "")
            if reason:
                counts[reason] += 1
    return dict(counts.most_common())


def compute_missing_data(
    snapshots: list[tuple[str, list[dict]]],
) -> dict[str, dict]:
    """Frequency of missing/blank values in key columns for top-60 names."""
    KEY_COLS = ["catalyst_days", "clinical_score", "de_drawdown",
                "catalyst_mode", "financial_score"]
    overall: dict[str, int] = defaultdict(int)
    top60: dict[str, int] = defaultdict(int)
    total_rows = 0
    total_top60 = 0

    for _, rows in snapshots:
        # Sort by composite_rank for top-60
        sorted_rows = sorted(rows, key=lambda r: _safe_int(r.get("composite_rank")) or 9999)
        top60_tickers = {r["ticker"] for r in sorted_rows[:60]}

        for r in rows:
            total_rows += 1
            in_top60 = r["ticker"] in top60_tickers
            if in_top60:
                total_top60 += 1
            for col in KEY_COLS:
                val = r.get(col, "")
                if val is None or val == "":
                    overall[col] += 1
                    if in_top60:
                        top60[col] += 1

    return {
        "total_rows": total_rows,
        "total_top60_rows": total_top60,
        "missing_overall": {col: overall[col] for col in KEY_COLS},
        "missing_pct_overall": {
            col: round(100 * overall[col] / total_rows, 2) if total_rows else 0
            for col in KEY_COLS
        },
        "missing_top60": {col: top60[col] for col in KEY_COLS},
        "missing_pct_top60": {
            col: round(100 * top60[col] / total_top60, 2) if total_top60 else 0
            for col in KEY_COLS
        },
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _format_markdown(diag: dict) -> str:
    lines = ["# Decision Engine Diagnostics Report\n"]

    # Tier distribution
    lines.append("## Tier Distribution (A/B/C/D per date)\n")
    lines.append(f"{'date':<14} {'total':>6} {'A':>5} {'B':>5} {'C':>5} {'D':>5} {'none':>5}")
    lines.append("-" * 52)
    for row in diag["tier_distribution"]:
        lines.append(
            f"{row['date']:<14} {row['total']:>6} {row['A']:>5} {row['B']:>5} "
            f"{row['C']:>5} {row['D']:>5} {row['no_tier']:>5}"
        )

    # Tier churn
    lines.append("\n## Tier Churn (consecutive dates)\n")
    lines.append(f"{'from':<14} {'to':<14} {'changes':>8} {'churn%':>7} {'A_up':>5} {'A_dn':>5}")
    lines.append("-" * 60)
    for row in diag["tier_churn"]:
        lines.append(
            f"{row['from_date']:<14} {row['to_date']:<14} {row['tier_changes']:>8} "
            f"{row['churn_pct']:>7.1f} {row['A_upgrades']:>5} {row['A_downgrades']:>5}"
        )

    # Top-K stability
    for k_label in ("top20_stability", "top60_stability"):
        k = 20 if "20" in k_label else 60
        lines.append(f"\n## Top-{k} Stability (Jaccard + Spearman)\n")
        lines.append(f"{'from':<14} {'to':<14} {'jaccard':>8} {'spearman':>9}")
        lines.append("-" * 50)
        for row in diag[k_label]:
            sp = f"{row['spearman']:.4f}" if row["spearman"] is not None else "N/A"
            lines.append(
                f"{row['from_date']:<14} {row['to_date']:<14} {row['jaccard']:>8.4f} {sp:>9}"
            )

    # Score contributions
    lines.append("\n## Score Contributions by Tier\n")
    for tier, scores in diag["score_contributions"].items():
        lines.append(f"\n### Tier {tier}\n")
        lines.append(f"{'score':<22} {'mean':>8} {'median':>8} {'p25':>8} {'p75':>8} {'n':>6}")
        lines.append("-" * 58)
        for col, st in scores.items():
            if st["mean"] is not None:
                lines.append(
                    f"{col:<22} {st['mean']:>8.2f} {st['median']:>8.2f} "
                    f"{st['p25']:>8.2f} {st['p75']:>8.2f} {st['n']:>6}"
                )

    # Tier reason distribution
    lines.append("\n## Tier Reason Distribution (all dates)\n")
    for reason, count in diag["tier_reason_distribution"].items():
        lines.append(f"  {reason}: {count}")

    # Missing data
    lines.append("\n## Missing Data Sensitivity\n")
    md = diag["missing_data"]
    lines.append(f"Total rows: {md['total_rows']}, Top-60 rows: {md['total_top60_rows']}\n")
    lines.append(f"{'column':<22} {'miss_all':>9} {'pct_all':>8} {'miss_t60':>9} {'pct_t60':>8}")
    lines.append("-" * 60)
    for col in md["missing_overall"]:
        lines.append(
            f"{col:<22} {md['missing_overall'][col]:>9} {md['missing_pct_overall'][col]:>7.1f}% "
            f"{md['missing_top60'][col]:>9} {md['missing_pct_top60'][col]:>7.1f}%"
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decision Engine Diagnostics")
    parser.add_argument("--from-date", default="2026-01-15")
    parser.add_argument("--to-date", default="2026-02-16")
    parser.add_argument("--snapshot-dir", default="data/snapshots")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args(argv)

    from_dt = date.fromisoformat(args.from_date)
    to_dt = date.fromisoformat(args.to_date)
    snap_base = _PROJECT_ROOT / args.snapshot_dir
    out_dir = _PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all snapshots
    snapshot_paths = _discover_snapshots(snap_base, from_dt, to_dt)
    print(f"Found {len(snapshot_paths)} snapshots in [{args.from_date}, {args.to_date}]")

    snapshots: list[tuple[str, list[dict]]] = []
    for date_str, path in snapshot_paths:
        rows = _load_rankings(path)
        if rows:
            snapshots.append((date_str, rows))

    if not snapshots:
        print("No snapshots found. Exiting.")
        return 1

    # Compute diagnostics
    diag: dict[str, Any] = {}
    diag["date_range"] = {"from": args.from_date, "to": args.to_date, "n_snapshots": len(snapshots)}
    diag["tier_distribution"] = compute_tier_distribution(snapshots)
    diag["tier_churn"] = compute_tier_churn(snapshots)
    diag["top20_stability"] = compute_topk_stability(snapshots, k=20)
    diag["top60_stability"] = compute_topk_stability(snapshots, k=60)
    diag["score_contributions"] = compute_score_contributions(snapshots)
    diag["tier_reason_distribution"] = compute_tier_reason_distribution(snapshots)
    diag["missing_data"] = compute_missing_data(snapshots)

    # Write JSON
    json_path = out_dir / f"decision_engine_diagnostics_{args.from_date}_{args.to_date}.json"
    json_path.write_text(
        json.dumps(diag, indent=2, sort_keys=False, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote: {json_path}")

    # Write markdown
    md_path = out_dir / "decision_engine_diagnostics.md"
    md_path.write_text(_format_markdown(diag), encoding="utf-8")
    print(f"Wrote: {md_path}")

    # Print summary
    print(f"\n--- Summary ---")
    td = diag["tier_distribution"]
    if td:
        latest = td[-1]
        print(f"Latest ({latest['date']}): A={latest['A']}, B={latest['B']}, C={latest['C']}, D={latest['D']}")
    if diag["tier_churn"]:
        avg_churn = sum(r["churn_pct"] for r in diag["tier_churn"]) / len(diag["tier_churn"])
        print(f"Avg tier churn: {avg_churn:.1f}%")
    if diag["top20_stability"]:
        avg_j20 = sum(r["jaccard"] for r in diag["top20_stability"]) / len(diag["top20_stability"])
        print(f"Avg top-20 Jaccard: {avg_j20:.3f}")
    if diag["top60_stability"]:
        avg_j60 = sum(r["jaccard"] for r in diag["top60_stability"]) / len(diag["top60_stability"])
        print(f"Avg top-60 Jaccard: {avg_j60:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
