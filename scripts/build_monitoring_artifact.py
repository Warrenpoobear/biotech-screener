#!/usr/bin/env python3
"""Build a daily monitoring artifact for a Phase-2 snapshot.

Produces coverage, stability, exposure distributions, and attribution
metrics in both JSON and Markdown formats.

Usage:
    python3 scripts/build_monitoring_artifact.py \\
        --as-of-date 2026-02-26 \\
        --snapshot-dir data/snapshots \\
        --out-json data/snapshots/2026-02-26/monitoring.json \\
        --out-md   data/snapshots/2026-02-26/monitoring.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

VERSION = "1.0.0"
SCHEMA = "monitoring.v1"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, default=str) + "\n"


def _read_rankings(path: Path) -> List[Dict[str, str]]:
    """Read rankings.csv into list of dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def _distribution(rows: List[Dict], col: str) -> Dict[str, int]:
    """Count values for a column, skipping empty."""
    counts: Counter = Counter()
    for r in rows:
        v = r.get(col, "")
        if v:
            counts[v] += 1
    return dict(sorted(counts.items()))


def _find_prior_snapshot(snapshot_dir: Path, as_of_date: str) -> Optional[Path]:
    """Find the most recent snapshot before as_of_date."""
    candidates = []
    for d in snapshot_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if len(name) != 10 or "__" in name:
            continue
        if name < as_of_date and (d / "rankings.csv").exists():
            candidates.append(name)
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return snapshot_dir / candidates[0]


def _compute_overlap(current_tickers: List[str], prior_tickers: List[str], k: int) -> Dict[str, Any]:
    """Compute top-K overlap metrics."""
    curr_set = set(current_tickers[:k])
    prior_set = set(prior_tickers[:k])
    if not curr_set or not prior_set:
        return {"top_k": k, "overlap_count": 0, "overlap_pct": 0.0}
    intersection = curr_set & prior_set
    union = curr_set | prior_set
    return {
        "top_k": k,
        "overlap_count": len(intersection),
        "overlap_pct": round(100 * len(intersection) / k, 1),
        "jaccard": round(len(intersection) / len(union), 4) if union else 0.0,
        "entrants": sorted(curr_set - prior_set),
        "exits": sorted(prior_set - curr_set),
    }


def _rank_correlation(current_ranks: Dict[str, int], prior_ranks: Dict[str, int]) -> Dict[str, Optional[float]]:
    """Compute Spearman rank correlation over common tickers."""
    common = sorted(set(current_ranks) & set(prior_ranks))
    if len(common) < 5:
        return {"spearman": None, "n_common": len(common)}

    curr_vals = [current_ranks[t] for t in common]
    prior_vals = [prior_ranks[t] for t in common]

    # Spearman: 1 - 6*sum(d^2) / (n*(n^2-1))
    n = len(common)
    d_sq = sum((c - p) ** 2 for c, p in zip(curr_vals, prior_vals))
    rho = 1 - (6 * d_sq) / (n * (n * n - 1)) if n > 1 else None

    return {"spearman": round(rho, 4) if rho is not None else None, "n_common": n}


def _eligible_rows(rows: List[Dict]) -> List[Dict]:
    """Filter to eligible rows."""
    result = []
    for r in rows:
        elig = r.get("eligible", "")
        if str(elig).strip().lower() in ("1", "true", "yes"):
            result.append(r)
    return result


def _ranked_tickers(rows: List[Dict], col: str = "actionable_rank") -> List[str]:
    """Extract tickers sorted by rank column."""
    ranked = []
    for r in rows:
        rank = _safe_int(r.get(col))
        if rank is not None:
            ranked.append((rank, r.get("ticker", "")))
    ranked.sort()
    return [t for _, t in ranked]


def _rank_map(rows: List[Dict], col: str = "actionable_rank") -> Dict[str, int]:
    """Build ticker -> rank mapping."""
    result = {}
    for r in rows:
        rank = _safe_int(r.get(col))
        ticker = r.get("ticker", "")
        if rank is not None and ticker:
            result[ticker] = rank
    return result


def build_monitoring(
    as_of_date: str,
    snapshot_dir: Path,
    top_k: int = 60,
) -> Dict[str, Any]:
    """Build the monitoring artifact dict."""
    snap_dir = snapshot_dir / as_of_date
    if not snap_dir.is_dir():
        raise FileNotFoundError(f"Snapshot not found: {snap_dir}")

    rankings_path = snap_dir / "rankings.csv"
    metadata_path = snap_dir / "metadata.json"
    if not rankings_path.exists():
        raise FileNotFoundError(f"rankings.csv not found in {snap_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {snap_dir}")

    metadata = _read_json(metadata_path)
    rows = _read_rankings(rankings_path)
    available_cols = set(rows[0].keys()) if rows else set()

    # --- Degraded detection ---
    cache_health_path = snap_dir / "cache_health.json"
    degraded_run = metadata.get("cache_degraded_run", False)
    cache_health_status = metadata.get("cache_health_overall_status", "unknown")
    if cache_health_path.exists():
        try:
            ch = _read_json(cache_health_path)
            degraded_run = degraded_run or ch.get("degraded_run", False)
        except Exception:
            pass

    # --- Context ---
    context = {
        "as_of_date": as_of_date,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ruleset_id": metadata.get("clinical_sort_telemetry", {}).get("ruleset_id", ""),
        "version": metadata.get("version", ""),
        "decision_mode": metadata.get("decision_mode", ""),
        "cache_health_overall_status": cache_health_status,
        "cache_degraded_run": degraded_run,
        "cache_refresh_had_rejections": metadata.get("cache_refresh_had_rejections", False),
        "cache_refresh_rejected_sources": metadata.get("cache_refresh_rejected_sources", []),
    }

    # Alpha coverage
    amt = metadata.get("alpha_modifier_telemetry", {})
    if amt:
        context["alpha_table_source"] = amt.get("alpha_table_source", "")
        context["alpha_score_present"] = amt.get("alpha_score_present", 0)
        context["alpha_score_missing"] = amt.get("alpha_score_missing", 0)

    # --- Coverage ---
    eligible = _eligible_rows(rows)
    n_universe = len(rows)
    n_eligible = len(eligible)

    coverage: Dict[str, Any] = {
        "n_universe": n_universe,
        "n_eligible": n_eligible,
    }

    # Catalyst coverage
    if "has_catalyst_signal" in available_cols:
        n_cat = sum(1 for r in eligible if str(r.get("has_catalyst_signal", "")).strip().lower() in ("1", "true"))
        coverage["catalyst_signal_pct"] = round(100 * n_cat / n_eligible, 1) if n_eligible else 0.0
    if "has_coinvest_signal" in available_cols:
        n_co = sum(1 for r in eligible if str(r.get("has_coinvest_signal", "")).strip().lower() in ("1", "true"))
        coverage["coinvest_signal_pct"] = round(100 * n_co / n_eligible, 1) if n_eligible else 0.0

    # From cache refresh if available
    refresh = metadata.get("cache_refresh_summary", {})
    if refresh:
        for source in ("sec_8k", "ctgov"):
            if source in refresh:
                coverage[f"{source}_count"] = refresh[source].get("count", 0)

    # --- Exposures (full eligible + top-K subset) ---
    current_tickers = _ranked_tickers(eligible)
    topk_rows = [r for r in eligible if r.get("ticker", "") in set(current_tickers[:top_k])]

    exposure_cols = {
        "tier_dev": "tier_distribution",
        "market_cap_bucket": "market_cap_distribution",
        "stage_bucket": "stage_distribution",
        "archetype": "archetype_distribution",
        "mom_state": "momentum_state_distribution",
        "size_band": "size_band_distribution",
    }

    exposures: Dict[str, Any] = {"missing_columns": []}
    for col, key in exposure_cols.items():
        if col in available_cols:
            exposures[key] = {
                "eligible": _distribution(eligible, col),
                f"top_{top_k}": _distribution(topk_rows, col),
            }
        else:
            exposures["missing_columns"].append(col)

    # --- Stability (skip if degraded or no prior) ---
    stability: Dict[str, Any] = {}
    prior_dir = None if degraded_run else _find_prior_snapshot(snapshot_dir, as_of_date)

    if degraded_run:
        stability["suppressed"] = True
        stability["reason"] = "degraded_run"
    elif prior_dir is None:
        stability["suppressed"] = True
        stability["reason"] = "no_prior_snapshot"
    else:
        prior_rows = _read_rankings(prior_dir / "rankings.csv")
        prior_eligible = _eligible_rows(prior_rows)
        prior_tickers = _ranked_tickers(prior_eligible)

        stability["prior_date"] = prior_dir.name
        stability["suppressed"] = False

        # Overlaps
        stability["top_20"] = _compute_overlap(current_tickers, prior_tickers, 20)
        stability[f"top_{top_k}"] = _compute_overlap(current_tickers, prior_tickers, top_k)

        # Name turnover (top-K)
        curr_set = set(current_tickers[:top_k])
        prior_set = set(prior_tickers[:top_k])
        n_entrants = len(curr_set - prior_set)
        stability["name_turnover_pct"] = round(100 * n_entrants / top_k, 1) if top_k else 0.0

        # Rank correlation (eligible)
        curr_ranks = _rank_map(eligible)
        prior_ranks = _rank_map(prior_eligible)
        stability["rank_correlation"] = _rank_correlation(curr_ranks, prior_ranks)

        # Max rank shift
        common = set(curr_ranks) & set(prior_ranks)
        if common:
            shifts = {t: abs(curr_ranks[t] - prior_ranks[t]) for t in common}
            max_ticker = max(shifts, key=shifts.get)
            stability["max_rank_shift"] = {
                "ticker": max_ticker,
                "shift": shifts[max_ticker],
                "from": prior_ranks[max_ticker],
                "to": curr_ranks[max_ticker],
            }
            n_changed = sum(1 for v in shifts.values() if v > 0)
            stability["pct_rows_rank_changed"] = round(100 * n_changed / len(common), 1)

    # --- Assemble output ---
    result = {
        "schema": SCHEMA,
        "version": VERSION,
        "context": context,
        "coverage": coverage,
        "exposures": exposures,
        "stability": stability,
    }

    return result


def _render_markdown(mon: Dict[str, Any], top_k: int = 60) -> str:
    """Render monitoring dict to Markdown."""
    lines: List[str] = []
    ctx = mon.get("context", {})

    lines.append(f"# Monitoring: {ctx.get('as_of_date', '?')}")
    lines.append("")

    if ctx.get("cache_degraded_run"):
        lines.append("> **DEGRADED RUN** — stability metrics suppressed.")
        lines.append("")

    # Context table
    lines.append("## Context")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    for k in ("as_of_date", "ruleset_id", "version", "decision_mode",
              "cache_health_overall_status", "cache_degraded_run"):
        lines.append(f"| {k} | `{ctx.get(k, '')}` |")
    if ctx.get("cache_refresh_had_rejections"):
        lines.append(f"| rejected_sources | `{', '.join(ctx.get('cache_refresh_rejected_sources', []))}` |")
    if ctx.get("alpha_table_source"):
        lines.append(f"| alpha_table_source | `{ctx['alpha_table_source']}` |")
        lines.append(f"| alpha_coverage | {ctx.get('alpha_score_present', '?')}/{ctx.get('alpha_score_present', 0) + ctx.get('alpha_score_missing', 0)} |")
    lines.append("")

    # Coverage
    cov = mon.get("coverage", {})
    lines.append("## Coverage")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Universe | {cov.get('n_universe', '?')} |")
    lines.append(f"| Eligible | {cov.get('n_eligible', '?')} |")
    if "catalyst_signal_pct" in cov:
        lines.append(f"| Catalyst signal | {cov['catalyst_signal_pct']}% |")
    if "coinvest_signal_pct" in cov:
        lines.append(f"| Coinvest signal | {cov['coinvest_signal_pct']}% |")
    if "sec_8k_count" in cov:
        lines.append(f"| SEC 8-K events | {cov['sec_8k_count']} |")
    if "ctgov_count" in cov:
        lines.append(f"| CTGov records | {cov['ctgov_count']} |")
    lines.append("")

    # Stability
    stab = mon.get("stability", {})
    lines.append("## Stability")
    lines.append("")
    if stab.get("suppressed"):
        lines.append(f"*Suppressed: {stab.get('reason', 'unknown')}*")
    else:
        lines.append(f"Prior: `{stab.get('prior_date', '?')}`")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")

        top20 = stab.get("top_20", {})
        lines.append(f"| Top-20 overlap | {top20.get('overlap_pct', '?')}% ({top20.get('overlap_count', '?')}/20) |")

        topk = stab.get(f"top_{top_k}", {})
        lines.append(f"| Top-{top_k} overlap | {topk.get('overlap_pct', '?')}% ({topk.get('overlap_count', '?')}/{top_k}) |")

        lines.append(f"| Name turnover | {stab.get('name_turnover_pct', '?')}% |")

        rc = stab.get("rank_correlation", {})
        lines.append(f"| Spearman rho | {rc.get('spearman', 'n/a')} (n={rc.get('n_common', '?')}) |")

        ms = stab.get("max_rank_shift", {})
        if ms:
            lines.append(f"| Max rank shift | {ms.get('ticker', '?')}: {ms.get('from', '?')} -> {ms.get('to', '?')} (delta={ms.get('shift', '?')}) |")

        lines.append(f"| % rows rank changed | {stab.get('pct_rows_rank_changed', '?')}% |")

        # Entrants/exits
        for label, subset_key in [("Top-20", "top_20"), (f"Top-{top_k}", f"top_{top_k}")]:
            subset = stab.get(subset_key, {})
            entrants = subset.get("entrants", [])
            exits = subset.get("exits", [])
            if entrants or exits:
                lines.append("")
                lines.append(f"**{label} changes:**")
                if entrants:
                    lines.append(f"- Entrants: {', '.join(entrants)}")
                if exits:
                    lines.append(f"- Exits: {', '.join(exits)}")
    lines.append("")

    # Exposures
    exp = mon.get("exposures", {})
    lines.append("## Exposures")
    lines.append("")
    missing = exp.get("missing_columns", [])
    if missing:
        lines.append(f"*Missing columns (skipped): {', '.join(missing)}*")
        lines.append("")

    for key in sorted(exp.keys()):
        if key == "missing_columns":
            continue
        data = exp[key]
        lines.append(f"### {key.replace('_', ' ').title()}")
        lines.append("")
        lines.append(f"| Value | Eligible | Top-{top_k} |")
        lines.append("|-------|----------|---------|")
        elig_dist = data.get("eligible", {})
        topk_dist = data.get(f"top_{top_k}", {})
        all_vals = sorted(set(list(elig_dist.keys()) + list(topk_dist.keys())))
        for v in all_vals:
            lines.append(f"| {v} | {elig_dist.get(v, 0)} | {topk_dist.get(v, 0)} |")
        lines.append("")

    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build daily monitoring artifact.")
    parser.add_argument("--as-of-date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "snapshots",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=60)
    args = parser.parse_args(argv)

    try:
        mon = build_monitoring(
            as_of_date=args.as_of_date,
            snapshot_dir=args.snapshot_dir,
            top_k=args.top_k,
        )

        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(_stable_json(mon), encoding="utf-8")

        md = _render_markdown(mon, top_k=args.top_k)
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")

        # Summary to stdout
        stab = mon.get("stability", {})
        cov = mon.get("coverage", {})
        print(f"Monitoring: {args.as_of_date}")
        print(f"  Eligible: {cov.get('n_eligible', '?')} / {cov.get('n_universe', '?')}")
        if not stab.get("suppressed"):
            top60 = stab.get(f"top_{args.top_k}", {})
            print(f"  Top-{args.top_k} overlap: {top60.get('overlap_pct', '?')}%")
            print(f"  Spearman: {stab.get('rank_correlation', {}).get('spearman', 'n/a')}")
        else:
            print(f"  Stability: suppressed ({stab.get('reason', '?')})")

        print(f"  Written: {args.out_json}, {args.out_md}")
        return 0

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
