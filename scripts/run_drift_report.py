#!/usr/bin/env python3
"""Daily Drift Report — post-promotion monitoring for ruleset eb833c56.

Loads the last N snapshots from the snapshot directory, computes per-snapshot
drift metrics (tier distribution, catalyst coverage, top-25 overlap, score
dispersion), evaluates hard guardrails, and produces JSON + Markdown reports.

Usage:
    python scripts/run_drift_report.py \
        --snapshot-dir data/snapshots \
        --output-dir output \
        [--window-size 5] \
        [--guardrails path/to/guardrails.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from dataclasses import fields as dc_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Imports from existing code (snapshot loading + helpers)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_phase2_snapshot_delta import (
    SnapshotData,
    _catalyst_coverage,
    _safe_float,
    _safe_int,
    _tier_counts,
    load_snapshot,
)

# ---------------------------------------------------------------------------
# Guardrails dataclass
# ---------------------------------------------------------------------------
MANIFEST_PATH = (
    PROJECT_ROOT / "production_data" / "decision_rulesets" / "manifest.json"
)


@dataclass(frozen=True)
class DriftGuardrails:
    """Immutable, versioned guardrail thresholds for drift monitoring.

    FAIL triggers indicate an immediate rollback candidate — the tier
    structure or data feed has diverged far enough from expectations that
    the active ruleset may be mis-specified.
    """

    # FAIL triggers
    fail_a_pct_low: float = 2.0        # A-tier % among dev < 2%
    fail_a_pct_high: float = 15.0      # A-tier % among dev > 15%
    fail_catalyst_missing_high: float = 85.0   # catalyst missing % among eligible > 85%
    fail_overlap_low: float = 50.0     # top-25 overlap vs prior < 50%
    fail_dispersion_low: float = 0.10  # optionality std < 0.10

    # Rolling window size
    window_size: int = 5

    @property
    def guardrails_id(self) -> str:
        """Deterministic 8-char SHA256 of all threshold fields."""
        parts = []
        for f in dc_fields(self):
            parts.append(f"{f.name}={getattr(self, f.name)}")
        blob = "|".join(parts).encode()
        return hashlib.sha256(blob).hexdigest()[:8]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["guardrails_id"] = self.guardrails_id
        return d

    @classmethod
    def from_json(cls, path: str) -> "DriftGuardrails":
        with open(path) as f:
            data = json.load(f)
        data.pop("guardrails_id", None)
        valid_fields = {f.name for f in dc_fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")


# ---------------------------------------------------------------------------
# Snapshot window loading
# ---------------------------------------------------------------------------
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def load_snapshot_window(
    snap_dir: Path, window_size: int
) -> List[SnapshotData]:
    """Load the last *window_size* snapshots sorted by date (ascending).

    Only considers directories whose name matches YYYY-MM-DD and that
    contain a loadable snapshot (rankings.csv with tier_dev).
    """
    if not snap_dir.is_dir():
        return []

    date_dirs = sorted(
        [d for d in snap_dir.iterdir() if d.is_dir() and _DATE_RE.match(d.name)],
        key=lambda d: d.name,
    )

    # Take the last N directories
    candidates = date_dirs[-window_size:] if window_size > 0 else date_dirs

    snapshots: List[SnapshotData] = []
    for d in candidates:
        snap = load_snapshot(d)
        if snap is not None:
            snapshots.append(snap)

    return snapshots


# ---------------------------------------------------------------------------
# Drift metrics computation
# ---------------------------------------------------------------------------
def _top_n_tickers(rankings: pd.DataFrame, n: int = 25) -> set:
    """Return the top-N tickers by actionable rank (dev-stage only)."""
    dev = rankings[rankings["archetype"] == "drug_developer"].copy()
    if "actionable_rank" not in dev.columns:
        return set()
    dev["_arank"] = dev["actionable_rank"].apply(lambda v: _safe_int(v, 9999))
    dev = dev.sort_values("_arank").head(n)
    return set(dev["ticker"])


def _optionality_std(rankings: pd.DataFrame) -> Optional[float]:
    """Standard deviation of clinical_optionality_pct_dev among dev tickers."""
    dev = rankings[rankings["archetype"] == "drug_developer"]
    if "clinical_optionality_pct_dev" not in dev.columns:
        return None
    vals = [
        _safe_float(v, None)
        for v in dev["clinical_optionality_pct_dev"]
        if _safe_float(v, None) is not None
    ]
    if len(vals) < 2:
        return None
    return round(statistics.stdev(vals), 4)


def _composite_iqr(rankings: pd.DataFrame) -> Optional[float]:
    """IQR of composite_score among dev tickers."""
    dev = rankings[rankings["archetype"] == "drug_developer"]
    if "composite_score" not in dev.columns:
        return None
    vals = sorted(
        _safe_float(v, None)
        for v in dev["composite_score"]
        if _safe_float(v, None) is not None
    )
    if len(vals) < 4:
        return None
    q1 = vals[len(vals) // 4]
    q3 = vals[3 * len(vals) // 4]
    return round(q3 - q1, 4)


def _catalyst_missing_pct_eligible(rankings: pd.DataFrame) -> Optional[float]:
    """Catalyst missing % among eligible dev tickers."""
    dev = rankings[rankings["archetype"] == "drug_developer"].copy()
    if "eligible" not in dev.columns:
        return None
    eligible = dev[dev["eligible"].astype(str).str.strip() == "1"]
    n_elig = len(eligible)
    if n_elig == 0:
        return None
    if "catalyst_mode" not in eligible.columns:
        return None
    n_missing = sum(
        1
        for m in eligible["catalyst_mode"]
        if str(m).strip() in ("missing", "")
    )
    return round(n_missing / n_elig * 100, 1)


def _drawdown_coverage_pct(rankings: pd.DataFrame) -> Optional[float]:
    """Drawdown coverage % among dev tickers (has a non-missing reason)."""
    dev = rankings[rankings["archetype"] == "drug_developer"]
    n_dev = len(dev)
    if n_dev == 0:
        return None
    if "de_drawdown_missing_reason" not in dev.columns:
        # Older snapshots without this column — assume coverage unknown
        return None
    n_covered = sum(
        1
        for r in dev["de_drawdown_missing_reason"]
        if str(r).strip() == ""
    )
    return round(n_covered / n_dev * 100, 1)


def compute_snapshot_metrics(snap: SnapshotData) -> Dict[str, Any]:
    """Compute drift metrics for a single snapshot."""
    rankings = snap.rankings
    tc = _tier_counts(rankings)
    n_dev = sum(tc.values())

    metrics: Dict[str, Any] = {
        "date": snap.date,
        "ruleset_id": snap.ruleset_id,
        "n_dev": n_dev,
    }

    # Tier counts and percentages
    for tier in ("A", "B", "C", "D"):
        metrics[f"tier_{tier}_count"] = tc[tier]
        metrics[f"tier_{tier}_pct"] = (
            round(tc[tier] / n_dev * 100, 1) if n_dev > 0 else 0.0
        )

    # Eligible percentage
    dev = rankings[rankings["archetype"] == "drug_developer"]
    if "eligible" in dev.columns:
        n_elig = sum(
            1 for e in dev["eligible"] if str(e).strip() == "1"
        )
        metrics["eligible_pct"] = round(n_elig / n_dev * 100, 1) if n_dev > 0 else 0.0
    else:
        metrics["eligible_pct"] = None

    # Catalyst missing % among eligible
    metrics["catalyst_missing_pct"] = _catalyst_missing_pct_eligible(rankings)

    # Drawdown coverage
    metrics["drawdown_coverage_pct"] = _drawdown_coverage_pct(rankings)

    # Score dispersion
    metrics["optionality_std"] = _optionality_std(rankings)
    metrics["composite_iqr"] = _composite_iqr(rankings)

    # Top-25 tickers (for overlap computation)
    metrics["_top25"] = _top_n_tickers(rankings, 25)

    return metrics


def compute_drift_metrics(
    snapshots: List[SnapshotData],
) -> Dict[str, Any]:
    """Compute per-snapshot metrics plus rolling aggregates.

    Returns:
        {
            "snapshots": [per-snapshot metrics (oldest first)],
            "rolling": {metric_name: {"min": ..., "max": ..., "mean": ..., "current": ...}},
            "current": latest snapshot metrics,
        }
    """
    if not snapshots:
        return {"snapshots": [], "rolling": {}, "current": {}}

    all_metrics: List[Dict[str, Any]] = []
    for snap in snapshots:
        m = compute_snapshot_metrics(snap)
        all_metrics.append(m)

    # Compute top-25 overlap (vs prior snapshot)
    for i, m in enumerate(all_metrics):
        if i == 0:
            m["top25_overlap_pct"] = None
        else:
            prev_top = all_metrics[i - 1].get("_top25", set())
            cur_top = m.get("_top25", set())
            if prev_top and cur_top:
                union = prev_top | cur_top
                intersection = prev_top & cur_top
                # Jaccard-style: intersection over 25 (fixed denominator)
                m["top25_overlap_pct"] = round(
                    len(intersection) / 25 * 100, 1
                )
            else:
                m["top25_overlap_pct"] = None

    # Remove internal _top25 sets (not serializable)
    for m in all_metrics:
        m.pop("_top25", None)

    current = all_metrics[-1]

    # Rolling aggregates for key numeric metrics
    roll_keys = [
        "tier_A_pct", "tier_B_pct", "tier_C_pct", "tier_D_pct",
        "eligible_pct", "catalyst_missing_pct", "top25_overlap_pct",
        "optionality_std", "composite_iqr", "drawdown_coverage_pct",
    ]
    rolling: Dict[str, Dict[str, Any]] = {}
    for key in roll_keys:
        vals = [m[key] for m in all_metrics if m.get(key) is not None]
        if vals:
            rolling[key] = {
                "min": round(min(vals), 2),
                "max": round(max(vals), 2),
                "mean": round(statistics.mean(vals), 2),
                "current": current.get(key),
            }

    return {
        "snapshots": all_metrics,
        "rolling": rolling,
        "current": current,
    }


# ---------------------------------------------------------------------------
# Guardrail evaluation
# ---------------------------------------------------------------------------
def find_rollback_candidate(
    manifest_path: Path = MANIFEST_PATH,
) -> Optional[Dict[str, Any]]:
    """Find the most recently retired ruleset from the manifest.

    Returns the manifest entry dict, or None if no retired entries exist.
    """
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        manifest = json.load(f)

    retired = [
        r for r in manifest.get("rulesets", []) if r.get("status") == "retired"
    ]
    if not retired:
        return None

    # Sort by updated_at (descending) to find the most recently retired
    def _sort_key(r: Dict) -> str:
        return r.get("updated_at", r.get("created_at", ""))

    retired.sort(key=_sort_key, reverse=True)
    return retired[0]


def evaluate_guardrails(
    metrics: Dict[str, Any],
    guardrails: DriftGuardrails,
) -> Tuple[str, List[str], Optional[Dict[str, Any]]]:
    """Evaluate drift guardrails against current snapshot metrics.

    Returns:
        (status, reasons, rollback_candidate)
        status: "OK" | "WARN" | "FAIL"
        reasons: list of human-readable reason strings
        rollback_candidate: manifest entry for most recently retired ruleset (on FAIL)
    """
    current = metrics.get("current", {})
    reasons: List[str] = []

    a_pct = current.get("tier_A_pct")
    cat_missing = current.get("catalyst_missing_pct")
    overlap = current.get("top25_overlap_pct")
    opt_std = current.get("optionality_std")

    # FAIL checks
    if a_pct is not None and a_pct < guardrails.fail_a_pct_low:
        reasons.append(
            f"A-tier % = {a_pct:.1f}% < {guardrails.fail_a_pct_low}% floor"
        )
    if a_pct is not None and a_pct > guardrails.fail_a_pct_high:
        reasons.append(
            f"A-tier % = {a_pct:.1f}% > {guardrails.fail_a_pct_high}% ceiling"
        )
    if cat_missing is not None and cat_missing > guardrails.fail_catalyst_missing_high:
        reasons.append(
            f"Catalyst missing = {cat_missing:.1f}% > {guardrails.fail_catalyst_missing_high}% ceiling"
        )
    if overlap is not None and overlap < guardrails.fail_overlap_low:
        reasons.append(
            f"Top-25 overlap = {overlap:.1f}% < {guardrails.fail_overlap_low}% floor"
        )
    if opt_std is not None and opt_std < guardrails.fail_dispersion_low:
        reasons.append(
            f"Optionality std = {opt_std:.4f} < {guardrails.fail_dispersion_low} floor"
        )

    if reasons:
        rollback = find_rollback_candidate()
        return "FAIL", reasons, rollback

    return "OK", [], None


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_drift_report_md(
    metrics: Dict[str, Any],
    status: str,
    reasons: List[str],
    guardrails: DriftGuardrails,
    rollback_candidate: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a Markdown drift report."""
    current = metrics.get("current", {})
    rolling = metrics.get("rolling", {})
    n_snaps = len(metrics.get("snapshots", []))

    lines: List[str] = []
    lines.append("# Daily Drift Report")
    lines.append(
        f"Date: {current.get('date', 'N/A')} | "
        f"Ruleset: {current.get('ruleset_id', 'N/A')} | "
        f"Window: {n_snaps} snapshots"
    )
    lines.append("")

    # Current snapshot table
    lines.append("## Current Snapshot")
    lines.append("")
    lines.append("| Metric                    | Value  |")
    lines.append("|---------------------------|--------|")

    def _fmt(val: Any, decimals: int = 1) -> str:
        if val is None:
            return "N/A"
        if isinstance(val, float):
            if abs(val) < 0.01 and val != 0.0:
                return f"{val:.4f}"
            return f"{val:.{decimals}f}"
        return str(val)

    rows = [
        ("A-tier count (dev)", current.get("tier_A_count")),
        ("A-tier % (dev)", f"{current.get('tier_A_pct', 'N/A')}%"
         if current.get("tier_A_pct") is not None else "N/A"),
        ("B-tier count (dev)", current.get("tier_B_count")),
        ("Eligible % (dev)", f"{current.get('eligible_pct', 'N/A')}%"
         if current.get("eligible_pct") is not None else "N/A"),
        ("Catalyst missing (elig)", f"{current.get('catalyst_missing_pct', 'N/A')}%"
         if current.get("catalyst_missing_pct") is not None else "N/A"),
        ("Drawdown coverage (dev)", f"{current.get('drawdown_coverage_pct', 'N/A')}%"
         if current.get("drawdown_coverage_pct") is not None else "N/A"),
        ("Top-25 overlap (vs prior)", f"{current.get('top25_overlap_pct', 'N/A')}%"
         if current.get("top25_overlap_pct") is not None else "N/A"),
        ("Optionality std", _fmt(current.get("optionality_std"), 2)),
        ("Composite IQR", _fmt(current.get("composite_iqr"))),
    ]
    for label, val in rows:
        lines.append(f"| {label:<25} | {str(val):>6} |")
    lines.append("")

    # Rolling window table
    if rolling:
        lines.append(f"## Rolling Window (last {n_snaps} runs)")
        lines.append("")
        lines.append("| Metric              | Min   | Max   | Mean  | Current |")
        lines.append("|---------------------|-------|-------|-------|---------|")
        for key in [
            "tier_A_pct", "catalyst_missing_pct",
            "top25_overlap_pct", "optionality_std",
        ]:
            if key not in rolling:
                continue
            r = rolling[key]
            label = key.replace("_", " ").replace("tier ", "").title()
            lines.append(
                f"| {label:<19} "
                f"| {r['min']:>5} "
                f"| {r['max']:>5} "
                f"| {r['mean']:>5} "
                f"| {_fmt(r['current']):>7} |"
            )
        lines.append("")

    # Mixed rulesets warning
    rulesets_in_window = set(
        s.get("ruleset_id", "") for s in metrics.get("snapshots", [])
    )
    rulesets_in_window.discard("")
    if len(rulesets_in_window) > 1:
        lines.append("## Warning: Mixed Rulesets in Window")
        lines.append(f"Rulesets observed: {', '.join(sorted(rulesets_in_window))}")
        lines.append("")

    # Guardrails
    lines.append(f"## Guardrails: {status}")
    if reasons:
        for r in reasons:
            lines.append(f"- {r}")
    if rollback_candidate:
        lines.append("")
        lines.append(f"**Rollback candidate**: {rollback_candidate['id']} "
                      f"({rollback_candidate.get('file', 'N/A')})")
        lines.append(f"  To rollback: `python scripts/promote_ruleset.py "
                      f"{rollback_candidate['id']}`")
    lines.append("")

    # Guardrails config
    lines.append("## Guardrails Config")
    lines.append(f"ID: {guardrails.guardrails_id}")
    lines.append(f"- fail_a_pct_low: {guardrails.fail_a_pct_low}%")
    lines.append(f"- fail_a_pct_high: {guardrails.fail_a_pct_high}%")
    lines.append(f"- fail_catalyst_missing_high: {guardrails.fail_catalyst_missing_high}%")
    lines.append(f"- fail_overlap_low: {guardrails.fail_overlap_low}%")
    lines.append(f"- fail_dispersion_low: {guardrails.fail_dispersion_low}")
    lines.append("")

    return "\n".join(lines)


def generate_drift_json(
    metrics: Dict[str, Any],
    status: str,
    reasons: List[str],
    guardrails: DriftGuardrails,
    rollback_candidate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a JSON-serializable drift report dict."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reasons": reasons,
        "rollback_candidate": rollback_candidate,
        "guardrails": guardrails.to_dict(),
        "current": metrics.get("current", {}),
        "rolling": metrics.get("rolling", {}),
        "window_size": len(metrics.get("snapshots", [])),
        "snapshots": metrics.get("snapshots", []),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate daily drift report for post-promotion monitoring."
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        required=True,
        help="Directory containing dated snapshot subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write drift_report.json and drift_report.md.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Number of most recent snapshots to include (default: 5).",
    )
    parser.add_argument(
        "--guardrails",
        type=str,
        default=None,
        help="Path to guardrails JSON override (optional).",
    )
    args = parser.parse_args()

    # Load guardrails
    if args.guardrails:
        guardrails = DriftGuardrails.from_json(args.guardrails)
    else:
        guardrails = DriftGuardrails(window_size=args.window_size)

    window_size = guardrails.window_size

    # Load snapshots
    snapshots = load_snapshot_window(args.snapshot_dir, window_size)
    if not snapshots:
        print("No loadable snapshots found.", file=sys.stderr)
        return 1

    print(f"Loaded {len(snapshots)} snapshots (window={window_size})")

    # Compute metrics
    metrics = compute_drift_metrics(snapshots)

    # Evaluate guardrails
    status, reasons, rollback = evaluate_guardrails(metrics, guardrails)

    # Generate reports
    md = generate_drift_report_md(metrics, status, reasons, guardrails, rollback)
    report_json = generate_drift_json(metrics, status, reasons, guardrails, rollback)

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.output_dir / "drift_report.md"
    json_path = args.output_dir / "drift_report.json"

    with open(md_path, "w") as f:
        f.write(md)

    with open(json_path, "w") as f:
        json.dump(report_json, f, indent=2, default=str)
        f.write("\n")

    print(f"Drift status: {status}")
    if reasons:
        for r in reasons:
            print(f"  - {r}")
    print(f"Reports written to {args.output_dir}/")

    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
