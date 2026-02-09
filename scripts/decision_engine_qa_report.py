#!/usr/bin/env python3
"""Decision Engine QA Report.

Reads the snapshot produced by run_screen.py and emits:
  - decision_engine_qa.json  — machine-readable, schema-stable
  - decision_engine_qa.md    — human-readable Markdown for IC review

CLI:
  python scripts/decision_engine_qa_report.py \
    --snapshot-dir data/snapshots/2026-02-09 [--output-dir output]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Reuse helpers from the snapshot-delta module
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_phase2_snapshot_delta import (
    CatalystCoverage,
    SnapshotData,
    _catalyst_coverage,
    _count_risk_flags,
    _parse_risk_flags,
    _safe_float,
    _safe_int,
    _size_band_counts,
    _tier_counts,
    load_snapshot,
)

# ---------------------------------------------------------------------------
# Section evaluators — each returns (status, metrics)
# ---------------------------------------------------------------------------

SectionResult = Tuple[str, Dict[str, Any]]


def _section_universe(snap: SnapshotData) -> SectionResult:
    """Universe size and archetype breakdown."""
    r = snap.rankings
    total = len(r)

    if total == 0:
        return "FAIL", {"ranked": 0, "dev_count": 0, "archetype_breakdown": {}}

    dev_count = int((r["archetype"] == "drug_developer").sum())
    archetype_counts: Dict[str, int] = {}
    for a in r["archetype"]:
        a = str(a).strip()
        archetype_counts[a] = archetype_counts.get(a, 0) + 1

    status = "OK"
    if total < 280 or total > 400:
        status = "WARN"

    return status, {
        "ranked": total,
        "dev_count": dev_count,
        "archetype_breakdown": archetype_counts,
    }


def _section_eligibility(snap: SnapshotData) -> SectionResult:
    """Eligibility counts and ineligible-reason breakdown."""
    r = snap.rankings
    total = len(r)
    eligible_mask = r["eligible"].astype(str).str.strip().isin(["1", "1.0", "True", "true"])
    eligible_count = int(eligible_mask.sum())
    eligible_pct = round(eligible_count / total * 100, 1) if total else 0.0

    dev = r[r["archetype"] == "drug_developer"]
    dev_elig = dev[dev["eligible"].astype(str).str.strip().isin(["1", "1.0", "True", "true"])]
    dev_eligible_pct = round(len(dev_elig) / len(dev) * 100, 1) if len(dev) else 0.0

    # Ineligible reason breakdown
    inelig = r[~eligible_mask]
    reason_counts: Dict[str, int] = {}
    for val in inelig["ineligible_reasons"]:
        if pd.isna(val) or str(val).strip() == "":
            continue
        for reason in str(val).split("|"):
            reason = reason.strip()
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

    status = "OK"
    if eligible_count == 0:
        status = "FAIL"
    elif dev_eligible_pct < 15 or dev_eligible_pct > 50:
        status = "WARN"

    return status, {
        "eligible_count": eligible_count,
        "eligible_pct": eligible_pct,
        "dev_eligible_pct": dev_eligible_pct,
        "ineligible_reason_breakdown": reason_counts,
    }


def _section_tier_distribution(snap: SnapshotData) -> SectionResult:
    """Tier A/B/C/D distribution among dev-stage names."""
    counts = _tier_counts(snap.rankings)
    dev_total = sum(counts.values())
    pcts = {}
    for tier, n in counts.items():
        pcts[tier] = round(n / dev_total * 100, 1) if dev_total else 0.0

    status = "OK"
    if counts["A"] == 0:
        status = "WARN"
    elif counts["B"] < 3:
        status = "WARN"

    return status, {
        "tier_counts": counts,
        "tier_pcts": pcts,
        "dev_total": dev_total,
    }


def _section_coverage(snap: SnapshotData) -> SectionResult:
    """Catalyst, sponsor, and momentum data coverage."""
    cat_cov: CatalystCoverage = _catalyst_coverage(snap.rankings)

    # Sponsor coverage: non-blank sponsor_tier1_count among dev
    dev = snap.rankings[snap.rankings["archetype"] == "drug_developer"]
    n_dev = len(dev)
    sponsor_present = 0
    for val in dev["sponsor_tier1_count"]:
        if not pd.isna(val) and str(val).strip() != "":
            sponsor_present += 1
    sponsor_pct = round(sponsor_present / n_dev * 100, 1) if n_dev else 0.0

    # Momentum coverage: non-blank mom_state among dev
    mom_present = 0
    for val in dev["mom_state"]:
        if not pd.isna(val) and str(val).strip() != "":
            mom_present += 1
    mom_pct = round(mom_present / n_dev * 100, 1) if n_dev else 0.0

    status = "OK"
    if cat_cov.pct == 0:
        status = "FAIL"
    elif cat_cov.pct < 20 or sponsor_pct < 70:
        status = "WARN"

    return status, {
        "catalyst_specific_pct": cat_cov.pct,
        "catalyst_mode_breakdown": cat_cov.mode_dist,
        "sponsor_coverage_pct": sponsor_pct,
        "momentum_coverage_pct": mom_pct,
    }


def _section_score_dispersion(snap: SnapshotData) -> SectionResult:
    """Score dispersion metrics to detect tier/band compression."""
    dev = snap.rankings[snap.rankings["archetype"] == "drug_developer"]

    # Optionality std
    opt_vals = dev["clinical_optionality_pct_dev"].apply(
        lambda v: _safe_float(v, default=float("nan"))
    ).dropna()
    opt_std = round(float(opt_vals.std()), 4) if len(opt_vals) > 1 else 0.0

    # Tier dispersion: number of distinct tiers present
    tier_vals = dev["tier_dev"].dropna().astype(str).str.strip()
    tier_distinct = tier_vals[tier_vals.isin(["A", "B", "C", "D"])].nunique()

    # Size band dispersion
    band_vals = dev["size_band"].dropna().astype(str).str.strip()
    band_distinct = band_vals[band_vals.isin(["L", "M", "S", "XS"])].nunique()

    # Composite score IQR
    comp = snap.rankings["composite_score"].apply(
        lambda v: _safe_float(v, default=float("nan"))
    ).dropna()
    if len(comp) > 3:
        q25 = float(comp.quantile(0.25))
        q75 = float(comp.quantile(0.75))
        comp_iqr = round(q75 - q25, 2)
    else:
        comp_iqr = 0.0

    status = "OK"
    if opt_std < 0.15:
        status = "WARN"

    return status, {
        "optionality_std": opt_std,
        "tier_distinct_count": tier_distinct,
        "band_distinct_count": band_distinct,
        "composite_iqr": comp_iqr,
    }


def _section_portfolio(snap: SnapshotData) -> SectionResult:
    """Portfolio composition and concentration metrics."""
    p = snap.portfolio
    n_positions = len(p)

    empty_tiers = {"A": 0, "B": 0, "C": 0, "D": 0}
    empty_bands = {"L": 0, "M": 0, "S": 0, "XS": 0}
    empty_flags = {f: 0 for f in ["deep_drawdown", "high_vol", "high_beta",
                                   "overbought_rsi", "low_confidence"]}

    if n_positions == 0:
        return "FAIL", {
            "n_positions": 0,
            "has_native_portfolio": snap.has_native_portfolio,
            "tier_split": empty_tiers,
            "band_split": empty_bands,
            "top5_weight_pct": 0.0,
            "risk_flag_counts": empty_flags,
        }

    # Tier split
    tier_split: Dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
    for t in p["tier_dev"]:
        t = str(t).strip()
        if t in tier_split:
            tier_split[t] += 1

    # Band split
    band_split = _size_band_counts(p)

    # Top-5 weight concentration
    weights = p["target_weight_pct"].apply(lambda v: _safe_float(v, 0.0))
    top5_weight = round(float(weights.nlargest(5).sum()), 1)

    # Risk flag counts
    risk_flags = _count_risk_flags(p)

    status = "OK"
    if n_positions < 10 or top5_weight > 50:
        status = "WARN"

    return status, {
        "n_positions": n_positions,
        "has_native_portfolio": snap.has_native_portfolio,
        "tier_split": tier_split,
        "band_split": band_split,
        "top5_weight_pct": top5_weight,
        "risk_flag_counts": risk_flags,
    }


def _section_health_gate(snap: SnapshotData) -> SectionResult:
    """Pass-through from phase2_health.json if present."""
    health_path = snap.path / "phase2_health.json"
    if not health_path.exists():
        return "OK", {"available": False, "note": "phase2_health.json not found"}

    with open(health_path, "r", encoding="utf-8") as f:
        health = json.load(f)

    h_status = health.get("status", "OK").upper()
    reasons = health.get("reasons", [])
    metrics = health.get("metrics", {})

    if h_status == "FAIL":
        status = "FAIL"
    elif h_status == "WARN":
        status = "WARN"
    else:
        status = "OK"

    return status, {
        "available": True,
        "health_status": h_status,
        "health_reasons": reasons,
        "health_metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Overall aggregation
# ---------------------------------------------------------------------------

SECTION_NAMES = [
    "universe",
    "eligibility",
    "tier_distribution",
    "coverage",
    "score_dispersion",
    "portfolio",
    "health_gate",
]

SECTION_FUNCS = {
    "universe": _section_universe,
    "eligibility": _section_eligibility,
    "tier_distribution": _section_tier_distribution,
    "coverage": _section_coverage,
    "score_dispersion": _section_score_dispersion,
    "portfolio": _section_portfolio,
    "health_gate": _section_health_gate,
}


def compute_overall_status(sections: Dict[str, Dict]) -> str:
    """FAIL if any section FAIL, WARN if any WARN, else OK."""
    for sec in sections.values():
        if sec.get("status") == "FAIL":
            return "FAIL"
    for sec in sections.values():
        if sec.get("status") == "WARN":
            return "WARN"
    return "OK"


def _get_git_sha() -> str:
    """Get short git SHA, with graceful fallback."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(snap: SnapshotData) -> Dict[str, Any]:
    """Run all section evaluators and assemble the QA report dict."""
    sections: Dict[str, Dict[str, Any]] = {}
    for name in SECTION_NAMES:
        status, metrics = SECTION_FUNCS[name](snap)
        sections[name] = {"status": status, "metrics": metrics}

    # Engine version from rankings
    ver_col = snap.rankings.get("decision_engine_version")
    engine_version = ""
    if ver_col is not None:
        vals = ver_col.dropna().unique()
        vals = [v for v in vals if str(v).strip()]
        if vals:
            engine_version = str(vals[0]).strip()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_date": snap.date,
        "overall_status": compute_overall_status(sections),
        "ruleset_id": snap.ruleset_id,
        "engine_version": engine_version,
        "git_sha": _get_git_sha(),
        "sections": sections,
    }
    return report


def _status_badge(status: str) -> str:
    """Return a text badge for Markdown."""
    if status == "OK":
        return "OK"
    elif status == "WARN":
        return "**WARN**"
    else:
        return "**FAIL**"


def format_markdown(report: Dict[str, Any]) -> str:
    """Format the QA report as Markdown."""
    lines: List[str] = []
    overall = report["overall_status"]
    lines.append(f"# Decision Engine QA Report")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Snapshot date | {report['snapshot_date']} |")
    lines.append(f"| Overall status | {_status_badge(overall)} |")
    lines.append(f"| Engine version | {report['engine_version']} |")
    lines.append(f"| Ruleset ID | {report['ruleset_id']} |")
    lines.append(f"| Git SHA | {report['git_sha']} |")
    lines.append(f"| Generated | {report['generated_at']} |")
    lines.append("")

    sections = report["sections"]

    # Universe
    sec = sections["universe"]
    m = sec["metrics"]
    lines.append(f"## Universe {_status_badge(sec['status'])}")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Ranked securities | {m['ranked']} |")
    lines.append(f"| Drug developers | {m['dev_count']} |")
    for arch, cnt in sorted(m["archetype_breakdown"].items()):
        lines.append(f"| {arch} | {cnt} |")
    lines.append("")

    # Eligibility
    sec = sections["eligibility"]
    m = sec["metrics"]
    lines.append(f"## Eligibility {_status_badge(sec['status'])}")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Eligible count | {m['eligible_count']} |")
    lines.append(f"| Eligible % | {m['eligible_pct']}% |")
    lines.append(f"| Dev eligible % | {m['dev_eligible_pct']}% |")
    if m["ineligible_reason_breakdown"]:
        lines.append(f"| | |")
        lines.append(f"| **Ineligible reasons** | |")
        for reason, cnt in sorted(m["ineligible_reason_breakdown"].items(),
                                   key=lambda x: -x[1]):
            lines.append(f"| {reason} | {cnt} |")
    lines.append("")

    # Tier Distribution
    sec = sections["tier_distribution"]
    m = sec["metrics"]
    lines.append(f"## Tier Distribution (Dev-Stage) {_status_badge(sec['status'])}")
    lines.append("")
    lines.append(f"| Tier | Count | % |")
    lines.append(f"|---|---|---|")
    for tier in ["A", "B", "C", "D"]:
        lines.append(f"| {tier} | {m['tier_counts'][tier]} | {m['tier_pcts'][tier]}% |")
    lines.append(f"| **Total** | **{m['dev_total']}** | |")
    lines.append("")

    # Coverage
    sec = sections["coverage"]
    m = sec["metrics"]
    lines.append(f"## Data Coverage {_status_badge(sec['status'])}")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Catalyst specific_days % | {m['catalyst_specific_pct']}% |")
    lines.append(f"| Sponsor coverage % | {m['sponsor_coverage_pct']}% |")
    lines.append(f"| Momentum coverage % | {m['momentum_coverage_pct']}% |")
    if m["catalyst_mode_breakdown"]:
        lines.append(f"| | |")
        lines.append(f"| **Catalyst modes** | |")
        for mode, cnt in sorted(m["catalyst_mode_breakdown"].items(),
                                 key=lambda x: -x[1]):
            lines.append(f"| {mode} | {cnt} |")
    lines.append("")

    # Score Dispersion
    sec = sections["score_dispersion"]
    m = sec["metrics"]
    lines.append(f"## Score Dispersion {_status_badge(sec['status'])}")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Optionality std | {m['optionality_std']} |")
    lines.append(f"| Tier distinct count | {m['tier_distinct_count']} |")
    lines.append(f"| Band distinct count | {m['band_distinct_count']} |")
    lines.append(f"| Composite IQR | {m['composite_iqr']} |")
    lines.append("")

    # Portfolio
    sec = sections["portfolio"]
    m = sec["metrics"]
    lines.append(f"## Portfolio {_status_badge(sec['status'])}")
    lines.append("")
    src = "native" if m["has_native_portfolio"] else "reconstructed"
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Positions | {m['n_positions']} ({src}) |")
    lines.append(f"| Top-5 weight % | {m['top5_weight_pct']}% |")
    for tier in ["A", "B", "C", "D"]:
        lines.append(f"| Tier {tier} | {m['tier_split'][tier]} |")
    for band in ["L", "M", "S", "XS"]:
        lines.append(f"| Band {band} | {m['band_split'][band]} |")
    if any(v > 0 for v in m["risk_flag_counts"].values()):
        lines.append(f"| | |")
        lines.append(f"| **Risk flags** | |")
        for flag, cnt in sorted(m["risk_flag_counts"].items()):
            if cnt > 0:
                lines.append(f"| {flag} | {cnt} |")
    lines.append("")

    # Health Gate
    sec = sections["health_gate"]
    m = sec["metrics"]
    lines.append(f"## Health Gate {_status_badge(sec['status'])}")
    lines.append("")
    if not m.get("available", False):
        lines.append(f"*{m.get('note', 'Not available')}*")
    else:
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| Status | {m['health_status']} |")
        if m.get("health_reasons"):
            lines.append(f"| Reasons | {', '.join(m['health_reasons'])} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Decision Engine QA report from a snapshot."
    )
    parser.add_argument(
        "--snapshot-dir", required=True,
        help="Path to the snapshot directory (e.g. data/snapshots/2026-02-09)",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory for output files (default: output)",
    )
    args = parser.parse_args()

    snap_path = Path(args.snapshot_dir)
    if not snap_path.is_dir():
        print(f"ERROR: snapshot directory not found: {snap_path}", file=sys.stderr)
        return 1

    snap = load_snapshot(snap_path)
    if snap is None:
        print(f"ERROR: could not load snapshot from {snap_path}", file=sys.stderr)
        return 1

    report = generate_report(snap)
    md_text = format_markdown(report)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "decision_engine_qa.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    md_path = out_dir / "decision_engine_qa.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    overall = report["overall_status"]
    print(f"QA report: {overall}")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
