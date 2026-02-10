#!/usr/bin/env python3
"""
Generate a release summary for a candidate ruleset promotion.

Reads calibration and walkforward artifacts to produce a one-pager
suitable for IC review or changelog finalization.

Outputs:
    artifacts/release_summary.md

Usage:
    python scripts/generate_release_summary.py [options]

    --calibration-json PATH  (default: artifacts/calibration_report.json)
    --walkforward-json PATH  (default: artifacts/walkforward_report.json)
    --output PATH            (default: artifacts/release_summary.md)
    --candidate-id ID        Candidate ruleset ID (auto-detected from calibration)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RULESETS_DIR = PROJECT_ROOT / "production_data" / "decision_rulesets"
MANIFEST_PATH = RULESETS_DIR / "manifest.json"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts"


# =============================================================================
# DATA LOADERS
# =============================================================================

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest() -> Dict[str, Any]:
    return load_json(MANIFEST_PATH)


def find_active_ruleset(manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for entry in manifest.get("rulesets", []):
        if entry.get("status") == "active":
            return entry
    return None


def find_candidate_ruleset(
    manifest: Dict[str, Any], candidate_id: str,
) -> Optional[Dict[str, Any]]:
    for entry in manifest.get("rulesets", []):
        if entry.get("id") == candidate_id and entry.get("status") == "candidate":
            return entry
    return None


# =============================================================================
# SUMMARY GENERATION
# =============================================================================

def generate_release_summary(
    calibration: Dict[str, Any],
    walkforward: Optional[Dict[str, Any]],
    active_entry: Optional[Dict[str, Any]],
    candidate_entry: Optional[Dict[str, Any]],
) -> str:
    """Generate the release summary markdown."""
    lines: List[str] = []

    # Extract key data
    config = calibration.get("config", {})
    baseline = calibration.get("baseline", {})
    best = calibration.get("best", {})
    candidates = calibration.get("candidates", [])

    active_id = active_entry.get("id", "unknown") if active_entry else "unknown"
    candidate_id = candidate_entry.get("id", best.get("a_floor", "unknown")) if candidate_entry else "n/a"

    # Header
    lines.append("# Ruleset Release Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Baseline ruleset | `{active_id}` ({active_entry.get('file', '?') if active_entry else '?'}) |")
    if candidate_entry:
        lines.append(f"| Candidate ruleset | `{candidate_id}` ({candidate_entry.get('file', '?')}) |")
    else:
        lines.append(f"| Candidate a_floor | `{best.get('a_floor', '?')}` |")
    lines.append(f"| Panel rows | {config.get('panel_rows', '?')} |")
    lines.append(f"| a_floor sweep | {config.get('a_floor_values', '?')} |")
    cat_near_values = config.get("catalyst_near_values")
    if cat_near_values:
        lines.append(f"| catalyst_near sweep | {cat_near_values} |")
    lines.append("")

    # Parameter changes
    lines.append("## Parameter Changes")
    lines.append("")
    baseline_a = config.get("baseline_a_floor", "?")
    candidate_a = best.get("a_floor", "?")
    baseline_cat = config.get("baseline_catalyst_near")
    candidate_cat = best.get("catalyst_near_days")
    a_changed = baseline_a != candidate_a
    cat_changed = baseline_cat is not None and candidate_cat is not None and baseline_cat != candidate_cat
    if not a_changed and not cat_changed:
        lines.append(f"**No change recommended.** Current parameters are already optimal.")
        lines.append("")
    else:
        lines.append("| Parameter | Baseline | Candidate | Delta |")
        lines.append("|-----------|----------|-----------|-------|")
        if a_changed and isinstance(candidate_a, (int, float)) and isinstance(baseline_a, (int, float)):
            lines.append(f"| `tier_a_optionality_floor` | {baseline_a} | {candidate_a} | {candidate_a - baseline_a:+.2f} |")
        elif a_changed:
            lines.append(f"| `tier_a_optionality_floor` | {baseline_a} | {candidate_a} | - |")
        if cat_changed:
            lines.append(f"| `catalyst_near_days` | {baseline_cat} | {candidate_cat} | {candidate_cat - baseline_cat:+d} |")
        lines.append("")

    # Objective improvement
    lines.append("## Objective Improvement")
    lines.append("")
    lines.append("| Metric | Baseline | Candidate | Delta |")
    lines.append("|--------|----------|-----------|-------|")

    metric_pairs = [
        ("AB median 60d return (%)", "ab_median_ret_60d"),
        ("CD median 60d return (%)", "cd_median_ret_60d"),
        ("AB-CD separation (%)", "separation_60d"),
        ("AB median max-DD (%)", "ab_median_dd_60d"),
        ("Mean A-count / date", "mean_a_count_per_date"),
        ("Top-25 overlap (%)", "mean_top25_overlap_pct"),
        ("Mean turnover (%)", "mean_turnover_pct"),
    ]

    # Find candidate data from candidates list (match on a_floor + catalyst_near_days for 2D)
    candidate_data = None
    best_cat = best.get("catalyst_near_days")
    for c in candidates:
        if c.get("a_floor") == best.get("a_floor"):
            if best_cat is None or c.get("catalyst_near_days") == best_cat:
                candidate_data = c
                break
    if candidate_data is None:
        candidate_data = best

    for label, key in metric_pairs:
        bv = baseline.get(key)
        cv = candidate_data.get(key)
        if bv is not None and cv is not None:
            delta = cv - bv
            lines.append(f"| {label} | {bv:.2f} | {cv:.2f} | {delta:+.2f} |")
        else:
            bv_s = f"{bv:.2f}" if bv is not None else "n/a"
            cv_s = f"{cv:.2f}" if cv is not None else "n/a"
            lines.append(f"| {label} | {bv_s} | {cv_s} | - |")
    lines.append("")

    # Tier distribution delta
    lines.append("## Tier Distribution")
    lines.append("")
    lines.append("| Tier | Baseline % | Candidate % | Delta |")
    lines.append("|------|-----------|------------|-------|")
    b_dist = baseline.get("tier_distribution", {})
    c_dist = candidate_data.get("tier_distribution", {})
    for tier in ["A", "B", "C", "D"]:
        bv = b_dist.get(tier, 0)
        cv = c_dist.get(tier, 0)
        lines.append(f"| {tier} | {bv:.1f} | {cv:.1f} | {cv - bv:+.1f} |")
    lines.append("")

    # Per-tier return detail
    lines.append("## Per-Tier 60d Return Detail (Candidate)")
    lines.append("")
    lines.append("| Tier | Count | Mean % | Median % | Hit Rate % | Mean Max-DD % |")
    lines.append("|------|-------|--------|----------|------------|---------------|")
    tier_metrics = candidate_data.get("tier_metrics", {})
    for tier in ["A", "B", "C", "D"]:
        tm = tier_metrics.get(tier, {})
        count = tm.get("count", 0)
        m = f"{tm['mean_ret_60d']:+.2f}" if tm.get("mean_ret_60d") is not None else "n/a"
        md = f"{tm['median_ret_60d']:+.2f}" if tm.get("median_ret_60d") is not None else "n/a"
        hr = f"{tm['hit_rate_60d']:.1f}" if tm.get("hit_rate_60d") is not None else "n/a"
        dd = f"{tm['mean_max_dd_60d']:.2f}" if tm.get("mean_max_dd_60d") is not None else "n/a"
        lines.append(f"| {tier} | {count} | {m} | {md} | {hr} | {dd} |")
    lines.append("")

    # Walkforward coverage (if available)
    if walkforward:
        wf_coverage = walkforward.get("coverage", {})
        wf_stability = walkforward.get("stability", {})
        lines.append("## Walkforward Panel Coverage")
        lines.append("")
        lines.append(f"- Total rows: {wf_coverage.get('total_rows', '?')}")
        lines.append(
            f"- With forward returns: {wf_coverage.get('with_fwd_returns', '?')} "
            f"({wf_coverage.get('with_fwd_returns_pct', '?')}%)"
        )
        lines.append(
            f"- With forward max-DD: {wf_coverage.get('with_fwd_max_dd', '?')} "
            f"({wf_coverage.get('with_fwd_max_dd_pct', '?')}%)"
        )
        overlap = wf_stability.get("mean_top_n_overlap")
        if overlap is not None:
            lines.append(f"- Top-25 overlap (Jaccard): {overlap:.1f}%")
        lines.append("")

    # Tradeoffs
    tradeoffs = best.get("tradeoffs", [])
    if tradeoffs:
        lines.append("## Tradeoffs")
        lines.append("")
        for t in tradeoffs:
            lines.append(f"- {t}")
        lines.append("")

    # Scoring
    lines.append("## Scoring")
    lines.append("")
    n_passing = sum(1 for c in candidates if c.get("passed"))
    lines.append(f"- Objective score: **{best.get('score', 'n/a')}**")
    lines.append(f"- Candidates evaluated: {len(candidates)}")
    lines.append(f"- Candidates passing constraints: {n_passing}")
    lines.append(f"- Constraints: A% >= {config.get('min_a_pct', '?')}%, "
                 f"turnover <= {config.get('max_turnover', '?')}%, "
                 f"separation > 0")
    passed = best.get("passed", False)
    lines.append(f"- Status: **{'PASS' if passed else 'FAIL'}**")
    if not passed:
        violations = []
        for c in candidates:
            if c.get("a_floor") == best.get("a_floor"):
                violations = c.get("violations", [])
                break
        if violations:
            for v in violations:
                lines.append(f"  - {v}")
    lines.append("")

    # Provenance
    prov = calibration.get("provenance")
    if prov:
        lines.append("## Provenance")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        price_sha = prov.get("price_history_sha256")
        panel_sha = prov.get("panel_sha256")
        lines.append(f"| Price data SHA256 | `{price_sha or 'n/a'}` |")
        lines.append(f"| Panel SHA256 | `{panel_sha or 'n/a'}` |")
        lines.append(f"| Ruleset ID | {prov.get('ruleset_id', 'n/a')} |")
        lines.append(f"| Code version | {prov.get('code_version', 'n/a')} |")
        reg_fp = prov.get("explanation_registry_fingerprint")
        if reg_fp:
            lines.append(f"| Explanation registry | `{reg_fp}` |")
        lines.append("")

    # QA checklist
    lines.append("## QA Checklist")
    lines.append("")
    lines.append("- [ ] Calibration report reviewed (`artifacts/calibration_report.md`)")
    lines.append("- [ ] Walkforward report reviewed (`artifacts/walkforward_report.md`)")
    lines.append("- [ ] `bump_ruleset.py --from-json artifacts/candidate_overrides.json` executed")
    lines.append("- [ ] Contract tests pass (`pytest tests/test_decision_engine_contract.py`)")
    lines.append("- [ ] Replay regression pass (`pytest tests/test_decision_engine_replay_regression.py`)")
    lines.append("- [ ] Golden records refreshed if needed (`scripts/refresh_goldens.py`)")
    lines.append("- [ ] Changelog finalized (remove [DRAFT] marker)")
    lines.append("- [ ] `promote_ruleset.py <candidate_id>` executed")
    lines.append("")

    return "\n".join(lines) + "\n"


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate release summary for ruleset promotion."
    )
    parser.add_argument(
        "--calibration-json", type=str,
        default=str(DEFAULT_ARTIFACTS / "calibration_report.json"),
        help="Path to calibration_report.json",
    )
    parser.add_argument(
        "--walkforward-json", type=str,
        default=str(DEFAULT_ARTIFACTS / "walkforward_report.json"),
        help="Path to walkforward_report.json",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(DEFAULT_ARTIFACTS / "release_summary.md"),
        help="Output path for release_summary.md",
    )
    parser.add_argument(
        "--candidate-id", type=str, default=None,
        help="Candidate ruleset ID (auto-detected from manifest if available)",
    )
    args = parser.parse_args()

    # Load calibration report (required)
    cal_path = Path(args.calibration_json)
    if not cal_path.exists():
        print(f"ERROR: Calibration report not found: {cal_path}", file=sys.stderr)
        print("Run calibrate_ruleset_from_panel.py first.", file=sys.stderr)
        return 1
    calibration = load_json(cal_path)
    print(f"Loaded calibration report: {cal_path}")

    # Load walkforward report (optional)
    wf_path = Path(args.walkforward_json)
    walkforward = None
    if wf_path.exists():
        walkforward = load_json(wf_path)
        print(f"Loaded walkforward report: {wf_path}")
    else:
        print(f"Walkforward report not found ({wf_path}), proceeding without it.")

    # Load manifest
    active_entry = None
    candidate_entry = None
    if MANIFEST_PATH.exists():
        manifest = load_manifest()
        active_entry = find_active_ruleset(manifest)
        if args.candidate_id:
            candidate_entry = find_candidate_ruleset(manifest, args.candidate_id)
    else:
        print("Manifest not found, proceeding without manifest context.")

    # Generate summary
    md_content = generate_release_summary(
        calibration, walkforward, active_entry, candidate_entry,
    )

    # Write
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\nWrote release summary: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
