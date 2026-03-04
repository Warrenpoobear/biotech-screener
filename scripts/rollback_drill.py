#!/usr/bin/env python3
"""Rollback drill — diagnose whether the active ruleset should be rolled back.

Reads the latest (or specified) production snapshot, checks ruleset_health, runs
a rerank-only cross-compare between the active ruleset and the last-known-good (LKG)
on the last N snapshots, and prints a structured verdict.

This is a read-only diagnostic. It never modifies anything. If rollback is
recommended, it prints the exact command to execute.

Usage:
    python3 scripts/rollback_drill.py
    python3 scripts/rollback_drill.py --as-of-date 2026-03-07
    python3 scripts/rollback_drill.py --n-snapshots 10
    python3 scripts/rollback_drill.py --json              # emit JSON to stdout
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_engine import DecisionRuleset
from scripts.eval_ruleset import evaluate_ruleset_rerank_only

SNAPSHOTS_ROOT = _PROJECT_ROOT / "data" / "snapshots"
MANIFEST_PATH = _PROJECT_ROOT / "production_data" / "decision_rulesets" / "manifest.json"
RULESETS_DIR = _PROJECT_ROOT / "production_data" / "decision_rulesets"

# Cross-compare thresholds: below these values → adds a reason
_CROSS_OVERLAP_WARN = 90.0   # top-20 overlap (active vs LKG)
_CROSS_OVERLAP60_WARN = 87.0  # top-60 overlap (active vs LKG)


# ── Manifest helpers ──────────────────────────────────────────────────────────

def _load_manifest() -> Dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _active_entry(manifest: Dict) -> Optional[Dict]:
    for r in manifest["rulesets"]:
        if r.get("status") == "active":
            return r
    return None


def _find_lkg(manifest: Dict) -> Optional[Dict]:
    """Walk manifest in reverse; return first retired entry promoted via promote_ruleset.py."""
    for entry in reversed(manifest["rulesets"]):
        if entry.get("status") != "retired":
            continue
        if entry.get("updated_by", "").startswith("promote_ruleset.py"):
            return entry
    return None


def _load_ruleset(entry: Dict) -> Optional[DecisionRuleset]:
    path = RULESETS_DIR / entry.get("file", "")
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return DecisionRuleset(**{k: v for k, v in data.items()
                               if k in DecisionRuleset.__dataclass_fields__})


# ── Snapshot helpers ──────────────────────────────────────────────────────────

def _production_snapshot_dates() -> List[str]:
    """Return sorted list of YYYY-MM-DD snapshot dirs."""
    dates = []
    for d in SNAPSHOTS_ROOT.iterdir():
        name = d.name
        if len(name) == 10 and name[4] == "-" and name[7] == "-":
            try:
                datetime.strptime(name, "%Y-%m-%d")
                dates.append(name)
            except ValueError:
                pass
    return sorted(dates)


def _load_ruleset_health(as_of_date: str) -> Dict:
    p = SNAPSHOTS_ROOT / as_of_date / "ruleset_health.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


# ── Reason builder ────────────────────────────────────────────────────────────

def _build_reasons(
    rh: Dict,
    cross: Dict,
    n_evaluated: int,
) -> List[str]:
    reasons: List[str] = []

    # From ruleset_health
    if rh.get("recommend_rollback"):
        consec = rh.get("consecutive_warn_days", 0)
        reasons.append(f"ruleset_health: {consec} consecutive WARN days (threshold 3)")

    warn_reasons = rh.get("warn_reasons") or []
    for wr in warn_reasons:
        reasons.append(f"drift: {wr}")

    # From cross-compare (independent signal)
    if n_evaluated > 0:
        top20 = cross.get("mean_top20_overlap")
        top60 = cross.get("mean_top60_overlap")
        if top20 is not None and top20 < _CROSS_OVERLAP_WARN:
            reasons.append(
                f"cross-compare: mean top-20 overlap vs LKG = {top20:.1f}% "
                f"(< {_CROSS_OVERLAP_WARN:.0f}% threshold)"
            )
        if top60 is not None and top60 < _CROSS_OVERLAP60_WARN:
            reasons.append(
                f"cross-compare: mean top-60 overlap vs LKG = {top60:.1f}% "
                f"(< {_CROSS_OVERLAP60_WARN:.0f}% threshold)"
            )

    return reasons


# ── Main drill ────────────────────────────────────────────────────────────────

def run_drill(
    as_of_date: str,
    n_snapshots: int = 5,
) -> Dict[str, Any]:
    """Run the rollback drill and return a structured result dict."""
    manifest = _load_manifest()
    active = _active_entry(manifest)
    lkg = _find_lkg(manifest)

    if active is None:
        return {"error": "No active ruleset found in manifest"}

    active_ruleset = _load_ruleset(active)
    lkg_ruleset = _load_ruleset(lkg) if lkg else None

    # All production snapshot dates up to and including as_of_date
    all_dates = _production_snapshot_dates()
    window = [d for d in all_dates if d <= as_of_date][-n_snapshots:]

    # ── Ruleset health (from the as_of_date snapshot) ───────────────
    rh = _load_ruleset_health(as_of_date)
    rh_recommend = rh.get("recommend_rollback", False)
    rh_consec = rh.get("consecutive_warn_days", 0)
    rh_status = rh.get("status", "unknown")
    rh_detail = rh.get("detail", "")
    rh_warn_reasons = rh.get("warn_reasons") or []

    # ── Cross-compare ────────────────────────────────────────────────
    cross: Dict[str, Any] = {}
    n_evaluated = 0
    n_skipped = 0
    cross_error: Optional[str] = None

    if active_ruleset and lkg_ruleset and window:
        try:
            result = evaluate_ruleset_rerank_only(
                candidate_ruleset=active_ruleset,
                baseline_ruleset=lkg_ruleset,
                dates=window,
                snapshot_dir=SNAPSHOTS_ROOT,
                k=60,
            )
            cross = result.get("cross_comparison", {})
            n_evaluated = result.get("n_evaluated", 0)
            n_skipped = result.get("n_skipped", 0)
        except Exception as exc:
            cross_error = str(exc)
    elif not lkg_ruleset:
        cross_error = "No LKG ruleset available for cross-compare"
    elif not active_ruleset:
        cross_error = "Could not load active ruleset file"

    # ── Build reasons + verdict ──────────────────────────────────────
    reasons = _build_reasons(rh, cross, n_evaluated)
    recommended = bool(reasons)  # any reason → recommend

    # Override: ruleset_health's own flag is authoritative
    if rh_recommend:
        recommended = True

    return {
        "schema": "rollback_drill.v1",
        "as_of_date": as_of_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "active": {
            "id": active.get("id"),
            "file": active.get("file"),
            "description": active.get("description", ""),
        },
        "lkg": {
            "id": lkg.get("id") if lkg else None,
            "file": lkg.get("file") if lkg else None,
            "description": lkg.get("description", "") if lkg else None,
        },
        "ruleset_health": {
            "status": rh_status,
            "consecutive_warn_days": rh_consec,
            "recommend_rollback": rh_recommend,
            "detail": rh_detail,
            "warn_reasons": rh_warn_reasons,
        },
        "cross_compare": {
            "window": window,
            "n_snapshots_requested": n_snapshots,
            "n_evaluated": n_evaluated,
            "n_skipped": n_skipped,
            "mean_top20_overlap_pct": cross.get("mean_top20_overlap"),
            "mean_top60_overlap_pct": cross.get("mean_top60_overlap"),
            "worst_top60_overlap_pct": cross.get("worst_top60_overlap"),
            "mean_spearman": cross.get("mean_spearman"),
            "mean_pct_rank_changed": cross.get("mean_pct_rank_changed"),
            "error": cross_error,
        },
        "reasons": reasons,
        "recommended": recommended,
    }


# ── Renderers ─────────────────────────────────────────────────────────────────

def _v(val: Any, fmt: str = "", suffix: str = "") -> str:
    if val is None:
        return "—"
    if isinstance(val, float) and fmt:
        return f"{val:{fmt}}{suffix}"
    return str(val) + suffix


def render_markdown(drill: Dict) -> str:
    """Render drill result as Markdown (for ROLLBACK.md)."""
    lines: List[str] = []
    rec = drill["recommended"]
    rh = drill["ruleset_health"]
    cc = drill["cross_compare"]

    verdict_str = "⚠ ROLLBACK RECOMMENDED" if rec else "✅ No rollback needed"
    lines += [
        f"# Rollback Drill — {drill['as_of_date']}",
        "",
        f"**Generated**: {drill['generated_at'][:19].replace('T', ' ')} UTC  ",
        f"**Recommendation**: {verdict_str}",
        "",
        "## Active / LKG",
        "",
        "| | ID | File |",
        "|--|--|--|",
        f"| Active | `{drill['active']['id']}` | `{drill['active']['file']}` |",
    ]
    if drill["lkg"]["id"]:
        lines.append(
            f"| LKG | `{drill['lkg']['id']}` | `{drill['lkg']['file']}` |"
        )
    else:
        lines.append("| LKG | — | _(none found in manifest)_ |")
    lines.append("")

    lines += [
        "## Ruleset Health",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Status | `{rh['status']}` |",
        f"| Consecutive WARNs | {rh['consecutive_warn_days']}d |",
        f"| recommend_rollback | {'**YES**' if rh['recommend_rollback'] else 'no'} |",
        f"| Detail | {rh['detail'] or '—'} |",
    ]
    if rh["warn_reasons"]:
        lines.append("")
        lines.append("**Warn reasons**:")
        for wr in rh["warn_reasons"]:
            lines.append(f"- {wr}")
    lines.append("")

    n_eval = cc["n_evaluated"]
    window = cc["window"]
    date_range = f"{window[0]} → {window[-1]}" if window else "—"
    lines += [
        "## Cross-Compare (active vs LKG)",
        "",
        f"_{n_eval}/{cc['n_snapshots_requested']} snapshots [{date_range}]_",
        "",
    ]
    if cc["error"]:
        lines += [f"> ⚠ Error: {cc['error']}", ""]
    else:
        lines += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Mean top-20 overlap | {_v(cc['mean_top20_overlap_pct'], '.1f', '%')} |",
            f"| Mean top-60 overlap | {_v(cc['mean_top60_overlap_pct'], '.1f', '%')} |",
            f"| Worst top-60 overlap | {_v(cc['worst_top60_overlap_pct'], '.1f', '%')} |",
            f"| Mean Spearman ρ | {_v(cc['mean_spearman'], '.4f')} |",
            f"| Mean % ranks changed | {_v(cc['mean_pct_rank_changed'], '.1f', '%')} |",
            "",
        ]

    if drill["reasons"]:
        lines += ["## Triggered By", ""]
        for i, r in enumerate(drill["reasons"], 1):
            lines.append(f"{i}. {r}")
        lines.append("")

    if rec:
        reason_str = drill["reasons"][0] if drill["reasons"] else "rollback drill triggered"
        lines += [
            "## Execute Rollback",
            "",
            "```bash",
            f'python3 scripts/promote_ruleset.py --rollback --reason "drill: {reason_str}"',
            "```",
            "",
        ]

    return "\n".join(lines)


def write_drill_artifacts(drill: Dict, out_dir: Path) -> tuple:
    """Write ROLLBACK.md + ROLLBACK.json to out_dir. Returns (md_path, json_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "ROLLBACK.md"
    json_path = out_dir / "ROLLBACK.json"
    md_path.write_text(render_markdown(drill), encoding="utf-8")
    json_path.write_text(json.dumps(drill, indent=2, default=str), encoding="utf-8")
    return md_path, json_path


def render_text(drill: Dict) -> str:
    lines: List[str] = []
    sep = "=" * 70

    lines += [
        sep,
        f"Rollback Drill — {drill['as_of_date']}",
        f"  Generated : {drill['generated_at'][:19].replace('T', ' ')} UTC",
        f"  Active    : {drill['active']['id']}  ({drill['active']['file']})",
    ]
    if drill["lkg"]["id"]:
        lines.append(f"  LKG       : {drill['lkg']['id']}  ({drill['lkg']['file']})")
    else:
        lines.append("  LKG       : none found in manifest")
    lines.append("")

    # Verdict banner
    if drill["recommended"]:
        lines += ["  ⚠  ROLLBACK RECOMMENDED", ""]
    else:
        lines += ["  ✅  No rollback needed", ""]

    # Ruleset health
    rh = drill["ruleset_health"]
    lines += [
        "Ruleset health:",
        f"  Status              : {rh['status']}",
        f"  Consecutive WARNs   : {rh['consecutive_warn_days']}d",
        f"  recommend_rollback  : {'YES' if rh['recommend_rollback'] else 'no'}",
    ]
    if rh["detail"]:
        lines.append(f"  Detail              : {rh['detail']}")
    if rh["warn_reasons"]:
        for wr in rh["warn_reasons"]:
            lines.append(f"  Warn reason         : {wr}")
    lines.append("")

    # Cross-compare
    cc = drill["cross_compare"]
    n_eval = cc["n_evaluated"]
    window = cc["window"]
    date_range = f"{window[0]} → {window[-1]}" if window else "—"
    lines += [
        f"Cross-compare (active vs LKG, {n_eval}/{cc['n_snapshots_requested']} snapshots [{date_range}]):",
    ]
    if cc["error"]:
        lines.append(f"  ERROR: {cc['error']}")
    else:
        lines += [
            f"  Mean top-20 overlap : {_v(cc['mean_top20_overlap_pct'], '.1f', '%')}",
            f"  Mean top-60 overlap : {_v(cc['mean_top60_overlap_pct'], '.1f', '%')}",
            f"  Worst top-60 overlap: {_v(cc['worst_top60_overlap_pct'], '.1f', '%')}",
            f"  Mean Spearman ρ     : {_v(cc['mean_spearman'], '.4f')}",
            f"  Mean % ranks changed: {_v(cc['mean_pct_rank_changed'], '.1f', '%')}",
        ]
    lines.append("")

    # Reasons
    if drill["reasons"]:
        lines += ["Reasons:"]
        for i, r in enumerate(drill["reasons"], 1):
            lines.append(f"  [{i}] {r}")
        lines.append("")

    # Rollback command
    if drill["recommended"]:
        reason_str = drill["reasons"][0] if drill["reasons"] else "rollback drill triggered"
        lines += [
            "To execute rollback:",
            f"  python3 scripts/promote_ruleset.py --rollback "
            f"--reason \"drill: {reason_str}\"",
            "",
        ]

    lines.append(sep)
    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rollback drill — read-only diagnostic for active ruleset health",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--as-of-date", default=None,
        help="YYYY-MM-DD snapshot date (default: latest production snapshot)",
    )
    parser.add_argument(
        "--n-snapshots", type=int, default=5,
        help="Number of recent snapshots to use for cross-compare (default: 5)",
    )
    parser.add_argument(
        "--json", dest="emit_json", action="store_true",
        help="Emit JSON to stdout instead of human-readable text",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help=(
            "Write ROLLBACK.md + ROLLBACK.json to this directory "
            "(always written regardless of --json flag)"
        ),
    )
    args = parser.parse_args()

    as_of_date = args.as_of_date
    if as_of_date is None:
        all_dates = _production_snapshot_dates()
        if not all_dates:
            print("ERROR: no production snapshots found", file=sys.stderr)
            sys.exit(1)
        as_of_date = all_dates[-1]
        print(f"Using latest snapshot: {as_of_date}", file=sys.stderr)

    drill = run_drill(as_of_date, n_snapshots=args.n_snapshots)

    if args.out_dir:
        md_path, json_path = write_drill_artifacts(drill, args.out_dir)
        print(f"  → {md_path}", file=sys.stderr)
        print(f"  → {json_path}", file=sys.stderr)

    if args.emit_json:
        print(json.dumps(drill, indent=2, default=str))
    else:
        print(render_text(drill))

    # Exit code: 2 if rollback recommended (mirrors gate exit codes)
    if drill.get("recommended"):
        sys.exit(2)


if __name__ == "__main__":
    main()
