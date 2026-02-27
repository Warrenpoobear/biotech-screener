#!/usr/bin/env python3
"""Build a unified daily triage bundle from all gate artifacts.

Reads monitoring, eligibility, and ruleset-eval gate outputs from a snapshot
directory and produces a single triage packet with a consolidated summary.

Output: data/snapshots/{date}/daily_triage/
    context.json          — overall verdict, per-gate summaries, git sha
    summary.md            — 1-screen consolidated overview
    health_summary.md     — compact block for $GITHUB_STEP_SUMMARY
    explains/             — copies of all gate artifacts
    movers.csv            — eligibility movers (if available)

CLI:
    python3 scripts/build_daily_triage.py \
        --as-of-date 2026-02-27 \
        --snapshot-dir data/snapshots \
        --git-sha abc1234
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

VERSION = "1.0.0"
SCHEMA = "daily_triage_context.v1"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

VERDICT_RANK = {
    "FAIL": 4,
    "WARN": 3,
    "DEGRADED_SKIP": 2,
    "PASS": 1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON file, returning None if missing or unparseable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"


def _worst_verdict(verdicts: List[str]) -> str:
    """Return the worst verdict from a list (FAIL > WARN > DEGRADED_SKIP > PASS)."""
    non_missing = [v for v in verdicts if v != "MISSING"]
    if not non_missing:
        return "MISSING"
    return max(non_missing, key=lambda v: VERDICT_RANK.get(v, 0))


def _verdict_icon(verdict: str) -> str:
    return {
        "PASS": "✅",
        "WARN": "⚠️",
        "FAIL": "❌",
        "DEGRADED_SKIP": "🔄",
        "MISSING": "❓",
    }.get(verdict, "❓")


# ---------------------------------------------------------------------------
# Collect gate artifacts
# ---------------------------------------------------------------------------


def collect_gate_artifacts(snap_dir: Path) -> Dict[str, Any]:
    """Read all available gate JSONs + markdown paths from a snapshot directory.

    Returns a dict with keys for each artifact (None if missing).
    """
    artifacts: Dict[str, Any] = {}

    # Monitoring
    artifacts["monitoring_json"] = _read_json(snap_dir / "monitoring.json")
    artifacts["monitoring_json_path"] = snap_dir / "monitoring.json"

    # Eligibility gate
    artifacts["eligibility_gate_json"] = _read_json(snap_dir / "eligibility_gate.json")
    artifacts["eligibility_gate_json_path"] = snap_dir / "eligibility_gate.json"

    # Eligibility delta
    artifacts["eligibility_delta_json"] = _read_json(snap_dir / "eligibility_delta.json")
    artifacts["eligibility_delta_json_path"] = snap_dir / "eligibility_delta.json"

    # Eligibility triage
    triage_dir = snap_dir / "eligibility_triage"
    artifacts["eligibility_triage_json"] = _read_json(triage_dir / "triage_summary.json")
    artifacts["eligibility_triage_json_path"] = triage_dir / "triage_summary.json"
    artifacts["eligibility_triage_md_path"] = triage_dir / "triage_summary.md"
    artifacts["movers_csv_path"] = triage_dir / "movers.csv"

    # Ruleset eval — CI writes to a nested subdirectory, use rglob
    artifacts["ruleset_eval_json"] = None
    artifacts["ruleset_eval_json_path"] = None
    candidates = list(snap_dir.rglob("ruleset_eval.json"))
    if candidates:
        # Prefer shortest path (most direct)
        best = min(candidates, key=lambda p: len(p.parts))
        artifacts["ruleset_eval_json"] = _read_json(best)
        artifacts["ruleset_eval_json_path"] = best

    return artifacts


# ---------------------------------------------------------------------------
# Derive monitoring verdict (calls evaluate() from the monitoring gate)
# ---------------------------------------------------------------------------


def _derive_monitoring_verdict(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Derive monitoring verdict by calling the evaluate function.

    Returns {"verdict": str, "key_metrics": {...}}.
    """
    monitoring = artifacts.get("monitoring_json")
    if monitoring is None:
        return {"verdict": "MISSING", "key_metrics": {}}

    thresholds_path = _PROJECT_ROOT / "production_data" / "monitoring_thresholds" / "v1.json"
    if not thresholds_path.exists():
        return {"verdict": "UNKNOWN", "key_metrics": {}}

    try:
        thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    except Exception:
        return {"verdict": "UNKNOWN", "key_metrics": {}}

    try:
        from scripts.evaluate_monitoring_gate import evaluate
        verdict, fail_reasons, warn_reasons = evaluate(monitoring, thresholds)
    except Exception:
        return {"verdict": "UNKNOWN", "key_metrics": {}}

    coverage = monitoring.get("coverage", {})
    n_universe = coverage.get("n_universe", 0)
    n_eligible = coverage.get("n_eligible", 0)
    eligible_pct = (100 * n_eligible / n_universe) if n_universe > 0 else 0.0

    return {
        "verdict": verdict,
        "fail_reasons": fail_reasons,
        "warn_reasons": warn_reasons,
        "key_metrics": {
            "eligible_pct": round(eligible_pct, 1),
            "n_eligible": n_eligible,
            "n_universe": n_universe,
        },
    }


# ---------------------------------------------------------------------------
# Build triage context
# ---------------------------------------------------------------------------


def build_triage_context(
    as_of_date: str,
    artifacts: Dict[str, Any],
    git_sha: str = "",
) -> Dict[str, Any]:
    """Build context.json content from collected artifacts."""
    gates: Dict[str, Any] = {}

    # -- Monitoring --
    mon = _derive_monitoring_verdict(artifacts)
    gates["monitoring"] = mon

    # -- Eligibility --
    elig_gate = artifacts.get("eligibility_gate_json")
    elig_delta = artifacts.get("eligibility_delta_json")
    if elig_gate is not None:
        elig_verdict = elig_gate.get("verdict", "MISSING")
        elig_metrics = elig_gate.get("metrics", {})
        elig_triggers = elig_gate.get("triggers", [])
        gates["eligibility"] = {
            "verdict": elig_verdict,
            "triggers": elig_triggers,
            "key_metrics": elig_metrics,
        }
    elif elig_delta is not None:
        # Delta exists but no gate — might be DEGRADED_SKIP
        delta_d = elig_delta.get("delta", {})
        gates["eligibility"] = {
            "verdict": "DEGRADED_SKIP",
            "triggers": elig_delta.get("notes", []),
            "key_metrics": {
                "eligible_delta": delta_d.get("eligible", 0),
                "ineligible_delta": delta_d.get("ineligible", 0),
            },
        }
    else:
        gates["eligibility"] = {"verdict": "MISSING", "triggers": [], "key_metrics": {}}

    # -- Ruleset eval --
    rs_eval = artifacts.get("ruleset_eval_json")
    if rs_eval is not None:
        rs_gate = rs_eval.get("gate", {})
        rs_verdict = rs_gate.get("verdict", "MISSING")
        ts = rs_eval.get("temporal_stability", {})
        gates["ruleset_eval"] = {
            "verdict": rs_verdict,
            "fail_reasons": rs_gate.get("fail_reasons", []),
            "warn_reasons": rs_gate.get("warn_reasons", []),
            "key_metrics": {
                "n_evaluated": rs_eval.get("n_evaluated", 0),
                "mean_top60_overlap": ts.get("mean_top60_overlap"),
            },
        }
    else:
        gates["ruleset_eval"] = {
            "verdict": "MISSING",
            "fail_reasons": [],
            "warn_reasons": [],
            "key_metrics": {},
        }

    # -- Overall verdict --
    all_verdicts = [g["verdict"] for g in gates.values()]
    overall = _worst_verdict(all_verdicts)
    non_pass = [
        name for name, g in gates.items() if g["verdict"] not in ("PASS", "MISSING")
    ]

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "as_of_date": as_of_date,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha,
        "overall_verdict": overall,
        "non_pass_gates": non_pass,
        "gates": gates,
    }


# ---------------------------------------------------------------------------
# Render markdown outputs
# ---------------------------------------------------------------------------


def render_summary_md(ctx: Dict[str, Any], artifacts: Dict[str, Any]) -> str:
    """Full 1-screen summary markdown."""
    lines: List[str] = []
    overall = ctx["overall_verdict"]
    icon = _verdict_icon(overall)

    lines.append(f"# Daily Triage — {ctx['as_of_date']}")
    lines.append("")
    lines.append(f"**Overall: {icon} {overall}**")
    if ctx.get("git_sha"):
        lines.append(f"  \nCommit: `{ctx['git_sha'][:8]}`")
    lines.append("")

    # Gate summary table
    lines.append("## Gate Summary")
    lines.append("")
    lines.append("| Gate | Verdict | Key Metric |")
    lines.append("|------|---------|------------|")
    for name, g in ctx["gates"].items():
        v = g["verdict"]
        vi = _verdict_icon(v)
        km = g.get("key_metrics", {})
        # Pick a headline metric
        if name == "monitoring":
            metric_str = f"eligible: {km.get('eligible_pct', '?')}%"
        elif name == "eligibility":
            ed = km.get("eligible_delta") or km.get("eligible", "?")
            metric_str = f"eligible Δ: {ed}"
        elif name == "ruleset_eval":
            ne = km.get("n_evaluated", "?")
            ov = km.get("mean_top60_overlap", "?")
            metric_str = f"n={ne}, top60 overlap={ov}"
        else:
            metric_str = str(km)[:40] if km else "—"
        lines.append(f"| {name} | {vi} {v} | {metric_str} |")
    lines.append("")

    # Per non-PASS gate details
    non_pass = ctx.get("non_pass_gates", [])
    if non_pass:
        lines.append("## Non-PASS Gates")
        lines.append("")
        for name in non_pass:
            g = ctx["gates"][name]
            lines.append(f"### {name} — {_verdict_icon(g['verdict'])} {g['verdict']}")
            triggers = (
                g.get("triggers", [])
                or g.get("fail_reasons", [])
                or g.get("warn_reasons", [])
            )
            if triggers:
                for t in triggers:
                    lines.append(f"- {t}")
            else:
                lines.append("- (no details)")
            lines.append("")

    # Mover highlights
    triage = artifacts.get("eligibility_triage_json")
    if triage and triage.get("movers"):
        movers = triage["movers"]
        newly_inelig = [m for m in movers if m.get("cur_eligible") == 0 and m.get("prior_eligible") == 1]
        lines.append("## Eligibility Movers")
        lines.append("")
        lines.append(f"Total movers: {len(movers)}, newly ineligible: {len(newly_inelig)}")
        if newly_inelig:
            top5 = newly_inelig[:5]
            tickers = ", ".join(m["ticker"] for m in top5)
            lines.append(f"Top newly ineligible: {tickers}")
        lines.append("")

    # Action recommendation
    lines.append("## Recommendation")
    lines.append("")
    if overall == "FAIL":
        lines.append("**BLOCKED** — Investigate FAIL gates before proceeding.")
    elif overall == "WARN":
        lines.append("**Investigate** — Review WARN gates; proceed with caution.")
    elif overall == "DEGRADED_SKIP":
        lines.append("**Investigate** — Degraded data; review coverage metrics.")
    else:
        lines.append("**Proceed** — All gates passed.")
    lines.append("")

    return "\n".join(lines)


def render_health_summary_md(ctx: Dict[str, Any]) -> str:
    """Compact ~10-line block for $GITHUB_STEP_SUMMARY."""
    lines: List[str] = []
    overall = ctx["overall_verdict"]
    icon = _verdict_icon(overall)

    lines.append("### Daily Health Summary")
    lines.append("")
    lines.append(f"**{icon} {overall}** | {ctx['as_of_date']}")
    lines.append("")
    lines.append("| Gate | Verdict |")
    lines.append("|------|---------|")
    for name, g in ctx["gates"].items():
        v = g["verdict"]
        vi = _verdict_icon(v)
        lines.append(f"| {name} | {vi} {v} |")
    lines.append("")
    if ctx.get("non_pass_gates"):
        lines.append(f"Non-pass: {', '.join(ctx['non_pass_gates'])}")
    lines.append("")
    lines.append("See `daily_triage/summary.md` in the snapshot artifact for details.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Assemble the bundle
# ---------------------------------------------------------------------------


_EXPLAIN_COPIES = {
    "monitoring_json_path": "monitoring.json",
    "eligibility_gate_json_path": "eligibility_gate.json",
    "eligibility_delta_json_path": "eligibility_delta.json",
    "eligibility_triage_json_path": "eligibility_triage.json",
    "eligibility_triage_md_path": "eligibility_triage.md",
    "ruleset_eval_json_path": "ruleset_eval.json",
}


def assemble_bundle(
    snap_dir: Path,
    out_dir: Path,
    artifacts: Dict[str, Any],
    ctx: Dict[str, Any],
) -> None:
    """Write context.json + summary.md + health_summary.md, copy gate artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    explains_dir = out_dir / "explains"
    explains_dir.mkdir(exist_ok=True)

    # Write context.json
    (out_dir / "context.json").write_text(_stable_json(ctx), encoding="utf-8")

    # Write summary.md
    summary_md = render_summary_md(ctx, artifacts)
    (out_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    # Write health_summary.md
    health_md = render_health_summary_md(ctx)
    (out_dir / "health_summary.md").write_text(health_md, encoding="utf-8")

    # Copy gate artifacts into explains/
    for key, dest_name in _EXPLAIN_COPIES.items():
        src = artifacts.get(key)
        if src is not None and isinstance(src, Path) and src.exists():
            shutil.copy2(src, explains_dir / dest_name)

    # Also generate markdown for monitoring if we have monitoring JSON
    if artifacts.get("monitoring_json") is not None:
        mon_gate = ctx["gates"].get("monitoring", {})
        mon_md_lines = [
            f"# Monitoring Gate — {ctx['as_of_date']}",
            "",
            f"**Verdict: {mon_gate.get('verdict', 'UNKNOWN')}**",
            "",
        ]
        for k, v in mon_gate.get("key_metrics", {}).items():
            mon_md_lines.append(f"- {k}: {v}")
        for r in mon_gate.get("fail_reasons", []):
            mon_md_lines.append(f"- FAIL: {r}")
        for r in mon_gate.get("warn_reasons", []):
            mon_md_lines.append(f"- WARN: {r}")
        mon_md_lines.append("")
        (explains_dir / "monitoring.md").write_text(
            "\n".join(mon_md_lines), encoding="utf-8"
        )

    # Generate markdown for eligibility delta
    elig_delta = artifacts.get("eligibility_delta_json")
    if elig_delta is not None:
        delta = elig_delta.get("delta", {})
        ed_lines = [
            f"# Eligibility Delta — {ctx['as_of_date']}",
            "",
            f"Eligible Δ: {delta.get('eligible', 0)}",
            f"Ineligible Δ: {delta.get('ineligible', 0)}",
            "",
        ]
        for note in elig_delta.get("notes", []):
            ed_lines.append(f"- {note}")
        ed_lines.append("")
        (explains_dir / "eligibility_delta.md").write_text(
            "\n".join(ed_lines), encoding="utf-8"
        )

    # Generate markdown for ruleset eval
    rs_eval = artifacts.get("ruleset_eval_json")
    if rs_eval is not None:
        rs_gate = rs_eval.get("gate", {})
        re_lines = [
            f"# Ruleset Eval — {ctx['as_of_date']}",
            "",
            f"**Verdict: {rs_gate.get('verdict', 'UNKNOWN')}**",
            f"Evaluated: {rs_eval.get('n_evaluated', 0)} dates",
            "",
        ]
        for r in rs_gate.get("fail_reasons", []):
            re_lines.append(f"- FAIL: {r}")
        for r in rs_gate.get("warn_reasons", []):
            re_lines.append(f"- WARN: {r}")
        re_lines.append("")
        (explains_dir / "ruleset_eval.md").write_text(
            "\n".join(re_lines), encoding="utf-8"
        )

    # Copy movers.csv if available
    movers_path = artifacts.get("movers_csv_path")
    if movers_path is not None and isinstance(movers_path, Path) and movers_path.exists():
        shutil.copy2(movers_path, out_dir / "movers.csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a unified daily triage bundle from gate artifacts."
    )
    parser.add_argument(
        "--as-of-date", required=True,
        help="Snapshot date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--snapshot-dir", required=True,
        help="Root snapshot directory (e.g. data/snapshots).",
    )
    parser.add_argument(
        "--git-sha", default="",
        help="Git commit SHA to include in context.",
    )
    parser.add_argument(
        "--out-dir", default="",
        help="Output directory (default: {snap_dir}/{date}/daily_triage).",
    )
    args = parser.parse_args(argv)

    snap_root = Path(args.snapshot_dir)
    snap_dir = snap_root / args.as_of_date
    if not snap_dir.is_dir():
        print(f"ERROR: Snapshot directory not found: {snap_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else snap_dir / "daily_triage"

    print(f"Building daily triage for {args.as_of_date} ...")
    artifacts = collect_gate_artifacts(snap_dir)
    ctx = build_triage_context(args.as_of_date, artifacts, git_sha=args.git_sha)
    assemble_bundle(snap_dir, out_dir, artifacts, ctx)

    print(f"  Overall verdict: {_verdict_icon(ctx['overall_verdict'])} {ctx['overall_verdict']}")
    if ctx["non_pass_gates"]:
        print(f"  Non-pass gates: {', '.join(ctx['non_pass_gates'])}")
    print(f"  Output: {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
