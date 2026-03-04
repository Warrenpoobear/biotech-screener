#!/usr/bin/env python3
"""Weekly health packet generator.

Reads a snapshot directory and produces:
  output/health_packets/health_YYYY-MM-DD.md   (human-readable)
  output/health_packets/health_YYYY-MM-DD.json  (machine-readable for charting)

Usage:
    python3 tools/weekly_health_packet.py                         # latest snapshot
    python3 tools/weekly_health_packet.py --as-of-date 2026-03-07
    python3 tools/weekly_health_packet.py --relaxed               # acknowledge non-strict run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SNAPSHOTS_ROOT = PROJECT_ROOT / "data" / "snapshots"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "health_packets"


# ── Snapshot discovery ───────────────────────────────────────────────

def _find_latest_snapshot_date() -> Optional[str]:
    """Return the most recent YYYY-MM-DD snapshot directory name."""
    candidates = []
    for d in SNAPSHOTS_ROOT.iterdir():
        name = d.name
        if len(name) == 10 and name[4] == "-" and name[7] == "-":
            try:
                datetime.strptime(name, "%Y-%m-%d")
                candidates.append(name)
            except ValueError:
                pass
    return max(candidates) if candidates else None


def _load_json(path: Path) -> Optional[Dict]:
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


# ── Turnover (current + trailing 4-week avg) ────────────────────────

def _turnover_from_delta(delta: List[Dict[str, str]]) -> Optional[float]:
    """Return turnover % for a delta CSV, or None if no prior portfolio."""
    entries = sum(1 for r in delta if r.get("in_current") == "1"
                  and r.get("in_prior") == "0")
    exits = sum(1 for r in delta if r.get("in_current") == "0"
                and r.get("in_prior") == "1")
    prior_active = sum(1 for r in delta if r.get("in_prior") == "1")
    if prior_active == 0:
        return None  # fresh-start run — no prior portfolio to compare against
    return (entries + exits) / (prior_active * 2) * 100


def _compute_turnover(snap_dir: Path, as_of_date: str) -> Dict:
    """Compute turnover from run_delta CSV; also trailing 4-week avg (skips fresh-start runs)."""
    delta = _load_csv(snap_dir / "phase2_run_delta.csv")
    entries = sum(1 for r in delta if r.get("in_current") == "1"
                  and r.get("in_prior") == "0")
    exits = sum(1 for r in delta if r.get("in_current") == "0"
                and r.get("in_prior") == "1")
    turnover_pct = _turnover_from_delta(delta)

    # Trailing 4-week snapshots (skip fresh-start runs where prior_active=0)
    trailing = []
    snap_dates = sorted(
        d.name for d in SNAPSHOTS_ROOT.iterdir()
        if d.is_dir() and len(d.name) == 10 and d.name[4] == "-" and d.name < as_of_date
    )
    for prior_date in reversed(snap_dates[-8:]):  # look back up to 8 to find 4 valid
        prior_delta = _load_csv(SNAPSHOTS_ROOT / prior_date / "phase2_run_delta.csv")
        if not prior_delta:
            continue
        t = _turnover_from_delta(prior_delta)
        if t is not None:
            trailing.append(t)
        if len(trailing) >= 4:
            break

    trailing_avg = sum(trailing) / len(trailing) if trailing else None
    return {
        "entries": entries,
        "exits": exits,
        "turnover_pct": round(turnover_pct, 2) if turnover_pct is not None else None,
        "trailing_4w_avg_pct": round(trailing_avg, 2) if trailing_avg is not None else None,
        "trailing_n": len(trailing),
        "fresh_start": turnover_pct is None,
    }


# ── Portfolio shape ──────────────────────────────────────────────────

def _portfolio_shape(snap_dir: Path) -> Dict:
    positions = _load_csv(snap_dir / "portfolio_positions.csv")
    portfolio = [
        r for r in positions
        if r.get("actionable_rank", "").strip()
        and r.get("eligible", "") == "1"
    ]
    portfolio.sort(key=lambda r: int(r.get("actionable_rank", "9999") or "9999"))

    portfolio_json = _load_json(snap_dir / "decision_portfolio.json") or {}
    n_eligible = portfolio_json.get("n_eligible", len(portfolio))
    top_k = portfolio[:20]

    weight_sum = sum(float(r.get("target_weight_pct") or 0) for r in top_k)

    top10 = [
        {
            "rank": r.get("actionable_rank"),
            "ticker": r.get("ticker"),
            "tier": r.get("tier_any"),
            "weight_pct": float(r.get("target_weight_pct") or 0),
            "size_band": r.get("size_band"),
            "risk_flags": r.get("risk_flags") or "",
        }
        for r in top_k[:10]
    ]

    exp = _load_json(snap_dir / "health_exposure_metrics.json") or {}
    exp_metrics = exp.get("metrics", {})

    return {
        "n_eligible": n_eligible,
        "n_portfolio": len(top_k),
        "weight_sum_pct": round(weight_sum, 2),
        "top10": top10,
        "top5_weight_pct": exp_metrics.get("top5_weight_pct"),
        "max_single_weight_pct": exp_metrics.get("max_single_weight_pct"),
        "high_vol_or_beta_pct": exp_metrics.get("high_vol_or_beta_weight_pct"),
        "headwind_pct": exp_metrics.get("headwind_weight_pct"),
        "high_risk_flags": exp.get("checks", {}),
    }


# ── Action-required items ────────────────────────────────────────────

def _action_items(
    *,
    gates: List[Dict],
    drift: Optional[Dict],
    ruleset_health: Optional[Dict],
    turnover: Dict,
    cache: Optional[Dict],
) -> List[Dict]:
    items: List[Dict] = []

    # New FAILs
    for g in gates:
        if g.get("status") == "FAIL":
            items.append({"severity": "FAIL", "type": "gate_fail",
                          "detail": f"{g['name']}: {g.get('detail', '')}"})

    # Drift WARN streak ≥ 2
    if ruleset_health and ruleset_health.get("consecutive_warn_days", 0) >= 2:
        items.append({
            "severity": "WARN",
            "type": "drift_warn_streak",
            "detail": (f"Ruleset health WARN streak = "
                       f"{ruleset_health['consecutive_warn_days']} days"),
        })

    # Rollback recommendation
    if ruleset_health and ruleset_health.get("recommend_rollback"):
        items.append({
            "severity": "FAIL",
            "type": "rollback_recommended",
            "detail": (
                "ruleset_health recommends rollback — "
                "run: python3 scripts/rollback_drill.py"
            ),
        })

    # Turnover spike vs 4-week avg (skip fresh-start runs where tc is None)
    t4 = turnover.get("trailing_4w_avg_pct")
    tc = turnover.get("turnover_pct")
    if tc is not None and t4 is not None and tc > t4 * 2.5 and tc > 5:
        items.append({
            "severity": "WARN",
            "type": "turnover_spike",
            "detail": f"Turnover {tc:.1f}% vs 4w avg {t4:.1f}% (>2.5x)",
        })

    # Cache health issues (skip "unknown" — old snapshots may not have cache data)
    if cache and cache.get("overall_status") not in ("ok", "unknown", None):
        items.append({"severity": "WARN", "type": "cache_health",
                      "detail": f"overall_status={cache['overall_status']}"})

    return items


# ── Preflight summary (from audit dir if present) ───────────────────

def _preflight_summary(snap_dir: Path) -> Optional[Dict]:
    """Read preflight_summary.json from the snapshot's audit dir, if present."""
    for subdir in ["audit", "preflight"]:
        p = snap_dir / subdir / "preflight_summary.json"
        if p.is_file():
            return _load_json(p)
    return None


# ── Main packet builder ──────────────────────────────────────────────

def build_health_packet(as_of_date: str, *, relaxed: bool = False) -> Dict:
    snap_dir = SNAPSHOTS_ROOT / as_of_date
    if not snap_dir.is_dir():
        raise FileNotFoundError(f"Snapshot directory not found: {snap_dir}")

    metadata = _load_json(snap_dir / "metadata.json") or {}
    run_manifest = _load_json(snap_dir / "run_manifest.json") or {}
    drift = _load_json(snap_dir / "drift_report.json") or {}
    ruleset_health = _load_json(snap_dir / "ruleset_health.json") or {}
    cache = _load_json(snap_dir / "cache_health.json") or {}
    phase2 = _load_json(snap_dir / "phase2_health.json") or {}

    # ── Provenance ───────────────────────────────────────────────────
    git_info = run_manifest.get("git", {})
    ruleset_info = run_manifest.get("ruleset", {})
    git_sha = git_info.get("commit_sha", "unknown")[:8]
    ruleset_id = ruleset_info.get("ruleset_hash") or ruleset_health.get("active_ruleset_id", "unknown")
    ruleset_path = snap_dir / "decision_ruleset.json"
    ruleset_filename = "unknown"
    if ruleset_path.is_file():
        try:
            dr = json.loads(ruleset_path.read_text())
            # No filename stored in the json; use the name from run_manifest if present
        except Exception:
            pass
    # Fallback: use run_manifest's ruleset_version field
    ruleset_version = ruleset_info.get("ruleset_version", "unknown")

    provenance = {
        "as_of_date": as_of_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "ruleset_id": ruleset_id,
        "ruleset_version": ruleset_version,
        "snapshot_root": str(SNAPSHOTS_ROOT),
        "mode": "relaxed" if relaxed else "strict",
        "overall_status": run_manifest.get("overall_status", "unknown"),
    }

    # ── Gates ────────────────────────────────────────────────────────
    gates = run_manifest.get("gates", [])
    fail_gates = [g for g in gates if g.get("status") == "FAIL"]
    warn_gates = [g for g in gates if g.get("status") == "WARN"]
    pass_gates = [g for g in gates if g.get("status") == "PASS"]

    # ── Preflight ────────────────────────────────────────────────────
    preflight = _preflight_summary(snap_dir)

    # ── Drift ────────────────────────────────────────────────────────
    drift_metrics = drift.get("metrics", {})
    drift_data = {
        "status": drift.get("status", "unknown"),
        "top20_overlap_pct": drift_metrics.get("top20_overlap_pct"),
        "top60_overlap_pct": drift_metrics.get("top60_overlap_pct"),
        "rank_spearman_rho": drift_metrics.get("rank_spearman_rho"),
        "warn_reasons": drift.get("warn_reasons", []),
    }

    # ── Ruleset health ───────────────────────────────────────────────
    rh_today = ruleset_health.get("today", {})
    rh_data = {
        "status": ruleset_health.get("status", "unknown"),
        "consecutive_warn_days": ruleset_health.get("consecutive_warn_days", 0),
        "recommend_rollback": ruleset_health.get("recommend_rollback", False),
        "days_since_promotion": ruleset_health.get("days_since_promotion"),
        "top60_overlap_pct": rh_today.get("top60_overlap_pct"),
        "max_rank_shift": rh_today.get("max_rank_shift"),
    }

    # ── Cache health ─────────────────────────────────────────────────
    cache_data = {
        "overall_status": cache.get("overall_status", "unknown"),
        "sec8k": cache.get("sec8k", {}),
        "ctgov": cache.get("ctgov", {}),
        "degraded_run": cache.get("degraded_run", False),
    }

    # ── Turnover ─────────────────────────────────────────────────────
    turnover = _compute_turnover(snap_dir, as_of_date)

    # ── Portfolio ────────────────────────────────────────────────────
    portfolio = _portfolio_shape(snap_dir)

    # ── Action items ─────────────────────────────────────────────────
    action_items = _action_items(
        gates=gates,
        drift=drift_data,
        ruleset_health=rh_data,
        turnover=turnover,
        cache=cache_data,
    )

    return {
        "schema": "health_packet.v1",
        "relaxed": relaxed,
        "provenance": provenance,
        "gates": {
            "pass_count": len(pass_gates),
            "warn_count": len(warn_gates),
            "fail_count": len(fail_gates),
            "warn_names": [g["name"] for g in warn_gates],
            "fail_names": [g["name"] for g in fail_gates],
            "fail_details": [{"name": g["name"], "detail": g.get("detail", "")}
                             for g in fail_gates],
        },
        "preflight": preflight,
        "drift": drift_data,
        "ruleset_health": rh_data,
        "cache": cache_data,
        "turnover": turnover,
        "portfolio": portfolio,
        "action_items": action_items,
    }


# ── Markdown renderer ────────────────────────────────────────────────

def _v(val, fmt=".2f", suffix="") -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:{fmt}}{suffix}"
    return str(val) + suffix


def render_markdown(packet: Dict) -> str:
    prov = packet["provenance"]
    gates = packet["gates"]
    drift = packet["drift"]
    rh = packet["ruleset_health"]
    cache = packet["cache"]
    turn = packet["turnover"]
    port = packet["portfolio"]
    actions = packet["action_items"]

    lines: List[str] = []

    # ── Banner ───────────────────────────────────────────────────────
    if packet["relaxed"]:
        lines += [
            "```",
            "⚠️  RELAXED MODE — results not under normal production guarantees",
            "⚠️  WARN-status snapshots included. Review manually before IC use.",
            "```",
            "",
        ]

    lines += [
        f"# Weekly Health Packet — {prov['as_of_date']}",
        "",
        f"**Generated**: {prov['generated_at'][:19].replace('T',' ')} UTC  ",
        f"**Overall status**: `{prov['overall_status'].upper()}`  ",
        f"**Mode**: `{prov['mode']}`",
        "",
    ]

    # ── Action required ──────────────────────────────────────────────
    if actions:
        lines += ["## ⚠ Action Required", ""]
        for item in actions:
            icon = "🔴" if item["severity"] == "FAIL" else "🟡"
            lines.append(f"- {icon} **{item['type']}**: {item['detail']}")
        lines.append("")
    else:
        lines += ["## ✅ No Action Required", "", "All thresholds green.", ""]

    # ── Provenance ───────────────────────────────────────────────────
    lines += [
        "## Provenance",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| as_of_date | `{prov['as_of_date']}` |",
        f"| git_sha | `{prov['git_sha']}` |",
        f"| ruleset_id | `{prov['ruleset_id']}` |",
        f"| ruleset_version | `{prov['ruleset_version']}` |",
        f"| snapshot_root | `{prov['snapshot_root']}` |",
        "",
    ]

    # ── Gates checklist ──────────────────────────────────────────────
    lines += [
        "## Gates Checklist",
        "",
        f"| Status | Count |",
        f"|--------|-------|",
        f"| PASS | {gates['pass_count']} |",
        f"| WARN | {gates['warn_count']} |",
        f"| FAIL | {gates['fail_count']} |",
        "",
    ]
    if gates["warn_names"]:
        lines.append(f"**WARN gates**: {', '.join(f'`{n}`' for n in gates['warn_names'])}")
        lines.append("")
    if gates["fail_details"]:
        lines += ["**FAIL details**:"]
        for fd in gates["fail_details"]:
            lines.append(f"- `{fd['name']}`: {fd['detail']}")
        lines.append("")

    # ── Preflight ────────────────────────────────────────────────────
    pf = packet.get("preflight")
    if pf:
        lines += [
            "## Preflight",
            "",
            f"| Status | Count |",
            f"|--------|-------|",
            f"| PASS | {pf.get('n_pass', '—')} |",
            f"| WARN | {pf.get('n_warn', '—')} |",
            f"| FAIL | {pf.get('n_fail', '—')} |",
            "",
        ]
        top_reasons = pf.get("top_warn_reasons") or pf.get("warn_reason_counts") or {}
        if top_reasons:
            lines.append("**Top WARN reasons**:")
            sorted_reasons = sorted(top_reasons.items(), key=lambda x: -x[1])
            for reason, count in sorted_reasons[:5]:
                lines.append(f"- `{reason}`: {count}")
            lines.append("")
    else:
        lines += ["## Preflight", "", "_No preflight artifact found for this snapshot._", ""]

    # ── Core gates ───────────────────────────────────────────────────
    lines += [
        "## Core Gates",
        "",
        "### Drift",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Status | `{drift['status']}` |",
        f"| Top-20 overlap | {_v(drift['top20_overlap_pct'], '.1f', '%')} |",
        f"| Top-60 overlap | {_v(drift['top60_overlap_pct'], '.1f', '%')} |",
        f"| Rank Spearman ρ | {_v(drift['rank_spearman_rho'], '.4f')} |",
        "",
    ]
    if drift["warn_reasons"]:
        lines.append(f"**Warn reasons**: {', '.join(drift['warn_reasons'])}")
        lines.append("")

    lines += [
        "### Ruleset Health",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Status | `{rh['status']}` |",
        f"| Consecutive WARN days | {rh['consecutive_warn_days']} |",
        f"| Days since promotion | {_v(rh['days_since_promotion'], 'd')} |",
        f"| Top-60 overlap vs baseline | {_v(rh['top60_overlap_pct'], '.1f', '%')} |",
        f"| Max rank shift | {_v(rh['max_rank_shift'], '.2f')} |",
        f"| Recommend rollback | {'**YES**' if rh['recommend_rollback'] else 'no'} |",
        "",
        "### Cache Health",
        "",
        f"| Cache | Count | Status |",
        f"|-------|-------|--------|",
    ]
    for key, label in [("sec8k", "SEC 8-K"), ("ctgov", "CTGov PIT")]:
        cd = cache.get(key, {})
        lines.append(f"| {label} | {cd.get('count','—')} | "
                     f"`{cd.get('status','unknown')}` |")
    lines += [
        f"",
        f"**Overall**: `{cache['overall_status'].upper()}`"
        + (" ⚠ Degraded run" if cache["degraded_run"] else ""),
        "",
    ]

    # ── Turnover ─────────────────────────────────────────────────────
    spike = ""
    t4 = turn.get("trailing_4w_avg_pct")
    tc = turn.get("turnover_pct")  # None for fresh-start runs
    if tc is not None and t4 is not None and tc > t4 * 2.5 and tc > 5:
        spike = " ⚠ SPIKE"
    fresh_note = " _(fresh start — no prior portfolio)_" if turn.get("fresh_start") else ""
    lines += [
        "### Turnover",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| This week | {_v(tc, '.2f', '%') + spike + fresh_note} |",
        f"| Trailing 4-week avg | {_v(t4, '.2f', '%')} (n={turn['trailing_n']}) |",
        f"| Entries | {turn['entries']} |",
        f"| Exits | {turn['exits']} |",
        "",
    ]

    # ── Portfolio shape ───────────────────────────────────────────────
    lines += [
        "## Portfolio Shape",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Eligible securities | {port['n_eligible']} |",
        f"| Portfolio (top-K) | {port['n_portfolio']} |",
        f"| Weight sum | {_v(port['weight_sum_pct'], '.2f', '%')} |",
        f"| Top-5 concentration | {_v(port['top5_weight_pct'], '.1f', '%')} |",
        f"| Max single weight | {_v(port['max_single_weight_pct'], '.2f', '%')} |",
        f"| High-vol or beta | {_v(port['high_vol_or_beta_pct'], '.1f', '%')} |",
        f"| Momentum headwind | {_v(port['headwind_pct'], '.1f', '%')} |",
        "",
        "### Top 10 Positions",
        "",
        "| Rank | Ticker | Tier | Weight | Size | Flags |",
        "|------|--------|------|--------|------|-------|",
    ]
    for p in port["top10"]:
        w = f"{p['weight_pct']:.2f}%" if p['weight_pct'] else "—"
        flags = p.get("risk_flags") or "—"
        lines.append(f"| {p['rank']} | `{p['ticker']}` | {p['tier']} | "
                     f"{w} | {p['size_band']} | {flags} |")
    lines.append("")

    return "\n".join(lines)


# ── Entry point ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate weekly health packet for a snapshot date",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--as-of-date", default=None,
        help="YYYY-MM-DD snapshot date (default: latest snapshot)",
    )
    parser.add_argument(
        "--relaxed", action="store_true", default=False,
        help="Acknowledge this is a non-strict / non-production run",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=OUTPUT_ROOT,
        help=f"Output directory (default: {OUTPUT_ROOT})",
    )
    args = parser.parse_args()

    as_of_date = args.as_of_date
    if as_of_date is None:
        as_of_date = _find_latest_snapshot_date()
        if as_of_date is None:
            print(f"ERROR: no snapshots found in {SNAPSHOTS_ROOT}", file=sys.stderr)
            sys.exit(1)
        print(f"Using latest snapshot: {as_of_date}")

    print(f"Building health packet for {as_of_date} ...")
    packet = build_health_packet(as_of_date, relaxed=args.relaxed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.out_dir / f"health_{as_of_date}.md"
    json_path = args.out_dir / f"health_{as_of_date}.json"

    # Write JSON (drop non-serialisable rich_risk_checks blob)
    serialisable = {k: v for k, v in packet.items() if k != "portfolio" or True}
    # Strip the nested checks dict from portfolio for cleaner JSON
    clean_packet = dict(packet)
    clean_portfolio = dict(packet["portfolio"])
    clean_portfolio.pop("high_risk_flags", None)
    clean_packet["portfolio"] = clean_portfolio
    json_path.write_text(json.dumps(clean_packet, indent=2, default=str))

    md_path.write_text(render_markdown(packet))

    # Print summary to stdout
    prov = packet["provenance"]
    actions = packet["action_items"]
    print(f"\n{'='*60}")
    print(f"Health Packet: {as_of_date}")
    print(f"  Status  : {prov['overall_status'].upper()}")
    print(f"  Ruleset : {prov['ruleset_id']} (v{prov['ruleset_version']})")
    print(f"  git SHA : {prov['git_sha']}")
    gates_s = packet["gates"]
    print(f"  Gates   : {gates_s['pass_count']} PASS, "
          f"{gates_s['warn_count']} WARN, {gates_s['fail_count']} FAIL")
    drift = packet["drift"]
    rh = packet["ruleset_health"]
    print(f"  Drift   : {drift['status']} "
          f"(top20={_v(drift['top20_overlap_pct'],'.1f','%')}, "
          f"top60={_v(drift['top60_overlap_pct'],'.1f','%')})")
    print(f"  RH      : {rh['status']} "
          f"(streak={rh['consecutive_warn_days']}d)")
    turn = packet["turnover"]
    print(f"  Turnover: {_v(turn['turnover_pct'],'.2f','%')} "
          f"(4w avg={_v(turn['trailing_4w_avg_pct'],'.2f','%')})")
    if actions:
        print(f"\n  ⚠  {len(actions)} action item(s):")
        for item in actions:
            print(f"    [{item['severity']}] {item['type']}: {item['detail']}")
    else:
        print("  ✅ No action required")
    print(f"\n  → {md_path}")
    print(f"  → {json_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
