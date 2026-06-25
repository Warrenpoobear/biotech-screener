#!/usr/bin/env python3
"""Freeze-lift forward evidence package (governance Tier 0).

Assembles Path C window status, forward-eval IC ledger, coinvest_score_z IC,
final_score IC (Spec 100), and optional coinvest shadow summary into a single
operator decision artifact. Does NOT modify production scoring or lift the freeze.

Requires explicit operator acknowledgment:
    export FREEZE_LIFT_ACK=1

Usage:
    python3 tools/forward_evidence_package.py --dry-run
    python3 tools/forward_evidence_package.py --write
    FREEZE_LIFT_ACK=1 python3 tools/forward_evidence_package.py --write --as-of-date 2026-06-24
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "artifacts" / "forward_evidence"
GOV_DIR = REPO / "artifacts" / "governance"
IC_LEDGER = REPO / "artifacts" / "forward_eval_ic_ledger.jsonl"
COINVEST_SHADOW_HISTORY = REPO / "artifacts" / "coinvest_shadow" / "history.csv"

PATH_C_WINDOW_END = "2026-06-03"
IC_FLOOR = 0.0200
SCHEMA = "forward_evidence_package.v1"
SHADOW_START = "2026-04-03"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _deterministic_timestamp(as_of_date: str) -> str:
    return f"{as_of_date}T00:00:00Z"


def require_freeze_lift_ack(*, dry_run: bool) -> None:
    if dry_run:
        return
    if os.environ.get("FREEZE_LIFT_ACK") != "1":
        raise SystemExit(
            "FREEZE_LIFT_ACK=1 required to write forward evidence artifacts.\n"
            "This package supports the freeze-lift decision; it does not lift the freeze.\n"
            "See docs/governance/FREEZE_LIFT_FORWARD_EVIDENCE_PACKAGE_2026_06_25.md"
        )


def read_ic_ledger(*, through_date: str | None = None) -> list[dict[str, Any]]:
    if not IC_LEDGER.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with IC_LEDGER.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            as_of = entry.get("as_of_date")
            if through_date and as_of and as_of > through_date:
                continue
            rows.append(entry)
    rows.sort(key=lambda r: r.get("as_of_date", ""))
    return rows


def check_ic_status(*, window_end: str) -> dict[str, Any]:
    """Path C / forward-eval IC observability through window_end."""
    observations = read_ic_ledger(through_date=window_end)
    if not observations:
        return {
            "observable": False,
            "reason": "IC ledger missing or empty through window end",
            "latest_date": None,
            "latest_ic": None,
            "status": "IC_UNOBSERVABLE",
            "observation_count": 0,
        }

    latest = observations[-1]
    latest_date = latest.get("as_of_date")
    latest_ic = latest.get("mean_ic")

    if latest_ic is None:
        return {
            "observable": False,
            "reason": "Forward eval gate cold start or horizons not yet filled",
            "latest_date": latest_date,
            "latest_ic": None,
            "status": "IC_UNOBSERVABLE",
            "observation_count": len(observations),
        }

    return {
        "observable": True,
        "reason": None,
        "latest_date": latest_date,
        "latest_ic": float(latest_ic),
        "status": "OBSERVABLE",
        "observation_count": len(observations),
        "above_floor": float(latest_ic) >= IC_FLOOR,
        "floor": IC_FLOOR,
    }


def path_c_close_decision(*, window_end: str = PATH_C_WINDOW_END) -> dict[str, Any]:
    ic_status = check_ic_status(window_end=window_end)
    if ic_status["observable"]:
        latest_ic = ic_status["latest_ic"]
        assert latest_ic is not None
        if latest_ic >= IC_FLOOR:
            decision = "PATH_C_VALID"
            action = "Override validated at window close; proceed to Path A design post-freeze."
        else:
            decision = "PATH_C_REVOKE"
            action = "Revert to HOLD; Path C override not supported by forward-eval IC."
    else:
        decision = "IC_UNOBSERVABLE"
        action = (
            "Operator must choose: extend observation window or revert to HOLD pending Path A."
        )

    return {
        "window_end": window_end,
        "ic_status": ic_status,
        "decision": decision,
        "action": action,
        "overdue_note": (
            f"Path C window closed {window_end}; no formal closure artifact was on file "
            "before this package run."
        ),
    }


def compute_signal_ic(
    *,
    score_field: str,
    start_date: str,
    end_date: str,
    snapshot_dir: Path,
    horizons: list[int],
) -> dict[str, Any]:
    from tools.measure_final_score_ic_spec100 import (
        discover_snapshots,
        load_snapshot,
        measure_final_score_ic,
    )

    snap_dates = discover_snapshots(snapshot_dir, start_date, end_date)
    if not snap_dates:
        return {
            "score_field": score_field,
            "status": "NO_SNAPSHOTS",
            "snapshot_count": 0,
            "horizons": {},
        }

    all_snapshots: dict[str, dict] = {}
    for snap_date in snap_dates:
        snap = load_snapshot(snapshot_dir / snap_date)
        if snap:
            all_snapshots[snap_date] = snap

    horizons_out: dict[str, Any] = {}
    for horizon in horizons:
        ic_values: list[float] = []
        for snap_date in snap_dates:
            snap = all_snapshots.get(snap_date)
            if not snap:
                continue
            result = measure_final_score_ic(snap, all_snapshots, horizon, score_field)
            if result and "error" not in result:
                ic = result.get("final_score_ic")
                if ic is not None and ic == ic:
                    ic_values.append(float(ic))

        mean_ic = statistics.mean(ic_values) if ic_values else None
        pct_positive = (
            sum(1 for x in ic_values if x > 0) / len(ic_values) if ic_values else None
        )
        horizons_out[f"T+{horizon}"] = {
            "n_observations": len(ic_values),
            "mean_ic": mean_ic,
            "pct_positive": pct_positive,
            "passes_floor_0_0200": mean_ic is not None and mean_ic >= IC_FLOOR,
        }

    primary = horizons_out.get(f"T+{horizons[0]}") if horizons else {}
    return {
        "score_field": score_field,
        "status": "OK" if primary.get("n_observations", 0) > 0 else "UNOBSERVABLE",
        "snapshot_count": len(all_snapshots),
        "start_date": start_date,
        "end_date": end_date,
        "horizons": horizons_out,
    }


def load_coinvest_shadow_summary() -> dict[str, Any]:
    if not COINVEST_SHADOW_HISTORY.is_file():
        return {"status": "MISSING", "path": str(COINVEST_SHADOW_HISTORY)}

    rows: list[dict[str, str]] = []
    with COINVEST_SHADOW_HISTORY.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {"status": "EMPTY", "path": str(COINVEST_SHADOW_HISTORY)}

    dates = sorted({r.get("as_of_date", "") for r in rows if r.get("as_of_date")})
    return {
        "status": "OK",
        "path": str(COINVEST_SHADOW_HISTORY),
        "row_count": len(rows),
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "shadow_start_policy": SHADOW_START,
    }


def advisory_verdict(
    *,
    coinvest_ic: dict[str, Any],
    final_score_ic: dict[str, Any],
    path_c: dict[str, Any],
) -> dict[str, Any]:
    """Advisory only — operator signs the governance memo."""

    def primary_mean(block: dict[str, Any]) -> float | None:
        horizons = block.get("horizons") or {}
        for key in ("T+20", "T+5", "T+10"):
            if key in horizons:
                return horizons[key].get("mean_ic")
        return None

    coinvest_mean = primary_mean(coinvest_ic)
    final_mean = primary_mean(final_score_ic)

    signals: list[str] = []
    if coinvest_mean is None and final_mean is None:
        verdict = "INSUFFICIENT_DATA"
        signals.append("No snapshot IC data on host for coinvest_score_z or final_score")
    elif (coinvest_mean or 0) >= IC_FLOOR and (final_mean or -1) >= IC_FLOOR:
        verdict = "POSITIVE"
        signals.append("Both coinvest_score_z and final_score mean IC >= 0.0200")
    elif (coinvest_mean or 0) < 0 and (final_mean or 0) < 0:
        verdict = "NEGATIVE"
        signals.append("Both signals negative at primary horizon — structural review warranted")
    else:
        verdict = "OBSERVE"
        if coinvest_mean is not None:
            signals.append(f"coinvest_score_z mean IC={coinvest_mean:.4f}")
        if final_mean is not None:
            signals.append(f"final_score mean IC={final_mean:.4f}")

    if path_c.get("decision") == "PATH_C_REVOKE":
        signals.append("Path C retrospective decision: REVOKE (IC below floor at window end)")
    elif path_c.get("decision") == "IC_UNOBSERVABLE":
        signals.append("Path C retrospective: IC unobservable at window close — operator choice required")

    return {
        "verdict": verdict,
        "advisory_only": True,
        "rationale": signals,
        "fork_positive": "Re-affirm coinvest-only selector thesis from forward evidence",
        "fork_negative": "Structural selector re-examination — do not tune plumbing",
    }


def build_package(
    *,
    as_of_date: str,
    snapshot_dir: Path,
    ic_start_date: str,
    horizons: list[int],
) -> dict[str, Any]:
    path_c_window = path_c_close_decision(window_end=PATH_C_WINDOW_END)
    path_c_current = path_c_close_decision(window_end=as_of_date)

    coinvest_ic = compute_signal_ic(
        score_field="coinvest_score_z",
        start_date=ic_start_date,
        end_date=as_of_date,
        snapshot_dir=snapshot_dir,
        horizons=horizons,
    )
    final_score_ic = compute_signal_ic(
        score_field="final_score",
        start_date=ic_start_date,
        end_date=as_of_date,
        snapshot_dir=snapshot_dir,
        horizons=horizons,
    )

    ledger_all = read_ic_ledger(through_date=as_of_date)
    advisory = advisory_verdict(
        coinvest_ic=coinvest_ic,
        final_score_ic=final_score_ic,
        path_c=path_c_window,
    )

    return {
        "schema": SCHEMA,
        "as_of_date": as_of_date,
        "generated_at": _deterministic_timestamp(as_of_date),
        "governance": {
            "freeze_lift_ack_required": True,
            "does_not_lift_freeze": True,
            "tier": 0,
            "memo": "docs/governance/FREEZE_LIFT_FORWARD_EVIDENCE_PACKAGE_2026_06_25.md",
        },
        "path_c": {
            "window_end": PATH_C_WINDOW_END,
            "retrospective_close": path_c_window,
            "status_through_as_of": path_c_current,
        },
        "forward_eval_ic_ledger": {
            "path": str(IC_LEDGER),
            "observation_count": len(ledger_all),
            "latest": ledger_all[-1] if ledger_all else None,
        },
        "coinvest_score_z_ic": coinvest_ic,
        "final_score_ic": final_score_ic,
        "coinvest_shadow": load_coinvest_shadow_summary(),
        "advisory_verdict": advisory,
        "prior_baseline": {
            "coinvest_pooled_ic": -0.031,
            "as_of": "2026-05-13",
            "prior_verdict": "OBSERVE",
            "note": "Pre-freeze-lift measurement; refresh supersedes when host data present",
        },
    }


def render_markdown(package: dict[str, Any]) -> str:
    adv = package["advisory_verdict"]
    path_c = package["path_c"]["retrospective_close"]
    lines = [
        "# Forward Evidence Package (Freeze-Lift Support)",
        "",
        f"**As-of:** {package['as_of_date']}",
        f"**Generated:** {package['generated_at']}",
        f"**Schema:** {package['schema']}",
        "",
        "> Advisory only. Does not lift the architecture freeze.",
        "",
        "## Advisory verdict",
        "",
        f"**{adv['verdict']}** — operator signs `docs/governance/FREEZE_LIFT_FORWARD_EVIDENCE_PACKAGE_2026_06_25.md`",
        "",
    ]
    for item in adv["rationale"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            f"- **If positive:** {adv['fork_positive']}",
            f"- **If negative:** {adv['fork_negative']}",
            "",
            "## Path C retrospective close",
            "",
            f"- Window end: {PATH_C_WINDOW_END}",
            f"- Decision: `{path_c['decision']}`",
            f"- Action: {path_c['action']}",
            f"- IC observable: {path_c['ic_status']['observable']}",
            f"- Latest IC: {path_c['ic_status'].get('latest_ic')}",
            "",
            "## Signal IC (host snapshots)",
            "",
        ]
    )

    for block_name in ("coinvest_score_z_ic", "final_score_ic"):
        block = package[block_name]
        lines.append(f"### {block['score_field']} ({block['status']})")
        for hkey, hval in (block.get("horizons") or {}).items():
            mean_ic = hval.get("mean_ic")
            mean_s = f"{mean_ic:.4f}" if mean_ic is not None else "n/a"
            lines.append(
                f"- {hkey}: n={hval.get('n_observations')}, mean_ic={mean_s}, "
                f"pct_positive={hval.get('pct_positive')}"
            )
        lines.append("")

    shadow = package["coinvest_shadow"]
    lines.extend(
        [
            "## Coinvest shadow",
            "",
            f"- Status: {shadow['status']}",
            f"- Path: `{shadow.get('path', '')}`",
            "",
            "## Operator next steps",
            "",
            "1. Review this artifact with the governance memo checklist",
            "2. If lifting freeze: document decision + date in governance memo",
            "3. Run Path A portfolio timing gate design (post-freeze)",
            "",
        ]
    )
    return "\n".join(lines)


def content_hash(package: dict[str, Any]) -> str:
    payload = json.dumps(package, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_package(package: dict[str, Any], *, as_of_date: str) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GOV_DIR.mkdir(parents=True, exist_ok=True)
    package["_governance"] = {
        "content_hash": content_hash(package),
        "as_of_date": as_of_date,
        "schema_version": SCHEMA,
    }
    json_path = OUT_DIR / f"{as_of_date}_package.json"
    md_path = OUT_DIR / f"{as_of_date}_package.md"
    json_path.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(package) + "\n", encoding="utf-8")

    path_c_path = GOV_DIR / f"path_c_window_close_{as_of_date}.json"
    path_c_path.write_text(
        json.dumps(package["path_c"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return json_path, md_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Freeze-lift forward evidence package")
    ap.add_argument("--as-of-date", default=date.today().isoformat())
    ap.add_argument("--snapshot-dir", type=Path, default=REPO / "data" / "snapshots")
    ap.add_argument("--ic-start-date", default=SHADOW_START)
    ap.add_argument("--horizons", default="20", help="Comma-separated forward horizons")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    package = build_package(
        as_of_date=args.as_of_date,
        snapshot_dir=args.snapshot_dir,
        ic_start_date=args.ic_start_date,
        horizons=horizons,
    )

    if args.json or args.dry_run:
        print(json.dumps(package, indent=2, sort_keys=True))
    if args.write:
        require_freeze_lift_ack(dry_run=False)
        json_path, md_path = write_package(package, as_of_date=args.as_of_date)
        print(f"Wrote {json_path}")
        print(f"Wrote {md_path}")
        print(f"Wrote {GOV_DIR / f'path_c_window_close_{args.as_of_date}.json'}")
    elif not args.json:
        print(render_markdown(package))

    return 0


if __name__ == "__main__":
    sys.exit(main())
