#!/usr/bin/env python3
"""Eligibility Drift Gate (v1).

Reads eligibility_delta.json (or builds it) and returns PASS/WARN/FAIL.

Exit semantics:
  0 = PASS or WARN (non-blocking, unless --fail-on-warn)
  1 = FAIL (or WARN with --fail-on-warn)

GitHub Actions friendly:
  - prints ::warning:: / ::error:: annotations when $GITHUB_ACTIONS=true
  - appends markdown to $GITHUB_STEP_SUMMARY when set

Usage:
    python3 scripts/eval_eligibility_gate.py --as-of-date 2026-02-27
    python3 scripts/eval_eligibility_gate.py --delta-json path/to/eligibility_delta.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

VERSION = "1.0.0"
SCHEMA = "eligibility_gate.v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"


def _gh_annot(level: str, msg: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{level}::{msg}")
    else:
        print(f"[{level.upper()}] {msg}")


def _append_step_summary(md: str) -> None:
    p = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if not p:
        return
    try:
        with open(p, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

def evaluate_eligibility_gate(
    delta: Dict[str, Any],
    *,
    warn_pct: float = 0.03,
    fail_pct: float = 0.06,
    warn_abs: int = 10,
    fail_abs: int = 25,
) -> Tuple[str, List[str]]:
    """Evaluate eligibility delta against thresholds.

    Returns (verdict, trigger_reasons).
    """
    d = delta.get("delta", {})
    reasons = d.get("reasons", {}) if isinstance(d.get("reasons"), dict) else {}

    triggers: List[str] = []
    verdict = "PASS"

    pct = float(d.get("pct_ineligible", 0.0))
    dinel = int(d.get("ineligible", 0))

    if abs(pct) >= fail_pct:
        verdict = "FAIL"
        triggers.append(f"pct_ineligible_delta={pct:+.4f} >= {fail_pct:.4f}")
    elif abs(pct) >= warn_pct:
        verdict = "WARN"
        triggers.append(f"pct_ineligible_delta={pct:+.4f} >= {warn_pct:.4f}")

    if abs(dinel) >= fail_abs:
        verdict = "FAIL"
        triggers.append(f"ineligible_delta={dinel:+d} >= {fail_abs}")
    elif verdict != "FAIL" and abs(dinel) >= warn_abs:
        if verdict != "FAIL":
            verdict = "WARN"
        triggers.append(f"ineligible_delta={dinel:+d} >= {warn_abs}")

    # Annotate top reason spikes (informational, doesn't escalate verdict)
    big = [(k, int(v)) for k, v in reasons.items() if int(v) != 0]
    big.sort(key=lambda kv: (-abs(kv[1]), kv[0]))
    if big:
        topk = ", ".join(f"{k}:{v:+d}" for k, v in big[:5])
        triggers.append(f"top_reason_deltas={topk}")

    return verdict, triggers


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Eligibility drift gate.")
    ap.add_argument("--as-of-date", default="", help="Build delta first, then evaluate.")
    ap.add_argument("--snapshot-dir", default="data/snapshots")
    ap.add_argument("--delta-json", default="", help="Path to pre-built eligibility_delta.json.")
    ap.add_argument("--prior-date", default="")
    ap.add_argument("--include-degraded", action="store_true")
    ap.add_argument("--warn-pct-delta", type=float, default=0.03)
    ap.add_argument("--fail-pct-delta", type=float, default=0.06)
    ap.add_argument("--warn-ineligible-abs", type=int, default=10)
    ap.add_argument("--fail-ineligible-abs", type=int, default=25)
    ap.add_argument("--fail-on-warn", action="store_true")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    if not args.delta_json and not args.as_of_date:
        raise SystemExit("Provide either --delta-json or --as-of-date.")

    delta_path: Path
    if args.as_of_date:
        from scripts.build_eligibility_delta import (
            build_delta,
            pick_prior_date,
            _read_json,
            _is_degraded,
        )

        as_of = args.as_of_date.strip()
        snapshot_root = Path(args.snapshot_dir)
        snap_dir = snapshot_root / as_of
        cur_path = snap_dir / "eligibility_summary.json"
        if not cur_path.exists():
            raise SystemExit(f"Missing: {cur_path}")

        prior = args.prior_date.strip()
        if not prior:
            prior = pick_prior_date(
                snapshot_root, as_of, include_degraded=args.include_degraded,
            )
            if not prior:
                raise SystemExit("Could not auto-select prior date.")
        prior_dir = snapshot_root / prior
        if not (prior_dir / "eligibility_summary.json").exists():
            raise SystemExit(f"Missing: {prior_dir / 'eligibility_summary.json'}")

        cur_raw = _read_json(cur_path)
        pri_raw = _read_json(prior_dir / "eligibility_summary.json")

        delta_obj = build_delta(
            cur_raw if isinstance(cur_raw, dict) else {},
            pri_raw if isinstance(pri_raw, dict) else {},
            as_of,
            prior,
            prior_degraded=_is_degraded(prior_dir),
        )

        # Persist delta sidecar
        delta_path = snap_dir / "eligibility_delta.json"
        delta_path.write_text(_stable_json(delta_obj), encoding="utf-8")

        from scripts.build_eligibility_delta import render_delta_md
        md_path = snap_dir / "eligibility_delta.md"
        md_path.write_text(render_delta_md(delta_obj) + "\n", encoding="utf-8")
    else:
        delta_path = Path(args.delta_json)
        if not delta_path.exists():
            raise SystemExit(f"Delta JSON not found: {delta_path}")
        delta_obj = json.loads(delta_path.read_text(encoding="utf-8"))

    verdict, triggers = evaluate_eligibility_gate(
        delta_obj,
        warn_pct=args.warn_pct_delta,
        fail_pct=args.fail_pct_delta,
        warn_abs=args.warn_ineligible_abs,
        fail_abs=args.fail_ineligible_abs,
    )

    gate_out = {
        "schema": SCHEMA,
        "verdict": verdict,
        "delta_path": str(delta_path),
        "thresholds": {
            "warn_pct_delta": args.warn_pct_delta,
            "fail_pct_delta": args.fail_pct_delta,
            "warn_ineligible_abs": args.warn_ineligible_abs,
            "fail_ineligible_abs": args.fail_ineligible_abs,
        },
        "triggers": triggers,
        "as_of_date": delta_obj.get("as_of_date"),
        "prior_date": delta_obj.get("prior_date"),
        "metrics": {
            "eligible_delta": int(delta_obj.get("delta", {}).get("eligible", 0)),
            "ineligible_delta": int(delta_obj.get("delta", {}).get("ineligible", 0)),
            "pct_ineligible_delta": float(delta_obj.get("delta", {}).get("pct_ineligible", 0.0)),
        },
    }

    # GH annotations
    if verdict == "FAIL":
        _gh_annot("error", f"ELIGIBILITY GATE FAIL -- {', '.join(triggers[:3])}")
    elif verdict == "WARN":
        _gh_annot("warning", f"ELIGIBILITY GATE WARN -- {', '.join(triggers[:3])}")
    else:
        print("ELIGIBILITY GATE PASS")

    # Step summary
    md_lines = [
        "## Eligibility Gate", "",
        "| Field | Value |", "|---|---|",
        f"| verdict | **{verdict}** |",
        f"| as_of_date | `{gate_out['as_of_date']}` |",
        f"| prior_date | `{gate_out['prior_date']}` |",
        f"| ineligible_delta | `{gate_out['metrics']['ineligible_delta']:+d}` |",
        f"| pct_ineligible_delta | `{gate_out['metrics']['pct_ineligible_delta']:+.4f}` |",
    ]
    if triggers:
        md_lines += ["", "**Triggers**"]
        md_lines += [f"- {t}" for t in triggers[:8]]
    _append_step_summary("\n".join(md_lines) + "\n")

    # Write gate json
    out_path = Path(args.out_json) if args.out_json else delta_path.parent / "eligibility_gate.json"
    out_path.write_text(_stable_json(gate_out), encoding="utf-8")
    print(f"Wrote: {out_path}")

    if verdict == "FAIL":
        return 1
    if verdict == "WARN" and args.fail_on_warn:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
