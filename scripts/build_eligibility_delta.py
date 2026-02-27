#!/usr/bin/env python3
"""Build Eligibility Delta (v1).

Compares eligibility_summary.json between as_of_date and a chosen prior date
and writes deterministic:
  - eligibility_delta.json
  - eligibility_delta.md

Intended use:
  - daily CI (phase2-daily-production) after run_screen writes eligibility_summary.json
  - promotion gate (ruleset-promotion-gate) to quantify eligibility drift

Schema-drift tolerant: tries canonical v1 keys first, then common alternatives.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

VERSION = "1.0.0"
SCHEMA = "eligibility_delta.v1"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, bool):
            return int(v)
        return int(float(str(v).strip() or default))
    except Exception:
        return default


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"


# ---------------------------------------------------------------------------
# Summary normalisation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormSummary:
    as_of_date: str
    total: int
    eligible: int
    ineligible: int
    pct_ineligible: float
    reasons: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of_date": self.as_of_date,
            "total": self.total,
            "eligible": self.eligible,
            "ineligible": self.ineligible,
            "pct_ineligible": round(self.pct_ineligible, 6),
            "reasons": dict(sorted(self.reasons.items())),
        }


def normalize_summary(raw: Dict[str, Any], as_of_date: str) -> NormSummary:
    """Normalize eligibility_summary.json (v1 or alternatives) to a stable shape."""
    # Canonical v1 keys (written by run_screen.py)
    total = _safe_int(raw.get("n_total") or raw.get("total"))
    eligible = _safe_int(raw.get("n_eligible") or raw.get("eligible"))
    ineligible = _safe_int(raw.get("n_ineligible") or raw.get("ineligible"))

    if total == 0 and (eligible or ineligible):
        total = eligible + ineligible

    pct_ineligible = ineligible / total if total > 0 else 0.0

    # Reason counts — try canonical v1 key first
    reasons: Dict[str, int] = {}
    for key in ("counts_by_reason", "reason_counts", "by_reason", "reasons"):
        val = raw.get(key)
        if isinstance(val, dict):
            for rk, rv in val.items():
                reasons[str(rk)] = _safe_int(rv)
            break

    return NormSummary(
        as_of_date=as_of_date,
        total=total,
        eligible=eligible,
        ineligible=ineligible,
        pct_ineligible=pct_ineligible,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Snapshot discovery
# ---------------------------------------------------------------------------

def _is_degraded(snapshot_dir: Path) -> bool:
    health = snapshot_dir / "cache_health.json"
    if not health.exists():
        return False
    try:
        h = _read_json(health)
        if isinstance(h, dict):
            return (
                bool(h.get("degraded_run", False))
                or str(h.get("overall_status", "")).lower() in ("bad", "degraded")
            )
    except Exception:
        pass
    return False


def list_snapshot_dates(snapshot_root: Path) -> List[str]:
    dates: List[str] = []
    if not snapshot_root.exists():
        return dates
    for p in snapshot_root.iterdir():
        if p.is_dir() and DATE_RE.match(p.name):
            dates.append(p.name)
    dates.sort()
    return dates


def pick_prior_date(
    snapshot_root: Path,
    as_of: str,
    *,
    include_degraded: bool = False,
) -> Optional[str]:
    all_dates = list_snapshot_dates(snapshot_root)
    if as_of not in all_dates:
        return None
    idx = all_dates.index(as_of)
    for j in range(idx - 1, -1, -1):
        d = all_dates[j]
        sd = snapshot_root / d
        if not include_degraded and _is_degraded(sd):
            continue
        if (sd / "eligibility_summary.json").exists():
            return d
    return None


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def _reason_delta(cur: Dict[str, int], pri: Dict[str, int]) -> Dict[str, int]:
    keys = set(cur) | set(pri)
    out = {k: _safe_int(cur.get(k)) - _safe_int(pri.get(k)) for k in keys}
    return dict(sorted(out.items(), key=lambda kv: (-abs(kv[1]), kv[0])))


def build_delta(
    cur_raw: Dict[str, Any],
    pri_raw: Dict[str, Any],
    as_of: str,
    prior: str,
    *,
    prior_degraded: bool = False,
) -> Dict[str, Any]:
    """Build a delta dict from two raw eligibility summary dicts."""
    cur = normalize_summary(cur_raw, as_of)
    pri = normalize_summary(pri_raw, prior)

    delta = {
        "eligible": cur.eligible - pri.eligible,
        "ineligible": cur.ineligible - pri.ineligible,
        "pct_ineligible": round(cur.pct_ineligible - pri.pct_ineligible, 6),
        "reasons": _reason_delta(cur.reasons, pri.reasons),
    }

    notes: List[str] = []
    if pri.total and cur.total and pri.total != cur.total:
        notes.append(f"total changed: {pri.total} -> {cur.total}")
    if prior_degraded:
        notes.append(f"prior snapshot was degraded: {prior}")

    return {
        "schema": SCHEMA,
        "as_of_date": as_of,
        "prior_date": prior,
        "current": cur.to_dict(),
        "prior": pri.to_dict(),
        "delta": delta,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def render_delta_md(delta_obj: Dict[str, Any]) -> str:
    cur = delta_obj["current"]
    pri = delta_obj["prior"]
    d = delta_obj["delta"]

    lines: List[str] = [
        f"# Eligibility Delta -- {delta_obj['as_of_date']} vs {delta_obj['prior_date']}",
        "",
        "| Field | Prior | Current | Delta |",
        "|---|---:|---:|---:|",
        f"| eligible | {pri['eligible']} | {cur['eligible']} | {d['eligible']:+d} |",
        f"| ineligible | {pri['ineligible']} | {cur['ineligible']} | {d['ineligible']:+d} |",
        f"| pct_ineligible | {pri['pct_ineligible']:.4f} | {cur['pct_ineligible']:.4f} | {d['pct_ineligible']:+.4f} |",
        "",
    ]

    reasons = [(k, v) for k, v in d["reasons"].items() if v != 0]
    if reasons:
        lines.append("## Reason deltas (non-zero)")
        lines.append("")
        lines.append("| Reason | Delta |")
        lines.append("|---|---:|")
        for k, v in reasons[:25]:
            lines.append(f"| {k} | {v:+d} |")
        lines.append("")
    else:
        lines.append("_No non-zero reason deltas._")
        lines.append("")

    if delta_obj.get("notes"):
        lines.append("## Notes")
        lines.append("")
        for n in delta_obj["notes"]:
            lines.append(f"- {n}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Build eligibility delta artifact.")
    ap.add_argument("--as-of-date", required=True)
    ap.add_argument("--snapshot-dir", default="data/snapshots")
    ap.add_argument("--prior-date", default="")
    ap.add_argument("--include-degraded", action="store_true")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    as_of = args.as_of_date.strip()
    snapshot_root = Path(args.snapshot_dir)
    snap_dir = snapshot_root / as_of
    if not snap_dir.exists():
        raise SystemExit(f"Snapshot dir not found: {snap_dir}")

    cur_path = snap_dir / "eligibility_summary.json"
    if not cur_path.exists():
        raise SystemExit(f"Missing eligibility summary: {cur_path}")

    prior = args.prior_date.strip()
    if prior:
        if not DATE_RE.match(prior):
            raise SystemExit(f"Invalid --prior-date: {prior}")
        prior_dir = snapshot_root / prior
        if not (prior_dir / "eligibility_summary.json").exists():
            raise SystemExit(f"Missing prior eligibility summary: {prior_dir / 'eligibility_summary.json'}")
    else:
        prior = pick_prior_date(snapshot_root, as_of, include_degraded=args.include_degraded)
        if not prior:
            raise SystemExit("Could not auto-select prior date (no suitable eligibility_summary.json found).")
        prior_dir = snapshot_root / prior

    cur_raw = _read_json(cur_path)
    pri_raw = _read_json(prior_dir / "eligibility_summary.json")

    out = build_delta(
        cur_raw if isinstance(cur_raw, dict) else {},
        pri_raw if isinstance(pri_raw, dict) else {},
        as_of,
        prior,
        prior_degraded=_is_degraded(prior_dir),
    )

    out_json = Path(args.out_json) if args.out_json else (snap_dir / "eligibility_delta.json")
    out_md = Path(args.out_md) if args.out_md else (snap_dir / "eligibility_delta.md")

    out_json.write_text(_stable_json(out), encoding="utf-8")
    out_md.write_text(render_delta_md(out) + "\n", encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")
    d = out["delta"]
    print(f"Prior: {prior}  Eligible d={d['eligible']:+d}  Ineligible d={d['ineligible']:+d}  pct d={d['pct_ineligible']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
