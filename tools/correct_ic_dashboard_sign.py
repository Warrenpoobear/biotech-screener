#!/usr/bin/env python3
"""Write sign-corrected copies of IC dashboard artifacts produced before the 2026-11-12 fix.

Until PR #652 (2026-11-12) ``tools/build_ic_dashboard.py`` treated ``score_rank_pct``
as lower-is-better, so every stored ``score_rank_pct`` IC, hit rate and health label
in ``artifacts/ic_dashboard/`` from before the fix has the wrong sign. The governance
memo (docs/governance/2026-11-12-ic-dashboard-sign-correction.md §4) keeps the
originals as written; this tool writes corrected copies to
``artifacts/ic_dashboard/sign_corrected/`` and never touches the originals.

Only the sign is corrected. The pre-fix dashboards also read unadjusted prices and,
for 2026-08-18 → 2026-10-14, regenerated (not live) snapshots; those defects remain
in the corrected copies, so treat them as descriptive history, not evidence.

A dashboard is pre-fix when it has no ``actionable_rank`` signal (added by the same
fix). Other signals were never negated and are copied unchanged. Health and
attention are re-derived from the corrected values with the dashboard's own
thresholds; INSUFFICIENT_DATES / NO_DATA labels do not depend on sign and are kept.

Usage:
    python tools/correct_ic_dashboard_sign.py [--dir artifacts/ic_dashboard]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.build_ic_dashboard import IC_ALERT, IC_HEALTHY, IC_WARN  # noqa: E402

FIELD = "score_rank_pct"
POST_FIX_MARKER = "actionable_rank"
OUT_SUBDIR = "sign_corrected"
SIGN_LABEL_INDEPENDENT = {"INSUFFICIENT_DATES", "NO_DATA"}
CORRECTION_NOTE = {
    "fields": [FIELD],
    "reason": "pre-2026-11-12 dashboard negated score_rank_pct (higher is better)",
    "memo": "docs/governance/2026-11-12-ic-dashboard-sign-correction.md",
    "caveat": "sign only; unadjusted prices and regenerated snapshots in the original are not corrected",
}


def _neg(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(-x, 4) + 0.0


def _health(mean_ic: Optional[float], original: Optional[str]) -> Optional[str]:
    if original in SIGN_LABEL_INDEPENDENT or mean_ic is None:
        return original
    if mean_ic >= IC_HEALTHY:
        return "HEALTHY"
    if mean_ic >= IC_WARN:
        return "WEAK"
    if mean_ic >= IC_ALERT:
        return "WARN"
    return "ALERT"


def _attention(healths, any_decaying: bool = False) -> str:
    if "ALERT" in healths:
        return "HIGH"
    if "WARN" in healths or any_decaying:
        return "MEDIUM"
    return "LOW"


def is_pre_fix_dashboard(d: Dict[str, Any]) -> bool:
    return POST_FIX_MARKER not in (d.get("signals") or {})


def correct_dashboard(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a corrected copy of one pre-fix dashboard dict."""
    out = copy.deepcopy(d)
    sig = (out.get("signals") or {}).get(FIELD)
    if isinstance(sig, dict):
        sig["mean_ic"] = _neg(sig.get("mean_ic"))
        sig["latest_ic"] = _neg(sig.get("latest_ic"))
        if "recent_ic" in sig:
            sig["recent_ic"] = _neg(sig.get("recent_ic"))
        if sig.get("hit_rate") is not None:
            sig["hit_rate"] = round(1.0 - sig["hit_rate"], 4) + 0.0
        for row in sig.get("per_date") or []:
            row["ic"] = _neg(row.get("ic"))
        sig["health"] = _health(sig["mean_ic"], sig.get("health"))
        if sig.get("trend") in ("decaying", "stable") and sig.get("recent_ic") is not None:
            decaying = sig["mean_ic"] >= IC_HEALTHY and sig["recent_ic"] <= IC_WARN
            sig["trend"] = "decaying" if decaying else "stable"
    signals = (out.get("signals") or {}).values()
    out["attention"] = _attention(
        [s.get("health") for s in signals if isinstance(s, dict)],
        any(isinstance(s, dict) and s.get("trend") == "decaying" for s in signals),
    )
    out["sign_correction"] = dict(CORRECTION_NOTE)
    return out


def correct_history_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return a corrected copy of one ``history.jsonl`` row (flat ``<signal>_ic`` keys)."""
    out = dict(row)
    if f"{POST_FIX_MARKER}_ic" in row:
        out["sign_corrected"] = False
        return out
    key = f"{FIELD}_ic"
    if key in out:
        out[key] = _neg(out[key])
        out[f"{FIELD}_health"] = _health(out[key], out.get(f"{FIELD}_health"))
        out["attention"] = _attention([v for k, v in out.items() if k.endswith("_health")])
    out["sign_corrected"] = True
    return out


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def write_corrected_copies(src_dir: Path) -> Dict[str, Any]:
    """Write corrected copies of every pre-fix dashboard and of history.jsonl."""
    out_dir = src_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    n_dash = 0
    for path in sorted(src_dir.glob("*_dashboard.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if not is_pre_fix_dashboard(d):
            continue
        (out_dir / path.name).write_text(_dump(correct_dashboard(d)), encoding="utf-8", newline="\n")
        n_dash += 1
    n_rows = n_flipped = 0
    hist = src_dir / "history.jsonl"
    if hist.exists():
        lines = []
        for line in hist.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = correct_history_row(json.loads(line))
            n_rows += 1
            n_flipped += int(row["sign_corrected"])
            lines.append(json.dumps(row, ensure_ascii=False))
        (out_dir / "history.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    summary = {
        "dashboards_corrected": n_dash,
        "history_rows": n_rows,
        "history_rows_corrected": n_flipped,
        "out_dir": str(out_dir),
        **CORRECTION_NOTE,
    }
    (out_dir / "README.json").write_text(_dump(summary), encoding="utf-8", newline="\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", type=Path, default=REPO_ROOT / "artifacts" / "ic_dashboard")
    args = ap.parse_args()
    print(json.dumps(write_corrected_copies(args.dir), indent=2))


if __name__ == "__main__":
    main()
