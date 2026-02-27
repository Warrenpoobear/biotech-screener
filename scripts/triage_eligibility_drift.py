#!/usr/bin/env python3
"""Eligibility Drift Triage (v1).

Given two snapshots (as_of_date + prior_date), produce a deterministic triage
bundle identifying which tickers changed eligibility and why.

Outputs (to out_dir):
  - context.json
  - triage_summary.json
  - triage_summary.md
  - movers.csv

"Movers" are tickers with eligibility flips (1->0 or 0->1).
Optionally includes reason-change-only movers (--include-reason-only).

Uses rankings.csv fields: ticker, eligible, ineligible_reasons (pipe-separated).
"""
from __future__ import annotations

import argparse
import csv
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
SCHEMA = "eligibility_triage.v1"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv_rows(path: Path) -> Dict[str, Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        out: Dict[str, Dict[str, str]] = {}
        for row in csv.DictReader(f):
            t = (row.get("ticker") or "").strip().upper()
            if not t:
                continue
            out[t] = {k: (v if v is not None else "") for k, v in row.items()}
        return out


def _split_reasons(s: str) -> List[str]:
    """Split pipe-separated reasons into sorted, deduped list."""
    parts = [p.strip() for p in (s or "").split("|") if p.strip()]
    return sorted(set(parts))


def _safe_bool01(v: str) -> Optional[int]:
    s = (v or "").strip()
    if s in ("1", "1.0", "True", "true"):
        return 1
    if s in ("0", "0.0", "False", "false"):
        return 0
    return None


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"


# ---------------------------------------------------------------------------
# Mover detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Mover:
    ticker: str
    prior_eligible: Optional[int]
    cur_eligible: Optional[int]
    prior_reasons: List[str]
    cur_reasons: List[str]
    archetype: str
    tier_any: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "prior_eligible": self.prior_eligible,
            "cur_eligible": self.cur_eligible,
            "prior_reasons": self.prior_reasons,
            "cur_reasons": self.cur_reasons,
            "archetype": self.archetype,
            "tier_any": self.tier_any,
        }


def find_movers(
    cur: Dict[str, Dict[str, str]],
    pri: Dict[str, Dict[str, str]],
    *,
    include_reason_only: bool = False,
    max_tickers: int = 50,
) -> List[Mover]:
    """Identify tickers with eligibility changes between two snapshots."""
    tickers = sorted(set(cur) | set(pri))
    movers: List[Mover] = []

    for t in tickers:
        cr = cur.get(t, {})
        pr = pri.get(t, {})

        ce = _safe_bool01(cr.get("eligible", ""))
        pe = _safe_bool01(pr.get("eligible", ""))
        crs = _split_reasons(cr.get("ineligible_reasons", ""))
        prs = _split_reasons(pr.get("ineligible_reasons", ""))

        archetype = (cr.get("archetype") or pr.get("archetype") or "").strip()
        tier_any = (cr.get("tier_any") or pr.get("tier_any") or "").strip()

        flip = pe is not None and ce is not None and pe != ce
        reason_change = prs != crs

        if flip or (include_reason_only and reason_change):
            movers.append(Mover(
                ticker=t,
                prior_eligible=pe,
                cur_eligible=ce,
                prior_reasons=prs,
                cur_reasons=crs,
                archetype=archetype,
                tier_any=tier_any,
            ))

    # Deterministic ordering: flips (newly ineligible first), then ticker
    movers.sort(key=lambda m: (
        0 if (m.prior_eligible == 1 and m.cur_eligible == 0) else
        1 if (m.prior_eligible == 0 and m.cur_eligible == 1) else 2,
        m.ticker,
    ))
    return movers[:max(0, max_tickers)]


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, movers: List[Mover]) -> None:
    fields = ["ticker", "prior_eligible", "cur_eligible",
              "prior_reasons", "cur_reasons", "archetype", "tier_any"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in movers:
            w.writerow({
                "ticker": m.ticker,
                "prior_eligible": "" if m.prior_eligible is None else m.prior_eligible,
                "cur_eligible": "" if m.cur_eligible is None else m.cur_eligible,
                "prior_reasons": "|".join(m.prior_reasons),
                "cur_reasons": "|".join(m.cur_reasons),
                "archetype": m.archetype,
                "tier_any": m.tier_any,
            })


def render_triage_md(ctx: Dict[str, Any], movers: List[Mover]) -> str:
    lines: List[str] = [
        f"# Eligibility Triage -- {ctx['as_of_date']} vs {ctx['prior_date']}",
        "",
        f"- movers: **{len(movers)}**",
        "",
    ]
    if not movers:
        lines.append("_No eligibility changes detected._")
        lines.append("")
        return "\n".join(lines)

    new_inel = [m for m in movers if m.prior_eligible == 1 and m.cur_eligible == 0]
    new_el = [m for m in movers if m.prior_eligible == 0 and m.cur_eligible == 1]
    reason_only = [m for m in movers
                   if m.prior_eligible == m.cur_eligible and m.prior_reasons != m.cur_reasons]

    lines += [
        "| Bucket | Count |", "|---|---:|",
        f"| newly_ineligible (1->0) | {len(new_inel)} |",
        f"| newly_eligible (0->1) | {len(new_el)} |",
        f"| reason_change_only | {len(reason_only)} |",
        "",
    ]

    def _table(title: str, items: List[Mover]) -> None:
        if not items:
            return
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Ticker | Prior reasons | Current reasons | Archetype | Tier |")
        lines.append("|---|---|---|---|---|")
        for m in items[:15]:
            lines.append(
                f"| {m.ticker} | `{'|'.join(m.prior_reasons)}` "
                f"| `{'|'.join(m.cur_reasons)}` | {m.archetype} | {m.tier_any} |"
            )
        lines.append("")

    _table("Newly ineligible", new_inel)
    _table("Newly eligible", new_el)
    _table("Reason changes (no flip)", reason_only)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _pick_prior_date(snapshot_root: Path, as_of: str) -> Optional[str]:
    dates: List[str] = []
    for p in snapshot_root.iterdir():
        if p.is_dir() and DATE_RE.match(p.name):
            dates.append(p.name)
    dates.sort()
    if as_of not in dates:
        return None
    i = dates.index(as_of)
    return dates[i - 1] if i > 0 else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Triage eligibility drift between snapshots.")
    ap.add_argument("--as-of-date", required=True)
    ap.add_argument("--snapshot-dir", default="data/snapshots")
    ap.add_argument("--prior-date", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--include-reason-only", action="store_true")
    ap.add_argument("--max-tickers", type=int, default=50)
    args = ap.parse_args()

    root = Path(args.snapshot_dir)
    as_of = args.as_of_date.strip()
    if not (root / as_of).exists():
        raise SystemExit(f"Missing snapshot dir: {root / as_of}")

    prior = args.prior_date.strip()
    if not prior:
        prior = _pick_prior_date(root, as_of) or ""
    if not prior:
        raise SystemExit("Could not determine prior date (set --prior-date).")
    if not (root / prior).exists():
        raise SystemExit(f"Missing prior snapshot dir: {root / prior}")

    cur_rank = root / as_of / "rankings.csv"
    pri_rank = root / prior / "rankings.csv"
    if not cur_rank.exists():
        raise SystemExit(f"Missing: {cur_rank}")
    if not pri_rank.exists():
        raise SystemExit(f"Missing: {pri_rank}")

    cur = _load_csv_rows(cur_rank)
    pri = _load_csv_rows(pri_rank)

    movers = find_movers(
        cur, pri,
        include_reason_only=args.include_reason_only,
        max_tickers=args.max_tickers,
    )

    out_dir = Path(args.out_dir) if args.out_dir else (root / as_of / "eligibility_triage")
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = {
        "schema": "eligibility_triage_context.v1",
        "as_of_date": as_of,
        "prior_date": prior,
        "snapshot_dir": str(root),
        "max_tickers": args.max_tickers,
        "include_reason_only": bool(args.include_reason_only),
    }

    (out_dir / "context.json").write_text(_stable_json(ctx), encoding="utf-8")

    payload = {
        "schema": SCHEMA,
        "context": ctx,
        "movers": [m.to_dict() for m in movers],
    }
    (out_dir / "triage_summary.json").write_text(_stable_json(payload), encoding="utf-8")
    (out_dir / "triage_summary.md").write_text(
        render_triage_md(ctx, movers) + "\n", encoding="utf-8",
    )
    _write_csv(out_dir / "movers.csv", movers)

    print(f"Triage complete: {out_dir}  movers={len(movers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
