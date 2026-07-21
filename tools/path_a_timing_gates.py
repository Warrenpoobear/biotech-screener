#!/usr/bin/env python3
"""Path A portfolio timing gates (Spec 106) — shadow overlay.

Post-selection construction overlay: T0 cap (catalyst_days <= 7) and T3+ floor
(catalyst_days >= 90 or no-catalyst). Does not modify selector/ranker/final_score.

Usage:
    python3 tools/path_a_timing_gates.py --dry-run --policy production_data/portfolio_policy_path_a_shadow.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO / "artifacts" / "portfolio_construction"
MANIFEST_SCHEMA = "path_a_manifest.v1"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_TWOPLACES = Decimal("0.01")
_FOURPLACES = Decimal("0.0001")


def _d(value: Any, default: str = "0") -> Decimal:
    if value is None or value == "":
        return Decimal(default)
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal(default)


def _rank(row: dict[str, Any]) -> int:
    try:
        return int(float(row.get("actionable_rank", "9999")))
    except (TypeError, ValueError):
        return 9999


def parse_catalyst_days(value: Any) -> int | None:
    if value is None or str(value).strip() in ("", "nan", "None"):
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def classify_path_a_zone(catalyst_days: int | None, catalyst_mode: str) -> str:
    """Classify a name into Path A zone T0..T4 (Spec 106 §4.1)."""
    mode = (catalyst_mode or "").strip().lower()
    if mode == "blended_window":
        return "T4"
    if mode in {"no_upcoming", "missing"}:
        return "T4"
    if catalyst_days is None:
        return "T4"
    if catalyst_days <= 0:
        return "T4"
    if catalyst_days <= 7:
        return "T0"
    if catalyst_days <= 30:
        return "T1"
    if catalyst_days <= 89:
        return "T2"
    if catalyst_days <= 180:
        return "T3"
    return "T4"


def zone_map_from_rows(rows: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        ticker = (row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        days = parse_catalyst_days(row.get("catalyst_days"))
        mode = row.get("catalyst_mode") or ""
        out[ticker] = classify_path_a_zone(days, mode)
    return out


@dataclass(frozen=True)
class PathATimingGates:
    enabled: bool = False
    t0_max_days: int = 7
    t0_max_weight_pct: Decimal = Decimal("30.0")
    t3_min_days: int = 90
    t3_plus_min_weight_pct: Decimal = Decimal("40.0")
    t3_plus_zones: frozenset[str] = frozenset({"T3", "T4"})
    infeasibility_relax_order: tuple[str, ...] = ("t3_plus_floor",)


@dataclass
class PathAResult:
    positions: list[dict[str, Any]]
    t0_weight_pct: Decimal
    t3_plus_weight_pct: Decimal
    compliant: bool
    swaps: list[dict[str, Any]] = field(default_factory=list)
    infeasibility: dict[str, Any] | None = None
    floor_relaxed: bool = False

    def gates_summary(self, gates: PathATimingGates) -> dict[str, Any]:
        return {
            "t0_weight_pct": float(self.t0_weight_pct),
            "t0_cap_pct": float(gates.t0_max_weight_pct),
            "t0_compliant": self.t0_weight_pct <= gates.t0_max_weight_pct,
            "t3_plus_weight_pct": float(self.t3_plus_weight_pct),
            "t3_plus_floor_pct": float(gates.t3_plus_min_weight_pct),
            "t3_plus_compliant": self.t3_plus_weight_pct >= gates.t3_plus_min_weight_pct,
        }


def gates_from_policy(policy: dict[str, Any]) -> PathATimingGates:
    block = policy.get("path_a_timing_gates") or {}
    t0 = block.get("t0_imminent") or {}
    t3 = block.get("t3_plus") or {}
    include = block.get("include_zones") or t3.get("include_zones") or ["T3", "T4"]
    relax = block.get("infeasibility_relax_order") or ["t3_plus_floor"]
    return PathATimingGates(
        enabled=bool(block.get("enabled")),
        t0_max_days=int(t0.get("max_days", 7)),
        t0_max_weight_pct=_d(t0.get("max_weight_pct", "30.0")),
        t3_min_days=int(t3.get("min_days", 90)),
        t3_plus_min_weight_pct=_d(t3.get("min_weight_pct", "40.0")),
        t3_plus_zones=frozenset(str(z) for z in include),
        infeasibility_relax_order=tuple(str(x) for x in relax),
    )


def measure_zone_weights(
    positions: list[dict[str, Any]],
    zones: dict[str, str],
    *,
    t0_zone: str = "T0",
    t3_plus_zones: frozenset[str],
) -> tuple[Decimal, Decimal]:
    t0 = Decimal("0")
    t3p = Decimal("0")
    for pos in positions:
        ticker = (pos.get("ticker") or "").strip().upper()
        wt = _d(pos.get("weight_pct", "0"))
        zone = zones.get(ticker, pos.get("path_a_zone", "T4"))
        if zone == t0_zone:
            t0 += wt
        if zone in t3_plus_zones:
            t3p += wt
    return t0.quantize(_FOURPLACES, ROUND_HALF_UP), t3p.quantize(_FOURPLACES, ROUND_HALF_UP)


def _equal_reweight(positions: list[dict[str, Any]], account_usd: Decimal) -> None:
    if not positions:
        return
    n = Decimal(len(positions))
    wt = (Decimal("100") / n).quantize(_FOURPLACES, ROUND_HALF_UP)
    for pos in positions:
        pos["weight_pct"] = float(wt)
        pos["target_dollars"] = float((account_usd * wt / Decimal("100")).quantize(_TWOPLACES, ROUND_HALF_UP))


def _position_from_row(row: dict[str, Any], *, weight_pct: Decimal, account_usd: Decimal) -> dict[str, Any]:
    from tools.build_action_lists import classify_action_bucket

    bucket = classify_action_bucket(row)
    cat_family = (row.get("catalyst_family") or "").upper()
    days_raw = row.get("catalyst_days", "")
    days_int = parse_catalyst_days(days_raw)
    gap = "LOW"
    if days_int is not None and days_int <= 7:
        gap = "MODERATE"
    wt_f = float(weight_pct.quantize(_FOURPLACES, ROUND_HALF_UP))
    zone = classify_path_a_zone(days_int, row.get("catalyst_mode") or "")
    return {
        "ticker": row.get("ticker", ""),
        "weight_pct": wt_f,
        "target_dollars": float((account_usd * weight_pct / Decimal("100")).quantize(_TWOPLACES, ROUND_HALF_UP)),
        "bucket": bucket,
        "tier": row.get("tier_dev", ""),
        "actionable_rank": _rank(row),
        "catalyst_days": days_raw if days_raw is not None else "",
        "catalyst_mode": row.get("catalyst_mode", ""),
        "catalyst_family": cat_family,
        "effective_family": cat_family,
        "path_a_zone": zone,
        "is_hard_catalyst": row.get("is_hard_catalyst", "") == "1",
        "mom_state": row.get("mom_state", ""),
        "gap_risk": gap,
        "size_band": "EW",
        "price_coverage": "OK",
        "entry_price": None,
        "entry_price_source": "CLOSE",
        "options_overlay_multiplier": 1.0,
        "options_overlay_reasons": "path_a_overlay",
        "opt_liquidity_state": row.get("opt_liquidity_state", "absent"),
        "regulatory_days": row.get("regulatory_days", ""),
        "regulatory_event_type": row.get("regulatory_event_type", ""),
        "regulatory_quality": row.get("regulatory_quality", ""),
        "regulatory_confidence": row.get("regulatory_confidence", ""),
        "regulatory_is_secondary": False,
        "reg_sub_bucket": "",
        "fill_qty": None,
        "fill_vwap": None,
        "fill_notional": None,
    }


def _stamp_zones(positions: list[dict[str, Any]], zones: dict[str, str]) -> None:
    for pos in positions:
        ticker = (pos.get("ticker") or "").strip().upper()
        days = parse_catalyst_days(pos.get("catalyst_days"))
        mode = pos.get("catalyst_mode") or ""
        pos["path_a_zone"] = zones.get(ticker) or classify_path_a_zone(days, mode)


def _pos_rank(pos: dict[str, Any], by_ticker: dict[str, dict[str, Any]]) -> int:
    t = (pos.get("ticker") or "").upper()
    row = by_ticker.get(t)
    if row is not None:
        return _rank(row)
    try:
        return int(pos.get("actionable_rank", 9999))
    except (TypeError, ValueError):
        return 9999


def enforce_path_a_gates(
    positions: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
    gates: PathATimingGates,
    *,
    account_usd: Decimal,
    target_n: int = 30,
) -> PathAResult:
    """Apply T0 cap and T3+ floor via deterministic rank-priority swaps."""
    zones = zone_map_from_rows(rankings)
    ranked_pool = sorted(rankings, key=lambda r: (_rank(r), r.get("ticker", "")))
    by_ticker = {(r.get("ticker") or "").upper(): r for r in ranked_pool}

    work = copy.deepcopy(positions)
    _stamp_zones(work, zones)
    swaps: list[dict[str, Any]] = []
    infeasibility: dict[str, Any] | None = None
    floor_relaxed = False

    def portfolio_tickers() -> set[str]:
        return {(p.get("ticker") or "").upper() for p in work}

    def enforce_t0_cap() -> bool:
        nonlocal infeasibility
        changed = False
        while True:
            t0_w, _ = measure_zone_weights(work, zones, t3_plus_zones=gates.t3_plus_zones)
            if t0_w <= gates.t0_max_weight_pct:
                break
            t0_positions = [p for p in work if zones.get((p.get("ticker") or "").upper(), "T4") == "T0"]
            if not t0_positions:
                break
            drop = max(t0_positions, key=lambda p: (_pos_rank(p, by_ticker), p.get("ticker", "")))
            add_row = None
            held = portfolio_tickers()
            for row in ranked_pool:
                t = (row.get("ticker") or "").upper()
                if t in held:
                    continue
                if zones.get(t, "T4") == "T0":
                    continue
                add_row = row
                break
            if add_row is None:
                infeasibility = {
                    "code": "T0_CAP_INFEASIBLE",
                    "detail": f"no non-T0 candidate to swap for {(drop.get('ticker') or '').upper()}",
                }
                break
            drop_t = (drop.get("ticker") or "").upper()
            add_t = (add_row.get("ticker") or "").upper()
            work.remove(drop)
            work.append(_position_from_row(add_row, weight_pct=Decimal("0"), account_usd=account_usd))
            swaps.append({"out": drop_t, "in": add_t, "reason": "t0_cap"})
            changed = True
        return changed

    def enforce_t3_floor(*, allow_relax: bool) -> bool:
        nonlocal floor_relaxed, infeasibility
        changed = False
        while True:
            _, t3_w = measure_zone_weights(work, zones, t3_plus_zones=gates.t3_plus_zones)
            if t3_w >= gates.t3_plus_min_weight_pct:
                break
            non_t3 = [p for p in work if zones.get((p.get("ticker") or "").upper(), "T4") not in gates.t3_plus_zones]
            candidates_in = [
                row
                for row in ranked_pool
                if zones.get((row.get("ticker") or "").upper(), "T4") in gates.t3_plus_zones
                and (row.get("ticker") or "").upper() not in portfolio_tickers()
            ]
            if not non_t3 or not candidates_in:
                if allow_relax and "t3_plus_floor" in gates.infeasibility_relax_order:
                    floor_relaxed = True
                    infeasibility = {
                        "code": "PATH_A_INFEASIBLE",
                        "detail": "t3_plus floor not achievable; floor relaxed per policy",
                        "t3_plus_weight_pct": float(t3_w),
                    }
                else:
                    infeasibility = {
                        "code": "T3_PLUS_FLOOR_INFEASIBLE",
                        "detail": "no swap candidates for t3_plus floor",
                        "t3_plus_weight_pct": float(t3_w),
                    }
                break
            drop = max(non_t3, key=lambda p: (_pos_rank(p, by_ticker), p.get("ticker", "")))
            add_row = candidates_in[0]
            drop_t = (drop.get("ticker") or "").upper()
            add_t = (add_row.get("ticker") or "").upper()
            work.remove(drop)
            work.append(_position_from_row(add_row, weight_pct=Decimal("0"), account_usd=account_usd))
            swaps.append({"out": drop_t, "in": add_t, "reason": "t3_plus_floor"})
            changed = True
        return changed

    enforce_t0_cap()
    enforce_t3_floor(allow_relax=True)

    if len(work) > target_n:
        work = sorted(work, key=lambda p: (_pos_rank(p, by_ticker), p.get("ticker", "")))[:target_n]

    _equal_reweight(work, account_usd)
    _stamp_zones(work, zones)
    t0_w, t3_w = measure_zone_weights(work, zones, t3_plus_zones=gates.t3_plus_zones)
    compliant = t0_w <= gates.t0_max_weight_pct and (t3_w >= gates.t3_plus_min_weight_pct or floor_relaxed)

    return PathAResult(
        positions=work,
        t0_weight_pct=t0_w,
        t3_plus_weight_pct=t3_w,
        compliant=compliant,
        swaps=swaps,
        infeasibility=infeasibility,
        floor_relaxed=floor_relaxed,
    )


def build_manifest(
    *,
    as_of_date: str,
    gates: PathATimingGates,
    result: PathAResult,
    policy: dict[str, Any],
) -> dict[str, Any]:
    policy_blob = json.dumps(policy, sort_keys=True)
    policy_hash = hashlib.sha256(policy_blob.encode("utf-8")).hexdigest()
    body = {
        "schema": MANIFEST_SCHEMA,
        "as_of_date": as_of_date,
        "generated_at": f"{as_of_date}T00:00:00Z",
        "gates": result.gates_summary(gates),
        "compliant": result.compliant,
        "swaps": result.swaps,
        "infeasibility": result.infeasibility,
        "floor_relaxed": result.floor_relaxed,
        "position_count": len(result.positions),
    }
    body["_governance"] = {
        "tier": 0,
        "shadow_only": True,
        "policy_hash": f"sha256:{policy_hash}",
        "pit_cutoff": as_of_date,
    }
    return body


def write_manifest(manifest: dict[str, Any], *, as_of_date: str, out_dir: Path | None = None) -> Path:
    root = out_dir or MANIFEST_DIR
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{as_of_date}_path_a_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def apply_path_a_overlay(
    positions: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
    policy: dict[str, Any],
    *,
    account_usd: float,
    target_n: int = 30,
) -> PathAResult | None:
    gates = gates_from_policy(policy)
    if not gates.enabled:
        return None
    acct = _d(account_usd)
    n = int(policy.get("ew_top_n", target_n))
    return enforce_path_a_gates(positions, rankings, gates, account_usd=acct, target_n=n)


def main() -> int:
    from tools.live_shadow_portfolio import build_positions, load_policy, load_rankings

    ap = argparse.ArgumentParser(description="Path A timing gates shadow overlay (Spec 106)")
    ap.add_argument("--as-of-date", required=True, help="YYYY-MM-DD")
    ap.add_argument(
        "--policy",
        default=str(REPO / "production_data" / "portfolio_policy_path_a_shadow.json"),
        help="Portfolio policy with path_a_timing_gates.enabled",
    )
    ap.add_argument("--snapshot-dir", help="Override snapshot directory")
    ap.add_argument("--write", action="store_true", help="Write manifest artifact")
    ap.add_argument("--dry-run", action="store_true", help="Print manifest JSON only")
    args = ap.parse_args()

    snap = Path(args.snapshot_dir) if args.snapshot_dir else REPO / "data" / "snapshots" / args.as_of_date
    policy = load_policy(Path(args.policy))
    rankings = load_rankings(snap)
    built = build_positions(rankings, policy)
    gates = gates_from_policy(policy)
    if not gates.enabled:
        print("path_a_timing_gates.enabled is false — nothing to do", file=sys.stderr)
        return 1

    acct = _d(policy.get("account_usd", "500000"))
    result = enforce_path_a_gates(
        built["positions"],
        rankings,
        gates,
        account_usd=acct,
        target_n=int(policy.get("ew_top_n", 30)),
    )
    manifest = build_manifest(as_of_date=args.as_of_date, gates=gates, result=result, policy=policy)

    if args.dry_run or not args.write:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.write:
        path = write_manifest(manifest, as_of_date=args.as_of_date)
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
