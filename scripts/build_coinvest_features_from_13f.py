#!/usr/bin/env python3
"""Build PIT-safe coinvest features from 13F cache.

Reads ONLY from quarterly PIT 13F caches to produce deterministic
per-ticker coinvest features, eliminating lookahead bias from the
smart-money factor during historical simulation.

Usage:
    python3 scripts/build_coinvest_features_from_13f.py \\
        --as-of-date 2025-12-31 \\
        --cache-root data/caches/sec_13f/PIT \\
        --out production_data/coinvest_features/2025-12-31.json \\
        --universe production_data/universe.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

VERSION = "1.1.0"
SCHEMA_VERSION = "coinvest_features.v1"

# Conviction formula constants — replicate run_screen.py:802-809 exactly.
TIER_WEIGHTS: Dict[int, float] = {1: 1.0, 2: 0.6, 3: 0.2, 0: 0.2}
CHANGE_WEIGHTS: Dict[str, float] = {
    "NEW": 1.25,
    "INCREASE": 1.25,
    "HOLD": 1.0,
    "DECREASE": 0.75,
    "EXIT": 0.5,
}
CHANGE_THRESHOLD = 0.10
RECENCY_FRESH_DAYS = 90

# A manager's selected filing is dropped once its period_of_report is older
# than this. The freshest legitimate filing can be ~137 days old just before
# a 13F deadline (quarter + 45-day filing window); one late or missed quarter
# adds ~92 days. Beyond that the filer has stopped (or moved CIK), and the PIT
# cache would otherwise carry its last book forever.
MAX_REPORT_AGE_DAYS = 230


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Optional[dict]:
    """Graceful JSON load; returns None on any error."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _load_cusip_map(path: Path) -> Dict[str, str]:
    """Load CUSIP->ticker static map. Returns empty dict on failure."""
    data = _load_json(path)
    return data if isinstance(data, dict) else {}


def _resolve_ticker(
    holding: dict,
    cusip_map: Dict[str, str],
    ca_registry: Any = None,
    as_of: str = "",
) -> str:
    """Resolve ticker from holding, with CUSIP fallback and rename resolution.

    If a corporate actions registry is provided, old ticker symbols (e.g.
    BGNE) are resolved to their current name (e.g. ONC) as of the given date.
    """
    tk = (holding.get("ticker") or "").strip().upper()
    if not tk:
        cusip = (holding.get("cusip") or "").strip()
        if cusip and cusip in cusip_map:
            tk = cusip_map[cusip].upper()
    if not tk:
        return ""
    # Rename resolution: map old tickers to current names (PIT-safe)
    if ca_registry and as_of:
        from common.corporate_actions import resolve_ticker as _ca_resolve

        tk = _ca_resolve(tk, as_of, ca_registry)
    return tk


def _pad_cik(cik: str) -> str:
    """Pad CIK to 10-digit zero-filled format."""
    return cik.lstrip("0").zfill(10)


def _get_dominant_period_of_report(index: dict) -> Optional[str]:
    """Most common period_of_report across selected managers."""
    counts: Dict[str, int] = {}
    for m in index.get("managers", []):
        if not m.get("selected"):
            continue
        por = m.get("period_of_report", "")
        if por:
            counts[por] = counts.get(por, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _get_prior_quarter_end(period_of_report: date) -> date:
    """Compute prior quarter-end date.

    Mar 31 -> Dec 31 (prior year), Jun 30 -> Mar 31,
    Sep 30 -> Jun 30, Dec 31 -> Sep 30.
    """
    m, y = period_of_report.month, period_of_report.year
    if m <= 3:
        return date(y - 1, 12, 31)
    elif m <= 6:
        return date(y, 3, 31)
    elif m <= 9:
        return date(y, 6, 30)
    else:
        return date(y, 9, 30)


def _find_prior_cache_dir(
    cache_root: Path,
    current_por: str,
) -> Optional[Path]:
    """Find cache dir whose dominant period_of_report matches the prior quarter.

    Scans all date-dirs in *cache_root*, loading each ``index.json`` to check
    the dominant ``period_of_report``.  Returns the first match (newest
    as_of_date first) or ``None``.
    """
    try:
        target = _get_prior_quarter_end(date.fromisoformat(current_por))
    except (ValueError, TypeError):
        return None
    target_str = target.isoformat()

    if not cache_root.is_dir():
        return None

    for d in sorted(cache_root.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        idx = _load_json(d / "index.json")
        if idx is None:
            continue
        if idx.get("schema_version") != "sec_13f_pit_index.v1":
            continue
        dom_por = _get_dominant_period_of_report(idx)
        if dom_por == target_str:
            return d

    return None


# ---------------------------------------------------------------------------
# Position-change classification
# ---------------------------------------------------------------------------


def _classify_position_change(
    current_val: float,
    prior_val: float,
    is_new: bool,
    is_exit: bool,
) -> str:
    """Classify position change.  Replicates run_screen.py:878-891."""
    if is_new:
        return "NEW"
    if is_exit:
        return "EXIT"
    if prior_val > 0:
        change_pct = (current_val - prior_val) / prior_val
        if change_pct > CHANGE_THRESHOLD:
            return "INCREASE"
        elif change_pct < -CHANGE_THRESHOLD:
            return "DECREASE"
        else:
            return "HOLD"
    else:
        return "NEW" if current_val > 0 else "HOLD"


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def _new_ticker_accum() -> Dict[str, Any]:
    """Fresh per-ticker accumulator."""
    return {
        "tier1_count": 0,
        "sponsor_tier1_count": 0,
        "sponsor_overlap_count": 0,
        "coinvest_overlap_count": 0,
        "conviction_overlap": 0.0,
        "tier1_conviction_overlap": 0.0,
        "max_tier1_position_pct": 0.0,
        "days_since_latest_filing": None,
        "latest_filed_at": None,
        "coinvest_recency_state": "stale",
        "coinvest_holders": [],
        "holder_tiers": {},
        "position_changes": {},
    }


def build_coinvest_features(
    as_of_date: str,
    cache_root: Path,
    universe_tickers: Set[str],
    manager_tiers: Dict[str, int],
    manager_names: Dict[str, str],
    cusip_map: Optional[Dict[str, str]] = None,
    prior_cache_dir: Optional[Path] = None,
    no_prior: bool = False,
    nearest_prior_days: int = 0,
    manager_groups: Optional[Dict[str, str]] = None,
    max_report_age_days: Optional[int] = MAX_REPORT_AGE_DAYS,
) -> Optional[dict]:
    """Build coinvest features from PIT 13F cache.

    Parameters
    ----------
    as_of_date : str
        ISO-format date (e.g. ``"2025-12-31"``).
    cache_root : Path
        Root of the 13F PIT cache tree (contains date-dirs).
    universe_tickers : set[str]
        Canonical ticker universe.
    manager_tiers : dict[str, int]
        Padded CIK -> tier (1/2/3/0).
    manager_names : dict[str, str]
        Padded CIK -> short display name.
    cusip_map : dict[str, str] | None
        CUSIP -> ticker fallback map.
    prior_cache_dir : Path | None
        Explicit prior-quarter cache directory (overrides scan).
    no_prior : bool
        If *True*, skip prior-quarter comparison entirely.
    manager_groups : dict[str, str] | None
        Padded CIK -> firm group name. Filer CIKs of one firm count as a
        single holder, using only the group's latest period_of_report.
    max_report_age_days : int | None
        Drop managers whose period_of_report is older than this many days
        before *as_of_date*. ``None`` disables the gate.

    Returns
    -------
    dict | None
        Full feature output, or *None* on cache-load failure.
    """
    cusip_map = cusip_map or {}

    # Resolve the current cache dir: exact-match for as_of_date, or the nearest
    # prior cache within nearest_prior_days if the exact date has no cache.
    # Backward-only — preserves PIT correctness.
    from common.pit_cache import resolve_pit_cache_dir

    date_dir, _cache_src = resolve_pit_cache_dir(cache_root, as_of_date, nearest_prior_days)
    if date_dir is None:
        return None

    # Corporate actions registry for ticker rename resolution
    try:
        from common.corporate_actions import load_actions as _load_ca

        _ca_registry = _load_ca()
    except Exception:
        _ca_registry = None

    # -- Load current cache ------------------------------------------------
    index = _load_json(date_dir / "index.json")
    if index is None:
        return None
    if index.get("schema_version") != "sec_13f_pit_index.v1":
        return None

    current_por = _get_dominant_period_of_report(index)
    as_of = date.fromisoformat(as_of_date)

    # -- Find prior cache --------------------------------------------------
    prior_index: Optional[dict] = None
    prior_date_dir: Optional[Path] = None
    prior_por: Optional[str] = None
    prior_cache_as_of: Optional[str] = None

    if not no_prior and current_por:
        if prior_cache_dir is not None:
            prior_date_dir = prior_cache_dir
        else:
            prior_date_dir = _find_prior_cache_dir(cache_root, current_por)

        if prior_date_dir is not None:
            prior_index = _load_json(prior_date_dir / "index.json")
            if prior_index and prior_index.get("schema_version") == "sec_13f_pit_index.v1":
                prior_por = _get_dominant_period_of_report(prior_index)
                prior_cache_as_of = prior_index.get("as_of_date")
            else:
                prior_index = None
                prior_date_dir = None

    # -- Load prior holdings per manager: {CIK: {ticker: value_k}} ---------
    prior_holdings_by_mgr: Dict[str, Dict[str, float]] = {}
    if prior_index and prior_date_dir:
        for m in prior_index.get("managers", []):
            if not m.get("selected"):
                continue
            cik = m["manager_cik"]
            json_path = m.get("manager_json_path", f"managers/{cik}.json")
            mgr_data = _load_json(prior_date_dir / json_path)
            if mgr_data is None:
                continue
            tkr_vals: Dict[str, float] = {}
            for h in mgr_data.get("holdings", []):
                tk = _resolve_ticker(h, cusip_map, _ca_registry, as_of_date)
                if not tk:
                    continue
                val = h.get("value_usd_thousands", 0) or 0
                tkr_vals[tk] = tkr_vals.get(tk, 0) + val
            prior_holdings_by_mgr[cik] = tkr_vals

    # -- Per-ticker accumulation -------------------------------------------
    ticker_data: Dict[str, Dict[str, Any]] = {}
    selected_managers = [m for m in index.get("managers", []) if m.get("selected")]
    managers_used: List[dict] = []
    change_summary: Dict[str, int] = {
        "NEW": 0,
        "INCREASE": 0,
        "HOLD": 0,
        "DECREASE": 0,
        "EXIT": 0,
    }
    unresolved_count = 0
    manager_groups = manager_groups or {}
    stale_excluded: List[dict] = []
    not_in_roster: List[dict] = []

    # -- Holder units: one per manager, or one per firm_group --------------
    # A firm's filer CIKs (e.g. Foresite IV/V/VI, or an old and new CIK after
    # a filer change) count once. Only members at the group's latest
    # period_of_report contribute; their books are summed.
    units: Dict[str, Dict[str, Any]] = {}
    for mgr_entry in selected_managers:
        cik = mgr_entry["manager_cik"]
        # Roster gate: caches written before a roster change still list
        # managers that have since been retired; manager_tiers is the roster.
        if manager_tiers and cik not in manager_tiers:
            not_in_roster.append({"cik": cik, "name": mgr_entry.get("manager_name", cik)})
            continue
        json_path = mgr_entry.get(
            "manager_json_path",
            f"managers/{cik}.json",
        )
        mgr_data = _load_json(date_dir / json_path)
        if mgr_data is None:
            continue

        short_name = manager_names.get(
            cik,
            mgr_entry.get("manager_name", cik),
        )
        tier = manager_tiers.get(cik, 0)
        filed_at_str = mgr_data.get("filed_at") or mgr_entry.get("filed_at", "")
        por_str = mgr_entry.get("period_of_report") or mgr_data.get("period_of_report", "")

        try:
            filed_at: Optional[date] = date.fromisoformat(filed_at_str) if filed_at_str else None
        except (ValueError, TypeError):
            filed_at = None

        # PIT safety: skip future filings.
        if filed_at and filed_at > as_of:
            continue

        # Staleness gate: an abandoned filer's last book must not persist.
        if max_report_age_days is not None and por_str:
            try:
                por_age = (as_of - date.fromisoformat(por_str)).days
            except (ValueError, TypeError):
                por_age = None
            if por_age is not None and por_age > max_report_age_days:
                stale_excluded.append(
                    {"cik": cik, "name": short_name, "period_of_report": por_str, "filed_at": filed_at_str}
                )
                continue

        # Aggregate current holdings by resolved ticker.
        holdings_by_ticker: Dict[str, float] = {}
        total_value = 0.0
        mgr_unresolved = 0
        for h in mgr_data.get("holdings", []):
            val = h.get("value_usd_thousands", 0) or 0
            total_value += val
            tk = _resolve_ticker(h, cusip_map, _ca_registry, as_of_date)
            if not tk:
                mgr_unresolved += 1
                continue
            holdings_by_ticker[tk] = holdings_by_ticker.get(tk, 0) + val

        group = manager_groups.get(cik)
        key = group or cik
        unit = units.get(key)
        if unit is None or por_str > unit["period_of_report"]:
            unit = {
                "name": group or short_name,
                "tier": tier,
                "filed_at": filed_at,
                "filed_at_str": filed_at_str,
                "period_of_report": por_str,
                "holdings": {},
                "total_value": 0.0,
                "prior": {},
                "unresolved": 0,
                "members": [],
            }
            units[key] = unit
        elif por_str < unit["period_of_report"]:
            continue  # superseded by a newer filing from the same firm

        if TIER_WEIGHTS.get(tier, 0.2) > TIER_WEIGHTS.get(unit["tier"], 0.2):
            unit["tier"] = tier
        if filed_at and (unit["filed_at"] is None or filed_at > unit["filed_at"]):
            unit["filed_at"] = filed_at
            unit["filed_at_str"] = filed_at_str
        for tk, val in holdings_by_ticker.items():
            unit["holdings"][tk] = unit["holdings"].get(tk, 0) + val
        for tk, val in prior_holdings_by_mgr.get(cik, {}).items():
            unit["prior"][tk] = unit["prior"].get(tk, 0) + val
        unit["total_value"] += total_value
        unit["unresolved"] += mgr_unresolved
        unit["members"].append(
            {
                "cik": cik,
                "name": short_name,
                "firm_group": group or "",
                "period_of_report": por_str,
                "filed_at": filed_at_str,
            }
        )

    for unit in units.values():
        short_name = unit["name"]
        tier = unit["tier"]
        filed_at = unit["filed_at"]
        filed_at_str = unit["filed_at_str"]
        holdings_by_ticker = unit["holdings"]
        total_value = unit["total_value"]
        prior_tkr_vals = unit["prior"]
        unresolved_count += unit["unresolved"]
        managers_used.extend(unit["members"])

        # Union of current + prior tickers (to detect EXIT).
        all_tickers = set(holdings_by_ticker.keys())
        if prior_tkr_vals:
            all_tickers |= set(prior_tkr_vals.keys())

        for tk in all_tickers:
            if tk not in universe_tickers:
                continue

            current_val = holdings_by_ticker.get(tk, 0)
            prior_val = prior_tkr_vals.get(tk, 0) if prior_tkr_vals else 0

            # Classify position change.
            is_new = current_val > 0 and (not prior_tkr_vals or tk not in prior_tkr_vals)
            is_exit = current_val == 0 and prior_val > 0

            if prior_tkr_vals:
                chg_type = _classify_position_change(
                    current_val,
                    prior_val,
                    is_new,
                    is_exit,
                )
            else:
                chg_type = ""  # No prior -> no classification

            if chg_type:
                change_summary[chg_type] = change_summary.get(chg_type, 0) + 1

            # EXIT positions: manager no longer holds the ticker.
            # Track in change_summary but exclude from per-ticker features.
            if is_exit:
                continue

            # -- Conviction components --
            tier_w = TIER_WEIGHTS.get(tier, 0.2)

            if total_value > 0:
                position_pct = (current_val / total_value) * 100
                pos_w = max(0.5, min(2.0, math.sqrt(position_pct / 1.0)))
            else:
                position_pct = 0.0
                pos_w = 0.5

            chg_w = CHANGE_WEIGHTS.get(chg_type, 1.0) if chg_type else 1.0

            if filed_at is not None:
                days_since = (as_of - filed_at).days
                recency_w = max(0.7, min(1.5, 1.5 - days_since / 180))
            else:
                days_since = None
                recency_w = 1.0

            holder_conviction = tier_w * pos_w * chg_w * recency_w

            # -- Accumulate --
            if tk not in ticker_data:
                ticker_data[tk] = _new_ticker_accum()

            td = ticker_data[tk]
            td["coinvest_holders"].append(short_name)
            td["coinvest_overlap_count"] += 1
            td["sponsor_overlap_count"] += 1
            td["holder_tiers"][short_name] = tier
            td["conviction_overlap"] += holder_conviction

            if chg_type:
                td["position_changes"][short_name] = chg_type

            if tier == 1:
                td["tier1_count"] += 1
                td["sponsor_tier1_count"] += 1
                td["tier1_conviction_overlap"] += holder_conviction
                td["max_tier1_position_pct"] = max(
                    td["max_tier1_position_pct"],
                    position_pct,
                )

            if days_since is not None:
                cur_min = td["days_since_latest_filing"]
                if cur_min is None or days_since < cur_min:
                    td["days_since_latest_filing"] = days_since
                    td["latest_filed_at"] = filed_at_str

    # -- Finalize per-ticker fields ----------------------------------------
    for td in ticker_data.values():
        td["conviction_overlap"] = round(td["conviction_overlap"], 4)
        td["tier1_conviction_overlap"] = round(
            td["tier1_conviction_overlap"],
            4,
        )
        td["max_tier1_position_pct"] = round(td["max_tier1_position_pct"], 2)
        dsf = td["days_since_latest_filing"]
        td["coinvest_recency_state"] = "fresh" if dsf is not None and dsf <= RECENCY_FRESH_DAYS else "stale"

    # -- Assemble output ---------------------------------------------------
    tickers_with_signal = len(ticker_data)
    tickers_in_universe = len(universe_tickers)

    output = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of_date": as_of_date,
        "cache_as_of_date": index.get("as_of_date", as_of_date),
        "period_of_report": current_por or "",
        "prior_period_of_report": prior_por or "",
        "prior_cache_as_of_date": prior_cache_as_of or "",
        "managers_total": index.get("total_managers", 0),
        "managers_with_filing": index.get("managers_with_filing", 0),
        "tickers_in_universe": tickers_in_universe,
        "tickers_with_signal": tickers_with_signal,
        "signal_coverage_pct": (
            round(tickers_with_signal / tickers_in_universe * 100, 1) if tickers_in_universe > 0 else 0.0
        ),
        "tickers": dict(sorted(ticker_data.items())),
        "provenance": {
            "builder": "build_coinvest_features_from_13f.py",
            "builder_version": VERSION,
            "cache_schema_version": "sec_13f_pit_index.v1",
            "cusip_map_entries": len(cusip_map),
            "unresolved_holdings": unresolved_count,
            "prior_available": prior_index is not None,
            "managers_used": managers_used,
            "stale_managers_excluded": stale_excluded,
            "managers_not_in_roster": not_in_roster,
            "max_report_age_days": max_report_age_days,
            "change_summary": change_summary,
        },
    }

    return output


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_coinvest_features_schema(
    features: dict,
) -> Tuple[bool, str]:
    """Validate output matches ``coinvest_features.v1`` schema.

    Returns ``(True, "OK")`` on success, ``(False, reason)`` on failure.
    """
    if not isinstance(features, dict):
        return False, "features must be a dict"

    required_top = [
        "schema_version",
        "version",
        "created_at",
        "as_of_date",
        "tickers_in_universe",
        "tickers_with_signal",
        "signal_coverage_pct",
        "tickers",
        "provenance",
    ]
    for k in required_top:
        if k not in features:
            return False, f"missing required field: {k}"

    if features["schema_version"] != SCHEMA_VERSION:
        return False, (f"expected schema {SCHEMA_VERSION}, " f"got {features['schema_version']}")

    # Coverage consistency.
    n_uni = features["tickers_in_universe"]
    n_sig = features["tickers_with_signal"]
    expected_pct = round(n_sig / n_uni * 100, 1) if n_uni > 0 else 0.0
    if abs(features["signal_coverage_pct"] - expected_pct) > 0.2:
        return False, (f"coverage inconsistent: {features['signal_coverage_pct']} " f"vs expected {expected_pct}")

    # Per-ticker required fields.
    required_ticker = [
        "tier1_count",
        "coinvest_overlap_count",
        "conviction_overlap",
        "coinvest_holders",
        "holder_tiers",
    ]
    for tk, td in features.get("tickers", {}).items():
        for f in required_ticker:
            if f not in td:
                return False, f"ticker {tk} missing field: {f}"

    # Provenance required fields.
    prov = features.get("provenance", {})
    prov_required = [
        "builder",
        "builder_version",
        "prior_available",
        "managers_used",
    ]
    for k in prov_required:
        if k not in prov:
            return False, f"provenance missing field: {k}"

    return True, "OK"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_universe(path: Path) -> Set[str]:
    """Load universe tickers from JSON (list-of-dicts or list-of-strings)."""
    data = _load_json(path)
    if data is None:
        return set()
    tickers: Set[str] = set()
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                tickers.add(item)
            elif isinstance(item, dict):
                tk = item.get("ticker", "")
                if tk:
                    tickers.add(tk)
    elif isinstance(data, dict):
        tickers = set(data.keys())
    tickers.discard("")
    return tickers


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build PIT-safe coinvest features from 13F cache",
    )
    parser.add_argument(
        "--as-of-date",
        required=True,
        help="As-of date (ISO format)",
    )
    parser.add_argument(
        "--cache-root",
        default="data/caches/sec_13f/PIT",
        help="13F cache root directory",
    )
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument(
        "--universe",
        default="production_data/universe.json",
        help="Universe JSON path",
    )
    parser.add_argument(
        "--cusip-map",
        default="production_data/cusip_static_map.json",
        help="CUSIP->ticker static map",
    )
    parser.add_argument(
        "--prior-cache-dir",
        default=None,
        help="Explicit prior-quarter cache directory",
    )
    parser.add_argument(
        "--no-prior",
        action="store_true",
        help="Skip prior-quarter comparison entirely",
    )
    parser.add_argument(
        "--nearest-prior-days",
        type=int,
        default=0,
        help="If no exact cache exists for --as-of-date, use the nearest prior "
        "cache within this many days (backward-only). Default 0 = exact-match only.",
    )
    args = parser.parse_args(argv)

    # Import project modules (add project root to path).
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    from elite_managers import get_all_managers  # noqa: E402

    cache_root = Path(args.cache_root)

    # Load universe.
    universe_tickers = _load_universe(Path(args.universe))
    if not universe_tickers:
        print(
            f"ERROR: empty universe from {args.universe}",
            file=sys.stderr,
        )
        return 1

    # Load CUSIP map.
    cusip_map = _load_cusip_map(Path(args.cusip_map))

    # Build CIK -> tier / name maps from elite_managers.
    manager_tiers: Dict[str, int] = {}
    manager_names: Dict[str, str] = {}
    manager_groups: Dict[str, str] = {}
    for m in get_all_managers():
        padded = _pad_cik(m["cik"])
        manager_tiers[padded] = m.get("tier", 0)
        manager_names[padded] = m.get("short_name", m.get("name", padded))
        if m.get("firm_group"):
            manager_groups[padded] = m["firm_group"]

    prior_dir = Path(args.prior_cache_dir) if args.prior_cache_dir else None

    result = build_coinvest_features(
        as_of_date=args.as_of_date,
        cache_root=cache_root,
        universe_tickers=universe_tickers,
        manager_tiers=manager_tiers,
        manager_names=manager_names,
        cusip_map=cusip_map,
        prior_cache_dir=prior_dir,
        no_prior=args.no_prior,
        nearest_prior_days=args.nearest_prior_days,
        manager_groups=manager_groups,
    )

    if result is None:
        print(
            f"ERROR: failed to build features for {args.as_of_date}",
            file=sys.stderr,
        )
        return 1

    ok, msg = validate_coinvest_features_schema(result)
    if not ok:
        print(f"ERROR: schema validation failed: {msg}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"Wrote {out_path} "
        f"({result['tickers_with_signal']}/{result['tickers_in_universe']} "
        f"tickers, coverage {result['signal_coverage_pct']}%)",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
