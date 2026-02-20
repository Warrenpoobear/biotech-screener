"""Institutional Validation v1 — per-ticker elite holder summary from 13F PIT cache.

Pure reporting sidecar. No ranking or decision engine changes.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

VERSION = "1.0.0"
SCHEMA_VERSION = "institutional_summary.v1"

_DEFAULT_CACHE_REL = Path("data") / "caches" / "sec_13f" / "PIT"


def _pad_cik(cik: str) -> str:
    """Pad CIK to 10-digit zero-filled format used by the 13F cache."""
    return cik.lstrip("0").zfill(10)


def _load_json(path: Path) -> Optional[Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Could not load %s: %s", path, e)
        return None


def build_institutional_summary(
    as_of_date: str,
    universe_tickers: Set[str],
    cache_base_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Build per-ticker institutional summary from the 13F PIT cache.

    Returns a dict suitable for JSON serialisation, or None if the cache is
    unavailable or malformed.
    """
    base = cache_base_dir or (Path(__file__).parent / _DEFAULT_CACHE_REL)
    date_dir = base / as_of_date
    index_path = date_dir / "index.json"

    index = _load_json(index_path)
    if index is None:
        return None

    if index.get("schema_version") != "sec_13f_pit_index.v1":
        logger.warning("Unexpected 13F cache schema: %s", index.get("schema_version"))
        return None

    # Build CIK → short_name map from elite_managers (lazy import)
    try:
        from elite_managers import get_elite_managers
        _mgrs = get_elite_managers()
        cik_to_short: Dict[str, str] = {
            _pad_cik(m["cik"]): m["short_name"] for m in _mgrs
        }
    except Exception:
        cik_to_short = {}

    # Accumulate per-ticker holdings across selected managers
    # Per-ticker: {holders: set(short_name), total_shares: int, total_value: int}
    ticker_accum: Dict[str, Dict[str, Any]] = {}

    selected_managers = [m for m in index.get("managers", []) if m.get("selected")]
    for mgr_entry in selected_managers:
        cik = mgr_entry["manager_cik"]
        mgr_json_path = date_dir / mgr_entry.get("manager_json_path", f"managers/{cik}.json")
        mgr_data = _load_json(mgr_json_path)
        if mgr_data is None:
            continue

        short_name = cik_to_short.get(cik, mgr_entry.get("manager_name", cik))

        for h in mgr_data.get("holdings", []):
            tk = h.get("ticker", "")
            if not tk or tk not in universe_tickers:
                continue

            if tk not in ticker_accum:
                ticker_accum[tk] = {
                    "holders": {},  # short_name -> shares (for top-10 sort)
                    "total_shares": 0,
                    "total_value": 0,
                }
            acc = ticker_accum[tk]
            shares = h.get("shares", 0) or 0
            value = h.get("value_usd_thousands", 0) or 0
            # A manager can hold multiple lines for the same ticker (e.g. common + call)
            acc["holders"][short_name] = acc["holders"].get(short_name, 0) + shares
            acc["total_shares"] += shares
            acc["total_value"] += value

    # Build per-ticker output
    tickers_out: Dict[str, Dict[str, Any]] = {}

    # Include ALL universe tickers (0 holders for those not held)
    for tk in sorted(universe_tickers):
        acc = ticker_accum.get(tk)
        if acc:
            holders_by_shares: List[tuple] = sorted(
                acc["holders"].items(), key=lambda x: (-x[1], x[0])
            )
            top_names = [name for name, _ in holders_by_shares[:10]]
            tickers_out[tk] = {
                "elite_holders_count": len(acc["holders"]),
                "elite_holder_names": sorted(top_names),
                "elite_total_shares": acc["total_shares"],
                "elite_total_value_usd_thousands": acc["total_value"],
                "inst_score_raw": len(acc["holders"]),
                "inst_score_z": 0.0,  # placeholder, computed below
            }
        else:
            tickers_out[tk] = {
                "elite_holders_count": 0,
                "elite_holder_names": [],
                "elite_total_shares": 0,
                "elite_total_value_usd_thousands": 0,
                "inst_score_raw": 0,
                "inst_score_z": 0.0,
            }

    # Compute z-scores cross-sectionally (ddof=0)
    raw_scores = [tickers_out[tk]["inst_score_raw"] for tk in sorted(universe_tickers)]
    n = len(raw_scores)
    if n > 0:
        mean = sum(raw_scores) / n
        var = sum((x - mean) ** 2 for x in raw_scores) / n
        std = math.sqrt(var) if var > 0 else 0.0
        for tk in tickers_out:
            if std > 0:
                tickers_out[tk]["inst_score_z"] = round(
                    (tickers_out[tk]["inst_score_raw"] - mean) / std, 4
                )
            else:
                tickers_out[tk]["inst_score_z"] = 0.0

    tickers_with_signal = sum(
        1 for v in tickers_out.values() if v["elite_holders_count"] > 0
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of_date": as_of_date,
        "cache_as_of_date": index.get("as_of_date", as_of_date),
        "cache_schema_version": index.get("schema_version", ""),
        "elite_managers_total": index.get("total_managers", 0),
        "elite_managers_with_filing": index.get("managers_with_filing", 0),
        "tickers_with_signal": tickers_with_signal,
        "tickers_in_universe": len(universe_tickers),
        "signal_coverage_pct": round(
            tickers_with_signal / len(universe_tickers) * 100, 2
        ) if universe_tickers else 0.0,
        "tickers": tickers_out,
    }
