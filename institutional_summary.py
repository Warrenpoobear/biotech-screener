"""Institutional Validation v1 — per-ticker elite holder summary from 13F PIT cache.

Pure reporting sidecar. No ranking or decision engine changes.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

VERSION = "1.0.0"
SCHEMA_VERSION = "institutional_summary.v1"
DELTA_SCHEMA_VERSION = "institutional_summary_delta.v1"

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
    *,
    nearest_prior_days: int = 0,
) -> Optional[Dict[str, Any]]:
    """Build per-ticker institutional summary from the 13F PIT cache.

    Returns a dict suitable for JSON serialisation, or None if the cache is
    unavailable or malformed.

    When `nearest_prior_days > 0`, falls back to the most recent prior cache
    within that many days if the exact as_of_date cache is missing. The
    returned dict's `cache_as_of_date` records the actual source cache date,
    so downstream PIT guards (e.g. `_find_prior_institutional_summary`) still
    work correctly — summaries built from a non-exact cache are rejected as
    priors for other monthly dates, and only true quarter-end summaries
    participate in the delta walk-back.
    """
    from common.pit_cache import resolve_pit_cache_dir

    base = cache_base_dir or (Path(__file__).parent / _DEFAULT_CACHE_REL)
    date_dir, _cache_src = resolve_pit_cache_dir(base, as_of_date, nearest_prior_days)
    if date_dir is None:
        return None
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
        cik_to_short: Dict[str, str] = {_pad_cik(m["cik"]): m["short_name"] for m in _mgrs}
    except Exception:
        cik_to_short = {}

    # Corporate actions: resolve renamed tickers to current names
    try:
        from common.corporate_actions import load_actions
        from common.corporate_actions import resolve_ticker as _ca_resolve

        _ca_reg = load_actions()
    except Exception:
        _ca_reg = None

    # CUSIP fallback map for holdings with blank tickers
    _cusip_map: Dict[str, str] = {}
    _cusip_map_path = Path(__file__).parent / "production_data" / "cusip_static_map.json"
    if _cusip_map_path.exists():
        try:
            _cusip_map = json.loads(_cusip_map_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    # Accumulate per-ticker holdings across selected managers
    # Per-ticker: {holders: set(short_name), total_shares: int, total_value: int}
    ticker_accum: Dict[str, Dict[str, Any]] = {}
    _cusip_resolved = 0

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
            # CUSIP fallback: resolve blank tickers from static map
            if not tk and _cusip_map:
                cusip = (h.get("cusip") or "").strip().upper()[:9]
                tk = _cusip_map.get(cusip, "")
                if tk:
                    _cusip_resolved += 1
            # Resolve renamed tickers (e.g. BGNE → ONC, BIOR → BBOT)
            if tk and _ca_reg:
                tk = _ca_resolve(tk, as_of_date, _ca_reg)
            # Second CUSIP fallback: if ticker is non-blank but not in universe
            # (stale ticker from old filing), try CUSIP resolution
            if tk and tk not in universe_tickers and _cusip_map:
                cusip = (h.get("cusip") or "").strip().upper()[:9]
                resolved = _cusip_map.get(cusip, "")
                if resolved and resolved in universe_tickers:
                    tk = resolved
                    _cusip_resolved += 1
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
            holders_by_shares: List[tuple] = sorted(acc["holders"].items(), key=lambda x: (-x[1], x[0]))
            top_names = [name for name, _ in holders_by_shares[:10]]
            tickers_out[tk] = {
                "elite_holders_count": len(acc["holders"]),
                "elite_holder_names": sorted(top_names),
                "elite_holder_shares": dict(sorted(acc["holders"].items())),
                "elite_total_shares": acc["total_shares"],
                "elite_total_value_usd_thousands": acc["total_value"],
                "inst_score_raw": len(acc["holders"]),
                "inst_score_z": 0.0,  # placeholder, computed below
            }
        else:
            tickers_out[tk] = {
                "elite_holders_count": 0,
                "elite_holder_names": [],
                "elite_holder_shares": {},
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
                tickers_out[tk]["inst_score_z"] = round((tickers_out[tk]["inst_score_raw"] - mean) / std, 4)
            else:
                tickers_out[tk]["inst_score_z"] = 0.0

    tickers_with_signal = sum(1 for v in tickers_out.values() if v["elite_holders_count"] > 0)

    if _cusip_resolved > 0:
        logger.info("CUSIP fallback resolved %d blank-ticker holdings from static map", _cusip_resolved)

    return {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "created_at": f"{as_of_date}T00:00:00Z",
        "as_of_date": as_of_date,
        "cache_as_of_date": index.get("as_of_date", as_of_date),
        "cache_schema_version": index.get("schema_version", ""),
        "elite_managers_total": index.get("total_managers", 0),
        "elite_managers_with_filing": index.get("managers_with_filing", 0),
        "tickers_with_signal": tickers_with_signal,
        "tickers_in_universe": len(universe_tickers),
        "signal_coverage_pct": round(tickers_with_signal / len(universe_tickers) * 100, 2) if universe_tickers else 0.0,
        "tickers": tickers_out,
    }


# =============================================================================
# SCHEMA VALIDATORS — pure functions, no I/O
# =============================================================================

_SUMMARY_REQUIRED_TOP_LEVEL = frozenset(
    {
        "schema_version",
        "version",
        "created_at",
        "as_of_date",
        "cache_as_of_date",
        "cache_schema_version",
        "elite_managers_total",
        "elite_managers_with_filing",
        "tickers_with_signal",
        "tickers_in_universe",
        "signal_coverage_pct",
        "tickers",
    }
)

_SUMMARY_REQUIRED_PER_TICKER = frozenset(
    {
        "elite_holders_count",
        "elite_holder_names",
        "elite_holder_shares",
        "elite_total_shares",
        "elite_total_value_usd_thousands",
        "inst_score_raw",
        "inst_score_z",
    }
)

_DELTA_REQUIRED_TOP_LEVEL = frozenset(
    {
        "schema_version",
        "version",
        "created_at",
        "as_of_date",
        "prior_date",
        "tickers_in_current",
        "tickers_in_prior",
        "tickers_common",
        "tickers",
    }
)

_DELTA_REQUIRED_PER_TICKER = frozenset(
    {
        "elite_new_count",
        "elite_exit_count",
        "elite_add_count",
        "elite_trim_count",
        "net_elite_holders_delta",
        "shares_delta_total",
        "value_delta_total",
    }
)


def validate_institutional_summary_schema_v1(
    data: Dict[str, Any],
    *,
    expected_as_of_date: Optional[str] = None,
) -> tuple:
    """Validate institutional_summary.json against institutional_summary.v1.

    Returns (ok: bool, detail: str). Pure function — no I/O.
    """
    # 1. Top-level required fields
    missing = _SUMMARY_REQUIRED_TOP_LEVEL - set(data.keys())
    if missing:
        return False, f"missing top-level fields: {sorted(missing)}"

    # 2. schema_version
    sv = data["schema_version"]
    if sv != SCHEMA_VERSION:
        return False, f"schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}"

    # 3. created_at must end with Z
    ca = data.get("created_at", "")
    if not isinstance(ca, str) or not ca.endswith("Z"):
        return False, f"created_at must end with 'Z': {ca!r}"

    # 4. as_of_date format (YYYY-MM-DD)
    aod = data["as_of_date"]
    if not isinstance(aod, str) or len(aod) != 10:
        return False, f"as_of_date not ISO date: {aod!r}"

    if expected_as_of_date and aod != expected_as_of_date:
        return False, f"as_of_date mismatch: got {aod}, expected {expected_as_of_date}"

    # 5. Numeric top-level fields — types and bounds
    tm = data["elite_managers_total"]
    mwf = data["elite_managers_with_filing"]
    tws = data["tickers_with_signal"]
    tiu = data["tickers_in_universe"]
    cov = data["signal_coverage_pct"]

    if not isinstance(tm, int) or tm < 0:
        return False, f"elite_managers_total must be int >= 0: {tm!r}"
    if not isinstance(mwf, int) or mwf < 0:
        return False, f"elite_managers_with_filing must be int >= 0: {mwf!r}"
    if mwf > tm:
        return False, f"elite_managers_with_filing ({mwf}) > elite_managers_total ({tm})"
    if not isinstance(tws, int) or tws < 0:
        return False, f"tickers_with_signal must be int >= 0: {tws!r}"
    if not isinstance(tiu, int) or tiu < 0:
        return False, f"tickers_in_universe must be int >= 0: {tiu!r}"
    if tws > tiu:
        return False, f"tickers_with_signal ({tws}) > tickers_in_universe ({tiu})"
    if not isinstance(cov, (int, float)) or cov < 0 or cov > 100:
        return False, f"signal_coverage_pct must be in [0, 100]: {cov!r}"

    # 6. Coverage consistency (±0.1 tolerance)
    if tiu > 0:
        expected_cov = round(tws / tiu * 100, 2)
        if abs(cov - expected_cov) > 0.1:
            return False, (f"signal_coverage_pct inconsistent: {cov} vs expected " f"{expected_cov:.2f} ({tws}/{tiu})")

    # 7. Tickers must be a dict
    tickers = data["tickers"]
    if not isinstance(tickers, dict):
        return False, "tickers must be a dict"

    # 8. Tickers dict size must match tickers_in_universe
    if len(tickers) != tiu:
        return False, f"tickers dict size ({len(tickers)}) != tickers_in_universe ({tiu})"

    # 9. Per-ticker required fields (spot-check first 5 + any held)
    checked = 0
    for tk, tv in tickers.items():
        if not isinstance(tv, dict):
            return False, f"tickers[{tk!r}] must be a dict"
        missing_tk = _SUMMARY_REQUIRED_PER_TICKER - set(tv.keys())
        if missing_tk:
            return False, f"tickers[{tk!r}] missing fields: {sorted(missing_tk)}"
        # Type checks
        if not isinstance(tv["elite_holders_count"], int):
            return False, f"tickers[{tk!r}].elite_holders_count must be int"
        if not isinstance(tv["elite_holder_names"], list):
            return False, f"tickers[{tk!r}].elite_holder_names must be list"
        if not isinstance(tv["elite_holder_shares"], dict):
            return False, f"tickers[{tk!r}].elite_holder_shares must be dict"
        if not isinstance(tv["elite_total_shares"], int):
            return False, f"tickers[{tk!r}].elite_total_shares must be int"
        # holders_count == len(elite_holder_shares) (all holders, not capped)
        if tv["elite_holders_count"] != len(tv["elite_holder_shares"]):
            return False, (
                f"tickers[{tk!r}].elite_holders_count ({tv['elite_holders_count']}) "
                f"!= len(elite_holder_shares) ({len(tv['elite_holder_shares'])})"
            )
        checked += 1
        if checked >= 5 and tv["elite_holders_count"] == 0:
            continue  # keep checking until we see a held ticker

    # 10. Tickers dict keys are sorted
    tk_keys = list(tickers.keys())
    if tk_keys != sorted(tk_keys):
        return False, "tickers dict keys not sorted alphabetically"

    # 11. tickers_with_signal consistency
    actual_with = sum(1 for v in tickers.values() if v.get("elite_holders_count", 0) > 0)
    if actual_with != tws:
        return False, (f"tickers_with_signal ({tws}) != actual count of held tickers ({actual_with})")

    return True, "ok"


def validate_institutional_summary_delta_schema_v1(
    data: Dict[str, Any],
    *,
    expected_as_of_date: Optional[str] = None,
) -> tuple:
    """Validate institutional_summary_delta.json against institutional_summary_delta.v1.

    Returns (ok: bool, detail: str). Pure function — no I/O.
    """
    # 1. Top-level required fields
    missing = _DELTA_REQUIRED_TOP_LEVEL - set(data.keys())
    if missing:
        return False, f"missing top-level fields: {sorted(missing)}"

    # 2. schema_version
    sv = data["schema_version"]
    if sv != DELTA_SCHEMA_VERSION:
        return False, f"schema_version mismatch: got {sv!r}, expected {DELTA_SCHEMA_VERSION!r}"

    # 3. created_at must end with Z
    ca = data.get("created_at", "")
    if not isinstance(ca, str) or not ca.endswith("Z"):
        return False, f"created_at must end with 'Z': {ca!r}"

    # 4. as_of_date format
    aod = data["as_of_date"]
    if not isinstance(aod, str) or len(aod) != 10:
        return False, f"as_of_date not ISO date: {aod!r}"

    if expected_as_of_date and aod != expected_as_of_date:
        return False, f"as_of_date mismatch: got {aod}, expected {expected_as_of_date}"

    # 5. prior_date format
    pd_ = data["prior_date"]
    if not isinstance(pd_, str) or len(pd_) != 10:
        return False, f"prior_date not ISO date: {pd_!r}"

    # 6. prior_date < as_of_date
    if pd_ >= aod:
        return False, f"prior_date ({pd_}) must be before as_of_date ({aod})"

    # 7. Numeric top-level fields
    tic = data["tickers_in_current"]
    tip = data["tickers_in_prior"]
    tc = data["tickers_common"]

    if not isinstance(tic, int) or tic < 0:
        return False, f"tickers_in_current must be int >= 0: {tic!r}"
    if not isinstance(tip, int) or tip < 0:
        return False, f"tickers_in_prior must be int >= 0: {tip!r}"
    if not isinstance(tc, int) or tc < 0:
        return False, f"tickers_common must be int >= 0: {tc!r}"
    if tc > min(tic, tip):
        return False, f"tickers_common ({tc}) > min(current={tic}, prior={tip})"

    # 8. Tickers must be a dict
    tickers = data["tickers"]
    if not isinstance(tickers, dict):
        return False, "tickers must be a dict"

    # 9. Tickers dict size must match tickers_in_current
    if len(tickers) != tic:
        return False, f"tickers dict size ({len(tickers)}) != tickers_in_current ({tic})"

    # 10. Per-ticker required fields (spot-check first 5)
    checked = 0
    for tk, tv in tickers.items():
        if not isinstance(tv, dict):
            return False, f"tickers[{tk!r}] must be a dict"
        missing_tk = _DELTA_REQUIRED_PER_TICKER - set(tv.keys())
        if missing_tk:
            return False, f"tickers[{tk!r}] missing fields: {sorted(missing_tk)}"
        # Type checks — all must be int
        for fld in _DELTA_REQUIRED_PER_TICKER:
            if not isinstance(tv[fld], int):
                return False, f"tickers[{tk!r}].{fld} must be int: {tv[fld]!r}"
        # net_elite_holders_delta == new - exit
        expected_net = tv["elite_new_count"] - tv["elite_exit_count"]
        if tv["net_elite_holders_delta"] != expected_net:
            return False, (
                f"tickers[{tk!r}].net_elite_holders_delta ({tv['net_elite_holders_delta']}) "
                f"!= elite_new_count - elite_exit_count ({expected_net})"
            )
        # Counts must be non-negative
        for fld in ("elite_new_count", "elite_exit_count", "elite_add_count", "elite_trim_count"):
            if tv[fld] < 0:
                return False, f"tickers[{tk!r}].{fld} must be >= 0: {tv[fld]}"
        checked += 1
        if checked >= 5:
            break

    return True, "ok"


def compute_institutional_delta(
    current: Dict[str, Any],
    prior: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Compute period-over-period delta between two institutional summaries.

    Returns None if either side lacks ``elite_holder_shares`` (v1 prior).
    """
    cur_tickers = current.get("tickers", {})
    pri_tickers = prior.get("tickers", {})

    # Require elite_holder_shares on both sides — if the first present ticker
    # on either side lacks it, this is a v1 prior without enrichment.
    for side, label in ((cur_tickers, "current"), (pri_tickers, "prior")):
        for _tk, _tv in side.items():
            if "elite_holder_shares" not in _tv:
                logger.info("[INST-DELTA] %s ticker %s lacks elite_holder_shares — skipping", label, _tk)
                return None
            break  # only need to check first ticker

    delta_tickers: Dict[str, Dict[str, Any]] = {}

    for tk, cur_data in cur_tickers.items():
        cur_holders = cur_data.get("elite_holder_shares", {})
        pri_data = pri_tickers.get(tk, {})
        pri_holders = pri_data.get("elite_holder_shares", {})

        cur_names = set(cur_holders)
        pri_names = set(pri_holders)

        new_names = cur_names - pri_names
        exit_names = pri_names - cur_names
        common_names = cur_names & pri_names

        add_count = sum(1 for n in common_names if cur_holders[n] > pri_holders[n])
        trim_count = sum(1 for n in common_names if cur_holders[n] < pri_holders[n])

        cur_total_shares = cur_data.get("elite_total_shares", 0)
        pri_total_shares = pri_data.get("elite_total_shares", 0)
        cur_total_value = cur_data.get("elite_total_value_usd_thousands", 0)
        pri_total_value = pri_data.get("elite_total_value_usd_thousands", 0)

        delta_tickers[tk] = {
            "elite_new_count": len(new_names),
            "elite_exit_count": len(exit_names),
            "elite_add_count": add_count,
            "elite_trim_count": trim_count,
            "net_elite_holders_delta": len(new_names) - len(exit_names),
            "shares_delta_total": cur_total_shares - pri_total_shares,
            "value_delta_total": cur_total_value - pri_total_value,
        }

    cur_tk_set = set(cur_tickers)
    pri_tk_set = set(pri_tickers)

    _cur_as_of = current.get("as_of_date", "")
    return {
        "schema_version": DELTA_SCHEMA_VERSION,
        "version": VERSION,
        "created_at": f"{_cur_as_of}T00:00:00Z" if _cur_as_of else "",
        "as_of_date": _cur_as_of,
        "prior_date": prior.get("as_of_date", ""),
        "current_cache_as_of_date": current.get("cache_as_of_date", ""),
        "prior_cache_as_of_date": prior.get("cache_as_of_date", ""),
        "tickers_in_current": len(cur_tk_set),
        "tickers_in_prior": len(pri_tk_set),
        "tickers_common": len(cur_tk_set & pri_tk_set),
        "tickers": delta_tickers,
    }


def _find_prior_institutional_summary(
    snapshot_dir: Path,
    current_date: str,
    *,
    max_candidates: int = 10,
    cross_quarter: bool = False,
    current_filing_period: str = "",
) -> Optional[Dict[str, Any]]:
    """Find most recent prior institutional_summary.json with elite_holder_shares.

    Walks backward from ``current_date`` through date-named subdirectories
    under ``snapshot_dir``. Returns the parsed dict or None.

    If ``cross_quarter=True``, skips summaries whose 13F cache has the same
    filing period as ``current_filing_period``. This ensures the delta captures
    actual portfolio changes between 13F filing periods, not day-over-day
    noise within the same quarter.
    """
    import re

    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    if not snapshot_dir.exists():
        return None

    all_dates = sorted(
        [d.name for d in snapshot_dir.iterdir() if d.is_dir() and date_re.match(d.name)],
        reverse=True,
    )

    candidates = [d for d in all_dates if d < current_date][:max_candidates]

    for date_str in candidates:
        sidecar = snapshot_dir / date_str / "institutional_summary.json"
        if not sidecar.exists():
            continue
        data = _load_json(sidecar)
        if data is None:
            continue
        # Verify this is an enriched v1 with elite_holder_shares
        tickers = data.get("tickers", {})
        if not tickers:
            continue
        first_ticker_data = next(iter(tickers.values()))
        if "elite_holder_shares" not in first_ticker_data:
            continue
        # PIT guard — cache_as_of_date should match the snapshot date
        cache_date = data.get("cache_as_of_date", "")
        if cache_date and cache_date != date_str:
            logger.warning(
                "[INST] Prior %s has cache_as_of_date=%s (expected %s) — skipping (PIT mismatch)",
                date_str,
                cache_date,
                date_str,
            )
            continue

        # Cross-quarter guard: skip if same filing period
        if cross_quarter and current_filing_period:
            prior_filing = _get_filing_period(snapshot_dir.parent / "caches" / "sec_13f" / "PIT", date_str)
            if prior_filing == current_filing_period:
                continue  # same quarter, keep looking
            logger.info(
                "[INST] Cross-quarter prior found: %s (filing %s vs current %s)",
                date_str,
                prior_filing,
                current_filing_period,
            )

        return data

    return None


def _get_filing_period(cache_base: Path, cache_date: str) -> str:
    """Get the 13F filing period_of_report for a given cache date.

    Returns the period reported by the MOST managers in the cache (ties go to
    the later period). A cache mixes quarters while filings arrive — e.g. the
    2026-08-13 cache held 51 managers at 2026-03-31, 4 at 2026-06-30 and 1 at
    2025-12-31 — so the first manager listed is not a reliable label.
    """
    from collections import Counter

    index_path = cache_base / cache_date / "index.json"
    if not index_path.exists():
        return ""
    idx = _load_json(index_path)
    if not idx:
        return ""
    counts = Counter(m.get("period_of_report", "") for m in idx.get("managers", []) if m.get("period_of_report"))
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def resolve_cross_quarter_delta(
    inst_summary: Dict[str, Any],
    snapshot_dir: Path,
    as_of_date: str,
    cache_base: Path,
    *,
    max_candidates: int = 200,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    """Pick the cross-quarter prior summary and compute the delta.

    Returns ``(prior, delta, status)``. ``delta`` is None whenever no real
    filing-period change can be measured, so callers report "no signal"
    (``has_inst_delta=False``) instead of a run of fake zeros.

    The filing period is looked up for the summary's ACTUAL source cache
    (``cache_as_of_date``), not ``as_of_date``. When the day's 13F cache is
    missing, the summary is built from an older cache via
    ``nearest_prior_days``; looking up ``as_of_date`` then returned "" and
    silently disabled the cross-quarter guard, so the prior was a summary of
    the same cache and every net delta was 0 (2026-09-18..09-30, 10-08, 10-09).

    status: "ok" | "no_prior" | "same_source_cache"
    """
    cur_cache = inst_summary.get("cache_as_of_date") or as_of_date
    cur_filing = _get_filing_period(cache_base, cur_cache)
    prior = _find_prior_institutional_summary(
        snapshot_dir,
        as_of_date,
        cross_quarter=True,
        current_filing_period=cur_filing,
        max_candidates=max_candidates,
    )
    if prior is not None and prior.get("cache_as_of_date", "") == cur_cache:
        # Same source data on both sides: the delta is identically zero and
        # carries no information. Report no signal rather than a measured 0.
        logger.warning(
            "[INST] Prior summary uses the same 13F cache (%s) as %s; no inst_delta signal this run",
            cur_cache,
            as_of_date,
        )
        return None, None, "same_source_cache"
    # An empty filing period means the 13F index is missing. The cross-quarter
    # guard is then skipped ("" is falsy), so a prior from a different cache
    # date of the same unknown quarter is accepted and every net delta is 0.
    # That 0 is not a measurement. Leave the signal missing.
    if not cur_filing:
        logger.warning(
            "[INST] No 13F filing period for cache %s (as_of %s); no inst_delta signal this run",
            cur_cache,
            as_of_date,
        )
        return None, None, "unknown_filing_period"
    if prior is None:
        return None, None, "no_prior"
    return prior, compute_institutional_delta(inst_summary, prior), "ok"
