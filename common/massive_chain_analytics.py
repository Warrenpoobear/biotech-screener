"""Massive chain analytics — skew, straddle pricing, OI concentration.

Consumes the per-strike chain snapshot from options_history_massive.py
to compute analytics that Tastytrade diagnostics cannot provide:
  - 25-delta risk reversal (RR_25d)
  - put/call skew from chain
  - ATM straddle actual pricing
  - OI concentration metrics
  - volume distribution by expiry bucket
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.options_quality import MAX_SPREAD_PCT, MIN_OI_THRESHOLD

logger = logging.getLogger(__name__)


def find_25delta_contracts(
    contracts: List[Dict[str, Any]],
    expiry: str,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Find the 25-delta put and 25-delta call for a specific expiry.

    Returns (put_contract, call_contract) or (None, None) if not found.
    """
    puts = [
        c
        for c in contracts
        if c.get("expiration_date") == expiry and c.get("contract_type") == "put" and c.get("delta") is not None
    ]
    calls = [
        c
        for c in contracts
        if c.get("expiration_date") == expiry and c.get("contract_type") == "call" and c.get("delta") is not None
    ]

    # 25-delta put: delta closest to -0.25
    best_put = None
    best_put_dist = float("inf")
    for p in puts:
        dist = abs(p["delta"] - (-0.25))
        if dist < best_put_dist:
            best_put_dist = dist
            best_put = p

    # 25-delta call: delta closest to +0.25
    best_call = None
    best_call_dist = float("inf")
    for c in calls:
        dist = abs(c["delta"] - 0.25)
        if dist < best_call_dist:
            best_call_dist = dist
            best_call = c

    # Reject if delta is too far from target (> 0.15 away)
    if best_put and best_put_dist > 0.15:
        best_put = None
    if best_call and best_call_dist > 0.15:
        best_call = None

    return best_put, best_call


def compute_rr_25d(
    put_contract: Optional[Dict],
    call_contract: Optional[Dict],
) -> Optional[float]:
    """Compute 25-delta risk reversal.

    RR_25d = IV(25d call) - IV(25d put)
    Positive = call skew (market pricing upside)
    Negative = put skew (market pricing downside)
    """
    if put_contract is None or call_contract is None:
        return None
    put_iv = put_contract.get("implied_volatility")
    call_iv = call_contract.get("implied_volatility")
    if put_iv is None or call_iv is None:
        return None
    return call_iv - put_iv


def compute_put_call_skew(
    put_contract: Optional[Dict],
    call_contract: Optional[Dict],
) -> Optional[float]:
    """Compute put/call skew: (put_iv - call_iv) / avg_iv.

    Positive = put premium (bearish sentiment / protection buying).
    """
    if put_contract is None or call_contract is None:
        return None
    put_iv = put_contract.get("implied_volatility")
    call_iv = call_contract.get("implied_volatility")
    if put_iv is None or call_iv is None:
        return None
    avg_iv = (put_iv + call_iv) / 2
    if avg_iv <= 0:
        return None
    return (put_iv - call_iv) / avg_iv


def find_atm_contracts(
    contracts: List[Dict[str, Any]],
    underlying_price: float,
    expiry: str,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Find ATM call and put for a specific expiry.

    ATM = strike closest to underlying_price.
    """
    expiry_contracts = [c for c in contracts if c.get("expiration_date") == expiry]
    if not expiry_contracts or underlying_price <= 0:
        return None, None

    # Find the strike closest to underlying price
    strikes = sorted(set(c.get("strike_price", 0) for c in expiry_contracts if c.get("strike_price")))
    if not strikes:
        return None, None
    atm_strike = min(strikes, key=lambda s: abs(s - underlying_price))

    atm_call = None
    atm_put = None
    for c in expiry_contracts:
        if c.get("strike_price") == atm_strike:
            if c.get("contract_type") == "call":
                atm_call = c
            elif c.get("contract_type") == "put":
                atm_put = c

    return atm_call, atm_put


def compute_atm_straddle(
    contracts: List[Dict[str, Any]],
    underlying_price: float,
    expiry: str,
) -> Dict[str, Any]:
    """Compute actual ATM straddle price from chain snapshot.

    Returns dict with straddle_price, implied_move, atm_strike,
    atm_call_close, atm_put_close.
    """
    empty: Dict[str, Any] = {
        "straddle_price": None,
        "actual_implied_move": None,
        "atm_strike": None,
        "atm_call_close": None,
        "atm_put_close": None,
    }

    atm_call, atm_put = find_atm_contracts(contracts, underlying_price, expiry)
    if atm_call is None or atm_put is None:
        return empty

    call_price = atm_call.get("day_close")
    put_price = atm_put.get("day_close")
    if call_price is None or put_price is None or call_price <= 0 or put_price <= 0:
        return empty

    straddle = call_price + put_price
    implied_move = straddle / underlying_price if underlying_price > 0 else None

    return {
        "straddle_price": round(straddle, 4),
        "actual_implied_move": round(implied_move, 4) if implied_move else None,
        "atm_strike": atm_call.get("strike_price"),
        "atm_call_close": round(call_price, 4),
        "atm_put_close": round(put_price, 4),
    }


def compute_oi_concentration(
    contracts: List[Dict[str, Any]],
    expiry: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute open interest concentration metrics.

    Distinguishes contracts with missing (None) open_interest from
    contracts with confirmed-zero open_interest: a missing OI means the
    vendor did not report a value (unknown), while a confirmed 0 means the
    contract genuinely has no open positions. Missing-OI contracts are
    excluded from all OI sums/ratios below (not coerced to 0), and their
    count is reported separately so callers can gate on data completeness.

    Returns:
        total_oi, max_oi_strike, oi_concentration (max/total),
        put_oi, call_oi, put_call_oi_ratio, n_contracts, n_oi_missing.
    """
    filtered = contracts
    if expiry:
        filtered = [c for c in contracts if c.get("expiration_date") == expiry]

    total_oi = 0
    max_oi = 0
    max_oi_strike = None
    put_oi = 0
    call_oi = 0
    n_oi_missing = 0

    for c in filtered:
        oi_raw = c.get("open_interest")
        if oi_raw is None:
            n_oi_missing += 1
            continue
        oi = oi_raw
        total_oi += oi
        if oi > max_oi:
            max_oi = oi
            max_oi_strike = c.get("strike_price")
        if c.get("contract_type") == "put":
            put_oi += oi
        elif c.get("contract_type") == "call":
            call_oi += oi

    return {
        "total_oi": total_oi,
        "max_oi": max_oi,
        "max_oi_strike": max_oi_strike,
        "oi_concentration": round(max_oi / total_oi, 4) if total_oi > 0 else None,
        "put_oi": put_oi,
        "call_oi": call_oi,
        "put_call_oi_ratio": round(put_oi / call_oi, 4) if call_oi > 0 else None,
        "n_contracts": len(filtered),
        "n_oi_missing": n_oi_missing,
    }


def compute_volume_by_expiry_bucket(
    contracts: List[Dict[str, Any]],
    as_of_date: str,
) -> Dict[str, int]:
    """Aggregate volume by expiry time bucket.

    Contracts with a missing (None) day_volume are excluded from the bucket
    sums (same numeric contribution as confirmed-zero-volume contracts,
    since both add 0 to the sum), but are counted separately in
    "n_volume_missing" so callers can distinguish "no reported volume" from
    "reported volume of exactly zero" when assessing data completeness.

    Returns: {near_0_30d: vol, mid_31_90d: vol, far_91_180d: vol,
    core_180d_plus: vol, n_volume_missing: int}
    """
    from datetime import date

    try:
        ref = date.fromisoformat(as_of_date)
    except (ValueError, TypeError):
        return {}

    buckets: Dict[str, int] = {
        "near_0_30d": 0,
        "mid_31_90d": 0,
        "far_91_180d": 0,
        "core_180d_plus": 0,
    }
    n_volume_missing = 0

    for c in contracts:
        vol_raw = c.get("day_volume")
        if vol_raw is None:
            n_volume_missing += 1
            continue
        vol = vol_raw
        if vol <= 0:
            continue
        exp_str = c.get("expiration_date", "")
        try:
            exp = date.fromisoformat(exp_str)
        except (ValueError, TypeError):
            continue
        days = (exp - ref).days
        if days <= 30:
            buckets["near_0_30d"] += vol
        elif days <= 90:
            buckets["mid_31_90d"] += vol
        elif days <= 180:
            buckets["far_91_180d"] += vol
        else:
            buckets["core_180d_plus"] += vol

    buckets["n_volume_missing"] = n_volume_missing
    return buckets


def _find_catalyst_aligned_expiry(
    expiries: List[str],
    as_of_date: str,
    catalyst_days: int,
) -> Optional[str]:
    """Find the expiry closest to catalyst_days from as_of_date.

    Returns the expiry whose DTE is closest to catalyst_days, preferring
    expiries AT or AFTER the catalyst date (so the straddle captures the
    event). Returns None if no expiries are within 2x catalyst_days.
    """
    if not expiries or catalyst_days <= 0 or not as_of_date:
        return None
    try:
        from datetime import date as _date

        ref = _date.fromisoformat(as_of_date)
    except (ValueError, TypeError):
        return None

    best_expiry = None
    best_gap = float("inf")
    for exp_str in expiries:
        try:
            exp = _date.fromisoformat(exp_str)
        except (ValueError, TypeError):
            continue
        dte = (exp - ref).days
        if dte < 7:  # too short, gamma-distorted
            continue
        gap: float = abs(dte - catalyst_days)
        # Prefer expiry at or after catalyst (captures event)
        if dte >= catalyst_days:
            gap -= 0.5  # slight preference for post-catalyst
        if gap < best_gap:
            best_gap = gap
            best_expiry = exp_str
    # Only use if within 2x catalyst_days (sanity bound)
    if best_gap > catalyst_days * 2:
        return None
    return best_expiry


def compute_chain_analytics(
    chain_snapshot: List[Dict[str, Any]],
    underlying_price: float,
    as_of_date: str = "",
    target_expiry: Optional[str] = None,
    catalyst_family: str = "",
    catalyst_days: int = 0,
) -> Dict[str, Any]:
    """Compute full chain analytics from a Massive chain snapshot.

    If target_expiry is provided, analytics are computed for that expiry.
    Otherwise, uses the nearest expiry with sufficient contracts.

    When catalyst_days > 0, also computes a catalyst-aligned straddle
    from the expiry closest to the catalyst date.

    Returns dict with all analytics fields.
    """
    if not chain_snapshot or underlying_price <= 0:
        return {"status": "no_data"}

    # Determine expiry
    expiries = sorted(set(c.get("expiration_date", "") for c in chain_snapshot if c.get("expiration_date")))
    if not expiries:
        return {"status": "no_expiries"}

    expiry = target_expiry if target_expiry and target_expiry in expiries else expiries[0]

    # Skew
    put_25d, call_25d = find_25delta_contracts(chain_snapshot, expiry)
    rr_25d = compute_rr_25d(put_25d, call_25d)
    skew = compute_put_call_skew(put_25d, call_25d)

    # Straddle (nearest expiry)
    straddle = compute_atm_straddle(chain_snapshot, underlying_price, expiry)

    # Catalyst-aligned straddle (expiry closest to catalyst date)
    catalyst_straddle: Dict[str, Any] = {}
    if catalyst_days > 0 and as_of_date:
        cat_expiry = _find_catalyst_aligned_expiry(expiries, as_of_date, catalyst_days)
        if cat_expiry and cat_expiry != expiry:
            cat_strad = compute_atm_straddle(chain_snapshot, underlying_price, cat_expiry)
            if cat_strad.get("straddle_price") is not None:
                catalyst_straddle = {
                    "catalyst_straddle_price": cat_strad["straddle_price"],
                    "catalyst_straddle_implied_move": cat_strad["actual_implied_move"],
                    "catalyst_straddle_expiry": cat_expiry,
                }
        elif cat_expiry == expiry and straddle.get("straddle_price") is not None:
            # Nearest expiry IS the catalyst-aligned one
            catalyst_straddle = {
                "catalyst_straddle_price": straddle["straddle_price"],
                "catalyst_straddle_implied_move": straddle["actual_implied_move"],
                "catalyst_straddle_expiry": expiry,
            }

    # OI
    # compute_oi_concentration() also returns "n_contracts" (OI-scope contract
    # count) and "n_oi_missing" (contracts with no reported open_interest).
    # "n_contracts" would collide with this function's own "n_contracts" key
    # (per-expiry contract count, set below) if unpacked directly, so rename
    # it distinctly before merging into `result`.
    oi_raw = compute_oi_concentration(chain_snapshot, expiry)
    oi = {k: v for k, v in oi_raw.items() if k != "n_contracts"}
    oi["n_oi_contracts"] = oi_raw["n_contracts"]

    # Volume by bucket
    # compute_volume_by_expiry_bucket() includes "n_volume_missing" alongside
    # the bucket sums; split it out so total_chain_volume (a sum of actual
    # volume) and volume_by_bucket (pure bucket data) aren't polluted by it.
    vol_buckets_raw = compute_volume_by_expiry_bucket(chain_snapshot, as_of_date) if as_of_date else {}
    n_volume_missing = vol_buckets_raw.pop("n_volume_missing", 0)
    vol_buckets = vol_buckets_raw
    total_vol = sum(vol_buckets.values())
    near_share = round(vol_buckets.get("near_0_30d", 0) / total_vol, 4) if total_vol > 0 else None

    # Spread quality (Stage 2, Spec 045 repair): assess bid/ask spread on the
    # ATM contracts used for the straddle, since that's the pair whose
    # tradability actually matters for the diagnostics this module feeds.
    # MAX_SPREAD_PCT gates "is this chain liquid enough to trust", separate
    # from the OI-based liquidity signal above -- a chain can have healthy OI
    # but a blown-out spread (stale market, no active MM), or vice versa.
    atm_call, atm_put = find_atm_contracts(chain_snapshot, underlying_price, expiry)
    spread_pcts = []
    n_quote_missing = 0
    for _c in (atm_call, atm_put):
        if _c is None:
            continue
        _sp = _c.get("bid_ask_spread_pct")
        if _sp is None:
            # No spread could be computed for this contract: either bid/ask
            # was never populated by the vendor, or was populated in a way
            # that failed compute_bid_ask_spread_pct's sanity checks. Either
            # way this is a MISSING quote, not a confirmed-tight (0%) spread.
            n_quote_missing += 1
        else:
            spread_pcts.append(_sp)
    max_atm_spread_pct = max(spread_pcts) if spread_pcts else None
    spread_gate_pass = (max_atm_spread_pct is not None) and (max_atm_spread_pct <= MAX_SPREAD_PCT)

    # OI floor on the specific ATM contracts used for the straddle (Stage 2,
    # Spec 045 repair): a chain's total_oi can be healthy while the specific
    # ATM call/put pair we quote here is itself thin (OI concentrated in
    # far-OTM/ITM strikes). Missing OI on either ATM leg fails the gate
    # (never treated as OI=0 passing or failing by coincidence).
    atm_ois = []
    n_atm_oi_missing = 0
    for _c in (atm_call, atm_put):
        if _c is None:
            continue
        _oi = _c.get("open_interest")
        if _oi is None:
            n_atm_oi_missing += 1
        else:
            atm_ois.append(_oi)
    min_atm_oi = min(atm_ois) if len(atm_ois) == 2 else None
    oi_gate_pass = (min_atm_oi is not None) and (min_atm_oi >= MIN_OI_THRESHOLD)

    # IV crush stress test
    crush: Dict[str, Any] = {}
    if underlying_price > 0:
        try:
            from common.options_greeks import iv_crush_stress_test

            # Estimate catalyst_days from as_of_date → expiry
            _cat_days = 30  # default
            if as_of_date and expiry:
                from datetime import date as _date

                try:
                    _cat_days = (_date.fromisoformat(expiry) - _date.fromisoformat(as_of_date)).days
                except (ValueError, TypeError):
                    pass
            crush = iv_crush_stress_test(
                chain_snapshot,
                underlying_price,
                max(_cat_days, 1),
                catalyst_family=catalyst_family,
            )
        except Exception:
            pass

    result = {
        "status": "ok",
        "expiry_used": expiry,
        "n_contracts": len([c for c in chain_snapshot if c.get("expiration_date") == expiry]),
        # Skew
        "rr_25d": round(rr_25d, 4) if rr_25d is not None else None,
        "put_call_skew": round(skew, 4) if skew is not None else None,
        "put_25d_iv": put_25d.get("implied_volatility") if put_25d else None,
        "call_25d_iv": call_25d.get("implied_volatility") if call_25d else None,
        # Straddle (nearest expiry)
        **straddle,
        # Catalyst-aligned straddle
        **catalyst_straddle,
        # OI
        **oi,
        # Volume distribution
        "volume_by_bucket": vol_buckets,
        "near_term_volume_share": near_share,
        "total_chain_volume": total_vol,
        "n_volume_missing": n_volume_missing,
        # Spread quality (ATM call/put; None = spread unknown, never fabricated)
        "max_atm_spread_pct": max_atm_spread_pct,
        "spread_gate_pass": spread_gate_pass,
        "n_atm_quote_missing": n_quote_missing,
        # OI floor on ATM call/put (Stage 2, Spec 045 repair; tightens, does
        # not replace, the total-OI-based liquidity signal in `oi` above)
        "min_atm_oi": min_atm_oi,
        "oi_gate_pass": oi_gate_pass,
        "n_atm_oi_missing": n_atm_oi_missing,
    }
    # Add crush metrics (prefixed to avoid key collision)
    if crush.get("confidence") == "ok":
        result["iv_crush_breakeven_pct"] = crush.get("breakeven_move_pct")
        result["crush_adjusted_implied_move"] = crush.get("crush_adjusted_implied_move")
        result["crush_loss_per_contract"] = crush.get("crush_loss_per_contract")
    return result


# ---------------------------------------------------------------------------
# Persistence + warm pass
# ---------------------------------------------------------------------------


def save_chain_snapshot(
    chain: List[Dict[str, Any]],
    snap_dir: Path,
    ticker: str,
) -> Path:
    """Persist raw chain snapshot to disk for downstream consumers."""
    chains_dir = snap_dir / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)
    path = chains_dir / f"{ticker.upper()}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chain, f, default=str)
        f.write("\n")
    return path


def load_chain_snapshot(snap_dir: Path, ticker: str) -> List[Dict[str, Any]]:
    """Load a persisted chain snapshot from disk."""
    path = snap_dir / "chains" / f"{ticker.upper()}.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


_massive_breaker = None


def _get_massive_breaker():
    """Lazy-init circuit breaker for Massive Finance API."""
    global _massive_breaker
    if _massive_breaker is None:
        try:
            from common.robustness_extended import StatefulCircuitBreaker, StatefulCircuitBreakerConfig

            _massive_breaker = StatefulCircuitBreaker(
                "massive_chain_api",
                config=StatefulCircuitBreakerConfig(
                    failure_threshold=10,
                    success_threshold=3,
                    timeout_seconds=180.0,
                    window_size_seconds=600.0,
                ),
            )
        except ImportError:
            _massive_breaker = None
    return _massive_breaker


def warm_chain_analytics(
    tickers: List[str],
    snap_dir: Path,
    as_of_date: str,
    prices: Optional[Dict[str, float]] = None,
    catalyst_days_by_ticker: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Fetch chain snapshots and compute analytics for a list of tickers.

    Args:
        tickers: List of underlying tickers to fetch.
        snap_dir: Snapshot directory to persist chains into.
        as_of_date: Date string for volume bucketing.
        prices: {ticker: last_close} for underlying prices. If None, ATM
                straddle and delta-based analytics may be less accurate.

    Returns:
        {ticker: analytics_dict} for each ticker with data.
    """
    try:
        from common.options_history_massive import fetch_chain_snapshot
    except ImportError:
        logger.warning("options_history_massive not available")
        return {}

    breaker = _get_massive_breaker()
    results: Dict[str, Dict[str, Any]] = {}
    n_fetched = 0
    n_failed = 0

    for ticker in tickers:
        # Circuit breaker: skip remaining tickers if API is consistently failing
        if breaker and not breaker.allow_request():
            logger.warning(
                "[MASSIVE_CHAIN] Circuit breaker OPEN — skipping %s and remaining tickers",
                ticker,
            )
            n_failed += len(tickers) - n_fetched - n_failed
            break

        try:
            chain = fetch_chain_snapshot(ticker.upper(), limit=250)
            if not chain:
                n_failed += 1
                continue

            # Persist raw chain
            save_chain_snapshot(chain, snap_dir, ticker)

            # Compute analytics
            underlying_price = (prices or {}).get(ticker.upper(), 0.0)
            cat_days = (catalyst_days_by_ticker or {}).get(ticker.upper(), 0)
            analytics = compute_chain_analytics(
                chain,
                underlying_price,
                as_of_date,
                catalyst_days=cat_days,
            )
            results[ticker.upper()] = analytics
            n_fetched += 1
            if breaker:
                breaker.record_success()
        except (ConnectionError, TimeoutError, OSError) as exc:
            logger.warning("Chain fetch network error for %s: %s", ticker, exc)
            n_failed += 1
            if breaker:
                breaker.record_failure(exc)
        except Exception as exc:
            logger.warning("Chain fetch failed for %s: %s", ticker, exc)
            n_failed += 1

    if n_failed > 0:
        logger.warning(
            "[MASSIVE_CHAIN] Warmed %d/%d tickers (%d failed)",
            n_fetched,
            len(tickers),
            n_failed,
        )
    else:
        logger.info(
            "[MASSIVE_CHAIN] Warmed %d/%d tickers (0 failed)",
            n_fetched,
            len(tickers),
        )
    return results
