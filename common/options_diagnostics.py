"""Options-implied vol/skew diagnostics from tastytrade.

Passive diagnostic output — does not affect rankings, weights, portfolio
construction, gates, or execution decisions.

Computes a compact set of options diagnostics for catalyst-relevant tickers
using tastytrade market metrics and option chain data.

Required environment variables:
    TT_SECRET   — tastytrade OAuth provider secret
    TT_REFRESH  — tastytrade refresh token

To disable: do not set env vars. The module degrades gracefully to empty
diagnostics with a logged warning.

Output columns (all prefixed ``opt_``):
    opt_has_data          — 1 if options data retrieved, 0 otherwise
    opt_quote_ts          — ISO timestamp of last quote update
    opt_nearest_expiry    — nearest liquid expiry date (YYYY-MM-DD)
    opt_dte               — days to nearest expiry
    opt_atm_iv            — at-the-money implied volatility (annualized)
    opt_front_iv          — front-month expiry IV
    opt_back_iv           — back-month expiry IV (next expiry after front)
    opt_term_slope        — (back_iv - front_iv) / front_iv; positive = contango
    opt_put_call_skew     — (ATM put IV - ATM call IV) / ATM IV;
                            positive = puts more expensive (fear premium)
    opt_rr_25d            — reserved for 25-delta risk reversal; empty in pilot
                            (requires per-strike greeks not in metrics API)
    opt_diagnostic_basis  — short note on data source / fallback reason

Operator flags (derived from raw diagnostics + catalyst context):
    opt_iv_regime         — NORMAL / ELEVATED / EXTREME / "" based on ATM IV
    opt_event_premium     — YES / NO / "" — front > back by > 10% (backwardation)
    opt_liquidity_ok      — 1 / 0 — chain quality sufficient for judgment
    opt_liquidity_state   — liquid / thin / absent — canonical three-state liquidity gate
    opt_use_for_judgment  — YES / NO / "" — composite: has data + liquid + not extreme junk
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import date as _date
from typing import Any, Dict, List, Optional, Tuple

from common.options_quality import MIN_OI_THRESHOLD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

OPTIONS_DIAGNOSTIC_COLUMNS = [
    "opt_has_data",
    "opt_quote_ts",
    "opt_nearest_expiry",
    "opt_dte",
    "opt_atm_iv",
    "opt_front_iv",
    "opt_back_iv",
    "opt_term_slope",
    "opt_put_call_skew",
    "opt_rr_25d",
    "opt_diagnostic_basis",
    # Operator flags
    "opt_iv_regime",
    "opt_event_premium",
    "opt_liquidity_ok",
    "opt_liquidity_state",
    "opt_use_for_judgment",
]

_EMPTY_DIAGNOSTICS: Dict[str, Any] = {col: "" for col in OPTIONS_DIAGNOSTIC_COLUMNS}
_EMPTY_DIAGNOSTICS["opt_has_data"] = "0"
_EMPTY_DIAGNOSTICS["opt_liquidity_ok"] = "0"
_EMPTY_DIAGNOSTICS["opt_liquidity_state"] = "absent"

# Track Massive API authorization failures (e.g., subscription limit)
_massive_auth_failed = False


def empty_diagnostics(reason: str = "") -> Dict[str, Any]:
    """Return empty diagnostic row with optional reason."""
    result = dict(_EMPTY_DIAGNOSTICS)
    result["opt_diagnostic_basis"] = reason
    return result


# ---------------------------------------------------------------------------
# Tastytrade session management
# ---------------------------------------------------------------------------

# Minimum liquidity rating to trust the chain (tastytrade scale: 1-5, 5=best)
MIN_LIQUIDITY_RATING = 1

# Maximum number of symbols per get_market_metrics batch
_METRICS_BATCH_SIZE = 50


def _has_credentials() -> bool:
    """Check if tastytrade credentials are available in environment."""
    return bool(os.environ.get("TT_SECRET") and os.environ.get("TT_REFRESH"))


async def _create_session(is_test: bool = False):
    """Create a tastytrade session. Returns None on failure."""
    try:
        from tastytrade import Session

        session = Session(is_test=is_test)
        return session
    except Exception as exc:
        logger.warning("tastytrade session creation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Per-expiry IV extraction and term structure
# ---------------------------------------------------------------------------


def select_front_back_expiries(
    expiry_ivs: List[Dict[str, Any]],
    as_of_date: _date,
    min_dte: int = 7,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Select front and back expiry from a list of {expiration_date, iv} dicts.

    Front = nearest future expiry with IV data and DTE >= min_dte.
    Back = next expiry after front with IV data.

    Returns (front, back) — either may be None.
    """
    # Filter to future expiries with IV
    candidates = []
    for item in expiry_ivs:
        exp_date = item.get("expiration_date")
        iv = item.get("implied_volatility")
        if exp_date is None or iv is None:
            continue
        if isinstance(exp_date, str):
            try:
                exp_date = _date.fromisoformat(exp_date)
            except (ValueError, TypeError):
                continue
        dte = (exp_date - as_of_date).days
        if dte < min_dte:
            continue
        candidates.append({"expiration_date": exp_date, "implied_volatility": iv, "dte": dte})

    candidates.sort(key=lambda x: x["dte"])

    front = candidates[0] if candidates else None
    back = candidates[1] if len(candidates) > 1 else None
    return front, back


def compute_term_slope(front_iv: float, back_iv: float) -> Optional[float]:
    """Compute term structure slope: (back - front) / front.

    Positive = contango (normal), negative = backwardation (event premium).
    Returns None if front_iv is zero or missing.
    """
    if not front_iv or front_iv <= 0:
        return None
    return round((back_iv - front_iv) / front_iv, 4)


# ---------------------------------------------------------------------------
# Operator flags — derived from raw diagnostics + catalyst context
# ---------------------------------------------------------------------------

# IV regime thresholds (annualized)
_IV_ELEVATED_THRESHOLD = 0.60
_IV_EXTREME_THRESHOLD = 2.00

# Event premium: term slope below this = meaningful backwardation
_EVENT_PREMIUM_SLOPE_THRESHOLD = -0.10

# Liquidity: tastytrade rating >= this is considered usable
# Scale is 1 (worst) to 5 (best); 3+ gives reasonable bid-ask spreads
_LIQUIDITY_OK_THRESHOLD = 3

# IV cap for "use for judgment" — above this, chain is likely too wide to trust
_IV_JUNK_CAP = 5.00


def classify_iv_regime(atm_iv: Optional[float]) -> str:
    """Classify ATM IV into operator-readable regime.

    NORMAL   — IV < 60% (typical for liquid large-cap biotech)
    ELEVATED — 60% <= IV < 200% (catalyst premium, small-cap vol)
    EXTREME  — IV >= 200% (micro-cap, illiquid, or imminent binary event)
    """
    if atm_iv is None:
        return ""
    if atm_iv >= _IV_EXTREME_THRESHOLD:
        return "EXTREME"
    if atm_iv >= _IV_ELEVATED_THRESHOLD:
        return "ELEVATED"
    return "NORMAL"


def classify_event_premium(term_slope: Optional[float]) -> str:
    """Detect event premium from term structure backwardation.

    YES — front IV > back IV by > 10% (slope < -0.10)
    NO  — flat or contango term structure
    """
    if term_slope is None:
        return ""
    return "YES" if term_slope < _EVENT_PREMIUM_SLOPE_THRESHOLD else "NO"


def classify_liquidity_ok(liquidity_rating: Optional[int]) -> str:
    """Return '1' if chain quality is sufficient for operator judgment."""
    if liquidity_rating is None:
        return "0"
    return "1" if liquidity_rating >= _LIQUIDITY_OK_THRESHOLD else "0"


def classify_liquidity_state(has_data: bool, liquidity_ok: bool) -> str:
    """Three-state liquidity gate: liquid / thin / absent.

    liquid  — chain data present with acceptable spreads and OI.
    thin    — chain data present but illiquid (wide spreads or low OI).
    absent  — no options chain data at all.
    """
    if not has_data:
        return "absent"
    return "liquid" if liquidity_ok else "thin"


def get_liquidity_state(row: Dict[str, Any]) -> str:
    """Read opt_liquidity_state from a row, inferring from legacy fields if absent.

    Use this instead of raw row.get("opt_liquidity_state", "absent") when
    reading from snapshots that may predate the field.
    """
    explicit = row.get("opt_liquidity_state", "")
    if explicit in ("liquid", "thin", "absent"):
        return explicit
    # Infer from legacy binary fields
    has_data = row.get("opt_has_data", "0") == "1"
    liq_ok = row.get("opt_liquidity_ok", "0") == "1"
    return classify_liquidity_state(has_data, liq_ok)


def classify_use_for_judgment(
    has_data: bool,
    liquidity_ok: bool,
    atm_iv: Optional[float],
) -> str:
    """Composite gate: should the operator use this for judgment?

    YES requires: has data + liquid chain + IV not in junk territory (>500%).
    Junk IV means the chain is so wide that the numbers are noise.
    """
    if not has_data:
        return "NO"
    if not liquidity_ok:
        return "NO"
    if atm_iv is not None and atm_iv >= _IV_JUNK_CAP:
        return "NO"
    return "YES"


def compute_operator_flags(
    diag: Dict[str, Any],
    liquidity_rating: Optional[int] = None,
) -> Dict[str, str]:
    """Derive operator flags from raw diagnostics.

    Mutates nothing — returns a dict of flag columns to merge.
    """
    has_data = diag.get("opt_has_data") == "1"
    atm_iv = diag.get("opt_atm_iv")
    if isinstance(atm_iv, str) and atm_iv == "":
        atm_iv = None
    elif atm_iv is not None:
        atm_iv = float(atm_iv)

    term_slope = diag.get("opt_term_slope")
    if isinstance(term_slope, str) and term_slope == "":
        term_slope = None
    elif term_slope is not None:
        term_slope = float(term_slope)

    liq_ok = classify_liquidity_ok(liquidity_rating)
    liq_state = classify_liquidity_state(has_data, liq_ok == "1")

    return {
        "opt_iv_regime": classify_iv_regime(atm_iv),
        "opt_event_premium": classify_event_premium(term_slope),
        "opt_liquidity_ok": liq_ok,
        "opt_liquidity_state": liq_state,
        "opt_use_for_judgment": classify_use_for_judgment(has_data, liq_ok == "1", atm_iv),
    }


# ---------------------------------------------------------------------------
# Put/call skew from option chain
# ---------------------------------------------------------------------------


def select_atm_strike(
    strikes: List[Dict[str, Any]],
    spot_price: float,
) -> Optional[Dict[str, Any]]:
    """Select the at-the-money strike nearest to spot price.

    Each strike dict has: strike_price, call (symbol), put (symbol).
    Returns the nearest strike dict, or None.
    """
    if not strikes or not spot_price or spot_price <= 0:
        return None

    best = None
    best_dist = float("inf")
    for s in strikes:
        sp = s.get("strike_price")
        if sp is None:
            continue
        try:
            dist = abs(float(sp) - spot_price)
        except (ValueError, TypeError):
            continue
        if dist < best_dist:
            best_dist = dist
            best = s

    return best


def compute_put_call_skew(
    put_iv: Optional[float],
    call_iv: Optional[float],
) -> Optional[float]:
    """Compute put/call skew at ATM: (put_iv - call_iv) / avg(put_iv, call_iv).

    Positive = puts more expensive than calls (fear premium).
    Returns None if either IV is missing.
    """
    if put_iv is None or call_iv is None:
        return None
    if put_iv <= 0 and call_iv <= 0:
        return None
    avg = (put_iv + call_iv) / 2.0
    if avg <= 0:
        return None
    return round((put_iv - call_iv) / avg, 4)


# ---------------------------------------------------------------------------
# Core async diagnostics fetcher
# ---------------------------------------------------------------------------


async def _fetch_metrics_batch(
    session,
    symbols: List[str],
) -> Dict[str, Any]:
    """Fetch market metrics for a batch of symbols.

    Returns {symbol: MarketMetricInfo} dict.
    """
    from tastytrade.metrics import get_market_metrics

    result = {}
    # Batch in chunks
    n_batches = (len(symbols) + _METRICS_BATCH_SIZE - 1) // _METRICS_BATCH_SIZE
    for i in range(0, len(symbols), _METRICS_BATCH_SIZE):
        batch = symbols[i : i + _METRICS_BATCH_SIZE]
        batch_num = i // _METRICS_BATCH_SIZE + 1
        try:
            metrics = await get_market_metrics(session, batch)
            for m in metrics:
                result[m.symbol] = m
            if batch_num < n_batches:
                await asyncio.sleep(0.5)  # rate-limit courtesy between batches
        except Exception as exc:
            logger.warning(
                "tastytrade metrics batch %d/%d failed for %d symbols: %s", batch_num, n_batches, len(batch), exc
            )
    logger.info("Fetched metrics for %d/%d symbols in %d batches", len(result), len(symbols), n_batches)
    return result


def compute_risk_reversal_25d(
    greeks_by_strike: Dict[float, Dict[str, Any]],
) -> Optional[float]:
    """Compute 25-delta risk reversal: IV(25d put) - IV(25d call).

    Positive = puts more expensive than calls at 25-delta wings.
    Returns None if no suitable strikes found.
    """
    best_call = None  # (strike, delta, iv)
    best_put = None

    for strike, data in greeks_by_strike.items():
        c_d = data.get("call_delta")
        c_iv = data.get("call_iv")
        p_d = data.get("put_delta")
        p_iv = data.get("put_iv")

        if c_d is not None and c_iv is not None and c_iv > 0:
            dist = abs(c_d - 0.25)
            if best_call is None or dist < abs(best_call[1] - 0.25):
                best_call = (strike, c_d, c_iv)

        if p_d is not None and p_iv is not None and p_iv > 0:
            dist = abs(p_d - (-0.25))
            if best_put is None or dist < abs(best_put[1] - (-0.25)):
                best_put = (strike, p_d, p_iv)

    if best_call is None or best_put is None:
        return None
    # Reject if nearest deltas are too far from 25d (>15d away)
    if abs(best_call[1] - 0.25) > 0.15 or abs(best_put[1] - (-0.25)) > 0.15:
        return None
    return round(best_put[2] - best_call[2], 4)


# Timeout for the streaming greeks fetch per symbol (seconds)
_STREAMING_TIMEOUT_PER_SYMBOL = 5.0
# Overall timeout for the entire streaming skew pass
_STREAMING_TIMEOUT_TOTAL = 60.0
# Maximum symbols to fetch skew for (keeps latency bounded)
_SKEW_MAX_SYMBOLS = 80


async def _fetch_skew_for_symbol(
    session,
    streamer,
    symbol: str,
    front_expiry: _date,
    spot_price: float,
) -> Dict[str, Any]:
    """Fetch per-option greeks via DXLinkStreamer for one symbol at one expiry.

    Returns dict with opt_put_call_skew and opt_rr_25d, or empty strings
    on failure/timeout.
    """
    from tastytrade.dxfeed import Greeks
    from tastytrade.instruments import get_option_chain

    empty = {"opt_put_call_skew": "", "opt_rr_25d": ""}

    try:
        chain = await get_option_chain(session, symbol)
    except Exception as exc:
        logger.debug("Skew: no chain for %s: %s", symbol, exc)
        return empty

    strikes_list = chain.get(front_expiry, [])
    if not strikes_list:
        # Try nearest available expiry within 14 days of front
        for exp in sorted(chain.keys()):
            if abs((exp - front_expiry).days) <= 14 and chain[exp]:
                strikes_list = chain[exp]
                break
    if not strikes_list:
        return empty

    # Pair calls and puts by strike
    calls = {}
    puts = {}
    for s in strikes_list:
        sp = float(s.strike_price)
        if "CALL" in str(s.option_type):
            calls[sp] = s
        elif "PUT" in str(s.option_type):
            puts[sp] = s

    common = sorted(set(calls) & set(puts))
    if not common:
        return empty

    # Discover spot from chain if not provided
    if spot_price <= 0:
        spot_price = common[len(common) // 2]

    # Subscribe to greeks for all paired strikes
    sub_syms = []
    for sp in common:
        sub_syms.append(calls[sp].streamer_symbol)
        sub_syms.append(puts[sp].streamer_symbol)

    await streamer.subscribe(Greeks, sub_syms)

    greeks_map: Dict[str, Any] = {}
    for _ in range(len(sub_syms)):
        try:
            g = await asyncio.wait_for(
                streamer.get_event(Greeks),
                timeout=_STREAMING_TIMEOUT_PER_SYMBOL,
            )
            greeks_map[g.event_symbol] = g
        except asyncio.TimeoutError:
            break

    if not greeks_map:
        return empty

    # Build per-strike IV/delta table
    greeks_by_strike: Dict[float, Dict[str, Any]] = {}
    atm_call_iv = None
    atm_put_iv = None
    best_atm_dist = float("inf")

    for sp in common:
        cg = greeks_map.get(calls[sp].streamer_symbol)
        pg = greeks_map.get(puts[sp].streamer_symbol)
        if not cg or not pg:
            continue
        if not cg.volatility or not pg.volatility:
            continue

        c_iv = float(cg.volatility)
        p_iv = float(pg.volatility)
        c_d = float(cg.delta) if cg.delta else None
        p_d = float(pg.delta) if pg.delta else None

        greeks_by_strike[sp] = {
            "call_iv": c_iv,
            "put_iv": p_iv,
            "call_delta": c_d,
            "put_delta": p_d,
        }

        # Track ATM strike
        dist = abs(sp - spot_price)
        if dist < best_atm_dist:
            best_atm_dist = dist
            atm_call_iv = c_iv
            atm_put_iv = p_iv

    result: Dict[str, Any] = {}

    # ATM put/call skew
    skew = compute_put_call_skew(atm_put_iv, atm_call_iv)
    result["opt_put_call_skew"] = round(skew, 4) if skew is not None else ""

    # 25-delta risk reversal
    rr = compute_risk_reversal_25d(greeks_by_strike)
    result["opt_rr_25d"] = round(rr, 4) if rr is not None else ""

    return result


async def _fetch_streaming_skew(
    session,
    diag_results: Dict[str, Dict[str, Any]],
    as_of_date: _date,
) -> None:
    """Second pass: populate opt_put_call_skew and opt_rr_25d via streaming.

    Only fetches for symbols where opt_use_for_judgment == YES and
    opt_has_data == '1'.  Mutates diag_results in place.

    Best-effort: any failure leaves the fields as empty strings (the
    current default).  Bounded by _STREAMING_TIMEOUT_TOTAL.
    """
    from tastytrade.streamer import DXLinkStreamer

    # Select liquid subset
    candidates = [
        sym
        for sym, diag in diag_results.items()
        if diag.get("opt_use_for_judgment") == "YES" and diag.get("opt_has_data") == "1"
    ]
    if not candidates:
        return

    candidates = candidates[:_SKEW_MAX_SYMBOLS]
    logger.info("Streaming skew fetch for %d liquid symbols", len(candidates))

    try:
        async with DXLinkStreamer(session) as streamer:  # type: ignore[var-annotated]
            for symbol in candidates:
                diag = diag_results[symbol]
                front_str = diag.get("opt_nearest_expiry", "")
                try:
                    front_expiry = _date.fromisoformat(front_str)
                except (ValueError, TypeError):
                    continue

                try:
                    skew_data = await asyncio.wait_for(
                        _fetch_skew_for_symbol(
                            session,
                            streamer,
                            symbol,
                            front_expiry,
                            spot_price=0.0,  # helper discovers spot from chain
                        ),
                        timeout=_STREAMING_TIMEOUT_PER_SYMBOL * 3,
                    )
                    diag.update(skew_data)
                    if skew_data.get("opt_put_call_skew") != "":
                        logger.debug(
                            "Skew for %s: skew=%s rr25d=%s",
                            symbol,
                            skew_data["opt_put_call_skew"],
                            skew_data["opt_rr_25d"],
                        )
                except asyncio.TimeoutError:
                    logger.debug("Skew timeout for %s", symbol)
                except Exception as exc:
                    logger.debug("Skew error for %s: %s", symbol, exc)

    except Exception as exc:
        logger.warning("Streaming skew pass failed: %s", exc)


async def _fetch_diagnostics_async(
    symbols: List[str],
    as_of_date: str,
    is_test: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Fetch options diagnostics for a list of symbols.

    Returns {symbol: diagnostics_dict}.
    """
    if not symbols:
        return {}

    session = await _create_session(is_test=is_test)
    if session is None:
        return {s: empty_diagnostics("no_session") for s in symbols}

    try:
        ref_date = _date.fromisoformat(as_of_date)
    except (ValueError, TypeError):
        ref_date = _date.today()

    async with session:
        # Batch fetch market metrics
        metrics = await _fetch_metrics_batch(session, symbols)

        result: Dict[str, Dict[str, Any]] = {}
        for symbol in symbols:
            m = metrics.get(symbol)
            if m is None:
                result[symbol] = empty_diagnostics("no_metrics")
                continue

            # Check liquidity
            liq = m.liquidity_rating
            if liq is not None and liq < MIN_LIQUIDITY_RATING:
                result[symbol] = empty_diagnostics(f"low_liquidity_{liq}")
                continue

            # Extract per-expiry IV
            expiry_ivs = []
            if m.option_expiration_implied_volatilities:
                for oiv in m.option_expiration_implied_volatilities:
                    if oiv.implied_volatility is not None:
                        expiry_ivs.append(
                            {
                                "expiration_date": oiv.expiration_date,
                                "implied_volatility": float(oiv.implied_volatility),
                            }
                        )

            front, back = select_front_back_expiries(expiry_ivs, ref_date, min_dte=7)

            # DTE relaxation: retry with min_dte=3 for weekly-only names
            basis = "tt_market_metrics"
            if not front and expiry_ivs:
                front, back = select_front_back_expiries(expiry_ivs, ref_date, min_dte=3)
                if front:
                    basis = "tt_weekly_fallback"

            if not front:
                result[symbol] = empty_diagnostics("no_liquid_expiry")
                continue

            # Detect event premium in sub-7-DTE expiries that were filtered out.
            # When a weekly drops below min_dte=7, the event premium signal
            # disappears from the primary term structure. This flag captures it.
            _nearby_event_premium = False
            if front["dte"] >= 7:
                for eiv in expiry_ivs:
                    exp_d = eiv.get("expiration_date")
                    iv_val = eiv.get("implied_volatility")
                    if exp_d is None or iv_val is None:
                        continue
                    if isinstance(exp_d, str):
                        try:
                            exp_d = _date.fromisoformat(exp_d)
                        except (ValueError, TypeError):
                            continue
                    nearby_dte = (exp_d - ref_date).days
                    if 0 < nearby_dte < 7 and iv_val > front["implied_volatility"] * 1.3:
                        _nearby_event_premium = True
                        break

            atm_iv = float(m.implied_volatility_index) if m.implied_volatility_index is not None else None
            # TT IVx occasionally returns values >= _IV_JUNK_CAP that are inconsistent
            # with the per-contract IV scale used elsewhere (front["implied_volatility"]
            # is always a decimal fraction).  Fall back to front_iv so that downstream
            # straddle scoring stays in a consistent unit.
            if atm_iv is not None and atm_iv >= _IV_JUNK_CAP:
                atm_iv = None
            front_iv = front["implied_volatility"]
            back_iv = back["implied_volatility"] if back else None
            term_slope = compute_term_slope(front_iv, back_iv) if back_iv else None

            quote_ts = ""
            if m.implied_volatility_updated_at:
                quote_ts = m.implied_volatility_updated_at.isoformat()
            elif m.updated_at:
                quote_ts = m.updated_at.isoformat()

            # Event premium: prefer primary term structure, fall back to nearby detection
            _ep_primary = classify_event_premium(term_slope)
            _ep_final = _ep_primary
            if _ep_primary == "NO" and _nearby_event_premium:
                _ep_final = "NEARBY"  # event premium in sub-7-DTE expiry, not in primary structure

            diag: Dict[str, Any] = {
                "opt_has_data": "1",
                "opt_quote_ts": quote_ts,
                "opt_nearest_expiry": front["expiration_date"].isoformat(),
                "opt_dte": front["dte"],
                "opt_atm_iv": round(atm_iv, 4) if atm_iv is not None else round(front_iv, 4),
                "opt_front_iv": round(front_iv, 4),
                "opt_back_iv": round(back_iv, 4) if back_iv is not None else "",
                "opt_term_slope": term_slope if term_slope is not None else "",
                "opt_put_call_skew": "",  # populated by streaming pass if liquid
                "opt_rr_25d": "",  # populated by streaming pass if liquid
                "opt_diagnostic_basis": basis,
            }
            # Derive operator flags
            diag.update(compute_operator_flags(diag, liquidity_rating=liq))
            # Override event premium with nearby-expiry detection
            if _ep_final != _ep_primary:
                diag["opt_event_premium"] = _ep_final
            result[symbol] = diag

        # Second pass: streaming skew for liquid subset
        try:
            await asyncio.wait_for(
                _fetch_streaming_skew(session, result, ref_date),
                timeout=_STREAMING_TIMEOUT_TOTAL,
            )
            skew_filled = sum(1 for d in result.values() if d.get("opt_put_call_skew") != "")
            if skew_filled:
                logger.info("Streaming skew: %d/%d symbols populated", skew_filled, len(result))
        except asyncio.TimeoutError:
            logger.warning("Streaming skew pass timed out after %ds", _STREAMING_TIMEOUT_TOTAL)
        except Exception as exc:
            logger.warning("Streaming skew pass failed: %s", exc)

        return result


# ---------------------------------------------------------------------------
# Public synchronous API
# ---------------------------------------------------------------------------


def fetch_options_diagnostics(
    symbols: List[str],
    as_of_date: str,
    is_test: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Fetch options diagnostics for a list of symbols (synchronous wrapper).

    Non-blocking: returns empty diagnostics on any failure.

    Parameters
    ----------
    symbols : list of ticker symbols (uppercase)
    as_of_date : evaluation date (YYYY-MM-DD)
    is_test : use tastytrade sandbox

    Returns
    -------
    {symbol: {opt_has_data, opt_quote_ts, ...}} for each symbol.
    Symbols without data get empty diagnostics with reason.
    """
    if not _has_credentials():
        logger.info("tastytrade credentials not set (TT_SECRET / TT_REFRESH) — options diagnostics disabled")
        return {s: empty_diagnostics("no_credentials") for s in symbols}

    if not symbols:
        return {}

    try:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_fetch_diagnostics_async(symbols, as_of_date, is_test))
        finally:
            loop.close()
    except Exception as exc:
        logger.warning("Options diagnostics fetch failed: %s", exc)
        return {s: empty_diagnostics(f"fetch_error: {exc}") for s in symbols}


def select_catalyst_tickers(
    rankings: List[Dict[str, Any]],
    max_tickers: int = 400,
) -> List[str]:
    """Select tickers eligible for options diagnostics from rankings.

    Fetches ALL ranked tickers to maximize options coverage. Priority ordering
    ensures the most important names are fetched first if the API cap is hit.

    Priority (lower = fetched first):
    1. Has an actionable_rank (ranked names)
    2. Has catalyst_days <= 180 (upcoming catalyst)
    3. All remaining universe tickers

    Returns sorted list of uppercase ticker symbols.
    """
    scored: List[Tuple[float, str]] = []
    for row in rankings:
        ticker = (row.get("ticker") or "").upper()
        if not ticker:
            continue

        # Priority: lower = more important
        ar = row.get("actionable_rank", "")
        cat_days = row.get("catalyst_days", "")

        try:
            ar_val = float(ar) if ar != "" else 999
        except (ValueError, TypeError):
            ar_val = 999

        try:
            cd_val = float(cat_days) if cat_days != "" else 999
        except (ValueError, TypeError):
            cd_val = 999

        # Score: actionable names first, then by catalyst proximity, then everyone else
        priority = ar_val * 0.1 + cd_val * 0.01
        scored.append((priority, ticker))

    scored.sort()
    tickers = [t for _, t in scored[:max_tickers]]
    return sorted(set(tickers))


# ---------------------------------------------------------------------------
# Massive/Polygon fallback chain
# ---------------------------------------------------------------------------


def _massive_fallback_batch(
    symbols: List[str],
    as_of_date: str,
) -> Dict[str, Dict[str, Any]]:
    """Fetch diagnostics from Massive/Polygon for tickers TT doesn't cover.

    Returns diagnostic dicts with opt_diagnostic_basis="massive_chain_snapshot".
    Gracefully degrades to empty diagnostics if Massive is unavailable.

    NOTE: If Massive returns NOT_AUTHORIZED (subscription limit), disables fallback
    for this run and returns empty diagnostics. Tastytrade will be primary source.
    """
    global _massive_auth_failed

    results: Dict[str, Dict[str, Any]] = {}

    # Check if Massive auth already failed in a prior call
    if _massive_auth_failed:
        logger.info("Massive authorization failed in prior attempt -- using Tastytrade only")
        return {s: empty_diagnostics("massive_auth_failed") for s in symbols}

    try:

        from common.options_history_massive import fetch_chain_snapshot
    except ImportError:
        logger.info("Massive/Polygon modules not available -- fallback disabled")
        return {s: empty_diagnostics("no_massive_module") for s in symbols}

    ref_date = _date.fromisoformat(as_of_date)

    for symbol in symbols:
        try:
            chain = fetch_chain_snapshot(symbol)
            if not chain:
                results[symbol] = empty_diagnostics("no_chain_any_source")
                continue

            # Aggregate per-expiry: use median IV weighted toward ATM as proxy
            from collections import defaultdict as _ddict

            expiry_contracts: Dict[str, List[float]] = _ddict(list)
            total_oi = 0
            n_oi_missing = 0  # contracts where open_interest is None (no data), not confirmed 0
            for contract in chain:
                exp = contract.get("expiration_date", "")
                iv = contract.get("implied_volatility")
                oi_raw = contract.get("open_interest")
                if oi_raw is None:
                    n_oi_missing += 1
                else:
                    total_oi += oi_raw
                if exp and iv is not None and 0.01 < float(iv) < 10.0:
                    expiry_contracts[exp].append(float(iv))

            # Per-expiry median IV (robust to OTM extremes)
            expiry_ivs = []
            for exp, ivs in expiry_contracts.items():
                ivs_sorted = sorted(ivs)
                median_iv = ivs_sorted[len(ivs_sorted) // 2]
                expiry_ivs.append({"expiration_date": exp, "implied_volatility": median_iv})

            front, back = select_front_back_expiries(expiry_ivs, ref_date, min_dte=3)
            if not front:
                results[symbol] = empty_diagnostics("no_chain_any_source")
                continue

            # If open_interest was missing (None) on every contract in the chain,
            # we cannot assess liquidity at all -- this is distinct from a chain
            # with confirmed-zero OI (e.g. a brand-new or abandoned contract).
            n_contracts_total = len(chain)
            if n_contracts_total > 0 and n_oi_missing == n_contracts_total:
                results[symbol] = empty_diagnostics("bad_quote")
                continue

            # Compute ATM IV from front expiry
            front_iv = front["implied_volatility"]
            back_iv = back["implied_volatility"] if back else None
            term_slope = None
            if back_iv and front_iv and front_iv > 0:
                term_slope = round((back_iv - front_iv) / front_iv, 4)

            # Synthetic liquidity from total OI (Stage 2, Spec 045 repair):
            # tighten the existing total-OI thresholds with an additional
            # MIN_OI_THRESHOLD-derived floor on OI per contract actually
            # observed (contracts with missing OI are excluded from the
            # denominator, not treated as 0-OI contracts). This can only
            # make the gate stricter (move liquid/thin -> thin/absent),
            # never looser.
            n_contracts_with_oi = n_contracts_total - n_oi_missing
            avg_oi_per_contract = (total_oi / n_contracts_with_oi) if n_contracts_with_oi > 0 else 0.0
            meets_min_oi_floor = avg_oi_per_contract >= MIN_OI_THRESHOLD

            if total_oi > 5000 and meets_min_oi_floor:
                liq_state = "liquid"
                liq_ok = "1"
            elif total_oi > 500 and meets_min_oi_floor:
                liq_state = "thin"
                liq_ok = "0"
            else:
                liq_state = "absent"
                liq_ok = "0"

            # Polygon median IV runs ~1.5-2x higher than TT IVx for same name.
            # Apply source-aware regime thresholds and cap extreme IVs.
            capped_iv = min(front_iv, 8.0)  # Cap at 800% for display
            iv_regime = "EXTREME" if capped_iv >= 4.0 else "ELEVATED" if capped_iv >= 1.20 else "NORMAL"
            # Event premium: Polygon median-IV term slopes are systematically
            # more negative than TT IVx (~80% < -0.20 vs TT's 28%). Calibrate
            # to match TT's event premium rate: -0.50 threshold gives ~28%.
            has_event_premium = term_slope is not None and term_slope < -0.50

            diag: Dict[str, Any] = {
                "opt_has_data": "1",
                "opt_quote_ts": "",
                "opt_nearest_expiry": (
                    front["expiration_date"].isoformat()
                    if isinstance(front["expiration_date"], _date)
                    else str(front["expiration_date"])
                ),
                "opt_dte": front["dte"],
                "opt_atm_iv": round(capped_iv, 4),
                "opt_front_iv": round(capped_iv, 4),
                "opt_back_iv": round(min(back_iv, 8.0), 4) if back_iv is not None else "",
                "opt_term_slope": term_slope if term_slope is not None else "",
                "opt_put_call_skew": "",
                "opt_rr_25d": "",
                "opt_diagnostic_basis": "massive_chain_snapshot",
                "opt_iv_regime": iv_regime,
                "opt_event_premium": "YES" if has_event_premium else "NO",
                "opt_liquidity_ok": liq_ok,
                "opt_liquidity_state": liq_state,
                "opt_use_for_judgment": "YES" if liq_ok == "1" and capped_iv < 5.0 else "NO",
                "opt_dte_warning": "short_dated" if front["dte"] < 7 else "",
            }
            results[symbol] = diag

        except Exception as exc:
            # Check for authorization errors (e.g., NOT_AUTHORIZED)
            exc_str = str(exc).lower()
            if "not_authorized" in exc_str or "not entitled" in exc_str:
                _massive_auth_failed = True
                logger.error(
                    "Massive API authorization failed (subscription limit?). "
                    "Disabling Massive fallback for this run. Using Tastytrade primary: %s",
                    exc,
                )
                # Return empty diagnostics for all remaining symbols
                for s in symbols:
                    if s not in results:
                        results[s] = empty_diagnostics("massive_auth_failed")
                break
            else:
                logger.warning("Massive fallback failed for %s: %s", symbol, exc)
                results[symbol] = empty_diagnostics(f"massive_error: {exc}")

    n_filled = sum(1 for d in results.values() if d.get("opt_has_data") == "1")
    logger.info("Massive fallback: %d/%d symbols got data", n_filled, len(symbols))
    return results


def _classify_iv_regime(iv: float) -> str:
    """Classify IV into regime bucket."""
    if iv >= 2.00:
        return "EXTREME"
    if iv >= 0.60:
        return "ELEVATED"
    return "NORMAL"


def fetch_options_with_fallback(
    symbols: List[str],
    as_of_date: str,
    is_test: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Fetch options diagnostics with Massive/Polygon fallback.

    1. Call fetch_options_diagnostics (Tastytrade)
    2. For absent tickers (no_metrics, no_credentials), try Massive
    3. Merge results

    PIT safeguard: when as_of_date is historical (>= 2 days before today),
    both Tastytrade and Massive fetch_chain_snapshot return CURRENT-state
    option chains, which would stamp today's IVs onto a historical row. No
    PIT-correct historical chain builder exists yet (that would require
    IV back-out from Massive day_aggs flat files). Gate the overlay off
    and return empty diagnostics with basis="no_historical_chain" instead
    of contaminating historical snapshots.

    Returns same schema as fetch_options_diagnostics.
    """
    try:
        _as_of = _date.fromisoformat(as_of_date)
        _today = _date.today()
        _is_historical = (_today - _as_of).days >= 2
    except (ValueError, TypeError):
        _is_historical = False

    if _is_historical:
        logger.info(
            "Options overlay gated off for historical as_of_date=%s "
            "(no PIT-correct chain source; see common/options_diagnostics.py:fetch_options_with_fallback)",
            as_of_date,
        )
        return {s: empty_diagnostics("no_historical_chain") for s in symbols}

    # Primary: Tastytrade (current / live runs only)
    logger.info("[OPTIONS_WARM] Starting Tastytrade fetch for %d symbols", len(symbols))
    result = fetch_options_diagnostics(symbols, as_of_date, is_test)

    # Identify absent tickers
    absent = [
        s
        for s in symbols
        if result.get(s, {}).get("opt_has_data") != "1"
        and result.get(s, {}).get("opt_diagnostic_basis", "") in ("no_metrics", "no_credentials", "")
    ]

    tt_filled = len(symbols) - len(absent)
    logger.info("[OPTIONS_WARM] Tastytrade: %d/%d symbols got data", tt_filled, len(symbols))

    if not absent:
        logger.info("[OPTIONS_WARM] All symbols covered by Tastytrade, skipping Massive fallback")
        return result

    # Fallback to Massive for absent tickers (will handle auth failures gracefully)
    logger.info("[OPTIONS_WARM] Attempting Massive fallback for %d absent tickers", len(absent))
    fallback = _massive_fallback_batch(absent, as_of_date)

    # Merge: only replace if fallback got data
    for s, diag in fallback.items():
        if diag.get("opt_has_data") == "1":
            result[s] = diag

    # Final summary
    final_filled = sum(1 for d in result.values() if d.get("opt_has_data") == "1")
    logger.info("[OPTIONS_WARM] Final coverage: %d/%d symbols", final_filled, len(symbols))

    return result
