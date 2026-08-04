"""CRT × Options join table — prep for catalyst EV model.

For each resolved CRT record, joins:
  - options surface state at time of prediction
  - event premium decomposition
  - implied vs realized move (1d, 5d, h20, h63)
  - price reaction
  - cohort flags (hard catalyst, regulatory, liquid)

This is the foundation for the eventual:
  predicted_hit_prob × implied_move × historical_hit_rate

Usage:
    python scripts/research/build_crt_options_join.py
"""

from __future__ import annotations

import bisect
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.hard_catalyst import is_hard_catalyst
from common.options_diagnostics import get_liquidity_state

SNAPSHOT_DIR = REPO_ROOT / "data" / "snapshots"
RESOLUTION_DIR = SNAPSHOT_DIR / "resolutions"
OUTPUT_DIR = REPO_ROOT / "output" / "catalyst_ev"
PRICE_CSV = REPO_ROOT / "production_data" / "price_history.csv"

# Regulatory catalyst types (subset of hard catalysts)
_REGULATORY_TYPES = frozenset(
    {
        "PDUFA_ACTION",
        "ADVISORY_COMMITTEE",
        "NDA_BLA_FILING",
        "REGULATORY_DESIGNATION",
        # Lowercase variants for robustness
        "pdufa",
        "fda_pdufa_date",
        "fda_decision",
        "fda_approval",
        "advisory_committee",
        "fda_adcom",
        "regulatory_decision",
        "approval_decision",
    }
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("crt_options_join")


def _sf(v) -> float:
    if v is None or v == "" or v == "None":
        return float("nan")
    try:
        return float(v)
    except (ValueError, TypeError):
        return float("nan")


# ---------------------------------------------------------------------------
# Price history loader (mirrors options_prospective_analysis.py)
# ---------------------------------------------------------------------------


def _load_prices(csv_path: Path) -> tuple[Dict[str, Dict[str, float]], List[str]]:
    """Load price_history.csv -> ({ticker: {date_str: close}}, sorted_dates)."""
    prices: Dict[str, Dict[str, float]] = {}
    all_dates: set[str] = set()
    if not csv_path.exists():
        log.warning("Price history not found: %s", csv_path)
        return prices, []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            close_str = (row.get("close") or "").strip()
            date_str = (row.get("date") or "").strip()
            if not ticker or not close_str or not date_str:
                continue
            try:
                close = float(close_str)
            except (ValueError, TypeError):
                continue
            prices.setdefault(ticker, {})[date_str] = close
            all_dates.add(date_str)
    return prices, sorted(all_dates)


def _horizon_return(
    ticker_prices: Dict[str, float],
    sorted_dates: List[str],
    anchor_date: str,
    horizon: int,
) -> Optional[float]:
    """Forward return from anchor_date + horizon trading days.

    anchor_date should be the day before the catalyst (price_t_minus_1 date).
    We find the first trading day on or after anchor_date as the base.
    """
    trade_date = None
    for d in sorted_dates:
        if d >= anchor_date:
            trade_date = d
            break
    if trade_date is None:
        return None
    p0 = ticker_prices.get(trade_date)
    if p0 is None or p0 <= 0:
        return None
    try:
        idx = sorted_dates.index(trade_date)
    except ValueError:
        return None
    target_idx = idx + horizon
    if target_idx >= len(sorted_dates):
        return None
    end_date = sorted_dates[target_idx]
    p1 = ticker_prices.get(end_date)
    if p1 is None or p1 <= 0:
        return None
    return p1 / p0 - 1.0


# ---------------------------------------------------------------------------
# CRT loaders
# ---------------------------------------------------------------------------


def load_crt_resolutions() -> list[dict]:
    """Load all CRT resolution records."""
    records = []
    if not RESOLUTION_DIR.exists():
        return records
    for month_dir in sorted(RESOLUTION_DIR.iterdir()):
        if not month_dir.is_dir() or not month_dir.name[:4].isdigit():
            continue
        for f in sorted(month_dir.glob("*.json")):
            try:
                with open(f, encoding="utf-8") as fh:
                    rec = json.load(fh)
                if rec.get("outcome") and rec.get("outcome") not in ("INFORMATIONAL",):
                    records.append(rec)
            except (json.JSONDecodeError, OSError):
                pass
    return records


_SNAPSHOT_DATES_CACHE: List[str] | None = None


def _snapshot_dates_with_rankings() -> List[str]:
    """Sorted names of every snapshot dir containing a rankings.csv.

    Scanned once per process. The previous implementation re-walked all ~223
    snapshot dirs and stat'd rankings.csv in each on every call; at ~1.05 s per
    call over 257 CRT resolutions that alone projected to ~270 s and guaranteed
    the caller's 120 s timeout. Directory contents do not change mid-run, so a
    single scan is equivalent.
    """
    global _SNAPSHOT_DATES_CACHE
    if _SNAPSHOT_DATES_CACHE is None:
        _SNAPSHOT_DATES_CACHE = sorted(
            d.name for d in SNAPSHOT_DIR.iterdir() if d.is_dir() and (d / "rankings.csv").exists()
        )
    return _SNAPSHOT_DATES_CACHE


def find_prediction_snapshot(ticker: str, catalyst_date: str) -> str | None:
    """Find the most recent snapshot before the catalyst date for this ticker.

    ``ticker`` is unused (kept for call-site compatibility) — the snapshot choice
    depends only on ``catalyst_date``.
    """
    dates = _snapshot_dates_with_rankings()
    # Largest name strictly less than catalyst_date, matching the original
    # string comparison and max() semantics exactly.
    idx = bisect.bisect_left(dates, catalyst_date)
    if idx == 0:
        return None
    return dates[idx - 1]


_OPTIONS_BY_SNAPSHOT: Dict[str, Dict[str, dict]] = {}


def _options_index(snapshot_date: str) -> Dict[str, dict]:
    """Parse one snapshot's rankings.csv into {TICKER: options fields}, memoized.

    Previously each call re-parsed the whole ~890 KB / 341-column rankings.csv
    and linear-scanned it for a single ticker. Records frequently share a
    snapshot, so parsing once per snapshot removes the repeated work while
    returning identical values. First row wins on duplicate tickers, matching
    the original early-return behaviour.
    """
    cached = _OPTIONS_BY_SNAPSHOT.get(snapshot_date)
    if cached is not None:
        return cached
    index: Dict[str, dict] = {}
    rpath = SNAPSHOT_DIR / snapshot_date / "rankings.csv"
    if rpath.exists():
        with open(rpath, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = row.get("ticker", "").upper()
                if key and key not in index:
                    index[key] = {
                        "opt_has_data": row.get("opt_has_data", ""),
                        "opt_atm_iv": _sf(row.get("opt_atm_iv")),
                        "opt_front_iv": _sf(row.get("opt_front_iv")),
                        "opt_back_iv": _sf(row.get("opt_back_iv")),
                        "opt_rr_25d": _sf(row.get("opt_rr_25d")),
                        "opt_term_slope": _sf(row.get("opt_term_slope")),
                        "opt_event_premium": row.get("opt_event_premium", ""),
                        "opt_iv_regime": row.get("opt_iv_regime", ""),
                        "opt_liquidity_ok": row.get("opt_liquidity_ok", ""),
                        "opt_liquidity_state": get_liquidity_state(row),
                        "actual_implied_move_pctile": _sf(row.get("actual_implied_move_pctile")),
                        "implied_event_move": _sf(row.get("implied_event_move")),
                        "opt_dte": _sf(row.get("opt_dte")),
                        "catalyst_days": _sf(row.get("catalyst_days")),
                        "actionable_rank": row.get("actionable_rank", ""),
                        "tier_any": row.get("tier_any", ""),
                    }
    _OPTIONS_BY_SNAPSHOT[snapshot_date] = index
    return index


def load_options_at_prediction(ticker: str, snapshot_date: str) -> dict:
    """Load options fields for one ticker from the prediction-date snapshot."""
    return _options_index(snapshot_date).get(ticker.upper(), {})


# ---------------------------------------------------------------------------
# Anchor date helper
# ---------------------------------------------------------------------------


def _infer_anchor_date(rec: dict) -> str | None:
    """Infer the anchor date (day before catalyst) for price lookups.

    Uses prediction_snapshot_date as proxy — it's the most recent snapshot
    before the catalyst, which is typically 1 trading day before.
    Falls back to catalyst_date - 1 calendar day.
    """
    snap = rec.get("prediction_snapshot_date", "")
    if snap:
        return snap
    cat = rec.get("catalyst_date", "")
    if not cat:
        return None
    # Crude fallback: subtract one calendar day
    from datetime import date as _d
    from datetime import timedelta

    try:
        dt = _d.fromisoformat(cat)
        return (dt - timedelta(days=1)).isoformat()
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Join builder
# ---------------------------------------------------------------------------


def build_join_table() -> dict:
    """Build the CRT × options join table with realized returns."""
    resolutions = load_crt_resolutions()
    log.info("Loaded %d CRT resolutions", len(resolutions))

    # Load price history for multi-horizon returns
    prices, all_dates = _load_prices(PRICE_CSV)
    log.info("Loaded price history: %d tickers, %d dates", len(prices), len(all_dates))

    joined = []
    for rec in resolutions:
        ticker = rec.get("ticker", "")
        catalyst_date = rec.get("catalyst_date", "")
        catalyst_type = rec.get("catalyst_type", "")
        if not ticker or not catalyst_date:
            continue

        # Find prediction snapshot
        snap_date = find_prediction_snapshot(ticker, catalyst_date)
        if not snap_date:
            continue

        # Load options state at prediction time
        opts = load_options_at_prediction(ticker, snap_date)

        # -----------------------------------------------------------------
        # Realized returns from CRT price fields
        # -----------------------------------------------------------------
        pt_minus_1 = _sf(rec.get("price_t_minus_1"))
        pt_0 = _sf(rec.get("price_t_0"))
        pt_plus_5 = _sf(rec.get("price_t_plus_5"))

        # 1-day return (event day vs prior close)
        if not math.isnan(pt_minus_1) and pt_minus_1 > 0 and not math.isnan(pt_0):
            realized_1d = pt_0 / pt_minus_1 - 1.0
        else:
            realized_1d = float("nan")

        # 5-day return (t+5 vs prior close)
        if not math.isnan(pt_minus_1) and pt_minus_1 > 0 and not math.isnan(pt_plus_5):
            realized_5d = pt_plus_5 / pt_minus_1 - 1.0
        else:
            realized_5d = float("nan")

        # -----------------------------------------------------------------
        # Multi-horizon returns from price_history.csv
        # -----------------------------------------------------------------
        anchor = _infer_anchor_date(rec)
        ticker_prices = prices.get(ticker.upper(), {})

        realized_h20 = None
        realized_h63 = None
        if anchor and ticker_prices:
            realized_h20 = _horizon_return(ticker_prices, all_dates, anchor, 20)
            realized_h63 = _horizon_return(ticker_prices, all_dates, anchor, 63)

        # -----------------------------------------------------------------
        # Implied vs realized
        # -----------------------------------------------------------------
        implied_move = opts.get("implied_event_move", float("nan"))
        if math.isnan(implied_move):
            implied_move = float("nan")

        implied_vs_realized_1d = None
        if not math.isnan(implied_move) and implied_move > 0 and not math.isnan(realized_1d):
            implied_vs_realized_1d = abs(realized_1d) / implied_move

        # Event premium ratio
        front_iv = opts.get("opt_front_iv", float("nan"))
        back_iv = opts.get("opt_back_iv", float("nan"))
        epr = None
        if not math.isnan(front_iv) and not math.isnan(back_iv) and back_iv > 0:
            epr = front_iv / back_iv

        # -----------------------------------------------------------------
        # Cohort flags
        # -----------------------------------------------------------------
        hard = is_hard_catalyst(catalyst_type)
        regulatory = catalyst_type.lower() in _REGULATORY_TYPES or catalyst_type in _REGULATORY_TYPES
        liquid = opts.get("opt_liquidity_state", "absent") == "liquid"

        # -----------------------------------------------------------------
        # Build joined record
        # -----------------------------------------------------------------
        joined.append(
            {
                "ticker": ticker,
                "catalyst_date": catalyst_date,
                "catalyst_type": catalyst_type,
                "outcome": rec.get("outcome", ""),
                "price_direction": rec.get("price_direction", ""),
                "prediction_dem_rank": rec.get("prediction_dem_rank"),
                "prediction_tier": rec.get("prediction_tier", ""),
                "prediction_snapshot_date": snap_date,
                # Options at prediction time
                "opt_has_data": opts.get("opt_has_data", ""),
                "opt_atm_iv": opts.get("opt_atm_iv"),
                "opt_rr_25d": opts.get("opt_rr_25d"),
                "opt_event_premium": opts.get("opt_event_premium", ""),
                "opt_iv_regime": opts.get("opt_iv_regime", ""),
                "opt_liquidity_ok": opts.get("opt_liquidity_ok", ""),
                "opt_liquidity_state": opts.get("opt_liquidity_state", "absent"),
                "implied_event_move": implied_move if not math.isnan(implied_move) else None,
                "actual_implied_move_pctile": opts.get("actual_implied_move_pctile"),
                "event_premium_ratio": epr,
                "catalyst_days_at_prediction": opts.get("catalyst_days"),
                # Realized returns
                "realized_1d_return": round(realized_1d, 6) if not math.isnan(realized_1d) else None,
                "realized_abs_1d_return": round(abs(realized_1d), 6) if not math.isnan(realized_1d) else None,
                "realized_5d_return": round(realized_5d, 6) if not math.isnan(realized_5d) else None,
                "realized_h20_return": round(realized_h20, 6) if realized_h20 is not None else None,
                "realized_h63_return": round(realized_h63, 6) if realized_h63 is not None else None,
                "implied_vs_realized_1d": (
                    round(implied_vs_realized_1d, 4) if implied_vs_realized_1d is not None else None
                ),
                "market_overpriced_1d": (implied_vs_realized_1d > 1.0 if implied_vs_realized_1d is not None else None),
                # Cohort flags
                "is_hard_catalyst": hard,
                "is_regulatory": regulatory,
                "is_liquid": liquid,
            }
        )

    # Summary stats
    n_with_options = sum(1 for j in joined if j.get("opt_has_data") == "1")
    n_with_implied = sum(1 for j in joined if j.get("implied_event_move") is not None)
    n_with_realized_1d = sum(1 for j in joined if j.get("realized_1d_return") is not None)
    n_with_realized_5d = sum(1 for j in joined if j.get("realized_5d_return") is not None)
    n_with_realized_h20 = sum(1 for j in joined if j.get("realized_h20_return") is not None)
    n_with_realized_h63 = sum(1 for j in joined if j.get("realized_h63_return") is not None)
    n_overpriced = sum(1 for j in joined if j.get("market_overpriced_1d") is True)
    n_underpriced = sum(1 for j in joined if j.get("market_overpriced_1d") is False)
    n_liquid = sum(1 for j in joined if j.get("is_liquid"))
    n_hard = sum(1 for j in joined if j.get("is_hard_catalyst"))

    return {
        "schema": "crt_options_join.v2",
        "n_resolutions": len(joined),
        "n_with_options": n_with_options,
        "n_with_implied_move": n_with_implied,
        "n_with_realized_1d": n_with_realized_1d,
        "n_with_realized_5d": n_with_realized_5d,
        "n_with_realized_h20": n_with_realized_h20,
        "n_with_realized_h63": n_with_realized_h63,
        "n_overpriced": n_overpriced,
        "n_underpriced": n_underpriced,
        "n_liquid_options": n_liquid,
        "n_hard_catalyst": n_hard,
        "records": joined,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = build_join_table()

    output_path = OUTPUT_DIR / "crt_options_join.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log.info("Wrote %s", output_path)

    print("\nCRT × OPTIONS JOIN TABLE (v2)")
    print(f"  Resolutions:      {result['n_resolutions']}")
    print(f"  With options:     {result['n_with_options']}")
    print(f"  With implied:     {result['n_with_implied_move']}")
    print(f"  Realized 1d:      {result['n_with_realized_1d']}")
    print(f"  Realized 5d:      {result['n_with_realized_5d']}")
    print(f"  Realized h20:     {result['n_with_realized_h20']}")
    print(f"  Realized h63:     {result['n_with_realized_h63']}")
    print(f"  Overpriced (1d):  {result['n_overpriced']}")
    print(f"  Underpriced (1d): {result['n_underpriced']}")
    print(f"  Hard catalyst:    {result['n_hard_catalyst']}")
    print(f"  Liquid options:   {result['n_liquid_options']}")

    # Print records
    if result["records"]:
        print(
            f"\n  {'Ticker':<7} {'Date':<12} {'Type':<20} {'Out':<6} {'EP':<5} {'Liq':<8} "
            f"{'Implied':<9} {'R_1d':<9} {'R_5d':<9} {'R_h20':<9} {'IvR':<6} {'Over?':<6}"
        )
        for r in result["records"]:
            imp = f"{r['implied_event_move']:.2f}" if r.get("implied_event_move") else "—"
            r1d = f"{r['realized_1d_return']:+.3f}" if r.get("realized_1d_return") is not None else "—"
            r5d = f"{r['realized_5d_return']:+.3f}" if r.get("realized_5d_return") is not None else "—"
            rh20 = f"{r['realized_h20_return']:+.3f}" if r.get("realized_h20_return") is not None else "—"
            ivr = f"{r['implied_vs_realized_1d']:.2f}" if r.get("implied_vs_realized_1d") is not None else "—"
            over = (
                "YES"
                if r.get("market_overpriced_1d") is True
                else "NO" if r.get("market_overpriced_1d") is False else "—"
            )
            liq = r.get("opt_liquidity_state", "absent")[:4]
            print(
                f"  {r['ticker']:<7} {r['catalyst_date']:<12} {r['catalyst_type']:<20} "
                f"{r['outcome']:<6} {r['opt_event_premium']:<5} {liq:<8} "
                f"{imp:<9} {r1d:<9} {r5d:<9} {rh20:<9} {ivr:<6} {over:<6}"
            )

    try:
        from tools.agent_skill_telemetry import log_agent_run
        from tools.record_skill_feedback import attach_outcome_verdict

        n_res = result.get("n_resolutions", 0)
        n_opts = result.get("n_with_options", 0)
        exec_id = log_agent_run(
            "crt_resolution_watcher",
            "CRT options join table refresh",
            outputs={"n_resolutions": n_res, "n_with_options": n_opts},
            success=True,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if exec_id and n_res > 0:
            join_rate = n_opts / n_res
            attach_outcome_verdict(
                exec_id,
                was_correct=join_rate >= 0.3,
                evidence=f"options joined {n_opts}/{n_res} resolutions",
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
