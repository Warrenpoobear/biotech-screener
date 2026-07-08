"""Transaction cost and rebalance threshold model.

Estimates spread + impact costs for portfolio rebalances and gates
small changes that don't justify the friction.

Usage:
    from common.rebalance_cost_model import (
        estimate_trade_cost_bps,
        compute_turnover_cost_drag,
        apply_rebalance_threshold,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

# Spread model: estimated half-spread by market-cap bucket (bps of notional)
# These are conservative estimates for small/mid-cap biotech via retail broker
SPREAD_BPS_BY_BUCKET = {
    "mega": 5,  # > $10B mcap
    "large": 10,  # $2B - $10B
    "mid": 20,  # $500M - $2B
    "small": 40,  # $100M - $500M
    "micro": 80,  # < $100M
}

# Market impact: additional cost for position establishment
# Modeled as sqrt(participation_rate) * volatility
# For simplicity, use fixed estimates by ADV bucket
IMPACT_BPS_BY_ADV = {
    "high": 5,  # ADV > $10M
    "medium": 15,  # ADV $1M - $10M
    "low": 40,  # ADV $100K - $1M
    "micro": 80,  # ADV < $100K
}


def _mcap_bucket(market_cap: float | None) -> str:
    if market_cap is None or market_cap <= 0:
        return "micro"
    if market_cap >= 10_000:  # $10B (in millions)
        return "mega"
    if market_cap >= 2_000:
        return "large"
    if market_cap >= 500:
        return "mid"
    if market_cap >= 100:
        return "small"
    return "micro"


def _adv_bucket(avg_dollar_volume: float | None) -> str:
    if avg_dollar_volume is None or avg_dollar_volume <= 0:
        return "micro"
    if avg_dollar_volume >= 10_000_000:
        return "high"
    if avg_dollar_volume >= 1_000_000:
        return "medium"
    if avg_dollar_volume >= 100_000:
        return "low"
    return "micro"


def estimate_trade_cost_bps(
    market_cap_mm: float | None = None,
    avg_dollar_volume: float | None = None,
) -> dict[str, Any]:
    """Estimate one-way trade cost in basis points.

    Returns dict with spread_bps, impact_bps, total_bps.
    """
    mc_bucket = _mcap_bucket(market_cap_mm)
    adv_bucket = _adv_bucket(avg_dollar_volume)

    spread = SPREAD_BPS_BY_BUCKET[mc_bucket]
    impact = IMPACT_BPS_BY_ADV[adv_bucket]

    return {
        "spread_bps": spread,
        "impact_bps": impact,
        "total_bps": spread + impact,
        "mcap_bucket": mc_bucket,
        "adv_bucket": adv_bucket,
    }


def estimate_portfolio_trade_cost(
    trades: list[dict],
    account_usd: float = 500_000,
) -> dict[str, Any]:
    """Estimate total cost of a set of trades.

    trades: list of dicts with:
      - ticker
      - trade_dollars (absolute value of notional change)
      - market_cap_mm (optional)
      - avg_dollar_volume (optional)

    Returns summary with total_cost_dollars, total_cost_bps, per-trade breakdown.
    """
    total_cost = 0.0
    breakdown = []

    for trade in trades:
        notional = abs(trade.get("trade_dollars", 0))
        if notional == 0:
            continue

        cost_est = estimate_trade_cost_bps(
            market_cap_mm=trade.get("market_cap_mm"),
            avg_dollar_volume=trade.get("avg_dollar_volume"),
        )
        cost_dollars = notional * cost_est["total_bps"] / 10_000
        total_cost += cost_dollars

        breakdown.append(
            {
                "ticker": trade.get("ticker", ""),
                "trade_dollars": notional,
                "cost_bps": cost_est["total_bps"],
                "cost_dollars": round(cost_dollars, 2),
            }
        )

    return {
        "total_cost_dollars": round(total_cost, 2),
        "total_cost_bps": round(total_cost / max(account_usd, 1) * 10_000, 1),
        "n_trades": len(breakdown),
        "breakdown": breakdown,
    }


# ---------------------------------------------------------------------------
# Turnover cost drag
# ---------------------------------------------------------------------------


def compute_turnover_cost_drag(
    prior_positions: list[dict],
    current_positions: list[dict],
    account_usd: float = 500_000,
    rankings: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Compute the estimated cost of rebalancing from prior to current positions.

    prior_positions, current_positions: lists of dicts with ticker, weight_pct
    rankings: optional {ticker: {market_cap_mm, avg_dollar_volume}} for cost estimation

    Returns cost summary.
    """
    prior_weights = {p["ticker"]: p["weight_pct"] for p in prior_positions}
    current_weights = {p["ticker"]: p["weight_pct"] for p in current_positions}

    all_tickers = set(prior_weights.keys()) | set(current_weights.keys())

    trades = []
    for ticker in all_tickers:
        w_old = prior_weights.get(ticker, 0)
        w_new = current_weights.get(ticker, 0)
        delta_pct = w_new - w_old
        if abs(delta_pct) < 0.01:  # skip trivial changes
            continue
        trade_dollars = abs(delta_pct) / 100 * account_usd

        r = (rankings or {}).get(ticker, {})
        trades.append(
            {
                "ticker": ticker,
                "trade_dollars": trade_dollars,
                "delta_pct": delta_pct,
                "market_cap_mm": r.get("market_cap_mm"),
                "avg_dollar_volume": r.get("avg_dollar_volume"),
            }
        )

    cost = estimate_portfolio_trade_cost(trades, account_usd)

    # Turnover metrics
    overlap = set(prior_weights.keys()) & set(current_weights.keys())
    n_added = len(set(current_weights.keys()) - set(prior_weights.keys()))
    n_removed = len(set(prior_weights.keys()) - set(current_weights.keys()))
    weight_turnover = sum(abs(current_weights.get(t, 0) - prior_weights.get(t, 0)) for t in all_tickers) / 2

    cost["n_added"] = n_added
    cost["n_removed"] = n_removed
    cost["n_overlap"] = len(overlap)
    cost["weight_turnover_pct"] = round(weight_turnover, 2)

    return cost


# ---------------------------------------------------------------------------
# Rebalance threshold gate
# ---------------------------------------------------------------------------


def apply_rebalance_threshold(
    prior_positions: list[dict],
    proposed_positions: list[dict],
    expected_alpha_bps: float = 0.0,
    cost_multiplier: float = 2.0,
    account_usd: float = 500_000,
    rankings: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Gate a rebalance: only trade if expected alpha > cost_multiplier × estimated cost.

    Returns dict with:
      - should_rebalance: bool
      - estimated_cost_bps: float
      - threshold_bps: float (cost × multiplier)
      - expected_alpha_bps: float
      - positions_to_use: list (either proposed or prior)
    """
    cost = compute_turnover_cost_drag(prior_positions, proposed_positions, account_usd, rankings)

    threshold_bps = cost["total_cost_bps"] * cost_multiplier
    should_rebalance = expected_alpha_bps > threshold_bps

    return {
        "should_rebalance": should_rebalance,
        "estimated_cost_bps": cost["total_cost_bps"],
        "estimated_cost_dollars": cost["total_cost_dollars"],
        "threshold_bps": round(threshold_bps, 1),
        "expected_alpha_bps": expected_alpha_bps,
        "cost_multiplier": cost_multiplier,
        "weight_turnover_pct": cost["weight_turnover_pct"],
        "n_trades": cost["n_trades"],
        "positions_to_use": proposed_positions if should_rebalance else prior_positions,
    }


# ---------------------------------------------------------------------------
# Historical cost drag estimation
# ---------------------------------------------------------------------------


def estimate_historical_cost_drag(
    periods: list[dict],
    account_usd: float = 500_000,
    avg_cost_bps: float = 50,
) -> dict[str, Any]:
    """Estimate cumulative cost drag from turnover in benchmark periods.

    periods: list of dicts with at least 'turnover' (fraction) and 'n_held'
    avg_cost_bps: average round-trip cost per name

    Returns cumulative cost drag estimate.
    """
    total_drag_pct = 0.0
    period_drags = []

    for p in periods:
        turnover = p.get("turnover", 0)
        n_held = p.get("n_held", 20)
        # Turnover fraction × number of names turned × avg cost
        # Each turned name costs ~avg_cost_bps one-way on exit + entry
        names_turned = turnover * n_held
        cost_pct = names_turned * avg_cost_bps * 2 / 10_000  # round trip
        total_drag_pct += cost_pct
        period_drags.append(
            {
                "date": p.get("date", ""),
                "turnover": turnover,
                "names_turned": round(names_turned, 1),
                "cost_pct": round(cost_pct, 4),
            }
        )

    return {
        "total_cost_drag_pct": round(total_drag_pct, 4),
        "avg_per_period_bps": round(total_drag_pct / max(len(periods), 1) * 100, 2),
        "n_periods": len(periods),
        "avg_cost_bps_assumption": avg_cost_bps,
        "periods": period_drags,
    }
