#!/usr/bin/env python3
"""
Portfolio Construction from Rankings - Morningstar Powered

Uses institutional-grade Morningstar Direct returns data.
No external API calls - all data from local returns_db_daily.json.

Usage:
    python scripts/generate_portfolio_morningstar.py --date 2026-01-15 --top-n 60
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_rankings(path: Path) -> list[dict]:
    """Load ranked tickers with momentum"""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif "securities" in data:
        return data["securities"]
    elif "module_5_composite" in data:
        # Full pipeline output format
        return data["module_5_composite"].get("ranked_securities", [])
    elif "ranked_securities" in data:
        return data["ranked_securities"]
    else:
        raise ValueError(f"Unexpected format in {path}")


def load_morningstar_returns(
    returns_db_path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """
    Load Morningstar daily returns database.

    Returns:
        (ticker_returns, benchmark_returns)
    """
    with open(returns_db_path) as f:
        data = json.load(f)

    ticker_returns = data.get("tickers", {})
    benchmark_returns = data.get("benchmark", {})

    return ticker_returns, benchmark_returns


def calculate_volatility_from_returns(
    returns_dict: dict[str, str], lookback_days: int = 60
) -> float:
    """
    Calculate annualized volatility from daily returns.

    Args:
        returns_dict: {date_str: return_decimal_str}
        lookback_days: Number of days to use for volatility calc

    Returns:
        Annualized volatility as decimal (e.g., 0.45 for 45%)
    """
    if not returns_dict:
        return 0.50  # Default 50% volatility

    # Sort by date and take most recent N days
    sorted_dates = sorted(returns_dict.keys(), reverse=True)[:lookback_days]

    if len(sorted_dates) < 20:
        return 0.50  # Not enough data

    # Convert to floats
    returns = [float(returns_dict[d]) for d in sorted_dates]

    # Calculate standard deviation of daily returns
    n = len(returns)
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    daily_std = math.sqrt(variance)

    # Annualize (sqrt(252) trading days)
    annual_vol = daily_std * math.sqrt(252)

    # Cap at reasonable bounds (10% - 200%)
    return max(0.10, min(2.0, annual_vol))


def estimate_price_from_returns(returns_dict: dict[str, str], base_price: float = 50.0) -> float:
    """
    Estimate a price level from cumulative returns.

    Note: This is a proxy since we don't have actual prices.
    For actual trading, you'd want real-time prices from another source.
    """
    if not returns_dict:
        return base_price

    # Get last 60 days of returns
    sorted_dates = sorted(returns_dict.keys(), reverse=True)[:60]

    # Calculate cumulative return
    cum_return = 1.0
    for date in sorted_dates:
        daily_ret = float(returns_dict[date])
        cum_return *= 1 + daily_ret

    estimated_price = base_price * cum_return

    # Cap at reasonable biotech price range ($5 - $500)
    return max(5.0, min(500.0, estimated_price))


def calculate_inverse_vol_weights(
    tickers: list[str], volatilities: dict[str, float]
) -> dict[str, float]:
    """
    Inverse volatility weighting
    - Lower volatility stocks get higher weight
    - Smooths out risk contribution across positions
    """
    inv_vols = {}
    for ticker in tickers:
        if ticker in volatilities:
            vol = volatilities[ticker]
            vol = max(0.10, min(2.0, vol))
            inv_vols[ticker] = 1.0 / vol
        else:
            inv_vols[ticker] = 2.0  # Default ~50% vol

    # Normalize to sum to 1.0
    total_inv_vol = sum(inv_vols.values())
    weights = {t: inv_vol / total_inv_vol for t, inv_vol in inv_vols.items()}

    return weights


def apply_concentration_limits(
    weights: dict[str, float], max_position_pct: float = 3.0
) -> dict[str, float]:
    """
    Apply concentration limits
    - No single position > max_position_pct
    - Redistribute excess to other positions
    """
    max_weight = max_position_pct / 100.0

    # Cap oversized positions
    capped = {}
    excess = 0.0

    for ticker, weight in weights.items():
        if weight > max_weight:
            capped[ticker] = max_weight
            excess += weight - max_weight
        else:
            capped[ticker] = weight

    # Redistribute excess proportionally to uncapped positions
    if excess > 0:
        uncapped_tickers = [t for t, w in capped.items() if w < max_weight]
        if uncapped_tickers:
            uncapped_total = sum(capped[t] for t in uncapped_tickers)
            if uncapped_total > 0:
                for ticker in uncapped_tickers:
                    boost = excess * (capped[ticker] / uncapped_total)
                    capped[ticker] = min(max_weight, capped[ticker] + boost)

    # Renormalize to ensure 100%
    total = sum(capped.values())
    normalized = {t: w / total for t, w in capped.items()}

    return normalized


def calculate_portfolio_metrics(positions: list[dict], volatilities: dict[str, float]) -> dict:
    """Calculate portfolio-level risk metrics"""

    # Handle empty portfolio
    if not positions:
        return {
            "num_positions": 0,
            "total_weight_pct": 0.0,
            "avg_momentum": 0.0,
            "max_position_pct": 0.0,
            "min_position_pct": 0.0,
            "portfolio_volatility": 0.0,
            "portfolio_beta_estimate": 0.0,
            "data_source": "morningstar_direct",
        }

    # Weighted average momentum
    total_weight = sum(p["weight_pct"] for p in positions)
    avg_momentum = sum(
        p.get("momentum_score", 50) * p["weight_pct"] / total_weight for p in positions
    )

    # Weighted average volatility
    weighted_vol = 0.0
    for pos in positions:
        ticker = pos["ticker"]
        if ticker in volatilities:
            weighted_vol += volatilities[ticker] * pos["weight_pct"] / 100.0

    # Portfolio beta estimate
    portfolio_beta = weighted_vol / 0.60 if weighted_vol > 0 else 0.0

    return {
        "num_positions": len(positions),
        "total_weight_pct": round(sum(p["weight_pct"] for p in positions), 2),
        "avg_momentum": round(avg_momentum, 1),
        "max_position_pct": round(max(p["weight_pct"] for p in positions), 2),
        "min_position_pct": round(min(p["weight_pct"] for p in positions), 2),
        "portfolio_volatility": round(weighted_vol * 100, 1),
        "portfolio_beta_estimate": round(portfolio_beta, 2),
        "data_source": "morningstar_direct",
    }


def generate_portfolio(
    ranked_path: Path,
    returns_db_path: Path,
    top_n: int,
    portfolio_value: float,
    date_str: str,
    output_path: Path,
    max_position_pct: float = 3.0,
):
    """Main portfolio generation logic - Morningstar powered"""

    print("=" * 70)
    print("PORTFOLIO GENERATION - MORNINGSTAR POWERED")
    print("=" * 70)
    print(f"Date: {date_str}")
    print(f"Portfolio value: ${portfolio_value:,.0f}")
    print(f"Target positions: {top_n}")
    print(f"Max position: {max_position_pct}%")
    print("Data source: Morningstar Direct")
    print()

    # Load rankings
    print("1. Loading rankings...")
    securities = load_rankings(ranked_path)
    print(f"   Found {len(securities)} ranked securities")

    # Take top N
    top_securities = securities[:top_n]
    tickers = [s["ticker"] for s in top_securities]
    print(f"   Selected top {len(tickers)} for portfolio")
    print()

    # Load Morningstar returns
    print("2. Loading Morningstar returns database...")
    ticker_returns, benchmark_returns = load_morningstar_returns(returns_db_path)
    print(f"   Loaded {len(ticker_returns)} tickers from Morningstar")

    # Calculate volatilities and prices from returns
    volatilities = {}
    prices = {}
    missing_tickers = []

    for ticker in tickers:
        if ticker in ticker_returns:
            returns = ticker_returns[ticker]
            volatilities[ticker] = calculate_volatility_from_returns(returns)
            prices[ticker] = estimate_price_from_returns(returns)
        else:
            missing_tickers.append(ticker)

    coverage_pct = (len(tickers) - len(missing_tickers)) / len(tickers) * 100
    print(
        f"   Morningstar coverage: {len(tickers) - len(missing_tickers)}/{len(tickers)} ({coverage_pct:.1f}%)"
    )

    if missing_tickers:
        print(
            f"   Missing tickers: {', '.join(missing_tickers[:10])}"
            + (f" (+{len(missing_tickers) - 10} more)" if len(missing_tickers) > 10 else "")
        )
    print()

    # Filter to tickers with data
    valid_tickers = [t for t in tickers if t in volatilities]
    valid_securities = [s for s in top_securities if s["ticker"] in volatilities]

    # Early exit if no valid tickers
    if not valid_tickers:
        print("❌ ERROR: No tickers have Morningstar data!")
        print()
        print("The returns database appears to be missing data for your universe.")
        print("To fix this, run the Morningstar fetch script first:")
        print()
        print("    python scripts/fetch_daily_returns_morningstar.py --years 1")
        print()
        print("This requires MD_AUTH_TOKEN environment variable to be set.")
        return None

    # Calculate weights
    print("3. Calculating position sizes...")
    weights = calculate_inverse_vol_weights(valid_tickers, volatilities)

    # Apply concentration limits
    final_weights = apply_concentration_limits(weights, max_position_pct)
    print(f"   Applied {max_position_pct}% concentration limit")
    print()

    # Build position list
    print("4. Building position list...")
    positions = []

    for i, security in enumerate(valid_securities):
        ticker = security["ticker"]
        weight_pct = final_weights.get(ticker, 0) * 100
        position_value = portfolio_value * weight_pct / 100

        # Get price estimate for share count
        price_estimate = prices.get(ticker, 50.0)
        shares = int(position_value / price_estimate) if price_estimate > 0 else 0

        # Get scores - support both old and new field names
        original_score = float(
            security.get("composite_score_original")
            or security.get("original_score")
            or 0
        )
        final_score = float(
            security.get("composite_score_with_momentum")
            or security.get("final_score")
            or 0
        )
        momentum = float(security.get("momentum_score", 50))

        position = {
            "rank": i + 1,
            "ticker": ticker,
            "weight_pct": round(weight_pct, 2),
            "position_value_usd": round(position_value, 2),
            "original_score": round(original_score, 2),
            "momentum_score": round(momentum, 1),
            "final_score": round(final_score, 2),
            "volatility": round(volatilities.get(ticker, 0.50), 4),
            "price_estimate": round(price_estimate, 2),
            "shares_estimate": shares,
            "data_source": "morningstar_direct",
        }

        positions.append(position)

    print(f"   Generated {len(positions)} positions")
    print()

    # Calculate portfolio metrics
    print("5. Calculating portfolio metrics...")
    metrics = calculate_portfolio_metrics(positions, volatilities)
    print(f"   Portfolio beta estimate: {metrics['portfolio_beta_estimate']}")
    print(f"   Portfolio volatility: {metrics['portfolio_volatility']}%")
    print(f"   Average momentum: {metrics['avg_momentum']}")
    print()

    # Build output
    output = {
        "date": date_str,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "portfolio_value_usd": portfolio_value,
        "num_positions": len(positions),
        "positions": positions,
        "metrics": metrics,
        "parameters": {
            "top_n": top_n,
            "max_position_pct": max_position_pct,
            "weighting_method": "inverse_volatility",
            "data_source": "morningstar_direct",
            "returns_db_path": str(returns_db_path),
        },
        "data_quality": {
            "tickers_requested": len(tickers),
            "tickers_with_morningstar_data": len(valid_tickers),
            "coverage_pct": round(coverage_pct, 1),
            "missing_tickers": missing_tickers,
        },
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✅ Portfolio saved to {output_path}")
    print()

    # Print summary
    print("=" * 70)
    print("TOP 10 POSITIONS")
    print("=" * 70)
    print(f"{'Rank':<6}{'Ticker':<8}{'Weight':>8}{'Value':>12}{'Vol':>8}{'Score':>8}")
    print("-" * 70)

    for pos in positions[:10]:
        print(
            f"{pos['rank']:<6}"
            f"{pos['ticker']:<8}"
            f"{pos['weight_pct']:>7.2f}%"
            f"${pos['position_value_usd']:>10,.0f}"
            f"{pos['volatility'] * 100:>7.1f}%"
            f"{pos['final_score']:>8.2f}"
        )

    print()
    print("=" * 70)
    print("PORTFOLIO METRICS")
    print("=" * 70)
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print()

    print("=" * 70)
    print("DATA QUALITY")
    print("=" * 70)
    print(f"  Tickers requested: {output['data_quality']['tickers_requested']}")
    print(f"  Tickers with data: {output['data_quality']['tickers_with_morningstar_data']}")
    print(f"  Coverage: {output['data_quality']['coverage_pct']}%")
    print()

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Generate portfolio from rankings (Morningstar powered)"
    )
    parser.add_argument("--date", required=True, help="Portfolio date (YYYY-MM-DD)")
    parser.add_argument("--top-n", type=int, default=60, help="Number of positions (default: 60)")
    parser.add_argument(
        "--portfolio-value",
        type=float,
        default=10_000_000,
        help="Portfolio value in USD (default: 10M)",
    )
    parser.add_argument(
        "--ranked-path",
        default="outputs/ranked_with_momentum.json",
        help="Path to ranked JSON file",
    )
    parser.add_argument(
        "--returns-db",
        default="data/returns/returns_db_daily.json",
        help="Path to Morningstar returns database",
    )
    parser.add_argument(
        "--output-path", help="Output path (default: outputs/portfolio_YYYYMMDD_morningstar.json)"
    )
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=3.0,
        help="Max position size as % of portfolio (default: 3.0)",
    )

    args = parser.parse_args()

    # Parse date
    date_str = args.date
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    date_compact = date_obj.strftime("%Y%m%d")

    # Set paths
    ranked_path = Path(args.ranked_path)
    returns_db_path = Path(args.returns_db)

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path(f"outputs/portfolio_{date_compact}_morningstar.json")

    # Validate inputs
    if not ranked_path.exists():
        print(f"❌ Rankings file not found: {ranked_path}")
        return 1

    if not returns_db_path.exists():
        print(f"❌ Morningstar returns not found: {returns_db_path}")
        print("   Run: python scripts/fetch_daily_returns_morningstar.py --years 1")
        return 1

    # Generate
    result = generate_portfolio(
        ranked_path=ranked_path,
        returns_db_path=returns_db_path,
        top_n=args.top_n,
        portfolio_value=args.portfolio_value,
        date_str=date_str,
        output_path=output_path,
        max_position_pct=args.max_position_pct,
    )

    if result is None:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
