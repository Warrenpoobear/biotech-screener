#!/usr/bin/env python3
"""
Generate portfolio from momentum-integrated rankings.

Takes the ranked output and constructs a portfolio with:
- Top N position selection
- Inverse volatility weighting
- Concentration limits (max 7% per position)
- Risk metrics calculation

Usage:
    python scripts/generate_portfolio_from_rankings.py
    python scripts/generate_portfolio_from_rankings.py --top-n 60 --date 2026-01-15
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from decimal import Decimal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def fetch_price_and_volatility(tickers: list[str]) -> dict:
    """Fetch current prices and 30-day volatility from Yahoo Finance."""
    try:
        import yfinance as yf
        import numpy as np
    except ImportError:
        print("Warning: yfinance not available, using placeholder prices")
        return {t: {"price": 50.0, "volatility": 0.5} for t in tickers}

    result = {}

    # Batch download for efficiency
    print(f"Fetching prices for {len(tickers)} tickers...")
    try:
        data = yf.download(tickers, period="60d", progress=False, threads=True)

        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    close = data["Close"]
                else:
                    close = data["Close"][ticker]

                if close is None or len(close) == 0:
                    result[ticker] = {"price": None, "volatility": 0.5}
                    continue

                # Current price (most recent)
                price = float(close.dropna().iloc[-1])

                # 30-day annualized volatility
                returns = close.pct_change().dropna()
                if len(returns) > 5:
                    vol = float(returns.std() * np.sqrt(252))
                else:
                    vol = 0.5  # Default volatility

                result[ticker] = {"price": price, "volatility": max(vol, 0.15)}  # Floor at 15%

            except Exception:
                result[ticker] = {"price": None, "volatility": 0.5}

    except Exception as e:
        print(f"Warning: Batch download failed: {e}")
        return {t: {"price": None, "volatility": 0.5} for t in tickers}

    return result


def calculate_inverse_volatility_weights(securities: list[dict], market_data: dict, max_weight: float = 0.07) -> list[dict]:
    """
    Calculate position weights using inverse volatility.

    Lower volatility = higher weight, capped at max_weight.
    """
    # Calculate raw inverse volatility weights
    total_inv_vol = 0
    for sec in securities:
        ticker = sec["ticker"]
        vol = market_data.get(ticker, {}).get("volatility", 0.5)
        sec["_inv_vol"] = 1.0 / vol
        total_inv_vol += sec["_inv_vol"]

    # Normalize to sum to 1
    for sec in securities:
        raw_weight = sec["_inv_vol"] / total_inv_vol
        # Cap at max_weight
        sec["weight"] = min(raw_weight, max_weight)
        del sec["_inv_vol"]

    # Renormalize after capping
    total_weight = sum(s["weight"] for s in securities)
    for sec in securities:
        sec["weight"] = sec["weight"] / total_weight

    return securities


def main():
    parser = argparse.ArgumentParser(description="Generate portfolio from rankings")
    parser.add_argument(
        "--rankings-path",
        default="outputs/ranked_full_with_momentum.json",
        help="Path to momentum-integrated rankings"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=60,
        help="Number of positions to include (default: 60)"
    )
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Portfolio date (default: today)"
    )
    parser.add_argument(
        "--portfolio-value",
        type=float,
        default=10_000_000,
        help="Total portfolio value in USD (default: 10M)"
    )
    parser.add_argument(
        "--max-position",
        type=float,
        default=0.07,
        help="Maximum position weight (default: 0.07 = 7%%)"
    )
    parser.add_argument(
        "--output-path",
        help="Output path (default: outputs/portfolio_YYYYMMDD.json)"
    )
    args = parser.parse_args()

    # Load rankings
    rankings_path = Path(args.rankings_path)
    if not rankings_path.exists():
        print(f"Error: Rankings file not found: {rankings_path}")
        return 1

    print(f"Loading rankings from {rankings_path}...")
    with open(rankings_path) as f:
        data = json.load(f)

    # Extract ranked securities
    if "module_5_composite" in data:
        securities = data["module_5_composite"].get("ranked_securities", [])
    elif "module_5_output" in data:
        securities = data["module_5_output"].get("ranked_securities", [])
    else:
        print("Error: No ranked_securities found")
        return 1

    print(f"Found {len(securities)} ranked securities")

    # Select top N
    top_securities = securities[:args.top_n]
    print(f"Selected top {len(top_securities)} positions")

    # Get tickers for price fetch
    tickers = [s["ticker"] for s in top_securities]

    # Filter out benchmark if present
    tickers = [t for t in tickers if not t.startswith("_")]
    top_securities = [s for s in top_securities if not s["ticker"].startswith("_")]

    # Fetch market data
    market_data = fetch_price_and_volatility(tickers)

    # Filter out tickers without prices
    valid_securities = []
    for sec in top_securities:
        ticker = sec["ticker"]
        price = market_data.get(ticker, {}).get("price")
        if price is not None and price > 0:
            valid_securities.append(sec)
        else:
            print(f"  Warning: Skipping {ticker} (no price data)")

    if len(valid_securities) < len(top_securities):
        print(f"  {len(top_securities) - len(valid_securities)} tickers dropped due to missing prices")

    # Calculate weights
    weighted_securities = calculate_inverse_volatility_weights(
        valid_securities,
        market_data,
        max_weight=args.max_position
    )

    # Build portfolio
    portfolio_value = args.portfolio_value
    positions = []

    for sec in weighted_securities:
        ticker = sec["ticker"]
        weight = sec["weight"]
        price = market_data[ticker]["price"]
        volatility = market_data[ticker]["volatility"]

        position_value = portfolio_value * weight
        shares = int(position_value / price)
        actual_value = shares * price

        positions.append({
            "ticker": ticker,
            "rank": sec.get("composite_rank_with_momentum", sec.get("composite_rank", 0)),
            "composite_score": float(sec.get("composite_score_with_momentum", sec.get("composite_score", 0))),
            "momentum_score": sec.get("momentum_score", 50),
            "weight": round(weight, 6),
            "weight_pct": f"{weight * 100:.2f}%",
            "price": round(price, 2),
            "shares": shares,
            "position_value": round(actual_value, 2),
            "volatility": round(volatility, 4)
        })

    # Calculate portfolio metrics
    total_value = sum(p["position_value"] for p in positions)
    avg_momentum = sum(p["momentum_score"] * p["weight"] for p in positions)
    portfolio_vol = sum(p["volatility"] * p["weight"] for p in positions)
    max_position_pct = max(p["weight"] for p in positions) * 100

    # Estimate portfolio beta (biotech typically 0.7-0.9 vs S&P)
    # Higher momentum stocks tend to have higher beta in bull markets
    avg_momentum_normalized = avg_momentum / 100  # 0-1 scale
    estimated_beta = 0.65 + (avg_momentum_normalized * 0.25)  # Range: 0.65-0.90

    portfolio = {
        "metadata": {
            "generation_date": args.date,
            "generation_timestamp": datetime.now().isoformat(),
            "rankings_source": str(rankings_path),
            "portfolio_value": portfolio_value,
            "max_position_limit": args.max_position,
            "target_positions": args.top_n,
            "actual_positions": len(positions)
        },
        "metrics": {
            "num_positions": len(positions),
            "total_invested": round(total_value, 2),
            "cash_remaining": round(portfolio_value - total_value, 2),
            "avg_momentum": round(avg_momentum, 2),
            "portfolio_volatility": round(portfolio_vol, 4),
            "portfolio_beta_estimate": round(estimated_beta, 3),
            "max_position_pct": round(max_position_pct, 2),
            "concentration_check": "PASS" if max_position_pct <= args.max_position * 100 else "FAIL"
        },
        "positions": positions
    }

    # Save portfolio
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        date_str = args.date.replace("-", "")
        output_path = Path(f"outputs/portfolio_{date_str}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(portfolio, f, indent=2)

    print(f"\n✅ Portfolio saved to {output_path}")

    # Display summary
    print("\n" + "=" * 70)
    print("TOP 10 POSITIONS")
    print("=" * 70)
    print(f"{'Rank':>4}  {'Ticker':<6}  {'Weight':>8}  {'Value':>12}  {'Shares':>8}  {'Price':>10}")
    print("-" * 70)

    for pos in positions[:10]:
        print(f"{pos['rank']:4d}  {pos['ticker']:<6}  {pos['weight_pct']:>8}  ${pos['position_value']:>10,.0f}  {pos['shares']:>8,}  ${pos['price']:>9,.2f}")

    print("\n" + "=" * 70)
    print("PORTFOLIO METRICS")
    print("=" * 70)
    for key, value in portfolio["metrics"].items():
        if isinstance(value, float):
            if "pct" in key or "volatility" in key:
                print(f"  {key}: {value:.2f}%")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
