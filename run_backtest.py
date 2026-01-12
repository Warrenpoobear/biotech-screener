#!/usr/bin/env python3
"""
Backtest Runner for Biotech Screener

Runs the point-in-time backtesting framework with available production data.
Integrates with the production scoring pipeline (Modules 1-5).

Usage:
    python run_backtest.py
    python run_backtest.py --verbose
    python run_backtest.py --output backtest_results.json
    python run_backtest.py --use-production-scorer
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

from backtest_engine import PointInTimeBacktester

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Suppress info logs during backtest

# Try to import production modules
try:
    from run_screen import run_screening_pipeline
    from common.date_utils import to_date_string
    HAS_PRODUCTION_SCORER = True
except ImportError as e:
    HAS_PRODUCTION_SCORER = False
    logging.warning(f"Production scorer not available: {e}")


def load_production_data(data_dir: Path) -> Dict[str, Any]:
    """Load all production data files."""
    data = {}

    # Load universe
    universe_file = data_dir / "universe.json"
    if universe_file.exists():
        with open(universe_file) as f:
            data["universe"] = json.load(f)
        print(f"Loaded universe: {len(data['universe'])} tickers")

    # Load trial records
    trial_file = data_dir / "trial_records.json"
    if trial_file.exists():
        with open(trial_file) as f:
            data["trials"] = json.load(f)
        print(f"Loaded trials: {len(data['trials'])} records")

    # Load financial data
    fin_file = data_dir / "financial_data.json"
    if fin_file.exists():
        with open(fin_file) as f:
            data["financials"] = json.load(f)
        print(f"Loaded financials: {len(data['financials'])} records")

    # Load market data (for price time series)
    market_file = data_dir / "market_data.json"
    if market_file.exists():
        with open(market_file) as f:
            data["market"] = json.load(f)
        print(f"Loaded market data")

    return data


def build_ticker_snapshot(
    ticker: str,
    prod_data: Dict[str, Any]
) -> Optional[Dict]:
    """
    Build a snapshot for a ticker from production data.

    Returns None if insufficient data.
    """
    # Find ticker in universe
    universe = prod_data.get("universe", [])
    ticker_data = None

    for item in universe:
        if item.get("ticker") == ticker:
            ticker_data = item
            break

    if not ticker_data:
        return None

    # Extract relevant fields
    snapshot = {
        "ticker": ticker,
        "market_cap": ticker_data.get("market_data", {}).get("market_cap"),
        "price": ticker_data.get("market_data", {}).get("price"),
        "volume_avg": ticker_data.get("market_data", {}).get("volume_avg_30d"),
        "sector": ticker_data.get("market_data", {}).get("sector"),
        "industry": ticker_data.get("market_data", {}).get("industry"),
    }

    # Add financial data
    financials = ticker_data.get("financials", {})
    snapshot["cash"] = financials.get("cash")
    snapshot["debt"] = financials.get("debt")
    snapshot["revenue"] = financials.get("revenue_ttm")

    # Add clinical data
    clinical = ticker_data.get("clinical", {})
    snapshot["total_trials"] = clinical.get("total_trials", 0)
    snapshot["active_trials"] = clinical.get("active_trials", 0)
    snapshot["lead_stage"] = clinical.get("lead_stage", "unknown")
    snapshot["by_phase"] = clinical.get("by_phase", {})

    # Add time series for price history
    time_series = ticker_data.get("time_series", {})
    snapshot["price_history"] = time_series.get("prices", [])

    return snapshot


def create_production_scorer():
    """
    Create a scoring function based on production screening logic.

    This mimics the composite scoring from run_screen.py.
    """
    def scorer(ticker: str, data: Dict, as_of_date: datetime) -> Dict:
        """Score a ticker based on available data."""
        score = Decimal("50")  # Base score
        components = {
            "base": 50,
            "market_cap": 0,
            "clinical_progress": 0,
            "trial_activity": 0,
            "momentum": 0,
        }

        # Market cap factor (prefer mid-cap biotechs)
        market_cap = data.get("market_cap")
        if market_cap:
            if 500_000_000 <= market_cap <= 5_000_000_000:  # $500M - $5B sweet spot
                score += Decimal("10")
                components["market_cap"] = 10
            elif 200_000_000 <= market_cap <= 10_000_000_000:
                score += Decimal("5")
                components["market_cap"] = 5

        # Clinical development stage
        lead_stage = data.get("lead_stage", "").lower()
        if "phase 3" in lead_stage or "phase3" in lead_stage:
            score += Decimal("20")
            components["clinical_progress"] = 20
        elif "phase 2" in lead_stage or "phase2" in lead_stage:
            score += Decimal("12")
            components["clinical_progress"] = 12
        elif "phase 1" in lead_stage or "phase1" in lead_stage:
            score += Decimal("5")
            components["clinical_progress"] = 5

        # Phase breakdown scoring
        by_phase = data.get("by_phase", {})
        phase_score = 0
        if by_phase.get("Phase 3", 0) > 0:
            phase_score += 15
        if by_phase.get("Phase 2", 0) >= 2:
            phase_score += 8
        if by_phase.get("Phase 1", 0) >= 3:
            phase_score += 3
        score += Decimal(str(min(phase_score, 20)))
        components["clinical_progress"] += min(phase_score, 20)

        # Trial activity (active trials = good)
        active_trials = data.get("active_trials", 0)
        if active_trials >= 5:
            score += Decimal("10")
            components["trial_activity"] = 10
        elif active_trials >= 2:
            score += Decimal("5")
            components["trial_activity"] = 5
        elif active_trials >= 1:
            score += Decimal("2")
            components["trial_activity"] = 2

        # Price momentum (if history available)
        prices = data.get("price_history", [])
        if len(prices) >= 20:
            # Simple momentum: recent vs earlier
            recent = prices[-5:]
            earlier = prices[-20:-15]
            if recent and earlier:
                try:
                    recent_avg = sum(p for p in recent if p) / len([p for p in recent if p])
                    earlier_avg = sum(p for p in earlier if p) / len([p for p in earlier if p])
                    if earlier_avg > 0:
                        momentum = (recent_avg - earlier_avg) / earlier_avg
                        if momentum > 0.1:
                            score += Decimal("5")
                            components["momentum"] = 5
                        elif momentum > 0:
                            score += Decimal("2")
                            components["momentum"] = 2
                        elif momentum < -0.2:
                            score -= Decimal("5")
                            components["momentum"] = -5
                except (TypeError, ZeroDivisionError):
                    pass

        # Cap score at 0-100
        final_score = max(Decimal("0"), min(Decimal("100"), score))

        return {
            "ticker": ticker,
            "final_score": float(final_score),
            "components": components,
            "as_of_date": as_of_date.isoformat()
        }

    return scorer


def run_production_scoring(
    data_dir: Path,
    as_of_date: str,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Run the full production scoring pipeline and extract scores.

    Args:
        data_dir: Path to production data directory
        as_of_date: Date string (YYYY-MM-DD)
        verbose: Print progress

    Returns:
        Dict mapping ticker to composite score
    """
    if not HAS_PRODUCTION_SCORER:
        raise RuntimeError("Production scorer modules not available")

    if verbose:
        print(f"  Running production pipeline for {as_of_date}...")

    try:
        # Run the full screening pipeline
        result = run_screening_pipeline(
            as_of_date=as_of_date,
            data_dir=data_dir,
            enable_coinvest=False,
            enable_enhancements=False,
            enable_short_interest=False,
        )

        # Extract scores from Module 5 output
        # Key is "module_5_composite" in the pipeline output
        scores = {}
        m5_result = result.get("module_5_composite", result.get("module_5", {}))
        ranked = m5_result.get("ranked_securities", [])

        for security in ranked:
            ticker = security.get("ticker")
            composite_score = security.get("composite_score")
            if ticker and composite_score is not None:
                # Convert Decimal or string to float
                if isinstance(composite_score, Decimal):
                    scores[ticker] = float(composite_score)
                elif isinstance(composite_score, str):
                    scores[ticker] = float(Decimal(composite_score))
                else:
                    scores[ticker] = float(composite_score)

        if verbose:
            print(f"    Scored {len(scores)} tickers")

        return scores

    except Exception as e:
        if verbose:
            print(f"    Error in production pipeline: {e}")
        return {}


def create_production_pipeline_scorer(data_dir: Path) -> Callable:
    """
    Create a scorer function that wraps the production pipeline.

    This caches results per date to avoid re-running the full pipeline
    for each ticker.
    """
    score_cache: Dict[str, Dict[str, float]] = {}

    def scorer(ticker: str, data: Dict, as_of_date: datetime) -> Dict:
        """Score using production pipeline (cached by date)."""
        date_str = as_of_date.strftime('%Y-%m-%d')

        # Check cache first
        if date_str not in score_cache:
            try:
                scores = run_production_scoring(data_dir, date_str, verbose=False)
                score_cache[date_str] = scores
            except Exception:
                score_cache[date_str] = {}

        # Get score for this ticker
        cached_scores = score_cache.get(date_str, {})
        score = cached_scores.get(ticker)

        if score is not None:
            return {
                "ticker": ticker,
                "final_score": score,
                "source": "production_pipeline",
                "as_of_date": date_str
            }

        # Fallback to None if not in cache
        return None

    return scorer


def extract_price_data(prod_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Extract price time series from production data.

    The production data has prices as a list without dates, so we generate
    dates going backwards from today based on trading days (weekdays).

    Returns: {ticker: {date_str: price}}
    """
    price_cache = {}
    universe = prod_data.get("universe", [])

    # Generate trading day dates going backwards from today
    def generate_trading_dates(num_days: int, end_date: datetime = None) -> List[str]:
        """Generate trading day dates (weekdays) going backwards."""
        if end_date is None:
            end_date = datetime.now()
        dates = []
        current = end_date
        while len(dates) < num_days:
            if current.weekday() < 5:  # Monday = 0, Friday = 4
                dates.append(current.strftime('%Y-%m-%d'))
            current -= timedelta(days=1)
        return list(reversed(dates))  # Oldest first

    for item in universe:
        ticker = item.get("ticker")
        if not ticker:
            continue

        time_series = item.get("time_series", {})
        prices = time_series.get("prices", [])

        if prices and len(prices) > 0:
            # Generate dates for the price series
            dates = generate_trading_dates(len(prices))

            price_cache[ticker] = {}
            for d, p in zip(dates, prices):
                if p is not None:
                    price_cache[ticker][d] = p

    return price_cache


def run_single_period_backtest(
    prod_data: Dict[str, Any],
    data_dir: Path = None,
    use_production_scorer: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Run a single-period backtest with current production data.

    Since we only have current snapshots, this validates the framework
    and scores the current universe.

    Args:
        prod_data: Loaded production data
        data_dir: Path to data directory (required for production scorer)
        use_production_scorer: Use full production pipeline (Modules 1-5)
        verbose: Print progress
    """
    # Get ticker list
    universe = prod_data.get("universe", [])
    tickers = [item["ticker"] for item in universe if item.get("ticker")]

    scorer_type = "PRODUCTION" if use_production_scorer else "DEMO"
    if verbose:
        print(f"\n{'='*70}")
        print(f"SINGLE-PERIOD BACKTEST ({scorer_type})")
        print(f"{'='*70}")
        print(f"Universe size: {len(tickers)} tickers")

    # For production scorer, run full pipeline once for current date
    if use_production_scorer and HAS_PRODUCTION_SCORER and data_dir:
        if verbose:
            print("Running full production pipeline...")
        today = datetime.now().strftime('%Y-%m-%d')
        production_scores = run_production_scoring(data_dir, today, verbose=verbose)

        if production_scores:
            results = []
            for ticker, score in production_scores.items():
                results.append({
                    "ticker": ticker,
                    "final_score": score,
                    "components": {"production_composite": score - 50},
                    "source": "production_pipeline"
                })

            results.sort(key=lambda x: x["final_score"], reverse=True)

            if verbose:
                print(f"Scored: {len(results)}, Errors: 0")
                print(f"\n{'-'*70}")
                print("TOP 20 SCORERS (PRODUCTION)")
                print(f"{'-'*70}")
                print(f"{'Rank':<6} {'Ticker':<10} {'Composite Score':<15}")
                print(f"{'-'*70}")

                for i, r in enumerate(results[:20], 1):
                    print(f"{i:<6} {r['ticker']:<10} {r['final_score']:<15.2f}")

            return {
                "period": today,
                "scorer": "production_pipeline",
                "universe_size": len(tickers),
                "scored_count": len(results),
                "error_count": 0,
                "scores": {r["ticker"]: r["final_score"] for r in results},
                "top_20": [{"ticker": r["ticker"], "score": r["final_score"]} for r in results[:20]],
                "bottom_20": [{"ticker": r["ticker"], "score": r["final_score"]} for r in results[-20:]],
            }

    # Fallback to demo scorer
    scorer = create_production_scorer()

    # Score all tickers
    results = []
    scored = 0
    errors = 0

    for ticker in tickers:
        snapshot = build_ticker_snapshot(ticker, prod_data)
        if snapshot is None:
            errors += 1
            continue

        try:
            score_result = scorer(
                ticker=ticker,
                data=snapshot,
                as_of_date=datetime.now()
            )
            results.append(score_result)
            scored += 1
        except Exception as e:
            if verbose:
                print(f"  Error scoring {ticker}: {e}")
            errors += 1

    if verbose:
        print(f"Scored: {scored}, Errors: {errors}")

    # Sort by score
    results.sort(key=lambda x: x["final_score"], reverse=True)

    # Display top scorers
    if verbose:
        print(f"\n{'-'*70}")
        print("TOP 20 SCORERS")
        print(f"{'-'*70}")
        print(f"{'Rank':<6} {'Ticker':<10} {'Score':<8} {'Components'}")
        print(f"{'-'*70}")

        for i, r in enumerate(results[:20], 1):
            comps = r.get("components", {})
            comp_str = f"MC:{comps.get('market_cap',0):+d} CP:{comps.get('clinical_progress',0):+d} TA:{comps.get('trial_activity',0):+d} MO:{comps.get('momentum',0):+d}"
            print(f"{i:<6} {r['ticker']:<10} {r['final_score']:<8.1f} {comp_str}")

    return {
        "period": datetime.now().isoformat(),
        "universe_size": len(tickers),
        "scored_count": scored,
        "error_count": errors,
        "scores": {r["ticker"]: r["final_score"] for r in results},
        "top_20": [{"ticker": r["ticker"], "score": r["final_score"]} for r in results[:20]],
        "bottom_20": [{"ticker": r["ticker"], "score": r["final_score"]} for r in results[-20:]],
    }


def run_simulated_historical_backtest(
    prod_data: Dict[str, Any],
    data_dir: Path = None,
    use_production_scorer: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Run simulated historical backtest using price time series.

    Uses current fundamental data but historical prices to calculate
    what returns would have been at different historical dates.

    Args:
        prod_data: Loaded production data
        data_dir: Path to data directory (required for production scorer)
        use_production_scorer: Use full production pipeline (Modules 1-5)
        verbose: Print progress
    """
    # Extract price data
    price_cache = extract_price_data(prod_data)

    if not price_cache:
        print("No price history available for historical backtest")
        return {}

    # Find common date range
    all_dates = set()
    for ticker_prices in price_cache.values():
        all_dates.update(ticker_prices.keys())

    if not all_dates:
        print("No price dates found")
        return {}

    sorted_dates = sorted(all_dates)

    scorer_type = "PRODUCTION PIPELINE" if use_production_scorer else "DEMO"
    if verbose:
        print(f"\n{'='*70}")
        print(f"SIMULATED HISTORICAL BACKTEST ({scorer_type})")
        print(f"{'='*70}")
        print(f"Price history: {sorted_dates[0]} to {sorted_dates[-1]}")
        print(f"Tickers with prices: {len(price_cache)}")

    # Generate test dates (monthly, from available data)
    test_dates = []
    current = datetime.fromisoformat(sorted_dates[0])
    end = datetime.fromisoformat(sorted_dates[-1]) - timedelta(days=90)  # Need 90d forward

    while current <= end:
        test_dates.append(current)
        current += timedelta(days=30)

    if verbose:
        print(f"Test periods: {len(test_dates)}")

    if len(test_dates) < 2:
        print("Insufficient date range for backtest")
        return {}

    # Create backtester with price data
    backtester = PointInTimeBacktester(
        start_date=test_dates[0].strftime('%Y-%m-%d'),
        end_date=test_dates[-1].strftime('%Y-%m-%d')
    )
    backtester.price_cache = price_cache

    # Create scorer based on mode
    if use_production_scorer and HAS_PRODUCTION_SCORER and data_dir:
        if verbose:
            print(f"Using PRODUCTION scorer (Modules 1-5)")
        scorer = create_production_pipeline_scorer(data_dir)
    else:
        if use_production_scorer and not HAS_PRODUCTION_SCORER:
            print("WARNING: Production scorer not available, falling back to demo scorer")
        scorer = create_production_scorer()

    # Build snapshots (using current data for all periods - simplified approach)
    universe = prod_data.get("universe", [])
    tickers = [item["ticker"] for item in universe if item.get("ticker")]

    # Filter to tickers with price data
    tickers_with_prices = [t for t in tickers if t in price_cache]

    if verbose:
        print(f"Tickers with price history: {len(tickers_with_prices)}")

    # Run scoring and return calculation for each period
    all_period_results = []
    horizons = [30, 60, 90]

    for i, test_date in enumerate(test_dates):
        if verbose:
            print(f"[{i+1}/{len(test_dates)}] Processing: {test_date.strftime('%Y-%m-%d')}")

        period_result = {
            "date": test_date.isoformat(),
            "scores": {},
            "returns_30d": {},
            "returns_60d": {},
            "returns_90d": {},
        }

        for ticker in tickers_with_prices:
            # Build snapshot
            snapshot = build_ticker_snapshot(ticker, prod_data)
            if snapshot is None:
                continue

            # Score
            try:
                score_result = scorer(ticker=ticker, data=snapshot, as_of_date=test_date)
                if score_result and "final_score" in score_result:
                    period_result["scores"][ticker] = score_result["final_score"]
            except Exception:
                continue

            # Calculate forward returns
            for horizon in horizons:
                ret = backtester.calculate_forward_returns(ticker, test_date, [horizon])
                key = f"{horizon}d"
                if ret.get(key) is not None:
                    period_result[f"returns_{key}"][ticker] = float(ret[key])

        if verbose:
            scored_count = len(period_result["scores"])
            print(f"    Scored: {scored_count} tickers")

        all_period_results.append(period_result)

    # Calculate IC statistics
    stats = backtester.calculate_backtest_statistics(all_period_results, horizons)

    if verbose:
        print(f"\n{backtester.generate_report(stats)}")

    return stats


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run backtest on biotech screener")
    parser.add_argument("--data-dir", default="production_data", help="Data directory")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--historical", action="store_true", help="Run historical simulation")
    parser.add_argument("--use-production-scorer", action="store_true",
                        help="Use full production pipeline (Modules 1-5) for scoring")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("="*70)
    print("BIOTECH SCREENER BACKTEST")
    print("="*70)

    if args.use_production_scorer:
        if HAS_PRODUCTION_SCORER:
            print("Mode: PRODUCTION SCORER (Modules 1-5)")
        else:
            print("WARNING: Production scorer not available")
    else:
        print("Mode: DEMO SCORER (basic heuristics)")

    # Load production data
    print("\nLoading production data...")
    prod_data = load_production_data(data_dir)

    if not prod_data.get("universe"):
        print("ERROR: No universe data found")
        sys.exit(1)

    # Run single-period backtest (production scorer works here)
    single_results = run_single_period_backtest(
        prod_data,
        data_dir=data_dir,
        use_production_scorer=args.use_production_scorer,
        verbose=True
    )

    # Run historical simulation
    # Note: Production scorer requires historical snapshots which we don't have
    # So historical simulation always uses demo scorer
    historical_results = {}
    if args.historical or True:  # Always run historical
        if args.use_production_scorer:
            print("\nNote: Historical simulation uses demo scorer (no historical snapshots available)")
        historical_results = run_simulated_historical_backtest(
            prod_data,
            data_dir=data_dir,
            use_production_scorer=False,  # Always use demo for historical
            verbose=True
        )

    # Combine results
    output = {
        "run_date": datetime.now().isoformat(),
        "single_period": single_results,
        "historical_simulation": historical_results,
    }

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

    # Summary
    print(f"\n{'='*70}")
    print("BACKTEST COMPLETE")
    print(f"{'='*70}")

    if historical_results:
        ic_30d = historical_results.get("ic_30d", {})
        if ic_30d.get("mean") is not None:
            print(f"30-Day IC: {ic_30d['mean']:.4f} (n={ic_30d['n_periods']} periods)")
            print(f"Assessment: {historical_results.get('assessment', 'N/A')}")

    return output


if __name__ == "__main__":
    main()
