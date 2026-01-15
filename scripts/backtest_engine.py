#!/usr/bin/env python3
"""
Point-in-Time Backtesting Framework

Simulates what the model would have scored at historical dates.
CRITICAL: Prevents look-ahead bias by only using data available at as_of_date.

Key Principles:
1. Only use data available at as_of_date (point-in-time discipline)
2. Calculate forward returns (30d, 60d, 90d post-scoring)
3. Track Information Coefficient over time
4. Identify regime dependencies

Usage:
    backtester = PointInTimeBacktester(
        start_date="2019-01-01",
        end_date="2024-12-31"
    )
    results = backtester.run_backtest(
        universe=["ARGX", "BBIO", "INCY", ...],
        scoring_function=your_composite_scorer
    )
    print(f"Mean IC (30d): {results['ic_30d']['mean']:.3f}")

Success Criteria:
    - IC > 0.05 = good
    - IC > 0.08 = excellent
    - Hit rate > 55% = viable strategy

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import json
import hashlib
from datetime import datetime, timedelta, date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
import math


# Module metadata
__version__ = "1.0.0"


class PointInTimeBacktester:
    """
    Validates model performance across historical periods.

    Key principles:
    1. Only use data available at as_of_date (point-in-time discipline)
    2. Calculate forward returns (30d, 60d, 90d post-scoring)
    3. Track Information Coefficient over time
    4. Identify regime dependencies
    """

    VERSION = "1.0.0"

    # Default horizons for forward return calculation
    DEFAULT_HORIZONS = [30, 60, 90]

    # Minimum tickers needed for valid IC calculation
    MIN_TICKERS_FOR_IC = 5

    def __init__(
        self,
        start_date: str,
        end_date: str,
        data_dir: str = "data/historical",
        price_file: Optional[str] = None
    ):
        """
        Initialize the backtester.

        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            data_dir: Directory containing historical snapshots
            price_file: Path to historical price data file
        """
        self.start_date = datetime.fromisoformat(start_date)
        self.end_date = datetime.fromisoformat(end_date)
        self.data_dir = Path(data_dir)
        self.price_file = price_file
        self.results: List[Dict] = []
        self.price_cache: Dict[str, Dict[str, float]] = {}
        self.audit_trail: List[Dict] = []

        # Load price data if available
        if price_file and Path(price_file).exists():
            self._load_price_data(price_file)

    def _load_price_data(self, price_file: str) -> None:
        """Load historical price data from file."""
        try:
            with open(price_file) as f:
                data = json.load(f)

            # Expected format: {"TICKER": {"2024-01-15": 45.23, ...}, ...}
            self.price_cache = data
            print(f"Loaded price data for {len(self.price_cache)} tickers")

        except Exception as e:
            print(f"Warning: Could not load price data: {e}")
            self.price_cache = {}

    def generate_test_dates(
        self,
        frequency_days: int = 30,
        weekday: int = 4  # Friday = 4
    ) -> List[datetime]:
        """
        Generate monthly scoring dates for backtest.

        Args:
            frequency_days: Days between test dates
            weekday: Preferred day of week (0=Mon, 4=Fri)

        Returns:
            List of test dates
        """
        test_dates = []
        current = self.start_date

        while current <= self.end_date:
            # Adjust to nearest weekday if needed
            adjusted = current
            while adjusted.weekday() > 4:  # Skip weekends
                adjusted -= timedelta(days=1)

            test_dates.append(adjusted)
            current += timedelta(days=frequency_days)

        return test_dates

    def load_historical_snapshot(
        self,
        ticker: str,
        as_of_date: datetime
    ) -> Optional[Dict]:
        """
        Load ticker data as it existed at as_of_date.

        Critical rules:
        - Financial data: Use most recent 10-Q/10-K filed BEFORE as_of_date
        - Trial data: Use status as of as_of_date (enrollment, phase)
        - Price data: Use price on as_of_date
        - News: Only events announced BEFORE as_of_date

        Returns None if insufficient data available.
        """
        date_str = as_of_date.strftime('%Y%m%d')

        # Try exact date first
        snapshot_path = self.data_dir / ticker / f"{date_str}.json"

        if snapshot_path.exists():
            try:
                with open(snapshot_path) as f:
                    return json.load(f)
            except Exception:
                pass

        # Try to find nearest prior snapshot
        ticker_dir = self.data_dir / ticker
        if not ticker_dir.exists():
            return None

        # Find all snapshots before as_of_date
        available = []
        for f in ticker_dir.glob("*.json"):
            try:
                file_date = datetime.strptime(f.stem, '%Y%m%d')
                if file_date <= as_of_date:
                    available.append((file_date, f))
            except ValueError:
                continue

        if not available:
            return None

        # Use most recent snapshot
        available.sort(key=lambda x: x[0], reverse=True)
        _, best_file = available[0]

        try:
            with open(best_file) as f:
                snapshot = json.load(f)
                snapshot["_snapshot_date"] = best_file.stem
                return snapshot
        except Exception:
            return None

    def get_price(
        self,
        ticker: str,
        as_of_date: datetime
    ) -> Optional[Decimal]:
        """
        Fetch historical price from database.

        Args:
            ticker: Stock ticker
            as_of_date: Date to get price for

        Returns:
            Price as Decimal, or None if unavailable
        """
        if ticker not in self.price_cache:
            return None

        ticker_prices = self.price_cache[ticker]
        date_str = as_of_date.strftime('%Y-%m-%d')

        # Try exact date
        if date_str in ticker_prices:
            return Decimal(str(ticker_prices[date_str]))

        # Try nearby dates (within 5 days)
        for offset in range(1, 6):
            for delta in [timedelta(days=-offset), timedelta(days=offset)]:
                try_date = (as_of_date + delta).strftime('%Y-%m-%d')
                if try_date in ticker_prices:
                    return Decimal(str(ticker_prices[try_date]))

        return None

    def calculate_forward_returns(
        self,
        ticker: str,
        as_of_date: datetime,
        horizons: Optional[List[int]] = None
    ) -> Dict[str, Optional[Decimal]]:
        """
        Calculate actual returns from as_of_date to future dates.

        Args:
            ticker: Stock ticker
            as_of_date: Starting date
            horizons: List of forward days [30, 60, 90]

        Returns:
            {"30d": Decimal("0.15"), "60d": Decimal("0.23"), ...}
        """
        if horizons is None:
            horizons = self.DEFAULT_HORIZONS

        returns = {}

        price_start = self.get_price(ticker, as_of_date)
        if price_start is None or price_start == 0:
            for days in horizons:
                returns[f"{days}d"] = None
            return returns

        for days in horizons:
            future_date = as_of_date + timedelta(days=days)
            price_end = self.get_price(ticker, future_date)

            if price_end is not None and price_start != 0:
                ret = (price_end - price_start) / price_start
                returns[f"{days}d"] = ret.quantize(Decimal("0.0001"))
            else:
                returns[f"{days}d"] = None

        return returns

    def run_backtest(
        self,
        universe: List[str],
        scoring_function: Callable,
        test_dates: Optional[List[datetime]] = None,
        horizons: Optional[List[int]] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Run full backtest across universe and time periods.

        For each test_date:
        1. Load point-in-time data for all tickers
        2. Score universe using scoring_function
        3. Calculate forward returns
        4. Store results for analysis

        Args:
            universe: List of tickers to backtest
            scoring_function: Function(ticker, data, as_of_date) -> Dict with "final_score"
            test_dates: Optional list of dates to test
            horizons: Forward return horizons [30, 60, 90]
            verbose: Print progress

        Returns:
            Backtest statistics including IC metrics
        """
        if test_dates is None:
            test_dates = self.generate_test_dates()

        if horizons is None:
            horizons = self.DEFAULT_HORIZONS

        all_results = []

        for i, test_date in enumerate(test_dates):
            if verbose:
                print(f"[{i+1}/{len(test_dates)}] Backtesting: {test_date.strftime('%Y-%m-%d')}")

            period_results = {
                "date": test_date.isoformat(),
                "scores": {},
                "data_coverage": {},
                "errors": []
            }

            # Initialize return dicts
            for h in horizons:
                period_results[f"returns_{h}d"] = {}

            scored_count = 0
            error_count = 0

            for ticker in universe:
                # Load historical snapshot
                snapshot = self.load_historical_snapshot(ticker, test_date)

                if snapshot is None:
                    period_results["data_coverage"][ticker] = "insufficient"
                    continue

                # Score using provided function
                try:
                    score_result = scoring_function(
                        ticker=ticker,
                        data=snapshot,
                        as_of_date=test_date
                    )

                    if score_result and "final_score" in score_result:
                        period_results["scores"][ticker] = float(score_result["final_score"])
                        period_results["data_coverage"][ticker] = "complete"
                        scored_count += 1
                    else:
                        period_results["data_coverage"][ticker] = "no_score"

                except Exception as e:
                    period_results["errors"].append({
                        "ticker": ticker,
                        "error": str(e)
                    })
                    period_results["data_coverage"][ticker] = "error"
                    error_count += 1
                    continue

                # Calculate forward returns
                forward_returns = self.calculate_forward_returns(
                    ticker, test_date, horizons
                )

                for horizon_key, ret in forward_returns.items():
                    if ret is not None:
                        period_results[f"returns_{horizon_key}"][ticker] = float(ret)

            if verbose:
                coverage = scored_count / len(universe) if universe else 0
                print(f"    Scored: {scored_count}/{len(universe)} ({coverage:.1%}), Errors: {error_count}")

            all_results.append(period_results)

        self.results = all_results

        # Aggregate statistics
        return self.calculate_backtest_statistics(all_results, horizons)

    def calculate_backtest_statistics(
        self,
        results: List[Dict],
        horizons: Optional[List[int]] = None
    ) -> Dict:
        """
        Calculate performance metrics across all periods.

        Key metrics:
        - Information Coefficient (IC): Rank correlation between scores and returns
        - Hit Rate: % of time top quintile outperforms bottom quintile
        - IC IR: IC / std(IC) - consistency of signal
        - Turnover: How often rankings change
        """
        if horizons is None:
            horizons = self.DEFAULT_HORIZONS

        # Collect IC values for each horizon
        ic_by_horizon = {f"{h}d": [] for h in horizons}

        for period in results:
            scores = period["scores"]

            for h in horizons:
                returns_key = f"returns_{h}d"
                returns = period.get(returns_key, {})

                # Get common tickers with both score and returns
                common_tickers = set(scores.keys()) & set(returns.keys())

                if len(common_tickers) >= self.MIN_TICKERS_FOR_IC:
                    score_values = [scores[t] for t in common_tickers]
                    return_values = [returns[t] for t in common_tickers]

                    ic = self._spearman_correlation(score_values, return_values)
                    if ic is not None:
                        ic_by_horizon[f"{h}d"].append(ic)

        # Calculate statistics for each horizon
        stats = {
            "num_periods": len(results),
            "periods_analyzed": [p["date"] for p in results],
            "version": self.VERSION
        }

        for horizon_key, ic_values in ic_by_horizon.items():
            if ic_values:
                stats[f"ic_{horizon_key}"] = {
                    "mean": sum(ic_values) / len(ic_values),
                    "median": self._median(ic_values),
                    "std": self._std(ic_values),
                    "min": min(ic_values),
                    "max": max(ic_values),
                    "n_periods": len(ic_values),
                    "ic_ir": (sum(ic_values) / len(ic_values)) / self._std(ic_values) if self._std(ic_values) > 0 else None,
                    "pct_positive": sum(1 for ic in ic_values if ic > 0) / len(ic_values)
                }
            else:
                stats[f"ic_{horizon_key}"] = {
                    "mean": None,
                    "n_periods": 0,
                    "reason": "Insufficient data"
                }

        # Calculate hit rate and turnover
        stats["hit_rate_top_quintile"] = self._calculate_hit_rate(results, horizons[0])
        stats["turnover_monthly"] = self._calculate_turnover(results)

        # Overall assessment
        primary_ic = stats.get(f"ic_{horizons[0]}d", {}).get("mean")
        if primary_ic is not None:
            if primary_ic > 0.08:
                stats["assessment"] = "EXCELLENT: Strong predictive signal"
            elif primary_ic > 0.05:
                stats["assessment"] = "GOOD: Viable for deployment"
            elif primary_ic > 0.02:
                stats["assessment"] = "WEAK: Marginal predictive value"
            else:
                stats["assessment"] = "FAILED: No predictive power"
        else:
            stats["assessment"] = "UNKNOWN: Insufficient data"

        return stats

    def _spearman_correlation(
        self,
        x: List[float],
        y: List[float]
    ) -> Optional[float]:
        """Calculate Spearman rank correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return None

        n = len(x)

        # Rank the values
        def rank(values):
            sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
            ranks = [0] * len(values)
            for rank_val, idx in enumerate(sorted_indices):
                ranks[idx] = rank_val + 1
            return ranks

        rank_x = rank(x)
        rank_y = rank(y)

        # Calculate Spearman correlation
        d_squared = sum((rx - ry) ** 2 for rx, ry in zip(rank_x, rank_y))
        rho = 1 - (6 * d_squared) / (n * (n ** 2 - 1))

        return rho

    def _calculate_hit_rate(
        self,
        results: List[Dict],
        horizon: int = 30
    ) -> Dict:
        """
        Calculate % of time top quintile beats bottom quintile.

        Returns:
            Dict with hit_rate and details
        """
        returns_key = f"returns_{horizon}d"
        hits = 0
        total = 0

        for period in results:
            scores = period["scores"]
            returns = period.get(returns_key, {})

            common = set(scores.keys()) & set(returns.keys())
            if len(common) < 10:  # Need at least 10 for quintiles
                continue

            # Sort by score
            sorted_tickers = sorted(common, key=lambda t: scores[t], reverse=True)

            # Top and bottom quintile
            quintile_size = len(sorted_tickers) // 5
            if quintile_size < 1:
                continue

            top_quintile = sorted_tickers[:quintile_size]
            bottom_quintile = sorted_tickers[-quintile_size:]

            # Average returns
            top_return = sum(returns[t] for t in top_quintile) / len(top_quintile)
            bottom_return = sum(returns[t] for t in bottom_quintile) / len(bottom_quintile)

            if top_return > bottom_return:
                hits += 1
            total += 1

        return {
            "hit_rate": hits / total if total > 0 else None,
            "hits": hits,
            "total_periods": total,
            "horizon_days": horizon
        }

    def _calculate_turnover(self, results: List[Dict]) -> Optional[float]:
        """
        Calculate how much ranking changes month-to-month.

        Returns rank correlation between consecutive periods.
        """
        if len(results) < 2:
            return None

        correlations = []

        for i in range(1, len(results)):
            prev_scores = results[i-1]["scores"]
            curr_scores = results[i]["scores"]

            common = set(prev_scores.keys()) & set(curr_scores.keys())
            if len(common) < 5:
                continue

            prev_vals = [prev_scores[t] for t in common]
            curr_vals = [curr_scores[t] for t in common]

            corr = self._spearman_correlation(prev_vals, curr_vals)
            if corr is not None:
                correlations.append(corr)

        if correlations:
            avg_corr = sum(correlations) / len(correlations)
            # Turnover = 1 - correlation (high correlation = low turnover)
            return 1 - avg_corr
        return None

    def _median(self, values: List[float]) -> float:
        """Calculate median of a list."""
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n % 2 == 0:
            return (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
        return sorted_vals[n//2]

    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def generate_report(self, stats: Dict) -> str:
        """Generate human-readable backtest report."""
        lines = [
            "=" * 70,
            "POINT-IN-TIME BACKTEST REPORT",
            "=" * 70,
            f"Periods Analyzed: {stats['num_periods']}",
            f"Date Range: {stats['periods_analyzed'][0]} to {stats['periods_analyzed'][-1]}",
            "",
            "-" * 70,
            "INFORMATION COEFFICIENT (IC) METRICS",
            "-" * 70,
        ]

        for key in ["ic_30d", "ic_60d", "ic_90d"]:
            if key in stats:
                ic_data = stats[key]
                if ic_data.get("mean") is not None:
                    lines.append(f"\n{key.upper()}:")
                    lines.append(f"  Mean IC:     {ic_data['mean']:.4f}")
                    lines.append(f"  Median IC:   {ic_data['median']:.4f}")
                    lines.append(f"  Std Dev:     {ic_data['std']:.4f}")
                    lines.append(f"  Min/Max:     {ic_data['min']:.4f} / {ic_data['max']:.4f}")
                    lines.append(f"  % Positive:  {ic_data['pct_positive']:.1%}")
                    if ic_data.get('ic_ir'):
                        lines.append(f"  IC IR:       {ic_data['ic_ir']:.2f}")
                else:
                    lines.append(f"\n{key.upper()}: Insufficient data")

        lines.extend([
            "",
            "-" * 70,
            "HIT RATE & TURNOVER",
            "-" * 70,
        ])

        hit_rate = stats.get("hit_rate_top_quintile", {})
        if hit_rate.get("hit_rate") is not None:
            lines.append(f"Top Quintile Hit Rate: {hit_rate['hit_rate']:.1%} ({hit_rate['hits']}/{hit_rate['total_periods']} periods)")
        else:
            lines.append("Top Quintile Hit Rate: Insufficient data")

        turnover = stats.get("turnover_monthly")
        if turnover is not None:
            lines.append(f"Monthly Turnover: {turnover:.1%}")

        lines.extend([
            "",
            "-" * 70,
            "ASSESSMENT",
            "-" * 70,
            stats.get("assessment", "Unknown"),
            "",
            "=" * 70,
        ])

        return "\n".join(lines)

    def save_results(self, filepath: str) -> None:
        """Save backtest results to file."""
        output = {
            "backtest_config": {
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "version": self.VERSION
            },
            "period_results": self.results
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"Results saved to {filepath}")


def create_sample_scoring_function():
    """
    Create a sample scoring function for testing.

    In production, replace with your actual composite scorer.
    """
    def sample_scorer(ticker: str, data: Dict, as_of_date: datetime) -> Dict:
        """Sample scorer that uses available data fields."""
        score = 50.0  # Base score

        # Financial health component
        cash = data.get("cash", 0)
        burn_rate = data.get("burn_rate", data.get("rd_expense", 0))

        if cash and burn_rate and burn_rate > 0:
            runway_months = cash / (burn_rate / 12)
            if runway_months > 24:
                score += 15
            elif runway_months > 12:
                score += 5
            else:
                score -= 10

        # Clinical progress component
        phase = data.get("phase", data.get("current_phase", ""))
        if "3" in str(phase):
            score += 20
        elif "2" in str(phase):
            score += 10

        # Trial status component
        trial_status = data.get("trial_status", data.get("status", ""))
        if trial_status.lower() in ["recruiting", "active"]:
            score += 10
        elif trial_status.lower() == "completed":
            score += 15

        return {
            "ticker": ticker,
            "final_score": max(0, min(100, score)),
            "components": {
                "base": 50,
                "financial": score - 50
            }
        }

    return sample_scorer


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("POINT-IN-TIME BACKTESTER - DEMO")
    print("=" * 70)

    # Create backtester
    backtester = PointInTimeBacktester(
        start_date="2023-01-01",
        end_date="2024-12-31"
    )

    # Generate test dates
    test_dates = backtester.generate_test_dates(frequency_days=30)
    print(f"\nGenerated {len(test_dates)} test dates")
    print(f"First: {test_dates[0].strftime('%Y-%m-%d')}")
    print(f"Last: {test_dates[-1].strftime('%Y-%m-%d')}")

    # Demo with sample scorer
    print("\nTo run a full backtest:")
    print("  1. Populate data/historical/{TICKER}/{YYYYMMDD}.json with historical snapshots")
    print("  2. Load price data into price_cache")
    print("  3. Call backtester.run_backtest(universe, scoring_function)")
    print("\nExample:")
    print("  results = backtester.run_backtest(")
    print("      universe=['ARGX', 'BBIO', 'INCY'],")
    print("      scoring_function=your_scorer")
    print("  )")
    print("  print(backtester.generate_report(results))")
