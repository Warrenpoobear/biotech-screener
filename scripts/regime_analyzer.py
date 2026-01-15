#!/usr/bin/env python3
"""
Market Regime Detection and Analysis

Identifies if signals work differently across market conditions.
Splits historical performance by regime to optimize signal weights.

Key Regimes:
1. Bull/Bear (market trend via SPY 200-day MA)
2. Risk-On/Risk-Off (volatility via VIX level)
3. Rate environment (Fed policy)
4. Sector sentiment (XBI relative performance)

Usage:
    analyzer = RegimeAnalyzer()
    regime = analyzer.classify_regime(datetime(2024, 1, 15))
    print(regime)  # {"trend": "bull", "risk": "on", "rates": "high"}

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import json
import math
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


# Module metadata
__version__ = "1.0.0"


class RegimeAnalyzer:
    """
    Splits historical performance by regime:
    1. Bull/Bear (market trend)
    2. Risk-On/Risk-Off (volatility environment)
    3. Rate environment (Fed policy)
    4. Sector sentiment (XBI relative performance)
    """

    VERSION = "1.0.0"

    # Historical regime definitions (manually curated)
    # Format: period_id -> {trend, risk, rates, xbi_sentiment}
    REGIME_DEFINITIONS = {
        "2019_q1": {"trend": "bull", "risk": "on", "rates": "low", "xbi": "neutral"},
        "2019_q2": {"trend": "bull", "risk": "on", "rates": "low", "xbi": "positive"},
        "2019_q3": {"trend": "bull", "risk": "mixed", "rates": "low", "xbi": "neutral"},
        "2019_q4": {"trend": "bull", "risk": "on", "rates": "low", "xbi": "positive"},
        "2020_q1": {"trend": "crash", "risk": "off", "rates": "zero", "xbi": "negative"},
        "2020_q2": {"trend": "recovery", "risk": "on", "rates": "zero", "xbi": "positive"},
        "2020_q3": {"trend": "bull", "risk": "on", "rates": "zero", "xbi": "positive"},
        "2020_q4": {"trend": "bull", "risk": "on", "rates": "zero", "xbi": "positive"},
        "2021_q1": {"trend": "bull", "risk": "on", "rates": "low", "xbi": "positive"},
        "2021_q2": {"trend": "bull", "risk": "on", "rates": "low", "xbi": "neutral"},
        "2021_q3": {"trend": "bull", "risk": "mixed", "rates": "low", "xbi": "negative"},
        "2021_q4": {"trend": "bear", "risk": "off", "rates": "low", "xbi": "negative"},
        "2022_q1": {"trend": "bear", "risk": "off", "rates": "rising", "xbi": "negative"},
        "2022_q2": {"trend": "bear", "risk": "off", "rates": "rising", "xbi": "negative"},
        "2022_q3": {"trend": "bear", "risk": "off", "rates": "rising", "xbi": "negative"},
        "2022_q4": {"trend": "bear", "risk": "mixed", "rates": "high", "xbi": "neutral"},
        "2023_q1": {"trend": "recovery", "risk": "mixed", "rates": "high", "xbi": "neutral"},
        "2023_q2": {"trend": "bull", "risk": "on", "rates": "high", "xbi": "neutral"},
        "2023_q3": {"trend": "bull", "risk": "mixed", "rates": "high", "xbi": "negative"},
        "2023_q4": {"trend": "bull", "risk": "on", "rates": "high", "xbi": "positive"},
        "2024_q1": {"trend": "bull", "risk": "on", "rates": "high", "xbi": "neutral"},
        "2024_q2": {"trend": "bull", "risk": "on", "rates": "high", "xbi": "neutral"},
        "2024_q3": {"trend": "bull", "risk": "mixed", "rates": "high", "xbi": "neutral"},
        "2024_q4": {"trend": "bull", "risk": "on", "rates": "falling", "xbi": "positive"},
    }

    # Thresholds for dynamic regime classification
    THRESHOLDS = {
        "vix_risk_off": 25,      # VIX > 25 = risk-off
        "vix_risk_on": 18,       # VIX < 18 = risk-on
        "spy_ma_days": 200,      # 200-day MA for trend
        "xbi_outperform": 0.02,  # XBI > SPY by 2% = positive sentiment
        "xbi_underperform": -0.02,
    }

    def __init__(self, market_data_file: Optional[str] = None):
        """
        Initialize the regime analyzer.

        Args:
            market_data_file: Path to historical market data (VIX, SPY, XBI)
        """
        self.market_data: Dict[str, Dict[str, float]] = {}
        if market_data_file and Path(market_data_file).exists():
            self._load_market_data(market_data_file)

    def _load_market_data(self, filepath: str) -> None:
        """Load historical market data for dynamic classification."""
        try:
            with open(filepath) as f:
                self.market_data = json.load(f)
            print(f"Loaded market data with {len(self.market_data)} dates")
        except Exception as e:
            print(f"Warning: Could not load market data: {e}")

    def _get_quarter_key(self, date: datetime) -> str:
        """Convert date to quarter key (e.g., '2024_q1')."""
        quarter = (date.month - 1) // 3 + 1
        return f"{date.year}_q{quarter}"

    def classify_regime(self, date: datetime) -> Dict[str, str]:
        """
        Classify market regime at given date.

        Uses static definitions first, falls back to dynamic calculation.

        Args:
            date: Date to classify

        Returns:
            Dict with trend, risk, rates, xbi keys
        """
        quarter_key = self._get_quarter_key(date)

        # Try static definition first
        if quarter_key in self.REGIME_DEFINITIONS:
            return self.REGIME_DEFINITIONS[quarter_key].copy()

        # Fall back to dynamic classification
        return self._classify_dynamic(date)

    def _classify_dynamic(self, date: datetime) -> Dict[str, str]:
        """
        Dynamically classify regime using market data.

        Requires market_data to be loaded.
        """
        date_str = date.strftime('%Y-%m-%d')

        if date_str not in self.market_data:
            return {
                "trend": "unknown",
                "risk": "unknown",
                "rates": "unknown",
                "xbi": "unknown"
            }

        data = self.market_data[date_str]

        # Trend: SPY vs 200-day MA
        spy_price = data.get("spy_close")
        spy_ma200 = data.get("spy_ma200")
        if spy_price and spy_ma200:
            if spy_price > spy_ma200 * 1.05:
                trend = "bull"
            elif spy_price < spy_ma200 * 0.95:
                trend = "bear"
            else:
                trend = "mixed"
        else:
            trend = "unknown"

        # Risk: VIX level
        vix = data.get("vix")
        if vix:
            if vix > self.THRESHOLDS["vix_risk_off"]:
                risk = "off"
            elif vix < self.THRESHOLDS["vix_risk_on"]:
                risk = "on"
            else:
                risk = "mixed"
        else:
            risk = "unknown"

        # Rates: Based on 10Y treasury
        rate_10y = data.get("rate_10y")
        if rate_10y:
            if rate_10y < 1.0:
                rates = "zero"
            elif rate_10y < 2.5:
                rates = "low"
            elif rate_10y < 4.0:
                rates = "rising"
            else:
                rates = "high"
        else:
            rates = "unknown"

        # XBI sentiment: XBI vs SPY relative performance (30-day)
        xbi_rel = data.get("xbi_spy_30d_rel")
        if xbi_rel:
            if xbi_rel > self.THRESHOLDS["xbi_outperform"]:
                xbi = "positive"
            elif xbi_rel < self.THRESHOLDS["xbi_underperform"]:
                xbi = "negative"
            else:
                xbi = "neutral"
        else:
            xbi = "unknown"

        return {
            "trend": trend,
            "risk": risk,
            "rates": rates,
            "xbi": xbi
        }

    def split_results_by_regime(
        self,
        backtest_results: List[Dict],
        regime_dimension: str = "trend"
    ) -> Dict[str, List[Dict]]:
        """
        Group backtest periods by regime.

        Args:
            backtest_results: List of period results from backtester
            regime_dimension: Which dimension to split by (trend, risk, rates, xbi)

        Returns:
            Dict mapping regime values to list of periods
        """
        regime_buckets: Dict[str, List[Dict]] = {}

        for period in backtest_results:
            date = datetime.fromisoformat(period["date"])
            regime = self.classify_regime(date)

            regime_value = regime.get(regime_dimension, "unknown")

            if regime_value not in regime_buckets:
                regime_buckets[regime_value] = []

            regime_buckets[regime_value].append(period)

        return regime_buckets

    def calculate_regime_ic(
        self,
        periods: List[Dict],
        horizon: str = "30d"
    ) -> Dict[str, Any]:
        """
        Calculate IC statistics for a group of periods.

        Args:
            periods: List of period results
            horizon: Return horizon (30d, 60d, 90d)

        Returns:
            Dict with IC mean, std, n_periods
        """
        ic_values = []

        for period in periods:
            scores = period.get("scores", {})
            returns = period.get(f"returns_{horizon}", {})

            common = set(scores.keys()) & set(returns.keys())
            if len(common) < 5:
                continue

            score_vals = [scores[t] for t in common]
            return_vals = [returns[t] for t in common]

            ic = self._spearman_correlation(score_vals, return_vals)
            if ic is not None:
                ic_values.append(ic)

        if not ic_values:
            return {
                "ic_mean": None,
                "ic_std": None,
                "n_periods": 0,
                "reason": "Insufficient data"
            }

        return {
            "ic_mean": sum(ic_values) / len(ic_values),
            "ic_std": self._std(ic_values),
            "ic_min": min(ic_values),
            "ic_max": max(ic_values),
            "n_periods": len(ic_values),
            "pct_positive": sum(1 for ic in ic_values if ic > 0) / len(ic_values)
        }

    def analyze_by_regime(
        self,
        backtest_results: List[Dict],
        dimensions: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Dict]]:
        """
        Full regime analysis across multiple dimensions.

        Args:
            backtest_results: Backtest period results
            dimensions: List of dimensions to analyze

        Returns:
            Nested dict: dimension -> regime_value -> IC stats
        """
        if dimensions is None:
            dimensions = ["trend", "risk", "rates", "xbi"]

        analysis = {}

        for dimension in dimensions:
            buckets = self.split_results_by_regime(backtest_results, dimension)

            dimension_stats = {}
            for regime_value, periods in buckets.items():
                if len(periods) >= 3:  # Need minimum periods
                    stats = self.calculate_regime_ic(periods)
                    dimension_stats[regime_value] = stats

            analysis[dimension] = dimension_stats

        return analysis

    def recommend_dynamic_weights(
        self,
        regime_analysis: Dict[str, Dict[str, Dict]],
        current_regime: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Suggest signal weight adjustments based on current regime.

        Args:
            regime_analysis: Output from analyze_by_regime
            current_regime: Current market regime

        Returns:
            Recommendations for weight adjustments
        """
        recommendations = {
            "current_regime": current_regime,
            "adjustments": [],
            "confidence": "low"
        }

        strong_signals = 0
        weak_signals = 0

        for dimension, stats_by_value in regime_analysis.items():
            current_value = current_regime.get(dimension, "unknown")

            if current_value in stats_by_value:
                ic_mean = stats_by_value[current_value].get("ic_mean")
                n_periods = stats_by_value[current_value].get("n_periods", 0)

                if ic_mean is not None and n_periods >= 3:
                    if ic_mean > 0.08:
                        recommendations["adjustments"].append({
                            "dimension": dimension,
                            "regime": current_value,
                            "action": "INCREASE weight by 20%",
                            "reason": f"Strong IC ({ic_mean:.3f}) in this regime",
                            "confidence": "high"
                        })
                        strong_signals += 1
                    elif ic_mean > 0.05:
                        recommendations["adjustments"].append({
                            "dimension": dimension,
                            "regime": current_value,
                            "action": "MAINTAIN current weight",
                            "reason": f"Adequate IC ({ic_mean:.3f}) in this regime",
                            "confidence": "medium"
                        })
                    elif ic_mean < 0.02:
                        recommendations["adjustments"].append({
                            "dimension": dimension,
                            "regime": current_value,
                            "action": "DECREASE weight by 30%",
                            "reason": f"Weak IC ({ic_mean:.3f}) in this regime",
                            "confidence": "medium"
                        })
                        weak_signals += 1

        # Overall confidence
        if strong_signals >= 2 and weak_signals == 0:
            recommendations["confidence"] = "high"
        elif weak_signals >= 2:
            recommendations["confidence"] = "low"
        else:
            recommendations["confidence"] = "medium"

        # Overall recommendation
        if recommendations["confidence"] == "high":
            recommendations["summary"] = "Current regime is favorable - increase exposure"
        elif recommendations["confidence"] == "low":
            recommendations["summary"] = "Current regime is unfavorable - reduce exposure"
        else:
            recommendations["summary"] = "Mixed signals - maintain standard exposure"

        return recommendations

    def _spearman_correlation(self, x: List[float], y: List[float]) -> Optional[float]:
        """Calculate Spearman rank correlation."""
        if len(x) != len(y) or len(x) < 2:
            return None

        n = len(x)

        def rank(values):
            sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
            ranks = [0] * len(values)
            for rank_val, idx in enumerate(sorted_indices):
                ranks[idx] = rank_val + 1
            return ranks

        rank_x = rank(x)
        rank_y = rank(y)

        d_squared = sum((rx - ry) ** 2 for rx, ry in zip(rank_x, rank_y))
        rho = 1 - (6 * d_squared) / (n * (n ** 2 - 1))

        return rho

    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def generate_report(self, analysis: Dict[str, Dict[str, Dict]]) -> str:
        """Generate human-readable regime analysis report."""
        lines = [
            "=" * 70,
            "REGIME ANALYSIS REPORT",
            "=" * 70,
        ]

        for dimension, stats_by_value in analysis.items():
            lines.append(f"\n{dimension.upper()} DIMENSION:")
            lines.append("-" * 40)

            for regime_value, stats in sorted(stats_by_value.items()):
                ic_mean = stats.get("ic_mean")
                n = stats.get("n_periods", 0)

                if ic_mean is not None:
                    assessment = "✓ STRONG" if ic_mean > 0.05 else "✗ WEAK" if ic_mean < 0.02 else "○ OK"
                    lines.append(
                        f"  {regime_value:12} | IC: {ic_mean:+.4f} | "
                        f"N: {n:3} | {assessment}"
                    )
                else:
                    lines.append(f"  {regime_value:12} | Insufficient data")

        lines.extend([
            "",
            "=" * 70,
        ])

        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("REGIME ANALYZER - DEMO")
    print("=" * 70)

    analyzer = RegimeAnalyzer()

    # Test regime classification
    test_dates = [
        datetime(2020, 3, 15),  # COVID crash
        datetime(2021, 6, 15),  # Bull market
        datetime(2022, 6, 15),  # Bear market
        datetime(2024, 1, 15),  # Current
    ]

    print("\nRegime Classification:")
    for date in test_dates:
        regime = analyzer.classify_regime(date)
        print(f"  {date.strftime('%Y-%m-%d')}: {regime}")

    print("\nTo run full regime analysis:")
    print("  1. Run backtester to get period results")
    print("  2. Call analyzer.analyze_by_regime(backtest_results)")
    print("  3. Call analyzer.recommend_dynamic_weights(analysis, current_regime)")
