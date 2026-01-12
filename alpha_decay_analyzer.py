#!/usr/bin/env python3
"""
Signal Decay Analysis

Measures how quickly alpha signals lose predictive power.

Key Metrics:
- Decay curve: IC at 5d, 10d, 20d, 30d, 60d, 90d horizons
- Half-life: Days until IC drops to 50% of initial value
- Recommended refresh frequency

Typical Patterns:
- Fast decay (5d IC >> 30d IC) = tactical signals, need daily refresh
- Slow decay (30d IC ≈ 60d IC) = strategic signals, weekly refresh OK

Usage:
    analyzer = AlphaDecayAnalyzer()
    results = analyzer.calculate_decay_curve(historical_scores)
    print(f"Half-life: {results['half_life_days']} days")
    print(f"Recommendation: {results['recommended_refresh']}")

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


class AlphaDecayAnalyzer:
    """
    Measures signal half-life: how long does a score remain predictive?

    Critical for:
    - Determining refresh frequency
    - Position holding periods
    - Portfolio turnover optimization
    """

    VERSION = "1.0.0"

    # Default horizons for decay analysis
    DEFAULT_HORIZONS = [5, 10, 20, 30, 60, 90]

    def __init__(self, price_data: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize the decay analyzer.

        Args:
            price_data: Historical price data {ticker: {date_str: price}}
        """
        self.price_data = price_data or {}

    def load_price_data(self, filepath: str) -> None:
        """Load price data from file."""
        try:
            with open(filepath) as f:
                self.price_data = json.load(f)
            print(f"Loaded price data for {len(self.price_data)} tickers")
        except Exception as e:
            print(f"Warning: Could not load price data: {e}")

    def _get_price(
        self,
        ticker: str,
        date: datetime
    ) -> Optional[float]:
        """Get price for ticker on date."""
        if ticker not in self.price_data:
            return None

        date_str = date.strftime('%Y-%m-%d')

        # Try exact date
        if date_str in self.price_data[ticker]:
            return self.price_data[ticker][date_str]

        # Try nearby dates (within 5 days)
        for offset in range(1, 6):
            for delta in [-offset, offset]:
                try_date = (date + timedelta(days=delta)).strftime('%Y-%m-%d')
                if try_date in self.price_data[ticker]:
                    return self.price_data[ticker][try_date]

        return None

    def _calculate_returns(
        self,
        score_date: datetime,
        horizon: int
    ) -> Dict[str, float]:
        """
        Calculate returns from score_date to score_date + horizon.

        Returns:
            Dict mapping ticker to return
        """
        returns = {}
        future_date = score_date + timedelta(days=horizon)

        for ticker in self.price_data:
            price_start = self._get_price(ticker, score_date)
            price_end = self._get_price(ticker, future_date)

            if price_start and price_end and price_start > 0:
                ret = (price_end - price_start) / price_start
                returns[ticker] = ret

        return returns

    def calculate_decay_curve(
        self,
        historical_scores: Dict[datetime, Dict[str, float]],
        horizons: Optional[List[int]] = None
    ) -> Dict:
        """
        Calculate IC at different forward-looking horizons.

        Args:
            historical_scores: {score_date: {ticker: score}}
            horizons: List of forward days [5, 10, 20, 30, 60, 90]

        Returns:
            Dict with decay_curve, half_life_days, recommended_refresh
        """
        if horizons is None:
            horizons = self.DEFAULT_HORIZONS

        decay_curve = {}

        for horizon in horizons:
            ic_values = []

            for score_date, scores in historical_scores.items():
                # Calculate returns at this horizon
                returns = self._calculate_returns(score_date, horizon)

                # Get common tickers
                common = set(scores.keys()) & set(returns.keys())
                if len(common) < 5:
                    continue

                score_vals = [scores[t] for t in common]
                return_vals = [returns[t] for t in common]

                ic = self._spearman_correlation(score_vals, return_vals)
                if ic is not None:
                    ic_values.append(ic)

            if ic_values:
                decay_curve[f"{horizon}d"] = {
                    "mean_ic": sum(ic_values) / len(ic_values),
                    "std_ic": self._std(ic_values),
                    "min_ic": min(ic_values),
                    "max_ic": max(ic_values),
                    "n_periods": len(ic_values),
                    "pct_positive": sum(1 for ic in ic_values if ic > 0) / len(ic_values)
                }
            else:
                decay_curve[f"{horizon}d"] = {
                    "mean_ic": None,
                    "n_periods": 0,
                    "reason": "Insufficient data"
                }

        # Calculate half-life
        half_life = self._estimate_half_life(decay_curve, horizons)

        # Generate recommendation
        recommendation = self._recommend_refresh_frequency(half_life)

        return {
            "decay_curve": decay_curve,
            "half_life_days": half_life,
            "recommended_refresh": recommendation,
            "analysis_summary": self._generate_summary(decay_curve, half_life),
            "version": self.VERSION
        }

    def _estimate_half_life(
        self,
        decay_curve: Dict,
        horizons: List[int]
    ) -> Optional[int]:
        """
        Estimate days until IC drops to 50% of initial value.

        Uses linear interpolation between horizons.
        """
        # Get ICs for each horizon
        ics = []
        for h in horizons:
            key = f"{h}d"
            if key in decay_curve and decay_curve[key].get("mean_ic") is not None:
                ics.append((h, decay_curve[key]["mean_ic"]))

        if len(ics) < 2:
            return None

        # Sort by horizon
        ics.sort(key=lambda x: x[0])

        initial_ic = ics[0][1]
        if initial_ic <= 0:
            return None

        target_ic = initial_ic * 0.5

        # Find where IC drops below target
        for i in range(1, len(ics)):
            current_horizon, current_ic = ics[i]
            prev_horizon, prev_ic = ics[i-1]

            if current_ic <= target_ic and prev_ic > target_ic:
                # Linear interpolation
                if prev_ic != current_ic:
                    fraction = (prev_ic - target_ic) / (prev_ic - current_ic)
                    half_life = prev_horizon + fraction * (current_horizon - prev_horizon)
                    return int(round(half_life))
                else:
                    return current_horizon

        # If never reaches 50%, return max horizon or None
        if ics[-1][1] > target_ic:
            return None  # Signal doesn't decay by 50% within measured period

        return horizons[-1]

    def _recommend_refresh_frequency(self, half_life: Optional[int]) -> str:
        """
        Recommend refresh frequency based on half-life.
        """
        if half_life is None:
            return "Insufficient data to determine - recommend weekly as baseline"

        if half_life < 7:
            return "Daily refresh REQUIRED - fast-decaying signal"
        elif half_life < 15:
            return "Twice-weekly refresh recommended"
        elif half_life < 30:
            return "Weekly refresh recommended"
        elif half_life < 60:
            return "Bi-weekly refresh sufficient"
        else:
            return "Monthly refresh sufficient - slow-decaying strategic signal"

    def _generate_summary(
        self,
        decay_curve: Dict,
        half_life: Optional[int]
    ) -> str:
        """Generate human-readable summary of decay analysis."""
        # Get key ICs
        ic_5d = decay_curve.get("5d", {}).get("mean_ic")
        ic_30d = decay_curve.get("30d", {}).get("mean_ic")
        ic_90d = decay_curve.get("90d", {}).get("mean_ic")

        parts = []

        if ic_5d is not None and ic_30d is not None:
            if ic_5d > 0 and ic_30d > 0:
                decay_ratio = ic_30d / ic_5d
                if decay_ratio < 0.5:
                    parts.append("FAST DECAY: Signal loses predictive power quickly")
                elif decay_ratio > 0.8:
                    parts.append("SLOW DECAY: Signal maintains predictive power")
                else:
                    parts.append("MODERATE DECAY: Typical signal decay pattern")

        if half_life:
            parts.append(f"Half-life of {half_life} days")

        if ic_5d is not None:
            if ic_5d > 0.10:
                parts.append("STRONG short-term signal (IC > 0.10)")
            elif ic_5d > 0.05:
                parts.append("GOOD short-term signal (IC > 0.05)")
            elif ic_5d > 0.02:
                parts.append("WEAK short-term signal (IC > 0.02)")
            else:
                parts.append("POOR short-term signal (IC < 0.02)")

        return " | ".join(parts) if parts else "Insufficient data for summary"

    def analyze_component_decay(
        self,
        component_scores: Dict[str, Dict[datetime, Dict[str, float]]],
        horizons: Optional[List[int]] = None
    ) -> Dict[str, Dict]:
        """
        Analyze decay for each scoring component separately.

        Useful for identifying which components have longer/shorter half-lives.

        Args:
            component_scores: {component_name: {score_date: {ticker: score}}}

        Returns:
            {component_name: decay_analysis}
        """
        if horizons is None:
            horizons = self.DEFAULT_HORIZONS

        results = {}

        for component_name, scores in component_scores.items():
            analysis = self.calculate_decay_curve(scores, horizons)
            results[component_name] = analysis

        # Sort by half-life (longest-lasting first)
        sorted_components = sorted(
            results.items(),
            key=lambda x: x[1].get("half_life_days") or 0,
            reverse=True
        )

        return {
            "by_component": dict(sorted_components),
            "ranking": [
                {
                    "component": name,
                    "half_life": data.get("half_life_days"),
                    "recommended_refresh": data.get("recommended_refresh")
                }
                for name, data in sorted_components
            ]
        }

    def calculate_optimal_holding_period(
        self,
        decay_curve: Dict
    ) -> Dict:
        """
        Suggest optimal holding period based on decay curve.

        Balances:
        - Signal strength (want high IC)
        - Transaction costs (favor longer holding)
        - Decay (don't hold past useful life)
        """
        horizons = []
        ics = []

        for key, data in decay_curve.items():
            if data.get("mean_ic") is not None:
                try:
                    horizon = int(key.replace("d", ""))
                    horizons.append(horizon)
                    ics.append(data["mean_ic"])
                except ValueError:
                    continue

        if not horizons:
            return {
                "optimal_days": None,
                "reason": "Insufficient data"
            }

        # Find horizon with best IC-adjusted-for-costs
        # Simple model: IC * sqrt(horizon) to favor longer periods
        adjusted_scores = [
            (h, ic * math.sqrt(h) if ic > 0 else 0)
            for h, ic in zip(horizons, ics)
        ]

        best = max(adjusted_scores, key=lambda x: x[1])

        return {
            "optimal_days": best[0],
            "raw_ic": ics[horizons.index(best[0])],
            "adjusted_score": best[1],
            "all_horizons": dict(zip(horizons, ics)),
            "recommendation": f"Hold positions for ~{best[0]} days for optimal risk-adjusted returns"
        }

    def _spearman_correlation(
        self,
        x: List[float],
        y: List[float]
    ) -> Optional[float]:
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

    def generate_report(self, analysis: Dict) -> str:
        """Generate human-readable decay analysis report."""
        lines = [
            "=" * 70,
            "ALPHA DECAY ANALYSIS REPORT",
            "=" * 70,
            "",
            f"Half-Life: {analysis.get('half_life_days', 'N/A')} days",
            f"Recommendation: {analysis.get('recommended_refresh', 'N/A')}",
            "",
            "-" * 70,
            "DECAY CURVE",
            "-" * 70,
        ]

        decay_curve = analysis.get("decay_curve", {})
        for key in sorted(decay_curve.keys(), key=lambda x: int(x.replace("d", ""))):
            data = decay_curve[key]
            if data.get("mean_ic") is not None:
                ic = data["mean_ic"]
                n = data.get("n_periods", 0)
                pct_pos = data.get("pct_positive", 0)

                # Visual bar
                bar_length = max(0, min(20, int(ic * 100)))
                bar = "█" * bar_length

                lines.append(
                    f"  {key:6} | IC: {ic:+.4f} | {bar:20} | "
                    f"N: {n:3} | +ve: {pct_pos:.0%}"
                )
            else:
                lines.append(f"  {key:6} | Insufficient data")

        lines.extend([
            "",
            "-" * 70,
            "SUMMARY",
            "-" * 70,
            analysis.get("analysis_summary", "N/A"),
            "",
            "=" * 70,
        ])

        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("ALPHA DECAY ANALYZER - DEMO")
    print("=" * 70)

    analyzer = AlphaDecayAnalyzer()

    # Create sample data for demo
    print("\nTo run decay analysis:")
    print("  1. Load historical price data")
    print("  2. Collect historical scores by date")
    print("  3. Call analyzer.calculate_decay_curve(scores)")
    print("")
    print("Example:")
    print("  historical_scores = {")
    print("      datetime(2024, 1, 15): {'NVAX': 85, 'IMCR': 82, ...},")
    print("      datetime(2024, 2, 15): {'NVAX': 83, 'IMCR': 80, ...},")
    print("      ...")
    print("  }")
    print("  results = analyzer.calculate_decay_curve(historical_scores)")
    print("  print(analyzer.generate_report(results))")

    # Demo with synthetic decay curve
    print("\n" + "-" * 70)
    print("SAMPLE OUTPUT (synthetic data)")
    print("-" * 70)

    sample_decay = {
        "5d": {"mean_ic": 0.12, "n_periods": 20, "pct_positive": 0.85},
        "10d": {"mean_ic": 0.10, "n_periods": 20, "pct_positive": 0.80},
        "20d": {"mean_ic": 0.08, "n_periods": 20, "pct_positive": 0.75},
        "30d": {"mean_ic": 0.06, "n_periods": 20, "pct_positive": 0.70},
        "60d": {"mean_ic": 0.04, "n_periods": 20, "pct_positive": 0.60},
        "90d": {"mean_ic": 0.02, "n_periods": 20, "pct_positive": 0.55},
    }

    sample_analysis = {
        "decay_curve": sample_decay,
        "half_life_days": 25,
        "recommended_refresh": "Weekly refresh recommended",
        "analysis_summary": "MODERATE DECAY | Half-life of 25 days | STRONG short-term signal"
    }

    print(analyzer.generate_report(sample_analysis))
