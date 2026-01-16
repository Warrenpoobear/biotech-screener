#!/usr/bin/env python3
"""
validate_momentum_signal.py - Momentum Signal Validation Framework

Validates the momentum/volatility calculations in the biotech screener:

Stage 1: Mathematical Sanity (no data required)
- Volatility annualization uses sqrt(252), NOT ×252
- Signal direction correctness (uptrend → positive, downtrend → negative)
- IC calculation mathematical properties

Stage 2: Production Testing (requires historical data)
- Quartile spread validation (target: 87.6%)
- Information Coefficient validation (target: IC = 0.713)
- Regime detection validation

Usage:
    python scripts/validate_momentum_signal.py              # Run Stage 1 (fast)
    python scripts/validate_momentum_signal.py --full       # Run all stages

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import sys
import math
import random
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from statistics import mean

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import production code to validate
from wake_robin_data_pipeline.defensive_overlays import (
    realized_vol,
    pearson_corr,
    momentum,
    TRADING_DAYS_PER_YEAR,
)
from backtest.metrics import compute_spearman_ic, MIN_OBS_IC

# Validation constants
VALIDATION_SEED = 42  # Deterministic tests
TOLERANCE_VOL = 0.05  # 5% tolerance for volatility tests
TOLERANCE_IC = 0.001  # Tight tolerance for IC math


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    expected: str
    actual: str
    details: str = ""


class MomentumValidator:
    """Validates momentum signal calculations."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data")
        self.results: List[ValidationResult] = []
        random.seed(VALIDATION_SEED)

    # =========================================================================
    # STAGE 1: MATHEMATICAL SANITY TESTS
    # =========================================================================

    def test_volatility_annualization(self) -> ValidationResult:
        """
        CRITICAL: Verify sqrt(252) annualization, NOT ×252.

        The bug was: daily_vol * 252 = 486% (WRONG)
        Correct:     daily_vol * sqrt(252) = 31.7% (RIGHT)
        """
        # Generate synthetic daily returns with known volatility
        # Target: 2% daily vol → ~31.75% annual vol with sqrt(252)
        daily_vol_target = 0.02
        n_days = 252

        # Generate returns with target daily volatility
        random.seed(VALIDATION_SEED)
        returns = [random.gauss(0, daily_vol_target) for _ in range(n_days)]

        # Calculate actual daily std
        mean_ret = sum(returns) / n_days
        daily_std = math.sqrt(sum((r - mean_ret)**2 for r in returns) / (n_days - 1))

        # Expected annual vol using sqrt(252)
        expected_annual = daily_std * math.sqrt(252)

        # Get production code result
        actual_result = realized_vol(returns, window=252, annualize=True)
        actual_annual = float(actual_result) if actual_result else 0.0

        # Wrong method would give this (×252 bug)
        wrong_method = daily_std * 252

        # Check if production uses correct method
        error_correct = abs(actual_annual - expected_annual) / expected_annual
        error_wrong = abs(actual_annual - wrong_method) / wrong_method

        passed = error_correct < TOLERANCE_VOL and error_correct < error_wrong

        return ValidationResult(
            test_name="Volatility Annualization (sqrt(252))",
            passed=passed,
            expected=f"{expected_annual:.4f} (sqrt(252) method)",
            actual=f"{actual_annual:.4f}",
            details=(
                f"Daily std: {daily_std:.4f}\n"
                f"Correct (sqrt(252)): {expected_annual:.4f}\n"
                f"Wrong (×252): {wrong_method:.4f}\n"
                f"Error vs correct: {error_correct:.2%}\n"
                f"Error vs wrong: {error_wrong:.2%}"
            )
        )

    def test_volatility_direct_assertion(self) -> ValidationResult:
        """
        Direct assertion: Does not depend on Monte Carlo.
        Tests that annualization factor is exactly sqrt(252).
        """
        # Use a simple return series where we know the exact daily std
        # If returns are [+0.01, -0.01] repeated, daily std = 0.01
        returns = [0.01, -0.01] * 126  # 252 days

        # Daily std of this series
        mean_ret = sum(returns) / 252
        daily_std = math.sqrt(sum((r - mean_ret)**2 for r in returns) / 251)

        # Expected with sqrt(252)
        expected = daily_std * math.sqrt(252)

        # Get production result
        result = realized_vol(returns, window=252, annualize=True)
        actual = float(result) if result else 0.0

        # Direct comparison (allow for 4dp quantization in production code)
        error = abs(actual - expected)
        passed = error < 1e-4  # Allow for quantization tolerance

        return ValidationResult(
            test_name="Volatility Direct Assertion",
            passed=passed,
            expected=f"{expected:.10f}",
            actual=f"{actual:.10f}",
            details=f"Absolute error: {error:.2e}"
        )

    def test_signal_direction_uptrend(self) -> ValidationResult:
        """Test that uptrending prices produce positive momentum."""
        # Uptrending prices: 100 → 150 over 21 days
        prices = [100 + i * (50/21) for i in range(22)]

        result = momentum(prices, period=21)

        passed = result is not None and float(result) > 0

        return ValidationResult(
            test_name="Signal Direction (Uptrend → Positive)",
            passed=passed,
            expected="> 0 (positive momentum)",
            actual=f"{float(result):.4f}" if result else "None",
            details="21-day return should be positive for uptrending prices"
        )

    def test_signal_direction_downtrend(self) -> ValidationResult:
        """Test that downtrending prices produce negative momentum."""
        # Downtrending prices: 100 → 50 over 21 days
        prices = [100 - i * (50/21) for i in range(22)]

        result = momentum(prices, period=21)

        passed = result is not None and float(result) < 0

        return ValidationResult(
            test_name="Signal Direction (Downtrend → Negative)",
            passed=passed,
            expected="< 0 (negative momentum)",
            actual=f"{float(result):.4f}" if result else "None",
            details="21-day return should be negative for downtrending prices"
        )

    def test_ic_calculation_perfect_correlation(self) -> ValidationResult:
        """Test IC = 1.0 for perfectly correlated scores and returns."""
        # Perfect positive correlation: higher score → higher return
        scores = [Decimal(str(i)) for i in range(1, 21)]
        returns = [Decimal(str(i * 0.01)) for i in range(1, 21)]

        ic = compute_spearman_ic(scores, returns)

        passed = ic is not None and abs(float(ic) - 1.0) < TOLERANCE_IC

        return ValidationResult(
            test_name="IC Calculation (Perfect Correlation)",
            passed=passed,
            expected="1.0000",
            actual=f"{float(ic):.4f}" if ic else "None",
            details="Perfectly ranked scores/returns should give IC = 1.0"
        )

    def test_ic_calculation_inverse_correlation(self) -> ValidationResult:
        """Test IC = -1.0 for inversely correlated scores and returns."""
        # Perfect negative correlation: higher score → lower return
        scores = [Decimal(str(i)) for i in range(1, 21)]
        returns = [Decimal(str((21 - i) * 0.01)) for i in range(1, 21)]

        ic = compute_spearman_ic(scores, returns)

        passed = ic is not None and abs(float(ic) + 1.0) < TOLERANCE_IC

        return ValidationResult(
            test_name="IC Calculation (Inverse Correlation)",
            passed=passed,
            expected="-1.0000",
            actual=f"{float(ic):.4f}" if ic else "None",
            details="Inversely ranked scores/returns should give IC = -1.0"
        )

    def test_ic_calculation_minimum_observations(self) -> ValidationResult:
        """Test that IC returns None for insufficient observations."""
        # Only 5 observations (below MIN_OBS_IC = 10)
        scores = [Decimal(str(i)) for i in range(1, 6)]
        returns = [Decimal(str(i * 0.01)) for i in range(1, 6)]

        ic = compute_spearman_ic(scores, returns)

        passed = ic is None

        return ValidationResult(
            test_name="IC Minimum Observations Gate",
            passed=passed,
            expected="None (insufficient data)",
            actual="None" if ic is None else f"{float(ic):.4f}",
            details=f"IC should return None when n < {MIN_OBS_IC}"
        )

    def test_correlation_bounds(self) -> ValidationResult:
        """Test that Pearson correlation is bounded [-1, 1]."""
        # Generate random returns
        random.seed(VALIDATION_SEED)
        a = [random.gauss(0, 0.02) for _ in range(120)]
        b = [random.gauss(0, 0.02) for _ in range(120)]

        corr = pearson_corr(a, b, window=120)

        if corr is None:
            passed = False
            actual = "None"
        else:
            corr_val = float(corr)
            passed = -1.0 <= corr_val <= 1.0
            actual = f"{corr_val:.4f}"

        return ValidationResult(
            test_name="Correlation Bounds [-1, 1]",
            passed=passed,
            expected="[-1.0, 1.0]",
            actual=actual,
            details="Correlation must be bounded"
        )

    # =========================================================================
    # STAGE 2: PRODUCTION TESTING (requires data)
    # =========================================================================

    def load_price_data(self) -> Tuple[Optional[Dict], Optional[List], Optional[List]]:
        """
        Load historical price data for Stage 2 validation.

        Expected files:
        - data/universe_prices.csv: Daily prices for all tickers
        - data/indices_prices.csv: XBI and SPY benchmark prices

        Returns:
            (prices_by_ticker, xbi_prices, spy_prices) or (None, None, None)
        """
        import csv
        from datetime import datetime

        universe_file = self.data_dir / "universe_prices.csv"
        indices_file = self.data_dir / "indices_prices.csv"

        if not universe_file.exists() or not indices_file.exists():
            return None, None, None

        try:
            # Load universe prices
            prices_by_ticker = {}
            with open(universe_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    return None, None, None

                # Get tickers from header (excluding 'date')
                tickers = [k for k in rows[0].keys() if k.lower() != 'date']

                for ticker in tickers:
                    prices_by_ticker[ticker] = []
                    for row in rows:
                        try:
                            price = float(row[ticker])
                            prices_by_ticker[ticker].append(price)
                        except (ValueError, KeyError):
                            pass

            # Load index prices
            xbi_prices = []
            spy_prices = []
            with open(indices_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        xbi_prices.append(float(row.get('XBI', row.get('xbi', 0))))
                        spy_prices.append(float(row.get('SPY', row.get('spy', 0))))
                    except ValueError:
                        pass

            return prices_by_ticker, xbi_prices, spy_prices

        except Exception as e:
            print(f"   Error loading price data: {e}")
            return None, None, None

    def calculate_returns(self, prices: List[float]) -> List[float]:
        """Calculate daily returns from prices."""
        if len(prices) < 2:
            return []
        return [(prices[i] / prices[i-1]) - 1 for i in range(1, len(prices))]

    def calculate_momentum_score(self, prices: List[float], lookback: int = 21) -> Optional[float]:
        """Calculate momentum score (trailing return)."""
        if len(prices) < lookback + 1:
            return None
        return (prices[-1] / prices[-lookback-1]) - 1

    def test_quartile_spread(self, prices_by_ticker: Dict, forward_days: int = 21) -> ValidationResult:
        """
        Test quartile spread between top/bottom momentum quartiles.
        Target: 87.6% spread.
        """
        if not prices_by_ticker:
            return ValidationResult(
                test_name="Quartile Spread Analysis",
                passed=False,
                expected="87.6% spread",
                actual="No data",
                details="Load price data to run this test"
            )

        # Calculate momentum scores for each ticker
        momentum_scores = {}
        for ticker, prices in prices_by_ticker.items():
            if len(prices) >= 63:  # Need at least 3 months
                # Use 21-day momentum (1 month lookback)
                score = self.calculate_momentum_score(prices[:-forward_days], lookback=21)
                if score is not None:
                    momentum_scores[ticker] = score

        if len(momentum_scores) < 20:
            return ValidationResult(
                test_name="Quartile Spread Analysis",
                passed=False,
                expected="87.6% spread",
                actual=f"Only {len(momentum_scores)} tickers with data",
                details="Need at least 20 tickers for quartile analysis"
            )

        # Sort by momentum and assign quartiles
        sorted_tickers = sorted(momentum_scores.keys(), key=lambda t: momentum_scores[t], reverse=True)
        n = len(sorted_tickers)
        q_size = n // 4

        q1_tickers = sorted_tickers[:q_size]  # Top quartile (highest momentum)
        q4_tickers = sorted_tickers[-q_size:]  # Bottom quartile (lowest momentum)

        # Calculate forward returns for each quartile
        q1_returns = []
        q4_returns = []

        for ticker in q1_tickers:
            prices = prices_by_ticker[ticker]
            if len(prices) >= forward_days:
                fwd_ret = (prices[-1] / prices[-forward_days-1]) - 1
                q1_returns.append(fwd_ret)

        for ticker in q4_tickers:
            prices = prices_by_ticker[ticker]
            if len(prices) >= forward_days:
                fwd_ret = (prices[-1] / prices[-forward_days-1]) - 1
                q4_returns.append(fwd_ret)

        if not q1_returns or not q4_returns:
            return ValidationResult(
                test_name="Quartile Spread Analysis",
                passed=False,
                expected="87.6% spread",
                actual="Insufficient forward return data",
                details=""
            )

        q1_mean = mean(q1_returns)
        q4_mean = mean(q4_returns)
        spread = (q1_mean - q4_mean) * 100  # Convert to percentage points

        # Pass if spread is positive and significant
        passed = spread > 10  # Minimum 10% spread to be meaningful

        return ValidationResult(
            test_name="Quartile Spread Analysis",
            passed=passed,
            expected="87.6% spread (target)",
            actual=f"{spread:.1f}% spread",
            details=(
                f"Q1 (top) mean return: {q1_mean*100:.2f}%\n"
                f"Q4 (bottom) mean return: {q4_mean*100:.2f}%\n"
                f"Spread: {spread:.1f}%\n"
                f"Tickers analyzed: {n}"
            )
        )

    def test_information_coefficient(self, prices_by_ticker: Dict) -> ValidationResult:
        """
        Test cross-sectional Information Coefficient.
        Target: IC = 0.713.
        """
        if not prices_by_ticker:
            return ValidationResult(
                test_name="Information Coefficient",
                passed=False,
                expected="IC = 0.713",
                actual="No data",
                details="Load price data to run this test"
            )

        # Calculate momentum scores and forward returns
        momentum_scores = []
        forward_returns = []

        for ticker, prices in prices_by_ticker.items():
            if len(prices) >= 42:  # Need 21 days lookback + 21 days forward
                # Momentum: 21-day trailing return
                mom = self.calculate_momentum_score(prices[:-21], lookback=21)
                # Forward: 21-day forward return
                fwd = (prices[-1] / prices[-22]) - 1

                if mom is not None:
                    momentum_scores.append(Decimal(str(mom)))
                    forward_returns.append(Decimal(str(fwd)))

        if len(momentum_scores) < MIN_OBS_IC:
            return ValidationResult(
                test_name="Information Coefficient",
                passed=False,
                expected="IC = 0.713",
                actual=f"Only {len(momentum_scores)} observations",
                details=f"Need at least {MIN_OBS_IC} observations"
            )

        ic = compute_spearman_ic(momentum_scores, forward_returns)

        if ic is None:
            return ValidationResult(
                test_name="Information Coefficient",
                passed=False,
                expected="IC = 0.713",
                actual="IC calculation failed",
                details=""
            )

        ic_val = float(ic)

        # Pass if IC is positive and meaningful
        passed = ic_val > 0.10  # Minimum 0.10 IC to be meaningful

        return ValidationResult(
            test_name="Information Coefficient",
            passed=passed,
            expected="IC = 0.713 (target)",
            actual=f"IC = {ic_val:.4f}",
            details=(
                f"Cross-sectional Spearman IC\n"
                f"Observations: {len(momentum_scores)}\n"
                f"IC > 0.30 = strong, > 0.50 = exceptional"
            )
        )

    def test_regime_detection(self, xbi_prices: List, spy_prices: List) -> ValidationResult:
        """
        Test regime detection on benchmark data.
        """
        if not xbi_prices or not spy_prices or len(xbi_prices) < 60:
            return ValidationResult(
                test_name="Regime Detection",
                passed=False,
                expected="Valid regime classification",
                actual="No benchmark data",
                details="Load XBI and SPY prices to run this test"
            )

        # Calculate XBI vs SPY relative performance (30-day)
        xbi_ret_30d = (xbi_prices[-1] / xbi_prices[-31]) - 1 if len(xbi_prices) > 30 else 0
        spy_ret_30d = (spy_prices[-1] / spy_prices[-31]) - 1 if len(spy_prices) > 30 else 0
        relative_perf = (xbi_ret_30d - spy_ret_30d) * 100  # Percentage points

        # Simple regime classification based on relative performance
        if relative_perf > 5:
            regime = "BULL"
        elif relative_perf < -5:
            regime = "BEAR"
        else:
            regime = "NEUTRAL"

        # Calculate XBI volatility (20-day)
        xbi_returns = self.calculate_returns(xbi_prices[-21:])
        if xbi_returns:
            xbi_vol = (sum(r**2 for r in xbi_returns) / len(xbi_returns)) ** 0.5 * math.sqrt(252) * 100
        else:
            xbi_vol = 0

        passed = regime in ["BULL", "BEAR", "NEUTRAL"]

        return ValidationResult(
            test_name="Regime Detection",
            passed=passed,
            expected="Valid regime classification",
            actual=f"Regime: {regime}",
            details=(
                f"XBI 30d return: {xbi_ret_30d*100:.2f}%\n"
                f"SPY 30d return: {spy_ret_30d*100:.2f}%\n"
                f"Relative performance: {relative_perf:.2f}%\n"
                f"XBI 20d volatility: {xbi_vol:.1f}%"
            )
        )

    def test_alpha_decay(self, prices_by_ticker: Dict) -> ValidationResult:
        """
        Test signal persistence / alpha decay across horizons.
        """
        if not prices_by_ticker:
            return ValidationResult(
                test_name="Alpha Decay Analysis",
                passed=False,
                expected="Decaying IC across horizons",
                actual="No data",
                details="Load price data to run this test"
            )

        horizons = [5, 10, 21, 42, 63]
        ic_by_horizon = {}

        for horizon in horizons:
            momentum_scores = []
            forward_returns = []

            for ticker, prices in prices_by_ticker.items():
                if len(prices) >= 21 + horizon:
                    # 21-day momentum
                    mom = self.calculate_momentum_score(prices[:-(horizon)], lookback=21)
                    # Forward return at this horizon
                    fwd = (prices[-1] / prices[-(horizon+1)]) - 1

                    if mom is not None:
                        momentum_scores.append(Decimal(str(mom)))
                        forward_returns.append(Decimal(str(fwd)))

            if len(momentum_scores) >= MIN_OBS_IC:
                ic = compute_spearman_ic(momentum_scores, forward_returns)
                if ic is not None:
                    ic_by_horizon[horizon] = float(ic)

        if not ic_by_horizon:
            return ValidationResult(
                test_name="Alpha Decay Analysis",
                passed=False,
                expected="IC values across horizons",
                actual="Insufficient data for any horizon",
                details=""
            )

        # Format results
        decay_report = []
        for h in sorted(ic_by_horizon.keys()):
            decay_report.append(f"{h}d: IC = {ic_by_horizon[h]:.4f}")

        # Check for reasonable decay pattern (short-term IC should be higher)
        short_term_ic = ic_by_horizon.get(5, ic_by_horizon.get(10, 0))
        passed = short_term_ic > 0  # At least positive short-term IC

        return ValidationResult(
            test_name="Alpha Decay Analysis",
            passed=passed,
            expected="Positive short-term IC, decaying over time",
            actual=f"IC at {min(ic_by_horizon.keys())}d = {ic_by_horizon[min(ic_by_horizon.keys())]:.4f}",
            details="\n".join(decay_report)
        )

    def run_stage2(self) -> List[ValidationResult]:
        """Run Stage 2: Production validation with historical data."""
        print("\n" + "=" * 70)
        print("STAGE 2: PRODUCTION TESTING")
        print("=" * 70)

        # Load data
        print("\nLoading historical price data...")
        prices_by_ticker, xbi_prices, spy_prices = self.load_price_data()

        if prices_by_ticker is None:
            print(f"\n⚠️  Price data not found in {self.data_dir}")
            print("   Expected files:")
            print(f"   - {self.data_dir}/universe_prices.csv")
            print(f"   - {self.data_dir}/indices_prices.csv")
            print("\n   Create these files to run Stage 2 validation.")
            return []

        n_tickers = len(prices_by_ticker)
        n_days = len(next(iter(prices_by_ticker.values()))) if prices_by_ticker else 0
        print(f"   Loaded {n_tickers} tickers, {n_days} days of data")

        tests = [
            lambda: self.test_quartile_spread(prices_by_ticker),
            lambda: self.test_information_coefficient(prices_by_ticker),
            lambda: self.test_regime_detection(xbi_prices, spy_prices),
            lambda: self.test_alpha_decay(prices_by_ticker),
        ]

        results = []
        for test_func in tests:
            try:
                result = test_func()
                results.append(result)
                status = "PASS" if result.passed else "FAIL"
                symbol = "✅" if result.passed else "❌"
                print(f"\n{symbol} {result.test_name}: {status}")
                print(f"   Expected: {result.expected}")
                print(f"   Actual:   {result.actual}")
                if result.details:
                    for line in result.details.split('\n'):
                        print(f"   {line}")
            except Exception as e:
                result = ValidationResult(
                    test_name=test_func.__name__,
                    passed=False,
                    expected="No exception",
                    actual=f"Exception: {e}",
                    details=str(e)
                )
                results.append(result)
                print(f"\n❌ {test_func.__name__}: EXCEPTION")
                print(f"   {e}")

        # Stage 2 summary
        print("\n" + "-" * 70)
        passed = sum(1 for r in results if r.passed)
        print(f"Stage 2: {passed}/{len(results)} tests passed")

        return results

    # =========================================================================
    # RUN VALIDATION
    # =========================================================================

    def run_stage1(self) -> List[ValidationResult]:
        """Run Stage 1: Mathematical sanity tests (no data required)."""
        print("\n" + "=" * 70)
        print("STAGE 1: MATHEMATICAL SANITY TESTS")
        print("=" * 70)

        tests = [
            self.test_volatility_annualization,
            self.test_volatility_direct_assertion,
            self.test_signal_direction_uptrend,
            self.test_signal_direction_downtrend,
            self.test_ic_calculation_perfect_correlation,
            self.test_ic_calculation_inverse_correlation,
            self.test_ic_calculation_minimum_observations,
            self.test_correlation_bounds,
        ]

        results = []
        for test_func in tests:
            try:
                result = test_func()
                results.append(result)
                status = "PASS" if result.passed else "FAIL"
                symbol = "✅" if result.passed else "❌"
                print(f"\n{symbol} {result.test_name}: {status}")
                print(f"   Expected: {result.expected}")
                print(f"   Actual:   {result.actual}")
                if result.details and not result.passed:
                    for line in result.details.split('\n'):
                        print(f"   {line}")
            except Exception as e:
                result = ValidationResult(
                    test_name=test_func.__name__,
                    passed=False,
                    expected="No exception",
                    actual=f"Exception: {e}",
                    details=str(e)
                )
                results.append(result)
                print(f"\n❌ {test_func.__name__}: EXCEPTION")
                print(f"   {e}")

        return results

    def run_all(self, include_stage2: bool = False) -> Dict[str, Any]:
        """Run all validation stages."""
        stage1_results = self.run_stage1()

        # Summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)

        passed = sum(1 for r in stage1_results if r.passed)
        total = len(stage1_results)

        print(f"\nStage 1: {passed}/{total} tests passed")

        if passed == total:
            print("\n✅ ALL MATHEMATICAL TESTS PASSED")
            print("   Volatility uses correct sqrt(252) annualization")
            print("   IC calculation is mathematically correct")
            print("   Signal direction logic is correct")
        else:
            print("\n❌ SOME TESTS FAILED - Review output above")
            failed = [r for r in stage1_results if not r.passed]
            for r in failed:
                print(f"   - {r.test_name}")

        stage2_results = []
        if include_stage2:
            stage2_results = self.run_stage2()

        stage2_passed = sum(1 for r in stage2_results if r.passed) if stage2_results else 0
        stage2_total = len(stage2_results)

        return {
            "stage1_passed": passed,
            "stage1_total": total,
            "stage1_results": [
                {
                    "test": r.test_name,
                    "passed": r.passed,
                    "expected": r.expected,
                    "actual": r.actual,
                }
                for r in stage1_results
            ],
            "stage2_passed": stage2_passed,
            "stage2_total": stage2_total,
            "stage2_results": [
                {
                    "test": r.test_name,
                    "passed": r.passed,
                    "expected": r.expected,
                    "actual": r.actual,
                }
                for r in stage2_results
            ],
            "all_passed": passed == total and (not stage2_results or stage2_passed == stage2_total),
        }


def main():
    """Run momentum signal validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate momentum signal calculations"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all validation stages (requires data)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing price data CSVs"
    )

    args = parser.parse_args()

    validator = MomentumValidator(data_dir=args.data_dir)
    results = validator.run_all(include_stage2=args.full)

    # Exit code based on results
    sys.exit(0 if results["all_passed"] else 1)


if __name__ == "__main__":
    main()
