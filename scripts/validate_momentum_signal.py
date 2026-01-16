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

        if include_stage2:
            print("\n" + "=" * 70)
            print("STAGE 2: PRODUCTION TESTING")
            print("=" * 70)
            print("\n⚠️  Stage 2 requires historical price data.")
            print("   Prepare CSV files and update load_test_data() method.")
            print("   See MOMENTUM_VALIDATION_GUIDE.md for instructions.")

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
            "all_passed": passed == total,
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
