"""
Momentum Signal Validation Script
==================================

Validates the momentum calculation system for Wake Robin biotech screening.

Critical validation checks:
1. Mathematical correctness of return calculations
2. Proper volatility annualization (sqrt(252) not 252)
3. Regime detection logic
4. Look-ahead bias detection
5. Information Coefficient calculation
6. Quartile spread analysis
7. Signal persistence and decay

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


# =============================================================================
# CONSTANTS
# =============================================================================

TRADING_DAYS_PER_YEAR = 252
DECIMAL_PLACES = Decimal('0.0001')
VALIDATION_SEED = 42  # Fixed seed for deterministic tests

# Regime thresholds (from memory)
REGIME_CONFIG = {
    'bull_market': {
        'xbi_spy_outperformance_threshold': 0.05,  # 5% outperformance over 60d
        'breadth_threshold': 0.40,  # >40% advancing
        'momentum_weight': 0.25
    },
    'bear_market': {
        'xbi_spy_underperformance_threshold': -0.05,  # 5% underperformance
        'breadth_threshold': 0.40,  # <40% advancing (60% declining)
        'momentum_weight': 0.05
    },
    'neutral': {
        'momentum_weight': 0.15
    }
}

# Expected performance metrics (from memory)
EXPECTED_METRICS = {
    'bull_quartile_spread': 0.876,  # 87.6% spread
    'information_coefficient': 0.713,
    'min_sharpe': 0.50
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MomentumSignal:
    """Momentum signal for a single ticker at a point in time."""
    ticker: str
    date: datetime
    raw_momentum: Decimal
    risk_adjusted_momentum: Decimal
    percentile_rank: float
    regime: str
    forward_30d_return: Optional[float] = None


@dataclass
class RegimeMetrics:
    """Market regime metrics at a point in time."""
    date: datetime
    xbi_return_60d: float
    spy_return_60d: float
    relative_performance: float
    xbi_breadth: float  # % advancing stocks
    regime: str  # 'bull', 'bear', 'neutral'


# =============================================================================
# MOMENTUM CALCULATION
# =============================================================================

class MomentumCalculator:
    """
    Calculate momentum signals with proper volatility normalization.
    
    CRITICAL: Uses sqrt(252) for annualization, not 252.
    """
    
    def __init__(self, lookback_days: int = 63):
        """
        Args:
            lookback_days: Number of trading days for momentum calculation
                          (default 63 = ~3 months)
        """
        self.lookback_days = lookback_days
        self.validation_errors = []
    
    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate log returns from price series.
        
        Args:
            prices: Series of prices indexed by date
            
        Returns:
            Series of log returns
        """
        # Validate input
        if prices.isnull().any():
            raise ValueError("Price series contains NaN values")
        
        if (prices <= 0).any():
            raise ValueError("Price series contains non-positive values")
        
        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1))
        
        return log_returns.dropna()
    
    def annualize_volatility(self, daily_std: float) -> float:
        """
        Properly annualize daily volatility.
        
        CRITICAL: This was the bug - must use sqrt(252), not 252.
        
        Args:
            daily_std: Daily standard deviation of returns
            
        Returns:
            Annualized volatility (decimal, not percentage)
        """
        annual_vol = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
        return annual_vol
    
    def calculate_momentum_signal(
        self,
        ticker: str,
        prices: pd.Series,
        as_of_date: datetime
    ) -> Tuple[Decimal, Dict]:
        """
        Calculate risk-adjusted momentum signal for a ticker.
        
        Formula:
            Momentum = (Cumulative Return) / (Annualized Volatility)
        
        This is the Sharpe-style risk-adjusted momentum.
        
        Args:
            ticker: Stock ticker
            prices: Price series indexed by date
            as_of_date: Date to calculate momentum as of
            
        Returns:
            (momentum_signal, diagnostics_dict)
        """
        # Get relevant price window
        end_date = as_of_date
        start_date = as_of_date - timedelta(days=self.lookback_days * 2)  # Buffer for holidays
        
        window = prices.loc[start_date:end_date].tail(self.lookback_days + 1)
        
        if len(window) < self.lookback_days:
            raise ValueError(
                f"{ticker}: Insufficient data. Need {self.lookback_days} days, "
                f"got {len(window)}"
            )
        
        # Calculate cumulative return
        total_return = (window.iloc[-1] / window.iloc[0]) - 1.0
        
        # Calculate daily returns
        log_returns = self.calculate_log_returns(window)
        
        if len(log_returns) < self.lookback_days - 5:  # Allow some missing days
            raise ValueError(f"{ticker}: Too many missing returns")
        
        # Calculate daily volatility
        daily_std = log_returns.std()
        
        if daily_std == 0:
            raise ValueError(f"{ticker}: Zero volatility (no price movement)")
        
        # Annualize volatility (CORRECT METHOD)
        annual_vol = self.annualize_volatility(daily_std)
        
        # Calculate risk-adjusted momentum
        # This gives us return per unit of risk
        momentum = total_return / annual_vol
        
        # Convert to Decimal for determinism
        momentum_decimal = Decimal(str(momentum)).quantize(
            DECIMAL_PLACES, 
            rounding=ROUND_HALF_UP
        )
        
        # Diagnostics
        diagnostics = {
            'ticker': ticker,
            'date': as_of_date.isoformat(),
            'lookback_days': len(window) - 1,
            'total_return': float(total_return),
            'daily_vol': float(daily_std),
            'annual_vol': float(annual_vol),
            'momentum_score': float(momentum_decimal),
            'data_points': len(log_returns)
        }
        
        return momentum_decimal, diagnostics


# =============================================================================
# REGIME DETECTION
# =============================================================================

class RegimeDetector:
    """Detect market regime for adaptive momentum weighting."""
    
    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
    
    def detect_regime(
        self,
        xbi_prices: pd.Series,
        spy_prices: pd.Series,
        universe_prices: pd.DataFrame,
        as_of_date: datetime
    ) -> RegimeMetrics:
        """
        Detect current market regime.
        
        Args:
            xbi_prices: XBI index prices
            spy_prices: SPY index prices
            universe_prices: DataFrame of prices for universe (columns = tickers)
            as_of_date: Date to detect regime as of
            
        Returns:
            RegimeMetrics object
        """
        # Get lookback window
        end_date = as_of_date
        start_date = as_of_date - timedelta(days=self.lookback_days * 2)
        
        # Calculate 60-day returns
        xbi_window = xbi_prices.loc[start_date:end_date].tail(self.lookback_days + 1)
        spy_window = spy_prices.loc[start_date:end_date].tail(self.lookback_days + 1)
        
        xbi_return_60d = (xbi_window.iloc[-1] / xbi_window.iloc[0]) - 1.0
        spy_return_60d = (spy_window.iloc[-1] / spy_window.iloc[0]) - 1.0
        
        # Relative performance
        relative_perf = xbi_return_60d - spy_return_60d
        
        # Calculate breadth (% of stocks advancing over period)
        universe_window = universe_prices.loc[start_date:end_date].tail(self.lookback_days + 1)
        
        if len(universe_window) < 2:
            breadth = 0.50  # Default to neutral
        else:
            returns = (universe_window.iloc[-1] / universe_window.iloc[0]) - 1.0
            breadth = (returns > 0).sum() / len(returns)
        
        # Determine regime
        if (relative_perf > REGIME_CONFIG['bull_market']['xbi_spy_outperformance_threshold'] and
            breadth > REGIME_CONFIG['bull_market']['breadth_threshold']):
            regime = 'bull'
        elif (relative_perf < REGIME_CONFIG['bear_market']['xbi_spy_underperformance_threshold'] or
              breadth < REGIME_CONFIG['bear_market']['breadth_threshold']):
            regime = 'bear'
        else:
            regime = 'neutral'
        
        return RegimeMetrics(
            date=as_of_date,
            xbi_return_60d=float(xbi_return_60d),
            spy_return_60d=float(spy_return_60d),
            relative_performance=float(relative_perf),
            xbi_breadth=float(breadth),
            regime=regime
        )


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class MomentumValidator:
    """Comprehensive validation of momentum signal calculations."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.calculator = MomentumCalculator()
        self.regime_detector = RegimeDetector()
        self.test_results = {}
    
    def load_test_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Load price data for testing."""
        # This would load your actual price data
        # For now, return placeholders
        raise NotImplementedError("Implement data loading for your system")
    
    def test_volatility_annualization(self):
        """
        Test that volatility is properly annualized.
        
        CRITICAL TEST: Ensures we use sqrt(252), not 252.
        """
        print("=" * 70)
        print("TEST 1: Volatility Annualization")
        print("=" * 70)
        
        # Create synthetic data with known volatility
        np.random.seed(VALIDATION_SEED)
        daily_vol = 0.02  # 2% daily
        expected_annual = daily_vol * np.sqrt(252)  # Should be ~31.7%
        
        # Direct check (no Monte Carlo noise): must be exactly sqrt(252) scaling
        direct_annual = self.calculator.annualize_volatility(daily_vol)
        if abs(direct_annual - expected_annual) > 1e-12:
            print("\n❌ FAIL: annualize_volatility() is not using sqrt(252)")
            print(f"   Expected: {expected_annual:.12f}")
            print(f"   Got: {direct_annual:.12f}")
            self.test_results['volatility_annualization'] = 'FAIL'
            return
        
        # Generate returns
        returns = np.random.normal(0, daily_vol, 252)
        actual_daily = np.std(returns)
        actual_annual = self.calculator.annualize_volatility(actual_daily)
        
        print(f"\nDaily volatility (input): {daily_vol:.4f}")
        print(f"Expected annual: {expected_annual:.4f} ({expected_annual*100:.2f}%)")
        print(f"Actual annual: {actual_annual:.4f} ({actual_annual*100:.2f}%)")
        
        # Check for the OLD BUG (multiplying by 252)
        wrong_method = actual_daily * 252
        print(f"\n❌ WRONG METHOD (×252): {wrong_method:.4f} ({wrong_method*100:.2f}%)")
        print(f"   This would be {wrong_method/actual_annual:.1f}x too high!")
        
        # Tolerance check
        tolerance = 0.05  # 5% tolerance for Monte Carlo
        error = abs(actual_annual - expected_annual) / expected_annual
        
        if error < tolerance:
            print(f"\n✅ PASS: Error {error*100:.2f}% < {tolerance*100:.0f}% threshold")
            self.test_results['volatility_annualization'] = 'PASS'
        else:
            print(f"\n❌ FAIL: Error {error*100:.2f}% >= {tolerance*100:.0f}% threshold")
            self.test_results['volatility_annualization'] = 'FAIL'
    
    def test_look_ahead_bias(self):
        """Test that momentum calculation doesn't use future data."""
        print("\n" + "=" * 70)
        print("TEST 2: Look-Ahead Bias Detection")
        print("=" * 70)
        
        # This test would verify that:
        # 1. Momentum at date T only uses data up to T
        # 2. Regime detection at T only uses data up to T
        # 3. No future returns leak into signal calculation
        
        print("\n⚠️  Implement with actual price data")
        print("    Check: signal(T) uses only data[start:T]")
        print("    Check: no forward returns in momentum calc")
        
        self.test_results['look_ahead_bias'] = 'PENDING'
    
    def test_quartile_spread(self):
        """
        Test that momentum achieves claimed 87.6% quartile spread.
        
        This validates the core alpha claim.
        """
        print("\n" + "=" * 70)
        print("TEST 3: Quartile Spread Analysis")
        print("=" * 70)
        
        expected_spread = EXPECTED_METRICS['bull_quartile_spread']
        
        print(f"\nExpected (from memory): {expected_spread*100:.1f}%")
        print("\n⚠️  Requires historical backtest data")
        print("    Steps:")
        print("    1. Calculate momentum for all tickers at T")
        print("    2. Sort into quartiles")
        print("    3. Calculate forward 30-day returns")
        print("    4. Compare Q1 vs Q4 performance")
        print("\n    Quartile Spread = avg_return(Q1) - avg_return(Q4)")
        
        self.test_results['quartile_spread'] = 'PENDING'
    
    def test_information_coefficient(self):
        """
        Test that IC matches claimed 0.713.
        
        IC = Spearman correlation between momentum ranks and forward returns.
        """
        print("\n" + "=" * 70)
        print("TEST 4: Information Coefficient")
        print("=" * 70)
        
        expected_ic = EXPECTED_METRICS['information_coefficient']
        
        print(f"\nExpected IC: {expected_ic:.3f}")
        print("\n⚠️  Requires historical data")
        print("    Formula: IC = corr(momentum_ranks, forward_returns)")
        print("\n    Good IC values:")
        print("    • >0.10 = useful signal")
        print("    • >0.20 = strong signal")
        print("    • >0.30 = exceptional (your claim: 0.713)")
        
        self.test_results['information_coefficient'] = 'PENDING'
    
    def test_regime_detection(self):
        """Test regime detection logic."""
        print("\n" + "=" * 70)
        print("TEST 5: Regime Detection Logic")
        print("=" * 70)
        
        print("\nRegime thresholds:")
        for regime, config in REGIME_CONFIG.items():
            print(f"\n{regime.upper()}:")
            for key, value in config.items():
                print(f"  • {key}: {value}")
        
        print("\n⚠️  Test with known regime periods:")
        print("    • 2020-2021: biotech bull market")
        print("    • 2022: biotech bear market")
        print("    • 2023-2024: mixed/neutral")
        
        self.test_results['regime_detection'] = 'PENDING'
    
    def test_signal_persistence(self):
        """Test how long momentum signal persists."""
        print("\n" + "=" * 70)
        print("TEST 6: Signal Decay Analysis")
        print("=" * 70)
        
        print("\nTest momentum predictive power over horizons:")
        print("  • 5 days")
        print("  • 10 days")
        print("  • 20 days (1 month)")
        print("  • 60 days (3 months)")
        print("\nExpected: Signal should decay over time")
        print("Optimal holding period: Where IC peaks")
        
        self.test_results['signal_persistence'] = 'PENDING'
    
    def test_mathematical_properties(self):
        """Test basic mathematical properties."""
        print("\n" + "=" * 70)
        print("TEST 7: Mathematical Properties")
        print("=" * 70)
        
        # Test with simple synthetic data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # Case 1: Constant price → zero momentum
        const_price = pd.Series(100.0, index=dates)
        try:
            signal, _ = self.calculator.calculate_momentum_signal(
                'TEST1', const_price, dates[-1]
            )
            print("❌ FAIL: Should reject zero volatility")
            self.test_results['math_properties'] = 'FAIL'
        except ValueError as e:
            print(f"✅ PASS: Correctly rejects zero volatility")
            print(f"   Error: {e}")
        
        np.random.seed(VALIDATION_SEED)
        # Case 2: Trending up → positive momentum
        uptrend = pd.Series(np.linspace(100, 120, 100), index=dates)
        uptrend = uptrend * np.random.lognormal(mean=0.0, sigma=0.01, size=100)  # Positive noise
        signal_up, diag_up = self.calculator.calculate_momentum_signal(
            'TEST2', uptrend, dates[-1]
        )
        
        print(f"\n✓ Uptrend momentum: {signal_up}")
        print(f"  Return: {diag_up['total_return']*100:.2f}%")
        print(f"  Vol: {diag_up['annual_vol']*100:.2f}%")
        
        # Case 3: Trending down → negative momentum
        downtrend = pd.Series(np.linspace(100, 80, 100), index=dates)
        downtrend = downtrend * np.random.lognormal(mean=0.0, sigma=0.01, size=100)
        signal_down, diag_down = self.calculator.calculate_momentum_signal(
            'TEST3', downtrend, dates[-1]
        )
        
        print(f"\n✓ Downtrend momentum: {signal_down}")
        print(f"  Return: {diag_down['total_return']*100:.2f}%")
        print(f"  Vol: {diag_down['annual_vol']*100:.2f}%")
        
        # Verify ordering
        if signal_up > 0 and signal_down < 0:
            print("\n✅ PASS: Positive trend → positive signal")
            print("         Negative trend → negative signal")
            self.test_results['math_properties'] = 'PASS'
        else:
            print("\n❌ FAIL: Signal direction incorrect")
            self.test_results['math_properties'] = 'FAIL'
    
    def run_all_tests(self):
        """Run complete validation suite."""
        print("\n")
        print("*" * 70)
        print("MOMENTUM SIGNAL VALIDATION SUITE")
        print("Wake Robin Capital Management")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("*" * 70)
        
        # Run tests
        self.test_volatility_annualization()
        self.test_mathematical_properties()
        self.test_look_ahead_bias()
        self.test_regime_detection()
        self.test_quartile_spread()
        self.test_information_coefficient()
        self.test_signal_persistence()
        
        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        for test_name, result in self.test_results.items():
            emoji = "✅" if result == "PASS" else "⚠️" if result == "PENDING" else "❌"
            print(f"{emoji} {test_name}: {result}")
        
        pass_count = sum(1 for r in self.test_results.values() if r == "PASS")
        pending_count = sum(1 for r in self.test_results.values() if r == "PENDING")
        fail_count = sum(1 for r in self.test_results.values() if r == "FAIL")
        
        print(f"\nResults: {pass_count} PASS | {pending_count} PENDING | {fail_count} FAIL")
        
        if fail_count > 0:
            print("\n❌ VALIDATION FAILED - DO NOT USE IN PRODUCTION")
        elif pending_count > 0:
            print("\n⚠️  VALIDATION INCOMPLETE - Implement pending tests")
        else:
            print("\n✅ ALL TESTS PASSED - Ready for production")
        
        return self.test_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Initialize validator
    validator = MomentumValidator(data_dir=Path('data'))
    
    # Run validation suite
    results = validator.run_all_tests()
    
    # Save results
    output_file = Path('momentum_validation_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'test_results': results,
            'expected_metrics': EXPECTED_METRICS,
            'regime_config': {k: dict(v) for k, v in REGIME_CONFIG.items()}
        }, f, indent=2)
    
    print(f"\n📊 Results saved to: {output_file}")
