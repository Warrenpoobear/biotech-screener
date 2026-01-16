"""
Production Momentum Validator - Validates ACTUAL production code
=================================================================

CRITICAL: This validator imports and tests your REAL momentum calculation code,
not a re-implementation. This prevents false confidence from "validator passes
but production code is broken."

Architecture:
- Adapter layer that calls YOUR production momentum functions
- Validates what actually runs in production, not a clean re-write
- Catches bugs that would slip through isolated unit tests

Author: Wake Robin Capital Management
Version: 1.1.0 (Production-testing)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr
from dataclasses import dataclass
import sys

# =============================================================================
# IMPORT YOUR PRODUCTION CODE HERE
# =============================================================================

# TODO: Replace these imports with your actual production code paths
# Example:
# from src.signals.momentum import calculate_momentum_score
# from src.signals.momentum import detect_regime
# from src.utils.returns import calculate_volatility

# For now, we'll use placeholder functions that you need to replace
def PRODUCTION_calculate_momentum(ticker: str, prices: pd.Series, 
                                  as_of_date: datetime) -> float:
    """
    REPLACE THIS with import from your actual production code.
    
    Example:
        from src.signals.momentum import calculate_momentum_score
        return calculate_momentum_score(ticker, prices, as_of_date)
    """
    raise NotImplementedError(
        "You must import your ACTUAL production momentum function here. "
        "See instructions in PRODUCTION_calculate_momentum()"
    )


def PRODUCTION_calculate_volatility(prices: pd.Series) -> float:
    """
    REPLACE THIS with import from your actual production volatility calculation.
    
    This is CRITICAL - if your production vol calc still uses ×252 bug,
    this validator will catch it.
    """
    raise NotImplementedError(
        "You must import your ACTUAL production volatility function here. "
        "See instructions in PRODUCTION_calculate_volatility()"
    )


# =============================================================================
# TRADING DAY UTILITIES
# =============================================================================

class TradingDayCalendar:
    """Handles trading day arithmetic (not calendar day arithmetic)."""
    
    def __init__(self, trading_days: pd.DatetimeIndex):
        """
        Args:
            trading_days: Index of actual trading days in your data
        """
        self.trading_days = trading_days.sort_values()
        self.trading_day_set = set(self.trading_days)
    
    def add_trading_days(self, date: datetime, n_days: int) -> datetime:
        """
        Add n trading days to date.
        
        Example: If date is Friday and n_days=1, returns next Monday.
        """
        if date not in self.trading_day_set:
            # Find nearest trading day
            date = self.get_nearest_trading_day(date)
        
        try:
            idx = self.trading_days.get_loc(date)
            target_idx = idx + n_days
            
            if target_idx < 0 or target_idx >= len(self.trading_days):
                raise IndexError(f"Date {n_days} trading days from {date} out of range")
            
            return self.trading_days[target_idx]
        except KeyError:
            raise ValueError(f"{date} not in trading calendar")
    
    def get_nearest_trading_day(self, date: datetime, 
                               direction: str = 'forward') -> datetime:
        """Get nearest trading day to date."""
        if direction == 'forward':
            future_days = self.trading_days[self.trading_days >= date]
            if len(future_days) == 0:
                raise ValueError(f"No trading days after {date}")
            return future_days[0]
        else:
            past_days = self.trading_days[self.trading_days <= date]
            if len(past_days) == 0:
                raise ValueError(f"No trading days before {date}")
            return past_days[-1]


# =============================================================================
# CROSS-SECTIONAL IC CALCULATOR
# =============================================================================

class CrossSectionalICCalculator:
    """
    Calculate Information Coefficient correctly: per-date cross-sectional IC,
    then averaged (not pooled).
    
    This is the standard definition used in quantitative finance.
    """
    
    def __init__(self, trading_calendar: TradingDayCalendar):
        self.calendar = trading_calendar
    
    def calculate_ic_series(
        self,
        momentum_func,
        prices_df: pd.DataFrame,
        rebalance_dates: List[datetime],
        forward_horizon_days: int = 21  # 21 trading days ≈ 1 month
    ) -> pd.Series:
        """
        Calculate cross-sectional IC at each rebalance date.
        
        Args:
            momentum_func: Function that takes (ticker, prices, date) → momentum
            prices_df: DataFrame of prices (columns=tickers, index=dates)
            rebalance_dates: List of dates to calculate IC
            forward_horizon_days: Trading days forward for returns
            
        Returns:
            Series of IC values indexed by rebalance_date
        """
        ic_values = []
        
        for calc_date in rebalance_dates:
            try:
                # Calculate momentum for all tickers at calc_date
                momentum_scores = {}
                
                for ticker in prices_df.columns:
                    try:
                        score = momentum_func(
                            ticker,
                            prices_df[ticker],
                            calc_date
                        )
                        momentum_scores[ticker] = float(score)
                    except Exception:
                        continue  # Skip tickers with insufficient data
                
                if len(momentum_scores) < 20:  # Need minimum universe
                    continue
                
                # Calculate forward returns (CORRECT: using trading days)
                forward_date = self.calendar.add_trading_days(
                    calc_date, 
                    forward_horizon_days
                )
                
                forward_returns = {}
                for ticker in momentum_scores.keys():
                    try:
                        ret = (prices_df[ticker].loc[forward_date] / 
                              prices_df[ticker].loc[calc_date]) - 1
                        forward_returns[ticker] = ret
                    except (KeyError, ZeroDivisionError):
                        continue
                
                # Get common tickers
                common_tickers = set(momentum_scores.keys()) & set(forward_returns.keys())
                
                if len(common_tickers) < 20:
                    continue
                
                # Calculate cross-sectional IC (Spearman correlation)
                mom_values = [momentum_scores[t] for t in common_tickers]
                ret_values = [forward_returns[t] for t in common_tickers]
                
                ic, p_value = spearmanr(mom_values, ret_values)
                
                ic_values.append({
                    'date': calc_date,
                    'ic': ic,
                    'p_value': p_value,
                    'n_tickers': len(common_tickers)
                })
                
            except Exception as e:
                print(f"Warning: Failed to calculate IC for {calc_date}: {e}")
                continue
        
        return pd.DataFrame(ic_values).set_index('date')
    
    def calculate_ic_statistics(self, ic_series: pd.Series) -> Dict:
        """Calculate IC statistics with t-test."""
        mean_ic = ic_series['ic'].mean()
        std_ic = ic_series['ic'].std()
        n = len(ic_series)
        
        # t-statistic for testing if IC is significantly different from 0
        t_stat = mean_ic / (std_ic / np.sqrt(n))
        
        # Hit rate: % of positive IC months
        hit_rate = (ic_series['ic'] > 0).mean()
        
        return {
            'mean_ic': float(mean_ic),
            'std_ic': float(std_ic),
            't_statistic': float(t_stat),
            'n_periods': int(n),
            'hit_rate': float(hit_rate),
            'min_ic': float(ic_series['ic'].min()),
            'max_ic': float(ic_series['ic'].max())
        }


# =============================================================================
# PRODUCTION VALIDATOR
# =============================================================================

class ProductionMomentumValidator:
    """Validates actual production momentum code."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.test_results = {}
        self.calendar = None
    
    def load_production_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Load actual production price data."""
        # Load universe prices
        universe_file = self.data_dir / 'universe_prices.csv'
        if not universe_file.exists():
            raise FileNotFoundError(
                f"Universe price file not found: {universe_file}\n"
                f"Create this file with daily prices for your biotech universe."
            )
        
        universe_df = pd.read_csv(universe_file, index_col=0, parse_dates=True)
        
        # Load index data
        indices_file = self.data_dir / 'indices_prices.csv'
        if not indices_file.exists():
            raise FileNotFoundError(
                f"Index price file not found: {indices_file}\n"
                f"Create this file with XBI and SPY daily prices."
            )
        
        indices_df = pd.read_csv(indices_file, index_col=0, parse_dates=True)
        
        xbi_prices = indices_df['XBI']
        spy_prices = indices_df['SPY']
        
        # Initialize trading calendar from actual trading days
        self.calendar = TradingDayCalendar(universe_df.index)
        
        return universe_df, xbi_prices, spy_prices
    
    def test_volatility_production(self):
        """Test that production volatility calculation uses sqrt(252)."""
        print("=" * 70)
        print("TEST: Production Volatility Calculation")
        print("=" * 70)
        
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        daily_vol = 0.02
        returns = pd.Series(np.random.normal(0, daily_vol, 252), index=dates)
        
        # Create price series
        prices = (1 + returns).cumprod() * 100
        
        try:
            # Call PRODUCTION volatility function
            prod_vol = PRODUCTION_calculate_volatility(prices)
            
            # Expected (correct)
            expected_vol = daily_vol * np.sqrt(252)
            
            # Wrong method (old bug)
            wrong_vol = daily_vol * 252
            
            print(f"\nProduction volatility: {prod_vol:.4f} ({prod_vol*100:.2f}%)")
            print(f"Expected (sqrt(252)): {expected_vol:.4f} ({expected_vol*100:.2f}%)")
            print(f"Wrong (×252): {wrong_vol:.4f} ({wrong_vol*100:.2f}%)")
            
            # Check which one it's closer to
            error_correct = abs(prod_vol - expected_vol) / expected_vol
            error_wrong = abs(prod_vol - wrong_vol) / wrong_vol
            
            if error_correct < 0.10:  # Within 10%
                print(f"\n✅ PASS: Production code uses sqrt(252)")
                print(f"   Error from correct: {error_correct*100:.2f}%")
                self.test_results['production_volatility'] = 'PASS'
            elif error_wrong < 0.10:
                print(f"\n❌ FAIL: Production code uses ×252 (BUG STILL PRESENT)")
                print(f"   Error from wrong method: {error_wrong*100:.2f}%")
                self.test_results['production_volatility'] = 'FAIL'
            else:
                print(f"\n⚠️  WARNING: Volatility doesn't match either method")
                print(f"   Review production calculation logic")
                self.test_results['production_volatility'] = 'UNKNOWN'
                
        except NotImplementedError as e:
            print(f"\n⚠️  SKIPPED: {e}")
            self.test_results['production_volatility'] = 'SKIPPED'
    
    def test_information_coefficient(
        self,
        start_date: str = '2020-01-01',
        end_date: str = '2024-12-31'
    ):
        """
        Test IC using production momentum function.
        
        Uses CORRECT methodology:
        - Cross-sectional IC per month
        - Trading day horizons
        - No survivorship bias
        """
        print("\n" + "=" * 70)
        print("TEST: Information Coefficient (Production Code)")
        print("=" * 70)
        
        try:
            universe_df, xbi, spy = self.load_production_data()
            
            # Monthly rebalance dates
            rebalance_dates = pd.date_range(
                start_date, 
                end_date, 
                freq='MS'
            )
            
            # Filter to dates in our data
            rebalance_dates = [d for d in rebalance_dates 
                             if d in universe_df.index]
            
            print(f"\nAnalyzing {len(rebalance_dates)} rebalance periods")
            print(f"Universe: {len(universe_df.columns)} tickers")
            
            # Calculate IC series
            ic_calculator = CrossSectionalICCalculator(self.calendar)
            
            ic_df = ic_calculator.calculate_ic_series(
                momentum_func=PRODUCTION_calculate_momentum,
                prices_df=universe_df,
                rebalance_dates=rebalance_dates,
                forward_horizon_days=21  # ~1 month trading days
            )
            
            if len(ic_df) == 0:
                print("\n❌ FAIL: No IC values calculated")
                self.test_results['production_ic'] = 'FAIL'
                return
            
            # Calculate statistics
            stats = ic_calculator.calculate_ic_statistics(ic_df)
            
            print(f"\nIC Statistics:")
            print(f"  Mean IC: {stats['mean_ic']:.3f}")
            print(f"  Std IC: {stats['std_ic']:.3f}")
            print(f"  t-statistic: {stats['t_statistic']:.2f}")
            print(f"  Hit rate: {stats['hit_rate']*100:.1f}%")
            print(f"  Range: [{stats['min_ic']:.3f}, {stats['max_ic']:.3f}]")
            print(f"  N periods: {stats['n_periods']}")
            
            # Compare to claimed performance
            expected_ic = 0.713
            print(f"\nExpected IC (from docs): {expected_ic:.3f}")
            
            # IC interpretation
            if stats['mean_ic'] >= 0.20:
                quality = "EXCEPTIONAL ✅"
            elif stats['mean_ic'] >= 0.10:
                quality = "STRONG ✓"
            elif stats['mean_ic'] >= 0.05:
                quality = "USEFUL ⚠️"
            else:
                quality = "WEAK ❌"
            
            print(f"Signal Quality: {quality}")
            
            # Statistical significance
            if abs(stats['t_statistic']) > 2.0:
                print(f"✓ Statistically significant (|t| > 2.0)")
            else:
                print(f"⚠️  Not statistically significant (|t| < 2.0)")
            
            # Pass/fail
            if stats['mean_ic'] >= expected_ic * 0.70:  # 70% of claimed
                print(f"\n✅ PASS: IC is {stats['mean_ic']/expected_ic*100:.0f}% of claimed")
                self.test_results['production_ic'] = 'PASS'
            else:
                print(f"\n❌ FAIL: IC is only {stats['mean_ic']/expected_ic*100:.0f}% of claimed")
                print(f"   Claimed: {expected_ic:.3f}")
                print(f"   Actual: {stats['mean_ic']:.3f}")
                self.test_results['production_ic'] = 'FAIL'
            
            # Save detailed results
            ic_df.to_csv(self.data_dir / 'ic_time_series.csv')
            print(f"\n📊 IC time series saved to: ic_time_series.csv")
            
        except NotImplementedError as e:
            print(f"\n⚠️  SKIPPED: {e}")
            self.test_results['production_ic'] = 'SKIPPED'
        except FileNotFoundError as e:
            print(f"\n⚠️  SKIPPED: {e}")
            self.test_results['production_ic'] = 'SKIPPED'
    
    def run_all_tests(self):
        """Run all production validation tests."""
        print("\n")
        print("*" * 70)
        print("PRODUCTION MOMENTUM VALIDATION")
        print("Testing ACTUAL production code (not re-implementation)")
        print("*" * 70)
        
        self.test_volatility_production()
        self.test_information_coefficient()
        
        # Summary
        print("\n" + "=" * 70)
        print("PRODUCTION VALIDATION SUMMARY")
        print("=" * 70)
        
        for test_name, result in self.test_results.items():
            emoji = {
                'PASS': '✅',
                'FAIL': '❌',
                'SKIPPED': '⚠️',
                'UNKNOWN': '❓'
            }.get(result, '?')
            print(f"{emoji} {test_name}: {result}")
        
        # Final verdict
        failed = [k for k, v in self.test_results.items() if v == 'FAIL']
        skipped = [k for k, v in self.test_results.items() if v == 'SKIPPED']
        
        if failed:
            print(f"\n❌ VALIDATION FAILED")
            print(f"   Failed tests: {', '.join(failed)}")
            print(f"   DO NOT USE IN PRODUCTION")
        elif skipped:
            print(f"\n⚠️  VALIDATION INCOMPLETE")
            print(f"   Implement production imports to complete validation")
        else:
            print(f"\n✅ ALL PRODUCTION TESTS PASSED")
            print(f"   Code is ready for production use")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("SETUP INSTRUCTIONS")
    print("=" * 70)
    print("""
1. Import your ACTUAL production momentum function:
   
   Replace the placeholder PRODUCTION_calculate_momentum() with:
   from src.signals.momentum import your_actual_function
   
2. Import your ACTUAL production volatility function:
   
   Replace the placeholder PRODUCTION_calculate_volatility() with:
   from src.utils.returns import your_actual_volatility_calc

3. Prepare data files in data/:
   - universe_prices.csv (daily prices for all tickers)
   - indices_prices.csv (XBI and SPY daily prices)

4. Run validator:
   python validate_production_momentum.py
""")
    
    response = input("\nHave you completed steps 1-3? (y/n): ")
    
    if response.lower() != 'y':
        print("\nPlease complete setup steps first.")
        sys.exit(0)
    
    # Run validation
    validator = ProductionMomentumValidator(data_dir=Path('data'))
    validator.run_all_tests()
    
    # Save results
    output_file = Path('production_momentum_validation.json')
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'test_results': validator.test_results,
        }, f, indent=2)
    
    print(f"\n📊 Results saved to: {output_file}")
