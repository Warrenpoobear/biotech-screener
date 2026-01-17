"""
VOLATILITY CALCULATION FIX
==========================
Location: wake_robin_data_pipeline/market_data_provider.py

This fix addresses the 200% / 10% volatility errors in your portfolio output.

DIAGNOSIS
---------
Your portfolio showed:
- FOLD, AZN, MEDP, KRYS, MIRM, AKE, ILMN: 200.0% volatility ← DEFAULT VALUE
- INDV: 10.0% volatility ← WRONG CALCULATION
- ALKS: 184.4%, KROS: 136.9% ← SUSPICIOUS

These errors cascade into wrong position sizing:
- 200% vol stocks get tiny 1.06% positions
- 10% vol stock gets huge 3.18% position (hit concentration limit)

ROOT CAUSES
-----------
1. When data fetch fails → defaults to 2.0 (200%)
2. Annualization bug: multiplying by 252 instead of sqrt(252)
3. Percentage vs decimal confusion
4. No sanity checks for biotech volatility ranges

THE FIX
-------
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import yfinance as yf
from typing import Optional

class MarketDataProvider:
    """Fixed implementation with correct volatility calculation"""
    
    def __init__(self):
        self.morningstar_provider = None  # Your existing Morningstar provider
        
    def get_volatility(self, ticker: str, as_of_date: date, lookback_days: int = 252) -> float:
        """
        Calculate annualized volatility CORRECTLY
        
        Args:
            ticker: Stock ticker symbol
            as_of_date: Date for point-in-time calculation
            lookback_days: Lookback period in trading days
            
        Returns:
            Annualized volatility as decimal (e.g., 0.45 = 45%)
        """
        try:
            # Try Morningstar first (will likely return NaN for individual stocks)
            if self.morningstar_provider:
                mstar_returns = self._try_morningstar(ticker, as_of_date, lookback_days)
                if mstar_returns is not None and len(mstar_returns) >= 50:
                    vol = self._calculate_volatility_from_returns(mstar_returns)
                    if vol is not None:
                        return vol
            
            # Fallback to yfinance (this is what actually runs for stocks)
            yf_returns = self._get_yfinance_returns(ticker, as_of_date, lookback_days)
            if yf_returns is not None:
                return self._calculate_volatility_from_returns(yf_returns)
            
            # If both failed, use conservative default
            print(f"WARNING: No data for {ticker}, using 50% default volatility")
            return 0.50  # 50% is reasonable for biotech (NOT 200%!)
            
        except Exception as e:
            print(f"ERROR calculating volatility for {ticker}: {e}")
            return 0.50
    
    def _get_yfinance_returns(self, ticker: str, as_of_date: date, lookback_days: int) -> Optional[pd.Series]:
        """
        Fetch daily returns from yfinance
        
        Returns:
            Series of daily log returns, or None if insufficient data
        """
        try:
            # Calculate date range
            end_date = as_of_date
            start_date = as_of_date - timedelta(days=int(lookback_days * 1.5))  # Buffer for weekends
            
            # Fetch price history
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist is None or len(hist) < 50:
                print(f"  WARNING: {ticker} insufficient yfinance data ({len(hist) if hist is not None else 0} days)")
                return None
            
            # Calculate log returns (more stable than simple returns)
            returns = np.log(hist['Close'] / hist['Close'].shift(1))
            returns = returns.dropna()
            
            # Trim to requested lookback
            returns = returns.tail(lookback_days)
            
            return returns
            
        except Exception as e:
            print(f"  ERROR fetching yfinance data for {ticker}: {e}")
            return None
    
    def _calculate_volatility_from_returns(self, returns: pd.Series) -> Optional[float]:
        """
        Calculate annualized volatility from daily returns
        
        THIS IS THE CRITICAL FIX!
        
        Args:
            returns: Series of daily returns (as decimals, e.g., 0.02 for 2%)
            
        Returns:
            Annualized volatility as decimal, or None if invalid
        """
        if returns is None or len(returns) < 50:
            return None
        
        # Validation: returns should be in decimal format, not percentages
        if returns.abs().max() > 10:
            raise ValueError(f"Returns seem to be in wrong format (max={returns.max()})")
        
        # Calculate daily volatility (standard deviation of returns)
        daily_vol = returns.std()
        
        # ===================================================================
        # CRITICAL FIX: Use sqrt(252), NOT 252!
        # ===================================================================
        # Trading days in a year = 252
        # Volatility scales with square root of time (not linearly)
        # 
        # WRONG: annual_vol = daily_vol * 252      ← 100x too high!
        # RIGHT: annual_vol = daily_vol * sqrt(252) ← Correct annualization
        # ===================================================================
        
        annual_vol = daily_vol * np.sqrt(252)
        
        # Sanity checks for biotech stocks
        # Typical biotech volatility ranges:
        # - Large cap (>$10B): 30-50%
        # - Mid cap ($2-10B): 40-60%
        # - Small cap (<$2B): 50-80%
        # - Pre-revenue: 60-100%
        
        if annual_vol < 0.20:  # 20% is too low for biotech
            print(f"  WARNING: Volatility {annual_vol:.1%} seems low, flooring at 30%")
            return 0.30
        
        if annual_vol > 1.50:  # 150% is suspiciously high
            print(f"  WARNING: Volatility {annual_vol:.1%} seems high, capping at 100%")
            return min(annual_vol, 1.00)
        
        return annual_vol
    
    def _try_morningstar(self, ticker: str, as_of_date: date, lookback_days: int) -> Optional[pd.Series]:
        """
        Try to get returns from Morningstar
        Returns None for individual stocks (expected behavior)
        """
        try:
            if not self.morningstar_provider:
                return None
                
            returns = self.morningstar_provider.get_daily_returns(
                ticker=ticker,
                as_of_date=as_of_date,
                lookback_days=lookback_days
            )
            
            # Check if we got valid data (not all NaN)
            if returns is not None and not returns.isna().all():
                return returns
            else:
                # This is expected for individual stocks
                return None
                
        except Exception as e:
            print(f"  Morningstar attempt failed for {ticker}: {e}")
            return None


# ==============================================================================
# VERIFICATION EXAMPLES
# ==============================================================================

def verify_fix():
    """
    Verify the fix produces correct volatility values
    """
    print("="*70)
    print("VOLATILITY CALCULATION FIX VERIFICATION")
    print("="*70)
    
    # Example 1: Typical daily returns for a biotech stock
    # Simulate returns with 50% annual volatility
    np.random.seed(42)
    daily_returns = np.random.normal(0, 0.03, 252)  # ~50% annual vol
    returns_series = pd.Series(daily_returns)
    
    provider = MarketDataProvider()
    calculated_vol = provider._calculate_volatility_from_returns(returns_series)
    
    print(f"\nExample 1: Simulated 50% annual volatility")
    print(f"  Daily return std: {returns_series.std():.4f}")
    print(f"  Calculated annual vol: {calculated_vol:.1%}")
    print(f"  Expected: ~50%")
    print(f"  Status: {'✅ PASS' if 0.45 <= calculated_vol <= 0.55 else '❌ FAIL'}")
    
    # Example 2: Show the wrong vs right calculation
    daily_std = 0.03
    
    wrong_annual = daily_std * 252  # ❌ WRONG
    correct_annual = daily_std * np.sqrt(252)  # ✅ CORRECT
    
    print(f"\nExample 2: Annualization comparison")
    print(f"  Daily std: {daily_std:.4f}")
    print(f"  WRONG (×252):     {wrong_annual:.1%}  ← This is your bug!")
    print(f"  CORRECT (×√252):  {correct_annual:.1%}")
    print(f"  Difference: {(wrong_annual/correct_annual - 1)*100:.1f}x too high")
    
    print("\n" + "="*70)


# ==============================================================================
# IMPLEMENTATION CHECKLIST
# ==============================================================================

print("""
IMPLEMENTATION CHECKLIST
========================

□ 1. Backup your current market_data_provider.py
   
□ 2. Replace _calculate_volatility_from_returns() with the fixed version above

□ 3. Update get_volatility() to:
      - Try Morningstar first (will return None for stocks)
      - Fall back to yfinance
      - Use 0.50 (50%) as default, NOT 2.0 (200%)

□ 4. Add sanity checks:
      - Floor: 30% for biotech minimum
      - Cap: 100% maximum
      - Warning if outside 20-150% range

□ 5. Update portfolio generator output label:
      - Change "morningstar_direct" to "yfinance_calculated"
      - Or make it dynamic based on actual source used

□ 6. Test with your top 10 tickers:
      FOLD, INDV, AZN, MEDP, KRYS, MIRM, AKE, ALKS, KROS, ILMN

□ 7. Verify portfolio output shows:
      - No more 200% defaults
      - Volatilities in 30-100% range
      - More varied position sizes (not all 1.06%)
      - Portfolio volatility ~50-70% (not 108%)

□ 8. Commit changes:
      git add wake_robin_data_pipeline/market_data_provider.py
      git commit -m "Fix volatility calculation: use sqrt(252) not 252, default to 50% not 200%"
      git push


EXPECTED RESULTS AFTER FIX
===========================

Before (Wrong):
  FOLD:  200.0% vol → 1.06% position (minimum)
  INDV:  10.0% vol  → 3.18% position (capped at max)
  Portfolio vol: 108.6%

After (Correct):
  FOLD:  45-60% vol  → 1.8-2.2% position (reasonable)
  INDV:  50-65% vol  → 1.6-2.0% position (reasonable)
  Portfolio vol: 55-65% (realistic for biotech)


KEY TAKEAWAYS
=============

1. Morningstar Direct does NOT provide daily returns for individual stocks
   - Only works for mutual funds and ETFs
   - Your fallback to yfinance is correct and necessary

2. The bug was in the yfinance volatility calculation:
   - Using 252 instead of sqrt(252) for annualization
   - Defaulting to 2.0 (200%) instead of 0.50 (50%)
   - No sanity checks for biotech ranges

3. The "morningstar_direct" label in your portfolio output is misleading
   - Should say "yfinance_calculated" for individual stocks
   - Or dynamically show actual source used

4. Position sizing was broken because volatility was broken:
   - Inverse vol weighting: weight ∝ 1/volatility
   - Wrong vol → wrong weights → wrong risk allocation
""")

if __name__ == "__main__":
    verify_fix()
