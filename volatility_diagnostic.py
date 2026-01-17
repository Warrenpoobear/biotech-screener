"""
Volatility Diagnostic & Fix Script
===================================
Run this to:
1. Calculate correct volatility for your top 10 holdings
2. Compare against your current (wrong) values
3. Identify the bug in your calculation

Usage:
    python volatility_diagnostic.py

Author: Claude
Date: 2026-01-15
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Your top 10 tickers with WRONG volatility from portfolio output
CURRENT_WRONG_DATA = {
    'FOLD': 2.00,   # 200% - definitely wrong
    'INDV': 0.10,   # 10% - definitely wrong  
    'AZN': 2.00,    # 200% - definitely wrong
    'MEDP': 2.00,   # 200% - definitely wrong
    'KRYS': 2.00,   # 200% - definitely wrong
    'MIRM': 2.00,   # 200% - definitely wrong
    'AKE': 2.00,    # 200% - definitely wrong
    'ALKS': 1.844,  # 184% - suspicious
    'KROS': 1.369,  # 137% - suspicious
    'ILMN': 2.00,   # 200% - definitely wrong (this is a $6B company!)
}

def calculate_correct_volatility(ticker, period='1y', method='log_returns'):
    """
    Calculate annualized volatility CORRECTLY
    
    Args:
        ticker: Stock ticker symbol
        period: Historical period ('1y', '2y', '3y')
        method: 'log_returns' (recommended) or 'simple_returns'
    
    Returns:
        Annual volatility as decimal (e.g., 0.45 = 45%)
    """
    try:
        # Download price history
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if len(hist) < 50:  # Need at least 50 days of data
            print(f"  WARNING: {ticker} only has {len(hist)} days of data")
            return None
        
        # Calculate returns
        if method == 'log_returns':
            # Log returns (preferred for vol calculation)
            returns = np.log(hist['Close'] / hist['Close'].shift(1))
        else:
            # Simple returns
            returns = hist['Close'].pct_change()
        
        # Drop NaN values
        returns = returns.dropna()
        
        # Calculate daily volatility
        daily_vol = returns.std()
        
        # Annualize correctly: multiply by sqrt(trading days), NOT by days!
        annual_vol = daily_vol * np.sqrt(252)
        
        return annual_vol
        
    except Exception as e:
        print(f"  ERROR: {ticker} - {str(e)}")
        return None


def diagnose_volatility_bug():
    """Main diagnostic function"""
    
    print("=" * 70)
    print("VOLATILITY DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Period: 1 year lookback")
    print(f"Method: Log returns, annualized with sqrt(252)")
    print()
    
    results = []
    
    print("Ticker | Current (WRONG) | Correct | Error | Issue")
    print("-" * 70)
    
    for ticker, wrong_vol in CURRENT_WRONG_DATA.items():
        correct_vol = calculate_correct_volatility(ticker)
        
        if correct_vol is not None:
            error = ((wrong_vol - correct_vol) / correct_vol) * 100
            
            # Diagnose the issue
            if wrong_vol == 2.00:
                issue = "Default/fallback value (200%)"
            elif abs(wrong_vol - correct_vol * 252) < 0.01:
                issue = "Multiplied by 252 instead of sqrt(252)!"
            elif abs(wrong_vol - correct_vol * 100) < 0.01:
                issue = "Treating % as decimal"
            elif abs(error) > 100:
                issue = "Calculation bug"
            else:
                issue = "Within tolerance"
            
            results.append({
                'ticker': ticker,
                'wrong_vol': wrong_vol,
                'correct_vol': correct_vol,
                'error_pct': error,
                'issue': issue
            })
            
            print(f"{ticker:6} | {wrong_vol:15.1%} | {correct_vol:7.1%} | {error:+6.0f}% | {issue}")
        else:
            print(f"{ticker:6} | {wrong_vol:15.1%} | ERROR   | N/A    | Data unavailable")
    
    print("=" * 70)
    
    # Summary statistics
    if results:
        df = pd.DataFrame(results)
        
        print("\nSUMMARY STATISTICS:")
        print(f"  Average error: {df['error_pct'].mean():+.0f}%")
        print(f"  Max error: {df['error_pct'].max():+.0f}%")
        print(f"  Tickers with 200% default: {sum(df['wrong_vol'] == 2.00)}/10")
        
        print("\nDIAGNOSED ISSUES:")
        issue_counts = df['issue'].value_counts()
        for issue, count in issue_counts.items():
            print(f"  • {issue}: {count} tickers")
        
        # Export correct values
        print("\n" + "=" * 70)
        print("CORRECTED VOLATILITY VALUES (copy to your code):")
        print("=" * 70)
        print("CORRECTED_VOLATILITY = {")
        for _, row in df.iterrows():
            print(f"    '{row['ticker']}': {row['correct_vol']:.4f},  # {row['correct_vol']:.1%}")
        print("}")
    
    return results


def suggest_fix():
    """Suggest code fixes based on diagnosis"""
    
    print("\n" + "=" * 70)
    print("SUGGESTED FIX FOR YOUR CODE:")
    print("=" * 70)
    
    print("""
In your market_data_provider.py (or wherever you calculate vol), replace with:

```python
def calculate_volatility(ticker, period='1y'):
    \"\"\"Calculate annualized volatility correctly\"\"\"
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        # Need sufficient data
        if len(hist) < 50:
            return 0.50  # Default to 50% for biotech (not 200%!)
        
        # CORRECT METHOD: Log returns
        returns = np.log(hist['Close'] / hist['Close'].shift(1))
        returns = returns.dropna()
        
        daily_vol = returns.std()
        
        # CRITICAL: Use sqrt(252), NOT 252!
        annual_vol = daily_vol * np.sqrt(252)
        
        # Sanity check for biotech
        if annual_vol < 0.20:  # 20% too low
            return 0.30  # Floor at 30% for biotech
        if annual_vol > 1.50:  # 150% suspiciously high
            print(f"WARNING: {ticker} vol {annual_vol:.1%} seems high")
            return min(annual_vol, 1.00)  # Cap at 100%
        
        return annual_vol
        
    except Exception as e:
        print(f"ERROR calculating vol for {ticker}: {e}")
        return 0.50  # Safe default
```

KEY CHANGES:
1. ✅ Use np.sqrt(252) not 252
2. ✅ Default to 50% not 200% 
3. ✅ Add sanity checks (20-150% range)
4. ✅ Use log returns (more stable)
""")


if __name__ == "__main__":
    # Run diagnostic
    results = diagnose_volatility_bug()
    
    # Suggest fix
    suggest_fix()
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Fix your volatility calculation using the code above")
    print("2. Re-run your portfolio generation")
    print("3. Verify position sizes make more sense")
    print("4. Remove the misleading 'morningstar_direct' label")
    print("=" * 70)
