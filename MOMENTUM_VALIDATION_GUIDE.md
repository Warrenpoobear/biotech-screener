# Momentum Signal Validation - Implementation Guide

## Overview

This guide explains how to validate your momentum signal calculations against the documented performance claims:
- **87.6% quartile spread** in bull markets
- **Information Coefficient of 0.713**
- Proper volatility annualization (sqrt(252) not 252)
- Regime-adaptive weighting

## Quick Start

```bash
# Run basic validation (mathematical tests only)
python validate_momentum_signal.py

# Expected output:
# ✅ Volatility annualization: PASS
# ✅ Mathematical properties: PASS
# ⚠️  Look-ahead bias: PENDING (needs data)
# ⚠️  Quartile spread: PENDING (needs backtest)
# ⚠️  Information Coefficient: PENDING (needs backtest)
```

## Step 1: Implement Data Loading

The validation script needs historical price data. Update the `load_test_data()` method:

```python
def load_test_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load price data for testing."""
    
    # Load universe prices
    universe_file = self.data_dir / 'universe_prices.csv'
    universe_df = pd.read_csv(universe_file, index_col=0, parse_dates=True)
    
    # Load index data
    indices_file = self.data_dir / 'indices_prices.csv'
    indices_df = pd.read_csv(indices_file, index_col=0, parse_dates=True)
    
    xbi_prices = indices_df['XBI']
    spy_prices = indices_df['SPY']
    
    return universe_df, xbi_prices, spy_prices
```

### Data Requirements

Your data files should contain:

**universe_prices.csv**:
```
date,MRNA,GILD,VRTX,...
2024-01-01,150.25,85.30,420.15,...
2024-01-02,152.10,84.90,418.50,...
...
```

**indices_prices.csv**:
```
date,XBI,SPY
2024-01-01,95.50,475.30
2024-01-02,96.20,476.15
...
```

## Step 2: Run Volatility Validation

This test runs immediately with synthetic data:

```python
validator = MomentumValidator(data_dir=Path('data'))
validator.test_volatility_annualization()
```

**Expected Output:**
```
==================================================
TEST 1: Volatility Annualization
==================================================

Daily volatility (input): 0.0200
Expected annual: 0.3175 (31.75%)
Actual annual: 0.3182 (31.82%)

❌ WRONG METHOD (×252): 5.0400 (504.00%)
   This would be 15.8x too high!

✅ PASS: Error 0.22% < 5% threshold
```

**If this fails**, your code is still using the old bug (multiplying by 252).

## Step 3: Implement Backtest for Quartile Spread

To validate the 87.6% claim, you need to run a historical backtest:

```python
def calculate_quartile_spread_backtest(
    validator: MomentumValidator,
    start_date: str = '2020-01-01',
    end_date: str = '2024-12-31'
):
    """
    Backtest quartile spread performance.
    
    Process:
    1. For each month T:
       a. Calculate momentum for all tickers
       b. Sort into quartiles (Q1 = highest momentum)
       c. Wait 30 days
       d. Calculate forward returns for each quartile
    2. Average results across all months
    3. Calculate Q1 - Q4 spread
    """
    
    universe_df, xbi, spy = validator.load_test_data()
    
    monthly_spreads = []
    
    # Iterate monthly
    dates = pd.date_range(start_date, end_date, freq='MS')
    
    for calc_date in dates:
        # Calculate momentum for all tickers
        momentum_scores = {}
        
        for ticker in universe_df.columns:
            try:
                signal, _ = validator.calculator.calculate_momentum_signal(
                    ticker,
                    universe_df[ticker],
                    calc_date
                )
                momentum_scores[ticker] = float(signal)
            except Exception:
                continue
        
        if len(momentum_scores) < 20:  # Need minimum universe
            continue
        
        # Sort into quartiles
        sorted_tickers = sorted(
            momentum_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        n = len(sorted_tickers)
        q1_tickers = [t for t, s in sorted_tickers[:n//4]]  # Top quartile
        q4_tickers = [t for t, s in sorted_tickers[-n//4:]]  # Bottom quartile
        
        # Calculate forward 30-day returns
        forward_date = calc_date + timedelta(days=30)
        
        try:
            q1_returns = []
            for ticker in q1_tickers:
                ret = (universe_df[ticker].loc[forward_date] / 
                       universe_df[ticker].loc[calc_date]) - 1
                q1_returns.append(ret)
            
            q4_returns = []
            for ticker in q4_tickers:
                ret = (universe_df[ticker].loc[forward_date] / 
                       universe_df[ticker].loc[calc_date]) - 1
                q4_returns.append(ret)
            
            # Calculate spread
            spread = np.mean(q1_returns) - np.mean(q4_returns)
            monthly_spreads.append({
                'date': calc_date,
                'q1_return': np.mean(q1_returns),
                'q4_return': np.mean(q4_returns),
                'spread': spread
            })
            
        except KeyError:
            continue
    
    # Analyze results
    df = pd.DataFrame(monthly_spreads)
    
    print("\nQUARTILE SPREAD BACKTEST RESULTS")
    print("=" * 50)
    print(f"Period: {start_date} to {end_date}")
    print(f"Months analyzed: {len(df)}")
    print(f"\nAverage Spread: {df['spread'].mean()*100:.2f}%")
    print(f"Expected: {EXPECTED_METRICS['bull_quartile_spread']*100:.1f}%")
    print(f"\nQ1 Avg Return: {df['q1_return'].mean()*100:.2f}%")
    print(f"Q4 Avg Return: {df['q4_return'].mean()*100:.2f}%")
    print(f"Hit Rate (Q1 > Q4): {(df['spread'] > 0).mean()*100:.1f}%")
    
    # Test vs expected
    actual_spread = df['spread'].mean()
    expected_spread = EXPECTED_METRICS['bull_quartile_spread']
    
    if actual_spread >= expected_spread * 0.8:  # Allow 20% tolerance
        print("\n✅ PASS: Spread meets expectations")
        return True
    else:
        print(f"\n❌ FAIL: Spread {actual_spread*100:.1f}% << Expected {expected_spread*100:.1f}%")
        return False
```

## Step 4: Calculate Information Coefficient

```python
def calculate_information_coefficient(
    validator: MomentumValidator,
    start_date: str = '2020-01-01',
    end_date: str = '2024-12-31'
):
    """
    Calculate IC = Spearman correlation between momentum ranks and forward returns.
    
    An IC of 0.713 is exceptionally high - this is the key test.
    """
    from scipy.stats import spearmanr
    
    universe_df, _, _ = validator.load_test_data()
    
    all_momentum = []
    all_returns = []
    
    dates = pd.date_range(start_date, end_date, freq='MS')
    
    for calc_date in dates:
        for ticker in universe_df.columns:
            try:
                # Calculate momentum
                signal, _ = validator.calculator.calculate_momentum_signal(
                    ticker,
                    universe_df[ticker],
                    calc_date
                )
                
                # Calculate forward return
                forward_date = calc_date + timedelta(days=30)
                fwd_return = (universe_df[ticker].loc[forward_date] / 
                             universe_df[ticker].loc[calc_date]) - 1
                
                all_momentum.append(float(signal))
                all_returns.append(fwd_return)
                
            except Exception:
                continue
    
    # Calculate Spearman correlation
    ic, p_value = spearmanr(all_momentum, all_returns)
    
    print("\nINFORMATION COEFFICIENT ANALYSIS")
    print("=" * 50)
    print(f"Data points: {len(all_momentum)}")
    print(f"IC (Spearman ρ): {ic:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"\nExpected IC: {EXPECTED_METRICS['information_coefficient']:.3f}")
    
    # Interpretation
    if ic >= 0.10:
        quality = "STRONG ✅" if ic >= 0.30 else "GOOD ✓" 
    else:
        quality = "WEAK ❌"
    
    print(f"Signal Quality: {quality}")
    
    # Test vs expected
    if ic >= EXPECTED_METRICS['information_coefficient'] * 0.8:
        print("\n✅ PASS: IC meets expectations")
        return True
    else:
        print(f"\n❌ FAIL: IC {ic:.3f} << Expected {EXPECTED_METRICS['information_coefficient']:.3f}")
        return False
```

## Step 5: Validate Regime Detection

```python
def validate_regime_detection(
    validator: MomentumValidator,
    known_bull_period: tuple = ('2020-01-01', '2021-12-31'),
    known_bear_period: tuple = ('2022-01-01', '2022-12-31')
):
    """
    Test regime detection against known market periods.
    """
    _, xbi, spy = validator.load_test_data()
    
    # Test bull period
    bull_dates = pd.date_range(known_bull_period[0], known_bull_period[1], freq='MS')
    bull_regimes = []
    
    for date in bull_dates:
        regime_metrics = validator.regime_detector.detect_regime(
            xbi, spy, None, date  # universe not needed for basic detection
        )
        bull_regimes.append(regime_metrics.regime)
    
    bull_pct = sum(r == 'bull' for r in bull_regimes) / len(bull_regimes)
    
    # Test bear period
    bear_dates = pd.date_range(known_bear_period[0], known_bear_period[1], freq='MS')
    bear_regimes = []
    
    for date in bear_dates:
        regime_metrics = validator.regime_detector.detect_regime(
            xbi, spy, None, date
        )
        bear_regimes.append(regime_metrics.regime)
    
    bear_pct = sum(r == 'bear' for r in bear_regimes) / len(bear_regimes)
    
    print("\nREGIME DETECTION VALIDATION")
    print("=" * 50)
    print(f"Bull period ({known_bull_period[0]} to {known_bull_period[1]}):")
    print(f"  Detected as 'bull': {bull_pct*100:.1f}%")
    print(f"\nBear period ({known_bear_period[0]} to {known_bear_period[1]}):")
    print(f"  Detected as 'bear': {bear_pct*100:.1f}%")
    
    # Should correctly identify >70% of periods
    if bull_pct > 0.7 and bear_pct > 0.7:
        print("\n✅ PASS: Regime detection works correctly")
        return True
    else:
        print("\n❌ FAIL: Regime detection needs tuning")
        return False
```

## Step 6: Run Complete Validation

```python
# Complete validation workflow
def main():
    validator = MomentumValidator(data_dir=Path('data'))
    
    print("Starting comprehensive momentum validation...")
    
    # 1. Mathematical tests (run immediately)
    validator.run_all_tests()
    
    # 2. Backtest tests (require historical data)
    print("\n" + "=" * 70)
    print("RUNNING BACKTEST VALIDATION")
    print("=" * 70)
    
    quartile_pass = calculate_quartile_spread_backtest(validator)
    ic_pass = calculate_information_coefficient(validator)
    regime_pass = validate_regime_detection(validator)
    
    # 3. Final verdict
    print("\n" + "*" * 70)
    print("FINAL VALIDATION VERDICT")
    print("*" * 70)
    
    all_passed = quartile_pass and ic_pass and regime_pass
    
    if all_passed:
        print("\n✅ ALL VALIDATIONS PASSED")
        print("   Momentum signal is production-ready")
    else:
        print("\n❌ VALIDATION FAILURES DETECTED")
        print("   DO NOT USE IN PRODUCTION until issues resolved")

if __name__ == '__main__':
    main()
```

## Expected Results

### Pass Criteria

| Test | Criterion | Tolerance |
|------|-----------|-----------|
| Volatility | Uses sqrt(252) | Exact |
| Quartile Spread | ≥70% of 87.6% | ≥61% |
| Information Coefficient | ≥0.57 | ≥80% of 0.713 |
| Regime Detection | >70% accuracy | >70% |
| Look-ahead Bias | None detected | Zero tolerance |

### Red Flags

🚩 **If volatility test fails**: You're still using the old bug (×252 instead of ×sqrt(252))

🚩 **If quartile spread <30%**: Momentum has no predictive power

🚩 **If IC <0.10**: Signal is noise, not alpha

🚩 **If regime detection <50% accuracy**: Thresholds need calibration

## Next Steps After Validation

Once all tests pass:

1. **Document baseline metrics** for monitoring drift
2. **Set up weekly validation** as part of screening run
3. **Implement alpha decay monitoring** (test monthly)
4. **Add kill switches** if IC drops below 0.20
5. **Integrate with full screening pipeline**

## Troubleshooting

### "Insufficient data" errors
- Need at least 63 trading days of history
- Check for delisted tickers
- Verify date range covers full period

### IC significantly below 0.713
- Check if you're testing only bull markets (IC varies by regime)
- Verify no look-ahead bias in calculation
- Ensure proper handling of corporate actions

### Regime detection unstable
- Try different threshold values
- Add smoothing (e.g., 3-month rolling average)
- Consider additional indicators (VIX, funding rates)

## Files Generated

After running validation:

```
momentum_validation_results.json  # Test results
quartile_spread_analysis.csv      # Monthly spread data
ic_time_series.csv                # Rolling IC over time
regime_history.csv                # Historical regime classifications
```

## Integration with Production

Once validated, integrate with your screening pipeline:

```python
# In weekly_screening.py
from validate_momentum_signal import MomentumCalculator, RegimeDetector

calculator = MomentumCalculator()
regime_detector = RegimeDetector()

# For each ticker in universe:
momentum_signal, diagnostics = calculator.calculate_momentum_signal(
    ticker, prices, as_of_date
)

# Detect current regime
regime = regime_detector.detect_regime(xbi, spy, universe_prices, as_of_date)

# Apply regime-adaptive weighting
if regime.regime == 'bull':
    momentum_weight = 0.25
elif regime.regime == 'bear':
    momentum_weight = 0.05
else:
    momentum_weight = 0.15

final_score += momentum_signal * momentum_weight
```
