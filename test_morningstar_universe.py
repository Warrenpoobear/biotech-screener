'''
Comprehensive Morningstar Data Feed Test for Full Biotech Universe
===================================================================
Tests Morningstar daily returns coverage and quality for all tickers
'''

import json
import statistics
import time
from datetime import date, timedelta
from collections import defaultdict
from wake_robin_data_pipeline.morningstar_data_provider import MorningstarDataProvider

def test_universe_coverage():
    print('='*70)
    print('MORNINGSTAR BIOTECH UNIVERSE COVERAGE TEST')
    print('='*70)
    
    # Load universe
    print('\n1. Loading universe...')
    with open('production_data/universe.json', 'r') as f:
        universe = json.load(f)
    
    tickers = sorted(set(s['ticker'] for s in universe if s.get('ticker')))
    print(f'   Found {len(tickers)} unique tickers')
    
    # Initialize provider
    provider = MorningstarDataProvider()
    
    # Results tracking
    results = {
        'success': [],
        'partial': [],
        'failed': [],
        'volatilities': {},
        'data_points': {},
    }
    
    # Test each ticker
    print('\n2. Testing data fetch (this may take a few minutes)...')
    print('   Ticker | Returns | Volatility | Status')
    print('   ' + '-'*60)
    
    start_time = time.time()
    
    for i, ticker in enumerate(tickers, 1):
        try:
            # Fetch data
            returns = provider.get_daily_returns(
                ticker=ticker,
                as_of_date=date.today(),
                lookback_days=252  # Full year
            )
            
            if not returns:
                results['failed'].append(ticker)
                print(f'   {ticker:6} | None    | N/A        | ❌ No data')
                continue
            
            num_returns = len(returns)
            results['data_points'][ticker] = num_returns
            
            # Calculate volatility if sufficient data
            if num_returns >= 50:
                daily_std = statistics.stdev(returns)
                annual_vol = daily_std * (252 ** 0.5)
                results['volatilities'][ticker] = annual_vol
                
                if num_returns >= 200:
                    results['success'].append(ticker)
                    status = '✅'
                else:
                    results['partial'].append(ticker)
                    status = '⚠️'
                
                print(f'   {ticker:6} | {num_returns:3}     | {annual_vol:5.1%}      | {status}')
            else:
                results['partial'].append(ticker)
                print(f'   {ticker:6} | {num_returns:3}     | N/A        | ⚠️  Low data')
            
            # Progress indicator
            if i % 20 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (len(tickers) - i) / rate
                print(f'   ... {i}/{len(tickers)} completed ({remaining:.0f}s remaining)')
        
        except Exception as e:
            results['failed'].append(ticker)
            error_msg = str(e)[:30]
            print(f'   {ticker:6} | Error   | N/A        | ❌ {error_msg}')
    
    elapsed = time.time() - start_time
    
    # Summary statistics
    print('\n' + '='*70)
    print('SUMMARY STATISTICS')
    print('='*70)
    
    total = len(tickers)
    success_count = len(results['success'])
    partial_count = len(results['partial'])
    failed_count = len(results['failed'])
    
    print(f'\nCoverage:')
    print(f'  Total tickers:        {total}')
    print(f'  ✅ Full data (≥200d):  {success_count} ({success_count/total*100:.1f}%)')
    print(f'  ⚠️  Partial (50-199d): {partial_count} ({partial_count/total*100:.1f}%)')
    print(f'  ❌ Failed/Low (<50d):  {failed_count} ({failed_count/total*100:.1f}%)')
    
    # Volatility distribution
    if results['volatilities']:
        vols = list(results['volatilities'].values())
        print(f'\nVolatility Distribution ({len(vols)} tickers):')
        print(f'  Min:    {min(vols):.1%}')
        print(f'  25th:   {sorted(vols)[len(vols)//4]:.1%}')
        print(f'  Median: {statistics.median(vols):.1%}')
        print(f'  75th:   {sorted(vols)[3*len(vols)//4]:.1%}')
        print(f'  Max:    {max(vols):.1%}')
        print(f'  Mean:   {statistics.mean(vols):.1%}')
        
        # Flag outliers
        outliers = [(t, v) for t, v in results['volatilities'].items() 
                    if v < 0.15 or v > 1.50]
        if outliers:
            print(f'\n  ⚠️  Volatility Outliers ({len(outliers)}):')
            for ticker, vol in sorted(outliers, key=lambda x: x[1])[:10]:
                print(f'     {ticker}: {vol:.1%}')
    
    # Data points distribution
    if results['data_points']:
        points = list(results['data_points'].values())
        print(f'\nData Points Distribution:')
        print(f'  Min:    {min(points)} days')
        print(f'  Median: {statistics.median(points):.0f} days')
        print(f'  Max:    {max(points)} days')
    
    print(f'\nPerformance:')
    print(f'  Total time: {elapsed:.1f}s')
    print(f'  Rate: {len(tickers)/elapsed:.1f} tickers/sec')
    
    # Save detailed results
    failed_tickers = results['failed']
    failed_len = len(failed_tickers)
    
    output = {
        'test_date': str(date.today()),
        'total_tickers': total,
        'success': {
            'tickers': results['success'],
            'count': success_count,
            'percentage': success_count/total*100
        },
        'partial': {
            'tickers': results['partial'],
            'count': partial_count,
            'percentage': partial_count/total*100
        },
        'failed': {
            'tickers': failed_tickers,
            'count': failed_count,
            'percentage': failed_count/total*100
        },
        'volatilities': {t: f'{v:.4f}' for t, v in results['volatilities'].items()},
        'data_points': results['data_points'],
        'elapsed_seconds': elapsed
    }
    
    output_file = 'morningstar_universe_test.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f'\n✅ Detailed results saved to: {output_file}')
    
    # Recommendations
    print('\n' + '='*70)
    print('RECOMMENDATIONS')
    print('='*70)
    
    if success_count / total >= 0.80:
        print('✅ Excellent coverage! Morningstar is production-ready.')
    elif success_count / total >= 0.60:
        print('⚠️  Good coverage, but some gaps. Review failed tickers.')
    else:
        print('❌ Low coverage. Consider keeping yfinance as primary.')
    
    if failed_count > 0:
        print(f'\n   Failed tickers may need fallback to yfinance:')
        for ticker in failed_tickers[:10]:
            print(f'     - {ticker}')
        if failed_len > 10:
            remaining = failed_len - 10
            print(f'     ... and {remaining} more')
    
    return results

if __name__ == '__main__':
    results = test_universe_coverage()
