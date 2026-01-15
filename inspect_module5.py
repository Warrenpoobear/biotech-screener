import json

with open('outputs/ranked_with_momentum.json') as f:
    data = json.load(f)

print("=== MODULE 5 COMPOSITE STRUCTURE ===")
m5 = data.get('module_5_composite', {})
print(f"Type: {type(m5)}")
print(f"Keys: {list(m5.keys())[:20]}")

# Check if it has a securities list
if 'securities' in m5:
    print(f"\nFound 'securities' key")
    securities = m5['securities']
    print(f"  Type: {type(securities)}")
    print(f"  Length: {len(securities) if isinstance(securities, list) else 'N/A'}")
    if isinstance(securities, list) and len(securities) > 0:
        print(f"  First item keys: {list(securities[0].keys())}")
        print(f"  Sample item: {securities[0]}")

# Check if it has ranked_securities
if 'ranked_securities' in m5:
    print(f"\nFound 'ranked_securities' key")
    ranked = m5['ranked_securities']
    print(f"  Type: {type(ranked)}")
    print(f"  Length: {len(ranked) if isinstance(ranked, list) else 'N/A'}")
    if isinstance(ranked, list) and len(ranked) > 0:
        print(f"  First item: {ranked[0]}")

# Check if it's a dict of tickers
if isinstance(m5, dict):
    ticker_like_keys = [k for k in m5.keys() if isinstance(k, str) and k.isupper() and 2 <= len(k) <= 6 and k not in ['USD']]
    if ticker_like_keys:
        print(f"\nFound {len(ticker_like_keys)} ticker-like keys")
        print(f"  Sample tickers: {ticker_like_keys[:10]}")
        sample_ticker = ticker_like_keys[0]
        print(f"  Sample data for {sample_ticker}: {m5[sample_ticker]}")
