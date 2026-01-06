import yfinance as yf
from datetime import date

ticker = 'VRTX'
as_of = date(2024, 12, 31)

print(f"Testing {ticker} as of {as_of}")
print(f"Today is: {date.today()}")

stock = yf.Ticker(ticker)
hist = stock.history(period='1y', auto_adjust=True, actions=False)

print(f"\nTotal rows: {len(hist)}")
print(f"\nFirst 5 rows:")
print(hist.head())
print(f"\nLast 5 rows:")
print(hist.tail())

# Check filtering
print(f"\nFiltering to dates <= {as_of}")
filtered = []
for idx, row in hist.iterrows():
    if hasattr(idx, 'date'):
        price_date = idx.date()
    else:
        price_date = idx.to_pydatetime().date()
    
    print(f"  Date: {price_date}, <= {as_of}? {price_date <= as_of}")
    
    if price_date <= as_of:
        filtered.append(price_date)

print(f"\nFiltered count: {len(filtered)}")
