import yfinance as yf
from datetime import date, timedelta

ticker = "VRTX"
as_of = date(2024, 12, 31)
start = as_of - timedelta(days=365)

print(f"Testing {ticker}")
print(f"Start: {start}")
print(f"End: {as_of}")

# Try method 1: Using period
print("\nMethod 1: Using period parameter")
stock = yf.Ticker(ticker)
hist1 = stock.history(period="1y")
print(f"  Rows: {len(hist1)}")

# Try method 2: Using start/end with string format
print("\nMethod 2: Using start/end dates")
hist2 = stock.history(start=start.strftime("%Y-%m-%d"), end=as_of.strftime("%Y-%m-%d"))
print(f"  Rows: {len(hist2)}")

# Try method 3: Most recent data
print("\nMethod 3: Recent data (no dates)")
hist3 = stock.history(period="1mo")
print(f"  Rows: {len(hist3)}")
print(f"  Last date: {hist3.index[-1] if len(hist3) > 0 else 'None'}")
