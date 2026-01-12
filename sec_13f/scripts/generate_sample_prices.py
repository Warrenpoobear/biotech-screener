"""
Generate sample daily price data for biotech tickers.
Creates realistic-looking price series with some variance.
"""
import csv
import random
from datetime import date, timedelta
from decimal import Decimal

# Seed for reproducibility
random.seed(42)

# Sample biotech tickers (mix of large, mid, small cap)
TICKERS = [
    # Large cap biotech
    "AMGN", "GILD", "VRTX", "REGN", "BIIB",
    # Mid cap
    "ALNY", "BMRN", "SGEN", "INCY", "EXEL",
    # Small/micro cap (higher volatility)
    "MRNA", "BNTX", "IONS", "SRPT", "RARE",
    "BLUE", "FOLD", "ACAD", "HALO", "KRTX",
    "IMVT", "ARWR", "PCVX", "BEAM", "EDIT",
]

# Starting prices (roughly realistic)
STARTING_PRICES = {
    "AMGN": 250.0, "GILD": 75.0, "VRTX": 350.0, "REGN": 750.0, "BIIB": 280.0,
    "ALNY": 180.0, "BMRN": 90.0, "SGEN": 150.0, "INCY": 70.0, "EXEL": 20.0,
    "MRNA": 120.0, "BNTX": 100.0, "IONS": 40.0, "SRPT": 120.0, "RARE": 35.0,
    "BLUE": 5.0, "FOLD": 15.0, "ACAD": 18.0, "HALO": 45.0, "KRTX": 55.0,
    "IMVT": 25.0, "ARWR": 30.0, "PCVX": 80.0, "BEAM": 35.0, "EDIT": 12.0,
}

# Volatility by ticker (daily std dev as %)
VOLATILITY = {
    "AMGN": 0.015, "GILD": 0.018, "VRTX": 0.020, "REGN": 0.022, "BIIB": 0.025,
    "ALNY": 0.028, "BMRN": 0.025, "SGEN": 0.022, "INCY": 0.030, "EXEL": 0.035,
    "MRNA": 0.045, "BNTX": 0.050, "IONS": 0.035, "SRPT": 0.040, "RARE": 0.045,
    "BLUE": 0.060, "FOLD": 0.040, "ACAD": 0.035, "HALO": 0.038, "KRTX": 0.042,
    "IMVT": 0.050, "ARWR": 0.045, "PCVX": 0.035, "BEAM": 0.055, "EDIT": 0.060,
}

# Drift (annualized expected return - varies by ticker)
DRIFT = {
    "AMGN": 0.05, "GILD": 0.02, "VRTX": 0.15, "REGN": 0.10, "BIIB": -0.05,
    "ALNY": 0.20, "BMRN": 0.08, "SGEN": 0.12, "INCY": 0.05, "EXEL": 0.15,
    "MRNA": -0.10, "BNTX": -0.15, "IONS": 0.10, "SRPT": 0.25, "RARE": 0.30,
    "BLUE": -0.40, "FOLD": 0.15, "ACAD": 0.08, "HALO": 0.20, "KRTX": 0.25,
    "IMVT": 0.35, "ARWR": 0.18, "PCVX": 0.12, "BEAM": -0.20, "EDIT": -0.25,
}

def generate_prices(start_date: date, end_date: date) -> list:
    """Generate daily prices for all tickers."""
    rows = []
    
    # Initialize prices
    prices = {t: STARTING_PRICES[t] for t in TICKERS}
    
    current = start_date
    while current <= end_date:
        # Skip weekends
        if current.weekday() < 5:
            for ticker in TICKERS:
                # Random walk with drift
                daily_drift = DRIFT[ticker] / 252  # Annualized → daily
                daily_vol = VOLATILITY[ticker]
                
                # Log-normal return
                shock = random.gauss(0, 1)
                daily_return = daily_drift + daily_vol * shock
                
                # Update price
                prices[ticker] = prices[ticker] * (1 + daily_return)
                
                # Floor at $0.50 (prevent negative/zero)
                prices[ticker] = max(0.50, prices[ticker])
                
                rows.append({
                    "date": current.isoformat(),
                    "ticker": ticker,
                    "adj_close": f"{prices[ticker]:.2f}",
                })
        
        current += timedelta(days=1)
    
    return rows


def main():
    # Generate 5 years of data (2020-2024)
    start = date(2020, 1, 2)  # First trading day 2020
    end = date(2024, 12, 31)
    
    print(f"Generating prices from {start} to {end}...")
    rows = generate_prices(start, end)
    
    # Write CSV
    output_path = "/home/user/biotech-screener/data/daily_prices.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "ticker", "adj_close"])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Tickers: {len(TICKERS)}")
    print(f"Date range: {rows[0]['date']} to {rows[-1]['date']}")


if __name__ == "__main__":
    main()
