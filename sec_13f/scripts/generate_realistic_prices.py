"""
Generate realistic 10-year historical price data for biotech tickers.
Uses actual historical biotech sector characteristics:
- Real market cycles (2015-2016 biotech bear, 2020 COVID rally, 2021-2022 crash)
- Realistic volatility regimes
- Sector correlation
- Event-driven moves (FDA approvals, trial results)
"""
import csv
import random
import math
from datetime import date, timedelta
from decimal import Decimal

# Seed for reproducibility
random.seed(42)

# Sample biotech tickers
TICKERS = [
    "AMGN", "GILD", "VRTX", "REGN", "BIIB",
    "ALNY", "BMRN", "SGEN", "INCY", "EXEL",
    "MRNA", "BNTX", "IONS", "SRPT", "RARE",
    "BLUE", "FOLD", "ACAD", "HALO", "KRTX",
    "IMVT", "ARWR", "PCVX", "BEAM", "EDIT",
]

# Realistic starting prices (approx Jan 2015 levels, adjusted)
STARTING_PRICES = {
    "AMGN": 159.0, "GILD": 94.0, "VRTX": 115.0, "REGN": 410.0, "BIIB": 340.0,
    "ALNY": 95.0, "BMRN": 95.0, "SGEN": 115.0, "INCY": 70.0, "EXEL": 5.0,
    "MRNA": 20.0, "BNTX": 15.0, "IONS": 55.0, "SRPT": 15.0, "RARE": 45.0,
    "BLUE": 120.0, "FOLD": 12.0, "ACAD": 32.0, "HALO": 8.0, "KRTX": 25.0,
    "IMVT": 15.0, "ARWR": 8.0, "PCVX": 30.0, "BEAM": 25.0, "EDIT": 35.0,
}

# Base volatility (daily)
BASE_VOLATILITY = {
    "AMGN": 0.016, "GILD": 0.018, "VRTX": 0.022, "REGN": 0.024, "BIIB": 0.026,
    "ALNY": 0.032, "BMRN": 0.028, "SGEN": 0.024, "INCY": 0.030, "EXEL": 0.038,
    "MRNA": 0.045, "BNTX": 0.048, "IONS": 0.032, "SRPT": 0.042, "RARE": 0.040,
    "BLUE": 0.055, "FOLD": 0.038, "ACAD": 0.035, "HALO": 0.040, "KRTX": 0.045,
    "IMVT": 0.050, "ARWR": 0.042, "PCVX": 0.038, "BEAM": 0.052, "EDIT": 0.055,
}

# Realistic 10-year returns (total, split by periods)
# Based on actual biotech performance patterns
PERIOD_RETURNS = {
    # 2015: Biotech peak then crash
    (date(2015, 1, 1), date(2015, 7, 20)): {"sector": 0.25, "vol_mult": 1.0},
    (date(2015, 7, 21), date(2016, 2, 11)): {"sector": -0.40, "vol_mult": 1.8},
    # 2016: Recovery
    (date(2016, 2, 12), date(2016, 12, 31)): {"sector": -0.10, "vol_mult": 1.3},
    # 2017: Strong year
    (date(2017, 1, 1), date(2017, 12, 31)): {"sector": 0.20, "vol_mult": 0.9},
    # 2018: Volatile
    (date(2018, 1, 1), date(2018, 12, 31)): {"sector": -0.08, "vol_mult": 1.2},
    # 2019: Recovery
    (date(2019, 1, 1), date(2019, 12, 31)): {"sector": 0.25, "vol_mult": 1.0},
    # 2020: COVID crash then mRNA rally
    (date(2020, 1, 1), date(2020, 3, 23)): {"sector": -0.25, "vol_mult": 2.5},
    (date(2020, 3, 24), date(2020, 12, 31)): {"sector": 0.50, "vol_mult": 1.5},
    # 2021: Peak then decline
    (date(2021, 1, 1), date(2021, 2, 9)): {"sector": 0.15, "vol_mult": 1.3},
    (date(2021, 2, 10), date(2021, 12, 31)): {"sector": -0.25, "vol_mult": 1.4},
    # 2022: Bear market
    (date(2022, 1, 1), date(2022, 12, 31)): {"sector": -0.30, "vol_mult": 1.5},
    # 2023: Stabilization
    (date(2023, 1, 1), date(2023, 12, 31)): {"sector": 0.05, "vol_mult": 1.1},
    # 2024: Recovery
    (date(2024, 1, 1), date(2024, 12, 31)): {"sector": 0.15, "vol_mult": 1.0},
}

# Individual stock alpha (outperformance vs sector)
STOCK_ALPHA = {
    "AMGN": 0.02, "GILD": -0.05, "VRTX": 0.15, "REGN": 0.08, "BIIB": -0.08,
    "ALNY": 0.12, "BMRN": -0.02, "SGEN": 0.05, "INCY": 0.03, "EXEL": 0.20,
    "MRNA": 0.30, "BNTX": 0.25, "IONS": 0.05, "SRPT": 0.25, "RARE": 0.10,
    "BLUE": -0.35, "FOLD": 0.08, "ACAD": -0.05, "HALO": 0.15, "KRTX": 0.10,
    "IMVT": 0.20, "ARWR": 0.12, "PCVX": 0.08, "BEAM": -0.15, "EDIT": -0.20,
}

def get_period_params(current_date):
    """Get sector return and volatility multiplier for current date."""
    for (start, end), params in PERIOD_RETURNS.items():
        if start <= current_date <= end:
            return params
    return {"sector": 0.0, "vol_mult": 1.0}

def generate_prices(start_date: date, end_date: date) -> list:
    """Generate realistic daily prices for all tickers."""
    rows = []

    # Initialize prices
    prices = {t: STARTING_PRICES[t] for t in TICKERS}

    # Track running sector return for correlation
    sector_noise = 0.0

    current = start_date
    trading_days = 0

    while current <= end_date:
        # Skip weekends
        if current.weekday() < 5:
            trading_days += 1

            # Get period parameters
            params = get_period_params(current)
            sector_daily_return = params["sector"] / 252  # Annualized to daily
            vol_multiplier = params["vol_mult"]

            # Generate correlated sector move (common factor)
            sector_noise = 0.7 * sector_noise + 0.3 * random.gauss(0, 1)
            sector_move = sector_daily_return + 0.02 * sector_noise

            for ticker in TICKERS:
                # Base volatility with regime adjustment
                daily_vol = BASE_VOLATILITY[ticker] * vol_multiplier

                # Stock-specific alpha (annualized to daily)
                alpha = STOCK_ALPHA[ticker] / 252

                # Idiosyncratic shock
                idio_shock = random.gauss(0, 1)

                # Combine: sector beta + alpha + idiosyncratic
                beta = 0.8 + 0.4 * random.random()  # Beta between 0.8-1.2
                daily_return = (beta * sector_move) + alpha + (daily_vol * idio_shock * 0.6)

                # Occasional large moves (FDA events, trial results)
                if random.random() < 0.005:  # ~1.25 events per year
                    event_move = random.choice([-1, 1]) * random.uniform(0.10, 0.30)
                    daily_return += event_move

                # Update price
                prices[ticker] = prices[ticker] * (1 + daily_return)

                # Floor at $0.50
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
    start = date(2020, 1, 2)
    end = date(2024, 12, 31)

    print(f"Generating realistic prices from {start} to {end}...")
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

    # Show final prices vs starting
    print("\nPrice evolution (Start → End):")
    final_prices = {}
    for row in rows[-len(TICKERS):]:
        final_prices[row['ticker']] = float(row['adj_close'])

    for ticker in sorted(TICKERS):
        start_p = STARTING_PRICES[ticker]
        end_p = final_prices.get(ticker, 0)
        ret = ((end_p / start_p) - 1) * 100
        print(f"  {ticker}: ${start_p:.2f} → ${end_p:.2f} ({ret:+.1f}%)")


if __name__ == "__main__":
    main()
