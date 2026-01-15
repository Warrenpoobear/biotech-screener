#!/usr/bin/env python3
"""
Create synthetic historical financial snapshots from current data.

This is a workaround when SEC API is not accessible. It creates quarterly
snapshots going back 5 years using current financial data as a base.

NOTE: This is NOT proper PIT data - all snapshots have the same values.
It's meant to allow the scoring modules to function for testing purposes.
"""
import json
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

def main():
    # Load current financial records
    data_dir = Path(__file__).parent / "production_data"
    financial_file = data_dir / "financial_records.json"

    if not financial_file.exists():
        print(f"Error: {financial_file} not found")
        return

    with open(financial_file) as f:
        current_records = json.load(f)

    print(f"Loaded {len(current_records)} current financial records")

    # Generate quarterly dates for 5 years (2020-2024)
    dates = []
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)

    current = start_date
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        current += relativedelta(months=3)

    print(f"Generated {len(dates)} quarterly dates from {dates[0]} to {dates[-1]}")

    # Create snapshots for each ticker at each date
    all_snapshots = []

    for record in current_records:
        ticker = record.get("ticker")
        if not ticker:
            continue

        for date in dates:
            snapshot = {
                "ticker": ticker,
                "date": date,
                "cash": record.get("Cash"),
                "debt": record.get("Liabilities"),  # Using Liabilities as proxy for Debt
                "assets": record.get("Assets"),
                "liabilities": record.get("Liabilities"),
                "rd_expense": record.get("R&D"),
                "revenue": record.get("Revenue"),
                "net_income": record.get("NetIncome"),
                "current_assets": record.get("CurrentAssets"),
                "current_liabilities": record.get("CurrentLiabilities"),
                "shareholders_equity": record.get("ShareholdersEquity"),
                "source": "synthetic",  # Mark as synthetic data
                "source_date": record.get("collected_at", "2026-01-07"),
            }
            all_snapshots.append(snapshot)

    # Sort by date, then ticker
    all_snapshots.sort(key=lambda x: (x['date'], x['ticker']))

    print(f"Created {len(all_snapshots)} total snapshots")
    print(f"  Tickers: {len(current_records)}")
    print(f"  Dates: {len(dates)}")

    # Save to production_data
    output_file = data_dir / "historical_financial_snapshots.json"
    with open(output_file, 'w') as f:
        json.dump(all_snapshots, f, indent=2)

    print(f"\nSaved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")

    # Verify structure
    sample = all_snapshots[0]
    print(f"\nSample snapshot:")
    print(f"  ticker: {sample['ticker']}")
    print(f"  date: {sample['date']}")
    print(f"  cash: {sample['cash']}")
    print(f"  assets: {sample['assets']}")
    print(f"  rd_expense: {sample['rd_expense']}")

if __name__ == "__main__":
    main()
