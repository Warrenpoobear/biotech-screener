#!/usr/bin/env python3
"""Debug script to diagnose validation return calculation."""

import json
from datetime import date, timedelta
from decimal import Decimal

# Load database
db_path = 'data/returns/returns_db_2020-01-01_2026-01-13.json'
with open(db_path) as f:
    data = json.load(f)

returns = data.get('returns', {})
print(f"Database has {len(returns)} tickers")

# Pick a sample ticker
sample_ticker = 'VRTX'
if sample_ticker not in returns:
    sample_ticker = list(returns.keys())[0] if returns else None

if not sample_ticker:
    print("No tickers in database!")
    exit(1)

ticker_returns = returns[sample_ticker]
print(f"\n=== {sample_ticker} has {len(ticker_returns)} monthly returns ===")
print("First 5:")
for r in ticker_returns[:5]:
    print(f"  {r['date']}: {r['return']:.4f}%")
print("Last 5:")
for r in ticker_returns[-5:]:
    print(f"  {r['date']}: {r['return']:.4f}%")

# Simulate forward return calculation
screen_date = "2025-01-01"
forward_months = 3
screen_dt = date.fromisoformat(screen_date)
end_dt = screen_dt + timedelta(days=forward_months * 30)

print(f"\n=== Forward return calculation ===")
print(f"Screen date: {screen_date}")
print(f"End date: {end_dt}")

forward_returns = []
for r in ticker_returns:
    r_date = date.fromisoformat(r["date"][:10])
    if screen_dt < r_date <= end_dt:
        forward_returns.append(r)
        print(f"  INCLUDED: {r['date']} = {r['return']:.4f}%")

print(f"\nTotal returns in window: {len(forward_returns)}")

if forward_returns:
    # Compound returns correctly
    cumulative = Decimal("1")
    for r in forward_returns:
        ret_pct = Decimal(str(r["return"]))
        cumulative *= (Decimal("1") + ret_pct / Decimal("100"))
        print(f"  After {r['date']}: cumulative = {cumulative:.6f}")

    total_return = (cumulative - Decimal("1")) * Decimal("100")
    print(f"\nFinal compounded return: {total_return:.2f}%")
