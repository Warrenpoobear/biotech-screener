#!/usr/bin/env python3
"""
Create a full universe CSV from momentum signals data.
"""

import json
from pathlib import Path

def main():
    # Load tickers from momentum signals
    momentum_path = Path("outputs/momentum_signals.json")
    if not momentum_path.exists():
        print("Error: Run calculate_momentum_batch.py first")
        return 1

    with open(momentum_path) as f:
        data = json.load(f)

    signals = data.get("signals", {})
    tickers = [t for t in signals.keys() if not t.startswith("_")]
    tickers.sort()

    print(f"Found {len(tickers)} tickers")

    # Create universe CSV
    output_path = Path("data/universe/biotech_universe_308.csv")
    with open(output_path, "w") as f:
        f.write("ticker,name,sector\n")
        for ticker in tickers:
            f.write(f"{ticker},{ticker} Inc,Biotechnology\n")

    print(f"Created {output_path} with {len(tickers)} tickers")

    # Also create a JSON version
    json_path = Path("data/universe/biotech_universe_308.json")
    with open(json_path, "w") as f:
        json.dump({"tickers": tickers, "count": len(tickers)}, f, indent=2)

    print(f"Created {json_path}")
    return 0

if __name__ == "__main__":
    exit(main())
