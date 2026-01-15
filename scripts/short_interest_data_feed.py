#!/usr/bin/env python3
"""
short_interest_data_feed.py

Generates or loads short interest data for the biotech universe.

For production use, replace the synthetic generation with actual data sources:
- FINRA Short Interest Data (bi-monthly)
- S3 Partners (daily)
- Ortex (real-time)

Design Philosophy:
- DETERMINISTIC: Uses seeded RNG for reproducible synthetic data
- STDLIB-ONLY: No external dependencies
- PIT DISCIPLINE: All data tagged with report_date

Usage:
    # Generate synthetic data for testing
    python short_interest_data_feed.py --universe production_data/universe.json \
        --output data/short_interest.json --as-of-date 2026-01-10

    # With seed for reproducibility
    python short_interest_data_feed.py --universe production_data/universe.json \
        --output data/short_interest.json --as-of-date 2026-01-10 --seed 42

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import argparse
import hashlib
import json
import random
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Any, Optional


__version__ = "1.0.0"


# Biotech SI distribution parameters (based on industry data)
# Biotech tends to have higher SI than market average due to binary outcomes
SI_DISTRIBUTION = {
    # SI% buckets with approximate probabilities
    "very_low": {"range": (0.5, 2.0), "probability": 0.20},    # 20% have <2% SI
    "low": {"range": (2.0, 5.0), "probability": 0.30},         # 30% have 2-5% SI
    "moderate": {"range": (5.0, 10.0), "probability": 0.25},   # 25% have 5-10% SI
    "high": {"range": (10.0, 20.0), "probability": 0.15},      # 15% have 10-20% SI
    "very_high": {"range": (20.0, 40.0), "probability": 0.08}, # 8% have 20-40% SI
    "extreme": {"range": (40.0, 60.0), "probability": 0.02},   # 2% have >40% SI
}

# Days to cover distribution
DTC_DISTRIBUTION = {
    "liquid": {"range": (0.5, 2.0), "probability": 0.35},
    "normal": {"range": (2.0, 5.0), "probability": 0.40},
    "elevated": {"range": (5.0, 10.0), "probability": 0.18},
    "high": {"range": (10.0, 20.0), "probability": 0.05},
    "extreme": {"range": (20.0, 40.0), "probability": 0.02},
}


class DeterministicRNG:
    """Simple deterministic RNG using hashlib for reproducibility."""

    def __init__(self, seed: str):
        self.seed = seed
        self.counter = 0

    def _next_bytes(self) -> bytes:
        """Generate next batch of random bytes."""
        data = f"{self.seed}:{self.counter}".encode()
        self.counter += 1
        return hashlib.sha256(data).digest()

    def random(self) -> float:
        """Return random float in [0, 1)."""
        b = self._next_bytes()[:8]
        n = int.from_bytes(b, 'big')
        return n / (2**64)

    def uniform(self, a: float, b: float) -> float:
        """Return random float in [a, b)."""
        return a + (b - a) * self.random()

    def gauss(self, mu: float, sigma: float) -> float:
        """Approximate Gaussian using Box-Muller (simplified)."""
        u1 = max(1e-10, self.random())  # Avoid log(0)
        u2 = self.random()
        import math
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return mu + sigma * z

    def choice(self, items: List[Any]) -> Any:
        """Choose random item from list."""
        idx = int(self.random() * len(items))
        return items[min(idx, len(items) - 1)]


def generate_si_for_ticker(
    ticker: str,
    rng: DeterministicRNG,
    as_of_date: date,
    market_cap_mm: Optional[float] = None
) -> Dict[str, Any]:
    """
    Generate synthetic short interest data for a single ticker.

    Args:
        ticker: Stock ticker symbol
        rng: Deterministic RNG instance
        as_of_date: Report date
        market_cap_mm: Market cap in millions (affects SI patterns)

    Returns:
        SI data dict
    """
    # Select SI bucket based on distribution
    r = rng.random()
    cumulative = 0
    si_bucket = "moderate"  # default
    for bucket, params in SI_DISTRIBUTION.items():
        cumulative += params["probability"]
        if r <= cumulative:
            si_bucket = bucket
            break

    si_range = SI_DISTRIBUTION[si_bucket]["range"]
    si_pct = rng.uniform(si_range[0], si_range[1])

    # Select DTC bucket
    r = rng.random()
    cumulative = 0
    dtc_bucket = "normal"
    for bucket, params in DTC_DISTRIBUTION.items():
        cumulative += params["probability"]
        if r <= cumulative:
            dtc_bucket = bucket
            break

    dtc_range = DTC_DISTRIBUTION[dtc_bucket]["range"]
    days_to_cover = rng.uniform(dtc_range[0], dtc_range[1])

    # SI change (trending)
    # 60% stable, 20% increasing, 20% decreasing
    trend_r = rng.random()
    if trend_r < 0.20:
        si_change = rng.uniform(5, 25)  # Increasing
    elif trend_r < 0.40:
        si_change = rng.uniform(-25, -5)  # Decreasing (covering)
    else:
        si_change = rng.gauss(0, 5)  # Stable with noise

    # Institutional ownership (inverse correlation with very high SI)
    if si_pct > 25:
        inst_long = rng.uniform(40, 70)
    elif si_pct > 15:
        inst_long = rng.uniform(55, 80)
    else:
        inst_long = rng.uniform(65, 95)

    # Average daily volume (rough synthetic)
    if market_cap_mm and market_cap_mm > 10000:
        adv = rng.uniform(2000000, 20000000)
    elif market_cap_mm and market_cap_mm > 1000:
        adv = rng.uniform(500000, 5000000)
    else:
        adv = rng.uniform(100000, 2000000)

    # Report date is typically settlement date - 2 business days
    report_date = as_of_date - timedelta(days=3)

    return {
        "ticker": ticker,
        "short_interest_pct": str(Decimal(str(si_pct)).quantize(Decimal("0.1"))),
        "days_to_cover": str(Decimal(str(days_to_cover)).quantize(Decimal("0.1"))),
        "short_interest_change_pct": str(Decimal(str(si_change)).quantize(Decimal("0.1"))),
        "institutional_long_pct": str(Decimal(str(inst_long)).quantize(Decimal("0.1"))),
        "avg_daily_volume": str(int(adv)),
        "report_date": report_date.isoformat(),
    }


def generate_universe_si_data(
    tickers: List[str],
    as_of_date: date,
    seed: Optional[int] = None,
    market_caps: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Generate short interest data for entire universe.

    Args:
        tickers: List of ticker symbols
        as_of_date: As-of date for SI data
        seed: Random seed for reproducibility
        market_caps: Optional dict of ticker -> market_cap_mm

    Returns:
        List of SI data records
    """
    # Create deterministic seed from date + provided seed
    seed_str = f"{as_of_date.isoformat()}:{seed or 'default'}"
    rng = DeterministicRNG(seed_str)

    market_caps = market_caps or {}

    results = []
    for ticker in sorted(tickers):  # Sort for determinism
        si_data = generate_si_for_ticker(
            ticker=ticker,
            rng=rng,
            as_of_date=as_of_date,
            market_cap_mm=market_caps.get(ticker)
        )
        results.append(si_data)

    return results


def load_universe_tickers(universe_path: Path) -> tuple:
    """
    Load tickers and market caps from universe file.

    Returns:
        Tuple of (tickers list, market_caps dict)
    """
    with open(universe_path, 'r') as f:
        data = json.load(f)

    tickers = []
    market_caps = {}

    # Handle both array and dict formats
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = data.get("active_securities", data.get("records", []))
    else:
        records = []

    for rec in records:
        ticker = rec.get("ticker")
        if ticker:
            tickers.append(ticker)
            if rec.get("market_cap_mm"):
                market_caps[ticker] = float(rec["market_cap_mm"])

    return tickers, market_caps


def main():
    parser = argparse.ArgumentParser(
        description="Generate or load short interest data for biotech universe"
    )
    parser.add_argument(
        "--universe",
        type=Path,
        required=True,
        help="Path to universe.json file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for short_interest.json"
    )
    parser.add_argument(
        "--as-of-date",
        required=True,
        help="As-of date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing file"
    )

    args = parser.parse_args()

    # Parse date
    as_of_date = date.fromisoformat(args.as_of_date)

    # Load universe
    print(f"Loading universe from {args.universe}...")
    tickers, market_caps = load_universe_tickers(args.universe)
    print(f"  Found {len(tickers)} tickers")

    # Generate SI data
    print(f"Generating SI data for {args.as_of_date}...")
    si_data = generate_universe_si_data(
        tickers=tickers,
        as_of_date=as_of_date,
        seed=args.seed,
        market_caps=market_caps
    )

    # Compute summary stats
    si_values = [float(r["short_interest_pct"]) for r in si_data]
    dtc_values = [float(r["days_to_cover"]) for r in si_data]

    print(f"\nSI Distribution Summary:")
    print(f"  Mean SI%: {sum(si_values)/len(si_values):.1f}%")
    print(f"  Max SI%: {max(si_values):.1f}%")
    print(f"  >10% SI: {sum(1 for v in si_values if v > 10)} tickers")
    print(f"  >20% SI: {sum(1 for v in si_values if v > 20)} tickers")
    print(f"  Mean DTC: {sum(dtc_values)/len(dtc_values):.1f} days")

    # Identify squeeze candidates
    squeeze_candidates = [
        r for r in si_data
        if float(r["short_interest_pct"]) > 15
        and float(r["days_to_cover"]) > 5
    ]
    print(f"\nPotential Squeeze Candidates ({len(squeeze_candidates)}):")
    for r in sorted(squeeze_candidates, key=lambda x: -float(x["short_interest_pct"]))[:10]:
        print(f"  {r['ticker']}: SI={r['short_interest_pct']}%, DTC={r['days_to_cover']}")

    if args.dry_run:
        print("\n[DRY-RUN] Not writing output file")
        return 0

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(si_data, f, indent=2)
        f.write('\n')

    print(f"\nWrote {len(si_data)} SI records to {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
