#!/usr/bin/env python3
"""
Compare this week vs last week momentum signals.

Shows:
- Regime changes
- Sweet spot entrants/exits
- Large momentum swings (±20 points)
- New high/low momentum tickers
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta


def find_weekly_file(base_pattern, target_date):
    """Find the closest weekly file to target date."""
    outputs_dir = Path('outputs')

    # Try exact date first
    date_suffix = target_date.strftime('%Y%m%d')
    exact_path = outputs_dir / f"{base_pattern}_{date_suffix}.json"
    if exact_path.exists():
        return exact_path

    # Search for closest file within 7 days
    for delta in range(1, 8):
        for direction in [-1, 1]:
            check_date = target_date + timedelta(days=delta * direction)
            check_suffix = check_date.strftime('%Y%m%d')
            check_path = outputs_dir / f"{base_pattern}_{check_suffix}.json"
            if check_path.exists():
                return check_path

    return None


def compare_weeks(current_date_str=None, previous_date_str=None):
    """Compare two weekly runs."""

    # Parse dates
    if current_date_str:
        current_date = datetime.strptime(current_date_str, '%Y-%m-%d')
    else:
        current_date = datetime.now()

    if previous_date_str:
        previous_date = datetime.strptime(previous_date_str, '%Y-%m-%d')
    else:
        previous_date = current_date - timedelta(days=7)

    # Find files
    curr_file = find_weekly_file('momentum_signals', current_date)
    prev_file = find_weekly_file('momentum_signals', previous_date)

    # Fall back to latest if dated files don't exist
    latest_file = Path('outputs/momentum_signals.json')

    if not curr_file and latest_file.exists():
        curr_file = latest_file
        print(f"Using latest momentum_signals.json for current week")

    if not curr_file:
        print("ERROR: No momentum signals file found for current week")
        return 1

    if not prev_file:
        print(f"No previous week file found - showing current week only")
        prev_file = None

    # Load current
    with open(curr_file, 'r') as f:
        current = json.load(f)

    curr_signals = current.get('signals', {})
    curr_regime = current.get('metadata', {}).get('regime', 'UNKNOWN')

    print("=" * 70)
    print(f"WEEKLY MOMENTUM REPORT")
    print("=" * 70)
    print(f"Current: {curr_file.name}")
    print(f"Regime: {curr_regime}")
    print(f"Tickers: {len(curr_signals)}")

    if not prev_file:
        # Just show current week summary
        print("\n" + "=" * 70)
        print("TOP 20 MOMENTUM (Current Week)")
        print("=" * 70)

        ranked = sorted(
            curr_signals.items(),
            key=lambda x: x[1].get('composite_momentum_score', 0),
            reverse=True
        )

        for i, (ticker, data) in enumerate(ranked[:20], 1):
            score = data.get('composite_momentum_score', 0)
            print(f"  {i:2d}. {ticker:6s}: {score:5.1f}")

        return 0

    # Load previous
    with open(prev_file, 'r') as f:
        previous = json.load(f)

    prev_signals = previous.get('signals', {})
    prev_regime = previous.get('metadata', {}).get('regime', 'UNKNOWN')

    print(f"Previous: {prev_file.name}")

    # Check regime change
    print("\n" + "=" * 70)
    if curr_regime != prev_regime:
        print(f"🚨 REGIME CHANGE: {prev_regime} → {curr_regime}")
    else:
        print(f"Regime: {curr_regime} (unchanged)")
    print("=" * 70)

    # Calculate changes
    entrants = []  # New to high momentum (>80)
    exits = []     # Dropped from high momentum
    large_up = []  # Large positive changes
    large_down = []  # Large negative changes

    for ticker, curr_data in curr_signals.items():
        curr_score = curr_data.get('composite_momentum_score', 50)

        if ticker in prev_signals:
            prev_score = prev_signals[ticker].get('composite_momentum_score', 50)
            delta = curr_score - prev_score

            # Large changes (>15 points)
            if delta > 15:
                large_up.append({
                    'ticker': ticker,
                    'prev': prev_score,
                    'curr': curr_score,
                    'delta': delta
                })
            elif delta < -15:
                large_down.append({
                    'ticker': ticker,
                    'prev': prev_score,
                    'curr': curr_score,
                    'delta': delta
                })

            # Entrants/exits from high momentum
            if curr_score > 80 and prev_score <= 80:
                entrants.append({'ticker': ticker, 'score': curr_score, 'prev': prev_score})
            elif curr_score <= 80 and prev_score > 80:
                exits.append({'ticker': ticker, 'score': curr_score, 'prev': prev_score})
        else:
            # New ticker
            if curr_score > 80:
                entrants.append({'ticker': ticker, 'score': curr_score, 'prev': None})

    # Display results
    if entrants:
        print(f"\n📈 NEW HIGH MOMENTUM ENTRANTS ({len(entrants)}):")
        entrants.sort(key=lambda x: x['score'], reverse=True)
        for e in entrants[:10]:
            prev_str = f"(was {e['prev']:.1f})" if e['prev'] else "(new)"
            print(f"   {e['ticker']:6s}: {e['score']:.1f} {prev_str}")

    if exits:
        print(f"\n📉 DROPPED FROM HIGH MOMENTUM ({len(exits)}):")
        exits.sort(key=lambda x: x['prev'], reverse=True)
        for e in exits[:10]:
            print(f"   {e['ticker']:6s}: {e['prev']:.1f} → {e['score']:.1f}")

    if large_up:
        print(f"\n🚀 LARGE MOMENTUM INCREASES ({len(large_up)}):")
        large_up.sort(key=lambda x: x['delta'], reverse=True)
        for c in large_up[:10]:
            print(f"   {c['ticker']:6s}: {c['prev']:.1f} → {c['curr']:.1f} (+{c['delta']:.1f})")

    if large_down:
        print(f"\n⚠️  LARGE MOMENTUM DECREASES ({len(large_down)}):")
        large_down.sort(key=lambda x: x['delta'])
        for c in large_down[:10]:
            print(f"   {c['ticker']:6s}: {c['prev']:.1f} → {c['curr']:.1f} ({c['delta']:.1f})")

    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"High momentum entrants: {len(entrants)}")
    print(f"High momentum exits: {len(exits)}")
    print(f"Large increases (>15): {len(large_up)}")
    print(f"Large decreases (<-15): {len(large_down)}")

    return 0


if __name__ == "__main__":
    current = sys.argv[1] if len(sys.argv) > 1 else None
    previous = sys.argv[2] if len(sys.argv) > 2 else None
    sys.exit(compare_weeks(current, previous))
