#!/usr/bin/env python3
"""
collect_corporate_catalysts.py - Corporate Catalyst Data Collector

Collects corporate catalyst events from various sources:
1. PDUFA dates (FDA approval decision dates) - from company filings/news
2. Earnings dates - from Yahoo Finance
3. Conference presentations - from major biotech conferences
4. Data readout expectations - from company guidance

Event types collected:
- FDA_PDUFA_DATE: FDA approval decision deadline
- FDA_ADCOM: Advisory committee meeting
- FDA_SUBMISSION: NDA/BLA submission
- EARNINGS_RELEASE: Quarterly earnings call
- CONFERENCE_PRESENTATION: Major conference presentation
- DATA_READOUT: Expected data readout

Usage:
    python collect_corporate_catalysts.py
    python collect_corporate_catalysts.py --universe production_data/universe.json
    python collect_corporate_catalysts.py --output production_data/corporate_catalysts.json
"""

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

# Check for yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# Major biotech conferences with typical dates
MAJOR_CONFERENCES = [
    {
        "name": "JPM Healthcare Conference",
        "abbrev": "JPM",
        "typical_month": 1,
        "typical_days": (8, 11),
        "tier": 1,
    },
    {
        "name": "ASCO Annual Meeting",
        "abbrev": "ASCO",
        "typical_month": 6,
        "typical_days": (1, 5),
        "tier": 1,
    },
    {
        "name": "ASH Annual Meeting",
        "abbrev": "ASH",
        "typical_month": 12,
        "typical_days": (7, 10),
        "tier": 1,
    },
    {
        "name": "AACR Annual Meeting",
        "abbrev": "AACR",
        "typical_month": 4,
        "typical_days": (5, 10),
        "tier": 1,
    },
    {
        "name": "ESMO Congress",
        "abbrev": "ESMO",
        "typical_month": 9,
        "typical_days": (20, 24),
        "tier": 1,
    },
    {
        "name": "AAN Annual Meeting",
        "abbrev": "AAN",
        "typical_month": 4,
        "typical_days": (15, 20),
        "tier": 2,
    },
    {
        "name": "EASL Congress",
        "abbrev": "EASL",
        "typical_month": 6,
        "typical_days": (15, 19),
        "tier": 2,
    },
    {
        "name": "ACR Convergence",
        "abbrev": "ACR",
        "typical_month": 11,
        "typical_days": (10, 15),
        "tier": 2,
    },
]


def get_earnings_date(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get next earnings date from Yahoo Finance.

    Returns dict with:
        - event_type: EARNINGS_RELEASE
        - event_date: Next earnings date
        - confidence: HIGH if confirmed, MED if estimated
    """
    try:
        stock = yf.Ticker(ticker)
        calendar = stock.calendar

        if calendar is None or calendar.empty:
            return None

        # Get earnings date
        earnings_date = None
        if 'Earnings Date' in calendar.index:
            earnings_dates = calendar.loc['Earnings Date']
            if isinstance(earnings_dates, (list, tuple)) and len(earnings_dates) > 0:
                earnings_date = earnings_dates[0]
            elif hasattr(earnings_dates, 'iloc'):
                earnings_date = earnings_dates.iloc[0] if len(earnings_dates) > 0 else None
            else:
                earnings_date = earnings_dates

        if earnings_date is None:
            return None

        # Convert to date string
        if hasattr(earnings_date, 'strftime'):
            event_date = earnings_date.strftime('%Y-%m-%d')
        elif hasattr(earnings_date, 'isoformat'):
            event_date = earnings_date.isoformat()[:10]
        else:
            event_date = str(earnings_date)[:10]

        return {
            'ticker': ticker,
            'event_type': 'EARNINGS_RELEASE',
            'event_date': event_date,
            'event_name': 'Quarterly Earnings Release',
            'confidence': 'MED',
            'source': 'yfinance',
        }

    except Exception:
        return None


def get_upcoming_conferences(as_of_date: date, lookahead_days: int = 180) -> List[Dict[str, Any]]:
    """
    Get list of upcoming major biotech conferences.

    Returns list of conference events within lookahead window.
    """
    conferences = []
    year = as_of_date.year

    for conf in MAJOR_CONFERENCES:
        # Check this year and next year
        for y in [year, year + 1]:
            try:
                start_day = conf['typical_days'][0]
                end_day = conf['typical_days'][1]
                conf_start = date(y, conf['typical_month'], start_day)
                conf_end = date(y, conf['typical_month'], end_day)

                # Check if within lookahead window
                days_until = (conf_start - as_of_date).days
                if 0 < days_until <= lookahead_days:
                    conferences.append({
                        'conference_name': conf['name'],
                        'conference_abbrev': conf['abbrev'],
                        'start_date': conf_start.isoformat(),
                        'end_date': conf_end.isoformat(),
                        'tier': conf['tier'],
                        'days_until': days_until,
                    })
            except ValueError:
                # Invalid date
                continue

    return sorted(conferences, key=lambda x: x['start_date'])


def load_pdufa_dates(pdufa_file: Optional[Path] = None) -> Dict[str, List[Dict]]:
    """
    Load PDUFA dates from file if available.

    Expected format:
    [
        {
            "ticker": "ACAD",
            "drug_name": "Pimavanserin",
            "indication": "Major Depressive Disorder",
            "pdufa_date": "2026-04-03",
            "submission_type": "sNDA",
            "confidence": "confirmed"
        }
    ]
    """
    if pdufa_file and pdufa_file.exists():
        with open(pdufa_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Index by ticker
            by_ticker = {}
            for item in data:
                ticker = item.get('ticker')
                if ticker:
                    if ticker not in by_ticker:
                        by_ticker[ticker] = []
                    by_ticker[ticker].append(item)
            return by_ticker
    return {}


def load_data_readouts(readouts_file: Optional[Path] = None) -> Dict[str, List[Dict]]:
    """
    Load expected data readout dates from file if available.

    Expected format:
    [
        {
            "ticker": "MRNA",
            "drug_name": "mRNA-1283",
            "indication": "RSV",
            "expected_date": "2026-Q2",
            "trial_phase": "Phase 3",
            "confidence": "guidance"
        }
    ]
    """
    if readouts_file and readouts_file.exists():
        with open(readouts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            by_ticker = {}
            for item in data:
                ticker = item.get('ticker')
                if ticker:
                    if ticker not in by_ticker:
                        by_ticker[ticker] = []
                    by_ticker[ticker].append(item)
            return by_ticker
    return {}


def collect_corporate_catalysts(
    universe_file: Path,
    output_file: Path,
    pdufa_file: Optional[Path] = None,
    readouts_file: Optional[Path] = None,
    lookahead_days: int = 180,
) -> int:
    """
    Collect corporate catalyst data for all tickers in universe.

    Args:
        universe_file: Path to universe.json
        output_file: Path to output corporate_catalysts.json
        pdufa_file: Optional path to PDUFA dates file
        readouts_file: Optional path to data readouts file
        lookahead_days: Days to look ahead for events

    Returns:
        Exit code (0 for success, 1 for error)
    """
    as_of_date = date.today()

    print("=" * 80)
    print("CORPORATE CATALYST DATA COLLECTION")
    print("=" * 80)
    print(f"Date: {as_of_date}")
    print(f"Lookahead: {lookahead_days} days")

    # Check yfinance
    if not YFINANCE_AVAILABLE:
        print("\n[ERROR] yfinance not installed")
        print("Install with: pip install yfinance")
        return 1

    print("[OK] yfinance library found")

    # Load universe
    if not universe_file.exists():
        print(f"\n[ERROR] Universe file not found: {universe_file}")
        return 1

    with open(universe_file, 'r', encoding='utf-8') as f:
        universe = json.load(f)

    # Extract tickers
    if isinstance(universe, list):
        tickers = [s.get('ticker') for s in universe if s.get('ticker')]
    elif isinstance(universe, dict):
        securities = universe.get('active_securities', universe.get('securities', []))
        tickers = [s.get('ticker') for s in securities if s.get('ticker')]
    else:
        print(f"\n[ERROR] Invalid universe format")
        return 1

    # Filter out benchmark ticker
    tickers = [t for t in tickers if not t.startswith('_')]

    print(f"\nUniverse: {len(tickers)} tickers")
    print(f"Output: {output_file}")

    # Load supplementary data
    pdufa_data = load_pdufa_dates(pdufa_file)
    readouts_data = load_data_readouts(readouts_file)

    if pdufa_data:
        print(f"PDUFA dates loaded: {sum(len(v) for v in pdufa_data.values())} events")
    if readouts_data:
        print(f"Data readouts loaded: {sum(len(v) for v in readouts_data.values())} events")

    # Get upcoming conferences
    conferences = get_upcoming_conferences(as_of_date, lookahead_days)
    print(f"Upcoming conferences: {len(conferences)}")
    for conf in conferences[:5]:
        print(f"  - {conf['conference_abbrev']}: {conf['start_date']} ({conf['days_until']} days)")

    print("\n" + "=" * 80)
    print("COLLECTING CORPORATE CATALYST DATA")
    print("=" * 80 + "\n")

    # Collect events
    all_events = []
    earnings_count = 0
    pdufa_count = 0
    readout_count = 0

    for i, ticker in enumerate(tickers):
        ticker_events = []

        # Get earnings date from Yahoo Finance
        earnings = get_earnings_date(ticker)
        if earnings:
            # Check if within lookahead window
            try:
                earnings_date = datetime.strptime(earnings['event_date'], '%Y-%m-%d').date()
                days_until = (earnings_date - as_of_date).days
                if 0 < days_until <= lookahead_days:
                    earnings['days_until'] = days_until
                    ticker_events.append(earnings)
                    earnings_count += 1
            except ValueError:
                pass

        # Add PDUFA dates if available
        if ticker in pdufa_data:
            for pdufa in pdufa_data[ticker]:
                try:
                    pdufa_date_str = pdufa.get('pdufa_date')
                    if pdufa_date_str:
                        pdufa_date = datetime.strptime(pdufa_date_str, '%Y-%m-%d').date()
                        days_until = (pdufa_date - as_of_date).days
                        if 0 < days_until <= lookahead_days:
                            ticker_events.append({
                                'ticker': ticker,
                                'event_type': 'FDA_PDUFA_DATE',
                                'event_date': pdufa_date_str,
                                'event_name': f"PDUFA: {pdufa.get('drug_name', 'Unknown')}",
                                'drug_name': pdufa.get('drug_name'),
                                'indication': pdufa.get('indication'),
                                'submission_type': pdufa.get('submission_type'),
                                'confidence': 'HIGH' if pdufa.get('confidence') == 'confirmed' else 'MED',
                                'source': 'pdufa_file',
                                'days_until': days_until,
                            })
                            pdufa_count += 1
                except ValueError:
                    pass

        # Add data readouts if available
        if ticker in readouts_data:
            for readout in readouts_data[ticker]:
                expected = readout.get('expected_date', '')
                # Handle quarterly dates (e.g., "2026-Q2")
                if '-Q' in expected:
                    year_q = expected.split('-Q')
                    if len(year_q) == 2:
                        try:
                            year = int(year_q[0])
                            quarter = int(year_q[1])
                            # Mid-quarter estimate
                            month = quarter * 3 - 1
                            readout_date = date(year, month, 15)
                            days_until = (readout_date - as_of_date).days
                            if 0 < days_until <= lookahead_days:
                                ticker_events.append({
                                    'ticker': ticker,
                                    'event_type': 'DATA_READOUT',
                                    'event_date': readout_date.isoformat(),
                                    'event_name': f"Data: {readout.get('drug_name', 'Unknown')}",
                                    'drug_name': readout.get('drug_name'),
                                    'indication': readout.get('indication'),
                                    'trial_phase': readout.get('trial_phase'),
                                    'confidence': 'LOW',  # Quarterly estimate
                                    'source': 'readouts_file',
                                    'days_until': days_until,
                                })
                                readout_count += 1
                        except (ValueError, TypeError):
                            pass
                else:
                    # Try direct date parsing
                    try:
                        readout_date = datetime.strptime(expected, '%Y-%m-%d').date()
                        days_until = (readout_date - as_of_date).days
                        if 0 < days_until <= lookahead_days:
                            ticker_events.append({
                                'ticker': ticker,
                                'event_type': 'DATA_READOUT',
                                'event_date': expected,
                                'event_name': f"Data: {readout.get('drug_name', 'Unknown')}",
                                'drug_name': readout.get('drug_name'),
                                'indication': readout.get('indication'),
                                'trial_phase': readout.get('trial_phase'),
                                'confidence': 'MED',
                                'source': 'readouts_file',
                                'days_until': days_until,
                            })
                            readout_count += 1
                    except ValueError:
                        pass

        all_events.extend(ticker_events)

        # Progress output
        event_str = ', '.join([e['event_type'] for e in ticker_events]) if ticker_events else 'None'
        print(f"[{i+1:>3}/{len(tickers)}] {ticker:<6} Events: {event_str[:50]}")

        # Rate limiting
        if (i + 1) % 10 == 0:
            time.sleep(0.3)

    print("\n" + "=" * 80)
    print("COLLECTION SUMMARY")
    print("=" * 80)
    print(f"Total tickers:       {len(tickers)}")
    print(f"Total events:        {len(all_events)}")
    print(f"  Earnings dates:    {earnings_count}")
    print(f"  PDUFA dates:       {pdufa_count}")
    print(f"  Data readouts:     {readout_count}")
    print(f"\nUpcoming conferences in window:")
    for conf in conferences:
        print(f"  - {conf['conference_abbrev']:<6} {conf['start_date']} - {conf['end_date']}")

    # Sort events by date
    all_events.sort(key=lambda x: (x.get('event_date', '9999-99-99'), x.get('ticker', '')))

    # Build output
    output_data = {
        'collection_date': as_of_date.isoformat(),
        'lookahead_days': lookahead_days,
        'summary': {
            'total_events': len(all_events),
            'earnings_events': earnings_count,
            'pdufa_events': pdufa_count,
            'readout_events': readout_count,
            'tickers_with_events': len(set(e['ticker'] for e in all_events)),
        },
        'upcoming_conferences': conferences,
        'events': all_events,
    }

    # Write output
    print(f"\nWriting to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"[OK] Saved {len(all_events)} events")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Collect corporate catalyst data (earnings, PDUFA dates, conferences)"
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=Path("production_data/universe.json"),
        help="Path to universe JSON file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("production_data/corporate_catalysts.json"),
        help="Output path for corporate catalyst data"
    )
    parser.add_argument(
        "--pdufa-file",
        type=Path,
        default=None,
        help="Optional path to PDUFA dates file"
    )
    parser.add_argument(
        "--readouts-file",
        type=Path,
        default=None,
        help="Optional path to data readouts file"
    )
    parser.add_argument(
        "--lookahead-days",
        type=int,
        default=180,
        help="Days to look ahead for events (default: 180)"
    )

    args = parser.parse_args()

    # Check for PDUFA file in default location
    pdufa_file = args.pdufa_file
    if pdufa_file is None:
        default_pdufa = Path("production_data/pdufa_dates.json")
        if default_pdufa.exists():
            pdufa_file = default_pdufa

    # Check for readouts file in default location
    readouts_file = args.readouts_file
    if readouts_file is None:
        default_readouts = Path("production_data/data_readouts.json")
        if default_readouts.exists():
            readouts_file = default_readouts

    return collect_corporate_catalysts(
        universe_file=args.universe,
        output_file=args.output,
        pdufa_file=pdufa_file,
        readouts_file=readouts_file,
        lookahead_days=args.lookahead_days,
    )


if __name__ == "__main__":
    sys.exit(main())
