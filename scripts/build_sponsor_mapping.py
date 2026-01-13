#!/usr/bin/env python3
"""
Build comprehensive ticker → sponsor name mapping for clinical trial searches.

Uses multiple data sources:
1. Yahoo Finance (primary) - longName field
2. SEC EDGAR (secondary) - company name from filings
3. Manual mappings (tertiary) - hardcoded for important tickers

Usage:
    python scripts/build_sponsor_mapping.py --universe outputs/rankings_FIXED.csv
    python scripts/build_sponsor_mapping.py --universe outputs/rankings_FIXED.csv --output data/ticker_to_sponsor.json

Author: Wake Robin Capital
Version: 2.0
"""

import argparse
import csv
import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Dict, List

# Try to import yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("Warning: yfinance not installed. Install with: pip install yfinance")


# =============================================================================
# Configuration
# =============================================================================

SEC_DATA_URL = "https://data.sec.gov"
SEC_WWW_URL = "https://www.sec.gov"
SEC_HEADERS = {
    'User-Agent': 'WakeRobinBiotechScreener/2.0 (research@wakerobincapital.com)',
    'Accept': 'application/json'
}

# Manual mappings for top biotech companies
# These are hardcoded to ensure accuracy for the most important tickers
MANUAL_MAPPINGS = {
    # Large Cap Biotech
    "VRTX": "Vertex Pharmaceuticals",
    "REGN": "Regeneron Pharmaceuticals",
    "GILD": "Gilead Sciences",
    "BIIB": "Biogen",
    "AMGN": "Amgen",
    "MRNA": "Moderna",
    "BNTX": "BioNTech",

    # Gene Therapy / Editing
    "CRSP": "CRISPR Therapeutics",
    "EDIT": "Editas Medicine",
    "NTLA": "Intellia Therapeutics",
    "BEAM": "Beam Therapeutics",
    "VERV": "Verve Therapeutics",
    "SGMO": "Sangamo Therapeutics",
    "BLUE": "bluebird bio",
    "RCKT": "Rocket Pharmaceuticals",
    "RGNX": "Regenxbio",
    "QURE": "uniQure",
    "VYGR": "Voyager Therapeutics",
    "TSHA": "Taysha Gene Therapies",

    # Oncology
    "EXEL": "Exelixis",
    "INCY": "Incyte",
    "KURA": "Kura Oncology",
    "IOVA": "Iovance Biotherapeutics",
    "ERAS": "Erasca",
    "ORIC": "ORIC Pharmaceuticals",
    "IDYA": "IDEAYA Biosciences",
    "MRSN": "Mersana Therapeutics",
    "RVMD": "Revolution Medicines",
    "NUVL": "Nuvalent",
    "CLDX": "Celldex Therapeutics",
    "ARVN": "Arvinas",
    "KYMR": "Kymera Therapeutics",

    # Rare Disease
    "ALNY": "Alnylam Pharmaceuticals",
    "BMRN": "BioMarin Pharmaceutical",
    "SRPT": "Sarepta Therapeutics",
    "RARE": "Ultragenyx Pharmaceutical",
    "FOLD": "Amicus Therapeutics",
    "PTCT": "PTC Therapeutics",
    "IONS": "Ionis Pharmaceuticals",
    "ARWR": "Arrowhead Pharmaceuticals",
    "INSM": "Insmed",
    "KRYS": "Krystal Biotech",

    # CNS / Neurology
    "NBIX": "Neurocrine Biosciences",
    "JAZZ": "Jazz Pharmaceuticals",
    "ALKS": "Alkermes",
    "AXSM": "Axsome Therapeutics",
    "ACAD": "ACADIA Pharmaceuticals",
    "PRAX": "Praxis Precision Medicines",
    "DNLI": "Denali Therapeutics",
    "PRTA": "Prothena",
    "NMRA": "Neumora Therapeutics",

    # Immunology
    "IMVT": "Immunovant",
    "APLS": "Apellis Pharmaceuticals",
    "AUPH": "Aurinia Pharmaceuticals",
    "ANAB": "AnaptysBio",
    "RCUS": "Arcus Biosciences",
    "XNCR": "Xencor",

    # Cardiovascular / Metabolic
    "CYTK": "Cytokinetics",
    "MDGL": "Madrigal Pharmaceuticals",
    "CRNX": "Crinetics Pharmaceuticals",

    # Cell Therapy
    "FATE": "Fate Therapeutics",
    "SANA": "Sana Biotechnology",
    "ACLX": "Arcellx",
    "VOR": "Vor Biopharma",
    "BCYC": "Bicycle Therapeutics",

    # Infectious Disease / Vaccines
    "NVAX": "Novavax",
    "DVAX": "Dynavax Technologies",
    "EBS": "Emergent BioSolutions",
    "AVIR": "Atea Pharmaceuticals",

    # Psychedelics / Mental Health
    "CMPS": "COMPASS Pathways",
    "ATAI": "ATAI Life Sciences",
    "MNMD": "Mind Medicine",

    # Platform / Tools
    "ILMN": "Illumina",
    "PACB": "Pacific Biosciences",
    "TXG": "10x Genomics",
    "TWST": "Twist Bioscience",
    "DNA": "Ginkgo Bioworks",
    "RXRX": "Recursion Pharmaceuticals",
    "ABSI": "Absci",

    # Other Important
    "BBIO": "BridgeBio Pharma",
    "ROIV": "Roivant Sciences",
    "HALO": "Halozyme Therapeutics",
    "UTHR": "United Therapeutics",
    "BCRX": "BioCryst Pharmaceuticals",
    "TVTX": "Travere Therapeutics",
    "TGTX": "TG Therapeutics",
    "OCUL": "Ocular Therapeutix",
    "GERN": "Geron",
    "IRWD": "Ironwood Pharmaceuticals",
    "CHRS": "Coherus BioSciences",
    "SUPN": "Supernus Pharmaceuticals",
    "MNKD": "MannKind",
    "ARDX": "Ardelyx",
    "AQST": "Aquestive Therapeutics",
    "AGIO": "Agios Pharmaceuticals",
    "SNDX": "Syndax Pharmaceuticals",
    "NUVB": "Nuvation Bio",
    "PGEN": "Precigen",
    "CRVS": "Corvus Pharmaceuticals",
    "SPRY": "Spruce Biosciences",
    "OLMA": "Olema Pharmaceuticals",
    "APGE": "Apogee Therapeutics",
    "PHAT": "Phathom Pharmaceuticals",
    "PCVX": "Vaxcyte",
    "SVRA": "Savara",
    "TKNO": "Alpha Teknova",
    "CGON": "CG Oncology",
    "ADPT": "Adaptive Biotechnologies",
    "XENE": "Xenon Pharmaceuticals",
    "VTYX": "Ventyx Biosciences",
    "PTGX": "Protagonist Therapeutics",
    "NRIX": "Nurix Therapeutics",
    "CTNM": "Contineum Therapeutics",
    "TNGX": "Tango Therapeutics",
    "ACRS": "Aclaris Therapeutics",
    "EPRX": "Eupraxia Pharmaceuticals",
    "TRVI": "Trevi Therapeutics",
    "ORKA": "Oruka Therapeutics",
    "SYRE": "Spyre Therapeutics",
    "DSGN": "Design Therapeutics",
    "ARQT": "Arcutis Biotherapeutics",
    "AVTR": "Avantor",
    "JANX": "Janux Therapeutics",
    "KYTX": "Kyverna Therapeutics",
    "CRBU": "Caribou Biosciences",
}


# =============================================================================
# Company Name Cleaning
# =============================================================================

def clean_company_name(name: str) -> str:
    """
    Clean company name by removing legal suffixes.

    Examples:
        "Vertex Pharmaceuticals Incorporated" → "Vertex Pharmaceuticals"
        "CRISPR Therapeutics AG" → "CRISPR Therapeutics"
        "Regeneron Pharmaceuticals, Inc." → "Regeneron Pharmaceuticals"
    """
    if not name:
        return ""

    # Remove common legal suffixes using regex
    suffixes = [
        r',?\s*Inc\.?$',
        r',?\s*Incorporated$',
        r',?\s*Corp\.?$',
        r',?\s*Corporation$',
        r',?\s*Ltd\.?$',
        r',?\s*Limited$',
        r',?\s*plc\.?$',
        r',?\s*PLC\.?$',
        r',?\s*N\.V\.?$',
        r',?\s*AG$',
        r',?\s*S\.A\.?$',
        r',?\s*SA$',
        r',?\s*GmbH$',
        r',?\s*SE$',
        r',?\s*LLC$',
        r',?\s*L\.L\.C\.?$',
        r',?\s*LP$',
        r',?\s*L\.P\.?$',
        r',?\s*Co\.?$',
        r',?\s*Company$',
    ]

    cleaned = name.strip()
    for suffix in suffixes:
        cleaned = re.sub(suffix, '', cleaned, flags=re.IGNORECASE)

    # Remove trailing commas and whitespace
    cleaned = cleaned.rstrip(',').strip()

    return cleaned


# =============================================================================
# Data Fetchers
# =============================================================================

def get_company_name_yfinance(ticker: str) -> Optional[str]:
    """Get company name from Yahoo Finance."""
    if not HAS_YFINANCE:
        return None

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Try longName first, then shortName
        long_name = info.get('longName', '')
        short_name = info.get('shortName', '')

        name = long_name or short_name

        if name:
            return clean_company_name(name)

    except Exception as e:
        pass  # Silently fail, will try other sources

    return None


def get_company_name_sec(ticker: str, ticker_to_cik: Dict[str, str]) -> Optional[str]:
    """Get company name from SEC EDGAR filings."""
    cik = ticker_to_cik.get(ticker)
    if not cik:
        return None

    # Pad CIK to 10 digits
    cik_padded = str(cik).zfill(10)
    url = f"{SEC_DATA_URL}/submissions/CIK{cik_padded}.json"

    try:
        req = urllib.request.Request(url, headers=SEC_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
            name = data.get('name', '')
            if name:
                return clean_company_name(name)
    except Exception:
        pass

    return None


def load_sec_cik_mapping() -> Dict[str, str]:
    """Load ticker to CIK mapping from SEC."""
    cache_file = Path("data/cache/sec_edgar/ticker_to_cik.json")

    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)

    # Fetch from SEC if not cached
    url = f"{SEC_WWW_URL}/files/company_tickers.json"

    try:
        req = urllib.request.Request(url, headers=SEC_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))

            mapping = {}
            for entry in data.values():
                ticker = entry.get('ticker', '').upper()
                cik = str(entry.get('cik_str', ''))
                if ticker and cik:
                    mapping[ticker] = cik

            # Cache it
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(mapping, f)

            return mapping

    except Exception as e:
        print(f"Warning: Could not fetch CIK mapping from SEC: {e}")
        return {}


# =============================================================================
# Main Builder
# =============================================================================

def load_universe(universe_file: str) -> List[str]:
    """Load tickers from universe file."""
    tickers = []

    with open(universe_file, 'r', encoding='utf-8', errors='ignore') as f:
        # Try to detect format
        first_line = f.readline()
        f.seek(0)

        if ',' in first_line:
            # CSV format
            reader = csv.reader(f)
            header = next(reader, None)

            # Find ticker column
            ticker_col = 0
            if header:
                for i, col in enumerate(header):
                    if col.lower() in ('ticker', 'symbol'):
                        ticker_col = i
                        break
                    # Also check if second column looks like tickers
                    if i == 1 and col.isupper() and len(col) <= 6:
                        ticker_col = 1
                        f.seek(0)  # Re-read without skipping header
                        break

            for row in reader:
                if len(row) > ticker_col:
                    ticker = row[ticker_col].strip().upper()
                    if ticker and ticker != 'TICKER' and len(ticker) <= 6 and ticker.isalpha():
                        tickers.append(ticker)
        else:
            # Plain text format
            for line in f:
                ticker = line.strip().upper()
                if ticker and len(ticker) <= 6 and ticker.isalpha():
                    tickers.append(ticker)

    return list(set(tickers))  # Remove duplicates


def build_sponsor_mapping(
    universe_file: str,
    output_file: str = "data/ticker_to_sponsor.json",
    use_yfinance: bool = True,
    use_sec: bool = True
) -> Dict[str, str]:
    """
    Build comprehensive ticker to sponsor mapping.

    Sources (in order of priority):
    1. Manual mappings (hardcoded for accuracy)
    2. Yahoo Finance (longName field)
    3. SEC EDGAR (company name from filings)
    """

    # Load universe
    tickers = load_universe(universe_file)
    print(f"Loaded {len(tickers)} tickers from {universe_file}")

    # Load existing mapping
    output_path = Path(output_file)
    if output_path.exists():
        with open(output_path, 'r') as f:
            mapping = json.load(f)
        print(f"Loaded {len(mapping)} existing mappings")
    else:
        mapping = {}

    # Start with manual mappings
    for ticker, name in MANUAL_MAPPINGS.items():
        if ticker in tickers or ticker in mapping:
            mapping[ticker] = name

    print(f"After manual mappings: {len(mapping)}")

    # Load SEC CIK mapping for fallback
    ticker_to_cik = {}
    if use_sec:
        print("Loading SEC CIK mapping...")
        ticker_to_cik = load_sec_cik_mapping()
        print(f"  Loaded {len(ticker_to_cik)} CIK mappings")

    # Process remaining tickers
    to_fetch = [t for t in tickers if t not in mapping]
    print(f"Need to fetch: {len(to_fetch)} tickers")

    success_count = 0
    fail_count = 0

    for i, ticker in enumerate(to_fetch, 1):
        print(f"[{i}/{len(to_fetch)}] {ticker}...", end=" ", flush=True)

        name = None
        source = ""

        # Try Yahoo Finance first
        if use_yfinance and HAS_YFINANCE:
            name = get_company_name_yfinance(ticker)
            if name:
                source = "YF"

        # Fall back to SEC EDGAR
        if not name and use_sec:
            name = get_company_name_sec(ticker, ticker_to_cik)
            if name:
                source = "SEC"

        if name:
            mapping[ticker] = name
            success_count += 1
            print(f"✓ {name} [{source}]")
        else:
            fail_count += 1
            print("✗ Not found")

        # Rate limiting
        time.sleep(0.3)

    # Save mapping
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2, sort_keys=True)

    # Report statistics
    coverage = len(mapping) / len(tickers) * 100 if tickers else 0

    print()
    print("=" * 60)
    print("MAPPING COMPLETE")
    print("=" * 60)
    print(f"Total tickers:      {len(tickers)}")
    print(f"Successfully mapped: {len(mapping)} ({coverage:.1f}%)")
    print(f"  Manual mappings:   {len([t for t in mapping if t in MANUAL_MAPPINGS])}")
    print(f"  Fetched new:       {success_count}")
    print(f"  Failed to find:    {fail_count}")
    print(f"Output file:        {output_path}")
    print()

    if coverage >= 70:
        print(f"✓ TARGET ACHIEVED: {coverage:.1f}% coverage (target: 70%)")
    else:
        needed = int(0.7 * len(tickers)) - len(mapping)
        print(f"✗ Below target: {coverage:.1f}% coverage (target: 70%)")
        print(f"  Need {needed} more mappings")

    return mapping


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build ticker to sponsor mapping for clinical trial searches'
    )
    parser.add_argument(
        '--universe',
        required=True,
        help='Path to universe file (CSV or text)'
    )
    parser.add_argument(
        '--output',
        default='data/ticker_to_sponsor.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--no-yfinance',
        action='store_true',
        help='Skip Yahoo Finance lookups'
    )
    parser.add_argument(
        '--no-sec',
        action='store_true',
        help='Skip SEC EDGAR lookups'
    )

    args = parser.parse_args()

    build_sponsor_mapping(
        universe_file=args.universe,
        output_file=args.output,
        use_yfinance=not args.no_yfinance,
        use_sec=not args.no_sec
    )


if __name__ == '__main__':
    main()
