#!/usr/bin/env python3
"""
Build ticker → company name mapping from SEC EDGAR data.

This mapping is used by ClinicalTrials.gov fetcher to search
for trials by sponsor name instead of ticker symbol.
"""

import json
import time
from pathlib import Path
import urllib.request
import urllib.error

SEC_DATA_URL = "https://data.sec.gov"
SEC_WWW_URL = "https://www.sec.gov"
SEC_HEADERS = {
    'User-Agent': 'BiotechScreener/1.0 (contact@example.com)',
    'Accept': 'application/json'
}

def get_company_name_from_sec(cik: str) -> str:
    """Get company name from SEC submissions."""
    url = f"{SEC_DATA_URL}/submissions/CIK{cik}.json"

    try:
        req = urllib.request.Request(url, headers=SEC_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            return data.get('name', '')
    except Exception as e:
        return ''


def build_sponsor_mapping(tickers: list, output_file: str = "data/ticker_to_sponsor.json"):
    """Build ticker to sponsor name mapping."""

    # Load existing CIK cache
    cik_cache = Path("data/cache/sec_edgar/ticker_to_cik.json")
    if cik_cache.exists():
        with open(cik_cache) as f:
            ticker_to_cik = json.load(f)
    else:
        print("Error: No CIK cache found. Run SEC EDGAR fetcher first.")
        return

    # Load existing mapping if present
    output_path = Path(output_file)
    if output_path.exists():
        with open(output_path) as f:
            mapping = json.load(f)
    else:
        mapping = {}

    # Add known mappings for common biotech companies
    known_mappings = {
        "VRTX": "Vertex Pharmaceuticals",
        "REGN": "Regeneron Pharmaceuticals",
        "ALNY": "Alnylam Pharmaceuticals",
        "BMRN": "BioMarin Pharmaceutical",
        "BIIB": "Biogen",
        "GILD": "Gilead Sciences",
        "AMGN": "Amgen",
        "CRSP": "CRISPR Therapeutics",
        "EDIT": "Editas Medicine",
        "NTLA": "Intellia Therapeutics",
        "BEAM": "Beam Therapeutics",
        "MRNA": "Moderna",
        "BNTX": "BioNTech",
        "NVAX": "Novavax",
        "IONS": "Ionis Pharmaceuticals",
        "SRPT": "Sarepta Therapeutics",
        "RARE": "Ultragenyx Pharmaceutical",
        "FOLD": "Amicus Therapeutics",
        "INCY": "Incyte",
        "EXEL": "Exelixis",
        "JAZZ": "Jazz Pharmaceuticals",
        "NBIX": "Neurocrine Biosciences",
        "UTHR": "United Therapeutics",
        "ALKS": "Alkermes",
        "HALO": "Halozyme Therapeutics",
        "PTCT": "PTC Therapeutics",
        "BCRX": "BioCryst Pharmaceuticals",
        "ARVN": "Arvinas",
        "ARWR": "Arrowhead Pharmaceuticals",
        "FATE": "Fate Therapeutics",
        "SANA": "Sana Biotechnology",
        "PRTA": "Prothena",
        "DNLI": "Denali Therapeutics",
        "TVTX": "Travere Therapeutics",
        "ROIV": "Roivant Sciences",
        "ERAS": "Erasca",
        "KURA": "Kura Oncology",
        "IOVA": "Iovance Biotherapeutics",
        "MRSN": "Mersana Therapeutics",
        "TSHA": "Taysha Gene Therapies",
        "RCKT": "Rocket Pharmaceuticals",
        "SGMO": "Sangamo Therapeutics",
        "VOR": "Vor Biopharma",
        "PSNL": "Personalis",
        "RGNX": "Regenxbio",
        "VYGR": "Voyager Therapeutics",
        "QURE": "uniQure",
        "ANAB": "AnaptysBio",
        "BCYC": "Bicycle Therapeutics",
        "XNCR": "Xencor",
        "RCUS": "Arcus Biosciences",
        "IDYA": "IDEAYA Biosciences",
        "PRAX": "Praxis Precision Medicines",
        "CMPS": "COMPASS Pathways",
        "ATAI": "ATAI Life Sciences",
        "MNMD": "Mind Medicine",
        "AXSM": "Axsome Therapeutics",
        "CRNX": "Crinetics Pharmaceuticals",
        "CYTK": "Cytokinetics",
        "MDGL": "Madrigal Pharmaceuticals",
        "KRYS": "Krystal Biotech",
        "INSM": "Insmed",
        "AGIO": "Agios Pharmaceuticals",
        "ORIC": "ORIC Pharmaceuticals",
        "KYMR": "Kymera Therapeutics",
        "NUVB": "Nuvation Bio",
        "PGEN": "Precigen",
        "CRVS": "Corvus Pharmaceuticals",
        "ABSI": "Absci",
        "SPRY": "Spruce Biosciences",
        "SNDX": "Syndax Pharmaceuticals",
        "RVMD": "Revolution Medicines",
        "NUVL": "Nuvalent",
        "CLDX": "Celldex Therapeutics",
        "OLMA": "Olema Pharmaceuticals",
        "ACLX": "Arcellx",
        "APLS": "Apellis Pharmaceuticals",
        "IMVT": "Immunovant",
        "BBIO": "BridgeBio Pharma",
        "AVIR": "Atea Pharmaceuticals",
        "DVAX": "Dynavax Technologies",
        "ACAD": "ACADIA Pharmaceuticals",
        "TGTX": "TG Therapeutics",
        "OCUL": "Ocular Therapeutix",
        "GERN": "Geron",
        "IRWD": "Ironwood Pharmaceuticals",
        "CHRS": "Coherus BioSciences",
        "HRTX": "Heron Therapeutics",
        "EBS": "Emergent BioSolutions",
        "SUPN": "Supernus Pharmaceuticals",
        "MNKD": "MannKind",
        "XERS": "Xeris Biopharma",
        "AQST": "Aquestive Therapeutics",
        "AUPH": "Aurinia Pharmaceuticals",
        "ARDX": "Ardelyx",
        "ARQT": "Arcus Therapeutics",
        "PACB": "Pacific Biosciences",
        "ILMN": "Illumina",
        "TXG": "10x Genomics",
        "TWST": "Twist Bioscience",
        "DNA": "Ginkgo Bioworks",
        "RXRX": "Recursion Pharmaceuticals",
    }

    # Start with known mappings
    mapping.update(known_mappings)

    # Fetch from SEC for remaining tickers
    to_fetch = [t for t in tickers if t not in mapping]
    print(f"Already mapped: {len(mapping)}")
    print(f"Need to fetch: {len(to_fetch)}")

    for i, ticker in enumerate(to_fetch, 1):
        cik = ticker_to_cik.get(ticker)
        if not cik:
            continue

        print(f"  [{i}/{len(to_fetch)}] {ticker}...", end=" ")
        name = get_company_name_from_sec(cik)

        if name:
            # Clean up the name for ClinicalTrials.gov search
            # Remove Inc., Corp., Ltd., etc.
            clean_name = name
            for suffix in [', Inc.', ' Inc.', ', Corp.', ' Corp.', ', Ltd.', ' Ltd.',
                          ' LLC', ', LLC', ' plc', ', plc', ' PLC', ', PLC',
                          ' SE', ', SE', ' N.V.', ', N.V.', ' AG', ', AG']:
                clean_name = clean_name.replace(suffix, '')

            mapping[ticker] = clean_name.strip()
            print(f"{clean_name}")
        else:
            print("No name found")

        time.sleep(0.1)  # Rate limit

    # Save mapping
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2, sort_keys=True)

    print(f"\nSaved {len(mapping)} mappings to {output_file}")
    return mapping


if __name__ == "__main__":
    import sys

    # Load tickers from rankings file
    rankings_file = "outputs/rankings_FIXED.csv"
    tickers = []

    with open(rankings_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.split(',')
            if len(parts) >= 2:
                ticker = parts[1].strip().strip('"')
                if ticker and ticker != 'Ticker' and len(ticker) <= 6 and ticker.isalpha():
                    tickers.append(ticker.upper())

    print(f"Loaded {len(tickers)} tickers from {rankings_file}")
    build_sponsor_mapping(tickers)
