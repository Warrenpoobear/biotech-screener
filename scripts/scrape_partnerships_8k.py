#!/usr/bin/env python3
"""
Scrape SEC 8-K filings for partnership announcements.

Usage:
    python scripts/scrape_partnerships_8k.py --tickers MRNA ALNY BEAM
    python scripts/scrape_partnerships_8k.py --universe production_data/universe.json
    python scripts/scrape_partnerships_8k.py --universe production_data/universe.json --output production_data/partnerships_new.json
"""

import argparse
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests

# SEC EDGAR API settings
SEC_BASE_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_FILING_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
USER_AGENT = "BiotechScreener Research contact@example.com"

# Partnership keywords to search for in 8-K filings
PARTNERSHIP_KEYWORDS = [
    "collaboration agreement",
    "license agreement",
    "licensing agreement",
    "partnership agreement",
    "co-development agreement",
    "joint venture",
    "strategic alliance",
    "exclusive license",
    "option agreement",
    "royalty agreement",
    "milestone payment",
    "upfront payment",
]

# Major pharma partners (for validation)
MAJOR_PARTNERS = [
    "pfizer", "merck", "roche", "novartis", "johnson", "j&j", "abbvie",
    "bristol-myers", "bms", "astrazeneca", "sanofi", "gsk", "glaxosmithkline",
    "eli lilly", "lilly", "gilead", "amgen", "regeneron", "biogen", "vertex",
    "takeda", "bayer", "boehringer", "novo nordisk", "astellas", "daiichi",
]


def get_cik_for_ticker(ticker: str) -> Optional[str]:
    """Get CIK number for a ticker symbol using SEC company tickers JSON."""
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        data = resp.json()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry.get("cik_str", ""))
                return cik.zfill(10)  # Pad to 10 digits
        return None
    except Exception as e:
        print(f"  Warning: Could not get CIK for {ticker}: {e}")
        return None


def fetch_8k_filings(cik: str, ticker: str, lookback_days: int = 730) -> List[Dict]:
    """Fetch 8-K filings for a company from SEC EDGAR."""
    filings = []

    try:
        # Use SEC submissions API
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        descriptions = recent.get("primaryDocument", [])

        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        for i, form in enumerate(forms):
            if form == "8-K" and dates[i] >= cutoff_date:
                filings.append({
                    "ticker": ticker,
                    "cik": cik,
                    "filing_date": dates[i],
                    "accession": accessions[i].replace("-", ""),
                    "document": descriptions[i],
                })

        return filings
    except Exception as e:
        print(f"  Warning: Could not fetch 8-K filings for {ticker}: {e}")
        return []


def fetch_filing_content(cik: str, accession: str, document: str) -> Optional[str]:
    """Fetch the actual content of an 8-K filing."""
    try:
        # Build URL to filing document
        url = f"{SEC_ARCHIVES_URL}/{cik}/{accession}/{document}"
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        content = resp.text

        # Strip HTML tags to get plain text
        # Simple regex-based HTML stripping
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<[^>]+>', ' ', content)
        content = re.sub(r'&nbsp;', ' ', content)
        content = re.sub(r'&amp;', '&', content)
        content = re.sub(r'&#160;', ' ', content)
        content = re.sub(r'\s+', ' ', content)

        return content.strip()
    except Exception as e:
        return None


def extract_partnership_info(content: str, filing: Dict) -> Optional[Dict]:
    """Extract partnership information from 8-K filing content."""
    content_lower = content.lower()

    # Check if this filing mentions partnership keywords
    matched_keywords = [kw for kw in PARTNERSHIP_KEYWORDS if kw in content_lower]
    if not matched_keywords:
        return None

    # Check for major partner names
    partner_found = None
    for partner in MAJOR_PARTNERS:
        if partner in content_lower:
            # Capitalize properly
            partner_found = partner.title()
            if partner == "j&j":
                partner_found = "Johnson & Johnson"
            elif partner == "bms":
                partner_found = "Bristol-Myers Squibb"
            elif partner == "gsk":
                partner_found = "GlaxoSmithKline"
            break

    if not partner_found:
        # Try to find partner from "agreement with" pattern
        patterns = [
            r"agreement with ([A-Z][A-Za-z\s&]+(?:Inc|Corp|Ltd|LLC|Company|Pharmaceuticals)?)",
            r"partnership with ([A-Z][A-Za-z\s&]+(?:Inc|Corp|Ltd|LLC|Company|Pharmaceuticals)?)",
            r"collaboration with ([A-Z][A-Za-z\s&]+(?:Inc|Corp|Ltd|LLC|Company|Pharmaceuticals)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                partner_found = match.group(1).strip()
                break

    if not partner_found:
        return None  # No identifiable partner

    # Determine deal type
    deal_type = "collaboration"  # default
    if "exclusive license" in content_lower or "exclusive licensing" in content_lower:
        deal_type = "exclusive_license"
    elif "co-development" in content_lower or "co-develop" in content_lower:
        deal_type = "co_development"
    elif "option agreement" in content_lower or "option to" in content_lower:
        deal_type = "option"
    elif "royalty" in content_lower:
        deal_type = "royalty"

    # Try to extract payment amounts
    upfront = None
    total_value = None

    # Look for dollar amounts
    amount_patterns = [
        r"\$(\d+(?:\.\d+)?)\s*(?:million|M)\s*(?:upfront|up-front|up front)",
        r"upfront.*?\$(\d+(?:\.\d+)?)\s*(?:million|M)",
        r"\$(\d+(?:\.\d+)?)\s*(?:billion|B).*?(?:total|potential)",
        r"(?:total|potential).*?\$(\d+(?:\.\d+)?)\s*(?:billion|B)",
    ]

    for pattern in amount_patterns:
        match = re.search(pattern, content_lower)
        if match:
            amount = float(match.group(1))
            if "billion" in pattern or "B" in pattern:
                amount *= 1000
            if "upfront" in pattern:
                upfront = amount
            else:
                total_value = amount

    # Try to find indication/therapeutic area
    indication = None
    indication_keywords = {
        "oncology": ["cancer", "tumor", "oncology", "carcinoma", "melanoma", "leukemia", "lymphoma"],
        "neurology": ["neurolog", "alzheimer", "parkinson", "brain", "cns", "central nervous"],
        "immunology": ["immun", "autoimmune", "inflammatory", "rheumatoid"],
        "rare disease": ["rare disease", "orphan", "genetic disorder"],
        "infectious": ["infectious", "viral", "bacterial", "vaccine", "covid", "influenza"],
        "cardiovascular": ["cardiovascular", "heart", "cardiac"],
        "metabolic": ["metabolic", "diabetes", "obesity"],
    }

    for ind, keywords in indication_keywords.items():
        if any(kw in content_lower for kw in keywords):
            indication = ind
            break

    return {
        "ticker": filing["ticker"],
        "partner_name": partner_found,
        "deal_type": deal_type,
        "announcement_date": filing["filing_date"],
        "upfront_payment": upfront,
        "total_potential_value": total_value,
        "indication": indication,
        "asset_name": None,  # Hard to extract reliably
        "is_active": True,
        "source": "SEC 8-K",
        "accession": filing["accession"],
        "keywords_matched": matched_keywords[:3],  # First 3 matched keywords
    }


def scrape_partnerships_for_ticker(ticker: str, lookback_days: int = 730) -> List[Dict]:
    """Scrape all partnership announcements for a single ticker."""
    partnerships = []

    print(f"  Processing {ticker}...")

    # Get CIK
    cik = get_cik_for_ticker(ticker)
    if not cik:
        print(f"    No CIK found for {ticker}")
        return []

    # Fetch 8-K filings
    filings = fetch_8k_filings(cik, ticker, lookback_days)
    print(f"    Found {len(filings)} 8-K filings in last {lookback_days} days")

    # Process each filing
    for filing in filings:
        time.sleep(0.2)  # Rate limiting

        content = fetch_filing_content(cik, filing["accession"], filing["document"])
        if not content:
            continue

        partnership = extract_partnership_info(content, filing)
        if partnership:
            partnerships.append(partnership)
            print(f"    Found partnership: {partnership['partner_name']} ({partnership['deal_type']})")

    return partnerships


def load_universe(universe_path: str) -> List[str]:
    """Load tickers from universe file."""
    with open(universe_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            return [d.get("ticker", "").upper() for d in data if d.get("ticker")]
        return [t.upper() for t in data if isinstance(t, str)]
    elif isinstance(data, dict):
        return list(data.keys())
    return []


def merge_with_existing(new_partnerships: List[Dict], existing_path: str) -> List[Dict]:
    """Merge new partnerships with existing file, avoiding duplicates."""
    existing = []
    if Path(existing_path).exists():
        with open(existing_path, 'r') as f:
            data = json.load(f)
            existing = data.get("partnerships", []) if isinstance(data, dict) else data

    # Create lookup key for deduplication
    existing_keys = set()
    for p in existing:
        key = (p.get("ticker"), p.get("partner_name", "").lower(), p.get("announcement_date"))
        existing_keys.add(key)

    # Add new partnerships that don't exist
    added = 0
    for p in new_partnerships:
        key = (p.get("ticker"), p.get("partner_name", "").lower(), p.get("announcement_date"))
        if key not in existing_keys:
            existing.append(p)
            existing_keys.add(key)
            added += 1

    print(f"\nMerged: {added} new partnerships added, {len(existing)} total")
    return existing


def main():
    parser = argparse.ArgumentParser(description="Scrape SEC 8-K filings for partnership announcements")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to process")
    parser.add_argument("--universe", help="Path to universe.json file")
    parser.add_argument("--output", default="production_data/partnerships_scraped.json", help="Output file path")
    parser.add_argument("--lookback-days", type=int, default=730, help="Days to look back (default: 730 = 2 years)")
    parser.add_argument("--merge", help="Merge with existing partnerships file")
    parser.add_argument("--limit", type=int, help="Limit number of tickers to process")

    args = parser.parse_args()

    # Get tickers to process
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.universe:
        tickers = load_universe(args.universe)
        print(f"Loaded {len(tickers)} tickers from universe")
    else:
        print("Error: Must specify --tickers or --universe")
        return 1

    if args.limit:
        tickers = tickers[:args.limit]

    print(f"Processing {len(tickers)} tickers...")
    print(f"Looking back {args.lookback_days} days")
    print()

    # Scrape partnerships
    all_partnerships = []
    for ticker in tickers:
        try:
            partnerships = scrape_partnerships_for_ticker(ticker, args.lookback_days)
            all_partnerships.extend(partnerships)
        except Exception as e:
            print(f"  Error processing {ticker}: {e}")

        time.sleep(0.5)  # Rate limiting between tickers

    print(f"\nFound {len(all_partnerships)} total partnerships")

    # Merge with existing if specified
    if args.merge:
        all_partnerships = merge_with_existing(all_partnerships, args.merge)

    # Save output
    output = {
        "metadata": {
            "description": "Biotech partnership data scraped from SEC 8-K filings",
            "source": "SEC EDGAR 8-K filings",
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "lookback_days": args.lookback_days,
            "tickers_processed": len(tickers),
        },
        "partnerships": all_partnerships,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(all_partnerships)} partnerships to {output_path}")

    # Summary by partner
    if all_partnerships:
        print("\nTop partners found:")
        partner_counts = {}
        for p in all_partnerships:
            partner = p.get("partner_name", "Unknown")
            partner_counts[partner] = partner_counts.get(partner, 0) + 1

        for partner, count in sorted(partner_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {partner}: {count}")

    return 0


if __name__ == "__main__":
    exit(main())
