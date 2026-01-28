#!/usr/bin/env python3
"""
collect_adr_data.py - ADR (American Depositary Receipt) Data Collector

Enriches universe with ADR and foreign company identification data.

Data sources:
1. Yahoo Finance - country, financialCurrency, city
2. SEC EDGAR - stateOfIncorporation, filing types (20-F/6-K vs 10-K/10-Q)

Fields collected:
- is_adr: Boolean - True if foreign company trading on US exchange
- is_foreign_private_issuer: Boolean - True if files 20-F/6-K instead of 10-K/10-Q
- country_of_origin: Country where company is headquartered
- reporting_currency: Currency used for financial statements
- headquarters_city: City of headquarters
- incorporation_country: Country/state of incorporation (from SEC)
- adr_detection_method: How ADR status was determined

Usage:
    python collect_adr_data.py
    python collect_adr_data.py --universe production_data/universe.json
    python collect_adr_data.py --output production_data/adr_data.json
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Any

# Check for yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# Country code mappings for SEC EDGAR stateOfIncorporation
SEC_COUNTRY_CODES = {
    "A0": "Alberta, Canada",
    "A1": "British Columbia, Canada",
    "A2": "Manitoba, Canada",
    "A3": "New Brunswick, Canada",
    "A4": "Newfoundland, Canada",
    "A5": "Nova Scotia, Canada",
    "A6": "Ontario, Canada",
    "A7": "Prince Edward Island, Canada",
    "A8": "Quebec, Canada",
    "A9": "Saskatchewan, Canada",
    "B0": "Yukon, Canada",
    "B2": "Afghanistan",
    "B3": "Albania",
    "B4": "Algeria",
    "B5": "American Samoa",
    "B6": "Andorra",
    "B7": "Angola",
    "B8": "Anguilla",
    "B9": "Antarctica",
    "C1": "Antigua and Barbuda",
    "C3": "Argentina",
    "C4": "Armenia",
    "C5": "Aruba",
    "C6": "Australia",
    "C7": "Austria",
    "C8": "Azerbaijan",
    "C9": "Bahamas",
    "D0": "Bahrain",
    "D1": "Bangladesh",
    "D2": "Barbados",
    "D3": "Belarus",
    "D4": "Belgium",
    "D5": "Belize",
    "D6": "Benin",
    "D7": "Bermuda",
    "D8": "Bhutan",
    "D9": "Bolivia",
    "E0": "Bosnia and Herzegovina",
    "E1": "Botswana",
    "E2": "Bouvet Island",
    "E3": "Brazil",
    "E4": "British Indian Ocean Territory",
    "E5": "Brunei Darussalam",
    "E6": "Bulgaria",
    "E8": "Burkina Faso",
    "E9": "Burundi",
    "F0": "Cambodia",
    "F1": "Cameroon",
    "F2": "Canada",
    "F3": "Cape Verde",
    "F4": "Cayman Islands",
    "F6": "Central African Republic",
    "F7": "Chad",
    "F8": "Chile",
    "F9": "China",
    "G0": "Christmas Island",
    "G1": "Cocos (Keeling) Islands",
    "G2": "Colombia",
    "G3": "Comoros",
    "G4": "Congo",
    "G5": "Congo, Democratic Republic",
    "G6": "Cook Islands",
    "G7": "Costa Rica",
    "G8": "Cote d'Ivoire",
    "G9": "Croatia",
    "H0": "Cuba",
    "H1": "Cyprus",
    "H2": "Czech Republic",
    "H3": "Denmark",
    "H4": "Djibouti",
    "H5": "Dominica",
    "H6": "Dominican Republic",
    "H7": "Ecuador",
    "H8": "Egypt",
    "H9": "El Salvador",
    "I0": "France",
    "I1": "Equatorial Guinea",
    "I2": "Eritrea",
    "I3": "Estonia",
    "I4": "Ethiopia",
    "I5": "Falkland Islands",
    "I6": "Faroe Islands",
    "I7": "Fiji",
    "I8": "Finland",
    "I9": "Gabon",
    "J0": "Gambia",
    "J1": "Georgia",
    "J2": "Germany",
    "J3": "Ghana",
    "J4": "Gibraltar",
    "J5": "Greece",
    "J6": "Greenland",
    "J7": "Grenada",
    "J8": "Guadeloupe",
    "J9": "Guam",
    "K0": "Guatemala",
    "K1": "Guinea",
    "K2": "Guinea-Bissau",
    "K3": "Guyana",
    "K4": "Haiti",
    "K5": "Heard and McDonald Islands",
    "K6": "Holy See",
    "K7": "Honduras",
    "K8": "Hong Kong",
    "K9": "Hungary",
    "L0": "Iceland",
    "L1": "India",
    "L2": "Indonesia",
    "L3": "Iran",
    "L4": "Iraq",
    "L5": "Ireland",
    "L6": "Israel",
    "L7": "Italy",
    "L8": "Jamaica",
    "L9": "Japan",
    "M0": "Jordan",
    "M1": "Kazakhstan",
    "M2": "Kenya",
    "M3": "Kiribati",
    "M4": "Korea, North",
    "M5": "Korea, South",
    "M6": "Kuwait",
    "M7": "Kyrgyzstan",
    "M8": "Laos",
    "M9": "Latvia",
    "N0": "Lebanon",
    "N1": "Lesotho",
    "N2": "Liberia",
    "N3": "Libya",
    "N4": "Liechtenstein",
    "N5": "Lithuania",
    "N6": "Luxembourg",
    "N7": "Macau",
    "N8": "Macedonia",
    "N9": "Madagascar",
    "O0": "Malawi",
    "O1": "Malaysia",
    "O2": "Maldives",
    "O3": "Mali",
    "O4": "Malta",
    "O5": "Marshall Islands",
    "O6": "Martinique",
    "O7": "Mauritania",
    "O8": "Mauritius",
    "O9": "Mayotte",
    "P0": "Mexico",
    "P1": "Micronesia",
    "P2": "Moldova",
    "P3": "Monaco",
    "P4": "Mongolia",
    "P5": "Montserrat",
    "P6": "Morocco",
    "P7": "Mozambique",
    "P8": "Myanmar",
    "P9": "Namibia",
    "Q0": "Nauru",
    "Q1": "Nepal",
    "Q2": "Netherlands",
    "Q3": "Netherlands Antilles",
    "Q4": "New Caledonia",
    "Q5": "New Zealand",
    "Q6": "Nicaragua",
    "Q7": "Niger",
    "Q8": "Nigeria",
    "Q9": "Niue",
    "R0": "Norfolk Island",
    "R1": "Northern Mariana Islands",
    "R2": "Norway",
    "R3": "Oman",
    "R4": "Pakistan",
    "R5": "Palau",
    "R6": "Panama",
    "R7": "Papua New Guinea",
    "R8": "Paraguay",
    "R9": "Peru",
    "S0": "Philippines",
    "S1": "Pitcairn",
    "S2": "Poland",
    "S3": "Portugal",
    "S4": "Puerto Rico",
    "S5": "Qatar",
    "S6": "Reunion",
    "S7": "Romania",
    "S8": "Russia",
    "S9": "Rwanda",
    "T0": "Saint Helena",
    "T1": "Saint Kitts and Nevis",
    "T2": "Saint Lucia",
    "T3": "Saint Pierre and Miquelon",
    "T4": "Saint Vincent and Grenadines",
    "T5": "Samoa",
    "T6": "San Marino",
    "T7": "Sao Tome and Principe",
    "T8": "Saudi Arabia",
    "T9": "Senegal",
    "U0": "Serbia and Montenegro",
    "U1": "Seychelles",
    "U2": "Sierra Leone",
    "U3": "Singapore",
    "U4": "Slovakia",
    "U5": "Slovenia",
    "U6": "Solomon Islands",
    "U7": "Somalia",
    "U8": "South Africa",
    "U9": "South Georgia",
    "V0": "Spain",
    "V1": "Sri Lanka",
    "V2": "Sudan",
    "V3": "Suriname",
    "V4": "Svalbard and Jan Mayen",
    "V5": "Swaziland",
    "V6": "Sweden",
    "V7": "Switzerland",
    "V8": "Syria",
    "V9": "Taiwan",
    "W0": "Tajikistan",
    "W1": "Tanzania",
    "W2": "Thailand",
    "W3": "Timor-Leste",
    "W4": "Togo",
    "W5": "Tokelau",
    "W6": "Tonga",
    "W7": "Trinidad and Tobago",
    "W8": "Tunisia",
    "W9": "Turkey",
    "X0": "Turkmenistan",
    "X1": "Turks and Caicos Islands",
    "X2": "Tuvalu",
    "X3": "Uganda",
    "X4": "Ukraine",
    "X5": "United Arab Emirates",
    "X6": "United Kingdom",
    "X9": "Uruguay",
    "Y0": "Uzbekistan",
    "Y1": "Vanuatu",
    "Y2": "Venezuela",
    "Y3": "Vietnam",
    "Y4": "Virgin Islands, British",
    "Y5": "Virgin Islands, U.S.",
    "Y6": "Wallis and Futuna",
    "Y7": "Western Sahara",
    "Y8": "Yemen",
    "Y9": "Zambia",
    "Z0": "Zimbabwe",
    "Z4": "Curacao",
}

# US state codes (not foreign)
US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO",
    "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
    "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "PR",
    "VI", "GU", "AS", "MP",
}


def get_sec_fpi_data(cik: str) -> Optional[Dict[str, Any]]:
    """
    Get foreign private issuer data from SEC EDGAR.

    Returns:
        Dict with incorporation_country, is_fpi, recent_forms
    """
    if not cik:
        return None

    # Ensure CIK is zero-padded to 10 digits
    cik_padded = cik.lstrip('0').zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'BiotechScreener/1.0 (research)'}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)

        # Get incorporation info
        state_code = data.get('stateOfIncorporation', '')
        state_desc = data.get('stateOfIncorporationDescription', '')

        # Determine if foreign incorporation
        is_foreign_incorporation = False
        incorporation_country = None

        if state_code:
            if state_code in US_STATE_CODES:
                incorporation_country = "United States"
            elif state_code in SEC_COUNTRY_CODES:
                incorporation_country = SEC_COUNTRY_CODES[state_code]
                is_foreign_incorporation = True
            elif state_desc:
                incorporation_country = state_desc
                # Assume foreign if not a US state
                is_foreign_incorporation = state_code not in US_STATE_CODES

        # Check recent filings for FPI forms (20-F, 6-K) vs US forms (10-K, 10-Q)
        filings = data.get('filings', {}).get('recent', {})
        forms = filings.get('form', [])[:100]  # Check last 100 filings

        has_20f = any(f in ['20-F', '20-F/A'] for f in forms)
        has_6k = any(f in ['6-K', '6-K/A'] for f in forms)
        has_40f = any(f in ['40-F', '40-F/A'] for f in forms)  # Canadian FPI
        has_10k = any(f in ['10-K', '10-K/A'] for f in forms)
        has_10q = any(f in ['10-Q', '10-Q/A'] for f in forms)

        # FPI if files 20-F/6-K/40-F and NOT 10-K/10-Q
        is_fpi = (has_20f or has_6k or has_40f) and not (has_10k or has_10q)

        return {
            'incorporation_country': incorporation_country,
            'incorporation_code': state_code,
            'is_foreign_incorporation': is_foreign_incorporation,
            'is_fpi_by_filings': is_fpi,
            'has_20f': has_20f,
            'has_6k': has_6k,
            'has_40f': has_40f,
            'has_10k': has_10k,
            'has_10q': has_10q,
        }

    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        return None
    except Exception:
        return None


def get_yfinance_country_data(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get country and currency data from Yahoo Finance.

    Returns:
        Dict with country, financial_currency, city, exchange
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        country = info.get('country')
        financial_currency = info.get('financialCurrency')
        city = info.get('city')
        exchange = info.get('exchange')
        market = info.get('market')
        quote_type = info.get('quoteType')

        # Determine if this looks like an ADR
        # Foreign country + US market = ADR
        is_foreign_by_yf = country and country != "United States"
        is_us_market = market == "us_market" or exchange in ['NMS', 'NYQ', 'NGM', 'NCM', 'ASE']

        return {
            'country': country,
            'financial_currency': financial_currency,
            'city': city,
            'exchange': exchange,
            'market': market,
            'quote_type': quote_type,
            'is_foreign_by_yfinance': is_foreign_by_yf,
            'is_us_market': is_us_market,
        }

    except Exception:
        return None


def collect_adr_data(
    universe_file: Path,
    output_file: Path,
    skip_sec: bool = False,
) -> int:
    """
    Collect ADR data for all tickers in universe.

    Args:
        universe_file: Path to universe.json
        output_file: Path to output adr_data.json
        skip_sec: If True, skip SEC EDGAR lookups (faster but less accurate)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    print("=" * 80)
    print("ADR DATA COLLECTION")
    print("=" * 80)
    print(f"Date: {date.today()}")

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

    # Extract tickers and CIKs
    if isinstance(universe, list):
        securities = universe
    elif isinstance(universe, dict):
        securities = universe.get('active_securities', universe.get('securities', []))
    else:
        print(f"\n[ERROR] Invalid universe format")
        return 1

    ticker_cik_map = {}
    for sec in securities:
        ticker = sec.get('ticker')
        cik = sec.get('cik')
        if ticker:
            ticker_cik_map[ticker] = cik

    tickers = list(ticker_cik_map.keys())
    print(f"\nUniverse: {len(tickers)} tickers")
    print(f"Output: {output_file}")

    if not skip_sec:
        print("Sources: Yahoo Finance + SEC EDGAR")
    else:
        print("Sources: Yahoo Finance only (--skip-sec)")

    print("\n" + "=" * 80)
    print("COLLECTING ADR DATA")
    print("=" * 80 + "\n")

    results = []
    adr_count = 0
    fpi_count = 0
    foreign_count = 0

    for i, ticker in enumerate(tickers):
        cik = ticker_cik_map.get(ticker)

        # Get Yahoo Finance data
        yf_data = get_yfinance_country_data(ticker)

        # Get SEC data (if CIK available and not skipped)
        sec_data = None
        if not skip_sec and cik:
            sec_data = get_sec_fpi_data(cik)
            time.sleep(0.15)  # Rate limit for SEC

        # Merge and determine ADR status
        record = {
            'ticker': ticker,
            'cik': cik,
            'collection_date': date.today().isoformat(),
        }

        # Country determination (prefer Yahoo Finance, fallback to SEC)
        country = None
        if yf_data and yf_data.get('country'):
            country = yf_data['country']
            record['country_source'] = 'yfinance'
        elif sec_data and sec_data.get('incorporation_country'):
            country = sec_data['incorporation_country']
            record['country_source'] = 'sec_edgar'

        record['country_of_origin'] = country
        record['reporting_currency'] = yf_data.get('financial_currency') if yf_data else None
        record['headquarters_city'] = yf_data.get('city') if yf_data else None
        record['exchange'] = yf_data.get('exchange') if yf_data else None

        # SEC-specific fields
        if sec_data:
            record['incorporation_country'] = sec_data.get('incorporation_country')
            record['incorporation_code'] = sec_data.get('incorporation_code')
            record['has_20f'] = sec_data.get('has_20f', False)
            record['has_6k'] = sec_data.get('has_6k', False)
            record['has_40f'] = sec_data.get('has_40f', False)

        # Determine ADR status
        is_foreign = country and country != "United States"
        is_us_market = yf_data.get('is_us_market', True) if yf_data else True
        is_fpi = sec_data.get('is_fpi_by_filings', False) if sec_data else False

        # ADR = foreign company trading on US market
        is_adr = is_foreign and is_us_market

        # Detection method
        detection_methods = []
        if yf_data and yf_data.get('is_foreign_by_yfinance'):
            detection_methods.append('yfinance_country')
        if sec_data and sec_data.get('is_foreign_incorporation'):
            detection_methods.append('sec_incorporation')
        if is_fpi:
            detection_methods.append('sec_fpi_filings')

        record['is_adr'] = is_adr
        record['is_foreign_private_issuer'] = is_fpi
        record['is_foreign_company'] = is_foreign
        record['adr_detection_method'] = detection_methods if detection_methods else None

        results.append(record)

        # Update counts
        if is_adr:
            adr_count += 1
        if is_fpi:
            fpi_count += 1
        if is_foreign:
            foreign_count += 1

        # Progress output
        status = ""
        if is_adr:
            status = f"ADR ({country})"
        elif is_foreign:
            status = f"Foreign ({country})"
        else:
            status = "US"

        currency = record.get('reporting_currency', 'N/A')
        print(f"[{i+1:>3}/{len(tickers)}] {ticker:<6} {status:<25} Currency: {currency}")

        # Rate limiting for Yahoo Finance
        if (i + 1) % 10 == 0:
            time.sleep(0.3)

    print("\n" + "=" * 80)
    print("COLLECTION SUMMARY")
    print("=" * 80)
    print(f"Total tickers:              {len(tickers)}")
    print(f"US companies:               {len(tickers) - foreign_count}")
    print(f"Foreign companies:          {foreign_count}")
    print(f"ADRs (foreign + US market): {adr_count}")
    print(f"Foreign Private Issuers:    {fpi_count}")

    # Country breakdown
    countries = {}
    for r in results:
        c = r.get('country_of_origin') or 'Unknown'
        countries[c] = countries.get(c, 0) + 1

    print(f"\nCountry breakdown:")
    for country, count in sorted(countries.items(), key=lambda x: -x[1])[:15]:
        pct = 100 * count / len(tickers)
        print(f"  {country:<25} {count:>4} ({pct:>5.1f}%)")

    # Currency breakdown for non-USD
    currencies = {}
    for r in results:
        curr = r.get('reporting_currency')
        if curr and curr != 'USD':
            currencies[curr] = currencies.get(curr, 0) + 1

    if currencies:
        print(f"\nNon-USD reporting currencies:")
        for curr, count in sorted(currencies.items(), key=lambda x: -x[1]):
            print(f"  {curr:<6} {count:>4}")

    # Sort results by ticker
    results.sort(key=lambda x: x.get('ticker', ''))

    # Write output
    print(f"\nWriting to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'collection_date': date.today().isoformat(),
        'total_securities': len(tickers),
        'adr_count': adr_count,
        'fpi_count': fpi_count,
        'foreign_count': foreign_count,
        'country_breakdown': countries,
        'records': results,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"[OK] Saved {len(results)} records")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Collect ADR and foreign company data from Yahoo Finance and SEC EDGAR"
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=Path("production_data/universe.json"),
        help="Path to universe JSON file (default: production_data/universe.json)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("production_data/adr_data.json"),
        help="Output path for ADR data (default: production_data/adr_data.json)"
    )
    parser.add_argument(
        "--skip-sec",
        action="store_true",
        help="Skip SEC EDGAR lookups (faster but less accurate FPI detection)"
    )

    args = parser.parse_args()

    return collect_adr_data(
        universe_file=args.universe,
        output_file=args.output,
        skip_sec=args.skip_sec,
    )


if __name__ == "__main__":
    sys.exit(main())
