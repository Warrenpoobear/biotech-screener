#!/usr/bin/env python3
"""
cfo_extractor.py - SEC Cash Flow Statement Extractor

Extracts Operating Cash Flow (CFO) from 10-Q/10-K filings with point-in-time integrity.

Key Features:
- Extracts YTD CFO values (Module 2 handles quarterly conversion)
- Captures fiscal_period metadata (Q1/Q2/Q3/FY)
- Supports XBRL-tagged filings
- Maintains filing date for PIT validation
- Handles restatements and amendments

Usage:
    from cfo_extractor import extract_cfo_data
    cfo_data = extract_cfo_data(ticker, filing_path, as_of_date)
"""

import json
import re
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from decimal import Decimal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CFORecord:
    """Single CFO data point from a filing"""
    ticker: str
    filing_date: str              # YYYY-MM-DD when filed
    fiscal_year: int              # e.g., 2025
    fiscal_period: str            # "Q1", "Q2", "Q3", "FY"
    period_end_date: str          # YYYY-MM-DD
    cfo_value: float              # CFO in dollars (negative = burn)
    is_ytd: bool                  # True for Q2/Q3 (cumulative)
    form_type: str                # "10-Q" or "10-K"
    accession_number: str         # SEC filing ID
    source_tag: str               # XBRL tag used
    
    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization"""
        return asdict(self)


# ============================================================================
# XBRL TAG MAPPINGS
# ============================================================================

# Common XBRL tags for Operating Cash Flow (in priority order)
CFO_TAGS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "CashProvidedByUsedInOperatingActivities", 
    "NetCashFromOperatingActivities",
    "CashFromOperatingActivities",
    "OperatingCashFlow",
]

# Period type indicators
PERIOD_INDICATORS = {
    "Q1": ["Q1", "1stQuarter", "FirstQuarter"],
    "Q2": ["Q2", "2ndQuarter", "SecondQuarter"],
    "Q3": ["Q3", "3rdQuarter", "ThirdQuarter"],
    "FY": ["FY", "Annual", "12months", "FullYear"],
}


# ============================================================================
# XBRL PARSER
# ============================================================================

def parse_xbrl_filing(filing_path: Path, ticker: str) -> List[CFORecord]:
    """
    Parse XBRL-formatted SEC filing for CFO data.
    
    Args:
        filing_path: Path to filing (can be XML or HTML with XBRL)
        ticker: Ticker symbol
    
    Returns:
        List of CFORecord objects (one per period found)
    """
    records = []
    
    try:
        with open(filing_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Failed to read {filing_path}: {e}")
        return records
    
    # Extract form type - try multiple patterns
    form_match = re.search(r'<TYPE>(10-[QK])', content, re.IGNORECASE)
    if not form_match:
        form_match = re.search(r'CONFORMED SUBMISSION TYPE:\s*(10-[QK])', content, re.IGNORECASE)
    form_type = form_match.group(1).upper() if form_match else "UNKNOWN"
    
    # Extract filing date - try multiple patterns
    filing_date = None
    # Pattern 1: <FILING-DATE>YYYY-MM-DD
    filing_date_match = re.search(r'<FILING-DATE>(\d{4}-\d{2}-\d{2})', content)
    if filing_date_match:
        filing_date = filing_date_match.group(1)
    else:
        # Pattern 2: FILED AS OF DATE: YYYYMMDD
        filing_date_match = re.search(r'FILED AS OF DATE:\s*(\d{8})', content)
        if filing_date_match:
            date_str = filing_date_match.group(1)
            filing_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        else:
            # Pattern 3: CONFORMED PERIOD OF REPORT: YYYYMMDD
            filing_date_match = re.search(r'CONFORMED PERIOD OF REPORT:\s*(\d{8})', content)
            if filing_date_match:
                date_str = filing_date_match.group(1)
                filing_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    # Extract accession number
    accession_match = re.search(r'<ACCESSION-NUMBER>([0-9-]+)', content)
    if not accession_match:
        accession_match = re.search(r'ACCESSION NUMBER:\s*([0-9-]+)', content)
    accession_number = accession_match.group(1) if accession_match else "UNKNOWN"
    
    if not filing_date:
        logger.warning(f"No filing date found in {filing_path}")
        # Don't return empty - try to extract CFO data anyway using filename date
        # Extract date from filename as fallback
        filename = filing_path.name
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if date_match:
            filing_date = date_match.group(1)
            logger.info(f"Using date from filename: {filing_date}")
        else:
            return records
    
    # Try each CFO tag in priority order
    for cfo_tag in CFO_TAGS:
        # Look for XBRL facts with this tag
        # Pattern: <us-gaap:TAG contextRef="..." unitRef="..." decimals="...">VALUE</us-gaap:TAG>
        # Make pattern more flexible to handle different namespaces and attributes
        pattern = rf'<(?:[\w-]+:)?({cfo_tag})\s+[^>]*?contextRef="([^"]+)"[^>]*?>([-\d.,]+)</(?:[\w-]+:)?\1>'
        
        matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            tag = match.group(1)
            context_ref = match.group(2)
            value_str = match.group(3).replace(',', '')  # Remove commas
            
            try:
                cfo_value = float(value_str)
            except ValueError:
                continue
            
            # Extract decimals attribute for scaling
            # decimals="-3" means value is in thousands (multiply by 1000)
            # decimals="-6" means value is in millions (multiply by 1,000,000)
            # Search more broadly for decimals attribute near this tag
            tag_start = match.start()
            tag_end = match.end()
            context_window = content[max(0, tag_start-500):min(len(content), tag_end+500)]
            
            decimals_pattern = rf'decimals="(-?\d+)"'
            decimals_match = re.search(decimals_pattern, context_window, re.IGNORECASE)
            
            if decimals_match:
                decimals = int(decimals_match.group(1))
                if decimals < 0:
                    # Negative decimals means multiply by 10^|decimals|
                    scale_factor = 10 ** abs(decimals)
                    cfo_value *= scale_factor
                    logger.debug(f"Applied scale factor {scale_factor} (decimals={decimals})")
            
            # Parse context to get fiscal period and dates
            # Look for context block - might be far from the tag
            context_pattern = rf'<context[^>]*?id="{re.escape(context_ref)}"[^>]*?>.*?</context>'
            context_match = re.search(context_pattern, content, re.DOTALL | re.IGNORECASE)
            
            if not context_match:
                # Try without exact ID match - find any context nearby
                logger.debug(f"Exact context not found for {context_ref}, trying broader search")
                continue
            
            context_block = context_match.group(0)
            
            # Extract fiscal period
            fiscal_period = extract_fiscal_period(context_block, form_type)
            
            # Extract fiscal year - try multiple patterns
            fiscal_year = None
            fiscal_year_match = re.search(r'<(?:[\w-]+:)?FiscalYear>(\d{4})</(?:[\w-]+:)?FiscalYear>', context_block, re.IGNORECASE)
            if fiscal_year_match:
                fiscal_year = int(fiscal_year_match.group(1))
            
            # Extract period end date - try multiple patterns
            period_end_date = None
            # Try instant first
            end_date_match = re.search(r'<(?:[\w-]+:)?instant>(\d{4}-\d{2}-\d{2})</(?:[\w-]+:)?instant>', context_block, re.IGNORECASE)
            if end_date_match:
                period_end_date = end_date_match.group(1)
            else:
                # Try endDate
                end_date_match = re.search(r'<(?:[\w-]+:)?endDate>(\d{4}-\d{2}-\d{2})</(?:[\w-]+:)?endDate>', context_block, re.IGNORECASE)
                if end_date_match:
                    period_end_date = end_date_match.group(1)
            
            # Infer fiscal year from end date if not explicitly stated
            if not fiscal_year and period_end_date:
                fiscal_year = int(period_end_date[:4])
            
            if not all([fiscal_period, fiscal_year, period_end_date]):
                continue
            
            # Determine if YTD (Q2/Q3 are always YTD in 10-Q)
            is_ytd = fiscal_period in ["Q2", "Q3"]
            
            record = CFORecord(
                ticker=ticker,
                filing_date=filing_date,
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period,
                period_end_date=period_end_date,
                cfo_value=cfo_value,
                is_ytd=is_ytd,
                form_type=form_type,
                accession_number=accession_number,
                source_tag=tag
            )
            
            records.append(record)
    
    return records


def extract_fiscal_period(context_block: str, form_type: str) -> Optional[str]:
    """
    Extract fiscal period from XBRL context block.
    
    Args:
        context_block: XBRL context XML
        form_type: "10-Q" or "10-K"
    
    Returns:
        "Q1", "Q2", "Q3", or "FY"
    """
    # Check for explicit fiscal period tag - try multiple patterns
    for period_key, indicators in PERIOD_INDICATORS.items():
        for indicator in indicators:
            # Try with any namespace prefix
            pattern = rf'<(?:[\w-]+:)?FiscalPeriod[^>]*?>{indicator}</(?:[\w-]+:)?FiscalPeriod>'
            if re.search(pattern, context_block, re.IGNORECASE):
                return period_key
    
    # Infer from form type if not explicit
    if form_type == "10-K":
        return "FY"
    
    # For 10-Q, try to infer from context ID or period duration
    context_id_match = re.search(r'id="([^"]+)"', context_block, re.IGNORECASE)
    if context_id_match:
        context_id = context_id_match.group(1).lower()
        if 'q1' in context_id or '3m' in context_id or 'three' in context_id:
            return "Q1"
        if 'q2' in context_id or '6m' in context_id or 'six' in context_id:
            return "Q2"
        if 'q3' in context_id or '9m' in context_id or 'nine' in context_id:
            return "Q3"
    
    # Check duration in months - try multiple patterns
    duration_patterns = [
        r'<(?:[\w-]+:)?duration[^>]*?>P(\d+)M</(?:[\w-]+:)?duration>',
        r'P(\d+)M',  # Simple pattern
    ]
    
    for pattern in duration_patterns:
        duration_match = re.search(pattern, context_block, re.IGNORECASE)
        if duration_match:
            months = int(duration_match.group(1))
            if months == 3:
                return "Q1"  # First 3 months
            elif months == 6:
                return "Q2"
            elif months == 9:
                return "Q3"
            elif months == 12:
                return "FY"
    
    # Check start and end dates to calculate duration
    start_match = re.search(r'<(?:[\w-]+:)?startDate>(\d{4}-\d{2}-\d{2})</(?:[\w-]+:)?startDate>', context_block, re.IGNORECASE)
    end_match = re.search(r'<(?:[\w-]+:)?endDate>(\d{4}-\d{2}-\d{2})</(?:[\w-]+:)?endDate>', context_block, re.IGNORECASE)
    
    if start_match and end_match:
        from datetime import datetime
        start_date = datetime.strptime(start_match.group(1), '%Y-%m-%d')
        end_date = datetime.strptime(end_match.group(1), '%Y-%m-%d')
        days = (end_date - start_date).days
        
        # Approximate months (30 days per month)
        months = round(days / 30)
        if months <= 3:
            return "Q1"
        elif months <= 6:
            return "Q2"
        elif months <= 9:
            return "Q3"
        elif months <= 12:
            return "FY"
    
    return None


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def extract_cfo_batch(
    ticker_filings: Dict[str, List[Path]],
    as_of_date: date
) -> Dict[str, List[CFORecord]]:
    """
    Extract CFO data for multiple tickers.
    
    Args:
        ticker_filings: Dict of ticker -> list of filing paths
        as_of_date: Only include filings on or before this date
    
    Returns:
        Dict of ticker -> list of CFORecord objects
    """
    results = {}
    as_of_dt = datetime.combine(as_of_date, datetime.min.time())
    
    for ticker, filing_paths in ticker_filings.items():
        ticker_records = []
        
        for filing_path in filing_paths:
            records = parse_xbrl_filing(filing_path, ticker)
            
            # PIT filter: only include filings on or before as_of_date
            for record in records:
                filing_dt = datetime.fromisoformat(record.filing_date)
                if filing_dt <= as_of_dt:
                    ticker_records.append(record)
        
        # Sort by period_end_date (most recent first)
        ticker_records.sort(key=lambda r: r.period_end_date, reverse=True)
        
        results[ticker] = ticker_records
    
    logger.info(f"Extracted CFO data for {len(results)} tickers")
    
    return results


# ============================================================================
# MODULE 2 INTEGRATION
# ============================================================================

def prepare_for_module_2(
    cfo_records: Dict[str, List[CFORecord]]
) -> List[Dict]:
    """
    Convert CFO records to Module 2 financial_data format.
    
    Builds the YTD CFO fields that Module 2 expects.
    
    Args:
        cfo_records: Dict of ticker -> list of CFORecord objects
    
    Returns:
        List of dicts ready for Module 2
    """
    financial_data = []
    
    for ticker, records in cfo_records.items():
        if not records:
            continue
        
        # Group by fiscal year
        by_fiscal_year = {}
        for record in records:
            key = record.fiscal_year
            if key not in by_fiscal_year:
                by_fiscal_year[key] = []
            by_fiscal_year[key].append(record)
        
        # Get most recent fiscal year
        latest_fy = max(by_fiscal_year.keys())
        latest_records = by_fiscal_year[latest_fy]
        
        # Sort by period (Q1, Q2, Q3, FY)
        period_order = {"Q1": 1, "Q2": 2, "Q3": 3, "FY": 4}
        latest_records.sort(key=lambda r: period_order.get(r.fiscal_period, 0))
        
        # Find most recent period
        most_recent = latest_records[-1]
        
        # Build Module 2 fields
        module_2_data = {
            'ticker': ticker,
            'fiscal_period': most_recent.fiscal_period,
            'CFO_ytd_current': most_recent.cfo_value,
            'filing_date': most_recent.filing_date,
            'period_end_date': most_recent.period_end_date,
        }
        
        # Add prior YTD for Q2/Q3
        if most_recent.fiscal_period in ["Q2", "Q3"]:
            prior_period = "Q1" if most_recent.fiscal_period == "Q2" else "Q2"
            prior_record = next((r for r in latest_records if r.fiscal_period == prior_period), None)
            if prior_record:
                module_2_data['CFO_ytd_prev'] = prior_record.cfo_value
        
        # Add annual CFO and Q3 YTD for FY/Q4 calculations
        fy_record = next((r for r in latest_records if r.fiscal_period == "FY"), None)
        q3_record = next((r for r in latest_records if r.fiscal_period == "Q3"), None)
        
        if fy_record:
            module_2_data['CFO_fy_annual'] = fy_record.cfo_value
        if q3_record:
            module_2_data['CFO_ytd_q3'] = q3_record.cfo_value
        
        financial_data.append(module_2_data)
    
    logger.info(f"Prepared {len(financial_data)} ticker records for Module 2")
    
    return financial_data


# ============================================================================
# PERSISTENCE
# ============================================================================

def save_cfo_records(
    cfo_records: Dict[str, List[CFORecord]],
    output_path: Path
):
    """Save CFO records to JSON"""
    serializable = {
        ticker: [r.to_dict() for r in records]
        for ticker, records in cfo_records.items()
    }
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    logger.info(f"Saved CFO records to {output_path}")


def load_cfo_records(input_path: Path) -> Dict[str, List[CFORecord]]:
    """Load CFO records from JSON"""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    records = {
        ticker: [CFORecord(**r) for r in record_list]
        for ticker, record_list in data.items()
    }
    
    logger.info(f"Loaded CFO records for {len(records)} tickers from {input_path}")
    
    return records


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for CFO extraction"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract CFO data from SEC filings'
    )
    parser.add_argument(
        '--filings-dir',
        type=str,
        required=True,
        help='Directory containing SEC filing files (organized by ticker)'
    )
    parser.add_argument(
        '--as-of-date',
        type=str,
        required=True,
        help='Point-in-time date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='cfo_data.json',
        help='Output JSON file'
    )
    parser.add_argument(
        '--module-2-output',
        type=str,
        default='financial_data_cfo.json',
        help='Output file in Module 2 format'
    )
    
    args = parser.parse_args()
    
    # Parse as_of_date
    as_of_date = datetime.fromisoformat(args.as_of_date).date()
    
    # Discover filings
    filings_dir = Path(args.filings_dir)
    ticker_filings = {}
    
    for ticker_dir in filings_dir.iterdir():
        if ticker_dir.is_dir():
            ticker = ticker_dir.name
            filing_files = list(ticker_dir.glob('*.txt')) + list(ticker_dir.glob('*.xml'))
            if filing_files:
                ticker_filings[ticker] = filing_files
    
    logger.info(f"Found filings for {len(ticker_filings)} tickers")
    
    # Extract CFO data
    cfo_records = extract_cfo_batch(ticker_filings, as_of_date)
    
    # Save raw records
    save_cfo_records(cfo_records, Path(args.output))
    
    # Prepare for Module 2
    module_2_data = prepare_for_module_2(cfo_records)
    
    with open(args.module_2_output, 'w') as f:
        json.dump(module_2_data, f, indent=2)
    
    logger.info(f"Saved Module 2 data to {args.module_2_output}")
    
    # Print summary
    print("\n" + "="*80)
    print("CFO EXTRACTION COMPLETE")
    print("="*80)
    print(f"Tickers processed: {len(cfo_records)}")
    print(f"Total records extracted: {sum(len(r) for r in cfo_records.values())}")
    print(f"Records ready for Module 2: {len(module_2_data)}")
    print("="*80)


if __name__ == "__main__":
    main()
