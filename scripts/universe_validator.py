#!/usr/bin/env python3
"""
universe_validator.py - Hard Universe Gate

CRITICAL FIX #1: Only allow tickers that are in your ETF constituent set.
Drop anything else, even if it has scores.

This prevents universe contamination (NAME, USD, CCCC, etc.) from polluting backtests.

Usage:
    from universe_validator import validate_universe, load_etf_constituents
    
    valid_universe = load_etf_constituents()
    filtered = validate_universe(tickers, valid_universe)
"""

import json
from pathlib import Path
from typing import List, Set, Dict, Tuple


def load_etf_constituents(data_dir: str = "production_data") -> Set[str]:
    """
    Load ETF constituent lists (XBI, IBB, NBI union).
    
    Returns:
        Set of valid ticker symbols (uppercase)
    """
    constituents = set()
    
    # Load from ETF files
    etf_files = [
        f"{data_dir}/xbi_constituents.json",
        f"{data_dir}/ibb_constituents.json", 
        f"{data_dir}/nbi_constituents.json",
    ]
    
    for filepath in etf_files:
        path = Path(filepath)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    # Handle different JSON formats
                    if isinstance(data, list):
                        # Direct list of tickers
                        tickers = [str(t).upper() for t in data]
                    elif isinstance(data, dict) and 'constituents' in data:
                        # Dict with 'constituents' key
                        tickers = [str(t).upper() for t in data['constituents']]
                    elif isinstance(data, dict):
                        # Dict where keys are tickers
                        tickers = [str(k).upper() for k in data.keys()]
                    else:
                        print(f"Warning: Unknown format in {filepath}")
                        continue
                    
                    constituents.update(tickers)
                    print(f"Loaded {len(tickers)} constituents from {path.name}")
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
    
    if not constituents:
        print("ERROR: No ETF constituents loaded! Check data_dir and file formats.")
        print("Expected files:")
        for f in etf_files:
            print(f"  - {f}")
    
    return constituents


def validate_universe(
    tickers: List[str], 
    valid_universe: Set[str],
    strict: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Validate tickers against ETF constituent universe.
    
    Args:
        tickers: List of tickers to validate
        valid_universe: Set of valid tickers from ETF constituents
        strict: If True, only allow exact matches
    
    Returns:
        (valid_tickers, rejected_tickers)
    """
    valid = []
    rejected = []
    
    for ticker in tickers:
        ticker_upper = ticker.upper()
        
        if ticker_upper in valid_universe:
            valid.append(ticker)
        else:
            rejected.append(ticker)
    
    return valid, rejected


def create_universe_gate(data_dir: str = "production_data") -> callable:
    """
    Create a universe validation function with loaded constituents.
    
    Returns:
        Function that validates tickers
    """
    valid_universe = load_etf_constituents(data_dir)
    
    def gate(tickers: List[str]) -> Tuple[List[str], List[str]]:
        return validate_universe(tickers, valid_universe)
    
    return gate


def apply_universe_gate(
    tickers: List[str],
    data_dir: str = "production_data",
    verbose: bool = True
) -> List[str]:
    """
    Apply hard universe gate and return only valid tickers.
    
    Args:
        tickers: Input ticker list
        data_dir: Directory with ETF constituent files
        verbose: Print statistics
    
    Returns:
        Filtered list of valid tickers only
    """
    valid_universe = load_etf_constituents(data_dir)
    
    if not valid_universe:
        print("WARNING: No valid universe loaded! Returning all tickers.")
        return tickers
    
    valid, rejected = validate_universe(tickers, valid_universe)
    
    if verbose:
        print(f"\n{'='*80}")
        print("UNIVERSE GATE - VALIDATION RESULTS")
        print(f"{'='*80}")
        print(f"Input tickers: {len(tickers)}")
        print(f"Valid (in ETF): {len(valid)} ({len(valid)/len(tickers)*100:.1f}%)")
        print(f"Rejected: {len(rejected)} ({len(rejected)/len(tickers)*100:.1f}%)")
        
        if rejected:
            print(f"\nRejected tickers (not in XBI/IBB/NBI):")
            for i, ticker in enumerate(sorted(rejected)[:20], 1):
                print(f"  {i}. {ticker}")
            if len(rejected) > 20:
                print(f"  ... and {len(rejected) - 20} more")
        
        print(f"{'='*80}\n")
    
    return valid


def main():
    """Test universe gate."""
    
    # Test tickers including some invalid ones
    test_tickers = [
        'CVAC', 'KROS', 'BIIB',  # Valid
        'NAME', 'USD', 'CCCC', '-',  # Invalid
        'VRTX', 'GILD', 'REGN',  # Valid
    ]
    
    print("Testing universe gate with sample tickers...")
    print(f"Input: {test_tickers}\n")
    
    valid = apply_universe_gate(test_tickers, verbose=True)
    
    print(f"\nResult: {len(valid)} valid tickers")
    print(f"Valid tickers: {sorted(valid)}")


if __name__ == "__main__":
    main()
