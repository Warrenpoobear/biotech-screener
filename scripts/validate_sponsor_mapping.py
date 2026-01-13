#!/usr/bin/env python3
"""
Validate sponsor mapping by testing against ClinicalTrials.gov.

Checks that each sponsor name in the mapping can find clinical trials,
and reports coverage statistics.

Usage:
    python scripts/validate_sponsor_mapping.py --mapping data/ticker_to_sponsor.json
    python scripts/validate_sponsor_mapping.py --mapping data/ticker_to_sponsor.json --sample 50

Author: Wake Robin Capital
Version: 1.0
"""

import argparse
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# Configuration
# =============================================================================

CTGOV_API_URL = "https://clinicaltrials.gov/api/v2/studies"
CTGOV_HEADERS = {
    'User-Agent': 'WakeRobinBiotechScreener/2.0 (research@wakerobincapital.com)',
    'Accept': 'application/json'
}

# Rate limiting
REQUEST_DELAY = 0.5  # seconds between requests


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ValidationResult:
    ticker: str
    sponsor: str
    trial_count: int
    status: str  # 'found', 'not_found', 'error'
    error_message: Optional[str] = None


# =============================================================================
# ClinicalTrials.gov API
# =============================================================================

def search_clinical_trials(sponsor_name: str, max_results: int = 1) -> Tuple[Optional[int], Optional[str]]:
    """
    Search ClinicalTrials.gov for trials by sponsor name.

    Returns:
        (trial_count, error_message)
        - trial_count is None if there was an error
        - error_message is None if successful
    """
    params = {
        "query.spons": sponsor_name,
        "pageSize": max_results,
        "format": "json"
    }

    # Build URL with proper encoding (same as working fetcher)
    url = f"{CTGOV_API_URL}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url)
        req.add_header('Accept', 'application/json')
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
            # Get total count from response
            total_count = data.get('totalCount', 0)
            # Also check studies array as fallback
            if total_count == 0:
                studies = data.get('studies', [])
                total_count = len(studies)
            return total_count, None

    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return None, f"URL Error: {e.reason}"
    except json.JSONDecodeError:
        return None, "Invalid JSON response"
    except Exception as e:
        return None, str(e)


# Need to import urllib.parse for URL encoding
import urllib.parse


# =============================================================================
# Validation Logic
# =============================================================================

def validate_mapping(
    mapping_file: str,
    sample_size: Optional[int] = None,
    verbose: bool = True
) -> List[ValidationResult]:
    """
    Validate sponsor mapping against ClinicalTrials.gov.

    Args:
        mapping_file: Path to ticker_to_sponsor.json
        sample_size: If set, only validate this many entries (random sample)
        verbose: Print progress to console

    Returns:
        List of ValidationResult objects
    """

    # Load mapping
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    if verbose:
        print(f"Loaded {len(mapping)} sponsor mappings from {mapping_file}")
        print()

    # Sample if requested
    items = list(mapping.items())
    if sample_size and sample_size < len(items):
        import random
        items = random.sample(items, sample_size)
        if verbose:
            print(f"Validating random sample of {sample_size} mappings")
            print()

    results = []

    for i, (ticker, sponsor) in enumerate(items, 1):
        if verbose:
            print(f"[{i}/{len(items)}] {ticker} → {sponsor}...", end=" ", flush=True)

        trial_count, error = search_clinical_trials(sponsor)

        if error:
            result = ValidationResult(
                ticker=ticker,
                sponsor=sponsor,
                trial_count=0,
                status='error',
                error_message=error
            )
            if verbose:
                print(f"⚠️  {error}")
        elif trial_count > 0:
            result = ValidationResult(
                ticker=ticker,
                sponsor=sponsor,
                trial_count=trial_count,
                status='found'
            )
            if verbose:
                print(f"✓ {trial_count} trials")
        else:
            result = ValidationResult(
                ticker=ticker,
                sponsor=sponsor,
                trial_count=0,
                status='not_found'
            )
            if verbose:
                print("✗ No trials found")

        results.append(result)

        # Rate limiting
        time.sleep(REQUEST_DELAY)

    return results


def print_summary(results: List[ValidationResult], mapping_file: str):
    """Print validation summary."""

    found = [r for r in results if r.status == 'found']
    not_found = [r for r in results if r.status == 'not_found']
    errors = [r for r in results if r.status == 'error']

    total = len(results)
    found_pct = len(found) / total * 100 if total else 0
    not_found_pct = len(not_found) / total * 100 if total else 0
    error_pct = len(errors) / total * 100 if total else 0

    print()
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Mapping file:     {mapping_file}")
    print(f"Mappings tested:  {total}")
    print()
    print(f"Found trials:     {len(found):>4} ({found_pct:.1f}%)")
    print(f"No trials found:  {len(not_found):>4} ({not_found_pct:.1f}%)")
    print(f"API errors:       {len(errors):>4} ({error_pct:.1f}%)")
    print()

    # Calculate trial statistics
    if found:
        trial_counts = [r.trial_count for r in found]
        avg_trials = sum(trial_counts) / len(trial_counts)
        max_trials = max(trial_counts)
        min_trials = min(trial_counts)

        print(f"Trial statistics (for found):")
        print(f"  Average:  {avg_trials:.1f} trials per company")
        print(f"  Maximum:  {max_trials} trials")
        print(f"  Minimum:  {min_trials} trial(s)")
        print()

    # Interpretation
    print("-" * 60)
    print("INTERPRETATION")
    print("-" * 60)

    # Not all biotech companies have clinical trials:
    # - Diagnostics companies
    # - Platform/tools companies
    # - Preclinical stage companies
    # - Companies that license out their programs
    # So 50-70% finding trials is actually good

    if found_pct >= 60:
        print(f"✓ EXCELLENT: {found_pct:.1f}% of mappings find clinical trials")
        print("  This indicates high-quality sponsor name mappings.")
    elif found_pct >= 50:
        print(f"✓ GOOD: {found_pct:.1f}% of mappings find clinical trials")
        print("  Many biotech companies don't have active trials.")
    elif found_pct >= 40:
        print(f"⚠️  ACCEPTABLE: {found_pct:.1f}% of mappings find clinical trials")
        print("  Some mappings may need manual review.")
    else:
        print(f"✗ LOW: {found_pct:.1f}% of mappings find clinical trials")
        print("  Many sponsor names may be incorrect.")

    print()

    # Show top performers
    if found:
        print("-" * 60)
        print("TOP 10 BY TRIAL COUNT")
        print("-" * 60)
        top_10 = sorted(found, key=lambda r: r.trial_count, reverse=True)[:10]
        for r in top_10:
            print(f"  {r.ticker:6} {r.sponsor:40} {r.trial_count:>4} trials")
        print()

    # Show companies with no trials
    if not_found:
        print("-" * 60)
        print(f"NO TRIALS FOUND ({len(not_found)} companies)")
        print("-" * 60)
        print("These may be: diagnostics, tools/platform, preclinical,")
        print("or companies that license programs to others.")
        print()

        # Show first 20
        for r in not_found[:20]:
            print(f"  {r.ticker:6} {r.sponsor}")

        if len(not_found) > 20:
            print(f"  ... and {len(not_found) - 20} more")
        print()

    # Show errors
    if errors:
        print("-" * 60)
        print(f"API ERRORS ({len(errors)} companies)")
        print("-" * 60)
        for r in errors[:10]:
            print(f"  {r.ticker:6} {r.sponsor}: {r.error_message}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        print()


def save_results(results: List[ValidationResult], output_file: str):
    """Save validation results to JSON."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total': len(results),
        'found': len([r for r in results if r.status == 'found']),
        'not_found': len([r for r in results if r.status == 'not_found']),
        'errors': len([r for r in results if r.status == 'error']),
        'results': [
            {
                'ticker': r.ticker,
                'sponsor': r.sponsor,
                'trial_count': r.trial_count,
                'status': r.status,
                'error': r.error_message
            }
            for r in results
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to: {output_path}")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Validate sponsor mapping against ClinicalTrials.gov'
    )
    parser.add_argument(
        '--mapping',
        required=True,
        help='Path to ticker_to_sponsor.json'
    )
    parser.add_argument(
        '--sample',
        type=int,
        help='Only validate a random sample of N entries'
    )
    parser.add_argument(
        '--output',
        help='Save detailed results to JSON file'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only show summary, not progress'
    )

    args = parser.parse_args()

    # Validate
    results = validate_mapping(
        mapping_file=args.mapping,
        sample_size=args.sample,
        verbose=not args.quiet
    )

    # Print summary
    print_summary(results, args.mapping)

    # Save results if requested
    if args.output:
        save_results(results, args.output)


if __name__ == '__main__':
    main()
