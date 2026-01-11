#!/usr/bin/env python3
"""
12-Quarter 13F History Extraction Runner

Extracts 13F holdings for a rolling window of N quarters,
writing governed per-quarter snapshots + manifest.

Point-in-Time Safety:
- Each snapshot represents holdings AS OF the filing date
- Prior quarter data is embedded for QoQ change analysis
- No future data leakage possible

Outputs:
- holdings_{YYYY-MM-DD}.json for each quarter
- manifest.json with run metadata and input hashes

Usage:
    python scripts/extract_13f_history.py \
        --quarter-end 2025-12-31 \
        --quarters 12 \
        --manager-registry production_data/manager_registry.json \
        --universe production_data/universe.json \
        --cusip-map production_data/cusip_static_map.json \
        --out-dir production_data/holdings_history/ \
        --mode live
"""

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.history.snapshots import (
    get_prior_quarter,
    get_quarter_sequence,
    is_valid_quarter_end,
    write_snapshot,
    write_manifest,
    SNAPSHOT_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
)
from governance.canonical_json import canonical_dumps
from governance.hashing import hash_file, hash_bytes, compute_input_hashes
from governance.run_id import compute_run_id
from governance.schema_registry import PIPELINE_VERSION


# =============================================================================
# EXTRACTION CORE (Adapted from edgar_13f_extractor_CORRECTED.py)
# =============================================================================

def import_extractor():
    """Import extraction functions from the corrected extractor."""
    # Import extraction module
    import edgar_13f_extractor_CORRECTED as extractor
    return extractor


def extract_quarter_holdings(
    quarter_end: date,
    managers: List[Dict[str, Any]],
    cusip_map: Dict[str, Any],
    universe_tickers: set,
    mode: str = "live",
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Extract holdings for all managers for a single quarter.

    Args:
        quarter_end: Quarter end date
        managers: List of manager dicts with 'cik' and 'name'
        cusip_map: CUSIP to ticker mapping
        universe_tickers: Set of tickers in universe
        mode: 'live' for SEC API calls, 'offline' for cached data

    Returns:
        Tuple of (holdings_by_ticker, warnings)
    """
    extractor = import_extractor()

    warnings = []
    all_holdings = {}  # {cik: {ticker: RawHolding}}
    all_filings_metadata = {}  # {cik: FilingInfo}

    for manager in managers:
        cik = manager['cik']
        name = manager['name']

        try:
            filing_info, ticker_holdings = extractor.extract_manager_holdings(
                cik=cik,
                manager_name=name,
                quarter_end=quarter_end,
                cusip_map=cusip_map,
                universe_tickers=universe_tickers,
            )

            if filing_info:
                all_holdings[cik] = ticker_holdings
                all_filings_metadata[cik] = filing_info
            else:
                warnings.append(f"No filing found for {name} ({cik}) Q{quarter_end}")

        except Exception as e:
            warnings.append(f"Error extracting {name} ({cik}): {e}")

    return all_holdings, all_filings_metadata, warnings


def build_quarter_snapshot(
    quarter_end: date,
    prior_quarter_end: date,
    current_holdings: Dict[str, Dict],  # {cik: {ticker: RawHolding}}
    prior_holdings: Dict[str, Dict],
    filings_metadata: Dict[str, Any],
    managers: List[Dict[str, Any]],
    universe: List[Dict[str, Any]],
    warnings: List[str],
    run_id: str,
    score_version: str,
    parameters_hash: str,
    input_lineage: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Build a quarter snapshot in contract format.
    """
    # Build ticker-centric structure
    universe_tickers = {
        s.get("ticker").strip().upper()
        for s in universe
        if isinstance(s.get("ticker"), str) and s.get("ticker").strip()
        and s.get("ticker") != "_XBI_BENCHMARK_"
    }

    ticker_data = {}

    for ticker in universe_tickers:
        # Get market cap
        ticker_info = next((s for s in universe if s.get('ticker') == ticker), None)
        market_cap = ticker_info.get('market_cap_usd', 0) if ticker_info else 0

        current = {}
        prior = {}
        filings = {}

        # Collect current holdings
        for cik, holdings in current_holdings.items():
            if ticker in holdings:
                holding = holdings[ticker]
                current[cik] = {
                    'quarter_end': quarter_end.isoformat(),
                    'state': 'KNOWN',
                    'shares': holding.shares,
                    'value_kusd': holding.value_kusd,
                    'put_call': holding.put_call,
                }

        # Collect prior holdings
        for cik, holdings in prior_holdings.items():
            if ticker in holdings:
                holding = holdings[ticker]
                prior[cik] = {
                    'quarter_end': prior_quarter_end.isoformat(),
                    'state': 'KNOWN',
                    'shares': holding.shares,
                    'value_kusd': holding.value_kusd,
                    'put_call': holding.put_call,
                }

        # Collect filings metadata
        for cik, info in filings_metadata.items():
            if ticker in current_holdings.get(cik, {}):
                filings[cik] = {
                    'quarter_end': info.quarter_end.isoformat(),
                    'accession': info.accession,
                    'total_value_kusd': info.total_value_kusd,
                    'filed_at': info.filed_at.isoformat(),
                    'is_amendment': info.is_amendment,
                }

        # Only include ticker if any manager holds it
        if current or prior:
            ticker_data[ticker] = {
                'market_cap_usd': market_cap,
                'holdings': {
                    'current': current,
                    'prior': prior,
                },
                'filings_metadata': filings,
            }

    # Build managers section
    managers_dict = {
        m['cik']: {
            'name': m['name'],
            'aum_b': m.get('aum_b', 0),
            'style': m.get('style', ''),
        }
        for m in managers
        if m['cik'] in current_holdings
    }

    # Build stats
    stats = {
        'tickers_count': len(ticker_data),
        'managers_count': len(managers_dict),
        'total_positions': sum(
            len(d['holdings']['current'])
            for d in ticker_data.values()
        ),
    }

    # Assemble snapshot
    snapshot = {
        '_schema': {
            'version': SNAPSHOT_SCHEMA_VERSION,
            'quarter_end': quarter_end.isoformat(),
            'prior_quarter_end': prior_quarter_end.isoformat(),
            'created_by': 'extract_13f_history.py',
        },
        '_governance': {
            'run_id': run_id,
            'score_version': score_version,
            'parameters_hash': parameters_hash,
            'input_lineage': input_lineage,
        },
        'tickers': ticker_data,
        'managers': managers_dict,
        'stats': stats,
        'warnings': warnings,
    }

    return snapshot


# =============================================================================
# MAIN EXTRACTION PIPELINE
# =============================================================================

def run_extraction(
    quarter_end: date,
    num_quarters: int,
    manager_registry_path: Path,
    universe_path: Path,
    cusip_map_path: Path,
    out_dir: Path,
    mode: str = "live",
    score_version: str = "v1",
) -> Dict[str, Any]:
    """
    Run full 12-quarter extraction pipeline.

    Args:
        quarter_end: Most recent quarter end
        num_quarters: Number of quarters to extract
        manager_registry_path: Path to manager registry JSON
        universe_path: Path to universe JSON
        cusip_map_path: Path to CUSIP mapping JSON
        out_dir: Output directory
        mode: 'live' or 'offline'
        score_version: Score version for governance

    Returns:
        Manifest dict
    """
    print(f"\n{'='*80}")
    print(f"13F HISTORY EXTRACTION")
    print(f"{'='*80}")
    print(f"Quarter End: {quarter_end}")
    print(f"Quarters: {num_quarters}")
    print(f"Mode: {mode}")
    print(f"{'='*80}\n")

    # Validate quarter end
    if not is_valid_quarter_end(quarter_end):
        raise ValueError(f"Invalid quarter end date: {quarter_end}")

    # Load inputs
    with open(manager_registry_path, 'r') as f:
        registry = json.load(f)

    with open(universe_path, 'r') as f:
        universe = json.load(f)

    with open(cusip_map_path, 'r') as f:
        cusip_map = json.load(f)

    # Get managers
    managers = registry.get('elite_core', [])
    if not managers:
        raise ValueError("No elite_core managers in registry")

    # Get universe tickers
    universe_tickers = {
        s.get("ticker").strip().upper()
        for s in universe
        if isinstance(s.get("ticker"), str) and s.get("ticker").strip()
        and s.get("ticker") != "_XBI_BENCHMARK_"
    }

    # Compute input hashes
    input_paths = [manager_registry_path, universe_path, cusip_map_path]
    input_hashes = compute_input_hashes(input_paths)

    # Compute parameters hash (simple - just the extraction params)
    params = {
        'quarter_end': quarter_end.isoformat(),
        'num_quarters': num_quarters,
        'mode': mode,
    }
    params_json = canonical_dumps(params)
    parameters_hash = hash_bytes(params_json.encode())[:16]

    # Compute run_id
    run_id = compute_run_id(
        as_of_date=quarter_end.isoformat(),
        score_version=score_version,
        parameters_hash=parameters_hash,
        input_hashes=input_hashes,
        pipeline_version=PIPELINE_VERSION,
    )

    print(f"Run ID: {run_id}")
    print(f"Parameters Hash: {parameters_hash}")
    print(f"Input Hashes: {len(input_hashes)} files")
    print()

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get quarter sequence
    quarters = get_quarter_sequence(quarter_end, num_quarters)

    # Storage for manifest
    quarters_info = []
    all_warnings = []

    # Cache holdings for prior quarter lookup
    holdings_cache = {}  # {quarter_end: {cik: {ticker: RawHolding}}}
    filings_cache = {}   # {quarter_end: {cik: FilingInfo}}

    # Extract each quarter
    for i, q_end in enumerate(quarters):
        prior_q = get_prior_quarter(q_end)

        print(f"\n--- Quarter {i+1}/{num_quarters}: {q_end} ---")

        # Extract current quarter if not cached
        if q_end not in holdings_cache:
            current_holdings, current_filings, warnings = extract_quarter_holdings(
                quarter_end=q_end,
                managers=managers,
                cusip_map=cusip_map,
                universe_tickers=universe_tickers,
                mode=mode,
            )
            holdings_cache[q_end] = current_holdings
            filings_cache[q_end] = current_filings
            all_warnings.extend(warnings)
        else:
            current_holdings = holdings_cache[q_end]
            current_filings = filings_cache[q_end]
            warnings = []

        # Extract prior quarter if not cached
        if prior_q not in holdings_cache:
            prior_holdings, prior_filings, prior_warnings = extract_quarter_holdings(
                quarter_end=prior_q,
                managers=managers,
                cusip_map=cusip_map,
                universe_tickers=universe_tickers,
                mode=mode,
            )
            holdings_cache[prior_q] = prior_holdings
            filings_cache[prior_q] = prior_filings
            # Don't add prior warnings to main list
        else:
            prior_holdings = holdings_cache[prior_q]

        # Build snapshot
        snapshot = build_quarter_snapshot(
            quarter_end=q_end,
            prior_quarter_end=prior_q,
            current_holdings=current_holdings,
            prior_holdings=prior_holdings,
            filings_metadata=current_filings,
            managers=managers,
            universe=universe,
            warnings=warnings,
            run_id=run_id,
            score_version=score_version,
            parameters_hash=parameters_hash,
            input_lineage=input_hashes,
        )

        # Write snapshot
        filepath, file_hash = write_snapshot(snapshot, q_end, out_dir)

        quarters_info.append({
            'quarter_end': q_end.isoformat(),
            'prior_quarter_end': prior_q.isoformat(),
            'filename': filepath.name,
            'sha256': file_hash,
            'tickers_count': snapshot['stats']['tickers_count'],
            'managers_count': snapshot['stats']['managers_count'],
            'warnings_count': len(warnings),
        })

        print(f"  Written: {filepath.name}")
        print(f"  Tickers: {snapshot['stats']['tickers_count']}")
        print(f"  Managers: {snapshot['stats']['managers_count']}")

    # Build manifest
    manifest = {
        '_schema': {
            'version': MANIFEST_SCHEMA_VERSION,
        },
        'run_id': run_id,
        'params': {
            'quarter_end': quarter_end.isoformat(),
            'quarters': num_quarters,
            'mode': mode,
            'manager_registry_hash': next(
                h['sha256'] for h in input_hashes
                if 'manager' in h['path'].lower()
            ),
            'universe_hash': next(
                h['sha256'] for h in input_hashes
                if 'universe' in h['path'].lower()
            ),
            'cusip_map_hash': next(
                h['sha256'] for h in input_hashes
                if 'cusip' in h['path'].lower()
            ),
        },
        'quarters': quarters_info,
        'input_hashes': input_hashes,
        'warnings_total': len(all_warnings),
    }

    # Write manifest
    manifest_path, manifest_hash = write_manifest(manifest, out_dir)

    print(f"\n{'='*80}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Run ID: {run_id}")
    print(f"Quarters extracted: {len(quarters_info)}")
    print(f"Output directory: {out_dir}")
    print(f"Manifest: {manifest_path.name} ({manifest_hash[:16]}...)")
    print(f"Total warnings: {len(all_warnings)}")
    print(f"{'='*80}\n")

    return manifest


# =============================================================================
# CLI
# =============================================================================

def parse_date(s: str) -> date:
    """Parse YYYY-MM-DD date string."""
    return datetime.strptime(s, '%Y-%m-%d').date()


def main():
    parser = argparse.ArgumentParser(
        description="Extract 13F holdings history for multiple quarters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--quarter-end',
        type=parse_date,
        required=True,
        help='Most recent quarter end date (YYYY-MM-DD)',
    )

    parser.add_argument(
        '--quarters',
        type=int,
        default=12,
        help='Number of quarters to extract (default: 12)',
    )

    parser.add_argument(
        '--manager-registry',
        type=Path,
        required=True,
        help='Path to manager_registry.json',
    )

    parser.add_argument(
        '--universe',
        type=Path,
        required=True,
        help='Path to universe.json',
    )

    parser.add_argument(
        '--cusip-map',
        type=Path,
        required=True,
        help='Path to cusip_static_map.json',
    )

    parser.add_argument(
        '--out-dir',
        type=Path,
        required=True,
        help='Output directory for holdings history',
    )

    parser.add_argument(
        '--mode',
        choices=['live', 'offline'],
        default='live',
        help='Extraction mode: live (SEC API) or offline (cached)',
    )

    parser.add_argument(
        '--score-version',
        default='v1',
        help='Score version for governance (default: v1)',
    )

    args = parser.parse_args()

    # Validate inputs exist
    for path in [args.manager_registry, args.universe, args.cusip_map]:
        if not path.exists():
            print(f"ERROR: Input file not found: {path}")
            sys.exit(1)

    # Run extraction
    try:
        manifest = run_extraction(
            quarter_end=args.quarter_end,
            num_quarters=args.quarters,
            manager_registry_path=args.manager_registry,
            universe_path=args.universe,
            cusip_map_path=args.cusip_map,
            out_dir=args.out_dir,
            mode=args.mode,
            score_version=args.score_version,
        )

        print("Extraction successful.")
        sys.exit(0)

    except Exception as e:
        print(f"ERROR: Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
