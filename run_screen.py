#!/usr/bin/env python3
"""
run_screen.py - Deterministic Biotech Screening Orchestrator

Coordinates Modules 1-5 to produce weekly ranked investment opportunities.

DETERMINISM GUARANTEES:
- as_of_date is REQUIRED (no today() defaults)
- All modules receive explicit as_of_date
- PIT discipline enforced throughout
- Decimal-only arithmetic
- Stable ordering on all outputs
- Content-hash verification on inputs (if enabled)

Usage:
    python run_screen.py --as-of-date 2024-12-15 --data-dir ./data --output results.json

Architecture:
    Module 1: Universe filtering
    Module 2: Financial health
    Module 3: Catalyst detection (NEW: Delta-based event detection)
    Module 4: Clinical development
    Module 5: Composite ranking
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
import hashlib
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

# Common utilities
from common.date_utils import normalize_date, to_date_string, to_date_object

# Module imports
from module_1_universe import compute_module_1_universe
from module_2_financial import compute_module_2_financial
from module_3_catalyst import compute_module_3_catalyst, Module3Config
from module_4_clinical_dev import compute_module_4_clinical_dev
from module_5_composite_with_defensive import compute_module_5_composite_with_defensive

# Module 3A specific imports
from event_detector import SimpleMarketCalendar

VERSION = "1.1.0"  # Bumped for Module 3A integration
DETERMINISTIC_TIMESTAMP_SUFFIX = "T00:00:00Z"


def _force_deterministic_generated_at(obj: Any, generated_at: str) -> None:
    """Recursively overwrite provenance.generated_at for byte-identical outputs."""
    if isinstance(obj, dict):
        prov = obj.get("provenance")
        if isinstance(prov, dict) and "generated_at" in prov:
            prov["generated_at"] = generated_at
        for v in obj.values():
            _force_deterministic_generated_at(v, generated_at)
    elif isinstance(obj, list):
        for item in obj:
            _force_deterministic_generated_at(item, generated_at)


def validate_as_of_date_param(as_of_date: str) -> None:
    """
    Validate as_of_date format.

    Raises:
        ValueError: If date format is invalid

    Note:
        Does not compare to date.today() to maintain time-invariance.
        Lookahead protection is enforced via PIT filters in modules.
    """
    try:
        normalize_date(as_of_date)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid as_of_date format '{as_of_date}': must be YYYY-MM-DD") from e

    # NOTE: Do not compare to date.today() here (wall-clock dependency breaks time-invariance).
    # Lookahead protection should be enforced via PIT filters and/or input snapshot dating.


def load_json_data(filepath: Path, description: str) -> List[Dict[str, Any]]:
    """
    Load JSON data file with validation.
    
    Args:
        filepath: Path to JSON file
        description: Human-readable description for error messages
    
    Returns:
        List of records
    
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if not filepath.exists():
        raise FileNotFoundError(f"{description} file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"{description} must be a JSON array, got {type(data)}")
    
    return data


def write_json_output(filepath: Path, data: Dict[str, Any]) -> None:
    """
    Write JSON output with deterministic formatting.
    
    Args:
        filepath: Output path
        data: Data to serialize
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(
            data,
            f,
            indent=2,
            sort_keys=True,  # Deterministic key ordering
            ensure_ascii=False,
        )
        f.write('\n')  # Trailing newline for diff-friendliness


def run_screening_pipeline(
    as_of_date: str,
    data_dir: Path,
    universe_tickers: Optional[List[str]] = None,
    enable_coinvest: bool = False,
) -> Dict[str, Any]:
    """
    Execute full screening pipeline with deterministic guarantees.
    
    Args:
        as_of_date: Analysis date (YYYY-MM-DD) - REQUIRED, no defaults
        data_dir: Directory containing input data files
        universe_tickers: Optional whitelist of tickers
        enable_coinvest: Include co-invest overlay (requires coinvest_signals.json)
    
    Returns:
        Complete screening results with provenance
    
    Raises:
        ValueError: If as_of_date is invalid
        FileNotFoundError: If required data files missing
    """
    # CRITICAL: Validate as_of_date FIRST (no implicit defaults)
    validate_as_of_date_param(as_of_date)
    
    print(f"[{as_of_date}] Starting screening pipeline...")
    print(f"  Data directory: {data_dir}")
    
    # Load input data
    print("\n[1/7] Loading input data...")
    raw_universe = load_json_data(data_dir / "universe.json", "Universe")
    financial_records = load_json_data(data_dir / "financial_records.json", "Financial")
    trial_records = load_json_data(data_dir / "trial_records.json", "Trials")
    market_records = load_json_data(data_dir / "market_data.json", "Market data")
    
    coinvest_signals = None
    if enable_coinvest:
        coinvest_file = data_dir / "coinvest_signals.json"
        if coinvest_file.exists():
            print("  Loading co-invest signals...")
            coinvest_signals = load_json_data(coinvest_file, "Co-invest signals")
        else:
            print(f"  Warning: --enable-coinvest specified but {coinvest_file} not found")
    
    # Module 1: Universe filtering
    print("\n[2/7] Module 1: Universe filtering...")
    m1_result = compute_module_1_universe(
        raw_records=raw_universe,
        as_of_date=as_of_date,  # Explicit threading
        universe_tickers=universe_tickers,
    )
    active_tickers = [s["ticker"] for s in m1_result["active_securities"]]
    print(f"  Active: {len(active_tickers)}, Excluded: {len(m1_result['excluded_securities'])}")
    
    # Module 2: Financial health
    print("\n[3/7] Module 2: Financial health...")
    m2_result = compute_module_2_financial(
        financial_records=financial_records,
        active_tickers=set(active_tickers),
        as_of_date=as_of_date,
        raw_universe=raw_universe,
        market_records=market_records,
    )
    diag = m2_result.get('diagnostic_counts', {})
    print(f"  Scored: {diag.get('scored', len(m2_result.get('scores', [])))}, "
          f"Missing: {diag.get('missing', 'N/A')}")
    
    # ========================================================================
    # Module 3: Catalyst Detection (NEW: Delta-based event detection)
    # ========================================================================
    
    print("\n[4/7] Module 3: Catalyst detection...")

    # Convert as_of_date string to date object for Module 3
    as_of_date_obj = to_date_object(as_of_date)
    
    # Create state directory if it doesn't exist
    state_dir = data_dir / "ctgov_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    
    # Run Module 3A with correct signature
    m3_result = compute_module_3_catalyst(
        trial_records_path=data_dir / "trial_records.json",  # Path, not list!
        state_dir=state_dir,  # State directory for snapshots
        active_tickers=set(active_tickers),  # Set of active tickers
        as_of_date=as_of_date_obj,  # Date object
        market_calendar=SimpleMarketCalendar(),  # Market calendar for weekends
        config=Module3Config(),  # Default configuration
        output_dir=data_dir  # Output directory for catalyst_events_*.json
    )
    
    # Extract results (summaries is already a dict keyed by ticker)
    catalyst_summaries = m3_result["summaries"]
    diag3 = m3_result.get("diagnostic_counts", {})
    
    # Print diagnostics
    print(f"  Events detected: {diag3.get('events_detected', 0)}, "
          f"Tickers with events: {diag3.get('tickers_with_events', 0)}/{diag3.get('tickers_analyzed', 0)}, "
          f"Severe negatives: {diag3.get('severe_negatives', 0)}")
    
    # ========================================================================
    # End Module 3
    # ========================================================================
    
    # Module 4: Clinical development
    print("\n[5/7] Module 4: Clinical development...")
    m4_result = compute_module_4_clinical_dev(
        trial_records=trial_records,
        active_tickers=active_tickers,
        as_of_date=as_of_date,  # Explicit threading
    )
    diag = m4_result.get('diagnostic_counts', {})
    print(f"  Scored: {diag.get('scored', len(m4_result.get('scores', [])))}, "
          f"Trials evaluated: {diag.get('total_trials', 'N/A')}, "
          f"PIT filtered: {diag.get('pit_filtered', 'N/A')}")
    
    # Module 5: Composite ranking
    print("\n[6/7] Module 5: Composite ranking...")
    m5_result = compute_module_5_composite_with_defensive(
        universe_result=m1_result,
        financial_result=m2_result,
        catalyst_result=m3_result,
        clinical_result=m4_result,
        as_of_date=as_of_date,  # Explicit threading
        normalization="rank",
        cohort_mode="stage_only",
        coinvest_signals=coinvest_signals,
        validate=True,
    )
    diag = m5_result.get('diagnostic_counts', {})
    print(f"  Rankable: {diag.get('rankable', len(m5_result.get('ranked_securities', [])))}, "
          f"Excluded: {diag.get('excluded', len(m5_result.get('excluded_securities', [])))}")
    
    # Final defensive overlay and top-N selection
    print("\n[7/7] Defensive overlay & top-N selection...")
    # (Assuming this is handled in Module 5 or separately)
    
    # Assemble results
    results = {
        "run_metadata": {
            "as_of_date": as_of_date,
            "version": VERSION,
            "deterministic_timestamp": as_of_date + DETERMINISTIC_TIMESTAMP_SUFFIX,
        },
        "module_1_universe": m1_result,
        "module_2_financial": m2_result,
        "module_3_catalyst": m3_result,
        "module_4_clinical": m4_result,
        "module_5_composite": m5_result,
        "summary": {
            "total_evaluated": len(raw_universe),
            "active_universe": len(active_tickers),
            "excluded": len(m1_result.get('excluded_securities', [])),
            "final_ranked": len(m5_result.get('ranked_securities', [])),
            "catalyst_events": diag3.get('events_detected', 0),
            "severe_negatives": diag3.get('severe_negatives', 0),
        }
    }
    
    # Force deterministic timestamps for byte-identical outputs
    deterministic_ts = as_of_date + DETERMINISTIC_TIMESTAMP_SUFFIX
    _force_deterministic_generated_at(results, deterministic_ts)
    
    return results


def compute_data_hash(data_dir: Path) -> str:
    """Compute hash of input data files for seed derivation."""
    h = hashlib.sha256()
    for json_file in sorted(data_dir.glob("*.json")):
        h.update(json_file.name.encode('utf-8'))
        h.update(json_file.read_bytes())
    return h.hexdigest()[:16]


def add_bootstrap_analysis(
    results: dict,
    as_of_date: str,
    data_dir: Path,
    n_bootstrap: int,
    run_id: str = None,
) -> dict:
    """Add bootstrap confidence intervals to results."""
    from common.random_state import derive_base_seed, DeterministicRNG
    from extensions.bootstrap_scoring import compute_bootstrap_ci_decimal
    
    # Derive deterministic seed
    data_hash = compute_data_hash(data_dir)
    if run_id is None:
        run_id = f"screen_{as_of_date}"
    base_seed = derive_base_seed(as_of_date, run_id, data_hash)
    
    # Create RNG
    rng = DeterministicRNG(base_seed, "bootstrap_ci")
    
    # Get composite scores
    ranked = results["module_5_composite"]["ranked_securities"]
    if not ranked:
        results["bootstrap_analysis"] = {"error": "no_securities_to_bootstrap"}
        return results
    
    scores = [Decimal(s["composite_score"]) for s in ranked]
    
    # Compute bootstrap CI
    bootstrap_result = compute_bootstrap_ci_decimal(
        scores=scores,
        rng=rng,
        n_bootstrap=n_bootstrap,
    )
    
    results["bootstrap_analysis"] = bootstrap_result
    return results


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Wake Robin Biotech Screening Pipeline (Deterministic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic screening
  python run_screen.py --as-of-date 2024-12-15 --data-dir ./data --output results.json
  
  # With co-invest overlay
  python run_screen.py --as-of-date 2024-12-15 --data-dir ./data --output results.json --enable-coinvest
  
  # Custom universe
  python run_screen.py --as-of-date 2024-12-15 --data-dir ./data --output results.json --tickers REGN VRTX ALNY

Determinism guarantees:
  - Identical inputs + as_of_date → identical outputs
  - No today() defaults (as_of_date is required)
  - PIT discipline enforced throughout
  - Stable ordering on all outputs
  
Module 3 Catalyst Detection:
  - Delta-based event detection (compares current vs prior CT.gov state)
  - First run: 0 events (no prior state)
  - Subsequent runs: Events detected if trial data changed
  - IMPORTANT: Refresh trial_records.json from CT.gov before each run!
        """,
    )
    
    parser.add_argument(
        "--as-of-date",
        required=True,
        help="Analysis date (YYYY-MM-DD). REQUIRED - no defaults to prevent nondeterminism.",
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing input data files (universe.json, financial.json, trial_records.json)",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file path",
    )
    
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Optional: Whitelist specific tickers (default: use all from universe.json)",
    )
    
    parser.add_argument(
        "--enable-coinvest",
        action="store_true",
        help="Enable co-invest overlay (requires coinvest_signals.json in data-dir)",
    )
    
    parser.add_argument(
        "--enable-bootstrap",
        action="store_true",
        help="Enable bootstrap confidence intervals (deterministic)",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap samples (default: 1000)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID for seed derivation (default: auto)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    
    args = parser.parse_args()
    
    try:
        # Run pipeline
        results = run_screening_pipeline(
            as_of_date=args.as_of_date,
            data_dir=args.data_dir,
            universe_tickers=args.tickers,
            enable_coinvest=args.enable_coinvest,
        )        
        # Add bootstrap analysis if requested
        if args.enable_bootstrap:
            print("\n[BOOTSTRAP] Computing confidence intervals...")
            results = add_bootstrap_analysis(
                results=results,
                as_of_date=args.as_of_date,
                data_dir=args.data_dir,
                n_bootstrap=args.bootstrap_samples,
                run_id=args.run_id,
            )
            if "error" not in results.get("bootstrap_analysis", {}):
                ba = results["bootstrap_analysis"]
                print(f"  Mean score: {ba['mean']}")
                print(f"  95% CI: [{ba['ci_lower']}, {ba['ci_upper']}]")
                print(f"  Bootstrap samples: {ba['bootstrap_samples']}")

        
        # Write output
        print(f"\n[OUTPUT] Writing results to {args.output}")
        write_json_output(args.output, results)
        
        # Print summary
        summary = results["summary"]
        print(f"\n{'='*60}")
        print(f"SCREENING SUMMARY ({args.as_of_date})")
        print(f"{'='*60}")
        print(f"Total evaluated:    {summary['total_evaluated']}")
        print(f"Active universe:    {summary['active_universe']}")
        print(f"Excluded:           {summary['excluded']}")
        print(f"Final ranked:       {summary['final_ranked']}")
        print(f"Catalyst events:    {summary.get('catalyst_events', 0)}")
        print(f"Severe negatives:   {summary.get('severe_negatives', 0)}")
        print(f"{'='*60}")
        
        return 0
        
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
