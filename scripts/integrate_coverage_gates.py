#!/usr/bin/env python3
"""
Integration script to add coverage gates to screening results.

Usage:
    python scripts/integrate_coverage_gates.py \
        --input results.json \
        --output results_with_coverage.json \
        --include-market-cap  # Optional
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from typing import Dict, Any

from extensions.coverage_gates import calculate_coverage_confidence


def enhance_with_coverage_gates(
    results: Dict[str, Any],
    include_market_cap: bool = False,
) -> Dict[str, Any]:
    """
    Add coverage confidence analysis to existing screening results.
    
    Adds to each security:
    - coverage_confidence: Overall data quality score (0.0-1.0)
    - data_quality_flag: OK / SPARSE_DATA / MISSING_CRITICAL
    - coverage_breakdown: Per-category details
    - critical_missing: List of critical missing fields
    """
    
    # Get ranked securities from Module 5
    if 'module_5_composite' not in results:
        print('Warning: No module_5_composite found in results')
        return results
    
    ranked = results['module_5_composite'].get('ranked_securities', [])
    
    # Stats
    stats = {
        'total': len(ranked),
        'ok': 0,
        'sparse_data': 0,
        'missing_critical': 0,
    }
    
    for security in ranked:
        # Calculate coverage confidence
        coverage_analysis = calculate_coverage_confidence(
            security_record=security,
            include_market_cap=include_market_cap,
        )
        
        # Add to security record
        security['coverage_confidence'] = coverage_analysis['coverage_confidence']
        security['data_quality_flag'] = coverage_analysis['data_quality_flag']
        security['coverage_breakdown'] = coverage_analysis['coverage_breakdown']
        security['critical_missing'] = coverage_analysis['critical_missing']
        security['coverage_audit_hash'] = coverage_analysis['audit_hash']
        
        # Update stats
        flag = coverage_analysis['data_quality_flag']
        if flag == 'OK':
            stats['ok'] += 1
        elif flag == 'SPARSE_DATA':
            stats['sparse_data'] += 1
        elif flag == 'MISSING_CRITICAL':
            stats['missing_critical'] += 1
    
    # Add summary
    results['coverage_gates_summary'] = {
        'total_securities': stats['total'],
        'data_quality_ok': stats['ok'],
        'data_quality_sparse': stats['sparse_data'],
        'data_quality_missing_critical': stats['missing_critical'],
        'ok_percentage': f"{(stats['ok'] / stats['total'] * 100):.1f}%" if stats['total'] > 0 else "N/A",
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Add coverage gates to screening results'
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input screening results JSON',
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output enhanced results JSON',
    )
    parser.add_argument(
        '--include-market-cap',
        action='store_true',
        help='Include market_cap_usd in coverage requirements',
    )
    
    args = parser.parse_args()
    
    # Load inputs
    print(f'Loading screening results from {args.input}...')
    with open(args.input, 'r') as f:
        results = json.load(f)
    
    # Enhance results
    print(f'\nAdding coverage gates analysis...')
    enhanced_results = enhance_with_coverage_gates(
        results,
        include_market_cap=args.include_market_cap,
    )
    
    # Show summary
    if 'coverage_gates_summary' in enhanced_results:
        summary = enhanced_results['coverage_gates_summary']
        print(f'\n📊 Coverage Gates Summary:')
        print(f'   Total Securities: {summary["total_securities"]}')
        print(f'   ✅ OK: {summary["data_quality_ok"]} ({summary["ok_percentage"]})')
        print(f'   ⚠️  SPARSE_DATA: {summary["data_quality_sparse"]}')
        print(f'   🚫 MISSING_CRITICAL: {summary["data_quality_missing_critical"]}')
    
    # Save output
    print(f'\nWriting enhanced results to {args.output}...')
    with open(args.output, 'w') as f:
        json.dump(enhanced_results, f, indent=2)
    
    print(f'\n✅ Coverage gates integration complete!')
    print(f'   Input:  {args.input}')
    print(f'   Output: {args.output}')


if __name__ == '__main__':
    main()