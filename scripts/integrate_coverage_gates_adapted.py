#!/usr/bin/env python3
"""
Adapted integration script using score-based coverage gates.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from typing import Dict, Any

from extensions.coverage_gates_adapted import calculate_coverage_confidence_adapted


def enhance_with_adapted_coverage_gates(results: Dict[str, Any]) -> Dict[str, Any]:
    """Add coverage confidence using adapted logic for score-based pipeline."""
    
    if 'module_5_composite' not in results:
        print('Warning: No module_5_composite found')
        return results
    
    ranked = results['module_5_composite'].get('ranked_securities', [])
    
    stats = {'total': len(ranked), 'ok': 0, 'sparse_data': 0, 'missing_critical': 0}
    
    for security in ranked:
        coverage = calculate_coverage_confidence_adapted(security)
        
        # Update security with adapted coverage
        security['coverage_confidence'] = coverage['coverage_confidence']
        security['data_quality_flag'] = coverage['data_quality_flag']
        security['coverage_breakdown'] = coverage['coverage_breakdown']
        security['coverage_audit_hash'] = coverage['audit_hash']
        
        # Update stats
        flag = coverage['data_quality_flag']
        if flag == 'OK':
            stats['ok'] += 1
        elif flag == 'SPARSE_DATA':
            stats['sparse_data'] += 1
        elif flag == 'MISSING_CRITICAL':
            stats['missing_critical'] += 1
    
    results['coverage_gates_summary'] = {
        'total_securities': stats['total'],
        'data_quality_ok': stats['ok'],
        'data_quality_sparse': stats['sparse_data'],
        'data_quality_missing_critical': stats['missing_critical'],
        'ok_percentage': f"{(stats['ok'] / stats['total'] * 100):.1f}%" if stats['total'] > 0 else "N/A",
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Add adapted coverage gates')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    
    print(f'Loading results from {args.input}...')
    with open(args.input, 'r') as f:
        results = json.load(f)
    
    print('\nAdding adapted coverage gates...')
    enhanced = enhance_with_adapted_coverage_gates(results)
    
    if 'coverage_gates_summary' in enhanced:
        summary = enhanced['coverage_gates_summary']
        print(f'\n📊 Coverage Summary:')
        print(f'   Total: {summary["total_securities"]}')
        print(f'   ✅ OK: {summary["data_quality_ok"]} ({summary["ok_percentage"]})')
        print(f'   ⚠️  SPARSE: {summary["data_quality_sparse"]}')
        print(f'   🚫 MISSING: {summary["data_quality_missing_critical"]}')
    
    print(f'\nWriting to {args.output}...')
    with open(args.output, 'w') as f:
        json.dump(enhanced, f, indent=2)
    
    print('\n✅ Adapted coverage gates complete!')


if __name__ == '__main__':
    main()