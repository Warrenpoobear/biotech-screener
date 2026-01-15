#!/usr/bin/env python3
"""
Wake Robin - Score Quality Diagnostic
Measures score quantization, tie mass, and universe quality.
"""

import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple


def load_results(filepath: str) -> Dict:
    """Load screening results."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_score_quality(securities: List[Dict]) -> Dict:
    """Analyze score quality metrics."""
    
    # Extract scores
    financial_scores = [float(s.get('financial_normalized', 0)) for s in securities]
    clinical_scores = [float(s.get('clinical_dev_normalized', 0)) for s in securities]
    catalyst_scores = [float(s.get('catalyst_normalized', 50)) for s in securities]
    composite_scores = [float(s.get('composite_score', 0)) for s in securities]
    
    num_tickers = len(securities)
    
    # Calculate uniqueness
    fin_unique = len(set(financial_scores))
    clin_unique = len(set(clinical_scores))
    cat_unique = len(set(catalyst_scores))
    comp_unique = len(set(composite_scores))
    
    # Find most common values
    fin_common = Counter(financial_scores).most_common(5)
    clin_common = Counter(clinical_scores).most_common(5)
    comp_common = Counter(composite_scores).most_common(5)
    
    # Calculate tie mass
    comp_rounded = [round(s, 2) for s in composite_scores]
    ties = Counter(comp_rounded)
    num_tied_scores = sum(1 for count in ties.values() if count > 1)
    max_tie = max(ties.values())
    
    # Score ranges
    results = {
        'num_tickers': num_tickers,
        'financial': {
            'unique': fin_unique,
            'uniqueness_pct': (fin_unique / num_tickers * 100) if num_tickers > 0 else 0,
            'most_common': fin_common,
            'min': min(financial_scores) if financial_scores else 0,
            'max': max(financial_scores) if financial_scores else 0,
        },
        'clinical': {
            'unique': clin_unique,
            'uniqueness_pct': (clin_unique / num_tickers * 100) if num_tickers > 0 else 0,
            'most_common': clin_common,
            'min': min(clinical_scores) if clinical_scores else 0,
            'max': max(clinical_scores) if clinical_scores else 0,
        },
        'catalyst': {
            'unique': cat_unique,
            'uniqueness_pct': (cat_unique / num_tickers * 100) if num_tickers > 0 else 0,
        },
        'composite': {
            'unique': comp_unique,
            'uniqueness_pct': (comp_unique / num_tickers * 100) if num_tickers > 0 else 0,
            'most_common': comp_common,
            'num_tied_scores': num_tied_scores,
            'max_tie_size': max_tie,
            'min': min(composite_scores) if composite_scores else 0,
            'max': max(composite_scores) if composite_scores else 0,
            'range': max(composite_scores) - min(composite_scores) if composite_scores else 0,
        }
    }
    
    return results


def analyze_universe_quality(securities: List[Dict]) -> Dict:
    """Check for universe contamination."""
    
    suspicious_tickers = []
    
    for sec in securities:
        ticker = sec.get('ticker', '')
        
        # Check for suspicious patterns
        is_suspicious = False
        reasons = []
        
        # Too short or generic
        if len(ticker) <= 2:
            is_suspicious = True
            reasons.append("too_short")
        
        # Known non-biotech
        non_biotech = ['USD', 'EUR', 'GBP', 'NAME', 'NA', 'XX', 'CCCC']
        if ticker in non_biotech:
            is_suspicious = True
            reasons.append("blacklisted")
        
        # All numbers (weird)
        if ticker.isdigit():
            is_suspicious = True
            reasons.append("all_digits")
        
        # Mixed case (weird for tickers)
        if not ticker.isupper() and ticker.isalpha():
            is_suspicious = True
            reasons.append("mixed_case")
        
        if is_suspicious:
            suspicious_tickers.append({
                'ticker': ticker,
                'rank': sec.get('composite_rank', 'N/A'),
                'score': float(sec.get('composite_score', 0)),
                'reasons': reasons
            })
    
    return {
        'total_tickers': len(securities),
        'suspicious_count': len(suspicious_tickers),
        'suspicious_pct': (len(suspicious_tickers) / len(securities) * 100) if securities else 0,
        'suspicious_tickers': suspicious_tickers
    }


def grade_score_quality(uniqueness_pct: float, max_tie: int, rank_uniqueness_pct: float = None) -> Tuple[str, int]:
    """Grade score quality - PRIORITIZES rank uniqueness over score uniqueness."""
    
    # PRIMARY: Check rank uniqueness (if available)
    if rank_uniqueness_pct is not None:
        if rank_uniqueness_pct == 100:
            # Perfect ranks - grade on score uniqueness for extra credit
            if uniqueness_pct >= 85:
                return 'A+', 10
            elif uniqueness_pct >= 75:
                return 'A', 9
            elif uniqueness_pct >= 65:
                return 'A-', 8
            else:
                return 'B+', 7  # Even low score uniqueness with perfect ranks = B+
        elif rank_uniqueness_pct >= 95:
            return 'B', 6
        elif rank_uniqueness_pct >= 90:
            return 'C', 5
        else:
            return 'D', 3  # Ranks have significant duplicates
    
    # FALLBACK: Old grading (for backwards compatibility)
    if uniqueness_pct >= 90 and max_tie <= 3:
        return 'A', 9
    elif uniqueness_pct >= 80 and max_tie <= 5:
        return 'B', 7
    elif uniqueness_pct >= 70 and max_tie <= 8:
        return 'C', 6
    elif uniqueness_pct >= 50 and max_tie <= 15:
        return 'D', 4
    else:
        return 'F', 2


def print_report(quality: Dict, universe: Dict, data: Dict = None):
    """Print formatted quality report."""
    
    print("=" * 80)
    print("WAKE ROBIN - SCORE QUALITY DIAGNOSTIC REPORT")
    print("=" * 80)
    print()
    
    # Universe Quality
    print("1. UNIVERSE QUALITY")
    print("-" * 80)
    print(f"   Total tickers: {universe['total_tickers']}")
    print(f"   Suspicious tickers: {universe['suspicious_count']} ({universe['suspicious_pct']:.1f}%)")
    
    if universe['suspicious_tickers']:
        print(f"\n   Top suspicious tickers:")
        for i, sus in enumerate(universe['suspicious_tickers'][:10], 1):
            print(f"      {i}. {sus['ticker']} (Rank {sus['rank']}) - {', '.join(sus['reasons'])}")
        
        if len(universe['suspicious_tickers']) > 10:
            print(f"      ... and {len(universe['suspicious_tickers']) - 10} more")
    
    # Score Uniqueness
    print()
    print("2A. RANK UNIQUENESS (Primary Quality Metric)")
    print("-" * 80)
    
    # Check Module 5 ranked securities for unique ranks
    if 'module_5_composite' in data and 'ranked_securities' in data['module_5_composite']:
        from collections import Counter
        ranked = data['module_5_composite']['ranked_securities']
        ranks = [s['composite_rank'] for s in ranked]
        unique_ranks = len(set(ranks))
        total_ranks = len(ranks)
        rank_uniqueness_pct = (unique_ranks / total_ranks * 100) if total_ranks > 0 else 0
        
        print(f'   Total securities: {total_ranks}')
        print(f'   Unique ranks: {unique_ranks}/{total_ranks} ({rank_uniqueness_pct:.1f}%)')
        
        # Check for duplicate ranks
        rank_counts = Counter(ranks)
        duplicates = [(rank, count) for rank, count in rank_counts.items() if count > 1]
        
        if duplicates:
            print(f'   ??  WARNING: {len(duplicates)} ranks have duplicates!')
            print(f'   Duplicate ranks:')
            for rank, count in sorted(duplicates, key=lambda x: -x[1])[:5]:
                print(f'      Rank {rank}: {count} tickers')
        else:
            print(f'   ? PERFECT: All ranks are unique (tiebreaker working!)')
    else:
        print('   ??  No ranked securities found')
    
    print()
    print("2B. SCORE UNIQUENESS (Secondary - informational only)")
    print("-" * 80)
    print()
    print("2. SCORE UNIQUENESS (Higher is better)")
    print("-" * 80)
    
    fin = quality['financial']
    print(f"   Financial: {fin['unique']}/{quality['num_tickers']} unique ({fin['uniqueness_pct']:.1f}%)")
    
    clin = quality['clinical']
    print(f"   Clinical:  {clin['unique']}/{quality['num_tickers']} unique ({clin['uniqueness_pct']:.1f}%)")
    
    cat = quality['catalyst']
    print(f"   Catalyst:  {cat['unique']}/{quality['num_tickers']} unique ({cat['uniqueness_pct']:.1f}%)")
    
    comp = quality['composite']
    print(f"   Composite: {comp['unique']}/{quality['num_tickers']} unique ({comp['uniqueness_pct']:.1f}%)")
    
    # Bucketing Evidence
    print()
    print("3. BUCKETING EVIDENCE (Repeated values indicate bucketing)")
    print("-" * 80)
    
    print(f"\n   Financial - Top 5 repeated values:")
    for val, count in fin['most_common']:
        pct = (count / quality['num_tickers'] * 100)
        print(f"      {val:.2f} appears {count} times ({pct:.1f}% of universe)")
    
    print(f"\n   Clinical - Top 5 repeated values:")
    for val, count in clin['most_common']:
        pct = (count / quality['num_tickers'] * 100)
        print(f"      {val:.2f} appears {count} times ({pct:.1f}% of universe)")
        if pct > 40:
            print(f"         ⚠️  WARNING: >40% of universe at same score!")
    
    # Tie Mass
    print()
    print("4. TIE MASS (Tickers with identical composite scores)")
    print("-" * 80)
    print(f"   Scores with ties: {comp['num_tied_scores']}")
    print(f"   Largest tie: {comp['max_tie_size']} tickers")
    print(f"   Score range: {comp['min']:.2f} - {comp['max']:.2f} ({comp['range']:.2f} points)")
    
    # Overall Grade
    print()
    print("5. OVERALL QUALITY GRADE")
    print("-" * 80)
    
    # Get rank uniqueness
    rank_uniqueness_pct = None
    if 'module_5_composite' in data and 'ranked_securities' in data['module_5_composite']:
        ranked = data['module_5_composite']['ranked_securities']
        ranks = [s['composite_rank'] for s in ranked]
        rank_uniqueness_pct = (len(set(ranks)) / len(ranks) * 100) if ranks else 0
    
    grade, score = grade_score_quality(comp['uniqueness_pct'], comp['max_tie_size'], rank_uniqueness_pct)
    
    print(f"   Composite Score Quality: {grade} ({score}/10)")
    print(f"   Uniqueness: {comp['uniqueness_pct']:.1f}%")
    print(f"   Max tie size: {comp['max_tie_size']}")
    print()
    
    # Interpretation
    if grade in ['A+', 'A', 'A-']:
        print("   ? EXCELLENT - Perfect rank uniqueness, production ready!")
    elif grade in ['B+', 'B']:
        print("   ? GOOD - High resolution, minimal tie issues")
    elif grade == 'C':
        print("   ??  ACCEPTABLE - Some rank ties, room for improvement")
    elif grade == 'D':
        print("   ??  POOR - Significant rank ties, limited resolution")
    else:
        print("   ? CRITICAL - Severe rank ties, system needs tiebreaker!")
        print("      Action required: Implement deterministic tiebreaker!")
    print("6. RECOMMENDATIONS")
    print("-" * 80)
    
    # Specific recommendations
    if universe['suspicious_pct'] > 1:
        print("   🔧 Clean universe - remove suspicious tickers")
    
    if clin['uniqueness_pct'] < 50:
        print("   🔧 Add clinical sub-scoring to break up stage buckets")
    
    if fin['uniqueness_pct'] < 50:
        print("   🔧 Use continuous runway calculation instead of buckets")
    
    if comp['max_tie_size'] > 10:
        print("   🔧 Increase score resolution - too many ties")
    
    if comp['range'] < 50:
        print("   🔧 Expand score range - currently compressed")
    
    print()
    print("=" * 80)


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnose score quality issues')
    parser.add_argument('--file', '-f', type=str, help='Results JSON file (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Find results file
    if args.file:
        filepath = args.file
    else:
        # Try common names
        candidates = ['test_no_overlay.json', 'results_*.json']
        filepath = None
        for pattern in candidates:
            import glob
            files = glob.glob(pattern)
            if files:
                filepath = sorted(files)[-1]  # Most recent
                break
        
        if not filepath:
            print("Error: No results file found!")
            print("Usage: python score_quality_diagnostic.py --file results.json")
            sys.exit(1)
    
    print(f"Analyzing: {filepath}\n")
    
    # Load data
    try:
        data = load_results(filepath)
        securities = data['module_5_composite']['ranked_securities']
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    
    # Analyze
    quality = analyze_score_quality(securities)
    universe = analyze_universe_quality(securities)
    
    # Report
    print_report(quality, universe, data)


if __name__ == '__main__':
    main()
