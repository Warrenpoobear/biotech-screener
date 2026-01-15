"""
Clinical Scoring Comparison: Before vs After Confidence Gating

Run this to see how confidence gating changes clinical scores.
"""

import sys
from pathlib import Path

# Example clinical data scenarios
examples = [
    {
        'ticker': 'SNY',
        'name': 'Sanofi (Big Pharma)',
        'trials': 97,
        'stage': 'COMMERCIAL'
    },
    {
        'ticker': 'VRTX',
        'name': 'Vertex (Large Biotech)',
        'trials': 45,
        'stage': 'PHASE3'
    },
    {
        'ticker': 'CRSP',
        'name': 'CRISPR (Mid-cap)',
        'trials': 8,
        'stage': 'PHASE2'
    },
    {
        'ticker': 'XERS',
        'name': 'Xeris (Small Biotech)',
        'trials': 1,
        'stage': 'PHASE2'
    },
    {
        'ticker': 'PAHC',
        'name': 'Phibro Animal Health',
        'trials': 0,
        'stage': 'UNKNOWN'
    }
]


def get_old_score(stage: str) -> int:
    """Old system: No confidence gating."""
    scores = {
        'COMMERCIAL': 10,
        'PHASE3': 20,
        'PHASE2': 30,
        'PHASE1': 40,
        'UNKNOWN': 30
    }
    return scores.get(stage, 30)


def get_confidence(trial_count: int) -> tuple:
    """New system: Confidence based on trial count."""
    if trial_count >= 10:
        return 'HIGH', 1.0
    elif trial_count >= 3:
        return 'MEDIUM', 0.5
    elif trial_count >= 1:
        return 'LOW', 0.25
    else:
        return 'UNKNOWN', 0.0


def get_new_score(stage: str, trial_count: int) -> tuple:
    """New system: Confidence-weighted scoring."""
    raw_score = get_old_score(stage)
    confidence, weight = get_confidence(trial_count)
    
    if confidence == 'UNKNOWN':
        weighted_score = 30  # Neutral
    else:
        # Blend toward neutral based on confidence
        weighted_score = (raw_score * weight) + (30 * (1 - weight))
    
    return confidence, weight, raw_score, weighted_score


def print_comparison():
    """Print before/after comparison."""
    
    print("=" * 80)
    print("CLINICAL SCORING COMPARISON: Before vs After Confidence Gating")
    print("=" * 80)
    print()
    print("Lower score = better (penalty-style scoring)")
    print()
    
    # Header
    print(f"{'Company':<25} {'Trials':<8} {'Stage':<12} {'Old':<6} {'New':<6} {'Conf':<8} {'Change':<8}")
    print("-" * 80)
    
    for ex in examples:
        old_score = get_old_score(ex['stage'])
        confidence, weight, raw, new_score = get_new_score(ex['stage'], ex['trials'])
        
        change = new_score - old_score
        change_str = f"{change:+.1f}"
        
        # Format confidence with icon
        conf_icons = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🟠', 'UNKNOWN': '⚪'}
        conf_str = f"{conf_icons[confidence]} {confidence}"
        
        print(f"{ex['name']:<25} {ex['trials']:<8} {ex['stage']:<12} {old_score:<6} {new_score:<6.1f} {conf_str:<15} {change_str:<8}")
    
    print()
    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print("✅ HIGH confidence (≥10 trials): Full signal strength")
    print("   → Sanofi, Vertex keep strong scores (no change)")
    print()
    print("⚠️  MEDIUM confidence (3-9 trials): Half signal strength")
    print("   → CRISPR score moves toward neutral (+0.0)")
    print()
    print("⚠️  LOW confidence (1-2 trials): Quarter signal strength")
    print("   → Xeris score moves strongly toward neutral (+0.0)")
    print()
    print("⚪ UNKNOWN (0 trials): No signal (neutral)")
    print("   → Phibro stays neutral (0.0)")
    print()
    print("=" * 80)
    print("IMPACT ON RANKINGS")
    print("=" * 80)
    print()
    print("BEFORE (no gating):")
    print("  Top picks: Big pharma (many trials, COMMERCIAL)")
    print("  Problem: Sanofi with 97 trials ranked same as focused biotech with 10 trials")
    print()
    print("AFTER (with gating):")
    print("  Top picks: Still favor strong clinical programs, but...")
    print("  - HIGH confidence (10+ trials) keeps full weight")
    print("  - LOW confidence (1-2 trials) gets reduced impact")
    print("  - UNKNOWN (0 trials) truly neutral")
    print()
    print("Expected result:")
    print("  - Less bias toward mega-pharma")
    print("  - More balanced weighting of clinical data")
    print("  - Better performance in 2024 mature bull markets")
    print()


if __name__ == '__main__':
    print_comparison()
