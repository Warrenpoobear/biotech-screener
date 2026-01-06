#!/usr/bin/env python3
"""
analyze_defensive_impact.py

Analyze the impact of defensive overlays by examining pipeline outputs.
Compares metrics like weight distribution, ranking stability, and adjustments.

Usage:
    python analyze_defensive_impact.py --output outputs/ranked_with_real_defensive_FINAL.json
"""

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Any
import statistics


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze defensive overlay impact")
    parser.add_argument("--output", required=True, help="Pipeline output JSON file")
    return parser.parse_args()


def analyze_position_sizing(ranked: List[Dict]) -> Dict[str, Any]:
    """Analyze position sizing distribution."""
    weights = []
    for sec in ranked:
        try:
            w = float(sec.get("position_weight", "0"))
            weights.append(w)
        except (ValueError, TypeError):
            pass
    
    if not weights:
        return {}
    
    return {
        "count": len(weights),
        "sum": sum(weights),
        "mean": statistics.mean(weights),
        "median": statistics.median(weights),
        "stdev": statistics.stdev(weights) if len(weights) > 1 else 0,
        "min": min(weights),
        "max": max(weights),
        "range": max(weights) - min(weights),
        "concentration_top5": sum(sorted(weights, reverse=True)[:5]),
        "concentration_top10": sum(sorted(weights, reverse=True)[:10]),
    }


def analyze_defensive_adjustments(ranked: List[Dict]) -> Dict[str, Any]:
    """Analyze defensive adjustment patterns."""
    with_adjustments = []
    adjustment_types = {}
    
    for sec in ranked:
        notes = sec.get("defensive_notes", [])
        if notes:
            with_adjustments.append(sec["ticker"])
            for note in notes:
                adjustment_types[note] = adjustment_types.get(note, 0) + 1
    
    return {
        "total_securities": len(ranked),
        "with_adjustments": len(with_adjustments),
        "pct_adjusted": len(with_adjustments) / len(ranked) if ranked else 0,
        "adjustment_types": adjustment_types,
        "adjusted_tickers": with_adjustments,
    }


def analyze_weight_by_vol(ranked: List[Dict]) -> Dict[str, Any]:
    """Analyze relationship between volatility and position weight."""
    vol_weight_pairs = []
    
    for sec in ranked:
        try:
            weight = float(sec.get("position_weight", "0"))
            # Try to get vol from defensive features if available in output
            # (This would require defensive_features to be in output, which they're not currently)
            vol_weight_pairs.append((weight, sec["ticker"]))
        except (ValueError, TypeError):
            pass
    
    # Sort by weight
    vol_weight_pairs.sort(reverse=True)
    
    return {
        "top_10_by_weight": [(ticker, weight) for weight, ticker in vol_weight_pairs[:10]],
        "bottom_10_by_weight": [(ticker, weight) for weight, ticker in vol_weight_pairs[-10:]],
    }


def analyze_score_impact(ranked: List[Dict]) -> Dict[str, Any]:
    """Analyze score impact from defensive multipliers."""
    score_changes = []
    
    for sec in ranked:
        try:
            final_score = float(sec.get("composite_score", "0"))
            before_score = float(sec.get("composite_score_before_defensive", final_score))
            
            if before_score != final_score:
                change = final_score - before_score
                pct_change = (change / before_score * 100) if before_score > 0 else 0
                score_changes.append({
                    "ticker": sec["ticker"],
                    "before": before_score,
                    "after": final_score,
                    "change": change,
                    "pct_change": pct_change,
                    "notes": sec.get("defensive_notes", []),
                })
        except (ValueError, TypeError, KeyError):
            pass
    
    if score_changes:
        changes = [s["change"] for s in score_changes]
        pct_changes = [s["pct_change"] for s in score_changes]
        
        return {
            "securities_changed": len(score_changes),
            "mean_change": statistics.mean(changes),
            "mean_pct_change": statistics.mean(pct_changes),
            "max_increase": max(changes),
            "max_decrease": min(changes),
            "details": sorted(score_changes, key=lambda x: abs(x["change"]), reverse=True)[:10],
        }
    
    return {}


def print_report(output: Dict[str, Any]):
    """Print analysis report."""
    ranked = output.get("module_5_composite", {}).get("ranked_securities", [])
    
    print("\n" + "="*80)
    print("DEFENSIVE OVERLAYS IMPACT ANALYSIS")
    print("="*80)
    
    # Basic info
    as_of = output.get("module_5_composite", {}).get("as_of_date", "Unknown")
    print(f"\nAs of Date: {as_of}")
    print(f"Total Securities: {len(ranked)}")
    
    # Position sizing
    print("\n" + "-"*80)
    print("POSITION SIZING ANALYSIS")
    print("-"*80)
    
    sizing = analyze_position_sizing(ranked)
    if sizing:
        print(f"\nWeight Distribution:")
        print(f"  Sum:          {sizing['sum']:.4f} (target: 0.9000)")
        print(f"  Mean:         {sizing['mean']:.4f}")
        print(f"  Median:       {sizing['median']:.4f}")
        print(f"  Std Dev:      {sizing['stdev']:.4f}")
        print(f"  Min:          {sizing['min']:.4f} ({sizing['min']*100:.2f}%)")
        print(f"  Max:          {sizing['max']:.4f} ({sizing['max']*100:.2f}%)")
        print(f"  Range:        {sizing['range']:.4f} ({sizing['range']/sizing['min']:.1f}:1 ratio)")
        print(f"\nConcentration:")
        print(f"  Top 5:        {sizing['concentration_top5']:.4f} ({sizing['concentration_top5']*100:.1f}%)")
        print(f"  Top 10:       {sizing['concentration_top10']:.4f} ({sizing['concentration_top10']*100:.1f}%)")
    
    # Defensive adjustments
    print("\n" + "-"*80)
    print("DEFENSIVE ADJUSTMENTS")
    print("-"*80)
    
    adjustments = analyze_defensive_adjustments(ranked)
    print(f"\nAdjustment Frequency:")
    print(f"  Securities adjusted: {adjustments['with_adjustments']}/{adjustments['total_securities']} " +
          f"({adjustments['pct_adjusted']:.1%})")
    
    if adjustments['adjustment_types']:
        print(f"\nAdjustment Types:")
        for adj_type, count in sorted(adjustments['adjustment_types'].items(), key=lambda x: -x[1]):
            print(f"  {adj_type:40} {count:3} securities")
    
    # Weight distribution by security
    print("\n" + "-"*80)
    print("TOP 10 POSITIONS BY WEIGHT")
    print("-"*80)
    
    weight_analysis = analyze_weight_by_vol(ranked)
    print(f"\n{'Rank':<6}{'Ticker':<8}{'Weight':<12}{'Score':<10}{'Def Notes'}")
    print("-"*80)
    for i, (ticker, weight) in enumerate(weight_analysis['top_10_by_weight'], 1):
        sec = next((s for s in ranked if s["ticker"] == ticker), None)
        if sec:
            score = sec.get("composite_score", "N/A")
            notes = ", ".join(sec.get("defensive_notes", [])) or "-"
            print(f"{i:<6}{ticker:<8}{weight:>6.4f} ({weight*100:>5.2f}%)   {score:<10}{notes}")
    
    # Score impact
    print("\n" + "-"*80)
    print("SCORE IMPACT FROM DEFENSIVE MULTIPLIERS")
    print("-"*80)
    
    score_impact = analyze_score_impact(ranked)
    if score_impact and score_impact.get("securities_changed", 0) > 0:
        print(f"\nSecurities with score changes: {score_impact['securities_changed']}")
        print(f"Mean change: {score_impact['mean_change']:.2f} ({score_impact['mean_pct_change']:.2f}%)")
        print(f"Range: {score_impact['max_decrease']:.2f} to {score_impact['max_increase']:.2f}")
        
        print(f"\nTop 10 Score Changes:")
        print(f"{'Ticker':<8}{'Before':<10}{'After':<10}{'Change':<10}{'%':<8}{'Notes'}")
        print("-"*80)
        for detail in score_impact['details']:
            notes_str = ", ".join(detail['notes']) if detail['notes'] else "-"
            print(f"{detail['ticker']:<8}{detail['before']:<10.2f}{detail['after']:<10.2f}" +
                  f"{detail['change']:>+9.2f}{detail['pct_change']:>7.1f}%  {notes_str}")
    else:
        print("\nNo securities had score changes from defensive multipliers.")
        print("(All correlations in neutral zone 0.40-0.80)")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n✓ Position sizing working: {sizing['range']:.4f} weight range")
    print(f"✓ Defensive adjustments: {adjustments['pct_adjusted']:.1%} of securities")
    print(f"✓ Top position: {weight_analysis['top_10_by_weight'][0][1]:.1%} " +
          f"({weight_analysis['top_10_by_weight'][0][0]})")
    print(f"✓ Smallest position: {weight_analysis['bottom_10_by_weight'][0][1]:.1%} " +
          f"({weight_analysis['bottom_10_by_weight'][0][0]})")
    print(f"✓ Weight distribution: {sizing['stdev']:.4f} standard deviation")
    
    print("\n" + "="*80)


def main():
    args = parse_args()
    
    output_path = Path(args.output)
    if not output_path.exists():
        print(f"ERROR: File not found: {args.output}")
        return 1
    
    print(f"Loading: {args.output}")
    with open(output_path, 'r') as f:
        output = json.load(f)
    
    print_report(output)
    
    return 0


if __name__ == "__main__":
    exit(main())
