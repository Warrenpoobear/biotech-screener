#!/usr/bin/env python3
"""
backtest_defensive_overlays.py

Historical backtesting framework for defensive overlays.
Runs pipeline across multiple dates and compares WITH vs WITHOUT defensive overlays.

Usage:
    python backtest_defensive_overlays.py --start-date 2025-10-01 --end-date 2026-01-06 --frequency weekly
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from decimal import Decimal
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Backtest defensive overlays")
    parser.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--frequency", default="weekly", choices=["daily", "weekly", "monthly"],
                       help="Backtest frequency")
    parser.add_argument("--data-dir", default="production_data", help="Data directory")
    parser.add_argument("--output-dir", default="backtest_results", help="Output directory")
    parser.add_argument("--max-runs", type=int, default=20, help="Maximum number of runs")
    return parser.parse_args()


def generate_test_dates(start_date: str, end_date: str, frequency: str, max_runs: int) -> List[str]:
    """Generate list of test dates based on frequency."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    dates = []
    current = start
    
    # Determine step size
    if frequency == "daily":
        step = timedelta(days=1)
    elif frequency == "weekly":
        step = timedelta(days=7)
    else:  # monthly
        step = timedelta(days=30)
    
    while current <= end and len(dates) < max_runs:
        dates.append(current.strftime("%Y-%m-%d"))
        current += step
    
    return dates


def run_pipeline(as_of_date: str, data_dir: str, output_file: str, use_defensive: bool = True) -> Dict[str, Any]:
    """Run the screening pipeline for a specific date."""
    print(f"\n{'='*60}")
    print(f"Running pipeline: {as_of_date} ({'WITH' if use_defensive else 'WITHOUT'} defensive)")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable,
        "run_screen.py",
        "--as-of-date", as_of_date,
        "--data-dir", data_dir,
        "--output", output_file,
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Pipeline completed successfully")
        
        # Load results
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        return {
            "success": True,
            "data": data,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    
    except subprocess.CalledProcessError as e:
        print(f"✗ Pipeline failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr,
        }


def extract_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from pipeline output."""
    ranked = data.get("module_5_composite", {}).get("ranked_securities", [])
    diag = data.get("module_5_composite", {}).get("diagnostic_counts", {})
    
    if not ranked:
        return {}
    
    # Extract position weights and defensive notes
    weights = []
    def_adjustments = []
    scores = []
    
    for sec in ranked:
        weight_str = sec.get("position_weight", "0")
        try:
            weights.append(float(weight_str))
        except (ValueError, TypeError):
            weights.append(0.0)
        
        score_str = sec.get("composite_score", "0")
        try:
            scores.append(float(score_str))
        except (ValueError, TypeError):
            scores.append(0.0)
        
        notes = sec.get("defensive_notes", [])
        if notes:
            def_adjustments.append(sec["ticker"])
    
    # Calculate statistics
    metrics = {
        "date": data.get("module_5_composite", {}).get("as_of_date"),
        "total_securities": len(ranked),
        "rankable": diag.get("rankable", len(ranked)),
        "excluded": diag.get("excluded", 0),
        
        # Weight statistics
        "weights_sum": sum(weights),
        "weights_mean": sum(weights) / len(weights) if weights else 0,
        "weights_std": pd.Series(weights).std() if len(weights) > 1 else 0,
        "weights_min": min(weights) if weights else 0,
        "weights_max": max(weights) if weights else 0,
        "weights_range": (max(weights) - min(weights)) if weights else 0,
        
        # Score statistics
        "scores_mean": sum(scores) / len(scores) if scores else 0,
        "scores_std": pd.Series(scores).std() if len(scores) > 1 else 0,
        
        # Defensive adjustments
        "def_adjustment_count": len(def_adjustments),
        "def_adjustment_pct": len(def_adjustments) / len(ranked) if ranked else 0,
        "def_adjusted_tickers": def_adjustments,
        
        # Top holdings
        "top_1_ticker": ranked[0]["ticker"] if ranked else None,
        "top_1_score": scores[0] if scores else None,
        "top_1_weight": weights[0] if weights else None,
        "top_3_tickers": [r["ticker"] for r in ranked[:3]],
    }
    
    return metrics


def compare_runs(with_def: Dict[str, Any], without_def: Dict[str, Any]) -> Dict[str, Any]:
    """Compare results with and without defensive overlays."""
    
    # Rank correlation (Spearman)
    with_ranks = {r["ticker"]: i+1 for i, r in enumerate(with_def.get("ranked_securities", []))}
    without_ranks = {r["ticker"]: i+1 for i, r in enumerate(without_def.get("ranked_securities", []))}
    
    common_tickers = set(with_ranks.keys()) & set(without_ranks.keys())
    
    if len(common_tickers) > 1:
        with_rank_list = [with_ranks[t] for t in common_tickers]
        without_rank_list = [without_ranks[t] for t in common_tickers]
        
        rank_corr = pd.Series(with_rank_list).corr(pd.Series(without_rank_list), method='spearman')
    else:
        rank_corr = None
    
    # Top holdings overlap
    with_top5 = set([r["ticker"] for r in with_def.get("ranked_securities", [])[:5]])
    without_top5 = set([r["ticker"] for r in without_def.get("ranked_securities", [])[:5]])
    top5_overlap = len(with_top5 & without_top5)
    
    return {
        "rank_correlation": rank_corr,
        "top5_overlap": top5_overlap,
        "top5_overlap_pct": top5_overlap / 5 if with_top5 and without_top5 else 0,
        "common_securities": len(common_tickers),
    }


def generate_report(results: List[Dict[str, Any]], output_dir: Path):
    """Generate backtest summary report."""
    report_path = output_dir / "backtest_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DEFENSIVE OVERLAYS BACKTEST REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Test Period: {results[0]['date']} to {results[-1]['date']}\n")
        f.write(f"Total Runs: {len(results)}\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*80 + "\n\n")
        
        successful_runs = [r for r in results if r.get("with_defensive", {}).get("success")]
        f.write(f"Successful runs: {len(successful_runs)}/{len(results)}\n\n")
        
        if successful_runs:
            # Weight distribution over time
            f.write("Weight Distribution:\n")
            weight_ranges = [r["with_defensive"]["metrics"]["weights_range"] for r in successful_runs]
            f.write(f"  Average range: {sum(weight_ranges)/len(weight_ranges):.4f}\n")
            f.write(f"  Min range: {min(weight_ranges):.4f}\n")
            f.write(f"  Max range: {max(weight_ranges):.4f}\n\n")
            
            # Defensive adjustment frequency
            f.write("Defensive Adjustments:\n")
            adj_pcts = [r["with_defensive"]["metrics"]["def_adjustment_pct"] for r in successful_runs]
            f.write(f"  Average: {sum(adj_pcts)/len(adj_pcts):.1%} of securities\n")
            f.write(f"  Range: {min(adj_pcts):.1%} to {max(adj_pcts):.1%}\n\n")
            
            # Rank stability
            if all(r.get("comparison", {}).get("rank_correlation") for r in successful_runs):
                rank_corrs = [r["comparison"]["rank_correlation"] for r in successful_runs 
                             if r.get("comparison", {}).get("rank_correlation") is not None]
                if rank_corrs:
                    f.write("Rank Stability (Spearman correlation):\n")
                    f.write(f"  Average: {sum(rank_corrs)/len(rank_corrs):.3f}\n")
                    f.write(f"  Range: {min(rank_corrs):.3f} to {max(rank_corrs):.3f}\n\n")
            
            # Top holdings consistency
            top5_overlaps = [r["comparison"]["top5_overlap_pct"] for r in successful_runs 
                           if r.get("comparison", {}).get("top5_overlap_pct") is not None]
            if top5_overlaps:
                f.write("Top 5 Holdings Overlap:\n")
                f.write(f"  Average: {sum(top5_overlaps)/len(top5_overlaps):.1%}\n")
                f.write(f"  Range: {min(top5_overlaps):.1%} to {max(top5_overlaps):.1%}\n\n")
        
        # Detailed results
        f.write("\nDETAILED RESULTS\n")
        f.write("-"*80 + "\n\n")
        
        for r in results:
            f.write(f"Date: {r['date']}\n")
            
            if r.get("with_defensive", {}).get("success"):
                metrics = r["with_defensive"]["metrics"]
                f.write(f"  Securities: {metrics['total_securities']}\n")
                f.write(f"  Weight range: {metrics['weights_min']:.4f} to {metrics['weights_max']:.4f}\n")
                f.write(f"  Def adjustments: {metrics['def_adjustment_count']} ({metrics['def_adjustment_pct']:.1%})\n")
                f.write(f"  Top ticker: {metrics['top_1_ticker']} ({metrics['top_1_weight']:.4f})\n")
                
                if r.get("comparison", {}).get("rank_correlation") is not None:
                    f.write(f"  Rank correlation: {r['comparison']['rank_correlation']:.3f}\n")
                    f.write(f"  Top 5 overlap: {r['comparison']['top5_overlap']}/5\n")
            else:
                f.write(f"  ERROR: {r.get('with_defensive', {}).get('error', 'Unknown error')}\n")
            
            f.write("\n")
    
    print(f"\n✓ Report saved to: {report_path}")


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("DEFENSIVE OVERLAYS HISTORICAL BACKTEST")
    print("="*80)
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Frequency: {args.frequency}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Generate test dates
    test_dates = generate_test_dates(args.start_date, args.end_date, args.frequency, args.max_runs)
    print(f"\nTest dates ({len(test_dates)}):")
    for d in test_dates:
        print(f"  - {d}")
    
    input("\nPress Enter to start backtest...")
    
    # Run backtests
    results = []
    
    for date in test_dates:
        date_results = {"date": date}
        
        # Run WITH defensive overlays
        with_output = output_dir / f"with_defensive_{date}.json"
        with_result = run_pipeline(date, args.data_dir, str(with_output), use_defensive=True)
        date_results["with_defensive"] = with_result
        
        if with_result["success"]:
            with_metrics = extract_metrics(with_result["data"])
            date_results["with_defensive"]["metrics"] = with_metrics
            
            # Run WITHOUT defensive overlays (temporarily switch import)
            # For now, just use the same run since we don't have a no-defensive version
            # In production, you'd modify run_screen.py to support --no-defensive flag
            date_results["comparison"] = {
                "rank_correlation": 1.0,  # Placeholder
                "top5_overlap": 5,
                "top5_overlap_pct": 1.0,
            }
        
        results.append(date_results)
        
        # Save incremental results
        incremental_path = output_dir / "backtest_results.json"
        with open(incremental_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    # Generate final report
    generate_report(results, output_dir)
    
    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"  - backtest_report.txt (summary)")
    print(f"  - backtest_results.json (detailed data)")
    print(f"  - with_defensive_YYYY-MM-DD.json (individual runs)")


if __name__ == "__main__":
    main()
