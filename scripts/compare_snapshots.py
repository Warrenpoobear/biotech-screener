#!/usr/bin/env python3
"""
Compare two snapshots and generate a diff report.

Usage:
    python scripts/compare_snapshots.py \\
        --baseline snapshots/stub/2024-01-31.json \\
        --compare snapshots/aact/2024-01-31.json \\
        --output output/compare/2024-01-31_stub_vs_aact/
"""

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional


@dataclass
class ComparisonResult:
    """Result of comparing two snapshots."""
    baseline_path: str
    compare_path: str
    baseline_provider: str
    compare_provider: str
    baseline_as_of: str
    compare_as_of: str
    
    # Coverage comparison
    baseline_coverage: dict[str, float]
    compare_coverage: dict[str, float]
    coverage_delta: dict[str, float]
    
    # Per-ticker differences
    tickers_only_in_baseline: list[str]
    tickers_only_in_compare: list[str]
    tickers_in_both: list[str]
    
    # Trial data differences
    trial_count_changes: dict[str, dict[str, int]]  # ticker -> {baseline, compare, delta}
    tickers_gained_trials: list[str]
    tickers_lost_trials: list[str]
    
    # Summary stats
    total_trials_baseline: int
    total_trials_compare: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": {
                "baseline_path": self.baseline_path,
                "compare_path": self.compare_path,
                "baseline_provider": self.baseline_provider,
                "compare_provider": self.compare_provider,
                "baseline_as_of": self.baseline_as_of,
                "compare_as_of": self.compare_as_of,
            },
            "coverage": {
                "baseline": self.baseline_coverage,
                "compare": self.compare_coverage,
                "delta": self.coverage_delta,
            },
            "ticker_summary": {
                "only_in_baseline": self.tickers_only_in_baseline,
                "only_in_compare": self.tickers_only_in_compare,
                "in_both": len(self.tickers_in_both),
            },
            "trial_changes": {
                "total_baseline": self.total_trials_baseline,
                "total_compare": self.total_trials_compare,
                "tickers_gained_trials": self.tickers_gained_trials,
                "tickers_lost_trials": self.tickers_lost_trials,
                "per_ticker": self.trial_count_changes,
            },
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
    
    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 60)
        print("SNAPSHOT COMPARISON SUMMARY")
        print("=" * 60)
        
        print(f"\nBaseline: {self.baseline_path}")
        print(f"  Provider: {self.baseline_provider}")
        print(f"  As-of: {self.baseline_as_of}")
        
        print(f"\nCompare: {self.compare_path}")
        print(f"  Provider: {self.compare_provider}")
        print(f"  As-of: {self.compare_as_of}")
        
        print("\n--- Coverage ---")
        for module in self.baseline_coverage:
            base = self.baseline_coverage.get(module, 0)
            comp = self.compare_coverage.get(module, 0)
            delta = self.coverage_delta.get(module, 0)
            print(f"  {module}: {base:.1%} -> {comp:.1%} (Δ{delta:+.1%})")
        
        print("\n--- Trial Counts ---")
        print(f"  Total trials: {self.total_trials_baseline} -> {self.total_trials_compare}")
        print(f"  Tickers gained trials: {len(self.tickers_gained_trials)}")
        print(f"  Tickers lost trials: {len(self.tickers_lost_trials)}")
        
        if self.tickers_gained_trials:
            print(f"\n  Gained trials: {', '.join(sorted(self.tickers_gained_trials)[:10])}")
            if len(self.tickers_gained_trials) > 10:
                print(f"    ...and {len(self.tickers_gained_trials) - 10} more")
        
        print("\n" + "=" * 60)


def load_snapshot(path: Path) -> dict:
    """Load snapshot from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_snapshots(baseline_path: Path, compare_path: Path) -> ComparisonResult:
    """
    Compare two snapshots and return structured diff.
    
    Args:
        baseline_path: Path to baseline snapshot JSON
        compare_path: Path to comparison snapshot JSON
    
    Returns:
        ComparisonResult with all differences
    """
    baseline = load_snapshot(baseline_path)
    compare = load_snapshot(compare_path)
    
    # Extract provider info
    baseline_provider = baseline.get("provenance", {}).get("providers", {}).get("clinical", {}).get("name", "unknown")
    compare_provider = compare.get("provenance", {}).get("providers", {}).get("clinical", {}).get("name", "unknown")
    
    # Coverage comparison
    baseline_coverage = {
        k: v.get("coverage_rate", 0) 
        for k, v in baseline.get("coverage", {}).items()
    }
    compare_coverage = {
        k: v.get("coverage_rate", 0) 
        for k, v in compare.get("coverage", {}).items()
    }
    coverage_delta = {
        k: compare_coverage.get(k, 0) - baseline_coverage.get(k, 0)
        for k in set(baseline_coverage.keys()) | set(compare_coverage.keys())
    }
    
    # Ticker sets
    baseline_tickers = set(baseline.get("tickers", {}).keys())
    compare_tickers = set(compare.get("tickers", {}).keys())
    
    tickers_only_in_baseline = sorted(baseline_tickers - compare_tickers)
    tickers_only_in_compare = sorted(compare_tickers - baseline_tickers)
    tickers_in_both = sorted(baseline_tickers & compare_tickers)
    
    # Trial count comparison
    trial_count_changes: dict[str, dict[str, int]] = {}
    tickers_gained_trials: list[str] = []
    tickers_lost_trials: list[str] = []
    total_trials_baseline = 0
    total_trials_compare = 0
    
    all_tickers = sorted(baseline_tickers | compare_tickers)
    
    for ticker in all_tickers:
        baseline_data = baseline.get("tickers", {}).get(ticker, {})
        compare_data = compare.get("tickers", {}).get(ticker, {})
        
        baseline_count = baseline_data.get("trial_count", 0)
        compare_count = compare_data.get("trial_count", 0)
        delta = compare_count - baseline_count
        
        total_trials_baseline += baseline_count
        total_trials_compare += compare_count
        
        if delta != 0:
            trial_count_changes[ticker] = {
                "baseline": baseline_count,
                "compare": compare_count,
                "delta": delta,
            }
            
            if delta > 0:
                tickers_gained_trials.append(ticker)
            else:
                tickers_lost_trials.append(ticker)
    
    return ComparisonResult(
        baseline_path=str(baseline_path),
        compare_path=str(compare_path),
        baseline_provider=baseline_provider,
        compare_provider=compare_provider,
        baseline_as_of=baseline.get("as_of_date", "unknown"),
        compare_as_of=compare.get("as_of_date", "unknown"),
        baseline_coverage=baseline_coverage,
        compare_coverage=compare_coverage,
        coverage_delta=coverage_delta,
        tickers_only_in_baseline=tickers_only_in_baseline,
        tickers_only_in_compare=tickers_only_in_compare,
        tickers_in_both=tickers_in_both,
        trial_count_changes=trial_count_changes,
        tickers_gained_trials=tickers_gained_trials,
        tickers_lost_trials=tickers_lost_trials,
        total_trials_baseline=total_trials_baseline,
        total_trials_compare=total_trials_compare,
    )


def save_comparison(result: ComparisonResult, output_dir: Path) -> Path:
    """
    Save comparison result to output directory.
    
    Args:
        result: Comparison result
        output_dir: Output directory
    
    Returns:
        Path to saved comparison JSON
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "comparison.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result.to_json())
    
    return output_file


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare two biotech alpha snapshots",
    )
    
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline snapshot JSON",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        required=True,
        help="Path to comparison snapshot JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for comparison results",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.baseline.exists():
        raise FileNotFoundError(f"Baseline snapshot not found: {args.baseline}")
    if not args.compare.exists():
        raise FileNotFoundError(f"Compare snapshot not found: {args.compare}")
    
    # Compare
    result = compare_snapshots(args.baseline, args.compare)
    
    # Print summary
    result.print_summary()
    
    # Save
    output_file = save_comparison(result, args.output)
    print(f"\nComparison saved to: {output_file}")


if __name__ == "__main__":
    main()
