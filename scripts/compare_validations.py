#!/usr/bin/env python3
"""
Validation Comparison Tool

Compares before/after validation results for clinical coverage expansion.
Tracks pending quarters and generates comparison reports.

Usage:
    python compare_validations.py --show-pending
    python compare_validations.py --output report.md
    python compare_validations.py --run-pending
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Baseline results (13% clinical coverage)
BASELINE_RESULTS = {
    "2023-Q1": {
        "screen_date": "2023-01-15",
        "q1_q5_spread": 8.66,
        "q1_return": 12.42,
        "q5_return": 3.76,
        "alpha": 12.18,
        "clinical_coverage": 13.0
    },
    "2023-Q2": {
        "screen_date": "2023-04-15",
        "q1_q5_spread": 3.79,
        "q1_return": 3.52,
        "q5_return": -0.27,
        "alpha": 7.66,
        "clinical_coverage": 13.0
    },
    "2023-Q3": {
        "screen_date": "2023-07-15",
        "q1_q5_spread": 12.97,
        "q1_return": 12.53,
        "q5_return": -0.44,
        "alpha": 5.29,
        "clinical_coverage": 13.0
    },
    "2023-Q4": {
        "screen_date": "2023-10-15",
        "q1_q5_spread": 18.43,
        "q1_return": 42.67,
        "q5_return": 24.23,
        "alpha": 12.58,
        "clinical_coverage": 13.0
    },
    "2024-Q1": {
        "screen_date": "2024-01-15",
        "q1_q5_spread": 13.19,
        "q1_return": 19.21,
        "q5_return": 6.02,
        "alpha": 19.21,
        "clinical_coverage": 13.0
    },
    "2024-Q2": {
        "screen_date": "2024-04-15",
        "q1_q5_spread": -14.53,
        "q1_return": 1.64,
        "q5_return": 16.16,
        "alpha": -2.54,
        "clinical_coverage": 13.0
    },
    "2024-Q3": {
        "screen_date": "2024-07-15",
        "q1_q5_spread": 22.62,
        "q1_return": 26.78,
        "q5_return": 4.16,
        "alpha": 29.57,
        "clinical_coverage": 13.0
    }
}

# Enhanced results (75%+ clinical coverage) - populated as validations complete
ENHANCED_RESULTS = {
    "2023-Q1": {
        "screen_date": "2023-01-15",
        "q1_q5_spread": 13.60,
        "q1_return": 17.05,
        "q5_return": 3.45,
        "alpha": 16.81,
        "clinical_coverage": 75.3
    }
}


class ValidationComparison:
    """Compare validation results before and after clinical coverage expansion."""

    def __init__(self):
        self.baseline = BASELINE_RESULTS
        self.enhanced = ENHANCED_RESULTS
        self.results_file = Path("data/validation_comparison.json")
        self._load_results()

    def _load_results(self):
        """Load any saved enhanced results."""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                    self.enhanced.update(saved.get('enhanced', {}))
            except Exception as e:
                print(f"Warning: Could not load saved results: {e}")

    def _save_results(self):
        """Save enhanced results to file."""
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'baseline': self.baseline,
                'enhanced': self.enhanced,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)

    def add_enhanced_result(self, quarter: str, result: Dict):
        """Add a new enhanced validation result."""
        self.enhanced[quarter] = result
        self._save_results()

    def get_pending_quarters(self) -> List[Tuple[str, str]]:
        """Get list of quarters pending re-validation."""
        pending = []
        for quarter, data in self.baseline.items():
            if quarter not in self.enhanced:
                pending.append((quarter, data['screen_date']))
        return sorted(pending, key=lambda x: x[1])

    def get_completed_quarters(self) -> List[str]:
        """Get list of completed re-validations."""
        return sorted([q for q in self.enhanced.keys()])

    def calculate_improvement(self, quarter: str) -> Optional[Dict]:
        """Calculate improvement for a specific quarter."""
        if quarter not in self.baseline or quarter not in self.enhanced:
            return None

        before = self.baseline[quarter]
        after = self.enhanced[quarter]

        return {
            'quarter': quarter,
            'spread_before': before['q1_q5_spread'],
            'spread_after': after['q1_q5_spread'],
            'spread_change': after['q1_q5_spread'] - before['q1_q5_spread'],
            'alpha_before': before.get('alpha', 0),
            'alpha_after': after.get('alpha', 0),
            'alpha_change': after.get('alpha', 0) - before.get('alpha', 0),
            'coverage_before': before['clinical_coverage'],
            'coverage_after': after['clinical_coverage']
        }

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate comparison report in markdown format."""
        lines = []
        lines.append("# Wake Robin Validation Comparison Report\n")
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        lines.append("---\n")

        # Enhancement description
        lines.append("## Enhancement: Clinical Coverage Expansion")
        lines.append("- **Baseline**: 13% clinical coverage (41/316 tickers)")
        lines.append("- **Enhanced**: 75%+ clinical coverage (238/316 tickers)")
        lines.append("- **Method**: Expanded ticker -> sponsor name mapping\n")

        # Validation status
        pending = self.get_pending_quarters()
        completed = self.get_completed_quarters()

        lines.append("## Validation Status\n")
        lines.append(f"- **Total Test Periods**: {len(self.baseline)}")
        lines.append(f"- **Baseline Complete**: {len(self.baseline)} [DONE]")
        lines.append(f"- **Enhanced Complete**: {len(completed)}")
        lines.append(f"- **Pending**: {len(pending)}\n")

        if pending:
            lines.append("### Quarters Pending Re-validation:")
            for quarter, date in pending:
                lines.append(f"- [ ] {quarter} ({date})")
            lines.append("")

        # Quarter-by-quarter comparison table
        lines.append("## Quarter-by-Quarter Comparison\n")
        lines.append("| Quarter | Screen Date | Q1-Q5 Spread (Before) | Q1-Q5 Spread (After) | Change | Q1 Alpha (Before) | Q1 Alpha (After) | Change |")
        lines.append("|---------|-------------|------------------------|----------------------|--------|-------------------|------------------|--------|")

        for quarter in sorted(self.baseline.keys()):
            before = self.baseline[quarter]
            after = self.enhanced.get(quarter)

            spread_before = f"+{before['q1_q5_spread']:.2f}%" if before['q1_q5_spread'] >= 0 else f"{before['q1_q5_spread']:.2f}%"
            alpha_before = f"+{before.get('alpha', 0):.2f}%" if before.get('alpha', 0) >= 0 else f"{before.get('alpha', 0):.2f}%"

            if after:
                spread_after = f"+{after['q1_q5_spread']:.2f}%" if after['q1_q5_spread'] >= 0 else f"{after['q1_q5_spread']:.2f}%"
                alpha_after = f"+{after.get('alpha', 0):.2f}%" if after.get('alpha', 0) >= 0 else f"{after.get('alpha', 0):.2f}%"

                spread_change = after['q1_q5_spread'] - before['q1_q5_spread']
                alpha_change = after.get('alpha', 0) - before.get('alpha', 0)

                # Use text indicators instead of emoji arrows
                spread_indicator = "[UP]" if spread_change > 0 else "[DOWN]" if spread_change < 0 else "[SAME]"
                alpha_indicator = "[UP]" if alpha_change > 0 else "[DOWN]" if alpha_change < 0 else "[SAME]"

                spread_change_str = f"{spread_indicator} {spread_change:+.2f}%"
                alpha_change_str = f"{alpha_indicator} {alpha_change:+.2f}%"
            else:
                spread_after = "*pending*"
                alpha_after = "*pending*"
                spread_change_str = "-"
                alpha_change_str = "-"

            lines.append(f"| {quarter} | {before['screen_date']} | {spread_before} | {spread_after} | {spread_change_str} | {alpha_before} | {alpha_after} | {alpha_change_str} |")

        lines.append("")

        # Aggregate statistics
        lines.append("## Aggregate Performance\n")

        # Calculate averages
        baseline_spreads = [d['q1_q5_spread'] for d in self.baseline.values()]
        enhanced_spreads = [d['q1_q5_spread'] for d in self.enhanced.values()]

        baseline_alphas = [d.get('alpha', 0) for d in self.baseline.values()]
        enhanced_alphas = [d.get('alpha', 0) for d in self.enhanced.values()]

        avg_baseline_spread = sum(baseline_spreads) / len(baseline_spreads)
        avg_enhanced_spread = sum(enhanced_spreads) / len(enhanced_spreads) if enhanced_spreads else 0

        avg_baseline_alpha = sum(baseline_alphas) / len(baseline_alphas)
        avg_enhanced_alpha = sum(enhanced_alphas) / len(enhanced_alphas) if enhanced_alphas else 0

        lines.append("### Average Q1-Q5 Spread")
        lines.append(f"- **Before**: {avg_baseline_spread:+.2f}%")
        lines.append(f"- **After**: {avg_enhanced_spread:+.2f}%")
        if enhanced_spreads:
            improvement = avg_enhanced_spread - avg_baseline_spread
            pct_improvement = (improvement / abs(avg_baseline_spread)) * 100 if avg_baseline_spread != 0 else 0
            lines.append(f"- **Improvement**: {improvement:+.2f}% ({pct_improvement:+.1f}%)\n")
        else:
            lines.append("")

        lines.append("### Average Alpha")
        lines.append(f"- **Before**: {avg_baseline_alpha:+.2f}%")
        lines.append(f"- **After**: {avg_enhanced_alpha:+.2f}%")
        if enhanced_alphas:
            improvement = avg_enhanced_alpha - avg_baseline_alpha
            pct_improvement = (improvement / abs(avg_baseline_alpha)) * 100 if avg_baseline_alpha != 0 else 0
            lines.append(f"- **Improvement**: {improvement:+.2f}% ({pct_improvement:+.1f}%)\n")
        else:
            lines.append("")

        lines.append("### Success Rate (Positive Spreads)")
        baseline_success = sum(1 for s in baseline_spreads if s > 0)
        enhanced_success = sum(1 for s in enhanced_spreads if s > 0)
        lines.append(f"- **Before**: {baseline_success}/{len(baseline_spreads)} ({100*baseline_success/len(baseline_spreads):.1f}%)")
        if enhanced_spreads:
            lines.append(f"- **After**: {enhanced_success}/{len(enhanced_spreads)} ({100*enhanced_success/len(enhanced_spreads):.1f}%)")
        lines.append("")

        # Biggest improvements
        if self.enhanced:
            lines.append("## Improvements by Quarter\n")
            improvements = []
            for quarter in self.enhanced:
                if quarter in self.baseline:
                    imp = self.calculate_improvement(quarter)
                    if imp:
                        improvements.append(imp)

            if improvements:
                lines.append("### Q1-Q5 Spread Changes:")
                for imp in sorted(improvements, key=lambda x: x['spread_change'], reverse=True):
                    change_dir = "improved" if imp['spread_change'] > 0 else "decreased"
                    lines.append(f"- **{imp['quarter']}**: {imp['spread_before']:+.2f}% -> {imp['spread_after']:+.2f}% (*{imp['spread_change']:+.2f}% {change_dir}*)")

        report_text = "\n".join(lines)

        # Write to file with explicit UTF-8 encoding
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Report saved to: {output_file}")

        return report_text


def run_validation_for_quarter(screen_date: str, quarter: str) -> Optional[Dict]:
    """Run validation for a specific quarter with expanded clinical coverage."""
    import subprocess
    import json

    try:
        print(f"\n{'='*60}")
        print(f"Running validation for {quarter} ({screen_date})")
        print(f"{'='*60}")

        # Step 1: Reconstruct snapshot and generate rankings
        print("\nStep 1: Reconstructing snapshot with expanded clinical coverage...")

        # Find tickers file
        tickers_file = Path("data/tickers/biotech_universe.csv")
        if not tickers_file.exists():
            tickers_file = Path("example_universe.csv")

        cmd = [
            sys.executable,
            "historical_fetchers/reconstruct_snapshot.py",
            "--date", screen_date,
            "--generate-rankings"
        ]
        if tickers_file.exists():
            cmd.extend(["--tickers-file", str(tickers_file)])

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).parent.parent))
        if result.returncode != 0:
            print(f"Error reconstructing snapshot: {result.stderr}")
            return None
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

        # Step 2: Run validation
        print("\nStep 2: Validating rankings against forward returns...")

        rankings_file = Path(f"data/snapshots/{screen_date}/rankings.csv")
        returns_db = Path("data/returns/returns_db.json")

        if not rankings_file.exists():
            print(f"Error: Rankings file not found: {rankings_file}")
            return None

        if not returns_db.exists():
            print(f"Warning: Returns database not found: {returns_db}")
            print("Skipping forward returns validation")
            return None

        output_file = Path(f"data/snapshots/{screen_date}/validation_results.json")

        cmd = [
            sys.executable,
            "validate_signals.py",
            "--database", str(returns_db),
            "--ranked-list", str(rankings_file),
            "--screen-date", screen_date,
            "--forward-months", "6",
            "--output", str(output_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).parent.parent))
        print(result.stdout)
        if result.stderr:
            print(f"Stderr: {result.stderr}")

        # Step 3: Parse results
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                validation = json.load(f)

            # Extract key metrics for comparison
            quintile_returns = validation.get('quintile_analysis', {}).get('quintile_returns', {})
            q1_return = quintile_returns.get('1', quintile_returns.get(1, 0)) or 0
            q5_return = quintile_returns.get('5', quintile_returns.get(5, 0)) or 0

            return {
                'screen_date': screen_date,
                'q1_q5_spread': (q1_return - q5_return) * 100,  # Convert to percentage
                'q1_return': q1_return * 100,
                'q5_return': q5_return * 100,
                'alpha': (validation.get('alpha_metrics', {}).get('avg_alpha') or 0) * 100,
                'clinical_coverage': 75.3  # Approximate with expanded coverage
            }

        return None

    except Exception as e:
        print(f"Error running validation: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Compare validation results')
    parser.add_argument('--show-pending', action='store_true',
                        help='Show pending quarters')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file for report')
    parser.add_argument('--run-pending', action='store_true',
                        help='Run all pending validations')
    parser.add_argument('--run-quarter', type=str,
                        help='Run validation for specific quarter (e.g., 2023-Q2)')

    args = parser.parse_args()

    comparison = ValidationComparison()

    if args.run_quarter:
        quarter = args.run_quarter
        if quarter in comparison.baseline:
            screen_date = comparison.baseline[quarter]['screen_date']
            result = run_validation_for_quarter(screen_date, quarter)
            if result:
                comparison.add_enhanced_result(quarter, result)
                print(f"\nCompleted validation for {quarter}")
                print(f"  Q1-Q5 Spread: {result['q1_q5_spread']:+.2f}%")
        else:
            print(f"Unknown quarter: {quarter}")
            print(f"Available: {list(comparison.baseline.keys())}")
        return

    if args.run_pending:
        pending = comparison.get_pending_quarters()
        print(f"Running {len(pending)} pending validations...")
        for quarter, screen_date in pending:
            result = run_validation_for_quarter(screen_date, quarter)
            if result:
                comparison.add_enhanced_result(quarter, result)
                print(f"  {quarter}: Q1-Q5 Spread = {result['q1_q5_spread']:+.2f}%")
        print("\nAll pending validations complete!")

    # Always generate report
    output = args.output or "reports/validation_comparison.md"
    report = comparison.generate_report(output)
    print(report)


if __name__ == "__main__":
    main()
