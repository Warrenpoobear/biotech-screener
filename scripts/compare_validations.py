"""
Compare validation results across different versions of the screening system.

Tracks improvements from enhancements like clinical coverage expansion.

Usage:
    python compare_validations.py --output reports/validation_comparison.md
    python compare_validations.py --show-pending
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import argparse


class ValidationComparison:
    """Compare validation results across system versions."""
    
    def __init__(self):
        self.results = {
            'baseline': {},  # Original results (13% clinical)
            'enhanced': {}   # Enhanced results (70% clinical)
        }
        
    def add_result(self, version: str, quarter: str, screen_date: str, 
                   q1_q5_spread: float, avg_alpha: float, q1_return: float,
                   q1_alpha: float, hit_rate: float, tickers_tested: int):
        """Add a validation result."""
        
        self.results[version][quarter] = {
            'screen_date': screen_date,
            'q1_q5_spread': q1_q5_spread,
            'avg_alpha': avg_alpha,
            'q1_return': q1_return,
            'q1_alpha': q1_alpha,
            'hit_rate': hit_rate,
            'tickers_tested': tickers_tested
        }
    
    def calculate_improvement(self, metric: str) -> Dict:
        """Calculate improvement for a specific metric."""
        
        improvements = {}
        
        for quarter in self.results['baseline'].keys():
            if quarter in self.results['enhanced']:
                baseline_val = self.results['baseline'][quarter][metric]
                enhanced_val = self.results['enhanced'][quarter][metric]
                
                improvement = enhanced_val - baseline_val
                pct_improvement = (improvement / abs(baseline_val) * 100) if baseline_val != 0 else 0
                
                improvements[quarter] = {
                    'baseline': baseline_val,
                    'enhanced': enhanced_val,
                    'absolute_change': improvement,
                    'pct_change': pct_improvement
                }
        
        return improvements
    
    def generate_summary_table(self) -> str:
        """Generate comparison summary table."""
        
        lines = []
        lines.append("## Quarter-by-Quarter Comparison\n")
        lines.append("| Quarter | Screen Date | Q1-Q5 Spread (Before) | Q1-Q5 Spread (After) | Change | Q1 Alpha (Before) | Q1 Alpha (After) | Change |")
        lines.append("|---------|-------------|------------------------|----------------------|--------|-------------------|------------------|--------|")
        
        for quarter in sorted(self.results['baseline'].keys()):
            baseline = self.results['baseline'][quarter]
            
            if quarter in self.results['enhanced']:
                enhanced = self.results['enhanced'][quarter]
                
                spread_change = enhanced['q1_q5_spread'] - baseline['q1_q5_spread']
                alpha_change = enhanced['q1_alpha'] - baseline['q1_alpha']
                
                spread_arrow = "ðŸ“ˆ" if spread_change > 0 else "ðŸ“‰" if spread_change < 0 else "â†’"
                alpha_arrow = "ðŸ“ˆ" if alpha_change > 0 else "ðŸ“‰" if alpha_change < 0 else "â†’"
                
                lines.append(
                    f"| {quarter} | {baseline['screen_date']} | "
                    f"{baseline['q1_q5_spread']:+.2f}% | "
                    f"{enhanced['q1_q5_spread']:+.2f}% | "
                    f"{spread_arrow} {spread_change:+.2f}% | "
                    f"{baseline['q1_alpha']:+.2f}% | "
                    f"{enhanced['q1_alpha']:+.2f}% | "
                    f"{alpha_arrow} {alpha_change:+.2f}% |"
                )
            else:
                lines.append(
                    f"| {quarter} | {baseline['screen_date']} | "
                    f"{baseline['q1_q5_spread']:+.2f}% | "
                    f"*pending* | - | "
                    f"{baseline['q1_alpha']:+.2f}% | "
                    f"*pending* | - |"
                )
        
        return "\n".join(lines)
    
    def generate_aggregate_stats(self) -> str:
        """Generate aggregate statistics comparison."""
        
        lines = []
        lines.append("\n## Aggregate Performance\n")
        
        # Calculate averages
        baseline_spreads = [r['q1_q5_spread'] for r in self.results['baseline'].values()]
        baseline_alphas = [r['avg_alpha'] for r in self.results['baseline'].values()]
        
        enhanced_quarters = [q for q in self.results['baseline'].keys() if q in self.results['enhanced']]
        if enhanced_quarters:
            enhanced_spreads = [self.results['enhanced'][q]['q1_q5_spread'] for q in enhanced_quarters]
            enhanced_alphas = [self.results['enhanced'][q]['avg_alpha'] for q in enhanced_quarters]
            
            avg_baseline_spread = sum(baseline_spreads) / len(baseline_spreads)
            avg_enhanced_spread = sum(enhanced_spreads) / len(enhanced_spreads)
            
            avg_baseline_alpha = sum(baseline_alphas) / len(baseline_alphas)
            avg_enhanced_alpha = sum(enhanced_alphas) / len(enhanced_alphas)
            
            spread_improvement = avg_enhanced_spread - avg_baseline_spread
            alpha_improvement = avg_enhanced_alpha - avg_baseline_alpha
            
            lines.append("### Average Q1-Q5 Spread")
            lines.append(f"- **Before**: {avg_baseline_spread:+.2f}%")
            lines.append(f"- **After**: {avg_enhanced_spread:+.2f}%")
            lines.append(f"- **Improvement**: {spread_improvement:+.2f}% ({spread_improvement/avg_baseline_spread*100:+.1f}%)\n")
            
            lines.append("### Average Alpha")
            lines.append(f"- **Before**: {avg_baseline_alpha:+.2f}%")
            lines.append(f"- **After**: {avg_enhanced_alpha:+.2f}%")
            lines.append(f"- **Improvement**: {alpha_improvement:+.2f}% ({alpha_improvement/avg_baseline_alpha*100:+.1f}%)\n")
            
            # Success rate
            baseline_positive = sum(1 for s in baseline_spreads if s > 0)
            enhanced_positive = sum(1 for s in enhanced_spreads if s > 0)
            
            lines.append("### Success Rate (Positive Spreads)")
            lines.append(f"- **Before**: {baseline_positive}/{len(baseline_spreads)} ({baseline_positive/len(baseline_spreads)*100:.1f}%)")
            lines.append(f"- **After**: {enhanced_positive}/{len(enhanced_spreads)} ({enhanced_positive/len(enhanced_spreads)*100:.1f}%)\n")
        else:
            lines.append("*No enhanced results available yet*\n")
        
        return "\n".join(lines)
    
    def generate_top_improvements(self) -> str:
        """Identify quarters with biggest improvements."""
        
        lines = []
        lines.append("\n## Biggest Improvements\n")
        
        spread_improvements = self.calculate_improvement('q1_q5_spread')
        
        if spread_improvements:
            sorted_improvements = sorted(
                spread_improvements.items(),
                key=lambda x: x[1]['absolute_change'],
                reverse=True
            )
            
            lines.append("### Q1-Q5 Spread Improvements:")
            for i, (quarter, data) in enumerate(sorted_improvements[:3], 1):
                lines.append(
                    f"{i}. **{quarter}**: {data['baseline']:+.2f}% â†’ {data['enhanced']:+.2f}% "
                    f"(*{data['absolute_change']:+.2f}% improvement*)"
                )
        else:
            lines.append("*No comparison data available yet*")
        
        return "\n".join(lines)
    
    def generate_status_summary(self) -> str:
        """Generate current status summary."""
        
        lines = []
        lines.append("\n## Validation Status\n")
        
        total_quarters = len(self.results['baseline'])
        completed_quarters = len([q for q in self.results['baseline'].keys() if q in self.results['enhanced']])
        pending_quarters = total_quarters - completed_quarters
        
        lines.append(f"- **Total Test Periods**: {total_quarters}")
        lines.append(f"- **Baseline Complete**: {total_quarters} âœ…")
        lines.append(f"- **Enhanced Complete**: {completed_quarters}")
        lines.append(f"- **Pending**: {pending_quarters}\n")
        
        if pending_quarters > 0:
            pending_list = [q for q in self.results['baseline'].keys() if q not in self.results['enhanced']]
            lines.append("### Quarters Pending Re-validation:")
            for quarter in pending_list:
                screen_date = self.results['baseline'][quarter]['screen_date']
                lines.append(f"- [ ] {quarter} ({screen_date})")
        
        return "\n".join(lines)
    
    def generate_report(self, output_file: str = None):
        """Generate complete comparison report."""
        
        report = []
        report.append("# Wake Robin Validation Comparison Report")
        report.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        report.append("---\n")
        report.append("## Enhancement: Clinical Coverage Expansion")
        report.append("- **Baseline**: 13% clinical coverage (41/316 tickers)")
        report.append("- **Enhanced**: 70% clinical coverage (220+/316 tickers)")
        report.append("- **Method**: Expanded ticker â†’ sponsor name mapping\n")
        
        report.append(self.generate_status_summary())
        report.append(self.generate_summary_table())
        report.append(self.generate_aggregate_stats())
        report.append(self.generate_top_improvements())
        
        report_text = "\n".join(report)
        
        # Print to console
        print(report_text)
        
        # Save to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\nâœ… Report saved to: {output_file}")
        
        return report_text


def load_baseline_results() -> Dict:
    """
    Load baseline validation results (before clinical enhancement).
    
    These are the original results from the initial 7-quarter validation.
    """
    return {
        '2023-Q1': {
            'screen_date': '2023-01-15',
            'q1_q5_spread': 8.66,
            'avg_alpha': 7.40,
            'q1_return': 12.42,
            'q1_alpha': 12.18,
            'hit_rate': 45.49,
            'tickers_tested': 277
        },
        '2023-Q2': {
            'screen_date': '2023-04-15',
            'q1_q5_spread': 3.79,
            'avg_alpha': 3.99,
            'q1_return': 3.52,
            'q1_alpha': 7.66,
            'hit_rate': 37.86,
            'tickers_tested': 280
        },
        '2023-Q3': {
            'screen_date': '2023-07-15',
            'q1_q5_spread': 12.97,
            'avg_alpha': 2.23,
            'q1_return': 12.53,
            'q1_alpha': 5.29,
            'hit_rate': 52.65,
            'tickers_tested': 283
        },
        '2023-Q4': {
            'screen_date': '2023-10-15',
            'q1_q5_spread': 18.43,
            'avg_alpha': 17.71,
            'q1_return': 42.67,
            'q1_alpha': 12.58,
            'hit_rate': 70.83,
            'tickers_tested': 288
        },
        '2024-Q1': {
            'screen_date': '2024-01-15',
            'q1_q5_spread': 13.19,
            'avg_alpha': 4.84,
            'q1_return': 23.20,
            'q1_alpha': 19.21,
            'hit_rate': 45.89,
            'tickers_tested': 292
        },
        '2024-Q2': {
            'screen_date': '2024-04-15',
            'q1_q5_spread': -14.53,
            'avg_alpha': -0.28,
            'q1_return': 1.64,
            'q1_alpha': -2.54,
            'hit_rate': 40.20,
            'tickers_tested': 296
        },
        '2024-Q3': {
            'screen_date': '2024-07-15',
            'q1_q5_spread': 22.62,
            'avg_alpha': 13.24,
            'q1_return': 26.78,
            'q1_alpha': 29.57,
            'hit_rate': 46.86,
            'tickers_tested': 303
        }
    }


def load_enhanced_results() -> Dict:
    """
    Load enhanced validation results (after clinical enhancement).
    
    Add results here as you complete re-validation of each quarter.
    """
    return {
        '2023-Q1': {
            'screen_date': '2023-01-15',
            'q1_q5_spread': 13.60,
            'avg_alpha': 7.40,
            'q1_return': 17.05,
            'q1_alpha': 16.81,
            'hit_rate': 45.49,
            'tickers_tested': 277
        },
        # Add more quarters as you complete them:
        # '2023-Q2': { ... },
        # '2023-Q3': { ... },
        # etc.
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare validation results before and after enhancements'
    )
    parser.add_argument(
        '--output',
        default='reports/validation_comparison.md',
        help='Output file for comparison report'
    )
    parser.add_argument(
        '--show-pending',
        action='store_true',
        help='Show commands to run pending validations'
    )
    args = parser.parse_args()
    
    # Load results
    baseline = load_baseline_results()
    enhanced = load_enhanced_results()
    
    # Create comparison
    comparison = ValidationComparison()
    
    # Add baseline results
    for quarter, data in baseline.items():
        comparison.add_result(
            version='baseline',
            quarter=quarter,
            **data
        )
    
    # Add enhanced results
    for quarter, data in enhanced.items():
        comparison.add_result(
            version='enhanced',
            quarter=quarter,
            **data
        )
    
    # Generate report
    comparison.generate_report(args.output)
    
    # Show pending validation commands
    if args.show_pending:
        print("\n" + "="*70)
        print("PENDING VALIDATIONS")
        print("="*70)
        print("\nRun these commands to complete the comparison:\n")
        
        pending = [q for q in baseline.keys() if q not in enhanced.keys()]
        
        date_map = {q: baseline[q]['screen_date'] for q in pending}
        
        for quarter in sorted(pending):
            screen_date = date_map[quarter]
            print(f"# {quarter}")
            print(f"python historical_fetchers/reconstruct_snapshot.py \\")
            print(f"  --date {screen_date} \\")
            print(f"  --tickers-file outputs/rankings_FIXED.csv \\")
            print(f"  --generate-rankings")
            print()
            print(f"python validate_signals.py \\")
            print(f"  --database data/returns/returns_db_2020-01-01_2026-01-13.json \\")
            print(f"  --ranked-list data/snapshots/{screen_date}/rankings.csv \\")
            print(f"  --screen-date {screen_date} \\")
            print(f"  --forward-months 6")
            print()


if __name__ == '__main__':
    main()

