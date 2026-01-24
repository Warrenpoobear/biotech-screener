#!/usr/bin/env python3
"""
Run Institutional-Grade Technical Audit

Usage:
    python run_audit.py                          # Run full audit
    python run_audit.py --tier 1                 # Run specific tier
    python run_audit.py --output report.json     # Custom output path
    python run_audit.py --format markdown        # Generate markdown report

Tiers:
    1: Determinism & Reproducibility
    2: Data Integrity & Provenance
    3: Performance & Resilience
    4: Testing & Regression
    5: Architecture & Security
    6: Deployment Readiness
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

# Add codebase to path
sys.path.insert(0, str(Path(__file__).parent))

from audit_framework.orchestrator import AuditOrchestrator, run_full_audit
from audit_framework.types import AuditTier, PassCriteria


def main():
    parser = argparse.ArgumentParser(
        description="Run institutional-grade technical audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help="Run specific tier only",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output file path",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "markdown"],
        default="json",
        help="Output format (default: json)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="production_data",
        help="Data directory (default: production_data)",
    )

    parser.add_argument(
        "--as-of-date",
        type=str,
        default=date.today().isoformat(),
        help="Reference date for PIT checks (default: today)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    # Get codebase path
    codebase_path = str(Path(__file__).parent)

    print("=" * 60)
    print("INSTITUTIONAL-GRADE TECHNICAL AUDIT")
    print("Wake Robin Capital Management - biotech-screener")
    print("=" * 60)
    print()

    # Create orchestrator
    orchestrator = AuditOrchestrator(
        codebase_path=codebase_path,
        data_dir=args.data_dir,
        as_of_date=args.as_of_date,
    )

    # Run audit
    if args.tier:
        # Map tier number to enum
        tier_map = {
            1: AuditTier.TIER_1_DETERMINISM,
            2: AuditTier.TIER_2_DATA_INTEGRITY,
            3: AuditTier.TIER_3_PERFORMANCE,
            4: AuditTier.TIER_4_TESTING,
            5: AuditTier.TIER_5_ARCHITECTURE,
            6: AuditTier.TIER_6_DEPLOYMENT,
        }

        tier = tier_map[args.tier]
        print(f"Running Tier {args.tier}: {tier.value}")
        print("-" * 40)

        result = orchestrator.run_tier(tier)

        # Print summary
        print(f"\nGrade: {result.grade.value}")
        print(f"Status: {'PASSED' if result.passed else 'FAILED'}")
        print(f"Findings: {len(result.findings)}")
        print(f"  - Critical: {result.critical_count}")
        print(f"  - High: {result.high_count}")

        if args.verbose:
            print("\n" + result.summary)

    else:
        print("Running Full Audit (all 6 tiers)")
        print("-" * 40)

        report = orchestrator.run_full_audit()

        # Print summary
        print(f"\n{'=' * 60}")
        print("AUDIT RESULTS")
        print("=" * 60)
        print(f"\nOverall Grade: {report.overall_grade.value}")
        print(f"Status: {'PASSED' if report.overall_passed else 'FAILED'}")
        print(f"Total Findings: {report.total_findings}")
        print(f"  - Critical: {len(report.critical_findings)}")
        print(f"  - High: {len(report.high_findings)}")

        print("\nTier Results:")
        for tr in report.tier_results:
            status = "PASS" if tr.passed else "FAIL"
            print(f"  {tr.tier.value}: Grade {tr.grade.value} [{status}]")

        # Save report
        output_path = orchestrator.save_report(
            report,
            output_path=args.output,
            format=args.format,
        )
        print(f"\nReport saved to: {output_path}")

        if args.verbose:
            print("\n" + report.executive_summary)

            if report.recommendations:
                print("\nRecommendations:")
                for i, rec in enumerate(report.recommendations, 1):
                    print(f"  {i}. {rec}")

    print("\n" + "=" * 60)
    print("Audit Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
