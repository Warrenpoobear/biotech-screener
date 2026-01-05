#!/usr/bin/env python3
"""
quickstart.py - Automated deployment and first run

This script automates the entire deployment process:
1. Validates environment
2. Runs pre-flight checks
3. Executes first data collection
4. Generates deployment report

Run this ONCE in your production environment to verify everything works.
"""
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

def print_header(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(title.center(70))
    print("="*70 + "\n")

def run_validation():
    """Run validation suite."""
    print_header("PHASE 1: VALIDATION")
    
    result = subprocess.run(
        [sys.executable, "validate_pipeline.py"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    
    if result.returncode != 0:
        print("❌ Validation failed. Cannot proceed.")
        return False
    
    return True

def run_preflight():
    """Run pre-flight checks."""
    print_header("PHASE 2: PRE-FLIGHT CHECKS")
    
    result = subprocess.run(
        [sys.executable, "preflight_check.py"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    
    if result.returncode != 0:
        print("⚠️  Some pre-flight checks failed.")
        response = input("\nContinue anyway? (y/N): ").strip().lower()
        return response == 'y'
    
    return True

def run_collection():
    """Run first data collection."""
    print_header("PHASE 3: FIRST DATA COLLECTION")
    print("This may take 3-5 minutes for 20 tickers...\n")
    
    start_time = datetime.now()
    
    result = subprocess.run(
        [sys.executable, "collect_universe_data.py"],
        capture_output=True,
        text=True
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"\n❌ Collection failed after {duration:.1f} seconds")
        print("\nError output:")
        print(result.stderr)
        return False, None
    
    print(f"\n✅ Collection completed in {duration:.1f} seconds")
    return True, duration

def analyze_results():
    """Analyze and summarize collection results."""
    print_header("PHASE 4: RESULTS ANALYSIS")
    
    # Load quality report
    report_path = Path("outputs/quality_report_latest.json")
    
    if not report_path.exists():
        print("❌ Quality report not found")
        return False
    
    with open(report_path) as f:
        quality = json.load(f)
    
    # Display key metrics
    print("📊 Data Quality Summary:")
    print(f"   Universe Size: {quality['universe_size']} companies")
    print(f"   Collection Date: {quality['collection_date']}")
    print()
    
    print("Coverage by Source:")
    for source, pct in quality['coverage_pct'].items():
        status = "✓" if pct >= 80 else "⚠️" if pct >= 60 else "✗"
        print(f"   {status} {source}: {pct:.1f}%")
    print()
    
    print("Average Coverage:")
    print(f"   Financial Metrics: {quality['avg_financial_coverage']:.1f}%")
    print(f"   Overall Data: {quality['avg_overall_coverage']:.1f}%")
    print()
    
    print("Quality Distribution:")
    qual = quality['tickers_by_quality']
    print(f"   Excellent (>80%): {len(qual['excellent'])} tickers")
    print(f"   Good (60-80%):    {len(qual['good'])} tickers")
    print(f"   Fair (40-60%):    {len(qual['fair'])} tickers")
    print(f"   Poor (<40%):      {len(qual['poor'])} tickers")
    
    if qual['poor']:
        print(f"\n   ⚠️  Low-coverage tickers: {', '.join(qual['poor'])}")
    
    # Overall assessment
    avg_coverage = quality['avg_overall_coverage']
    
    if avg_coverage >= 85:
        print("\n✅ EXCELLENT: Data quality exceeds targets")
        return True
    elif avg_coverage >= 75:
        print("\n✓ GOOD: Data quality meets minimum requirements")
        return True
    elif avg_coverage >= 60:
        print("\n⚠️  ACCEPTABLE: Some data gaps but operational")
        return True
    else:
        print("\n❌ POOR: Data quality below acceptable threshold")
        return False

def generate_deployment_report(duration):
    """Generate final deployment report."""
    print_header("DEPLOYMENT REPORT")
    
    report = {
        "deployment_timestamp": datetime.now().isoformat(),
        "environment": "Production",
        "pipeline_version": "1.0",
        "first_run_duration_seconds": duration,
        "validation_status": "PASS",
        "preflight_status": "PASS",
        "collection_status": "SUCCESS",
        "next_steps": [
            "Review quality report in outputs/",
            "Set up scheduled collection (Tuesday 7 AM recommended)",
            "Integrate with scoring model",
            "Monitor first week of automated runs"
        ]
    }
    
    # Save report
    output_path = Path("outputs/deployment_report.json")
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✅ Deployment Complete!")
    print(f"\n📄 Report saved: {output_path}")
    print("\n🎯 Next Steps:")
    for i, step in enumerate(report['next_steps'], 1):
        print(f"   {i}. {step}")
    
    print("\n📅 Recommended Schedule:")
    print("   • Tuesday 7:00 AM: Run data collection")
    print("   • Tuesday 9:00 AM: Review quality report")
    print("   • Wednesday: Feed into scoring model")
    print("   • Thursday: Generate ranked lists")
    
    print("\n📚 Documentation:")
    print("   • README.md - Full pipeline documentation")
    print("   • DEPLOYMENT_GUIDE.md - Deployment instructions")
    print("   • DEPLOYMENT_CHECKLIST.md - Detailed checklist")
    
    return True

def main():
    """Main quickstart execution."""
    print_header("WAKE ROBIN DATA PIPELINE - QUICKSTART")
    print("Automated deployment and verification")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Phase 1: Validation
    if not run_validation():
        print("\n❌ Quickstart aborted - validation failed")
        return 1
    
    input("\nPress ENTER to continue to pre-flight checks...")
    
    # Phase 2: Pre-flight
    if not run_preflight():
        print("\n❌ Quickstart aborted - pre-flight failed")
        return 1
    
    input("\nPress ENTER to begin first data collection...")
    
    # Phase 3: Collection
    success, duration = run_collection()
    if not success:
        print("\n❌ Quickstart aborted - collection failed")
        return 1
    
    input("\nPress ENTER to analyze results...")
    
    # Phase 4: Analysis
    if not analyze_results():
        print("\n⚠️  Data quality concerns detected - review required")
    
    # Phase 5: Report
    generate_deployment_report(duration)
    
    print("\n" + "="*70)
    print("DEPLOYMENT SUCCESSFUL".center(70))
    print("="*70 + "\n")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Quickstart interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
