#!/usr/bin/env python3
"""
Wake Robin Biotech Screener - Weekly Screening Pipeline

This script runs the complete weekly screening workflow:
1. Captures current point-in-time snapshot
2. Generates rankings from fundamentals
3. Extracts top quintile for portfolio
4. Generates summary report
5. Optionally sends email notification

Usage:
    python scripts/weekly_screen.py
    python scripts/weekly_screen.py --date 2024-01-15
    python scripts/weekly_screen.py --dry-run
    python scripts/weekly_screen.py --email user@example.com

Schedule with Windows Task Scheduler:
    Run every Tuesday at 9:00 AM EST

Author: Wake Robin Capital
Version: 1.0
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Configuration
CONFIG = {
    'universe_file': 'outputs/rankings_FIXED.csv',
    'returns_db': 'data/returns/returns_db_2020-01-01_2026-01-13.json',
    'output_dir': 'data/snapshots',
    'reports_dir': 'reports/weekly',
    'top_quintile_pct': 0.20,  # Top 20%
    'email_enabled': False,
    'email_recipient': None,
}


def log(message: str, level: str = 'INFO') -> None:
    """Print timestamped log message."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {message}")


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    log(f"Running: {description}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"Error: {result.stderr}", level='ERROR')
            return False
        return True
    except Exception as e:
        log(f"Exception: {e}", level='ERROR')
        return False


def capture_snapshot(screen_date: str, dry_run: bool = False) -> bool:
    """
    Step 1: Capture point-in-time snapshot.

    Fetches current financials from SEC EDGAR and clinical data
    from ClinicalTrials.gov.
    """
    log(f"Step 1: Capturing snapshot for {screen_date}")

    if dry_run:
        log("DRY RUN - Skipping snapshot capture")
        return True

    cmd = [
        'python', 'historical_fetchers/reconstruct_snapshot.py',
        '--date', screen_date,
        '--tickers-file', CONFIG['universe_file'],
        '--generate-rankings'
    ]

    return run_command(cmd, "Snapshot capture")


def load_rankings(screen_date: str) -> List[Dict]:
    """Load rankings from CSV file."""
    rankings_file = Path(CONFIG['output_dir']) / screen_date / 'rankings.csv'

    if not rankings_file.exists():
        log(f"Rankings file not found: {rankings_file}", level='ERROR')
        return []

    rankings = []
    with open(rankings_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rankings.append(row)

    return rankings


def extract_top_quintile(rankings: List[Dict]) -> List[Dict]:
    """
    Step 2: Extract top quintile for portfolio.

    Returns the top 20% of ranked tickers.
    """
    log("Step 2: Extracting top quintile")

    n_total = len(rankings)
    n_top = int(n_total * CONFIG['top_quintile_pct'])

    top_quintile = rankings[:n_top]

    log(f"  Total tickers: {n_total}")
    log(f"  Top quintile: {n_top} tickers")

    return top_quintile


def generate_trade_list(top_quintile: List[Dict], screen_date: str) -> Path:
    """
    Step 3: Generate trade list CSV.

    Creates a CSV file with tickers ready for trading.
    """
    log("Step 3: Generating trade list")

    reports_dir = Path(CONFIG['reports_dir'])
    reports_dir.mkdir(parents=True, exist_ok=True)

    trade_file = reports_dir / f"trade_list_{screen_date}.csv"

    with open(trade_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Ticker', 'Composite_Score', 'Action', 'Weight'])

        n_tickers = len(top_quintile)
        equal_weight = 1.0 / n_tickers if n_tickers > 0 else 0

        for row in top_quintile:
            writer.writerow([
                row.get('Rank', ''),
                row.get('Ticker', ''),
                row.get('Composite_Score', ''),
                'BUY',
                f"{equal_weight:.4f}"
            ])

    log(f"  Trade list saved: {trade_file}")
    return trade_file


def generate_summary_report(
    screen_date: str,
    rankings: List[Dict],
    top_quintile: List[Dict]
) -> str:
    """
    Step 4: Generate summary report.

    Creates a text summary of the screening results.
    """
    log("Step 4: Generating summary report")

    report_lines = [
        "=" * 60,
        "WAKE ROBIN BIOTECH SCREENER - WEEKLY REPORT",
        "=" * 60,
        "",
        f"Screen Date:     {screen_date}",
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "-" * 60,
        "UNIVERSE SUMMARY",
        "-" * 60,
        f"  Total Tickers:    {len(rankings)}",
        f"  Top Quintile:     {len(top_quintile)} tickers",
        "",
        "-" * 60,
        "TOP 10 RANKED TICKERS",
        "-" * 60,
    ]

    for i, row in enumerate(rankings[:10], 1):
        ticker = row.get('Ticker', 'N/A')
        score = row.get('Composite_Score', 'N/A')
        report_lines.append(f"  {i:2}. {ticker:<8} Score: {score}")

    report_lines.extend([
        "",
        "-" * 60,
        "TOP QUINTILE TICKERS (For Portfolio)",
        "-" * 60,
    ])

    # Show first 20 of top quintile
    for row in top_quintile[:20]:
        ticker = row.get('Ticker', 'N/A')
        report_lines.append(f"  {ticker}")

    if len(top_quintile) > 20:
        report_lines.append(f"  ... and {len(top_quintile) - 20} more")

    report_lines.extend([
        "",
        "-" * 60,
        "ACTIONS REQUIRED",
        "-" * 60,
        "  1. Review trade list CSV",
        "  2. Compare with current holdings",
        "  3. Generate rebalancing orders",
        "  4. Execute trades by market close",
        "",
        "=" * 60,
        "END OF REPORT",
        "=" * 60,
    ])

    report_text = "\n".join(report_lines)

    # Save report
    reports_dir = Path(CONFIG['reports_dir'])
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / f"summary_{screen_date}.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    log(f"  Summary saved: {report_file}")

    return report_text


def send_email_notification(
    recipient: str,
    subject: str,
    body: str
) -> bool:
    """
    Step 5: Send email notification (optional).

    Requires SMTP configuration.
    """
    if not CONFIG['email_enabled']:
        log("Email notifications disabled")
        return True

    log(f"Step 5: Sending email to {recipient}")

    # TODO: Implement email sending
    # This would require SMTP configuration
    # For now, just log that we would send
    log("  Email sending not implemented - see report file")

    return True


def validate_previous_performance(screen_date: str) -> Optional[Dict]:
    """
    Bonus: Validate previous screen performance (if possible).

    Compares previous screen's predictions against actual returns.
    """
    log("Validating previous screen performance...")

    # This would compare previous predictions to actual returns
    # Requires historical tracking

    return None


def main():
    """Main entry point for weekly screening."""
    parser = argparse.ArgumentParser(
        description='Wake Robin Weekly Biotech Screening'
    )
    parser.add_argument(
        '--date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='Screen date (YYYY-MM-DD), default: today'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without fetching new data'
    )
    parser.add_argument(
        '--email',
        type=str,
        help='Email recipient for notifications'
    )
    parser.add_argument(
        '--skip-snapshot',
        action='store_true',
        help='Skip snapshot capture, use existing data'
    )

    args = parser.parse_args()

    screen_date = args.date

    # Banner
    print("\n" + "=" * 60)
    print("WAKE ROBIN BIOTECH SCREENER")
    print("Weekly Screening Pipeline")
    print("=" * 60)
    print(f"Screen Date: {screen_date}")
    print(f"Dry Run: {args.dry_run}")
    print("=" * 60 + "\n")

    # Configure email if provided
    if args.email:
        CONFIG['email_enabled'] = True
        CONFIG['email_recipient'] = args.email

    # Step 1: Capture snapshot
    if not args.skip_snapshot:
        if not capture_snapshot(screen_date, dry_run=args.dry_run):
            log("Snapshot capture failed", level='ERROR')
            sys.exit(1)
    else:
        log("Skipping snapshot capture (--skip-snapshot)")

    # Load rankings
    rankings = load_rankings(screen_date)
    if not rankings:
        log("No rankings available", level='ERROR')
        sys.exit(1)

    # Step 2: Extract top quintile
    top_quintile = extract_top_quintile(rankings)

    # Step 3: Generate trade list
    trade_file = generate_trade_list(top_quintile, screen_date)

    # Step 4: Generate summary report
    report = generate_summary_report(screen_date, rankings, top_quintile)

    # Print report to console
    print("\n" + report)

    # Step 5: Send email (if configured)
    if CONFIG['email_enabled'] and CONFIG['email_recipient']:
        send_email_notification(
            CONFIG['email_recipient'],
            f"Wake Robin Screen - {screen_date}",
            report
        )

    # Final summary
    print("\n" + "=" * 60)
    print("SCREENING COMPLETE")
    print("=" * 60)
    print(f"  Snapshot:    data/snapshots/{screen_date}/")
    print(f"  Rankings:    data/snapshots/{screen_date}/rankings.csv")
    print(f"  Trade List:  reports/weekly/trade_list_{screen_date}.csv")
    print(f"  Summary:     reports/weekly/summary_{screen_date}.txt")
    print("=" * 60 + "\n")

    log("Weekly screening complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
