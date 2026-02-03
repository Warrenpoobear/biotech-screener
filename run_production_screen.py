#!/usr/bin/env python3
"""
run_production_screen.py - Production wrapper for biotech screener

Runs the screening pipeline and automatically saves results to production_data/:
  - results.json (latest results, overwritten each run)
  - results.csv (latest CSV export, overwritten each run)
  - results_YYYY-MM-DD.json (date-stamped archive)
  - results_YYYY-MM-DD.csv (date-stamped archive)
  - run_log_YYYY-MM-DD.json (run metadata)

Usage:
    python run_production_screen.py --as-of-date 2024-12-15
    python run_production_screen.py --as-of-date 2024-12-15 --archive
    python run_production_screen.py --as-of-date 2024-12-15 --no-csv
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import CSV exporter
from export_results_csv import export_to_csv


def get_production_data_dir() -> Path:
    """Get the production_data directory path."""
    return Path(__file__).parent / "production_data"


def run_screen(
    as_of_date: str,
    data_dir: Path,
    output_path: Path,
    output_dir: Optional[Path] = None,
    extra_args: Optional[list] = None,
) -> int:
    """Run the screening pipeline.

    Args:
        as_of_date: The as-of date for PIT screening (YYYY-MM-DD)
        data_dir: Directory containing input data
        output_path: Path for the main results JSON output
        output_dir: Directory for side outputs (catalyst_events, etc.)
        extra_args: Additional arguments to pass to run_screen.py

    Returns:
        Return code from run_screen.py (0 = success)
    """
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_screen.py"),
        "--as-of-date", as_of_date,
        "--data-dir", str(data_dir),
        "--output", str(output_path),
    ]

    if output_dir:
        cmd.extend(["--output-dir", str(output_dir)])

    if extra_args:
        cmd.extend(extra_args)

    print(f"[PRODUCTION] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def save_run_log(
    production_dir: Path,
    as_of_date: str,
    success: bool,
    results_path: Path,
    csv_path: Optional[Path] = None,
    archive_json_path: Optional[Path] = None,
    archive_csv_path: Optional[Path] = None,
) -> Path:
    """Save a run log with metadata about this screening run.

    Returns:
        Path to the saved run log
    """
    run_log = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "as_of_date": as_of_date,
        "success": success,
        "outputs": {
            "results_json": str(results_path),
            "results_csv": str(csv_path) if csv_path else None,
            "archive_json": str(archive_json_path) if archive_json_path else None,
            "archive_csv": str(archive_csv_path) if archive_csv_path else None,
        },
        "python_version": sys.version,
    }

    log_path = production_dir / f"run_log_{as_of_date}.json"
    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=2)

    return log_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Production wrapper for biotech screener",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run screen for a specific date
  python run_production_screen.py --as-of-date 2024-12-15

  # Run with date-stamped archives
  python run_production_screen.py --as-of-date 2024-12-15 --archive

  # Skip CSV generation
  python run_production_screen.py --as-of-date 2024-12-15 --no-csv

  # Pass additional arguments to run_screen.py
  python run_production_screen.py --as-of-date 2024-12-15 -- --enable-coinvest --enable-pos
""",
    )

    parser.add_argument(
        "--as-of-date",
        required=True,
        help="Point-in-time date for screening (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Input data directory (default: production_data)",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Also save date-stamped archive copies of results",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV export generation",
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Additional arguments to pass to run_screen.py (use -- before them)",
    )

    args = parser.parse_args()

    # Setup paths
    production_dir = get_production_data_dir()
    production_dir.mkdir(exist_ok=True)

    data_dir = args.data_dir or production_dir

    # Output paths
    results_json = production_dir / "results.json"
    results_csv = production_dir / "results.csv"

    # Archive paths (date-stamped)
    archive_json = production_dir / f"results_{args.as_of_date}.json" if args.archive else None
    archive_csv = production_dir / f"results_{args.as_of_date}.csv" if args.archive else None

    print(f"[PRODUCTION] As-of date: {args.as_of_date}")
    print(f"[PRODUCTION] Data directory: {data_dir}")
    print(f"[PRODUCTION] Output directory: {production_dir}")
    print()

    # Run the screening pipeline
    return_code = run_screen(
        as_of_date=args.as_of_date,
        data_dir=data_dir,
        output_path=results_json,
        output_dir=production_dir,
        extra_args=args.extra_args if args.extra_args else None,
    )

    if return_code != 0:
        print(f"\n[PRODUCTION] ERROR: run_screen.py failed with code {return_code}")
        save_run_log(
            production_dir=production_dir,
            as_of_date=args.as_of_date,
            success=False,
            results_path=results_json,
        )
        return return_code

    print(f"\n[PRODUCTION] Results saved to: {results_json}")

    # Generate CSV export
    csv_path = None
    if not args.no_csv:
        try:
            count = export_to_csv(results_json, results_csv)
            print(f"[PRODUCTION] CSV exported ({count} securities): {results_csv}")
            csv_path = results_csv
        except Exception as e:
            print(f"[PRODUCTION] WARNING: CSV export failed: {e}")

    # Create archive copies if requested
    if args.archive:
        if archive_json:
            shutil.copy2(results_json, archive_json)
            print(f"[PRODUCTION] Archive JSON: {archive_json}")

        if archive_csv and csv_path and csv_path.exists():
            shutil.copy2(csv_path, archive_csv)
            print(f"[PRODUCTION] Archive CSV: {archive_csv}")

    # Save run log
    log_path = save_run_log(
        production_dir=production_dir,
        as_of_date=args.as_of_date,
        success=True,
        results_path=results_json,
        csv_path=csv_path,
        archive_json_path=archive_json,
        archive_csv_path=archive_csv,
    )
    print(f"[PRODUCTION] Run log: {log_path}")

    print("\n[PRODUCTION] Complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
