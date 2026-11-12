#!/usr/bin/env python3
"""Monitor 13F filing progress for a target quarter.

The target quarter is selected with --quarter (a calendar quarter-end date);
it defaults to the most recently completed quarter. Progress is tracked in a
per-quarter status file so one quarter's results can never be read as another's.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running as `python3 tools/monitor_13f_filing_progress.py` — Python puts the
# script's own directory on sys.path, not the repo root, so the imports below would
# otherwise require PYTHONPATH to be set by the caller.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.date_utils import latest_completed_quarter_end, parse_quarter_end  # noqa: E402
from elite_managers import get_all_managers  # noqa: E402
from sec_13f.edgar_13f import SEC13FFetcher  # noqa: E402

CACHE_DIR = Path("data/13f_cache")
STATUS_FILE_TEMPLATE = "production_data/13f_filing_status_{quarter}.json"

# A manager's most recent filing is not necessarily the one for the target quarter —
# it can be an amendment or a later-period form. Fetch a few and look for the quarter.
RECENT_FILINGS_TO_SCAN = 4

# Fraction of the registry that must have filed before cohort validation is rerun.
MILESTONE_THRESHOLD_PCT = 70.0


def parse_quarter(value):
    """argparse adapter around parse_quarter_end."""
    try:
        return parse_quarter_end(value)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def status_path_for(quarter, override=None):
    """Status file for a quarter; per-quarter by default so results cannot mix."""
    return Path(override) if override else Path(STATUS_FILE_TEMPLATE.format(quarter=quarter))


def load_status(status_file, quarter):
    """Load filing status, discarding any state recorded for a different quarter."""
    if not status_file.exists():
        return {"target_quarter": quarter, "last_check": None, "filed": {}, "pending": {}}

    with open(status_file) as f:
        status = json.load(f)

    recorded = status.get("target_quarter")
    if recorded != quarter:
        # Reusing another quarter's `filed` map would skip every manager in it and
        # report them as having filed for this quarter.
        print(f"  NOTE: {status_file} tracks {recorded or 'an unrecorded quarter'}; starting fresh for {quarter}")
        return {"target_quarter": quarter, "last_check": None, "filed": {}, "pending": {}}

    status.setdefault("filed", {})
    status.setdefault("pending", {})
    return status


def save_status(status, status_file):
    """Save filing status"""
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)


def check_manager_filings(quarter=None, status_file_override=None):
    """Check which managers have filings for the target quarter."""
    quarter = quarter or latest_completed_quarter_end().isoformat()
    status_file = status_path_for(quarter, status_file_override)

    managers = get_all_managers()
    fetcher = SEC13FFetcher(cache_dir=str(CACHE_DIR))

    print(f"\n[{datetime.now().isoformat()}] Checking {len(managers)} managers for {quarter} filings...")
    print(f"  status file: {status_file}")

    status = load_status(status_file, quarter)
    status["last_check"] = datetime.now().isoformat()

    newly_filed = []

    for m in managers:
        cik = m["cik"].zfill(10)
        name = m["name"]

        # Skip if already processed
        if cik in status["filed"]:
            continue

        try:
            filings = fetcher.get_recent_filings(m["cik"], count=RECENT_FILINGS_TO_SCAN)
            match = next((f for f in (filings or []) if f.report_date.isoformat() == quarter), None)
            if match:
                status["filed"][cik] = {
                    "name": name,
                    "filed_date": match.filing_date.isoformat(),
                    "accession": match.accession_number,
                }
                status["pending"].pop(cik, None)
                newly_filed.append((cik, name, match.filing_date.isoformat()))
                print(f"  ✓ {name:30s} ({cik}) filed {match.filing_date}")
            else:
                status["pending"][cik] = {"name": name, "checked_at": datetime.now().isoformat()}

            time.sleep(0.1)  # Rate limit

        except Exception as e:
            status["pending"][cik] = {"name": name, "error": str(e)[:50]}

    save_status(status, status_file)

    filed_total = len(status["filed"])
    print("\nSummary:")
    print(f"  Target quarter: {quarter}")
    print(f"  Total filed: {filed_total}/{len(managers)}")
    print(f"  Newly filed this check: {len(newly_filed)}")

    if newly_filed:
        print("\nNew filings:")
        for cik, name, filed_date in newly_filed:
            print(f"  - {name} ({cik}) on {filed_date}")

    # Check if we should lift quarantine (70%+ filed)
    filing_pct = filed_total / len(managers) * 100
    required = MILESTONE_THRESHOLD_PCT / 100 * len(managers)
    print(f"\nFiling progress: {filing_pct:.1f}%")

    if filing_pct >= MILESTONE_THRESHOLD_PCT:
        print(f"\n⚠️  MILESTONE: {MILESTONE_THRESHOLD_PCT:.0f}% filing threshold reached ({filed_total} managers)")
        print("    Ready for cohort validation rerun")
    else:
        print(f"  Below the {MILESTONE_THRESHOLD_PCT:.0f}% milestone — need {required:.0f} of {len(managers)}")

    return newly_filed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--quarter",
        type=parse_quarter,
        default=None,
        help="Target quarter end YYYY-MM-DD (default: most recently completed quarter)",
    )
    parser.add_argument(
        "--status-file",
        default=None,
        help=f"Override the status file path (default: {STATUS_FILE_TEMPLATE})",
    )
    args = parser.parse_args()
    check_manager_filings(quarter=args.quarter, status_file_override=args.status_file)


if __name__ == "__main__":
    main()
