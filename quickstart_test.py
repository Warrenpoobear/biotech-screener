#!/usr/bin/env python3
"""
Quickstart Test for Morningstar Returns Integration

Verifies that:
1. MD_AUTH_TOKEN environment variable is set
2. morningstar-data package is installed
3. Morningstar Direct connection works
4. Returns data can be fetched

Run this first to verify your setup before building the full database.

Usage:
    $env:MD_AUTH_TOKEN="your-token-here"  # Windows PowerShell
    python quickstart_test.py
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def test_token_set() -> bool:
    """Test 1: Check if MD_AUTH_TOKEN is set."""
    print("Test 1: Checking MD_AUTH_TOKEN environment variable...")
    token = os.environ.get("MD_AUTH_TOKEN")
    if token:
        print(f"  [PASS] MD_AUTH_TOKEN is set ({len(token)} chars)")
        return True
    else:
        print("  [FAIL] MD_AUTH_TOKEN is not set")
        print()
        print("  Set your token with:")
        print("    Windows PowerShell: $env:MD_AUTH_TOKEN='your-token-here'")
        print("    Linux/Mac bash:     export MD_AUTH_TOKEN='your-token-here'")
        return False


def test_module_import() -> bool:
    """Test 2: Check if morningstar_returns module can be imported."""
    print("Test 2: Importing morningstar_returns module...")
    try:
        from morningstar_returns import MorningstarReturnsFetcher, ReturnsDatabase
        print("  [PASS] Module imported successfully")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_morningstar_package() -> bool:
    """Test 3: Check if morningstar-data package is installed."""
    print("Test 3: Checking morningstar-data package...")
    try:
        import morningstar_data as md
        print(f"  [PASS] morningstar-data package available")

        # Check for direct module
        if hasattr(md, "direct"):
            print(f"  [PASS] md.direct module available")
            return True
        else:
            print(f"  [WARN] md.direct module not found - may need different API")
            return True  # Still pass - might work with different API
    except ImportError as e:
        print(f"  [FAIL] morningstar-data not installed: {e}")
        print()
        print("  Install with:")
        print("    pip install morningstar-data")
        return False


def test_connection() -> bool:
    """Test 4: Test Morningstar Direct connection."""
    print("Test 4: Testing Morningstar Direct connection...")

    token = os.environ.get("MD_AUTH_TOKEN")
    if not token:
        print("  [SKIP] No token set")
        return False

    try:
        import morningstar_data as md
        md.init(token)
        print("  [PASS] Morningstar Direct initialized")
        return True
    except Exception as e:
        print(f"  [FAIL] Connection failed: {e}")
        print()
        print("  Possible causes:")
        print("    - Token expired (tokens typically last 24 hours)")
        print("    - Network connectivity issues")
        print("    - Invalid token format")
        return False


def test_fetch_returns() -> bool:
    """Test 5: Test fetching returns data."""
    print("Test 5: Fetching sample returns data...")

    token = os.environ.get("MD_AUTH_TOKEN")
    if not token:
        print("  [SKIP] No token set")
        return False

    try:
        import morningstar_data as md

        # Try to fetch XBI returns for last year
        end_date = date.today()
        start_date = end_date - timedelta(days=365)

        print(f"  Fetching XBI returns from {start_date} to {end_date}...")

        df = md.direct.returns(
            investments=["XBI:US"],
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            frequency="monthly",
        )

        if df is not None and not df.empty:
            num_obs = len(df)
            print(f"  [PASS] Retrieved {num_obs} observations")
            return True
        else:
            print("  [FAIL] No data returned")
            return False

    except Exception as e:
        print(f"  [FAIL] Fetch failed: {e}")
        return False


def test_database_class() -> bool:
    """Test 6: Test ReturnsDatabase class (no token needed)."""
    print("Test 6: Testing ReturnsDatabase class...")

    try:
        from morningstar_returns import ReturnsDatabase

        # Check if any database exists
        db_dir = Path("data/returns")
        if db_dir.exists():
            db_files = list(db_dir.glob("returns_db_*.json"))
            if db_files:
                db_path = db_files[0]
                print(f"  Found existing database: {db_path.name}")

                db = ReturnsDatabase(db_path)
                print(f"  [PASS] Database loaded: {len(db.available_tickers)} tickers")
                print(f"  [PASS] Date range: {db.date_range[0]} to {db.date_range[1]}")
                return True
            else:
                print("  [INFO] No database files found (build with build_returns_database.py)")
                return True  # Not a failure
        else:
            print("  [INFO] data/returns directory not found (will be created by build_returns_database.py)")
            return True  # Not a failure

    except Exception as e:
        print(f"  [FAIL] Database class error: {e}")
        return False


def main():
    print("=" * 60)
    print("MORNINGSTAR RETURNS INTEGRATION - QUICKSTART TEST")
    print("=" * 60)
    print()

    results = []

    # Run tests
    results.append(("Token Set", test_token_set()))
    print()

    results.append(("Module Import", test_module_import()))
    print()

    results.append(("Package Install", test_morningstar_package()))
    print()

    results.append(("Connection", test_connection()))
    print()

    results.append(("Fetch Returns", test_fetch_returns()))
    print()

    results.append(("Database Class", test_database_class()))
    print()

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    critical_failed = False

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

        if not passed:
            all_passed = False
            if name in ["Token Set", "Package Install", "Connection"]:
                critical_failed = True

    print()

    if all_passed:
        print("[SUCCESS] All tests passed!")
        print()
        print("Next steps:")
        print("  1. Build the returns database:")
        print("     python build_returns_database.py --universe example_universe.csv --start-date 2020-01-01")
        print()
        print("  2. Run validation (no token needed after database is built):")
        print("     python validate_signals.py --database data/returns/returns_db.json --ranked-list screen.csv")
        return 0

    elif critical_failed:
        print("[FAILED] Critical tests failed. Fix the issues above before proceeding.")
        return 1

    else:
        print("[PARTIAL] Some tests failed, but setup may still work.")
        print("Try building the database to see if it works.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
