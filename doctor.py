#!/usr/bin/env python3
"""
doctor.py - Health Check for Biotech Screener Pipeline

Checks:
1. Python version compatibility
2. Required dependencies installed
3. Required input files exist
4. Input file schemas are valid
5. Configuration is consistent

Usage:
    python doctor.py
    python doctor.py --data-dir ./production_data
    python doctor.py --verbose
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Version requirements
REQUIRED_PYTHON_VERSION = (3, 10)
REQUIRED_PACKAGES = [
    "yaml",  # PyYAML
]

# Schema definitions for input validation
UNIVERSE_REQUIRED_FIELDS = {"ticker"}
UNIVERSE_RECOMMENDED_FIELDS = {"company_name", "sector", "market_cap_mm", "is_active"}

FINANCIAL_REQUIRED_FIELDS = {"ticker"}
FINANCIAL_RECOMMENDED_FIELDS = {"Cash", "NetIncome", "CFO", "R&D"}

TRIAL_REQUIRED_FIELDS = {"ticker", "nct_id"}
TRIAL_RECOMMENDED_FIELDS = {"overall_status", "phase", "last_update_posted", "primary_completion_date"}

MARKET_REQUIRED_FIELDS = {"ticker"}
MARKET_RECOMMENDED_FIELDS = {"market_cap", "price", "avg_volume"}


class HealthCheck:
    """Health check results container"""

    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def error(self, msg: str):
        self.errors.append(msg)
        self.passed = False

    def warning(self, msg: str):
        self.warnings.append(msg)

    def add_info(self, msg: str):
        self.info.append(msg)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}"


def check_python_version() -> HealthCheck:
    """Check Python version compatibility"""
    check = HealthCheck("Python Version")

    current = sys.version_info[:2]
    required = REQUIRED_PYTHON_VERSION

    check.add_info(f"Current: Python {current[0]}.{current[1]}")
    check.add_info(f"Required: Python {required[0]}.{required[1]}+")

    if current < required:
        check.error(f"Python {required[0]}.{required[1]}+ required, got {current[0]}.{current[1]}")

    return check


def check_dependencies() -> HealthCheck:
    """Check required packages are installed"""
    check = HealthCheck("Dependencies")

    for package in REQUIRED_PACKAGES:
        try:
            __import__(package)
            check.add_info(f"{package}: installed")
        except ImportError:
            check.warning(f"{package}: not installed (optional)")

    # Check core imports
    core_modules = [
        ("common.date_utils", "Date utilities"),
        ("common.pit_enforcement", "PIT enforcement"),
        ("common.integration_contracts", "Integration contracts"),
        ("module_1_universe", "Module 1"),
        ("module_2_financial", "Module 2"),
        ("module_3_catalyst", "Module 3"),
        ("module_4_clinical_dev", "Module 4"),
        ("module_5_composite_with_defensive", "Module 5"),
    ]

    for module_name, description in core_modules:
        try:
            __import__(module_name)
            check.add_info(f"{description}: OK")
        except ImportError as e:
            check.error(f"{description}: Import failed - {e}")

    return check


def check_file_exists(path: Path, description: str, required: bool = True) -> Tuple[bool, str]:
    """Check if file exists"""
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        return True, f"{description}: OK ({size_mb:.2f} MB)"
    elif required:
        return False, f"{description}: MISSING (required)"
    else:
        return True, f"{description}: not found (optional)"


def validate_json_schema(
    data: List[Dict[str, Any]],
    required_fields: set,
    recommended_fields: set,
    name: str
) -> Tuple[List[str], List[str]]:
    """Validate JSON data against expected schema"""
    errors = []
    warnings = []

    if not data:
        errors.append(f"{name}: Empty dataset")
        return errors, warnings

    # Sample first record
    sample = data[0]
    present_fields = set(sample.keys())

    # Check required fields
    missing_required = required_fields - present_fields
    if missing_required:
        errors.append(f"{name}: Missing required fields: {missing_required}")

    # Check recommended fields
    missing_recommended = recommended_fields - present_fields
    if missing_recommended:
        warnings.append(f"{name}: Missing recommended fields: {missing_recommended}")

    # Check all records have required fields
    records_missing_required = 0
    for i, record in enumerate(data):
        if not required_fields.issubset(set(record.keys())):
            records_missing_required += 1

    if records_missing_required > 0:
        errors.append(f"{name}: {records_missing_required}/{len(data)} records missing required fields")

    return errors, warnings


def check_input_files(data_dir: Path) -> HealthCheck:
    """Check all required input files exist and have valid schemas"""
    check = HealthCheck("Input Files")

    required_files = [
        ("universe.json", "Universe data", UNIVERSE_REQUIRED_FIELDS, UNIVERSE_RECOMMENDED_FIELDS),
        ("financial_records.json", "Financial records", FINANCIAL_REQUIRED_FIELDS, FINANCIAL_RECOMMENDED_FIELDS),
        ("trial_records.json", "Trial records", TRIAL_REQUIRED_FIELDS, TRIAL_RECOMMENDED_FIELDS),
        ("market_data.json", "Market data", MARKET_REQUIRED_FIELDS, MARKET_RECOMMENDED_FIELDS),
    ]

    for filename, description, required_fields, recommended_fields in required_files:
        filepath = data_dir / filename
        exists, msg = check_file_exists(filepath, description)

        if exists and filepath.exists():
            check.add_info(msg)

            # Validate schema
            try:
                with open(filepath) as f:
                    data = json.load(f)

                if isinstance(data, list):
                    check.add_info(f"  Records: {len(data)}")
                    errors, warnings = validate_json_schema(data, required_fields, recommended_fields, description)

                    for err in errors:
                        check.error(err)
                    for warn in warnings:
                        check.warning(warn)
                else:
                    check.error(f"{description}: Expected JSON array, got {type(data).__name__}")

            except json.JSONDecodeError as e:
                check.error(f"{description}: Invalid JSON - {e}")
            except Exception as e:
                check.error(f"{description}: Read error - {e}")
        else:
            check.error(msg)

    # Check optional files
    optional_files = [
        ("coinvest_signals.json", "Co-invest signals"),
        ("short_interest.json", "Short interest"),
        ("market_snapshot.json", "Market snapshot"),
    ]

    for filename, description in optional_files:
        filepath = data_dir / filename
        exists, msg = check_file_exists(filepath, description, required=False)
        check.add_info(msg)

    return check


def check_ctgov_state(data_dir: Path) -> HealthCheck:
    """Check CT.gov state directory"""
    check = HealthCheck("CT.gov State")

    state_dir = data_dir / "ctgov_state"

    if not state_dir.exists():
        check.warning("State directory not found (will be created on first run)")
        return check

    # Count state files
    state_files = list(state_dir.glob("state_*.jsonl"))
    check.add_info(f"State snapshots: {len(state_files)}")

    if state_files:
        # Check latest snapshot
        latest = max(state_files, key=lambda p: p.stem)
        size_kb = latest.stat().st_size / 1024
        check.add_info(f"Latest snapshot: {latest.name} ({size_kb:.1f} KB)")

        # Check if manifest exists
        manifest = state_dir / "manifest.json"
        if manifest.exists():
            check.add_info("Manifest: OK")
        else:
            check.warning("Manifest: not found")

    return check


def check_config() -> HealthCheck:
    """Check configuration file"""
    check = HealthCheck("Configuration")

    config_path = Path("config.yml")

    if not config_path.exists():
        check.warning("config.yml not found - using defaults")
        return check

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        check.add_info(f"Config version: {config.get('version', 'unknown')}")

        # Check required sections
        required_sections = ["paths", "module_1", "module_2", "module_3", "module_4", "module_5"]
        for section in required_sections:
            if section not in config:
                check.warning(f"Missing section: {section}")
            else:
                check.add_info(f"Section {section}: OK")

        # Validate weights sum
        m5_weights = config.get("module_5", {}).get("weights", {})
        if m5_weights:
            total = sum(m5_weights.values())
            if abs(total - 1.0) > 0.001:
                check.error(f"Module 5 weights sum to {total}, expected 1.0")
            else:
                check.add_info(f"Module 5 weights: {total} (OK)")

    except ImportError:
        check.warning("PyYAML not installed - cannot validate config.yml")
    except Exception as e:
        check.error(f"Config error: {e}")

    return check


def check_output_directory(data_dir: Path) -> HealthCheck:
    """Check output directory permissions"""
    check = HealthCheck("Output Directory")

    if not data_dir.exists():
        check.error(f"Data directory does not exist: {data_dir}")
        return check

    # Check write permissions
    test_file = data_dir / ".doctor_test"
    try:
        test_file.write_text("test")
        test_file.unlink()
        check.add_info("Write permissions: OK")
    except Exception as e:
        check.error(f"Cannot write to data directory: {e}")

    return check


def run_all_checks(data_dir: Path, verbose: bool = False) -> bool:
    """Run all health checks"""

    print("=" * 60)
    print("BIOTECH-SCREENER HEALTH CHECK")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Current date: {date.today()}")
    print()

    checks = [
        check_python_version(),
        check_dependencies(),
        check_config(),
        check_input_files(data_dir),
        check_ctgov_state(data_dir),
        check_output_directory(data_dir),
    ]

    all_passed = True

    for check in checks:
        print(check)

        if verbose:
            for info in check.info:
                print(f"  [INFO] {info}")

        for warning in check.warnings:
            print(f"  [WARN] {warning}")

        for error in check.errors:
            print(f"  [ERR]  {error}")

        if not check.passed:
            all_passed = False

        print()

    # Summary
    print("=" * 60)
    if all_passed:
        print("RESULT: ALL CHECKS PASSED")
        print()
        print("You can run the pipeline with:")
        print(f"  python run_screen.py --as-of-date {date.today()} --data-dir {data_dir} --output results.json")
    else:
        print("RESULT: SOME CHECKS FAILED")
        print()
        print("Fix the errors above before running the pipeline.")
    print("=" * 60)

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Health check for biotech screener pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python doctor.py                           # Check with default data dir
    python doctor.py --data-dir ./my_data      # Check specific data directory
    python doctor.py --verbose                 # Show all info messages

Fix instructions:
    - Missing files: Ensure all required input files are present
    - Invalid JSON: Check file encoding and syntax
    - Missing fields: Update input files with required fields
    - Write permissions: Check directory ownership and permissions
        """
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("production_data"),
        help="Data directory to check (default: production_data)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose output (all info messages)"
    )

    args = parser.parse_args()

    success = run_all_checks(args.data_dir, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
