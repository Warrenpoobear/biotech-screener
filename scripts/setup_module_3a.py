#!/usr/bin/env python3
"""
setup_module_3a.py - Setup and Verification Script for Module 3A

Verifies installation, dependencies, and data readiness.
"""

import sys
import json
from pathlib import Path
from datetime import date


def print_header(text):
    """Print formatted header"""
    print()
    print("=" * 80)
    print(text)
    print("=" * 80)
    print()


def print_success(text):
    """Print success message"""
    print(f"✅ {text}")


def print_warning(text):
    """Print warning message"""
    print(f"⚠️  {text}")


def print_error(text):
    """Print error message"""
    print(f"❌ {text}")


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 9:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} ✓")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor} detected - need 3.9+")
        return False


def check_dependencies():
    """Check required Python packages"""
    print("\nChecking dependencies...")
    
    required = {
        'json': 'stdlib',
        'pathlib': 'stdlib',
        'datetime': 'stdlib',
        'dataclasses': 'stdlib',
        'enum': 'stdlib',
        'typing': 'stdlib',
        'logging': 'stdlib',
        'hashlib': 'stdlib'
    }
    
    all_ok = True
    for package, source in required.items():
        try:
            __import__(package)
            print_success(f"{package} ({source})")
        except ImportError:
            print_error(f"{package} - MISSING")
            all_ok = False
    
    return all_ok


def check_module_files():
    """Check Module 3A files exist"""
    print("\nChecking Module 3A files...")
    
    required_files = [
        'ctgov_adapter.py',
        'state_management.py',
        'event_detector.py',
        'catalyst_summary.py',
        'module_3_catalyst.py',
        'test_module_3a.py'
    ]
    
    all_ok = True
    for filename in required_files:
        if Path(filename).exists():
            print_success(f"{filename}")
        else:
            print_error(f"{filename} - MISSING")
            all_ok = False
    
    return all_ok


def check_data_files():
    """Check required data files"""
    print("\nChecking data files...")
    
    data_dir = Path("production_data")
    
    if not data_dir.exists():
        print_error("production_data/ directory not found")
        return False
    
    trial_records = data_dir / "trial_records.json"
    
    if not trial_records.exists():
        print_error("trial_records.json not found")
        return False
    
    print_success("production_data/ directory exists")
    
    # Check trial_records.json has dates
    try:
        with open(trial_records) as f:
            records = json.load(f)
        
        if not records:
            print_error("trial_records.json is empty")
            return False
        
        sample = records[0]
        has_dates = sample.get('last_update_posted') is not None
        
        if has_dates:
            print_success(f"trial_records.json ({len(records)} trials with dates)")
        else:
            print_warning(f"trial_records.json ({len(records)} trials) - MISSING DATES!")
            print("         Run: python backfill_ctgov_dates.py")
            return False
    
    except Exception as e:
        print_error(f"Failed to read trial_records.json: {e}")
        return False
    
    return True


def test_imports():
    """Test importing Module 3A components"""
    print("\nTesting imports...")
    
    components = [
        ('ctgov_adapter', ['CTGovStatus', 'CanonicalTrialRecord']),
        ('state_management', ['StateStore', 'StateSnapshot']),
        ('event_detector', ['EventDetector', 'EventType']),
        ('catalyst_summary', ['CatalystAggregator', 'TickerCatalystSummary']),
        ('module_3_catalyst', ['compute_module_3_catalyst', 'Module3Config'])
    ]
    
    all_ok = True
    for module_name, classes in components:
        try:
            module = __import__(module_name)
            for class_name in classes:
                if hasattr(module, class_name):
                    print_success(f"{module_name}.{class_name}")
                else:
                    print_error(f"{module_name}.{class_name} - NOT FOUND")
                    all_ok = False
        except ImportError as e:
            print_error(f"{module_name} - IMPORT FAILED: {e}")
            all_ok = False
    
    return all_ok


def verify_data_schema():
    """Verify trial_records.json schema"""
    print("\nVerifying data schema...")
    
    trial_records = Path("production_data/trial_records.json")
    
    if not trial_records.exists():
        print_error("trial_records.json not found")
        return False
    
    try:
        with open(trial_records) as f:
            records = json.load(f)
        
        sample = records[0]
        
        required_fields = [
            'ticker',
            'nct_id',
            'status',
            'last_update_posted'
        ]
        
        optional_fields = [
            'primary_completion_date',
            'primary_completion_type',
            'completion_date',
            'completion_type',
            'results_first_posted'
        ]
        
        all_ok = True
        
        print("\n  Required fields:")
        for field in required_fields:
            if field in sample:
                value = sample[field]
                if value is not None:
                    print_success(f"  {field}: {value}")
                else:
                    print_error(f"  {field}: NULL")
                    all_ok = False
            else:
                print_error(f"  {field}: MISSING")
                all_ok = False
        
        print("\n  Optional fields:")
        for field in optional_fields:
            if field in sample:
                value = sample[field]
                if value is not None:
                    print_success(f"  {field}: {value}")
                else:
                    print_warning(f"  {field}: NULL (ok)")
            else:
                print_warning(f"  {field}: MISSING (ok)")
        
        return all_ok
    
    except Exception as e:
        print_error(f"Schema verification failed: {e}")
        return False


def run_quick_test():
    """Run a quick smoke test"""
    print("\nRunning quick smoke test...")
    
    try:
        from ctgov_adapter import process_trial_records_batch, AdapterConfig
        
        # Load trial records
        with open("production_data/trial_records.json") as f:
            records = json.load(f)
        
        # Test adapter on first 5 records
        test_records = records[:5]
        
        config = AdapterConfig()
        canonical_records, stats = process_trial_records_batch(
            test_records,
            date.today(),
            config
        )
        
        print_success(f"Processed {len(canonical_records)}/{len(test_records)} records")
        print(f"         Success rate: {stats['success_count']}/{stats['total_records']}")
        
        return len(canonical_records) > 0
    
    except Exception as e:
        print_error(f"Smoke test failed: {e}")
        return False


def main():
    """Run all checks"""
    print_header("MODULE 3A SETUP & VERIFICATION")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Module Files", check_module_files),
        ("Data Files", check_data_files),
        ("Import Tests", test_imports),
        ("Data Schema", verify_data_schema),
        ("Smoke Test", run_quick_test)
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"{name} check crashed: {e}")
            results.append((name, False))
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    all_passed = all(result for _, result in results)
    
    for name, result in results:
        if result:
            print_success(f"{name}")
        else:
            print_error(f"{name}")
    
    print()
    
    if all_passed:
        print("="*80)
        print("🎉 ALL CHECKS PASSED - MODULE 3A IS READY!")
        print("="*80)
        print()
        print("Next steps:")
        print("1. Run standalone: python module_3_catalyst.py --as-of-date 2026-01-06")
        print("2. Run tests: python test_module_3a.py")
        print("3. Integrate with run_screen.py")
        print()
        return 0
    else:
        print("="*80)
        print("⚠️  SOME CHECKS FAILED - SEE ERRORS ABOVE")
        print("="*80)
        print()
        print("Fix the errors and run setup_module_3a.py again")
        print()
        return 1


if __name__ == "__main__":
    exit(main())
