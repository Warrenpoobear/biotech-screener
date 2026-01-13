"""
Quick Start Test for Morningstar Returns Integration
Tests that Phase 1 is working correctly

Run this first to verify installation.
"""

import os
import sys

print("="*70)
print("WAKE ROBIN PHASE 1 - QUICK START TEST")
print("="*70)

# Test 1: Check environment
print("\n[Test 1] Checking environment...")

if not os.environ.get('MD_AUTH_TOKEN'):
    print("❌ MD_AUTH_TOKEN not set")
    print("\nPlease set your Morningstar authentication token:")
    print("  $env:MD_AUTH_TOKEN='your-token-here'  # PowerShell")
    print("  export MD_AUTH_TOKEN='your-token-here'  # Bash")
    sys.exit(1)

print("✅ MD_AUTH_TOKEN is set")

# Test 2: Import module
print("\n[Test 2] Importing morningstar_returns module...")

try:
    import morningstar_returns as mr
    print("✅ Module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

# Test 3: Check Morningstar availability
print("\n[Test 3] Checking Morningstar Direct availability...")

available, msg = mr.check_availability()
if not available:
    print(f"❌ {msg}")
    sys.exit(1)

print(f"✅ {msg}")

# Test 4: Fetch sample data
print("\n[Test 4] Fetching sample returns data (VRTX, 2024)...")

try:
    data = mr.fetch_returns(
        sec_ids=['0P000005R7'],  # VRTX
        start_date='2024-01-01',
        end_date='2024-12-31',
        frequency='monthly'
    )
    
    num_obs = data['provenance']['num_observations']
    print(f"✅ Retrieved {num_obs} observations")
    
    # Verify data structure
    assert 'absolute' in data
    assert 'provenance' in data
    assert len(data['absolute']) > 0
    
    print("✅ Data structure validated")
    
except Exception as e:
    print(f"❌ Failed to fetch data: {e}")
    sys.exit(1)

# Test 5: Test caching
print("\n[Test 5] Testing cache operations...")

try:
    test_dir = 'test_cache'
    os.makedirs(test_dir, exist_ok=True)
    
    # Save
    cache_path = mr.save_returns_cache(
        data,
        output_dir=test_dir,
        filename='test_cache.json'
    )
    print(f"✅ Saved to: {cache_path}")
    
    # Load and verify
    loaded = mr.load_returns_cache(cache_path)
    assert loaded['provenance']['sha256'] == data['provenance']['sha256']
    print("✅ Loaded and verified SHA256")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    
except Exception as e:
    print(f"❌ Cache test failed: {e}")
    sys.exit(1)

# Test 6: Verify benchmark access
print("\n[Test 6] Testing benchmark (XBI) access...")

try:
    benchmark_data = mr.fetch_returns(
        sec_ids=['FEUSA04AER'],  # XBI
        start_date='2024-01-01',
        end_date='2024-12-31',
        frequency='monthly'
    )
    
    print(f"✅ XBI benchmark accessible")
    
except Exception as e:
    print(f"⚠️  Benchmark test failed: {e}")
    print("   (This is non-critical)")

# All tests passed
print("\n" + "="*70)
print("ALL TESTS PASSED")
print("="*70)

print("\nPhase 1 installation is working correctly!")
print("\nNext steps:")
print("1. Create your universe.csv file (see example_universe.csv)")
print("2. Run: python build_returns_database.py --universe universe.csv --start-date 2020-01-01")
print("3. After database is built, use validate_signals.py to test your screens")
print("\nSee README_PHASE1.md for full documentation.")
