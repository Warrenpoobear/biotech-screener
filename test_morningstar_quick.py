"""
Quick test for Morningstar Returns - matches morningstar_returns_fixed.py API
"""
import os
import sys

print("="*60)
print("MORNINGSTAR RETURNS - QUICK TEST")
print("="*60)

# Test 1: Token
print("\n[Test 1] Checking MD_AUTH_TOKEN...")
if not os.environ.get('MD_AUTH_TOKEN'):
    print("FAIL: MD_AUTH_TOKEN not set")
    sys.exit(1)
print(f"PASS: Token set ({len(os.environ['MD_AUTH_TOKEN'])} chars)")

# Test 2: Import module
print("\n[Test 2] Importing morningstar_returns...")
try:
    import morningstar_returns as mr
    print("PASS: Module imported")
except ImportError as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# Test 3: Check availability
print("\n[Test 3] Checking availability...")
try:
    available, msg = mr.check_availability()
    if available:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}")
        sys.exit(1)
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# Test 4: Fetch returns
print("\n[Test 4] Fetching returns (XBI, 2023)...")
try:
    data = mr.fetch_returns(
        sec_ids=['XBI:US'],
        start_date='2023-01-01',
        end_date='2023-12-31',
        frequency='monthly'
    )

    num_records = len(data.get('absolute', []))
    print(f"PASS: Retrieved {num_records} records")

    if num_records > 0:
        print(f"      Sample: {data['absolute'][0]}")

    print(f"      SHA256: {data['provenance']['sha256'][:16]}...")

except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Cache operations
print("\n[Test 5] Testing cache operations...")
try:
    import tempfile
    import shutil

    test_dir = tempfile.mkdtemp()

    # Save
    cache_path = mr.save_returns_cache(data, test_dir, 'test.json')
    print(f"PASS: Saved to {cache_path}")

    # Load
    loaded = mr.load_returns_cache(cache_path)
    assert loaded['provenance']['sha256'] == data['provenance']['sha256']
    print("PASS: Loaded and verified SHA256")

    # Cleanup
    shutil.rmtree(test_dir)

except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
print("\nYour Morningstar integration is working correctly.")
print("Next: Run build_returns_database.py to build your returns database.")
