"""
Debug script to discover correct Morningstar API usage
"""
import os
print("="*60)
print("MORNINGSTAR API DIAGNOSTIC")
print("="*60)

# Check token
print("\n[1] Checking MD_AUTH_TOKEN...")
token = os.environ.get('MD_AUTH_TOKEN')
if token:
    print(f"    Token set ({len(token)} chars)")
else:
    print("    Token NOT set - set it with:")
    print("    $env:MD_AUTH_TOKEN='your-token'")
    exit(1)

# Import package
print("\n[2] Importing morningstar_data...")
try:
    import morningstar_data as md
    print(f"    Package imported")
    print(f"    Version: {getattr(md, '__version__', 'unknown')}")
except ImportError as e:
    print(f"    FAILED: {e}")
    exit(1)

# Check direct module
print("\n[3] Checking md.direct module...")
if hasattr(md, 'direct'):
    print(f"    md.direct exists")

    # List available methods
    methods = [m for m in dir(md.direct) if not m.startswith('_')]
    print(f"    Available methods: {methods[:10]}...")
else:
    print("    md.direct NOT found")
    exit(1)

# Check Frequency enum
print("\n[4] Looking for Frequency enum...")

# Try different import paths
freq_locations = [
    'morningstar_data.direct.data_type',
    'morningstar_data.direct',
    'morningstar_data',
]

Frequency = None
for loc in freq_locations:
    try:
        module = __import__(loc, fromlist=['Frequency'])
        if hasattr(module, 'Frequency'):
            Frequency = module.Frequency
            print(f"    Found at: {loc}")
            print(f"    Values: {[v for v in dir(Frequency) if not v.startswith('_')]}")
            break
    except (ImportError, AttributeError):
        continue

if Frequency is None:
    print("    Frequency enum NOT found in common locations")

# Check returns function signature
print("\n[5] Checking md.direct.returns signature...")
import inspect
if hasattr(md.direct, 'returns'):
    sig = inspect.signature(md.direct.returns)
    print(f"    Signature: {sig}")
    print(f"    Parameters: {list(sig.parameters.keys())}")

# Check get_returns function signature
print("\n[6] Checking md.direct.get_returns signature...")
if hasattr(md.direct, 'get_returns'):
    sig = inspect.signature(md.direct.get_returns)
    print(f"    Signature: {sig}")
    print(f"    Parameters: {list(sig.parameters.keys())}")
else:
    print("    get_returns NOT found")

# Try a simple fetch
print("\n[7] Attempting simple returns fetch...")
try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Try with minimal parameters
        df = md.direct.returns(
            investments=['XBI:US'],
            start_date='2024-01-01',
            end_date='2024-06-30',
            freq='monthly'  # Note: 'freq' not 'frequency'
        )

        print(f"    SUCCESS!")
        print(f"    DataFrame shape: {df.shape}")
        print(f"    Columns: {list(df.columns)}")
        print(f"    Index: {df.index[:3].tolist()}...")
        print(f"\n    Sample data:")
        print(df.head(3))

except Exception as e:
    print(f"    FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
