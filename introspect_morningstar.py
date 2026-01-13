"""
Introspect morningstar_data package to discover available API methods
This will show us what we can actually call
"""
import os
import sys
import inspect

# Set your token here
TOKEN = "YOUR_TOKEN_HERE"  # Replace with your current token
os.environ['MD_AUTH_TOKEN'] = TOKEN

print("="*70)
print("MORNINGSTAR_DATA PACKAGE INTROSPECTION")
print("="*70)

try:
    import morningstar_data as md
    print("✅ Package imported successfully\n")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

# Discover main module structure
print("\n" + "="*70)
print("MAIN MODULE ATTRIBUTES")
print("="*70)
main_attrs = [attr for attr in dir(md) if not attr.startswith('_')]
for attr in main_attrs:
    obj = getattr(md, attr)
    obj_type = type(obj).__name__
    print(f"  • {attr:30s} ({obj_type})")

# Explore the 'direct' submodule in detail
if hasattr(md, 'direct'):
    print("\n" + "="*70)
    print("DIRECT SUBMODULE - Available Methods")
    print("="*70)
    
    direct_methods = [m for m in dir(md.direct) if not m.startswith('_')]
    
    for method_name in sorted(direct_methods):
        method = getattr(md.direct, method_name)
        
        if callable(method):
            try:
                # Try to get signature
                sig = inspect.signature(method)
                print(f"\n  📌 {method_name}{sig}")
                
                # Try to get docstring
                doc = inspect.getdoc(method)
                if doc:
                    # Print first line of docstring
                    first_line = doc.split('\n')[0]
                    print(f"     {first_line}")
            except:
                print(f"\n  📌 {method_name}()")
        else:
            print(f"\n  • {method_name} (attribute, not callable)")

# Check for other useful submodules
print("\n" + "="*70)
print("OTHER USEFUL MODULES/CLASSES")
print("="*70)

useful_items = ['investment', 'data_set', 'lookup', 'search_criteria']
for item_name in useful_items:
    if hasattr(md, item_name):
        print(f"\n  Found: md.{item_name}")
        item = getattr(md, item_name)
        if hasattr(item, '__all__'):
            print(f"    Exports: {getattr(item, '__all__')}")
        
        # Show key methods/attributes
        item_attrs = [a for a in dir(item) if not a.startswith('_')][:10]
        print(f"    Key attributes: {', '.join(item_attrs)}")

# Try to find search/query methods
print("\n" + "="*70)
print("SEARCHING FOR DATA RETRIEVAL METHODS")
print("="*70)

keywords = ['search', 'get', 'lookup', 'query', 'investment', 'data', 'security']
for keyword in keywords:
    matching = []
    for attr in dir(md):
        if keyword.lower() in attr.lower() and not attr.startswith('_'):
            matching.append(attr)
    
    if matching:
        print(f"\n  Methods containing '{keyword}':")
        for m in matching:
            print(f"    • md.{m}")

# Check if there's a way to list available data points
print("\n" + "="*70)
print("CHECKING FOR DATA CATALOG/SCHEMA INFO")
print("="*70)

catalog_methods = ['get_data_sets', 'list_data_sets', 'get_schema', 'get_data_points']
for method_name in catalog_methods:
    if hasattr(md.direct, method_name):
        print(f"\n  Found: md.direct.{method_name}")
        try:
            method = getattr(md.direct, method_name)
            sig = inspect.signature(method)
            print(f"    Signature: {sig}")
        except:
            pass

print("\n" + "="*70)
print("INTROSPECTION COMPLETE")
print("="*70)
print("\nNext: Use the methods discovered above to build data retrieval code")
