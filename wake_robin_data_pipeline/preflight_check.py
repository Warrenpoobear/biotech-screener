#!/usr/bin/env python3
"""
preflight_check.py - Pre-flight verification before running production collection

Tests network connectivity to all data sources and validates API access.
Run this BEFORE collect_universe_data.py to catch issues early.
"""
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

def test_yahoo_finance_access():
    """Test Yahoo Finance connectivity with a known ticker."""
    print("\n1. Testing Yahoo Finance access...")
    print("   Target: finance.yahoo.com")
    
    try:
        import yfinance as yf
        
        # Test with a highly liquid, reliable ticker
        print("   └─ Fetching AAPL (test ticker)...", end=" ")
        stock = yf.Ticker("AAPL")
        
        # Try to get basic info
        info = stock.info
        price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        
        if price > 0:
            print(f"✓")
            print(f"      Current price: ${price:.2f}")
            print(f"      Status: Yahoo Finance accessible")
            return True, "Yahoo Finance operational"
        else:
            print(f"✗")
            print(f"      Status: No price data returned")
            return False, "Yahoo Finance returned no data"
            
    except Exception as e:
        print(f"✗")
        print(f"      Error: {e}")
        return False, f"Yahoo Finance failed: {e}"

def test_sec_edgar_access():
    """Test SEC EDGAR API connectivity."""
    print("\n2. Testing SEC EDGAR access...")
    print("   Target: data.sec.gov")
    
    try:
        import requests
        
        # Test with Apple's CIK
        cik = "0000320193"
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        headers = {'User-Agent': 'Wake Robin Research contact@wakerobincapital.com'}
        
        print(f"   └─ Fetching CIK {cik} (Apple test)...", end=" ")
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓")
            print(f"      Status: SEC EDGAR API accessible")
            print(f"      Entity: {data.get('entityName', 'Unknown')}")
            return True, "SEC EDGAR operational"
        else:
            print(f"✗")
            print(f"      HTTP {response.status_code}")
            return False, f"SEC EDGAR returned HTTP {response.status_code}"
            
    except Exception as e:
        print(f"✗")
        print(f"      Error: {e}")
        return False, f"SEC EDGAR failed: {e}"

def test_clinicaltrials_gov_access():
    """Test ClinicalTrials.gov API connectivity."""
    print("\n3. Testing ClinicalTrials.gov access...")
    print("   Target: clinicaltrials.gov")
    
    try:
        import requests
        
        # Test with a simple search
        url = "https://clinicaltrials.gov/api/v2/studies"
        params = {
            'query.spons': 'Pfizer',
            'pageSize': 1,
            'format': 'json'
        }
        
        print(f"   └─ Testing API v2 endpoint...", end=" ")
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            total = data.get('totalCount', 0)
            print(f"✓")
            print(f"      Status: ClinicalTrials.gov API accessible")
            print(f"      Test query returned {total} results")
            return True, "ClinicalTrials.gov operational"
        else:
            print(f"✗")
            print(f"      HTTP {response.status_code}")
            return False, f"ClinicalTrials.gov returned HTTP {response.status_code}"
            
    except Exception as e:
        print(f"✗")
        print(f"      Error: {e}")
        return False, f"ClinicalTrials.gov failed: {e}"

def test_network_general():
    """Test general internet connectivity."""
    print("\n4. Testing general network connectivity...")
    
    try:
        import requests
        
        print("   └─ Checking DNS resolution...", end=" ")
        response = requests.get("https://www.google.com", timeout=10)
        
        if response.status_code == 200:
            print(f"✓")
            print(f"      Status: Internet connection active")
            return True, "Network operational"
        else:
            print(f"✗")
            return False, "Network connectivity issues"
            
    except Exception as e:
        print(f"✗")
        print(f"      Error: {e}")
        return False, f"Network test failed: {e}"

def test_dependencies():
    """Test that all required packages are installed."""
    print("\n5. Testing Python dependencies...")
    
    required = {
        'yfinance': 'Yahoo Finance API client',
        'requests': 'HTTP library for APIs',
    }
    
    missing = []
    
    for package, description in required.items():
        try:
            print(f"   └─ {package} ({description})...", end=" ")
            __import__(package)
            print("✓")
        except ImportError:
            print("✗")
            missing.append(package)
    
    if missing:
        return False, f"Missing packages: {', '.join(missing)}"
    
    return True, "All dependencies installed"

def main():
    """Run all pre-flight checks."""
    print("\n" + "="*70)
    print("WAKE ROBIN DATA PIPELINE - PRE-FLIGHT CHECK")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print("Environment: Production deployment verification")
    
    checks = [
        ("Network Connectivity", test_network_general),
        ("Python Dependencies", test_dependencies),
        ("Yahoo Finance API", test_yahoo_finance_access),
        ("SEC EDGAR API", test_sec_edgar_access),
        ("ClinicalTrials.gov API", test_clinicaltrials_gov_access),
    ]
    
    results = {}
    passed = 0
    failed = 0
    
    for check_name, check_func in checks:
        success, message = check_func()
        results[check_name] = {'success': success, 'message': message}
        
        if success:
            passed += 1
        else:
            failed += 1
        
        # Brief pause between checks
        time.sleep(1)
    
    # Summary
    print("\n" + "="*70)
    print(f"PRE-FLIGHT RESULTS: {passed}/{len(checks)} checks passed")
    print("="*70 + "\n")
    
    if failed == 0:
        print("✅ ALL SYSTEMS GO - Ready for data collection")
        print("\nYou can now run:")
        print("  python collect_universe_data.py\n")
        return 0
    else:
        print("⚠️  SOME CHECKS FAILED - Review issues before proceeding")
        print("\nFailed checks:")
        for check_name, result in results.items():
            if not result['success']:
                print(f"  • {check_name}: {result['message']}")
        
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Verify firewall/proxy settings")
        print("  3. Install missing dependencies: pip install -r requirements.txt")
        print("  4. If network restrictions exist, contact IT support\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
