#!/usr/bin/env python3
"""
validate_pipeline.py - Comprehensive validation for Wake Robin data pipeline

Tests all components and generates deployment readiness report.
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

class PipelineValidator:
    """Validates data pipeline readiness for production deployment."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "overall_status": "UNKNOWN",
            "deployment_ready": False
        }
    
    def test_imports(self) -> Tuple[bool, str]:
        """Test that all required modules can be imported."""
        try:
            import yfinance
            import requests
            from collectors import yahoo_collector, sec_collector, trials_collector
            return True, "All required modules imported successfully"
        except ImportError as e:
            return False, f"Import failed: {e}"
    
    def test_universe_config(self) -> Tuple[bool, str]:
        """Test universe configuration file."""
        universe_path = Path(__file__).parent / "universe" / "pilot_universe.json"
        
        if not universe_path.exists():
            return False, f"Universe file not found: {universe_path}"
        
        try:
            with open(universe_path) as f:
                universe = json.load(f)
            
            # Validate structure
            if 'tickers' not in universe:
                return False, "Universe missing 'tickers' key"
            
            tickers = universe['tickers']
            if len(tickers) != 20:
                return False, f"Expected 20 tickers, found {len(tickers)}"
            
            # Validate each ticker entry
            required_fields = ['ticker', 'company', 'stage', 'primary_indication']
            for ticker_info in tickers:
                missing = [f for f in required_fields if f not in ticker_info]
                if missing:
                    return False, f"Ticker {ticker_info.get('ticker', 'UNKNOWN')} missing fields: {missing}"
            
            return True, f"Universe configuration valid: {len(tickers)} tickers"
            
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def test_directory_structure(self) -> Tuple[bool, str]:
        """Test that all required directories exist or can be created."""
        base_dir = Path(__file__).parent
        
        required_dirs = [
            base_dir / "collectors",
            base_dir / "universe",
            base_dir / "cache",
            base_dir / "outputs"
        ]
        
        missing = []
        for dir_path in required_dirs:
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    missing.append(f"{dir_path.name}: {e}")
        
        if missing:
            return False, f"Could not create directories: {', '.join(missing)}"
        
        return True, "All required directories present or created"
    
    def test_cache_system(self) -> Tuple[bool, str]:
        """Test cache read/write operations."""
        cache_dir = Path(__file__).parent / "cache"
        test_file = cache_dir / "test_cache.json"
        
        try:
            # Test write
            test_data = {"test": "data", "timestamp": datetime.now().isoformat()}
            with open(test_file, 'w') as f:
                json.dump(test_data, f)
            
            # Test read
            with open(test_file) as f:
                loaded = json.load(f)
            
            if loaded != test_data:
                return False, "Cache data mismatch"
            
            # Cleanup
            test_file.unlink()
            
            return True, "Cache system operational"
            
        except Exception as e:
            return False, f"Cache test failed: {e}"
    
    def test_collector_interfaces(self) -> Tuple[bool, str]:
        """Test that collector modules have correct interfaces."""
        try:
            from collectors import yahoo_collector, sec_collector, trials_collector
            
            # Test Yahoo collector interface
            if not hasattr(yahoo_collector, 'collect_yahoo_data'):
                return False, "yahoo_collector missing collect_yahoo_data function"
            if not hasattr(yahoo_collector, 'collect_batch'):
                return False, "yahoo_collector missing collect_batch function"
            
            # Test SEC collector interface
            if not hasattr(sec_collector, 'collect_sec_data'):
                return False, "sec_collector missing collect_sec_data function"
            if not hasattr(sec_collector, 'collect_batch'):
                return False, "sec_collector missing collect_batch function"
            
            # Test Trials collector interface
            if not hasattr(trials_collector, 'collect_trials_data'):
                return False, "trials_collector missing collect_trials_data function"
            if not hasattr(trials_collector, 'collect_batch'):
                return False, "trials_collector missing collect_batch function"
            
            return True, "All collector interfaces validated"
            
        except Exception as e:
            return False, f"Interface validation failed: {e}"
    
    def test_orchestrator(self) -> Tuple[bool, str]:
        """Test main orchestrator script."""
        orchestrator_path = Path(__file__).parent / "collect_universe_data.py"
        
        if not orchestrator_path.exists():
            return False, "Orchestrator script not found"
        
        # Basic syntax check
        try:
            with open(orchestrator_path) as f:
                code = f.read()
            compile(code, str(orchestrator_path), 'exec')
            return True, "Orchestrator script validated"
        except SyntaxError as e:
            return False, f"Syntax error in orchestrator: {e}"
        except Exception as e:
            return False, f"Orchestrator validation failed: {e}"
    
    def test_output_permissions(self) -> Tuple[bool, str]:
        """Test that output directory is writable."""
        output_dir = Path(__file__).parent / "outputs"
        test_file = output_dir / "test_write.json"
        
        try:
            with open(test_file, 'w') as f:
                json.dump({"test": "write"}, f)
            test_file.unlink()
            return True, "Output directory writable"
        except Exception as e:
            return False, f"Cannot write to output directory: {e}"
    
    def run_all_tests(self) -> Dict:
        """Run all validation tests."""
        print("\n" + "="*70)
        print("WAKE ROBIN DATA PIPELINE - VALIDATION SUITE")
        print("="*70 + "\n")
        
        tests = [
            ("Module Imports", self.test_imports),
            ("Universe Configuration", self.test_universe_config),
            ("Directory Structure", self.test_directory_structure),
            ("Cache System", self.test_cache_system),
            ("Collector Interfaces", self.test_collector_interfaces),
            ("Orchestrator Script", self.test_orchestrator),
            ("Output Permissions", self.test_output_permissions),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            print(f"Testing {test_name}...", end=" ")
            
            try:
                success, message = test_func()
                
                if success:
                    print(f"✓ PASS")
                    print(f"  └─ {message}")
                    passed += 1
                    self.results['tests'][test_name] = {
                        'status': 'PASS',
                        'message': message
                    }
                else:
                    print(f"✗ FAIL")
                    print(f"  └─ {message}")
                    failed += 1
                    self.results['tests'][test_name] = {
                        'status': 'FAIL',
                        'message': message
                    }
            except Exception as e:
                print(f"✗ ERROR")
                print(f"  └─ Unexpected error: {e}")
                failed += 1
                self.results['tests'][test_name] = {
                    'status': 'ERROR',
                    'message': str(e)
                }
            
            print()
        
        # Overall assessment
        print("="*70)
        print(f"RESULTS: {passed} passed, {failed} failed")
        print("="*70 + "\n")
        
        if failed == 0:
            self.results['overall_status'] = "PASS"
            self.results['deployment_ready'] = True
            print("✅ PIPELINE VALIDATED - READY FOR DEPLOYMENT")
            print("\nNext steps:")
            print("  1. Run: python collect_universe_data.py")
            print("  2. Review quality report in outputs/")
            print("  3. Validate data coverage meets requirements\n")
        else:
            self.results['overall_status'] = "FAIL"
            self.results['deployment_ready'] = False
            print("❌ VALIDATION FAILED - FIX ERRORS BEFORE DEPLOYMENT")
            print("\nFailed tests:")
            for test_name, result in self.results['tests'].items():
                if result['status'] in ['FAIL', 'ERROR']:
                    print(f"  • {test_name}: {result['message']}")
            print()
        
        return self.results
    
    def save_report(self):
        """Save validation report to outputs."""
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"validation_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"📄 Validation report saved: {report_file.name}\n")

def main():
    """Main validation execution."""
    validator = PipelineValidator()
    results = validator.run_all_tests()
    validator.save_report()
    
    # Exit with appropriate code
    sys.exit(0 if results['deployment_ready'] else 1)

if __name__ == "__main__":
    main()
