#!/usr/bin/env python3
"""
Portfolio Validation - Pre-Deployment Checks
Validates portfolio meets risk constraints before deployment

Usage:
    python scripts/validate_portfolio.py --portfolio outputs/portfolio_20260115.json
"""

import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

class PortfolioValidator:
    """Validates portfolio against risk and quality constraints"""
    
    def __init__(self, portfolio_path: Path):
        self.portfolio_path = portfolio_path
        self.portfolio = self._load_portfolio()
        self.errors = []
        self.warnings = []
        self.passed = []
        
    def _load_portfolio(self) -> Dict:
        """Load portfolio JSON"""
        with open(self.portfolio_path) as f:
            return json.load(f)
    
    def validate_all(self) -> bool:
        """Run all validation checks"""
        checks = [
            self.check_data_freshness,
            self.check_position_count,
            self.check_concentration_limits,
            self.check_weight_sum,
            self.check_missing_prices,
            self.check_beta_target,
            self.check_momentum_sanity,
            self.check_rank_uniqueness,
            self.check_no_nan_scores
        ]
        
        for check in checks:
            check()
        
        return len(self.errors) == 0
    
    def check_data_freshness(self):
        """Verify portfolio was generated recently"""
        generated_str = self.portfolio.get('generated_at', '')
        
        if not generated_str:
            self.errors.append("Missing generated_at timestamp")
            return
        
        try:
            generated_dt = datetime.fromisoformat(generated_str.replace('Z', '+00:00'))
            age_hours = (datetime.now() - generated_dt).total_seconds() / 3600
            
            if age_hours > 24:
                self.warnings.append(f"Portfolio is {age_hours:.1f} hours old (>24 hours)")
            else:
                self.passed.append(f"✅ Data freshness OK ({age_hours:.1f} hours old)")
        
        except Exception as e:
            self.errors.append(f"Invalid generated_at timestamp: {e}")
    
    def check_position_count(self):
        """Verify position count is reasonable"""
        positions = self.portfolio.get('positions', [])
        num_positions = len(positions)
        
        if num_positions < 20:
            self.warnings.append(f"Only {num_positions} positions (target: 60)")
        elif num_positions > 80:
            self.warnings.append(f"Too many positions: {num_positions} (target: 60)")
        else:
            self.passed.append(f"✅ Position count OK ({num_positions} positions)")
    
    def check_concentration_limits(self):
        """Verify no single position exceeds max concentration"""
        positions = self.portfolio.get('positions', [])
        max_position_pct = self.portfolio.get('parameters', {}).get('max_position_pct', 3.0)
        
        violations = []
        for pos in positions:
            weight_pct = pos.get('weight_pct', 0)
            if weight_pct > max_position_pct:
                violations.append(f"{pos['ticker']}: {weight_pct:.2f}%")
        
        if violations:
            self.errors.append(f"Concentration limit violations (>{max_position_pct}%): {', '.join(violations)}")
        else:
            max_weight = max(p.get('weight_pct', 0) for p in positions) if positions else 0
            self.passed.append(f"✅ Concentration limits OK (max: {max_weight:.2f}%)")
    
    def check_weight_sum(self):
        """Verify weights sum to 100%"""
        positions = self.portfolio.get('positions', [])
        total_weight = sum(p.get('weight_pct', 0) for p in positions)
        
        if abs(total_weight - 100.0) > 0.01:  # Allow 0.01% rounding error
            self.errors.append(f"Weights sum to {total_weight:.2f}% (expected: 100%)")
        else:
            self.passed.append(f"✅ Weight sum OK ({total_weight:.2f}%)")
    
    def check_missing_prices(self):
        """Check for positions without current prices"""
        positions = self.portfolio.get('positions', [])
        missing = [p['ticker'] for p in positions if not p.get('current_price')]
        
        if missing:
            self.errors.append(f"Missing prices for {len(missing)} tickers: {', '.join(missing[:10])}")
        else:
            self.passed.append(f"✅ All positions have current prices")
    
    def check_beta_target(self):
        """Verify portfolio beta is within target range"""
        metrics = self.portfolio.get('metrics', {})
        beta = metrics.get('portfolio_beta_estimate', 0)
        
        # Target: 0.60-0.85 (defensive positioning)
        if beta < 0.50:
            self.warnings.append(f"Beta unusually low: {beta:.2f} (target: 0.60-0.85)")
        elif beta > 0.90:
            self.warnings.append(f"Beta too high: {beta:.2f} (target: 0.60-0.85)")
        else:
            self.passed.append(f"✅ Beta within target: {beta:.2f}")
    
    def check_momentum_sanity(self):
        """Verify momentum scores are reasonable"""
        positions = self.portfolio.get('positions', [])
        
        if not positions:
            self.errors.append("No positions found")
            return
        
        momentum_scores = [p.get('momentum_score', 50) for p in positions]
        avg_momentum = sum(momentum_scores) / len(momentum_scores)
        
        # In RISK_ON regime, expect avg momentum > 50
        if avg_momentum < 40:
            self.warnings.append(f"Low average momentum: {avg_momentum:.1f} (expected: >50 in RISK_ON)")
        else:
            self.passed.append(f"✅ Momentum sanity OK (avg: {avg_momentum:.1f})")
        
        # Check for extreme values
        extremes = [p['ticker'] for p in positions if p.get('momentum_score', 50) < 10 or p.get('momentum_score', 50) > 95]
        if extremes:
            self.warnings.append(f"Extreme momentum values in: {', '.join(extremes[:5])}")
    
    def check_rank_uniqueness(self):
        """Verify all positions have unique ranks"""
        positions = self.portfolio.get('positions', [])
        ranks = [p.get('rank') for p in positions]
        
        if len(ranks) != len(set(ranks)):
            self.errors.append("Duplicate ranks found")
        else:
            self.passed.append(f"✅ Rank uniqueness OK")
    
    def check_no_nan_scores(self):
        """Verify no NaN or null scores"""
        positions = self.portfolio.get('positions', [])
        
        invalid = []
        for pos in positions:
            ticker = pos['ticker']
            if pos.get('final_score') is None or pos.get('final_score') != pos.get('final_score'):  # Check for NaN
                invalid.append(ticker)
        
        if invalid:
            self.errors.append(f"Invalid final scores for: {', '.join(invalid)}")
        else:
            self.passed.append(f"✅ No NaN scores")
    
    def print_report(self):
        """Print validation report"""
        print("="*70)
        print("PORTFOLIO VALIDATION REPORT")
        print("="*70)
        print(f"Portfolio: {self.portfolio_path.name}")
        print(f"Date: {self.portfolio.get('date', 'N/A')}")
        print(f"Positions: {self.portfolio.get('num_positions', 0)}")
        print()
        
        # Passed checks
        if self.passed:
            print("PASSED CHECKS:")
            for msg in self.passed:
                print(f"  {msg}")
            print()
        
        # Warnings
        if self.warnings:
            print("⚠️  WARNINGS:")
            for msg in self.warnings:
                print(f"  ⚠️  {msg}")
            print()
        
        # Errors
        if self.errors:
            print("❌ ERRORS (BLOCKING):")
            for msg in self.errors:
                print(f"  ❌ {msg}")
            print()
        
        # Summary
        print("="*70)
        if self.errors:
            print("❌ VALIDATION FAILED - DO NOT DEPLOY")
            print(f"   {len(self.errors)} blocking error(s)")
        elif self.warnings:
            print("⚠️  VALIDATION PASSED WITH WARNINGS")
            print(f"   {len(self.warnings)} warning(s) - review before deploying")
        else:
            print("✅ VALIDATION PASSED - READY TO DEPLOY")
        print("="*70)
        print()
        
        return len(self.errors) == 0

def main():
    parser = argparse.ArgumentParser(description='Validate portfolio before deployment')
    parser.add_argument('--portfolio', required=True, help='Path to portfolio JSON file')
    
    args = parser.parse_args()
    
    portfolio_path = Path(args.portfolio)
    
    if not portfolio_path.exists():
        print(f"❌ Portfolio file not found: {portfolio_path}")
        return 1
    
    # Run validation
    validator = PortfolioValidator(portfolio_path)
    validator.validate_all()
    is_valid = validator.print_report()
    
    return 0 if is_valid else 1

if __name__ == '__main__':
    exit(main())
