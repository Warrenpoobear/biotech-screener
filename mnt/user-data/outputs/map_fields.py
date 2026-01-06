#!/usr/bin/env python3
"""
Map data from existing fields to expected field names.
This fixes the field name mismatch without re-collecting data.
"""

import json
import sys
from pathlib import Path

def map_fields_to_expected_names(
    input_path="production_data/universe.json",
    output_path="production_data/universe_mapped.json",
    backup=True
):
    """
    Map existing fields to expected names for screening.
    
    Existing → Expected:
      financials → financial_data
      clinical → clinical_data
      (create catalyst_data from available info)
    """
    
    print("="*80)
    print("FIELD MAPPING SCRIPT")
    print("="*80)
    print()
    
    # Load universe
    with open(input_path) as f:
        universe = json.load(f)
    
    print(f"Loaded {len(universe)} securities from {input_path}")
    print()
    
    # Backup if requested
    if backup:
        backup_path = Path(input_path).with_suffix('.bak')
        with open(backup_path, 'w') as f:
            json.dump(universe, f, indent=2)
        print(f"✅ Backup saved to: {backup_path}")
        print()
    
    # Track what we're doing
    stats = {
        'mapped_financial': 0,
        'mapped_clinical': 0,
        'created_catalyst': 0,
        'no_data': 0,
    }
    
    print("Mapping fields...")
    print()
    
    for i, security in enumerate(universe, 1):
        ticker = security.get('ticker', f'UNKNOWN_{i}')
        
        # Map financials → financial_data
        if 'financials' in security and security['financials']:
            if not security.get('financial_data'):
                security['financial_data'] = security['financials']
                stats['mapped_financial'] += 1
        
        # Map clinical → clinical_data
        if 'clinical' in security and security['clinical']:
            if not security.get('clinical_data'):
                security['clinical_data'] = security['clinical']
                stats['mapped_clinical'] += 1
        
        # Create catalyst_data from available info
        # (This is a placeholder - you may want to customize)
        if not security.get('catalyst_data'):
            # Try to extract catalyst info from clinical trials or financials
            catalyst_data = {}
            
            # If clinical data exists, use trial info as catalyst
            if security.get('clinical') or security.get('clinical_data'):
                clinical = security.get('clinical_data') or security.get('clinical')
                if isinstance(clinical, dict):
                    trials = clinical.get('trials', [])
                    if trials:
                        # Use first active trial as catalyst
                        for trial in trials:
                            if isinstance(trial, dict):
                                status = trial.get('status', '').upper()
                                if 'ACTIVE' in status or 'RECRUITING' in status:
                                    catalyst_data = {
                                        'has_catalyst': True,
                                        'catalyst_type': 'TRIAL_READOUT',
                                        'trial_nct': trial.get('nct_id'),
                                        'trial_phase': trial.get('phase'),
                                        'note': 'Generated from clinical trial data',
                                    }
                                    break
            
            # If we created anything, save it
            if catalyst_data:
                security['catalyst_data'] = catalyst_data
                stats['created_catalyst'] += 1
            else:
                stats['no_data'] += 1
    
    # Save mapped universe
    with open(output_path, 'w') as f:
        json.dump(universe, f, indent=2)
    
    print()
    print("="*80)
    print("MAPPING SUMMARY")
    print("="*80)
    print()
    print(f"✅ Mapped financials → financial_data: {stats['mapped_financial']} stocks")
    print(f"✅ Mapped clinical → clinical_data: {stats['mapped_clinical']} stocks")
    print(f"⚠️  Created catalyst_data: {stats['created_catalyst']} stocks")
    print(f"❌ No data to map: {stats['no_data']} stocks")
    print()
    print(f"✅ Saved to: {output_path}")
    print()
    
    # Calculate expected improvement
    before_coverage = 0  # Was 0%
    after_financial = stats['mapped_financial'] / len(universe) * 100
    after_clinical = stats['mapped_clinical'] / len(universe) * 100
    after_catalyst = stats['created_catalyst'] / len(universe) * 100
    
    print("="*80)
    print("EXPECTED IMPROVEMENT")
    print("="*80)
    print()
    print(f"Financial data: {before_coverage:.0f}% → {after_financial:.0f}%")
    print(f"Clinical data:  {before_coverage:.0f}% → {after_clinical:.0f}%")
    print(f"Catalyst data:  {before_coverage:.0f}% → {after_catalyst:.0f}%")
    print()
    
    if after_financial > 80 and after_clinical > 80:
        print("🎉 EXCELLENT! Should have >80% coverage after mapping!")
    elif after_financial > 50 or after_clinical > 50:
        print("✅ GOOD! Should have >50% coverage after mapping!")
    else:
        print("⚠️  WARNING: Low coverage even after mapping.")
        print("   May need to run actual data collection.")
    
    print()
    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print()
    print("1. Review the mapped file:")
    print(f"   {output_path}")
    print()
    print("2. If it looks good, replace the original:")
    print(f"   Copy-Item {output_path} {input_path} -Force")
    print()
    print("3. Re-run the diagnostic:")
    print("   python diagnose_data_collection.py")
    print()
    print("4. Re-run screening:")
    print("   python run_screen.py --as-of-date 2026-01-06 --data-dir production_data --output screening_MAPPED.json")
    print()
    
    return universe, stats

if __name__ == "__main__":
    map_fields_to_expected_names()
