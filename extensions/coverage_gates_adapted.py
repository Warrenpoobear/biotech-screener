# extensions/coverage_gates_adapted.py
"""
Adapted Coverage Gates for Pipeline That Outputs Scores (Not Detailed Fields)

Works with pipeline structure:
- financial_normalized/financial_raw (scores exist)
- clinical_dev_normalized/clinical_dev_raw (scores exist)
- catalyst_normalized/catalyst_raw (scores exist)
"""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict

Q = Decimal('0.0001')

def _q(x: Decimal) -> str:
    return str(x.quantize(Q, rounding=ROUND_HALF_UP))

def calculate_coverage_confidence_adapted(
    security_record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate coverage based on what scores are present in the pipeline output.
    
    Categories:
    - Financial: Checks if financial_normalized exists
    - Clinical: Checks if clinical_dev_normalized exists
    - Catalyst: Checks if catalyst_normalized exists
    """
    
    # Check which scores are present
    has_financial = security_record.get('financial_normalized') is not None
    has_clinical = security_record.get('clinical_dev_normalized') is not None
    has_catalyst = security_record.get('catalyst_normalized') is not None
    
    # Calculate category confidence (binary: 1.0 if present, 0.0 if missing)
    fin_conf = Decimal('1.0') if has_financial else Decimal('0.0')
    cli_conf = Decimal('1.0') if has_clinical else Decimal('0.0')
    cat_conf = Decimal('1.0') if has_catalyst else Decimal('0.0')
    
    # Weighted composite (Financial 40%, Clinical 40%, Catalyst 20%)
    composite = (
        fin_conf * Decimal('0.40') +
        cli_conf * Decimal('0.40') +
        cat_conf * Decimal('0.20')
    )
    
    # Determine flag
    if composite >= Decimal('0.80'):
        flag = 'OK'
    elif composite >= Decimal('0.40'):
        flag = 'SPARSE_DATA'
    else:
        flag = 'MISSING_CRITICAL'
    
    # Build breakdown
    breakdown = {
        'financial': {
            'present': has_financial,
            'confidence': _q(fin_conf),
            'score': str(security_record.get('financial_normalized', 'N/A'))
        },
        'clinical': {
            'present': has_clinical,
            'confidence': _q(cli_conf),
            'score': str(security_record.get('clinical_dev_normalized', 'N/A'))
        },
        'catalyst': {
            'present': has_catalyst,
            'confidence': _q(cat_conf),
            'score': str(security_record.get('catalyst_normalized', 'N/A'))
        }
    }
    
    # Create audit hash
    audit_data = {
        'breakdown': breakdown,
        'composite': _q(composite),
        'flag': flag
    }
    canonical = json.dumps(audit_data, sort_keys=True, separators=(',', ':'))
    audit_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]
    
    return {
        'coverage_confidence': _q(composite),
        'data_quality_flag': flag,
        'coverage_breakdown': breakdown,
        'audit_hash': audit_hash,
    }


if __name__ == '__main__':
    # Test with actual structure
    test_sec = {
        'ticker': 'TEST',
        'financial_normalized': '50.00',
        'clinical_dev_normalized': '64.00',
        'catalyst_normalized': '35.00',
    }
    
    result = calculate_coverage_confidence_adapted(test_sec)
    print(json.dumps(result, indent=2))