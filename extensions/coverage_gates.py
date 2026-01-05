# extensions/coverage_gates.py
"""
Data Coverage Gates - Quality Scoring for Screening Results

Purpose:
- Make missingness transparent and auditable
- Compute deterministic coverage_confidence (0.0..1.0)
- Flag sparse data for IC review
- (Optional) provide downweight factor to apply to scores

Determinism guarantees:
- Decimal-only arithmetic for scoring math
- Stable quantization at serialization boundary only
- Canonical audit hashing
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Tuple

# ---- Configuration ----

REQUIRED_FIELDS = {
    'financial': ['cash_usd', 'debt_usd', 'ttm_revenue_usd', 'market_cap_usd'],
    'clinical': ['lead_stage', 'lead_indication', 'next_catalyst_date', 'catalyst_type'],
    'institutional': ['manager_count', 'total_position_value'],
}

CATEGORY_WEIGHTS = {
    'financial': Decimal('0.40'),
    'clinical': Decimal('0.40'),
    'institutional': Decimal('0.20'),
}

# Output quantization (stable JSON)
Q = Decimal('0.0001')  # 4dp keeps stability without band-edge jitter


def _is_present(v: Any) -> bool:
    return v is not None and v != '' and v != []


def _q(x: Decimal) -> str:
    return str(x.quantize(Q, rounding=ROUND_HALF_UP))


def calculate_field_coverage(record: Dict[str, Any], fields: List[str]) -> Tuple[int, int, List[str]]:
    """
    Returns:
      present_count, total_count, missing_fields
    """
    if not fields:
        return 0, 0, []
    missing = [f for f in fields if not _is_present(record.get(f))]
    present = len(fields) - len(missing)
    return present, len(fields), missing


def calculate_category_confidence(coverage: Decimal, category: str) -> Decimal:
    """
    Convert coverage to confidence.
    - financial: piecewise buckets
    - clinical: quadratic penalty (coverage^2)
    - institutional: linear (coverage)
    """
    if category == 'financial':
        if coverage >= Decimal('0.75'):
            return Decimal('1.0')
        if coverage >= Decimal('0.50'):
            return Decimal('0.7')
        if coverage >= Decimal('0.25'):
            return Decimal('0.4')
        return Decimal('0.1')

    if category == 'clinical':
        return coverage * coverage

    return coverage


def calculate_coverage_confidence(
    security_record: Dict[str, Any],
    include_market_cap: bool = False,
) -> Dict[str, Any]:
    """
    security_record is the per-ticker object that contains module outputs.

    Returns:
      - coverage_confidence (str)
      - data_quality_flag (OK / SPARSE_DATA / MISSING_CRITICAL)
      - coverage_breakdown (per category)
      - critical_missing (list)
      - audit_hash (str)
    """
    # Build category records from module outputs (adjust these keys at integration time if needed)
    mod2 = security_record.get('module_2_financial', {}) or {}
    mod3 = security_record.get('module_3_catalyst', {}) or {}
    mod4 = (
        security_record.get('module_4_clinical_dev', {})
        or security_record.get('module_4_clinical', {})
        or {}
    )
    inst = security_record.get('institutional_signals', {}) or {}

    # Choose financial fields
    fin_fields = list(REQUIRED_FIELDS['financial'])
    if not include_market_cap and 'market_cap_usd' in fin_fields:
        fin_fields.remove('market_cap_usd')

    # Coverage: financial
    fin_present, fin_total, fin_missing = calculate_field_coverage(mod2, fin_fields)
    fin_cov = (Decimal(fin_present) / Decimal(fin_total)) if fin_total else Decimal('1.0')
    fin_conf = calculate_category_confidence(fin_cov, 'financial')

    # Coverage: clinical (merge mod3 + mod4)
    clin_record = {**mod3, **mod4}
    cli_present, cli_total, cli_missing = calculate_field_coverage(clin_record, REQUIRED_FIELDS['clinical'])
    cli_cov = (Decimal(cli_present) / Decimal(cli_total)) if cli_total else Decimal('1.0')
    cli_conf = calculate_category_confidence(cli_cov, 'clinical')

    # Coverage: institutional
    if inst:
        inst_present, inst_total, inst_missing = calculate_field_coverage(inst, REQUIRED_FIELDS['institutional'])
        inst_cov = (Decimal(inst_present) / Decimal(inst_total)) if inst_total else Decimal('1.0')
        inst_conf = calculate_category_confidence(inst_cov, 'institutional')
    else:
        inst_present, inst_total, inst_missing = 0, len(REQUIRED_FIELDS['institutional']), list(REQUIRED_FIELDS['institutional'])
        inst_cov = Decimal('0.0')
        inst_conf = Decimal('0.0')

    # Weighted composite confidence
    composite = (
        fin_conf * CATEGORY_WEIGHTS['financial']
        + cli_conf * CATEGORY_WEIGHTS['clinical']
        + inst_conf * CATEGORY_WEIGHTS['institutional']
    )

    # Hard gate: require minimum clinical coverage
    if cli_cov < Decimal('0.50'):
        composite = min(composite, Decimal('0.30'))

    # Critical missing rules (field-level, not category-level)
    critical_missing: List[str] = []
    for f in ['cash_usd', 'debt_usd', 'ttm_revenue_usd']:
        if not _is_present(mod2.get(f)):
            critical_missing.append(f)
    for f in ['lead_stage', 'next_catalyst_date']:
        if not _is_present(clin_record.get(f)):
            critical_missing.append(f)
    if include_market_cap and not _is_present(mod2.get('market_cap_usd')):
        critical_missing.append('market_cap_usd')

    if critical_missing:
        flag = 'MISSING_CRITICAL'
    elif composite < Decimal('0.60'):
        flag = 'SPARSE_DATA'
    else:
        flag = 'OK'

    breakdown = {
        'financial': {
            'coverage': _q(fin_cov),
            'confidence': _q(fin_conf),
            'required_fields': fin_total,
            'present_fields': fin_present,
            'missing': fin_missing,
        },
        'clinical': {
            'coverage': _q(cli_cov),
            'confidence': _q(cli_conf),
            'required_fields': cli_total,
            'present_fields': cli_present,
            'missing': cli_missing,
        },
        'institutional': {
            'coverage': _q(inst_cov),
            'confidence': _q(inst_conf),
            'required_fields': inst_total,
            'present_fields': inst_present,
            'missing': inst_missing,
        },
    }

    audit_hash = _compute_audit_hash(
        breakdown=breakdown,
        composite=_q(composite),
        critical_missing=critical_missing,
        include_market_cap=include_market_cap,
    )

    return {
        'coverage_confidence': _q(composite),
        'data_quality_flag': flag,
        'coverage_breakdown': breakdown,
        'critical_missing': critical_missing,
        'audit_hash': audit_hash,
    }


def apply_coverage_adjustment(
    composite_score: Decimal,
    coverage_confidence: Decimal,
    min_confidence_threshold: Decimal = Decimal('0.60'),
) -> Dict[str, Any]:
    """
    Optional downweighting. Keep separate so you can toggle it.
    """
    if coverage_confidence < min_confidence_threshold:
        adjusted = (composite_score * coverage_confidence).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        return {
            'adjustment_applied': True,
            'original_score': str(composite_score),
            'adjusted_score': str(adjusted),
            'adjustment_factor': str(coverage_confidence),
        }

    return {
        'adjustment_applied': False,
        'original_score': str(composite_score),
        'adjusted_score': str(composite_score),
        'adjustment_factor': '1.0',
    }


def _compute_audit_hash(
    breakdown: Dict[str, Any],
    composite: str,
    critical_missing: List[str],
    include_market_cap: bool,
) -> str:
    payload = {
        'breakdown': breakdown,
        'composite': composite,
        'critical_missing': sorted(critical_missing),
        'include_market_cap': include_market_cap,
        'weights': {k: str(v) for k, v in CATEGORY_WEIGHTS.items()},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]


if __name__ == '__main__':
    print('==' * 35)
    print('COVERAGE GATES - SELF TEST')
    print('==' * 35)

    # Example security record shaped like a merged per-ticker object
    sec = {
        'module_2_financial': {
            'cash_usd': '1000000000',
            'debt_usd': '100000000',
            'ttm_revenue_usd': '500000000',
            'market_cap_usd': '5000000000',
        },
        'module_3_catalyst': {
            'next_catalyst_date': '2026-03-15',
            'catalyst_type': 'phase_3_topline',
        },
        'module_4_clinical_dev': {
            'lead_stage': 'phase_3',
            'lead_indication': 'oncology',
        },
        'institutional_signals': {
            'manager_count': 12,
            'total_position_value': '250000000',
        },
    }

    r = calculate_coverage_confidence(sec, include_market_cap=True)
    print(json.dumps(r, indent=2, sort_keys=True))
    print('✅ OK')