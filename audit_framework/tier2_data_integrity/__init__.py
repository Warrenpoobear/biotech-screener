"""
Tier 2: Data Integrity & Provenance Architecture.

Validates data quality, provenance tracking, and failure mode handling
for institutional-grade compliance.

Queries Implemented:
    2.1 - Tier-0 Provenance Lock Validation
    2.2 - Data Quality & Coverage Analysis
    2.3 - Failure Mode Catalog & Edge Case Handling
"""

from audit_framework.tier2_data_integrity.provenance import (
    ProvenanceValidator,
    validate_provenance,
)

from audit_framework.tier2_data_integrity.coverage import (
    CoverageValidator,
    validate_data_coverage,
)

from audit_framework.tier2_data_integrity.failure_modes import (
    FailureModeValidator,
    validate_failure_modes,
)

from audit_framework.tier2_data_integrity.runner import (
    run_tier2_audit,
    Tier2Result,
)

__all__ = [
    "ProvenanceValidator",
    "validate_provenance",
    "CoverageValidator",
    "validate_data_coverage",
    "FailureModeValidator",
    "validate_failure_modes",
    "run_tier2_audit",
    "Tier2Result",
]
