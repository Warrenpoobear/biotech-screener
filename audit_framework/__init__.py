"""
Institutional-Grade Technical Audit Framework for Wake Robin Capital Management.

This framework provides comprehensive validation infrastructure for SEC/FINRA compliance,
investment committee review, and institutional-grade quality assurance.

Framework Tiers:
    - Tier 1: Determinism & Reproducibility Validation
    - Tier 2: Data Integrity & Provenance Architecture
    - Tier 3: Performance, Scalability & Operational Robustness
    - Tier 4: Testing, Validation & Regression Prevention
    - Tier 5: Architecture & Code Quality
    - Tier 6: Deployment & Operational Readiness

Version: 1.0.0
Author: Wake Robin Capital Management
"""

from audit_framework.types import (
    AuditResult,
    AuditSeverity,
    AuditTier,
    ComplianceGrade,
    ValidationFinding,
    AuditReport,
    PassCriteria,
)

from audit_framework.orchestrator import (
    AuditOrchestrator,
    run_full_audit,
    run_tier_audit,
)

__version__ = "1.0.0"
__all__ = [
    "AuditResult",
    "AuditSeverity",
    "AuditTier",
    "ComplianceGrade",
    "ValidationFinding",
    "AuditReport",
    "PassCriteria",
    "AuditOrchestrator",
    "run_full_audit",
    "run_tier_audit",
]
