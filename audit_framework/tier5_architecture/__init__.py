"""
Tier 5: Architecture & Code Quality.

Validates codebase architecture, maintainability, and security posture.

Queries Implemented:
    5.1 - Architecture Review for Maintainability
    5.2 - Security & Access Control
"""

from audit_framework.tier5_architecture.maintainability import (
    MaintainabilityValidator,
    validate_maintainability,
)

from audit_framework.tier5_architecture.security import (
    SecurityValidator,
    validate_security,
)

from audit_framework.tier5_architecture.runner import (
    run_tier5_audit,
    Tier5Result,
)

__all__ = [
    "MaintainabilityValidator",
    "validate_maintainability",
    "SecurityValidator",
    "validate_security",
    "run_tier5_audit",
    "Tier5Result",
]
