"""
Tier 3: Performance, Scalability & Operational Robustness.

Validates pipeline performance, error handling, and dependency security
for institutional production deployment.

Queries Implemented:
    3.1 - Production Pipeline Performance Profiling
    3.2 - Error Handling & Resilience Audit
    3.3 - Dependency & Supply Chain Security
"""

from audit_framework.tier3_performance.profiling import (
    PerformanceValidator,
    validate_performance,
)

from audit_framework.tier3_performance.resilience import (
    ResilienceValidator,
    validate_resilience,
)

from audit_framework.tier3_performance.dependencies import (
    DependencyValidator,
    validate_dependencies,
)

from audit_framework.tier3_performance.runner import (
    run_tier3_audit,
    Tier3Result,
)

__all__ = [
    "PerformanceValidator",
    "validate_performance",
    "ResilienceValidator",
    "validate_resilience",
    "DependencyValidator",
    "validate_dependencies",
    "run_tier3_audit",
    "Tier3Result",
]
