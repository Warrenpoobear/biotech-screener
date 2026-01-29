"""
common - Shared utilities for biotech screener modules.

Provides:
- date_utils: Date normalization and validation
- data_quality: Data quality gates and validation
- pit_enforcement: Point-in-time discipline
- provenance: Provenance tracking
- types: Common type definitions
- integration_contracts: Module boundary types and schema validation
- hash_utils: Deterministic hashing utilities
- input_validation: Pipeline input validation
- score_utils: Score clamping and normalization
- null_safety: Defensive null handling
- robustness: Data staleness, consistency checks, retry logic, memory guards
- score_to_er: Score to Expected Return conversion (institutional methodology)
- clustering: Correlated security clustering (return correlation based)
"""

from common.date_utils import normalize_date, to_date_string, to_date_object, validate_as_of_date

# Score to Expected Return conversion (institutional methodology)
from common.score_to_er import (
    compute_expected_returns,
    attach_rank_and_z,
    attach_expected_return,
    validate_er_output,
    DEFAULT_LAMBDA_ANNUAL,
    ER_MODEL_ID,
    ER_MODEL_VERSION,
)

# Correlated security clustering
from common.clustering import (
    build_corr_clusters,
    attach_cluster_ids,
    attach_indication_clusters,
    compute_pairwise_correlations,
    DEFAULT_CORR_THRESHOLD,
    DEFAULT_CORR_WINDOW,
    CLUSTER_MODEL_ID,
    CLUSTER_MODEL_VERSION,
)

# Hash utilities
from common.hash_utils import (
    compute_hash,
    compute_snapshot_id,
    compute_trial_facts_hash,
    stable_json_dumps,
)
from common.data_quality import (
    DataQualityGates,
    DataQualityConfig,
    ValidationResult,
    QualityGateResult,
    validate_financial_staleness,
    validate_liquidity,
)
from common.types import Severity

# Integration contracts - module boundary types and validation
from common.integration_contracts import (
    # Type aliases
    TickerSet,
    TickerList,
    TickerCollection,
    DateLike,
    # Normalization helpers
    normalize_date_input,
    normalize_date_string,
    normalize_ticker_set,
    # Score field names
    ScoreFieldNames,
    # Schema validation
    SchemaValidationError,
    validate_module_1_output,
    validate_module_2_output,
    validate_module_3_output,
    validate_module_4_output,
    validate_module_5_output,
    validate_pipeline_handoff,
    # Score extraction helpers
    extract_financial_score,
    extract_catalyst_score,
    extract_clinical_score,
    extract_market_cap_mm,
    # Version checking
    check_schema_version,
    SUPPORTED_SCHEMA_VERSIONS,
    # Validation mode
    get_validation_mode,
    set_validation_mode,
    is_strict_mode,
    is_validation_enabled,
    # Schema migration
    migrate_module_output,
    ensure_dual_field_names,
    # Re-exported Module 3 types
    EventType,
    EventSeverity,
    ConfidenceLevel,
    CatalystWindowBucket,
    CatalystEventV2,
    TickerCatalystSummaryV2,
)

# Robustness utilities
from common.robustness import (
    # Data staleness
    DataFreshnessConfig,
    DataFreshnessResult,
    validate_data_freshness,
    validate_record_freshness,
    # Cross-module consistency
    ConsistencyReport,
    validate_ticker_coverage,
    validate_module_handoff,
    # Retry logic
    RetryConfig,
    retry_with_backoff,
    RetryExhaustedError,
    # Memory guards
    MemoryGuardConfig,
    chunk_universe,
    estimate_memory_usage,
    # Structured logging
    CorrelationContext,
    get_correlation_id,
    set_correlation_id,
    with_correlation_id,
    CorrelatedLogger,
    # Graceful degradation
    DegradationReport,
    GracefulDegradationConfig,
    compute_with_degradation,
)

__all__ = [
    # Date utilities
    "normalize_date",
    "to_date_string",
    "to_date_object",
    "validate_as_of_date",
    # Hash utilities
    "compute_hash",
    "compute_snapshot_id",
    "compute_trial_facts_hash",
    "stable_json_dumps",
    # Data quality
    "DataQualityGates",
    "DataQualityConfig",
    "ValidationResult",
    "QualityGateResult",
    "validate_financial_staleness",
    "validate_liquidity",
    # Types
    "Severity",
    # Integration contracts - type aliases
    "TickerSet",
    "TickerList",
    "TickerCollection",
    "DateLike",
    # Integration contracts - normalization
    "normalize_date_input",
    "normalize_date_string",
    "normalize_ticker_set",
    # Integration contracts - score field names
    "ScoreFieldNames",
    # Integration contracts - schema validation
    "SchemaValidationError",
    "validate_module_1_output",
    "validate_module_2_output",
    "validate_module_3_output",
    "validate_module_4_output",
    "validate_module_5_output",
    "validate_pipeline_handoff",
    # Integration contracts - score extraction
    "extract_financial_score",
    "extract_catalyst_score",
    "extract_clinical_score",
    "extract_market_cap_mm",
    # Integration contracts - version checking
    "check_schema_version",
    "SUPPORTED_SCHEMA_VERSIONS",
    # Integration contracts - validation mode
    "get_validation_mode",
    "set_validation_mode",
    "is_strict_mode",
    "is_validation_enabled",
    # Integration contracts - schema migration
    "migrate_module_output",
    "ensure_dual_field_names",
    # Integration contracts - Module 3 types
    "EventType",
    "EventSeverity",
    "ConfidenceLevel",
    "CatalystWindowBucket",
    "CatalystEventV2",
    "TickerCatalystSummaryV2",
    # Robustness - data staleness
    "DataFreshnessConfig",
    "DataFreshnessResult",
    "validate_data_freshness",
    "validate_record_freshness",
    # Robustness - cross-module consistency
    "ConsistencyReport",
    "validate_ticker_coverage",
    "validate_module_handoff",
    # Robustness - retry logic
    "RetryConfig",
    "retry_with_backoff",
    "RetryExhaustedError",
    # Robustness - memory guards
    "MemoryGuardConfig",
    "chunk_universe",
    "estimate_memory_usage",
    # Robustness - structured logging
    "CorrelationContext",
    "get_correlation_id",
    "set_correlation_id",
    "with_correlation_id",
    "CorrelatedLogger",
    # Robustness - graceful degradation
    "DegradationReport",
    "GracefulDegradationConfig",
    "compute_with_degradation",
    # Score to Expected Return
    "compute_expected_returns",
    "attach_rank_and_z",
    "attach_expected_return",
    "validate_er_output",
    "DEFAULT_LAMBDA_ANNUAL",
    "ER_MODEL_ID",
    "ER_MODEL_VERSION",
    # Clustering
    "build_corr_clusters",
    "attach_cluster_ids",
    "attach_indication_clusters",
    "compute_pairwise_correlations",
    "DEFAULT_CORR_THRESHOLD",
    "DEFAULT_CORR_WINDOW",
    "CLUSTER_MODEL_ID",
    "CLUSTER_MODEL_VERSION",
]
