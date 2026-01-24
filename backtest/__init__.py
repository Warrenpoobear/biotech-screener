"""
Backtest Framework for Biotech Screener

Provides metrics calculation, IC measurement, returns providers,
and validation utilities for evaluating screener performance.
"""

from backtest.metrics import (
    compute_spearman_ic,
    compute_bucket_returns,
    compute_quintile_returns,
    compute_hit_rate,
    compute_period_metrics,
    aggregate_metrics,
    run_metrics_suite,
    METRICS_VERSION,
)

from backtest.ic_measurement import (
    # Main system
    ICMeasurementSystem,
    WeeklyICReportGenerator,
    ICTimeSeriesDatabase,
    # Engines
    ForwardReturnEngine,
    ICCalculationEngine,
    BootstrapEngine,
    ICStabilityAnalyzer,
    OutOfSampleValidator,
    # Core functions
    calculate_ic,
    analyze_ic_trend,
    # Enums
    ICQuality,
    MarketCapBucket,
    SectorCategory,
    RegimeType,
    # Data classes
    ForwardReturn,
    ICResult,
    BootstrapCI,
    RollingICResult,
    # Constants
    IC_EXCELLENT,
    IC_GOOD,
    IC_WEAK,
    HORIZON_TRADING_DAYS,
)

from backtest.returns_provider import (
    BaseReturnsProvider,
    CSVReturnsProvider,
    NullReturnsProvider,
    FixedReturnsProvider,
    ShuffledReturnsProvider,
    LaggedReturnsProvider,
    create_csv_provider,
    create_shuffled_provider,
    create_lagged_provider,
)

__all__ = [
    # Metrics
    "compute_spearman_ic",
    "compute_bucket_returns",
    "compute_quintile_returns",
    "compute_hit_rate",
    "compute_period_metrics",
    "aggregate_metrics",
    "run_metrics_suite",
    "METRICS_VERSION",
    # IC Measurement System
    "ICMeasurementSystem",
    "WeeklyICReportGenerator",
    "ICTimeSeriesDatabase",
    "ForwardReturnEngine",
    "ICCalculationEngine",
    "BootstrapEngine",
    "ICStabilityAnalyzer",
    "OutOfSampleValidator",
    "calculate_ic",
    "analyze_ic_trend",
    "ICQuality",
    "MarketCapBucket",
    "SectorCategory",
    "RegimeType",
    "ForwardReturn",
    "ICResult",
    "BootstrapCI",
    "RollingICResult",
    "IC_EXCELLENT",
    "IC_GOOD",
    "IC_WEAK",
    "HORIZON_TRADING_DAYS",
    # Returns providers
    "BaseReturnsProvider",
    "CSVReturnsProvider",
    "NullReturnsProvider",
    "FixedReturnsProvider",
    "ShuffledReturnsProvider",
    "LaggedReturnsProvider",
    "create_csv_provider",
    "create_shuffled_provider",
    "create_lagged_provider",
]
