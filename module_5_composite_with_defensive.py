"""
module_5_composite_with_defensive.py

Production wrapper that adds defensive overlays to Module 5.

v1.2.0 (2026-01-18): Promoted v3 to default production scorer:
  - All v2 features retained (monotonic caps, confidence weighting, etc.)
  - V3 IC enhancements: momentum, valuation, catalyst decay, smart money
  - Sanity override mechanism for pathological rankings
  - Feature flags for experimental features (interaction terms, adaptive weights)
  - Fallback to v2 when low-confidence or experimental signals dominate

v1.1.0 (2026-01-18): Merged v2 features into production.

The v3 scorer is now the default. Set use_v3_scoring=False for v2,
or use_v2_scoring=False AND use_v3_scoring=False for v1 (legacy).
"""

import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Optional, List, Dict, Any

from module_5_composite import compute_module_5_composite
from module_5_composite_v2 import compute_module_5_composite_v2
from module_5_composite_v3 import compute_module_5_composite_v3
from defensive_overlay_adapter import enrich_with_defensive_overlays, validate_defensive_integration

# V3 production configuration
from config.v3_production_integration import (
    FeatureFlags,
    FallbackConfig,
    SanityOverrideConfig,
    check_sanity_override,
    V3_PRODUCTION_DEFAULTS,
)

logger = logging.getLogger(__name__)


def _apply_sanity_overrides(output_v3: dict, output_v2: dict) -> dict:
    """
    Apply sanity overrides to v3 output based on v2 comparison.

    If a ticker has massive rank divergence between v3 and v2, and the
    divergence is driven by low-confidence or experimental signals,
    fall back to v2 score for that ticker.

    Args:
        output_v3: V3 scoring output
        output_v2: V2 scoring output for comparison

    Returns:
        V3 output with sanity overrides applied
    """
    config = SanityOverrideConfig()

    # Build v2 rank lookup
    v2_ranks = {}
    v2_by_ticker = {}
    for sec in output_v2.get("ranked_securities", []):
        ticker = sec.get("ticker")
        if ticker:
            v2_ranks[ticker] = sec.get("composite_rank", 999)
            v2_by_ticker[ticker] = sec

    # Track overrides
    overrides_applied = []
    modified_securities = []

    for sec in output_v3.get("ranked_securities", []):
        ticker = sec.get("ticker")
        v3_rank = sec.get("composite_rank", 999)
        v2_rank = v2_ranks.get(ticker, 999)

        # Check sanity override
        override_result = check_sanity_override(
            ticker=ticker,
            v3_rank=v3_rank,
            v2_rank=v2_rank,
            score_breakdown=sec.get("score_breakdown", {}),
            config=config,
        )

        if override_result.fallback_to_v2 and ticker in v2_by_ticker:
            # Substitute v2 data for this ticker
            v2_sec = v2_by_ticker[ticker].copy()
            v2_sec["sanity_override"] = {
                "original_v3_rank": v3_rank,
                "v2_rank": v2_rank,
                "override_reason": override_result.override_reason,
                "driving_factor": override_result.driving_factor,
                "confidence_level": str(override_result.confidence_level),
            }
            v2_sec["flags"] = v2_sec.get("flags", []) + ["sanity_override_applied"]
            modified_securities.append(v2_sec)

            overrides_applied.append({
                "ticker": ticker,
                "v3_rank": v3_rank,
                "v2_rank": v2_rank,
                "reason": override_result.override_reason,
                "driving_factor": override_result.driving_factor,
            })

            logger.warning(
                f"Sanity override for {ticker}: v3_rank={v3_rank}, v2_rank={v2_rank}, "
                f"reason={override_result.override_reason}"
            )
        else:
            # Keep v3 data but add sanity check info
            sec_copy = sec.copy()
            if override_result.override_applied:
                sec_copy["sanity_check"] = {
                    "v2_rank": v2_rank,
                    "rank_divergence": override_result.rank_divergence,
                    "driving_factor": override_result.driving_factor,
                    "fallback_triggered": False,
                }
            modified_securities.append(sec_copy)

    # Re-rank after any substitutions
    modified_securities.sort(
        key=lambda x: Decimal(x.get("composite_score", "0")),
        reverse=True
    )
    for i, sec in enumerate(modified_securities):
        sec["composite_rank"] = i + 1

    # Update output
    output_v3["ranked_securities"] = modified_securities
    output_v3["sanity_overrides"] = {
        "enabled": True,
        "overrides_count": len(overrides_applied),
        "overrides": overrides_applied,
    }

    if overrides_applied:
        logger.info(f"Module 5: Applied {len(overrides_applied)} sanity overrides")

    return output_v3


# Default search paths for universe file with defensive features
DEFAULT_UNIVERSE_PATHS = [
    "production_data/universe.json",
    "wake_robin_data_pipeline/outputs/universe_snapshot_latest.json",
    "wake_robin_data_pipeline/outputs/universe.json",
    "data/universe.json",
]


def compute_module_5_composite_with_defensive(
    universe_result: dict,
    financial_result: dict,
    catalyst_result: dict,
    clinical_result: dict,
    as_of_date: str,
    weights: dict = None,
    normalization: str = "rank",
    coinvest_signals: dict = None,
    cohort_mode: str = "stage_only",
    apply_defensive_multiplier: bool = False,
    apply_position_sizing: bool = False,  # Deprecated: use expected_return for alpha signal
    validate: bool = False,
    universe_path: Optional[str] = None,
    universe_search_paths: Optional[List[str]] = None,
    enhancement_result: dict = None,
    use_v3_scoring: bool = True,
    use_v2_scoring: bool = True,
    market_data_by_ticker: Optional[Dict[str, Dict[str, Any]]] = None,
    raw_financial_data: Optional[List[Dict[str, Any]]] = None,
    enable_sanity_override: bool = True,
    historical_scores: Optional[List[Dict]] = None,
    historical_returns: Optional[Dict] = None,
    use_adaptive_weights: bool = False,
) -> dict:
    """
    Rank securities with defensive overlays integrated.

    Production wrapper for Module 5 composite scoring with v3 IC enhancements.

    Args:
        universe_result: Module 1 output
        financial_result: Module 2 output
        catalyst_result: Module 3 output
        clinical_result: Module 4 output
        as_of_date: Analysis date (YYYY-MM-DD)
        weights: Override default weights
        normalization: "rank" (default) or "zscore" (v1 only)
        coinvest_signals: Optional co-invest overlay data
        cohort_mode: "stage_only" (recommended) or "stage_mcap" (v1 only)
        apply_defensive_multiplier: Apply defensive score multiplier
        apply_position_sizing: Apply position sizing (deprecated, use expected_return)
        validate: Run validation checks
        universe_path: Explicit path to universe file with defensive_features
        universe_search_paths: Custom list of paths to search for universe file
        enhancement_result: Optional enhancement module results (PoS, regime, SI)
        use_v3_scoring: Use v3 IC-enhanced scoring (default True)
        use_v2_scoring: Use v2 if v3 disabled (default True)
        market_data_by_ticker: Optional market data for volatility/momentum
        enable_sanity_override: Apply sanity override for pathological rankings
        historical_scores: For adaptive weight learning (experimental)
        historical_returns: For adaptive weight learning (experimental)
        use_adaptive_weights: Enable adaptive weight learning (experimental)

    Returns:
        Module 5 output enriched with defensive overlay fields

    V3 Features (default):
        - All v2 features (caps, confidence, hybrid, hash, vol adjustment)
        - Catalyst signal decay (time-based IC modeling)
        - Price momentum (60-day relative strength vs XBI)
        - Peer-relative valuation (MCap-per-asset comparison)
        - Smart money signal (13F overlap with tier weighting)
        - Shrinkage normalization (Bayesian cohort adjustment)
        - Sanity override (fallback for pathological rankings)
        - Expected Return (score → rank → z-score → ER, λ=8%/σ/yr)
    """
    output_v2_for_sanity = None

    if use_v3_scoring:
        logger.info("Module 5: Using v3 scoring (IC-enhanced)")

        # Run v3
        output = compute_module_5_composite_v3(
            universe_result=universe_result,
            financial_result=financial_result,
            catalyst_result=catalyst_result,
            clinical_result=clinical_result,
            as_of_date=as_of_date,
            weights=weights,
            coinvest_signals=coinvest_signals,
            enhancement_result=enhancement_result,
            market_data_by_ticker=market_data_by_ticker,
            raw_financial_data=raw_financial_data,
            historical_scores=historical_scores,
            historical_returns=historical_returns,
            use_adaptive_weights=use_adaptive_weights,
            validate_inputs=validate,
        )

        # If sanity override enabled, also run v2 for comparison
        if enable_sanity_override:
            logger.info("Module 5: Running v2 for sanity comparison")
            output_v2_for_sanity = compute_module_5_composite_v2(
                universe_result=universe_result,
                financial_result=financial_result,
                catalyst_result=catalyst_result,
                clinical_result=clinical_result,
                as_of_date=as_of_date,
                weights=weights,
                coinvest_signals=coinvest_signals,
                enhancement_result=enhancement_result,
                market_data_by_ticker=market_data_by_ticker,
            )

            # Apply sanity override
            output = _apply_sanity_overrides(output, output_v2_for_sanity)

    elif use_v2_scoring:
        logger.info("Module 5: Using v2 scoring (caps/confidence/hybrid/hash)")
        output = compute_module_5_composite_v2(
            universe_result=universe_result,
            financial_result=financial_result,
            catalyst_result=catalyst_result,
            clinical_result=clinical_result,
            as_of_date=as_of_date,
            weights=weights,
            coinvest_signals=coinvest_signals,
            enhancement_result=enhancement_result,
            market_data_by_ticker=market_data_by_ticker,
        )
    else:
        logger.info("Module 5: Using v1 scoring (legacy)")
        output = compute_module_5_composite(
            universe_result=universe_result,
            financial_result=financial_result,
            catalyst_result=catalyst_result,
            clinical_result=clinical_result,
            as_of_date=as_of_date,
            weights=weights,
            normalization=normalization,
            coinvest_signals=coinvest_signals,
            cohort_mode=cohort_mode,
            enhancement_result=enhancement_result,
        )

    # Build defensive_features lookup from RAW universe file
    # Module 1 strips out defensive_features, so we load the raw file
    defensive_by_ticker = {}
    raw_universe = None
    loaded_from = None

    # If explicit path provided, use it
    if universe_path:
        path = Path(universe_path)
        if path.exists():
            with open(path, 'r') as f:
                raw_universe = json.load(f)
            loaded_from = path
        else:
            logger.warning(f"Specified universe_path does not exist: {universe_path}")

    # Otherwise search configured paths
    if raw_universe is None:
        search_paths = universe_search_paths or DEFAULT_UNIVERSE_PATHS
        for path_str in search_paths:
            path = Path(path_str)
            if path.exists():
                with open(path, 'r') as f:
                    raw_universe = json.load(f)
                loaded_from = path
                break

    if raw_universe:
        logger.info(f"Loaded defensive features from {loaded_from}")

        # Handle both dict and array formats
        if isinstance(raw_universe, dict):
            securities = raw_universe.get("active_securities", [])
        else:
            securities = raw_universe  # Direct array

        for sec in securities:
            ticker = sec.get("ticker")
            if ticker and "defensive_features" in sec:
                defensive_by_ticker[ticker] = {"defensive_features": sec["defensive_features"]}

        logger.info(f"Extracted defensive_features for {len(defensive_by_ticker)} tickers")
    else:
        logger.warning(
            "Could not find raw universe file for defensive features. "
            f"Searched: {universe_search_paths or DEFAULT_UNIVERSE_PATHS}"
        )

    # Add defensive overlays
    enrich_with_defensive_overlays(
        output,
        defensive_by_ticker,
        apply_multiplier=apply_defensive_multiplier,
        apply_position_sizing=apply_position_sizing,
        top_n=None,  # Show all ranked securities (no limit)
    )
    
    # Optionally validate
    if validate:
        validate_defensive_integration(output)
    
    return output


__version__ = "1.2.0"

# Convenience exports
__all__ = [
    "compute_module_5_composite_with_defensive",
    "enrich_with_defensive_overlays",
    "validate_defensive_integration",
    "compute_module_5_composite_v3",  # Direct access to v3 scorer
    "compute_module_5_composite_v2",  # Direct access to v2 scorer
    "V3_PRODUCTION_DEFAULTS",         # V3 configuration
]


if __name__ == "__main__":
    print(f"module_5_composite_with_defensive.py v{__version__}")
    print("Production wrapper - v3 IC-enhanced scoring with sanity overrides")
    print()
    print("V3 Features (default):")
    print("  - All v2 features (caps, confidence, hybrid, hash, vol adjustment)")
    print("  - Catalyst signal decay (time-based IC modeling)")
    print("  - Price momentum (60-day alpha vs XBI)")
    print("  - Peer-relative valuation (MCap-per-asset)")
    print("  - Smart money signal (13F tier-weighted overlap)")
    print("  - Shrinkage normalization (Bayesian cohort)")
    print("  - Sanity override (fallback for pathological rankings)")
    print()
    print("Feature Flags:")
    print("  STABLE (on):    catalyst_decay, momentum, valuation, smart_money")
    print("  MONITORED (on): volatility_adjustment, shrinkage_norm, smart_money_tiers")
    print("  EXPERIMENTAL:   interaction_terms, adaptive_weights, regime_adaptation")
    print()
    print("Usage:")
    print("  # Default (v3 with sanity override)")
    print("  result = compute_module_5_composite_with_defensive(...)")
    print()
    print("  # Disable sanity override")
    print("  result = compute_module_5_composite_with_defensive(..., enable_sanity_override=False)")
    print()
    print("  # Revert to v2")
    print("  result = compute_module_5_composite_with_defensive(..., use_v3_scoring=False)")
    print()
    print("  # Revert to v1 (legacy)")
    print("  result = compute_module_5_composite_with_defensive(..., use_v3_scoring=False, use_v2_scoring=False)")
    print()
    print("Configuration: config/v3_production_integration.py")
    print("Tests: python -m pytest tests/test_module_5_v3_regression.py -v")
