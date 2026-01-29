"""
Elite Biotech Manager Registry - Re-export from canonical source

IMPORTANT: This module re-exports from the canonical manager registry.
DO NOT add manager data here - update production_data/manager_registry.json instead.

The canonical source is loaded by the root elite_managers.py module.
"""

# Re-export everything from canonical source
from elite_managers import (
    # Manager lists
    ELITE_MANAGERS,
    get_elite_managers,
    get_conditional_managers,
    get_all_managers,
    # Lookup functions
    get_manager_by_cik,
    get_manager_by_short_name,
    get_tier_1_managers,
    # CIK lists
    get_all_ciks,
    get_elite_ciks,
    get_conditional_ciks,
    get_ciks_by_tier,
    # Weighting
    TIER_WEIGHTS,
    STYLE_CONVICTION_MULTIPLIER,
    get_manager_weight,
    # Validation
    validate_registry,
    get_registry_info,
)

__all__ = [
    'ELITE_MANAGERS',
    'get_elite_managers',
    'get_conditional_managers',
    'get_all_managers',
    'get_manager_by_cik',
    'get_manager_by_short_name',
    'get_tier_1_managers',
    'get_all_ciks',
    'get_elite_ciks',
    'get_conditional_ciks',
    'get_ciks_by_tier',
    'TIER_WEIGHTS',
    'STYLE_CONVICTION_MULTIPLIER',
    'get_manager_weight',
    'validate_registry',
    'get_registry_info',
]
