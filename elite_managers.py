"""
Elite Biotech Manager Registry for Wake Robin Biotech Alpha System

This module provides access to the canonical list of elite biotech-focused hedge funds
whose 13F filings we track for co-investment signals.

IMPORTANT: The canonical source of truth is production_data/manager_registry.json
This module loads from that file - DO NOT hardcode manager lists here.

CIK (Central Index Key) is the SEC's unique identifier for filers.
Find CIKs at: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany

Usage:
    from elite_managers import get_elite_managers, get_manager_by_cik

    for manager in get_elite_managers():
        print(f"{manager['name']}: CIK {manager['cik']}")
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any

# =============================================================================
# REGISTRY LOADING
# =============================================================================

REGISTRY_PATH = Path(__file__).parent / "production_data" / "manager_registry.json"

_registry_cache: Optional[Dict[str, Any]] = None


def _load_registry() -> Dict[str, Any]:
    """Load manager registry from canonical JSON source."""
    global _registry_cache
    if _registry_cache is None:
        with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
            _registry_cache = json.load(f)
    return _registry_cache


def _normalize_cik(cik: str) -> str:
    """Normalize CIK to 10-digit format with leading zeros."""
    return cik.lstrip('0').zfill(10)


def _manager_to_legacy_format(mgr: Dict[str, Any], tier: int) -> Dict[str, Any]:
    """
    Convert registry format to legacy format for backwards compatibility.

    Registry format: {cik, name, aum_b, style}
    Legacy format: {cik, name, short_name, style, tier, ...}
    """
    name = mgr['name']

    # Generate short name from full name
    # "Baker Bros Advisors" -> "Baker Bros"
    # "RA Capital Management" -> "RA Capital"
    words = name.split()
    if len(words) >= 2:
        # Check for common suffixes to exclude
        suffixes = {'Advisors', 'Management', 'Partners', 'Capital', 'Group',
                    'Investments', 'Asset', 'LP', 'LLC', 'Inc', 'Ltd'}
        short_words = []
        for w in words:
            if w in suffixes and len(short_words) >= 2:
                break
            short_words.append(w)
        short_name = ' '.join(short_words[:3])  # Max 3 words
    else:
        short_name = name

    return {
        'cik': mgr['cik'].lstrip('0'),  # Legacy format without leading zeros
        'name': name,
        'short_name': short_name,
        'style': mgr.get('style', 'diversified_healthcare'),
        'aum_b': mgr.get('aum_b', 0),
        'tier': tier,
    }


# =============================================================================
# MANAGER ACCESS FUNCTIONS
# =============================================================================

def get_elite_managers() -> List[Dict[str, Any]]:
    """
    Get all Elite Core managers (primary signal source).

    Returns list of manager dicts in legacy format.
    """
    registry = _load_registry()
    return [_manager_to_legacy_format(m, tier=1) for m in registry['elite_core']]


def get_conditional_managers() -> List[Dict[str, Any]]:
    """
    Get all Conditional managers (secondary breadth signal).

    Returns list of manager dicts in legacy format.
    """
    registry = _load_registry()
    return [_manager_to_legacy_format(m, tier=2) for m in registry['conditional']]


def get_all_managers() -> List[Dict[str, Any]]:
    """Get all managers (Elite Core + Conditional)."""
    return get_elite_managers() + get_conditional_managers()


# Backwards compatibility: ELITE_MANAGERS as a property-like access
# This allows existing code using `from elite_managers import ELITE_MANAGERS` to work
def _get_elite_managers_compat():
    """Backwards compatible access to elite managers."""
    return get_elite_managers()


# For backwards compatibility with existing imports
# WARNING: This is loaded at import time - prefer using get_elite_managers()
ELITE_MANAGERS = property(lambda self: get_elite_managers())


# Create a class to make ELITE_MANAGERS work as expected
class _ManagersProxy:
    """Proxy class to make ELITE_MANAGERS behave like a list."""

    def __iter__(self):
        return iter(get_elite_managers())

    def __len__(self):
        return len(get_elite_managers())

    def __getitem__(self, idx):
        return get_elite_managers()[idx]

    def __contains__(self, item):
        return item in get_elite_managers()


# This is the backwards-compatible ELITE_MANAGERS "list"
ELITE_MANAGERS = _ManagersProxy()


# =============================================================================
# LOOKUP FUNCTIONS
# =============================================================================

def get_manager_by_cik(cik: str) -> Optional[Dict[str, Any]]:
    """Look up manager metadata by CIK."""
    cik_clean = cik.lstrip('0')
    for manager in get_all_managers():
        if manager['cik'].lstrip('0') == cik_clean:
            return manager
    return None


def get_manager_by_short_name(short_name: str) -> Optional[Dict[str, Any]]:
    """Look up manager by short name (case-insensitive)."""
    short_lower = short_name.lower()
    for manager in get_all_managers():
        if manager['short_name'].lower() == short_lower:
            return manager
        # Also check if query is contained in name
        if short_lower in manager['name'].lower():
            return manager
    return None


def get_tier_1_managers() -> List[Dict[str, Any]]:
    """Return only Elite Core (Tier 1) managers."""
    return get_elite_managers()


def get_all_ciks() -> List[str]:
    """Return list of all manager CIKs."""
    return [m['cik'] for m in get_all_managers()]


def get_elite_ciks() -> List[str]:
    """Return list of Elite Core manager CIKs."""
    return [m['cik'] for m in get_elite_managers()]


def get_conditional_ciks() -> List[str]:
    """Return list of Conditional manager CIKs."""
    return [m['cik'] for m in get_conditional_managers()]


def get_ciks_by_tier(tier: int) -> List[str]:
    """Return CIKs for managers of a specific tier (1=Elite, 2=Conditional)."""
    if tier == 1:
        return get_elite_ciks()
    elif tier == 2:
        return get_conditional_ciks()
    return []


# =============================================================================
# CONVICTION WEIGHTING
# =============================================================================
#
# When scoring positions, we weight by manager tier and style:
# - Elite Core (Tier 1) managers get full weight
# - Conditional (Tier 2) managers get reduced weight
# - Conditional signal is also capped at 30% of total (see ic_enhancements.py)
# =============================================================================

TIER_WEIGHTS = {
    1: 1.0,   # Full weight for Elite Core
    2: 0.3,   # Reduced weight for Conditional (also capped at 30% in scoring)
}

STYLE_CONVICTION_MULTIPLIER = {
    # High conviction styles (biotech specialists)
    'concentrated_conviction': 1.2,
    'concentrated_life_sciences': 1.2,
    'scientific_deep_dive': 1.1,
    'clinical_probability_engine': 1.1,
    'clinical_stage_specialists': 1.1,
    'physician_led_fundamental': 1.1,
    # Standard biotech styles
    'event_driven': 1.0,
    'oncology_focused': 1.0,
    'genomics_focused': 1.0,
    'healthcare_fundamental': 1.0,
    'fundamental_long_short': 1.0,
    'life_sciences_value': 1.0,
    'value_healthcare': 1.0,
    'biotech_value': 1.0,
    'garp_healthcare': 1.0,
    # Crossover/diversified styles
    'venture_crossover': 0.9,
    'life_sciences_crossover': 0.9,
    'growth_equity': 0.9,
    'healthcare_long_short': 0.9,
    'diversified_healthcare': 0.8,
    'multi_strategy': 0.8,
    'multi_stage_healthcare': 0.8,
    'private_equity_crossover': 0.8,
    'asia_biotech': 0.8,
    'scientific_data_driven': 0.9,
    # Multi-strategy/quant (Conditional tier)
    'multi_strategy_macro': 0.5,
    'multi_strategy_platform': 0.4,
    'quantitative': 0.3,
}


def get_manager_weight(manager: Dict[str, Any]) -> float:
    """
    Calculate conviction weight for a manager.

    Used when aggregating signals across multiple managers.
    """
    tier_weight = TIER_WEIGHTS.get(manager.get('tier', 2), 0.3)
    style = manager.get('style', 'diversified_healthcare')
    style_mult = STYLE_CONVICTION_MULTIPLIER.get(style, 0.8)
    return tier_weight * style_mult


# =============================================================================
# VALIDATION
# =============================================================================

def validate_registry() -> bool:
    """Validate registry integrity."""
    registry = _load_registry()

    all_mgrs = registry['elite_core'] + registry['conditional']
    ciks = [m['cik'] for m in all_mgrs]

    # Check for duplicate CIKs
    if len(ciks) != len(set(ciks)):
        raise ValueError("Duplicate CIKs in registry")

    # Check required fields
    required = ['cik', 'name', 'style']
    for manager in all_mgrs:
        for field in required:
            if field not in manager:
                raise ValueError(f"Missing {field} for manager: {manager}")

    return True


def get_registry_info() -> Dict[str, Any]:
    """Get registry metadata."""
    registry = _load_registry()
    return {
        'elite_core_count': len(registry['elite_core']),
        'conditional_count': len(registry['conditional']),
        'total_count': len(registry['elite_core']) + len(registry['conditional']),
        'version': registry.get('metadata', {}).get('version', 'unknown'),
        'last_updated': registry.get('metadata', {}).get('last_updated', 'unknown'),
        'elite_aum_b': registry.get('metadata', {}).get('total_elite_aum_b', 0),
    }


if __name__ == '__main__':
    # Quick validation
    validate_registry()

    info = get_registry_info()
    print("Elite Biotech Manager Registry")
    print("=" * 50)
    print(f"Version: {info['version']}")
    print(f"Last Updated: {info['last_updated']}")
    print(f"Elite Core: {info['elite_core_count']} managers (~${info['elite_aum_b']}B)")
    print(f"Conditional: {info['conditional_count']} managers")
    print(f"Total: {info['total_count']} managers")
    print()

    print("ELITE CORE (Tier 1):")
    for m in get_elite_managers()[:10]:
        weight = get_manager_weight(m)
        print(f"  {m['short_name']:25} CIK {m['cik']:10} weight={weight:.2f}")
    if len(get_elite_managers()) > 10:
        print(f"  ... and {len(get_elite_managers()) - 10} more")
    print()

    print("CONDITIONAL (Tier 2):")
    for m in get_conditional_managers():
        weight = get_manager_weight(m)
        print(f"  {m['short_name']:25} CIK {m['cik']:10} weight={weight:.2f}")
