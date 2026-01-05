"""
Elite Biotech Manager Registry for Wake Robin Biotech Alpha System

This module maintains the canonical list of elite biotech-focused hedge funds
whose 13F filings we track for co-investment signals.

CIK (Central Index Key) is the SEC's unique identifier for filers.
Find CIKs at: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany

Usage:
    from wake_robin.providers.sec_13f.elite_managers import ELITE_MANAGERS, get_manager_by_cik
    
    for manager in ELITE_MANAGERS:
        print(f"{manager['name']}: CIK {manager['cik']}")
"""

from typing import Optional

# =============================================================================
# ELITE BIOTECH MANAGER REGISTRY
# =============================================================================
# 
# Selection criteria:
# - Biotech/healthcare specialist (>50% portfolio in life sciences)
# - Long-term track record (10+ years)
# - Significant AUM ($1B+ in 13F securities)
# - Known for deep scientific/clinical due diligence
#
# These represent the "Mount Rushmore" of biotech investing.
# =============================================================================

ELITE_MANAGERS = [
    # =========================================================================
    # TIER 1: Core biotech specialists with 20+ year track records
    # =========================================================================
    {
        'cik': '1074999',
        'name': 'Baker Bros. Advisors LP',
        'short_name': 'Baker Bros',
        'style': 'concentrated_conviction',
        'focus': ['rare_disease', 'oncology', 'genetic_medicine'],
        'typical_position_size': 'large',  # Often 10-20% positions
        'holding_period': 'long',  # 3-10 year holds
        'tier': 1,
    },
    {
        'cik': '1535392',
        'name': 'RA Capital Management, L.P.',
        'short_name': 'RA Capital',
        'style': 'crossover_specialist',
        'focus': ['platform_technologies', 'rare_disease', 'oncology'],
        'typical_position_size': 'medium',
        'holding_period': 'long',
        'tier': 1,
    },
    {
        'cik': '1303382',
        'name': 'Perceptive Advisors LLC',
        'short_name': 'Perceptive',
        'style': 'diversified_biotech',
        'focus': ['broad_biotech', 'clinical_stage', 'commercial'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 1,
    },
    {
        'cik': '1056831',
        'name': 'Biotechnology Value Fund, L.P.',
        'short_name': 'BVF',
        'style': 'value_activist',
        'focus': ['undervalued_biotech', 'activist_situations'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 1,
    },
    {
        'cik': '1633642',
        'name': 'EcoR1 Capital, LLC',
        'short_name': 'EcoR1',
        'style': 'scientific_deep_dive',
        'focus': ['genetic_medicine', 'cell_therapy', 'rare_disease'],
        'typical_position_size': 'medium',
        'holding_period': 'long',
        'tier': 1,
    },
    
    # =========================================================================
    # TIER 2: Excellent biotech specialists
    # =========================================================================
    {
        'cik': '1167483',
        'name': 'OrbiMed Advisors LLC',
        'short_name': 'OrbiMed',
        'style': 'diversified_healthcare',
        'focus': ['broad_healthcare', 'biotech', 'medtech'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 2,
    },
    {
        'cik': '1510387',
        'name': 'Redmile Group, LLC',
        'short_name': 'Redmile',
        'style': 'crossover_specialist',
        'focus': ['clinical_stage', 'platform_technologies'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 2,
    },
    {
        'cik': '1454502',
        'name': 'Deerfield Management Company, L.P.',
        'short_name': 'Deerfield',
        'style': 'multi_strategy_healthcare',
        'focus': ['royalties', 'structured_finance', 'equity'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 2,
    },
    {
        'cik': '1358706',
        'name': 'Farallon Capital Management, L.L.C.',
        'short_name': 'Farallon',
        'style': 'event_driven',
        'focus': ['healthcare_events', 'M&A', 'special_situations'],
        'typical_position_size': 'medium',
        'holding_period': 'short_to_medium',
        'tier': 2,
    },
    {
        'cik': '1040273',
        'name': 'Citadel Advisors LLC',
        'short_name': 'Citadel',
        'style': 'quantitative_fundamental',
        'focus': ['broad_market', 'healthcare_allocation'],
        'typical_position_size': 'small',
        'holding_period': 'short',
        'tier': 2,
        'note': 'Multi-strategy; healthcare is one sleeve',
    },
    
    # =========================================================================
    # TIER 3: Notable biotech allocators (smaller or narrower focus)
    # =========================================================================
    {
        'cik': '1603466',
        'name': 'Avoro Capital Advisors LLC',
        'short_name': 'Avoro',
        'style': 'concentrated_biotech',
        'focus': ['clinical_stage', 'rare_disease'],
        'typical_position_size': 'large',
        'holding_period': 'medium',
        'tier': 3,
    },
    {
        'cik': '1495584',
        'name': 'Venrock Healthcare Capital Partners',
        'short_name': 'Venrock HCP',
        'style': 'venture_crossover',
        'focus': ['early_stage', 'platform_technologies'],
        'typical_position_size': 'medium',
        'holding_period': 'long',
        'tier': 3,
    },
    {
        'cik': '1802994',
        'name': 'Cormorant Asset Management, LP',
        'short_name': 'Cormorant',
        'style': 'healthcare_specialist',
        'focus': ['biotech', 'medtech', 'healthcare_services'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 3,
    },
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_manager_by_cik(cik: str) -> Optional[dict]:
    """Look up manager metadata by CIK."""
    cik_clean = cik.lstrip('0')  # Normalize leading zeros
    for manager in ELITE_MANAGERS:
        if manager['cik'].lstrip('0') == cik_clean:
            return manager
    return None


def get_manager_by_short_name(short_name: str) -> Optional[dict]:
    """Look up manager by short name (case-insensitive)."""
    for manager in ELITE_MANAGERS:
        if manager['short_name'].lower() == short_name.lower():
            return manager
    return None


def get_tier_1_managers() -> list[dict]:
    """Return only Tier 1 (highest conviction) managers."""
    return [m for m in ELITE_MANAGERS if m['tier'] == 1]


def get_all_ciks() -> list[str]:
    """Return list of all elite manager CIKs."""
    return [m['cik'] for m in ELITE_MANAGERS]


def get_ciks_by_tier(tier: int) -> list[str]:
    """Return CIKs for managers of a specific tier."""
    return [m['cik'] for m in ELITE_MANAGERS if m['tier'] == tier]


# =============================================================================
# CONVICTION WEIGHTING
# =============================================================================
# 
# When scoring positions, we weight by manager tier and style:
# - Tier 1 managers get higher weight (they're pure biotech specialists)
# - Concentrated styles get higher weight (higher conviction per position)
# =============================================================================

TIER_WEIGHTS = {
    1: 1.0,   # Full weight for tier 1
    2: 0.7,  # 70% weight for tier 2
    3: 0.4,  # 40% weight for tier 3
}

STYLE_CONVICTION_MULTIPLIER = {
    'concentrated_conviction': 1.2,
    'scientific_deep_dive': 1.1,
    'crossover_specialist': 1.0,
    'value_activist': 1.0,
    'diversified_biotech': 0.9,
    'diversified_healthcare': 0.8,
    'multi_strategy_healthcare': 0.7,
    'quantitative_fundamental': 0.5,
    'event_driven': 0.6,
    'venture_crossover': 0.8,
    'healthcare_specialist': 0.9,
}


def get_manager_weight(manager: dict) -> float:
    """
    Calculate conviction weight for a manager.
    
    Used when aggregating signals across multiple managers.
    """
    tier_weight = TIER_WEIGHTS.get(manager.get('tier', 3), 0.4)
    style = manager.get('style', 'diversified_biotech')
    style_mult = STYLE_CONVICTION_MULTIPLIER.get(style, 0.8)
    return tier_weight * style_mult


# =============================================================================
# VALIDATION
# =============================================================================

def validate_registry():
    """Validate registry integrity."""
    ciks = [m['cik'] for m in ELITE_MANAGERS]
    
    # Check for duplicate CIKs
    if len(ciks) != len(set(ciks)):
        raise ValueError("Duplicate CIKs in registry")
    
    # Check required fields
    required = ['cik', 'name', 'short_name', 'tier']
    for manager in ELITE_MANAGERS:
        for field in required:
            if field not in manager:
                raise ValueError(f"Missing {field} for manager: {manager}")
    
    return True


if __name__ == '__main__':
    # Quick validation
    validate_registry()
    
    print("Elite Biotech Manager Registry")
    print("=" * 50)
    print(f"Total managers: {len(ELITE_MANAGERS)}")
    print(f"Tier 1: {len(get_tier_1_managers())}")
    print()
    
    for tier in [1, 2, 3]:
        managers = [m for m in ELITE_MANAGERS if m['tier'] == tier]
        print(f"TIER {tier}:")
        for m in managers:
            weight = get_manager_weight(m)
            print(f"  {m['short_name']:20} CIK {m['cik']:10} weight={weight:.2f}")
        print()
