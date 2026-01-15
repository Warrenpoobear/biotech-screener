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
    # TIER 1: Elite Biotech-Dedicated
    # =========================================================================
    {
        'cik': '0001263508',
        'name': 'Baker Bros. Advisors LP',
        'short_name': 'Baker Bros',
        'style': 'concentrated_conviction',
        'focus': ['rare_disease', 'oncology', 'genetic_medicine'],
        'typical_position_size': 'large',
        'holding_period': 'long',
        'tier': 1,
    },
    {
        'cik': '0001346824',
        'name': 'RA Capital Management, L.P.',
        'short_name': 'RA Capital',
        'style': 'crossover_specialist',
        'focus': ['platform_technologies', 'rare_disease', 'oncology'],
        'typical_position_size': 'medium',
        'holding_period': 'long',
        'tier': 1,
    },
    {
        'cik': '0001224962',
        'name': 'Perceptive Advisors LLC',
        'short_name': 'Perceptive',
        'style': 'diversified_biotech',
        'focus': ['broad_biotech', 'clinical_stage', 'commercial'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 1,
    },
    {
        'cik': '0001056807',
        'name': 'BVF Partners (BVF Inc/IL)',
        'short_name': 'BVF',
        'style': 'value_activist',
        'focus': ['undervalued_biotech', 'activist_situations'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 1,
    },
    {
        'cik': '0001587114',
        'name': 'EcoR1 Capital, LLC',
        'short_name': 'EcoR1',
        'style': 'scientific_deep_dive',
        'focus': ['genetic_medicine', 'cell_therapy', 'rare_disease'],
        'typical_position_size': 'medium',
        'holding_period': 'long',
        'tier': 1,
    },
    
    # =========================================================================
    # TIER 2B: Biotech-Active Specialists
    # =========================================================================
    {
        'cik': '0001343781',
        'name': 'HealthCor Management, L.P.',
        'short_name': 'HealthCor',
        'style': 'healthcare_specialist',
        'focus': ['biotech', 'medical_devices', 'healthcare_services'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 2,
    },
    {
        'cik': '0001792126',
        'name': 'Logos Global Management LLC',
        'short_name': 'Logos Capital',
        'style': 'biotech_active',
        'focus': ['clinical_stage', 'platform_technologies'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 2,
    },
    {
        'cik': '0001544773',
        'name': 'Consonance Capital Management LP',
        'short_name': 'Consonance',
        'style': 'healthcare_specialist',
        'focus': ['biotech', 'healthcare_services', 'medtech'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 2,
    },
    {
        'cik': '0001274413',
        'name': 'Sectoral Asset Management Inc.',
        'short_name': 'Sectoral',
        'style': 'sector_specialist',
        'focus': ['biotech', 'pharmaceuticals'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 2,
    },
    {
        'cik': '0001482416',
        'name': 'Sio Capital Management LLC',
        'short_name': 'Sio Capital',
        'style': 'healthcare_specialist',
        'focus': ['biotech', 'specialty_pharma'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 2,
    },
    {
        'cik': '0001595725',
        'name': 'Rock Springs Capital Management LP',
        'short_name': 'Rock Springs',
        'style': 'healthcare_specialist',
        'focus': ['biotech', 'medical_devices'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 2,
    },
    {
        'cik': '0001511901',
        'name': 'Broadfin Capital LLC',
        'short_name': 'Broadfin',
        'style': 'healthcare_specialist',
        'focus': ['biotech', 'rare_disease'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 2,
    },
    {
        'cik': '0002018299',
        'name': 'Boxer Capital, LLC',
        'short_name': 'Boxer Capital',
        'style': 'healthcare_specialist',
        'focus': ['biotech', 'clinical_stage'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 2,
    },
    
    # =========================================================================
    # TIER 2A: Healthcare Platforms & Crossover Specialists
    # =========================================================================
    {
        'cik': '0001055951',
        'name': 'OrbiMed Advisors LLC',
        'short_name': 'OrbiMed',
        'style': 'diversified_healthcare',
        'focus': ['broad_healthcare', 'biotech', 'medtech', 'devices'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 2,
    },
    {
        'cik': '0001009258',
        'name': 'Deerfield Management Company, L.P.',
        'short_name': 'Deerfield',
        'style': 'multi_strategy_healthcare',
        'focus': ['royalties', 'structured_finance', 'equity', 'credit'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 2,
    },
    {
        'cik': '0001425738',
        'name': 'Redmile Group, LLC',
        'short_name': 'Redmile',
        'style': 'crossover_specialist',
        'focus': ['clinical_stage', 'platform_technologies', 'commercial'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 2,
    },
    {
        'cik': '0001493215',
        'name': 'RTW Investments, LP',
        'short_name': 'RTW',
        'style': 'crossover_specialist',
        'focus': ['biotech', 'genetic_medicine', 'global_health'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 2,
    },
    {
        'cik': '0001583977',
        'name': 'Cormorant Asset Management, LP',
        'short_name': 'Cormorant',
        'style': 'healthcare_specialist',
        'focus': ['biotech', 'medtech', 'healthcare_services'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 2,
    },
    {
        'cik': '0001674712',
        'name': 'Vivo Capital LLC',
        'short_name': 'Vivo Capital',
        'style': 'venture_crossover',
        'focus': ['asia_healthcare', 'biotech', 'medtech'],
        'typical_position_size': 'medium',
        'holding_period': 'long',
        'tier': 2,
    },
    
    # =========================================================================
    # TIER 2C-M: Multi-Strategy Pod Shops
    # =========================================================================
    {
        'cik': '0001423053',
        'name': 'Citadel Advisors LLC',
        'short_name': 'Citadel',
        'style': 'quantitative_fundamental',
        'focus': ['broad_market', 'healthcare_allocation'],
        'typical_position_size': 'small',
        'holding_period': 'short',
        'tier': 2,
        'note': 'Multi-strategy; healthcare is one sleeve',
    },
    {
        'cik': '0001273087',
        'name': 'Millennium Management LLC',
        'short_name': 'Millennium',
        'style': 'multi_strategy',
        'focus': ['broad_market', 'healthcare_allocation'],
        'typical_position_size': 'small',
        'holding_period': 'short',
        'tier': 2,
        'note': 'Multi-strategy; healthcare is one sleeve',
    },
    {
        'cik': '0001009207',
        'name': 'D.E. Shaw & Co., L.P.',
        'short_name': 'D.E. Shaw',
        'style': 'quantitative_fundamental',
        'focus': ['broad_market', 'healthcare_allocation'],
        'typical_position_size': 'small',
        'holding_period': 'short',
        'tier': 2,
        'note': 'Multi-strategy; healthcare is one sleeve',
    },
    {
        'cik': '0001218710',
        'name': 'Balyasny Asset Management L.P.',
        'short_name': 'Balyasny',
        'style': 'multi_strategy',
        'focus': ['broad_market', 'healthcare_allocation'],
        'typical_position_size': 'small',
        'holding_period': 'short',
        'tier': 2,
        'note': 'Multi-strategy; healthcare is one sleeve',
    },
    {
        'cik': '0001603466',
        'name': 'Point72 Asset Management, L.P.',
        'short_name': 'Point72',
        'style': 'multi_strategy',
        'focus': ['broad_market', 'healthcare_allocation'],
        'typical_position_size': 'small',
        'holding_period': 'short',
        'tier': 2,
        'note': 'Multi-strategy; healthcare is one sleeve',
    },
    {
        'cik': '0001665241',
        'name': 'Schonfeld Strategic Advisors LLC',
        'short_name': 'Schonfeld',
        'style': 'multi_strategy',
        'focus': ['broad_market', 'healthcare_allocation'],
        'typical_position_size': 'small',
        'holding_period': 'short',
        'tier': 2,
        'note': 'Multi-strategy; healthcare is one sleeve',
    },
    
    # =========================================================================
    # TIER 2C-Q: Quantitative Mega-Cap
    # =========================================================================
    {
        'cik': '0001037389',
        'name': 'Renaissance Technologies LLC',
        'short_name': 'Renaissance',
        'style': 'quantitative',
        'focus': ['broad_market', 'statistical_arbitrage'],
        'typical_position_size': 'small',
        'holding_period': 'short',
        'tier': 2,
        'note': 'Quantitative; healthcare is statistical signal',
    },
    {
        'cik': '0001478735',
        'name': 'Two Sigma Investments, LP',
        'short_name': 'Two Sigma',
        'style': 'quantitative',
        'focus': ['broad_market', 'data_driven'],
        'typical_position_size': 'small',
        'holding_period': 'short',
        'tier': 2,
        'note': 'Quantitative; healthcare is data signal',
    },
    
    # =========================================================================
    # TIER 2D: Generalist Fundamental L/S
    # =========================================================================
    {
        'cik': '0001103804',
        'name': 'Viking Global Investors LP',
        'short_name': 'Viking',
        'style': 'fundamental_long_short',
        'focus': ['broad_market', 'fundamental_analysis'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 2,
        'note': 'Generalist; healthcare is one sector',
    },
    {
        'cik': '0001135730',
        'name': 'Coatue Management, L.P.',
        'short_name': 'Coatue',
        'style': 'fundamental_long_short',
        'focus': ['tech_healthcare_crossover', 'growth_companies'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 2,
        'note': 'Generalist; healthcare/tech crossover focus',
    },
    {
        'cik': '0001279936',
        'name': 'Cantillon Capital Management LLC',
        'short_name': 'Cantillon',
        'style': 'fundamental_long_short',
        'focus': ['broad_market', 'quality_companies'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 2,
        'note': 'Generalist; healthcare is one sector',
    },
    {
        'cik': '0001061165',
        'name': 'Lone Pine Capital LLC',
        'short_name': 'Lone Pine',
        'style': 'fundamental_long_short',
        'focus': ['broad_market', 'concentrated_growth'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 2,
        'note': 'Generalist; healthcare is one sector',
    },
    
    # =========================================================================
    # TIER 3: Activist/Special Situations
    # =========================================================================
    {
        'cik': '0001577524',
        'name': 'Sarissa Capital Management LP',
        'short_name': 'Sarissa',
        'style': 'activist',
        'focus': ['biotech_activism', 'corporate_governance'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 3,
    },
    {
        'cik': '0001040273',
        'name': 'Third Point LLC',
        'short_name': 'Third Point',
        'style': 'activist',
        'focus': ['event_driven', 'activist_situations'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 3,
    },
    
    # =========================================================================
    # APPENDIX E: Conditional Add-Ons (Pending Verification)
    # =========================================================================
    {
        'cik': '0001856083',
        'name': 'Deep Track Capital, LP',
        'short_name': 'Deep Track',
        'style': 'biotech_specialist',
        'focus': ['biotech', 'clinical_stage'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 3,
        'note': 'Conditional add-on; pending verification',
    },
    {
        'cik': '0001633313',
        'name': 'Avoro Capital Advisors LLC',
        'short_name': 'Avoro',
        'style': 'concentrated_biotech',
        'focus': ['clinical_stage', 'rare_disease'],
        'typical_position_size': 'large',
        'holding_period': 'medium',
        'tier': 3,
        'note': 'Conditional add-on; pending verification',
    },
    {
        'cik': '0001569064',
        'name': 'Suvretta Capital Management, LLC',
        'short_name': 'Suvretta',
        'style': 'healthcare_specialist',
        'focus': ['biotech', 'healthcare'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 3,
        'note': 'Conditional add-on; pending verification',
    },
    {
        'cik': '0001802630',
        'name': 'Soleus Capital Management, L.P.',
        'short_name': 'Soleus',
        'style': 'healthcare_specialist',
        'focus': ['biotech', 'healthcare'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 3,
        'note': 'Conditional add-on; pending verification',
    },
    {
        'cik': '0001776382',
        'name': 'venBio Partners, LLC',
        'short_name': 'venBio',
        'style': 'biotech_specialist',
        'focus': ['biotech', 'therapeutics'],
        'typical_position_size': 'medium',
        'holding_period': 'medium',
        'tier': 3,
        'note': 'Conditional add-on; pending verification',
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
# 
# GUARDRAIL: Interpret within-tier only (don't average signals across tiers)
# =============================================================================

TIER_WEIGHTS = {
    1: 1.0,   # Full weight for tier 1
    2: 0.7,   # 70% weight for tier 2
    3: 0.4,   # 40% weight for tier 3
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
    'biotech_active': 0.9,
    'sector_specialist': 0.9,
    'multi_strategy': 0.6,
    'quantitative': 0.4,
    'fundamental_long_short': 0.7,
    'activist': 0.6,
    'biotech_specialist': 0.5,
    'concentrated_biotech': 0.8,
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
        duplicates = set([cik for cik in ciks if ciks.count(cik) > 1])
        raise ValueError(f"Duplicate CIKs in registry: {duplicates}")
    
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
    print(f"Tier 2: {len([m for m in ELITE_MANAGERS if m['tier'] == 2])}")
    print(f"Tier 3: {len([m for m in ELITE_MANAGERS if m['tier'] == 3])}")
    print()
    
    for tier in [1, 2, 3]:
        managers = [m for m in ELITE_MANAGERS if m['tier'] == tier]
        print(f"TIER {tier} ({len(managers)} managers):")
        for m in managers:
            weight = get_manager_weight(m)
            note = f" - {m.get('note', '')}" if 'note' in m else ''
            print(f"  {m['short_name']:20} CIK {m['cik']} weight={weight:.2f}{note}")
        print()