"""
AACT Enum Normalization for Wake Robin Biotech Alpha System

AACT switched from Title Case to SCREAMING_SNAKE_CASE around late 2025.
This module normalizes raw AACT values to canonical internal values.

Usage:
    from wake_robin.providers.aact.normalize import normalize_phase, normalize_status
    
    canonical_phase = normalize_phase(raw_phase)  # "PHASE2" -> "Phase 2"
"""

from typing import Optional

# =============================================================================
# PHASE NORMALIZATION
# =============================================================================

PHASE_MAP = {
    # New SCREAMING_SNAKE_CASE format (current AACT)
    'PHASE1': 'Phase 1',
    'PHASE2': 'Phase 2',
    'PHASE3': 'Phase 3',
    'PHASE4': 'Phase 4',
    'PHASE1/PHASE2': 'Phase 1/Phase 2',
    'PHASE2/PHASE3': 'Phase 2/Phase 3',
    'EARLY_PHASE1': 'Early Phase 1',
    'NA': 'Not Applicable',
    
    # Old Title Case format (historical AACT, may appear in archives)
    'Phase 1': 'Phase 1',
    'Phase 2': 'Phase 2',
    'Phase 3': 'Phase 3',
    'Phase 4': 'Phase 4',
    'Phase 1/Phase 2': 'Phase 1/Phase 2',
    'Phase 2/Phase 3': 'Phase 2/Phase 3',
    'Early Phase 1': 'Early Phase 1',
    'Not Applicable': 'Not Applicable',
    'N/A': 'Not Applicable',
}

# Phases relevant for biotech clinical trial scoring (Modules 3 & 4)
CLINICAL_PHASES = {'Phase 1', 'Phase 2', 'Phase 3', 'Phase 1/Phase 2', 'Phase 2/Phase 3'}


def normalize_phase(raw: Optional[str]) -> Optional[str]:
    """
    Normalize AACT phase value to canonical format.
    
    Returns None for null/empty/unknown phases (kills clinical scoring).
    """
    if raw is None or raw.strip() == '':
        return None
    
    cleaned = raw.strip()
    return PHASE_MAP.get(cleaned)  # Returns None if not in map


def is_clinical_phase(phase: Optional[str]) -> bool:
    """Check if normalized phase is relevant for clinical trial scoring."""
    return phase in CLINICAL_PHASES


# =============================================================================
# STATUS NORMALIZATION
# =============================================================================

STATUS_MAP = {
    # New SCREAMING_SNAKE_CASE format
    'RECRUITING': 'Recruiting',
    'ACTIVE_NOT_RECRUITING': 'Active, not recruiting',
    'COMPLETED': 'Completed',
    'TERMINATED': 'Terminated',
    'SUSPENDED': 'Suspended',
    'WITHDRAWN': 'Withdrawn',
    'ENROLLING_BY_INVITATION': 'Enrolling by invitation',
    'NOT_YET_RECRUITING': 'Not yet recruiting',
    'UNKNOWN_STATUS': 'Unknown status',
    'NO_LONGER_AVAILABLE': 'No longer available',
    'APPROVED_FOR_MARKETING': 'Approved for marketing',
    'AVAILABLE': 'Available',
    'TEMPORARILY_NOT_AVAILABLE': 'Temporarily not available',
    'WITHHELD': 'Withheld',
    
    # Old Title Case format
    'Recruiting': 'Recruiting',
    'Active, not recruiting': 'Active, not recruiting',
    'Completed': 'Completed',
    'Terminated': 'Terminated',
    'Suspended': 'Suspended',
    'Withdrawn': 'Withdrawn',
    'Enrolling by invitation': 'Enrolling by invitation',
    'Not yet recruiting': 'Not yet recruiting',
    'Unknown status': 'Unknown status',
    'No longer available': 'No longer available',
    'Approved for marketing': 'Approved for marketing',
    'Available': 'Available',
    'Temporarily not available': 'Temporarily not available',
}

# Active statuses (trial is ongoing, catalyst potential)
ACTIVE_STATUSES = {
    'Recruiting',
    'Active, not recruiting',
    'Enrolling by invitation',
    'Not yet recruiting',
}

# Terminal statuses (trial concluded)
TERMINAL_STATUSES = {
    'Completed',
    'Terminated',
    'Suspended',
    'Withdrawn',
}


def normalize_status(raw: Optional[str]) -> Optional[str]:
    """Normalize AACT status value to canonical format."""
    if raw is None or raw.strip() == '':
        return None
    
    cleaned = raw.strip()
    return STATUS_MAP.get(cleaned)


def is_active_trial(status: Optional[str]) -> bool:
    """Check if normalized status indicates an active/ongoing trial."""
    return status in ACTIVE_STATUSES


# =============================================================================
# INTERVENTION TYPE NORMALIZATION
# =============================================================================

INTERVENTION_TYPE_MAP = {
    # New SCREAMING_SNAKE_CASE format
    'DRUG': 'Drug',
    'BIOLOGICAL': 'Biological',
    'DEVICE': 'Device',
    'PROCEDURE': 'Procedure',
    'RADIATION': 'Radiation',
    'BEHAVIORAL': 'Behavioral',
    'GENETIC': 'Genetic',
    'DIETARY_SUPPLEMENT': 'Dietary Supplement',
    'COMBINATION_PRODUCT': 'Combination Product',
    'DIAGNOSTIC_TEST': 'Diagnostic Test',
    'OTHER': 'Other',
    
    # Old Title Case format
    'Drug': 'Drug',
    'Biological': 'Biological',
    'Device': 'Device',
    'Procedure': 'Procedure',
    'Radiation': 'Radiation',
    'Behavioral': 'Behavioral',
    'Genetic': 'Genetic',
    'Dietary Supplement': 'Dietary Supplement',
    'Combination Product': 'Combination Product',
    'Diagnostic Test': 'Diagnostic Test',
    'Other': 'Other',
}

# Intervention types relevant for biotech (drug/biological focus)
BIOTECH_INTERVENTION_TYPES = {'Drug', 'Biological', 'Combination Product'}


def normalize_intervention_type(raw: Optional[str]) -> Optional[str]:
    """Normalize AACT intervention_type to canonical format."""
    if raw is None or raw.strip() == '':
        return None
    
    cleaned = raw.strip()
    return INTERVENTION_TYPE_MAP.get(cleaned)


def is_biotech_intervention(intervention_type: Optional[str]) -> bool:
    """Check if normalized intervention type is biotech-relevant."""
    return intervention_type in BIOTECH_INTERVENTION_TYPES


# =============================================================================
# STUDY TYPE NORMALIZATION
# =============================================================================

STUDY_TYPE_MAP = {
    # New format
    'INTERVENTIONAL': 'Interventional',
    'OBSERVATIONAL': 'Observational',
    'EXPANDED_ACCESS': 'Expanded Access',
    
    # Old format
    'Interventional': 'Interventional',
    'Observational': 'Observational',
    'Expanded Access': 'Expanded Access',
}


def normalize_study_type(raw: Optional[str]) -> Optional[str]:
    """Normalize AACT study_type to canonical format."""
    if raw is None or raw.strip() == '':
        return None
    
    cleaned = raw.strip()
    return STUDY_TYPE_MAP.get(cleaned)


# =============================================================================
# CONVENIENCE: NORMALIZE FULL ROW
# =============================================================================

def normalize_study_row(row: dict) -> dict:
    """
    Apply all normalizations to a raw AACT study row.
    
    Returns a new dict with normalized values. Original row unchanged.
    """
    normalized = row.copy()
    
    normalized['phase'] = normalize_phase(row.get('phase'))
    normalized['overall_status'] = normalize_status(row.get('overall_status'))
    normalized['study_type'] = normalize_study_type(row.get('study_type'))
    
    return normalized


def normalize_intervention_row(row: dict) -> dict:
    """Apply normalizations to a raw AACT intervention row."""
    normalized = row.copy()
    normalized['intervention_type'] = normalize_intervention_type(row.get('intervention_type'))
    return normalized
