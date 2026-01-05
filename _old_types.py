"""
common/types.py - Shared type definitions for Wake Robin screening modules.

Provides enums and type aliases used across all modules.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Any, Optional
from decimal import Decimal


class Severity(Enum):
    """Risk severity levels for scoring penalties and exclusions."""
    NONE = "none"
    SEV1 = "sev1"  # Minor concern, 10% penalty
    SEV2 = "sev2"  # Significant concern, 50% penalty
    SEV3 = "sev3"  # Critical, excluded from ranking


class StatusGate(Enum):
    """Security status classifications for universe filtering."""
    ACTIVE = "active"
    EXCLUDED_DELISTED = "excluded_delisted"
    EXCLUDED_ACQUIRED = "excluded_acquired"
    EXCLUDED_SHELL = "excluded_shell"
    EXCLUDED_OTHER = "excluded_other"


class TrialStatus(Enum):
    """Clinical trial status from ClinicalTrials.gov."""
    RECRUITING = "recruiting"
    ACTIVE_NOT_RECRUITING = "active_not_recruiting"
    COMPLETED = "completed"
    TERMINATED = "terminated"
    WITHDRAWN = "withdrawn"
    SUSPENDED = "suspended"
    UNKNOWN = "unknown"


class Phase(Enum):
    """Clinical trial phases."""
    PRECLINICAL = "preclinical"
    PHASE_1 = "phase 1"
    PHASE_1_2 = "phase 1/2"
    PHASE_2 = "phase 2"
    PHASE_2_3 = "phase 2/3"
    PHASE_3 = "phase 3"
    APPROVED = "approved"
    UNKNOWN = "unknown"


# Type aliases for clarity
Ticker = str
CUSIP = str
NCTId = str
Score = Decimal
