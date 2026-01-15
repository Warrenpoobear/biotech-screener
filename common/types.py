"""
Shared type definitions for biotech screener.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional
from enum import Enum


class Severity(Enum):
    """Data quality severity levels."""
    NONE = "none"
    SEV1 = "sev1"  # Minor: 10% penalty
    SEV2 = "sev2"  # Soft gate: 50% penalty
    SEV3 = "sev3"  # Hard gate: excluded


class StatusGate(Enum):
    """Universe status gates."""
    ACTIVE = "active"
    EXCLUDED_SHELL = "excluded_shell"
    EXCLUDED_DELISTED = "excluded_delisted"
    EXCLUDED_ACQUIRED = "excluded_acquired"
    NOT_FOUND = "not_found"


@dataclass
class SecurityRecord:
    """Core security record."""
    ticker: str
    status: StatusGate = StatusGate.ACTIVE
    market_cap_mm: Optional[Decimal] = None
    cash_mm: Optional[Decimal] = None
    debt_mm: Optional[Decimal] = None
    runway_months: Optional[Decimal] = None
    severity: Severity = Severity.NONE
    flags: List[str] = None
    
    def __post_init__(self):
        if self.flags is None:
            self.flags = []


@dataclass
class CatalystRecord:
    """Catalyst/trial record."""
    ticker: str
    nct_id: str
    phase: str
    primary_completion_date: Optional[str] = None
    study_status: Optional[str] = None
    indication: Optional[str] = None
    
    
@dataclass  
class ClinicalScore:
    """Clinical development score."""
    ticker: str
    phase_score: Decimal
    design_score: Decimal
    execution_score: Decimal
    endpoint_score: Decimal
    total_score: Decimal
    lead_phase: str
    flags: List[str] = None
    
    def __post_init__(self):
        if self.flags is None:
            self.flags = []


@dataclass
class CompositeRecord:
    """Final composite ranking record."""
    ticker: str
    composite_score: Decimal
    composite_rank: int
    clinical_dev_raw: Optional[Decimal]
    financial_raw: Optional[Decimal]
    catalyst_raw: Optional[Decimal]
    clinical_dev_normalized: Optional[Decimal]
    financial_normalized: Optional[Decimal]
    catalyst_normalized: Optional[Decimal]
    uncertainty_penalty: Decimal
    missing_subfactor_pct: Decimal
    market_cap_bucket: str
    stage_bucket: str
    severity: Severity
    flags: List[str]
    rankable: bool = True
