"""
Shared type definitions for biotech screener.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Union
from typing_extensions import TypedDict, NotRequired
from enum import Enum


# =============================================================================
# TYPE ALIASES
# =============================================================================

Ticker = str
CUSIP = str
NCTId = str
Score = Decimal
DateString = str  # Format: YYYY-MM-DD


# =============================================================================
# UNIVERSE RECORD TYPES (Module 1)
# =============================================================================

class RawUniverseRecord(TypedDict, total=False):
    """Input record for universe filtering."""
    ticker: str
    company_name: str
    status: str
    market_cap_mm: Union[float, int, str, None]
    sector: str
    exchange: str


class ActiveSecurityRecord(TypedDict):
    """Output record for active securities."""
    ticker: str
    status: str
    market_cap_mm: Optional[str]
    company_name: Optional[str]


class ExcludedSecurityRecord(TypedDict):
    """Output record for excluded securities."""
    ticker: str
    reason: str


class UniverseDiagnosticCounts(TypedDict):
    """Diagnostic counts for universe module."""
    total_input: int
    active: int
    excluded: int
    excluded_by_reason: Dict[str, int]


class UniverseResult(TypedDict):
    """Complete result from Module 1."""
    as_of_date: str
    active_securities: List[ActiveSecurityRecord]
    excluded_securities: List[ExcludedSecurityRecord]
    diagnostic_counts: UniverseDiagnosticCounts
    provenance: Dict[str, object]


# =============================================================================
# FINANCIAL RECORD TYPES (Module 2)
# =============================================================================

class FinancialInputRecord(TypedDict, total=False):
    """Input record for financial scoring."""
    ticker: str
    Cash: float
    MarketableSecurities: float
    ShortTermInvestments: float
    CFO: float
    CFO_quarterly: float
    CFO_YTD: float
    FCF: float
    FCF_quarterly: float
    NetIncome: float
    R_D: float  # Note: R&D is escaped as R_D
    Debt: float
    cash_mm: float
    burn_rate_mm: float
    rd_mm: float
    market_cap_mm: float
    source_date: str


class MarketDataRecord(TypedDict, total=False):
    """Market data for a security."""
    ticker: str
    market_cap: float
    price: float
    avg_volume: float
    volume_avg_30d: float


class FinancialScoreRecord(TypedDict, total=False):
    """Output record for financial scoring."""
    ticker: str
    financial_normalized: float
    runway_months: Optional[float]
    runway_score: Optional[float]
    dilution_score: Optional[float]
    liquidity_score: Optional[float]
    cash_to_mcap: Optional[float]
    monthly_burn: Optional[float]
    has_financial_data: bool
    severity: str
    flags: List[str]
    burn_source: str
    burn_confidence: str
    burn_period: str
    liquidity_gate: bool
    dollar_adv: Optional[float]
    liquid_assets: float
    liquid_components: List[str]
    financial_data_state: str
    missing_fields: List[str]
    inputs_used: Dict[str, str]


class FinancialDiagnosticCounts(TypedDict):
    """Diagnostic counts for financial module."""
    scored: int
    missing: int


class FinancialResult(TypedDict):
    """Complete result from Module 2."""
    scores: List[FinancialScoreRecord]
    diagnostic_counts: FinancialDiagnosticCounts


# =============================================================================
# TRIAL RECORD TYPES (Module 3 & 4)
# =============================================================================

class TrialRecord(TypedDict, total=False):
    """Clinical trial record from CT.gov."""
    ticker: str
    nct_id: str
    phase: str
    study_status: str
    primary_completion_date: Optional[str]
    study_first_posted_date: Optional[str]
    results_first_posted_date: Optional[str]
    conditions: List[str]
    interventions: List[str]
    sponsor: str
    indication: str
    enrollment: int
    study_type: str


class CatalystDiagnosticCounts(TypedDict):
    """Diagnostic counts for catalyst module."""
    events_detected: int
    events_deduped: NotRequired[int]
    severe_negatives: int
    tickers_with_events: int
    tickers_analyzed: int


# =============================================================================
# CLINICAL SCORE TYPES (Module 4)
# =============================================================================

class ClinicalScoreRecord(TypedDict, total=False):
    """Output record for clinical scoring."""
    ticker: str
    clinical_normalized: float
    phase_score: float
    design_score: float
    execution_score: float
    endpoint_score: float
    lead_phase: str
    lead_indication: Optional[str]
    trial_count: int
    flags: List[str]
    severity: str


class ClinicalDiagnosticCounts(TypedDict):
    """Diagnostic counts for clinical module."""
    scored: int
    total_trials: int
    pit_filtered: int


class ClinicalResult(TypedDict):
    """Complete result from Module 4."""
    scores: List[ClinicalScoreRecord]
    diagnostic_counts: ClinicalDiagnosticCounts


# =============================================================================
# COMPOSITE RECORD TYPES (Module 5)
# =============================================================================

class CompositeRankedRecord(TypedDict, total=False):
    """Ranked security in composite output."""
    ticker: str
    composite_score: str
    composite_rank: int
    clinical_dev_raw: Optional[str]
    financial_raw: Optional[str]
    catalyst_raw: Optional[str]
    clinical_dev_normalized: Optional[str]
    financial_normalized: Optional[str]
    catalyst_normalized: Optional[str]
    uncertainty_penalty: str
    missing_subfactor_pct: str
    market_cap_bucket: str
    stage_bucket: str
    severity: str
    flags: List[str]
    rankable: bool


class CompositeDiagnosticCounts(TypedDict):
    """Diagnostic counts for composite module."""
    rankable: int
    excluded: int


class CompositeResult(TypedDict):
    """Complete result from Module 5."""
    ranked_securities: List[CompositeRankedRecord]
    excluded_securities: List[ExcludedSecurityRecord]
    diagnostic_counts: CompositeDiagnosticCounts


# =============================================================================
# PROVENANCE TYPES
# =============================================================================

class ProvenanceRecord(TypedDict, total=False):
    """Provenance information for audit trail."""
    version: str
    generated_at: str
    input_hash: str
    as_of_date: str
    module: str


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
