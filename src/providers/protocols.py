"""
Provider protocols and data structures for clinical trials data.

This module defines the interface that all clinical trials providers must implement,
ensuring consistent PIT-safe data delivery to Modules 3 and 4.
"""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol, Optional, TypedDict, Union


class TrialRowDict(TypedDict):
    """Dictionary representation of a TrialRow."""
    nct_id: str
    phase: str
    overall_status: str
    primary_completion_date: Optional[str]
    primary_completion_date_type: str
    last_update_posted_date: Optional[str]
    lead_sponsor: str
    study_type: str
    pcd_pushes_18m: int
    status_flips_18m: int
    flags: list[str]


class TrialDiffDict(TypedDict):
    """Dictionary representation of a TrialDiff."""
    nct_id: str
    snapshot_date_prev: str
    snapshot_date_curr: str
    pcd_prev: Optional[str]
    pcd_curr: Optional[str]
    pcd_type_prev: str
    pcd_type_curr: str
    pcd_pushed: bool
    status_prev: str
    status_curr: str
    status_flipped: bool


class Phase(Enum):
    """Normalized clinical trial phase buckets."""
    P1 = "P1"
    P1_2 = "P1_2"
    P2 = "P2"
    P2_3 = "P2_3"
    P3 = "P3"
    P4 = "P4"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_aact(cls, phase_str: Optional[str]) -> "Phase":
        """Convert AACT phase string to normalized Phase enum."""
        if not phase_str:
            return cls.UNKNOWN
        
        phase_str = phase_str.upper().strip()
        
        # AACT phase mappings
        mappings = {
            "PHASE 1": cls.P1,
            "PHASE1": cls.P1,
            "EARLY PHASE 1": cls.P1,
            "PHASE 1/PHASE 2": cls.P1_2,
            "PHASE 1/2": cls.P1_2,
            "PHASE 2": cls.P2,
            "PHASE2": cls.P2,
            "PHASE 2/PHASE 3": cls.P2_3,
            "PHASE 2/3": cls.P2_3,
            "PHASE 3": cls.P3,
            "PHASE3": cls.P3,
            "PHASE 4": cls.P4,
            "PHASE4": cls.P4,
            "N/A": cls.UNKNOWN,
            "NOT APPLICABLE": cls.UNKNOWN,
        }
        
        return mappings.get(phase_str, cls.UNKNOWN)


class TrialStatus(Enum):
    """Normalized clinical trial status."""
    ACTIVE = "active"
    RECRUITING = "recruiting"
    ENROLLING = "enrolling_by_invitation"
    COMPLETED = "completed"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"
    WITHDRAWN = "withdrawn"
    NOT_YET_RECRUITING = "not_yet_recruiting"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_aact(cls, status_str: Optional[str]) -> "TrialStatus":
        """Convert AACT overall_status string to normalized TrialStatus enum."""
        if not status_str:
            return cls.UNKNOWN
        
        status_str = status_str.upper().strip()
        
        mappings = {
            "ACTIVE, NOT RECRUITING": cls.ACTIVE,
            "RECRUITING": cls.RECRUITING,
            "ENROLLING BY INVITATION": cls.ENROLLING,
            "COMPLETED": cls.COMPLETED,
            "SUSPENDED": cls.SUSPENDED,
            "TERMINATED": cls.TERMINATED,
            "WITHDRAWN": cls.WITHDRAWN,
            "NOT YET RECRUITING": cls.NOT_YET_RECRUITING,
            "APPROVED FOR MARKETING": cls.COMPLETED,
            "AVAILABLE": cls.ACTIVE,
            "NO LONGER AVAILABLE": cls.TERMINATED,
            "TEMPORARILY NOT AVAILABLE": cls.SUSPENDED,
            "WITHHELD": cls.WITHDRAWN,
            "UNKNOWN STATUS": cls.UNKNOWN,
        }
        
        return mappings.get(status_str, cls.UNKNOWN)
    
    def is_terminal(self) -> bool:
        """Check if status is terminal (completed, terminated, or withdrawn)."""
        return self in (self.COMPLETED, self.TERMINATED, self.WITHDRAWN)
    
    def is_active(self) -> bool:
        """Check if status indicates an active/ongoing trial."""
        return self in (self.ACTIVE, self.RECRUITING, self.ENROLLING, self.NOT_YET_RECRUITING)


class PCDType(Enum):
    """Primary completion date type."""
    ACTUAL = "actual"
    ANTICIPATED = "anticipated"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_aact(cls, type_str: Optional[str]) -> "PCDType":
        """Convert AACT primary_completion_date_type to normalized PCDType."""
        if not type_str:
            return cls.UNKNOWN
        
        type_str = type_str.upper().strip()
        
        if "ACTUAL" in type_str:
            return cls.ACTUAL
        elif "ANTICIPATED" in type_str or "ESTIMATED" in type_str:
            return cls.ANTICIPATED
        
        return cls.UNKNOWN


@dataclass(frozen=True, order=True)
class TrialRow:
    """
    Canonical trial data structure.
    
    Frozen and ordered for deterministic hashing and sorting.
    Sort order is by nct_id (ascending) for stable output.
    
    Flags track data quality and processing state:
        - pcd_missing: Primary completion date not present
        - pcd_type_missing: PCD type not present or unknown
        - diffs_disabled: Diff computation was disabled via config
        - diffs_unavailable_insufficient_snapshots: Not enough snapshots for diffs
    """
    nct_id: str
    phase: Phase
    overall_status: TrialStatus
    primary_completion_date: Optional[date]
    primary_completion_date_type: PCDType
    last_update_posted_date: Optional[date]
    lead_sponsor: str
    study_type: str = "Interventional"
    
    # Derived fields set during provider processing
    pcd_pushes_18m: int = 0
    status_flips_18m: int = 0
    
    # Flags for data quality and processing state (frozen tuple for hashability)
    flags: tuple[str, ...] = ()
    
    def to_dict(self) -> TrialRowDict:
        """Convert to dictionary for JSON serialization."""
        return {
            "nct_id": self.nct_id,
            "phase": self.phase.value,
            "overall_status": self.overall_status.value,
            "primary_completion_date": self.primary_completion_date.isoformat() if self.primary_completion_date else None,
            "primary_completion_date_type": self.primary_completion_date_type.value,
            "last_update_posted_date": self.last_update_posted_date.isoformat() if self.last_update_posted_date else None,
            "lead_sponsor": self.lead_sponsor,
            "study_type": self.study_type,
            "pcd_pushes_18m": self.pcd_pushes_18m,
            "status_flips_18m": self.status_flips_18m,
            "flags": list(self.flags),  # Convert tuple to list for JSON
        }

    @classmethod
    def from_dict(cls, data: TrialRowDict) -> "TrialRow":
        """Create from dictionary."""
        return cls(
            nct_id=data["nct_id"],
            phase=Phase(data["phase"]) if isinstance(data["phase"], str) else data["phase"],
            overall_status=TrialStatus(data["overall_status"]) if isinstance(data["overall_status"], str) else data["overall_status"],
            primary_completion_date=date.fromisoformat(data["primary_completion_date"]) if data.get("primary_completion_date") else None,
            primary_completion_date_type=PCDType(data["primary_completion_date_type"]) if isinstance(data["primary_completion_date_type"], str) else data["primary_completion_date_type"],
            last_update_posted_date=date.fromisoformat(data["last_update_posted_date"]) if data.get("last_update_posted_date") else None,
            lead_sponsor=data["lead_sponsor"],
            study_type=data.get("study_type", "Interventional"),
            pcd_pushes_18m=data.get("pcd_pushes_18m", 0),
            status_flips_18m=data.get("status_flips_18m", 0),
            flags=tuple(data.get("flags", [])),  # Convert list back to tuple
        )


@dataclass(frozen=True)
class TrialDiff:
    """
    Represents changes between two snapshots for a single trial.
    Used to compute pcd_pushes and status_flips.
    """
    nct_id: str
    snapshot_date_prev: date
    snapshot_date_curr: date
    
    # PCD changes
    pcd_prev: Optional[date]
    pcd_curr: Optional[date]
    pcd_type_prev: PCDType
    pcd_type_curr: PCDType
    pcd_pushed: bool  # True if PCD moved later
    
    # Status changes
    status_prev: TrialStatus
    status_curr: TrialStatus
    status_flipped: bool  # True if status changed meaningfully
    
    def to_dict(self) -> TrialDiffDict:
        """Convert to dictionary for serialization."""
        return {
            "nct_id": self.nct_id,
            "snapshot_date_prev": self.snapshot_date_prev.isoformat(),
            "snapshot_date_curr": self.snapshot_date_curr.isoformat(),
            "pcd_prev": self.pcd_prev.isoformat() if self.pcd_prev else None,
            "pcd_curr": self.pcd_curr.isoformat() if self.pcd_curr else None,
            "pcd_type_prev": self.pcd_type_prev.value,
            "pcd_type_curr": self.pcd_type_curr.value,
            "pcd_pushed": self.pcd_pushed,
            "status_prev": self.status_prev.value,
            "status_curr": self.status_curr.value,
            "status_flipped": self.status_flipped,
        }


@dataclass
class ProviderResult:
    """
    Result from a clinical trials provider query.
    Includes both data and metadata for provenance tracking.
    """
    trials_by_ticker: dict[str, list[TrialRow]]
    snapshot_date_used: date
    snapshots_root: str
    provider_name: str
    pit_cutoff_applied: date
    
    # Coverage statistics
    tickers_total: int = 0
    tickers_with_trials: int = 0
    trials_total: int = 0
    
    # Diff computation status (for provenance)
    compute_diffs_enabled: bool = False
    compute_diffs_available: bool = False  # True only if enabled AND sufficient snapshots
    snapshots_available_count: int = 0  # How many snapshots were available for diffs
    
    def compute_coverage(self) -> None:
        """Compute coverage statistics from trials_by_ticker."""
        self.tickers_total = len(self.trials_by_ticker)
        self.tickers_with_trials = sum(1 for trials in self.trials_by_ticker.values() if trials)
        self.trials_total = sum(len(trials) for trials in self.trials_by_ticker.values())
    
    @property
    def coverage_rate(self) -> Decimal:
        """Return coverage rate as Decimal for deterministic output."""
        if self.tickers_total == 0:
            return Decimal("0")
        return Decimal(self.tickers_with_trials) / Decimal(self.tickers_total)


class ClinicalTrialsProvider(Protocol):
    """
    Protocol for clinical trials data providers.
    
    All providers must implement get_trials_as_of() which returns
    PIT-safe trial data. The provider owns the PIT boundary enforcement,
    so downstream modules can remain pure and deterministic.
    """
    
    def get_trials_as_of(
        self,
        as_of_date: date,
        pit_cutoff: date,
        tickers: list[str],
        trial_mapping: dict[str, list[str]],  # ticker -> list of nct_ids
    ) -> ProviderResult:
        """
        Return trials per ticker, filtered to PIT-safe snapshot <= pit_cutoff.
        
        Args:
            as_of_date: The date we're generating the snapshot for
            pit_cutoff: Latest allowed data date (typically as_of_date - lag_days)
            tickers: List of tickers to get trials for
            trial_mapping: Mapping from ticker to NCT IDs (from trial_mapping.csv)
        
        Returns:
            ProviderResult with trials_by_ticker and metadata
        """
        ...
    
    @property
    def provider_name(self) -> str:
        """Return the provider name for provenance tracking."""
        ...
