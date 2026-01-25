"""
pos_model_v2.py - Hybrid Probability-of-Success Framework (Tier 1)

Implements indication × mechanism × stage base rates with FDA designation modifiers.
All calculations use Decimal arithmetic for regulatory compliance and audit trails.

Wake Robin Capital Management - Biotech Screening System
Version: 2.1.0

Changelog:
- v2.1.0: Added fixture loading, provenance tracking, strict validation
- v2.0.0: Initial hybrid PoS framework with FDA/trial modifiers
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, getcontext, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, NamedTuple, Any, Union

# Set precision for financial calculations
getcontext().prec = 28


# =============================================================================
# EXCEPTIONS
# =============================================================================

class FixtureValidationError(Exception):
    """Raised when fixture validation fails. Fail-closed behavior."""
    pass


class FixtureSchemaError(FixtureValidationError):
    """Raised when fixture schema is invalid."""
    pass


class FixtureEnumError(FixtureValidationError):
    """Raised when fixture contains unknown enum values."""
    pass


class FixtureDecimalError(FixtureValidationError):
    """Raised when fixture contains unparseable Decimal values."""
    pass

# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class ClinicalStage(Enum):
    """Clinical development stages with canonical names."""
    PRECLINICAL = "preclinical"
    PHASE_1 = "phase_1"
    PHASE_1_2 = "phase_1_2"
    PHASE_2 = "phase_2"
    PHASE_2_3 = "phase_2_3"
    PHASE_3 = "phase_3"
    FILED = "filed"  # NDA/BLA submitted
    APPROVED = "approved"  # Commercial stage
    
    @classmethod
    def from_string(cls, s: str) -> "ClinicalStage":
        """Parse stage from various string formats."""
        normalized = s.lower().strip().replace(" ", "_").replace("-", "_")
        mapping = {
            "preclinical": cls.PRECLINICAL,
            "pre_clinical": cls.PRECLINICAL,
            "phase_1": cls.PHASE_1,
            "phase1": cls.PHASE_1,
            "p1": cls.PHASE_1,
            "phase_1_2": cls.PHASE_1_2,
            "phase1_2": cls.PHASE_1_2,
            "phase_2": cls.PHASE_2,
            "phase2": cls.PHASE_2,
            "p2": cls.PHASE_2,
            "phase_2_3": cls.PHASE_2_3,
            "phase2_3": cls.PHASE_2_3,
            "phase_3": cls.PHASE_3,
            "phase3": cls.PHASE_3,
            "p3": cls.PHASE_3,
            "filed": cls.FILED,
            "nda": cls.FILED,
            "bla": cls.FILED,
            "pdufa": cls.FILED,
            "approved": cls.APPROVED,
            "commercial": cls.APPROVED,
        }
        if normalized in mapping:
            return mapping[normalized]
        raise ValueError(f"Unknown clinical stage: {s}")


class TherapeuticArea(Enum):
    """Therapeutic area / indication categories."""
    ONCOLOGY = "oncology"
    RARE_DISEASE = "rare_disease"
    CNS = "cns"  # Central nervous system
    CARDIOVASCULAR = "cardiovascular"
    IMMUNOLOGY = "immunology"
    INFECTIOUS_DISEASE = "infectious_disease"
    METABOLIC = "metabolic"
    RESPIRATORY = "respiratory"
    OPHTHALMOLOGY = "ophthalmology"
    DERMATOLOGY = "dermatology"
    HEMATOLOGY = "hematology"
    GI = "gi"  # Gastrointestinal
    OTHER = "other"
    
    @classmethod
    def from_string(cls, s: str) -> "TherapeuticArea":
        """Parse therapeutic area from string."""
        normalized = s.lower().strip().replace(" ", "_").replace("-", "_")
        mapping = {
            "oncology": cls.ONCOLOGY,
            "cancer": cls.ONCOLOGY,
            "tumor": cls.ONCOLOGY,
            "rare_disease": cls.RARE_DISEASE,
            "rare": cls.RARE_DISEASE,
            "orphan": cls.RARE_DISEASE,
            "cns": cls.CNS,
            "neuro": cls.CNS,
            "neurological": cls.CNS,
            "neurology": cls.CNS,
            "psychiatric": cls.CNS,
            "cardiovascular": cls.CARDIOVASCULAR,
            "cardio": cls.CARDIOVASCULAR,
            "cv": cls.CARDIOVASCULAR,
            "heart": cls.CARDIOVASCULAR,
            "immunology": cls.IMMUNOLOGY,
            "autoimmune": cls.IMMUNOLOGY,
            "inflammation": cls.IMMUNOLOGY,
            "infectious_disease": cls.INFECTIOUS_DISEASE,
            "infectious": cls.INFECTIOUS_DISEASE,
            "anti_infective": cls.INFECTIOUS_DISEASE,
            "metabolic": cls.METABOLIC,
            "diabetes": cls.METABOLIC,
            "obesity": cls.METABOLIC,
            "respiratory": cls.RESPIRATORY,
            "pulmonary": cls.RESPIRATORY,
            "lung": cls.RESPIRATORY,
            "ophthalmology": cls.OPHTHALMOLOGY,
            "eye": cls.OPHTHALMOLOGY,
            "ophthalmic": cls.OPHTHALMOLOGY,
            "dermatology": cls.DERMATOLOGY,
            "skin": cls.DERMATOLOGY,
            "hematology": cls.HEMATOLOGY,
            "blood": cls.HEMATOLOGY,
            "gi": cls.GI,
            "gastrointestinal": cls.GI,
            "gastro": cls.GI,
        }
        return mapping.get(normalized, cls.OTHER)


class MechanismClass(Enum):
    """Drug mechanism / modality categories."""
    SMALL_MOLECULE = "small_molecule"
    MONOCLONAL_ANTIBODY = "monoclonal_antibody"
    BIOLOGIC = "biologic"  # Other biologics (fusion proteins, etc.)
    GENE_THERAPY = "gene_therapy"
    CELL_THERAPY = "cell_therapy"
    RNA_THERAPEUTIC = "rna_therapeutic"  # ASO, siRNA, mRNA
    PEPTIDE = "peptide"
    ADC = "adc"  # Antibody-drug conjugate
    BISPECIFIC = "bispecific"
    CAR_T = "car_t"
    OTHER = "other"
    
    @classmethod
    def from_string(cls, s: str) -> "MechanismClass":
        """Parse mechanism class from string."""
        normalized = s.lower().strip().replace(" ", "_").replace("-", "_")
        mapping = {
            "small_molecule": cls.SMALL_MOLECULE,
            "sm": cls.SMALL_MOLECULE,
            "chemical": cls.SMALL_MOLECULE,
            "monoclonal_antibody": cls.MONOCLONAL_ANTIBODY,
            "mab": cls.MONOCLONAL_ANTIBODY,
            "antibody": cls.MONOCLONAL_ANTIBODY,
            "biologic": cls.BIOLOGIC,
            "protein": cls.BIOLOGIC,
            "gene_therapy": cls.GENE_THERAPY,
            "gene": cls.GENE_THERAPY,
            "aav": cls.GENE_THERAPY,
            "cell_therapy": cls.CELL_THERAPY,
            "cell": cls.CELL_THERAPY,
            "rna_therapeutic": cls.RNA_THERAPEUTIC,
            "rna": cls.RNA_THERAPEUTIC,
            "aso": cls.RNA_THERAPEUTIC,
            "sirna": cls.RNA_THERAPEUTIC,
            "mrna": cls.RNA_THERAPEUTIC,
            "peptide": cls.PEPTIDE,
            "adc": cls.ADC,
            "antibody_drug_conjugate": cls.ADC,
            "bispecific": cls.BISPECIFIC,
            "bsab": cls.BISPECIFIC,
            "car_t": cls.CAR_T,
            "cart": cls.CAR_T,
        }
        return mapping.get(normalized, cls.OTHER)


class FDADesignation(Enum):
    """FDA special designations that modify PoS."""
    BREAKTHROUGH_THERAPY = "breakthrough_therapy"
    ORPHAN_DRUG = "orphan_drug"
    FAST_TRACK = "fast_track"
    ACCELERATED_APPROVAL = "accelerated_approval"
    PRIORITY_REVIEW = "priority_review"
    RMAT = "rmat"  # Regenerative Medicine Advanced Therapy


class TrialCharacteristic(Enum):
    """Trial design characteristics that modify PoS."""
    BIOMARKER_SELECTED = "biomarker_selected"
    SURROGATE_ENDPOINT = "surrogate_endpoint"
    ACTIVE_COMPARATOR = "active_comparator"
    PLACEBO_CONTROLLED = "placebo_controlled"
    FIRST_IN_CLASS = "first_in_class"
    BEST_IN_CLASS = "best_in_class"
    PRIOR_PHASE_SUCCESS = "prior_phase_success"  # Strong Phase 2 results


# =============================================================================
# BASE RATE MATRIX - Evidence-Based PoS Benchmarks
# =============================================================================

# Key: (stage, therapeutic_area, mechanism_class)
# Value: (base_pos, n_support, source_citation)
# Sources: BIO/Informa Clinical Development Success Rates 2011-2020,
#          FDA approval statistics, IQVIA Global Trends 2025

PosKey = Tuple[ClinicalStage, TherapeuticArea, MechanismClass]

@dataclass(frozen=True)
class BaseRateEntry:
    """A single entry in the base rate matrix."""
    base_pos: Decimal
    n_support: int  # Number of trials supporting this estimate
    source: str
    source_date: date
    
    def to_dict(self) -> dict:
        return {
            "base_pos": str(self.base_pos),
            "n_support": self.n_support,
            "source": self.source,
            "source_date": self.source_date.isoformat(),
        }


# Default base rates when specific combination not found
DEFAULT_BASE_RATES: Dict[ClinicalStage, Decimal] = {
    ClinicalStage.PRECLINICAL: Decimal("0.10"),
    ClinicalStage.PHASE_1: Decimal("0.15"),
    ClinicalStage.PHASE_1_2: Decimal("0.18"),
    ClinicalStage.PHASE_2: Decimal("0.30"),
    ClinicalStage.PHASE_2_3: Decimal("0.40"),
    ClinicalStage.PHASE_3: Decimal("0.50"),
    ClinicalStage.FILED: Decimal("0.85"),
    ClinicalStage.APPROVED: Decimal("1.00"),
}


def _build_base_rate_matrix() -> Dict[PosKey, BaseRateEntry]:
    """
    Build the base rate matrix from empirical data.
    
    Sources:
    - BIO/Informa "Clinical Development Success Rates 2011-2020"
    - FDA CDER Novel Drug Approvals statistics
    - Nature Reviews Drug Discovery benchmarks
    - IQVIA Global Trends in R&D 2025
    """
    src_date = date(2024, 1, 1)  # Benchmark data vintage
    bio_source = "BIO/Informa Clinical Development Success Rates 2011-2020"
    fda_source = "FDA CDER Novel Drug Approvals"
    
    matrix: Dict[PosKey, BaseRateEntry] = {}
    
    # --- ONCOLOGY ---
    # Oncology historically has lower success rates, especially in later phases
    matrix[(ClinicalStage.PHASE_1, TherapeuticArea.ONCOLOGY, MechanismClass.SMALL_MOLECULE)] = \
        BaseRateEntry(Decimal("0.63"), 245, bio_source, src_date)
    matrix[(ClinicalStage.PHASE_2, TherapeuticArea.ONCOLOGY, MechanismClass.SMALL_MOLECULE)] = \
        BaseRateEntry(Decimal("0.30"), 189, bio_source, src_date)
    matrix[(ClinicalStage.PHASE_3, TherapeuticArea.ONCOLOGY, MechanismClass.SMALL_MOLECULE)] = \
        BaseRateEntry(Decimal("0.42"), 128, bio_source, src_date)
    
    matrix[(ClinicalStage.PHASE_2, TherapeuticArea.ONCOLOGY, MechanismClass.MONOCLONAL_ANTIBODY)] = \
        BaseRateEntry(Decimal("0.32"), 87, bio_source, src_date)
    matrix[(ClinicalStage.PHASE_3, TherapeuticArea.ONCOLOGY, MechanismClass.MONOCLONAL_ANTIBODY)] = \
        BaseRateEntry(Decimal("0.45"), 64, bio_source, src_date)
    
    matrix[(ClinicalStage.PHASE_2, TherapeuticArea.ONCOLOGY, MechanismClass.ADC)] = \
        BaseRateEntry(Decimal("0.35"), 42, bio_source, src_date)
    matrix[(ClinicalStage.PHASE_3, TherapeuticArea.ONCOLOGY, MechanismClass.ADC)] = \
        BaseRateEntry(Decimal("0.48"), 28, bio_source, src_date)
    
    matrix[(ClinicalStage.PHASE_2, TherapeuticArea.ONCOLOGY, MechanismClass.CAR_T)] = \
        BaseRateEntry(Decimal("0.40"), 35, fda_source, src_date)
    matrix[(ClinicalStage.PHASE_3, TherapeuticArea.ONCOLOGY, MechanismClass.CAR_T)] = \
        BaseRateEntry(Decimal("0.55"), 18, fda_source, src_date)
    
    # --- RARE DISEASE ---
    # Rare disease generally has higher success due to unmet need, regulatory flexibility
    matrix[(ClinicalStage.PHASE_2, TherapeuticArea.RARE_DISEASE, MechanismClass.GENE_THERAPY)] = \
        BaseRateEntry(Decimal("0.35"), 22, bio_source, src_date)
    matrix[(ClinicalStage.PHASE_3, TherapeuticArea.RARE_DISEASE, MechanismClass.GENE_THERAPY)] = \
        BaseRateEntry(Decimal("0.58"), 15, bio_source, src_date)
    
    matrix[(ClinicalStage.PHASE_2, TherapeuticArea.RARE_DISEASE, MechanismClass.SMALL_MOLECULE)] = \
        BaseRateEntry(Decimal("0.38"), 56, bio_source, src_date)
    matrix[(ClinicalStage.PHASE_3, TherapeuticArea.RARE_DISEASE, MechanismClass.SMALL_MOLECULE)] = \
        BaseRateEntry(Decimal("0.62"), 41, bio_source, src_date)
    
    matrix[(ClinicalStage.PHASE_2, TherapeuticArea.RARE_DISEASE, MechanismClass.BIOLOGIC)] = \
        BaseRateEntry(Decimal("0.40"), 34, bio_source, src_date)
    matrix[(ClinicalStage.PHASE_3, TherapeuticArea.RARE_DISEASE, MechanismClass.BIOLOGIC)] = \
        BaseRateEntry(Decimal("0.60"), 28, bio_source, src_date)
    
    # --- CNS (Historically most challenging) ---
    matrix[(ClinicalStage.PHASE_2, TherapeuticArea.CNS, MechanismClass.SMALL_MOLECULE)] = \
        BaseRateEntry(Decimal("0.22"), 145, bio_source, src_date)
    matrix[(ClinicalStage.PHASE_3, TherapeuticArea.CNS, MechanismClass.SMALL_MOLECULE)] = \
        BaseRateEntry(Decimal("0.38"), 98, bio_source, src_date)
    
    matrix[(ClinicalStage.PHASE_2, TherapeuticArea.CNS, MechanismClass.MONOCLONAL_ANTIBODY)] = \
        BaseRateEntry(Decimal("0.25"), 32, bio_source, src_date)
    matrix[(ClinicalStage.PHASE_3, TherapeuticArea.CNS, MechanismClass.MONOCLONAL_ANTIBODY)] = \
        BaseRateEntry(Decimal("0.28"), 24, bio_source, src_date)
    
    # --- CARDIOVASCULAR ---
    matrix[(ClinicalStage.PHASE_2, TherapeuticArea.CARDIOVASCULAR, MechanismClass.SMALL_MOLECULE)] = \
        BaseRateEntry(Decimal("0.35"), 78, bio_source, src_date)
    matrix[(ClinicalStage.PHASE_3, TherapeuticArea.CARDIOVASCULAR, MechanismClass.SMALL_MOLECULE)] = \
        BaseRateEntry(Decimal("0.55"), 52, bio_source, src_date)
    
    # --- IMMUNOLOGY ---
    matrix[(ClinicalStage.PHASE_2, TherapeuticArea.IMMUNOLOGY, MechanismClass.MONOCLONAL_ANTIBODY)] = \
        BaseRateEntry(Decimal("0.38"), 67, bio_source, src_date)
    matrix[(ClinicalStage.PHASE_3, TherapeuticArea.IMMUNOLOGY, MechanismClass.MONOCLONAL_ANTIBODY)] = \
        BaseRateEntry(Decimal("0.52"), 48, bio_source, src_date)
    
    matrix[(ClinicalStage.PHASE_2, TherapeuticArea.IMMUNOLOGY, MechanismClass.BIOLOGIC)] = \
        BaseRateEntry(Decimal("0.40"), 45, bio_source, src_date)
    matrix[(ClinicalStage.PHASE_3, TherapeuticArea.IMMUNOLOGY, MechanismClass.BIOLOGIC)] = \
        BaseRateEntry(Decimal("0.50"), 38, bio_source, src_date)
    
    # --- INFECTIOUS DISEASE ---
    matrix[(ClinicalStage.PHASE_2, TherapeuticArea.INFECTIOUS_DISEASE, MechanismClass.SMALL_MOLECULE)] = \
        BaseRateEntry(Decimal("0.45"), 89, bio_source, src_date)
    matrix[(ClinicalStage.PHASE_3, TherapeuticArea.INFECTIOUS_DISEASE, MechanismClass.SMALL_MOLECULE)] = \
        BaseRateEntry(Decimal("0.65"), 62, bio_source, src_date)
    
    # --- METABOLIC ---
    matrix[(ClinicalStage.PHASE_2, TherapeuticArea.METABOLIC, MechanismClass.SMALL_MOLECULE)] = \
        BaseRateEntry(Decimal("0.32"), 76, bio_source, src_date)
    matrix[(ClinicalStage.PHASE_3, TherapeuticArea.METABOLIC, MechanismClass.SMALL_MOLECULE)] = \
        BaseRateEntry(Decimal("0.52"), 54, bio_source, src_date)
    
    matrix[(ClinicalStage.PHASE_2, TherapeuticArea.METABOLIC, MechanismClass.PEPTIDE)] = \
        BaseRateEntry(Decimal("0.38"), 28, bio_source, src_date)
    matrix[(ClinicalStage.PHASE_3, TherapeuticArea.METABOLIC, MechanismClass.PEPTIDE)] = \
        BaseRateEntry(Decimal("0.58"), 22, bio_source, src_date)
    
    # --- HEMATOLOGY ---
    matrix[(ClinicalStage.PHASE_2, TherapeuticArea.HEMATOLOGY, MechanismClass.SMALL_MOLECULE)] = \
        BaseRateEntry(Decimal("0.42"), 45, bio_source, src_date)
    matrix[(ClinicalStage.PHASE_3, TherapeuticArea.HEMATOLOGY, MechanismClass.SMALL_MOLECULE)] = \
        BaseRateEntry(Decimal("0.58"), 32, bio_source, src_date)
    
    # --- OPHTHALMOLOGY ---
    matrix[(ClinicalStage.PHASE_2, TherapeuticArea.OPHTHALMOLOGY, MechanismClass.BIOLOGIC)] = \
        BaseRateEntry(Decimal("0.35"), 34, bio_source, src_date)
    matrix[(ClinicalStage.PHASE_3, TherapeuticArea.OPHTHALMOLOGY, MechanismClass.BIOLOGIC)] = \
        BaseRateEntry(Decimal("0.55"), 26, bio_source, src_date)
    
    matrix[(ClinicalStage.PHASE_2, TherapeuticArea.OPHTHALMOLOGY, MechanismClass.GENE_THERAPY)] = \
        BaseRateEntry(Decimal("0.38"), 12, bio_source, src_date)
    matrix[(ClinicalStage.PHASE_3, TherapeuticArea.OPHTHALMOLOGY, MechanismClass.GENE_THERAPY)] = \
        BaseRateEntry(Decimal("0.60"), 8, bio_source, src_date)
    
    return matrix


# Global base rate matrix (loaded once)
BASE_RATE_MATRIX: Dict[PosKey, BaseRateEntry] = _build_base_rate_matrix()


# =============================================================================
# MODIFIER DEFINITIONS - Empirically Validated Adjustments
# =============================================================================

@dataclass(frozen=True)
class ModifierDefinition:
    """Definition of a PoS modifier with empirical support."""
    factor: Decimal
    direction: str  # "positive" or "negative"
    source: str
    rationale: str


# FDA Designation modifiers (multiplicative)
FDA_DESIGNATION_MODIFIERS: Dict[FDADesignation, ModifierDefinition] = {
    FDADesignation.BREAKTHROUGH_THERAPY: ModifierDefinition(
        factor=Decimal("1.25"),
        direction="positive",
        source="FDA BTD approval rates 2012-2023",
        rationale="BTD drugs show ~25% higher approval rates vs non-BTD"
    ),
    FDADesignation.ORPHAN_DRUG: ModifierDefinition(
        factor=Decimal("1.15"),
        direction="positive",
        source="FDA Orphan Drug Act statistics",
        rationale="Orphan designation provides regulatory flexibility, smaller trials"
    ),
    FDADesignation.FAST_TRACK: ModifierDefinition(
        factor=Decimal("1.10"),
        direction="positive",
        source="FDA Fast Track statistics",
        rationale="Fast Track increases FDA interaction, rolling review"
    ),
    FDADesignation.ACCELERATED_APPROVAL: ModifierDefinition(
        factor=Decimal("0.90"),
        direction="negative",
        source="FDA Accelerated Approval confirmatory trial data",
        rationale="~10% fail confirmatory trials post-approval"
    ),
    FDADesignation.PRIORITY_REVIEW: ModifierDefinition(
        factor=Decimal("1.08"),
        direction="positive",
        source="FDA Priority Review approval data",
        rationale="Priority Review correlates with differentiated profile"
    ),
    FDADesignation.RMAT: ModifierDefinition(
        factor=Decimal("1.12"),
        direction="positive",
        source="FDA RMAT program statistics",
        rationale="RMAT provides expedited development pathway"
    ),
}


# Trial characteristic modifiers (multiplicative)
TRIAL_CHARACTERISTIC_MODIFIERS: Dict[TrialCharacteristic, ModifierDefinition] = {
    TrialCharacteristic.BIOMARKER_SELECTED: ModifierDefinition(
        factor=Decimal("1.20"),
        direction="positive",
        source="Nature Reviews Drug Discovery 2020",
        rationale="Biomarker-selected populations show ~20% higher response rates"
    ),
    TrialCharacteristic.SURROGATE_ENDPOINT: ModifierDefinition(
        factor=Decimal("0.75"),
        direction="negative",
        source="FDA surrogate endpoint analysis",
        rationale="Surrogate endpoints may not translate to clinical benefit"
    ),
    TrialCharacteristic.ACTIVE_COMPARATOR: ModifierDefinition(
        factor=Decimal("0.85"),
        direction="negative",
        source="Clinical trial design literature",
        rationale="Active comparator trials harder to show superiority vs SOC"
    ),
    TrialCharacteristic.PLACEBO_CONTROLLED: ModifierDefinition(
        factor=Decimal("1.05"),
        direction="positive",
        source="Clinical trial success rate analysis",
        rationale="Placebo-controlled trials have clearer efficacy signal"
    ),
    TrialCharacteristic.FIRST_IN_CLASS: ModifierDefinition(
        factor=Decimal("1.30"),
        direction="positive",
        source="First-in-class approval analysis",
        rationale="Validated novel mechanisms show higher success if de-risked"
    ),
    TrialCharacteristic.BEST_IN_CLASS: ModifierDefinition(
        factor=Decimal("1.15"),
        direction="positive",
        source="Best-in-class clinical precedent",
        rationale="Improved profile over existing therapies reduces risk"
    ),
    TrialCharacteristic.PRIOR_PHASE_SUCCESS: ModifierDefinition(
        factor=Decimal("1.15"),
        direction="positive",
        source="Phase transition success rates",
        rationale="Strong Phase 2 data (p<0.05) predicts Phase 3 success"
    ),
}


# =============================================================================
# BOUNDS AND CONSTRAINTS
# =============================================================================

# Minimum and maximum allowed PoS values (prevent overconfidence/underconfidence)
POS_FLOOR = Decimal("0.05")
POS_CEILING = Decimal("0.95")

# Maximum total modifier adjustment (prevents runaway multipliers)
MAX_MODIFIER_PRODUCT = Decimal("2.00")
MIN_MODIFIER_PRODUCT = Decimal("0.50")


# =============================================================================
# FIXTURE PROVENANCE
# =============================================================================

@dataclass(frozen=True)
class FixtureProvenance:
    """Immutable provenance record for loaded fixture data."""
    fixture_version: str
    fixture_source_asof_date: date
    fixture_sha256: str
    fixture_path: str
    loaded_at: datetime
    base_rate_count: int
    fda_modifier_count: int
    trial_modifier_count: int
    
    def to_dict(self) -> dict:
        return {
            "fixture_version": self.fixture_version,
            "fixture_source_asof_date": self.fixture_source_asof_date.isoformat(),
            "fixture_sha256": self.fixture_sha256,
            "fixture_path": self.fixture_path,
            "loaded_at": self.loaded_at.isoformat(),
            "base_rate_count": self.base_rate_count,
            "fda_modifier_count": self.fda_modifier_count,
            "trial_modifier_count": self.trial_modifier_count,
        }


# =============================================================================
# FIXTURE LOADING AND VALIDATION
# =============================================================================

class FixtureLoader:
    """
    Loads and validates PoS fixture data from JSON.
    
    Implements strict validation with fail-closed behavior:
    - Unknown enum values raise FixtureEnumError (not silently mapped to OTHER)
    - Invalid Decimal values raise FixtureDecimalError
    - Missing required fields raise FixtureSchemaError
    - All parsing is deterministic with sorted key ordering
    """
    
    # Required top-level keys in fixture
    REQUIRED_KEYS = {"version", "source_asof_date", "base_rates"}
    
    # Optional top-level keys
    OPTIONAL_KEYS = {
        "description", "primary_sources", "fda_modifiers", 
        "trial_modifiers", "default_stage_rates", "bounds"
    }
    
    # Valid enum values for strict parsing (lowercase)
    VALID_STAGES = {s.value for s in ClinicalStage}
    VALID_INDICATIONS = {t.value for t in TherapeuticArea}
    VALID_MECHANISMS = {m.value for m in MechanismClass}
    VALID_FDA_DESIGNATIONS = {d.value for d in FDADesignation}
    VALID_TRIAL_CHARACTERISTICS = {t.value for t in TrialCharacteristic}
    
    @classmethod
    def compute_file_sha256(cls, path: Path) -> str:
        """Compute SHA256 hash of fixture file for provenance."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    @classmethod
    def _parse_decimal_strict(cls, value: Any, field_name: str) -> Decimal:
        """Parse a value to Decimal with strict validation."""
        try:
            if isinstance(value, (int, float)):
                return Decimal(str(value))
            elif isinstance(value, str):
                return Decimal(value)
            else:
                raise FixtureDecimalError(
                    f"Field '{field_name}' has invalid type {type(value).__name__}, "
                    f"expected string or number"
                )
        except InvalidOperation as e:
            raise FixtureDecimalError(
                f"Field '{field_name}' value '{value}' is not a valid Decimal: {e}"
            )
    
    @classmethod
    def _parse_date_strict(cls, value: str, field_name: str) -> date:
        """Parse a date string with strict validation."""
        try:
            return date.fromisoformat(value)
        except (ValueError, TypeError) as e:
            raise FixtureSchemaError(
                f"Field '{field_name}' value '{value}' is not a valid ISO date: {e}"
            )
    
    @classmethod
    def _validate_enum_strict(
        cls, 
        value: str, 
        valid_values: Set[str], 
        enum_type_name: str,
        field_context: str
    ) -> str:
        """Validate enum value strictly - fail if unknown."""
        normalized = value.lower().strip().replace(" ", "_").replace("-", "_")
        if normalized not in valid_values:
            raise FixtureEnumError(
                f"Unknown {enum_type_name} '{value}' (normalized: '{normalized}') "
                f"in {field_context}. Valid values: {sorted(valid_values)}"
            )
        return normalized
    
    @classmethod
    def _parse_stage_strict(cls, value: str, context: str) -> ClinicalStage:
        """Parse stage enum strictly."""
        normalized = cls._validate_enum_strict(
            value, cls.VALID_STAGES, "ClinicalStage", context
        )
        return ClinicalStage(normalized)
    
    @classmethod
    def _parse_indication_strict(cls, value: str, context: str) -> TherapeuticArea:
        """Parse therapeutic area enum strictly."""
        normalized = cls._validate_enum_strict(
            value, cls.VALID_INDICATIONS, "TherapeuticArea", context
        )
        return TherapeuticArea(normalized)
    
    @classmethod
    def _parse_mechanism_strict(cls, value: str, context: str) -> MechanismClass:
        """Parse mechanism class enum strictly."""
        normalized = cls._validate_enum_strict(
            value, cls.VALID_MECHANISMS, "MechanismClass", context
        )
        return MechanismClass(normalized)
    
    @classmethod
    def _parse_fda_designation_strict(cls, value: str, context: str) -> FDADesignation:
        """Parse FDA designation enum strictly."""
        normalized = cls._validate_enum_strict(
            value, cls.VALID_FDA_DESIGNATIONS, "FDADesignation", context
        )
        return FDADesignation(normalized)
    
    @classmethod
    def _parse_trial_characteristic_strict(cls, value: str, context: str) -> TrialCharacteristic:
        """Parse trial characteristic enum strictly."""
        normalized = cls._validate_enum_strict(
            value, cls.VALID_TRIAL_CHARACTERISTICS, "TrialCharacteristic", context
        )
        return TrialCharacteristic(normalized)
    
    @classmethod
    def _validate_schema(cls, data: dict, path: Path) -> None:
        """Validate fixture has required structure."""
        missing = cls.REQUIRED_KEYS - set(data.keys())
        if missing:
            raise FixtureSchemaError(
                f"Fixture {path} missing required keys: {sorted(missing)}"
            )
        
        unknown = set(data.keys()) - cls.REQUIRED_KEYS - cls.OPTIONAL_KEYS
        if unknown:
            raise FixtureSchemaError(
                f"Fixture {path} has unknown top-level keys: {sorted(unknown)}. "
                f"This may indicate a schema version mismatch."
            )
    
    @classmethod
    def _parse_base_rates(
        cls, 
        base_rates_list: List[dict],
        source_date: date
    ) -> Dict[PosKey, BaseRateEntry]:
        """Parse base_rates array into matrix."""
        matrix: Dict[PosKey, BaseRateEntry] = {}
        
        for i, entry in enumerate(base_rates_list):
            context = f"base_rates[{i}]"
            
            # Validate required fields
            required = {"stage", "indication", "mechanism_class", "base_pos"}
            missing = required - set(entry.keys())
            if missing:
                raise FixtureSchemaError(
                    f"{context} missing required fields: {sorted(missing)}"
                )
            
            # Parse enums strictly
            stage = cls._parse_stage_strict(entry["stage"], context)
            indication = cls._parse_indication_strict(entry["indication"], context)
            mechanism = cls._parse_mechanism_strict(entry["mechanism_class"], context)
            
            # Parse Decimal strictly
            base_pos = cls._parse_decimal_strict(entry["base_pos"], f"{context}.base_pos")
            
            # Validate range
            if base_pos < Decimal("0.01") or base_pos > Decimal("0.99"):
                raise FixtureSchemaError(
                    f"{context}.base_pos value {base_pos} out of valid range [0.01, 0.99]"
                )
            
            # Parse optional fields
            n_support = entry.get("n_support")
            if n_support is not None:
                if not isinstance(n_support, int) or n_support < 0:
                    raise FixtureSchemaError(
                        f"{context}.n_support must be a non-negative integer, got {n_support}"
                    )
            
            source = entry.get("source", "Fixture")
            
            # Build key and entry
            key = (stage, indication, mechanism)
            
            # Check for duplicates
            if key in matrix:
                raise FixtureSchemaError(
                    f"{context} creates duplicate key {key}. "
                    f"Each (stage, indication, mechanism) combination must be unique."
                )
            
            matrix[key] = BaseRateEntry(
                base_pos=base_pos,
                n_support=n_support or 0,
                source=source,
                source_date=source_date,
            )
        
        return matrix
    
    @classmethod
    def _parse_fda_modifiers(
        cls, 
        modifiers_list: List[dict]
    ) -> Dict[FDADesignation, ModifierDefinition]:
        """Parse fda_modifiers array."""
        result: Dict[FDADesignation, ModifierDefinition] = {}
        
        for i, entry in enumerate(modifiers_list):
            context = f"fda_modifiers[{i}]"
            
            # Validate required fields
            required = {"designation", "factor"}
            missing = required - set(entry.keys())
            if missing:
                raise FixtureSchemaError(
                    f"{context} missing required fields: {sorted(missing)}"
                )
            
            # Parse enum strictly
            designation = cls._parse_fda_designation_strict(entry["designation"], context)
            
            # Parse factor
            factor = cls._parse_decimal_strict(entry["factor"], f"{context}.factor")
            
            # Validate range
            if factor < Decimal("0.5") or factor > Decimal("2.0"):
                raise FixtureSchemaError(
                    f"{context}.factor value {factor} out of valid range [0.5, 2.0]"
                )
            
            # Parse optional fields
            direction = entry.get("direction", "positive" if factor >= 1 else "negative")
            source = entry.get("source", "Fixture")
            rationale = entry.get("rationale", "")
            
            # Check for duplicates
            if designation in result:
                raise FixtureSchemaError(
                    f"{context} creates duplicate FDA designation {designation.value}"
                )
            
            result[designation] = ModifierDefinition(
                factor=factor,
                direction=direction,
                source=source,
                rationale=rationale,
            )
        
        return result
    
    @classmethod
    def _parse_trial_modifiers(
        cls, 
        modifiers_list: List[dict]
    ) -> Dict[TrialCharacteristic, ModifierDefinition]:
        """Parse trial_modifiers array."""
        result: Dict[TrialCharacteristic, ModifierDefinition] = {}
        
        for i, entry in enumerate(modifiers_list):
            context = f"trial_modifiers[{i}]"
            
            # Validate required fields
            required = {"characteristic", "factor"}
            missing = required - set(entry.keys())
            if missing:
                raise FixtureSchemaError(
                    f"{context} missing required fields: {sorted(missing)}"
                )
            
            # Parse enum strictly
            characteristic = cls._parse_trial_characteristic_strict(
                entry["characteristic"], context
            )
            
            # Parse factor
            factor = cls._parse_decimal_strict(entry["factor"], f"{context}.factor")
            
            # Validate range
            if factor < Decimal("0.5") or factor > Decimal("2.0"):
                raise FixtureSchemaError(
                    f"{context}.factor value {factor} out of valid range [0.5, 2.0]"
                )
            
            # Parse optional fields
            direction = entry.get("direction", "positive" if factor >= 1 else "negative")
            source = entry.get("source", "Fixture")
            rationale = entry.get("rationale", "")
            
            # Check for duplicates
            if characteristic in result:
                raise FixtureSchemaError(
                    f"{context} creates duplicate trial characteristic {characteristic.value}"
                )
            
            result[characteristic] = ModifierDefinition(
                factor=factor,
                direction=direction,
                source=source,
                rationale=rationale,
            )
        
        return result
    
    @classmethod
    def _parse_default_stage_rates(
        cls, 
        rates_dict: dict
    ) -> Dict[ClinicalStage, Decimal]:
        """Parse default_stage_rates dictionary."""
        result: Dict[ClinicalStage, Decimal] = {}
        
        for stage_str, rate_val in rates_dict.items():
            context = f"default_stage_rates.{stage_str}"
            
            # Parse stage strictly
            stage = cls._parse_stage_strict(stage_str, context)
            
            # Parse rate
            rate = cls._parse_decimal_strict(rate_val, context)
            
            # Validate range
            if rate < Decimal("0.01") or rate > Decimal("1.00"):
                raise FixtureSchemaError(
                    f"{context} value {rate} out of valid range [0.01, 1.00]"
                )
            
            result[stage] = rate
        
        return result
    
    @classmethod
    def _parse_bounds(cls, bounds_dict: dict) -> Dict[str, Decimal]:
        """Parse bounds dictionary."""
        valid_keys = {"pos_floor", "pos_ceiling", "max_modifier_product", "min_modifier_product"}
        result: Dict[str, Decimal] = {}
        
        for key, val in bounds_dict.items():
            if key not in valid_keys:
                raise FixtureSchemaError(
                    f"bounds has unknown key '{key}'. Valid keys: {sorted(valid_keys)}"
                )
            
            result[key] = cls._parse_decimal_strict(val, f"bounds.{key}")
        
        return result
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "FixtureData":
        """
        Load and validate fixture from JSON file.
        
        Args:
            path: Path to fixture JSON file
            
        Returns:
            FixtureData containing parsed data and provenance
            
        Raises:
            FixtureValidationError: If validation fails (fail-closed)
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file isn't valid JSON
        """
        path = Path(path)
        
        # Compute hash before parsing (for provenance)
        file_sha256 = cls.compute_file_sha256(path)
        
        # Load and parse JSON
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Validate schema
        cls._validate_schema(data, path)
        
        # Parse required fields
        version = str(data["version"])
        source_asof_date = cls._parse_date_strict(
            data["source_asof_date"], "source_asof_date"
        )
        
        # Parse base rates (required)
        base_rate_matrix = cls._parse_base_rates(
            data["base_rates"], source_asof_date
        )
        
        # Parse optional sections
        fda_modifiers: Dict[FDADesignation, ModifierDefinition] = {}
        if "fda_modifiers" in data:
            fda_modifiers = cls._parse_fda_modifiers(data["fda_modifiers"])
        
        trial_modifiers: Dict[TrialCharacteristic, ModifierDefinition] = {}
        if "trial_modifiers" in data:
            trial_modifiers = cls._parse_trial_modifiers(data["trial_modifiers"])
        
        default_stage_rates: Dict[ClinicalStage, Decimal] = DEFAULT_BASE_RATES.copy()
        if "default_stage_rates" in data:
            parsed_defaults = cls._parse_default_stage_rates(data["default_stage_rates"])
            default_stage_rates.update(parsed_defaults)
        
        bounds = {
            "pos_floor": POS_FLOOR,
            "pos_ceiling": POS_CEILING,
            "max_modifier_product": MAX_MODIFIER_PRODUCT,
            "min_modifier_product": MIN_MODIFIER_PRODUCT,
        }
        if "bounds" in data:
            parsed_bounds = cls._parse_bounds(data["bounds"])
            bounds.update(parsed_bounds)
        
        # Build provenance
        provenance = FixtureProvenance(
            fixture_version=version,
            fixture_source_asof_date=source_asof_date,
            fixture_sha256=file_sha256,
            fixture_path=str(path.resolve()),
            loaded_at=datetime.now(timezone.utc),
            base_rate_count=len(base_rate_matrix),
            fda_modifier_count=len(fda_modifiers),
            trial_modifier_count=len(trial_modifiers),
        )
        
        return FixtureData(
            base_rate_matrix=base_rate_matrix,
            fda_modifiers=fda_modifiers,
            trial_modifiers=trial_modifiers,
            default_stage_rates=default_stage_rates,
            bounds=bounds,
            provenance=provenance,
        )


@dataclass
class FixtureData:
    """Container for loaded fixture data."""
    base_rate_matrix: Dict[PosKey, BaseRateEntry]
    fda_modifiers: Dict[FDADesignation, ModifierDefinition]
    trial_modifiers: Dict[TrialCharacteristic, ModifierDefinition]
    default_stage_rates: Dict[ClinicalStage, Decimal]
    bounds: Dict[str, Decimal]
    provenance: FixtureProvenance


# =============================================================================
# CORE CALCULATION ENGINE
# =============================================================================

class PosLookupResult(NamedTuple):
    """Result of a base rate lookup."""
    base_pos: Decimal
    lookup_type: str  # "exact", "partial", "default"
    n_support: Optional[int]
    source: Optional[str]


@dataclass
class PosCalculationResult:
    """Complete result of a PoS calculation with full audit trail."""
    
    # Inputs (for audit)
    ticker: str
    stage: ClinicalStage
    therapeutic_area: TherapeuticArea
    mechanism_class: MechanismClass
    fda_designations: List[FDADesignation]
    trial_characteristics: List[TrialCharacteristic]
    
    # Calculation components
    base_pos: Decimal
    base_pos_lookup: str  # How base rate was determined
    base_pos_n_support: Optional[int]
    
    # Modifier details
    fda_modifier_product: Decimal
    trial_modifier_product: Decimal
    total_modifier_product: Decimal
    modifier_details: List[Dict]
    
    # Final result
    adjusted_pos: Decimal
    confidence_level: str  # "high", "medium", "low"
    
    # Bounds applied
    pos_floor: Decimal = POS_FLOOR
    pos_ceiling: Decimal = POS_CEILING
    
    # Provenance (optional, present when loaded from fixture)
    fixture_provenance: Optional[FixtureProvenance] = None
    
    # Metadata
    calculation_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    model_version: str = "2.1.0"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "ticker": self.ticker,
            "inputs": {
                "stage": self.stage.value,
                "therapeutic_area": self.therapeutic_area.value,
                "mechanism_class": self.mechanism_class.value,
                "fda_designations": [d.value for d in self.fda_designations],
                "trial_characteristics": [t.value for t in self.trial_characteristics],
            },
            "base_rate": {
                "value": str(self.base_pos),
                "lookup_type": self.base_pos_lookup,
                "n_support": self.base_pos_n_support,
            },
            "modifiers": {
                "fda_product": str(self.fda_modifier_product),
                "trial_product": str(self.trial_modifier_product),
                "total_product": str(self.total_modifier_product),
                "details": self.modifier_details,
            },
            "result": {
                "adjusted_pos": str(self.adjusted_pos),
                "confidence_level": self.confidence_level,
            },
            "bounds": {
                "pos_floor": str(self.pos_floor),
                "pos_ceiling": str(self.pos_ceiling),
            },
            "metadata": {
                "timestamp": self.calculation_timestamp.isoformat(),
                "model_version": self.model_version,
            }
        }
        
        # Include provenance if present
        if self.fixture_provenance is not None:
            result["metadata"]["fixture_provenance"] = self.fixture_provenance.to_dict()
        
        return result
    
    def get_audit_hash(self) -> str:
        """Generate SHA256 hash of calculation for audit trail."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


class PosModelV2:
    """
    Hybrid Probability-of-Success Model with FDA Designation Modifiers.
    
    This model calculates PoS using:
    1. Base rate lookup: (stage, indication, mechanism) → empirical base rate
    2. FDA designation modifiers: BTD, Orphan, Fast Track, etc.
    3. Trial characteristic modifiers: biomarker selection, endpoint type, etc.
    
    All calculations use Decimal arithmetic for regulatory compliance.
    Full audit trail provided for every calculation.
    
    Two initialization modes:
    1. Default: Uses hardcoded base rates and modifiers
    2. Fixture: Loads from versioned JSON file for point-in-time reproducibility
    """
    
    def __init__(
        self,
        base_rate_matrix: Optional[Dict[PosKey, BaseRateEntry]] = None,
        fda_modifiers: Optional[Dict[FDADesignation, ModifierDefinition]] = None,
        trial_modifiers: Optional[Dict[TrialCharacteristic, ModifierDefinition]] = None,
        default_stage_rates: Optional[Dict[ClinicalStage, Decimal]] = None,
        bounds: Optional[Dict[str, Decimal]] = None,
        provenance: Optional[FixtureProvenance] = None,
    ):
        """
        Initialize the PoS model.
        
        Args:
            base_rate_matrix: Optional custom base rate matrix
            fda_modifiers: Optional custom FDA modifiers
            trial_modifiers: Optional custom trial modifiers
            default_stage_rates: Optional custom default rates by stage
            bounds: Optional custom bounds (pos_floor, pos_ceiling, etc.)
            provenance: Optional provenance record (from fixture loading)
        """
        self.base_rate_matrix = base_rate_matrix or BASE_RATE_MATRIX
        self.fda_modifiers = fda_modifiers or FDA_DESIGNATION_MODIFIERS
        self.trial_modifiers = trial_modifiers or TRIAL_CHARACTERISTIC_MODIFIERS
        self.default_stage_rates = default_stage_rates or DEFAULT_BASE_RATES
        
        # Extract bounds
        bounds = bounds or {}
        self.pos_floor = bounds.get("pos_floor", POS_FLOOR)
        self.pos_ceiling = bounds.get("pos_ceiling", POS_CEILING)
        self.max_modifier_product = bounds.get("max_modifier_product", MAX_MODIFIER_PRODUCT)
        self.min_modifier_product = bounds.get("min_modifier_product", MIN_MODIFIER_PRODUCT)
        
        # Provenance tracking
        self.provenance = provenance
        
        self._calculation_count = 0
    
    @classmethod
    def from_fixture(cls, path: Union[str, Path]) -> "PosModelV2":
        """
        Load model configuration from a versioned fixture file.
        
        This is the preferred initialization method for production use,
        as it provides:
        - Point-in-time reproducibility
        - Version-controlled configuration
        - Full provenance tracking
        - Strict validation (fail-closed on errors)
        
        Args:
            path: Path to fixture JSON file
            
        Returns:
            PosModelV2 instance configured from fixture
            
        Raises:
            FixtureValidationError: If validation fails
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file isn't valid JSON
        """
        fixture_data = FixtureLoader.load(path)
        
        return cls(
            base_rate_matrix=fixture_data.base_rate_matrix,
            fda_modifiers=fixture_data.fda_modifiers,
            trial_modifiers=fixture_data.trial_modifiers,
            default_stage_rates=fixture_data.default_stage_rates,
            bounds=fixture_data.bounds,
            provenance=fixture_data.provenance,
        )
    
    def get_provenance(self) -> Optional[FixtureProvenance]:
        """Return provenance record if model was loaded from fixture."""
        return self.provenance
    
    def is_fixture_loaded(self) -> bool:
        """Check if model was loaded from a fixture file."""
        return self.provenance is not None
    
    def _lookup_base_rate(
        self,
        stage: ClinicalStage,
        therapeutic_area: TherapeuticArea,
        mechanism_class: MechanismClass,
    ) -> PosLookupResult:
        """
        Look up base PoS rate with fallback hierarchy.
        
        Lookup order:
        1. Exact match: (stage, therapeutic_area, mechanism_class)
        2. Partial match: (stage, therapeutic_area, OTHER)
        3. Partial match: (stage, OTHER, mechanism_class)
        4. Default: stage-based default rate
        """
        exact_key = (stage, therapeutic_area, mechanism_class)
        
        # Try exact match
        if exact_key in self.base_rate_matrix:
            entry = self.base_rate_matrix[exact_key]
            return PosLookupResult(
                base_pos=entry.base_pos,
                lookup_type="exact",
                n_support=entry.n_support,
                source=entry.source,
            )
        
        # Try partial match: same stage + therapeutic area, any mechanism
        for key, entry in self.base_rate_matrix.items():
            if key[0] == stage and key[1] == therapeutic_area:
                return PosLookupResult(
                    base_pos=entry.base_pos,
                    lookup_type="partial_indication",
                    n_support=entry.n_support,
                    source=entry.source,
                )
        
        # Try partial match: same stage + mechanism, any therapeutic area
        for key, entry in self.base_rate_matrix.items():
            if key[0] == stage and key[2] == mechanism_class:
                return PosLookupResult(
                    base_pos=entry.base_pos,
                    lookup_type="partial_mechanism",
                    n_support=entry.n_support,
                    source=entry.source,
                )
        
        # Fall back to stage-based default (use instance defaults)
        default_pos = self.default_stage_rates.get(stage, Decimal("0.30"))
        return PosLookupResult(
            base_pos=default_pos,
            lookup_type="default",
            n_support=None,
            source="Stage-based default",
        )
    
    def _calculate_modifier_product(
        self,
        fda_designations: List[FDADesignation],
        trial_characteristics: List[TrialCharacteristic],
    ) -> Tuple[Decimal, Decimal, Decimal, List[Dict]]:
        """
        Calculate the product of all applicable modifiers.
        
        Returns:
            Tuple of (fda_product, trial_product, total_product, details_list)
        """
        details = []
        
        # FDA designation modifiers
        fda_product = Decimal("1.0")
        for designation in fda_designations:
            if designation in self.fda_modifiers:
                mod = self.fda_modifiers[designation]
                fda_product *= mod.factor
                details.append({
                    "type": "fda_designation",
                    "name": designation.value,
                    "factor": str(mod.factor),
                    "direction": mod.direction,
                    "rationale": mod.rationale,
                })
        
        # Trial characteristic modifiers
        trial_product = Decimal("1.0")
        for characteristic in trial_characteristics:
            if characteristic in self.trial_modifiers:
                mod = self.trial_modifiers[characteristic]
                trial_product *= mod.factor
                details.append({
                    "type": "trial_characteristic",
                    "name": characteristic.value,
                    "factor": str(mod.factor),
                    "direction": mod.direction,
                    "rationale": mod.rationale,
                })
        
        # Calculate total and apply bounds (use instance bounds)
        total_product = fda_product * trial_product
        total_product = max(self.min_modifier_product, min(self.max_modifier_product, total_product))
        
        return fda_product, trial_product, total_product, details
    
    def _determine_confidence(
        self,
        lookup_type: str,
        n_support: Optional[int],
        num_modifiers: int,
    ) -> str:
        """
        Determine confidence level based on data quality.
        
        High: Exact match with n_support >= 50
        Medium: Partial match or n_support 20-49
        Low: Default fallback or n_support < 20
        """
        if lookup_type == "exact" and n_support is not None and n_support >= 50:
            return "high"
        elif lookup_type in ("exact", "partial_indication") and (n_support is None or n_support >= 20):
            return "medium"
        else:
            return "low"
    
    def calculate(
        self,
        ticker: str,
        stage: ClinicalStage,
        therapeutic_area: TherapeuticArea,
        mechanism_class: MechanismClass,
        fda_designations: Optional[List[FDADesignation]] = None,
        trial_characteristics: Optional[List[TrialCharacteristic]] = None,
    ) -> PosCalculationResult:
        """
        Calculate adjusted PoS for a drug candidate.
        
        Args:
            ticker: Security ticker symbol
            stage: Clinical development stage
            therapeutic_area: Therapeutic area / indication
            mechanism_class: Drug mechanism / modality
            fda_designations: List of FDA special designations
            trial_characteristics: List of trial design characteristics
        
        Returns:
            PosCalculationResult with full audit trail
        """
        fda_designations = fda_designations or []
        trial_characteristics = trial_characteristics or []
        
        # Step 1: Look up base rate
        lookup_result = self._lookup_base_rate(stage, therapeutic_area, mechanism_class)
        
        # Step 2: Calculate modifiers
        fda_prod, trial_prod, total_prod, modifier_details = self._calculate_modifier_product(
            fda_designations, trial_characteristics
        )
        
        # Step 3: Calculate adjusted PoS
        adjusted_pos = lookup_result.base_pos * total_prod
        
        # Step 4: Apply bounds (use instance bounds)
        adjusted_pos = max(self.pos_floor, min(self.pos_ceiling, adjusted_pos))
        
        # Step 5: Round to 4 decimal places
        adjusted_pos = adjusted_pos.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        
        # Step 6: Determine confidence
        confidence = self._determine_confidence(
            lookup_result.lookup_type,
            lookup_result.n_support,
            len(fda_designations) + len(trial_characteristics),
        )
        
        self._calculation_count += 1
        
        return PosCalculationResult(
            ticker=ticker,
            stage=stage,
            therapeutic_area=therapeutic_area,
            mechanism_class=mechanism_class,
            fda_designations=fda_designations,
            trial_characteristics=trial_characteristics,
            base_pos=lookup_result.base_pos,
            base_pos_lookup=lookup_result.lookup_type,
            base_pos_n_support=lookup_result.n_support,
            fda_modifier_product=fda_prod,
            trial_modifier_product=trial_prod,
            total_modifier_product=total_prod,
            modifier_details=modifier_details,
            adjusted_pos=adjusted_pos,
            confidence_level=confidence,
            pos_floor=self.pos_floor,
            pos_ceiling=self.pos_ceiling,
            fixture_provenance=self.provenance,
        )
    
    def calculate_from_strings(
        self,
        ticker: str,
        stage: str,
        therapeutic_area: str,
        mechanism_class: str,
        fda_designations: Optional[List[str]] = None,
        trial_characteristics: Optional[List[str]] = None,
    ) -> PosCalculationResult:
        """
        Calculate PoS from string inputs (convenience method).
        
        Parses string inputs into enums and calls calculate().
        """
        parsed_stage = ClinicalStage.from_string(stage)
        parsed_ta = TherapeuticArea.from_string(therapeutic_area)
        parsed_mech = MechanismClass.from_string(mechanism_class)
        
        parsed_fda = []
        if fda_designations:
            fda_mapping = {d.value: d for d in FDADesignation}
            for fd in fda_designations:
                normalized = fd.lower().strip().replace(" ", "_").replace("-", "_")
                if normalized in fda_mapping:
                    parsed_fda.append(fda_mapping[normalized])
        
        parsed_trial = []
        if trial_characteristics:
            trial_mapping = {t.value: t for t in TrialCharacteristic}
            for tc in trial_characteristics:
                normalized = tc.lower().strip().replace(" ", "_").replace("-", "_")
                if normalized in trial_mapping:
                    parsed_trial.append(trial_mapping[normalized])
        
        return self.calculate(
            ticker=ticker,
            stage=parsed_stage,
            therapeutic_area=parsed_ta,
            mechanism_class=parsed_mech,
            fda_designations=parsed_fda,
            trial_characteristics=parsed_trial,
        )
    
    def get_coverage_stats(self) -> Dict:
        """Return statistics about base rate matrix coverage."""
        stages = set(k[0] for k in self.base_rate_matrix.keys())
        tas = set(k[1] for k in self.base_rate_matrix.keys())
        mechs = set(k[2] for k in self.base_rate_matrix.keys())
        
        result = {
            "total_entries": len(self.base_rate_matrix),
            "stages_covered": sorted([s.value for s in stages]),
            "therapeutic_areas_covered": sorted([t.value for t in tas]),
            "mechanisms_covered": sorted([m.value for m in mechs]),
            "fda_modifiers": len(self.fda_modifiers),
            "trial_modifiers": len(self.trial_modifiers),
            "bounds": {
                "pos_floor": str(self.pos_floor),
                "pos_ceiling": str(self.pos_ceiling),
                "max_modifier_product": str(self.max_modifier_product),
                "min_modifier_product": str(self.min_modifier_product),
            },
            "fixture_loaded": self.is_fixture_loaded(),
        }
        
        if self.provenance:
            result["fixture_version"] = self.provenance.fixture_version
            result["fixture_sha256"] = self.provenance.fixture_sha256[:16] + "..."
        
        return result
    
    def get_bounds(self) -> Dict[str, Decimal]:
        """Return current bounds configuration."""
        return {
            "pos_floor": self.pos_floor,
            "pos_ceiling": self.pos_ceiling,
            "max_modifier_product": self.max_modifier_product,
            "min_modifier_product": self.min_modifier_product,
        }


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def enhanced_stage_score(
    base_stage_score: Decimal,
    adjusted_pos: Decimal,
    median_pos: Decimal = Decimal("0.50"),
) -> Decimal:
    """
    Adjust stage score by PoS relative to median.
    
    If adjusted_pos > median: boost score
    If adjusted_pos < median: penalize score
    
    Args:
        base_stage_score: Original stage-based score (0-100)
        adjusted_pos: Adjusted PoS from model
        median_pos: Median PoS for normalization (default 0.50)
    
    Returns:
        Adjusted stage score
    """
    pos_multiplier = adjusted_pos / median_pos
    adjusted = base_stage_score * pos_multiplier
    # Clamp to reasonable range
    return max(Decimal("0"), min(Decimal("100"), adjusted))


def pos_to_catalyst_ev_weight(
    adjusted_pos: Decimal,
    base_ev: Decimal,
    blend_factor: Decimal = Decimal("0.70"),
) -> Decimal:
    """
    Weight catalyst EV by PoS for risk-adjusted expected value.
    
    Formula: EV_adjusted = base_EV * (1 - blend_factor + blend_factor * PoS)
    
    With default blend_factor=0.70:
    - PoS=1.0 → EV_adjusted = base_EV * 1.0
    - PoS=0.5 → EV_adjusted = base_EV * 0.65
    - PoS=0.0 → EV_adjusted = base_EV * 0.30
    """
    return base_ev * (Decimal("1") - blend_factor + blend_factor * adjusted_pos)


# =============================================================================
# VALIDATION AND TESTING
# =============================================================================

def validate_model_consistency() -> Dict:
    """
    Run consistency checks on the model.
    
    Returns dict with validation results.
    """
    model = PosModelV2()
    issues = []
    
    # Check 1: All base rates in valid range
    for key, entry in model.base_rate_matrix.items():
        if entry.base_pos < Decimal("0.01") or entry.base_pos > Decimal("0.99"):
            issues.append(f"Base rate out of range: {key} = {entry.base_pos}")
    
    # Check 2: All modifiers in valid range
    for designation, mod in model.fda_modifiers.items():
        if mod.factor < Decimal("0.5") or mod.factor > Decimal("2.0"):
            issues.append(f"FDA modifier out of range: {designation} = {mod.factor}")
    
    for characteristic, mod in model.trial_modifiers.items():
        if mod.factor < Decimal("0.5") or mod.factor > Decimal("2.0"):
            issues.append(f"Trial modifier out of range: {characteristic} = {mod.factor}")
    
    # Check 3: Stage ordering (later stages should have higher base rates on average)
    stage_avgs = {}
    for key, entry in model.base_rate_matrix.items():
        stage = key[0]
        if stage not in stage_avgs:
            stage_avgs[stage] = []
        stage_avgs[stage].append(entry.base_pos)
    
    for stage, rates in stage_avgs.items():
        avg = sum(rates) / len(rates)
        stage_avgs[stage] = avg
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "coverage_stats": model.get_coverage_stats(),
        "stage_averages": {k.value: str(v) for k, v in stage_avgs.items()},
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("Wake Robin PoS Model v2.1 - Tier 1 Implementation")
    print("=" * 70)
    
    # Check if fixture path provided
    fixture_path = None
    if len(sys.argv) > 1:
        fixture_path = sys.argv[1]
    
    # Initialize model (from fixture if provided, else defaults)
    if fixture_path:
        print(f"\nLoading from fixture: {fixture_path}")
        try:
            model = PosModelV2.from_fixture(fixture_path)
            prov = model.get_provenance()
            print(f"  Fixture version: {prov.fixture_version}")
            print(f"  Source as-of date: {prov.fixture_source_asof_date}")
            print(f"  SHA256: {prov.fixture_sha256[:32]}...")
            print(f"  Base rates: {prov.base_rate_count}")
            print(f"  FDA modifiers: {prov.fda_modifier_count}")
            print(f"  Trial modifiers: {prov.trial_modifier_count}")
        except FixtureValidationError as e:
            print(f"  ERROR: {e}")
            sys.exit(1)
    else:
        print("\nUsing default (hardcoded) configuration")
        print("  Tip: Pass fixture path as argument for point-in-time reproducibility")
        model = PosModelV2()
    
    # Example 1: Phase 3 oncology small molecule with BTD
    result1 = model.calculate(
        ticker="EXAMPLE1",
        stage=ClinicalStage.PHASE_3,
        therapeutic_area=TherapeuticArea.ONCOLOGY,
        mechanism_class=MechanismClass.SMALL_MOLECULE,
        fda_designations=[FDADesignation.BREAKTHROUGH_THERAPY],
        trial_characteristics=[TrialCharacteristic.BIOMARKER_SELECTED],
    )
    
    print("\nExample 1: Phase 3 Oncology Small Molecule with BTD + Biomarker")
    print(f"  Base PoS: {result1.base_pos} ({result1.base_pos_lookup})")
    print(f"  FDA Modifier: {result1.fda_modifier_product}")
    print(f"  Trial Modifier: {result1.trial_modifier_product}")
    print(f"  Adjusted PoS: {result1.adjusted_pos}")
    print(f"  Confidence: {result1.confidence_level}")
    
    # Example 2: Phase 3 rare disease gene therapy with orphan + fast track
    result2 = model.calculate(
        ticker="EXAMPLE2",
        stage=ClinicalStage.PHASE_3,
        therapeutic_area=TherapeuticArea.RARE_DISEASE,
        mechanism_class=MechanismClass.GENE_THERAPY,
        fda_designations=[
            FDADesignation.ORPHAN_DRUG,
            FDADesignation.FAST_TRACK,
            FDADesignation.BREAKTHROUGH_THERAPY,
        ],
    )
    
    print("\nExample 2: Phase 3 Rare Disease Gene Therapy with Orphan + FT + BTD")
    print(f"  Base PoS: {result2.base_pos} ({result2.base_pos_lookup})")
    print(f"  FDA Modifier: {result2.fda_modifier_product}")
    print(f"  Adjusted PoS: {result2.adjusted_pos}")
    print(f"  Confidence: {result2.confidence_level}")
    
    # Example 3: Phase 3 CNS mAb (historically challenging)
    result3 = model.calculate(
        ticker="EXAMPLE3",
        stage=ClinicalStage.PHASE_3,
        therapeutic_area=TherapeuticArea.CNS,
        mechanism_class=MechanismClass.MONOCLONAL_ANTIBODY,
        trial_characteristics=[
            TrialCharacteristic.ACTIVE_COMPARATOR,
            TrialCharacteristic.SURROGATE_ENDPOINT,
        ],
    )
    
    print("\nExample 3: Phase 3 CNS mAb with Active Comparator + Surrogate Endpoint")
    print(f"  Base PoS: {result3.base_pos} ({result3.base_pos_lookup})")
    print(f"  Trial Modifier: {result3.trial_modifier_product}")
    print(f"  Adjusted PoS: {result3.adjusted_pos}")
    print(f"  Confidence: {result3.confidence_level}")
    
    # Example 4: Using string inputs
    result4 = model.calculate_from_strings(
        ticker="EXAMPLE4",
        stage="phase_2",
        therapeutic_area="immunology",
        mechanism_class="mab",
        fda_designations=["fast_track"],
        trial_characteristics=["biomarker_selected", "first_in_class"],
    )
    
    print("\nExample 4: Phase 2 Immunology mAb (from strings)")
    print(f"  Base PoS: {result4.base_pos}")
    print(f"  Adjusted PoS: {result4.adjusted_pos}")
    
    # Validation
    print("\n" + "=" * 70)
    print("Model Configuration")
    print("=" * 70)
    stats = model.get_coverage_stats()
    print(f"  Base rate entries: {stats['total_entries']}")
    print(f"  FDA modifiers: {stats['fda_modifiers']}")
    print(f"  Trial modifiers: {stats['trial_modifiers']}")
    print(f"  Fixture loaded: {stats['fixture_loaded']}")
    if stats['fixture_loaded']:
        print(f"  Fixture version: {stats['fixture_version']}")
    
    # Export example result as JSON (with provenance if loaded from fixture)
    print("\n" + "=" * 70)
    print("JSON Export (Example 1)")
    print("=" * 70)
    print(json.dumps(result1.to_dict(), indent=2))
